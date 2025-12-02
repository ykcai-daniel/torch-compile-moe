import os
from typing import List

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
import torch.fx as fx

from torch.profiler import profile, record_function, ProfilerActivity

try:
    from functorch.compile import aot_module, make_boxed_func 
except Exception:
    from torch._functorch.aot_autograd import aot_module, make_boxed_func 


# -------------------------------
# 1. Traceable 3-layer MoE with all_to_all
# -------------------------------
class ThreeLayerAllToAllMoE(nn.Module):
    """
      (all_to_all -> MLP1 -> all_to_all)
      -> (all_to_all -> MLP2 -> all_to_all)
      -> (all_to_all -> MLP3 -> all_to_all)
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, num_layers: int = 3):
        super().__init__()
        assert num_layers == 3

        # Three independent MLP experts
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, ffn_dim),
                    nn.GELU(),
                    nn.Linear(ffn_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (local_tokens, hidden_dim) local tokens on each rank
        """
        assert dist.is_initialized(), "Need dist.init_process_group first!"
        world_size = dist.get_world_size()

        local_tokens, hidden_dim = x.shape
        assert local_tokens % world_size == 0
        chunk = local_tokens // world_size
        split_sizes = [chunk] * world_size

        group = dist.group.WORLD.group_name  # Use WORLD group name as the group identifier

        out = x
        for mlp in self.mlps:
            # 1) all-to-all gather tokens
            out_gathered = ft_c.all_to_all_single_autograd(
                out,
                split_sizes,  # input_split_sizes
                split_sizes,  # output_split_sizes
                group,
            )

            # 2) MLP expert
            out_mlp = mlp(out_gathered)

            # 3) all-to-all scatter tokens back
            out = ft_c.all_to_all_single_autograd(
                out_mlp,
                split_sizes,
                split_sizes,
                group,
            )

        return out


# -----------------------
# 2. Distributed init/cleanup
# -----------------------
def setup_dist():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"[rank {dist.get_rank()}] init done, using cuda:{local_rank}")


def cleanup_dist():
    dist.destroy_process_group()


# ------------------------------------
# 3. AOTAutograd compile module + print BW graph only
# ------------------------------------
def make_aot_compiled_model(model: ThreeLayerAllToAllMoE, rank: int, enable_overlap_pass: bool) -> nn.Module:
    def fw_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        if rank == 0:
            print("=" * 80)
            print("AOTAutograd FORWARD GRAPH (fw_graph):")
            print("=" * 80)
            print(gm.graph)
            print("=" * 80, flush=True)
        return make_boxed_func(gm.forward)

    def bw_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        if rank == 0:
            print("=== BEFORE PASS ===")
            print(gm.graph)

        new_gm = gm
        if enable_overlap_pass:
            new_gm = apply_moe_overlap_pass(gm)

        if rank == 0:
            print("=== AFTER PASS ===")
            print(new_gm.graph, flush=True)

        return make_boxed_func(new_gm.forward)

    compiled_model = aot_module(
        model,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
    )
    return compiled_model

def apply_moe_overlap_pass(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Generalized MoE backward‐overlap pass.

    For arbitrary MoE backward graphs:
      - Identify all patterns of:  all_to_all_single(mm_*, split_sizes..., group)
      - For each communication node, move it immediately after its corresponding mm_* node
      - Do NOT modify the position of wait_tensor nodes

    The goal is to launch communication as early as possible so that subsequent
    compute that does not depend on the communication can overlap with it.
    """

    graph = gm.graph
    target_comm = torch.ops._c10d_functional.all_to_all_single.default
    target_mm = torch.ops.aten.mm.default

    # 1. Collect a snapshot of all all_to_all_single nodes
    comm_nodes = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target == target_comm
    ]

    if not comm_nodes:
        return gm

    # 2. Process each communication node
    for comm_node in comm_nodes:
        # The first argument is the communication input. It may be:
        # placeholder, mm node, wait_tensor node, etc.
        if not comm_node.args:
            continue
        comm_input = comm_node.args[0]

        # Handle only the specific pattern where the input is an mm node (typical dW)
        if not isinstance(comm_input, fx.Node):
            # e.g., the top-level all_to_all_single(tangents_1, ...) whose input is a placeholder
            continue
        if not (
            comm_input.op == "call_function"
            and comm_input.target == target_mm
        ):
            # e.g., all_to_all_single(wait_tensor_k, ...) — not handled in this prototype
            continue

        # If comm_node already follows comm_input, you could skip,
        # but recreating it is safe and simpler.
        kwargs_copy = dict(comm_node.kwargs) if comm_node.kwargs else {}

        # Insert new comm node right after the mm_* node
        with graph.inserting_after(comm_input):
            new_comm = graph.call_function(
                target_comm,
                comm_node.args,
                kwargs_copy,
            )

        # Copy meta/name for easier debugging
        new_comm.meta = dict(getattr(comm_node, "meta", {}))
        new_comm.name = comm_node.name

        # Replace all uses of old comm with the new comm (typically consumed by wait_tensor)
        comm_node.replace_all_uses_with(new_comm)

        # Remove the old communication node
        graph.erase_node(comm_node)

    graph.lint()
    gm.recompile()
    return gm

def train_one_step_with_profiler(compiled_model, x, logdir: str, rank: int):
    torch.cuda.synchronize()

    rank_logdir = os.path.join(logdir, f"rank{rank}")
    os.makedirs(rank_logdir, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(rank_logdir),
    ) as prof:
        with record_function("moe_train_step"):
            out = compiled_model(x)
            loss = out.sum()
            loss.backward()
        torch.cuda.synchronize()
        prof.step()
    if rank == 0:
        print(f"[rank 0] profile written to {rank_logdir}", flush=True)


def main():
    setup_dist()

    device = torch.device("cuda")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    hidden_dim = 1024
    ffn_dim = 17408
    local_tokens = 4 * world_size  # number of tokens per rank

    # ---------- BEFORE PASS ----------
    model = ThreeLayerAllToAllMoE(hidden_dim, ffn_dim, num_layers=3).to(device)
    x = torch.randn(local_tokens, hidden_dim, device=device, requires_grad=True)

    compiled_before = make_aot_compiled_model(model, rank, enable_overlap_pass=False).to(device)
    # compiled_model = make_aot_compiled_model(model, rank).to(device)

    out = compiled_before(x)
    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize()
    if rank == 0:
        print(f"[rank 0] Warmup BEFORE done, loss = {loss.item():.4f}", flush=True)

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    if rank == 0:
        print("[rank 0] Profiling BEFORE pass...", flush=True)
    train_one_step_with_profiler(compiled_before, x, logdir="./log_before", rank=rank)

    # ---------- AFTER PASS ----------
    model2 = ThreeLayerAllToAllMoE(hidden_dim, ffn_dim).to(device)
    x2 = torch.randn(local_tokens, hidden_dim, device=device, requires_grad=True)

    compiled_after = make_aot_compiled_model(model2, rank, enable_overlap_pass=True).to(device)

    out2 = compiled_after(x2)
    loss2 = out2.sum()
    loss2.backward()
    torch.cuda.synchronize()
    if rank == 0:
        print(f"[rank 0] Warmup AFTER done, loss = {loss2.item():.4f}", flush=True)

    for p in model2.parameters():
        if p.grad is not None:
            p.grad = None

    if rank == 0:
        print("[rank 0] Profiling AFTER pass...", flush=True)
    train_one_step_with_profiler(compiled_after, x2, logdir="./log_after", rank=rank)

    if rank == 0:
        print("[rank 0] Done profiling BEFORE & AFTER.", flush=True)

    cleanup_dist()


if __name__ == "__main__":
    main()
