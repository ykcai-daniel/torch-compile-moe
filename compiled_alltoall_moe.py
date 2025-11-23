# import os
# from typing import List

# import torch
# import torch.nn as nn
# import torch.distributed as dist
# import torch.distributed._functional_collectives as ft_c


# # -------------------------------
# # 1. Traceable MoE + all_to_all
# # -------------------------------
# class AllToAllMoE(nn.Module):
#     """
#     Minimal MoE: all_to_all -> MLP -> all_to_all
#
#     Input: local tokens on each rank, shape (local_tokens, hidden_dim)
#     We use all_to_all_single_autograd on dim=0 to make it torch.compile traceable.
#     """

#     def __init__(self, hidden_dim: int, ffn_dim: int):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_dim, ffn_dim),
#             nn.GELU(),
#             nn.Linear(ffn_dim, hidden_dim),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (local_tokens, hidden_dim) on each rank
#         """
#         assert dist.is_initialized(), "Need dist.init_process_group first!"
#         world_size = dist.get_world_size()

#         local_tokens, hidden_dim = x.shape
#         assert local_tokens % world_size == 0, (
#             f"local_tokens={local_tokens} must be divisible by world_size={world_size}"
#         )
#         chunk = local_tokens // world_size
#         split_sizes = [chunk] * world_size

#         # Same usage as in PyTorch tests: group = dist.group.WORLD.group_name :contentReference[oaicite:0]{index=0}
#         group = dist.group.WORLD.group_name

#         # 1) all-to-all gather tokens
#         x_gathered = ft_c.all_to_all_single_autograd(
#             x,           # (local_tokens, hidden_dim)
#             split_sizes, # input_split_sizes
#             split_sizes, # output_split_sizes
#             group,       # group identifier
#         )

#         # 2) MLP expert
#         y = self.mlp(x_gathered)

#         # 3) all-to-all scatter tokens back
#         y_scattered = ft_c.all_to_all_single_autograd(
#             y,
#             split_sizes,
#             split_sizes,
#             group,
#         )

#         return y_scattered


# # ---------------------------------
# # 2. Custom backend: print FX IR
# # ---------------------------------
# def print_ir_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
#     """
#     torch.compile backend:
#     - Print the FX graph
#     - Return gm.forward as the real execution function
#     """
#     rank = dist.get_rank() if dist.is_initialized() else 0
#     if rank == 0:
#         print("=" * 80)
#         print("FX GRAPH:")
#         print("=" * 80)
#         print(gm.graph)
#         print("=" * 80)

#     return gm.forward


# # -----------------------
# # 3. Distributed init/cleanup
# # -----------------------
# def setup_dist():
#     dist.init_process_group(backend="nccl")
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)
#     print(f"[rank {dist.get_rank()}] init done, using cuda:{local_rank}")


# def cleanup_dist():
#     dist.destroy_process_group()


# # --------------
# # 4. main logic
# # --------------
# def main():
#     setup_dist()

#     device = torch.device("cuda")
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()

#     hidden_dim = 256
#     ffn_dim = 1024

#     # local tokens per rank; must be divisible by world_size
#     local_tokens = 4 * world_size  # e.g., world_size=2 → 8

#     model = AllToAllMoE(hidden_dim, ffn_dim).to(device)

#     # Input shape: (local_tokens, hidden_dim)
#     x = torch.randn(local_tokens, hidden_dim, device=device, requires_grad=True)

#     # Compile model, fullgraph=True ensures collectives stay in one graph
#     compiled_model = torch.compile(
#         model,
#         backend=print_ir_backend,
#         fullgraph=True,
#     )

#     # Forward + backward
#     out = compiled_model(x)
#     loss = out.sum()
#     loss.backward()

#     if rank == 0:
#         print(f"[rank {rank}] Done. loss = {loss.item():.4f}")

#     cleanup_dist()


# if __name__ == "__main__":
#     main()

import os
from typing import List

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c

# --- AOTAutograd: wrap module with aot_module ---
try:
    # Legacy functorch
    from functorch.compile import aot_module
except Exception:
    # PyTorch 2.x built-in
    from torch._functorch.aot_autograd import aot_module


# -------------------------------
# 1. Traceable MoE with all_to_all
# -------------------------------
class AllToAllMoE(nn.Module):
    """
    all_to_all -> MLP -> all_to_all
    """

    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
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

        # Use WORLD group name as the group identifier
        group = dist.group.WORLD.group_name

        # 1) all-to-all gather tokens
        x_gathered = ft_c.all_to_all_single_autograd(
            x,
            split_sizes,  # input_split_sizes
            split_sizes,  # output_split_sizes
            group,
        )

        # 2) MLP expert
        y = self.mlp(x_gathered)

        # 3) all-to-all scatter tokens back
        y_scattered = ft_c.all_to_all_single_autograd(
            y,
            split_sizes,
            split_sizes,
            group,
        )

        return y_scattered


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
# 3. AOTAutograd compile module + print FW/BW graphs
# ------------------------------------
def make_aot_compiled_model(model: AllToAllMoE, rank: int) -> nn.Module:
    """
    Use aot_module to compile AllToAllMoE with AOTAutograd:
    - fw_compiler prints forward FX graph
    - bw_compiler prints backward FX graph
    The returned compiled_model(x) behaves the same but both fwd & bwd use AOT graphs.
    """

    def fw_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        if rank == 0:
            print("=" * 80)
            print("AOTAutograd FORWARD GRAPH (fw_graph):")
            print("=" * 80)
            print(gm.graph)
            print("=" * 80)
        return gm.forward

    def bw_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        if rank == 0:
            print("=" * 80)
            print("AOTAutograd BACKWARD GRAPH (bw_graph):")
            print("=" * 80)
            print(gm.graph)
            print("=" * 80)
        return gm.forward

    # aot_module lifts params/buffers as inputs and invokes aot_function,
    # automatically handling FakeTensor so no "params are real tensor" errors occur.
    compiled_model = aot_module(
        model,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
    )

    return compiled_model


# --------------
# 4. main logic
# --------------
def main():
    setup_dist()

    device = torch.device("cuda")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    hidden_dim = 256
    ffn_dim = 1024
    local_tokens = 4 * world_size  # number of tokens per rank

    # Original MoE model
    model = AllToAllMoE(hidden_dim, ffn_dim).to(device)

    # AOTAutograd-compiled model (prints fw_graph / bw_graph)
    compiled_model = make_aot_compiled_model(model, rank).to(device)

    # Input: (local_tokens, hidden_dim)
    x = torch.randn(local_tokens, hidden_dim, device=device, requires_grad=True)

    # Forward + backward
    out = compiled_model(x)
    loss = out.sum()
    loss.backward()

    if rank == 0:
        print(f"[rank {rank}] Done. loss = {loss.item():.4f}")

    cleanup_dist()


if __name__ == "__main__":
    main()
