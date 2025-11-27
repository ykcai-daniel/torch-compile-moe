import os
from typing import List

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c

try:
    from functorch.compile import aot_module 
except Exception:
    from torch._functorch.aot_autograd import aot_module


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
def make_aot_compiled_model(model: nn.Module, rank: int) -> nn.Module:
    def fw_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        return gm.forward

    def bw_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        if rank == 0:
            print("=" * 80)
            print("AOTAutograd BACKWARD GRAPH (bw_graph):")
            print("=" * 80)
            print(gm.graph)
            print("=" * 80)
        return gm.forward

    compiled_model = aot_module(
        model,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
    )

    return compiled_model


def main():
    setup_dist()

    device = torch.device("cuda")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    hidden_dim = 256
    ffn_dim = 1024
    local_tokens = 4 * world_size  # number of tokens per rank

    model = ThreeLayerAllToAllMoE(hidden_dim, ffn_dim, num_layers=3).to(device)
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
