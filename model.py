from grouped_gemm_op import grouped_gemm

import torch
import torch.nn as nn


class SwiGLUMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, top_k: int = 2):
        super(SwiGLUMoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # two weights: value & gate
        self.expert_weights_v = nn.Parameter(torch.randn(num_experts, input_dim, hidden_dim))  # [num_experts, d_model, hidden_dim]
        self.expert_weights_g = nn.Parameter(torch.randn(num_experts, input_dim, hidden_dim))  # [num_experts, d_model, hidden_dim]
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [batch, input_dim]
        expert_indices (optional): [batch, top_k] expert ids to use. If not provided we run topk on gate.
        """
        batch, d_model = x.shape

        # If caller did not provide routing, use gate to pick top-k experts and weights.
        router_logits = self.gate(x)  # [batch, num_experts]
        router_probs = torch.softmax(router_logits, dim=-1)

        if expert_indices is None:
            top_k = min(self.top_k, self.num_experts)
            topk_scores, topk_indices = torch.topk(router_probs, k=top_k, dim=-1)
        else:
            topk_indices = expert_indices.to(torch.long)
            top_k = topk_indices.shape[1]
            topk_scores = torch.gather(router_probs, 1, topk_indices)

        grouped_val = grouped_gemm(
            tokens=x,
            weights=self.expert_weights_v,
            top_k_expert_activation=topk_indices,
        ).view(batch, top_k, -1) # call the grouped_gemm twice to calculate value and gate

        grouped_gate = grouped_gemm(
            tokens=x,
            weights=self.expert_weights_g,
            top_k_expert_activation=topk_indices,
        ).view(batch, top_k, -1)

        # SwiGLU: val * σ(gate)
        activated = grouped_val * torch.sigmoid(grouped_gate)

        # Weight expert outputs by router probabilities and combine.
        weighted_sum = (activated * topk_scores.unsqueeze(-1)).sum(dim=1)
        return weighted_sum


if __name__ == "__main__":
    # Minimal driver to exercise grouped_gemm + IR/guard dumping
    from dump_ir import dump_ir_and_guards

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run grouped_gemm; no GPU detected.")

    torch.manual_seed(0)
    model = SwiGLUMoE(input_dim=16, hidden_dim=8, num_experts=4, top_k=2).cuda()
    example = torch.randn(3, 16, device="cuda")

    def identity_pass(gm, example_inputs):
        return gm

    dump_ir_and_guards(model, (example,), backend_pass=identity_pass)
