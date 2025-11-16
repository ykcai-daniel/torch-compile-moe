from grouped_gemm_op import grouped_gemm

import torch
import torch.nn as nn


class SwiGLUMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        # allocate expert a single weight tensor
        super(SwiGLUMoE, self).__init__()
        self.num_experts = num_experts
        self.expert_weights = nn.Parameter(torch.randn(num_experts, input_dim, hidden_dim * 2))
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x, expert_indices):
        # TODO

        # dispatch with grouped gemm
        expert_outputs = grouped_gemm(expert_inputs, expert_weights)