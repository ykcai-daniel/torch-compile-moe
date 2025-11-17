from grouped_gemm_op import grouped_gemm

import torch
import torch.nn as nn


class SwiGLUMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, top_k: int = 2):
        super(SwiGLUMoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim
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
        router_probs = torch.softmax(router_logits, dim=1)

        if expert_indices is None:
            top_k = min(self.top_k, self.num_experts)
            topk_scores, topk_indices = torch.topk(router_probs, k=top_k, dim=1)
        else:
            topk_indices = expert_indices.to(torch.long)
            top_k = topk_indices.shape[1]
            topk_scores = torch.gather(router_probs, 1, topk_indices)

        # Note: to make it torch compile friendly, we must not use -1 in view/reshape.
        grouped_val = grouped_gemm(
            tokens=x,
            weights=self.expert_weights_v,
            top_k_expert_activation=topk_indices,
        ).view(batch, top_k, self.hidden_dim) 

        grouped_gate = grouped_gemm(
            tokens=x,
            weights=self.expert_weights_g,
            top_k_expert_activation=topk_indices,
        ).view(batch, top_k, self.hidden_dim)

        # SwiGLU: val * σ(gate)
        activated = grouped_val * torch.sigmoid(grouped_gate)

        # Weight expert outputs by router probabilities and combine.
        weighted_sum = (activated * topk_scores.unsqueeze(2)).sum(dim=1)
        return weighted_sum


def demo_compiled_backward(model: nn.Module, example: torch.Tensor, *,
                           use_fullgraph: bool = True,
                           backend: str | None = None):
    """
    Compile the model (torch.compile) and run forward + backward to exercise the compiled backward path.
    - use_fullgraph=True will compile forward+backward together (recommended if supported)
    - backend: optional name passed to torch.compile(..., backend=backend)
    """
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build.")

    # Move model and example to device already done by caller; assume CUDA.
    model = model.train()
    model.zero_grad()

    compile_kwargs = {"fullgraph": use_fullgraph}
    if backend is not None:
        compile_kwargs["backend"] = backend

    print(f"Compiling model with torch.compile(fullgraph={use_fullgraph}, backend={backend}) ...")
    compiled_model = torch.compile(model, **compile_kwargs)

    # Run forward
    out = compiled_model(example)
    # create a simple scalar loss to backprop
    loss = out.sum()
    print("Running backward on compiled model...")
    loss.backward()

    # Show that gradients are present (simple check)
    v_grad_norm = model.expert_weights_v.grad.norm().item() if model.expert_weights_v.grad is not None else 0.0
    g_grad_norm = model.expert_weights_g.grad.norm().item() if model.expert_weights_g.grad is not None else 0.0
    gate_grad_norm = model.gate.weight.grad.norm().item() if model.gate.weight.grad is not None else 0.0

    print(f"Gradient norms after backward -- expert_weights_v: {v_grad_norm:.6f}, "
          f"expert_weights_g: {g_grad_norm:.6f}, gate.weight: {gate_grad_norm:.6f}")

    return compiled_model, {"v_grad_norm": v_grad_norm, "g_grad_norm": g_grad_norm, "gate_grad_norm": gate_grad_norm}


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

    # 1) Original IR + guards dump (forward tracing demo)
    print("=== Dumping IR and guards for original (uncompiled) forward ===")
    dump_ir_and_guards(model, (example,), backend_pass=identity_pass)

    # 2) Compiled-backward demo
    print("\n=== Demo: compiled backward (torch.compile, forward+backward) ===")
    try:
        compiled_model, grads = demo_compiled_backward(model, example, use_fullgraph=True, backend=None)
        # Optionally dump IR/guards for the compiled model as well (if you want to inspect compiled graph).
        # Some dump utilities expect an nn.Module; compiled_model is callable as well.
        # If dump_ir_and_guards can accept the compiled model, you can uncomment:
        # dump_ir_and_guards(compiled_model, (example,), backend_pass=identity_pass)
    except Exception as e:
        print("Compiled-backward demo failed:", e)
        print("You can try setting use_fullgraph=False or specifying a backend supported by your PyTorch build.")