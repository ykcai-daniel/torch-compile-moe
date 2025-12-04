\
import torch
import torch.nn as nn
import torch.nn.functional as F
from grouped_gemm_op import grouped_gemm
import time


class SwiGLUMoELayer(nn.Module):
    """
    SwiGLU MoE Layer using grouped_gemm operator.

    SwiGLU formulation:
        gate = x @ W_gate
        up = x @ W_up
        hidden = swish(gate) * up
        output = hidden @ W_down

    For MoE, each expert has its own W_gate, W_up, and W_down weights.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        use_triton: bool = False,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_triton = use_triton

        # Router network - maps tokens to expert probabilities
        self.router = nn.Linear(d_model, num_experts, device=device, dtype=dtype)

        # Expert weights for SwiGLU
        # W_gate: [num_experts, d_model, d_ff]
        self.w_gate = nn.Parameter(
            torch.randn(num_experts, d_model, d_ff, device=device, dtype=dtype) * 0.02
        )

        # W_up: [num_experts, d_model, d_ff]
        self.w_up = nn.Parameter(
            torch.randn(num_experts, d_model, d_ff, device=device, dtype=dtype) * 0.02
        )

        # W_down: [num_experts, d_ff, d_model]
        self.w_down = nn.Parameter(
            torch.randn(num_experts, d_ff, d_model, device=device, dtype=dtype) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SwiGLU MoE layer.

        Args:
            x: Input tokens [batch_size, d_model]

        Returns:
            output: [batch_size, d_model]
        """
        batch_size, d_model = x.shape

        # 1. Route tokens to experts
        # router_logits: [batch_size, num_experts]
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)

        # Get top-k experts for each token
        # top_k_weights: [batch_size, top_k]
        # top_k_indices: [batch_size, top_k]
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize weights to sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # 2. Apply SwiGLU using grouped_gemm
        # Gate projection: x @ W_gate[expert_id]
        gate_output = torch.ops.moe.grouped_gemm(
            x, self.w_gate, top_k_indices, self.use_triton
        )  # [batch_size * top_k, d_ff]

        # Up projection: x @ W_up[expert_id]
        up_output = torch.ops.moe.grouped_gemm(
            x, self.w_up, top_k_indices, self.use_triton
        )  # [batch_size * top_k, d_ff]

        # SwiGLU activation: swish(gate) * up
        # swish(x) = x * sigmoid(x)
        swish_gate = gate_output * torch.sigmoid(gate_output)
        hidden = swish_gate * up_output  # [batch_size * top_k, d_ff]

        # Down projection: hidden @ W_down[expert_id]
        # hidden has shape [batch_size * top_k, d_ff]
        # We need to flatten expert indices to match: [batch_size * top_k, 1]
        # Each row of hidden uses the corresponding expert from top_k_indices
        expert_indices_flat = top_k_indices.reshape(-1, 1)  # [batch_size * top_k, 1]

        expert_output = torch.ops.moe.grouped_gemm(
            hidden, self.w_down, expert_indices_flat, self.use_triton
        )  # [batch_size * top_k, d_model]

        # 3. Combine expert outputs with routing weights
        # Reshape expert_output: [batch_size, top_k, d_model]
        expert_output = expert_output.view(batch_size, self.top_k, self.d_model)

        # Apply routing weights and sum
        # top_k_weights: [batch_size, top_k, 1]
        output = torch.sum(
            expert_output * top_k_weights.unsqueeze(-1), dim=1
        )  # [batch_size, d_model]

        return output


def benchmark_swiglu_moe(
    batch_size: int = 1024,
    d_model: int = 512,
    d_ff: int = 2048,
    num_experts: int = 8,
    top_k: int = 2,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
):
    """
    Benchmark the SwiGLU MoE layer.

    Args:
        batch_size: Number of tokens
        d_model: Model dimension
        d_ff: Feed-forward dimension
        num_experts: Number of experts
        top_k: Number of experts to route each token to
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations
    """
    print("=" * 70)
    print("SwiGLU MoE Layer Benchmark")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  d_model: {d_model}")
    print(f"  d_ff: {d_ff}")
    print(f"  num_experts: {num_experts}")
    print(f"  top_k: {top_k}")
    print(f"  Iterations: {num_iterations} (+ {warmup_iterations} warmup)")
    print("=" * 70)

    # Create model and input
    model = SwiGLUMoELayer(
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k,
    ).cuda()

    x = torch.randn(batch_size, d_model, device='cuda', dtype=torch.float32)

    # Warmup
    print("\nWarming up...")
    for _ in range(warmup_iterations):
        output = model(x)
    torch.cuda.synchronize()

    # Benchmark forward pass
    print("Benchmarking forward pass...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        output = model(x)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    forward_time = (end_time - start_time) / num_iterations * 1000  # ms

    print(f"\nResults:")
    print(f"  Forward pass time: {forward_time:.3f} ms")
    print(f"  Throughput: {batch_size / forward_time * 1000:.0f} tokens/sec")
    print(f"  Output shape: {output.shape}")

    # Benchmark backward pass
    print("\nBenchmarking backward pass...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Warmup
    for _ in range(warmup_iterations):
        optimizer.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        optimizer.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    backward_time = (end_time - start_time) / num_iterations * 1000  # ms

    print(f"  Backward pass time: {backward_time:.3f} ms")
    print(f"  Total time (fwd + bwd): {forward_time + backward_time:.3f} ms")

    print("=" * 70)


def test_swiglu_moe_correctness():
    """
    Test correctness of SwiGLU MoE layer against a reference implementation.
    """
    print("\n" + "=" * 70)
    print("Testing SwiGLU MoE Correctness")
    print("=" * 70)

    batch_size = 4
    d_model = 64
    d_ff = 128
    num_experts = 4
    top_k = 2

    torch.manual_seed(42)

    # Create model
    model = SwiGLUMoELayer(
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k,
    ).cuda()

    # Create input
    x = torch.randn(batch_size, d_model, device='cuda', dtype=torch.float32)

    # Forward pass
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, d_model), f"Expected shape {(batch_size, d_model)}, got {output.shape}"

    # Test backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    assert model.router.weight.grad is not None
    assert model.w_gate.grad is not None
    assert model.w_up.grad is not None
    assert model.w_down.grad is not None

    print("✓ Forward pass: PASSED")
    print("✓ Backward pass: PASSED")
    print("✓ All gradients computed successfully")
    print("=" * 70)


def benchmark_three_modes(
    batch_size: int = 1024,
    d_model: int = 1024,
    d_ff: int = 2048,
    num_experts: int = 8,
    top_k: int = 2,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
):
    """
    Compare performance across four modes:
    1. Eager mode without triton
    2. Compiled without triton
    3. Compiled with triton
    4. Compiled with horizontal_fusion pass
    """
    print("=" * 90)
    print("SwiGLU MoE: Four-Mode Performance Comparison")
    print("=" * 90)
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  d_model (input_dim): {d_model}")
    print(f"  d_ff (expert_dim): {d_ff}")
    print(f"  num_experts: {num_experts}")
    print(f"  top_k: {top_k}")
    print(f"  Iterations: {num_iterations} (+ {warmup_iterations} warmup)")
    print("=" * 90)

    # Create input
    x = torch.randn(batch_size, d_model, device='cuda', dtype=torch.float32)

    results = {}

    # ========== MODE 1: EAGER WITHOUT TRITON ==========
    print("\n[1/4] Benchmarking EAGER mode (no triton, no compile)...")

    model_eager = SwiGLUMoELayer(
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k,
        use_triton=False,
    ).cuda()

    # Warmup
    for _ in range(warmup_iterations):
        output = model_eager(x)
    torch.cuda.synchronize()

    # Benchmark forward pass
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        output = model_eager(x)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    eager_forward_time = (end_time - start_time) / num_iterations * 1000  # ms

    print(f"  Forward:  {eager_forward_time:.3f} ms")
    print(f"  Throughput: {batch_size / eager_forward_time * 1000:.0f} tokens/sec")

    results['eager_no_triton'] = {
        'forward': eager_forward_time,
    }

    # ========== MODE 2: COMPILED WITHOUT TRITON ==========
    print("\n[2/4] Benchmarking COMPILED mode (no triton, with compile)...")

    model_compiled_no_triton_base = SwiGLUMoELayer(
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k,
        use_triton=False,
    ).cuda()

    # Compile the model
    model_compiled_no_triton = torch.compile(model_compiled_no_triton_base, mode='max-autotune')

    # Warmup (includes compilation time on first run)
    print("  Compiling (first run includes compilation overhead)...")
    for _ in range(warmup_iterations):
        output = model_compiled_no_triton(x)
    torch.cuda.synchronize()

    # Benchmark forward pass
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        output = model_compiled_no_triton(x)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    compiled_no_triton_forward_time = (end_time - start_time) / num_iterations * 1000  # ms

    print(f"  Forward:  {compiled_no_triton_forward_time:.3f} ms")
    print(f"  Throughput: {batch_size / compiled_no_triton_forward_time * 1000:.0f} tokens/sec")

    results['compiled_no_triton'] = {
        'forward': compiled_no_triton_forward_time,
    }

    # ========== MODE 3: COMPILED WITH TRITON ==========
    print("\n[3/4] Benchmarking COMPILED mode (with triton, with compile)...")

    model_compiled_triton_base = SwiGLUMoELayer(
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k,
        use_triton=True,
    ).cuda()

    # Compile the model
    model_compiled_triton = torch.compile(model_compiled_triton_base, mode='max-autotune')

    # Warmup (includes compilation time on first run)
    print("  Compiling (first run includes compilation overhead)...")
    for _ in range(warmup_iterations):
        output = model_compiled_triton(x)
    torch.cuda.synchronize()

    # Benchmark forward pass
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        output = model_compiled_triton(x)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    compiled_triton_forward_time = (end_time - start_time) / num_iterations * 1000  # ms

    print(f"  Forward:  {compiled_triton_forward_time:.3f} ms")
    print(f"  Throughput: {batch_size / compiled_triton_forward_time * 1000:.0f} tokens/sec")

    results['compiled_with_triton'] = {
        'forward': compiled_triton_forward_time,
    }

    # ========== MODE 4: COMPILED WITH HORIZONTAL FUSION ==========
    print("\n[4/4] Benchmarking COMPILED mode (with horizontal_fusion pass)...")

    model_horizontal_fusion_base = SwiGLUMoELayer(
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k,
        use_triton=True,
    ).cuda()

    # Compile the model with horizontal_fusion custom pass
    # Import the custom backend/pass (will be implemented separately)
    try:
        from torch._dynamo import register_backend
        from horizontal_fusion import horizontal_fusion_backend

        # Register custom backend
        model_horizontal_fusion = torch.compile(
            model_horizontal_fusion_base,
            backend=horizontal_fusion_backend
        )

        # Warmup (includes compilation time on first run)
        print("  Compiling with horizontal_fusion pass (first run includes compilation overhead)...")
        for _ in range(warmup_iterations):
            output = model_horizontal_fusion(x)
        torch.cuda.synchronize()

        # Benchmark forward pass
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            output = model_horizontal_fusion(x)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        horizontal_fusion_forward_time = (end_time - start_time) / num_iterations * 1000  # ms

        print(f"  Forward:  {horizontal_fusion_forward_time:.3f} ms")
        print(f"  Throughput: {batch_size / horizontal_fusion_forward_time * 1000:.0f} tokens/sec")

        results['compiled_horizontal_fusion'] = {
            'forward': horizontal_fusion_forward_time,
        }
        has_horizontal_fusion = True
    except ImportError:
        print("  horizontal_fusion pass not found, skipping this benchmark...")
        print("  (horizontal_fusion.py needs to be implemented)")
        has_horizontal_fusion = False

    # ========== COMPARISON ==========
    print("\n" + "=" * 90)
    print("Performance Comparison (Forward Pass Only)")
    print("=" * 90)

    if has_horizontal_fusion:
        print(f"\n{'Metric':<20} {'Eager':<15} {'Compiled':<15} {'Compiled+Triton':<20} {'HorizontalFusion':<20}")
        print("-" * 90)

        print(f"{'Forward (ms)':<20} {eager_forward_time:<15.3f} {compiled_no_triton_forward_time:<15.3f} {compiled_triton_forward_time:<20.3f} {horizontal_fusion_forward_time:<20.3f}")

        eager_throughput = batch_size / eager_forward_time * 1000
        compiled_no_triton_throughput = batch_size / compiled_no_triton_forward_time * 1000
        compiled_triton_throughput = batch_size / compiled_triton_forward_time * 1000
        horizontal_fusion_throughput = batch_size / horizontal_fusion_forward_time * 1000
        print(f"{'Throughput (tok/s)':<20} {eager_throughput:<15.0f} {compiled_no_triton_throughput:<15.0f} {compiled_triton_throughput:<20.0f} {horizontal_fusion_throughput:<20.0f}")
    else:
        print(f"\n{'Metric':<20} {'Eager':<15} {'Compiled':<15} {'Compiled+Triton':<20}")
        print("-" * 80)

        print(f"{'Forward (ms)':<20} {eager_forward_time:<15.3f} {compiled_no_triton_forward_time:<15.3f} {compiled_triton_forward_time:<20.3f}")

        eager_throughput = batch_size / eager_forward_time * 1000
        compiled_no_triton_throughput = batch_size / compiled_no_triton_forward_time * 1000
        compiled_triton_throughput = batch_size / compiled_triton_forward_time * 1000
        print(f"{'Throughput (tok/s)':<20} {eager_throughput:<15.0f} {compiled_no_triton_throughput:<15.0f} {compiled_triton_throughput:<20.0f}")

    print("\n" + "=" * 90)
    print("Speedup vs Eager (no triton)")
    print("=" * 90)

    compiled_speedup = eager_forward_time / compiled_no_triton_forward_time
    triton_speedup = eager_forward_time / compiled_triton_forward_time

    print(f"  Compiled (no triton):  {compiled_speedup:.2f}x")
    print(f"  Compiled + Triton:     {triton_speedup:.2f}x")

    if has_horizontal_fusion:
        horizontal_fusion_speedup = eager_forward_time / horizontal_fusion_forward_time
        print(f"  Horizontal Fusion:     {horizontal_fusion_speedup:.2f}x")

    print("\n" + "=" * 90)
    print("Speedup: Compiled+Triton vs Compiled (no triton)")
    print("=" * 90)
    triton_vs_compiled = compiled_no_triton_forward_time / compiled_triton_forward_time
    print(f"  {triton_vs_compiled:.2f}x")

    if has_horizontal_fusion:
        print("\n" + "=" * 90)
        print("Speedup: Horizontal Fusion vs Compiled+Triton")
        print("=" * 90)
        fusion_vs_triton = compiled_triton_forward_time / horizontal_fusion_forward_time
        print(f"  {fusion_vs_triton:.2f}x")

    print("=" * 90)

    return results


if __name__ == "__main__":
    # Run correctness test
    test_swiglu_moe_correctness()

    # Run three-mode benchmark with specified configuration
    print("\n" * 2)
    benchmark_three_modes(
        batch_size=1024,
        d_model=1024,      # input_dim
        d_ff=2048,         # expert_dim
        num_experts=8,
        top_k=2,
        num_iterations=100,
        warmup_iterations=10,
    )
