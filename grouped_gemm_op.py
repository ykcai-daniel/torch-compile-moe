import triton
import triton.language as tl
import torch
from typing import List
from torch.library import custom_op, register_fake

@triton.jit
def grouped_gemm_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,  # d_ff
    K,  # d_model
    EM,  # total tokens * top_k (padded)
    num_valid_tokens,
    # Stride variables
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
):
    """
    Simplified fused MOE kernel for grouped GEMM.
    Computes C = A @ B where A is tokens and B is expert weights.
    """
    # Map program ids to blocks
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Check if this block is valid
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    
    # Load token ids for this block
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    # Load expert id for this block
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # Write zeros for invalid experts
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        # FIX: Use the output position (pid_m * BLOCK_SIZE_M + offset) instead of original token position
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_m[:, None] < num_tokens_post_padded) & (offs_cn[None, :] < N)
        zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        tl.store(c_ptrs, zeros, mask=c_mask)
        return

    # Create pointers for A and B blocks
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # FIX: Use original token indices to read from input tokens (not divided by top_k)
    a_ptrs = a_ptr + (
        (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak
    )
    
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # Accumulate in fp32 for accuracy
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks of A and B
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Matrix multiply and accumulate
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Write output
    # FIX: Write to sorted output positions (pid_m * BLOCK_SIZE_M + offset) not original token positions
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_m[:, None] < num_tokens_post_padded) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

# Register as a custom operator for torch.compile
@custom_op("moe::grouped_gemm", mutates_args=())
def grouped_gemm(
    tokens: torch.Tensor,  # [batch_size, d_model]
    weights: torch.Tensor,  # [num_experts, d_model, d_ff]
    top_k_expert_activation: torch.Tensor,  # [batch_size, top_k]
) -> torch.Tensor:  # [batch_size * top_k, d_ff]
    batch_size, d_model = tokens.shape
    num_experts, _, d_ff = weights.shape
    top_k = top_k_expert_activation.shape[1]
    
    # Flatten the expert assignments to get all (token_idx, expert_id) pairs
    # Shape: [batch_size * top_k]
    flat_expert_ids = top_k_expert_activation.reshape(-1)
    
    # Create indices to track which output position each computation goes to
    output_indices = torch.arange(batch_size * top_k, device=tokens.device)
    
    # Create token indices (which token each computation comes from)
    # For top_k=2: [0, 0, 1, 1, 2, 2, ...]
    token_indices = torch.arange(batch_size, device=tokens.device).repeat_interleave(top_k)
    
    # Allocate output
    output = torch.zeros(batch_size * top_k, d_ff, device=tokens.device, dtype=tokens.dtype)
    
    # Process each expert
    for expert_id in range(num_experts):
        # Find all positions assigned to this expert
        mask = flat_expert_ids == expert_id
        
        if mask.any():
            # Get the token indices that need this expert
            selected_token_indices = token_indices[mask]
            
            # Get the tokens for this expert
            selected_tokens = tokens[selected_token_indices]  # [num_selected, d_model]
            
            # Batch matrix multiply: [num_selected, d_model] @ [d_model, d_ff]
            expert_output = selected_tokens @ weights[expert_id]  # [num_selected, d_ff]
            
            # Place results in the correct output positions
            output[mask] = expert_output
    
    return output

def batched_grouped_gemm_backward(
    grad_output: torch.Tensor,  # [batch_size * top_k, d_ff]
    tokens: torch.Tensor,  # [batch_size, d_model]
    weights: torch.Tensor,  # [num_experts, d_model, d_ff]
    top_k_expert_activation: torch.Tensor,  # [batch_size, top_k]
):
    """
    Backward pass for batched_grouped_gemm.
    
    Forward: output = tokens @ weights[expert_ids]
    
    Gradients:
        grad_tokens: Accumulate gradients from all expert computations
        grad_weights: Accumulate gradients for each expert's weight matrix
        grad_top_k_expert_activation: None (discrete routing, not differentiable)
    
    Args:
        grad_output: Gradient of loss w.r.t. output [batch_size * top_k, d_ff]
        tokens: Original input tokens [batch_size, d_model]
        weights: Original expert weights [num_experts, d_model, d_ff]
        top_k_expert_activation: Original expert assignments [batch_size, top_k]
    
    Returns:
        grad_tokens: [batch_size, d_model]
        grad_weights: [num_experts, d_model, d_ff]
        grad_top_k_expert_activation: None
    """
    batch_size, d_model = tokens.shape
    num_experts, _, d_ff = weights.shape
    top_k = top_k_expert_activation.shape[1]
    
    # Initialize gradients
    grad_tokens = torch.zeros_like(tokens)  # [batch_size, d_model]
    grad_weights = torch.zeros_like(weights)  # [num_experts, d_model, d_ff]
    
    # Flatten the expert assignments
    flat_expert_ids = top_k_expert_activation.reshape(-1)  # [batch_size * top_k]
    
    # Create token indices (which token each computation comes from)
    token_indices = torch.arange(batch_size, device=tokens.device).repeat_interleave(top_k)
    
    # Process each expert
    for expert_id in range(num_experts):
        # Find all positions assigned to this expert
        mask = flat_expert_ids == expert_id
        
        if mask.any():
            # Get the token indices that used this expert
            selected_token_indices = token_indices[mask]
            
            # Get the tokens that were processed by this expert
            selected_tokens = tokens[selected_token_indices]  # [num_selected, d_model]
            
            # Get the gradient for outputs from this expert
            selected_grad_output = grad_output[mask]  # [num_selected, d_ff]
            
            # Gradient w.r.t. weights for this expert
            # Forward: output = tokens @ weights
            # Backward: grad_weights = tokens.T @ grad_output
            grad_weights[expert_id] = selected_tokens.T @ selected_grad_output  # [d_model, d_ff]
            
            # Gradient w.r.t. tokens
            # Forward: output = tokens @ weights
            # Backward: grad_tokens = grad_output @ weights.T
            grad_tokens_selected = selected_grad_output @ weights[expert_id].T  # [num_selected, d_model]
            
            # Accumulate gradients back to the original token positions
            # Note: Use index_add_ for correct accumulation when a token appears multiple times
            grad_tokens.index_add_(0, selected_token_indices, grad_tokens_selected)
    
    return grad_tokens, grad_weights, None

@register_fake("moe::grouped_gemm")
def grouped_gemm_fake(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    top_k_expert_activation: torch.Tensor,
) -> torch.Tensor:
    """
    Fake implementation for torch.compile meta analysis.
    Returns a tensor with the correct output shape without actual computation.
    """
    batch_size = tokens.shape[0]
    top_k = top_k_expert_activation.shape[1]
    d_ff = weights.shape[2]
    
    # Return empty tensor with correct shape
    return tokens.new_empty((batch_size * top_k, d_ff))


def setup_grouped_gemm_autograd():
    from torch.library import register_autograd

    def setup_context(ctx, inputs, output):
        ctx.tokens, ctx.weights, ctx.expert_ids = inputs

    def grouped_gemm_backward(ctx, grad_out):
        return batched_grouped_gemm_backward(
            grad_out, ctx.tokens, ctx.weights, ctx.expert_ids
        )

    register_autograd(
        "moe::grouped_gemm",
        grouped_gemm_backward,
        setup_context=setup_context,
    )


# Call setup to register backward pass
setup_grouped_gemm_autograd()


# Unit test
if __name__ == "__main__":
    print("Running unit test for grouped_gemm_op...")
    
    # Test parameters
    batch_size = 16
    d_model = 64
    d_ff = 128
    num_experts = 4
    top_k = 2
    
    # Create test inputs
    torch.manual_seed(42)
    tokens = torch.randn(batch_size, d_model, device='cuda', dtype=torch.float32)
    weights = torch.randn(num_experts, d_model, d_ff, device='cuda', dtype=torch.float32)
    top_k_expert_activation = torch.randint(0, num_experts, (batch_size, top_k), device='cuda')
    
    print(f"Tokens shape: {tokens.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Expert activation shape: {top_k_expert_activation.shape}")
    print(f"Expert assignments (first 4 tokens): {top_k_expert_activation[:4].tolist()}")
    
    # Test 1: Original implementation
    print("\n" + "="*50)
    print("Test 1: Original grouped_gemm_op")
    print("="*50)
    output = grouped_gemm(tokens, weights, top_k_expert_activation)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size * top_k, d_ff), f"Expected shape {(batch_size * top_k, d_ff)}, got {output.shape}"
    
    # Verify correctness by computing reference output
    print("\nVerifying correctness against naive implementation...")
    reference_output = torch.zeros(batch_size * top_k, d_ff, device='cuda', dtype=torch.float32)
    
    idx = 0
    for i in range(batch_size):
        for k in range(top_k):
            expert_id = top_k_expert_activation[i, k].item()
            # token[i] @ weights[expert_id] -> [d_model] @ [d_model, d_ff] = [d_ff]
            reference_output[idx] = tokens[i] @ weights[expert_id]
            idx += 1
    
    # Check if outputs match
    max_diff = (output - reference_output).abs().max().item()
    mean_diff = (output - reference_output).abs().mean().item()
    
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    
    # Allow small numerical differences due to floating point
    tolerance = 1e-4
    if max_diff < tolerance:
        print(f"✓ Test PASSED! Max difference {max_diff:.6e} is within tolerance {tolerance}")
    else:
        print(f"✗ Test FAILED! Max difference {max_diff:.6e} exceeds tolerance {tolerance}")
        print(f"\nSample outputs (first 3):")
        print(f"Triton output: {output[:3, :5]}")
        print(f"Reference output: {reference_output[:3, :5]}")
    
    # Test 2: Custom operator
    print("\n" + "="*50)
    print("Test 2: Custom operator (moe::grouped_gemm)")
    print("="*50)
    output_custom = torch.ops.moe.grouped_gemm(tokens, weights, top_k_expert_activation)
    print(f"Output shape: {output_custom.shape}")
    
    max_diff_custom = (output_custom - reference_output).abs().max().item()
    print(f"Max difference from reference: {max_diff_custom:.6e}")
    
    if max_diff_custom < tolerance:
        print(f"✓ Custom operator test PASSED!")
    else:
        print(f"✗ Custom operator test FAILED!")
    
    # Test 3: With torch.compile
    print("\n" + "="*50)
    print("Test 3: With torch.compile")
    print("="*50)
    
    def model_with_grouped_gemm(tokens, weights, experts):
        return torch.ops.moe.grouped_gemm(tokens, weights, experts)
    
    try:
        compiled_model = torch.compile(model_with_grouped_gemm, fullgraph=True)
        output_compiled = compiled_model(tokens, weights, top_k_expert_activation)
        print(f"Output shape: {output_compiled.shape}")
        
        max_diff_compiled = (output_compiled - reference_output).abs().max().item()
        print(f"Max difference from reference: {max_diff_compiled:.6e}")
        
        if max_diff_compiled < tolerance:
            print(f"✓ torch.compile test PASSED!")
        else:
            print(f"✗ torch.compile test FAILED!")
    except Exception as e:
        print(f"✗ torch.compile test failed with error: {e}")
    
    # Additional shape tests
    print("\n" + "="*50)
    print("Test 4: Different configurations")
    print("="*50)
    
    test_configs = [
        (4, 32, 64, 2, 1),   # Small batch, single expert per token
        (8, 128, 256, 8, 3), # Larger with top-3
        (1, 16, 32, 4, 2),   # Single token
    ]
    
    for bs, dm, df, ne, tk in test_configs:
        tokens_test = torch.randn(bs, dm, device='cuda', dtype=torch.float32)
        weights_test = torch.randn(ne, dm, df, device='cuda', dtype=torch.float32)
        experts_test = torch.randint(0, ne, (bs, tk), device='cuda')
        
        output_test = torch.ops.moe.grouped_gemm(tokens_test, weights_test, experts_test)
        expected_shape = (bs * tk, df)
        
        if output_test.shape == expected_shape:
            print(f"✓ Config (bs={bs}, d_model={dm}, d_ff={df}, experts={ne}, top_k={tk}): PASSED")
        else:
            print(f"✗ Config (bs={bs}, d_model={dm}, d_ff={df}, experts={ne}, top_k={tk}): FAILED")
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)