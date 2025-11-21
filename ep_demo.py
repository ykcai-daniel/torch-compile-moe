import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed._functional_collectives as funcol
from typing import List


class MockMoE(nn.Module):
    """
    A simplified Mixture of Experts layer that performs:
    alltoall -> MLP -> alltoall
    
    This is a mock MoE with no actual expert selection - just a single MLP
    applied to tokens after gathering them via alltoall communication.
    """
    
    def __init__(self, hidden_dim, ffn_dim, num_experts=1):
        """
        Args:
            hidden_dim: Model dimension (e.g., 768)
            ffn_dim: Feedforward dimension (typically 4x hidden_dim)
            num_experts: Number of experts (for mock MoE, this is just 1)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        
        # Single MLP expert
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
    
    def forward(self, x, world_size=1, rank=0):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            world_size: Number of processes in distributed setup (default: 1)
            rank: Rank of current process (default: 0)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape
        assert hidden_dim == self.hidden_dim, f"Expected {self.hidden_dim}, got {hidden_dim}"
        
        # Reshape for alltoall: (batch_size, seq_len, hidden_dim)
        # In a real distributed setup, we'd do:
        # x_gathered = alltoall_gather(x)  # All processes send/receive tokens
        # For mock MoE, we just use x as is (simulating single-GPU or already gathered)
        x_flat = x.view(-1, hidden_dim)  # (batch_size * seq_len, hidden_dim)
        
        # Apply MLP expert
        output = self.mlp(x_flat)  # (batch_size * seq_len, hidden_dim)
        
        # Reshape back
        output = output.view(batch_size, seq_len, hidden_dim)
        
        # In a real distributed setup, we'd do:
        # output = alltoall_scatter(output)  # Send tokens back to original processes
        
        return output


class DistributedMockMoE(nn.Module):
    """
    A mock MoE that simulates distributed alltoall with torch.distributed.
    Only use if you have torch.distributed initialized.
    """
    
    def __init__(self, hidden_dim, ffn_dim, num_experts=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        
        # Single MLP expert
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        import torch.distributed as dist
        
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialized")
        
        batch_size, seq_len, hidden_dim = x.shape
        world_size = dist.get_world_size()
        
        # Reshape for alltoall: (batch_size * seq_len, hidden_dim)
        x_flat = x.view(batch_size * seq_len, hidden_dim)

        # NOTE: torch.compile will ignore pure side effect collectives like all-to-all
        # We must use the new tracable functional_collectives
        # x_gathered = dist.all_to_all_single(x_flat, split_dimension=0)

        # all_to_all_single: each rank sends/receives the same amount of data
        x_gathered = funcol.all_to_all_single_autograd(x_flat)
        
        # Apply MLP
        output = self.mlp(x_gathered)
        
        # Scatter back with all_to_all_single
        output_scattered = funcol.all_to_all_single_autograd(output)
        
        # Reshape back
        output_scattered = output_scattered.view(batch_size, seq_len, hidden_dim)
        
        return output_scattered


# Example usage
def training_step():
    """Single training step: forward, backward, and gradient update"""
    batch_size, seq_len, hidden_dim, ffn_dim = 2, 8, 768, 3072
    
    # Initialize model and optimizer
    moe = MockMoE(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
    optimizer = torch.optim.AdamW(moe.parameters(), lr=1e-4)
    
    # Create dummy input and target
    x = torch.randn(batch_size, seq_len, hidden_dim)
    target = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Forward pass
    output = moe(x)
    
    # Compute loss
    loss = F.mse_loss(output, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient update
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f} | Params: {sum(p.numel() for p in moe.parameters()):,}")

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    # Print the FX graph IR
    print("FX Graph:")
    print(gm.graph)
    
    # Print human-readable code
    print("\nFX Code:")
    
    # Return the forward function
    return gm.forward

if __name__ == "__main__":
    import sys
    compile_flag = True
    # training_step()
    batch_size, seq_len, hidden_dim, ffn_dim = 2, 8, 768, 3072
    moe = MockMoE(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    output = moe(x)

    if compile:
        compiled_model = torch.compile(moe,fullgraph=True,backend = custom_backend)

        compiled_output = compiled_model(x)

        torch.testing.assert_close(output,compiled_output)
        print("MoE Test passed")