import torch
import torch.distributed as dist
import torch.nn as nn



def setup_distributed():
    """Initialize distributed environment for single-node multi-GPU"""
    if not dist.is_initialized():
        # For single node, multi-GPU setup
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    return rank, world_size


class AllToAllModule(nn.Module):
    """Simple module that uses all-to-all collective"""
    
    def __init__(self, dim_size, num_experts):
        super().__init__()
        self.dim_size = dim_size
        self.num_experts = num_experts
        self.linear = nn.Linear(dim_size, dim_size)
    
    def forward(self, x):
        # x shape: [batch, seq_len, dim]
        world_size = dist.get_world_size()
        
        # Apply linear transformation
        x = self.linear(x)
        
        # Prepare for all-to-all: split along batch dimension
        # Shape: [batch, seq_len, dim]
        split_size = x.size(0) // world_size
        input_list = list(x.split(split_size, dim=0))
        output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
        
        # Perform all-to-all
        dist.all_to_all(output_list, input_list)
        
        # Concatenate results
        output = torch.cat(output_list, dim=0)
        
        # Apply another transformation
        output = output * 2.0 + 1.0
        
        return output


def dump_ir_callback(name, ir_type="forward"):
    """Create a callback to dump IR"""
    counter = {"count": 0}
    
    def callback(gm, example_inputs):
        counter["count"] += 1
        filename = f"ir_{ir_type}_{name}_pass_{counter['count']}.txt"
        
        print(f"\n{'='*60}")
        print(f"Dumping {ir_type.upper()} IR - Pass {counter['count']}: {name}")
        print(f"Saving to: {filename}")
        print(f"{'='*60}")
        
        with open(filename, "w") as f:
            f.write(f"=== {ir_type.upper()} PASS - {name} ===\n\n")
            f.write(f"Graph:\n{gm.graph}\n\n")
            f.write(f"Code:\n{gm.code}\n\n")
            
            # Dump module structure
            f.write("Module Structure:\n")
            for node in gm.graph.nodes:
                f.write(f"  {node.op}: {node.name} -> {node.target}\n")
        
        print(f"IR dumped successfully to {filename}\n")
        return gm
    
    return callback


def custom_backend(gm, example_inputs):
    """Custom backend that dumps IR before compilation"""
    print("\n" + "="*60)
    print("CUSTOM BACKEND: Processing graph")
    print("="*60)
    
    # Dump the graph
    with open("ir_custom_backend.txt", "w") as f:
        f.write("=== CUSTOM BACKEND GRAPH ===\n\n")
        f.write(f"Graph:\n{gm.graph}\n\n")
        f.write(f"Code:\n{gm.code}\n\n")
    
    print("Graph dumped to ir_custom_backend.txt")
    
    # Use inductor as the actual backend
    return torch.compile(gm, backend="inductor")


def main():
    # Setup distributed
    rank, world_size = setup_distributed()
    device = f"cuda:{rank}"
    
    print(f"Rank {rank}/{world_size} initialized on {device}")
    
    # Create model
    batch_size = 8
    seq_len = 16
    dim_size = 64
    num_experts = 4
    
    model = AllToAllModule(dim_size, num_experts).to(device)
    
    # Create input
    x = torch.randn(batch_size, seq_len, dim_size, device=device, requires_grad=True)
    
    print(f"\nRank {rank}: Input shape: {x.shape}")
    
    # Method 1: Using torch._dynamo.optimize with custom backend
    print("\n" + "="*80)
    print("METHOD 1: Using torch._dynamo.optimize with IR dumping")
    print("="*80)
    
    # Register compilation callbacks for forward pass
    # torch._dynamo.config.post_grad_fusion = True
    
    # Compile with custom backend
    compiled_model_1 = torch.compile(
        model,
        backend="inductor",
        mode="default",
    )
    
    # Enable verbose mode to see compilation details
    torch._dynamo.config.verbose = True
    
    # Forward pass
    print(f"\nRank {rank}: Running FORWARD pass (Method 1)...")
    output_1 = compiled_model_1(x)
    
    # Backward pass
    print(f"\nRank {rank}: Running BACKWARD pass (Method 1)...")
    loss_1 = output_1.sum()
    loss_1.backward()
    
    print(f"Rank {rank}: Method 1 completed. Output shape: {output_1.shape}")
    
    # Method 2: Using AOTAutograd to explicitly capture forward and backward
    print("\n" + "="*80)
    print("METHOD 2: Explicit AOT Autograd IR dumping")
    print("="*80)
    
    # Reset gradients
    model.zero_grad()
    x_2 = torch.randn(batch_size, seq_len, dim_size, device=device, requires_grad=True)
    
    # Use explain to get detailed IR
    from torch._dynamo import explain
    
    explanation = explain(model)(x_2)
    
    if rank == 0:
        print("\n" + "="*60)
        print("EXPLANATION OUTPUT:")
        print("="*60)
        print(explanation)
        
        # Save explanation to file
        with open("ir_explanation.txt", "w") as f:
            f.write(str(explanation))
        print("Explanation saved to ir_explanation.txt")
    
    # Method 3: Manual IR extraction using export
    print("\n" + "="*80)
    print("METHOD 3: Using torch.export for IR inspection")
    print("="*80)
    
    try:
        from torch.export import export
        
        # Export the model
        print(f"\nRank {rank}: Exporting model...")
        exported_program = export(model, (x_2,))
        
        if rank == 0:
            # Save exported graph
            with open("ir_exported_graph.txt", "w") as f:
                f.write("=== EXPORTED GRAPH MODULE ===\n\n")
                f.write(str(exported_program.graph_module))
                f.write("\n\n=== GRAPH SIGNATURE ===\n\n")
                f.write(str(exported_program.graph_signature))
            
            print("Exported graph saved to ir_exported_graph.txt")
    except Exception as e:
        print(f"Export failed (expected with distributed ops): {e}")
    
    # Synchronize
    dist.barrier()
    
    if rank == 0:
        print("\n" + "="*80)
        print("SCRIPT COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nGenerated IR files:")
        print("  - ir_explanation.txt: Dynamo explanation output")
        print("  - ir_exported_graph.txt: Exported graph module (if available)")
        print("\nCheck the current directory for all generated IR files.")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run this script with torchrun:
    
    torchrun --nproc_per_node=2 script_name.py
    
    Or with multiple nodes:
    torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=<addr> --master_port=<port> script_name.py
    """
    
    # Check if distributed is available
    if not dist.is_available():
        print("ERROR: torch.distributed is not available")
        exit(1)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        exit(1)
    
    main()