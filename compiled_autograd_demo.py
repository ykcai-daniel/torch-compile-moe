import torch
import torch.nn as nn
import torch.distributed as dist
import torch._dynamo
import os

# Enable compiled autograd
torch._dynamo.config.compiled_autograd = True

# Define a simple 2-layer MLP with allreduce
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 4)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Custom compiler to capture and print IR
def inspect_ir(gm, example_inputs):
    print("=" * 60)
    print("GRAPH IR:")
    print("=" * 60)
    print(gm.graph)
    print("\n" + "=" * 60)
    print("DETAILED NODES:")
    print("=" * 60)
    for node in gm.graph.nodes:
        print(f"Op: {node.op}, Target: {node.target}")
    print("=" * 60)
    
    # Your compiler pass goes here
    
    return gm.forward

# Initialize DDP
def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(42)

def cleanup():
    dist.destroy_process_group()

# Create model
model = MLP().cuda()
x = torch.randn(32, 10, requires_grad=True, device="cuda")
target = torch.randn(32, 4, device="cuda")

# Compile with custom inspector
@torch.compile(backend=inspect_ir)
def train(model, x, target):
    output = model(x)
    # dist.all_reduce(output)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()
    return loss

if __name__ == "__main__":
    setup()
    
    # Run to trigger compilation
    loss = train(model, x, target)
    print(f"\nLoss: {loss.item():.4f}")
    
    cleanup()