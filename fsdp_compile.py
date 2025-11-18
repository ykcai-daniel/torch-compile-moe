import argparse, os, time, torch, torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class TinyMLP(nn.Module):
    def __init__(self, d_in=1024, d_hidden=2048, d_out=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--fsdp", type=int, default=1)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--fullgraph", type=int, default=0)
    return ap.parse_args()

def setup_ddp():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

def wrap_model(model, use_fsdp):
    if not use_fsdp:
        return model
    return FSDP(model, use_orig_params=True)

def main():
    args = parse_args()
    setup_ddp()
    rank = dist.get_rank()
    device = torch.device("cuda", rank)

    torch.manual_seed(0)
    model = TinyMLP().to(device)
    model = wrap_model(model, bool(args.fsdp))

    if args.compile:
        model = torch.compile(
            model,
            fullgraph=bool(args.fullgraph),
            backend="inductor",
        )

    opt = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # synthetic data
    x = torch.randn(args.bs, 1024, device=device)
    y = torch.randint(0, 16, (args.bs,), device=device)

    torch.cuda.synchronize()
    start = time.time()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        if rank == 0:
            print(f"step {step} loss {loss.item():.4f}")
    torch.cuda.synchronize()
    elapsed = time.time() - start
    if rank == 0:
        print(f"Ran {args.steps} steps | FSDP={bool(args.fsdp)} | compile={bool(args.compile)} | fullgraph={bool(args.fullgraph)} | time={elapsed:.2f}s")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
