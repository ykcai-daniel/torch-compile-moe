import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Tuple, List, Dict
import json
import os

os.environ["TORCH_LOGS"] = "recompiles" # Enable torch dynamo logs

class Expert(nn.Module):
    """Single expert network"""
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class MoELayer(nn.Module):
    """Mixture of Experts Layer"""
    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router/gating network
        self.gate = nn.Linear(hidden_dim, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]
        
        # Router logits
        router_logits = self.gate(x_flat)  # [batch_size * seq_len, num_experts]
        
        # Get top-k experts
        router_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # Both: [batch_size * seq_len, top_k]
        router_weights = F.softmax(router_weights, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == i).any(dim=-1)  # [batch_size * seq_len]
            if not expert_mask.any():
                continue
            
            # Get tokens for this expert
            expert_input = x_flat[expert_mask]
            expert_output = expert(expert_input)
            
            # Get weights for this expert
            expert_weights = torch.zeros(expert_mask.sum(), device=x.device)
            for k in range(self.top_k):
                mask_k = selected_experts[expert_mask, k] == i
                expert_weights[mask_k] = router_weights[expert_mask, k][mask_k]
            
            # Add weighted output
            output[expert_mask] += expert_output * expert_weights.unsqueeze(-1)
        
        return output.view(batch_size, seq_len, hidden_dim)

class MoELayerOptimized(nn.Module):
    """Optimized MoE implementation for better compilation"""
    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.gate = nn.Linear(hidden_dim, num_experts)
        
        # Batched expert weights for parallel computation
        self.w1 = nn.Parameter(torch.randn(num_experts, hidden_dim, ffn_dim))
        self.w2 = nn.Parameter(torch.randn(num_experts, ffn_dim, hidden_dim))
        self.activation = nn.GELU()
    
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        
        # Router
        router_logits = self.gate(x_flat)
        router_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        router_weights = F.softmax(router_weights, dim=-1)
        
        # Batch expert computation
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = selected_experts[:, k]
            weights = router_weights[:, k].unsqueeze(-1)
            
            # Gather expert weights
            w1_selected = self.w1[expert_idx]  # [tokens, hidden_dim, ffn_dim]
            w2_selected = self.w2[expert_idx]  # [tokens, ffn_dim, hidden_dim]
            
            # Apply expert
            h = torch.bmm(x_flat.unsqueeze(1), w1_selected).squeeze(1)
            h = self.activation(h)
            expert_out = torch.bmm(h.unsqueeze(1), w2_selected).squeeze(1)
            
            output += expert_out * weights
        
        return output.view(batch_size, seq_len, hidden_dim)

def benchmark_model(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = "cuda"
) -> Dict[str, float]:
    """Benchmark a model with given configuration"""
    model = model.to(device)
    model.eval()
    
    # Generate random input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark forward pass
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            output = model(x)
            
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    # Calculate memory usage
    if device == "cuda":
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        torch.cuda.reset_peak_memory_stats()
    else:
        memory_allocated = 0
    
    return {
        "mean_time_ms": np.mean(times),
        "std_time_ms": np.std(times),
        "median_time_ms": np.median(times),
        "min_time_ms": np.min(times),
        "max_time_ms": np.max(times),
        "throughput_tokens_per_sec": (batch_size * seq_len * num_iterations) / (np.sum(times) / 1000),
        "memory_gb": memory_allocated
    }

def run_comprehensive_benchmark():
    """Run comprehensive benchmarks across different configurations"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("\n" + "="*80 + "\n")
    
    # Configuration
    hidden_dim = 1024
    ffn_dim = 4096
    num_experts = 64
    top_k = 4
    
    # Test configurations
    configs = [
        {"batch_size": 1, "seq_len": 128, "name": "Small (B=1, S=128)"},
        {"batch_size": 4, "seq_len": 512, "name": "Medium (B=4, S=512)"},
        {"batch_size": 8, "seq_len": 1024, "name": "Large (B=8, S=1024)"},
        {"batch_size": 16, "seq_len": 1024, "name": "XLarge (B=16, S=1024)"},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Configuration: {config['name']}")
        print(f"{'='*80}\n")
        
        batch_size = config["batch_size"]
        seq_len = config["seq_len"]
        total_tokens = batch_size * seq_len
        
        print(f"Total tokens: {total_tokens:,}")
        print(f"Parameters: hidden_dim={hidden_dim}, ffn_dim={ffn_dim}, "
              f"num_experts={num_experts}, top_k={top_k}\n")
        
        config_results = {}
        
        # Test 1: Baseline (no compilation)
        print("1. Baseline MoE (no compilation)...")
        try:
            model_baseline = MoELayer(hidden_dim, ffn_dim, num_experts, top_k)
            metrics = benchmark_model(model_baseline, batch_size, seq_len, hidden_dim, device=device)
            config_results["baseline"] = metrics
            print(f"   Mean time: {metrics['mean_time_ms']:.2f} ± {metrics['std_time_ms']:.2f} ms")
            print(f"   Throughput: {metrics['throughput_tokens_per_sec']:.0f} tokens/sec")
            if device == "cuda":
                print(f"   Memory: {metrics['memory_gb']:.2f} GB")
            del model_baseline
        except Exception as e:
            print(f"   Failed: {e}")
            config_results["baseline"] = None
        
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Test 2: torch.compile with default settings
        print("\n2. torch.compile (default) with dynamic ...")
        try:
            model_compiled = MoELayer(hidden_dim, ffn_dim, num_experts, top_k)
            model_compiled = torch.compile(model_compiled,dynamic=True)
            metrics = benchmark_model(model_compiled, batch_size, seq_len, hidden_dim, device=device)
            config_results["compiled_default"] = metrics
            print(f"   Mean time: {metrics['mean_time_ms']:.2f} ± {metrics['std_time_ms']:.2f} ms")
            print(f"   Throughput: {metrics['throughput_tokens_per_sec']:.0f} tokens/sec")
            if device == "cuda":
                print(f"   Memory: {metrics['memory_gb']:.2f} GB")
            
            if config_results["baseline"]:
                speedup = config_results["baseline"]["mean_time_ms"] / metrics["mean_time_ms"]
                print(f"   Speedup vs baseline: {speedup:.2f}x")
            del model_compiled
        except Exception as e:
            print(f"   Failed: {e}")
            config_results["compiled_default"] = None
        
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Test 3: torch.compile with reduce-overhead mode
        print("\n3. torch.compile (reduce-overhead)...")
        try:
            model_compiled_ro = MoELayer(hidden_dim, ffn_dim, num_experts, top_k)
            model_compiled_ro = torch.compile(model_compiled_ro, mode="reduce-overhead")
            metrics = benchmark_model(model_compiled_ro, batch_size, seq_len, hidden_dim, device=device)
            config_results["compiled_reduce_overhead"] = metrics
            print(f"   Mean time: {metrics['mean_time_ms']:.2f} ± {metrics['std_time_ms']:.2f} ms")
            print(f"   Throughput: {metrics['throughput_tokens_per_sec']:.0f} tokens/sec")
            if device == "cuda":
                print(f"   Memory: {metrics['memory_gb']:.2f} GB")
            
            if config_results["baseline"]:
                speedup = config_results["baseline"]["mean_time_ms"] / metrics["mean_time_ms"]
                print(f"   Speedup vs baseline: {speedup:.2f}x")
            del model_compiled_ro
        except Exception as e:
            print(f"   Failed: {e}")
            config_results["compiled_reduce_overhead"] = None
        
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Test 4: Optimized MoE with compilation
        print("\n4. Optimized MoE + torch.compile (default)...")
        try:
            model_opt = MoELayerOptimized(hidden_dim, ffn_dim, num_experts, top_k)
            model_opt = torch.compile(model_opt)
            metrics = benchmark_model(model_opt, batch_size, seq_len, hidden_dim, device=device)
            config_results["optimized_compiled"] = metrics
            print(f"   Mean time: {metrics['mean_time_ms']:.2f} ± {metrics['std_time_ms']:.2f} ms")
            print(f"   Throughput: {metrics['throughput_tokens_per_sec']:.0f} tokens/sec")
            if device == "cuda":
                print(f"   Memory: {metrics['memory_gb']:.2f} GB")
            
            if config_results["baseline"]:
                speedup = config_results["baseline"]["mean_time_ms"] / metrics["mean_time_ms"]
                print(f"   Speedup vs baseline: {speedup:.2f}x")
            del model_opt
        except Exception as e:
            print(f"   Failed: {e}")
            config_results["optimized_compiled"] = None
        
        if device == "cuda":
            torch.cuda.empty_cache()
        
        results[config["name"]] = config_results
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    
    for config_name, config_results in results.items():
        print(f"{config_name}:")
        if config_results.get("baseline"):
            baseline_time = config_results["baseline"]["mean_time_ms"]
            print(f"  Baseline: {baseline_time:.2f} ms")
            
            for variant in ["compiled_default", "compiled_reduce_overhead", "optimized_compiled"]:
                if config_results.get(variant):
                    time_ms = config_results[variant]["mean_time_ms"]
                    speedup = baseline_time / time_ms
                    print(f"  {variant}: {time_ms:.2f} ms ({speedup:.2f}x)")
        print()
    
    # Save results to JSON
    with open("moe_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to: moe_benchmark_results.json")

import torch
import torch.fx as fx
from torch._inductor import config as inductor_config
from torch._inductor.compile_fx import compile_fx
import copy

def compile_model(model: torch.nn.Module, example_input: torch.Tensor):
    """
    Compile a model and dump FX graphs before and after optimization passes.
    
    Args:
        model: PyTorch model to compile
        example_input: Example input tensor for tracing
    
    Returns:
        Compiled model
    """
    print("=" * 80)
    print("COMPILING MODEL")
    print("=" * 80)
    
    # Store graphs at different stages
    graphs = {}
    
    def capturing_backend(gm: fx.GraphModule, example_inputs):
        """Custom backend that captures graphs before optimization"""
        print("\n" + "=" * 80)
        print("INITIAL FX GRAPH (Before Optimizations)")
        print("=" * 80)
        print(gm.graph)
        print("\n--- Generated Python Code ---")
        print(gm.code)
        
        # Store the initial graph
        graphs['initial'] = copy.deepcopy(gm)
        
        # Now apply TorchInductor's optimization passes
        print("\n" + "=" * 80)
        print("APPLYING OPTIMIZATION PASSES...")
        print("=" * 80)
        
        # Use inductor's compilation pipeline
        from torch._inductor.decomposition import decompositions
        from torch._functorch.aot_autograd import aot_module_simplified
        
        # AOT Autograd performs lowering and optimization
        def fw_compiler(gm: fx.GraphModule, example_inputs):
            print("\n" + "=" * 80)
            print("OPTIMIZED FX GRAPH (After Passes)")
            print("=" * 80)
            print(gm.graph)
            print("\n--- Optimized Python Code ---")
            print(gm.code)
            
            graphs['optimized'] = copy.deepcopy(gm)
            
            # Continue with actual compilation
            return compile_fx(gm, example_inputs)
        
        # Apply decompositions and optimizations
        optimized = aot_module_simplified(
            gm,
            example_inputs,
            fw_compiler=fw_compiler
        )
        
        return optimized
    
    # Compile with our custom backend
    compiled_model = torch.compile(model, backend=capturing_backend,dynamic=True)
    
    # Trigger compilation by running forward pass
    print("\n" + "=" * 80)
    print("TRIGGERING COMPILATION (First Forward Pass)")
    print("=" * 80)
    
    with torch.no_grad():
        output = compiled_model(example_input)
    
    print("\n" + "=" * 80)
    print("COMPILATION COMPLETE")
    print("=" * 80)
    print(f"Output shape: {output.shape}")
    
    # Optionally save graphs to files
    if graphs:
        print("\n--- Saving graphs to files ---")
        if 'initial' in graphs:
            with open("initial_graph.txt", "w") as f:
                f.write(str(graphs['initial'].graph))
                f.write("\n\n=== CODE ===\n\n")
                f.write(graphs['initial'].code)
            print("✓ Saved initial_graph.txt")
        
        if 'optimized' in graphs:
            with open("optimized_graph.txt", "w") as f:
                f.write(str(graphs['optimized'].graph))
                f.write("\n\n=== CODE ===\n\n")
                f.write(graphs['optimized'].code)
            print("✓ Saved optimized_graph.txt")
    
    return compiled_model

if __name__ == "__main__":
    # run_comprehensive_benchmark()
    
    hidden_dim = 1024
    ffn_dim = 512
    num_expert = 16
    top_k = 4
    moe_model_basic = MoELayer(hidden_dim, ffn_dim, num_expert, top_k)
    batch_size = 10
    sec_len = 40
    example_tensor = torch.rand(batch_size,sec_len, hidden_dim)

    compile_model(moe_model_basic,example_tensor)