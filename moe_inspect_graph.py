import torch
import torch.fx as fx
from torch._decomp import core_aten_decompositions
from torch.fx.experimental.proxy_tensor import make_fx
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor

def inspect_decomposed_graph_correct(model: torch.nn.Module, example_input):
    """
    Properly trace with FakeTensorMode to handle model parameters.
    """
    print("="*80)
    print("HIGH-LEVEL GRAPH")
    print("="*80)
    
    # Create FakeTensorMode context
    fake_mode = FakeTensorMode()
    
    with fake_mode:
        # Convert example input to fake tensor
        fake_input = fake_mode.from_tensor(example_input)
        
        # Trace the model
        traced = make_fx(model, tracing_mode='symbolic')(fake_input)
    
    print(traced.graph)
    
    print("\n" + "="*80)
    print("DECOMPOSED GRAPH (ATen primitives)")
    print("="*80)
    
    with fake_mode:
        fake_input = fake_mode.from_tensor(example_input)
        
        decomposed = make_fx(
            model,
            tracing_mode='symbolic',
            decomposition_table=core_aten_decompositions()
        )(fake_input)
    
    print(decomposed.graph)
    
    return traced, decomposed


def inspect_moe_operations_correct(model: torch.nn.Module, example_input):
    """
    Find MoE operations with proper FakeTensor handling.
    """
    print("="*80)
    print("TRACING MOE MODEL WITH FAKETENSORS")
    print("="*80)
    
    fake_mode = FakeTensorMode()
    
    with fake_mode:
        fake_input = fake_mode.from_tensor(example_input)
        
        decomposed = make_fx(
            model,
            tracing_mode='symbolic',
            decomposition_table=core_aten_decompositions()
        )(fake_input)
    
    print(decomposed.graph)
    
    print("\n" + "="*80)
    print("MOE-RELEVANT OPERATIONS")
    print("="*80)
    
    moe_keywords = {
        'topk': 'Expert Selection',
        'gather': 'Expert Routing',
        'scatter': 'Expert Routing',
        'softmax': 'Gate Weighting',
        'mm': 'Matrix Multiply',
        'addmm': 'Linear Layer',
        'bmm': 'Batched MatMul',
        'stack': 'Tensor Stacking',
        'cat': 'Concatenation',
        'index': 'Indexing',
        'expand': 'Broadcasting'
    }
    
    found_ops = {}
    
    for node in decomposed.graph.nodes:
        if node.op == 'call_function':
            target_str = str(node.target).lower()
            
            for keyword, description in moe_keywords.items():
                if keyword in target_str:
                    if description not in found_ops:
                        found_ops[description] = []
                    
                    node_info = {
                        'name': node.name,
                        'target': str(node.target),
                        'shape': None
                    }
                    
                    if 'val' in node.meta and hasattr(node.meta['val'], 'shape'):
                        node_info['shape'] = tuple(node.meta['val'].shape)
                    
                    found_ops[description].append(node_info)
                    break
    
    for op_type, nodes in sorted(found_ops.items()):
        print(f"\n{op_type}: ({len(nodes)} occurrences)")
        for node_info in nodes[:3]:  # Show first 3
            print(f"  - {node_info['name']}: {node_info['target']}")
            if node_info['shape']:
                print(f"    Shape: {node_info['shape']}")
    
    # Statistics
    print("\n" + "="*80)
    print("OPERATION COUNTS")
    print("="*80)
    
    op_counts = {}
    for node in decomposed.graph.nodes:
        if node.op == 'call_function':
            op_name = str(node.target).split('.')[-2] if '.' in str(node.target) else str(node.target)
            op_counts[op_name] = op_counts.get(op_name, 0) + 1
    
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {op}: {count}x")
    
    return decomposed


def find_batchable_matmuls_in_graph(gm: fx.GraphModule):
    """
    Analyze decomposed graph to find matmuls that can be batched.
    """
    print("\n" + "="*80)
    print("ANALYZING MATMUL OPERATIONS FOR BATCHING")
    print("="*80)
    
    matmuls = []
    
    for node in gm.graph.nodes:
        if node.op == 'call_function':
            target_str = str(node.target)
            
            # Look for matmul operations
            if any(op in target_str for op in ['aten.mm', 'aten.addmm', 'aten.bmm', 'aten.matmul']):
                matmul_info = {
                    'node': node,
                    'target': target_str,
                    'name': node.name,
                    'input_shapes': [],
                    'output_shape': None
                }
                
                # Get input shapes
                for arg in node.args:
                    if isinstance(arg, fx.Node) and 'val' in arg.meta:
                        if hasattr(arg.meta['val'], 'shape'):
                            matmul_info['input_shapes'].append(tuple(arg.meta['val'].shape))
                
                # Get output shape
                if 'val' in node.meta and hasattr(node.meta['val'], 'shape'):
                    matmul_info['output_shape'] = tuple(node.meta['val'].shape)
                
                matmuls.append(matmul_info)
    
    print(f"Found {len(matmuls)} matmul operations:\n")
    
    # Group by shapes for batching analysis
    from collections import defaultdict
    shape_groups = defaultdict(list)
    
    for matmul in matmuls:
        if len(matmul['input_shapes']) >= 2:
            # Create signature for grouping
            lhs_shape = matmul['input_shapes'][0]
            rhs_shape = matmul['input_shapes'][1]
            
            # For 2D matmuls
            if len(lhs_shape) == 2 and len(rhs_shape) == 2:
                signature = (lhs_shape[0], lhs_shape[1], rhs_shape[1])  # (M, K, N)
                shape_groups[signature].append(matmul)
    
    print("Matmuls grouped by shape signature (M, K, N):")
    for signature, group in shape_groups.items():
        print(f"\n  Shape {signature}: {len(group)} matmuls")
        for matmul in group:
            print(f"    - {matmul['name']}: {matmul['target']}")
            print(f"      Inputs: {matmul['input_shapes']}")
            print(f"      Output: {matmul['output_shape']}")
        
        if len(group) >= 2:
            print(f"    ✓ Can batch {len(group)} matmuls together!")
    
    return matmuls, shape_groups


# Complete workflow for your use case
def complete_moe_analysis(model: torch.nn.Module, example_input):
    """
    Complete analysis pipeline for MoE model.
    """
    print("="*80)
    print("COMPLETE MOE MODEL ANALYSIS")
    print("="*80)
    
    # Step 1: Trace and decompose
    fake_mode = FakeTensorMode()
    
    with fake_mode:
        fake_input = fake_mode.from_tensor(example_input)
        
        print("\nStep 1: Creating decomposed graph...")
        decomposed = make_fx(
            model,
            tracing_mode='symbolic',
            decomposition_table=core_aten_decompositions()
        )(fake_input)
    
    # Step 2: Find MoE operations
    print("\nStep 2: Identifying MoE operations...")
    inspect_moe_operations_correct(model, example_input)
    
    # Step 3: Analyze matmuls for batching
    print("\nStep 3: Analyzing matmul batching opportunities...")
    matmuls, shape_groups = find_batchable_matmuls_in_graph(decomposed)
    
    # Step 4: Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total operations in graph: {len(list(decomposed.graph.nodes))}")
    print(f"Total matmul operations: {len(matmuls)}")
    print(f"Unique matmul shape patterns: {len(shape_groups)}")
    
    batchable_count = sum(1 for group in shape_groups.values() if len(group) >= 2)
    print(f"Batchable matmul groups: {batchable_count}")
    
    return decomposed, matmuls, shape_groups


# Vectorized MoE model (same as before)
class VectorizedMoE(torch.nn.Module):
    def __init__(self, hidden_dim=128, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = torch.nn.Linear(hidden_dim, num_experts)
        self.experts = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        batch_size, hidden_dim = x.shape
        
        # Gate scoring
        gate_logits = self.gate(x)
        
        # TopK selection
        top_k_weights, top_k_indices = torch.topk(gate_logits, k=self.top_k, dim=-1)
        top_k_weights = torch.softmax(top_k_weights, dim=-1)
        
        # All expert outputs
        expert_outputs = torch.stack([
            expert(x) for expert in self.experts
        ], dim=1)
        
        # Gather selected experts
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(
            batch_size, self.top_k, hidden_dim
        )
        
        selected_expert_outputs = torch.gather(
            expert_outputs, 
            dim=1, 
            index=top_k_indices_expanded
        )
        
        # Weight and combine
        weighted_outputs = top_k_weights.unsqueeze(-1) * selected_expert_outputs
        output = weighted_outputs.sum(dim=1)
        
        return output


if __name__ == "__main__":
    print("Testing MoE Analysis with FakeTensors")
    print("="*80 + "\n")
    
    model = VectorizedMoE(hidden_dim=128, num_experts=4, top_k=2)
    example_input = torch.randn(32, 128)
    
    # Run complete analysis
    decomposed, matmuls, shape_groups = complete_moe_analysis(model, example_input)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)