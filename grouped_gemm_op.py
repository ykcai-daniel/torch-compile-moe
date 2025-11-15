import torch
import torch.fx as fx
from typing import Dict, Tuple, List, Optional
import operator

# --- 1. THE CUSTOM FUSED OPERATOR (STANDARD PYTHON FUNCTION) ---

def custom_grouped_gemm_impl(a: torch.Tensor, b_group: torch.Tensor, num_groups: int) -> torch.Tensor:
    """
    This function represents your custom, fused Triton kernel.
    
    In a real application, this is where you would call your triton.kernel.launch().
    For this demo, it uses standard PyTorch ops as a runnable fallback/simulation
    to ensure the shape and output are correct for compilation.
    
    NOTE: Since we couldn't use torch.library, the compiler relies on symbolically 
    tracing this Python fallback for shape inference.
    """
    
    assert a.size(0) % num_groups == 0, "Input batch must be divisible by num_groups"
    
    # Simulate the unrolled batched matmul result
    input_groups = torch.split(a, a.size(0) // num_groups, dim=0)
    output_parts = []
    
    for i in range(num_groups):
        output_parts.append(torch.matmul(input_groups[i], b_group[i]))
        
    return torch.cat(output_parts, dim=0)

# The function used as the target in the FX graph replacement
GROUPED_GEMM_FUNCTION = custom_grouped_gemm_impl

# -------------------------------------------------------------------

# --- 2. EXAMPLE MODEL (THE PATTERN TARGET) ---

class SimpleGroupedPattern(torch.nn.Module):
    """
    A model simulating a grouped matmul/MoE dispatch pattern that the pass will replace.
    The sequence is: split -> [matmul, matmul, ...] -> cat.
    """
    def __init__(self, in_features, out_features, num_groups):
        super().__init__()
        self.num_groups = num_groups
        # Weight tensor is grouped: [num_groups, in_features, out_features]
        self.weight = torch.nn.Parameter(
            torch.randn(num_groups, in_features, out_features)
        )

    def forward(self, x):
        # 1. Split the input into N groups
        x_groups = torch.split(x, x.size(0) // self.num_groups, dim=0)
        
        output_parts = []
        for i in range(self.num_groups):
            # 2. Perform N independent matmuls
            out = torch.matmul(x_groups[i], self.weight[i])
            output_parts.append(out)
        
        # 3. Concatenate the results (This is the end of the pattern)
        return torch.cat(output_parts, dim=0)

# -------------------------------------------------------------------

# --- 3. FX GRAPH REPLACEMENT PASS ---

def  _find_grouped_gemm_pattern(graph: fx.Graph) -> Optional[Tuple[fx.Node, fx.Node, int, List[fx.Node]]]:
    """
    Matcher to find the specific pattern created by SimpleGroupedPattern.
    Returns: (Input_A_Node, Input_B_Node, Num_Groups, Nodes_To_Delete)
    """
    nodes_to_delete = []
    
    for node in reversed(graph.nodes):
        if node.op == 'call_function' and node.target == torch.cat:
            cat_node = node
            cat_inputs = cat_node.args[0]
            
            if not isinstance(cat_inputs, tuple) or not all(isinstance(n, fx.Node) for n in cat_inputs):
                continue

            num_groups = len(cat_inputs)
            matmul_nodes = []
            
            # Check if all inputs to cat are matmuls
            for matmul_node in cat_inputs:
                if matmul_node.op == 'call_function' and matmul_node.target == torch.matmul:
                    matmul_nodes.append(matmul_node)
                else:
                    nodes_to_delete.clear()
                    continue
            
            if len(matmul_nodes) != num_groups:
                continue

            # Identify B_GROUP (weight) and A_INPUT (original tensor)
            b_group_node = matmul_nodes[0].args[1]
            a_split_node = matmul_nodes[0].args[0].args[0] 
            
            if a_split_node.target != torch.split:
                 continue
            
            a_node = a_split_node.args[0]
            
            if b_group_node.op != 'get_attr':
                continue
            
            # Pattern found: build the list of nodes to remove
            nodes_to_delete.append(cat_node)
            nodes_to_delete.append(a_split_node)
            nodes_to_delete.extend(matmul_nodes)
            
            # Add the intermediate 'getitem' nodes (slices)
            for matmul_node in matmul_nodes:
                getitem_node = matmul_node.args[0]
                if getitem_node.target == operator.getitem:
                    nodes_to_delete.append(getitem_node)
            
            return (a_node, b_group_node, num_groups, nodes_to_delete)
    
    return None

def grouped_gemm_replacement_pass(graph_module: fx.GraphModule) -> fx.GraphModule:
    """FX pass to replace the grouped GEMM pattern with the custom function."""
    graph = graph_module.graph
    new_graph = fx.Graph()
    node_map: Dict[fx.Node, fx.Node] = {}
    
    pattern_match = _find_grouped_gemm_pattern(graph)

    print(pattern_match)
    
    if pattern_match is None:
        # If pattern not found, just copy the graph
        for node in graph.nodes:
            new_node = new_graph.node_copy(node, lambda x: node_map.get(x, x))
            node_map[node] = new_node
        new_graph.lint()
        return graph_module
        
    print(f"[PASS] Found Grouped GEMM Pattern! Replacing with {GROUPED_GEMM_FUNCTION.__name__}.")
        
    (input_a_node, input_b_group_node, num_groups, nodes_to_delete) = pattern_match
    nodes_to_delete_set = set(nodes_to_delete)
    
    # Get the original output node (torch.cat) for remapping downstream users
    cat_node = [n for n in nodes_to_delete if n.target == torch.cat][0]

    # Iterate and copy/replace
    for node in graph.nodes:
        if node in nodes_to_delete_set:
            continue

        # Copy non-pattern nodes
        new_node = new_graph.node_copy(node, lambda x: node_map.get(x, x))
        node_map[node] = new_node
        
        # When we process the last of the dependencies, insert the replacement call
        # We use input_b_group_node as a trigger for insertion.
        if node == input_b_group_node: 
            
            # 1. Map old inputs to the new graph's nodes
            new_a = node_map[input_a_node]
            new_b_group = node_map[input_b_group_node]
            
            # 2. Call the custom, fused operator!
            with new_graph.inserting_after(new_b_group):
                new_replacement_node = new_graph.call_function(
                    GROUPED_GEMM_FUNCTION, (new_a, new_b_group, num_groups)
                )
            
            # 3. CRITICAL: Map the original pattern output node (cat_node) 
            # to the new replacement node.
            node_map[cat_node] = new_replacement_node

    # Finalize and compile the new graph
    new_graph.lint()
    graph_module.graph = new_graph
    graph_module.recompile()
    
    return graph_module

# -------------------------------------------------------------------

# --- 4. COMPILATION INTEGRATION ---

def custom_moe_compiler(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    Custom Inductor backend that applies the graph pass before execution.
    FIXED: Returns the transformed module's forward method instead of calling
    the unstable torch._inductor.compile_fx.
    """
    # 1. Apply your custom pass
    gm_transformed = grouped_gemm_replacement_pass(gm)
    
    # 2. Hand off the transformed graph. The backend must return a callable.
    # Returning gm_transformed.forward satisfies the contract for now.
    return gm_transformed.forward


if __name__ == '__main__':
    print("--- Grouped GEMM FX Pass Demo ---")
    
    # --- Setup ---
    BATCH_SIZE = 8
    IN_FEATURES = 16
    OUT_FEATURES = 32
    NUM_GROUPS = 2
    
    # Create the model instance
    model = SimpleGroupedPattern(IN_FEATURES, OUT_FEATURES, NUM_GROUPS)
    
    # Input tensor (batch size must be divisible by NUM_GROUPS for this demo)
    x = torch.randn(BATCH_SIZE, IN_FEATURES)
    
    # 1. Verify Eager mode output
    eager_output = model(x)
    print(f"\n[Eager] Model Output shape: {eager_output.shape}")
    
    # 2. Trace the model to see the initial graph
    traced_model = fx.symbolic_trace(model)
    print("\n[Initial FX Graph (Eager Pattern)]")
    traced_model.graph.print_tabular()
    
    # 3. Compile the model with the custom pass
    try:
        compiled_model = torch.compile(model, backend=custom_moe_compiler)
        compiled_output = compiled_model(x)
        print(f"\n[Compiled] Model Output shape: {compiled_output.shape}")

        # 4. Verify correctness
        assert torch.allclose(eager_output, compiled_output, atol=1e-6), "Compiled output does not match Eager output!"
        print("\n[SUCCESS] Compiled output matches Eager output.")
        
        # 5. Show Transformed Graph by running the pass manually (for visualization)
        transformed_model = grouped_gemm_replacement_pass(fx.symbolic_trace(model))
        
        print("\n[Transformed FX Graph (Fusing Applied)]")
        transformed_model.graph.print_tabular()
        
    except Exception as e:
        print(f"\n[ERROR] Could not complete torch.compile demo.")
        print(f"Error details: {e}")