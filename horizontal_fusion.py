import torch
from collections import defaultdict
from typing import Dict, List, Tuple


# Given two grouped_gemm: grouped_gemm(tokens, w1, index) and grouped_gemm(tokens, w2, index),
# fuse them into one by concatnating the weights grouped_gemm(tokens, [w1|w2], index) and reducing memory load.
def horizontal_fusion_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # Find all grouped_gemm calls and group by (tokens, top_k_expert_activation, use_triton)
    gemm_groups: Dict[Tuple, List[torch.fx.Node]] = defaultdict(list)

    for node in gm.graph.nodes:
        if node.op == 'call_function' and node.target == torch.ops.moe.grouped_gemm:
            # Extract arguments: (tokens, weights, top_k_expert_activation, use_triton)
            if len(node.args) >= 3:
                tokens = node.args[0]
                top_k_expert_activation = node.args[2]
                use_triton = node.args[3] if len(node.args) > 3 else False

                # Group by the shared inputs
                key = (tokens, top_k_expert_activation, use_triton)
                gemm_groups[key].append(node)

    # Fuse groups with multiple gemm operations
    modifications_made = False
    for key, nodes in gemm_groups.items():
        if len(nodes) <= 1:
            # No fusion possible
            continue

        modifications_made = True
        tokens, top_k_expert_activation, use_triton = key

        # Get the weights for each gemm
        weights_list = [node.args[1] for node in nodes]

        # Insert concatenation of weights before the first gemm
        with gm.graph.inserting_before(nodes[0]):
            # Concatenate weights along dim=2 (d_ff dimension)
            # weights shape: [num_experts, d_model, d_ff]
            fused_weights = gm.graph.call_function(
                torch.cat,
                args=(weights_list,),
                kwargs={'dim': 2}
            )

            # Create single fused grouped_gemm call
            fused_gemm = gm.graph.call_function(
                torch.ops.moe.grouped_gemm,
                args=(tokens, fused_weights, top_k_expert_activation, use_triton),
            )

        # Calculate the output sizes for splitting
        # Each weight has shape [num_experts, d_model, d_ff_i]
        # We need to split the output along dim=1 (d_ff dimension)

        # Create getattr nodes to get the size of each weight's last dimension
        split_sizes = []
        with gm.graph.inserting_before(nodes[0]):
            for weight_node in weights_list:
                # Get the size of the last dimension
                # weight.shape[2] gets the d_ff size
                shape_node = gm.graph.call_function(
                    getattr,
                    args=(weight_node, 'shape')
                )
                size_node = gm.graph.call_function(
                    lambda s: s[2],
                    args=(shape_node,)
                )
                split_sizes.append(size_node)

        # Insert split operation after the fused gemm
        with gm.graph.inserting_after(fused_gemm):
            # Split the fused output back into individual outputs
            # torch.split(tensor, split_sizes, dim=1)
            split_sizes_list = gm.graph.call_function(
                list,
                args=(split_sizes,)
            )

            split_result = gm.graph.call_function(
                torch.split,
                args=(fused_gemm, split_sizes_list),
                kwargs={'dim': 1}
            )

        # Replace uses of original gemm nodes with the split outputs
        for i, node in enumerate(nodes):
            with gm.graph.inserting_after(split_result):
                # Extract the i-th element from the split result
                getitem_node = gm.graph.call_function(
                    lambda tuple_result, idx: tuple_result[idx],
                    args=(split_result, i),
                )

            # Replace all uses of the original node with the getitem result
            node.replace_all_uses_with(getitem_node)

    # Clean up unused nodes if modifications were made
    if modifications_made:
        gm.graph.eliminate_dead_code()
        gm.recompile()

    return gm


# Backend to enable grouped_gemm
def horizontal_fusion_backend(gm: torch.fx.GraphModule, example_inputs):
    print(f"[Horizontal Fusion] Applying fusion pass to graph with {len(list(gm.graph.nodes))} nodes")

    # Apply horizontal fusion pass
    gm = horizontal_fusion_pass(gm)

    print(f"[Horizontal Fusion] After fusion: {len(list(gm.graph.nodes))} nodes")

    # Compile with default inductor backend
    from torch._inductor.compile_fx import compile_fx

    return compile_fx(gm, example_inputs)
