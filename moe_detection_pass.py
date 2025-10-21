import torch
import torch.fx as fx
from torch.fx import GraphModule, Node
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F

@dataclass
class MoEPattern:
    """Represents a detected MoE pattern in the graph."""
    router_node: Node  # Node that computes routing/gating
    expert_nodes: List[Node]  # Expert computation nodes
    combine_node: Node  # Node that combines expert outputs
    topk_node: Node = None  # Optional top-k selection node
    
    def __repr__(self):
        return (f"MoEPattern(router={self.router_node.name}, "
                f"experts={[e.name for e in self.expert_nodes]}, "
                f"combine={self.combine_node.name})")


class MoEPatternDetector(fx.Transformer):
    """
    FX Pass that detects Mixture of Experts patterns in a graph.
    
    Common MoE patterns include:
    1. Router/Gate → Top-K Selection → Expert Computation → Weighted Combination
    2. Softmax → Expert Selection → Parallel Expert Execution → Aggregation
    """
    
    def __init__(self, module: GraphModule):
        super().__init__(module)
        self.detected_patterns: List[MoEPattern] = []
        self.visited_nodes: Set[Node] = set()
        
    def detect_patterns(self) -> List[MoEPattern]:
        """Main entry point to detect all MoE patterns."""
        self.detected_patterns = []
        self.visited_nodes = set()
        
        for node in self.module.graph.nodes:
            if node not in self.visited_nodes:
                pattern = self._try_detect_moe_at_node(node)
                if pattern:
                    self.detected_patterns.append(pattern)
                    self._mark_pattern_visited(pattern)
        
        return self.detected_patterns
    
    def _try_detect_moe_at_node(self, node: Node) -> MoEPattern:
        """Try to detect an MoE pattern starting from this node."""
        # Pattern 1: Look for router/gating with softmax or linear
        if self._is_router_node(node):
            return self._detect_from_router(node)
        
        # Pattern 2: Look for top-k selection (common in sparse MoE)
        if self._is_topk_node(node):
            return self._detect_from_topk(node)
        
        return None
    
    def _is_router_node(self, node: Node) -> bool:
        """Check if node looks like a router/gating mechanism."""
        if node.op == 'call_function':
            # Check for softmax (common in routing)
            if node.target in [torch.nn.functional.softmax, torch.softmax]:
                return True
        elif node.op == 'call_module':
            # Check for Linear layer that might be a router
            module = self._get_module(node)
            if isinstance(module, torch.nn.Linear):
                # Check if this linear has multiple downstream branches (experts)
                users = list(node.users.keys())
                if len(users) > 1:
                    return True
        return False
    
    def _is_topk_node(self, node: Node) -> bool:
        """Check if node is a top-k selection."""
        # if node.op == 'call_function':
        #     # if node.target in [torch.topk]:
        #     #     return True
        # elif node.op == 'call_method' and node.target == 'topk':
        #     return True
        return False
    
    def _detect_from_router(self, router_node: Node) -> MoEPattern:
        """Detect MoE pattern starting from a router node."""
        # Look for top-k in immediate children
        topk_node = None
        for user in router_node.users.keys():
            if self._is_topk_node(user):
                topk_node = user
                break
        
        # Find expert nodes - look for parallel computation paths
        expert_nodes = self._find_expert_nodes(topk_node or router_node)
        
        if len(expert_nodes) < 2:  # Need at least 2 experts for MoE
            return None
        
        # Find combine/aggregation node
        combine_node = self._find_combine_node(expert_nodes)
        
        if combine_node:
            return MoEPattern(
                router_node=router_node,
                expert_nodes=expert_nodes,
                combine_node=combine_node,
                topk_node=topk_node
            )
        
        return None
    
    def _detect_from_topk(self, topk_node: Node) -> MoEPattern:
        """Detect MoE pattern starting from a top-k node."""
        # Find the router (should be an input to top-k)
        router_node = None
        for arg in topk_node.args:
            if isinstance(arg, Node) and self._is_router_node(arg):
                router_node = arg
                break
        
        if not router_node:
            return None
        
        return self._detect_from_router(router_node)
    
    def _find_expert_nodes(self, start_node: Node) -> List[Node]:
        """Find parallel expert computation nodes."""
        experts = []
        
        # Strategy: Look for nodes that are accessed via indexing/slicing
        # or multiple similar computational branches
        for user in start_node.users.keys():
            # Check for getitem (indexing into expert outputs)
            if user.op == 'call_function' and user.target == 'operator.getitem':
                experts.append(user)
            # Check for parallel module calls
            elif user.op == 'call_module':
                module = self._get_module(user)
                # Look for similar modules (experts typically have same architecture)
                if self._is_expert_like(module):
                    experts.append(user)
        
        # Also look one level deeper for expert computations
        if not experts:
            expert_candidates = []
            for user in start_node.users.keys():
                for sub_user in user.users.keys():
                    if sub_user.op == 'call_module':
                        module = self._get_module(sub_user)
                        if self._is_expert_like(module):
                            expert_candidates.append(sub_user)
            
            # Group by similar architecture
            experts = self._deduplicate_experts(expert_candidates)
        
        return experts
    
    def _is_expert_like(self, module) -> bool:
        """Check if a module looks like an expert."""
        # Experts are typically Linear layers or small MLPs
        return isinstance(module, (torch.nn.Linear, torch.nn.Sequential))
    
    def _find_combine_node(self, expert_nodes: List[Node]) -> Node:
        """Find the node that combines expert outputs."""
        # Find common downstream node that uses multiple experts
        common_users = None
        
        for expert in expert_nodes:
            users = set(expert.users.keys())
            if common_users is None:
                common_users = users
            else:
                common_users = common_users.intersection(users)
        
        if common_users:
            # Look for aggregation operations
            for user in common_users:
                if self._is_aggregation_node(user):
                    return user
        
        # Alternatively, find nodes that use outputs from all experts
        potential_combines = []
        for expert in expert_nodes:
            for user in expert.users.keys():
                # Check if this user depends on multiple experts
                expert_deps = sum(1 for e in expert_nodes 
                                 if self._node_depends_on(user, e))
                if expert_deps >= 2:
                    potential_combines.append(user)
        
        return potential_combines[0] if potential_combines else None
    
    def _is_aggregation_node(self, node: Node) -> bool:
        """Check if node performs aggregation (sum, weighted sum, etc)."""
        if node.op == 'call_function':
            if node.target in [torch.sum, torch.mean, torch.add, 
                              torch.mul, operator.add, operator.mul]:
                return True
        elif node.op == 'call_method':
            if node.target in ['sum', 'mean', 'add', 'mul']:
                return True
        return False
    
    def _node_depends_on(self, node: Node, target: Node) -> bool:
        """Check if node depends on target node."""
        visited = set()
        
        def dfs(n: Node) -> bool:
            if n == target:
                return True
            if n in visited:
                return False
            visited.add(n)
            
            for arg in n.args:
                if isinstance(arg, Node) and dfs(arg):
                    return True
            return False
        
        return dfs(node)
    
    def _deduplicate_experts(self, candidates: List[Node]) -> List[Node]:
        """Remove duplicate expert nodes."""
        seen = set()
        unique = []
        for node in candidates:
            if node not in seen:
                seen.add(node)
                unique.append(node)
        return unique
    
    def _mark_pattern_visited(self, pattern: MoEPattern):
        """Mark all nodes in a pattern as visited."""
        self.visited_nodes.add(pattern.router_node)
        if pattern.topk_node:
            self.visited_nodes.add(pattern.topk_node)
        self.visited_nodes.add(pattern.combine_node)
        for expert in pattern.expert_nodes:
            self.visited_nodes.add(expert)
    
    def _get_module(self, node: Node):
        """Get the actual module for a call_module node."""
        if node.op == 'call_module':
            return self.module.get_submodule(node.target)
        return None


def detect_moe_patterns(model: torch.nn.Module) -> Tuple[GraphModule, List[MoEPattern]]:
    """
    Convenience function to trace a model and detect MoE patterns.
    
    Args:
        model: PyTorch module to analyze
        
    Returns:
        Tuple of (traced GraphModule, list of detected MoE patterns)
    """
    # Trace the model
    traced = fx.symbolic_trace(model)
    print(traced.graph)
    
    # Detect patterns
    detector = MoEPatternDetector(traced)
    patterns = detector.detect_patterns()
    
    return traced, patterns

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
    def __init__(self, hidden_dim: int = 1024, ffn_dim : int= 512, num_experts: int = 16, top_k: int = 4):
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
    def __init__(self, hidden_dim: int = 1024, ffn_dim : int= 512, num_experts: int = 16, top_k: int = 4):
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


# Example usage
if __name__ == "__main__":
    # Example: Simple MoE model
    class SimpleMoE(torch.nn.Module):
        def __init__(self, input_dim=128, hidden_dim=256, num_experts=4):
            super().__init__()
            self.router = torch.nn.Linear(input_dim, num_experts)
            self.experts = torch.nn.ModuleList([
                torch.nn.Linear(input_dim, hidden_dim) 
                for _ in range(num_experts)
            ])
            
        def forward(self, x):
            # Router computes gating weights
            router_logits = self.router(x)
            routing_weights = torch.softmax(router_logits, dim=-1)
            
            # Select top-2 experts
            topk_weights, topk_indices = torch.topk(routing_weights, k=2, dim=-1)
            
            # Compute expert outputs (simplified)
            expert_outputs = []
            for i, expert in enumerate(self.experts):
                expert_outputs.append(expert(x))
            
            # Stack and select
            stacked = torch.stack(expert_outputs, dim=1)
            selected = stacked.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, stacked.size(-1)))
            
            # Weighted combination
            output = (selected * topk_weights.unsqueeze(-1)).sum(dim=1)
            
            return output
    
    # Detect patterns
    model = SimpleMoE()
    traced, patterns = detect_moe_patterns(model)

    model_basic = MoELayer()

    model_optimized = MoELayerOptimized()

    #traced_basic, patterns_basic = detect_moe_patterns(model_basic)

    traced_opt, patterns_opt = detect_moe_patterns(model_optimized)

