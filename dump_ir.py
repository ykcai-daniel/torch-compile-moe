import torch
import torch.fx as fx
from torch._inductor import config as inductor_config
from torch._inductor.compile_fx import compile_fx


def dump_ir_and_guards(model: torch.nn.Module, inputs, backend_pass=None):
    """
    Compile the model once and print:
      - Dynamo guards
      - Dynamo FX graph
      - FX graph before/after backend_pass
    """
    if backend_pass is None:
        backend_pass = lambda gm, _: gm

    # Reset Dynamo state to ensure a fresh compile per call.
    torch._dynamo.reset()

    # Handle both legacy tuple return and newer ExplainOutput object.
    try:
        explain_out = torch._dynamo.explain(model)(*inputs)
    except TypeError:
        # Fallback for older API signature
        explain_out = torch._dynamo.explain(model, *inputs)

    if isinstance(explain_out, tuple):
        gm_explained, guards, guard_str, ops_per_graph, guard_failures = explain_out
    else:
        gm_explained = getattr(explain_out, "graph", None)
        if gm_explained is None:
            graphs = getattr(explain_out, "graphs", None)
            gm_explained = graphs[0] if graphs else None
        guards = getattr(explain_out, "guards", None)
        guard_str = getattr(explain_out, "guard_str", None)
        if callable(guard_str):
            guard_str = guard_str()
        ops_per_graph = getattr(explain_out, "ops_per_graph", None)
        guard_failures = getattr(explain_out, "guard_failures", None)

    print("=" * 80)
    print("DYNAMO GUARDS")
    print("=" * 80)
    if guard_str:
        print(guard_str)
    else:
        print(guards)

    print("\n" + "=" * 80)
    print("DYNAMO FX GRAPH")
    print("=" * 80)
    if gm_explained is not None and hasattr(gm_explained, "graph"):
        print(gm_explained.graph)
    else:
        print("N/A (Dynamo explain did not return a GraphModule)")

    # Enable extra Inductor tracing output (goes to temp dir)
    inductor_config.trace.enabled = True
    inductor_config.debug = True

    def backend(gm: fx.GraphModule, example_inputs):
        print("\n" + "=" * 80)
        print("FX BEFORE BACKEND PASS")
        print("=" * 80)
        print(gm.graph)

        gm_after = backend_pass(gm, example_inputs)
        gm_after_graph = gm_after.graph if isinstance(gm_after, fx.GraphModule) else gm.graph

        print("\n" + "=" * 80)
        print("FX AFTER BACKEND PASS")
        print("=" * 80)
        print(gm_after_graph)

        compiled_gm = gm_after if isinstance(gm_after, fx.GraphModule) else gm
        return compile_fx(compiled_gm, example_inputs)

    compiled = torch.compile(model, backend=backend, dynamic=True, fullgraph=True)
    compiled(*inputs)


def identity_pass(gm, example_inputs):
    """Simple identity backend pass for quick checks."""
    return gm


__all__ = ["dump_ir_and_guards", "identity_pass"]
