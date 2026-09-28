"""Unit tests for the bare matmul→matmul chain (linear attention `Q (Kᵀ V)`).

A bare chain is two matmuls with NO combine node between them: the first
matmul's result is directly an operand of the second. Before this change the
matmul emitter refused it — its multi-matmul paths only handled a matmul feeding
a *non-matmul* combine (silu/relu MLP) or a softmax reduce (attention). These
tests pin the target behavior: such a chain routes to the staged path, resolves
its four schedule roles (the 4th, `k`, coming from the EARLIER matmul's
contraction rather than a reduce), keeps the intermediate as a materialized
moving operand of the second matmul, and emits two matmuls.

The graph is the ISA form of `out = Q (Kᵀ V)` in CROSS-attention shape (distinct
query length `sq` and key length `sk`): that keeps the sequence dim off both
`mm2`'s output partition (m) and `mm1`'s contraction (k), so no single dim takes
two roles. `nc_matmul` computes ``inputs[0]ᵀ @ inputs[1]`` (stationary, moving).
"""

from __future__ import annotations

import ast

from axon.codegen import emit
from axon.codegen.plan import build_emission_plan
from axon.ir import Node, nuGraph


def _undefined_names(code):
    """Names read but never bound (params, assignments, imports, comprehension
    targets, or builtins). Catches an emitted reference to a buffer no stage
    produces — e.g. a materialized transpose left unemitted — which `compile()`
    (syntax only) misses and which would NameError at trace time."""
    import builtins

    tree = ast.parse(code)
    bound = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            bound.update(a.arg for a in node.args.args)
            bound.update(a.arg for a in node.args.kwonlyargs)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
    read = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    return sorted(read - bound)


def _matmul_site_count(code):
    """Distinct matmuls the source realizes (each emits a peeled + a loop-body
    nc_matmul, so count unique PSUM accumulator targets, not call sites)."""
    names = set()
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Call):
            continue
        if "nc_matmul" not in ast.unparse(node.func) or not node.args:
            continue
        names.add(ast.unparse(node.args[0]).split("[")[0])
    return len(names)


def _inp(node_id, shape):
    return Node(node_id, "input", [], {"shape": shape})


def _graph(nodes):
    id_to_node = {n.id: n for n in nodes}
    inputs = {n.id for n in nodes if n.op == "input"}
    compute = [n for n in nodes if n.op != "input"]
    return compute, id_to_node, inputs


# Cross-attention extents: sq/sk on the 128 partition base (m/k), dv on the 512
# wide base (n), dk the contraction (p) that clamps to a single 128 tile.
SQ, SK, DK, DV = 512, 512, 128, 512


def _bare_chain_nodes(sq=SQ, sk=SK, dk=DK, dv=DV):
    """ISA graph for out = Q (Kᵀ V):

    state = nc_matmul(k, v)   # kᵀ@v            -> (dk, dv)
    qt    = nc_transpose(q)                     -> (dk, sq)
    out   = nc_matmul(qt, state)  # qtᵀ@state = q@state -> (sq, dv)
    """
    return [
        _inp("q", (sq, dk)),
        _inp("key", (sk, dk)),
        _inp("v", (sk, dv)),
        Node("state", "nc_matmul", ["key", "v"], shape=(dk, dv)),
        Node("qt", "nc_transpose", ["q"], shape=(dk, sq)),
        Node("out", "nc_matmul", ["qt", "state"], shape=(sq, dv)),
    ]


def test_bare_chain_builds_a_plan_via_the_stage_path():
    # Before the change this raised "staged emission supports exactly one
    # reduction, got 0"; after, it builds and routes to the staged path.
    plan = build_emission_plan(*_graph(_bare_chain_nodes()), output_id="out")
    assert plan.is_multi
    assert plan.stages != ()  # routed to _build_multi_stage, not the chained path
    assert plan.reduces == ()  # a bare chain carries no reduction
    assert {mp.mm_id for mp in plan.matmuls} == {"state", "out"}


def test_bare_chain_second_matmul_reads_the_first_as_a_materialized_operand():
    plan = build_emission_plan(*_graph(_bare_chain_nodes()), output_id="out")
    out_mm = plan.matmul("out")
    # The intermediate matmul is the SECOND matmul's moving operand, staged
    # (materialized) via stop_ids rather than walked as a raw-input load chain.
    assert out_mm.mov.root_id == "state"
    assert out_mm.mov.is_materialized
    # The stationary operand is a transpose of the raw query input, produced and
    # consumed in the same stage, so it is walked inline (root q, transpose step),
    # NOT staged as a cross-stage materialized value.
    assert out_mm.stat.root_id == "q"
    assert not out_mm.stat.is_materialized
    assert "qt" in {step.node_id for step in out_mm.stat.steps}


def test_bare_chain_resolves_four_schedule_roles():
    # The 4th role (k) is fixed by the EARLIER matmul's contraction (sk), since
    # there is no reduce to fix it. All four m/n/k/p roles must be assigned.
    plan = build_emission_plan(*_graph(_bare_chain_nodes()), output_id="out")
    assert set(plan.dim_aliases.values()) == {"m", "n", "k", "p"}
    assert set(plan.role_dims) == {"m", "n", "k", "p"}


def test_bare_chain_emits_two_matmuls_and_compiles():
    graph = nuGraph(
        nodes=_bare_chain_nodes(),
        input_ids=("q", "key", "v"),
        output_ids=("out",),
    )
    code = emit(graph, kernel_name="bare_chain_linear_attention")
    compile(code, "<bare_chain>", "exec")  # emitted NKI is valid Python
    # No reference to a buffer no stage produces (the materialized-transpose bug).
    assert _undefined_names(code) == []
    assert _matmul_site_count(code) == 2
