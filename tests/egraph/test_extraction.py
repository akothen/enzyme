"""Extraction tests (plan section 11 table).

Covers: shared-class consistency (a shared e-class uses the same selected
e-node in every output and occurrence); output order and duplication;
productive cycles (a cyclic e-class still yields finite acyclic selections that
terminate and are DAGs); uninhabited cycles (an SCC with no productive exit is
uninhabited, a declared uninhabited output fails extraction, a non-output
uninhabited class reachable only via an unselected alternative is fine); acyclic
branch pruning (cyclic branches are rejected and backtracked, not the whole
class); and one-at-a-time materialization (lazy streaming).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from axon import cli
from axon.egraph.adapter import EClassRef, EGraphAdapter, ENodeRef, Snapshot
from axon.egraph.analysis import decode_isa, ensure_isa_semantics_registered
from axon.egraph.codec import encode_isa_enode, encode_isa_input
from axon.egraph.extraction import (
    ExtractionError,
    compute_inhabited_classes,
    iter_isa_selections,
    iter_materialized_isa_graphs,
    iter_program_selections,
    materialize_isa_graph,
    selection_would_cycle,
    validate_materialized_graph,
)
from axon.egraph.lowering import lower_tensor_egraph
from axon.egraph.pipeline import build_egraph_search
from axon.egraph.proof import ProofStore
from axon.egraph.tensor import ingest_tensor_graph
from axon.ir import build_graph_from_kernel, nuGraph
from axon.isa_semantics import engine, nl

_SHAPE = (4, 4)
_DIMS = {"m": 4, "k": 4, "n": 4}
_KERNELS = Path(__file__).resolve().parents[2] / "kernels"


def _mul_graph() -> nuGraph:
    spec = cli._load_spec(str(_KERNELS / "mul"))
    dim_sizes = dict(zip(spec.dim_vars, (4, 8), strict=True))
    return build_graph_from_kernel(
        spec.axon_kernel, *spec.input_specs, dim_sizes=dim_sizes
    )


def test_pipeline_truncates_lowering_when_wall_limit_is_lapsed() -> None:
    search = build_egraph_search(
        _mul_graph(),
        max_hw_size=2,
        timeout=3000,
        wall_clock_seconds=0.0,
    )
    # A lapsed wall truncates discovery, but lowering is mandatory finalization
    # work: every declared output still gets a realization off the wall, so the
    # search truncates with extractable roots instead of failing with none.
    assert search.tensor_status.status == "truncated"
    assert search.tensor_status.reason == "wall_clock_seconds"
    assert search.tensor_status.rounds == 0
    assert search.lowering_status.status == "lowered"
    assert not search.lowering_status.unrealized_outputs
    assert search.isa_status.status == "truncated"
    assert search.isa_status.reason == "wall_clock_seconds"
    assert search.isa_output_roots
    assert search.terminal_status.status == "truncated"
    assert search.terminal_status.stop_reason == "wall_clock_seconds"
    assert search.terminal_status.truncated_stage == "tensor_propagation"


_MISSING_OUTPUT = EClassRef(snapshot_id=-1, value="missing", sort="TensorExpr")


def _patch_unrealized(
    monkeypatch: pytest.MonkeyPatch, *, calls_to_fail: tuple[int, ...]
) -> list[int]:
    """Report the numbered lowering passes as truncated with an output missing."""
    import axon.egraph.pipeline as pipeline
    from axon.egraph.lowering import LoweringStatus

    original = pipeline.lower_tensor_egraph
    seen: list[int] = []

    def lower(*args: Any, **kwargs: Any) -> Any:
        seen.append(len(seen))
        isa_adapter, L, status = original(*args, **kwargs)
        if seen[-1] in calls_to_fail:
            status = LoweringStatus(
                status="truncated",
                unrealized_outputs=(_MISSING_OUTPUT,),
                reason="wall_clock_seconds",
            )
        return isa_adapter, L, status

    monkeypatch.setattr(pipeline, "lower_tensor_egraph", lower)
    return seen


def test_a_partial_fallback_keeps_the_main_pass_realizations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both passes fall short, so the main pass's roots must survive.

    A fallback that also leaves an output unrealized must not rebind the lowered
    graph: its refs belong to the ingested snapshot, so pairing them with the
    saturated roots would resolve nothing.
    """
    seen = _patch_unrealized(monkeypatch, calls_to_fail=(0, 1))
    search = build_egraph_search(_mul_graph(), max_hw_size=2, timeout=3000)

    assert len(seen) == 2, "the fallback pass did not run"
    assert search.tensor_output_roots
    assert all(root in search.L for root in search.tensor_output_roots)
    assert search.isa_output_roots
    # The kept roots come from the saturated snapshot the search reports.
    assert {root.snapshot_id for root in search.tensor_output_roots} == {
        search.tensor_snapshot.snapshot_id
    }


def test_a_successful_fallback_pairs_its_roots_with_its_own_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback's ingested-snapshot roots must be persisted with that snapshot."""
    from axon.egraph import persist

    seen = _patch_unrealized(monkeypatch, calls_to_fail=(0,))
    search = build_egraph_search(_mul_graph(), max_hw_size=2, timeout=3000)

    assert len(seen) == 2, "the fallback pass did not run"
    assert not search.lowering_status.unrealized_outputs
    assert search.tensor_output_roots
    assert all(
        root in search.tensor_snapshot.classes for root in search.tensor_output_roots
    )

    cache = persist.build_cache_file(
        search,
        kernel_name="mul",
        dim_sizes={"m": 4, "n": 8},
        graph_identity="test",
        run_stem="mul__m4_n8",
        options={},
    )
    rebuilt = cache.tensor.rebuild()
    assert cache.tensor_output_roots
    assert all(root in rebuilt.classes for root in cache.tensor_output_roots)


def _reciprocal(x: Any) -> Any:
    return encode_isa_enode("reciprocal", {}, [x])


def _tensor_copy(x: Any) -> Any:
    return encode_isa_enode("tensor_copy", {"engine": engine.unknown}, [x])


def _nodes_by_op(G: Any, op: str) -> list[Any]:
    return [n for n in G.nodes if n.op == op]


def _node_by_id(G: Any, node_id: str) -> Any:
    return next(n for n in G.nodes if n.id == node_id)


def _is_dag(G: Any) -> bool:
    """Every node's inputs must refer to nodes defined earlier (topo order)."""
    defined: set[str] = set()
    for node in G.nodes:
        if any(inp not in defined for inp in node.inputs):
            return False
        defined.add(node.id)
    return True


# ---------------------------------------------------------------------------
# Synthetic-snapshot helpers for the pure enumeration and inhabitance logic.
# ---------------------------------------------------------------------------

_SID = 4242


def _cref(name: str) -> EClassRef:
    return EClassRef(_SID, name, "IsaExpr")


def _enode(name: str, *children: EClassRef) -> ENodeRef:
    return ENodeRef(callable=name, egg_fn=name, sort="IsaExpr", args=tuple(children))


def _snapshot(classes: dict[EClassRef, tuple[ENodeRef, ...]]) -> Snapshot:
    return Snapshot(_SID, classes, {})


# ---------------------------------------------------------------------------
# selection_would_cycle unit behavior
# ---------------------------------------------------------------------------


def test_selection_would_cycle_detects_back_edge() -> None:
    p, q = _cref("P"), _cref("Q")
    p_to_q = _enode("fP", q)
    q_to_p = _enode("fQ", p)
    # P already selects an e-node depending on Q; selecting Q -> P closes P->Q->P.
    assert selection_would_cycle({p: p_to_q}, q, q_to_p) is True
    # A leaf e-node adds no edges, so it can never close a cycle.
    assert selection_would_cycle({p: p_to_q}, q, _enode("leaf")) is False
    # A dependency that does not lead back to the new class is acyclic.
    r = _cref("R")
    assert selection_would_cycle({}, p, _enode("fP2", r)) is False


# ---------------------------------------------------------------------------
# Uninhabited cycles
# ---------------------------------------------------------------------------


def test_uninhabited_scc_is_excluded_and_output_fails() -> None:
    x = _cref("X")
    a = _cref("A")
    b = _cref("B")
    leaf = _enode("leaf")
    classes = {
        x: (leaf,),
        a: (_enode("fA", b),),  # A depends only on B
        b: (_enode("fB", a),),  # B depends only on A -> SCC with no leaf
    }
    snap = _snapshot(classes)

    inhabited, heights = compute_inhabited_classes(snap)
    assert inhabited == {x}
    assert a not in inhabited and b not in inhabited
    assert heights[x] == 1

    # A declared output that is uninhabited fails extraction.
    with pytest.raises(ExtractionError):
        list(iter_isa_selections(snap, [a]))


def test_non_output_uninhabited_via_unselected_alternative_is_fine() -> None:
    x = _cref("X")
    a = _cref("A")
    b = _cref("B")
    leaf = _enode("leaf")
    # X has a productive leaf member and an alternative member depending on the
    # uninhabited SCC. The alternative is simply never selectable.
    classes = {
        x: (leaf, _enode("fX", a)),
        a: (_enode("fA", b),),
        b: (_enode("fB", a),),
    }
    snap = _snapshot(classes)
    inhabited, _ = compute_inhabited_classes(snap)

    selections = list(iter_program_selections(snap, [x], inhabited))
    assert len(selections) == 1
    assert selections[0][x] == leaf


# ---------------------------------------------------------------------------
# Acyclic branch pruning (real ISA snapshot with Transpose(Transpose(x)) == x)
# ---------------------------------------------------------------------------


def _transpose_cycle_adapter() -> tuple[EGraphAdapter, Any, Any]:
    """Build an ISA e-graph with a productive transpose-pair cycle on x.

    Returns the adapter plus the ``x`` and ``reciprocal(x)`` handles.
    """
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    x = adapter.intern_expr(encode_isa_input("x", _SHAPE), "x").handle
    t1 = adapter.intern_expr(
        encode_isa_enode("nc_transpose", {"engine": engine.unknown}, [x]), "t1"
    ).handle
    t2 = adapter.intern_expr(
        encode_isa_enode("nc_transpose", {"engine": engine.unknown}, [t1]), "t2"
    ).handle
    # Transpose(Transpose(x)) == x makes x's class self-reachable.
    adapter.union_if_distinct(t2, x)
    out = adapter.intern_expr(_reciprocal(x), "out").handle
    return adapter, x, out


def test_cyclic_branch_pruned_but_class_survives() -> None:
    adapter, x, out = _transpose_cycle_adapter()
    snap = adapter.freeze_snapshot()
    x_class = adapter.resolve_handle(snap, x)

    # x's class genuinely holds two members: the input leaf and a transpose.
    ops = {r.egg_fn for r in snap.members(x_class)}
    assert ops == {"axIInput", "axIOp1"}

    inhabited, _ = compute_inhabited_classes(snap)
    assert x_class in inhabited

    # Enumerating x directly: the cyclic transpose member is pruned, but the
    # class is not rejected; the input leaf still produces one selection.
    selections = list(iter_program_selections(snap, [x_class], inhabited))
    assert len(selections) == 1
    assert selections[0][x_class].egg_fn == "axIInput"


def test_productive_cycle_yields_finite_acyclic_graphs() -> None:
    adapter, _x, out = _transpose_cycle_adapter()
    snap = adapter.freeze_snapshot()
    root = adapter.resolve_handle(snap, out)

    graphs = list(iter_materialized_isa_graphs(snap, [root], {"x": {"shape": _SHAPE}}))
    # Enumeration terminates with exactly one acyclic representative.
    assert len(graphs) == 1
    G = graphs[0]
    assert _is_dag(G)
    assert {n.op for n in G.nodes} == {"input", "reciprocal"}
    # The reciprocal consumes the input directly; the cycle unrolling is gone.
    (recip,) = _nodes_by_op(G, "reciprocal")
    (inp,) = _nodes_by_op(G, "input")
    assert recip.inputs == [inp.id]


# ---------------------------------------------------------------------------
# Shared-class consistency (diamond)
# ---------------------------------------------------------------------------


def _diamond_adapter() -> tuple[EGraphAdapter, Any]:
    """A diamond: a shared class S (two e-nodes) feeds two distinct consumers.

    Returns the adapter and the root ``tensor_tensor(A, B)`` handle, where
    ``A = tensor_copy(S)`` and ``B = reciprocal(S)`` both read the shared S.
    """
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    x = adapter.intern_expr(encode_isa_input("x", _SHAPE), "x").handle
    r1 = adapter.intern_expr(_reciprocal(x), "r1").handle
    c1 = adapter.intern_expr(_tensor_copy(x), "c1").handle
    # Union makes S = {reciprocal(x), tensor_copy(x)}: one class, two e-nodes.
    adapter.union_if_distinct(r1, c1)
    a = adapter.intern_expr(_tensor_copy(r1), "a").handle
    b = adapter.intern_expr(_reciprocal(r1), "b").handle
    top = adapter.intern_expr(
        encode_isa_enode(
            "tensor_tensor", {"op": nl.add, "engine": engine.unknown}, [a, b]
        ),
        "top",
    ).handle
    return adapter, top


def test_shared_class_resolved_consistently() -> None:
    adapter, top = _diamond_adapter()
    snap = adapter.freeze_snapshot()
    root = adapter.resolve_handle(snap, top)

    graphs = list(iter_materialized_isa_graphs(snap, [root], {"x": {"shape": _SHAPE}}))
    # Exactly two selections: one per member of the shared class S.
    assert len(graphs) == 2

    shared_ops: set[str] = set()
    for G in graphs:
        assert _is_dag(G)
        (top_node,) = _nodes_by_op(G, "tensor_tensor")
        a_node = _node_by_id(G, top_node.inputs[0])
        b_node = _node_by_id(G, top_node.inputs[1])
        assert a_node.op == "tensor_copy"
        assert b_node.op == "reciprocal"
        # Both consumers read the SAME materialized node for the shared class.
        assert a_node.inputs[0] == b_node.inputs[0]
        shared = _node_by_id(G, a_node.inputs[0])
        assert shared.op in {"reciprocal", "tensor_copy"}
        shared_ops.add(shared.op)

    # Across the two selections both e-nodes of the shared class are exercised,
    # and no single graph ever splits the class into two different e-nodes.
    assert shared_ops == {"reciprocal", "tensor_copy"}


# ---------------------------------------------------------------------------
# Output order and duplication
# ---------------------------------------------------------------------------


def test_output_order_and_duplicates_preserved() -> None:
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    x = adapter.intern_expr(encode_isa_input("x", _SHAPE), "x").handle
    y = adapter.intern_expr(encode_isa_input("y", _SHAPE), "y").handle
    o1 = adapter.intern_expr(_reciprocal(x), "o1").handle
    o2 = adapter.intern_expr(_tensor_copy(y), "o2").handle
    snap = adapter.freeze_snapshot()
    c1 = adapter.resolve_handle(snap, o1)
    c2 = adapter.resolve_handle(snap, o2)

    # Ordered roots with a duplicated first output.
    roots = [c1, c2, c1]
    selections = list(iter_isa_selections(snap, roots))
    assert len(selections) == 1
    G = materialize_isa_graph(
        snap, selections[0], roots, {"x": {"shape": _SHAPE}, "y": {"shape": _SHAPE}}
    )

    assert len(G.output_ids) == 3
    # Order preserved and the duplicate is the identical node id.
    assert G.output_ids[0] == G.output_ids[2]
    assert G.output_ids[0] != G.output_ids[1]
    assert _node_by_id(G, G.output_ids[0]).op == "reciprocal"
    assert _node_by_id(G, G.output_ids[1]).op == "tensor_copy"


# ---------------------------------------------------------------------------
# Declared input order
# ---------------------------------------------------------------------------


def _reversed_first_use_adapter() -> tuple[EGraphAdapter, Any]:
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    a = adapter.intern_expr(encode_isa_input("a", _SHAPE), "a").handle
    b = adapter.intern_expr(encode_isa_input("b", _SHAPE), "b").handle
    adapter.intern_expr(
        encode_isa_input("unused", _SHAPE),
        "unused",
    )
    out = adapter.intern_expr(
        encode_isa_enode(
            "tensor_tensor", {"op": nl.subtract, "engine": engine.unknown}, [b, a]
        ),
        "out",
    ).handle
    return adapter, out


def test_materialization_preserves_declared_input_order() -> None:
    adapter, out = _reversed_first_use_adapter()
    snapshot = adapter.freeze_snapshot()
    root = adapter.resolve_handle(snapshot, out)
    (selection,) = iter_isa_selections(snapshot, [root])

    graph = materialize_isa_graph(
        snapshot,
        selection,
        [root],
        {
            "a": {"shape": _SHAPE},
            "unused": {"shape": _SHAPE},
            "b": {"shape": _SHAPE},
        },
        declared_input_ids=("a", "unused", "b"),
    )

    assert tuple(node.id for node in graph.nodes if node.op == "input") == ("b", "a")
    assert graph.input_ids == ("a", "b")


def test_reversed_first_use_emits_declared_argument_binding() -> None:
    import ast

    from axon.codegen import emit

    adapter, out = _reversed_first_use_adapter()
    snapshot = adapter.freeze_snapshot()
    root = adapter.resolve_handle(snapshot, out)
    (graph,) = iter_materialized_isa_graphs(
        snapshot,
        [root],
        {"a": {"shape": _SHAPE}, "b": {"shape": _SHAPE}},
        declared_input_ids=("a", "b"),
    )

    assert graph.input_ids == ("a", "b")
    (subtract,) = _nodes_by_op(graph, "tensor_tensor")
    assert subtract.inputs == ["b", "a"]

    code = emit(graph, kernel_name="reversed_first_use")
    module = ast.parse(code)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "reversed_first_use"
    )
    assert [arg.arg for arg in function.args.args[:2]] == ["a", "b"]
    assert (
        ", b_tiles[0:TILE_M, tile_m, 0:BLOCK_N], "
        "a_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.subtract)"
    ) in code


# ---------------------------------------------------------------------------
# One-at-a-time (lazy) materialization
# ---------------------------------------------------------------------------


def test_materialization_is_lazy_and_one_at_a_time() -> None:
    adapter, top = _diamond_adapter()  # two selections
    snap = adapter.freeze_snapshot()
    root = adapter.resolve_handle(snap, top)

    calls: list[str] = []

    def counting_decode(s: Snapshot, enode: ENodeRef) -> Any:
        calls.append(enode.egg_fn or "")
        return decode_isa(s, enode)

    gen = iter_materialized_isa_graphs(
        snap, [root], {"x": {"shape": _SHAPE}}, decode_isa_enode=counting_decode
    )
    # Nothing is materialized before the consumer advances the generator.
    assert calls == []

    _first = next(gen)
    after_first = len(calls)
    assert after_first > 0

    # The second graph is not built until iteration advances again.
    _second = next(gen)
    assert len(calls) > after_first

    with pytest.raises(StopIteration):
        next(gen)


# ---------------------------------------------------------------------------
# End-to-end materialize + validate against a traced tensor graph
# ---------------------------------------------------------------------------


def test_materialized_graph_validates_against_tensor_graph() -> None:
    def kernel(x, y):
        return x * y

    G = build_graph_from_kernel(
        kernel, ("x", ("m", "k")), ("y", ("m", "k")), dim_sizes=_DIMS
    )
    tensor_adapter = EGraphAdapter("t")
    ingest = ingest_tensor_graph(tensor_adapter, G)
    tsnap = tensor_adapter.freeze_snapshot()
    (out_handle,) = ingest.output_handles
    out_class = tensor_adapter.resolve_handle(tsnap, out_handle)

    isa_adapter, L, status = lower_tensor_egraph(
        tsnap, [out_class], EGraphAdapter("isa"), ProofStore(), d=1, timeout=6000
    )
    assert status.status == "lowered"

    isa_snap = isa_adapter.freeze_snapshot()
    isa_root = isa_adapter.resolve_handle(isa_snap, L[out_class])

    graphs = list(
        iter_materialized_isa_graphs(
            isa_snap,
            [isa_root],
            ingest.input_metadata,
        )
    )
    assert len(graphs) >= 1
    for materialized in graphs:
        # Every op is an ISA op or an input; the graph is a valid DAG.
        assert _is_dag(materialized)
        assert "tensor_tensor" in {n.op for n in materialized.nodes}
        assert validate_materialized_graph(materialized) is None
        assert len(materialized.output_ids) == 1


# ---------------------------------------------------------------------------
# The two solver-free checks validation retains
# ---------------------------------------------------------------------------


def test_validation_rejects_graph_illegal_at_concrete_sizes() -> None:
    # Every op is ISA and correctly applied symbolically, but the reduce drops to
    # rank 1 at these concrete sizes and dma_transpose has no rank-1 shape.
    from axon.ir import Node, nuGraph
    from axon.isa_semantics import nl

    ensure_isa_semantics_registered()
    a = Node(id="a", op="input", inputs=[], attrs={"shape": _SHAPE})
    reduce_node = Node(
        id="red",
        op="tensor_reduce",
        inputs=["a"],
        attrs={"op": nl.add, "axis": 1, "keepdims": False},
    )
    transpose = Node(id="tr", op="dma_transpose", inputs=["red"], attrs={})
    G = nuGraph(nodes=[a, reduce_node, transpose], output_ids=("tr",), input_ids=("a",))

    reason = validate_materialized_graph(G)
    assert reason is not None
    assert "shape" in reason

    # The same graph with a rank-preserving reduce is accepted.
    keepdims = Node(
        id="red",
        op="tensor_reduce",
        inputs=["a"],
        attrs={"op": nl.add, "axis": 1, "keepdims": True},
    )
    legal = nuGraph(
        nodes=[
            a,
            keepdims,
            Node(id="tr", op="dma_transpose", inputs=["red"], attrs={}),
        ],
        output_ids=("tr",),
        input_ids=("a",),
    )
    assert validate_materialized_graph(legal) is None


# ---------------------------------------------------------------------------
# Emitter destination binding (M5, plan section 4g)
# ---------------------------------------------------------------------------


def test_emit_activation_reciprocal_string() -> None:
    # The Scalar reciprocal arrives as an activation(op=reciprocal) node; the
    # emitter must print nisa.activation(..., nl.reciprocal, ...) unchanged.
    from axon.codegen.context import EmitCtx
    from axon.codegen.ops import _emit_activation
    from axon.ir import Node
    from axon.isa_semantics import nl

    node = Node(
        id="activation_0",
        op="activation",
        inputs=["den"],
        attrs={"op": nl.reciprocal},
    )
    call = _emit_activation(EmitCtx(), node, {"den": "den_tile"})
    assert call == "nisa.activation({DST}, nl.reciprocal, den_tile)"


# ---------------------------------------------------------------------------
# Extraction completeness for the new Scalar realizations (M5)
# ---------------------------------------------------------------------------


def test_div_extraction_yields_reciprocal_and_emits() -> None:
    # A lowered div graph extracts to materialized ISA graphs, one of which
    # contains a `reciprocal` node. Each passes the standard validation and emits
    # valid NKI printing nisa.reciprocal(...).
    #
    # It must be the dedicated instruction, not activation(op=reciprocal): the
    # Activation engine's reciprocal returns 0.0 for operands at or above ~1e14
    # on trn2, so `_activation_pool_templates` no longer offers it.
    from axon.codegen import emit

    def kernel(num, den):
        return num / den

    G = build_graph_from_kernel(
        kernel, ("num", ("m", "k")), ("den", ("m", "k")), dim_sizes=_DIMS
    )
    tensor_adapter = EGraphAdapter("t")
    ingest = ingest_tensor_graph(tensor_adapter, G)
    tsnap = tensor_adapter.freeze_snapshot()
    (out_handle,) = ingest.output_handles
    out_class = tensor_adapter.resolve_handle(tsnap, out_handle)

    isa_adapter, L, status = lower_tensor_egraph(
        tsnap, [out_class], EGraphAdapter("isa"), ProofStore(), d=2, timeout=8000
    )
    assert status.status == "lowered"

    isa_snap = isa_adapter.freeze_snapshot()
    isa_root = isa_adapter.resolve_handle(isa_snap, L[out_class])

    graphs = list(
        iter_materialized_isa_graphs(isa_snap, [isa_root], ingest.input_metadata)
    )
    assert graphs

    recip_graphs = [g for g in graphs if any(n.op == "reciprocal" for n in g.nodes)]
    assert recip_graphs, "expected a reciprocal materialization"
    for g in recip_graphs:
        assert _is_dag(g)
        assert validate_materialized_graph(g) is None
        code = emit(g, kernel_name="div_recip")
        assert "nisa.reciprocal(" in code
        # The unsound Activation-engine form must not reappear.
        assert "nl.reciprocal" not in code

    # No materialization anywhere may route the reciprocal through activation.
    assert not [
        g
        for g in graphs
        if any(
            n.op == "activation"
            and getattr(n.attrs.get("op"), "name", None) == "reciprocal"
            for n in g.nodes
        )
    ]


def _activation_reduce_isa_adapter() -> tuple[EGraphAdapter, Any]:
    """Build an ISA e-graph whose root is
    ``activation_reduce(op=square, reduce_op=add)(h)`` directly, so the
    materialize + emit path is exercised without the slow fusion round."""
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    h = adapter.intern_expr(encode_isa_input("h", _SHAPE), "h").handle
    ar = adapter.intern_expr(
        encode_isa_enode(
            "activation_reduce",
            {"op": nl.square, "reduce_op": nl.add, "bias_const": None, "scale": 1.0},
            [h],
        ),
        "ar",
    ).handle
    return adapter, ar


def test_activation_reduce_extraction_materializes_reduce_only_graph() -> None:
    # The M4 saturation test proves the fused member exists; here we exercise
    # materialization by building the ISA term directly (the slow fusion round,
    # ~250s, is intentionally not run in this milestone).
    #
    # Emission is a separate contract, and a reduce-ONLY graph is refused: it
    # matches neither the single-reduce body (which wants reduce-then-apply) nor
    # the n-reduce body (which wants several reductions), and the flat body would
    # emit a degenerate (M, 1) kernel that computes the wrong result. The
    # emission of `nisa.activation_reduce` in a SUPPORTED shape is covered by
    # tests/test_activation_reduce_codegen.py.
    from axon.codegen import emit
    from axon.codegen.ops import UnsupportedEmission

    adapter, ar = _activation_reduce_isa_adapter()
    snap = adapter.freeze_snapshot()
    root = adapter.resolve_handle(snap, ar)

    graphs = list(iter_materialized_isa_graphs(snap, [root], {"h": {"shape": _SHAPE}}))
    assert len(graphs) == 1
    G = graphs[0]
    assert _is_dag(G)
    (node,) = _nodes_by_op(G, "activation_reduce")
    assert getattr(node.attrs.get("op"), "name", None) == "square"
    assert getattr(node.attrs.get("reduce_op"), "name", None) == "add"

    with pytest.raises(UnsupportedEmission, match="unrecognized reduction structure"):
        emit(G, kernel_name="sum_of_squares")


# ---------------------------------------------------------------------------
# Wall-clock stop behavior of the extraction stream (moved from the removed
# test_extraction_budget.py, rewritten against the direct deadline check).
# ---------------------------------------------------------------------------


def test_stream_preserves_graph_yielded_before_wall_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axon.egraph.extraction as extraction
    from axon.egraph.proof import WallClockExceeded

    adapter, top = _diamond_adapter()  # two selections
    snap = adapter.freeze_snapshot()
    root = adapter.resolve_handle(snap, top)

    clock = iter([0.0] * 1000)
    monkeypatch.setattr(extraction.time, "monotonic", lambda: next(clock))
    graphs = extraction.iter_materialized_isa_graphs(
        snap,
        [root],
        {"x": {"shape": _SHAPE}},
        deadline=10.0,
    )

    first = next(graphs)
    assert first is not None

    # The wall limit lapses only after the first graph reached the consumer.
    monkeypatch.setattr(extraction.time, "monotonic", lambda: 20.0)
    with pytest.raises(WallClockExceeded, match="extraction: wall_clock_seconds"):
        next(graphs)


def test_lapsed_deadline_ends_stream_before_any_graph() -> None:
    import time

    from axon.egraph.proof import WallClockExceeded

    adapter, top = _diamond_adapter()
    snap = adapter.freeze_snapshot()
    root = adapter.resolve_handle(snap, top)

    graphs = iter_materialized_isa_graphs(
        snap,
        [root],
        {"x": {"shape": _SHAPE}},
        deadline=time.monotonic() - 1.0,
    )
    with pytest.raises(WallClockExceeded):
        next(graphs)


def test_production_stream_stops_cleanly_after_extraction_wall_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import axon.egraph.pipeline as pipeline
    from axon.egraph.proof import WallClockExceeded
    from axon.ir import Node

    def input_graph(*node_ids: str) -> nuGraph:
        return nuGraph(
            nodes=[Node(nid, "input", [], {"shape": (1,)}) for nid in node_ids],
            output_ids=tuple(node_ids),
            input_ids=tuple(node_ids),
        )

    graph = input_graph("x")
    yielded = input_graph("result")

    monkeypatch.setattr(
        pipeline,
        "build_egraph_search",
        lambda *_args, **_kwargs: SimpleNamespace(
            lowering_status=SimpleNamespace(unrealized_outputs=[]),
            terminal_status=pipeline.SynthesisStatus(status="completed"),
            isa_snapshot=object(),
            isa_output_roots=[],
            input_metadata={},
            declared_input_ids=(),
            deadline=None,
        ),
    )

    def materialize(*_args: Any, **_kwargs: Any) -> Any:
        yield yielded
        raise WallClockExceeded("extraction")

    monkeypatch.setattr(pipeline, "iter_materialized_isa_graphs", materialize)

    graphs = pipeline.iter_synthesized_hw_graphs(graph)
    assert list(graphs) == [yielded]
    assert graphs.outcome.extraction_stage == "extraction"
    assert graphs.outcome.extraction_exhaustion == "extraction: wall_clock_seconds"


def _product_chain_adapter(k: int) -> tuple[EGraphAdapter, Any]:
    """A depth-k chain whose every class has two members: 2**k selections."""
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    prev = adapter.intern_expr(encode_isa_input("x", _SHAPE), "x").handle
    for i in range(k):
        r = adapter.intern_expr(_reciprocal(prev), f"r{i}").handle
        c = adapter.intern_expr(_tensor_copy(prev), f"c{i}").handle
        adapter.union_if_distinct(r, c)
        prev = r
    return adapter, prev


def _graph_identity(G: nuGraph) -> tuple[Any, ...]:
    return (
        tuple((n.id, n.op, tuple(n.inputs)) for n in G.nodes),
        G.output_ids,
        G.input_ids,
    )


def test_parallel_stream_matches_serial_order_and_content() -> None:
    adapter, out = _product_chain_adapter(6)  # 64 selections, window of 8
    snap = adapter.freeze_snapshot()
    root = adapter.resolve_handle(snap, out)
    meta = {"x": {"shape": _SHAPE}}

    serial = [
        _graph_identity(g)
        for g in iter_materialized_isa_graphs(snap, [root], meta, workers=1)
    ]
    parallel = [
        _graph_identity(g)
        for g in iter_materialized_isa_graphs(snap, [root], meta, workers=4)
    ]
    assert len(serial) == 64
    assert parallel == serial


def test_parallel_deadline_yields_ordered_prefix_then_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axon.egraph.extraction as extraction
    from axon.egraph.proof import WallClockExceeded

    adapter, out = _product_chain_adapter(6)  # 64 selections
    snap = adapter.freeze_snapshot()
    root = adapter.resolve_handle(snap, out)
    meta = {"x": {"shape": _SHAPE}}

    serial = [
        _graph_identity(g)
        for g in iter_materialized_isa_graphs(snap, [root], meta, workers=1)
    ]

    # The parent clock lapses only after the first graph reached the consumer.
    # Patch extraction's own time binding: freezing the global time.monotonic
    # would stall multiprocessing.connection.wait during pool shutdown.
    from types import SimpleNamespace

    now = {"t": 0.0}
    monkeypatch.setattr(extraction, "time", SimpleNamespace(monotonic=lambda: now["t"]))
    stream = extraction.iter_materialized_isa_graphs(
        snap, [root], meta, deadline=10.0, workers=2
    )
    yielded = [_graph_identity(next(stream))]
    now["t"] = 20.0
    with pytest.raises(WallClockExceeded, match="extraction: wall_clock_seconds"):
        for graph in stream:
            yielded.append(_graph_identity(graph))

    # Everything yielded is the exact ordered serial prefix, nothing dropped.
    assert 1 <= len(yielded) < len(serial)
    assert yielded == serial[: len(yielded)]


def test_realized_partial_snapshot_extracts_first_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import axon.egraph.pipeline as pipeline
    from axon.ir import Node

    graph = nuGraph(
        nodes=[Node("x", "input", [], {"shape": (1,)})],
        output_ids=("x",),
        input_ids=("x",),
    )
    yielded = nuGraph(
        nodes=[Node("result", "input", [], {"shape": (1,)})],
        output_ids=("result",),
        input_ids=("result",),
    )
    snapshot = object()
    monkeypatch.setattr(
        pipeline,
        "build_egraph_search",
        lambda *_args, **_kwargs: SimpleNamespace(
            lowering_status=SimpleNamespace(unrealized_outputs=[]),
            terminal_status=pipeline.SynthesisStatus(
                status="truncated",
                truncated_stage="isa_fusion",
                stop_reason="wall_clock_seconds",
            ),
            isa_snapshot=snapshot,
            isa_output_roots=[object()],
            input_metadata={},
            declared_input_ids=(),
            deadline=None,
        ),
    )

    def materialize(actual_snapshot: object, *_args: Any, **_kwargs: Any) -> Any:
        assert actual_snapshot is snapshot
        yield yielded

    monkeypatch.setattr(pipeline, "iter_materialized_isa_graphs", materialize)
    graphs = pipeline.iter_synthesized_hw_graphs(graph)

    assert list(graphs) == [yielded]
    assert graphs.outcome.status == "partial_extracted"
    assert graphs.outcome.stop_reason == "wall_clock_seconds"
    assert graphs.outcome.emitted_graph_count == 1
