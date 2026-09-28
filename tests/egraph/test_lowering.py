"""Compositional lowering tests (plan section 11 table).

Covers: one attempt per ready e-node; multiple members of one tensor class
realizing and unioning into one ISA class; a rejected candidate leaving the
ISA e-graph unchanged; an unresolved non-output class not being an error; and
a missing output realization being an error.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest
import z3

import axon.egraph.lowering as lowering
import axon.synthesizer as synthesizer
from axon.egraph.adapter import EGraphAdapter, Snapshot, reachable_classes
from axon.egraph.codec import (
    DecodedTensorENode,
    decode_isa_enode,
    encode_isa_enode,
    encode_isa_input,
)
from axon.egraph.lowering import lower_tensor_egraph
from axon.egraph.proof import ProofStore
from axon.egraph.tensor import ingest_tensor_graph
from axon.ir import build_graph_from_kernel
from axon.isa_semantics import (
    EquivalenceVerdict,
    SymExpr,
    SymTensor,
    _operand_to_expr,
    nl,
)
from axon.isa_semantics import engine as _isa_engine
from axon.synthesizer import SketchNode

# Small dim sizes keep the proofs cheap so remote runs stay fast.
_DIMS = {"m": 4, "k": 4, "n": 4}


def _ingest(
    kernel: Any, *specs: Any, dim_sizes: dict[str, int]
) -> tuple[EGraphAdapter, Any]:
    G = build_graph_from_kernel(kernel, *specs, dim_sizes=dim_sizes)
    adapter = EGraphAdapter("t")
    ingest = ingest_tensor_graph(adapter, G)
    return adapter, ingest


def _isa_member_ops(
    snapshot: Snapshot, adapter: EGraphAdapter, handle: Any
) -> list[str]:
    ref = adapter.resolve_handle(snapshot, handle)
    ops: list[str] = []
    for row in snapshot.members(ref):
        try:
            ops.append(decode_isa_enode(snapshot, row).op)
        except Exception:
            continue
    return ops


def _reachable_isa_enodes(
    snapshot: Snapshot, adapter: EGraphAdapter, handle: Any
) -> list[Any]:
    """Every decoded ISA e-node reachable from ``handle``'s e-class."""
    root = adapter.resolve_handle(snapshot, handle)
    decoded: list[Any] = []
    for ref in reachable_classes(snapshot, [root]):
        for row in snapshot.members(ref):
            try:
                decoded.append(decode_isa_enode(snapshot, row))
            except Exception:
                continue
    return decoded


def _isa_signature(snapshot: Snapshot) -> list[tuple[str, tuple[Any, ...]]]:
    """A stable structural signature of every ISA e-node row in a snapshot."""
    sig: list[tuple[str, tuple[Any, ...]]] = []
    for _ref, rows in snapshot.classes.items():
        for row in rows:
            sig.append((row.egg_fn or "", tuple(repr(a) for a in row.args)))
    sig.sort()
    return sig


def test_input_identity_seeding_uses_no_solver_dispatch() -> None:
    def kernel(x):
        return x

    tensor_adapter, ingest = _ingest(kernel, ("x", ("m", "k")), dim_sizes=_DIMS)
    snapshot = tensor_adapter.freeze_snapshot()
    (output_handle,) = ingest.output_handles
    output_class = tensor_adapter.resolve_handle(snapshot, output_handle)
    store = ProofStore()

    _isa, L, status = lower_tensor_egraph(
        snapshot,
        [output_class],
        EGraphAdapter("isa"),
        store,
        d=1,
        timeout=6000,
    )

    assert status.status == "lowered"
    assert output_class in L
    assert store.dispatch_count == 0


def test_input_identity_fast_path_requires_exact_registered_identity(
    monkeypatch,
) -> None:
    def kernel(x):
        return x

    tensor_adapter, ingest = _ingest(kernel, ("x", ("m", "k")), dim_sizes=_DIMS)
    snapshot = tensor_adapter.freeze_snapshot()
    (output_handle,) = ingest.output_handles
    output_class = tensor_adapter.resolve_handle(snapshot, output_handle)
    store = ProofStore()
    calls = 0

    def mismatched_input(source_id, shape):
        child = encode_isa_input(source_id, shape)
        return encode_isa_enode("tensor_copy", {"engine": _isa_engine.unknown}, [child])

    def reject_unknown(obligations, **_kwargs):
        from axon.egraph.proof_parallel import LocalProofResult

        nonlocal calls
        calls += 1
        (obligation,) = tuple(obligations)
        current = obligation.current
        candidate = obligation.candidate
        assert current.expr.op == "input"
        assert candidate.expr.op == "tensor_copy"
        assert candidate.expr.attrs
        assert len(candidate.expr.inputs) == 1
        return (
            LocalProofResult(
                sequence=0,
                proof_key=obligation.proof_key,
                verdict=EquivalenceVerdict(
                    proved=False,
                    stage="value",
                    detail="unknown",
                ),
                cache_hit=False,
                deduplicated=False,
                timeout=obligation.timeout,
            ),
        )

    monkeypatch.setattr(lowering, "encode_isa_input", mismatched_input)
    monkeypatch.setattr(lowering, "run_local_proof_batch", reject_unknown)

    _isa, L, status = lower_tensor_egraph(
        snapshot,
        [output_class],
        EGraphAdapter("isa"),
        store,
        d=1,
        timeout=6000,
    )

    assert status.status == "failed"
    assert output_class not in L
    assert calls == 1
    assert store.dispatch_count == 1


def test_input_codec_solver_error_rejects_candidate(monkeypatch) -> None:
    def kernel(x):
        return x

    tensor_adapter, ingest = _ingest(kernel, ("x", ("m", "k")), dim_sizes=_DIMS)
    snapshot = tensor_adapter.freeze_snapshot()
    output_class = tensor_adapter.resolve_handle(
        snapshot,
        ingest.output_handles[0],
    )
    store = ProofStore()

    def mismatched_input(source_id, shape):
        child = encode_isa_input(source_id, shape)
        return encode_isa_enode("tensor_copy", {}, [child])

    def solver_error(*_args, **_kwargs):
        raise z3.Z3Exception("malformed codec candidate")

    monkeypatch.setattr(lowering, "encode_isa_input", mismatched_input)
    monkeypatch.setattr(lowering, "run_local_proof_batch", solver_error)

    _isa, lowered, status = lower_tensor_egraph(
        snapshot,
        [output_class],
        EGraphAdapter("isa"),
        store,
        d=1,
    )

    assert status.status == "failed"
    assert output_class not in lowered
    assert store.dispatch_count == 1


# -- one attempt per ready e-node; single op lowers -------------------------


def test_single_op_lowers_once_per_enode() -> None:
    def kernel(x, y):
        return x * y

    tensor_adapter, ingest = _ingest(
        kernel, ("x", ("m", "k")), ("y", ("m", "k")), dim_sizes=_DIMS
    )
    snap = tensor_adapter.freeze_snapshot()
    (out_handle,) = ingest.output_handles
    out_class = tensor_adapter.resolve_handle(snap, out_handle)

    isa_adapter, L, status = lower_tensor_egraph(
        snap, [out_class], EGraphAdapter("isa"), ProofStore(), d=1, timeout=6000
    )

    assert status.status == "lowered"
    assert out_class in L
    # Exactly one non-input reachable e-node (the mul), attempted exactly once.
    assert status.attempted == 1
    isa_snap = isa_adapter.freeze_snapshot()
    assert "tensor_tensor" in _isa_member_ops(isa_snap, isa_adapter, L[out_class])


# -- two members of one tensor class union into one ISA class ---------------


def test_two_members_union_into_one_isa_class() -> None:
    def kernel(x, y):
        return x * y

    tensor_adapter, ingest = _ingest(
        kernel, ("x", ("m", "k")), ("y", ("m", "k")), dim_sizes=_DIMS
    )
    (out_handle,) = ingest.output_handles

    # Give the output tensor class a second member (the commuted product) and
    # union it in, so the class holds two tensor e-nodes over the same inputs.
    input_handles = {
        sid: h for sid, h in ingest.node_handles.items() if sid in ingest.input_metadata
    }
    assert len(input_handles) == 2
    ordered = list(input_handles.values())
    from axon.egraph import tensor_language as tl
    from axon.egraph.payload import encode_attrs

    commuted = tl.t_op2("mul", encode_attrs({}), ordered[1], ordered[0])
    commuted_res = tensor_adapter.intern_expr(commuted, provenance="commuted")
    tensor_adapter.union_if_distinct(commuted_res.handle, out_handle)

    snap = tensor_adapter.freeze_snapshot()
    out_class = tensor_adapter.resolve_handle(snap, out_handle)
    assert len(snap.members(out_class)) == 2

    isa_adapter, L, status = lower_tensor_egraph(
        snap, [out_class], EGraphAdapter("isa"), ProofStore(), d=1, timeout=6000
    )

    assert status.status == "lowered"
    assert out_class in L
    # Both mul e-nodes were attempted (one per e-node), and both lowered
    # programs unioned into the single ISA e-class mapped by L[out_class].
    assert status.attempted == 2
    assert status.proved_programs >= 2
    isa_snap = isa_adapter.freeze_snapshot()
    ops = _isa_member_ops(isa_snap, isa_adapter, L[out_class])
    assert ops.count("tensor_tensor") >= 2


# -- rejected candidate leaves the ISA e-graph unchanged --------------------


def test_rejected_candidate_leaves_isa_unchanged() -> None:
    # softmax needs more than one ISA instruction, so with d=1 every candidate
    # is rejected and nothing is interned beyond the seeded ISA inputs.
    def kernel(z):
        return z.softmax(axis=-1)

    tensor_adapter, ingest = _ingest(kernel, ("z", ("m", "k")), dim_sizes=_DIMS)
    snap = tensor_adapter.freeze_snapshot()
    (out_handle,) = ingest.output_handles
    out_class = tensor_adapter.resolve_handle(snap, out_handle)

    isa_adapter, L, status = lower_tensor_egraph(
        snap, [out_class], EGraphAdapter("isa"), ProofStore(), d=1, timeout=6000
    )

    # The output could not be realized, and the only ISA rows are the seeded
    # inputs: no rejected candidate interned any instruction e-node.
    assert status.status == "failed"
    isa_snap = isa_adapter.freeze_snapshot()
    sig = _isa_signature(isa_snap)
    egg_fns = {egg_fn for egg_fn, _ in sig}
    instruction_fns = {f for f in egg_fns if f.startswith("axI") and f != "axIInput"}
    assert instruction_fns == set(), instruction_fns


# -- unresolved non-output class is not an error ----------------------------


def test_unresolved_nonoutput_class_is_not_error() -> None:
    def kernel(x, y):
        return x * y

    tensor_adapter, ingest = _ingest(
        kernel, ("x", ("m", "k")), ("y", ("m", "k")), dim_sizes=_DIMS
    )
    (out_handle,) = ingest.output_handles
    input_handles = [
        h for sid, h in ingest.node_handles.items() if sid in ingest.input_metadata
    ]
    assert len(input_handles) == 2

    from axon.egraph import tensor_language as tl
    from axon.egraph.codec import encode_shape
    from axon.egraph.payload import encode_attrs

    # An alternative output member that depends on softmax(z): with d=1 the
    # softmax class never realizes, so this alternative e-node is never
    # attempted. It must not turn a realized output into a failure.
    z_expr = tl.t_input("z", encode_shape(("m", "k")))
    z_res = tensor_adapter.intern_expr(z_expr, provenance="z")
    sm_expr = tl.t_op1("softmax", encode_attrs({"axis": 1}), z_res.handle)
    sm_res = tensor_adapter.intern_expr(sm_expr, provenance="sm")
    alt_expr = tl.t_op2("mul", encode_attrs({}), sm_res.handle, input_handles[1])
    alt_res = tensor_adapter.intern_expr(alt_expr, provenance="alt")
    tensor_adapter.union_if_distinct(alt_res.handle, out_handle)

    snap = tensor_adapter.freeze_snapshot()
    out_class = tensor_adapter.resolve_handle(snap, out_handle)
    sm_class = tensor_adapter.resolve_handle(snap, sm_res.handle)

    isa_adapter, L, status = lower_tensor_egraph(
        snap, [out_class], EGraphAdapter("isa"), ProofStore(), d=1, timeout=6000
    )

    assert status.status == "lowered"
    assert out_class in L
    # The unlowerable non-output softmax class simply has no realization.
    assert sm_class not in L


# -- missing output realization is an error ---------------------------------


def test_missing_output_realization_fails() -> None:
    def kernel(z):
        return z.softmax(axis=-1)

    tensor_adapter, ingest = _ingest(kernel, ("z", ("m", "k")), dim_sizes=_DIMS)
    snap = tensor_adapter.freeze_snapshot()
    (out_handle,) = ingest.output_handles
    out_class = tensor_adapter.resolve_handle(snap, out_handle)

    isa_adapter, L, status = lower_tensor_egraph(
        snap, [out_class], EGraphAdapter("isa"), ProofStore(), d=1, timeout=6000
    )

    assert status.status == "failed"
    assert out_class in status.unrealized_outputs
    assert out_class not in L


# -- div lowers to both the divide form and the Scalar reciprocal form ------


def test_div_lowers_to_reciprocal_multiply() -> None:
    # A tensor div e-node's ISA class holds both the one-instruction
    # tensor_tensor(op=divide) form and the size-2 realization
    # tensor_tensor(multiply, num, reciprocal(den)).
    #
    # The reciprocal must be the dedicated `reciprocal` instruction, never
    # activation(op=reciprocal): the Activation engine's reciprocal returns 0.0
    # for operands at or above ~1e14 on trn2, so that form is unsound and is no
    # longer offered by `_activation_pool_templates`. The two-instruction
    # realization itself is unaffected.
    def kernel(num, den):
        return num / den

    tensor_adapter, ingest = _ingest(
        kernel, ("num", ("m", "k")), ("den", ("m", "k")), dim_sizes=_DIMS
    )
    snap = tensor_adapter.freeze_snapshot()
    (out_handle,) = ingest.output_handles
    out_class = tensor_adapter.resolve_handle(snap, out_handle)

    isa_adapter, L, status = lower_tensor_egraph(
        snap, [out_class], EGraphAdapter("isa"), ProofStore(), d=2, timeout=8000
    )

    assert status.status == "lowered"
    assert out_class in L
    isa_snap = isa_adapter.freeze_snapshot()
    enodes = _reachable_isa_enodes(isa_snap, isa_adapter, L[out_class])
    # The existing Vector divide form.
    assert any(
        e.op == "tensor_tensor" and _operand_to_expr(e.attrs.get("op")) == "divide"
        for e in enodes
    )
    # The two-instruction member: a `reciprocal` e-node feeding a multiply.
    assert any(e.op == "reciprocal" for e in enodes)
    assert any(
        e.op == "tensor_tensor" and _operand_to_expr(e.attrs.get("op")) == "multiply"
        for e in enodes
    )
    # No activation function may carry the reciprocal, and none leaks in at all.
    act_ops = {
        _operand_to_expr(e.attrs.get("op")) for e in enodes if e.op == "activation"
    }
    assert "reciprocal" not in act_ops, act_ops
    assert act_ops == set(), act_ops


def _proved_verdict() -> EquivalenceVerdict:
    return EquivalenceVerdict(proved=True, stage="proved", detail="")


def _rejected_verdict() -> EquivalenceVerdict:
    return EquivalenceVerdict(proved=False, stage="value", detail="test rejection")


def test_recipe_cache_reuses_alpha_renamed_dimensions() -> None:
    m, k, n = z3.Ints("alpha_m alpha_k alpha_n")
    x, y, z, unrelated = z3.Ints("renamed_x renamed_y renamed_z renamed_unrelated")
    first = [
        SymTensor("a", shape=(m, k)),
        SymTensor("b", shape=(k, n)),
    ]
    renamed = [
        SymTensor("c", shape=(x, y)),
        SymTensor("d", shape=(y, z)),
    ]

    assert synthesizer._make_lowering_cache_key("matmul", {}, first, 2) == (
        synthesizer._make_lowering_cache_key("matmul", {}, renamed, 2)
    )
    different_equality_classes = [
        SymTensor("e", shape=(x, y)),
        SymTensor("f", shape=(unrelated, z)),
    ]
    assert synthesizer._make_lowering_cache_key(
        "matmul", {}, first, 2
    ) != synthesizer._make_lowering_cache_key(
        "matmul", {}, different_equality_classes, 2
    )
    assert synthesizer._make_lowering_cache_key(
        "matmul", {}, first, 1
    ) != synthesizer._make_lowering_cache_key("matmul", {}, first, 2)


def test_recipe_key_preserves_repeated_operand_pattern() -> None:
    m, n = z3.Ints("pattern_m pattern_n")
    x = SymTensor("pattern_x", shape=(m, n))
    y = SymTensor("pattern_y", shape=(m, n))

    distinct = synthesizer._make_lowering_cache_key("mul", {}, [x, y], 1)
    repeated = synthesizer._make_lowering_cache_key("mul", {}, [x, x], 1)

    assert distinct != repeated


def test_canonical_shallow_prefix_precedes_depth_two_subtrees() -> None:
    m, k, n = z3.Ints("prefix_m prefix_k prefix_n")
    inputs = [
        SymTensor("prefix_lhs", shape=(m, k)),
        SymTensor("prefix_rhs", shape=(k, n)),
    ]
    recipes = synthesizer._build_complete_recipe_cache_entry("matmul", {}, inputs, 2)
    sketches = [synthesizer._denormalize_sketch(recipe, inputs) for recipe in recipes]
    assert all(sketch is not None for sketch in sketches)
    concrete = [sketch for sketch in sketches if sketch is not None]

    assert synthesizer._format_sketch(concrete[0]) == (
        "nc_matmul(transpose(IN:prefix_lhs), IN:prefix_rhs)"
    )
    noncanonical = concrete[1:]
    sizes = [sketch.hw_size() for sketch in noncanonical]
    assert sizes == sorted(sizes)
    first_depth_two = next(
        index
        for index, sketch in enumerate(noncanonical)
        if synthesizer._sketch_derivation_depth(sketch) >= 2
    )
    assert all(sketch.hw_size() <= 1 for sketch in noncanonical[:first_depth_two])


def test_passthrough_cache_requires_exact_proof_key() -> None:
    def kernel(x, y):
        return x.broadcast_like(y), x.broadcast_like(x)

    tensor_adapter, ingest = _ingest(
        kernel,
        ("x", ("m", "k")),
        ("y", ("m", "k")),
        dim_sizes=_DIMS,
    )
    snapshot = tensor_adapter.freeze_snapshot()
    outputs = [
        tensor_adapter.resolve_handle(snapshot, handle)
        for handle in ingest.output_handles
    ]
    store = ProofStore()

    _isa, _lowering, status = lower_tensor_egraph(
        snapshot,
        outputs,
        EGraphAdapter("isa"),
        store,
        d=1,
        workers=1,
    )

    direct = [
        (key, verdict)
        for key, verdict in store.local_cache.items()
        if verdict.proved
        and verdict.detail == "exact registered passthrough codec identity"
    ]
    assert status.status == "lowered"
    assert len(direct) == 2
    assert direct[0][0] != direct[1][0]


@pytest.mark.parametrize(
    "mutation",
    [
        "operation",
        "shape",
        "attributes",
        "source",
        "operand_order",
        "codec_rejection",
    ],
)
def test_passthrough_direct_identity_rejects_any_semantic_mutation(
    monkeypatch,
    mutation,
) -> None:
    def kernel(x, y):
        return x.broadcast_like(y)

    tensor_adapter, ingest = _ingest(
        kernel,
        ("x", ("m", "k")),
        ("y", ("m", "k")),
        dim_sizes=_DIMS,
    )
    snapshot = tensor_adapter.freeze_snapshot()
    output = tensor_adapter.resolve_handle(snapshot, ingest.output_handles[0])
    original_eval = lowering.isa_symbolic_eval
    normal_proofs = 0

    def mutate(op, inputs, attrs):
        if mutation == "codec_rejection":
            return lowering.SYM_EVAL_REJECTED
        candidate = original_eval(op, inputs, attrs)
        assert isinstance(candidate, SymTensor)
        expr = candidate.expr
        mutated_op = "mutated" if mutation == "operation" else expr.op
        mutated_shape = (
            (*expr.shape[:-1], expr.shape[-1] + 1)
            if mutation == "shape"
            else expr.shape
        )
        mutated_attrs = dict(expr.attrs)
        if mutation == "attributes":
            mutated_attrs["mutated"] = True
        mutated_inputs = list(expr.inputs)
        if mutation == "source":
            mutated_inputs[0] = SymTensor(
                "mutated_source",
                shape=tuple(inputs[0].shape),
            ).expr
        elif mutation == "operand_order":
            mutated_inputs.reverse()
        return SymTensor(
            "mutated_passthrough",
            expr=SymExpr(
                op=mutated_op,
                inputs=mutated_inputs,
                shape=mutated_shape,
                attrs=mutated_attrs,
                name=expr.name,
            ),
        )

    def reject(*_args, **_kwargs):
        nonlocal normal_proofs
        normal_proofs += 1
        return EquivalenceVerdict(
            proved=False,
            stage="value",
            detail="semantic mutation",
        )

    monkeypatch.setattr(lowering, "isa_symbolic_eval", mutate)
    monkeypatch.setattr(lowering, "prove_candidate", reject)

    _isa, lowered, status = lower_tensor_egraph(
        snapshot,
        [output],
        EGraphAdapter("isa"),
        ProofStore(),
        d=1,
        workers=1,
    )

    assert status.status == "failed"
    assert output not in lowered
    assert normal_proofs == 1


def test_dependency_chain_reaches_output_before_first_node_exhaustion(
    monkeypatch,
) -> None:
    def kernel(x, y, z):
        return (x * y) + z

    tensor_adapter, ingest = _ingest(
        kernel,
        ("x", ("m", "k")),
        ("y", ("m", "k")),
        ("z", ("m", "k")),
        dim_sizes=_DIMS,
    )
    snapshot = tensor_adapter.freeze_snapshot()
    output = tensor_adapter.resolve_handle(snapshot, ingest.output_handles[0])
    events: list[tuple[str, str]] = []
    original_candidates = lowering._candidate_sketches_for
    original_admit = lowering.admit_lowering

    def observed(task, *args, **kwargs):
        for candidate in original_candidates(task, *args, **kwargs):
            events.append(("candidate", task.decoded.op))
            yield candidate
        events.append(("exhausted", task.decoded.op))

    def observed_admit(*args, **kwargs):
        root = original_admit(*args, **kwargs)
        if args[2] == output:
            events.append(("realized", "output"))
        return root

    monkeypatch.setattr(lowering, "_candidate_sketches_for", observed)
    monkeypatch.setattr(lowering, "admit_lowering", observed_admit)
    monkeypatch.setattr(lowering, "_LOWERING_CANDIDATE_WAVE_SIZE", 1)

    _isa, lowered, status = lower_tensor_egraph(
        snapshot,
        [output],
        EGraphAdapter("isa"),
        ProofStore(),
        d=2,
        timeout=6000,
        workers=1,
    )

    assert status.status == "lowered"
    assert output in lowered
    output_realized = events.index(("realized", "output"))
    assert ("exhausted", "mul") not in events[:output_realized]


def _run_rejected_fair_schedule(
    monkeypatch,
) -> list[str]:
    def kernel(a, b, c, d):
        return a * b, c + d

    tensor_adapter, ingest = _ingest(
        kernel,
        ("a", ("m", "k")),
        ("b", ("m", "k")),
        ("c", ("m", "k")),
        ("d", ("m", "k")),
        dim_sizes=_DIMS,
    )
    snapshot = tensor_adapter.freeze_snapshot()
    outputs = [
        tensor_adapter.resolve_handle(snapshot, handle)
        for handle in ingest.output_handles
    ]
    events: list[str] = []

    def candidates(task, _target, formal_syms, _d, **_kwargs):
        inputs = [
            SketchNode.make_input(formal_syms[child])
            for child in task.decoded.child_classes
        ]
        operation = nl.multiply if task.decoded.op == "mul" else nl.add
        sketch = SketchNode.make_op(
            "tensor_tensor",
            inputs,
            {"op": operation},
        )
        for _ in range(5):
            events.append(task.decoded.op)
            yield sketch, None

    monkeypatch.setattr(lowering, "_candidate_sketches_for", candidates)
    monkeypatch.setattr(lowering, "_LOWERING_CANDIDATE_WAVE_SIZE", 2)
    monkeypatch.setattr(
        lowering,
        "prove_candidate",
        lambda *_args, **_kwargs: _rejected_verdict(),
    )

    _isa, _lowered, status = lower_tensor_egraph(
        snapshot,
        outputs,
        EGraphAdapter("isa"),
        ProofStore(),
        d=1,
        workers=1,
    )

    assert status.status == "failed"
    return events


def test_ready_nodes_advance_in_bounded_waves(monkeypatch) -> None:
    events = _run_rejected_fair_schedule(monkeypatch)

    assert events.count("mul") == 5
    assert events.count("add") == 5
    runs = [len(list(group)) for _operation, group in itertools.groupby(events)]
    assert max(runs) <= 2


def test_lowering_consumes_fast_completion_before_slow_peer(
    monkeypatch,
) -> None:
    def kernel(x, y):
        return x * y

    tensor_adapter, ingest = _ingest(
        kernel,
        ("x", ("m", "k")),
        ("y", ("m", "k")),
        dim_sizes=_DIMS,
    )
    snapshot = tensor_adapter.freeze_snapshot()
    output = tensor_adapter.resolve_handle(snapshot, ingest.output_handles[0])
    store = ProofStore()
    observed_admission_counts: list[int] = []
    admissions = 0
    original_admit = lowering.admit_lowering

    def counting_admit(*args, **kwargs):
        nonlocal admissions
        admissions += 1
        return original_admit(*args, **kwargs)

    def candidates(task, _target, formal_syms, _d, **_kwargs):
        inputs = [
            SketchNode.make_input(formal_syms[child])
            for child in task.decoded.child_classes
        ]
        sketch = SketchNode.make_op(
            "tensor_tensor",
            inputs,
            {"op": nl.multiply},
        )
        yield sketch, None
        yield sketch, None

    def complete_out_of_order(*_args, on_verdict=None, **_kwargs):
        verdicts = [_proved_verdict() for _ in _args[4]]
        if len(verdicts) == 1:
            if on_verdict is not None:
                on_verdict(0, verdicts[0])
            return verdicts
        baseline = admissions
        on_verdict(1, verdicts[1])
        observed_admission_counts.append(admissions - baseline)
        on_verdict(0, verdicts[0])
        observed_admission_counts.append(admissions - baseline)
        return verdicts

    monkeypatch.setattr(lowering, "admit_lowering", counting_admit)
    monkeypatch.setattr(lowering, "_candidate_sketches_for", candidates)
    monkeypatch.setattr(lowering, "prove_candidate_batch", complete_out_of_order)

    _isa, lowered, status = lower_tensor_egraph(
        snapshot,
        [output],
        EGraphAdapter("isa"),
        store,
        d=1,
        workers=2,
    )

    assert status.status == "lowered"
    assert output in lowered
    assert observed_admission_counts == [0, 1]


def test_unbounded_lowering_reaches_the_same_fixed_point() -> None:
    # The truncation drain must not change an unbounded run: the div e-node
    # still reaches its recorded exhaustive fixed point, no wave is reported as
    # truncated, and repeated runs stay identical.
    def kernel(num, den):
        return num / den

    def run() -> tuple[Any, ...]:
        tensor_adapter, ingest = _ingest(
            kernel, ("num", ("m", "k")), ("den", ("m", "k")), dim_sizes=_DIMS
        )
        snap = tensor_adapter.freeze_snapshot()
        (out_handle,) = ingest.output_handles
        out_class = tensor_adapter.resolve_handle(snap, out_handle)
        isa_adapter, lowered, status = lower_tensor_egraph(
            snap,
            [out_class],
            EGraphAdapter("isa"),
            ProofStore(),
            d=2,
            timeout=8000,
        )
        return (
            status,
            len(lowered),
            _isa_signature(isa_adapter.freeze_snapshot()),
        )

    first = run()
    second = run()

    status, realized, signature = first
    assert status.status == "lowered"
    assert status.unrealized_outputs == ()
    # The recorded exhaustive fixed point for this e-node. `proved_programs` fell
    # from 7 to 4 when `nl.reciprocal` left the activation pool: the three
    # activation(reciprocal) realizations are gone. `realized` is unchanged,
    # because the two-instruction form survives on the dedicated `reciprocal`
    # instruction, so no capability was lost.
    assert status.attempted == 1
    assert status.proved_programs == 4
    assert realized == 3
    # The recorded exhaustive fixed point under the generic ISA language: the same
    # expression rows, without the retired closed-sort attr rows. 17 -> 13 for the
    # same reason as `proved_programs` above: the activation(reciprocal) rows are
    # gone with that lowering option.
    assert len(signature) == 13
    assert first == second


def test_fair_lowering_is_deterministic(monkeypatch) -> None:
    first = _run_rejected_fair_schedule(monkeypatch)
    monkeypatch.undo()
    second = _run_rejected_fair_schedule(monkeypatch)

    assert first == second


def test_recipe_cache_preserves_bounded_candidate_set() -> None:
    m, n = z3.Ints("coverage_recipe_m coverage_recipe_n")
    inputs = [
        SymTensor("coverage_recipe_x", shape=(m, n)),
        SymTensor("coverage_recipe_y", shape=(m, n)),
    ]
    target = synthesizer._invoke_hw_op(
        "tensor_tensor",
        inputs,
        {"op": synthesizer.nl.multiply},
    )
    assert target is not None
    pool = synthesizer._build_synthesis_pool("mul", {}, inputs)

    uncached = {
        synthesizer._normalized_recipe_key(
            synthesizer._normalize_sketch(sketch, inputs)
        )
        for sketch, _candidate in synthesizer.iter_complete_sketches(
            target,
            pool,
            max_hw_size=2,
            input_syms=inputs,
        )
    }
    recipes = synthesizer._build_complete_recipe_cache_entry("mul", {}, inputs, 2)
    cached = {
        synthesizer._normalized_recipe_key(
            synthesizer._normalize_sketch(sketch, inputs)
        )
        for sketch, _candidate in synthesizer._iter_rebound_recipes(
            target,
            recipes,
            inputs,
        )
    }

    assert cached == uncached


def test_fair_lowering_preserves_bounded_eventual_coverage(
    monkeypatch,
) -> None:
    def occurrence(
        snapshot_id: int,
        shape_names: tuple[str, str],
    ) -> tuple[Any, SymTensor, dict[lowering.EClassRef, SymTensor], list[SymTensor]]:
        m, n = z3.Ints(" ".join(shape_names))
        inputs = [
            SymTensor(f"x{snapshot_id}", shape=(m, n)),
            SymTensor(f"y{snapshot_id}", shape=(m, n)),
        ]
        refs = tuple(
            lowering.EClassRef(snapshot_id, f"arg{index}", "TensorExpr")
            for index in range(2)
        )
        task = lowering._ENodeTask(
            index=0,
            owner=lowering.EClassRef(snapshot_id, "owner", "TensorExpr"),
            decoded=DecodedTensorENode(
                op="mul",
                child_classes=refs,
                attrs={},
            ),
        )
        formals = dict(zip(refs, inputs, strict=True))
        target = lowering._sym_expr_from_graph_node(
            lowering._target_node(task.decoded, inputs),
            inputs,
        )
        return task, target, formals, inputs

    first = occurrence(700, ("fair_m", "fair_n"))
    second = occurrence(701, ("renamed_m", "renamed_n"))
    expected = list(
        lowering._candidate_sketches_for(
            first[0],
            first[1],
            first[2],
            2,
        )
    )
    builds = 0
    original = lowering._iter_complete_recipe_cache_entry

    def count_builds(*args, **kwargs):
        nonlocal builds
        builds += 1
        yield from original(*args, **kwargs)

    monkeypatch.setattr(
        lowering,
        "_iter_complete_recipe_cache_entry",
        count_builds,
    )
    cache: dict[tuple[Any, ...], Any] = {}
    streams = [
        iter(
            lowering._candidate_sketches_for(
                task,
                target,
                formals,
                2,
                recipe_cache=cache,
            )
        )
        for task, target, formals, _inputs in (first, second)
    ]
    observed: list[list[Any]] = [[], []]
    active = {0, 1}
    while active:
        for index in sorted(active):
            for _ in range(3):
                try:
                    observed[index].append(next(streams[index]))
                except StopIteration:
                    active.remove(index)
                    break

    def keys(
        candidates: list[tuple[Any, SymTensor | None]],
        inputs: list[SymTensor],
    ) -> list[Any]:
        return [
            synthesizer._normalized_recipe_key(
                synthesizer._normalize_sketch(sketch, inputs)
            )
            for sketch, _candidate in candidates
        ]

    expected_keys = keys(expected, first[3])
    assert keys(observed[0], first[3]) == expected_keys
    assert keys(observed[1], second[3]) == expected_keys
    assert builds == 1
