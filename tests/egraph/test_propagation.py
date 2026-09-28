"""Propagation tests: design examples, repeated operands, degree coverage,
multi-consumer eligibility, later-round reactivation, and saturation status."""

from __future__ import annotations

from typing import Any

import pytest
import z3

import axon.egraph.isa as isa_module
import axon.egraph.propagation as propagation_module
from axon.egraph.adapter import EClassRef, EGraphAdapter, Snapshot
from axon.egraph.analysis import (
    EClassAnalysis,
    analyze_snapshot,
    decode_isa,
    decode_tensor,
    ensure_isa_semantics_registered,
)
from axon.egraph.codec import (
    decode_tensor_enode,
    encode_isa_enode,
    encode_isa_input,
)
from axon.egraph.context import SemanticContext
from axon.egraph.fusion import (
    encode_isa_candidate,
    is_eligible_isa_op,
    isa_candidate_prefilter,
)
from axon.egraph.isa import OPTIONAL_PROPAGATION_TIMEOUT_MS, saturate_isa
from axon.egraph.proof import (
    ProofStore,
    TermApp,
    TermRef,
    WallClockExceeded,
)
from axon.egraph.propagation import (
    Candidate,
    eligible_occurrences,
    iter_occurrence_candidates,
    run_propagation_round,
    swap_with_successor,
)
from axon.egraph.tensor import (
    encode_tensor_candidate,
    ingest_tensor_graph,
    is_eligible_tensor_op,
    saturate_tensor,
)
from axon.ir import build_graph_from_kernel
from axon.isa_semantics import EquivalenceVerdict, nl

_ISA_OUT_KEY = EClassRef(
    snapshot_id=-1,
    value="propagation_output",
    sort="TensorExpr",
)


def _ingest(
    kernel: Any, *specs: Any, dim_sizes: dict[str, int]
) -> tuple[EGraphAdapter, Any]:
    G = build_graph_from_kernel(kernel, *specs, dim_sizes=dim_sizes)
    adapter = EGraphAdapter("t")
    ingest = ingest_tensor_graph(adapter, G)
    return adapter, ingest


def _class_root_ops(
    snapshot: Snapshot, adapter: EGraphAdapter, handle: Any
) -> list[str]:
    ref = adapter.resolve_handle(snapshot, handle)
    ops: list[str] = []
    for row in snapshot.members(ref):
        try:
            ops.append(decode_tensor_enode(snapshot, row).op)
        except Exception:
            continue
    return ops


def _run_round(adapter: EGraphAdapter, store: ProofStore) -> Any:
    return run_propagation_round(
        adapter,
        decode_tensor,
        encode_tensor_candidate,
        is_eligible_tensor_op,
        store,
        analyze=lambda s, d: analyze_snapshot(s, d),
        stage="tensor_propagation",
    )


def _isa_matmul_transpose() -> tuple[EGraphAdapter, Any]:
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa_prefilter")
    x = adapter.intern_expr(encode_isa_input("x", (2, 3)), provenance="x").handle
    y = adapter.intern_expr(encode_isa_input("y", (3, 4)), provenance="y").handle
    matmul = adapter.intern_expr(
        encode_isa_enode("nc_matmul", {}, [x, y]), provenance="matmul"
    ).handle
    out = adapter.intern_expr(
        encode_isa_enode("nc_transpose", {}, [matmul]),
        provenance="transpose",
    ).handle
    return adapter, out


def test_isa_prefilter_rejects_without_solver_dispatch() -> None:
    adapter, _out = _isa_matmul_transpose()
    store = ProofStore()

    result = run_propagation_round(
        adapter,
        decode_isa,
        encode_isa_candidate,
        is_eligible_isa_op,
        store,
        analyze=lambda snapshot, decode: analyze_snapshot(snapshot, decode),
        stage="isa_propagation",
        candidate_prefilter=isa_candidate_prefilter,
    )

    assert result.candidates > 0
    assert store.dispatch_count == 0


def test_isa_prefilter_preserves_candidates_valid_under_guaranteed_facts() -> None:
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa_prefilter_facts")
    x = adapter.intern_expr(encode_isa_input("x", ("p", "f")), provenance="x").handle
    y = adapter.intern_expr(encode_isa_input("y", ("p", "g")), provenance="y").handle
    snapshot = adapter.freeze_snapshot()
    x_ref = adapter.resolve_handle(snapshot, x)
    y_ref = adapter.resolve_handle(snapshot, y)
    p, f, g = z3.Ints("p f g")
    guaranteed = (p > 0, f > 0, g > 0, f == g)
    analyses = {
        x_ref: EClassAnalysis((p, f), guaranteed),
        y_ref: EClassAnalysis((p, g), guaranteed),
    }
    current = TermApp.make(
        "tensor_scalar",
        {"op0": nl.add, "operand0_const": 0.0},
        (TermRef(x_ref),),
    )
    candidate = TermApp.make(
        "tensor_tensor",
        {"op": nl.add},
        (TermRef(x_ref), TermRef(y_ref)),
    )

    assert not isa_candidate_prefilter(
        snapshot,
        analyses,
        current,
        candidate,
    )


def test_optional_propagation_uses_50ms_timeout(monkeypatch) -> None:
    adapter, out = _isa_matmul_transpose()
    observed: list[int] = []

    def capture_timeout(*args, **kwargs):
        observed.append(kwargs["timeout"])
        return []

    monkeypatch.setattr(
        propagation_module,
        "prove_candidate_batch",
        capture_timeout,
    )
    # The real ISA prefilter rejects every candidate this fixture produces, and
    # empty proof chunks are no longer dispatched, so leave the prefilter off to
    # force candidates into prove_candidate_batch where the timeout is observed.
    monkeypatch.setattr(isa_module, "isa_candidate_prefilter", lambda *_args: False)
    saturate_isa(
        adapter,
        {_ISA_OUT_KEY: out},
        [_ISA_OUT_KEY],
        ProofStore(),
        max_rounds=1,
    )

    assert observed
    assert set(observed) == {OPTIONAL_PROPAGATION_TIMEOUT_MS}


# -- design examples --------------------------------------------------------


def test_scalar_commutes_through_matmul() -> None:
    """(a) MatMul(Multiply(X, scale), W) -> Multiply(MatMul(X, W), scale).

    The scale is a per-row tensor operand ``(m, 1)`` because the proof engine
    lifts a two-input multiply through the matmul reduction; that is the
    ``Multiply(X, scale)`` form the design example names.
    """

    def kernel(x, scale, w):
        return (x * scale) @ w

    adapter, ingest = _ingest(
        kernel,
        ("x", ("m", "k")),
        ("scale", ("m", 1)),
        ("w", ("k", "n")),
        dim_sizes={"m": 4, "k": 8, "n": 6},
    )
    store = ProofStore()
    status = saturate_tensor(adapter, store)
    assert status.status == "fixed_point"
    snap = adapter.freeze_snapshot()
    (out_handle,) = ingest.output_handles
    assert "mul" in _class_root_ops(snap, adapter, out_handle)
    assert any(
        verdict.used_reduction_fallback for verdict in store.local_cache.values()
    )


def test_distributivity_through_matmul() -> None:
    """(b) MatMul(Add(X, Y), W) -> Add(MatMul(X, W), MatMul(Y, W))."""

    def kernel(x, y, w):
        return (x + y) @ w

    adapter, ingest = _ingest(
        kernel,
        ("x", ("m", "k")),
        ("y", ("m", "k")),
        ("w", ("k", "n")),
        dim_sizes={"m": 4, "k": 8, "n": 6},
    )
    status = saturate_tensor(adapter)
    assert status.status == "fixed_point"
    snap = adapter.freeze_snapshot()
    (out_handle,) = ingest.output_handles
    assert "add" in _class_root_ops(snap, adapter, out_handle)


def test_transpose_permutes_matmul_operands() -> None:
    """(c) Transpose(MatMul(X, Y)) -> MatMul(Transpose(Y), Transpose(X))."""

    def kernel(x, y):
        return (x @ y).transpose()

    adapter, ingest = _ingest(
        kernel, ("x", ("m", "k")), ("y", ("k", "n")), dim_sizes={"m": 4, "k": 8, "n": 6}
    )
    status = saturate_tensor(adapter)
    assert status.status == "fixed_point"
    snap = adapter.freeze_snapshot()
    (out_handle,) = ingest.output_handles
    assert "matmul" in _class_root_ops(snap, adapter, out_handle)


# -- repeated operands ------------------------------------------------------


def test_repeated_operand_positions_stay_separate() -> None:
    """One e-class in two operand positions shares one symbol but is two
    selectable positions, and distributivity still applies."""

    def kernel(x, w):
        return (x + x) @ w

    adapter, ingest = _ingest(
        kernel, ("x", ("m", "k")), ("w", ("k", "n")), dim_sizes={"m": 4, "k": 8, "n": 6}
    )
    snap = adapter.freeze_snapshot()
    analyses = analyze_snapshot(snap, decode_tensor)
    occurrences = eligible_occurrences(
        snap, analyses, decode_tensor, is_eligible_tensor_op
    )
    # The add producer feeds the matmul at one position, but its two operands
    # (both the same X class) remain two indexed selectable positions.
    add_occ = next(o for o in occurrences if o.producer_op == "add")
    assert len(add_occ.producer_children) == 2
    assert add_occ.producer_children[0] == add_occ.producer_children[1]

    status = saturate_tensor(adapter)
    assert status.status == "fixed_point"
    snap = adapter.freeze_snapshot()
    (out_handle,) = ingest.output_handles
    assert "add" in _class_root_ops(snap, adapter, out_handle)


# -- all candidates at the first successful degree --------------------------


def test_all_candidates_at_first_successful_degree_are_tried() -> None:
    """Two provable degree-2 candidates (both permutations of Add) both union.

    Scaling a sum distributes, and Add is symbolically commutative, so both
    Add(mul(X), mul(Y)) and Add(mul(Y), mul(X)) prove at degree two.
    """

    def kernel(x, y):
        return (x + y) * 2.0

    adapter, ingest = _ingest(
        kernel, ("x", ("m", "k")), ("y", ("m", "k")), dim_sizes={"m": 4, "k": 8}
    )
    store = ProofStore()
    result = _run_round(adapter, store)
    assert result.equalities_added >= 2
    snap = adapter.freeze_snapshot()
    (out_handle,) = ingest.output_handles
    ref = adapter.resolve_handle(snap, out_handle)
    add_members = [row for row in snap.members(ref) if _safe_op(snap, row) == "add"]
    assert len(add_members) >= 2


def _safe_op(snapshot: Snapshot, row: Any) -> str | None:
    try:
        return decode_tensor_enode(snapshot, row).op
    except Exception:
        return None


# -- multi-consumer producers -----------------------------------------------


def test_shared_producer_is_guarded_out_but_eligible_unguarded() -> None:
    """A producer feeding two consumers is NOT eligible under the default guards.

    `eligible_occurrences` refuses a producer whose e-class has more than one
    successor e-node, because propagating it duplicates a shared subexpression and
    each duplicate seeds further propagation (measured on the qkv_cte kernels).
    With the guards off the occurrences come back, and both consumers receive the
    propagated alternative, so this test pins both sides of that trade.
    """

    def kernel(x, scale, w, v):
        a = x * scale
        return a @ w, a @ v

    adapter, ingest = _ingest(
        kernel,
        ("x", ("m", "k")),
        ("scale", ("m", 1)),
        ("w", ("k", "n")),
        ("v", ("k", "n")),
        dim_sizes={"m": 4, "k": 8, "n": 6},
    )
    snap = adapter.freeze_snapshot()
    analyses = analyze_snapshot(snap, decode_tensor)

    def mul_into_matmul(occurrences):
        return [
            o
            for o in occurrences
            if o.producer_op == "mul" and o.consumer_op == "matmul"
        ]

    guarded = eligible_occurrences(snap, analyses, decode_tensor, is_eligible_tensor_op)
    assert mul_into_matmul(guarded) == []

    unguarded = eligible_occurrences(
        snap,
        analyses,
        decode_tensor,
        is_eligible_tensor_op,
        guard_shared_producers=False,
    )
    # The shared producer feeds two distinct matmul consumers.
    assert len(mul_into_matmul(unguarded)) == 2

    status = saturate_tensor(adapter, guard_shared_producers=False)
    assert status.status == "fixed_point"
    snap = adapter.freeze_snapshot()
    # Unguarded, both outputs get a propagated mul-rooted alternative.
    for out_handle in ingest.output_handles:
        assert "mul" in _class_root_ops(snap, adapter, out_handle)
    # The original producer member is never removed: its class still holds the
    # scalar mul over the X input.
    mul_id = next(nid for nid in ingest.node_handles if nid.startswith("mul"))
    assert "mul" in _class_root_ops(snap, adapter, ingest.node_handles[mul_id])


# -- later-round reactivation ----------------------------------------------


def test_later_round_reactivation() -> None:
    """A union in round one exposes an occurrence handled in round two.

    Round one turns MatMul(mul(X), W) into mul(MatMul(X, W)); that new scalar
    mul under the Transpose consumer is a fresh occurrence only reachable in
    round two.
    """

    def kernel(x, scale, w):
        return ((x * scale) @ w).transpose()

    adapter, _ = _ingest(
        kernel,
        ("x", ("m", "k")),
        ("scale", ("m", 1)),
        ("w", ("k", "n")),
        dim_sizes={"m": 4, "k": 8, "n": 6},
    )
    store = ProofStore()
    round1 = _run_round(adapter, store)
    assert round1.equalities_added > 0
    round2 = _run_round(adapter, store)
    # Round two considers strictly more occurrences and adds equalities that
    # round one could not, because they rest on e-nodes round one introduced.
    assert round2.occurrences > round1.occurrences
    assert round2.equalities_added > 0


def test_semi_naive_full_scan_and_parallel_profiles_match() -> None:
    def kernel(x, scale, w):
        return ((x * scale) @ w).transpose()

    def run(semi_naive: bool, workers: int) -> tuple[Any, ...]:
        adapter, ingest = _ingest(
            kernel,
            ("x", ("m", "k")),
            ("scale", ("m", 1)),
            ("w", ("k", "n")),
            dim_sizes={"m": 4, "k": 8, "n": 6},
        )
        status = saturate_tensor(
            adapter,
            ProofStore(),
            max_rounds=3,
            semi_naive=semi_naive,
            workers=workers,
        )
        snapshot = adapter.freeze_snapshot()
        output_ops = tuple(
            tuple(sorted(_class_root_ops(snapshot, adapter, handle)))
            for handle in ingest.output_handles
        )
        return (
            status.status,
            status.reason,
            status.enodes_added,
            status.equalities_added,
            output_ops,
        )

    oracle = run(False, 1)
    serial = run(True, 1)
    assert serial == oracle
    parallel = run(True, 2)
    assert parallel == oracle


def test_chunked_completion_matches_unbounded_oracle() -> None:
    def kernel(x, scale, w):
        return ((x * scale) @ w).transpose()

    def run(proof_chunk_size: int | None) -> tuple[Any, ...]:
        adapter, ingest = _ingest(
            kernel,
            ("x", ("m", "k")),
            ("scale", ("m", 1)),
            ("w", ("k", "n")),
            dim_sizes={"m": 4, "k": 8, "n": 6},
        )
        store = ProofStore()
        worklist = propagation_module.PropagationWorklist()
        total_enodes = 0
        total_equalities = 0
        round_count = 0
        for _ in range(20):
            round_count += 1
            result = run_propagation_round(
                adapter,
                decode_tensor,
                encode_tensor_candidate,
                is_eligible_tensor_op,
                store,
                analyze=lambda snapshot, decode: analyze_snapshot(snapshot, decode),
                stage="tensor_propagation",
                worklist=worklist,
                workers=1,
                proof_chunk_size=proof_chunk_size,
            )
            assert result.complete
            total_enodes += result.enodes_added
            total_equalities += result.equalities_added
            if result.enodes_added == 0 and result.equalities_added == 0:
                break
        else:
            pytest.fail("tensor propagation did not reach a fixed point")
        snapshot = adapter.freeze_snapshot()
        output_ops = tuple(
            tuple(sorted(_class_root_ops(snapshot, adapter, handle)))
            for handle in ingest.output_handles
        )
        return (
            round_count,
            total_enodes,
            total_equalities,
            output_ops,
        )

    oracle = run(None)
    assert oracle[0] >= 3
    assert run(1) == oracle
    assert run(64) == oracle


# -- truncation and fixed point ---------------------------------------------


def test_saturation_reports_fixed_point_on_saturating_graph() -> None:
    def kernel(x, y):
        return x + y

    adapter, _ = _ingest(
        kernel, ("x", ("m", "k")), ("y", ("m", "k")), dim_sizes={"m": 4, "k": 8}
    )
    status = saturate_tensor(adapter)
    assert status.status == "fixed_point"
    assert status.equalities_added == 0


def test_saturation_truncates_on_max_rounds() -> None:
    def kernel(x, w):
        return (x * 2.0) @ w

    adapter, _ = _ingest(
        kernel, ("x", ("m", "k")), ("w", ("k", "n")), dim_sizes={"m": 4, "k": 8, "n": 6}
    )
    status = saturate_tensor(adapter, max_rounds=0)
    assert status.status == "truncated"
    assert status.reason == "max_rounds"
    assert status.rounds == 0


def test_saturation_truncates_on_wall_clock() -> None:
    def kernel(x, w):
        return (x * 2.0) @ w

    adapter, _ = _ingest(
        kernel, ("x", ("m", "k")), ("w", ("k", "n")), dim_sizes={"m": 4, "k": 8, "n": 6}
    )
    status = saturate_tensor(adapter, wall_clock_seconds=0.0)
    assert status.status == "truncated"
    assert status.reason == "wall_clock_seconds"


def test_wall_clock_truncation_skips_expensive_admission(
    monkeypatch,
) -> None:
    def kernel(x, w):
        return (x * 2.0) @ w

    adapter, _ = _ingest(
        kernel,
        ("x", ("m", "k")),
        ("w", ("k", "n")),
        dim_sizes={"m": 4, "k": 8, "n": 6},
    )
    proof_calls = 0

    def expire_during_proof(*_args, **_kwargs):
        nonlocal proof_calls
        proof_calls += 1
        raise WallClockExceeded("tensor_propagation")

    monkeypatch.setattr(
        propagation_module,
        "prove_candidate_batch",
        expire_during_proof,
    )
    monkeypatch.setattr(
        propagation_module,
        "_admit_propagation_batch",
        lambda *_args, **_kwargs: pytest.fail(
            "wall-clock truncation must not start an admission batch"
        ),
    )

    result = run_propagation_round(
        adapter,
        decode_tensor,
        encode_tensor_candidate,
        is_eligible_tensor_op,
        ProofStore(),
        analyze=lambda snapshot, decode: analyze_snapshot(snapshot, decode),
        stage="tensor_propagation",
    )

    assert proof_calls == 1
    assert result.truncated_reason == "wall_clock_seconds"
    assert result.equalities_added == 0


def test_wall_clock_truncation_preserves_admitted_proof_prefix(
    monkeypatch,
) -> None:
    def kernel(x, w):
        return (x * 2.0) @ w

    adapter, _ = _ingest(
        kernel,
        ("x", ("m", "k")),
        ("w", ("k", "n")),
        dim_sizes={"m": 4, "k": 8, "n": 6},
    )
    snapshot = adapter.freeze_snapshot()
    context = SemanticContext.build(
        adapter,
        snapshot,
        decode_tensor,
        lambda current, decode: analyze_snapshot(current, decode),
    )
    occurrence = eligible_occurrences(
        snapshot,
        context.analyses,
        decode_tensor,
        is_eligible_tensor_op,
    )[0]
    candidate = next(iter_occurrence_candidates(occurrence))
    store = ProofStore()
    admitted_sequences: list[tuple[int, int]] = []

    monkeypatch.setattr(
        propagation_module,
        "eligible_occurrences",
        lambda *_args, **_kwargs: [occurrence],
    )

    def candidate_groups(*_args, **_kwargs):
        yield [candidate] * 9

    prove_calls = 0

    def prove_batch(*args, **kwargs):
        nonlocal prove_calls
        prove_calls += 1
        if prove_calls > 1:
            # The wall limit lapses before the second chunk can dispatch.
            raise WallClockExceeded("tensor_propagation")
        verdicts = []
        for index, _candidate in enumerate(args[4]):
            verdict = EquivalenceVerdict(proved=True, stage="proved")
            verdicts.append(verdict)
            kwargs["on_verdict"](index, verdict)
        return verdicts

    def admit_prefix(_adapter, _context, _encode, proved):
        admitted_sequences.extend(item.sequence for item in proved)
        return len(proved), len(proved)

    monkeypatch.setattr(
        propagation_module,
        "_candidate_degree_groups",
        candidate_groups,
    )
    monkeypatch.setattr(
        propagation_module,
        "prove_candidate_batch",
        prove_batch,
    )
    monkeypatch.setattr(
        propagation_module,
        "_admit_propagation_batch",
        admit_prefix,
    )

    result = run_propagation_round(
        adapter,
        decode_tensor,
        encode_tensor_candidate,
        is_eligible_tensor_op,
        store,
        analyze=lambda current, decode: analyze_snapshot(current, decode),
        stage="tensor_propagation",
        context=context,
        workers=2,
        proof_chunk_size=8,
    )

    assert result.truncated_reason == "wall_clock_seconds"
    assert result.enodes_added == 8
    assert result.equalities_added == 8
    assert admitted_sequences == [(0, index) for index in range(8)]


def test_proof_chunks_do_not_lose_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def kernel(x, w):
        return (x * 2.0) @ w

    adapter, _ = _ingest(
        kernel,
        ("x", ("m", "k")),
        ("w", ("k", "n")),
        dim_sizes={"m": 4, "k": 8, "n": 6},
    )
    snapshot = adapter.freeze_snapshot()
    context = SemanticContext.build(
        adapter,
        snapshot,
        decode_tensor,
        lambda current, decode: analyze_snapshot(current, decode),
    )
    occurrence = eligible_occurrences(
        snapshot,
        context.analyses,
        decode_tensor,
        is_eligible_tensor_op,
    )[0]
    candidate = next(iter_occurrence_candidates(occurrence))
    candidates = [
        Candidate(
            degree=1 if index < 3 else 2,
            current=candidate.current,
            new=TermApp.make(
                "ordered_candidate",
                {"index": index},
                (candidate.new,),
            ),
        )
        for index in range(5)
    ]
    submitted: list[int] = []
    submitted_pairs: list[tuple[Any, Any]] = []

    monkeypatch.setattr(
        propagation_module,
        "eligible_occurrences",
        lambda *_args, **_kwargs: [occurrence],
    )
    monkeypatch.setattr(
        propagation_module,
        "_candidate_degree_groups",
        lambda *_args, **_kwargs: iter((candidates[:3], candidates[3:])),
    )

    def reject_batch(*args: Any, **kwargs: Any) -> list[EquivalenceVerdict]:
        submitted.append(len(args[4]))
        submitted_pairs.extend(args[4])
        verdicts: list[EquivalenceVerdict] = []
        for index in range(len(args[4])):
            verdict = EquivalenceVerdict(proved=False, stage="value")
            verdicts.append(verdict)
            kwargs["on_verdict"](index, verdict)
        return verdicts

    monkeypatch.setattr(
        propagation_module,
        "prove_candidate_batch",
        reject_batch,
    )
    worklist = propagation_module.PropagationWorklist()
    result = run_propagation_round(
        adapter,
        decode_tensor,
        encode_tensor_candidate,
        is_eligible_tensor_op,
        ProofStore(),
        analyze=lambda current, decode: analyze_snapshot(current, decode),
        stage="tensor_propagation",
        context=context,
        worklist=worklist,
        proof_chunk_size=2,
    )

    assert submitted == [2, 1, 2]
    assert submitted_pairs == [(ordered.current, ordered.new) for ordered in candidates]
    assert result.candidates == 5
    assert result.complete
    assert result.queued_continuations == 0
    assert len(worklist.processed) == 1


def test_delivered_propagation_proof_is_admitted_on_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def kernel(x, w):
        return (x * 2.0) @ w

    adapter, _ = _ingest(
        kernel,
        ("x", ("m", "k")),
        ("w", ("k", "n")),
        dim_sizes={"m": 4, "k": 8, "n": 6},
    )
    snapshot = adapter.freeze_snapshot()
    context = SemanticContext.build(
        adapter,
        snapshot,
        decode_tensor,
        lambda current, decode: analyze_snapshot(current, decode),
    )
    occurrence = eligible_occurrences(
        snapshot,
        context.analyses,
        decode_tensor,
        is_eligible_tensor_op,
    )[0]
    candidate = next(iter_occurrence_candidates(occurrence))
    worklist = propagation_module.PropagationWorklist()
    admitted: list[tuple[int, int]] = []
    calls = 0

    monkeypatch.setattr(
        propagation_module,
        "eligible_occurrences",
        lambda *_args, **_kwargs: [occurrence],
    )
    monkeypatch.setattr(
        propagation_module,
        "_candidate_degree_groups",
        lambda *_args, **_kwargs: iter(([candidate] * 4,)),
    )
    monkeypatch.setattr(
        propagation_module,
        "_admit_propagation_batch",
        lambda _adapter, _context, _encode, proved: (
            admitted.extend(item.sequence for item in proved) or len(proved),
            len(proved),
        ),
    )

    def expire_second_chunk(*args: Any, **kwargs: Any) -> list[EquivalenceVerdict]:
        nonlocal calls
        calls += 1
        verdicts = [
            EquivalenceVerdict(proved=True, stage="proved")
            for _index in range(len(args[4]))
        ]
        completed = verdicts if calls == 1 else verdicts[:1]
        for index, verdict in enumerate(completed):
            kwargs["on_verdict"](index, verdict)
        if calls == 2:
            raise WallClockExceeded("tensor_propagation")
        return verdicts

    monkeypatch.setattr(
        propagation_module,
        "prove_candidate_batch",
        expire_second_chunk,
    )
    first = run_propagation_round(
        adapter,
        decode_tensor,
        encode_tensor_candidate,
        is_eligible_tensor_op,
        ProofStore(),
        analyze=lambda current, decode: analyze_snapshot(current, decode),
        stage="tensor_propagation",
        context=context,
        worklist=worklist,
        proof_chunk_size=2,
    )

    assert first.truncated_reason == "wall_clock_seconds"
    assert not first.complete
    assert first.queued_continuations == 1
    assert admitted == [(0, 0), (0, 1), (0, 2)]
    assert not worklist.processed


# -- candidate construction unit checks -------------------------------------


def test_swap_with_successor_returns_distinct_candidates() -> None:
    def kernel(x, y):
        return (x @ y).transpose()

    adapter, _ = _ingest(
        kernel, ("x", ("m", "k")), ("y", ("k", "n")), dim_sizes={"m": 4, "k": 8, "n": 6}
    )
    snap = adapter.freeze_snapshot()
    analyses = analyze_snapshot(snap, decode_tensor)
    occurrences = eligible_occurrences(
        snap, analyses, decode_tensor, is_eligible_tensor_op
    )
    occ = next(o for o in occurrences if o.producer_op == "matmul")
    # Degree two over both matmul operands yields two distinct permutations.
    cands = swap_with_successor(occ, frozenset({0, 1}))
    assert len(cands) == 2
    assert cands[0].new != cands[1].new
    # Candidates iterate ascending by degree.
    degrees = [c.degree for c in iter_occurrence_candidates(occ)]
    assert degrees == sorted(degrees)
