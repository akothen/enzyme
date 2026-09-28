"""ISA instruction-fusion and shared-driver saturation tests (plan section 11).

Covers exactly:

* single-node zero-operation simplification (an ISA node equal to one of its
  own inputs fuses, so its class unions with the input class);
* propagation ENABLES fusion (a propagation union exposes a fusion occurrence
  proved only in a later round);
* fusion ENABLES propagation (a fusion union exposes a propagation occurrence
  proved only in a later round);
* the fixed point REQUIRES a shared no-change round (``saturate_isa`` keeps
  running while EITHER producer changed, returns ``fixed_point`` only after a
  complete round in which propagation and fusion both add nothing, and returns
  a specific ``truncated`` reason under a tight limit).

Every saturation test asserts either a complete no-change ``fixed_point`` or a
specific ``truncated`` reason, per plan section 13. The graphs are tiny (2x2
inputs, elementwise ops) so the proofs stay cheap on the remote.
"""

from __future__ import annotations

from typing import Any

import axon.egraph.fusion as fusion_module
import axon.egraph.propagation as propagation_module
from axon.egraph.adapter import (
    EClassRef,
    EGraphAdapter,
    Snapshot,
    reachable_classes,
)
from axon.egraph.analysis import (
    analyze_snapshot,
    decode_isa,
    ensure_isa_semantics_registered,
)
from axon.egraph.codec import (
    decode_isa_enode,
    encode_isa_enode,
    encode_isa_input,
)
from axon.egraph.context import SemanticContext
from axon.egraph.fusion import (
    OPTIONAL_FUSION_TIMEOUT_MS,
    encode_isa_candidate,
    is_eligible_isa_op,
    run_fusion_round,
)
from axon.egraph.isa import active_isa_classes, saturate_isa
from axon.egraph.proof import ProofStore
from axon.egraph.propagation import RoundResult, run_propagation_round
from axon.isa_semantics import EquivalenceVerdict, _operand_to_expr, nl

_TIMEOUT = 2500

# A stable synthetic tensor-output key: ``L`` maps declared tensor output
# classes to ISA handles, and the ISA driver only reads ``L[out]`` for each
# ``out`` in ``tensor_outputs``, so any hashable EClassRef works as the key.
_OUT_KEY = EClassRef(snapshot_id=-1, value="declared_output", sort="TensorExpr")


# ---------------------------------------------------------------------------
# Small ISA e-graph builders
# ---------------------------------------------------------------------------


def _seed_input(adapter: EGraphAdapter, source_id: str, shape: tuple[int, ...]) -> Any:
    return adapter.intern_expr(
        encode_isa_input(source_id, shape), provenance=source_id
    ).handle


def _tensor_scalar(
    adapter: EGraphAdapter, child: Any, op0: Any, const: float, name: str
) -> Any:
    return adapter.intern_expr(
        encode_isa_enode(
            "tensor_scalar", {"op0": op0, "operand0_const": const}, [child]
        ),
        provenance=name,
    ).handle


def _tensor_tensor(
    adapter: EGraphAdapter, lhs: Any, rhs: Any, op: Any, name: str
) -> Any:
    return adapter.intern_expr(
        encode_isa_enode("tensor_tensor", {"op": op}, [lhs, rhs]), provenance=name
    ).handle


def _tensor_reduce(
    adapter: EGraphAdapter, child: Any, op: Any, axis: int, keepdims: bool, name: str
) -> Any:
    return adapter.intern_expr(
        encode_isa_enode(
            "tensor_reduce", {"op": op, "axis": axis, "keepdims": keepdims}, [child]
        ),
        provenance=name,
    ).handle


def _broadcast(
    adapter: EGraphAdapter,
    data: Any,
    shape_source: Any,
    name: str,
) -> Any:
    return adapter.intern_expr(
        encode_isa_enode("broadcast", {}, [data, shape_source]),
        provenance=name,
    ).handle


def _analyze(snapshot: Snapshot, decode: Any) -> dict[Any, Any]:
    return analyze_snapshot(snapshot, decode)


def _active_from_handle(adapter: EGraphAdapter, out_handle: Any):
    """An ``active_classes`` closure reaching from one declared ISA output."""

    def active(snapshot: Snapshot) -> list[EClassRef]:
        ref = adapter.resolve_handle(snapshot, out_handle)
        return reachable_classes(snapshot, [ref])

    return active


def _prop_round(
    adapter: EGraphAdapter,
    store: ProofStore,
    *,
    guard_shared_producers: bool = True,
) -> RoundResult:
    return run_propagation_round(
        adapter,
        decode_isa,
        encode_isa_candidate,
        is_eligible_isa_op,
        store,
        analyze=_analyze,
        stage="isa_propagation",
        timeout=_TIMEOUT,
        guard_shared_producers=guard_shared_producers,
    )


def _fusion_round(
    adapter: EGraphAdapter, store: ProofStore, out_handle: Any
) -> RoundResult:
    return run_fusion_round(
        adapter,
        decode_isa,
        store,
        analyze=_analyze,
        active_classes=_active_from_handle(adapter, out_handle),
        timeout=_TIMEOUT,
    )


def _run_to_fixed(step, max_rounds: int = 8) -> int:
    """Run ``step`` until a round adds nothing; return the round count."""
    rounds = 0
    while rounds < max_rounds:
        result = step()
        rounds += 1
        if result.enodes_added == 0 and result.equalities_added == 0:
            return rounds
    raise AssertionError("did not reach a single-producer fixed point in time")


def test_optional_fusion_uses_100ms_timeout(monkeypatch) -> None:
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    x = _seed_input(adapter, "x", (2, 2))
    inner = _tensor_scalar(adapter, x, nl.multiply, 2.0, "inner")
    out = _tensor_scalar(adapter, inner, nl.add, 3.0, "out")
    observed: list[int] = []
    obligation_counts: list[int] = []

    def capture_timeout(*args, **kwargs):
        observed.append(kwargs["timeout"])
        obligation_counts.append(len(args[4]))
        return []

    monkeypatch.setattr(
        fusion_module,
        "prove_candidate_batch",
        capture_timeout,
    )
    run_fusion_round(
        adapter,
        decode_isa,
        ProofStore(),
        analyze=_analyze,
        active_classes=_active_from_handle(adapter, out),
    )

    assert observed
    assert set(observed) == {OPTIONAL_FUSION_TIMEOUT_MS}
    assert any(count > 0 for count in obligation_counts)


def test_fusion_uses_proof_workers(monkeypatch) -> None:
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    x = _seed_input(adapter, "x", (2, 2))
    inner = _tensor_scalar(adapter, x, nl.multiply, 2.0, "inner")
    out = _tensor_scalar(adapter, inner, nl.add, 3.0, "out")
    observed: list[int] = []
    obligation_counts: list[int] = []

    def capture_workers(*args, **kwargs):
        observed.append(kwargs["workers"])
        obligation_counts.append(len(args[4]))
        return []

    monkeypatch.setattr(
        fusion_module,
        "prove_candidate_batch",
        capture_workers,
    )
    monkeypatch.setattr(
        propagation_module,
        "prove_candidate_batch",
        lambda *args, **kwargs: [],
    )
    saturate_isa(
        adapter,
        {_OUT_KEY: out},
        [_OUT_KEY],
        ProofStore(),
        max_rounds=1,
        workers=3,
    )

    assert observed
    assert set(observed) == {3}
    assert any(count > 1 for count in obligation_counts)


def test_single_enode_fusion_uses_proof_workers(monkeypatch) -> None:
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    x = _seed_input(adapter, "x", (2, 2))
    out = _tensor_scalar(adapter, x, nl.multiply, 1.0, "out")
    observed: list[tuple[int, int]] = []

    def capture_batch(*args, **kwargs):
        observed.append((kwargs["workers"], len(args[4])))
        return []

    monkeypatch.setattr(fusion_module, "prove_candidate_batch", capture_batch)
    run_fusion_round(
        adapter,
        decode_isa,
        ProofStore(),
        analyze=_analyze,
        active_classes=_active_from_handle(adapter, out),
        workers=4,
    )

    assert observed == [(4, 1)]


def test_one_node_fusion_proves_in_worker_sized_waves(
    monkeypatch,
) -> None:
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    x = _seed_input(adapter, "x", (2, 2))
    inner = _tensor_scalar(adapter, x, nl.multiply, 2.0, "inner")
    out = _tensor_scalar(adapter, inner, nl.add, 3.0, "out")
    snapshot = adapter.freeze_snapshot()
    context = SemanticContext.build(adapter, snapshot, decode_isa, _analyze)
    out_class = adapter.resolve_handle(snapshot, out)
    occurrences = fusion_module.eligible_fusion_occurrences(
        snapshot,
        context.analyses,
        decode_isa,
        set(reachable_classes(snapshot, [out_class])),
    )
    occurrence = next(item for item in occurrences if item.consumer_class == out_class)
    store = ProofStore()
    batch_sizes: list[int] = []

    def enumerate_three(*args, **kwargs):
        for sketch in range(3):
            yield sketch, None

    def prove_batch(*args, **kwargs):
        candidates = args[4]
        batch_sizes.append(len(candidates))
        verdicts = []
        for index, _candidate in enumerate(candidates):
            verdict = EquivalenceVerdict(proved=True, stage="value", detail="proved")
            verdicts.append(verdict)
            kwargs["on_verdict"](index, verdict)
        return verdicts

    monkeypatch.setattr(
        fusion_module,
        "eligible_fusion_occurrences",
        lambda *args, **kwargs: [occurrence],
    )
    monkeypatch.setattr(fusion_module, "_external_arg_classes", lambda _: [])
    monkeypatch.setattr(
        fusion_module,
        "_allowed_fusion_hw_ops",
        lambda _: ["tensor_scalar"],
    )
    monkeypatch.setattr(
        fusion_module,
        "_build_general_simplification_pool",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(fusion_module, "iter_complete_sketches", enumerate_three)
    monkeypatch.setattr(
        fusion_module,
        "_sketch_to_term",
        lambda _sketch, formal_to_class: fusion_module.TermRef(
            next(iter(formal_to_class.values()))
        ),
    )
    monkeypatch.setattr(
        fusion_module,
        "isa_candidate_prefilter",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(fusion_module, "prove_candidate_batch", prove_batch)

    result = run_fusion_round(
        adapter,
        decode_isa,
        store,
        analyze=_analyze,
        active_classes=lambda _snapshot: [],
        context=context,
        workers=2,
    )

    assert result.truncated_reason is None
    assert batch_sizes == [2, 1]
    assert result.equalities_added == 1


# ---------------------------------------------------------------------------
# 1. Single-node zero-operation simplification
# ---------------------------------------------------------------------------


def test_single_node_zero_op_simplification_unions_input() -> None:
    """``tensor_scalar(x, *1.0)`` equals ``x``, so its class unions with x.

    The single-e-node zero-instruction producer proposes each direct child of
    an active ISA node and proves it; a multiply-by-one copy is an identity, so
    the node's class fuses into its input class.
    """
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    x = _seed_input(adapter, "x", (2, 2))
    node = _tensor_scalar(adapter, x, nl.multiply, 1.0, "copy")

    status = saturate_isa(
        adapter,
        {_OUT_KEY: node},
        [_OUT_KEY],
        ProofStore(),
        max_rounds=8,
        timeout=_TIMEOUT,
        workers=8,
    )

    assert status.status == "fixed_point"
    snapshot = adapter.freeze_snapshot()
    assert adapter.resolve_handle(snapshot, x) == adapter.resolve_handle(snapshot, node)


def test_identity_broadcast_unions_only_data_input() -> None:
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    data = _seed_input(adapter, "data", (2, 2))
    shape_source = _seed_input(adapter, "shape_source", (2, 2))
    node = _broadcast(adapter, data, shape_source, "identity_broadcast")

    status = saturate_isa(
        adapter,
        {_OUT_KEY: node},
        [_OUT_KEY],
        ProofStore(),
        max_rounds=8,
        timeout=_TIMEOUT,
        workers=8,
    )

    assert status.status == "fixed_point"
    snapshot = adapter.freeze_snapshot()
    assert adapter.resolve_handle(snapshot, data) == adapter.resolve_handle(
        snapshot,
        node,
    )
    assert adapter.resolve_handle(snapshot, shape_source) != adapter.resolve_handle(
        snapshot,
        node,
    )


def test_expanding_broadcast_does_not_union_data_input() -> None:
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    data = _seed_input(adapter, "data", (1, 2))
    shape_source = _seed_input(adapter, "shape_source", (3, 2))
    node = _broadcast(adapter, data, shape_source, "expanding_broadcast")

    status = saturate_isa(
        adapter,
        {_OUT_KEY: node},
        [_OUT_KEY],
        ProofStore(),
        max_rounds=8,
        timeout=_TIMEOUT,
        workers=8,
    )
    assert status.status == "fixed_point"
    snapshot = adapter.freeze_snapshot()
    assert adapter.resolve_handle(snapshot, data) != adapter.resolve_handle(
        snapshot,
        node,
    )


# ---------------------------------------------------------------------------
# 2. Fusion enables propagation
# ---------------------------------------------------------------------------


def test_fusion_enables_propagation() -> None:
    """A fusion union in one round exposes a propagation union in a later round.

    On ``add(ts(x, *2), ts(y, +0))`` propagation alone reaches its own fixed
    point. A single fusion round then rewrites the identity ``ts(y, +0)`` into
    its input (and fuses the scalar into a nested form), which exposes fresh
    propagation occurrences: the next propagation round proves unions that the
    propagation-only fixed point could not reach.

    The reactivation is visible only with the shared-producer guards off. Fusion
    grows the classes it touches, so every producer here gains a second successor
    e-node, and the guarded enumeration then refuses it. That is the documented
    trade of `guard_shared_producers` (it exists to stop shared-subexpression
    blowup on the qkv_cte kernels), and the guarded round is asserted below so the
    cost of the guard stays visible rather than looking like a dead rule.
    """
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    x = _seed_input(adapter, "x", (2, 2))
    y = _seed_input(adapter, "y", (2, 2))
    scaled = _tensor_scalar(adapter, x, nl.multiply, 2.0, "scaled")
    identity = _tensor_scalar(adapter, y, nl.add, 0.0, "identity")
    out = _tensor_tensor(adapter, scaled, identity, nl.add, "out")

    store = ProofStore()

    # Propagation reaches its own fixed point: an extra propagation round adds
    # nothing more on its own.
    _run_to_fixed(lambda: _prop_round(adapter, store, guard_shared_producers=False))
    settled = _prop_round(adapter, store, guard_shared_producers=False)
    assert settled.enodes_added == 0 and settled.equalities_added == 0

    # One fusion round adds unions...
    fusion = _fusion_round(adapter, store, out)
    assert fusion.equalities_added > 0

    # ...which expose propagation occurrences the prior fixed point could not:
    # the next propagation round proves new equalities.
    reactivated = _prop_round(adapter, store, guard_shared_producers=False)
    assert reactivated.equalities_added > 0


def test_guards_cost_the_fusion_reactivation() -> None:
    """Run the same graph entirely guarded: fusion then buys no new equality.

    This is the price of `guard_shared_producers`, which exists to stop
    shared-subexpression blowup on the qkv_cte kernels. Pinning it here keeps the
    trade visible: the guarded pipeline reaches a weaker fixed point than the
    unguarded one, rather than the guard looking like a no-op.
    """
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    x = _seed_input(adapter, "x", (2, 2))
    y = _seed_input(adapter, "y", (2, 2))
    scaled = _tensor_scalar(adapter, x, nl.multiply, 2.0, "scaled")
    identity = _tensor_scalar(adapter, y, nl.add, 0.0, "identity")
    out = _tensor_tensor(adapter, scaled, identity, nl.add, "out")

    store = ProofStore()
    _run_to_fixed(lambda: _prop_round(adapter, store))
    settled = _prop_round(adapter, store)
    assert settled.enodes_added == 0 and settled.equalities_added == 0

    fusion = _fusion_round(adapter, store, out)
    assert fusion.equalities_added > 0

    guarded = _prop_round(adapter, store)
    assert guarded.equalities_added == 0


# ---------------------------------------------------------------------------
# 3. Propagation enables fusion
# ---------------------------------------------------------------------------


def test_propagation_enables_fusion() -> None:
    """A propagation union in one round exposes a fusion union in a later round.

    On ``ts(add(ts(x, +0), y), *2)`` instruction fusion alone reaches its own
    fixed point. A single propagation round then distributes the outer scalar
    and moves operations into new adjacencies, which exposes fresh fusion
    occurrences: the next fusion round proves unions the fusion-only fixed point
    could not reach.
    """
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    x = _seed_input(adapter, "x", (2, 2))
    y = _seed_input(adapter, "y", (2, 2))
    identity = _tensor_scalar(adapter, x, nl.add, 0.0, "identity")
    summed = _tensor_tensor(adapter, identity, y, nl.add, "summed")
    out = _tensor_scalar(adapter, summed, nl.multiply, 2.0, "out")

    store = ProofStore()

    # Fusion reaches its own fixed point: an extra fusion round adds nothing.
    _run_to_fixed(lambda: _fusion_round(adapter, store, out))
    settled = _fusion_round(adapter, store, out)
    assert settled.enodes_added == 0 and settled.equalities_added == 0

    # One propagation round adds unions...
    propagation = _prop_round(adapter, store)
    assert propagation.equalities_added > 0

    # ...which expose fusion occurrences the prior fixed point could not: the
    # next fusion round proves new equalities.
    reactivated = _fusion_round(adapter, store, out)
    assert reactivated.equalities_added > 0


# ---------------------------------------------------------------------------
# 4. Fixed point requires a shared no-change round
# ---------------------------------------------------------------------------


def test_fixed_point_requires_shared_no_change_round() -> None:
    """``saturate_isa`` runs while EITHER producer changed and needs a full
    no-change round.

    On ``add(x, ts(y, +0))`` the two producers interleave: fusion rewrites the
    identity and propagation moves the surviving scalar, and neither settles in
    a single round. The driver therefore runs more than one round and reports
    ``fixed_point`` only after a complete round in which propagation and fusion
    both add nothing.
    """
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    x = _seed_input(adapter, "x", (2, 2))
    y = _seed_input(adapter, "y", (2, 2))
    identity = _tensor_scalar(adapter, y, nl.add, 0.0, "identity")
    out = _tensor_tensor(adapter, x, identity, nl.add, "out")

    status = saturate_isa(
        adapter,
        {_OUT_KEY: out},
        [_OUT_KEY],
        ProofStore(),
        max_rounds=12,
        timeout=_TIMEOUT,
        workers=8,
    )

    assert status.status == "fixed_point"
    # A single round could not have been enough: the two producers had to
    # interleave, so the shared driver ran more than one round before the
    # complete no-change round that established the fixed point.
    assert status.rounds > 1


def test_saturation_truncates_on_max_rounds() -> None:
    """A tight round limit stops saturation with a specific truncated reason."""
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    x = _seed_input(adapter, "x", (2, 2))
    y = _seed_input(adapter, "y", (2, 2))
    scaled = _tensor_scalar(adapter, x, nl.multiply, 2.0, "scaled")
    identity = _tensor_scalar(adapter, y, nl.add, 0.0, "identity")
    out = _tensor_tensor(adapter, scaled, identity, nl.add, "out")

    status = saturate_isa(
        adapter,
        {_OUT_KEY: out},
        [_OUT_KEY],
        ProofStore(),
        max_rounds=1,
        timeout=_TIMEOUT,
        workers=8,
    )

    assert status.status == "truncated"
    assert status.reason == "max_rounds"
    assert status.rounds == 1
    assert status.stage == "isa_saturation"


def test_saturation_truncates_on_wall_clock() -> None:
    """A zero wall-clock limit truncates before any round runs."""
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    x = _seed_input(adapter, "x", (2, 2))
    node = _tensor_scalar(adapter, x, nl.multiply, 1.0, "copy")

    status = saturate_isa(
        adapter,
        {_OUT_KEY: node},
        [_OUT_KEY],
        ProofStore(),
        wall_clock_seconds=0.0,
        timeout=_TIMEOUT,
        workers=8,
    )

    assert status.status == "truncated"
    assert status.reason == "wall_clock_seconds"


def test_active_isa_classes_reaches_only_from_declared_outputs() -> None:
    """Only classes reachable from ``L[out]`` are active; others are excluded."""
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    x = _seed_input(adapter, "x", (2, 2))
    out = _tensor_scalar(adapter, x, nl.multiply, 2.0, "out")
    # A disconnected island not reachable from the declared output.
    island_in = _seed_input(adapter, "z", (2, 2))
    island = _tensor_scalar(adapter, island_in, nl.add, 1.0, "island")

    snapshot = adapter.freeze_snapshot()
    active = set(active_isa_classes(adapter, snapshot, {_OUT_KEY: out}, [_OUT_KEY]))

    out_ref = adapter.resolve_handle(snapshot, out)
    x_ref = adapter.resolve_handle(snapshot, x)
    island_ref = adapter.resolve_handle(snapshot, island)
    assert out_ref in active
    assert x_ref in active
    assert island_ref not in active


# ---------------------------------------------------------------------------
# Sum-of-squares fuses to activation_reduce(square, add)
# ---------------------------------------------------------------------------


def test_sum_of_squares_fuses_to_activation_reduce_square() -> None:
    """RMS-style ``reduce_add(mul(h, h), axis=1, keepdims)`` gains a fused
    ``activation_reduce(op=square, reduce_op=add)`` member in the reduce class.

    The producer ``tensor_tensor(mul, h, h)`` and consumer
    ``tensor_reduce(add, keepdims)`` fuse through the one-node sketch search into
    the one-instruction Scalar form, unioned into the consumer class alongside
    the existing tensor_tensor+tensor_reduce realization. One fusion round is
    enough for this direct occurrence; the sum-of-squares reduction obligation
    needs a generous per-proof budget (the M3 proof test proves it at 30000).
    """
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    h = _seed_input(adapter, "h", (2, 2))
    sq = _tensor_tensor(adapter, h, h, nl.multiply, "sq")
    out = _tensor_reduce(adapter, sq, nl.add, axis=1, keepdims=True, name="out")

    store = ProofStore()
    result = run_fusion_round(
        adapter,
        decode_isa,
        store,
        analyze=_analyze,
        active_classes=_active_from_handle(adapter, out),
        timeout=30000,
    )
    assert result.equalities_added > 0

    snapshot = adapter.freeze_snapshot()
    out_ref = adapter.resolve_handle(snapshot, out)
    decoded = []
    for row in snapshot.members(out_ref):
        try:
            decoded.append(decode_isa_enode(snapshot, row))
        except Exception:
            continue
    # The existing reduce form survives.
    assert any(e.op == "tensor_reduce" for e in decoded)
    # The fused Scalar form appears with op=square, reduce_op=add.
    fused = [e for e in decoded if e.op == "activation_reduce"]
    assert any(
        _operand_to_expr(e.attrs.get("op")) == "square"
        and _operand_to_expr(e.attrs.get("reduce_op")) == "add"
        for e in fused
    ), [
        (
            _operand_to_expr(e.attrs.get("op")),
            _operand_to_expr(e.attrs.get("reduce_op")),
        )
        for e in fused
    ]
