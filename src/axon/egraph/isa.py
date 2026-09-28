"""Run operator propagation and instruction fusion to an ISA fixed point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from axon.egraph.adapter import (
    EClassRef,
    EGraphAdapter,
    EGraphAdapterError,
    Snapshot,
    reachable_classes,
)
from axon.egraph.analysis import (
    analyze_snapshot,
    decode_isa,
    ensure_isa_semantics_registered,
)
from axon.egraph.context import SemanticContext
from axon.egraph.fusion import (
    OPTIONAL_FUSION_TIMEOUT_MS,
    FusionWorklist,
    encode_isa_candidate,
    is_eligible_isa_op,
    isa_candidate_prefilter,
    run_fusion_round,
)
from axon.egraph.proof import ProofStore
from axon.egraph.propagation import (
    PropagationWorklist,
    RoundResult,
    run_propagation_round,
)
from axon.egraph.saturation import (
    RoundOutcome,
    run_saturation_loop,
)
from axon.egraph.saturation import (
    SaturationStatus as BaseSaturationStatus,
)
from axon.egraph.workers import resolve_worker_count

OPTIONAL_PROPAGATION_TIMEOUT_MS = 50


def active_isa_classes(
    isa_adapter: EGraphAdapter,
    snapshot: Snapshot,
    L: dict[EClassRef, Any],
    tensor_outputs: list[EClassRef],
) -> list[EClassRef]:
    """Return classes reachable from the lowering roots of declared outputs."""
    roots: list[EClassRef] = []
    seen: set[EClassRef] = set()
    for out in tensor_outputs:
        handle = L.get(out)
        if handle is None:
            continue
        try:
            ref = isa_adapter.resolve_handle(snapshot, handle)
        except EGraphAdapterError:
            continue
        if ref not in seen:
            seen.add(ref)
            roots.append(ref)
    return reachable_classes(snapshot, roots)


def _analyze_isa(snapshot: Snapshot, decode: Any) -> dict[EClassRef, Any]:
    return analyze_snapshot(snapshot, decode)


@dataclass(frozen=True)
class SaturationStatus(BaseSaturationStatus):
    """Reports whether ISA saturation reached a fixed point or a limit.
    ``pass_limit`` is separate from resource truncation."""

    stage: str = "isa_saturation"


def saturate_isa(
    isa_adapter: EGraphAdapter,
    L: dict[EClassRef, Any],
    tensor_outputs: list[EClassRef],
    store: ProofStore | None = None,
    *,
    max_rounds: int = 50,
    max_passes: int | None = None,
    wall_clock_seconds: float = 300.0,
    timeout: int = 10000,
    propagation_timeout: int = OPTIONAL_PROPAGATION_TIMEOUT_MS,
    fusion_timeout: int = OPTIONAL_FUSION_TIMEOUT_MS,
    deadline: float | None = None,
    semi_naive: bool = True,
    workers: int | None = None,
) -> SaturationStatus:
    """Run ISA propagation and fusion until a fixed point or resource limit."""
    ensure_isa_semantics_registered()
    workers = resolve_worker_count(workers)
    if propagation_timeout <= 0 or fusion_timeout <= 0:
        raise ValueError("optional proof timeouts must be positive")
    propagation_timeout = min(timeout, propagation_timeout)
    fusion_timeout = min(timeout, fusion_timeout)
    if store is None:
        store = ProofStore()

    def active_classes(snapshot: Snapshot) -> list[EClassRef]:
        return active_isa_classes(isa_adapter, snapshot, L, tensor_outputs)

    propagation_worklist = PropagationWorklist() if semi_naive else None
    fusion_worklist = FusionWorklist() if semi_naive else None

    def run_round(round_deadline: float | None) -> RoundOutcome:
        snapshot = isa_adapter.freeze_snapshot()

        context = SemanticContext.build(
            isa_adapter,
            snapshot,
            decode_isa,
            _analyze_isa,
        )
        prop: RoundResult = run_propagation_round(
            isa_adapter,
            decode_isa,
            encode_isa_candidate,
            is_eligible_isa_op,
            store,
            analyze=_analyze_isa,
            stage="isa_propagation",
            timeout=propagation_timeout,
            context=context,
            worklist=propagation_worklist,
            deadline=round_deadline,
            workers=workers,
            candidate_prefilter=isa_candidate_prefilter,
        )
        if prop.truncated_reason is not None:
            return RoundOutcome(
                enodes_added=prop.enodes_added,
                equalities_added=prop.equalities_added,
                truncated_reason=prop.truncated_reason,
                truncated_stage=prop.truncated_stage or "isa_propagation",
            )
        fusion: RoundResult = run_fusion_round(
            isa_adapter,
            decode_isa,
            store,
            analyze=_analyze_isa,
            active_classes=active_classes,
            timeout=fusion_timeout,
            context=context,
            worklist=fusion_worklist,
            deadline=round_deadline,
            workers=workers,
        )
        # A shared no-change round (propagation AND fusion both added nothing)
        # establishes the ISA fixed point.
        return RoundOutcome(
            enodes_added=prop.enodes_added + fusion.enodes_added,
            equalities_added=prop.equalities_added + fusion.equalities_added,
            truncated_reason=fusion.truncated_reason,
            truncated_stage=(
                (fusion.truncated_stage or "isa_fusion")
                if fusion.truncated_reason is not None
                else None
            ),
        )

    return run_saturation_loop(
        run_round,
        status_cls=SaturationStatus,
        stage="isa_saturation",
        store=store,
        max_rounds=max_rounds,
        max_passes=max_passes,
        wall_clock_seconds=wall_clock_seconds,
        deadline=deadline,
    )
