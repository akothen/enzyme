"""Shared status shape and loop mechanics for e-graph saturation stages."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from axon.egraph.proof import ProofStore


@dataclass(frozen=True)
class SaturationStatus:
    """Reports whether saturation reached a fixed point or a limit.
    ``pass_limit`` is separate from resource truncation."""

    status: str
    rounds: int
    enodes_added: int
    equalities_added: int
    dispatch_count: int
    elapsed_seconds: float
    reason: str | None = None
    stage: str = ""
    passes: int = 0


@dataclass(frozen=True)
class RoundOutcome:
    """One saturation round's additions and optional truncation."""

    enodes_added: int
    equalities_added: int
    truncated_reason: str | None = None
    truncated_stage: str | None = None


StatusT = TypeVar("StatusT", bound=SaturationStatus)


def run_saturation_loop(
    run_round: Callable[[float | None], RoundOutcome],
    *,
    status_cls: type[StatusT],
    stage: str,
    store: ProofStore,
    max_rounds: int,
    max_passes: int | None,
    wall_clock_seconds: float,
    deadline: float | None,
) -> StatusT:
    """Drive rounds until a fixed point, a round or pass limit, or the wall.
    ``max_passes`` counts only rounds that change the graph."""
    if max_passes is not None and max_passes < 0:
        raise ValueError("max_passes must be nonnegative")
    start = time.monotonic()
    if deadline is None and math.isfinite(wall_clock_seconds):
        deadline = start + max(0.0, wall_clock_seconds)
    rounds = 0
    passes = 0
    total_enodes = 0
    total_equalities = 0

    def status(kind: str, reason: str | None, at_stage: str) -> SaturationStatus:
        return status_cls(
            status=kind,
            rounds=rounds,
            enodes_added=total_enodes,
            equalities_added=total_equalities,
            dispatch_count=store.dispatch_count,
            elapsed_seconds=time.monotonic() - start,
            reason=reason,
            stage=at_stage,
            passes=passes,
        )

    while True:
        if max_passes is not None and passes >= max_passes:
            return status("pass_limit", "max_passes", stage)
        if rounds >= max_rounds:
            return status("truncated", "max_rounds", stage)
        if deadline is not None and time.monotonic() >= deadline:
            return status("truncated", "wall_clock_seconds", stage)
        outcome = run_round(deadline)
        total_enodes += outcome.enodes_added
        total_equalities += outcome.equalities_added
        if outcome.truncated_reason is not None:
            return status(
                "truncated",
                outcome.truncated_reason,
                outcome.truncated_stage or stage,
            )
        rounds += 1
        if outcome.enodes_added == 0 and outcome.equalities_added == 0:
            return status("fixed_point", None, stage)
        # Only a round that changed the graph consumes a pass.
        passes += 1
