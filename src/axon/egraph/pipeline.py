"""Builds and streams concrete ISA graphs from one tensor e-graph search."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from axon.egraph.adapter import (
    EClassRef,
    EGraphAdapter,
    Snapshot,
)
from axon.egraph.analysis import ensure_isa_semantics_registered
from axon.egraph.extraction import ExtractionError, iter_materialized_isa_graphs
from axon.egraph.isa import SaturationStatus as IsaSaturationStatus
from axon.egraph.isa import saturate_isa
from axon.egraph.lowering import (
    LOWERING_TIMEOUT_MS,
    LoweringStatus,
    _RecipeCache,
    lower_tensor_egraph,
)
from axon.egraph.persist import EGraphCacheFile
from axon.egraph.proof import ProofStore, WallClockExceeded
from axon.egraph.tensor import SaturationStatus as TensorSaturationStatus
from axon.egraph.tensor import ingest_tensor_graph, saturate_tensor
from axon.egraph.workers import resolve_worker_count
from axon.ir import nuGraph

DEFAULT_SYNTHESIS_WALL_CLOCK_SECONDS = 3600.0

# Extraction renames only resource truncation.
_EXTRACTED_STATUS = {"truncated": "partial_extracted"}


@dataclass(frozen=True)
class SynthesisStatus:
    """Structured terminal status for the staged e-graph search."""

    status: str
    truncated_stage: str | None = None
    stop_reason: str | None = None
    unrealized_outputs: tuple[EClassRef, ...] = ()


@dataclass
class _ExtractionOutcome:
    """Mutable private status populated as the graph iterator is consumed."""

    status: str = "pending"
    truncated_stage: str | None = None
    stop_reason: str | None = None
    unrealized_outputs: tuple[EClassRef, ...] = ()
    extraction_exhaustion: str | None = None
    extraction_stage: str | None = None
    emitted_graph_count: int = 0


@dataclass
class EGraphSearch:
    """The state and status from ``build_egraph_search``."""

    isa_adapter: EGraphAdapter
    isa_snapshot: Snapshot
    isa_output_roots: list[EClassRef]
    tensor_snapshot: Snapshot
    tensor_output_roots: list[EClassRef]
    input_metadata: dict[str, dict[str, Any]]
    declared_input_ids: tuple[str, ...]
    L: dict[EClassRef, Any]
    store: ProofStore
    tensor_status: TensorSaturationStatus
    lowering_status: LoweringStatus
    isa_status: IsaSaturationStatus
    deadline: float | None
    terminal_status: SynthesisStatus


@dataclass(frozen=True)
class _ExtractionInputs:
    isa_snapshot: Snapshot
    isa_output_roots: list[EClassRef]
    input_metadata: dict[str, dict[str, Any]]
    declared_input_ids: tuple[str, ...]
    terminal_status: SynthesisStatus


def _extraction_inputs(search: EGraphSearch) -> _ExtractionInputs:
    return _ExtractionInputs(
        isa_snapshot=search.isa_snapshot,
        isa_output_roots=search.isa_output_roots,
        input_metadata=search.input_metadata,
        declared_input_ids=search.declared_input_ids,
        terminal_status=search.terminal_status,
    )


def build_egraph_search(
    G: nuGraph,
    *,
    max_hw_size: int = 2,
    timeout: int = 3000,
    lowering_timeout_ms: int = LOWERING_TIMEOUT_MS,
    store: ProofStore | None = None,
    tensor_max_rounds: int = 50,
    isa_max_rounds: int = 50,
    tensor_max_passes: int | None = None,
    isa_max_passes: int | None = None,
    wall_clock_seconds: float | None = DEFAULT_SYNTHESIS_WALL_CLOCK_SECONDS,
    workers: int | None = None,
    semi_naive: bool = True,
) -> EGraphSearch:
    """Runs tensor saturation, lowering, and ISA saturation.
    The result contains the search state and terminal status."""
    ensure_isa_semantics_registered()
    workers = resolve_worker_count(workers)
    if store is None:
        store = ProofStore()
    recipe_cache: _RecipeCache = {}

    # One shared wall-clock deadline spans every discovery stage.
    deadline: float | None = None
    if wall_clock_seconds is not None and math.isfinite(wall_clock_seconds):
        deadline = time.monotonic() + max(0.0, wall_clock_seconds)

    # Tensor ingest
    tensor_adapter = EGraphAdapter("tensor")
    ingest = ingest_tensor_graph(tensor_adapter, G)
    declared_input_ids = tuple(G.input_ids) or tuple(ingest.input_metadata)

    # Freeze the ingested state before saturation mutates the adapter. Lowering
    # this snapshot is the fallback route to extractable content, used only when
    # the wall lapses before the saturated state realizes every output.
    ingested_snapshot = tensor_adapter.freeze_snapshot()
    ingested_outputs = [
        tensor_adapter.resolve_handle(ingested_snapshot, handle)
        for handle in ingest.output_handles
    ]

    # -- exhaustive tensor saturation ---------------------------------------
    # Saturation runs before the only lowering pass.
    tensor_status = saturate_tensor(
        tensor_adapter,
        store,
        max_rounds=tensor_max_rounds,
        max_passes=tensor_max_passes,
        wall_clock_seconds=math.inf,
        timeout=timeout,
        deadline=deadline,
        workers=workers,
        semi_naive=semi_naive,
    )

    # Freeze once after tensor saturation and resolve the declared output
    # handles (stable let handles) into this snapshot, preserving duplicates.
    tensor_snapshot = tensor_adapter.freeze_snapshot()
    tensor_outputs = [
        tensor_adapter.resolve_handle(tensor_snapshot, handle)
        for handle in ingest.output_handles
    ]

    # Compositional lowering into a fresh ISA e-graph. Lowering shares the
    # synthesis wall clock, so a lapsed wall truncates it like the baseline.
    isa_adapter, L, lowering_status = lower_tensor_egraph(
        tensor_snapshot,
        tensor_outputs,
        EGraphAdapter("isa"),
        store,
        max_hw_size,
        timeout=timeout,
        lowering_timeout=lowering_timeout_ms,
        workers=workers,
        deadline=deadline,
        recipe_cache=recipe_cache,
    )
    # A wall that lapses before any output is realized would leave nothing to
    # extract. Lowering is mandatory finalization work, so fall back to the
    # ingested state off the wall: the wall truncates the search, it does not
    # zero out the result.
    if lowering_status.unrealized_outputs:
        fallback_adapter, fallback_L, fallback_status = lower_tensor_egraph(
            ingested_snapshot,
            ingested_outputs,
            EGraphAdapter("isa"),
            store,
            max_hw_size,
            timeout=timeout,
            lowering_timeout=lowering_timeout_ms,
            workers=workers,
            deadline=None,
            stop_when_outputs_realized=True,
            recipe_cache=recipe_cache,
        )
        # The fallback replaces the main pass only when it realizes every
        # output; a partial fallback would discard the main pass's roots, and
        # refs from the two snapshots never compare equal.
        if not fallback_status.unrealized_outputs:
            isa_adapter, L, lowering_status = (
                fallback_adapter,
                fallback_L,
                fallback_status,
            )
            # The roots and the persisted snapshot must be the same snapshot.
            tensor_snapshot = ingested_snapshot
            tensor_outputs = ingested_outputs

    # Tensor truncation does not prevent eligible ISA work.
    stop_after_lowering = lowering_status.status in {"failed", "truncated"}
    if stop_after_lowering:
        # An unrealized declared output is a finalization failure.
        if lowering_status.unrealized_outputs:
            terminal_status = SynthesisStatus(
                status="failed",
                truncated_stage="finalization",
                stop_reason=lowering_status.reason or "unrealized_outputs",
                unrealized_outputs=lowering_status.unrealized_outputs,
            )
        else:
            terminal_status = SynthesisStatus(
                status="truncated",
                truncated_stage=lowering_status.stage,
                stop_reason=lowering_status.reason,
            )
        isa_status = IsaSaturationStatus(
            status="truncated",
            rounds=0,
            enodes_added=0,
            equalities_added=0,
            dispatch_count=store.dispatch_count,
            elapsed_seconds=0.0,
            reason=terminal_status.stop_reason,
            stage=terminal_status.truncated_stage or "isa_saturation",
        )
        # Only the outputs lowering realized can resolve into the snapshot.
        resolvable_outputs = [out for out in tensor_outputs if out in L]
    else:
        # -- joint ISA propagation + instruction fusion ----------------------
        isa_status = saturate_isa(
            isa_adapter,
            L,
            tensor_outputs,
            store,
            max_rounds=isa_max_rounds,
            max_passes=isa_max_passes,
            wall_clock_seconds=math.inf,
            timeout=timeout,
            deadline=deadline,
            workers=workers,
            semi_naive=semi_naive,
        )
        resolvable_outputs = list(tensor_outputs)

        # Report the first stage that stopped before its fixed point.
        stopped = next(
            (
                stage_status
                for stage_status in (tensor_status, isa_status)
                if stage_status.status != "fixed_point"
            ),
            None,
        )
        if stopped is None:
            terminal_status = SynthesisStatus(status="completed")
        else:
            terminal_status = SynthesisStatus(
                status=stopped.status,
                truncated_stage=stopped.stage,
                stop_reason=stopped.reason,
            )

    # Freeze the ISA graph once and resolve the ordered declared-output roots.
    isa_snapshot = isa_adapter.freeze_snapshot()
    isa_output_roots = [
        isa_adapter.resolve_handle(isa_snapshot, L[out]) for out in resolvable_outputs
    ]
    return EGraphSearch(
        isa_adapter=isa_adapter,
        isa_snapshot=isa_snapshot,
        isa_output_roots=isa_output_roots,
        tensor_snapshot=tensor_snapshot,
        tensor_output_roots=tensor_outputs,
        input_metadata=ingest.input_metadata,
        declared_input_ids=declared_input_ids,
        L=L,
        store=store,
        tensor_status=tensor_status,
        lowering_status=lowering_status,
        isa_status=isa_status,
        deadline=deadline,
        terminal_status=terminal_status,
    )


class _SynthesisGraphIterator(Iterator[nuGraph]):
    """Lazy graph iterator using either a graph or prepared frozen inputs."""

    def __init__(
        self,
        G: nuGraph | None,
        options: dict[str, Any],
        inputs: _ExtractionInputs | None = None,
    ) -> None:
        self._graph = G
        self._options = options
        self._inputs = inputs
        self._iterator: Iterator[nuGraph] | None = None
        self.outcome = _ExtractionOutcome()

    def __iter__(self) -> _SynthesisGraphIterator:
        return self

    def __next__(self) -> nuGraph:
        if self._iterator is None:
            self._iterator = self._run()
        return next(self._iterator)

    def _run(self) -> Iterator[nuGraph]:
        inputs = self._inputs
        if inputs is None:
            assert self._graph is not None
            search = build_egraph_search(self._graph, **self._options)
            inputs = _extraction_inputs(search)
            del search

        terminal = inputs.terminal_status
        self.outcome.truncated_stage = terminal.truncated_stage
        self.outcome.stop_reason = terminal.stop_reason
        self.outcome.unrealized_outputs = terminal.unrealized_outputs
        if terminal.status == "failed" or terminal.unrealized_outputs:
            self.outcome.status = "failed"
            return

        try:
            # The synthesis wall bounds saturation only. Extraction streams
            # already-proved content, so the caller decides when to stop it.
            for graph in iter_materialized_isa_graphs(
                inputs.isa_snapshot,
                inputs.isa_output_roots,
                inputs.input_metadata,
                declared_input_ids=inputs.declared_input_ids,
                workers=self._options["workers"],
            ):
                self.outcome.emitted_graph_count += 1
                # Preserve pass limits, but rename resource truncation.
                self.outcome.status = _EXTRACTED_STATUS.get(
                    terminal.status, terminal.status
                )
                yield graph
        except WallClockExceeded as exc:
            self.outcome.extraction_exhaustion = f"{exc.stage}: {exc.reason}"
            self.outcome.extraction_stage = exc.stage
        except ExtractionError as exc:
            self.outcome.extraction_exhaustion = str(exc)
            self.outcome.extraction_stage = "extraction_selection"

        if self.outcome.emitted_graph_count == 0:
            self.outcome.status = "failed"
            if self.outcome.extraction_exhaustion is None:
                self.outcome.extraction_exhaustion = "no materialized ISA graph"


def iter_hw_graphs_from_cache(
    cache: EGraphCacheFile,
    *,
    workers: int | None = None,
) -> Iterator[nuGraph]:
    """Stream the same graphs the cached run streamed, skipping synthesis."""
    return _SynthesisGraphIterator(
        None,
        {"workers": resolve_worker_count(workers)},
        inputs=_ExtractionInputs(
            isa_snapshot=cache.isa.rebuild(),
            isa_output_roots=list(cache.isa_output_roots),
            input_metadata=cache.input_metadata,
            declared_input_ids=cache.declared_input_ids,
            terminal_status=cache.terminal_status,
        ),
    )


def iter_hw_graphs_from_search(
    search: EGraphSearch,
    *,
    workers: int | None = None,
) -> Iterator[nuGraph]:
    """Extract hardware graphs from a finished search."""
    return _SynthesisGraphIterator(
        None,
        {"workers": resolve_worker_count(workers)},
        inputs=_extraction_inputs(search),
    )


def iter_synthesized_hw_graphs(
    G: nuGraph,
    *,
    max_hw_size: int = 2,
    timeout: int = 3000,
    lowering_timeout_ms: int = LOWERING_TIMEOUT_MS,
    store: ProofStore | None = None,
    tensor_max_rounds: int = 50,
    isa_max_rounds: int = 50,
    tensor_max_passes: int | None = None,
    isa_max_passes: int | None = None,
    wall_clock_seconds: float | None = DEFAULT_SYNTHESIS_WALL_CLOCK_SECONDS,
    workers: int | None = None,
    semi_naive: bool = True,
) -> Iterator[nuGraph]:
    """Stream only validated graphs, with terminal detail kept private."""
    workers = resolve_worker_count(workers)
    return _SynthesisGraphIterator(
        G,
        {
            "max_hw_size": max_hw_size,
            "timeout": timeout,
            "lowering_timeout_ms": lowering_timeout_ms,
            "store": store,
            "tensor_max_rounds": tensor_max_rounds,
            "isa_max_rounds": isa_max_rounds,
            "tensor_max_passes": tensor_max_passes,
            "isa_max_passes": isa_max_passes,
            "wall_clock_seconds": wall_clock_seconds,
            "workers": workers,
            "semi_naive": semi_naive,
        },
    )


__all__ = [
    "DEFAULT_SYNTHESIS_WALL_CLOCK_SECONDS",
    "LOWERING_TIMEOUT_MS",
    "EGraphSearch",
    "SynthesisStatus",
    "build_egraph_search",
    "iter_hw_graphs_from_cache",
    "iter_hw_graphs_from_search",
    "iter_synthesized_hw_graphs",
]
