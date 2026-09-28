"""Tests for the lowering timeout."""

from __future__ import annotations

from typing import Any

import pytest

import axon.egraph.lowering as lowering
import axon.egraph.pipeline as pipeline
from axon.egraph.adapter import EGraphAdapter
from axon.egraph.isa import OPTIONAL_PROPAGATION_TIMEOUT_MS
from axon.egraph.lowering import LOWERING_TIMEOUT_MS, lower_tensor_egraph
from axon.egraph.proof import ProofStore
from axon.egraph.tensor import ingest_tensor_graph
from axon.ir import build_graph_from_kernel

_DIMS = {"m": 4, "k": 4, "n": 4}


def _kernel(x: Any, w: Any) -> Any:
    return (x * 2.0) @ w


def _ingest() -> tuple[Any, list[Any]]:
    graph = build_graph_from_kernel(
        _kernel, ("x", ("m", "k")), ("w", ("k", "n")), dim_sizes=_DIMS
    )
    adapter = EGraphAdapter("tensor")
    ingest = ingest_tensor_graph(adapter, graph)
    snapshot = adapter.freeze_snapshot()
    outputs = [
        adapter.resolve_handle(snapshot, handle) for handle in ingest.output_handles
    ]
    return snapshot, outputs


def _record_batch_timeouts(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []
    original = lowering.prove_candidate_batch

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["timeout"])
        return original(*args, **kwargs)

    monkeypatch.setattr(lowering, "prove_candidate_batch", spy)
    return seen


def _lower(**kwargs: Any) -> Any:
    snapshot, outputs = _ingest()
    return lower_tensor_egraph(
        snapshot,
        outputs,
        EGraphAdapter("isa"),
        ProofStore(),
        2,
        workers=1,
        **kwargs,
    )


def test_lowering_narrows_the_caller_timeout_to_the_stage_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _record_batch_timeouts(monkeypatch)
    _lower(timeout=3000)
    assert seen, "lowering dispatched no proof batch"
    assert set(seen) == {LOWERING_TIMEOUT_MS}


def test_a_smaller_caller_timeout_still_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _record_batch_timeouts(monkeypatch)
    _lower(timeout=25)
    assert seen
    assert set(seen) == {25}


def test_an_explicit_stage_deadline_is_honored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _record_batch_timeouts(monkeypatch)
    _lower(timeout=3000, lowering_timeout=100)
    assert seen
    assert set(seen) == {100}


def test_a_nonpositive_stage_deadline_is_refused() -> None:
    for value in (0, -1):
        with pytest.raises(ValueError, match="lowering proof timeout must be positive"):
            _lower(timeout=3000, lowering_timeout=value)


def test_the_stage_constant_is_the_default() -> None:
    signature = lowering.lower_tensor_egraph.__kwdefaults__
    assert signature["lowering_timeout"] == LOWERING_TIMEOUT_MS


def test_the_stage_constant_has_margin_over_the_measured_breakpoint() -> None:
    assert LOWERING_TIMEOUT_MS >= 3 * 100


def test_input_seeding_uses_the_same_narrowed_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []
    original = lowering.seed_isa_inputs

    def spy(*args: Any, **kwargs: Any) -> Any:
        # The fifth positional argument is the timeout.
        seen.append(args[4] if len(args) > 4 else kwargs["timeout"])
        return original(*args, **kwargs)

    monkeypatch.setattr(lowering, "seed_isa_inputs", spy)
    _lower(timeout=3000)
    assert seen == [LOWERING_TIMEOUT_MS]


def test_pipeline_threads_the_stage_deadline_and_leaves_tensor_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lowering_timeouts: list[int] = []
    tensor_timeouts: list[int] = []

    original_lower = pipeline.lower_tensor_egraph
    original_saturate = pipeline.saturate_tensor

    def lower_spy(*args: Any, **kwargs: Any) -> Any:
        lowering_timeouts.append(kwargs["lowering_timeout"])
        return original_lower(*args, **kwargs)

    def saturate_spy(*args: Any, **kwargs: Any) -> Any:
        tensor_timeouts.append(kwargs["timeout"])
        return original_saturate(*args, **kwargs)

    monkeypatch.setattr(pipeline, "lower_tensor_egraph", lower_spy)
    monkeypatch.setattr(pipeline, "saturate_tensor", saturate_spy)

    graph = build_graph_from_kernel(
        _kernel, ("x", ("m", "k")), ("w", ("k", "n")), dim_sizes=_DIMS
    )
    pipeline.build_egraph_search(
        graph,
        timeout=3000,
        lowering_timeout_ms=250,
        tensor_max_rounds=0,
        wall_clock_seconds=60.0,
        workers=1,
    )

    assert lowering_timeouts, "no lowering call was made"
    assert set(lowering_timeouts) == {250}
    assert tensor_timeouts == [3000]


def test_isa_propagation_keeps_its_own_constant() -> None:
    assert OPTIONAL_PROPAGATION_TIMEOUT_MS == 50
