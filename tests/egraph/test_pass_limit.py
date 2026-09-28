"""Tests for saturation pass limits."""

from __future__ import annotations

from typing import Any

import pytest

import axon.egraph.pipeline as pipeline
from axon.egraph.adapter import EGraphAdapter
from axon.egraph.tensor import ingest_tensor_graph, saturate_tensor
from axon.ir import build_graph_from_kernel

_DIMS = {"m": 4, "k": 8, "n": 6}


def _kernel(x: Any, w: Any) -> Any:
    return (x * 2.0) @ w


def _tensor_adapter() -> EGraphAdapter:
    graph = build_graph_from_kernel(
        _kernel, ("x", ("m", "k")), ("w", ("k", "n")), dim_sizes=_DIMS
    )
    adapter = EGraphAdapter("tensor")
    ingest_tensor_graph(adapter, graph)
    return adapter


def _search_graph() -> Any:
    return build_graph_from_kernel(
        _kernel, ("x", ("m", "k")), ("w", ("k", "n")), dim_sizes=_DIMS
    )


def test_pass_limit_is_not_reported_as_completed() -> None:
    status = saturate_tensor(_tensor_adapter(), max_passes=1, wall_clock_seconds=60.0)
    assert status.status == "pass_limit"
    assert status.status != "fixed_point"
    assert status.reason == "max_passes"
    assert status.passes == 1

    search = pipeline.build_egraph_search(
        _search_graph(),
        tensor_max_passes=1,
        wall_clock_seconds=60.0,
        workers=1,
    )
    assert search.tensor_status.status == "pass_limit"
    assert search.terminal_status.status == "pass_limit"
    assert search.terminal_status.status != "completed"
    assert search.terminal_status.truncated_stage == "tensor_propagation"
    assert search.terminal_status.stop_reason == "max_passes"


def test_pass_limit_is_not_reported_as_truncated() -> None:
    search = pipeline.build_egraph_search(
        _search_graph(),
        tensor_max_passes=1,
        wall_clock_seconds=60.0,
        workers=1,
    )
    assert search.terminal_status.status not in {"truncated", "failed"}


def test_a_zero_pass_limit_stops_before_the_first_round() -> None:
    status = saturate_tensor(_tensor_adapter(), max_passes=0, wall_clock_seconds=60.0)
    assert status.status == "pass_limit"
    assert status.rounds == 0
    assert status.enodes_added == 0


def test_an_unproductive_round_still_reaches_a_fixed_point() -> None:
    status = saturate_tensor(
        _tensor_adapter(), max_passes=1000, wall_clock_seconds=120.0
    )
    assert status.status == "fixed_point"
    assert status.reason is None
    assert status.passes == status.rounds - 1


def test_a_negative_pass_limit_is_refused() -> None:
    with pytest.raises(ValueError, match="max_passes must be nonnegative"):
        saturate_tensor(_tensor_adapter(), max_passes=-1)


def test_pass_limit_continues_through_lowering_and_isa(monkeypatch: Any) -> None:
    stages: list[str] = []
    original_lower = pipeline.lower_tensor_egraph
    original_isa = pipeline.saturate_isa

    def lower(*args: Any, **kwargs: Any) -> Any:
        stages.append("lowering")
        return original_lower(*args, **kwargs)

    def isa(*args: Any, **kwargs: Any) -> Any:
        stages.append("isa")
        return original_isa(*args, **kwargs)

    monkeypatch.setattr(pipeline, "lower_tensor_egraph", lower)
    monkeypatch.setattr(pipeline, "saturate_isa", isa)

    search = pipeline.build_egraph_search(
        _search_graph(),
        tensor_max_passes=1,
        wall_clock_seconds=120.0,
        workers=1,
    )

    assert stages == ["lowering", "isa"]
    assert search.tensor_status.status == "pass_limit"
    assert search.isa_output_roots


def test_an_isa_pass_limit_does_not_suppress_lowering(monkeypatch: Any) -> None:
    stages: list[str] = []
    original_lower = pipeline.lower_tensor_egraph

    def lower(*args: Any, **kwargs: Any) -> Any:
        stages.append("lowering")
        return original_lower(*args, **kwargs)

    monkeypatch.setattr(pipeline, "lower_tensor_egraph", lower)
    search = pipeline.build_egraph_search(
        _search_graph(),
        isa_max_passes=0,
        wall_clock_seconds=120.0,
        workers=1,
    )

    assert stages == ["lowering"]
    assert search.isa_status.status == "pass_limit"
    assert search.terminal_status.status == "pass_limit"
    assert search.terminal_status.truncated_stage == "isa_saturation"
    assert search.isa_output_roots


def test_a_tensor_pass_limit_is_reported_over_a_later_isa_fixed_point() -> None:
    search = pipeline.build_egraph_search(
        _search_graph(),
        tensor_max_passes=1,
        wall_clock_seconds=120.0,
        workers=1,
    )
    assert search.tensor_status.status == "pass_limit"
    assert search.terminal_status.truncated_stage == "tensor_propagation"


def test_the_default_has_no_pass_limit() -> None:
    assert pipeline.build_egraph_search.__kwdefaults__["tensor_max_passes"] is None
    assert pipeline.build_egraph_search.__kwdefaults__["isa_max_passes"] is None
    assert pipeline.iter_synthesized_hw_graphs.__kwdefaults__["tensor_max_passes"] is (
        None
    )
    assert pipeline.iter_synthesized_hw_graphs.__kwdefaults__["isa_max_passes"] is None

    search = pipeline.build_egraph_search(
        _search_graph(), wall_clock_seconds=120.0, workers=1
    )
    assert search.tensor_status.status == "fixed_point"
    assert search.terminal_status.status == "completed"


def test_the_pass_limit_is_not_a_cli_flag() -> None:
    import argparse

    from axon.cli import _build_parser

    parser = _build_parser()
    base = ["kernels/rmsnorm", "--sizes", "128", "128"]
    assert not hasattr(parser.parse_args(base), "tensor_passes")
    assert not hasattr(parser.parse_args(base), "isa_passes")
    for flag in ("--tensor-passes", "--isa-passes"):
        with pytest.raises(SystemExit):
            parser.parse_args([*base, flag, "2"])
    assert isinstance(parser, argparse.ArgumentParser)


def test_pass_limit_survives_the_graph_iterator(monkeypatch: Any) -> None:
    graphs = pipeline.iter_synthesized_hw_graphs(
        _search_graph(),
        tensor_max_passes=1,
        wall_clock_seconds=120.0,
        workers=1,
    )
    produced = list(graphs)

    assert produced
    assert graphs.outcome.status == "pass_limit"
    assert graphs.outcome.truncated_stage == "tensor_propagation"
    assert graphs.outcome.stop_reason == "max_passes"
