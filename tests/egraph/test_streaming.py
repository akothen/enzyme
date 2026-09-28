"""Streaming production-contract tests for the CLI trace helpers (plan
section 11 table).

These tests do not run real synthesis or emission. They drive
``cli._trace_single`` and ``cli._trace_lnc2`` with an instrumented generator
(a side-effect counter recording exactly when each hardware graph is drawn) and
an instrumented per-graph emitter, and assert:

* both helpers count zero and nonzero streams (the returned
  ``(consumed_graph_count, emitted_module_count)`` pair), and
* graph two is not materialized before graph one has finished all its tiles
  and, for ``lnc=2``, all its retained sharding plans (one-at-a-time
  consumption).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import axon.egraph.pipeline as pipeline
from axon import cli
from axon.codegen import EmitErr, EmitOk
from axon.egraph.extraction import iter_materialized_isa_graphs
from axon.egraph.proof import ProofStore, WallClockExceeded
from axon.ir import Node, build_graph_from_kernel, nuGraph
from axon.paths import run_paths

_KERNELS = Path(__file__).resolve().parents[2] / "kernels"


def _mul_spec():
    return cli._load_spec(str(_KERNELS / "mul"))


class _Graph:
    """A stand-in hardware graph; identity is all the helpers need here."""

    def __init__(self, tag: str) -> None:
        self.tag_str = tag


def _instrumented_stream(graphs, log):
    """Yield each graph, appending a ``("produce", tag)`` event as it is drawn.

    The event is appended at the moment the consumer advances the iterator, so
    the event log proves how far the consumer has drained the stream relative
    to its emission work."""
    for g in graphs:
        log.append(("produce", g.tag_str))
        yield g


# ---------------------------------------------------------------------------
# _trace_single
# ---------------------------------------------------------------------------


def test_trace_single_counts_zero_stream(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    paths.run_dir.mkdir(parents=True)

    def _boom(*a, **kw):
        raise AssertionError("run_nki_bench must not run for an empty stream")

    monkeypatch.setattr(cli, "run_nki_bench", _boom)

    consumed, emitted, _expected = cli._trace_single(
        spec,
        {"m": 128, "n": 128},
        iter([]),
        "mul",
        paths=paths,
        warmup=1,
        bench=1,
        target=None,
        dtype=None,
        rtol=1e-2,
        atol=1e-2,
        bench_variants=False,
    )
    assert (consumed, emitted) == (0, 0)


def test_trace_single_counts_nonzero_and_streams_one_at_a_time(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    paths.run_dir.mkdir(parents=True)
    monkeypatch.setattr(cli, "run_nki_bench", lambda *a, **kw: set())

    log: list = []

    def _emit(hw_variants, name):
        # Two tile configs per graph; record an emit event for each.
        for i, g_hw in enumerate(hw_variants):
            for j in range(2):
                log.append(("emit", g_hw.tag_str, j))
                yield EmitOk(i, j, f"# {g_hw.tag_str} t{j}\n")

    monkeypatch.setattr(cli, "emit_nki_code_variants", _emit)

    graphs = _instrumented_stream([_Graph("g0"), _Graph("g1")], log)
    consumed, emitted, _expected = cli._trace_single(
        spec,
        {"m": 128, "n": 128},
        graphs,
        "mul",
        paths=paths,
        warmup=1,
        bench=1,
        target=None,
        dtype=None,
        rtol=1e-2,
        atol=1e-2,
        bench_variants=False,
    )
    assert (consumed, emitted) == (2, 4)
    # Graph one finishes all its tiles before graph two is materialized.
    assert log == [
        ("produce", "g0"),
        ("emit", "g0", 0),
        ("emit", "g0", 1),
        ("produce", "g1"),
        ("emit", "g1", 0),
        ("emit", "g1", 1),
    ]


def test_trace_single_emit_error_is_counted_as_consumed_not_emitted(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    paths.run_dir.mkdir(parents=True)
    monkeypatch.setattr(cli, "run_nki_bench", lambda *a, **kw: set())

    def _emit(hw_variants, name):
        for i, _g_hw in enumerate(hw_variants):
            yield EmitErr(i, RuntimeError("boom"))

    monkeypatch.setattr(cli, "emit_nki_code_variants", _emit)

    consumed, emitted, _expected = cli._trace_single(
        spec,
        {"m": 128, "n": 128},
        iter([_Graph("g0")]),
        "mul",
        paths=paths,
        warmup=1,
        bench=1,
        target=None,
        dtype=None,
        rtol=1e-2,
        atol=1e-2,
        bench_variants=False,
    )
    assert (consumed, emitted) == (1, 0)


# ---------------------------------------------------------------------------
# _trace_lnc2
# ---------------------------------------------------------------------------


class _Plan:
    def __init__(self, tag: str) -> None:
        self._tag = tag

    def tag(self) -> str:
        return self._tag


def _lnc2_common(monkeypatch):
    """Retain two fake sharding plans and a no-op bench for the lnc=2 helper."""
    plans = [_Plan("planA"), _Plan("planB")]
    monkeypatch.setattr(cli, "shardings", lambda G0: list(plans))
    monkeypatch.setattr(cli, "prune_plans", lambda plans_, G0, dims: list(plans_))
    monkeypatch.setattr(cli, "run_nki_bench", lambda *a, **kw: set())
    return plans


def test_trace_lnc2_counts_zero_stream(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    paths.run_dir.mkdir(parents=True)
    _lnc2_common(monkeypatch)

    consumed, emitted, _expected = cli._trace_lnc2(
        spec,
        {"m": 128, "n": 128},
        _Graph("G0"),
        iter([]),
        "mul",
        paths=paths,
        lnc=2,
        warmup=1,
        bench=1,
        target=None,
        dtype=None,
        rtol=1e-2,
        atol=1e-2,
    )
    assert (consumed, emitted) == (0, 0)


def test_trace_lnc2_counts_nonzero_and_finishes_all_plans_per_graph(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    paths.run_dir.mkdir(parents=True)
    plans = _lnc2_common(monkeypatch)

    log: list = []

    def _emit(hw_variants, name, plan, plan_graph):
        # The helper hands every retained plan to the emitter, one per call.
        assert plan in plans
        for i, g_hw in enumerate(hw_variants):
            log.append(("emit", g_hw.tag_str, plan.tag()))
            yield EmitOk(i, 0, f"# {g_hw.tag_str} {plan.tag()}\n")

    monkeypatch.setattr(cli, "emit_nki_code_lnc2_variants", _emit)

    graphs = _instrumented_stream([_Graph("g0"), _Graph("g1")], log)
    consumed, emitted, _expected = cli._trace_lnc2(
        spec,
        {"m": 128, "n": 128},
        _Graph("G0"),
        graphs,
        "mul",
        paths=paths,
        lnc=2,
        warmup=1,
        bench=1,
        target=None,
        dtype=None,
        rtol=1e-2,
        atol=1e-2,
    )
    assert (consumed, emitted) == (2, 4)
    # Graph one finishes all retained plans before graph two is materialized.
    assert log == [
        ("produce", "g0"),
        ("emit", "g0", "planA"),
        ("emit", "g0", "planB"),
        ("produce", "g1"),
        ("emit", "g1", "planA"),
        ("emit", "g1", "planB"),
    ]


def test_trace_lnc2_emit_error_is_counted_as_consumed_not_emitted(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    paths.run_dir.mkdir(parents=True)
    _lnc2_common(monkeypatch)

    def _emit(hw_variants, name, plan, plan_graph):
        for i, _g_hw in enumerate(hw_variants):
            yield EmitErr(i, RuntimeError("boom"))

    monkeypatch.setattr(cli, "emit_nki_code_lnc2_variants", _emit)

    consumed, emitted, _expected = cli._trace_lnc2(
        spec,
        {"m": 128, "n": 128},
        _Graph("G0"),
        iter([_Graph("g0")]),
        "mul",
        paths=paths,
        lnc=2,
        warmup=1,
        bench=1,
        target=None,
        dtype=None,
        rtol=1e-2,
        atol=1e-2,
    )
    assert (consumed, emitted) == (1, 0)


def _input_graph(node_id: str = "x") -> nuGraph:
    return nuGraph(
        nodes=[Node(node_id, "input", [], {"shape": (1,)})],
        output_ids=(node_id,),
        input_ids=(node_id,),
    )


def _search(
    *,
    status: str = "completed",
    unrealized_outputs: tuple[Any, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        lowering_status=SimpleNamespace(unrealized_outputs=unrealized_outputs),
        terminal_status=pipeline.SynthesisStatus(
            status=status,
            truncated_stage="lowering" if status != "completed" else None,
            stop_reason="wall_clock_seconds" if status != "completed" else None,
            unrealized_outputs=unrealized_outputs,
        ),
        isa_snapshot=object(),
        isa_output_roots=[object()] if not unrealized_outputs else [],
        input_metadata={},
        declared_input_ids=(),
        deadline=None,
    )


def test_build_search_reports_truncated_stage_and_reason() -> None:
    search = pipeline.build_egraph_search(
        _input_graph(),
        tensor_max_rounds=0,
        wall_clock_seconds=10.0,
    )

    assert search.terminal_status.status == "truncated"
    assert search.terminal_status.truncated_stage == "tensor_propagation"
    assert search.terminal_status.stop_reason == "max_rounds"
    assert search.terminal_status.unrealized_outputs == ()
    assert len(search.isa_output_roots) == 1


def test_tensor_round_truncation_preserves_lowering_variants() -> None:
    def kernel(x, y):
        return x * y

    graph = build_graph_from_kernel(
        kernel,
        ("x", ("m", "k")),
        ("y", ("m", "k")),
        dim_sizes={"m": 4, "k": 4},
    )
    search = pipeline.build_egraph_search(
        graph,
        max_hw_size=2,
        tensor_max_rounds=0,
        wall_clock_seconds=10.0,
        workers=2,
    )

    materialized = list(
        iter_materialized_isa_graphs(
            search.isa_snapshot,
            search.isa_output_roots,
            search.input_metadata,
            declared_input_ids=search.declared_input_ids,
        )
    )
    operand_orders = {
        tuple(node.inputs)
        for graph_variant in materialized
        for node in graph_variant.nodes
        if node.op == "tensor_tensor"
    }

    assert search.terminal_status.stop_reason == "max_rounds"
    assert operand_orders == {("x", "y"), ("y", "x")}


def test_lowering_truncation_supersedes_tensor_round_limit(
    monkeypatch: Any,
) -> None:
    original_lower = pipeline.lower_tensor_egraph

    def lower(*args: Any, **kwargs: Any) -> Any:
        _adapter, lowered, status = original_lower(*args, **kwargs)
        return (
            _adapter,
            lowered,
            pipeline.LoweringStatus(
                status="truncated",
                unrealized_outputs=status.unrealized_outputs,
                attempted=status.attempted,
                realized=status.realized,
                proved_programs=status.proved_programs,
                reason="max_candidate_waves",
                stage="lowering",
            ),
        )

    monkeypatch.setattr(pipeline, "lower_tensor_egraph", lower)
    search = pipeline.build_egraph_search(
        _input_graph(),
        tensor_max_rounds=0,
        wall_clock_seconds=10.0,
    )

    assert search.tensor_status.reason == "max_rounds"
    assert search.lowering_status.reason == "max_candidate_waves"
    assert search.terminal_status.truncated_stage == "lowering"
    assert search.terminal_status.stop_reason == "max_candidate_waves"


def test_production_path_has_no_bootstrap_fallback(monkeypatch: Any) -> None:
    """The production path lowers once without a bootstrap adapter."""
    adapters: list[str] = []
    original_lower = pipeline.lower_tensor_egraph

    def lower(*args: Any, **kwargs: Any) -> Any:
        adapters.append(args[2].label)
        return original_lower(*args, **kwargs)

    monkeypatch.setattr(pipeline, "lower_tensor_egraph", lower)
    search = pipeline.build_egraph_search(
        _input_graph(),
        tensor_max_rounds=1,
        isa_max_rounds=0,
        wall_clock_seconds=10.0,
    )

    assert adapters == ["isa"]
    assert search.isa_adapter.label == "isa"


def test_proof_store_has_no_journal() -> None:
    # The store keeps only verdict caches and counters; journals are gone.
    assert not hasattr(ProofStore, "fork_journal")
    assert not hasattr(ProofStore(), "records")


def test_lowering_follows_tensor_propagation(monkeypatch: Any) -> None:
    events: list[str] = []
    original_lower = pipeline.lower_tensor_egraph
    original_saturate = pipeline.saturate_tensor

    def lower(*args: Any, **kwargs: Any) -> Any:
        events.append("lower")
        return original_lower(*args, **kwargs)

    def saturate(*args: Any, **kwargs: Any) -> Any:
        events.append("tensor")
        return original_saturate(*args, **kwargs)

    monkeypatch.setattr(pipeline, "lower_tensor_egraph", lower)
    monkeypatch.setattr(pipeline, "saturate_tensor", saturate)

    search = pipeline.build_egraph_search(
        _input_graph(),
        tensor_max_rounds=0,
        wall_clock_seconds=10.0,
    )

    assert events == ["tensor", "lower"]
    assert search.terminal_status.truncated_stage == "tensor_propagation"
    assert len(search.isa_output_roots) == 1


def test_lowering_uses_the_callers_store_directly(monkeypatch: Any) -> None:
    stores: list[ProofStore] = []
    original_lower = pipeline.lower_tensor_egraph

    def lower(*args: Any, **kwargs: Any) -> Any:
        stores.append(args[3])
        return original_lower(*args, **kwargs)

    monkeypatch.setattr(pipeline, "lower_tensor_egraph", lower)
    store = ProofStore()
    search = pipeline.build_egraph_search(
        _input_graph(),
        store=store,
        tensor_max_rounds=1,
        isa_max_rounds=0,
        wall_clock_seconds=10.0,
    )

    assert stores == [store]
    assert search.store is store


def _exhaust_tensor_search(reason: str) -> Any:
    """Returns a saturation stub that reports a resource limit."""

    def exhaust(
        _adapter: Any,
        proof_store: ProofStore,
        **_kwargs: Any,
    ) -> Any:
        return pipeline.TensorSaturationStatus(
            status="truncated",
            rounds=0,
            enodes_added=0,
            equalities_added=0,
            dispatch_count=proof_store.dispatch_count,
            elapsed_seconds=0.0,
            reason=reason,
            stage="tensor_propagation",
        )

    return exhaust


def test_tensor_timeout_finalizes_latest_stable_checkpoint(monkeypatch: Any) -> None:
    """A resource limit lowers the latest tensor state."""
    lowered_snapshots: list[Any] = []
    original_lower = pipeline.lower_tensor_egraph

    def lower(*args: Any, **kwargs: Any) -> Any:
        lowered_snapshots.append(args[0])
        return original_lower(*args, **kwargs)

    monkeypatch.setattr(
        pipeline, "saturate_tensor", _exhaust_tensor_search("wall_clock_seconds")
    )
    monkeypatch.setattr(pipeline, "lower_tensor_egraph", lower)
    search = pipeline.build_egraph_search(
        _input_graph(),
        wall_clock_seconds=10.0,
    )

    assert len(lowered_snapshots) == 1
    assert search.isa_adapter.label == "isa"
    assert len(search.isa_output_roots) == 1
    assert search.terminal_status.truncated_stage == "tensor_propagation"
    assert search.terminal_status.stop_reason == "wall_clock_seconds"


def test_latest_checkpoint_keeps_tensor_verdict_cache(monkeypatch: Any) -> None:
    from axon.isa_semantics import EquivalenceVerdict

    tensor_verdict = EquivalenceVerdict(proved=True, stage="proved")

    def exhaust(
        _adapter: Any,
        proof_store: ProofStore,
        **_kwargs: Any,
    ) -> Any:
        proof_store.local_cache["tensor_key"] = tensor_verdict
        return pipeline.TensorSaturationStatus(
            status="truncated",
            rounds=0,
            enodes_added=0,
            equalities_added=0,
            dispatch_count=proof_store.dispatch_count,
            elapsed_seconds=0.0,
            reason="wall_clock_seconds",
            stage="tensor_propagation",
        )

    monkeypatch.setattr(pipeline, "saturate_tensor", exhaust)
    store = ProofStore()
    search = pipeline.build_egraph_search(
        _input_graph(),
        store=store,
        wall_clock_seconds=10.0,
    )

    assert search.store is store
    assert store.local_cache.get("tensor_key") is tensor_verdict


def test_older_checkpoint_is_diagnostic_not_success(monkeypatch: Any) -> None:
    def unrealized(*args: Any, **kwargs: Any) -> Any:
        outputs = list(args[1])
        return (
            args[2],
            {},
            pipeline.LoweringStatus(
                status="failed",
                unrealized_outputs=tuple(outputs),
                realized=0,
            ),
        )

    monkeypatch.setattr(pipeline, "lower_tensor_egraph", unrealized)
    search = pipeline.build_egraph_search(
        _input_graph(),
        tensor_max_rounds=0,
        wall_clock_seconds=10.0,
    )

    assert search.terminal_status.status == "failed"
    assert search.terminal_status.unrealized_outputs
    assert search.isa_output_roots == []

    graphs = pipeline.iter_synthesized_hw_graphs(_input_graph(), tensor_max_rounds=0)
    monkeypatch.setattr(
        pipeline,
        "build_egraph_search",
        lambda *_args, **_kwargs: search,
    )
    assert list(graphs) == []
    assert graphs.outcome.status == "failed"


def test_tensor_round_limit_still_runs_isa_search(monkeypatch: Any) -> None:
    """A tensor round limit does not skip eligible ISA work."""
    isa_calls = 0
    original_isa = pipeline.saturate_isa

    def saturate(*args: Any, **kwargs: Any) -> Any:
        nonlocal isa_calls
        isa_calls += 1
        return original_isa(*args, **kwargs)

    monkeypatch.setattr(pipeline, "saturate_isa", saturate)
    search = pipeline.build_egraph_search(
        _input_graph(),
        tensor_max_rounds=0,
        wall_clock_seconds=10.0,
    )

    assert isa_calls == 1
    assert search.terminal_status.status == "truncated"
    assert search.terminal_status.truncated_stage == "tensor_propagation"
    assert search.terminal_status.stop_reason == "max_rounds"


def test_shared_resource_tensor_truncation_still_runs_isa_search(
    monkeypatch: Any,
) -> None:
    isa_calls = 0
    original_isa = pipeline.saturate_isa

    def saturate(*args: Any, **kwargs: Any) -> Any:
        nonlocal isa_calls
        isa_calls += 1
        return original_isa(*args, **kwargs)

    monkeypatch.setattr(
        pipeline, "saturate_tensor", _exhaust_tensor_search("wall_clock_seconds")
    )
    monkeypatch.setattr(pipeline, "saturate_isa", saturate)
    search = pipeline.build_egraph_search(
        _input_graph(),
        wall_clock_seconds=10.0,
    )

    assert isa_calls == 1
    assert search.terminal_status.truncated_stage == "tensor_propagation"
    assert search.terminal_status.stop_reason == "wall_clock_seconds"


def test_truncated_tensor_stage_is_not_reported_as_completed(
    monkeypatch: Any,
) -> None:
    """Tensor truncation remains visible after ISA saturation."""
    monkeypatch.setattr(
        pipeline, "saturate_tensor", _exhaust_tensor_search("wall_clock_seconds")
    )
    search = pipeline.build_egraph_search(
        _input_graph(),
        wall_clock_seconds=10.0,
    )

    assert search.isa_status.status == "fixed_point"
    assert search.terminal_status.status == "truncated"
    assert search.terminal_status.truncated_stage == "tensor_propagation"


def test_private_extraction_outcome_distinguishes_empty_stream_and_exhaustion(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "build_egraph_search",
        lambda *_args, **_kwargs: _search(),
    )

    def empty(*_args: Any, **_kwargs: Any) -> Any:
        if False:
            yield _input_graph()

    monkeypatch.setattr(pipeline, "iter_materialized_isa_graphs", empty)
    drained = pipeline.iter_synthesized_hw_graphs(_input_graph())
    assert list(drained) == []
    assert drained.outcome.extraction_exhaustion == "no materialized ISA graph"
    assert drained.outcome.extraction_stage is None

    def exhaust(*_args: Any, **_kwargs: Any) -> Any:
        raise WallClockExceeded("extraction")
        yield

    monkeypatch.setattr(pipeline, "iter_materialized_isa_graphs", exhaust)
    exhausted = pipeline.iter_synthesized_hw_graphs(_input_graph())
    assert list(exhausted) == []
    assert exhausted.outcome.extraction_exhaustion == "extraction: wall_clock_seconds"
    assert exhausted.outcome.extraction_stage == "extraction"


def test_graph_iterator_yields_only_graphs(monkeypatch: Any) -> None:
    graph = _input_graph("result")
    monkeypatch.setattr(
        pipeline,
        "build_egraph_search",
        lambda *_args, **_kwargs: _search(),
    )
    monkeypatch.setattr(
        pipeline,
        "iter_materialized_isa_graphs",
        lambda *_args, **_kwargs: iter([graph]),
    )

    graphs = pipeline.iter_synthesized_hw_graphs(_input_graph())
    yielded = list(graphs)

    assert yielded == [graph]
    assert all(isinstance(item, nuGraph) for item in yielded)
    assert graphs.outcome.status == "completed"


def test_graph_iterator_remains_empty_on_terminal_failure(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "build_egraph_search",
        lambda *_args, **_kwargs: _search(
            status="failed",
            unrealized_outputs=("missing",),
        ),
    )

    def materialize(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("terminal failure must not start extraction")

    monkeypatch.setattr(pipeline, "iter_materialized_isa_graphs", materialize)
    graphs = pipeline.iter_synthesized_hw_graphs(_input_graph())

    assert list(graphs) == []
    assert graphs.outcome.status == "failed"
    assert graphs.outcome.unrealized_outputs == ("missing",)


def test_unrealized_outputs_report_failed_at_finalization(monkeypatch: Any) -> None:
    """An unrealized output is a finalization failure."""

    def truncate_with_unrealized(*args: Any, **kwargs: Any) -> Any:
        outputs = list(args[1])
        return (
            args[2],
            {},
            pipeline.LoweringStatus(
                status="truncated",
                unrealized_outputs=tuple(outputs),
                realized=0,
                reason="wall_clock_seconds",
                stage="lowering",
            ),
        )

    monkeypatch.setattr(pipeline, "lower_tensor_egraph", truncate_with_unrealized)
    search = pipeline.build_egraph_search(
        _input_graph(),
        tensor_max_rounds=0,
        wall_clock_seconds=10.0,
    )

    assert search.terminal_status.status == "failed"
    assert search.terminal_status.truncated_stage == "finalization"
    assert search.terminal_status.stop_reason == "wall_clock_seconds"
    assert search.terminal_status.unrealized_outputs

    monkeypatch.setattr(pipeline, "build_egraph_search", lambda *_a, **_k: search)
    graphs = pipeline.iter_synthesized_hw_graphs(_input_graph())
    assert list(graphs) == []
    assert graphs.outcome.status == "failed"
    assert graphs.outcome.truncated_stage == "finalization"


def test_lowering_truncation_with_every_output_realized_stays_truncated(
    monkeypatch: Any,
) -> None:
    original_lower = pipeline.lower_tensor_egraph

    def truncate_but_realize(*args: Any, **kwargs: Any) -> Any:
        adapter, lowered, status = original_lower(*args, **kwargs)
        return (
            adapter,
            lowered,
            pipeline.LoweringStatus(
                status="truncated",
                unrealized_outputs=(),
                attempted=status.attempted,
                realized=status.realized,
                proved_programs=status.proved_programs,
                reason="wall_clock_seconds",
                stage="lowering",
            ),
        )

    monkeypatch.setattr(pipeline, "lower_tensor_egraph", truncate_but_realize)
    search = pipeline.build_egraph_search(
        _input_graph(),
        tensor_max_rounds=0,
        wall_clock_seconds=10.0,
    )

    assert search.terminal_status.status == "truncated"
    assert search.terminal_status.truncated_stage == "lowering"
    assert search.terminal_status.unrealized_outputs == ()
    assert search.isa_output_roots
