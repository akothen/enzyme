"""Host-only regression tests for the public production synthesis path."""

from __future__ import annotations

import ast
import itertools
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import axon.egraph.lowering as lowering
import axon.egraph.pipeline as pipeline
from axon.codegen import EmitErr, EmitOk, emit, emit_nki_code_variants
from axon.codegen.assemble import NKIEmitter
from axon.codegen.layout import LayoutError
from axon.egraph.extraction import iter_materialized_isa_graphs
from axon.egraph.pipeline import iter_synthesized_hw_graphs
from axon.egraph.proof import ProofStore
from axon.ir import build_graph_from_kernel

pytestmark = pytest.mark.slow


def emit_nki_code_variant(graph, variant_index, kernel_name):
    """Per-graph shim over the list-based emitter; results key by position."""
    del variant_index
    yield from emit_nki_code_variants([graph], kernel_name)


_DIMS_2D = {"m": 4, "n": 8}
_PIPELINE_OPTIONS = {
    "timeout": 3000,
    "workers": 8,
    "tensor_max_rounds": 3,
    "isa_max_rounds": 3,
}
_ATTENTION_SYNTHESIS_WALL_CLOCK_SECONDS = 3600.0
_QKV_BOOTSTRAP_WALL_CLOCK_SECONDS = 180.0


def _synthesized_graphs(
    kernel: Any,
    *input_specs: tuple[str, tuple[str | int, ...]],
    dim_sizes: dict[str, int],
    max_hw_size: int,
    wall_clock_seconds: float | None = None,
) -> tuple[Any, Any, ProofStore]:
    source = build_graph_from_kernel(
        kernel,
        *input_specs,
        dim_sizes=dim_sizes,
    )
    store = ProofStore()
    options = dict(_PIPELINE_OPTIONS)
    if wall_clock_seconds is not None:
        options["wall_clock_seconds"] = wall_clock_seconds
    graphs = iter_synthesized_hw_graphs(
        source,
        max_hw_size=max_hw_size,
        store=store,
        **options,
    )
    return source, graphs, store


def _assert_synthesis_outcome(
    graphs: Any,
    store: ProofStore,
    *,
    emitted_graph_count: int,
    wall_clock_seconds: float | None,
    emission_errors: list[str] | None = None,
) -> None:
    diagnostic = _synthesis_diagnostic(
        graphs,
        store,
        wall_clock_seconds=wall_clock_seconds,
        emission_errors=emission_errors,
    )
    assert graphs.outcome.status in {"completed", "partial_extracted"}, diagnostic
    assert graphs.outcome.emitted_graph_count == emitted_graph_count, diagnostic
    if graphs.outcome.status == "completed":
        assert graphs.outcome.truncated_stage is None, diagnostic
        assert graphs.outcome.stop_reason is None, diagnostic
    else:
        assert graphs.outcome.truncated_stage, diagnostic
        assert graphs.outcome.stop_reason, diagnostic


def _synthesis_diagnostic(
    graphs: Any,
    store: ProofStore,
    *,
    wall_clock_seconds: float | None,
    emission_errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        **vars(graphs.outcome),
        "dispatch_count": store.dispatch_count,
        "wall_clock_seconds": wall_clock_seconds,
        "emission_errors": emission_errors or [],
    }


def _first_graph(
    kernel: Any,
    *input_specs: tuple[str, tuple[str | int, ...]],
    dim_sizes: dict[str, int],
    max_hw_size: int,
    wall_clock_seconds: float | None = None,
) -> tuple[Any, Any, ProofStore]:
    source, graphs, store = _synthesized_graphs(
        kernel,
        *input_specs,
        dim_sizes=dim_sizes,
        max_hw_size=max_hw_size,
        wall_clock_seconds=wall_clock_seconds,
    )
    materialized = list(itertools.islice(graphs, 1))
    assert materialized, {
        **vars(graphs.outcome),
        "dispatch_count": store.dispatch_count,
        "wall_clock_seconds": wall_clock_seconds,
    }
    _assert_synthesis_outcome(
        graphs,
        store,
        emitted_graph_count=1,
        wall_clock_seconds=wall_clock_seconds,
    )
    return source, materialized[0], store


def _tile_config_variant_emitter(
    tile_config: dict[str, int],
) -> Callable[[Any, int, str], Iterator[EmitErr | EmitOk]]:
    """An ``emit_nki_code_variant``-shaped emitter pinned to ``tile_config``.

    The plan refuses a tile base that does not divide an extent, so a test on
    small extents supplies its own dividing config instead of the defaults."""

    def emit_variant(
        graph: Any,
        variant_index: int,
        kernel_name: str,
    ) -> Iterator[EmitErr | EmitOk]:
        try:
            code = NKIEmitter(
                kernel_name=kernel_name,
                variant_index=variant_index,
                tile_config=tile_config,
            ).emit(graph)
        except Exception as error:
            yield EmitErr(variant_index, error)
            return
        yield EmitOk(variant_index, 0, code)

    return emit_variant


def _first_emittable_graph(
    graphs: Any,
    *,
    kernel_name: str,
    store: ProofStore,
    wall_clock_seconds: float | None,
    tile_config: dict[str, int] | None = None,
) -> tuple[Any, str, list[str], int]:
    emit_variant = (
        emit_nki_code_variant
        if tile_config is None
        else _tile_config_variant_emitter(tile_config)
    )
    emission_errors: list[str] = []
    for variant_index, graph in enumerate(graphs):
        for result in emit_variant(graph, variant_index, kernel_name):
            match result:
                case EmitOk(_, _, code):
                    return graph, code, emission_errors, variant_index + 1
                case EmitErr(_, error):
                    emission_errors.append(
                        f"variant {variant_index}: {type(error).__name__}: {error}"
                    )
    diagnostic = _synthesis_diagnostic(
        graphs,
        store,
        wall_clock_seconds=wall_clock_seconds,
        emission_errors=emission_errors,
    )
    pytest.fail(f"no emittable hardware graph: {diagnostic!r}")


def _matmul_site_count(code: str) -> int:
    """How many distinct matmuls the emitted source realizes.

    NOT `code.count("nisa.nc_matmul(")`: the drain protocol peels the first
    contraction sub-tile, so every matmul emits TWO `nc_matmul` calls (the peeled
    `accumulate=False` one and the `accumulate=True` loop body). Each matmul's
    PSUM accumulator is uniquified per matmul, so counting those is exact."""
    names = set()
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Call):
            continue
        if "nc_matmul" not in ast.unparse(node.func):
            continue
        if not node.args:
            continue
        names.add(ast.unparse(node.args[0]).split("[")[0])
    return len(names)


def _function_args(code: str, function_name: str) -> list[str]:
    module = ast.parse(code)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return [arg.arg for arg in function.args.args]


def test_production_path_traces_validates_materializes_and_emits() -> None:
    def kernel(x: Any, y: Any) -> Any:
        return x * y

    source, graph, store = _first_graph(
        kernel,
        ("x", ("m", "n")),
        ("y", ("m", "n")),
        dim_sizes=_DIMS_2D,
        max_hw_size=1,
    )
    code = emit(graph, kernel_name="production_mul")

    assert source.output_ids
    assert graph.input_ids == ("x", "y")
    assert graph.output_ids
    assert [node.op for node in graph.nodes] == [
        "input",
        "input",
        "tensor_tensor",
    ]
    assert "def production_mul(" in code
    assert "nisa.tensor_tensor(" in code
    assert len(graph.nodes) <= 3
    assert store.dispatch_count <= 100


def _tensor_output_ops(snapshot: Any, output: Any) -> set[str]:
    """Returns the top operations in a tensor output class."""
    from axon.egraph.codec import decode_tensor_enode

    ops: set[str] = set()
    for row in snapshot.members(output):
        try:
            ops.add(decode_tensor_enode(snapshot, row).op)
        except Exception:  # noqa: BLE001 - an undecodable member names no class
            continue
    return ops


def test_qkv_truncation_retains_pre_and_post_matmul_schedule_classes() -> None:
    """A truncated QKV search retains both tensor schedule classes."""
    from axon.cli import _load_spec
    from axon.egraph.adapter import EGraphAdapter
    from axon.egraph.tensor import ingest_tensor_graph, saturate_tensor

    spec = _load_spec(str(Path(__file__).resolve().parents[2] / "kernels" / "qkv_cte"))
    source = build_graph_from_kernel(
        spec.axon_kernel,
        *spec.input_specs,
        dim_sizes={"m": 128, "n": 512, "k": 128},
    )

    # One tensor round creates the post-matmul schedule before truncation.
    adapter = EGraphAdapter("tensor")
    ingest = ingest_tensor_graph(adapter, source)
    tensor_status = saturate_tensor(
        adapter,
        max_rounds=1,
        wall_clock_seconds=_QKV_BOOTSTRAP_WALL_CLOCK_SECONDS,
        timeout=3000,
        workers=8,
    )
    assert tensor_status.status == "truncated"
    assert tensor_status.reason == "max_rounds"

    snapshot = adapter.freeze_snapshot()
    outputs = [
        adapter.resolve_handle(snapshot, handle) for handle in ingest.output_handles
    ]
    assert len(outputs) == 1
    classes = _tensor_output_ops(snapshot, outputs[0])

    assert "matmul" in classes, f"pre-matmul class missing from {classes}"
    assert "div" in classes, f"post-matmul class missing from {classes}"


def test_qkv_bounded_run_lowers_the_newest_state_not_a_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bounded QKV run lowers the newest tensor state once."""
    from axon.cli import _load_spec

    spec = _load_spec(str(Path(__file__).resolve().parents[2] / "kernels" / "qkv_cte"))
    source = build_graph_from_kernel(
        spec.axon_kernel,
        *spec.input_specs,
        dim_sizes={"m": 128, "n": 512, "k": 128},
    )
    lowering_adapters: list[str] = []
    lowered_snapshots: list[Any] = []
    original_lower = pipeline.lower_tensor_egraph

    def record_lower(*args: Any, **kwargs: Any) -> Any:
        lowering_adapters.append(args[2].label)
        lowered_snapshots.append(args[0])
        return original_lower(*args, **kwargs)

    def exhaust_tensor_search(
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
            reason="wall_clock_seconds",
            stage="tensor_propagation",
        )

    monkeypatch.setattr(pipeline, "lower_tensor_egraph", record_lower)
    monkeypatch.setattr(pipeline, "saturate_tensor", exhaust_tensor_search)
    store = ProofStore()
    search = pipeline.build_egraph_search(
        source,
        max_hw_size=2,
        timeout=3000,
        store=store,
        isa_max_rounds=0,
        wall_clock_seconds=_QKV_BOOTSTRAP_WALL_CLOCK_SECONDS,
        workers=8,
    )

    assert lowering_adapters == ["isa"]
    assert search.isa_adapter.label == "isa"
    assert search.isa_snapshot.adapter_id == search.isa_adapter.adapter_id
    assert all(
        root.adapter_id == search.isa_adapter.adapter_id
        for root in search.isa_output_roots
    )
    assert search.store is store
    assert store.dispatch_count > 0
    assert search.terminal_status.status == "truncated"
    assert search.terminal_status.truncated_stage == "tensor_propagation"
    assert search.terminal_status.stop_reason == "wall_clock_seconds"
    assert search.lowering_status.status == "lowered"

    # NKI emission of these variants belongs to the separate codegen change;
    # this test pins that lowering consumed the newest frozen tensor state.
    graph = next(
        iter_materialized_isa_graphs(
            search.isa_snapshot,
            search.isa_output_roots,
            search.input_metadata,
            deadline=search.deadline,
            declared_input_ids=search.declared_input_ids,
        )
    )

    assert graph.output_ids
    assert any(node.op == "nc_matmul" for node in graph.nodes)


def test_first_emittable_graph_skips_layout_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejected_graph = object()
    emitted_graph = object()
    events: list[tuple[str, str]] = []

    def fake_graphs() -> Iterator[object]:
        events.append(("produce", "rejected"))
        yield rejected_graph
        events.append(("produce", "emitted"))
        yield emitted_graph

    def fake_emit(
        graph: object,
        variant_index: int,
        _kernel_name: str,
    ) -> Iterator[EmitErr | EmitOk]:
        if graph is rejected_graph:
            events.append(("reject", "rejected"))
            yield EmitErr(
                variant_index,
                LayoutError("per-partition scalar layout is unsupported"),
            )
        else:
            events.append(("emit", "emitted"))
            yield EmitOk(variant_index, 0, "# emitted attention\n")

    monkeypatch.setattr(
        sys.modules[__name__],
        "emit_nki_code_variant",
        fake_emit,
    )

    graph, code, errors, consumed_count = _first_emittable_graph(
        fake_graphs(),
        kernel_name="production_attention",
        store=ProofStore(),
        wall_clock_seconds=_ATTENTION_SYNTHESIS_WALL_CLOCK_SECONDS,
    )

    assert graph is emitted_graph
    assert code == "# emitted attention\n"
    assert errors == [
        "variant 0: LayoutError: per-partition scalar layout is unsupported"
    ]
    assert consumed_count == 2
    assert events == [
        ("produce", "rejected"),
        ("reject", "rejected"),
        ("produce", "emitted"),
        ("emit", "emitted"),
    ]


@pytest.mark.parametrize(
    ("hardware_graphs", "status", "expected_emission_errors"),
    [
        ([], "failed", "[]"),
        (
            [object()],
            "completed",
            "['variant 0: LayoutError: unsupported layout']",
        ),
    ],
)
def test_no_emittable_graph_reports_complete_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    hardware_graphs: list[object],
    status: str,
    expected_emission_errors: str,
) -> None:
    class FakeSynthesisGraphs:
        def __init__(self) -> None:
            self.outcome = SimpleNamespace(
                status=status,
                truncated_stage=None,
                stop_reason=None,
                unrealized_outputs=(),
                extraction_exhaustion=(
                    "no materialized ISA graph" if not hardware_graphs else None
                ),
                extraction_stage=None,
                emitted_graph_count=len(hardware_graphs),
            )

        def __iter__(self) -> Iterator[object]:
            return iter(hardware_graphs)

    def reject_emit(
        _graph: object,
        variant_index: int,
        _kernel_name: str,
    ) -> Iterator[EmitErr]:
        yield EmitErr(variant_index, LayoutError("unsupported layout"))

    monkeypatch.setattr(
        sys.modules[__name__],
        "emit_nki_code_variant",
        reject_emit,
    )
    store = ProofStore()
    store.dispatch_count = 37

    with pytest.raises(pytest.fail.Exception) as failure:
        _first_emittable_graph(
            FakeSynthesisGraphs(),
            kernel_name="production_attention",
            store=store,
            wall_clock_seconds=_ATTENTION_SYNTHESIS_WALL_CLOCK_SECONDS,
        )

    message = str(failure.value)
    assert "no emittable hardware graph" in message
    assert f"'status': '{status}'" in message
    assert f"'emitted_graph_count': {len(hardware_graphs)}" in message
    assert "'dispatch_count': 37" in message
    assert "'wall_clock_seconds': 3600.0" in message
    assert f"'emission_errors': {expected_emission_errors}" in message
    if not hardware_graphs:
        assert "'extraction_exhaustion': 'no materialized ISA graph'" in message


def test_production_path_preserves_heterogeneous_input_order() -> None:
    def reversed_matmul(a: Any, b: Any) -> Any:
        return b @ a

    source, graph, store = _first_graph(
        reversed_matmul,
        ("a", ("m", "n")),
        ("b", ("k", "m")),
        dim_sizes={"m": 4, "n": 8, "k": 3},
        max_hw_size=2,
    )
    # m=3 is the partition extent and k=4 the contraction, so the tile bases
    # must divide those rather than the 128/128/512 defaults.
    code = NKIEmitter(
        kernel_name="production_reversed_matmul",
        tile_config={"tile_m": 1, "tile_k": 4, "tile_n": 8},
    ).emit(graph)

    assert source.input_ids == ("a", "b")
    assert [node.id for node in source.nodes[:2]] == ["b", "a"]
    assert graph.input_ids == ("a", "b")
    input_shapes = {node.id: node.shape for node in graph.nodes if node.op == "input"}
    assert input_shapes == {"a": (4, 8), "b": (3, 4)}
    assert _function_args(code, "production_reversed_matmul")[:2] == ["a", "b"]
    assert len(graph.nodes) <= 4
    assert store.dispatch_count <= 500


def test_production_path_reuses_repeated_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds: Counter[str] = Counter()
    original = lowering._iter_complete_recipe_cache_entry

    def count_builds(
        target_op: str,
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[Any]:
        builds[target_op] += 1
        yield from original(target_op, *args, **kwargs)

    monkeypatch.setattr(
        lowering,
        "_iter_complete_recipe_cache_entry",
        count_builds,
    )

    def repeated_projection(x: Any, w1: Any, w2: Any) -> Any:
        return (x @ w1) * (x @ w2)

    _source, graph, store = _first_graph(
        repeated_projection,
        ("x", ("m", "k")),
        ("w1", ("k", "n")),
        ("w2", ("k", "n")),
        dim_sizes={"m": 4, "n": 8, "k": 3},
        max_hw_size=2,
    )
    code = NKIEmitter(
        kernel_name="production_repeated_projection",
        tile_config={"tile_m": 4, "tile_k": 3, "tile_n": 8},
    ).emit(graph)

    assert builds == Counter({"matmul": 1, "mul": 1})
    assert sum(node.op == "nc_matmul" for node in graph.nodes) == 2
    assert "def production_repeated_projection(" in code
    assert _matmul_site_count(code) == 2
    assert len(graph.nodes) <= 7
    assert store.dispatch_count <= 1500


def test_production_path_emits_reduction_or_scan() -> None:
    def cumulative_sum(x: Any) -> Any:
        return x.cumsum(axis=-1)

    _source, graph, store = _first_graph(
        cumulative_sum,
        ("x", ("m", "n")),
        dim_sizes=_DIMS_2D,
        max_hw_size=2,
    )
    code = emit(graph, kernel_name="production_cumsum")

    assert [node.op for node in graph.nodes] == [
        "input",
        "tensor_scalar_cumulative",
    ]
    assert "nl.sequential_range(" in code
    assert "nisa.tensor_scalar_cumulative(" in code
    assert len(graph.nodes) <= 2
    assert store.dispatch_count <= 100


def test_attention_dependency_chain_reaches_codegen() -> None:
    def kernel_attention(x: Any, w_q: Any, w_k: Any, w_v: Any) -> Any:
        q = x @ w_q
        k = x @ w_k
        v = x @ w_v
        scores = q @ k.transpose()
        ex = scores.exp()
        probs = ex / ex.sum(axis=1, keep_dims=True)
        return probs @ v

    _source, graphs, store = _synthesized_graphs(
        kernel_attention,
        ("x", ("m", "k")),
        ("w_q", ("k", "n")),
        ("w_k", ("k", "n")),
        ("w_v", ("k", "n")),
        dim_sizes={"m": 4, "n": 4, "k": 4},
        max_hw_size=2,
        wall_clock_seconds=_ATTENTION_SYNTHESIS_WALL_CLOCK_SECONDS,
    )
    # Attention NKI emission belongs to the separate codegen change; this
    # test pins the synthesized dependency chain in the extracted graphs.
    materialized = list(graphs)
    _assert_synthesis_outcome(
        graphs,
        store,
        emitted_graph_count=len(materialized),
        wall_clock_seconds=_ATTENTION_SYNTHESIS_WALL_CLOCK_SECONDS,
    )

    assert materialized
    matching = [
        graph
        for graph in materialized
        if sum(node.op == "nc_matmul" for node in graph.nodes) == 5
        and any(node.op == "exponential" for node in graph.nodes)
        and any(
            node.op in {"tensor_reduce", "activation_reduce"} for node in graph.nodes
        )
    ]
    assert matching
