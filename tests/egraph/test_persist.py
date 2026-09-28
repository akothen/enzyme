"""Persistence and validation tests for saved e-graph searches."""

from __future__ import annotations

import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from axon.egraph import persist
from axon.egraph.extraction import iter_materialized_isa_graphs
from axon.egraph.persist import (
    CacheError,
    build_cache_file,
    cache_file_path,
    encode_snapshot,
    load_cache,
    save_cache,
)
from axon.egraph.pipeline import build_egraph_search
from axon.ir import build_graph_from_kernel, nuGraph

_DIM_SIZES = {"m": 4, "k": 4}


def _kernel(x: Any, y: Any) -> Any:
    return x * y


def _graph() -> nuGraph:
    return build_graph_from_kernel(
        _kernel,
        ("x", ("m", "k")),
        ("y", ("m", "k")),
        dim_sizes=_DIM_SIZES,
    )


def _search(graph: nuGraph | None = None) -> Any:
    return build_egraph_search(
        graph if graph is not None else _graph(),
        max_hw_size=2,
        tensor_max_rounds=0,
        wall_clock_seconds=30.0,
        workers=2,
    )


_RUN_STEM = "mul_case"


def _cache(search: Any, graph: nuGraph) -> persist.EGraphCacheFile:
    return build_cache_file(
        search,
        kernel_name="mul",
        dim_sizes=_DIM_SIZES,
        graph_identity=graph.identity(),
        run_stem=_RUN_STEM,
        options={"max_hw_size": 2},
    )


def _identities(graphs: list[nuGraph]) -> list[str]:
    return [g.identity() for g in graphs]


def test_snapshot_round_trip_preserves_classes_and_row_order() -> None:
    search = _search()
    for live in (search.isa_snapshot, search.tensor_snapshot):
        decoded = encode_snapshot(live).rebuild()
        assert decoded.snapshot_id == live.snapshot_id
        assert decoded.adapter_id == live.adapter_id
        assert len(decoded.classes) == len(live.classes)
        # Class order and per-class row order both drive extraction, so compare
        # the sequences, not sets.
        for (live_ref, live_rows), (ref, rows) in zip(
            live.classes.items(), decoded.classes.items(), strict=True
        ):
            assert repr(ref) == repr(live_ref)
            assert len(rows) == len(live_rows)
            for row, live_row in zip(rows, live_rows, strict=True):
                assert row.egg_fn == live_row.egg_fn
                assert row.sort == live_row.sort
                assert row.callable == live_row.callable
                assert row._sort_key() == live_row._sort_key()
        assert [(name, repr(ref)) for name, ref in decoded.let_bindings.items()] == [
            (name, repr(ref)) for name, ref in live.let_bindings.items()
        ]


def test_round_trip_through_a_file_preserves_graph_identities(tmp_path: Path) -> None:
    graph = _graph()
    search = _search(graph)
    path = tmp_path / "mul.bin"
    save_cache(path, _cache(search, graph))
    loaded = load_cache(path)

    live = _identities(
        list(
            iter_materialized_isa_graphs(
                search.isa_snapshot,
                search.isa_output_roots,
                search.input_metadata,
                declared_input_ids=search.declared_input_ids,
                workers=1,
            )
        )
    )
    replayed = _identities(
        list(
            iter_materialized_isa_graphs(
                loaded.isa.rebuild(),
                list(loaded.isa_output_roots),
                loaded.input_metadata,
                declared_input_ids=loaded.declared_input_ids,
                workers=1,
            )
        )
    )
    assert replayed == live
    assert loaded.terminal_status == search.terminal_status
    assert loaded.tensor.classes  # both e-graphs are saved, per the plan


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    graph = _graph()
    path = tmp_path / "nested" / "mul.bin"
    save_cache(path, _cache(_search(graph), graph))
    assert path.is_file()
    assert sorted(p.name for p in path.parent.iterdir()) == ["mul.bin"]


def test_concurrent_saves_use_distinct_temp_files(
    tmp_path: Path, monkeypatch: Any
) -> None:
    graph = _graph()
    cache = _cache(_search(graph), graph)
    caches = [
        replace(cache, run_stem="writer_one"),
        replace(cache, run_stem="writer_two"),
    ]
    path = tmp_path / "mul.bin"
    barrier = threading.Barrier(2)
    real_dump = persist.pickle.dump

    def synchronized_dump(*args: Any, **kwargs: Any) -> None:
        barrier.wait()
        real_dump(*args, **kwargs)

    monkeypatch.setattr(persist.pickle, "dump", synchronized_dump)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(save_cache, path, cache_) for cache_ in caches]
        for future in futures:
            future.result()

    assert load_cache(path).run_stem in {"writer_one", "writer_two"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mul.bin"]


def test_cache_file_path_names_kernel_and_sizes(tmp_path: Path) -> None:
    assert cache_file_path(tmp_path, "rmsnorm", {"m": 1024, "n": 4096}) == (
        tmp_path / "rmsnorm__m1024_n4096.bin"
    )


def test_cache_file_path_prefers_the_run_stem(tmp_path: Path) -> None:
    # Two dtype twins share (kernel, sizes); the stem is what separates them.
    plain = cache_file_path(tmp_path, "cumsum", {"m": 1024, "n": 1024}, "cumsum_fast")
    bf16 = cache_file_path(
        tmp_path, "cumsum", {"m": 1024, "n": 1024}, "cumsum_fast_bf16"
    )
    assert plain == tmp_path / "cumsum_fast.bin"
    assert plain != bf16


# -- validation -------------------------------------------------------------


def test_missing_file_names_the_expected_path(tmp_path: Path) -> None:
    with pytest.raises(CacheError, match="no e-graph cache at"):
        load_cache(tmp_path / "absent.bin")


def test_corrupt_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mul.bin"
    path.write_bytes(b"not a pickle")
    with pytest.raises(CacheError, match="unreadable e-graph cache"):
        load_cache(path)


def test_non_cache_pickle_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mul.bin"
    path.write_bytes(pickle.dumps({"format_version": persist.FORMAT_VERSION}))
    with pytest.raises(CacheError, match="not an e-graph cache file"):
        load_cache(path)


def test_format_version_mismatch_is_rejected(tmp_path: Path, monkeypatch: Any) -> None:
    graph = _graph()
    path = tmp_path / "mul.bin"
    save_cache(path, _cache(_search(graph), graph))
    monkeypatch.setattr(persist, "FORMAT_VERSION", persist.FORMAT_VERSION + 1)
    with pytest.raises(CacheError, match="cache format version"):
        load_cache(path)


def test_egglog_version_mismatch_is_rejected(tmp_path: Path, monkeypatch: Any) -> None:
    graph = _graph()
    path = tmp_path / "mul.bin"
    save_cache(path, _cache(_search(graph), graph))
    monkeypatch.setattr(persist, "installed_egglog_version", lambda: "0.0.0-test")
    with pytest.raises(CacheError, match="written against egglog"):
        load_cache(path)


def test_kernel_and_size_mismatch_are_rejected() -> None:
    graph = _graph()
    cache = _cache(_search(graph), graph)
    path = Path("cache/mul.bin")
    with pytest.raises(CacheError, match="cache holds kernel"):
        cache.require_matches(
            kernel_name="rmsnorm",
            dim_sizes=_DIM_SIZES,
            graph_identity=graph.identity(),
            run_stem=_RUN_STEM,
            path=path,
        )
    with pytest.raises(CacheError, match="cache holds sizes"):
        cache.require_matches(
            kernel_name="mul",
            dim_sizes={"m": 8, "k": 4},
            graph_identity=graph.identity(),
            run_stem=_RUN_STEM,
            path=path,
        )


def test_another_runs_cache_is_rejected() -> None:
    # A file that landed under this run's path but was written by a different
    # case (dtype twins share a key) must not be replayed.
    graph = _graph()
    cache = _cache(_search(graph), graph)
    with pytest.raises(CacheError, match="written by run"):
        cache.require_matches(
            kernel_name="mul",
            dim_sizes=_DIM_SIZES,
            graph_identity=graph.identity(),
            run_stem="mul_other_case",
            path=Path("cache/mul.bin"),
        )


def test_changed_kernel_body_is_rejected_as_stale() -> None:
    # Same kernel name and sizes, different traced graph: the identity guard is
    # the only thing standing between a codegen test and a stale search.
    graph = _graph()
    cache = _cache(_search(graph), graph)
    other = build_graph_from_kernel(
        lambda x, y: x + y,
        ("x", ("m", "k")),
        ("y", ("m", "k")),
        dim_sizes=_DIM_SIZES,
    )
    assert other.identity() != graph.identity()
    with pytest.raises(CacheError, match="traced graph changed"):
        cache.require_matches(
            kernel_name="mul",
            dim_sizes=_DIM_SIZES,
            graph_identity=other.identity(),
            run_stem=_RUN_STEM,
            path=Path("cache/mul.bin"),
        )


def test_partially_realized_failed_search_is_saved_for_inspection(
    tmp_path: Path,
) -> None:
    graph = _graph()
    cache = _cache(_search(graph), graph)
    failed = replace(
        cache,
        terminal_status=replace(
            cache.terminal_status,
            status="failed",
            unrealized_outputs=(cache.tensor_output_roots[0],),
        ),
    )
    path = tmp_path / "partial.bin"
    save_cache(path, failed)
    loaded = load_cache(path)

    assert loaded.isa_output_roots
    assert loaded.terminal_status.status == "failed"
    with pytest.raises(CacheError, match="realized no output"):
        loaded.require_matches(
            kernel_name="mul",
            dim_sizes=_DIM_SIZES,
            graph_identity=graph.identity(),
            run_stem=_RUN_STEM,
            path=path,
        )


def test_matching_cache_passes_validation() -> None:
    graph = _graph()
    cache = _cache(_search(graph), graph)
    cache.require_matches(
        kernel_name="mul",
        dim_sizes=_DIM_SIZES,
        graph_identity=graph.identity(),
        run_stem=_RUN_STEM,
        path=Path("cache/mul.bin"),
    )
