"""Host tests for the streamed compile/bench pipeline and its resume store.

`run_nki_bench` writes one terminal row per key to `out/<stem>/results.csv` and
deletes that key's artifact dir right after the append, so an interrupted case
keeps every finished row, a resume skips those keys, and peak disk stays bounded
by `AXON_MAX_LIVE_NEFFS`. `out/<stem>.csv` is published only once a case has one
terminal row for every expected key and no others. Every test here drives the
pipeline through in-process thread pools with stub compile/bench workers, so no
device, toolchain, or NEFF is involved.
"""

from __future__ import annotations

import csv
import dataclasses
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import pytest

from axon import bench_runner, cli
from axon.bench_runner import BenchStats
from axon.paths import run_paths

_KERNELS = Path(__file__).resolve().parents[1] / "kernels"
_DIMS = {"m": 128, "n": 128}


def _mul_spec(tile_options=(1, 2)):
    # Shrink the tile-option space: the real spec has 6 options over 2 tile
    # args (36 keys per module), which is too slow for a host test.
    spec = cli._load_spec(str(_KERNELS / "mul"))
    return dataclasses.replace(spec, tile_options=tile_options)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("AXON_KEEP_NEFF", "AXON_MAX_LIVE_NEFFS", "AXON_COMPILE_WORKERS"):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------
# stubs and seams
# --------------------------------------------------------------------------


def _write_neff(neff_path: str, size: int = 64) -> None:
    """Mirror the real compile worker: materialize the artifact dir and a NEFF
    of known size, so live-artifact-dir counting is observable."""
    os.makedirs(os.path.dirname(neff_path), exist_ok=True)
    with open(neff_path, "wb") as fh:
        fh.write(b"\0" * size)


def _ok_compile(args):
    tile_values, neff_path = args[2], args[5]
    _write_neff(neff_path)
    return (tile_values, neff_path, None)


def _ok_bench(args):
    del args
    return BenchStats(
        mean_ms=1.0,
        median_ms=1.0,
        min_ms=1.0,
        max_ms=1.0,
        std_dev_ms=0.0,
        outputs=None,
    )


def _install_pools(monkeypatch, *, compile_workers=2, device_workers=1):
    """Swap the spawn process pools for in-process thread pools so stub
    workers (and their closures) stay reachable from the pipeline."""

    def _make(*, n_compile, core_specs):
        return (
            ThreadPoolExecutor(max_workers=n_compile),
            [ThreadPoolExecutor(max_workers=1) for _ in core_specs],
        )

    monkeypatch.setattr(bench_runner, "_make_pipeline_pools", _make)
    monkeypatch.setattr(
        bench_runner, "_visible_neuron_core_count", lambda: device_workers
    )
    monkeypatch.setenv("AXON_COMPILE_WORKERS", str(compile_workers))


def test_compile_worker_does_not_publish_an_empty_core_pin(monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "")
    bench_runner._init_compile_worker()
    assert "NEURON_RT_VISIBLE_CORES" not in os.environ


def _install_stubs(monkeypatch, compile_fn=_ok_compile, bench_fn=_ok_bench, **kw):
    _install_pools(monkeypatch, **kw)
    monkeypatch.setattr(bench_runner, "_compile_one_nki_kernel", compile_fn)
    monkeypatch.setattr(bench_runner, "_bench_one_neff", bench_fn)


def _bench(spec, paths, *, hw_variant=0, tile_variant=0, **kw):
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    return bench_runner.run_nki_bench(
        spec,
        _DIMS,
        hw_variant=hw_variant,
        tile_variant=tile_variant,
        module_path=str(paths.variant_module(spec.name, hw_variant, tile_variant)),
        function_name="kernel_tensor_mul",
        warmup=1,
        bench=1,
        paths=paths,
        rtol=1e-2,
        atol=1e-2,
        **kw,
    )


# --------------------------------------------------------------------------
# assertion helpers
# --------------------------------------------------------------------------


def _expected_keys(tile_options, *, hw_variant=0, tile_variant=0):
    return {
        ("1", "lnc1", str(hw_variant), str(tile_variant), str(a), str(b))
        for a, b in product(tile_options, repeat=2)
    }


def _row_keys(spec, path: Path) -> list[tuple[str, ...]]:
    """Key tuple of every record in a results/published CSV, in file order."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == bench_runner.result_columns(spec)
        return [tuple(row[c] for c in bench_runner.key_columns(spec)) for row in reader]


def _rows(spec, path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _artifact_dirs(paths) -> list[str]:
    return sorted(p.name for p in paths.run_dir.glob("__v*_tiles_*") if p.is_dir())


def _tiles_of(key: tuple[str, ...]) -> tuple[str, ...]:
    return key[-2:]


def _record_line(spec, tiles, *, hw_variant=0, tile_variant=0) -> str:
    """One complete results.csv record for `tiles`, in result_columns order."""
    vals = {
        "m": 128,
        "n": 128,
        "lnc": 1,
        "sharding": "lnc1",
        "hw_variant": hw_variant,
        "tile_variant": tile_variant,
        "TILES_M": tiles[0],
        "TILES_N": tiles[1],
        "mean_ms": 1.0,
        "median_ms": 1.0,
        "min_ms": 1.0,
        "max_ms": 1.0,
        "std_dev_ms": 0.0,
        "correct": "",
        "max_abs_err": "",
        "error": "",
        "mode": "",
    }
    return ",".join(str(vals[c]) for c in bench_runner.result_columns(spec))


# --------------------------------------------------------------------------
# 0. column contract
# --------------------------------------------------------------------------


def test_column_helpers_agree_on_one_layout():
    # The resume key is read back out of the CSV by column name, so the row
    # writer, the key reader and the winner reader must share one formula.
    spec = _mul_spec()
    tcols = bench_runner.tile_cols(spec.tile_args)
    assert tcols == ["TILES_M", "TILES_N"]
    assert bench_runner.KEY_PREFIX_COLS == [
        "lnc",
        "sharding",
        "hw_variant",
        "tile_variant",
    ]
    assert bench_runner.STATS_COLS == [
        "mean_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "std_dev_ms",
    ]
    assert bench_runner.key_columns(spec) == [*bench_runner.KEY_PREFIX_COLS, *tcols]
    assert bench_runner.result_columns(spec) == [
        *spec.dim_vars,
        *bench_runner.KEY_PREFIX_COLS,
        *tcols,
        *bench_runner.STATS_COLS,
        "correct",
        "max_abs_err",
        "error",
        "mode",
    ]


def test_read_result_keys_on_missing_store_is_empty(tmp_path):
    spec = _mul_spec()
    assert bench_runner.read_result_keys(spec, tmp_path / "results.csv") == set()
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert bench_runner.read_result_keys(spec, empty) == set()


def test_publish_results_is_a_noop_without_expected_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    bench_runner.publish_results(spec, paths, set())
    assert not paths.csv.exists()


# --------------------------------------------------------------------------
# 1. full case publication
# --------------------------------------------------------------------------


def test_full_case_publishes_one_row_per_expected_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    _install_stubs(monkeypatch)

    expected: set[tuple[str, ...]] = set()
    for tile_variant in (0, 1):
        got = _bench(spec, paths, tile_variant=tile_variant)
        assert got == _expected_keys(spec.tile_options, tile_variant=tile_variant)
        expected |= got

    assert not paths.csv.exists(), "run_nki_bench must never publish the case CSV"
    bench_runner.publish_results(spec, paths, expected)

    assert paths.csv.exists()
    assert paths.csv.read_text() == paths.results.read_text()
    counts = Counter(_row_keys(spec, paths.csv))
    assert set(counts) == expected
    assert set(counts.values()) == {1}


def test_candidate_filter_prunes_before_compile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_filtered.csv")
    module = paths.variant_module(spec.name, 0, 0)
    module.parent.mkdir(parents=True)
    module.write_text(
        "__axon_tile_constraints__ = ("
        "{'tile_arg': 'TILES_IN_BLOCK_M', 'extent': 128, 'tile_size': 128, "
        "'require_divisible': False}, "
        "{'tile_arg': 'TILES_IN_BLOCK_N', 'extent': 512, 'tile_size': 512, "
        "'require_divisible': False},)\n"
    )
    _install_stubs(monkeypatch)

    expected = _bench(spec, paths, candidate_filter=True)
    assert expected == {
        ("1", "lnc1", "0", "0", "1", "1"),
    }
    assert _row_keys(spec, paths.results) == list(expected)


def test_candidate_filter_prunes_a_non_dividing_block(tmp_path, monkeypatch):
    """The elementwise and reduce bodies loop `extent // BLOCK` with no remainder
    pass, so they publish `require_divisible`. A block that leaves a tail drops
    rows and can only fail the correctness check, so the filter must reject it
    before a compile worker starts."""
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_divisible.csv")
    module = paths.variant_module(spec.name, 0, 0)
    module.parent.mkdir(parents=True)
    module.write_text(
        "__axon_tile_constraints__ = ("
        "{'tile_arg': 'TILES_IN_BLOCK_M', 'extent': 128, 'tile_size': 128, "
        "'require_divisible': True}, "
        # 768 = 512 * 1 leaves 256, so only TILES_IN_BLOCK_N == 1 survives the
        # extent check, and it does not divide 768 either.
        "{'tile_arg': 'TILES_IN_BLOCK_N', 'extent': 768, 'tile_size': 512, "
        "'require_divisible': True},)\n"
    )
    _install_stubs(monkeypatch)

    assert _bench(spec, paths, candidate_filter=True) == set()


def test_candidate_filter_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_unfiltered.csv")
    module = paths.variant_module(spec.name, 0, 0)
    module.parent.mkdir(parents=True)
    module.write_text(
        "__axon_tile_constraints__ = ("
        "{'tile_arg': 'TILES_IN_BLOCK_M', 'extent': 128, 'tile_size': 128, "
        "'require_divisible': False},)\n"
    )
    _install_stubs(monkeypatch)

    expected = _bench(spec, paths, candidate_filter=False)
    assert expected == _expected_keys(spec.tile_options)


# --------------------------------------------------------------------------
# 2. compile/bench overlap
# --------------------------------------------------------------------------


def test_bench_overlaps_a_blocked_compile(tmp_path, monkeypatch):
    # Two phases would stall every bench behind the slowest compile. One key's
    # compile is held open; a different key must reach the device meanwhile.
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    monkeypatch.setenv("AXON_MAX_LIVE_NEFFS", "4")

    blocker = (1, 1)  # first key in product order, so it is submitted first
    entered = threading.Event()
    benched_other = threading.Event()
    release = threading.Event()

    def _compile(args):
        tile_values, neff_path = args[2], args[5]
        if tile_values == blocker:
            entered.set()
            release.wait(30)
        _write_neff(neff_path)
        return (tile_values, neff_path, None)

    def _bench_fn(args):
        if args[2] != blocker:
            benched_other.set()
        return _ok_bench(args)

    _install_stubs(
        monkeypatch,
        _compile,
        _bench_fn,
        compile_workers=2,
        device_workers=1,
    )

    outcome: dict[str, bool] = {}

    def _referee():
        # Always release, even on timeout, so a failure reports instead of hanging.
        outcome["entered"] = entered.wait(30)
        outcome["overlapped"] = benched_other.wait(30)
        release.set()

    referee = threading.Thread(target=_referee)
    referee.start()
    try:
        got = _bench(spec, paths)
    finally:
        release.set()
        referee.join(60)

    assert outcome.get("entered"), "blocked compile stub never ran"
    assert outcome.get("overlapped"), (
        "no bench executed while a compile was still blocked: compile and bench "
        "are still running as two phases"
    )
    assert got == _expected_keys(spec.tile_options)
    assert len(_row_keys(spec, paths.results)) == 4


# --------------------------------------------------------------------------
# 3./4. interruption and resume
# --------------------------------------------------------------------------


class _Stop(BaseException):
    """Stands in for a SIGTERM: a BaseException the pipeline must not swallow."""


def _interrupt_after(monkeypatch, k: int):
    """Let `k` rows land durably, then raise out of the append seam (which runs
    in the parent, like a signal would)."""
    real = bench_runner._append_result_row
    seen = {"n": 0}

    def _wrapper(*a, **kw):
        if seen["n"] >= k:
            raise _Stop("simulated SIGTERM")
        out = real(*a, **kw)
        seen["n"] += 1
        return out

    monkeypatch.setattr(bench_runner, "_append_result_row", _wrapper)


def test_interruption_after_k_rows_keeps_exactly_those_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    _install_stubs(monkeypatch, compile_workers=1, device_workers=1)
    _interrupt_after(monkeypatch, 2)

    with pytest.raises(_Stop):
        _bench(spec, paths)

    text = paths.results.read_text()
    assert text.endswith("\n"), "results.csv ends mid-record after an interruption"
    assert len(text.splitlines()) == 3  # header + 2 durable rows
    assert len(_row_keys(spec, paths.results)) == 2
    assert not paths.csv.exists(), "a partial run must not publish the case CSV"


def test_resume_runs_only_the_remaining_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    expected = _expected_keys(spec.tile_options)

    with monkeypatch.context() as first:
        _install_stubs(first, compile_workers=1, device_workers=1)
        _interrupt_after(first, 2)
        with pytest.raises(_Stop):
            _bench(spec, paths)

    done = bench_runner.read_result_keys(spec, paths.results)
    assert len(done) == 2
    remaining = {_tiles_of(k) for k in expected - done}

    compiled: list[tuple[str, ...]] = []
    benched: list[tuple[str, ...]] = []

    def _compile(args):
        compiled.append(tuple(str(v) for v in args[2]))
        return _ok_compile(args)

    def _bench_fn(args):
        benched.append(tuple(str(v) for v in args[2]))
        return _ok_bench(args)

    _install_stubs(monkeypatch, _compile, _bench_fn)
    got = _bench(spec, paths)

    assert got == expected, "resume must still report the full expected key set"
    assert sorted(compiled) == sorted(remaining)
    assert sorted(benched) == sorted(remaining)
    counts = Counter(_row_keys(spec, paths.results))
    assert set(counts) == expected
    assert set(counts.values()) == {1}


# --------------------------------------------------------------------------
# 5./6. terminal rows for failures
# --------------------------------------------------------------------------


@pytest.mark.parametrize("failure", ["neff_none", "future_raises"])
def test_compile_failure_writes_exactly_one_terminal_row(
    tmp_path, monkeypatch, failure
):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec(tile_options=(1,))
    paths = run_paths(spec, f"out/mul_{failure}.csv")

    def _compile(args):
        tile_values, neff_path = args[2], args[5]
        # Partial output on disk, as a real failed compile leaves behind.
        os.makedirs(os.path.dirname(neff_path), exist_ok=True)
        if failure == "future_raises":
            raise RuntimeError("compile blew up")
        return (tile_values, None, "neff_missing")

    def _bench_fn(args):
        raise AssertionError("a failed compile must never reach the device")

    _install_stubs(monkeypatch, _compile, _bench_fn)
    got = _bench(spec, paths)

    assert got == _expected_keys(spec.tile_options)
    rows = _rows(spec, paths.results)
    assert len(rows) == 1
    assert rows[0]["error"], "compile failure wrote no error text"
    assert not _artifact_dirs(paths), "partial compile output survived its row"


def test_bench_failure_writes_exactly_one_terminal_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec(tile_options=(1,))
    paths = run_paths(spec, "out/mul_t.csv")

    def _bench_fn(args):
        raise RuntimeError("device fell over")

    _install_stubs(monkeypatch, _ok_compile, _bench_fn)
    got = _bench(spec, paths)

    assert got == _expected_keys(spec.tile_options)
    rows = _rows(spec, paths.results)
    assert len(rows) == 1
    assert rows[0]["error"], "bench failure wrote no error text"
    assert not _artifact_dirs(paths)


# --------------------------------------------------------------------------
# 7./8. store integrity
# --------------------------------------------------------------------------


def test_torn_final_record_is_truncated_before_the_next_append(tmp_path, monkeypatch):
    # A kill mid-append leaves a record with no trailing newline. It is not a
    # terminal row, so it must be dropped rather than parsed or appended after.
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    paths.run_dir.mkdir(parents=True)
    torn = _record_line(spec, (1, 2))[:20]
    paths.results.write_text(
        ",".join(bench_runner.result_columns(spec))
        + "\n"
        + _record_line(spec, (1, 1))
        + "\n"
        + torn
    )

    assert bench_runner.read_result_keys(spec, paths.results) == {
        ("1", "lnc1", "0", "0", "1", "1")
    }
    text = paths.results.read_text()
    assert text.endswith("\n")
    assert len(text.splitlines()) == 2

    _install_stubs(monkeypatch)
    got = _bench(spec, paths)
    counts = Counter(_row_keys(spec, paths.results))
    assert set(counts) == got == _expected_keys(spec.tile_options)
    assert set(counts.values()) == {1}


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("malformed_interior", "malformed record"),
        ("duplicate_key", "duplicate key"),
        ("drift", "column drift"),
    ],
)
def test_damaged_store_stops_publication(tmp_path, monkeypatch, damage, message):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, f"out/mul_{damage}.csv")
    paths.run_dir.mkdir(parents=True)
    cols = bench_runner.result_columns(spec)

    if damage == "malformed_interior":
        lines = [
            ",".join(cols),
            _record_line(spec, (1, 1)),
            ",".join(_record_line(spec, (1, 2)).split(",")[:-3]),
            _record_line(spec, (2, 1)),
        ]
    elif damage == "duplicate_key":
        lines = [
            ",".join(cols),
            _record_line(spec, (1, 1)),
            _record_line(spec, (1, 1)),
        ]
    else:
        lines = [",".join([*cols, "extra"]), _record_line(spec, (1, 1)) + ",0"]
    paths.results.write_text("\n".join(lines) + "\n")

    # Match the integrity message: a bare AssertionError would also be raised by
    # the downstream coverage check, so the damage itself must be what fires.
    with pytest.raises(AssertionError, match=message):
        bench_runner.read_result_keys(spec, paths.results)
    with pytest.raises(AssertionError, match=message):
        bench_runner.publish_results(spec, paths, _expected_keys(spec.tile_options))
    assert not paths.csv.exists()


def test_incomplete_coverage_stops_publication(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    _install_stubs(monkeypatch)
    got = _bench(spec, paths)

    with pytest.raises(AssertionError):
        bench_runner.publish_results(
            spec, paths, got | {("2", "lnc2", "0", "0", "1", "1")}
        )
    assert not paths.csv.exists()

    extra = next(iter(got))
    with pytest.raises(AssertionError):
        bench_runner.publish_results(spec, paths, got - {extra})
    assert not paths.csv.exists()


# --------------------------------------------------------------------------
# 9./10./11./12. artifact lifetime
# --------------------------------------------------------------------------


def test_startup_clears_artifacts_from_an_interrupted_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec(tile_options=(1,))
    paths = run_paths(spec, "out/mul_t.csv")
    stale = paths.run_dir / "__v0_t0_tiles_9_9"
    stale.mkdir(parents=True)
    (stale / "kernel.neff").write_bytes(b"orphan from a killed run")

    _install_stubs(monkeypatch)
    _bench(spec, paths)

    assert not stale.exists(), "an orphaned artifact dir survived startup cleanup"
    assert not _artifact_dirs(paths)


def test_artifact_dir_outlives_its_incomplete_row_append(tmp_path, monkeypatch):
    # The delete must happen strictly after the append reports success, or an
    # interrupted append loses both the row and the NEFF it could resume from.
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    monkeypatch.setenv("AXON_MAX_LIVE_NEFFS", "2")

    ok_tiles, compile_fail, bench_fail = (1, 1), (1, 2), (2, 1)

    def _compile(args):
        tile_values, neff_path = args[2], args[5]
        if tile_values == compile_fail:
            os.makedirs(os.path.dirname(neff_path), exist_ok=True)
            return (tile_values, None, "neff_missing")
        _write_neff(neff_path)
        return (tile_values, neff_path, None)

    def _bench_fn(args):
        if args[2] == bench_fail:
            raise RuntimeError("device fell over")
        return _ok_bench(args)

    _install_stubs(
        monkeypatch, _compile, _bench_fn, compile_workers=1, device_workers=1
    )

    # Sample inside the append, at the fsync: the row is on disk by then and the
    # delete has not run yet. Wrapping _append_result_row instead would sample
    # before it and pass whichever order the real function used.
    real_fsync = os.fsync
    live: list[list[str]] = []

    def _spy_fsync(fd):
        out = real_fsync(fd)
        live.append(_artifact_dirs(paths))
        return out

    monkeypatch.setattr(os, "fsync", _spy_fsync)
    _bench(spec, paths)

    assert len(live) == 4
    for dirs in live:
        assert dirs, "artifact dir was deleted before its row reached disk"
    seen = {name for dirs in live for name in dirs}
    for tiles in (ok_tiles, compile_fail, bench_fail):
        tag = f"_tiles_{tiles[0]}_{tiles[1]}"
        assert any(name.endswith(tag) for name in seen), (
            f"no live artifact dir observed for {tiles}"
        )
    assert not _artifact_dirs(paths), "artifact dirs survived their terminal rows"


def test_keep_neff_preserves_every_artifact_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    monkeypatch.setenv("AXON_KEEP_NEFF", "1")
    _install_stubs(monkeypatch)

    _bench(spec, paths)

    assert len(_row_keys(spec, paths.results)) == 4
    assert _artifact_dirs(paths) == sorted(
        f"__v0_t0_tiles_{a}_{b}" for a, b in product(spec.tile_options, repeat=2)
    )


def test_live_artifact_dirs_never_exceed_the_max_live_bound(tmp_path, monkeypatch):
    # The bound is the whole point of streaming: a 216-key case at ~200 MB per
    # NEFF needs ~1 TB if every artifact dir stays live to the end.
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec(tile_options=(1, 2, 4, 8))
    paths = run_paths(spec, "out/mul_t.csv")
    monkeypatch.setenv("AXON_MAX_LIVE_NEFFS", "2")

    peak = {"n": 0}
    lock = threading.Lock()

    def _compile(args):
        result = _ok_compile(args)
        with lock:
            peak["n"] = max(peak["n"], len(_artifact_dirs(paths)))
        return result

    _install_stubs(monkeypatch, _compile, compile_workers=4, device_workers=2)
    got = _bench(spec, paths)

    assert got == _expected_keys(spec.tile_options)
    assert len(_row_keys(spec, paths.results)) == 16
    assert peak["n"] <= 2, f"{peak['n']} artifact dirs live with AXON_MAX_LIVE_NEFFS=2"
    assert not _artifact_dirs(paths)
