"""Host integration tests for e-graph cache save and replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import axon.egraph.pipeline as pipeline
from axon import cli
from axon.egraph.persist import cache_file_path, load_cache
from axon.paths import run_paths

_KERNELS = Path(__file__).resolve().parents[1] / "kernels"
_SIZES = {"m": 128, "n": 128}
_LIVE_OUT = "out/live/mul_case.csv"
_CACHED_OUT = "out/cached/mul_case.csv"


def _mul_spec():
    return cli._load_spec(str(_KERNELS / "mul"))


def _emit(spec, *, out: str, from_cache: bool = False) -> Any:
    return cli.trace_kernel(
        spec,
        dict(_SIZES),
        warmup=1,
        bench=1,
        out=out,
        phase="emit",
        cache_dir="cache",
        from_cache=from_cache,
        synth_workers=2,
    )


def _cache_path() -> Path:
    return cache_file_path(Path("cache"), "mul", _SIZES, "mul_case")


def _modules(spec, out: str) -> dict[str, bytes]:
    run_dir = run_paths(spec, out).run_dir
    return {path.name: path.read_bytes() for path in sorted(run_dir.glob("*.py"))}


def test_synthesis_saves_both_egraphs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _emit(_mul_spec(), out=_LIVE_OUT)

    cache = load_cache(_cache_path())
    assert cache.tensor.classes
    assert cache.isa.classes
    assert cache.run_stem == "mul_case"


def test_cached_run_emits_identical_modules_without_synthesis(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    _emit(spec, out=_LIVE_OUT)
    live = _modules(spec, _LIVE_OUT)

    def _no_synthesis(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("--from-cache must not synthesize")

    monkeypatch.setattr(cli, "build_egraph_search", _no_synthesis)
    _emit(spec, out=_CACHED_OUT, from_cache=True)

    assert live
    assert _modules(spec, _CACHED_OUT) == live


def test_missing_cache_is_an_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        _emit(_mul_spec(), out=_CACHED_OUT, from_cache=True)

    assert exc.value.code == 2
    assert "no e-graph cache at" in capsys.readouterr().err


def test_cache_is_saved_before_extraction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        pipeline,
        "iter_materialized_isa_graphs",
        lambda *args, **kwargs: iter(()),
    )

    assert _emit(_mul_spec(), out=_LIVE_OUT) == 1
    assert _cache_path().is_file()


def test_save_failure_does_not_fail_emission(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()

    def _fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(cli, "save_cache", _fail)
    assert _emit(spec, out=_LIVE_OUT) is None
    assert _modules(spec, _LIVE_OUT)
    assert "e-graph cache not saved" in capsys.readouterr().err


def test_failed_search_is_saved_for_inspection_but_not_replayed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _fail_lowering(*args: Any, **kwargs: Any) -> Any:
        return (
            args[2],
            {},
            pipeline.LoweringStatus(
                status="failed",
                unrealized_outputs=tuple(args[1]),
                realized=0,
                reason="wall_clock_seconds",
            ),
        )

    monkeypatch.setattr(pipeline, "lower_tensor_egraph", _fail_lowering)
    assert _emit(_mul_spec(), out=_LIVE_OUT) == 1
    cache = load_cache(_cache_path())
    assert cache.tensor.classes
    assert cache.isa_output_roots == ()

    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _emit(_mul_spec(), out=_LIVE_OUT, from_cache=True)
    assert exc.value.code == 2
