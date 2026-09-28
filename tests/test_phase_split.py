"""Host tests for the emit/bench phase split (no device).

`--phase emit` runs stages 1-5 (synthesize -> emit -> write per-variant
modules) and must never touch the bench; `--phase bench` skips synthesis,
discovers the emitted modules in the run dir, and benches them; `--phase all`
(default) is the legacy single-shot loop.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from axon import cli
from axon.paths import run_paths

_KERNELS = Path(__file__).resolve().parents[1] / "kernels"


def _mul_spec():
    return cli._load_spec(str(_KERNELS / "mul"))


def _fake_emit(hw_variants, name):
    # One graph's worth of fake emitted code (two tile configs), matching the
    # list-based EmitOk contract of `emit_nki_code_variants`.
    from axon.codegen import EmitOk

    for i, _g_hw in enumerate(hw_variants):
        yield EmitOk(i, 0, "# fake kernel v0 t0\n")
        yield EmitOk(i, 1, "# fake kernel v0 t1\n")


def _patch_emit_pipeline(monkeypatch, *, emitter=_fake_emit):
    monkeypatch.setattr(cli, "build_egraph_search", lambda G, **kw: object())
    monkeypatch.setattr(cli, "_save_search_cache", lambda *a, **kw: None)
    monkeypatch.setattr(
        cli, "iter_hw_graphs_from_search", lambda search, **kw: iter([object()])
    )
    monkeypatch.setattr(cli, "emit_nki_code_variants", emitter)
    monkeypatch.setattr(cli, "print_graph", lambda G: None)


def test_phase_emit_writes_modules_and_never_benches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    # The streamed synthesis path yields one hardware graph; the fake per-graph
    # emitter turns it into two tile modules.
    _patch_emit_pipeline(monkeypatch)

    def _boom(*a, **kw):
        raise AssertionError("run_nki_bench must not be called in --phase emit")

    monkeypatch.setattr(cli, "run_nki_bench", _boom)

    cli.trace_kernel(
        spec,
        {"m": 128, "n": 128},
        warmup=1,
        bench=1,
        out="out/mul_t.csv",
        phase="emit",
    )
    paths = run_paths(spec, "out/mul_t.csv")
    written = sorted(p.name for p in paths.run_dir.glob("*.py"))
    assert written == ["mul__v0_t0.py", "mul__v0_t1.py"]


def test_phase_emit_forwards_synthesis_controls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    seen: dict = {}

    def _synthesize(G, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "build_egraph_search", _synthesize)
    monkeypatch.setattr(cli, "_save_search_cache", lambda *a, **kw: None)
    monkeypatch.setattr(
        cli, "iter_hw_graphs_from_search", lambda search, **kw: iter([object()])
    )
    monkeypatch.setattr(cli, "emit_nki_code_variants", _fake_emit)
    monkeypatch.setattr(cli, "print_graph", lambda G: None)
    cli.trace_kernel(
        spec,
        {"m": 128, "n": 128},
        warmup=1,
        bench=1,
        out="out/mul_t.csv",
        phase="emit",
        solver_timeout_ms=7000,
        synthesis_timeout_seconds=7200.0,
        synth_workers=16,
    )
    assert seen["timeout"] == 7000
    assert seen["wall_clock_seconds"] == 7200.0
    assert seen["workers"] == 16


def test_cli_reports_structured_synthesis_failure(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()

    class FailedSynthesis:
        outcome = SimpleNamespace(
            status="failed",
            truncated_stage="isa_fusion",
            stop_reason="wall_clock_seconds",
            extraction_exhaustion="no materialized ISA graph",
            extraction_stage="extraction_materialization",
        )

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(
        cli,
        "build_egraph_search",
        lambda G, **kwargs: object(),
    )
    monkeypatch.setattr(cli, "_save_search_cache", lambda *a, **kw: None)
    monkeypatch.setattr(
        cli,
        "iter_hw_graphs_from_search",
        lambda search, **kwargs: FailedSynthesis(),
    )
    monkeypatch.setattr(cli, "print_graph", lambda G: None)

    assert (
        cli.trace_kernel(
            spec,
            {"m": 128, "n": 128},
            warmup=1,
            bench=1,
            out="out/mul_t.csv",
            phase="emit",
        )
        == 1
    )

    output = capsys.readouterr().out
    assert "synthesis status: failed" in output
    assert "stage=extraction_materialization" in output
    assert "reason=no materialized ISA graph" in output


def test_phase_bench_discovers_modules_and_benches_each(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    paths.run_dir.mkdir(parents=True)
    (paths.run_dir / "mul__v0_t0.py").write_text("# v0t0\n")
    (paths.run_dir / "mul__v0_t1.py").write_text("# v0t1\n")

    benched: list[tuple[int, int]] = []

    def _record(spec_, dims, *, hw_variant, tile_variant, **kw):
        benched.append((hw_variant, tile_variant))
        # run_nki_bench returns its expected key set for the caller to union.
        return {(str(hw_variant), str(tile_variant))}

    monkeypatch.setattr(cli, "run_nki_bench", _record)
    monkeypatch.setattr(cli, "publish_results", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "pick_winner", lambda *a, **kw: None)

    def _no_synth(*a, **kw):
        raise AssertionError("synthesis must not run in --phase bench")

    monkeypatch.setattr(cli, "build_egraph_search", _no_synth)

    with pytest.raises(AssertionError, match="no timed variants"):
        cli.trace_kernel(
            spec,
            {"m": 128, "n": 128},
            warmup=1,
            bench=1,
            out="out/mul_t.csv",
            phase="bench",
        )
    assert benched == [(0, 0), (0, 1)]


def test_phase_bench_no_emitted_modules_exits_loud(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    with pytest.raises(SystemExit):
        cli.trace_kernel(
            spec,
            {"m": 128, "n": 128},
            warmup=1,
            bench=1,
            out="out/mul_t.csv",
            phase="bench",
        )


def test_phase_emit_rejects_lnc2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    with pytest.raises(SystemExit):
        cli.trace_kernel(
            spec,
            {"m": 128, "n": 128},
            warmup=1,
            bench=1,
            out="out/mul_t.csv",
            phase="emit",
            lnc=2,
        )


def test_out_accepts_bare_stem(tmp_path, monkeypatch):
    # `--out out/matmul_sq1k` (no .csv) is the stem: CSV becomes <stem>.csv,
    # run dir <stem>/ — identical to passing out/matmul_sq1k.csv. Must hold
    # even when the run dir already exists (a second run after an emit), which
    # under the old existing-dir-container rule silently switched to
    # <dir>/nki_<name>.csv.
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    rp = run_paths(spec, "out/mul_t")
    assert rp.csv.name == "mul_t.csv"
    assert rp.run_dir.name == "mul_t"
    rp.run_dir.mkdir(parents=True)  # second-run scenario
    rp2 = run_paths(spec, "out/mul_t")
    assert rp2.csv == rp.csv


def test_cli_zero_emitted_modules_exits_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _patch_emit_pipeline(monkeypatch, emitter=lambda *args: iter(()))

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                str(_KERNELS / "mul"),
                "--sizes",
                "128",
                "128",
                "--out",
                "out/mul_case.csv",
                "--phase",
                "emit",
            ]
        )

    assert exc.value.code == 1
    assert "emission failed for mul: no modules emitted" in capsys.readouterr().out


def test_cli_zero_synthesized_graphs_exits_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli,
        "build_egraph_search",
        lambda graph, **kwargs: object(),
    )
    monkeypatch.setattr(cli, "_save_search_cache", lambda *a, **kw: None)
    monkeypatch.setattr(
        cli,
        "iter_hw_graphs_from_search",
        lambda search, **kwargs: iter(()),
    )
    monkeypatch.setattr(cli, "print_graph", lambda graph: None)

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                str(_KERNELS / "mul"),
                "--sizes",
                "128",
                "128",
                "--out",
                "out/mul_case.csv",
                "--phase",
                "emit",
            ]
        )

    assert exc.value.code == 1
    assert "synthesis failed for mul: no hw graphs synthesized" in (
        capsys.readouterr().out
    )


def test_phase_emit_clears_stale_modules(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _mul_spec()
    paths = run_paths(spec, "out/mul_t.csv")
    paths.run_dir.mkdir(parents=True)
    stale = paths.run_dir / "mul__v99_t99.py"
    stale.write_bytes(b"stale\n")
    _patch_emit_pipeline(monkeypatch)

    cli.trace_kernel(
        spec,
        {"m": 128, "n": 128},
        warmup=1,
        bench=1,
        out="out/mul_t.csv",
        phase="emit",
    )

    assert not stale.exists()
    assert sorted(path.name for path in paths.run_dir.glob("*.py")) == [
        "mul__v0_t0.py",
        "mul__v0_t1.py",
    ]
