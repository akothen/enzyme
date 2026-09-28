"""Unit tests for the flat `axon kernels/foo --sizes ...` interface."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from axon import cli
from axon.paths import out_dir, run_paths

BENCHMARK_NAME = "none"
DIMS = ("m", "k", "n")


@pytest.fixture(autouse=True)
def _stub_spec(monkeypatch):
    monkeypatch.setattr(
        cli, "_load_spec", lambda path: SimpleNamespace(name=path, dim_vars=DIMS)
    )


def _run(monkeypatch, argv: list[str]) -> dict[str, int]:
    captured: dict[str, int] = {}
    monkeypatch.setattr(
        cli, "trace_kernel", lambda spec, dim_sizes, **kw: captured.update(dim_sizes)
    )
    with pytest.raises(SystemExit) as exc:
        cli.main([BENCHMARK_NAME, *argv])
    assert exc.value.code == 0
    return captured


@pytest.mark.parametrize(
    ("sizes", "expected"),
    [
        (["1024", "16384", "512"], {"m": 1024, "k": 16384, "n": 512}),
        # a second valid shape: positional binding to dim_vars order
        (["4096", "4096", "4096"], {"m": 4096, "k": 4096, "n": 4096}),
    ],
    ids=["canonical-order", "cube"],
)
def test_sizes_accepted(monkeypatch, sizes, expected):
    assert _run(monkeypatch, ["--sizes", *sizes]) == expected


@pytest.mark.parametrize(
    "sizes",
    [
        ["1024", "16384"],  # too few sizes (wrong count -> exit 2)
        ["1024", "16384", "512", "7"],  # too many sizes (wrong count -> exit 2)
        ["1024", "abc", "512"],  # non-int value (argparse type=int rejects)
        ["m=1024", "k=16384", "n=512"],  # NAME=VALUE tokens (argparse type=int rejects)
    ],
    ids=["too-few", "too-many", "non-int", "name-value"],
)
def test_sizes_rejected(sizes):
    with pytest.raises(SystemExit) as exc:
        cli.main([BENCHMARK_NAME, "--sizes", *sizes])
    assert exc.value.code == 2


def _run_kwargs(monkeypatch, argv: list[str]) -> dict:
    seen: dict = {}
    monkeypatch.setattr(
        cli, "trace_kernel", lambda spec, dim_sizes, **kw: seen.update(kw)
    )
    with pytest.raises(SystemExit) as exc:
        cli.main([BENCHMARK_NAME, "--sizes", "1024", "1024", "512", *argv])
    assert exc.value.code == 0
    return seen


def test_synth_workers_default_is_none(monkeypatch):
    # None lets the synthesis pipeline use its automatic CPU-count policy.
    assert _run_kwargs(monkeypatch, [])["synth_workers"] is None


def test_synth_workers_explicit_value(monkeypatch):
    assert _run_kwargs(monkeypatch, ["--synth-workers", "8"])["synth_workers"] == 8


@pytest.mark.parametrize("value", ["0", "-4"], ids=["zero", "negative"])
def test_synth_workers_rejects_nonpositive(value):
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                BENCHMARK_NAME,
                "--sizes",
                "1024",
                "1024",
                "512",
                "--synth-workers",
                value,
            ]
        )
    assert exc.value.code == 2


def test_head_to_head_case_id_from_out_stem(monkeypatch):
    # --rtol/--atol mark a head-to-head run; its case id is the --out stem minus
    # the `<kernel>_` prefix (no separate --case-id flag).
    def _mk(*, m, n, dtype="float32", rng):  # signature read by dtype fallback
        return ()

    monkeypatch.setattr(
        cli,
        "_load_spec",
        lambda path: SimpleNamespace(name="mul", dim_vars=("m", "n"), make_inputs=_mk),
    )
    seen: dict = {}
    monkeypatch.setattr(
        cli, "trace_kernel", lambda spec, dim_sizes, **kw: seen.update(kw)
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "mul",
                "--sizes",
                "4096",
                "4096",
                "--rtol",
                "1e-3",
                "--atol",
                "1e-3",
                "--out",
                "out/mul_bw_4k.csv",
            ]
        )
    assert exc.value.code == 0
    assert seen["case_id"] == "bw_4k"
    # dtype falls back to the spec's make_inputs default for the per-case harness.
    assert seen["case_dtype"] == "float32"


def test_bench_only_run_has_no_case_id(monkeypatch):
    # Without --rtol/--atol the run is bench-only: no case id, no dtype fallback
    # (so a spec whose make_inputs cannot be introspected still runs).
    monkeypatch.setattr(
        cli, "_load_spec", lambda path: SimpleNamespace(name="mul", dim_vars=("m", "n"))
    )
    seen: dict = {}
    monkeypatch.setattr(
        cli, "trace_kernel", lambda spec, dim_sizes, **kw: seen.update(kw)
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(["mul", "--sizes", "1024", "1024", "--out", "out/mul_sq1k.csv"])
    assert exc.value.code == 0
    assert seen["case_id"] is None
    assert seen["case_dtype"] is None


def test_synthesis_controls_are_forwarded(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        cli, "trace_kernel", lambda spec, dim_sizes, **kw: seen.update(kw)
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                BENCHMARK_NAME,
                "--sizes",
                "1024",
                "1024",
                "1024",
                "--synthesis-timeout-seconds",
                "7200",
                "--synth-workers",
                "16",
            ]
        )
    assert exc.value.code == 0
    assert seen["synthesis_timeout_seconds"] == 7200.0
    assert seen["synth_workers"] == 16


def test_synthesis_control_defaults():
    args = cli._build_parser().parse_args(
        [BENCHMARK_NAME, "--sizes", "1024", "1024", "1024"]
    )
    assert args.synthesis_timeout_seconds == 3600.0
    assert args.synth_workers is None


def test_proof_workers_alias_is_removed():
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(
            [BENCHMARK_NAME, "--sizes", "1024", "1024", "1024", "--proof-workers", "9"]
        )


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--synthesis-timeout-seconds", "inf"),
        ("--synth-workers", "-1"),
    ],
)
def test_synthesis_controls_reject_nonpositive_or_nonfinite_values(flag, value):
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                BENCHMARK_NAME,
                "--sizes",
                "1024",
                "1024",
                "1024",
                flag,
                value,
            ]
        )
    assert exc.value.code == 2


def test_run_paths_with_out():
    spec = SimpleNamespace(name="mul")
    p = run_paths(spec, "out/mul_bw_4k.csv")
    # The CSV path is resolved absolute so spawn workers (own cwd) agree on it.
    base = Path("out/mul_bw_4k.csv").resolve()
    assert p.csv == base
    assert p.run_dir == base.with_suffix("")
    assert p.variant_module("fn", 0, 2) == base.with_suffix("") / "fn__v0_t2.py"
    assert (
        p.neff(0, 2, (1, 4))
        == base.with_suffix("") / "__v0_t2_tiles_1_4" / "kernel.neff"
    )
    assert p.baseline_dir == base.parent / "baseline_mul_bw_4k"
    assert p.baseline_csv == base.parent / "baseline_mul_bw_4k.csv"


def test_run_paths_default_stem():
    spec = SimpleNamespace(name="mul")
    p = run_paths(spec, None)
    assert p.csv == (out_dir() / "nki_mul.csv").resolve()
    assert p.run_dir == (out_dir() / "nki_mul").resolve()


def test_run_paths_bare_stem_equals_csv_form():
    # `--out out/foo` is the run stem: same paths as `--out out/foo.csv`.
    spec = SimpleNamespace(name="mul")
    assert run_paths(spec, "out/foo") == run_paths(spec, "out/foo.csv")


def test_run_paths_trailing_slash_is_container_dir():
    # `--out results/` names a container: the default CSV lands inside it.
    spec = SimpleNamespace(name="mul")
    p = run_paths(spec, "results/")
    assert p.csv.name == "nki_mul.csv"
    assert p.csv.parent.name == "results"
