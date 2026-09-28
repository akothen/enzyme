"""Tests for the folded-in export tail (winner-pick + per-case harness
emission), device-free via a fabricated bench CSV + variant modules."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from axon.codegen import nki_safe_var
from axon.paths import run_paths
from axon.winner import export_winner, pick_winner

REPO = Path(__file__).resolve().parents[1]


def _load_mul_spec():
    # Load through the real CLI loader; the kernel is a package dir now, and
    # export_winner/case_gen take that directory as the kernel_path (case_gen
    # resolves refs.py inside it for the harness).
    from axon.cli import _load_spec

    p = REPO / "kernels" / "mul"
    return _load_spec(str(p)), p


_COLUMNS = [
    "m",
    "n",
    "hw_variant",
    "tile_variant",
    "TILES_M",
    "TILES_N",
    "mean_ms",
    "median_ms",
    "min_ms",
    "max_ms",
    "std_dev_ms",
    "correct",
    "max_abs_err",
    "error",
    "mode",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=_COLUMNS).to_csv(path, index=False)


def _timed_row(hw, tile, tm, tn, median):
    return {
        "m": 1024,
        "n": 1024,
        "hw_variant": hw,
        "tile_variant": tile,
        "TILES_M": tm,
        "TILES_N": tn,
        "mean_ms": median + 0.1,
        "median_ms": median,
        "min_ms": median - 0.1,
        "max_ms": median + 0.2,
        "std_dev_ms": 0.01,
        "correct": True,
        "max_abs_err": 0.0,
        "error": None,
        "mode": None,
    }


def _null_row(hw, tile, tm, tn):
    return {
        "m": 1024,
        "n": 1024,
        "hw_variant": hw,
        "tile_variant": tile,
        "TILES_M": tm,
        "TILES_N": tn,
        "mean_ms": None,
        "median_ms": None,
        "min_ms": None,
        "max_ms": None,
        "std_dev_ms": None,
        "correct": None,
        "max_abs_err": None,
        "error": "boom",
        "mode": None,
    }


def _make_variant_module(paths, fn_name, hw, tile) -> Path:
    p = paths.variant_module(fn_name, hw, tile)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"# fake variant v{hw}_t{tile}\ndef {fn_name}():\n    return 0\n")
    return p


def test_export_winner_full(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec, kpath = _load_mul_spec()
    fn_name = nki_safe_var(spec.name)
    out = "out/mul_bw_4k.csv"
    paths = run_paths(spec, out)

    _write_csv(
        paths.csv,
        [
            _timed_row(0, 1, 2, 2, median=5.0),
            _timed_row(1, 2, 4, 8, median=3.0),
        ],
    )
    _make_variant_module(paths, fn_name, 0, 1)
    win_src = _make_variant_module(paths, fn_name, 1, 2)

    winner = pick_winner(paths.csv, spec, "median", rtol=1e-2, atol=1e-2)
    assert winner is not None
    assert winner.hw_variant == 1 and winner.tile_variant == 2

    export_winner(
        spec,
        winner,
        case_id="bw_4k",
        kernel_path=kpath,
        paths=paths,
        dtype="bfloat16",
        rtol=1e-2,
        atol=1e-2,
    )

    win_kernel = tmp_path / "out" / "winners" / "mul_bw_4k.py"
    win_case = tmp_path / "out" / "winners" / "mul_bw_4k_case.py"
    manifest = tmp_path / "out" / "axon_kernel_specs.json"
    assert win_kernel.is_file()
    assert win_kernel.read_text() == win_src.read_text()
    assert win_case.is_file()
    # The aggregate manifest is gone; each case is self-describing.
    assert not manifest.exists()

    # Import the generated case module and inspect its CASE metadata + callables.
    cspec = importlib.util.spec_from_file_location("_mul_bw_4k_case", win_case)
    cmod = importlib.util.module_from_spec(cspec)
    cspec.loader.exec_module(cmod)

    case = cmod.CASE
    for key in (
        "name",
        "entry_point",
        "kernel_rel",
        "refs_rel",
        "dtype",
        "input_names",
        "shape_args",
        "rtol",
        "atol",
    ):
        assert key in case, key
    assert case["name"] == "mul_bw_4k"
    assert case["entry_point"] == fn_name
    assert case["kernel_rel"] == "mul_bw_4k.py"
    assert case["dtype"] == "bfloat16"
    assert case["input_names"] == ["x", "y"]
    assert (win_case.parent / case["refs_rel"]).is_file()
    assert case["rtol"] == 1e-2 and case["atol"] == 1e-2
    assert case["shape_args"]["m"] == 1024 and case["shape_args"]["n"] == 1024
    assert case["shape_args"]["TILES_IN_BLOCK_M"] == 4
    assert case["shape_args"]["TILES_IN_BLOCK_N"] == 8

    # The sibling kernel the case points at exists, and the three callables
    # nkilib consumes are present.
    assert (win_case.parent / case["kernel_rel"]).is_file()
    assert callable(cmod.torch_ref)
    assert callable(cmod.make_inputs)
    assert callable(cmod.output_tensors)


def test_pick_winner_empty_csv_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec, _ = _load_mul_spec()
    paths = run_paths(spec, "out/mul_bw_4k.csv")
    _write_csv(paths.csv, [_null_row(0, 1, 2, 2), _null_row(1, 2, 4, 8)])
    assert pick_winner(paths.csv, spec, "median", rtol=1e-2, atol=1e-2) is None


def test_export_winner_missing_variant_module_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec, kpath = _load_mul_spec()
    paths = run_paths(spec, "out/mul_bw_4k.csv")
    _write_csv(paths.csv, [_timed_row(1, 2, 4, 8, median=3.0)])
    # No variant module written, so export must fail loud before writing anything.

    winner = pick_winner(paths.csv, spec, "median", rtol=1e-2, atol=1e-2)
    with pytest.raises(FileNotFoundError, match="winning variant module not found"):
        export_winner(
            spec,
            winner,
            case_id="bw_4k",
            kernel_path=kpath,
            paths=paths,
            dtype="bfloat16",
            rtol=1e-2,
            atol=1e-2,
        )


def test_pick_winner_no_csv_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec, _ = _load_mul_spec()
    paths = run_paths(spec, "out/mul_bw_4k.csv")
    assert pick_winner(paths.csv, spec, "median", rtol=1e-2, atol=1e-2) is None


def test_pick_winner_verified_wrong_raises(tmp_path, monkeypatch):
    # Fastest variant is timed but verified incorrect -> fail loud.
    monkeypatch.chdir(tmp_path)
    spec, _ = _load_mul_spec()
    paths = run_paths(spec, "out/mul_bw_4k.csv")
    wrong = _timed_row(1, 2, 4, 8, median=3.0)
    wrong["correct"] = False
    wrong["max_abs_err"] = 9.9
    _write_csv(paths.csv, [wrong])
    with pytest.raises(AssertionError, match="failed on-device correctness"):
        pick_winner(paths.csv, spec, "median", rtol=1e-2, atol=1e-2)


def _wrong_row(hw, tile, tm, tn, median):
    row = _timed_row(hw, tile, tm, tn, median)
    row["correct"] = False
    row["max_abs_err"] = 9.9
    return row


def _unverified_row(hw, tile, tm, tn, median):
    row = _timed_row(hw, tile, tm, tn, median)
    row["correct"] = None
    row["max_abs_err"] = None
    return row


def test_pick_winner_wrong_loses_to_slower_correct(tmp_path, monkeypatch):
    # The fastest row is verified wrong, so it loses to the slower correct one
    # instead of aborting the case.
    monkeypatch.chdir(tmp_path)
    spec, _ = _load_mul_spec()
    paths = run_paths(spec, "out/mul_bw_4k.csv")
    _write_csv(
        paths.csv,
        [_wrong_row(1, 2, 4, 8, median=3.0), _timed_row(0, 1, 2, 2, median=5.0)],
    )
    winner = pick_winner(paths.csv, spec, "median", rtol=1e-2, atol=1e-2)
    assert winner is not None
    assert winner.hw_variant == 0 and winner.tile_variant == 1
    assert winner.median_ms == 5.0


def test_pick_winner_wrong_loses_to_slower_unverified(tmp_path, monkeypatch, capsys):
    # NaN correctness is kept, not dropped, so an unverified row still beats a
    # verified-wrong one and ships with the unverified warning.
    monkeypatch.chdir(tmp_path)
    spec, _ = _load_mul_spec()
    paths = run_paths(spec, "out/mul_bw_4k.csv")
    _write_csv(
        paths.csv,
        [_wrong_row(1, 2, 4, 8, median=3.0), _unverified_row(0, 1, 2, 2, median=5.0)],
    )
    winner = pick_winner(paths.csv, spec, "median", rtol=1e-2, atol=1e-2)
    assert winner is not None
    assert winner.hw_variant == 0 and winner.tile_variant == 1
    assert "correctness unverified" in capsys.readouterr().out


def test_pick_winner_skipped_count_excludes_slower_wrong_rows(
    tmp_path, monkeypatch, capsys
):
    # Only the wrong rows ranked ahead of the winner were skipped; the wrong row
    # behind it never competed and must not inflate the count.
    monkeypatch.chdir(tmp_path)
    spec, _ = _load_mul_spec()
    paths = run_paths(spec, "out/mul_bw_4k.csv")
    _write_csv(
        paths.csv,
        [
            _wrong_row(1, 2, 4, 8, median=3.0),
            _timed_row(0, 1, 2, 2, median=5.0),
            _wrong_row(2, 3, 4, 4, median=7.0),
        ],
    )
    winner = pick_winner(paths.csv, spec, "median", rtol=1e-2, atol=1e-2)
    assert winner is not None
    assert winner.hw_variant == 0 and winner.tile_variant == 1
    assert "skipped 1 faster but incorrect variant(s)" in capsys.readouterr().out


def test_pick_winner_fastest_correct_reports_no_skips(tmp_path, monkeypatch, capsys):
    # A wrong row behind the winner prints nothing at all.
    monkeypatch.chdir(tmp_path)
    spec, _ = _load_mul_spec()
    paths = run_paths(spec, "out/mul_bw_4k.csv")
    _write_csv(
        paths.csv,
        [_timed_row(0, 1, 2, 2, median=3.0), _wrong_row(1, 2, 4, 8, median=5.0)],
    )
    winner = pick_winner(paths.csv, spec, "median", rtol=1e-2, atol=1e-2)
    assert winner is not None
    assert winner.hw_variant == 0
    assert "skipped" not in capsys.readouterr().out


def test_pick_winner_all_wrong_still_raises(tmp_path, monkeypatch):
    # With no correct or unverified fallback, keep the loud failure.
    monkeypatch.chdir(tmp_path)
    spec, _ = _load_mul_spec()
    paths = run_paths(spec, "out/mul_bw_4k.csv")
    _write_csv(
        paths.csv,
        [_wrong_row(1, 2, 4, 8, median=3.0), _wrong_row(0, 1, 2, 2, median=5.0)],
    )
    with pytest.raises(AssertionError, match="failed on-device correctness") as exc:
        pick_winner(paths.csv, spec, "median", rtol=1e-2, atol=1e-2)
    # The message names the fastest wrong row, its err and the tolerances.
    msg = str(exc.value)
    assert "v1_t2" in msg and "9.9" in msg and "rtol=0.01" in msg


def test_pick_winner_unverified_beats_slower_correct(tmp_path, monkeypatch, capsys):
    # NaN is ranked, not penalized: the fastest row wins even unverified.
    monkeypatch.chdir(tmp_path)
    spec, _ = _load_mul_spec()
    paths = run_paths(spec, "out/mul_bw_4k.csv")
    _write_csv(
        paths.csv,
        [_unverified_row(1, 2, 4, 8, median=3.0), _timed_row(0, 1, 2, 2, median=5.0)],
    )
    winner = pick_winner(paths.csv, spec, "median", rtol=1e-2, atol=1e-2)
    assert winner is not None
    assert winner.hw_variant == 1 and winner.tile_variant == 2
    assert "correctness unverified" in capsys.readouterr().out


def test_pick_winner_unverified_warns_but_returns(tmp_path, monkeypatch, capsys):
    # Fastest variant is timed but correctness is unverified (NaN: no baseline
    # outputs) -> warn and ship the winner rather than silently passing.
    monkeypatch.chdir(tmp_path)
    spec, _ = _load_mul_spec()
    paths = run_paths(spec, "out/mul_bw_4k.csv")
    unchecked = _timed_row(1, 2, 4, 8, median=3.0)
    unchecked["correct"] = None
    unchecked["max_abs_err"] = None
    _write_csv(paths.csv, [unchecked])
    winner = pick_winner(paths.csv, spec, "median", rtol=1e-2, atol=1e-2)
    assert winner is not None and winner.hw_variant == 1
    assert "correctness unverified" in capsys.readouterr().out
