"""Unit tests for the pure logic in agent_opt/ (optimize, summarize, seed).

Covers the code that runs off-device with no creds: response parsing, model-id
resolution, retry/auth classification, the nkilib pass-count guard, the CSV
aggregation helpers, and the seeder's refuse-on-diff / no-op / force behavior.
The device- and creds-bound orchestration (call_agent, verify_and_profile,
run_nkilib's subprocess, run_neuron_profile) is exercised by live runs, not here.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

_AGENT_OPT = Path(__file__).resolve().parent.parent / "agent_opt"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_agentopt_{name}", _AGENT_OPT / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


optimize = _load("optimize")
summarize = _load("summarize")
seed = _load("seed")
install_adapter = _load("install_adapter")


# --------------------------------------------------------------------------
# optimize.extract_kernel — pull the kernel source out of the agent reply
# --------------------------------------------------------------------------


def test_extract_kernel_no_fence_returns_stripped_body():
    assert optimize.extract_kernel("  just some text  ") == "just some text\n"


def test_extract_kernel_single_python_fence():
    resp = "here you go:\n```python\ndef f():\n    return 1\n```\nthanks"
    assert optimize.extract_kernel(resp) == "def f():\n    return 1\n"


def test_extract_kernel_bare_fence_no_lang():
    assert optimize.extract_kernel("```\nx = 1\n```") == "x = 1\n"


def test_extract_kernel_prefers_block_with_code_over_prose():
    resp = (
        "```\njust a note, no code here at all really\n```\n"
        "```python\n@nki.jit\ndef k():\n    pass\n```"
    )
    assert optimize.extract_kernel(resp) == "@nki.jit\ndef k():\n    pass\n"


def test_extract_kernel_picks_longest_among_code_blocks():
    short = "def a():\n    pass"
    long = "def b():\n    x = 1\n    y = 2\n    return x + y"
    resp = f"```python\n{short}\n```\nand better:\n```python\n{long}\n```"
    assert optimize.extract_kernel(resp) == long + "\n"


# --------------------------------------------------------------------------
# optimize.resolve_model — Bedrock cross-region inference-profile prefixing
# --------------------------------------------------------------------------


def test_resolve_model_non_bedrock_unchanged():
    assert optimize.resolve_model("claude-opus-4-8", "api") == "claude-opus-4-8"


def test_resolve_model_bedrock_bare_gets_us_anthropic():
    assert (
        optimize.resolve_model("claude-opus-4-8", "bedrock")
        == "us.anthropic.claude-opus-4-8"
    )


def test_resolve_model_bedrock_anthropic_prefixed_gets_us():
    assert (
        optimize.resolve_model("anthropic.claude-x", "bedrock")
        == "us.anthropic.claude-x"
    )


@pytest.mark.parametrize(
    "m", ["us.anthropic.x", "global.anthropic.x", "eu.anthropic.x", "apac.x"]
)
def test_resolve_model_bedrock_already_regioned_unchanged(m):
    assert optimize.resolve_model(m, "bedrock") == m


# --------------------------------------------------------------------------
# optimize retry/auth classification
# --------------------------------------------------------------------------


class _Err(Exception):
    def __init__(self, msg="", status_code=None):
        super().__init__(msg)
        if status_code is not None:
            self.status_code = status_code


def test_auth_error_detected_by_message():
    e = _Err("The security token included in the request is expired")
    assert optimize._looks_like_auth_error(e)
    assert not optimize._is_transient(e)  # auth is never retried


def test_access_denied_is_auth_not_transient():
    e = _Err("AccessDenied: not authorized to perform bedrock:InvokeModel")
    assert optimize._looks_like_auth_error(e)
    assert not optimize._is_transient(e)


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_transient_by_status_code(code):
    e = _Err("boom", status_code=code)
    assert optimize._is_transient(e)


def test_transient_by_message_overloaded():
    assert optimize._is_transient(_Err("service is Overloaded, try again"))


def test_plain_client_error_not_transient_not_auth():
    e = _Err("validation error: bad request", status_code=400)
    assert not optimize._is_transient(e)
    assert not optimize._looks_like_auth_error(e)


def test_load_meta_missing_lists_available(tmp_path, monkeypatch):
    kernels = tmp_path / "kernels"
    (kernels / "known").mkdir(parents=True)
    monkeypatch.setattr(optimize, "KERNELS", kernels)
    with pytest.raises(SystemExit, match=r"available: known"):
        optimize.load_meta("missing")


def test_load_case_meta_reads_generated_contract(tmp_path):
    refs = tmp_path / "refs.py"
    refs.write_text("# refs\n")
    kernel = tmp_path / "winner.py"
    kernel.write_text("def k(): pass\n")
    case = tmp_path / "winner_case.py"
    case.write_text(
        "CASE = {"
        "'name': 'winner', 'entry_point': 'k', 'kernel_rel': 'winner.py', "
        "'refs_rel': 'refs.py', 'dtype': 'bfloat16', 'input_names': ['x'], "
        "'shape_args': {'m': 128}, 'rtol': 0.01, 'atol': 0.01}\n"
    )
    meta = optimize.load_case_meta(case)
    assert meta["_name"] == "winner"
    assert meta["_src"] == kernel.read_text()
    # load_case_meta resolves the path, so compare resolved: a pytest tmp dir
    # behind a symlink (macOS /var -> /private/var) would fail otherwise.
    assert meta["_refs_path"] == refs.resolve()


def test_load_case_meta_rejects_legacy_incomplete_contract(tmp_path):
    case = tmp_path / "old_case.py"
    case.write_text("CASE = {'name': 'old'}\n")
    with pytest.raises(SystemExit, match="regenerate it with current Axon"):
        optimize.load_case_meta(case)


# --------------------------------------------------------------------------
# optimize._passed_count — nkilib exactly-one-match guard input
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "out,expected",
    [
        ("===== 1 passed in 3.21s =====", 1),
        ("34 passed, 2 warnings in 40s", 34),
        ("no tests ran in 0.01s", 0),
        ("2 failed, 0 passed in 1s", 0),
        ("0 passed, 41 deselected in 0.6s", 0),
        ("", 0),
    ],
)
def test_passed_count(out, expected):
    assert optimize._passed_count(out) == expected


def test_verify_and_profile_requires_exactly_one_pass():
    """A deselect-everything run exits 0 and prints " passed", so the verdict must
    come from the count, not from the substring."""
    source = inspect.getsource(optimize.verify_and_profile)
    assert "_passed_count(out) == 1" in source
    assert '" passed" in out' not in source


def test_newest_qor_ignores_stale_files(tmp_path):
    out = tmp_path / "neuron_test_output"
    out.mkdir()
    stale = out / "qor_data_stale.csv"
    stale.write_text("InferenceTime\n0.001\n")
    cutoff = stale.stat().st_mtime_ns
    assert optimize._newest_qor_us(tmp_path, newer_than_ns=cutoff) is None


def test_run_nkilib_without_selector_skips(tmp_path, capsys):
    meta = {"nkilib": {"test": "test.py"}}
    assert optimize.run_nkilib(meta, tmp_path, "trn2") is None
    assert "has no selector" in capsys.readouterr().out


def test_install_adapter_refuses_different_existing_file(tmp_path):
    destination = tmp_path / install_adapter.DESTINATION
    destination.parent.mkdir(parents=True)
    destination.write_text("different\n")
    with pytest.raises(SystemExit, match="exists and differs"):
        install_adapter.install(tmp_path)
    assert destination.read_text() == "different\n"


def test_install_adapter_install_noop_and_force(tmp_path):
    assert install_adapter.install(tmp_path) == "installed"
    destination = tmp_path / install_adapter.DESTINATION
    assert destination.read_bytes() == install_adapter.SOURCE.read_bytes()
    assert install_adapter.install(tmp_path) == "noop"
    destination.write_text("different\n")
    assert install_adapter.install(tmp_path, force=True) == "installed"
    assert destination.read_bytes() == install_adapter.SOURCE.read_bytes()


# --------------------------------------------------------------------------
# summarize CSV helpers
# --------------------------------------------------------------------------


def test_fmt_sizes_none_and_empty():
    assert summarize._fmt_sizes(None) == ""
    assert summarize._fmt_sizes({}) == ""


def test_fmt_sizes_drops_tile_args():
    sizes = {"m": 1024, "n": 8192, "TILES_IN_BLOCK_M": 1, "TILES_IN_BLOCK_N": 8}
    assert summarize._fmt_sizes(sizes) == "[1024,8192]"


def test_ratio_normal_and_guards():
    assert summarize._ratio(2.0, 4.0) == 0.5
    assert summarize._ratio(None, 4.0) == ""
    assert summarize._ratio(2.0, None) == ""
    assert summarize._ratio(2.0, 0) == ""


def test_row_full_record():
    d = {
        "kernel": "fused_adam_opt_1m",
        "shape_args": {"p": 128, "f": 8192, "TILES_IN_BLOCK_M": 1},
        "baseline_active_us": 100.0,
        "best_active_us": 80.0,
        "nkilib_inference_us": 160.0,
    }
    row = summarize._row(d)
    assert row["kernel"] == "fused_adam_opt_1m"
    assert row["input_sizes"] == "[128,8192]"
    assert row["axon_us"] == 100.0
    assert row["best_agent_us"] == 80.0
    assert row["nkilib_us"] == 160.0
    assert row["axon_over_nkilib"] == 0.625
    assert row["agent_over_axon"] == 0.8
    assert row["agent_over_nkilib"] == 0.5
    assert set(row) == set(summarize.COLUMNS)


def test_row_carries_timings_when_present():
    d = {
        "kernel": "k",
        "baseline_active_us": 10.0,
        "best_active_us": 9.0,
        "timings": {
            "total_seconds": 300.0,
            "generate_seconds": 40.0,
            "measure_seconds": 250.0,
        },
    }
    row = summarize._row(d)
    assert row["postprocess_total_s"] == 300.0
    assert row["generate_s"] == 40.0
    assert row["measure_s"] == 250.0


def test_row_without_timings_leaves_timing_columns_blank():
    row = summarize._row({"kernel": "k", "baseline_active_us": 10.0})
    assert row["postprocess_total_s"] == ""
    assert row["generate_s"] == ""
    assert row["measure_s"] == ""


# --------------------------------------------------------------------------
# optimize phase-2 timing rows
# --------------------------------------------------------------------------


def test_phase2_timing_rows_cover_baseline_hint_and_iterations():
    records = [
        {
            "iter": 0,
            "role": "baseline",
            "generate_seconds": None,
            "measure_seconds": 5.0,
        },
        {"iter": 1, "role": "agent", "generate_seconds": 2.0, "measure_seconds": 6.0},
        {"iter": 2, "role": "agent", "generate_seconds": 3.0, "measure_seconds": 7.0},
    ]
    rows = optimize.phase2_timing_rows(5.0, 4.0, 1.5, records)
    assert [(r["step"], r["iter"], r["seconds"]) for r in rows] == [
        ("baseline_measure", 0, 5.0),
        ("nkilib_measure", 0, 4.0),
        ("hint_generate", 0, 1.5),
        ("generate", 1, 2.0),
        ("measure", 1, 6.0),
        ("generate", 2, 3.0),
        ("measure", 2, 7.0),
    ]
    assert all(set(r) == set(optimize.PHASE2_TIMING_FIELDS) for r in rows)


def test_phase2_timing_rows_drop_absent_nkilib_and_hint():
    rows = optimize.phase2_timing_rows(5.0, None, None, [])
    assert [r["step"] for r in rows] == ["baseline_measure"]


def test_row_no_nkilib_leaves_nkilib_columns_blank():
    d = {"kernel": "k", "baseline_active_us": 10.0, "best_active_us": 9.0}
    row = summarize._row(d)
    assert row["nkilib_us"] == ""
    assert row["axon_over_nkilib"] == ""
    assert row["agent_over_nkilib"] == ""
    assert row["agent_over_axon"] == 0.9


def _write_results(root: Path, kernel: str, ts: str, extra: dict | None = None):
    d = {"kernel": kernel, "timestamp": ts}
    d.update(extra or {})
    p = root / kernel / ts / "results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


def test_latest_per_kernel_picks_newest_timestamp(tmp_path):
    _write_results(tmp_path, "k1", "2024-01-01T00-00-00", {"best_active_us": 1.0})
    _write_results(tmp_path, "k1", "2024-06-01T00-00-00", {"best_active_us": 2.0})
    _write_results(tmp_path, "k2", "2024-03-01T00-00-00", {"best_active_us": 3.0})
    latest = summarize._latest_per_kernel(tmp_path)
    assert set(latest) == {"k1", "k2"}
    assert latest["k1"]["best_active_us"] == 2.0  # newer, not the smaller-us one
    assert latest["k2"]["best_active_us"] == 3.0


def test_latest_per_kernel_skips_corrupt_json(tmp_path):
    _write_results(tmp_path, "k1", "2024-01-01T00-00-00", {"best_active_us": 1.0})
    bad = tmp_path / "k1" / "2024-09-09T00-00-00" / "results.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not json")
    latest = summarize._latest_per_kernel(tmp_path)
    assert latest["k1"]["best_active_us"] == 1.0


# --------------------------------------------------------------------------
# seed.seed_kernel — refuse-on-diff / no-op / force / stamp
# --------------------------------------------------------------------------


def _kdir(tmp_path: Path, meta: dict | None = None) -> Path:
    d = tmp_path / "kern"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps(meta if meta is not None else {}))
    return d


def _winner(tmp_path: Path, body: str, name: str = "w.py") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def test_seed_fresh_writes_and_stamps(tmp_path):
    kdir = _kdir(tmp_path, {"entry_point": "k"})
    src = _winner(tmp_path, "def k():\n    return 1\n")
    assert seed.seed_kernel(kdir, src) == "seeded"
    assert (kdir / "input_kernel.py").read_text() == src.read_text()
    meta = json.loads((kdir / "meta.json").read_text())
    assert meta["entry_point"] == "k"  # preserved
    assert meta["input_sha"] == seed._sha(src.read_text())
    assert meta["provenance"]["source"] == str(src)
    assert meta["provenance"]["seeded_utc"].endswith("Z")


def test_seed_identical_is_noop(tmp_path):
    kdir = _kdir(tmp_path)
    src = _winner(tmp_path, "x = 1\n")
    seed.seed_kernel(kdir, src)
    before = (kdir / "meta.json").read_text()
    assert seed.seed_kernel(kdir, src) == "noop"
    assert (kdir / "meta.json").read_text() == before  # untouched


def test_seed_different_without_force_refuses(tmp_path):
    kdir = _kdir(tmp_path)
    seed.seed_kernel(kdir, _winner(tmp_path, "x = 1\n", "a.py"))
    src2 = _winner(tmp_path, "x = 2\n", "b.py")
    with pytest.raises(SystemExit):
        seed.seed_kernel(kdir, src2)
    assert (kdir / "input_kernel.py").read_text() == "x = 1\n"  # unchanged


def test_seed_different_with_force_overwrites(tmp_path):
    kdir = _kdir(tmp_path)
    seed.seed_kernel(kdir, _winner(tmp_path, "x = 1\n", "a.py"))
    src2 = _winner(tmp_path, "x = 2\n", "b.py")
    assert seed.seed_kernel(kdir, src2, force=True) == "seeded"
    assert (kdir / "input_kernel.py").read_text() == "x = 2\n"
    meta = json.loads((kdir / "meta.json").read_text())
    assert meta["input_sha"] == seed._sha("x = 2\n")


def test_seed_missing_kdir_raises(tmp_path):
    with pytest.raises(SystemExit):
        seed.seed_kernel(tmp_path / "nope", _winner(tmp_path, "x=1\n"))


def test_seed_missing_source_raises(tmp_path):
    kdir = _kdir(tmp_path)
    with pytest.raises(SystemExit):
        seed.seed_kernel(kdir, tmp_path / "missing.py")
