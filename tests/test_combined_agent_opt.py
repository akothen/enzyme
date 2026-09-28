"""Host-only tests for the combined Axon + agent workflow driver."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "agent_opt" / "run_combined.py"


def _load():
    spec = importlib.util.spec_from_file_location("_combined_agent_opt", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "target": "cumsum_fast_small",
        "emit_on": "host",
        "remote_config": "trn2",
        "candidate_filter": True,
        "candidate_budget": 8,
        "iters": 3,
        "num_runs": 8,
        "nkilib": "/nkilib",
        "target_host": "trn2",
        "test_adapter": "test_adapter.py",
        "out_root": "/results",
        "hint": "auto",
        "backend": "bedrock",
        "model": "model",
        "region": "us-west-2",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_case_command_threads_filter_and_remote_config():
    combined = _load()
    command = combined.build_run_case_command(_args(candidate_filter=False))
    assert command[0] == sys.executable
    assert "--no-candidate-filter" in command
    assert command[command.index("--candidate-budget") + 1] == "8"
    assert command[command.index("-c") : command.index("-c") + 2] == ["-c", "trn2"]


def test_agent_command_consumes_generated_case(tmp_path):
    combined = _load()
    case = tmp_path / "winner_case.py"
    command = combined.build_agent_command(_args(), case)
    assert command[command.index("--case") + 1] == str(case)
    assert command[command.index("--iters") + 1] == "3"
    assert command[command.index("--target-host") + 1] == "trn2"


def test_run_case_command_carries_phase1_timing_path(tmp_path):
    combined = _load()
    path = tmp_path / "phase1.csv"
    command = combined.build_run_case_command(_args(phase1_timings=path))
    assert command[command.index("--timings") + 1] == str(path)


def test_run_case_command_omits_timings_when_unset():
    combined = _load()
    assert "--timings" not in combined.build_run_case_command(_args())


def test_spec_probe_runs_through_uv_with_the_kernel_path():
    combined = _load()
    command = combined.spec_probe_command("rmsnorm")
    assert command[:4] == ["uv", "run", "python", "-c"]
    assert command[-1].endswith("kernels/rmsnorm")


def test_spec_probe_asks_for_the_nki_safe_name():
    combined = _load()
    # The emitted modules are named `<nki_safe_var(spec.name)>__v<i>_t<j>.py`, so
    # a probe that returned the raw spec name would make the glob miss.
    probe = combined.spec_probe_command("rmsnorm")[4]
    assert "nki_safe_var(spec.name)" in probe


def test_parse_spec_facts_takes_the_last_json_line():
    combined = _load()
    stdout = (
        'Resolved 1 package\n\n{"name": "rmsnorm", "tile_options": 6, "tile_args": 2}\n'
    )
    assert combined.parse_spec_facts(stdout) == {
        "name": "rmsnorm",
        "tile_options": 6,
        "tile_args": 2,
    }


def test_parse_spec_facts_empty_output_raises():
    combined = _load()
    with pytest.raises(RuntimeError):
        combined.parse_spec_facts("\n \n")


def test_emitted_module_count_prefers_local_modules(tmp_path):
    combined = _load()
    (tmp_path / "rmsnorm__v0_t0.py").write_text("# module\n")
    (tmp_path / "rmsnorm__v1_t0.py").write_text("# module\n")
    assert combined.emitted_module_count(tmp_path, "rmsnorm") == (2, "emitted_modules")


def test_emitted_module_count_falls_back_to_the_csv(tmp_path):
    """`--emit-on box` emits remotely and pulls back only results.csv, so the
    module count has to come from the distinct variant pairs in that CSV."""
    combined = _load()
    path = tmp_path / "results.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["hw_variant", "tile_variant"])
        writer.writeheader()
        writer.writerows(
            [
                {"hw_variant": "0", "tile_variant": "0"},
                {"hw_variant": "0", "tile_variant": "0"},
                {"hw_variant": "0", "tile_variant": "1"},
                {"hw_variant": "1", "tile_variant": "0"},
            ]
        )
    assert combined.emitted_module_count(tmp_path, "rmsnorm") == (3, "results_csv")


def test_emitted_module_count_without_modules_or_csv(tmp_path):
    combined = _load()
    assert combined.emitted_module_count(tmp_path, "rmsnorm") == (0, "unavailable")


def test_phase1_timings_are_relabeled(tmp_path):
    combined = _load()
    path = tmp_path / "phase1.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["step", "seconds", "returncode", "command"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {"step": "emit", "seconds": "12.5", "returncode": "0", "command": "a"},
                {"step": "bench", "seconds": "40.0", "returncode": "0", "command": "b"},
            ]
        )
    rows = combined.read_phase1_timings(path)
    assert [r["step"] for r in rows] == ["emit", "bench"]
    assert all(r["phase"] == "axon" for r in rows)
    assert rows[0]["seconds"] == 12.5


def test_phase1_timings_missing_file_is_empty(tmp_path):
    combined = _load()
    assert combined.read_phase1_timings(tmp_path / "absent.csv") == []


def test_phase2_timings_prefer_the_csv(tmp_path):
    combined = _load()
    result_path = tmp_path / "results.json"
    csv_path = tmp_path / "phase2_timings.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "iter", "seconds"])
        writer.writeheader()
        writer.writerows([{"step": "generate", "iter": "1", "seconds": "9.0"}])
    rows = combined.read_phase2_timings(result_path, {"iterations": []})
    assert rows == [
        {"phase": "postprocess", "step": "generate", "iter": "1", "seconds": 9.0}
    ]


def test_phase2_timings_fall_back_to_iteration_records(tmp_path):
    combined = _load()
    result = {
        "iterations": [
            {"iter": 0, "generate_seconds": None, "measure_seconds": 5.0},
            {"iter": 1, "generate_seconds": 2.0, "measure_seconds": 6.0},
        ]
    }
    rows = combined.read_phase2_timings(tmp_path / "results.json", result)
    assert [(r["step"], r["iter"], r["seconds"]) for r in rows] == [
        ("measure", 0, 5.0),
        ("generate", 1, 2.0),
        ("measure", 1, 6.0),
    ]


def test_phase_totals_split_emit_device_and_exclude_wall():
    combined = _load()
    rows = [
        {"phase": "axon", "step": "emit", "iter": "", "seconds": 10.0},
        {"phase": "axon", "step": "sync", "iter": "", "seconds": 1.0},
        {"phase": "axon", "step": "bench", "iter": "", "seconds": 30.0},
        {"phase": "axon", "step": "wall", "iter": "", "seconds": 42.0},
        {"phase": "postprocess", "step": "hint_generate", "iter": 0, "seconds": 4.0},
        {"phase": "postprocess", "step": "generate", "iter": 1, "seconds": 3.0},
        {"phase": "postprocess", "step": "measure", "iter": 1, "seconds": 20.0},
        {"phase": "postprocess", "step": "wall", "iter": "", "seconds": 28.0},
    ]
    totals = combined.phase_totals(rows)
    assert totals["axon_seconds"] == 41.0  # wall excluded, emit+sync+bench
    assert totals["axon_emit_seconds"] == 10.0
    assert totals["axon_device_seconds"] == 30.0
    assert totals["postprocess_generate_seconds"] == 7.0
    assert totals["postprocess_measure_seconds"] == 20.0


def test_row_count_counts_only_timed_correct(tmp_path):
    combined = _load()
    path = tmp_path / "results.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["median_ms", "correct", "error"])
        writer.writeheader()
        writer.writerows(
            [
                {"median_ms": "1.0", "correct": "True", "error": ""},
                {"median_ms": "2.0", "correct": "False", "error": ""},
                {"median_ms": "", "correct": "", "error": "compile"},
            ]
        )
    assert combined._row_count(path) == (3, 1)
