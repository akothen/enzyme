"""Host tests for tools/run_case.py — the split-run case driver.

`run_case.py <kernel>_<case_id>` orchestrates one declared case with the
emit phase on THIS host and the bench phase on the remote box:
  1. uv run axon <case args> --phase emit          (local)
  2. python3 tools/remote.py sync                  (ship emitted modules)
  3. remote run -- uv run axon <case args> --phase bench
  4. remote sync --pull
`--emit-on box` instead runs the whole case on the box (legacy loop).
The case args (sizes/dtype/rtol/atol/out) come from eval/sizes.json — the
same single source of truth the Makefile generator reads.
"""

import csv
import importlib.util
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load_run_case():
    spec = importlib.util.spec_from_file_location("run_case", _TOOLS / "run_case.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_case_longest_kernel_prefix_wins():
    rc = _load_run_case()
    sizes = {
        "matmul": {"sq1k": {"sizes": [1024, 1024, 1024]}},
        "matmul_red_div": {"sq1k": {"sizes": [1024, 1024, 1024]}},
    }
    kernel, case_id, cfg = rc.resolve_case("matmul_red_div_sq1k", sizes)
    assert kernel == "matmul_red_div" and case_id == "sq1k"
    kernel, case_id, cfg = rc.resolve_case("matmul_sq1k", sizes)
    assert kernel == "matmul" and case_id == "sq1k"


def test_resolve_case_unknown_target_raises():
    rc = _load_run_case()
    with pytest.raises(SystemExit):
        rc.resolve_case("nope_xyz", {"matmul": {"sq1k": {"sizes": [1]}}})


def test_axon_args_carry_case_config():
    rc = _load_run_case()
    cfg = {"sizes": [128, 8192], "dtype": "float32", "rtol": 1e-4, "atol": 1e-4}
    args = rc.axon_args("fused_adam", "opt_1m_fp32", cfg)
    joined = " ".join(args)
    assert "kernels/fused_adam" in joined
    assert "--sizes 128 8192" in joined
    assert "--dtype float32" in joined
    assert "--rtol 0.0001" in joined and "--atol 0.0001" in joined
    assert "--out out/fused_adam_opt_1m_fp32.csv" in joined


def test_host_mode_command_sequence():
    rc = _load_run_case()
    cfg = {"sizes": [1024, 1024, 1024]}
    cmds = rc.plan_commands("matmul", "sq1k", cfg, emit_on="host", config=None)
    # exactly: local emit, sync, remote bench, pull
    assert len(cmds) == 4
    assert cmds[0][:3] == ["uv", "run", "axon"] and "--phase" in cmds[0]
    assert cmds[0][cmds[0].index("--phase") + 1] == "emit"
    assert cmds[1][-1] == "sync"
    assert "--phase" in cmds[2] and cmds[2][cmds[2].index("--phase") + 1] == "bench"
    assert "remote.py" in " ".join(cmds[2])
    assert cmds[3][-2:] == ["sync", "--pull"]


def test_box_mode_is_single_remote_all_run():
    rc = _load_run_case()
    cfg = {"sizes": [1024, 1024, 1024]}
    cmds = rc.plan_commands("matmul", "sq1k", cfg, emit_on="box", config=None)
    assert len(cmds) == 3  # sync, one remote all-phase run, pull
    assert cmds[0][-1] == "sync"
    joined = " ".join(cmds[1])
    assert "remote.py" in joined and "--phase" not in joined
    assert cmds[2][-2:] == ["sync", "--pull"]


def test_lnc2_case_forces_box_mode():
    rc = _load_run_case()
    cfg = {"sizes": [1024, 1024, 16384], "lnc": 2}
    cmds = rc.plan_commands("matmul", "ck_16k_l2", cfg, emit_on="host", config=None)
    # emit can't split for lnc=2 -> falls back to box-side all with a warning
    joined = " ".join(cmds[-1])
    assert "sync --pull" in joined
    remote_run = " ".join(cmds[-2])
    assert "--lnc 2" in remote_run and "--phase" not in remote_run


def test_candidate_filter_can_be_disabled():
    rc = _load_run_case()
    cfg = {"sizes": [1024, 1024]}
    args = rc.axon_args("mul", "sq1k", cfg, candidate_filter=False)
    assert "--no-candidate-filter" in args


def test_candidate_budget_is_forwarded():
    rc = _load_run_case()
    args = rc.axon_args("mul", "sq1k", {"sizes": [1024, 1024]}, candidate_budget=4)
    assert args[args.index("--candidate-budget") + 1] == "4"


def test_step_labels_name_each_planned_command():
    rc = _load_run_case()
    cfg = {"sizes": [1024, 1024, 1024]}
    host = rc.plan_commands("matmul", "sq1k", cfg, emit_on="host", config=None)
    assert [rc.step_label(cmd) for cmd in host] == ["emit", "sync", "bench", "pull"]
    box = rc.plan_commands("matmul", "sq1k", cfg, emit_on="box", config=None)
    assert [rc.step_label(cmd) for cmd in box] == ["sync", "bench_all", "pull"]


def test_write_timings_round_trips(tmp_path):
    rc = _load_run_case()
    path = tmp_path / "nested" / "phase1.csv"
    rows = [{"step": "emit", "seconds": 1.5, "returncode": 0, "command": "uv run axon"}]
    rc.write_timings(path, rows)
    with path.open(newline="") as handle:
        read = list(csv.DictReader(handle))
    assert read == [
        {"step": "emit", "seconds": "1.5", "returncode": "0", "command": "uv run axon"}
    ]
