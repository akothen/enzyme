#!/usr/bin/env python3
"""Run filtered Axon synthesis followed by agentic winner post-processing."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCHEMA_VERSION = 2  # 2 adds the phase-1/phase-2 wall-time CSV
HERE = Path(__file__).resolve().parent
AXON = HERE.parent
DEFAULT_RESULTS = HERE / "results"
COMBINED_TIMING_FIELDS = ("phase", "step", "iter", "seconds")


def _load_run_case():
    path = AXON / "tools" / "run_case.py"
    spec = importlib.util.spec_from_file_location("_axon_run_case", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_run_case_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(AXON / "tools" / "run_case.py"),
        args.target,
        "--emit-on",
        args.emit_on,
    ]
    if args.remote_config:
        command += ["-c", args.remote_config]
    if not args.candidate_filter:
        command += ["--no-candidate-filter"]
    command += ["--candidate-budget", str(args.candidate_budget)]
    if getattr(args, "phase1_timings", None):
        command += ["--timings", str(args.phase1_timings)]
    return command


def build_agent_command(args: argparse.Namespace, case_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(HERE / "optimize.py"),
        "--case",
        str(case_path),
        "--iters",
        str(args.iters),
        "--num-runs",
        str(args.num_runs),
        "--nkilib",
        str(Path(args.nkilib).expanduser()),
        "--target-host",
        args.target_host,
        "--test-adapter",
        args.test_adapter,
        "--out-root",
        str(Path(args.out_root).expanduser()),
        "--hint",
        args.hint,
        "--backend",
        args.backend,
        "--model",
        args.model,
        "--region",
        args.region,
    ]
    return command


def _row_count(path: Path) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    correct = sum(
        bool(row.get("median_ms")) and row.get("correct", "").lower() == "true"
        for row in rows
    )
    return len(rows), correct


def emitted_module_count(run_dir: Path, fn_name: str) -> tuple[int, str]:
    """How many `(hw variant, schedule family)` modules the run emitted, and where
    the number came from.

    `--emit-on box` emits on the remote box and pulls back only `results.csv`, so
    the local run dir holds no modules. Counting the distinct
    `(hw_variant, tile_variant)` pairs in the CSV gives the same number without
    them, and it keeps the module count consistent with the row counts beside it.
    """
    modules = list(run_dir.glob(f"{fn_name}__v*_t*.py"))
    if modules:
        return len(modules), "emitted_modules"
    results = run_dir / "results.csv"
    if not results.is_file():
        return 0, "unavailable"
    with results.open(newline="") as handle:
        pairs = {
            (row.get("hw_variant"), row.get("tile_variant"))
            for row in csv.DictReader(handle)
        }
    return len(pairs), "results_csv"


_SPEC_PROBE = """
import json, sys
from axon.cli import _load_spec
from axon.codegen import nki_safe_var

spec = _load_spec(sys.argv[1])
print(json.dumps({
    # The emitted modules are named after the NKI-safe function name, not the
    # raw spec name, so the caller's glob must use that form.
    "name": nki_safe_var(spec.name),
    "tile_options": len(spec.tile_options),
    "tile_args": len(spec.tile_args),
}))
"""


def spec_probe_command(kernel: str) -> list[str]:
    """`uv run` argv that prints one kernel spec's tile-space size as JSON.

    This driver runs under whatever interpreter invoked it, and importing `axon`
    needs the uv-managed 3.11 environment, so the spec is read out of process
    exactly like the `uv run axon` steps in `tools/run_case.py`."""
    return ["uv", "run", "python", "-c", _SPEC_PROBE, str(AXON / "kernels" / kernel)]


def parse_spec_facts(stdout: str) -> dict:
    """Take the probe's JSON line. `uv` may print progress first, so read the
    last non-empty line rather than the whole stream."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("spec probe printed nothing")
    return json.loads(lines[-1])


def read_spec_facts(kernel: str) -> dict:
    completed = subprocess.run(
        spec_probe_command(kernel), cwd=AXON, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"could not read the spec for {kernel}: {completed.stderr.strip()}"
        )
    return parse_spec_facts(completed.stdout)


def axon_candidate_summary(target: str) -> dict:
    run_case = _load_run_case()
    kernel, _case_id, _cfg = run_case.resolve_case(target, run_case.load_sizes())
    spec = read_spec_facts(kernel)
    run_dir = AXON / "out" / target
    module_count, module_count_source = emitted_module_count(run_dir, spec["name"])
    full_leaf_count = module_count * (spec["tile_options"] ** spec["tile_args"])
    selected, correct = _row_count(run_dir / "results.csv")
    return {
        "emitted_modules": module_count,
        "emitted_modules_source": module_count_source,
        "source_leaf_candidates": full_leaf_count,
        "selected_leaf_candidates": selected,
        "correct_leaf_candidates": correct,
    }


def read_phase1_timings(path: Path) -> list[dict]:
    """`run_case.py --timings` rows, relabeled for the combined CSV. `emit` is
    synthesis + proving + emission; `bench` is compile + benchmark on the box."""
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "phase": "axon",
            "step": row["step"],
            "iter": "",
            "seconds": float(row["seconds"]),
        }
        for row in rows
        if row.get("seconds")
    ]


def read_phase2_timings(result_path: Path, result: dict) -> list[dict]:
    """The post-process wall times. Prefer the CSV `optimize.py` writes beside
    its results.json; fall back to the per-iteration seconds in the JSON, which
    a results schema older than 2 does not carry (rows are then empty)."""
    csv_path = result_path.with_name("phase2_timings.csv")
    if csv_path.is_file():
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return [
            {
                "phase": "postprocess",
                "step": row["step"],
                "iter": row.get("iter", ""),
                "seconds": float(row["seconds"]),
            }
            for row in rows
            if row.get("seconds")
        ]
    out = []
    for rec in result.get("iterations", []):
        for step in ("generate", "measure"):
            seconds = rec.get(f"{step}_seconds")
            if seconds is not None:
                out.append(
                    {
                        "phase": "postprocess",
                        "step": step,
                        "iter": rec.get("iter", ""),
                        "seconds": float(seconds),
                    }
                )
    return out


def write_combined_timings(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMBINED_TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def phase_totals(rows: list[dict]) -> dict:
    """Wall seconds per phase, plus the emit/device split inside phase 1. The
    driver-measured `wall` rows are excluded, because they contain the steps."""

    def total(phase: str, steps: tuple[str, ...] | None = None) -> float:
        return round(
            sum(
                row["seconds"]
                for row in rows
                if row["phase"] == phase
                and row["step"] != "wall"
                and (steps is None or row["step"] in steps)
            ),
            3,
        )

    return {
        "axon_seconds": total("axon"),
        "axon_emit_seconds": total("axon", ("emit",)),
        "axon_device_seconds": total("axon", ("bench", "bench_all")),
        "postprocess_seconds": total("postprocess"),
        "postprocess_generate_seconds": total(
            "postprocess", ("generate", "hint_generate")
        ),
        "postprocess_measure_seconds": total(
            "postprocess", ("measure", "baseline_measure", "nkilib_measure")
        ),
    }


def _new_agent_result(root: Path, target: str, prior: set[Path]) -> Path:
    candidates = {
        path.resolve()
        for path in (root / target).glob("*/results.json")
        if path.is_file()
    }
    new = sorted(candidates - prior, key=lambda path: path.stat().st_mtime_ns)
    if not new:
        raise RuntimeError(
            f"post-processor produced no new results under {root / target}"
        )
    return new[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="<kernel>_<case_id> from eval/sizes.json")
    parser.add_argument("--nkilib", default=os.environ.get("NKILIB", ""))
    parser.add_argument(
        "--target-host", default=os.environ.get("AXON_TARGET_HOST", "trn2")
    )
    parser.add_argument("--remote-config")
    parser.add_argument("--emit-on", choices=("host", "box"), default="host")
    parser.add_argument(
        "--candidate-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--candidate-budget", type=int, default=8)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--num-runs", type=int, default=8)
    parser.add_argument("--hint", choices=("auto", "none"), default="auto")
    parser.add_argument("--backend", default="bedrock")
    parser.add_argument(
        "--model", default=os.environ.get("AGENT_MODEL", "claude-opus-4-8")
    )
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    parser.add_argument(
        "--test-adapter",
        default="test/integration/nkilib/utils/test_axon_emitted.py",
    )
    parser.add_argument("--out-root", default=str(DEFAULT_RESULTS))
    args = parser.parse_args()

    if not args.nkilib:
        raise SystemExit("need --nkilib (or $NKILIB)")
    run_case = _load_run_case()
    kernel, case_id, cfg = run_case.resolve_case(args.target, run_case.load_sizes())
    if "rtol" not in cfg or "atol" not in cfg:
        raise SystemExit(f"{args.target} is not a head-to-head Axon case")
    if args.target != f"{kernel}_{case_id}":
        raise SystemExit(f"non-canonical target name: {args.target}")

    result_root = Path(args.out_root).expanduser().resolve()
    prior = {
        path.resolve() for path in (result_root / args.target).glob("*/results.json")
    }
    args.phase1_timings = AXON / "out" / f"{args.target}_phase1_timings.csv"
    axon_command = build_run_case_command(args)
    print("->", " ".join(axon_command), file=sys.stderr)
    axon_started = time.perf_counter()
    completed = subprocess.run(axon_command, cwd=AXON)
    axon_wall_s = round(time.perf_counter() - axon_started, 3)
    if completed.returncode != 0:
        return completed.returncode

    case_path = AXON / "out" / "winners" / f"{args.target}_case.py"
    if not case_path.is_file():
        raise SystemExit(f"Axon run produced no pulled winner case: {case_path}")

    agent_command = build_agent_command(args, case_path)
    print("->", " ".join(agent_command), file=sys.stderr)
    agent_started = time.perf_counter()
    completed = subprocess.run(agent_command, cwd=AXON)
    agent_wall_s = round(time.perf_counter() - agent_started, 3)
    if completed.returncode != 0:
        return completed.returncode

    agent_result_path = _new_agent_result(result_root, args.target, prior)
    agent_result = json.loads(agent_result_path.read_text())
    timing_rows = [
        *read_phase1_timings(Path(args.phase1_timings)),
        {"phase": "axon", "step": "wall", "iter": "", "seconds": axon_wall_s},
        *read_phase2_timings(agent_result_path, agent_result),
        {"phase": "postprocess", "step": "wall", "iter": "", "seconds": agent_wall_s},
    ]
    timings_path = agent_result_path.with_name("phase_timings.csv")
    write_combined_timings(timings_path, timing_rows)
    combined = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "target": args.target,
        "candidate_filter": {
            "enabled": args.candidate_filter,
            "budget_per_schedule_family": args.candidate_budget,
        },
        "axon": {
            **axon_candidate_summary(args.target),
            "winner_case_path": str(case_path.resolve()),
        },
        "timings": {
            **phase_totals(timing_rows),
            "axon_wall_seconds": axon_wall_s,
            "postprocess_wall_seconds": agent_wall_s,
            "total_wall_seconds": round(axon_wall_s + agent_wall_s, 3),
            "csv_path": str(timings_path),
        },
        "postprocess": {
            "result_path": str(agent_result_path),
            "result": agent_result,
        },
        "best_kernel_path": agent_result["best_kernel_path"],
    }
    combined_path = agent_result_path.with_name("combined_result.json")
    combined_path.write_text(json.dumps(combined, indent=2) + "\n")
    print(combined_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
