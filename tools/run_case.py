#!/usr/bin/env python3
"""Run one declared case with the emit phase on THIS host and the bench
phase on the remote Trainium box — the default, because emission (synthesis
+ Z3 proving) is pure CPU work that shouldn't occupy a device host.

    python3 tools/run_case.py matmul_sq1k
    python3 tools/run_case.py fused_adam_opt_1m --emit-on box   # legacy: all on box
    python3 tools/run_case.py mul_bw_1k -c trn2                 # pick a remote config

Sequence (default `--emit-on host`):
  1. uv run axon kernels/<kernel> --sizes ... --out out/<stem>.csv --phase emit   (local)
  2. python3 tools/remote.py sync                       (ship the emitted modules)
  3. remote run -- uv run axon <same args> --phase bench                    (box)
  4. python3 tools/remote.py sync --pull                (retrieve winner/results)

`--emit-on box` skips the local phase and runs the single-shot legacy loop on
the box (one remote `--phase all` run after a sync). Cases declaring `lnc: 2`
in eval/sizes.json always take the box path (the SPMD sweep cannot split).

`--timings out/<stem>_phase1.csv` records the wall time of each step, so a
combined run can report emit (synthesis) time separately from device time.

Case config (sizes/dtype/rtol/atol/lnc) comes from eval/sizes.json — the same
single source of truth the Makefile generator reads — so this driver and
`make <stem>` agree on every argument. Local emit needs a one-time `uv sync`
in this repo (the emit phase imports only the host-side deps).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SIZES_JSON = REPO / "eval" / "sizes.json"
TIMING_FIELDS = ("step", "seconds", "returncode", "command")


def load_sizes() -> dict:
    return json.loads(SIZES_JSON.read_text())


def resolve_case(target: str, sizes: dict) -> tuple[str, str, dict]:
    """Split `<kernel>_<case_id>` against sizes.json. Kernel names may contain
    underscores, so match the LONGEST declared kernel that prefixes the target
    (matmul_red_div_sq1k -> matmul_red_div, not matmul)."""
    kernels = [k for k in sizes if not k.startswith("_")]
    best = None
    for kernel in kernels:
        prefix = kernel + "_"
        if target.startswith(prefix):
            case_id = target[len(prefix) :]
            if case_id in sizes[kernel] and (
                best is None or len(kernel) > len(best[0])
            ):
                best = (kernel, case_id)
    if best is None:
        sys.exit(
            f"error: '{target}' is not a declared <kernel>_<case_id> in "
            f"{SIZES_JSON} (kernels: {', '.join(sorted(kernels))})"
        )
    kernel, case_id = best
    return kernel, case_id, sizes[kernel][case_id]


def axon_args(
    kernel: str,
    case_id: str,
    cfg: dict,
    *,
    candidate_filter: bool = True,
    candidate_budget: int = 8,
) -> list[str]:
    """The `axon ...` argv for one case — mirrors eval/gen_size_targets.py so
    this driver and `make` produce identical runs."""
    args = [
        f"kernels/{kernel}",
        "--sizes",
        *[str(s) for s in cfg["sizes"]],
    ]
    if "dtype" in cfg:
        args += ["--dtype", cfg["dtype"]]
    if "rtol" in cfg and "atol" in cfg:
        args += ["--rtol", str(cfg["rtol"]), "--atol", str(cfg["atol"])]
    if "lnc" in cfg:
        args += ["--lnc", str(cfg["lnc"])]
    if not candidate_filter:
        args += ["--no-candidate-filter"]
    args += ["--candidate-budget", str(candidate_budget)]
    args += ["--out", f"out/{kernel}_{case_id}.csv"]
    return args


def plan_commands(
    kernel: str,
    case_id: str,
    cfg: dict,
    *,
    emit_on: str,
    config: str | None,
    candidate_filter: bool = True,
    candidate_budget: int = 8,
) -> list[list[str]]:
    """The command sequence for one case. Pure planning (no execution) so it
    is unit-testable; main() runs the plan."""
    remote = [sys.executable, str(REPO / "tools" / "remote.py")]
    if config:
        remote += ["-c", config]
    base = axon_args(
        kernel,
        case_id,
        cfg,
        candidate_filter=candidate_filter,
        candidate_budget=candidate_budget,
    )

    if "lnc" in cfg and int(cfg["lnc"]) >= 2 and emit_on == "host":
        print(
            f"[note] {kernel}_{case_id} declares lnc={cfg['lnc']}: the SPMD "
            f"sweep cannot split at emit, running the whole case on the box.",
            file=sys.stderr,
        )
        emit_on = "box"

    if emit_on == "box":
        return [
            [*remote, "sync"],
            [*remote, "run", "--", "uv", "run", "axon", *base],
            [*remote, "sync", "--pull"],
        ]

    return [
        ["uv", "run", "axon", *base, "--phase", "emit"],
        [*remote, "sync"],
        [*remote, "run", "--", "uv", "run", "axon", *base, "--phase", "bench"],
        [*remote, "sync", "--pull"],
    ]


def step_label(cmd: list[str]) -> str:
    """Name the phase a planned command belongs to, for the timing CSV. Pure
    string work on the plan, so the labels stay in sync with plan_commands
    without threading a second list through it."""
    joined = " ".join(cmd)
    is_remote = "remote.py" in joined
    if is_remote and cmd[-2:] == ["sync", "--pull"]:
        return "pull"
    if is_remote and cmd[-1] == "sync":
        return "sync"
    if "--phase" in cmd:
        # emit runs locally, bench runs through `remote run --`.
        return cmd[cmd.index("--phase") + 1]
    return "bench_all" if is_remote else "axon"


def write_timings(path: Path, rows: list[dict]) -> None:
    """One row per executed step: the wall time of `<kernel>_<case_id>` split
    at emit / sync / bench / pull. `emit` is synthesis + proving + emission;
    `bench` is compile + benchmark + correctness on the device."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(
        prog="run_case", description=(__doc__ or "").splitlines()[0]
    )
    p.add_argument("target", help="<kernel>_<case_id> declared in eval/sizes.json")
    p.add_argument(
        "--emit-on",
        choices=("host", "box"),
        default="host",
        help=(
            "Where the emit phase (synthesis + Z3 + NKI emission) runs. "
            "'host' (default): emit locally, sync, bench on the box. "
            "'box': single-shot legacy loop entirely on the box."
        ),
    )
    p.add_argument("-c", "--config", help="remote.toml host config (default: default)")
    p.add_argument(
        "--candidate-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="filter impossible tile candidates before compile (default: enabled)",
    )
    p.add_argument(
        "--candidate-budget",
        type=int,
        default=8,
        help="legal tile configurations selected per emitted schedule family",
    )
    p.add_argument(
        "--timings",
        help="write per-step wall time to this CSV (step,seconds,returncode,command)",
    )
    args = p.parse_args()

    kernel, case_id, cfg = resolve_case(args.target, load_sizes())
    cmds = plan_commands(
        kernel,
        case_id,
        cfg,
        emit_on=args.emit_on,
        config=args.config,
        candidate_filter=args.candidate_filter,
        candidate_budget=args.candidate_budget,
    )
    timings: list[dict] = []
    try:
        for cmd in cmds:
            print("→", " ".join(cmd), file=sys.stderr)
            started = time.perf_counter()
            rc = subprocess.run(cmd, cwd=REPO).returncode
            timings.append(
                {
                    "step": step_label(cmd),
                    "seconds": round(time.perf_counter() - started, 3),
                    "returncode": rc,
                    "command": " ".join(cmd),
                }
            )
            if rc != 0:
                print(f"error: step failed (exit {rc}); aborting case", file=sys.stderr)
                return rc
        return 0
    finally:
        # Write what completed even on an aborted case, so a partial run still
        # reports where its time went.
        if args.timings:
            write_timings(Path(args.timings), timings)


if __name__ == "__main__":
    sys.exit(main())
