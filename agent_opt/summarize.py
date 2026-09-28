#!/usr/bin/env python3
"""Aggregate feedback_loop results.json files into results_summary.csv.

Walks agent_opt/results/<kernel>/<ts>/results.json, takes the most recent
run per kernel, and emits one row per kernel: the Axon baseline, the best agent
kernel, the (optional) nkilib latency, and the three latency ratios.

Ratios are latency ratios (< 1 = numerator faster):
  axon_over_nkilib  = axon_us       / nkilib_us
  agent_over_axon = best_agent_us / axon_us
  agent_over_nkilib = best_agent_us / nkilib_us

NOTE the metric mix: axon_us/best_agent_us are ActiveInferenceTime, while
nkilib_us is the InferenceTime nkilib's harness reports, so the nkilib ratios can
be apples-to-oranges (and nkilib kernels are often general-purpose / lnc=2). The
confound-free number is agent_over_axon (both single-core, same shape).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

COLUMNS = [
    "kernel",
    "input_sizes",
    "axon_us",
    "best_agent_us",
    "nkilib_us",
    "axon_over_nkilib",
    "agent_over_axon",
    "agent_over_nkilib",
    "postprocess_total_s",
    "generate_s",
    "measure_s",
]


def _latest_per_kernel(results_root: Path) -> dict[str, dict]:
    """Most-recent results.json per kernel (by the recorded timestamp)."""
    latest: dict[str, tuple[str, dict]] = {}
    for rj in results_root.glob("*/*/results.json"):
        try:
            d = json.loads(rj.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        kernel = d.get("kernel") or rj.parent.parent.name
        ts = d.get("timestamp", "")
        if kernel not in latest or ts > latest[kernel][0]:
            latest[kernel] = (ts, d)
    return {k: v[1] for k, v in latest.items()}


def _fmt_sizes(shape_args: dict | None) -> str:
    """The dim extents (tile args dropped), e.g. [1024,8192]."""
    if not shape_args:
        return ""
    dims = [k for k in shape_args if not k.startswith("TILES_IN_BLOCK")]
    return "[" + ",".join(str(shape_args[k]) for k in dims) + "]"


def _ratio(num: float | None, den: float | None) -> float | str:
    return round(num / den, 3) if (num and den) else ""


def _row(d: dict) -> dict:
    axon = d.get("baseline_active_us")
    best = d.get("best_active_us")
    nkilib = d.get("nkilib_inference_us")
    # `timings` exists from results schema 2 on; older runs leave these blank.
    t = d.get("timings") or {}
    return {
        "kernel": d.get("kernel", ""),
        "input_sizes": _fmt_sizes(d.get("shape_args")),
        "axon_us": round(axon, 1) if axon else "",
        "best_agent_us": round(best, 1) if best else "",
        "nkilib_us": round(nkilib, 1) if nkilib else "",
        "axon_over_nkilib": _ratio(axon, nkilib),
        "agent_over_axon": _ratio(best, axon),
        "agent_over_nkilib": _ratio(best, nkilib),
        "postprocess_total_s": t.get("total_seconds", ""),
        "generate_s": t.get("generate_seconds", ""),
        "measure_s": t.get("measure_seconds", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "results" / "results_summary.csv"))
    args = ap.parse_args()

    runs = _latest_per_kernel(Path(args.results_root))
    rows = [_row(d) for _, d in sorted(runs.items())]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} kernel(s))")


if __name__ == "__main__":
    main()
