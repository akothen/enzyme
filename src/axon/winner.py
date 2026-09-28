"""Winner selection + per-case nkilib harness emission, folded into `axon`'s run
tail (the surviving half of the deleted `axon export` subcommand)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from axon.bench_runner import tile_cols
from axon.case_gen import generate_case_module
from axon.codegen import nki_safe_var
from axon.kernel_spec import KernelSpec
from axon.paths import RunPaths, _fs_safe

PickBy = Literal["median", "mean", "min"]


@dataclass(frozen=True)
class WinnerRow:
    hw_variant: int
    tile_variant: int
    mean_ms: float | None
    median_ms: float | None
    min_ms: float | None
    sizes: tuple[int, ...]
    tile_values: tuple[int, ...]
    lnc: int = 1
    sharding: str = "lnc1"


def _verified_wrong(correct: object) -> bool:
    """True only for a `correct` cell that is explicitly false. NaN means
    unverified, not wrong."""
    if pd.isna(correct):
        return False
    return not bool(correct)


def _tag(row) -> str:
    return f"v{int(row['hw_variant'])}_t{int(row['tile_variant'])}"


def pick_winner(
    csv_path: Path, spec: KernelSpec, pick_by: PickBy, *, rtol: float, atol: float
) -> WinnerRow | None:
    """Pick the fastest timed variant that is not verified incorrect, so a
    fast-but-wrong variant loses to a correct one instead of aborting the case.
    Only when every timed variant is verified incorrect does this fail loud."""
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    sort_col = f"{pick_by}_ms"
    if sort_col not in df.columns:
        return None
    valid = df[df[sort_col].notna()].copy()
    if valid.empty:
        return None

    cols = tile_cols(spec.tile_args)
    ranked = valid.sort_values(sort_col)
    wrong = ranked["correct"].map(_verified_wrong).astype(bool)
    survivors = ranked[~wrong]

    if survivors.empty:
        # No fallback exists: report the fastest of the wrong rows.
        worst = ranked.iloc[0]
        raise AssertionError(
            f"{spec.name} winner {_tag(worst)} failed on-device correctness: "
            f"max_abs_err={worst['max_abs_err']} exceeds rtol={rtol} atol={atol}"
        )

    row = survivors.iloc[0]
    tag = _tag(row)
    # The winner is the first survivor in rank order, so the rows it passed are
    # exactly the leading verified-wrong ones. Slower wrong rows are not skipped.
    faster_wrong = list(wrong).index(False)
    if faster_wrong:
        print(
            f"[info] {spec.name} skipped {faster_wrong} faster but "
            f"incorrect variant(s); winner is {tag}"
        )
    if pd.isna(row["correct"]):
        print(
            f"[warn] {spec.name} winner {tag} correctness unverified "
            f"(no baseline_op or no device outputs); shipping unchecked"
        )

    return WinnerRow(
        hw_variant=int(row["hw_variant"]),
        tile_variant=int(row["tile_variant"]),
        mean_ms=float(row["mean_ms"]),
        median_ms=float(row["median_ms"]),
        min_ms=float(row["min_ms"]),
        sizes=tuple(int(row[c]) for c in spec.dim_vars),
        tile_values=tuple(int(row[c]) for c in cols),
        lnc=int(row["lnc"]) if "lnc" in row else 1,
        sharding=str(row["sharding"]) if "sharding" in row else "lnc1",
    )


def export_winner(
    spec: KernelSpec,
    winner: WinnerRow,
    *,
    case_id: str,
    kernel_path: Path | str,
    paths: RunPaths,
    dtype: str,
    rtol: float,
    atol: float,
) -> None:
    """Emit the head-to-head artifacts for an already-picked winner: the winner
    kernel module and a self-describing per-case nkilib harness next to it. Pure
    per-case writer — reads nothing, mutates no shared state."""
    entry_name = f"{spec.name}_{case_id}"
    print(f"\n=== export {entry_name} ===")
    print(
        f"  winner: hw_variant={winner.hw_variant}, "
        f"tile_variant={winner.tile_variant}, median_ms={winner.median_ms}, "
        f"mean_ms={winner.mean_ms}, min_ms={winner.min_ms}"
    )

    win_kernel = paths.winner_kernel(entry_name)
    win_case = paths.winner_case(entry_name)

    fn_name = nki_safe_var(spec.name)
    # An lnc=2 winner lives under the SPMD plan-tagged module name; an lnc=1
    # winner under the plain per-variant name.
    if winner.lnc >= 2:
        src_module = paths.spmd_variant_module(
            fn_name, winner.hw_variant, winner.tile_variant, _fs_safe(winner.sharding)
        )
    else:
        src_module = paths.variant_module(
            fn_name, winner.hw_variant, winner.tile_variant
        )
    if not src_module.is_file():
        raise FileNotFoundError(
            f"winning variant module not found at {src_module}; the CSV row "
            f"references a variant whose source file is missing — re-run axon."
        )
    win_kernel.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_module, win_kernel)
    print(f"  wrote {win_kernel}")

    shape_args = dict(zip(spec.dim_vars, winner.sizes, strict=True))
    tile_kwargs = {
        name: int(val)
        for name, val in zip(spec.tile_args, winner.tile_values, strict=True)
    }
    generate_case_module(
        spec,
        kernel_path,
        dtype,
        win_case,
        name=entry_name,
        entry_point=fn_name,
        shape_args={**shape_args, **tile_kwargs},
        rtol=rtol,
        atol=atol,
    )
    print(f"  wrote {win_case}")
