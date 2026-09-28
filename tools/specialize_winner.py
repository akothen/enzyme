#!/usr/bin/env python3
"""Specialize an exported winner kernel to its winning tile configuration.

`axon` exports the head-to-head winner as a *template*: the kernel keeps its
`TILES_IN_BLOCK_*` parameters, and the winning values live separately in the
generated `<stem>_case.py` under `CASE["shape_args"]`. The kernel is therefore
not a standalone artifact — a reader needs a second file to run it correctly,
and the case dict mixes tile configuration with real tensor dims.

This tool runs AFTER a bench completes. It reads the exported winner plus its
case module, removes every parameter that the case declares (the tile args),
and binds those parameters as literal constants at the top of the body. Nothing
else in the source changes: the transform rewrites the signature line and
inserts assignments, then verifies the remaining body is byte-identical in AST
terms to the input.

It does not modify the pipeline and does not replace the exported winner. The
parameterized form stays in `out/winners/`, because the bench sweep passes those
args per config, and `out/` is gitignored scratch.

The specialized kernel is the deliverable, so it lands in the tracked top-level
`winners/` directory instead, with its provenance in the module docstring: the
shape, the tile config, the measured median, the machine it was built on, when
the source winner was synthesized, and the axon commit. That keeps a stale
artifact detectable by reading it, which matters because `make clean-<kernel>`
removes only `out/winners/<case>.py` and `out/winners/<case>_case.py` by name.

Two rules govern `winners/`:

  * **Measured only.** A case with no correct benched row in `out/<stem>.csv` is
    refused. Every artifact in `winners/` carries a latency somebody can check.
  * **Best wins.** A re-run overwrites the stored artifact only when it measures
    faster. A slower or tied re-run is reported as `KEEP` and changes nothing,
    so a noisy or contended run cannot demote a good kernel. `--force` overrides.

Usage:
    python3 tools/specialize_winner.py rmsnorm_h1024
    python3 tools/specialize_winner.py --all
    python3 tools/specialize_winner.py rmsnorm_h1024 --force
    python3 tools/specialize_winner.py --all --from-csv

Stdlib only: it parses `CASE` with `ast.literal_eval` instead of importing the
case module, which would pull in torch/numpy and need the venv.
"""

from __future__ import annotations

import argparse
import ast
import csv
import getpass
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class SpecializeError(RuntimeError):
    """A winner could not be specialized. The message names the reason."""


class KeptExisting(RuntimeError):
    """The stored artifact was already at least as fast, so nothing was written."""


def read_case(case_path: Path) -> dict:
    """Return the CASE dict from a generated `<stem>_case.py`, without importing it."""
    tree = ast.parse(case_path.read_text(), filename=str(case_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "CASE" in names:
            try:
                return ast.literal_eval(node.value)
            except ValueError as exc:  # pragma: no cover - generated source is literal
                raise SpecializeError(
                    f"{case_path}: CASE is not a literal: {exc}"
                ) from exc
    raise SpecializeError(f"{case_path}: no CASE assignment found")


def find_kernel(tree: ast.Module, entry_point: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            return node
    raise SpecializeError(f"no `def {entry_point}` at module level")


def specialize_source(
    src: str, entry_point: str, bindings: dict[str, int]
) -> tuple[str, list[str]]:
    """Drop `bindings` from the kernel signature and bind them as literals.

    Returns the new source and the ordered list of parameters that were bound.
    """
    tree = ast.parse(src)
    fn = find_kernel(tree, entry_point)
    args = fn.args

    # Refuse anything the simple rewrite cannot represent faithfully.
    if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg:
        raise SpecializeError(
            f"{entry_point}: only plain positional parameters are supported"
        )
    if args.defaults:
        raise SpecializeError(
            f"{entry_point}: parameters with defaults are not supported"
        )
    if any(a.annotation is not None for a in args.args):
        raise SpecializeError(f"{entry_point}: annotated parameters are not supported")

    names = [a.arg for a in args.args]
    to_bind = [n for n in names if n in bindings]
    if not to_bind:
        raise SpecializeError(
            f"{entry_point}: no parameter matches a case arg; nothing to specialize"
        )
    keep = [n for n in names if n not in bindings]
    if not keep:
        raise SpecializeError(f"{entry_point}: every parameter is a case arg, refusing")

    lines = src.splitlines(keepends=True)
    # The signature spans the `def` line through the line before the first
    # statement of the body. Replacing that whole span collapses a multi-line
    # signature onto one line, which is the only formatting this tool changes.
    def_idx = fn.lineno - 1
    body_idx = fn.body[0].lineno - 1
    if body_idx <= def_idx:
        raise SpecializeError(f"{entry_point}: body starts on the signature line")

    fn_indent = " " * fn.col_offset
    body_indent = " " * fn.body[0].col_offset
    new_def = f"{fn_indent}def {entry_point}({', '.join(keep)}):\n"

    const_lines = [
        f"{body_indent}# Specialized to the benched winner's tile configuration.\n"
    ]
    const_lines += [f"{body_indent}{n} = {bindings[n]!r}\n" for n in to_bind]
    const_lines.append("\n")

    out = lines[:def_idx] + [new_def] + const_lines + lines[body_idx:]
    new_src = "".join(out)

    verify(new_src, entry_point, keep, to_bind, fn)
    return new_src, to_bind


def verify(
    new_src: str,
    entry_point: str,
    keep: list[str],
    bound: list[str],
    original_fn: ast.FunctionDef,
) -> None:
    """Fail unless the rewrite changed only the signature and prepended constants."""
    fn = find_kernel(ast.parse(new_src), entry_point)
    got = [a.arg for a in fn.args.args]
    if got != keep:
        raise SpecializeError(f"{entry_point}: signature is {got}, expected {keep}")

    n = len(bound)
    prefix, rest = fn.body[:n], fn.body[n:]
    for stmt, name in zip(prefix, bound, strict=True):
        ok = (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name
            and isinstance(stmt.value, ast.Constant)
        )
        if not ok:
            raise SpecializeError(
                f"{entry_point}: expected a constant binding for {name}"
            )

    # Line numbers shift, so compare structure only.
    want = [ast.dump(s, annotate_fields=True) for s in original_fn.body]
    have = [ast.dump(s, annotate_fields=True) for s in rest]
    if want != have:
        raise SpecializeError(
            f"{entry_point}: body changed by the rewrite ({len(want)} vs {len(have)} statements)"
        )


def device_id() -> str:
    """The Trainium device the kernel was measured on, from `neuron-ls`.

    A hostname or ssh alias says nothing about the hardware, and the latency in
    this artifact only means something next to the device that produced it. So
    record the instance type, the NeuronCore layout, and the logical-core config,
    and keep the host only as a way back to the run directory.
    """
    host = f"{getpass.getuser()}@{socket.gethostname()}"
    try:
        out = subprocess.run(
            ["neuron-ls", "--json-output"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        devices = json.loads(out.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return f"unknown device (host {host})"
    if not devices:
        return f"no Neuron device (host {host})"
    first = devices[0]
    kind = first.get("instance_type") or "unknown instance"
    per = first.get("nc_count")
    lnc = first.get("logical_neuroncore_config")
    parts = [kind, f"{len(devices)} neuron devices"]
    if per:
        parts.append(f"{per} cores each")
    if lnc:
        parts.append(f"lnc={lnc}")
    return f"{', '.join(parts)} (host {host})"


def winner_from_csv(
    csv_path: Path, module_dir: Path, name: str
) -> tuple[Path, dict[str, int], float]:
    """Pick the fastest correct row from a run CSV and return its module.

    This reads the run's own record instead of trusting `<stem>_case.py`, so it
    also works for a case that exported no harness. Tile columns drop the
    `TILES_IN_BLOCK_` prefix (see `bench_runner.tile_cols`), and the emitted
    module for a row is `<name>__v<hw>_t<tile>.py`.
    """
    if not csv_path.is_file():
        raise SpecializeError(f"{csv_path} not found")
    best: tuple[float, dict[str, str]] | None = None
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if row.get("correct", "").lower() != "true":
            continue
        raw = row.get("median_ms") or ""
        if not raw:
            continue
        ms = float(raw)
        if best is None or ms < best[0]:
            best = (ms, row)
    if best is None:
        raise SpecializeError(f"{csv_path}: no correct benched row")
    ms, row = best
    module = module_dir / f"{name}__v{row['hw_variant']}_t{row['tile_variant']}.py"
    if not module.is_file():
        raise SpecializeError(f"{module} not found (was out/<stem>/ cleaned?)")
    tiles = {
        col: int(val)
        for col, val in row.items()
        if col.startswith("TILES_") and val not in (None, "")
    }
    return module, tiles, ms


def source_commit(repo: Path) -> str:
    """Short HEAD sha of the tree that produced the winner, or a marker.

    `AXON_COMMIT` wins when set. That matters on a remote Trainium box, where
    `.git` is excluded from the rsync, so git cannot answer there: export
    `AXON_COMMIT=$(git rev-parse --short HEAD)` on the driving machine and pass
    it through, or accept `unknown` in the artifact.
    """
    env = os.environ.get("AXON_COMMIT", "").strip()
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = out.stdout.strip()
    if out.returncode != 0 or not sha:
        return "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    return f"{sha}-dirty" if dirty.stdout.strip() else sha


MEASURED_RE = re.compile(
    r"^\s*measured\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*us\b", re.MULTILINE
)


def measured_median_us(csv_path: Path, tile_values: dict[str, int]) -> float:
    """Median time in microseconds of the winner row, read from the run CSV.

    The CSV's tile columns drop the `TILES_IN_BLOCK_` prefix (see
    `bench_runner.tile_cols`), so map the case's parameter names onto them. Only
    a row that passed the on-device correctness check counts.

    Raises `SpecializeError` when there is no measurement, because an unmeasured
    kernel must never enter `winners/`: the whole point of that directory is that
    every artifact in it carries a number somebody can check.
    """
    if not csv_path.is_file():
        raise SpecializeError(
            f"no measurement: {csv_path} is absent, so this run's latency is unknown"
        )
    wanted = {
        "TILES_" + name.removeprefix("TILES_IN_BLOCK_").removeprefix("TILES_"): str(val)
        for name, val in tile_values.items()
    }
    best: float | None = None
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("correct", "").lower() != "true":
                continue
            if any(row.get(col) != val for col, val in wanted.items()):
                continue
            raw = row.get("median_ms") or ""
            if not raw:
                continue
            ms = float(raw)
            if best is None or ms < best:
                best = ms
    if best is None:
        raise SpecializeError(
            f"no measurement: {csv_path.name} has no correct benched row at {wanted}"
        )
    return best * 1000.0


def recorded_measurement(path: Path) -> float | None:
    """The `measured` figure already stored in an artifact, in microseconds."""
    if not path.is_file():
        return None
    match = MEASURED_RE.search(path.read_text())
    return float(match.group(1)) if match else None


def decide_write(out_path: Path, new_us: float, force: bool) -> str | None:
    """Return a skip reason, or None when the new artifact should be written.

    `winners/` keeps the best measured kernel per case, so a slower re-run must
    not overwrite a faster recorded one. A re-run that ties is treated as not an
    improvement and is skipped, which keeps the stored provenance stable.
    """
    if force or not out_path.is_file():
        return None
    old = recorded_measurement(out_path)
    if old is None:
        return None  # unmeasured or hand-edited: a measured artifact supersedes it
    # Compare at the precision the header stores. The CSV carries more digits
    # than the artifact records, so comparing raw against rounded would make an
    # identical re-run look like a tiny improvement and rewrite the file forever.
    new_rounded = round(new_us, 2)
    if new_rounded < old:
        return None
    return (
        f"kept existing {old:.2f} us; this run measured {new_rounded:.2f} us "
        f"({'tie' if new_rounded == old else 'slower'}). Pass --force to overwrite."
    )


def specialize_from_csv(
    stem: str, out_dir: Path, repo: Path, force: bool = False
) -> Path:
    """Derive the winner from the run CSV and the emitted candidates, then specialize.

    This path does not read `<stem>_case.py`. It re-picks the fastest correct row
    from `out/<stem>.csv`, finds that row's emitted module under `out/<stem>/`,
    and binds the row's own tile values. Use it when there is no exported case
    harness, or to confirm the exported winner really is the fastest correct row.
    """
    csv_path = repo / "out" / f"{stem}.csv"
    run_dir = repo / "out" / stem
    if not run_dir.is_dir():
        raise SpecializeError(f"{run_dir} not found (was the run cleaned?)")
    # Candidate modules are named "<kernel>__v<i>_t<j>.py"; the kernel half is
    # spec.name, which is the stem minus its case id, so read it off a candidate
    # instead of guessing where the case id starts.
    candidates = sorted(run_dir.glob("*__v*_t*.py"))
    if not candidates:
        raise SpecializeError(f"no emitted candidates in {run_dir}")
    name = candidates[0].name.split("__v")[0]

    module, csv_tiles, ms = winner_from_csv(csv_path, run_dir, name)

    # Map CSV tile columns back onto the kernel's parameter names.
    tree = ast.parse(module.read_text())
    fn = find_kernel(tree, name)
    bindings: dict[str, int] = {}
    for param in (a.arg for a in fn.args.args):
        col = "TILES_" + param.removeprefix("TILES_IN_BLOCK_").removeprefix("TILES_")
        if col in csv_tiles:
            bindings[param] = csv_tiles[col]
    if not bindings:
        raise SpecializeError(
            f"{module.name}: no parameter maps to a CSV tile column {sorted(csv_tiles)}"
        )

    measured_us = ms * 1000.0
    out_path = out_dir / f"{stem}.py"
    skip = decide_write(out_path, measured_us, force)
    if skip:
        raise KeptExisting(skip)

    new_src, bound = specialize_source(module.read_text(), name, bindings)
    stamp = time.strftime(
        "%Y-%m-%d %H:%M:%S %Z", time.localtime(module.stat().st_mtime)
    )
    header = (
        f'"""Axon winner for case {stem!r}, specialized to its tile configuration.\n\n'
        f"Generated by tools/specialize_winner.py --from-csv. Do not edit by hand.\n\n"
        f"The winner was re-derived here from the run CSV: the fastest row whose\n"
        f"on-device output passed the correctness check. The tile configuration is\n"
        f"bound as constants, so this module takes only tensor inputs.\n\n"
        f"Provenance\n"
        f"----------\n"
        f"    entry point   : {name}\n"
        f"    tile config   : {', '.join(f'{k}={bindings[k]}' for k in bound)}\n"
        f"    measured      : {measured_us:.2f} us median (fastest correct row)\n"
        f"    emitted       : {stamp} (mtime of the candidate module)\n"
        f"    axon commit   : {source_commit(repo)}\n"
        f"    measured on   : {device_id()}\n\n"
        f"Where the run artifacts live on that host\n"
        f"----------------------------------------\n"
        f"    candidate     : {module}\n"
        f"    run CSV       : {csv_path}\n"
        f"    variants+NEFF : {run_dir}/\n"
        f'"""\n\n'
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + new_src)
    return out_path


def specialize_one(
    stem: str, source: Path, out_dir: Path, repo: Path, force: bool = False
) -> Path:
    case_path = source / f"{stem}_case.py"
    if not case_path.is_file():
        raise SpecializeError(f"{case_path} not found (is {stem} a head-to-head case?)")
    case = read_case(case_path)

    entry_point = case.get("entry_point")
    kernel_rel = case.get("kernel_rel")
    shape_args = case.get("shape_args") or {}
    if not entry_point or not kernel_rel:
        raise SpecializeError(f"{case_path}: CASE lacks entry_point/kernel_rel")

    kernel_path = (source / kernel_rel).resolve()
    if not kernel_path.is_file():
        raise SpecializeError(f"{kernel_path} not found")

    src = kernel_path.read_text()
    new_src, bound = specialize_source(src, entry_point, shape_args)

    tile_values = {n: int(shape_args[n]) for n in bound}
    dims = {k: v for k, v in shape_args.items() if k not in tile_values}
    stamp = time.strftime(
        "%Y-%m-%d %H:%M:%S %Z", time.localtime(kernel_path.stat().st_mtime)
    )
    csv_path = repo / "out" / f"{stem}.csv"
    measured_us = measured_median_us(csv_path, tile_values)
    out_path = out_dir / f"{stem}.py"
    skip = decide_write(out_path, measured_us, force)
    if skip:
        raise KeptExisting(skip)

    # Absolute paths, because the run directory usually lives on a remote
    # Trainium box: a reader needs the machine AND the path to go back to it.
    run_dir = repo / "out" / stem
    header = (
        f'"""Axon winner for case {stem!r}, specialized to its tile configuration.\n\n'
        f"Generated by tools/specialize_winner.py. Do not edit by hand.\n\n"
        f"This is a standalone kernel: it takes only tensor inputs. The winning tile\n"
        f"configuration is bound as constants in the body, so no caller has to supply\n"
        f"it and the file cannot disagree with the run it came from.\n\n"
        f"Provenance\n"
        f"----------\n"
        f"    entry point   : {entry_point}\n"
        f"    shape         : {', '.join(f'{k}={v}' for k, v in dims.items()) or 'n/a'}\n"
        f"    tile config   : {', '.join(f'{k}={v}' for k, v in tile_values.items())}\n"
        f"    measured      : {measured_us:.2f} us median (on-device, correct)\n"
        f"    synthesized   : {stamp} (mtime of the source kernel)\n"
        f"    axon commit   : {source_commit(repo)}\n"
        f"    measured on   : {device_id()}\n\n"
        f"Where the run artifacts live on that host\n"
        f"----------------------------------------\n"
        f"    winner kernel : {kernel_path}\n"
        f"    case metadata : {case_path}\n"
        f"    run CSV       : {csv_path}\n"
        f"    variants+NEFF : {run_dir}/\n"
        f'"""\n\n'
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.py"
    out_path.write_text(header + new_src)
    return out_path


def discover(winners: Path) -> list[str]:
    return sorted(p.name[: -len("_case.py")] for p in winners.glob("*_case.py"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stems", nargs="*", help="case stems, e.g. rmsnorm_h1024")
    ap.add_argument(
        "--all", action="store_true", help="specialize every exported winner"
    )
    ap.add_argument(
        "--source-dir",
        default=REPO / "out" / "winners",
        type=Path,
        help="where axon exported the winners (scratch; default out/winners)",
    )
    ap.add_argument(
        "--out-dir",
        default=REPO / "winners",
        type=Path,
        help="where to write the tracked deliverable (default winners/)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing artifact even when this run is not faster",
    )
    ap.add_argument(
        "--from-csv",
        action="store_true",
        help="re-derive the winner from out/<stem>.csv and the emitted candidates "
        "instead of reading <stem>_case.py",
    )
    args = ap.parse_args(argv)

    source: Path = args.source_dir
    if not args.from_csv and not source.is_dir():
        print(
            f"error: {source} is not a directory (run a bench first)", file=sys.stderr
        )
        return 2

    if args.all:
        stems = (
            sorted(p.stem for p in (REPO / "out").glob("*.csv"))
            if args.from_csv
            else discover(source)
        )
    else:
        stems = args.stems
    if not stems:
        print("error: name at least one stem, or pass --all", file=sys.stderr)
        return 2

    failed = 0
    for stem in stems:
        try:
            if args.from_csv:
                out = specialize_from_csv(stem, args.out_dir, REPO, args.force)
            else:
                out = specialize_one(stem, source, args.out_dir, REPO, args.force)
        except KeptExisting as exc:
            # Not a failure: the stored kernel is still the best measured one.
            print(f"KEEP {stem}: {exc}")
        except SpecializeError as exc:
            print(f"SKIP {stem}: {exc}", file=sys.stderr)
            failed += 1
        else:
            print(f"wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    return 1 if failed and failed == len(stems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
