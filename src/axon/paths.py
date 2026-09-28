from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from axon.kernel_spec import KernelSpec


def out_dir() -> Path:
    """Where axon writes generated artifacts."""
    return Path.cwd() / "out"


def _fs_safe(tag: str) -> str:
    """Filesystem-safe form of a Σ tag (`shard=k:P+` -> `shard_k_Pplus`)."""
    return tag.replace("=", "_").replace(":", "_").replace("+", "plus").replace("#", "")


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Single source of truth for every per-run path. Only the CSV path is
    stored (absolute); every other path is derived from it, so the parts cannot
    drift out of sync."""

    csv: Path

    @property
    def run_dir(self) -> Path:
        """Per-variant modules + NEFFs live here (the CSV path minus `.csv`)."""
        return self.csv.with_suffix("")

    @property
    def results(self) -> Path:
        """The durable per-key row store. Not a Make target: it survives an
        interrupted run and is published to `csv` only on full coverage."""
        return self.run_dir / "results.csv"

    @property
    def baseline_dir(self) -> Path:
        return self.run_dir.parent / ("baseline_" + self.run_dir.name)

    @property
    def baseline_csv(self) -> Path:
        return self.baseline_dir.parent / (self.baseline_dir.name + ".csv")

    @property
    def winners_dir(self) -> Path:
        """Head-to-head winners (the winner kernel + its per-case harness)."""
        return self.csv.parent / "winners"

    def variant_module(self, fn_name: str, i: int, j: int) -> Path:
        return self.run_dir / f"{fn_name}__v{i}_t{j}.py"

    def spmd_variant_module(self, fn_name: str, i: int, j: int, plan_tag: str) -> Path:
        """An SPMD `(hw_variant, tile_variant, Σ)` module. The Σ tag keeps
        sibling shardings of one (v, t) cell from colliding."""
        return self.run_dir / f"{fn_name}__v{i}_t{j}_{plan_tag}.py"

    def neff(self, i: int, j: int, tile_values, plan_tag: str = "") -> Path:
        tile_tag = "_".join(map(str, tile_values))
        suffix = f"_{_fs_safe(plan_tag)}" if plan_tag else ""
        return self.run_dir / f"__v{i}_t{j}{suffix}_tiles_{tile_tag}" / "kernel.neff"

    def winner_kernel(self, entry_name: str) -> Path:
        return self.winners_dir / f"{entry_name}.py"

    def winner_case(self, entry_name: str) -> Path:
        return self.winners_dir / f"{entry_name}_case.py"


def run_paths(spec: KernelSpec, out: str | None) -> RunPaths:
    """Compute every per-run path from `out`, the run STEM: `out/<stem>.csv`
    is the per-case CSV and `out/<stem>/` the run dir. `--out out/matmul_sq1k`
    and `--out out/matmul_sq1k.csv` are equivalent (the `.csv` suffix is
    optional). Omitted, the stem defaults to `out/nki_<name>`. The CSV path is
    resolved absolute so every derived path is absolute too (compile workers
    are spawned with their own cwd).

    A trailing slash (`--out results/`) names a container directory instead:
    the default `nki_<name>.csv` is placed inside it. A bare stem is NOT
    treated as a container even if a directory of that name exists —
    `out/<stem>/` is the run dir the split phases share, so an emit-then-bench
    sequence must derive the same CSV from the same `--out` both times."""
    default_name = f"nki_{spec.name}.csv"
    if out is None:
        csv = out_dir() / default_name
    elif out.endswith("/"):
        csv = Path(out) / default_name
    else:
        p = Path(out)
        csv = p if p.suffix == ".csv" else p.with_name(p.name + ".csv")
    return RunPaths(csv=csv.resolve())
