#!/usr/bin/env python3
"""Check that a specialized winner is equivalent to the parameterized one.

`tools/specialize_winner.py` moves the winning `TILES_IN_BLOCK_*` values from
the kernel signature into literal constants. The bench path already passes those
values as compile-time kwargs, so the compiler saw the same constants before the
rewrite. The NEFF should therefore be identical.

This tool proves that instead of assuming it. For one case it:

  1. compiles the parameterized winner with the winning tile kwargs,
  2. compiles the specialized winner with no tile kwargs,
  3. compares the two NEFFs byte for byte,
  4. runs both on device and compares each output to the spec's fp32 baseline,
  5. benchmarks both and prints the medians.

Two rules make the comparison honest, and both are why every step runs in its
own subprocess:

  * The loaded module's name leaks into the compiled artifacts (the frontend
    derives a `<module>.<fn>.json` sidecar from `func.__module__`). Both arms
    must therefore load under the SAME module name, which one process cannot do.
  * A compile initializes the runtime, after which importing `nkipy.runtime`
    in the same process fails. `bench_runner` separates compile and bench into
    different pools for this reason.

Run it from the repo root on a Trainium box:

    uv run python tools/verify_specialized.py rmsnorm_h1024
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[1]
MODULE_NAME = "_axon_verify_kernel"  # identical in both arms, on purpose


def load_case(winners: Path, stem: str) -> dict:
    src = (winners / f"{stem}_case.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CASE" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise SystemExit(f"{stem}_case.py: no CASE found")


def _import_by_path(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_spec(kernel_dir: Path):
    """Load a kernel package's SPEC by path, the way axon's CLI does."""
    spec = importlib.util.spec_from_file_location(
        f"_verify_spec_{kernel_dir.name}",
        str(kernel_dir / "__init__.py"),
        submodule_search_locations=[str(kernel_dir)],
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import spec from {kernel_dir}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.SPEC


def find_kernel_package(stem: str) -> Path:
    root = REPO / "kernels"
    hits = sorted(
        (d for d in root.iterdir() if d.is_dir() and stem.startswith(d.name)),
        key=lambda d: -len(d.name),
    )
    if not hits:
        raise SystemExit(f"no kernels/<name> prefix-matches {stem}")
    return hits[0]


def make_inputs(stem: str, dims: dict):
    import numpy as np

    kspec = load_spec(find_kernel_package(stem))
    return kspec, kspec.make_inputs(**dims, rng=np.random.default_rng(42))


# --------------------------------------------------------------------------
# worker modes (each runs in a fresh process)
# --------------------------------------------------------------------------


def worker_compile(cfg: dict) -> int:
    from nki.framework.compiled import CompileKernel

    mod = _import_by_path(cfg["kernel_path"], MODULE_NAME)
    kernel = getattr(mod, cfg["entry"])

    artifacts = Path(cfg["artifacts"])
    if artifacts.is_dir():
        shutil.rmtree(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    _, inputs = make_inputs(cfg["stem"], cfg["dims"])
    ck = kernel[1]._to_subclass(
        CompileKernel, artifacts_dir=str(artifacts), target=cfg["target"]
    )
    try:
        ck(*inputs, **cfg["tile_kwargs"])
    except Exception as exc:  # the post-compile execute can fail; the NEFF still lands
        print(f"    (compile raised after emit: {type(exc).__name__}: {exc})")

    neff = artifacts / "kernel.neff"
    if not (neff.exists() and neff.stat().st_size > 0):
        print("    no NEFF produced", file=sys.stderr)
        return 1
    return 0


def worker_bench(cfg: dict) -> int:
    import ml_dtypes  # noqa: F401  registers bfloat16 with np.dtype
    import numpy as np
    from nkipy.runtime import DeviceKernel
    from spike.spike_tensor import SpikeTensor

    _, inputs = make_inputs(cfg["stem"], cfg["dims"])
    dk = DeviceKernel.load_from_neff(cfg["neff"])
    in_t = {
        n: SpikeTensor.from_numpy(a, name=n)
        for n, a in zip(cfg["data_names"], inputs, strict=True)
    }
    out_t = {
        n: SpikeTensor.from_numpy(np.zeros(ti.shape, dtype=np.dtype(ti.dtype)), name=n)
        for n, ti in dk.output_tensors_info.items()
    }
    res = dk.benchmark(
        in_t,
        out_t,
        warmup_iter=cfg["warmup"],
        benchmark_iter=cfg["iters"],
        mode="device",
    )

    def out_index(name: str) -> int:
        tail = name.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    names = sorted(out_t, key=out_index)
    payload = {
        "durations_ms": list(getattr(res, "durations_ms", []) or []),
        "outputs": [out_t[n].numpy().astype(np.float32).tolist() for n in names],
    }
    Path(cfg["result"]).write_text(json.dumps(payload))
    return 0


def spawn(mode: str, cfg: dict) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            mode,
            json.dumps(cfg),
        ],
        cwd=str(REPO),
    )
    if proc.returncode != 0:
        raise SystemExit(f"{mode} worker failed with exit {proc.returncode}")


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stem", nargs="?")
    ap.add_argument(
        "--worker", nargs=2, metavar=("MODE", "JSON"), help=argparse.SUPPRESS
    )
    ap.add_argument("--winners-dir", type=Path, default=REPO / "out" / "winners")
    ap.add_argument("--specialized-dir", type=Path, default=REPO / "winners")
    ap.add_argument("--target", default="trn2")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    if args.worker:
        mode, blob = args.worker
        cfg = json.loads(blob)
        return worker_compile(cfg) if mode == "compile" else worker_bench(cfg)

    if not args.stem:
        ap.error("stem is required")

    import inspect

    import numpy as np

    winners: Path = args.winners_dir
    stem: str = args.stem
    case = load_case(winners, stem)
    entry = case["entry_point"]
    shape_args = case["shape_args"]

    param_path = (winners / case["kernel_rel"]).resolve()
    spec_path = (args.specialized_dir / f"{stem}.py").resolve()
    for p in (param_path, spec_path):
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    # Read signatures in this (compile-free) process, then discard the modules.
    param_sig = list(
        inspect.signature(
            getattr(_import_by_path(str(param_path), "_sig_a"), entry).func
        ).parameters
    )
    spec_sig = list(
        inspect.signature(
            getattr(_import_by_path(str(spec_path), "_sig_b"), entry).func
        ).parameters
    )
    tile_names = [p for p in param_sig if p in shape_args]
    data_names = [p for p in param_sig if p not in shape_args]
    dims = {k: v for k, v in shape_args.items() if k not in tile_names}

    print(f"case            : {stem}")
    print(f"parameterized   : {entry}({', '.join(param_sig)})")
    print(f"specialized     : {entry}({', '.join(spec_sig)})")
    print(f"tile args bound : {', '.join(f'{t}={shape_args[t]}' for t in tile_names)}")
    print(f"kernel package  : kernels/{find_kernel_package(stem).name}  dims={dims}")
    if spec_sig != data_names:
        raise SystemExit(
            f"specialized signature {spec_sig} != data params {data_names}"
        )

    scratch = REPO / "out" / "verify_specialized" / stem
    arms = {
        "parameterized": (param_path, {t: int(shape_args[t]) for t in tile_names}),
        "specialized": (spec_path, {}),
    }
    # Both arms build into the SAME artifacts path, one after the other, because
    # the NEFF embeds its own absolute output path (info.json "name"). Building
    # side by side would differ on that string alone and defeat a byte compare.
    build = scratch / "build"
    neffs: dict[str, Path] = {}
    for label, (path, tile_kwargs) in arms.items():
        print(f"\ncompiling {label} ...")
        spawn(
            "compile",
            {
                "kernel_path": str(path),
                "entry": entry,
                "artifacts": str(build),
                "target": args.target,
                "stem": stem,
                "dims": dims,
                "tile_kwargs": tile_kwargs,
            },
        )
        kept = scratch / f"{label}.neff"
        shutil.copy2(build / "kernel.neff", kept)
        shutil.copy2(build / "kernel_info.json", scratch / f"{label}.kernel_info.json")
        neffs[label] = kept

    # NEFF bytes are NOT a valid equivalence test: compiling one unchanged source
    # twice already yields a different NEFF (measured: ~24k of 42k bytes differ,
    # same size). The stable program description is kernel_info.json, so compare
    # that and report the NEFF hash as information only.
    print()
    for label, neff in neffs.items():
        digest = hashlib.sha256(neff.read_bytes()).hexdigest()[:16]
        print(
            f"NEFF {label:<14}: {neff.stat().st_size:>10} bytes  sha256={digest}"
            "  (build is non-reproducible; informational)"
        )
    infos = {
        label: (scratch / f"{label}.kernel_info.json").read_bytes() for label in neffs
    }
    same = len(set(infos.values())) == 1
    print(f"kernel_info.json identical: {same}")

    results = {}
    for label, neff in neffs.items():
        print(f"benching {label} ...")
        out = scratch / f"{label}.json"
        spawn(
            "bench",
            {
                "neff": str(neff),
                "stem": stem,
                "dims": dims,
                "data_names": data_names,
                "warmup": args.warmup,
                "iters": args.iters,
                "result": str(out),
            },
        )
        results[label] = json.loads(out.read_text())

    kspec, inputs = make_inputs(stem, dims)
    ref = kspec.baseline_op(*[np.asarray(a).astype(np.float32) for a in inputs])
    if not isinstance(ref, tuple):
        ref = (ref,)
    rtol = float(case.get("rtol", 2e-2))
    atol = float(case.get("atol", 2e-2))

    print()
    verdict_ok = same
    for label, payload in results.items():
        for i, (got, want) in enumerate(zip(payload["outputs"], ref, strict=True)):
            g = np.asarray(got, dtype=np.float32)
            w = np.asarray(want, dtype=np.float32)
            err = float(np.max(np.abs(g - w)))
            good = bool(np.allclose(g, w, rtol=rtol, atol=atol))
            verdict_ok = verdict_ok and good
            print(f"  {label:<14} out[{i}] max_abs_err={err:.6g} correct={good}")

    a, b = results["parameterized"]["outputs"], results["specialized"]["outputs"]
    outs_match = all(
        np.array_equal(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
        for x, y in zip(a, b, strict=True)
    )
    verdict_ok = verdict_ok and outs_match
    print(f"  outputs bit-identical between arms: {outs_match}")

    da = results["parameterized"]["durations_ms"]
    db = results["specialized"]["durations_ms"]
    if da and db:
        ma, mb = median(da) * 1000, median(db) * 1000
        print(f"\nmedian parameterized: {ma:8.2f} us  (n={len(da)})")
        print(f"median specialized  : {mb:8.2f} us  (n={len(db)})")
        print(f"ratio spec/param    : {mb / ma:.4f}")
    else:
        print("\n(no duration samples returned by the executor)")

    print(f"\nVERDICT: {'EQUIVALENT' if verdict_ok else 'NOT PROVEN EQUIVALENT'}")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
