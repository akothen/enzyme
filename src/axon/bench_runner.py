"""In-process NKI compile/bench driver invoked by `axon`.

`axon.cli:trace_kernel` calls `run_nki_bench` once per emitted
`(hw_variant, tile_variant)` and `run_baseline` once per baseline
invocation. Both functions write/append to CSVs under `<cwd>/out/`.

The emitted kernels are `@nki.jit` functions; they compile via
`nki.framework.compiled.CompileKernel` (which writes
`<artifacts_dir>/kernel.neff`) and bench via the runtime
`DeviceKernel.load_from_neff` NEFF-direct path.

nkipy / spike / nki imports MUST stay inside function bodies — bench workers
(`_bench_one_neff`) run under a spawn pool whose initializer pins
`NEURON_RT_VISIBLE_CORES` *before* nkipy is imported. A top-level nkipy
import would re-import in each child before the initializer runs and
defeat per-worker core pinning.
"""

import csv
import importlib.util
import io
import multiprocessing
import os
import shutil
import sys
from collections import Counter, deque
from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Executor, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from axon.candidate_filter import read_manifest, rejection_reason, select_diverse
from axon.kernel_spec import SEED, KernelSpec
from axon.paths import RunPaths

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

__all__ = [
    "run_nki_bench",
    "run_baseline",
    "BenchStats",
    "BenchError",
    "clear_stale_artifacts",
    "publish_results",
]


def _with_target(compile_opt: str, target: str | None) -> str:
    """Append ``--target=<target>`` to a compile-opt string.

    No-op when ``target`` is None or the opt string already names a target —
    a per-spec compile-opt that pins its own target is left untouched.
    `target` None means "don't pass --target": the toolchain picks its own
    default (trn2 today).
    """
    if not target or "--target" in compile_opt:
        return compile_opt
    return f"{compile_opt} --target={target}".strip()


@dataclass(frozen=True)
class BenchStats:
    mean_ms: float | None
    median_ms: float | None
    min_ms: float | None
    max_ms: float | None
    std_dev_ms: float | None
    outputs: tuple[np.ndarray, ...] | None = None

    @classmethod
    def from_executor(
        cls, stats, outputs: tuple[np.ndarray, ...] | None = None
    ) -> "BenchStats":
        return cls(
            mean_ms=stats.mean_ms,
            median_ms=median(stats.durations_ms) if stats.durations_ms else None,
            min_ms=stats.min_ms,
            max_ms=stats.max_ms,
            std_dev_ms=stats.std_dev_ms,
            outputs=outputs,
        )


@dataclass(frozen=True)
class BenchError:
    message: str

    @classmethod
    def from_exception(cls, e: BaseException) -> "BenchError":
        return cls(message=f"{type(e).__name__}: {e}")


BenchResult = BenchStats | BenchError


def _load_kernel_from_path(kernel_path: str, kernel_attr: str):
    # Load the per-variant `<name>__v<i>_t<j>.py` file by absolute path so we
    # don't need the output dir on sys.path. Each variant gets a unique cache key
    # (its absolute path) to prevent collisions across (hw_variant,
    # tile_variant) combos within one process / spawn child. Path separators are
    # squashed to underscores so the key is also a valid module `__name__`: the
    # `CompileKernel` frontend derives a `<module>.<fn>.json` sidecar path from
    # `func.__module__`, and slashes there would point at a nonexistent nested
    # dir.
    cache_key = "_axon_variant_" + kernel_path.replace("/", "_").replace("\\", "_")
    cached = sys.modules.get(cache_key)
    if cached is None:
        loader_spec = importlib.util.spec_from_file_location(cache_key, kernel_path)
        if loader_spec is None or loader_spec.loader is None:
            raise ImportError(f"could not load variant module from {kernel_path}")
        cached = importlib.util.module_from_spec(loader_spec)
        sys.modules[cache_key] = cached
        loader_spec.loader.exec_module(cached)
    return getattr(cached, kernel_attr)


def _init_visible_cores(core_spec) -> None:
    # Spawn-pool initializer for `_bench_one_neff`. Must set
    # NEURON_RT_VISIBLE_CORES BEFORE nkipy/spike is imported in the child;
    # nkipy/spike imports are deferred to the worker function body for this
    # reason. `core_spec` is an int (single-core lnc=1) or a range string like
    # "0-1" (an lnc=2 SPMD launch needs both physical cores visible to one
    # worker).
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_spec)


def _init_compile_worker() -> None:
    """Compile workers never execute NEFFs and must not initialize NRT."""
    os.environ.pop("NEURON_RT_VISIBLE_CORES", None)


def _compile_only_executor(_compiled, _inputs, _outputs) -> None:
    """NKI executor hook: compilation has completed; intentionally do nothing."""


def _bench_one_neff(args) -> BenchStats:
    """Bench one NEFF in a worker. Single packed-tuple arg mirrors
    `_compile_one_nki_kernel`, keeping the call site uniform whether the
    caller uses `pool.map` (one positional) or `submit`. Exceptions
    propagate to the parent via `Future.result()`.

    Drives the NEFF-direct `DeviceKernel.load_from_neff` flow: an emitted
    `@nki.jit` kernel is a `nki.framework.kernel.Kernel` (no `.specialize`, not
    an nkipy traced kernel), so we load the compiled NEFF and run it with spike
    tensors. nkipy/spike imports stay inside the worker body so the spawn-pool
    initializer's NEURON_RT_VISIBLE_CORES pin fires before nkipy/spike load
    (see module docstring). The tile args (TILES_IN_BLOCK_*) are baked into the
    NEFF — NOT runtime inputs — so the NEFF's input tensors are exactly the
    kernel's data params. We map `inputs` (positional, in data-param order) to
    NEFF tensor names *by name*, not by dict order: `dk.input_tensors_info`
    keys are not guaranteed to be in signature order. Data-param names come
    from the emitted kernel's signature (`kernel.func`), dropping the tile
    args."""
    (
        kernel_path,
        kernel_attr,
        tile_values,
        tile_args,
        inputs,
        neff_path,
        warmup,
        bench,
    ) = args

    import inspect

    import ml_dtypes  # noqa: F401  (registers bfloat16 et al. with np.dtype)
    import numpy as np
    from nkipy.runtime import DeviceKernel
    from spike.spike_tensor import SpikeTensor

    kernel = _load_kernel_from_path(kernel_path, kernel_attr)

    sig_params = list(inspect.signature(kernel.func).parameters)
    tile_arg_set = set(tile_args)
    data_param_names = [p for p in sig_params if p not in tile_arg_set]
    if len(data_param_names) != len(inputs):
        raise ValueError(
            f"bench: kernel has {len(data_param_names)} data params "
            f"{data_param_names} but got {len(inputs)} input arrays"
        )

    dk = DeviceKernel.load_from_neff(neff_path)
    input_tensors = {
        name: SpikeTensor.from_numpy(arr, name=name)
        for name, arr in zip(data_param_names, inputs, strict=True)
    }
    output_tensors = {
        name: SpikeTensor.from_numpy(
            np.zeros(ti.shape, dtype=np.dtype(ti.dtype)), name=name
        )
        for name, ti in dk.output_tensors_info.items()
    }
    res = dk.benchmark(
        input_tensors,
        output_tensors,
        warmup_iter=warmup,
        benchmark_iter=bench,
        mode="device",
    )

    # output_tensors_info iterates in reverse insertion order; reorder by the
    # numeric suffix of the codegen output names ("out_0", "out_1", ...) so the
    # positional tuple matches baseline_op's return order. Lexical sort would
    # misorder "out_10" before "out_2"; a single-output kernel is named plain
    # "output" (no suffix -> sorts first, which is correct for one output).
    def _out_index(name: str) -> int:
        _, _, suffix = name.rpartition("_")
        return int(suffix) if suffix.isdigit() else -1

    device_outputs = tuple(
        output_tensors[name].numpy()
        for name in sorted(dk.output_tensors_info, key=_out_index)
    )
    return BenchStats.from_executor(res, outputs=device_outputs)


def _compile_one_nki_kernel(args):
    # Truth is on disk: workers can raise *after* the NEFF lands (nrt_init,
    # post-process dereferences). Classify by file existence + size, not by
    # whether the worker raised. AssertionError = invalid tile config (skip).
    #
    # Compile path: an emitted `@nki.jit` kernel -> CompileKernel with a no-op
    # executor. The recipe writes <artifacts_dir>/kernel.neff, prepares host
    # tensors, then calls our executor without initializing NRT. `neff_path` is
    # the full path to that `kernel.neff`; its parent is the artifacts dir.
    kernel_path, kernel_attr, key, call_args, call_kwargs, neff_path, target, lnc = args

    # NEFF dirs persist across runs, and "NEFF exists" is this worker's
    # success criterion — so a leftover from a previous emission would be
    # reported as THIS compile's product whenever the compile fails or skips.
    # Worse, stale toolchain state (sg00/, penguin.py) in the dir can itself
    # make the recompile fail. Start from an empty artifacts dir: existence of
    # the NEFF after this point can only mean the current source produced it.
    artifacts_dir_path = os.path.dirname(neff_path)
    if os.path.isdir(artifacts_dir_path):
        shutil.rmtree(artifacts_dir_path)
    os.makedirs(artifacts_dir_path, exist_ok=True)
    kernel_obj = _load_kernel_from_path(kernel_path, kernel_attr)

    from nki.framework.compiled import CompileKernel

    artifacts_dir = os.path.dirname(neff_path)
    err = None
    try:
        # kernel_obj is the @nki.jit-wrapped function; `[lnc]` (LncSubscriptable)
        # selects the SPMD program count — `[1]` single-core, `[2]` a 2-core SPMD
        # NEFF (the sharding prototype's launch path).
        ck = kernel_obj[lnc]._to_subclass(
            CompileKernel,
            artifacts_dir=artifacts_dir,
            target=target,
            _executor=_compile_only_executor,
        )
        ck(*call_args, **call_kwargs)
    except AssertionError:
        if not (os.path.exists(neff_path) and os.path.getsize(neff_path) > 0):
            return (key, None, "assert")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    if os.path.exists(neff_path) and os.path.getsize(neff_path) > 0:
        return (key, neff_path, None)
    return (key, None, err or "neff_missing")


def _drop_blank_visible_cores() -> None:
    """Remove NEURON_RT_VISIBLE_CORES when it is set but blank.

    A blank value is not the same as unset to the runtime. NRT rejects it
    (`Invalid configuration: NEURON_RT_VISIBLE_CORES=`) and then reports **zero**
    visible cores, while unset reports all of them. This module's own reader
    treats blank as "no pin requested", so the two disagree and the bench fans
    out over no cores. Deleting the blank entry makes them agree. Call this
    before anything imports or queries spike.
    """
    raw = os.environ.get("NEURON_RT_VISIBLE_CORES")
    if raw is not None and not raw.strip():
        os.environ.pop("NEURON_RT_VISIBLE_CORES", None)


def _external_core_ids() -> list[int] | None:
    """Absolute core ids from a caller-set NEURON_RT_VISIBLE_CORES ("2",
    "4-7", "0,2"), or None when unset. Disjoint per-case pins depend on it."""
    _drop_blank_visible_cores()
    raw = os.environ.get("NEURON_RT_VISIBLE_CORES", "").strip()
    if not raw:
        return None
    ids: list[int] = []
    for part in raw.split(","):
        lo, _, hi = part.partition("-")
        ids.extend(range(int(lo), int(hi or lo) + 1))
    return ids


def _visible_neuron_core_count() -> int:
    # Honors NEURON_RT_VISIBLE_CORES — spike reads it on first init. A blank
    # value would make spike report 0 cores, so drop it first.
    # Returns logical cores post-LNC fusion, which is what we want for
    # one-worker-per-core scheduling.
    _drop_blank_visible_cores()
    try:
        from spike._spike import Spike

        return max(1, int(Spike.get_visible_neuron_core_count()))
    except Exception:
        return 1


def _make_pipeline_pools(
    *, n_compile: int, core_specs: Sequence
) -> tuple[Executor, list[Executor]]:
    """The compile pool plus one single-worker device pool per core spec."""
    # spawn, not fork: each worker must re-import nkipy under its own core pin.
    spawn_ctx = multiprocessing.get_context("spawn")
    compile_pool = ProcessPoolExecutor(
        max_workers=n_compile,
        initializer=_init_compile_worker,
        mp_context=spawn_ctx,
    )
    device_pools: list[Executor] = [
        ProcessPoolExecutor(
            max_workers=1,
            initializer=_init_visible_cores,
            initargs=(core_spec,),
            mp_context=spawn_ctx,
        )
        for core_spec in core_specs
    ]
    return compile_pool, device_pools


def clear_stale_artifacts(paths: RunPaths) -> None:
    """Drop every per-key artifact dir under the run dir. Retention wins."""
    if os.environ.get("AXON_KEEP_NEFF") == "1":
        return
    for d in sorted(paths.run_dir.glob("__v*_tiles_*")):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def _append_csv(df, path):
    if df.empty:
        return
    if not os.path.exists(path):
        df.to_csv(path, index=False)
        return
    existing_cols = list(pd.read_csv(path, nrows=0).columns)
    assert existing_cols == list(df.columns), (
        f"column drift on {path}: existing {existing_cols} != new {list(df.columns)}; "
        f"delete the file to start fresh."
    )
    df.to_csv(path, mode="a", header=False, index=False)


def tile_cols(tile_args: Sequence[str]) -> list[str]:
    """CSV tile-column names. Shared so the bench writer and winner reader
    derive identical column names from one formula."""
    return [
        f"TILES_{a.upper().replace('TILES_IN_BLOCK_', '').replace('TILES_', '')}"
        for a in tile_args
    ]


STATS_COLS = ["mean_ms", "median_ms", "min_ms", "max_ms", "std_dev_ms"]
KEY_PREFIX_COLS = ["lnc", "sharding", "hw_variant", "tile_variant"]


def result_columns(spec: KernelSpec) -> list[str]:
    """Column order of the durable row store (and of the published CSV)."""
    return [
        *spec.dim_vars,
        *KEY_PREFIX_COLS,
        *tile_cols(spec.tile_args),
        *STATS_COLS,
        "correct",
        "max_abs_err",
        "error",
        "mode",
    ]


def key_columns(spec: KernelSpec) -> list[str]:
    """The columns whose values identify one benched key. A terminal row under
    this key means "finished", so a resumed case skips it."""
    return [*KEY_PREFIX_COLS, *tile_cols(spec.tile_args)]


ResultKey = tuple[str, ...]


def read_result_keys(spec: KernelSpec, path: Path) -> set[ResultKey]:
    """Key tuples of every complete record, after truncating a torn final one."""
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    # A record is durable only once its newline is on disk, so drop anything
    # after the last one: that is a torn append from a killed run.
    with path.open("rb+") as f:
        _ = f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":
            _ = f.seek(0)
            _ = f.truncate(f.read().rfind(b"\n") + 1)
    columns = result_columns(spec)
    key_idx = [columns.index(c) for c in key_columns(spec)]
    keys: set[ResultKey] = set()
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return set()
        assert header == columns, (
            f"column drift on {path}: found {header} != expected {columns}; "
            f"delete {path.parent} to start fresh."
        )
        for lineno, record in enumerate(reader, start=2):
            assert len(record) == len(columns), (
                f"malformed record at {path}:{lineno}: {len(record)} fields, "
                f"expected {len(columns)}; delete {path.parent} to start fresh."
            )
            key = tuple(record[i] for i in key_idx)
            assert key not in keys, (
                f"duplicate key {key} at {path}:{lineno}; "
                f"delete {path.parent} to start fresh."
            )
            keys.add(key)
    return keys


def _append_result_row(
    path: Path, columns: Sequence[str], row: dict, artifact_dir: str | None
) -> None:
    """Make one terminal row durable, then reclaim its artifact dir."""
    # Delete only after the append succeeds, or a resume loses row and NEFF both.
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    if is_new:
        writer.writerow(columns)
    # Fold line breaks (error text carries tracebacks) so one record is one
    # physical line, which is what makes the torn-tail rule above decidable.
    writer.writerow(
        [" ".join(str(row[c]).split()) if row[c] is not None else "" for c in columns]
    )
    with path.open("a", newline="") as f:
        _ = f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())
    if artifact_dir and os.environ.get("AXON_KEEP_NEFF") != "1":
        shutil.rmtree(artifact_dir, ignore_errors=True)


def publish_results(
    spec: KernelSpec, paths: RunPaths, expected_keys: set[ResultKey]
) -> None:
    """Publish the durable store to the Make target, but only on exact
    case-level coverage: one terminal row per expected key and nothing else."""
    if not expected_keys:
        return
    have = read_result_keys(spec, paths.results)
    missing = sorted(expected_keys - have)
    unexpected = sorted(have - expected_keys)
    assert not missing and not unexpected, (
        f"{spec.name}: {paths.results} does not cover the run: "
        f"{len(missing)} missing key(s) {missing[:5]}, "
        f"{len(unexpected)} unexpected key(s) {unexpected[:5]}"
    )
    tmp = paths.csv.with_name(paths.csv.name + ".publish.tmp")
    shutil.copyfile(paths.results, tmp)
    os.replace(tmp, paths.csv)
    print(f"\npublished {len(expected_keys)} row(s) to {paths.csv}")


def _ordered_sizes(
    spec: KernelSpec, dim_sizes: dict[str, int]
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Resolve `dim_sizes` into a deterministic ordering for CSV columns."""
    ordered_dims = spec.dim_vars
    ordered_sizes = tuple(dim_sizes[d] for d in ordered_dims)
    return ordered_dims, ordered_sizes


def _stream_compile_bench(
    tasks: Sequence[tuple[tuple[int, ...], tuple, tuple]],
    *,
    n_compile: int,
    core_specs: Sequence,
    max_live: int,
    desc: str,
) -> Iterator[tuple[tuple[int, ...], BenchResult, str]]:
    """Stream `tasks` (each `(key, compile_args, bench_args)`) through compile
    workers into device workers, yielding one terminal triple per task."""
    # Caller must make the row durable before resuming: the yield is the only
    # point at which a key's artifact dir may be reclaimed.
    compile_pool, device_pools = _make_pipeline_pools(
        n_compile=n_compile, core_specs=core_specs
    )
    idle_devices: deque[Executor] = deque(device_pools)
    ready: deque[tuple] = deque()
    compiling: dict[Any, tuple] = {}
    # Each entry also carries the single-worker pool the bench occupies, so it
    # returns to the idle set the moment its future completes.
    benching: dict[Any, tuple[tuple, Executor]] = {}
    task_iter = iter(tasks)
    progress = tqdm(total=len(tasks), desc=desc)

    try:
        while True:
            while len(compiling) + len(ready) + len(benching) < max_live:
                task = next(task_iter, None)
                if task is None:
                    break
                compiling[compile_pool.submit(_compile_one_nki_kernel, task[1])] = task
            while idle_devices and ready:
                task = ready.popleft()
                pool = idle_devices.popleft()
                benching[pool.submit(_bench_one_neff, task[2])] = (task, pool)
            if not compiling and not benching:
                return
            done, _ = wait(set(compiling) | set(benching), return_when=FIRST_COMPLETED)
            for fut in done:
                if fut in compiling:
                    task = compiling.pop(fut)
                    try:
                        _, neff, err = fut.result()
                    except Exception as e:  # worker died / payload unpicklable
                        neff, err = None, f"{type(e).__name__}: {e}"
                    if neff is not None:
                        ready.append(task)
                        continue
                    progress.update(1)
                    yield (
                        task[0],
                        BenchError(message=f"compile: {err or 'neff_missing'}"),
                        os.path.dirname(task[1][5]),
                    )
                else:
                    task, pool = benching.pop(fut)
                    idle_devices.append(pool)
                    try:
                        result: BenchResult = fut.result()
                    except Exception as e:
                        tqdm.write(f"Error benchmarking NKI tiles {task[0]}: {e}")
                        result = BenchError.from_exception(e)
                    progress.update(1)
                    yield task[0], result, os.path.dirname(task[1][5])
    finally:
        progress.close()
        compile_pool.shutdown(wait=False, cancel_futures=True)
        for pool in device_pools:
            pool.shutdown(wait=False, cancel_futures=True)


def _compute_ref_outputs(baseline_op, inputs: tuple) -> tuple[np.ndarray, ...] | None:
    """Run baseline_op in fp32 and normalize to a tuple of fp32 arrays."""
    if baseline_op is None:
        return None
    ref_inputs = [np.asarray(a, dtype=np.float32) for a in inputs]
    out = baseline_op(*ref_inputs)
    arrs = (
        out
        if isinstance(out, tuple)
        else ((out,) if not isinstance(out, list) else tuple(out))
    )
    return tuple(np.asarray(a, dtype=np.float32) for a in arrs)


def run_nki_bench(
    spec: KernelSpec,
    dim_sizes: dict[str, int],
    *,
    hw_variant: int,
    tile_variant: int,
    module_path: str,
    function_name: str,
    warmup: int,
    bench: int,
    paths: RunPaths,
    target: str | None = None,
    dtype: Any | None = None,
    rtol: float,
    atol: float,
    lnc: int = 1,
    sharding: str = "lnc1",
    candidate_filter: bool = True,
    candidate_budget: int = 8,
    candidate_seed: int = SEED,
) -> set[ResultKey]:
    """Compile + bench every tile config of one emitted module, appending each
    terminal row to `paths.results`. Returns this invocation's expected keys."""
    print("\n" + "=" * 80)
    print(
        f"NKI {spec.name.upper()} BENCHMARK (hw_variant={hw_variant}, "
        f"tile_variant={tile_variant}, lnc={lnc}, sharding={sharding})"
    )
    print("=" * 80)

    # Pass dtype only when --dtype was given; else the spec's own default stands.
    inputs = spec.make_inputs(
        **dim_sizes,
        **({"dtype": dtype} if dtype else {}),
        rng=np.random.default_rng(SEED),
    )
    ordered_dims, ordered_sizes = _ordered_sizes(spec, dim_sizes)
    all_keys = list(product(spec.tile_options, repeat=len(spec.tile_args)))
    if candidate_filter:
        constraints = read_manifest(module_path)
        if constraints:
            source_count = len(all_keys)
            legal_keys = [
                key
                for key in all_keys
                if rejection_reason(spec.tile_args, key, constraints) is None
            ]
            all_keys = select_diverse(
                legal_keys, budget=candidate_budget, seed=candidate_seed
            )
            print(
                f"\ncandidate filter: {source_count} source -> "
                f"{len(legal_keys)} legal -> {len(all_keys)} selected "
                f"(budget={candidate_budget or 'unlimited'})"
            )
    tcols = tile_cols(spec.tile_args)
    columns = result_columns(spec)
    key_idx = [columns.index(c) for c in key_columns(spec)]

    def _row(
        tile_values,
        *,
        stats: BenchStats | None = None,
        correct: bool | None = None,
        max_abs_err: float | None = None,
        error: str | None = None,
        mode: str | None = None,
    ) -> dict:
        # Every stats column is present either way: the row writer indexes
        # `columns` directly, so an untimed row needs explicit empty cells.
        timing = (
            {k: v for k, v in asdict(stats).items() if k != "outputs"}
            if stats is not None
            else dict.fromkeys(STATS_COLS)
        )
        return {
            **dict(zip(ordered_dims, ordered_sizes, strict=True)),
            "lnc": lnc,
            "sharding": sharding,
            "hw_variant": hw_variant,
            "tile_variant": tile_variant,
            **dict(zip(tcols, tile_values, strict=True)),
            **timing,
            "correct": correct,
            "max_abs_err": max_abs_err,
            "error": error,
            "mode": mode,
        }

    def _key(tile_values) -> ResultKey:
        # The resume key as it reads back out of the CSV: same columns, same
        # stringification the row writer applies.
        row = _row(tile_values)
        return tuple(str(row[columns[i]]) for i in key_idx)

    expected_keys = {_key(tv) for tv in all_keys}
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    done_keys = read_result_keys(spec, paths.results)
    todo = [tv for tv in all_keys if _key(tv) not in done_keys]
    if todo != all_keys:
        print(
            f"\nresuming: {len(all_keys) - len(todo)}/{len(all_keys)} "
            f"key(s) already terminal"
        )

    def _terminal(tile_values, artifact_dir: str | None, **row_kwargs) -> None:
        _append_result_row(
            paths.results, columns, _row(tile_values, **row_kwargs), artifact_dir
        )

    # n_cores == 1 still goes through the pipeline, just with one device worker.
    # Worth one extra spawn vs. a duplicated bench loop.
    n_cores = _visible_neuron_core_count()
    if lnc > n_cores:
        msg = (
            f"lnc={lnc} requires >= {lnc} visible NeuronCores but only {n_cores} "
            f"detected; skipping {spec.name} {sharding} bench"
        )
        print(f"\n[skip] {msg}")
        clear_stale_artifacts(paths)
        for tv in todo:
            _terminal(tv, None, error=msg)
        return expected_keys

    # Per-variant correctness: compare device output to fp32 baseline.
    ref_outputs = _compute_ref_outputs(spec.baseline_op, inputs)

    def _check(
        outputs: tuple[np.ndarray, ...] | None,
    ) -> tuple[bool | None, float | None]:
        if ref_outputs is None or outputs is None:
            return None, None
        diffs = [
            float(np.abs(np.asarray(d, dtype=np.float32) - r).max())
            for d, r in zip(outputs, ref_outputs, strict=True)
        ]
        max_abs_err = max(diffs)
        correct = all(
            np.allclose(np.asarray(d, dtype=np.float32), r, rtol=rtol, atol=atol)
            for d, r in zip(outputs, ref_outputs, strict=True)
        )
        return correct, max_abs_err

    # A stale artifact dir from an interrupted run is not this run's product,
    # and its NEFF would otherwise be counted against the in-flight cap.
    clear_stale_artifacts(paths)

    # lnc=1 pins one core per device worker; lnc=2 gives one worker a contiguous
    # core *pair* (an SPMD NEFF needs both visible). A caller-set
    # NEURON_RT_VISIBLE_CORES supplies the absolute ids to pin: the worker
    # initializer overwrites the env var, so indexing from 0 would silently
    # unpin externally pinned cases onto cores 0..n-1, breaking the disjoint
    # per-case pins concurrent case runs depend on.
    ids = _external_core_ids() or list(range(n_cores))
    if lnc <= 1:
        core_specs: list = list(ids)
    else:
        core_specs = [
            f"{ids[i]}-{ids[i + lnc - 1]}" for i in range(0, len(ids) - lnc + 1, lnc)
        ] or [f"{ids[0]}-{ids[0] + lnc - 1}"]
    # The in-flight cap, not the key count, is what bounds peak disk.
    max_live = int(os.environ.get("AXON_MAX_LIVE_NEFFS") or 0) or max(
        4, 2 * len(core_specs)
    )
    # neuronx-cc is itself multithreaded, so oversubscribing thrashes small boxes.
    n_compile = int(os.environ.get("AXON_COMPILE_WORKERS") or 0) or max(
        1, multiprocessing.cpu_count() // 2
    )
    print(
        f"\nStreaming {len(todo)} key(s): {n_compile} compile worker(s) "
        f"(AXON_COMPILE_WORKERS) -> {len(core_specs)} device worker(s), "
        f"<= {max_live} live NEFF dir(s) (AXON_MAX_LIVE_NEFFS)"
    )

    # Both worker payloads are known up front: the NEFF path is derived from the
    # key, and the compile worker writes exactly that path.
    tasks: list[tuple[tuple[int, ...], tuple, tuple]] = []
    for tv in todo:
        neff = str(
            paths.neff(
                hw_variant, tile_variant, tv, plan_tag=sharding if lnc > 1 else ""
            )
        )
        tile_kwargs = dict(zip(spec.tile_args, tv, strict=True))
        tasks.append(
            (
                tv,
                (
                    module_path,
                    function_name,
                    tv,
                    tuple(inputs),
                    tile_kwargs,
                    neff,
                    target,
                    lnc,
                ),
                (
                    module_path,
                    function_name,
                    tv,
                    tuple(spec.tile_args),
                    inputs,
                    neff,
                    warmup,
                    bench,
                ),
            )
        )

    n_failed = 0
    n_incorrect = 0
    err_counts: Counter[str] = Counter()
    for tv, result, artifact_dir in _stream_compile_bench(
        tasks,
        n_compile=n_compile,
        core_specs=core_specs,
        max_live=max_live,
        desc=f"NKI {spec.name} v{hw_variant}t{tile_variant}",
    ):
        match result:
            case BenchError(message=msg):
                _terminal(tv, artifact_dir, error=msg)
                err_counts[msg.split(":")[0]] += 1
                n_failed += 1
            case BenchStats(outputs=outputs):
                correct, max_abs_err = _check(outputs)
                n_incorrect += not correct if correct is not None else 0
                _terminal(
                    tv,
                    artifact_dir,
                    stats=result,
                    correct=correct,
                    max_abs_err=max_abs_err,
                )

    if n_failed:
        print(
            f"\n[warn] {n_failed}/{len(todo)} key(s) for {spec.name} failed; "
            f"see `error` column in {paths.results}"
        )
        for tag, count in err_counts.most_common():
            print(f"  {count:4d}  {tag}")
    if n_incorrect:
        print(
            f"\n[warn] {n_incorrect}/{len(todo)} bench runs for {spec.name} "
            f"failed correctness (rtol={rtol}, atol={atol})"
        )
    print(f"\n{spec.name} rows appended to {paths.results}")
    return expected_keys


def run_baseline(
    spec: KernelSpec,
    dim_sizes: dict[str, int],
    *,
    warmup: int,
    bench: int,
    paths: RunPaths,
    target: str | None = None,
    dtype: Any | None = None,
) -> str:
    if spec.baseline_op is None:
        raise ValueError(f"kernel '{spec.name}' has no baseline_op set")

    from nkipy.core.compile import compile_to_neff, trace
    from nkipy.runtime import BaremetalExecutor, CompiledKernel

    print("\n" + "=" * 80)
    print(f"NEURONPY BASELINE {spec.name.upper()}")
    print("=" * 80)

    ordered_dims, ordered_sizes = _ordered_sizes(spec, dim_sizes)
    size_dir = paths.baseline_dir
    size_dir.mkdir(parents=True, exist_ok=True)

    # Pass dtype only when --dtype was given; else the spec's own default stands.
    inputs = spec.make_inputs(
        **dim_sizes,
        **({"dtype": dtype} if dtype else {}),
        rng=np.random.default_rng(SEED),
    )
    traced = trace(spec.baseline_op)
    traced.specialize(*inputs)
    neff_path = compile_to_neff(
        trace_kernel=traced,
        output_dir=str(size_dir),
        additional_compiler_args=_with_target(spec.baseline_compiler_args, target),
    )

    kernel = CompiledKernel(traced, neff_path)
    with BaremetalExecutor(verbose=0) as executor:
        print(f"\nRunning NeuronPy {spec.name} baseline...")
        stats = executor.benchmark(
            kernel,
            *inputs,
            warmup_iterations=warmup,
            benchmark_iterations=bench,
        )
    row = {
        **{
            k: v
            for k, v in asdict(BenchStats.from_executor(stats)).items()
            if k != "outputs"
        },
        **dict(zip(ordered_dims, ordered_sizes, strict=True)),
    }
    csv_path = str(paths.baseline_csv)
    df = pd.DataFrame([row])
    print(f"\nBaseline {spec.name} Results:")
    print(df.to_string(index=False))
    _append_csv(df, csv_path)
    print(f"\n{spec.name} baseline appended to {csv_path}")
    return csv_path
