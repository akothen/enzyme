from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ml_dtypes  # noqa: F401 — registers bfloat16 so np.dtype("bfloat16") resolves
import numpy as np

from axon.bench_runner import (
    ResultKey,
    clear_stale_artifacts,
    publish_results,
    run_baseline,
    run_nki_bench,
)
from axon.codegen import (
    EmitErr,
    EmitOk,
    emit_nki_code_variants,
    nki_safe_var,
    print_graph,
)
from axon.codegen.assemble import emit_nki_code_lnc2_variants
from axon.codegen.combine_emit import SpmdEmitError
from axon.egraph.persist import (
    CacheError,
    build_cache_file,
    cache_file_path,
    load_cache,
    save_cache,
)
from axon.egraph.pipeline import (
    DEFAULT_SYNTHESIS_WALL_CLOCK_SECONDS,
    LOWERING_TIMEOUT_MS,
    EGraphSearch,
    build_egraph_search,
    iter_hw_graphs_from_cache,
    iter_hw_graphs_from_search,
)
from axon.egraph.workers import resolve_worker_count
from axon.ir import build_graph_from_kernel, nuGraph
from axon.isa_semantics import _start_kernel_synthesis_cache
from axon.kernel_spec import KernelSpec
from axon.paths import RunPaths, _fs_safe, run_paths
from axon.sharding import ShardingPlan, shardings
from axon.sharding_cost import prune_plans
from axon.winner import export_winner, pick_winner

DTYPE_CHOICES: tuple[str, ...] = ("bfloat16", "float16", "float32")

# Permissive default tolerances for bench-only cases (no per-case rtol/atol):
# the correctness check catches gross codegen miscompiles, not tol drift.
_DEFAULT_RTOL = 2e-2
_DEFAULT_ATOL = 2e-2
_DEFAULT_SOLVER_TIMEOUT_MS = 3000
_DEFAULT_CACHE_DIR = "cache"
# Matches the saturation drivers' own default, so the flags below change nothing
# unless they are passed.
_DEFAULT_SATURATION_ROUNDS = 50


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def _positive_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite value greater than zero")
    return parsed


def _spec_default_dtype_name(spec: KernelSpec) -> str:
    """The dtype name baked into make_inputs' `dtype=` default, for when a
    head-to-head case names no dtype and no `--dtype` is given."""
    param = inspect.signature(spec.make_inputs).parameters.get("dtype")
    if param is None or param.default is inspect.Parameter.empty:
        return "float32"
    return np.dtype(param.default).name


def _load_spec(path: str) -> KernelSpec:
    """Load a kernel spec from `kernels/<name>.py` or a `kernels/<name>/`
    package directory (an `__init__.py` that assembles `SPEC`, typically from
    sibling `kernel.py` / `refs.py` / `inputs.py` modules)."""
    p = Path(path).expanduser().resolve()
    search_locations: list[str] | None = None
    if p.is_dir():
        init = p / "__init__.py"
        if not init.is_file():
            print(f"kernel-spec dir has no __init__.py: {p}", file=sys.stderr)
            sys.exit(2)
        # Package load: __path__ makes the package's relative imports
        # (`from .refs import ...`) resolve against the directory.
        search_locations = [str(p)]
        module_name = f"_axon_kernel_spec_{nki_safe_var(p.name)}"
        target = init
    elif p.is_file():
        module_name = f"_axon_kernel_spec_{nki_safe_var(p.stem)}"
        target = p
    else:
        print(f"kernel-spec file not found: {p}", file=sys.stderr)
        sys.exit(2)
    spec = importlib.util.spec_from_file_location(
        module_name, target, submodule_search_locations=search_locations
    )
    if spec is None or spec.loader is None:
        print(f"could not load kernel-spec from {p}", file=sys.stderr)
        sys.exit(2)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    obj = getattr(module, "SPEC", None)
    if not isinstance(obj, KernelSpec):
        print(f"{p} does not define a top-level `SPEC: KernelSpec`", file=sys.stderr)
        sys.exit(2)
    return obj


# Extraction streams every acyclic selection of the saturated ISA e-graph, a
# product that is astronomically large for kernels like fused_adam. The caller
# owns the decision to stop the lazy iterator, so the CLI takes a bounded prefix.
DEFAULT_MAX_HW_GRAPHS = 256


class _BoundedStream(Iterator[nuGraph]):
    """The first ``limit`` graphs of a stream, forwarding its live ``outcome``."""

    def __init__(self, source: Iterator[nuGraph], limit: int) -> None:
        self._source = source
        self._cursor = iter(source)
        self._limit = limit
        self._count = 0
        self._source_exhausted = False
        self._surplus_seen = False

    def __iter__(self) -> _BoundedStream:
        return self

    @property
    def outcome(self) -> Any:
        return getattr(self._source, "outcome", None)

    @property
    def truncated_by_limit(self) -> bool:
        """True only once the source is known to hold a graph past the limit."""
        return self._surplus_seen

    def __next__(self) -> nuGraph:
        if self._surplus_seen or self._source_exhausted:
            raise StopIteration
        try:
            graph = next(self._cursor)
        except StopIteration:
            self._source_exhausted = True
            raise
        if self._count >= self._limit:
            # One pull past the limit tells a drained source from a truncated
            # one. The surplus graph is discarded, never yielded downstream.
            self._surplus_seen = True
            raise StopIteration
        self._count += 1
        return graph


def _synthesis_status_message(outcome: Any, *, no_graphs: bool) -> str | None:
    if outcome is None:
        return None
    if no_graphs:
        stage = (
            getattr(outcome, "extraction_stage", None)
            or outcome.truncated_stage
            or "extraction"
        )
        reason = (
            outcome.extraction_exhaustion
            or outcome.stop_reason
            or "no hw graphs synthesized"
        )
    elif outcome.status != "completed":
        stage = outcome.truncated_stage or "unknown"
        reason = outcome.stop_reason or "unknown"
    else:
        return f"synthesis status: {outcome.status}"
    return f"synthesis status: {outcome.status}; stage={stage}; reason={reason}"


def _report_synthesis_result(
    spec_name: str,
    outcome: Any,
    *,
    consumed: int,
    emitted: int,
) -> int | None:
    status = _synthesis_status_message(outcome, no_graphs=consumed == 0)
    if consumed == 0:
        failure = status or (
            f"synthesis failed for {spec_name}: no hw graphs synthesized"
        )
        print(f"  [{failure}]")
        return 1
    if status is not None:
        print(f"--- {status} ---")
    print(
        f"--- {spec_name} :: consumed {consumed} synthesized hw graph(s), "
        f"emitted {emitted} module(s) ---"
    )
    if emitted > 0:
        return None
    print(f"  [emission failed for {spec_name}: no modules emitted]")
    return 1


def _save_search_cache(
    path: Path,
    search: EGraphSearch,
    *,
    kernel_name: str,
    dim_sizes: dict[str, int],
    graph_identity: str,
    run_stem: str,
    options: dict[str, Any],
) -> None:
    """Save a finished search immediately. Cache failure does not fail the run."""
    try:
        cache = build_cache_file(
            search,
            kernel_name=kernel_name,
            dim_sizes=dim_sizes,
            graph_identity=graph_identity,
            run_stem=run_stem,
            options=options,
        )
        save_cache(path, cache)
        size_mb = path.stat().st_size / (1024 * 1024)
    except Exception as exc:
        print(f"  [e-graph cache not saved to {path}: {exc}]", file=sys.stderr)
        return
    print(f"  [saved e-graph cache to {path} ({size_mb:.1f} MiB)]")


def _write_synth_time_sidecar(
    path: Path,
    *,
    stem: str,
    kernel: str,
    sizes: str,
    dtype: str,
    synth_wall_s: float,
    emit_wall_s: float,
    consumed_graphs: int,
    emitted_modules: int,
    status: str,
) -> None:
    """Record one synth/emit timing row next to the cache. Never fails the run."""
    try:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "stem",
                    "kernel",
                    "sizes",
                    "dtype",
                    "synth_wall_s",
                    "emit_wall_s",
                    "consumed_graphs",
                    "emitted_modules",
                    "status",
                ]
            )
            writer.writerow(
                [
                    stem,
                    kernel,
                    sizes,
                    dtype,
                    f"{synth_wall_s:.3f}",
                    f"{emit_wall_s:.3f}",
                    consumed_graphs,
                    emitted_modules,
                    status,
                ]
            )
    except Exception as exc:
        print(f"  [synth-time sidecar not written to {path}: {exc}]", file=sys.stderr)


def _bench_existing_modules(
    spec: KernelSpec,
    dim_sizes: dict[str, int],
    paths: RunPaths,
    function_name: str,
    *,
    warmup: int,
    bench: int,
    target: str | None,
    dtype: Any | None,
    rtol: float,
    atol: float,
    candidate_filter: bool,
    candidate_budget: int,
) -> set[ResultKey]:
    """Benchmark the variant modules already present in the run directory."""
    modules = sorted(paths.run_dir.glob(f"{function_name}__v*_t*.py"))
    if not modules:
        print(
            f"--phase bench: no emitted variant modules under "
            f"{paths.run_dir}; run --phase emit first",
            file=sys.stderr,
        )
        sys.exit(2)

    paths.csv.unlink(missing_ok=True)
    expected: set[ResultKey] = set()
    for module_path in modules:
        stem_tail = module_path.stem.rsplit("__v", 1)[1]
        i_s, j_s = stem_tail.split("_t", 1)
        expected |= run_nki_bench(
            spec,
            dim_sizes,
            hw_variant=int(i_s),
            tile_variant=int(j_s),
            module_path=str(module_path),
            function_name=function_name,
            warmup=warmup,
            bench=bench,
            target=target,
            dtype=dtype,
            paths=paths,
            rtol=rtol,
            atol=atol,
            candidate_filter=candidate_filter,
            candidate_budget=candidate_budget,
        )
    return expected


def _load_or_synthesize_graphs(
    G0: nuGraph,
    spec_name: str,
    dim_sizes: dict[str, int],
    paths: RunPaths,
    *,
    cache_dir: str,
    from_cache: bool,
    workers: int,
    solver_timeout_ms: int,
    lowering_timeout_ms: int,
    synthesis_timeout_seconds: float,
    tensor_rounds: int,
    isa_rounds: int,
) -> tuple[Iterator[nuGraph], float | None]:
    """Load cached hardware graphs or synthesize and cache a fresh search.

    The second tuple element is the synthesis wall in seconds, or None on a
    --from-cache replay where no synthesis happened.
    """
    run_stem = paths.csv.stem
    cache_path = cache_file_path(Path(cache_dir), spec_name, dim_sizes, run_stem)
    if from_cache:
        try:
            cache = load_cache(cache_path)
            cache.require_matches(
                kernel_name=spec_name,
                dim_sizes=dim_sizes,
                graph_identity=G0.identity(),
                run_stem=run_stem,
                path=cache_path,
            )
        except CacheError as exc:
            print(f"--from-cache: {exc}", file=sys.stderr)
            sys.exit(2)
        print(
            f"--- {spec_name} :: replaying e-graphs from {cache_path} "
            f"(no synthesis; cached options: {cache.describe_options()}; "
            f"stages: {cache.describe_stages()}) ---"
        )
        return iter_hw_graphs_from_cache(cache, workers=workers), None

    print(f"--- {spec_name} :: e-graph synthesis ---")
    synthesis_options = {
        "max_hw_size": 2,
        "timeout": solver_timeout_ms,
        "lowering_timeout_ms": lowering_timeout_ms,
        "wall_clock_seconds": synthesis_timeout_seconds,
        "tensor_max_rounds": tensor_rounds,
        "isa_max_rounds": isa_rounds,
    }
    synth_start = time.perf_counter()
    search = build_egraph_search(
        G0,
        **synthesis_options,
        workers=workers,
    )
    synth_wall_s = time.perf_counter() - synth_start
    _save_search_cache(
        cache_path,
        search,
        kernel_name=spec_name,
        dim_sizes=dim_sizes,
        graph_identity=G0.identity(),
        run_stem=run_stem,
        options={**synthesis_options, "workers": workers},
    )
    return iter_hw_graphs_from_search(search, workers=workers), synth_wall_s


def _publish_and_finish(
    spec: KernelSpec,
    paths: RunPaths,
    expected: set[ResultKey],
    *,
    pick_by: str,
    rtol: float,
    atol: float,
    kernel_path: str | None,
    case_id: str | None,
    case_dtype: str | None,
) -> None:
    """Publish results, clear stale artifacts, and export the winner."""
    publish_results(spec, paths, expected)
    clear_stale_artifacts(paths)
    _finish_run(
        spec,
        paths,
        pick_by=pick_by,
        rtol=rtol,
        atol=atol,
        kernel_path=kernel_path,
        case_id=case_id,
        case_dtype=case_dtype,
    )


def trace_kernel(
    spec: KernelSpec,
    dim_sizes: dict[str, int],
    *,
    warmup: int,
    bench: int,
    target: str | None = None,
    dtype: Any | None = None,
    out: str | None = None,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
    kernel_path: str | None = None,
    case_id: str | None = None,
    case_dtype: str | None = None,
    pick_by: str = "median",
    lnc: int = 1,
    phase: str = "all",
    solver_timeout_ms: int = _DEFAULT_SOLVER_TIMEOUT_MS,
    lowering_timeout_ms: int = LOWERING_TIMEOUT_MS,
    synthesis_timeout_seconds: float = DEFAULT_SYNTHESIS_WALL_CLOCK_SECONDS,
    synth_workers: int | None = None,
    tensor_rounds: int = _DEFAULT_SATURATION_ROUNDS,
    isa_rounds: int = _DEFAULT_SATURATION_ROUNDS,
    cache_dir: str = _DEFAULT_CACHE_DIR,
    from_cache: bool = False,
    max_hw_graphs: int | None = DEFAULT_MAX_HW_GRAPHS,
    candidate_filter: bool = True,
    candidate_budget: int = 8,
) -> int | None:
    """Run one kernel through emit, bench, or both, optionally replaying a cache."""
    synth_workers = resolve_worker_count(synth_workers)
    print("\n" + "=" * 80)
    print(f"Tracing kernel: {spec.name} (phase={phase})")
    print("=" * 80)
    paths = run_paths(spec, out)
    fn_name_early = nki_safe_var(spec.name)

    if phase == "emit" and lnc >= 2:
        # The SPMD sweep interleaves plan lowering with bench dispatch; a
        # split emit is not supported for it (yet). Fail loud, not silent.
        print("--phase emit does not support --lnc 2", file=sys.stderr)
        sys.exit(2)

    if phase == "bench":
        expected = _bench_existing_modules(
            spec,
            dim_sizes,
            paths,
            fn_name_early,
            warmup=warmup,
            bench=bench,
            target=target,
            dtype=dtype,
            rtol=rtol,
            atol=atol,
            candidate_filter=candidate_filter,
            candidate_budget=candidate_budget,
        )
        _publish_and_finish(
            spec,
            paths,
            expected,
            pick_by=pick_by,
            rtol=rtol,
            atol=atol,
            kernel_path=kernel_path,
            case_id=case_id,
            case_dtype=case_dtype,
        )
        return

    # Both paths reset ids because lnc=2 plan tags embed them even though graph
    # identity does not.
    _start_kernel_synthesis_cache(kernel_name=spec.name, verbose=not from_cache)
    G0 = build_graph_from_kernel(
        spec.axon_kernel,
        *spec.input_specs,
        dim_sizes=dim_sizes,
    )
    print(f"=== {spec.name} :: Original graph ===")
    print_graph(G0)
    print()

    graphs, synth_wall_s = _load_or_synthesize_graphs(
        G0,
        spec.name,
        dim_sizes,
        paths,
        cache_dir=cache_dir,
        from_cache=from_cache,
        workers=synth_workers,
        solver_timeout_ms=solver_timeout_ms,
        lowering_timeout_ms=lowering_timeout_ms,
        synthesis_timeout_seconds=synthesis_timeout_seconds,
        tensor_rounds=tensor_rounds,
        isa_rounds=isa_rounds,
    )
    if max_hw_graphs is not None:
        print(
            f"--- {spec.name} :: consuming at most {max_hw_graphs} hardware "
            f"graph(s) of the extraction stream ---"
        )
        graphs = _BoundedStream(graphs, max_hw_graphs)

    print(f"--- {spec.name} :: NKI code emission (lnc={lnc}) ---")
    paths.csv.unlink(missing_ok=True)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    fn_name = nki_safe_var(spec.name)
    # Clear the previous run's modules alongside its CSV: what the run dir holds
    # is then this run's output, so a refused shape leaves it empty.
    for stale in paths.run_dir.glob(f"{fn_name}__v*_t*.py"):
        stale.unlink()
    # ``lnc`` selects either the single-core or sharded sweep. The extraction
    # stream is lazy, so consuming it here is the emit wall (extraction + codegen).
    emit_start = time.perf_counter()
    if lnc >= 2:
        consumed, emitted, expected = _trace_lnc2(
            spec,
            dim_sizes,
            G0,
            graphs,
            fn_name,
            paths=paths,
            lnc=lnc,
            warmup=warmup,
            bench=bench,
            target=target,
            dtype=dtype,
            rtol=rtol,
            atol=atol,
            candidate_filter=candidate_filter,
            candidate_budget=candidate_budget,
        )
    else:
        consumed, emitted, expected = _trace_single(
            spec,
            dim_sizes,
            graphs,
            fn_name,
            paths=paths,
            warmup=warmup,
            bench=bench,
            target=target,
            dtype=dtype,
            rtol=rtol,
            atol=atol,
            bench_variants=(phase != "emit"),
            candidate_filter=candidate_filter,
            candidate_budget=candidate_budget,
        )
    emit_wall_s = time.perf_counter() - emit_start

    outcome = getattr(graphs, "outcome", None)
    if synth_wall_s is not None:
        # Real synthesis run (not --from-cache): drop a timing sidecar next to
        # the cache bin. A failure here must not fail the run.
        sizes = "_".join(f"{dim}{size}" for dim, size in dim_sizes.items())
        dtype_name = case_dtype or (np.dtype(dtype).name if dtype is not None else "")
        sidecar = cache_file_path(
            Path(cache_dir), spec.name, dim_sizes, paths.csv.stem
        ).with_suffix(".time.csv")
        _write_synth_time_sidecar(
            sidecar,
            stem=paths.csv.stem,
            kernel=spec.name,
            sizes=sizes,
            dtype=dtype_name,
            synth_wall_s=synth_wall_s,
            emit_wall_s=emit_wall_s,
            consumed_graphs=consumed,
            emitted_modules=emitted,
            status=getattr(outcome, "status", "") or "",
        )

    if getattr(graphs, "truncated_by_limit", False):
        print(
            f"--- {spec.name} :: stopped at the --max-hw-graphs limit "
            f"({max_hw_graphs}); more variants remain in the enumeration ---"
        )
    synthesis_failure = _report_synthesis_result(
        spec.name,
        outcome,
        consumed=consumed,
        emitted=emitted,
    )
    if synthesis_failure is not None:
        return synthesis_failure

    if phase == "emit":
        print(
            f"\n--phase emit done: {emitted} variant module(s) under "
            f"{paths.run_dir}. Bench them on a device with the same "
            "case args plus --phase bench."
        )
        return

    _publish_and_finish(
        spec,
        paths,
        expected,
        pick_by=pick_by,
        rtol=rtol,
        atol=atol,
        kernel_path=kernel_path,
        case_id=case_id,
        case_dtype=case_dtype,
    )


def _finish_run(
    spec: KernelSpec,
    paths: RunPaths,
    *,
    pick_by: str,
    rtol: float,
    atol: float,
    kernel_path: str | None,
    case_id: str | None,
    case_dtype: str | None,
) -> None:
    """Pick the winner and export head-to-head artifacts when requested."""
    winner = pick_winner(paths.csv, spec, pick_by, rtol=rtol, atol=atol)  # type: ignore[arg-type]
    assert winner is not None, f"{spec.name}: no timed variants in {paths.csv}"

    if case_id is not None:
        if spec.torch_ref is None:
            raise ValueError(
                f"head-to-head case {spec.name}_{case_id} requires spec.torch_ref"
            )
        if kernel_path is None or case_dtype is None:
            raise ValueError("head-to-head export needs kernel_path and case_dtype")
        export_winner(
            spec,
            winner,
            case_id=case_id,
            kernel_path=kernel_path,
            paths=paths,
            dtype=case_dtype,
            rtol=rtol,
            atol=atol,
        )


def _trace_single(
    spec: KernelSpec,
    dim_sizes: dict[str, int],
    graphs: Iterator[nuGraph],
    fn_name: str,
    *,
    paths: RunPaths,
    warmup: int,
    bench: int,
    target: str | None,
    dtype: Any | None,
    rtol: float,
    atol: float,
    bench_variants: bool = True,
    candidate_filter: bool = True,
    candidate_budget: int = 8,
) -> tuple[int, int, set[ResultKey]]:
    """Emit and optionally benchmark the single-core variant sweep."""
    consumed = 0
    emitted = 0
    expected: set[ResultKey] = set()
    for i, g_hw in enumerate(graphs):
        consumed += 1
        # The list-based emitter keys results by list position; the stream
        # index ``i`` names the variant, so a one-element list is safe.
        for result in emit_nki_code_variants([g_hw], spec.name):
            match result:
                case EmitOk(_, j, code):
                    module_path = paths.variant_module(fn_name, i, j)
                    module_path.write_text(code)
                    emitted += 1
                    # The module file IS the output; don't dump its source to
                    # stdout too (a full sweep prints thousands of lines).
                    print(
                        f"  +-- NKI hw variant {i}, tile config {j}: "
                        f"wrote {module_path}"
                    )
                    if not bench_variants:
                        continue
                    expected |= run_nki_bench(
                        spec,
                        dim_sizes,
                        hw_variant=i,
                        tile_variant=j,
                        module_path=str(module_path),
                        function_name=fn_name,
                        warmup=warmup,
                        bench=bench,
                        target=target,
                        dtype=dtype,
                        paths=paths,
                        rtol=rtol,
                        atol=atol,
                        candidate_filter=candidate_filter,
                        candidate_budget=candidate_budget,
                    )
                case EmitErr(_, exc):
                    print(f"  +-- NKI hw variant {i} [EMISSION FAILED: {exc}] ---")
    return consumed, emitted, expected


def _trace_lnc2(
    spec: KernelSpec,
    dim_sizes: dict[str, int],
    G0: nuGraph,
    graphs: Iterator[nuGraph],
    fn_name: str,
    *,
    paths: RunPaths,
    lnc: int,
    warmup: int,
    bench: int,
    target: str | None,
    dtype: Any | None,
    rtol: float,
    atol: float,
    candidate_filter: bool = True,
    candidate_budget: int = 8,
) -> tuple[int, int, set[ResultKey]]:
    """Emit and benchmark the SPMD hardware and sharding product."""
    expected: set[ResultKey] = set()
    plans: list[ShardingPlan] = list(shardings(G0))
    kept = prune_plans(plans, G0, dim_sizes)
    print(
        f"--- {spec.name} :: SPMD emission (lnc={lnc}): {len(kept)}/{len(plans)} "
        f"shardings after prune ---"
    )
    consumed = 0
    emitted = 0
    for i, g_hw in enumerate(graphs):
        consumed += 1
        for plan in kept:
            for result in emit_nki_code_lnc2_variants([g_hw], spec.name, plan, G0):
                match result:
                    case EmitOk(_, j, code):
                        plan_tag = _fs_safe(plan.tag())
                        module_path = paths.spmd_variant_module(fn_name, i, j, plan_tag)
                        module_path.write_text(code)
                        emitted += 1
                        print(
                            f"  +-- SPMD [{plan.tag()}] hw variant {i}, tile {j}: "
                            f"wrote {module_path}"
                        )
                        expected |= run_nki_bench(
                            spec,
                            dim_sizes,
                            hw_variant=i,
                            tile_variant=j,
                            module_path=str(module_path),
                            function_name=fn_name,
                            warmup=warmup,
                            bench=bench,
                            target=target,
                            dtype=dtype,
                            paths=paths,
                            rtol=rtol,
                            atol=atol,
                            lnc=lnc,
                            sharding=plan.tag(),
                            candidate_filter=candidate_filter,
                            candidate_budget=candidate_budget,
                        )
                    case EmitErr(_, exc):
                        kind = "skipped" if isinstance(exc, SpmdEmitError) else "FAILED"
                        print(f"  +-- SPMD [{plan.tag()}] v{i} {kind}: {exc}")
    return consumed, emitted, expected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Axon synthesis + NKI emission for a kernel-spec file.",
    )
    parser.add_argument(
        "kernel", help="Path to a kernel-spec .py file with a top-level SPEC."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        required=True,
        metavar="N",
        help=(
            "Concrete dimension sizes (required) as bare ints, one per symbolic "
            "dim the spec declares, bound positionally to the spec's dim_vars in "
            "order (e.g. for dim_vars (m, n, k): `--sizes 1024 512 16384` binds "
            "m=1024 n=512 k=16384)."
        ),
    )
    parser.add_argument(
        "--target",
        default="trn2",
        help=(
            "neuronx-cc compile target (e.g. trn2, trn3, trn3pre). Defaults to "
            "trn2 — the real bench/eval target — so compile-only runs on a "
            "no-device box match it instead of cross-compiling to the host's "
            "trn1 (which rejects trn2-only codegen)."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_CHOICES,
        default=None,
        help=(
            "Override the input dtype produced by the spec's make_inputs. "
            "Omit to use the dtype baked into the spec (the default). dtype is "
            "a pure make_inputs concern for these memory-bound, matmul-free "
            "kernels."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help=(
            "Run stem: out/<stem>.csv is the per-case CSV and out/<stem>/ the "
            "run dir (per-variant modules, NEFFs; baseline dir alongside). The "
            ".csv suffix is optional — `--out out/matmul_sq1k` and `--out "
            "out/matmul_sq1k.csv` are the same run. A trailing slash names a "
            "container dir for the default CSV. Defaults to out/nki_<name>. "
            "Split phases (--phase emit/bench) must pass the same --out."
        ),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--bench", type=int, default=100)
    parser.add_argument(
        "--synthesis-timeout-seconds",
        type=_positive_finite_float,
        default=DEFAULT_SYNTHESIS_WALL_CLOCK_SECONDS,
        metavar="SECONDS",
        help=(
            "Shared wall-clock limit for the complete e-graph synthesis "
            f"pipeline (default: {DEFAULT_SYNTHESIS_WALL_CLOCK_SECONDS:g}, one hour)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("variants", "baseline", "both"),
        default="variants",
        help=(
            "What to benchmark: synthesized NKI 'variants' (default), the "
            "NeuronPy 'baseline', or 'both'. The baseline requires Trainium "
            "hardware."
        ),
    )
    parser.add_argument(
        "--pick-by",
        choices=("median", "mean", "min"),
        default="median",
        help="Timing column used to sort variants when picking the winner.",
    )
    parser.add_argument(
        "--phase",
        choices=("emit", "bench", "all"),
        default="all",
        help=(
            "Which half of the pipeline to run. 'emit' is pure host work: "
            "synthesize + prove + emit the per-variant NKI modules under the "
            "run dir, no Neuron deps, no device. 'bench' skips synthesis, "
            "discovers the emitted modules, and compiles/benches/picks the "
            "winner on the device (invoke with the SAME kernel/--sizes/--out). "
            "'all' (default) is the single-shot legacy loop."
        ),
    )
    parser.add_argument(
        "--lnc",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "Logical NeuronCore count, selecting exactly one target. 1 (default) "
            "emits/benches the single-core sweep only. 2 emits/benches the sharded "
            "lnc=2 V × Σ sweep only (the sharding search × hw variants); it does "
            "not also run the lnc=1 sweep. Comparing lnc=1 vs lnc=2 is an "
            "eval-layer concern (run the case at each --lnc and diff the CSVs)."
        ),
    )
    parser.add_argument(
        "--synth-workers",
        dest="synth_workers",
        type=_positive_int,
        default=None,
        metavar="N",
        help=(
            "Worker processes used for independent proof obligations during "
            "synthesis. Defaults automatically from the available CPU count; "
            "set N to bound it. Does not affect compilation or device workers."
        ),
    )
    parser.add_argument(
        "--max-hw-graphs",
        dest="max_hw_graphs",
        type=_nonnegative_int,
        default=DEFAULT_MAX_HW_GRAPHS,
        metavar="N",
        help=(
            "How many hardware graphs to consume from the extraction stream "
            f"(default: {DEFAULT_MAX_HW_GRAPHS}). The synthesis wall bounds "
            "saturation only; extraction then enumerates every acyclic selection "
            "of the saturated e-graph, a product that is effectively unbounded "
            "for kernels with many equivalent ISA alternatives. 0 removes the "
            "cap and consumes the whole enumeration, which may not terminate."
        ),
    )
    parser.add_argument(
        "--candidate-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply exact generated tile constraints before compilation "
            "(default: enabled). Use --no-candidate-filter to benchmark the "
            "full tile-option product."
        ),
    )
    parser.add_argument(
        "--candidate-budget",
        type=_nonnegative_int,
        default=8,
        metavar="N",
        help=(
            "Maximum legal tile configurations selected per emitted schedule "
            "family when filtering is enabled (default: 8; 0 means unlimited)."
        ),
    )
    parser.add_argument(
        "--tensor-rounds",
        dest="tensor_rounds",
        type=_positive_int,
        default=_DEFAULT_SATURATION_ROUNDS,
        metavar="N",
        help=(
            "Cap tensor saturation at N rounds, bounding a search that would "
            f"otherwise run to its wall (default: {_DEFAULT_SATURATION_ROUNDS})."
        ),
    )
    parser.add_argument(
        "--isa-rounds",
        dest="isa_rounds",
        type=_positive_int,
        default=_DEFAULT_SATURATION_ROUNDS,
        metavar="N",
        help="Cap ISA saturation at N rounds. See --tensor-rounds "
        f"(default: {_DEFAULT_SATURATION_ROUNDS}).",
    )
    parser.add_argument(
        "--from-cache",
        dest="from_cache",
        action="store_true",
        help=(
            "Replay this case's saved e-graphs instead of synthesizing, to test "
            "codegen cheaply. Missing or stale is an error, not a fallback."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        dest="cache_dir",
        default=_DEFAULT_CACHE_DIR,
        metavar="PATH",
        help="Where per-case e-graph caches are written and read from "
        f"(default: {_DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help=(
            "Relative tolerance for the on-device correctness check. Together "
            "with --atol, marks a head-to-head case: the winner + per-case "
            "nkilib harness are emitted, named from the --out stem "
            "(out/<kernel>_<case_id>.csv -> out/winners/<kernel>_<case_id>"
            "{,_case}.py). Omit for a bench-only run "
            f"(permissive default rtol={_DEFAULT_RTOL})."
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help=(
            "Absolute tolerance for the on-device correctness check. See "
            f"--rtol. Omit for a bench-only run (default atol={_DEFAULT_ATOL})."
        ),
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    spec = _load_spec(args.kernel)
    if len(args.sizes) != len(spec.dim_vars):
        print(
            f"{spec.name}: --sizes expects {len(spec.dim_vars)} values for dims "
            f"{list(spec.dim_vars)} in order, got {len(args.sizes)}",
            file=sys.stderr,
        )
        sys.exit(2)
    dim_sizes = dict(zip(spec.dim_vars, args.sizes, strict=True))

    if args.from_cache and args.phase == "bench":
        print(
            "--from-cache does not apply to --phase bench: it discovers the "
            "modules --phase emit wrote and never synthesizes",
            file=sys.stderr,
        )
        sys.exit(2)

    paths = run_paths(spec, args.out)

    dtype = np.dtype(args.dtype) if args.dtype else None
    sim_rtol = args.rtol if args.rtol is not None else _DEFAULT_RTOL
    sim_atol = args.atol if args.atol is not None else _DEFAULT_ATOL

    # Both tolerances given marks a head-to-head case; its id is the --out stem
    # minus the `<kernel>_` prefix (out/<kernel>_<case_id>.csv).
    case_id = case_dtype = None
    status = 0
    if args.rtol is not None and args.atol is not None:
        prefix = f"{spec.name}_"
        if not paths.csv.stem.startswith(prefix):
            print(
                f"{spec.name}: head-to-head run needs an --out stem of the form "
                f"'{spec.name}_<case_id>.csv', got {paths.csv.stem!r}",
                file=sys.stderr,
            )
            sys.exit(2)
        case_id = paths.csv.stem[len(prefix) :]
        case_dtype = args.dtype or _spec_default_dtype_name(spec)

    if args.mode in ("variants", "both"):
        variant_status = trace_kernel(
            spec,
            dim_sizes,
            warmup=args.warmup,
            bench=args.bench,
            target=args.target,
            dtype=dtype,
            out=args.out,
            rtol=sim_rtol,
            atol=sim_atol,
            kernel_path=str(Path(args.kernel).expanduser().resolve()),
            case_id=case_id,
            case_dtype=case_dtype,
            pick_by=args.pick_by,
            lnc=args.lnc,
            phase=args.phase,
            synthesis_timeout_seconds=args.synthesis_timeout_seconds,
            synth_workers=args.synth_workers,
            tensor_rounds=args.tensor_rounds,
            isa_rounds=args.isa_rounds,
            cache_dir=args.cache_dir,
            from_cache=args.from_cache,
            max_hw_graphs=args.max_hw_graphs or None,
            candidate_filter=args.candidate_filter,
            candidate_budget=args.candidate_budget,
        )
        if variant_status is not None:
            status = variant_status
    if args.mode in ("baseline", "both"):
        if spec.baseline_op is None:
            print(f"kernel '{spec.name}' has no baseline_op set", file=sys.stderr)
            return 2
        paths.baseline_csv.unlink(missing_ok=True)
        run_baseline(
            spec,
            dim_sizes,
            warmup=args.warmup,
            bench=args.bench,
            target=args.target,
            dtype=dtype,
            paths=paths,
        )
    return status


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    sys.exit(_run(args))


if __name__ == "__main__":
    main()
