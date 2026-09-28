# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## What this is

Axon is a synthesis-based superoptimizer that lowers a high-level tensor IR to NKI kernels for AWS Trainium. The Python entry point is `axon.cli:main` (installed as the `axon` script via `pyproject.toml`). The package lives at `src/axon/`; `kernels/` at the repo root holds per-kernel spec packages (`kernels/<name>/`).

## Environment

- Python 3.11 only (`requires-python = "==3.11.*"`). Dependencies are managed by `uv`; use `uv sync` and `uv run`, not `pip` / bare `python`.
- Heavy Neuron deps (`neuronx-cc`, `nki`, `torch-neuronx`) come from `pip.repos.neuron.amazonaws.com`; `nkipy` and `spike` come from upstream `aws-neuron/nkipy` (see `[tool.uv.sources]`).
- The local dev desktop has **no Neuron device**. A full `axon` run compiles **and** benches on a device, so it needs Trainium — but the pipeline splits at the host/device boundary: `--phase emit` (synthesize + emit, pure host, no Neuron deps) runs on the desktop, and only `--phase bench` (compile + bench + winner) needs the box. Use the `run-on-remote` skill, which reads `.remote.toml` (gitignored, copied from `.claude/skills/run-on-remote/.remote.toml.example`), syncs the workspace, and runs commands on a remote Trainium host. The bundled driver is `tools/remote.py` (stdlib-only, not on PATH; invoke as `python3 tools/remote.py ...`).
- The Axon remote loop is three commands: `python3 tools/remote.py sync` (push the workspace), `python3 tools/remote.py task make -- <kernel>_<case_id>` (synth + tile sweep + bench + correctness on-device), `python3 tools/remote.py sync --pull` (copy the `out/` tree back). Conventions: **drive a declared case through the `make` task, not bare `axon`** (the `make` target reads the case from `eval/sizes.json` and writes the canonical stem; a bare `axon kernels/<name> --sizes ...` run is fine only for an ad-hoc one-off, see "Cases + sizes.json + Makefile"); **no quotes in `task`/`run` args** (the driver word-splits, so `remote task make -- mul_bw_1k`, not a quoted string). The default host is `trn2`; select `trn3` (pre-production silicon, a different arch rev) with `-c trn3`, which must compile for `--target trn3pre` (see `--target` below). Switching a build's arch leaves a stale NEFF that loads on the old arch and fails on the new one: delete `out/<stem>/` on the building box first.

## Common commands

Run from `axon/` (this directory). `axon` has **no subcommands**: `axon
kernels/<name> --sizes <ints>` is the whole interface.

```
# Synthesize + bench one kernel-spec file. --sizes is required: bare ints bound
# positionally to the spec's dim_vars. --out names the CSV and derives the run
# stem. mul's dim_vars are (m, n), so this binds m=1024 n=1024.
uv run axon kernels/mul --sizes 1024 1024 --out out/mul_bw_1k.csv

# matmul's dim_vars are (m, n, k), so these bind m=1024 n=512 k=16384.
uv run axon kernels/matmul --sizes 1024 512 16384

# Compile for a specific Neuron target (defaults to trn2).
# Needed on other hardware: a trn3pre box rejects a trn2-targeted NEFF at load.
uv run axon kernels/matmul --sizes 1024 1024 1024 --target trn3pre

# Override the spec's input dtype (defaults to whatever make_inputs bakes in).
# Choices: bfloat16, float16, float32.
uv run axon kernels/matmul --sizes 1024 1024 1024 --dtype float32

# Bench the NeuronPy baseline instead of (or alongside) the NKI variants.
# `--mode baseline` runs only the baseline; `both` runs variants then baseline.
# The baseline REQUIRES Trainium hardware.
uv run axon kernels/matmul --sizes 1024 1024 1024 --mode baseline

# Pick the winner by a different timing column (default median).
uv run axon kernels/mul --sizes 1024 1024 --out out/mul_bw_1k.csv --pick-by min

# Emit + bench the lnc=2 (dual-core SPMD) sharding sweep instead of single-core.
# Enumerates sharding plans, prunes by cost, emits each through the SPMD assembler.
uv run axon kernels/matmul --sizes 1024 1024 16384 --lnc 2

# Split the pipeline at the host/device boundary. `--phase emit` is pure host
# work (synthesize + prove + emit the per-variant modules; no Neuron deps, no
# device — runs on the dev desktop). `--phase bench` skips synthesis,
# discovers the emitted modules under the run dir, and compiles + benches +
# picks the winner on the device. Run both with the SAME kernel/--sizes/--out.
uv run axon kernels/matmul --sizes 1024 1024 1024 --out out/matmul_sq1k --phase emit
uv run axon kernels/matmul --sizes 1024 1024 1024 --out out/matmul_sq1k --phase bench

# Or drive a declared case split-by-default with the case driver: emit HERE
# (host), sync, bench on the box — one command. `--emit-on box` opts back into
# the single-shot legacy loop on the box; lnc=2 cases always take the box path.
python3 tools/run_case.py matmul_sq1k
python3 tools/run_case.py fused_adam_opt_1m --emit-on box
```

`--sizes` is **required** on every run as bare ints (`type=int, nargs="+"`),
one per dim the spec declares, bound **positionally** to the spec's `dim_vars`
in order (e.g. for matmul's `dim_vars=(m, n, k)`, `--sizes 1024 512 16384` binds
m=1024 n=512 k=16384). The dim names live in the spec, not on the CLI. A wrong
count exits 2 naming the expected dims in order. There is no implicit default
shape.

`--out PATH` is the CSV path; the run stem (NEFF dir, per-variant modules,
baseline dir) all derive from it, defaulting to `out/nki_<name>.csv`. `--dtype`
(choices `bfloat16`, `float16`, `float32`) overrides the spec's `make_inputs`
dtype. `--mode {variants,baseline,both}` selects what to bench. `--pick-by
{median,mean,min}` selects the timing column used to pick the head-to-head
winner.

`--target` defaults to `trn2`, the real bench/eval target, so a run on a
trn3pre box must pass `--target trn3pre` (which otherwise rejects a trn2-targeted
NEFF at load). It applies to both the NKI variants and the `--mode baseline`
path. A per-spec `KernelSpec.baseline_compiler_args` that already names a
`--target` wins over the flag (for the baseline path).

`--lnc {1,2}` selects exactly one target. `1` (default) emits/benches the
single-core sweep. `2` runs the sharding search (enumerate plans from the
coloring generator, prune by cost, emit each surviving plan through the SPMD
assembler). It does NOT also run lnc=1; comparing the two is an eval concern.

`--candidate-filter` is enabled by default. Codegen embeds structured
`TILES_IN_BLOCK_*` constraints in every emitted module, so either a same-process
run or a later `--phase bench` can reject impossible block sizes before
compilation. `--no-candidate-filter` restores the full `tile_options` product;
`--candidate-budget` limits legal tiles per emitted schedule family (default 8,
0 unlimited). Missing metadata fails open.

`--phase {emit,bench,all}` splits one case at the host/device boundary. `emit`
runs stages 1-5 (lift, prove, synthesize, emit) and writes the per-variant
modules under `out/<stem>/`, then stops — pure host work, so it runs on the
dev desktop. `bench` skips synthesis entirely: it discovers the emitted
`<name>__v<i>_t<j>.py` modules in the run dir (exit 2 if none — run emit
first), then compiles, benches, correctness-checks, and picks the winner on
the device. `all` (default) is the single-shot legacy loop. The two split
phases must be invoked with the same kernel, `--sizes`, and `--out` so the
run-stem paths line up (the emitted modules are how the phases hand off;
`remote sync` carries them to the box). `--phase emit` rejects `--lnc 2` (the
SPMD sweep interleaves plan lowering with bench dispatch and cannot split yet).

`tools/run_case.py <kernel>_<case_id>` is the split-by-default case driver:
it reads the case from `eval/sizes.json` (same source of truth as the
Makefile generator), runs `--phase emit` locally, `remote sync`s the emitted
modules (the `.remote.toml` include `out/*/*.py` whitelists them through the
`out/***` exclude; NEFFs/CSVs stay excluded), and runs `--phase bench` on the
box, then `remote sync --pull` to retrieve the winner and results.
`--emit-on box` opts back into one single-shot remote run; cases
declaring `lnc: 2` always fall back to the box path. Note `make
<kernel>_<case_id>` (and `remote task make -- ...`) still runs single-shot on
whichever machine invokes it — prefer run_case.py from the desktop so the box
CPU is not spent on synthesis. `--timings <csv>` records one row per step
(`step,seconds,returncode,command`) and writes it even when a step fails, so a
partial case still reports where its time went. `emit` is synthesis; `bench` is
compile plus benchmark on the box.

`agent_opt/` holds the agentic post-process, which takes one Axon winner and
rewrites it. `agent_opt/optimize.py --case out/winners/<stem>_case.py` measures
the baseline through nkilib's harness, generates the optimization hint from the
measured profile, then rewrites, verifies, and profiles each iteration on
device. `agent_opt/run_combined.py <kernel>_<case_id>` is the joiner: it runs
`tools/run_case.py` (so filtering applies), then `optimize.py` on the winner it
produced, and writes `combined_result.json` plus a merged `phase_timings.csv`.
Keep the split of concerns: filtering lives in `axon.candidate_filter` and
`run_case.py`, and `optimize.py` knows nothing about it. `run_combined.py` runs
under the invoking interpreter, so it reads a kernel spec through `uv run` in a
subprocess rather than importing `axon` itself. See `agent_opt/README.md`.

The Makefile is data-driven from `eval/sizes.json` (see `Cases + sizes.json +
Makefile` below): `make <kernel>_<case_id>` runs one declared case (caching on
`out/<stem>.csv`), `make all` runs every case, a bare `make <kernel>` prints
that kernel's case targets and exits 2, and `make clean-<kernel>` wipes its
per-case artifacts.


Linting / type-checking dev tools are declared in `pyproject.toml` `[dependency-groups].dev` at **exact pins** (`ruff`, `basedpyright`); a floating pin is not a gate, because ruff 0.16 widened its default rule set from 2 findings on this tree to 197. **Run `make ci` before you commit** — it is the one definition of green (format check, lint, and the whole host suite, ~55 s), and the tracked git hooks in `scripts/hooks/` call the same targets (`make hooks` enables them; `make test-fast` is the ~2 s subset the commit hook uses). `make typecheck` is not in `ci` yet: 8 pre-existing errors. `make host-env` builds a 1.3 GB environment that runs the whole suite without the Neuron wheels, and a device-only test must carry `@pytest.mark.device`, which `ci` deselects. Host-only tests live in `tests/` and run with `uv run python -m pytest tests/` (no device required). They cover the sharding algebra, monoid laws, and SPMD codegen source assertions.

## Architecture (pipeline)

The compiler is a five-stage pipeline; each stage is a single module (or package) under `src/axon/`:

1. **Frontend IR (`axon.ir`)** defines `AxonArray` (a NumPy-ish tensor builder), `Node`, `nuGraph`, and shape annotation. `build_graph_from_kernel(kernel, *input_specs, dim_sizes=...)` lifts a Python `kernel` (a function on `AxonArray`s) into a `nuGraph`. Per-kernel spec files in `kernels/` call this via the dispatcher; symbolic dim names that repeat across inputs (e.g. `k` shared between `x=(m, k)` and `w=(k, n)`) are unified. `axon/__init__.py` re-exports `AxonArray` and `build_graph_from_kernel`, so kernel packages use `from axon import AxonArray` in their `kernel.py`.

2. **E-graph search (`axon.egraph`)** holds the high-level graph and every ISA lowering of it in one e-graph, so rewrites are saturated in place instead of enumerated as whole-graph variants. `build_egraph_search` constructs the search; saturation applies the ISA axioms (`egraph/isa.py`, `egraph/propagation.py`, `egraph/fusion.py`); `egraph/extraction.py` streams hardware graphs out of the saturated e-graph; `egraph/proof.py` and `egraph/proof_parallel.py` prove a lowering equivalent to its source; `egraph/persist.py` saves a completed search under `cache/` so a later run replays it instead of re-searching. Extraction, lowering, and the whole search each stop on a direct wall-clock check.

3. **Synthesis (`axon.synthesizer`)** lowers the high-level `nuGraph` to hardware graphs of NeuronCore ISA ops (`ISA_POOL_OP_NAMES`, keyed off `_POOL_TEMPLATE_REGISTRY`: `nc_matmul`, `activation`, `tensor_tensor`, `tensor_reduce`, `dma_*`, …), driving the e-graph search above. `axon.isa_semantics` defines the symbolic semantics of those ISA ops (`SymTensor`, `ShapeExpr`, `SymExpr`) and the locks its module globals need: `_Z3_LOCK` serializes z3 formula construction (`z3.main_ctx` is not thread-safe), while the uninterpreted-function caches (`_BINARY_UFS`, `_UNARY_UFS`, `_COMPARE_UFS`) are filled under `_UF_LOCK` and `_SEMANTICS` under `_SEMANTICS_LOCK` (`_FOLD_FAMILIES` is filled at import time). `axon.lang_semantics` defines high-level op semantics (broadcast/reduce shape rules) used by both the frontend and the equivalence checker.

4. **Sharding (`axon.sharding`, `axon.sharding_monoids`, `axon.sharding_cost`)** enumerates LNC=2 SPMD plans for the high-level graph G0. A *coloring* fixes one active axis class per graph segment; PROPAGATE derives every node's output label (Replicated, Sharded, or Partial) and ASSEMBLE reads cross-core combines off label mismatches. The cost model prunes to top-K plans by traffic vs sharded extent. Three combine patterns are codegen-emittable: P (free-axis barrier), C (contraction all-reduce), S-moment (accumulator all-reduce + column gather). The monoid library (`sharding_monoids`) is the extension seam for adding new reduction combines. This stage is pure host code with no device or synthesizer dependency; `--lnc 2` activates it.

5. **NKI codegen (`axon.codegen`)** turns each synthesized hardware-graph variant into NKI Python source. It is a pure package: no disk writes, no subprocesses, no knowledge of bench dispatch. `_NL_OP_MAP` maps internal op names (`add`, `mul`, `relu`, …) to `nl.*` calls. `emit_nki_code_variants` is a generator yielding `EmitOk(variant_index, tile_index, code) | EmitErr(variant_index, error)` per emitted variant, and the caller pattern-matches. `emit_nki_code_lnc2_variants` is the SPMD counterpart: it takes a `ShardingPlan` and emits each (hw variant, tile config) through `emit_lnc2`, which splices the plan's shard prelude (program id, operand slices) and combine (barrier / all-reduce / gather) around the shared single-core body. `combine_emit.py` holds the NKI line-templates; `combine_plan.py` holds the plan analysis that selects which template to use.

   After benchmarking, each variant's **on-device output** is compared to the spec's `baseline_op` (computed in fp32) at the case tolerances (`rtol`/`atol`). The correctness verdict (`correct: bool`, `max_abs_err: float`) is recorded per-variant in the CSV. Winner selection picks the fastest timed variant that is not verified incorrect and reports how many faster-but-wrong variants it skipped; the run fails loudly only when every timed variant is verified incorrect, so a fast-but-wrong variant is never silently shipped.

Per-kernel data lives in `kernels/<name>/` package directories, each exposing a top-level `SPEC: KernelSpec` (see `axon.kernel_spec`) from its `__init__.py`. The package splits by consumer runtime: `kernel.py` holds the Axon math (imports `axon`); `refs.py` holds the numpy `baseline_op`, the torch `torch_ref`, and the rng-driven `make_inputs` and is **axon/nkipy-free** (numpy/torch/ml_dtypes only) because nkilib's harness loads it directly; `__init__.py` assembles `SPEC` (dim_vars, input specs, tile-arg names, bench knobs). The problem shape is passed at the CLI via the required `--sizes`. `axon` loads one spec by path with `importlib.util.spec_from_file_location` (a directory loads as a package via `submodule_search_locations`; a legacy single-file `kernels/<name>.py` still loads). There is no in-tree registry.

The orchestrating loop ("synthesize, then for each emitted variant: write file, compile, bench + correctness-check; then assert the picked winner is correct") is `trace_kernel` in `axon.cli`. Synthesis runs once in the parent process. Each emitted `(hw_variant, tile_variant)` is written under the run stem (`out/<stem>/<name>__v<i>_t<j>.py`) and the dispatcher calls `bench_runner.run_nki_bench` **in-process** with the variant's file path; `run_nki_bench` benchmarks the NEFF and then compares the on-device output to the fp32 baseline, recording `correct`/`max_abs_err` in the CSV. After all variants are benched, `winner.pick_winner` reads the CSV, sorts by the chosen timing column, and picks the fastest row that is not verified incorrect, reporting how many faster-but-wrong rows it skipped; it raises only when every timed row is verified wrong. Per-variant modules are loaded by absolute path via `importlib.util.spec_from_file_location` (no `sys.path` mutation). No subprocess wrapping; no canonical `kernel_<name>.py` file. `axon --mode baseline` (or `--mode both`) calls `bench_runner.run_baseline` instead, which compiles the spec's `baseline_op` via `nkipy.core.compile.trace`. The run paths (CSV, run dir, per-variant module, NEFF, baseline dir) are all derived from one stem by `axon.paths.run_paths`, the single source of truth.

`axon.bench_runner` parallel-compiles the tile-config sweep with `nki.framework.compiled.CompileKernel`. That recipe writes `<artifacts_dir>/kernel.neff` **before** its post-compile execute step fails on a compiler worker with no device, so the worker catches the exception and classifies the attempt by NEFF presence instead of claiming a device. It then benchmarks each NEFF through `DeviceKernel.load_from_neff` and **appends per-leaf rows** to the durable per-key store `out/<stem>/results.csv`; `publish_results` copies that store to the Make target `out/<stem>.csv` only once every expected key has a terminal row. The bench phase auto-detects visible NeuronCores via `spike._spike.Spike.get_visible_neuron_core_count()` and fans tile configs out across them, one single-worker `ProcessPoolExecutor` per core, each pinned via `NEURON_RT_VISIBLE_CORES` set in the pool initializer before nkipy is imported. With one core (or detection failure) it falls back to one device worker. nkipy/spike imports stay inside function bodies in `bench_runner.py` so spawn workers' initializer fires before nkipy is loaded. Bench always runs (there is no compile-only path). Per-variant source files are regenerated on every synthesis run.

After benching, if the case carries `rtol`/`atol` in `eval/sizes.json` (a head-to-head case), `trace_kernel`'s tail (`axon.winner.export_winner`) picks the winner row from the CSV (by `--pick-by`), copies the winning variant module to `out/winners/<kernel>_<case_id>.py`, and generates a self-describing per-case nkilib harness `out/winners/<kernel>_<case_id>_case.py` (via `axon.case_gen`). That module loads the spec, wraps `spec.torch_ref` + `spec.make_inputs` + `spec.baseline_op`, and carries the case run-metadata in a `CASE` dict (`name`, `entry_point`, `kernel_rel`, `shape_args`, `rtol`, `atol`); it points at the sibling winner kernel by relative path. There is **no aggregate manifest**: nkilib discovers a case by being pointed at one `_case.py` module (one benchmark) or the `out/winners/` directory (the full set), imports it, reads `CASE`, and compiles the kernel from source. Bench-only cases (no `rtol`/`atol`) and empty synthesis emit no winner/case module.

**`winners/` is the deliverable; `out/` is scratch.** `out/winners/<stem>.py` is a *template*: it keeps the `TILES_IN_BLOCK_*` parameters (the sweep gives one module a different tile config per row) and the winning values sit in a second file, `<stem>_case.py`, under `CASE["shape_args"]`. `tools/specialize_winner.py` turns that pair into a standalone kernel at the tracked top-level `winners/<kernel>_<case_id>.py`, binding the winning tile values as constants in the body so the module takes only tensor inputs. The Makefile runs it after every head-to-head bench (see `eval/gen_size_targets.py`), so `make <kernel>_<case_id>` refreshes the deliverable; lnc=2 cases are excluded, because `winners/` holds single-core kernels only. Specializing is not a performance change — `bench_runner` already passes the tile args to `CompileKernel` as compile-time kwargs, so they are baked into the NEFF and the compiler saw the same constants beforehand (measured spec/param 0.9986 on `rmsnorm_h1024`, 1.0050 on `attention_nkilib_d128`, outputs bit-identical). Two rules hold: an artifact enters `winners/` only with a **measured** latency (a case with no correct benched row in `out/<stem>.csv` is refused), and a re-run overwrites only when it measures **faster** (a slower or tied re-run prints `KEEP`; `--force` overrides). Each file records its own provenance in the docstring: shape, tile config, measured median, the **device** it ran on from `neuron-ls` (instance type, core layout, lnc), the axon commit (`AXON_COMMIT` when git is unavailable, as on a synced remote box), and absolute paths back to the run. `tools/verify_specialized.py <stem>` compares the specialized and parameterized forms on device; note NEFF bytes are not a valid equivalence test, since one unchanged source compiled twice already yields a different NEFF, so it compares `kernel_info.json` and the outputs instead.

All axon output goes under `<cwd>/out/`, one self-contained directory per case keyed by the stem `<kernel>_<case_id>` (derived from `--out`): the per-case CSV `out/<stem>.csv`, per-variant modules + NEFFs under `out/<stem>/` (`out/<stem>/<name>__v<i>_t<j>.py` and `out/<stem>/__v<i>_t<j>_tiles_<tile_tag>/kernel.neff`, no `nki_` prefix and no size tag), baseline NEFFs under `out/baseline_<stem>/`, and head-to-head winners under `out/winners/` (the winner kernel `out/winners/<stem>.py` plus its self-describing per-case nkilib harness `out/winners/<stem>_case.py`). There is no aggregate manifest. The path is resolved at call time from `Path.cwd()`, so always invoke `axon` from the repo root. The `out/` tree is gitignored.

## Cases + sizes.json + Makefile

`eval/sizes.json` declares the per-kernel workload **cases**: a map of `<kernel> -> {<case_id>: {sizes, dtype?, lnc?, rtol?, atol?}}`. `sizes` is the int tuple in `dim_vars` order (the same order `--sizes` binds positionally). `dtype` is optional and passed through as `--dtype` (it distinguishes same-shape different-dtype cases). The **presence** of `rtol`/`atol` marks a case as a head-to-head point: it gets a generated per-case `_case.py` harness, and the correctness check uses those tolerances; a case without them is bench-only (the generic 1024-cube synthesis kernels) and uses a permissive default. A leading `_comment` key documents the schema; readers skip `_`-prefixed keys at both levels. A case *is* the complete eval point.

The `Makefile` is data-driven via `eval/gen_size_targets.py`, which emits `out/size_targets.mk` (gitignored; `make` auto-remakes it from `sizes.json` + the generator before reading). For each case it produces a **file target** `out/<kernel>_<case_id>.csv` (recipe runs `axon ... --sizes ... --out $@`, prereqs are the spec + `sizes.json`, so `make` caches when the CSV is newer than both), a phony alias `<kernel>_<case_id>`, a per-kernel `CASES_<kernel>`, and the aggregate `CASE_TARGETS`. `make all` builds every `CASE_TARGETS`; a bare `make <kernel>` prints that kernel's case targets and exits 2; `make clean-<kernel>` loops over `CASES_<kernel>` (not a `out/<kernel>_*` glob, which would wipe a sibling kernel sharing a `_`-prefix) wiping each case's `out/<case>.csv`, `out/<case>/`, `out/baseline_<case>/`, and `out/winners/<case>.py` + `_case.py`. It does **not** touch the tracked `winners/` deliverable, which is deliberate: that file records its own measurement and is replaced only by a faster measured run.

## Adding a new kernel

One package directory, three files. The split is by **consumer runtime**:
`refs.py` is loaded directly (standalone, by path) by nkilib's harness, whose
runtime has no `axon`/`nkipy` — so it must import only numpy/torch/ml_dtypes,
and must not use relative imports. `kernel.py` is axon-side only.

Create `kernels/foo/kernel.py` — the Axon math:

```python
from axon import AxonArray


def kernel_foo(x: AxonArray, w: AxonArray) -> AxonArray:
    return x @ w
```

Create `kernels/foo/refs.py` — references + inputs (axon-free):

```python
import numpy as np
from ml_dtypes import bfloat16  # NOT nkipy: same numpy bf16 dtype, no nkipy dep


# Numpy reference: powers the on-device correctness check (and `--mode baseline`
# timing). Required for any head-to-head case. The correctness check upcasts
# inputs to fp32 before calling this (so it need not upcast itself), matching
# the fp32-reference convention.
def baseline_op(x, w):
    return np.matmul(x, w)


# Torch reference: the head-to-head reference math, PURE MATH ONLY. axon's
# case_gen wraps it into the per-case module nkilib consumes and coerces every
# input to an fp32 torch tensor BEFORE calling it (`_to_torch_f32`), then casts
# the output to the case dtype after. So no torch.as_tensor / astype boilerplate
# here — inputs arrive as fp32 torch tensors. (fp32 keeps the reference from
# adding bf16 intermediate-rounding error the kernel does not incur.)
def torch_ref(x, w):
    return x @ w


# rng-driven, seeded by the caller: axon always calls make_inputs by name
# (make_inputs(**dim_sizes, dtype=..., rng=...)). Dim params are keyword-only so
# a positional signature can't silently bind dims in the wrong order; `rng` is a
# required np.random.Generator the bench and correctness check both seed identically.
def make_inputs(*, m, n, k, dtype=bfloat16, rng):
    return (rng.standard_normal((m, k)).astype(dtype),
            rng.standard_normal((k, n)).astype(dtype))
```

Create `kernels/foo/__init__.py` — assemble the SPEC:

```python
from axon.kernel_spec import KernelSpec

from .kernel import kernel_foo
from .refs import baseline_op, make_inputs, torch_ref

SPEC = KernelSpec(
    name="foo",
    axon_kernel=kernel_foo,
    # dim_vars is the ordered canonical dim namespace: each name is a
    # `make_inputs` keyword param, a `--sizes` position, and a CSV column. Order
    # is significant (it is `make_inputs` keyword order). Shared dims (the
    # contraction `k` here) use the same name across input_specs so they unify.
    dim_vars=("m", "n", "k"),
    input_specs=(("x", ("m", "k")), ("w", ("k", "n"))),
    baseline_op=baseline_op,
    torch_ref=torch_ref,
    make_inputs=make_inputs,
    tile_args=("TILES_IN_BLOCK_M", "TILES_IN_BLOCK_N", "TILES_IN_BLOCK_K"),
)
```

Then `uv run axon kernels/foo --sizes 1024 1024 1024` (`--sizes` is required: bare ints bound positionally to `dim_vars`). To run it as a declared case, add an entry under `foo` in `eval/sizes.json` and `make foo_<case_id>`. The kernel package can live anywhere; Axon takes a path, not a registry key. Bench-only kernels (no head-to-head case) omit `torch_ref`. See `kernels/mul/` or `kernels/matmul/` for live templates.

## Conventions worth knowing

- `KernelSpec.name` is the short, no-`kernel_`-prefix name. It is the kernel half of every per-case stem (`<kernel>_<case_id>`): the run CSV `out/<stem>.csv`, the default CSV `out/nki_<name>.csv`, the `CASE["name"]` in the per-case harness, and the per-variant filename stem (`<name>__v<i>_t<j>.py`). The synthesized NKI function inside each variant module is always named `nki_safe_var(spec.name)` (i.e. it matches `spec.name` after the safe-filename transform).
- All axon output goes under `Path.cwd() / "out"`. Run `axon` from the repo root so this resolves to the gitignored `out/`.
- Z3 calls are not thread-safe, so anything touching the solver must hold `_Z3_LOCK` (imported from `axon.isa_semantics` as `_Z3_LOCK` by both `axon.ir` and `axon.synthesizer`).
- Shapes flow as both concrete `tuple[int, ...]` and symbolic (`SymTensor` / `ShapeExpr`). `spec.dim_vars` is the single canonical, ordered dim namespace: each name is a `make_inputs` keyword param, a `--sizes` position (positional `--sizes` binds to `dim_vars` in order), a `sizes.json` case-tuple position, and a CSV size column. `__post_init__` validates that `set(dim_vars)` equals the set of string labels in `input_specs` (a disagreement fails at spec load). `cli._run` builds `dim_sizes = dict(zip(spec.dim_vars, args.sizes))` after a count check, and `build_graph_from_kernel` in `axon.ir` binds each label to its concrete size from that **required** `dim_sizes` map. `dim_sizes` must be total over every label a spec declares; a missing binding raises `ValueError` (a missing size is a spec bug, never an invented value, since the concrete size feeds trace-time `.shape` constant folds like RMSNorm's `1.0 / x.shape[1]`). Symbolic identity for the equivalence checker is carried separately via each input's `sym_shape` (a fresh `z3.Int` per name), not via these concrete ints, so distinct names may share a size (a cube) without losing distinctness.
- Synthesis caches are global and process-wide, and nothing clears them between kernels. `_start_kernel_synthesis_cache(...)` resets the node and reduction identity counters at each kernel boundary; respect that boundary when adding instrumentation.
- **The proof does not model operand ranges, so a hardware instruction with a limited domain must stay out of the lowering pool.** `_activation_pool_templates` in `axon.synthesizer` deliberately omits `nl.reciprocal`: the Activation engine's reciprocal returns exactly 0.0 at and above 1e14 on trn2, while `isa_semantics` models it as 1/x for every nonzero real (x = 0 maps to 0) and attaches no range assumption, unlike `exp`. Axon proves over the reals, and the simulator returns the exact value, so only a device run disagrees; a softmax denominator reaches that range and the kernel silently produces zeros. `1/x` therefore lowers to the Vector-engine `nisa.reciprocal`, which is correct over the whole range. Apply the same rule to any instruction whose accuracy depends on the value, not only on the shape or dtype.
