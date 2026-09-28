# Axon

Axon is a synthesis-based compiler for Trainium. It lowers a high-level
tensor IR to NKI kernels and benches them on Neuron hardware.

## Prerequisites

- Python 3.11 (pinned via `.python-version`).
- [`uv`](https://docs.astral.sh/uv/) for dependency and environment management.
- A Neuron development host with Trainium hardware for the device leg. A
  full `axon` run compiles **and** benches on a device, but the pipeline
  splits: `--phase emit` (synthesize + emit NKI modules) is pure host work
  and runs on the dev desktop; `--phase bench` (compile + bench + winner)
  needs Trainium — route it through the `run-on-remote` skill (it syncs the
  repo, including the emitted modules, to a Trainium host and runs there).

Then, use the following commands to set up and activate the venv:

```
uv sync
source .venv/bin/activate
```

All commands below assume the venv is active.

## Checks before you commit

This repository has no server-side CI, so the checks are local and you enable
them once per clone:

```
make hooks        # point git at scripts/hooks (pre-commit + pre-push)
```

Three layers, all defined in the `Makefile` so a hook and a human run the same
thing:

| command | what it checks | cost |
|---|---|---|
| `make test-fast` | 142 pure-logic tests, no solver or simulation | ~2 s |
| `make lint` | `ruff format --check` + `ruff check` | ~1 s |
| `make ci` | `make lint` plus the whole host suite (810 tests) | ~55 s |

`pre-commit` runs `make lint` over the **staged** Python files plus
`make test-fast`. `pre-push` runs `make ci`. Both are bypassable with
`--no-verify`, and a hook only exists in a clone that ran `make hooks`, so treat
them as a habit rather than a guarantee.

`make typecheck` (`basedpyright src`) is deliberately **not** in `make ci` yet: it
reports 8 pre-existing errors, mostly `Optional` narrowing in `codegen/plan.py`
and `codegen/bodies/matmul_generic.py`. Fix those, then fold it in.

The gate tools are pinned exactly in `[dependency-groups].dev`. That is not
fussiness: ruff 0.16 widened its default rule set, and a floating `>=` pin turns
2 findings on this tree into 197, so an unrelated ruff release would break every
commit.

### Running the tests without the Neuron toolchain

`uv sync` installs `neuronx-cc`, `torch-neuronx`, `nkipy`, and `spike` — about
9.5 GB — because they are hard dependencies of the package. **No test needs
them.** The suite needs `nki` only for `nki.simulate` (a CPU simulator) and
`torch` only for the kernels' torch references, so:

```
make host-env     # builds .venv-host, ~1.3 GB, no device
make test         # all 810 tests pass in it
```

The `Makefile` picks `.venv-host` up automatically when it exists. This is also
what a GitHub Actions job would install, if the repo ever gets one: about 2.5
minutes for the suite on two cores.

### Tests that need a device

Anything requiring Trainium is marked `@pytest.mark.device` and deselected by
default (see `addopts` in `pyproject.toml`). Run those with `make test-device` on
a Trainium host. No such tests exist yet; the marker is registered so a device
test can never be collected by a hook or by `make ci` on a machine with no
device.

## Remote setup (running on Trainium)

The dev desktop has no Neuron device, so run synthesis and bench on a remote
Trainium box via the bundled `tools/remote.py` driver. Do the one-time setup
below, then follow AGENTS.md "Environment" for the day-to-day commands. For the
underlying sync + ssh mechanics, see the vendored `run-on-remote` skill under
`.claude/skills/`.

One-time setup specific to your machine:

1. Define an SSH alias for your box in `~/.ssh/config` (the driver passes an
   alias to `ssh`/`rsync`, never an IP).
2. Copy the template to `.remote.toml` and set `host` to that alias. The
   template documents the schema and the SSH-alias convention; `.remote.toml` is
   gitignored (per-person), only the template is committed:

   ```
   cp .claude/skills/run-on-remote/.remote.toml.example .remote.toml
   ```

3. Install `uv` once on the remote box (its `post_sync` runs `uv sync`):
   `curl -LsSf https://astral.sh/uv/install.sh | sh`.

## Running synthesis

The CLI entry point is `axon` (defined in `pyproject.toml` →
`axon.cli:main`). There are no subcommands: `axon kernels/<name> --sizes
<ints>` is the whole interface. It takes a path to one kernel package
directory (or a single-file spec) and the problem shape as bare ints, bound
positionally to the spec's `dim_vars`:

```
# Synthesize + bench mul at 1024x1024, writing the CSV to out/mul_bw_1k.csv.
axon kernels/mul --sizes 1024 1024 --out out/mul_bw_1k.csv

# matmul has dim_vars (m, n, k), so these bind m=1024 n=512 k=16384.
axon kernels/matmul --sizes 1024 512 16384

# Override the input dtype produced by the spec's make_inputs.
# Choices: bfloat16, float16, float32.
axon kernels/matmul --sizes 1024 1024 1024 --dtype float32

# Compile + bench the kernel's NeuronPy baseline. `--mode baseline` runs only
# the baseline; `--mode both` runs the variants then the baseline.
axon kernels/matmul --sizes 1024 1024 1024 --mode baseline

# Pick the winner by a different timing column (default median).
axon kernels/mul --sizes 1024 1024 --out out/mul_bw_1k.csv --pick-by min
```

`--sizes` is **required**: bare ints (one per dim the spec declares) bound
positionally to `spec.dim_vars` in order. A wrong count exits 2 naming the
expected dims. `--out PATH` is the CSV path; the run stem (NEFF dir,
per-variant modules, baseline dir) all derive from it. It defaults to
`out/nki_<name>.csv`.

Exact candidate filtering is enabled by default. Emitted modules carry
structured `TILES_IN_BLOCK_*` constraints derived from the same codegen plan
that emits their block guards; the bench phase rejects tile values whose blocks
cannot fit or divide the concrete extent before starting a compiler worker.
Pass `--no-candidate-filter` to benchmark the full `tile_options` Cartesian
product. `--candidate-budget` limits legal tile configurations per emitted
schedule family (default 8; 0 is unlimited) using deterministic
quality/diversity selection. Missing or unknown constraint metadata fails open
to the full sweep.

### Agentic post-processing

`agent_opt/` takes the winner Axon picked and asks a model for a faster kernel
with the same semantics, measuring every attempt on device. One command runs
both halves:

```bash
uv run --extra agent python agent_opt/run_combined.py rmsnorm_h1024 \
  --nkilib "$NKILIB" --target-host trn2 --remote-config trn2 --iters 3
```

`--target-host` is your own ssh alias and `--remote-config` is a host key in
`.remote.toml`; the example uses the `trn2` of the template config.

It writes `combined_result.json` (candidate counts, per-phase wall time, best
kernel path) and `phase_timings.csv` (`phase,step,iter,seconds`) into
`agent_opt/results/<target>/<ts>/`, beside the post-process `results.json`. For a
synthesis-only run, `tools/run_case.py --timings <csv>` records the per-step wall
time in its own schema (`step,seconds,returncode,command`), which the combined
driver relabels into `phase_timings.csv`. Read `agent_opt/README.md` before the
first run: the post-process needs an nkilib clone with the Axon test adapter
installed, and Bedrock or Anthropic credentials.

### Cases and the Makefile

`eval/sizes.json` declares the workload **cases** per kernel: a map of
`<kernel> -> {<case_id>: {sizes, dtype?, rtol?, atol?}}`. `sizes` is the int
tuple in `dim_vars` order; `dtype` is optional (distinguishes same-shape
different-dtype cases); the presence of `rtol`/`atol` marks a case as a
head-to-head point (it gets a generated per-case nkilib harness, see
below). A case *is* the complete eval point.

The `Makefile` is data-driven: `eval/gen_size_targets.py` turns each
kernel/case into a file target. Run one case with `make <kernel>_<case_id>`;
it caches on `out/<kernel>_<case_id>.csv` (a no-op when the CSV is newer than
the spec and `sizes.json`):

```
# Run one case (caches on out/mul_bw_1k.csv).
make mul_bw_1k

# Run every case across all kernels.
make all

# A bare `make <kernel>` is ambiguous (a kernel has several cases), so it
# prints that kernel's case targets and exits 2.
make mul

# Wipe every per-case artifact for one kernel (CSV, run dir, baseline dir,
# winners), or clean everything.
make clean-mul
make clean

# Lint / format.
make format
make lint

# Pass extra flags through to `axon` (e.g. an alternate compile target).
make mul_bw_1k AXON_ARGS="--target trn3pre"
```

The generated fragment lives in `out/size_targets.mk` (gitignored; `make`
auto-remakes it from `eval/sizes.json` + the generator before reading it).

## Output layout

Everything for one case lives under a single self-contained stem
(`<kernel>_<case_id>`, derived from `--out`):

```
out/
  <stem>.csv                                   # final output (copied on full coverage)
  <stem>/
    results.csv                                 # one row per config (survives an interrupt)
    <name>__v<i>_t<j>.py                        # per-variant module
    __v<i>_t<j>_tiles_<tile_tag>/kernel.neff    # per-config NEFF (deleted once its row lands)
  baseline_<stem>/                              # baseline NEFFs (--mode baseline)
  winners/
    <kernel>_<case_id>.py                       # winning variant (head-to-head)
    <kernel>_<case_id>_case.py                  # self-describing per-case nkilib harness
```

`out/` is gitignored, resolved at call time from the current directory, so
always invoke `axon` from the repo root.

## Deliverables: `winners/`

`out/` is scratch. The deliverable for a case is the **specialized** kernel. It
goes in the tracked top-level `winners/` directory:

```
winners/
  <kernel>_<case_id>.py    # standalone kernel, tile config bound as constants
```

`out/winners/<stem>.py` is a template, not a kernel. It keeps the
`TILES_IN_BLOCK_*` parameters, because the bench sweep gives the same module a
different tile config on each row. The winning values are in a second file,
`<stem>_case.py`, under `CASE["shape_args"]`. You cannot run the template
correctly unless you read that second file.

`winners/<stem>.py` binds the winning values as constants. It takes only tensor
inputs:

```python
@nki.jit
def rmsnorm(x):
    # Specialized to the benched winner's tile configuration.
    TILES_IN_BLOCK_M = 1
    TILES_IN_BLOCK_N = 2
```

This does not change performance. The tile args are already compile-time kwargs:
`bench_runner` gives them to `CompileKernel`, which bakes them into the NEFF. The
compiler therefore saw the same constants before the rewrite. Measured
specialized/parameterized ratios are 0.9986 on `rmsnorm_h1024` and 1.0050 on
`attention_nkilib_d128`. Outputs are bit-identical.

`tools/specialize_winner.py` writes these files. The Makefile runs it after each
head-to-head bench, so `make <kernel>_<case_id>` updates the deliverable. lnc=2
cases are excluded: `winners/` holds single-core kernels only.

Two rules govern the directory:

- **Measured only.** The tool refuses a case that has no correct benched row in
  `out/<stem>.csv`. Every artifact carries a latency you can check.
- **Best wins.** A re-run replaces the artifact only if it measures faster. A
  slower or equal re-run prints `KEEP` and writes nothing. A contended box
  therefore cannot demote a good kernel. Use `--force` to overwrite.

Each artifact records its own provenance. It names the device, because a latency
means nothing without the hardware that produced it. It also gives absolute
paths, because the run usually stays on a remote Trainium box:

```
    measured      : 34.53 us median (on-device, correct)
    axon commit   : 8617490
    measured on   : trn2.48xlarge, 16 neuron devices, 4 cores each, lnc=2 (host ubuntu@ip-172-31-34-20)
    run CSV       : /home/ubuntu/chunghs-axon/axon/out/rmsnorm_h1024.csv
```

`axon commit` reads `unknown` on a remote box, because the sync excludes `.git`.
To record it, export `AXON_COMMIT=$(git rev-parse --short HEAD)`.

`tools/verify_specialized.py <stem>` compares a specialized kernel with the
parameterized one. It compiles both, compares `kernel_info.json`, runs both on
device, then compares outputs and medians. NEFF **bytes** are not a valid
equivalence test: one unchanged source compiled twice already gives a different
NEFF, about 24k of 42k bytes, at the same size. The tool therefore prints the
NEFF hash for information only.

Each synthesized `(hw_variant, tile_variant)` is written under `out/<stem>/`, then compiled + benched in-process by `bench_runner.run_nki_bench`. Each variant's on-device output is compared to the spec's fp32 `baseline_op` at the case tolerances.

`run_nki_bench` streams each tile config through a bounded pipeline. Compile workers hand each finished NEFF to the next idle device worker. The parent appends that row to `out/<stem>/results.csv`, then deletes the artifact directory it came from. A row means the config is finished, including compile and benchmark failures, so a re-run of an interrupted case skips it. The run copies `results.csv` to `out/<stem>.csv`, the final output, only after every expected config has a row. That keeps `.DELETE_ON_ERROR` from discarding a run's progress.

Three environment variables tune the pipeline:

- `AXON_COMPILE_WORKERS` sets the compile pool size (default: half the CPU count).
- `AXON_MAX_LIVE_NEFFS` caps live artifact directories, which bounds peak disk (default: twice the device worker count, minimum 4).
- `AXON_KEEP_NEFF=1` keeps every artifact directory instead of reclaiming it.

Resume assumes the emitted source and invocation arguments are unchanged under the same output stem. Delete `out/<stem>/` before a changed run.

## Adding a new kernel

Create a `kernels/<name>/` package with three files (see `kernels/mul/` for
the live template, and the project `CLAUDE.md` for the full walkthrough):

- `kernel.py` — the Axon math (`kernel_*` on `AxonArray`s; imports `axon`).
- `refs.py` — the numpy `baseline_op`, the torch `torch_ref` (head-to-head
  kernels only), and the rng-driven `make_inputs`. **Axon-free**: only
  numpy / torch / ml_dtypes imports, because nkilib's harness loads this
  file directly in a runtime that has no `axon`/`nkipy`. `torch_ref` is pure
  math — the generated harness coerces inputs to fp32 torch tensors first.
- `__init__.py` — assembles the top-level `SPEC: KernelSpec` (see
  `axon.kernel_spec`) from the two modules.

No registry to update: `axon kernels/<name> --sizes ...` loads the package
by path (single-file `kernels/<name>.py` specs also still load).

## Targets supported

Trainium 2 and 3.
