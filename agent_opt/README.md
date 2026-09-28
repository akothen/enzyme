# agent_opt — agentic optimization of Axon-generated kernels

An LLM agent takes an **Axon-synthesized NKI kernel** and iteratively rewrites it
into an equivalent but faster kernel, verifying and profiling each attempt on
Trainium and feeding the measured profile + attempt history back into the next
attempt. It answers: can an agent further optimize the winning Axon kernel
 — measured as **agent-vs-Axon**, with an optional **agent/Axon-vs-nkilib**
comparison when a trusted hand-written baseline kernel exists.

## Files

| File | Role |
|---|---|
| `optimize.py` | The post-process workflow (self-contained: driver + model client, prompt loading, agent calls, on-the-fly nkilib case-module generation, on-device verify/time, the nkilib bridge). Knows nothing about candidate filtering. |
| `run_combined.py` | The joiner: filtered Axon synthesis through `tools/run_case.py`, then `optimize.py`, then one `combined_result.json` with candidate counts and per-phase wall time. |
| `summarize.py` | Aggregates `results.json` across runs into `results_summary.csv`. |
| `seed.py` | Copies an Axon winner into a kernel's pinned `input_kernel.py` (+ provenance & content hash). |
| `install_adapter.py` | Copies `test_axon_emitted.py` into an nkilib clone (one-time, per clone). |
| `test_axon_emitted.py` | The pytest adapter nkilib runs; it reads `AXON_CASE_PATH` and times the kernel through nkilib's own harness. |
| `prompts/shared/` | Prompt parts: `system.txt` + `task.md` for the optimizer pass, `analyst_system.txt` + `analyst.md` for the iteration-0 analyst pass, `axon_overview.md` for both. |
| `prompts/kernels/<kernel>/` | Per-kernel inputs: `input_kernel.py` + `meta.json`. |
| `results/<kernel>/<ts>/` | Outputs (gitignored via `agent_opt/results/`). |

## Prerequisites

1. **A local nkilib clone** — the `KaenaNeuronKernelLibrary` Brazil package (e.g.
   `~/workspaces/nkilib-eval/src/KaenaNeuronKernelLibrary`). `brazil-build` must
   work in it. Pass its path as `--nkilib`.
2. **The versioned Axon test adapter installed into that clone** (one-time):
   ```bash
   python3 agent_opt/install_adapter.py --nkilib "$NKILIB"
   ```
   The loop selects `-k "TestAxonEmitted and <name>"` against this file; without
   it, verification matches nothing.
3. **A trn2 box** reachable by ssh alias (default `trn2`, override with
   `--target-host` or `AXON_TARGET_HOST`). The actual
   compile-to-NEFF + execution + neuron-profile happen there; `brazil-build
   --target-host` dispatches to it.
4. **Credentials** for the agent: `ANTHROPIC_API_KEY` (`--backend api`) or AWS
   credentials + region for Bedrock (`--backend bedrock`, the default). Install
   the optional agent dependency with `uv sync --extra agent`. Bedrock needs
   `bedrock:InvokeModel` on the model, which a dev desktop role has and a
   Trainium instance role usually does not, so run the agent steps on the
   desktop. `--model` takes the bare id (`claude-opus-4-8`); the Bedrock backend
   prefixes the cross-region profile itself (`us.anthropic.claude-opus-4-8`).
   `--region` defaults to `$AWS_REGION`, or `us-west-2` when that is unset.

**Three faults in the nkilib clone block every measurement, each with its own
message.** Fix them in this order: `MissingPackageVersionException` for a
dependency the version set lacks, which `brazil ws merge --local` resolves; an
`ImportError` from `nkilib_testing`, whose built copy goes stale and which a
plain `brazil-build` refreshes; and `SSH connection error ... Authentication
failed` followed by `marking 0 (ineligible)`, which needs an ssh-agent holding
the key plus an `IdentityFile` line in the host's ssh-config block. The harness
uses fabric2/paramiko and cannot read a Midway certificate directly.

## Setting up a kernel

Create `prompts/kernels/<kernel>/` with two files:

- **`input_kernel.py`** — the Axon *winning* NKI kernel to optimize. Seed it from
  a synth winner (`python3 tools/run_case.py <kernel>_<case>` writes
  `out/winners/<stem>.py`) with:
  ```bash
  python3 agent_opt/seed.py --kernel <kernel> --from out/winners/<stem>.py
  ```
  It stamps `provenance` + a content hash (`input_sha`) into `meta.json`, and
  **refuses to overwrite a different pinned input without `--force`** (a
  re-seed can't silently change what an experiment optimized). `optimize.py`
  records the input's hash in `results.json` and warns if it drifts from
  `meta.input_sha`.
- **`meta.json`**:
  ```json
  {
    "entry_point": "fused_adam",
    "dtype": "bfloat16",
    "input_names": ["param", "grad", "exp_avg", "exp_avg_sq", "step_size", "inv_bc2_sqrt", "wd"],
    "shape_args": {"p": 128, "f": 8192, "TILES_IN_BLOCK_M": 1, "TILES_IN_BLOCK_N": 8},
    "rtol": 1e-2, "atol": 1e-2,
    "refs": "kernels/fused_adam/refs.py",

    "nkilib": {"selector": "<-k row id>", "test": "test/integration/nkilib/core/<area>/test_<kernel>.py"}
  }
  ```
  - `shape_args` = the `make_inputs` dim kwargs **plus** the winner's
    `TILES_IN_BLOCK_*`. `input_sizes` in the CSV is the dim extents (tile args
    dropped).
  - `refs` points at the kernel's `refs.py` (`torch_ref` / `baseline_op` /
    `make_inputs`) — the correctness oracle.
  - `nkilib` (**optional**) is the hand-tuned-kernel bridge: the nkilib
    integration test + the `-k` selector for the matched shape row. Present only
    for kernels with a comparable standalone nkilib kernel (attention, qkv_cte,
    silu_mlp, etc.); omit for synthetic/new kernels. Get the exact selector
    with `brazil-build integration-test --collect-only -k "<fn>" <test.py>`.

For a winner generated by current Axon, no checked-in prompt bundle is needed:

```bash
uv run --extra agent python agent_opt/optimize.py \
  --case out/winners/cumsum_fast_small_case.py \
  --iters 3 --nkilib "$NKILIB" --target-host trn2
```

The generated case carries the kernel, reference, dtype, input names, shape,
tile values, and tolerances as one literal handoff contract.

## Combined Axon + post-processing

Run synthesis, default-on candidate filtering, device winner selection, and
agent post-processing as one workflow:

```bash
uv run --extra agent python agent_opt/run_combined.py cumsum_fast_small \
  --nkilib "$NKILIB" --target-host trn2 --remote-config trn2
```

Use `--no-candidate-filter` for the exhaustive tile-sweep control. The command
writes `combined_result.json` beside the post-processing `results.json`, with
Axon source/selected/correct candidate counts, per-phase wall time, and the
final best kernel path. `--candidate-budget N` controls legal tile candidates
per emitted schedule family (default 8; 0 unlimited).

`combined_result.json.axon.emitted_modules_source` says where the module count
came from: `emitted_modules` when the local run dir holds the modules, or
`results_csv` after `--emit-on box`, which emits remotely and pulls back only
`results.csv`.

It also writes `phase_timings.csv` (`phase,step,iter,seconds`), which merges
`run_case.py --timings` rows (`emit`, `sync`, `bench`, `pull`) with the
post-process rows (`baseline_measure`, `nkilib_measure`, `hint_generate`,
`generate`, `measure`), plus one driver-measured `wall` row per phase. The
per-phase totals exclude the `wall` rows, because a `wall` row already contains
its phase's steps; summing the whole `seconds` column double-counts.
`combined_result.json.timings` totals them and splits Axon into emit versus
device time and the post-process into model versus device time.

Two cautions on reading those numbers. A resumed bench reports almost no device
time, because `out/<stem>/results.csv` on the box carries the earlier rows;
delete `out/<stem>/` there for a true re-measure. And fewer candidates buys little wall
time, because an illegal tile configuration fails its own block assertion early
in compilation. Measured on trn2-large-2: `rmsnorm_h1024` filtered 16 candidates
in 74.7 s against 72 candidates in 71.0 s, and `attention_nkilib_d128` filtered
96 candidates in 358.6 s against 7,500 in 379.6 s, a saving of 5.5 percent. Treat
filtering as a way to keep the data and the workers clean, and report the
candidate ratio and the wall time as separate numbers.


## Running

```bash
uv run --extra agent python agent_opt/optimize.py \
    --kernel fused_adam_opt_1m --iters 3 \
    --nkilib /home/$USER/workspaces/nkilib-eval/src/KaenaNeuronKernelLibrary \
    --target-host trn2
```

The nkilib hand-tuned kernel is timed automatically whenever the kernel's
`meta.json` includes a `nkilib` bridge  (e.g.
`qkv_cte_llama405b_1k`, `attention_nkilib_s4k_d128`, `silu_mlp_qwen32b_1k`).

```bash
uv run --extra agent python agent_opt/optimize.py \
    --kernel qkv_cte_llama405b_1k --iters 3 \
    --nkilib /home/$USER/workspaces/nkilib-eval/src/KaenaNeuronKernelLibrary \
    --target-host trn2
```

What happens:
- **Iter 0** — measure the Axon baseline (ActiveInferenceTime), time the nkilib
  kernel if a bridge is declared, and (default `--hint auto`) run an **analyst**
  agent pass over the baseline kernel + its neuron-profile to generate the
  optimization hint that seeds the loop.
- **Iters 1..N** — prompt the agent with the current-best kernel + its profile +
  the attempt history; verify + measure + profile each result on trn2; keep the
  fastest correct one.

Key args:

| arg | default | meaning |
|---|---|---|
| `--kernel` / `--case` | (exactly one required) | a bundle dir under `prompts/kernels/`, or a generated Axon winner `*_case.py` |
| `--iters` | 3 | agent iterations (iter 0 is the measured baseline) |
| `--num-runs` | 8 | timed executions per kernel (jitter-robust) |
| `--hint` | `auto` | `auto`: analyst generates the hint from the baseline profile; `none`: no hint (ablation) |
| `--backend` | `bedrock` | `bedrock` (AWS creds) or `api` (`ANTHROPIC_API_KEY`) |
| `--model` | `claude-opus-4-8` | agent model id |
| `--nkilib` | `$NKILIB` | the nkilib package path (required) |
| `--target-host` | `$AXON_TARGET_HOST` or `trn2` | trn2 ssh alias |
| `--test-adapter` | `test/integration/nkilib/utils/test_axon_emitted.py` | nkilib-relative pytest adapter |

## Outputs & summary

Each run writes `results/<kernel>/<ts>/`: `results.json` (baseline / best /
per-iter records + `nkilib_inference_us` + a `timings` block),
`phase2_timings.csv` (`step,iter,seconds`), `prompt_iter*.txt`,
`response_iter*.md`, `kernels/iter*.py`, `best.py`, and `generated_hint.md`.
Pass `--iters 0` for baseline-only reproduction without an agent client.

`results.json` is at `schema_version` 2, which added `timings` and the
per-iteration `generate_seconds` / `measure_seconds`. Schema 1 files stay
readable: the summarizer leaves their timing columns blank.

Expect the post-process to cost several times the synthesis. One measured run
(`rmsnorm_h1024`, two iterations) spent 350 s against Axon's 65 s, split almost
evenly between model calls and device measurement, because every iteration pays
a full nkilib harness run.

Aggregate the latest run per kernel into a CSV:
```bash
uv run python agent_opt/summarize.py
```
Columns: `kernel, input_sizes, axon_us, best_agent_us, nkilib_us,
axon_over_nkilib, agent_over_axon, agent_over_nkilib, postprocess_total_s,
generate_s, measure_s`. Ratios are **latency ratios (< 1 = numerator faster)**.

`axon_us` and `best_agent_us` are both ActiveInferenceTime from nkilib's
harness, so `agent_over_axon` is the one confound-free ratio. Axon's own bench
reports a different number for the same kernel: about 34.5 us for the
`rmsnorm_h1024` winner, against 38.0 us on the nkilib basis. Never build a ratio
across the two harnesses.
