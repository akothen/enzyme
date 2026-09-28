#!/usr/bin/env python3
"""The agent-optimize workflow: iteratively rewrite one Axon-synthesized NKI
kernel into an equivalent but faster one, verifying + profiling each attempt on
Trainium and feeding the profile + attempt history back into the next attempt.

Iteration 0 measures the Axon baseline (and, if meta.json declares an `nkilib`
bridge, the nkilib hand-tuned kernel) and — with --hint auto (default) — runs an
analyst agent pass over the baseline profile to generate the optimization hint
that seeds the loop. Each later iteration prompts the agent with the current-best
kernel + its neuron-profile + the history, then verifies + measures + profiles.

Self-contained: model client (Anthropic API vs Amazon Bedrock), prompt-part
loading, agent calls with retry, on-the-fly nkilib case-module generation,
on-device verify/time, and the nkilib bridge all live in this file.
`summarize.py` aggregates results.json into a CSV. See README.md.

Usage:
    AWS_REGION=us-west-2 python3 agent_opt/optimize.py \
        --kernel layernorm_llama70b_s1024 --iters 3 \
        --nkilib <nkilib-eval package path> --target-host trn2
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv as _csv
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
AXON = HERE.parent  # <axon>/agent_opt -> <axon>
SHARED = HERE / "prompts" / "shared"
KERNELS = HERE / "prompts" / "kernels"
DEFAULT_RESULTS = HERE / "results"

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "claude-opus-4-8")
_SEED = 42
RESULT_SCHEMA_VERSION = 2  # 2 adds the `timings` block + per-iteration seconds
DEFAULT_TEST_ADAPTER = "test/integration/nkilib/utils/test_axon_emitted.py"


# --------------------------------------------------------------------------
# Prompt assembly (shared parts + per-kernel parts)
# --------------------------------------------------------------------------


def _read(p: Path) -> str:
    return p.read_text()


# --------------------------------------------------------------------------
# Model client (Anthropic API vs Amazon Bedrock)
# --------------------------------------------------------------------------


def make_client(backend: str, region: str):
    try:
        import anthropic  # noqa: F401
    except ModuleNotFoundError as e:
        raise SystemExit("install the SDK: `uv pip install anthropic`") from e
    if backend == "auto":
        backend = "api" if os.environ.get("ANTHROPIC_API_KEY") else "bedrock"
    if backend == "api":
        from anthropic import Anthropic

        return Anthropic(), "api"
    if backend == "bedrock":
        # Legacy bedrock-runtime client + a cross-region inference-profile id
        # (us.anthropic.<model>, produced by resolve_model). The Mantle endpoint
        # is not enabled in every account, so this is the portable choice.
        from anthropic import AnthropicBedrock

        return AnthropicBedrock(aws_region=region), "bedrock"
    raise SystemExit(f"unknown backend {backend!r}")


def resolve_model(model: str, backend: str) -> str:
    if backend != "bedrock":
        return model
    if model.startswith(("us.", "global.", "eu.", "apac.")):
        return model
    if model.startswith("anthropic."):
        return "us." + model
    return "us.anthropic." + model


def _looks_like_auth_error(e: Exception) -> bool:
    msg = str(e).lower()
    needles = (
        "credential",
        "could not resolve aws",
        "unable to locate credentials",
        "authentication",
        "unauthorized",
        "accessdenied",
        "access denied",
        "security token",
        "expired",
        "not authorized",
    )
    return type(e).__name__ in (
        "AuthenticationError",
        "PermissionDeniedError",
        "NoCredentialsError",
    ) or any(n in msg for n in needles)


def _is_transient(e: Exception) -> bool:
    """A retryable server/transport hiccup (503/429/500/502/504, overloaded,
    connection/timeout) — as opposed to an auth error (never retry) or a real
    client error (4xx other than 429)."""
    if _looks_like_auth_error(e):
        return False
    sc = getattr(e, "status_code", None)
    if sc is not None:
        return sc in (429, 500, 502, 503, 504)
    if type(e).__name__ in (
        "ServiceUnavailableError",
        "RateLimitError",
        "InternalServerError",
        "APIConnectionError",
        "APITimeoutError",
        "OverloadedError",
    ):
        return True
    msg = str(e).lower()
    return any(
        n in msg
        for n in (
            "503",
            "429",
            "overloaded",
            "unable to process",
            "service unavailable",
            "rate limit",
            "timeout",
            "temporarily",
        )
    )


def call_agent(
    client,
    backend: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 32000,
    max_retries: int = 4,
) -> str:
    kwargs = {
        "model": resolve_model(model, backend),
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }

    def _once():
        try:
            with client.messages.stream(**kwargs) as stream:
                return stream.get_final_message()
        except TypeError:
            kwargs.pop("output_config", None)
            kwargs.pop("thinking", None)
            with client.messages.stream(**kwargs) as stream:
                return stream.get_final_message()

    for attempt in range(max_retries + 1):
        try:
            msg = _once()
            return "".join(
                b.text for b in msg.content if getattr(b, "type", None) == "text"
            )
        except Exception as e:  # noqa: BLE001
            # Auth errors and real client errors propagate immediately (main
            # turns auth into a clean exit); only transient server hiccups retry.
            if attempt == max_retries or not _is_transient(e):
                raise
            wait = min(60, 5 * 2**attempt)  # 5, 10, 20, 40, 60s
            print(
                f"    transient API error ({type(e).__name__}); "
                f"retry {attempt + 1}/{max_retries} in {wait}s ..."
            )
            time.sleep(wait)


_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_kernel(response: str) -> str:
    blocks = _CODE_BLOCK.findall(response)
    if not blocks:
        return response.strip() + "\n"
    scored = [b for b in blocks if "@nki.jit" in b or "def " in b]
    return max(scored or blocks, key=len).strip() + "\n"


# --------------------------------------------------------------------------
# Verification on Trainium
# --------------------------------------------------------------------------

_CASE_TEMPLATE = '''\
"""Auto-generated verification case for {name}. Do not edit."""
import importlib.util as _ilu
import ml_dtypes  # noqa: F401
import numpy as np
import torch

_SEED = {seed}
_DTYPE = "{dtype}"
_KERNEL_PATH = {refs_abs!r}
_INPUT_NAMES = {input_names!r}

CASE = {{
    "name": "{name}",
    "entry_point": "{entry_point}",
    "kernel_rel": "{kernel_rel}",
    "shape_args": {shape_args!r},
    "rtol": {rtol!r},
    "atol": {atol!r},
}}

_torch_dt = getattr(torch, _DTYPE)
_np_dt = np.dtype(_DTYPE)


def _to_torch_f32(a):
    if isinstance(a, torch.Tensor):
        return a.to(torch.float32)
    return torch.from_numpy(np.asarray(a).astype(np.float32))


def _load_spec():
    _s = _ilu.spec_from_file_location("_genref", _KERNEL_PATH)
    _m = _ilu.module_from_spec(_s)
    _s.loader.exec_module(_m)
    return getattr(_m, "SPEC", _m)


_SPEC = _load_spec()


def torch_ref({ref_params}):
{ref_del}
    result = _SPEC.torch_ref({ref_call})
    if not isinstance(result, tuple):
        result = (result,)
    return {{f"output_{{i}}": t.to(_torch_dt) for i, t in enumerate(result)}}


def make_inputs({mk_params}):
    data = _SPEC.make_inputs({mk_call}, dtype=_np_dt, rng=np.random.default_rng(_SEED))
    out = {{name: arr for name, arr in zip(_INPUT_NAMES, data)}}
{mk_update}
    return out


def output_tensors(kernel_input):
    ref = _SPEC.baseline_op(*[kernel_input[name] for name in _INPUT_NAMES])
    if not isinstance(ref, tuple):
        ref = (ref,)
    return {{f"output_{{i}}": np.zeros(np.asarray(a).shape, dtype=_np_dt)
            for i, a in enumerate(ref)}}
'''


def _split_shape_args(shape_args: dict) -> tuple[list[str], list[str]]:
    tiles = [k for k in shape_args if k.startswith("TILES_IN_BLOCK")]
    dims = [k for k in shape_args if k not in tiles]
    return dims, tiles


def write_case_module(dest: Path, meta: dict, kernel_rel: str, name: str) -> None:
    dims, tiles = _split_shape_args(meta["shape_args"])
    inputs = meta["input_names"]
    refs_abs = str(
        Path(meta["_refs_path"]).resolve()
        if meta.get("_refs_path")
        else (AXON / meta["refs"]).resolve()
    )
    ref_params = ", ".join(inputs + tiles)
    ref_del = ("    del " + ", ".join(tiles)) if tiles else "    pass"
    ref_call = ", ".join(f"_to_torch_f32({n})" for n in inputs)
    mk_params = ", ".join(dims + tiles)
    mk_call = ", ".join(f"{d}={d}" for d in dims)
    mk_update = (
        ("    out.update({" + ", ".join(f'"{t}": int({t})' for t in tiles) + "})")
        if tiles
        else "    # no tile kwargs"
    )
    dest.write_text(
        _CASE_TEMPLATE.format(
            name=name,
            seed=_SEED,
            dtype=meta["dtype"],
            refs_abs=refs_abs,
            input_names=inputs,
            entry_point=meta["entry_point"],
            kernel_rel=kernel_rel,
            shape_args=meta["shape_args"],
            rtol=meta["rtol"],
            atol=meta["atol"],
            ref_params=ref_params,
            ref_del=ref_del,
            ref_call=ref_call,
            mk_params=mk_params,
            mk_call=mk_call,
            mk_update=mk_update,
        )
    )


def _read_inference_us(csv_path: Path) -> float | None:
    with csv_path.open() as f:
        rows = list(_csv.reader(f))
    if len(rows) < 2:
        return None
    for h, v in zip(rows[0], rows[1], strict=False):
        if h == "InferenceTime" and v:
            try:
                return float(v) * 1e6
            except ValueError:
                return None
    return None


def _newest_qor_us(nkilib: Path, *, newer_than_ns: int = 0) -> float | None:
    csvs = sorted(
        (nkilib / "neuron_test_output").glob("qor_data_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    fresh = [path for path in csvs if path.stat().st_mtime_ns > newer_than_ns]
    return _read_inference_us(fresh[-1]) if fresh else None


def _passed_count(pytest_output: str) -> int:
    """Number of tests pytest reported as passed (0 if none / no match)."""
    m = re.search(r"(\d+) passed", pytest_output)
    return int(m.group(1)) if m else 0


def run_nkilib(meta: dict, nkilib: Path, host: str) -> float | None:
    """Time the nkilib kernel via its test bridge; return InferenceTime
    µs (or None if meta.json declares no `nkilib` bridge / there is no match)."""
    bridge = meta.get("nkilib")
    if not bridge:
        return None
    selector = bridge.get("selector")
    if not selector:
        print("  WARNING: nkilib bridge has no selector; skipping nkilib timing")
        return None
    started_ns = time.time_ns()
    cmd = [
        "brazil-build",
        "integration-test",
        "--platform-target=trn2",
        f"--target-host={host}",
        "-k",
        selector,
        bridge["test"],
    ]
    proc = subprocess.run(
        cmd, cwd=str(nkilib), env=dict(os.environ), capture_output=True, text=True
    )
    out = proc.stdout + "\n" + proc.stderr
    # The selector MUST resolve to exactly one row: 0 -> a dead/typo'd selector,
    # >1 -> ambiguous (which lnc/shape got timed is nondeterministic — _newest_qor
    # would pick whichever ran last). Either way the nkilib number is untrustworthy,
    # so refuse it loudly rather than return a wrong comparison.
    n_passed = _passed_count(out)
    if proc.returncode != 0 or n_passed != 1:
        print(
            f"  WARNING: nkilib selector {selector!r} matched "
            f"{n_passed} passing test(s) (need exactly 1); skipping nkilib timing"
        )
        return None
    return _newest_qor_us(nkilib, newer_than_ns=started_ns)


_PROFILE_KEYS = [
    "total_time",
    "total_active_time",
    "dma_active_time",
    "dma_active_time_percent",
    "scalar_engine_active_time_percent",
    "gpsimd_engine_active_time_percent",
    "tensor_engine_active_time",
    "sbuf_write_bytes",
    "software_dynamic_dma_packet_count",
    "mfu_estimated_percent",
]


def load_meta(kernel: str) -> dict:
    kdir = KERNELS / kernel
    if not kdir.is_dir():
        available = sorted(path.name for path in KERNELS.iterdir() if path.is_dir())
        raise SystemExit(
            f"no prompt bundle at {kdir}; available: {', '.join(available)}"
        )
    meta = json.loads((kdir / "meta.json").read_text())
    meta["_dir"] = kdir
    meta["_src"] = (kdir / "input_kernel.py").read_text()
    meta["_hint"] = ""  # generated at iter 0 by the analyst pass
    meta["_name"] = kernel
    return meta


def load_case_meta(case_path: str | Path) -> dict:
    """Load a generated Axon winner/case as an agent optimization input."""
    path = Path(case_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Axon winner case not found: {path}")
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
        case = None
        for statement in tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == "CASE"
                for target in statement.targets
            ):
                case = ast.literal_eval(statement.value)
                break
    except (OSError, SyntaxError, ValueError) as exc:
        raise SystemExit(f"could not read literal CASE from {path}: {exc}") from exc
    if not isinstance(case, dict):
        raise SystemExit(f"{path} has no literal CASE dict")

    required = {
        "name",
        "entry_point",
        "kernel_rel",
        "refs_rel",
        "dtype",
        "input_names",
        "shape_args",
        "rtol",
        "atol",
    }
    missing = sorted(required - set(case))
    if missing:
        raise SystemExit(
            f"{path} CASE lacks {missing}; regenerate it with current Axon"
        )
    kernel_path = (path.parent / case["kernel_rel"]).resolve()
    refs_path = (path.parent / case["refs_rel"]).resolve()
    if not kernel_path.is_file():
        raise SystemExit(f"winner kernel referenced by CASE is missing: {kernel_path}")
    if not refs_path.is_file():
        raise SystemExit(f"reference module referenced by CASE is missing: {refs_path}")
    return {
        "entry_point": case["entry_point"],
        "dtype": case["dtype"],
        "input_names": case["input_names"],
        "shape_args": case["shape_args"],
        "rtol": case["rtol"],
        "atol": case["atol"],
        "_dir": path.parent,
        "_src": kernel_path.read_text(),
        "_hint": "",
        "_name": case["name"],
        "_refs_path": refs_path,
        "_case_path": path,
    }


def run_neuron_profile(nkilib: Path, case_name: str, host: str) -> dict | None:
    """scp the emitted NEFF+NTFF to the box and parse neuron-profile summary-text."""
    out = nkilib / "neuron_test_output" / f"out-test_axon_kernel_trn2_{case_name}"
    neff = out / "file.neff"
    ntff = out / "infer_result" / "profile.ntff"
    if not (neff.is_file() and ntff.is_file()):
        return None
    rd = f"/tmp/prof_{case_name}"
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {rd}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    for src, dst in ((neff, "file.neff"), (ntff, "profile.ntff")):
        subprocess.run(
            ["scp", "-o", "BatchMode=yes", str(src), f"{host}:{rd}/{dst}"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    p = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            f"/opt/aws/neuron/bin/neuron-profile view -n {rd}/file.neff "
            f"-s {rd}/profile.ntff --output-format summary-text --disable-ui",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    prof = {}
    for line in (p.stdout + p.stderr).splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in _PROFILE_KEYS:
            with contextlib.suppress(ValueError):
                prof[parts[0]] = float(parts[1])
    return prof or None


def fmt_profile(prof: dict | None) -> str:
    if not prof:
        return "  (no profile captured)"

    def us(k):
        return f"{prof[k] * 1e6:.1f} us" if k in prof else "n/a"

    def pct(k):
        return f"{prof[k] * 100:.1f}%" if k in prof else "n/a"

    return (
        f"  total_time            = {us('total_time')}\n"
        f"  DMA active time       = {us('dma_active_time')} ({pct('dma_active_time_percent')} of total)  <- HBM traffic\n"
        f"  scalar(act) engine    = {pct('scalar_engine_active_time_percent')} of total\n"
        f"  gpsimd engine         = {pct('gpsimd_engine_active_time_percent')} of total\n"
        f"  tensor engine         = {us('tensor_engine_active_time')} (matmul; ~0 for norm)\n"
        f"  sbuf_write_bytes      = {prof.get('sbuf_write_bytes', 0):.0f}\n"
        f"  dyn DMA packet count  = {prof.get('software_dynamic_dma_packet_count', 0):.0f}\n"
        f"  MFU                   = {pct('mfu_estimated_percent')}"
    )


def verify_and_profile(
    kernel_src: str,
    meta: dict,
    name: str,
    nkilib: Path,
    host: str,
    num_runs: int,
    test_adapter: str = DEFAULT_TEST_ADAPTER,
) -> dict:
    """Write kernel + case, run the integration-test (num_runs, profiling on),
    return {compiles, correct, active_us, inference_us, profile, log_tail}."""
    work = Path(f"/tmp/fbloop/{name}.py")
    work.parent.mkdir(exist_ok=True)
    work.write_text(kernel_src)
    write_case_module(work.with_name(f"{name}_case.py"), meta, work.name, name)
    env = dict(
        os.environ,
        AXON_CASE_PATH=str(work.with_name(f"{name}_case.py")),
        AXON_NUM_RUNS=str(num_runs),
    )
    started_ns = time.time_ns()
    cmd = [
        "brazil-build",
        "integration-test",
        "--platform-target=trn2",
        f"--target-host={host}",
        "-k",
        f"TestAxonEmitted and {name}",
        test_adapter,
    ]
    proc = subprocess.run(
        cmd, cwd=str(nkilib), env=env, capture_output=True, text=True, timeout=900
    )
    out = proc.stdout + "\n" + proc.stderr
    # Exactly one test must pass, the same guard `run_nkilib` uses. pytest exits 0
    # when the `-k` selector deselects everything, and "0 passed, N deselected"
    # contains " passed", so a looser check would record an unverified kernel as
    # correct and let it win.
    passed = _passed_count(out) == 1 and proc.returncode == 0
    active_us = inference_us = None
    if passed:
        csvs = sorted(
            (
                path
                for path in (nkilib / "neuron_test_output").glob("qor_data_*.csv")
                if path.stat().st_mtime_ns > started_ns
            ),
            key=lambda p: p.stat().st_mtime,
        )
        if csvs:
            import csv as _csv

            rows = list(_csv.reader(csvs[-1].open()))
            if len(rows) >= 2:
                d = dict(zip(rows[0], rows[1], strict=False))
                if d.get("ActiveInferenceTime"):
                    active_us = float(d["ActiveInferenceTime"]) * 1e6
                if d.get("InferenceTime"):
                    inference_us = float(d["InferenceTime"]) * 1e6
    prof = run_neuron_profile(nkilib, name, host) if passed else None
    return {
        "compiles": passed or ("error" not in out.lower()),
        "correct": passed,
        "active_us": active_us,
        "inference_us": inference_us,
        "profile": prof,
        "log_tail": "\n".join(out.strip().splitlines()[-6:]),
    }


def build_iter_prompt(
    meta: dict,
    current_src: str,
    baseline_us: float,
    best_us: float,
    latest_profile: dict | None,
    history: list[dict],
    iteration: int,
    iters: int,
) -> str:
    parts = [_read(SHARED / "task.md")]
    parts.append(
        "## Background: how Axon (the synthesizer that produced this "
        "kernel) works\n\n" + _read(SHARED / "axon_overview.md")
    )
    if meta["_hint"]:
        parts.append("## Hint: where this kernel's time goes\n\n" + meta["_hint"])

    fb = [
        f"## Iterative optimization — iteration {iteration} of {iters}",
        "You are in a feedback loop. Below is the CURRENT BEST kernel, its measured",
        "on-device profile, and the history of what has been tried. Propose a",
        "change, and output the FULL kernel. Use the profile to decide what to",
        "target next.",
        "",
        f"Baseline (Axon) ActiveInferenceTime: {baseline_us:.1f} us",
        f"Current best ActiveInferenceTime:    {best_us:.1f} us",
        "",
        "Profile of the CURRENT BEST kernel:",
        fmt_profile(latest_profile),
        "",
    ]
    if history:
        fb.append("Attempt history (most recent last):")
        for h in history:
            status = (
                "correct"
                if h["correct"]
                else ("compiled-but-wrong" if h["compiles"] else "did-not-compile")
            )
            lat = f"{h['active_us']:.1f} us" if h["active_us"] else "n/a"
            fb.append(f"  - iter {h['iter']}: {status}, {lat} — {h['note']}")
        fb.append("")
    fb.append(
        "Begin your kernel file with a single comment line "
        "`# ITER_NOTE: <what you changed vs the current kernel and why>`."
    )
    parts.append("\n".join(fb))

    cfg = ", ".join(f"{k}={v}" for k, v in meta["shape_args"].items())
    parts.append(
        f"## The current best kernel to improve (`{meta['entry_point']}`, "
        f"config: {cfg})\n\n```python\n{current_src}\n```\n"
    )
    return "\n\n".join(parts)


def generate_hint(
    client,
    backend: str,
    model: str,
    meta: dict,
    baseline_profile: dict | None,
) -> str:
    """Iteration-0 analyst pass: inspect the baseline Axon kernel + its measured
    on-device profile and produce the optimization hint — the agentic replacement
    for a human-written hint. Returns the hint text.

    The analyst gets its OWN system prompt: the optimizer's (`system.txt`) demands
    exactly one kernel file and nothing else, which is the opposite of what
    `analyst.md` asks for."""
    cfg = ", ".join(f"{k}={v}" for k, v in meta["shape_args"].items())
    user = "\n\n".join(
        [
            _read(SHARED / "analyst.md"),
            "## Background: how Axon (the synthesizer that produced this kernel) "
            "works\n\n" + _read(SHARED / "axon_overview.md"),
            f"## The kernel (`{meta['entry_point']}`, config: {cfg})\n\n"
            f"```python\n{meta['_src']}\n```\n",
            "## Its measured on-device profile\n\n" + fmt_profile(baseline_profile),
        ]
    )
    analyst_system = _read(SHARED / "analyst_system.txt").strip()
    return call_agent(client, backend, model, analyst_system, user).strip()


PHASE2_TIMING_FIELDS = ("step", "iter", "seconds")


def phase2_timing_rows(
    baseline_measure_s: float,
    nkilib_measure_s: float | None,
    hint_s: float | None,
    records: list[dict],
) -> list[dict]:
    """Flatten the post-process wall times into CSV rows. `generate` is the model
    call; `measure` is the on-device verify + profile, which dominates."""
    rows = [{"step": "baseline_measure", "iter": 0, "seconds": baseline_measure_s}]
    if nkilib_measure_s is not None:
        rows.append({"step": "nkilib_measure", "iter": 0, "seconds": nkilib_measure_s})
    if hint_s is not None:
        rows.append({"step": "hint_generate", "iter": 0, "seconds": hint_s})
    for rec in records:
        if rec["role"] != "agent":
            continue
        rows.append(
            {
                "step": "generate",
                "iter": rec["iter"],
                "seconds": rec["generate_seconds"],
            }
        )
        rows.append(
            {"step": "measure", "iter": rec["iter"], "seconds": rec["measure_seconds"]}
        )
    return rows


def write_phase2_timings(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=PHASE2_TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--kernel", help="versioned bundle under prompts/kernels")
    source.add_argument("--case", help="generated Axon winner *_case.py")
    ap.add_argument(
        "--iters",
        type=int,
        default=3,
        help="rewrite iterations after the measured baseline; 0 is baseline-only",
    )
    ap.add_argument("--num-runs", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="bedrock")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--nkilib", default=os.environ.get("NKILIB", ""))
    ap.add_argument("--target-host", default=os.environ.get("AXON_TARGET_HOST", "trn2"))
    ap.add_argument(
        "--test-adapter",
        default=DEFAULT_TEST_ADAPTER,
        help="nkilib-relative Axon-emitted pytest adapter path",
    )
    ap.add_argument("--out-root", default=str(AXON / "agent_opt" / "results"))
    ap.add_argument(
        "--hint",
        choices=["auto", "none"],
        default="auto",
        help="auto (default): an agent analyst generates the hint "
        "from the baseline profile; none: run with no hint "
        "(ablation).",
    )
    args = ap.parse_args()

    if args.iters < 0:
        raise SystemExit("--iters must be >= 0")
    if args.num_runs <= 0:
        raise SystemExit("--num-runs must be > 0")
    if not args.nkilib:
        raise SystemExit("need --nkilib (or $NKILIB)")
    nkilib = Path(args.nkilib).expanduser().resolve()
    if not nkilib.is_dir():
        raise SystemExit(f"nkilib package directory not found: {nkilib}")
    adapter = nkilib / args.test_adapter
    if not adapter.is_file():
        raise SystemExit(
            f"Axon-emitted test adapter not found: {adapter}. "
            "Install it or pass --test-adapter."
        )
    meta = load_case_meta(args.case) if args.case else load_meta(args.kernel)
    run_name = meta["_name"]
    input_sha = hashlib.sha256(meta["_src"].encode()).hexdigest()
    if meta.get("input_sha") and meta["input_sha"] != input_sha:
        print(
            f"  WARNING: input_kernel.py sha {input_sha[:12]} != "
            f"meta.input_sha {meta['input_sha'][:12]} "
            "(input changed since seeding)"
        )
    system = _read(SHARED / "system.txt").strip()
    client = backend = None
    # `_dt.UTC` needs 3.11; `timezone.utc` is the portable spelling. This script
    # runs under whatever `python3` the caller has (NAD's adapter can even pass
    # its own `python`), so it must not need the repo's pinned interpreter.
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime(  # noqa: UP017
        "%Y%m%d_%H%M%S"
    )
    run_dir = Path(args.out_root) / run_name / stamp
    (run_dir / "kernels").mkdir(parents=True, exist_ok=True)

    # Iteration 0: measure the Axon baseline.
    print("iter 0: measuring Axon baseline ...")
    run_started = time.perf_counter()
    _t0 = time.perf_counter()
    b = verify_and_profile(
        meta["_src"],
        meta,
        f"{run_name}_fb0",
        nkilib,
        args.target_host,
        args.num_runs,
        args.test_adapter,
    )
    baseline_measure_s = round(time.perf_counter() - _t0, 3)
    if not b["correct"] or b["active_us"] is None:
        raise SystemExit(f"baseline failed to verify:\n{b['log_tail']}")
    baseline_us = b["active_us"]
    (run_dir / "kernels" / "iter0_baseline.py").write_text(meta["_src"])
    print(f"  baseline ActiveInferenceTime = {baseline_us:.1f} us")

    # Optional: time the nkilib kernel for the Axon-vs-nkilib comparison
    # (only when meta.json declares an `nkilib` test bridge; None otherwise).
    _t0 = time.perf_counter()
    nkilib_us = run_nkilib(meta, nkilib, args.target_host)
    nkilib_measure_s = (
        round(time.perf_counter() - _t0, 3) if nkilib_us is not None else None
    )
    if nkilib_us is not None:
        print(f"  nkilib InferenceTime         = {nkilib_us:.1f} us")

    # Resolve the hint that seeds every iteration. `auto` (default) generates it
    # agentically from the baseline profile; `none` runs with no hint.
    if args.iters > 0:
        client, backend = make_client(args.backend, args.region)

    hint_s = None
    if args.iters > 0 and args.hint == "auto":
        assert client is not None and backend is not None
        print("  generating hint (analyst pass) ...")
        _t0 = time.perf_counter()
        meta["_hint"] = generate_hint(client, backend, args.model, meta, b["profile"])
        hint_s = round(time.perf_counter() - _t0, 3)
        (run_dir / "generated_hint.md").write_text(meta["_hint"])
    elif args.hint == "none" or args.iters == 0:
        meta["_hint"] = ""

    best = {
        "src": meta["_src"],
        "active_us": baseline_us,
        "profile": b["profile"],
        "iter": 0,
    }
    history = [
        {
            "iter": 0,
            "note": "Axon baseline",
            "compiles": True,
            "correct": True,
            "active_us": baseline_us,
            "profile": b["profile"],
        }
    ]
    records = [
        {
            "iter": 0,
            "role": "baseline",
            **{
                k: b[k]
                for k in ("compiles", "correct", "active_us", "inference_us", "profile")
            },
            "generate_seconds": None,
            "measure_seconds": baseline_measure_s,
        }
    ]

    for i in range(1, args.iters + 1):
        print(f"iter {i}: generating ...")
        user = build_iter_prompt(
            meta,
            best["src"],
            baseline_us,
            best["active_us"],
            best["profile"],
            history,
            i,
            args.iters,
        )
        (run_dir / f"prompt_iter{i}.txt").write_text(
            f"# SYSTEM\n{system}\n\n# USER\n{user}\n"
        )
        _t0 = time.perf_counter()
        resp = call_agent(client, backend, args.model, system, user)
        generate_s = round(time.perf_counter() - _t0, 3)
        (run_dir / f"response_iter{i}.md").write_text(resp)
        code = extract_kernel(resp)
        (run_dir / "kernels" / f"iter{i}.py").write_text(code)
        # ITER_NOTE may land inside the code block (as a comment) or outside it
        # (as prose) — search the code first, then the whole response.
        m = re.search(r"#?\s*ITER_NOTE:\s*(.+)", code) or re.search(
            r"#?\s*ITER_NOTE:\s*(.+)", resp
        )
        note = m.group(1).strip() if m else "(no note)"
        print(f"  change: {note}")

        _t0 = time.perf_counter()
        r = verify_and_profile(
            code,
            meta,
            f"{run_name}_fb{i}",
            nkilib,
            args.target_host,
            args.num_runs,
            args.test_adapter,
        )
        measure_s = round(time.perf_counter() - _t0, 3)
        speedup = (baseline_us / r["active_us"]) if r["active_us"] else None
        print(
            f"  compiles={r['compiles']} correct={r['correct']} "
            f"active={r['active_us']} us"
            + (f" ({speedup:.3f}x vs baseline)" if speedup else "")
        )
        history.append(
            {
                "iter": i,
                "note": note,
                "compiles": r["compiles"],
                "correct": r["correct"],
                "active_us": r["active_us"],
                "profile": r["profile"],
            }
        )
        records.append(
            {
                "iter": i,
                "role": "agent",
                "note": note,
                **{
                    k: r[k]
                    for k in (
                        "compiles",
                        "correct",
                        "active_us",
                        "inference_us",
                        "profile",
                    )
                },
                "generate_seconds": generate_s,
                "measure_seconds": measure_s,
            }
        )
        if r["correct"] and r["active_us"] and r["active_us"] < best["active_us"]:
            best = {
                "src": code,
                "active_us": r["active_us"],
                "profile": r["profile"],
                "iter": i,
            }
            print(f"  -> new best ({r['active_us']:.1f} us)")

    best_kernel_path = run_dir / "best.py"
    best_kernel_path.write_text(best["src"])
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "success",
        "kernel": run_name,
        "timestamp": stamp,
        "model": args.model,
        "backend": backend or "not_run",
        "target_host": args.target_host,
        "num_runs": args.num_runs,
        "hint_mode": args.hint,
        "input_sha": input_sha,
        "shape_args": meta["shape_args"],
        "baseline_kernel_path": str(run_dir / "kernels" / "iter0_baseline.py"),
        "baseline_active_us": baseline_us,
        "best_active_us": best["active_us"],
        "best_iter": best["iter"],
        "best_kernel_path": str(best_kernel_path),
        "best_speedup_vs_baseline": (baseline_us / best["active_us"])
        if best["active_us"]
        else None,
        "nkilib_inference_us": nkilib_us,
        "timings": {
            "baseline_measure_seconds": baseline_measure_s,
            "nkilib_measure_seconds": nkilib_measure_s,
            "hint_generate_seconds": hint_s,
            "generate_seconds": sum(rec["generate_seconds"] or 0.0 for rec in records)
            + (hint_s or 0.0),
            "measure_seconds": sum(rec["measure_seconds"] or 0.0 for rec in records),
            "total_seconds": round(time.perf_counter() - run_started, 3),
            "csv_path": str(run_dir / "phase2_timings.csv"),
        },
        "iterations": records,
    }
    write_phase2_timings(
        run_dir / "phase2_timings.csv",
        phase2_timing_rows(baseline_measure_s, nkilib_measure_s, hint_s, records),
    )
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\n=== done. best = iter {best['iter']}: {best['active_us']:.1f} us "
        f"({summary['best_speedup_vs_baseline']:.3f}x vs baseline {baseline_us:.1f} us) ==="
    )
    print(f"wrote {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
