"""Kernel package for `qkv_cte` (QKV Context-Encoding projection core).

Distilled from `nkilib.core.qkv.qkv_cte.qkv_cte`, single-core (LNC=1) path:
residual add -> RMSNorm over the hidden dim -> fused QKV projection.

Dropped vs. the hand kernel (each a tracked Axon gap): the norm gamma scale
and projection bias (both `(1, N)` partition-broadcasts Axon can't emit yet)
and RoPE (head-structured rotation, outside the rank-2 surface). Dropping
gamma makes this the kernel's RMS_NORM_SKIP_GAMMA mode.

Layout (the standard directory-kernel split):
  kernel.py — the Axon math (imports axon; axon-runtime only)
  refs.py   — baseline_op / torch_ref / make_inputs (numpy/torch/ml_dtypes
              only; loaded directly by nkilib's harness, no import shims)
  __init__  — assembles SPEC (what `axon kernels/qkv_cte` loads)
"""

from axon.kernel_spec import KernelSpec

from .kernel import kernel_qkv_cte
from .refs import baseline_op, make_inputs, torch_ref

SPEC = KernelSpec(
    name="qkv_cte",
    axon_kernel=kernel_qkv_cte,
    dim_vars=("m", "n", "k"),
    input_specs=(
        ("x", ("m", "k")),
        ("mlp_prev", ("m", "k")),
        ("attention_prev", ("m", "k")),
        ("w", ("k", "n")),
    ),
    baseline_op=baseline_op,
    torch_ref=torch_ref,
    make_inputs=make_inputs,
    tile_args=("TILES_IN_BLOCK_M", "TILES_IN_BLOCK_N", "TILES_IN_BLOCK_K"),
)
