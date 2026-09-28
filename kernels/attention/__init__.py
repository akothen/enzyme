"""Kernel package for `attention` (fused QKV projection + softmax attention).

Layout (the standard directory-kernel split):
  kernel.py — the Axon math (imports axon; axon-runtime only)
  refs.py   — baseline_op / make_inputs (numpy/torch/ml_dtypes only; loaded
              directly by nkilib's harness, no import shims)
  __init__  — assembles SPEC (what `axon kernels/attention` loads)
"""

from axon.kernel_spec import KernelSpec

from .kernel import kernel_attention
from .refs import baseline_op, make_inputs, torch_ref

SPEC = KernelSpec(
    name="attention",
    axon_kernel=kernel_attention,
    dim_vars=("m", "n", "k"),
    input_specs=(
        ("x", ("m", "k")),
        ("w_q", ("k", "n")),
        ("w_k", ("k", "n")),
        ("w_v", ("k", "n")),
    ),
    baseline_op=baseline_op,
    torch_ref=torch_ref,
    make_inputs=make_inputs,
    tile_args=(
        "TILES_IN_BLOCK_M",
        "TILES_IN_BLOCK_N",
        "TILES_IN_BLOCK_K",
        "TILES_IN_BLOCK_P",
    ),
    tile_options=(1, 2, 4, 8, 16),
)
