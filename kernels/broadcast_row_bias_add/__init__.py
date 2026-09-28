"""Kernel package for `broadcast_row_bias_add` (row-reduce bias broadcast +
matmul).

Layout (the standard directory-kernel split):
  kernel.py — the Axon math (imports axon; axon-runtime only)
  refs.py   — baseline_op / make_inputs (numpy/torch/ml_dtypes only; loaded
              directly by nkilib's harness, no import shims)
  __init__  — assembles SPEC (what `axon kernels/broadcast_row_bias_add` loads)
"""

from axon.kernel_spec import KernelSpec

from .kernel import kernel_broadcast_row_bias_add
from .refs import baseline_op, make_inputs

SPEC = KernelSpec(
    name="broadcast_row_bias_add",
    axon_kernel=kernel_broadcast_row_bias_add,
    dim_vars=("m", "n", "k"),
    input_specs=(
        ("x", ("m", "k")),
        ("y", ("m", "k")),
        ("w", ("k", "n")),
    ),
    baseline_op=baseline_op,
    make_inputs=make_inputs,
    tile_args=("TILES_IN_BLOCK_M", "TILES_IN_BLOCK_N", "TILES_IN_BLOCK_K"),
)
