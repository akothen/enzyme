"""Kernel package for `matmul_red_div` (row-sum divide then matmul).

Layout (the standard directory-kernel split):
  kernel.py — the Axon math (imports axon; axon-runtime only)
  refs.py   — baseline_op / make_inputs (numpy/torch/ml_dtypes only;
              loaded directly by nkilib's harness, no import shims)
  __init__  — assembles SPEC (what `axon kernels/matmul_red_div` loads)
"""

from axon.kernel_spec import KernelSpec

from .kernel import kernel_matmul_red_div
from .refs import baseline_op, make_inputs

SPEC = KernelSpec(
    name="matmul_red_div",
    axon_kernel=kernel_matmul_red_div,
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
