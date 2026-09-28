"""Kernel package for `rmsnorm` (sum-based RMS normalization).

Layout (the standard directory-kernel split):
  kernel.py — the Axon math (imports axon; axon-runtime only)
  refs.py   — baseline_op / torch_ref / make_inputs (numpy/torch/ml_dtypes
              only; loaded directly by nkilib's harness, no import shims)
  __init__  — assembles SPEC (what `axon kernels/rmsnorm` loads)
"""

from axon.kernel_spec import KernelSpec

from .kernel import kernel_rmsnorm
from .refs import baseline_op, make_inputs, torch_ref

SPEC = KernelSpec(
    name="rmsnorm",
    axon_kernel=kernel_rmsnorm,
    dim_vars=("m", "n"),
    input_specs=(("x", ("m", "n")),),
    baseline_op=baseline_op,
    torch_ref=torch_ref,
    make_inputs=make_inputs,
    tile_args=("TILES_IN_BLOCK_M", "TILES_IN_BLOCK_N"),
)
