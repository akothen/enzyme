"""Kernel package for `silu` (elementwise SiLU).

Layout (the standard directory-kernel split):
  kernel.py — the Axon math (imports axon; axon-runtime only)
  refs.py   — baseline_op / torch_ref / make_inputs (numpy/torch/ml_dtypes
              only; loaded directly by nkilib's harness, no import shims)
  __init__  — assembles SPEC (what `axon kernels/silu` loads)
"""

from axon.kernel_spec import KernelSpec

from .kernel import kernel_silu
from .refs import baseline_op, make_inputs

SPEC = KernelSpec(
    name="silu",
    axon_kernel=kernel_silu,
    dim_vars=("m", "n"),
    input_specs=(("x", ("m", "n")),),
    baseline_op=baseline_op,
    make_inputs=make_inputs,
    tile_args=("TILES_IN_BLOCK_M", "TILES_IN_BLOCK_N"),
)
