"""Kernel package for `part_relu_mlp` (partial ReLU-gated MLP: two projections,
relu gate, elementwise product).

Layout (the standard directory-kernel split):
  kernel.py — the Axon math (imports axon; axon-runtime only)
  refs.py   — baseline_op / make_inputs (numpy/torch/ml_dtypes only; loaded
              directly by nkilib's harness, no import shims)
  __init__  — assembles SPEC (what `axon kernels/part_relu_mlp` loads)
"""

from axon.kernel_spec import KernelSpec

from .kernel import kernel_relu_mlp_part
from .refs import baseline_op, make_inputs

SPEC = KernelSpec(
    name="part_relu_mlp",
    axon_kernel=kernel_relu_mlp_part,
    dim_vars=("m", "n", "k"),
    input_specs=(
        ("x", ("m", "k")),
        ("w1", ("k", "n")),
        ("w2", ("k", "n")),
    ),
    baseline_op=baseline_op,
    make_inputs=make_inputs,
    tile_args=("TILES_IN_BLOCK_M", "TILES_IN_BLOCK_N", "TILES_IN_BLOCK_K"),
    tile_options=(1, 2, 4, 8, 16),
)
