"""Kernel package for `silu_mlp` (full SiLU-gated MLP: two projections, silu
gate, elementwise product, output projection).

Layout (the standard directory-kernel split):
  kernel.py — the Axon math (imports axon; axon-runtime only)
  refs.py   — baseline_op / make_inputs (numpy/torch/ml_dtypes only; loaded
              directly by nkilib's harness, no import shims)
  __init__  — assembles SPEC (what `axon kernels/silu_mlp` loads)
"""

from axon.kernel_spec import KernelSpec

from .kernel import kernel_silu_mlp_full
from .refs import baseline_op, make_inputs

SPEC = KernelSpec(
    name="silu_mlp",
    axon_kernel=kernel_silu_mlp_full,
    dim_vars=("m", "n", "k", "p"),
    input_specs=(
        ("x", ("m", "k")),
        ("w1", ("k", "n")),
        ("w2", ("k", "n")),
        ("w3", ("n", "p")),
    ),
    baseline_op=baseline_op,
    make_inputs=make_inputs,
    tile_args=(
        "TILES_IN_BLOCK_M",
        "TILES_IN_BLOCK_N",
        "TILES_IN_BLOCK_K",
        "TILES_IN_BLOCK_P",
    ),
)
