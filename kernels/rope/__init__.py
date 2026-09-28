"""Kernel package for `rope` (math-only Rotary Position Embedding).

Layout (the standard directory-kernel split):
  kernel.py — the Axon math (imports axon; axon-runtime only)
  refs.py   — baseline_op / torch_ref / make_inputs (numpy/torch/ml_dtypes
              only; loaded directly by nkilib's harness, no import shims)
  __init__  — assembles SPEC (what `axon kernels/rope` loads)
"""

from axon.kernel_spec import KernelSpec

from .kernel import kernel_rope
from .refs import baseline_op, make_inputs, torch_ref

SPEC = KernelSpec(
    name="rope",
    axon_kernel=kernel_rope,
    dim_vars=("half_d", "free"),
    # half_d * 2 = d_head (typically 128); free = B * n_heads * S.
    input_specs=(
        ("x_even", ("half_d", "free")),
        ("x_odd", ("half_d", "free")),
        ("cos", ("half_d", "free")),
        ("sin", ("half_d", "free")),
    ),
    # half_d=128 (Axon codegen's TILE_M=128 floors the partition dim, so
    # half_d=64 would fail NUM_BLOCK_M > 0). free = 4096 matches
    # B*n_heads*S = 1*32*128 = 4096 in the hand-side parametrize entry.
    baseline_op=baseline_op,
    torch_ref=torch_ref,
    make_inputs=make_inputs,
    tile_args=("TILES_IN_BLOCK_M", "TILES_IN_BLOCK_N"),
)
