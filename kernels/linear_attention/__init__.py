"""Kernel package for `linear_attention` — the unnormalized core of global
(non-causal) linear attention with an identity feature map: `out = Q (Kᵀ V)`.

Layout (the standard directory-kernel split):
  kernel.py — the Axon math (imports axon; axon-runtime only)
  refs.py   — baseline_op / torch_ref / make_inputs (numpy/torch/ml_dtypes only)
  __init__  — assembles SPEC (what `axon kernels/linear_attention` loads)
"""

from axon.kernel_spec import KernelSpec

from .kernel import kernel_linear_attention
from .refs import baseline_op, make_inputs, torch_ref

SPEC = KernelSpec(
    name="linear_attention",
    axon_kernel=kernel_linear_attention,
    dim_vars=("sq", "sk", "dk", "dv"),
    input_specs=(
        ("q", ("sq", "dk")),
        # named `key`, not `k`: `k` collides with the emitter's K-block loop var.
        ("key", ("sk", "dk")),
        ("v", ("sk", "dv")),
    ),
    baseline_op=baseline_op,
    torch_ref=torch_ref,
    make_inputs=make_inputs,
    # Chained multi-matmul graph (state = kᵀ@v, then q@state). The assembler names
    # tile params canonically m/n/k/p; a chained matmul graph takes all four.
    tile_args=(
        "TILES_IN_BLOCK_M",
        "TILES_IN_BLOCK_N",
        "TILES_IN_BLOCK_K",
        "TILES_IN_BLOCK_P",
    ),
    tile_options=(1, 2, 4, 8, 16),
)
