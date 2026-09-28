"""Kernel package for `attention_nkilib` (softmax(q @ k_t) @ v).

Layout (the standard directory-kernel split):
  kernel.py — the Axon math (imports axon; axon-runtime only)
  refs.py   — baseline_op / torch_ref / make_inputs (numpy/torch/ml_dtypes
              only; loaded directly by nkilib's harness, no import shims)
  __init__  — assembles SPEC (what `axon kernels/attention_nkilib` loads)

This is the apples-to-apples twin of `kernels/attention`. It takes q, k_t, and
v as inputs, exactly the contract nkilib's `attention_cte` uses, so a
head-to-head measures the same arithmetic on both sides. `kernels/attention`
additionally fuses the three QKV projections and is the right spec when the
question is "can Axon fuse a whole attention block", not "how does Axon's
attention core compare".
"""

from axon.kernel_spec import KernelSpec

from .kernel import kernel_attention_nkilib
from .refs import baseline_op, make_inputs, torch_ref

SPEC = KernelSpec(
    name="attention_nkilib",
    axon_kernel=kernel_attention_nkilib,
    dim_vars=("s", "d"),
    input_specs=(
        ("q", ("s", "d")),
        ("k_t", ("d", "s")),
        ("v", ("s", "d")),
    ),
    baseline_op=baseline_op,
    torch_ref=torch_ref,
    make_inputs=make_inputs,
    # The assembler names tile params canonically (m/n/k/p), not after dim_vars,
    # and a chained multi-matmul graph takes all four. See assemble.py:489.
    tile_args=(
        "TILES_IN_BLOCK_M",
        "TILES_IN_BLOCK_N",
        "TILES_IN_BLOCK_K",
        "TILES_IN_BLOCK_P",
    ),
    tile_options=(1, 2, 4, 8, 16),
)
