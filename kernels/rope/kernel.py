"""The Axon math for `rope`: math-only Rotary Position Embedding (RoPE).

Modeled after `nkilib.core.embeddings.rope.RoPE` in
`KaenaNeuronKernelLibrary`. The hand kernel does an extra DMA-relayout pass
to move between interleaved and split layouts; here we present Axon with
already-split inputs and ask only for the pointwise rotation math:

    out_even = x_even * cos - x_odd * sin
    out_odd  = x_odd  * cos + x_even * sin

`cos` and `sin` are pre-tiled to the full free-axis extent (the hand kernel
broadcasts them across `n_heads`). The free axis flattens
`(B, n_heads, S)` to a single dim so the rank-2 elementwise codegen path
applies.

(axon-side only; nkilib's harness never imports this module.)
"""

import os

os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

from axon import AxonArray


def kernel_rope(
    x_even: AxonArray,
    x_odd: AxonArray,
    cos: AxonArray,
    sin: AxonArray,
) -> tuple[AxonArray, AxonArray]:
    # All inputs share shape (half_d, free); pointwise rotation only.
    out_even = x_even * cos - x_odd * sin
    out_odd = x_odd * cos + x_even * sin
    return (out_even, out_odd)
