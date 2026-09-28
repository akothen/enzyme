"""The Axon math for `attention_nkilib` (axon-side only; nkilib's harness never
imports this module).

This spec exists to make the Axon-vs-nkilib head-to-head apples-to-apples.
`kernels/attention` fuses three QKV projections that nkilib's `attention_cte`
never performs, so it does ~2.5x the arithmetic and the ratio is not a quality
measure. Here q, k_t, and v are inputs, exactly as `attention_cte` receives
them, so both sides compute the same two matmuls.
"""

from axon import AxonArray


def kernel_attention_nkilib(q: AxonArray, k_t: AxonArray, v: AxonArray) -> AxonArray:
    # k arrives pre-transposed (d, s), matching attention_cte's tp_k=False
    # layout, so no transpose is needed on either side.
    qk = q @ k_t
    ex = qk.exp()
    den = ex.sum(axis=1, keep_dims=True)
    probs = ex / den
    return probs @ v
