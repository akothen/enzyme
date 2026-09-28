"""The Axon math for `attention` (axon-side only; nkilib's harness never
imports this module)."""

from axon import AxonArray


def kernel_attention(
    x: AxonArray, w_q: AxonArray, w_k: AxonArray, w_v: AxonArray
) -> AxonArray:
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v
    k_t = k.transpose()
    qk = q @ k_t
    ex = qk.exp()
    den = ex.sum(axis=1, keep_dims=True)
    probs = ex / den
    return probs @ v
