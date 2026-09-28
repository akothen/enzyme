"""The Axon math for `softmax_mm` (axon-side only; nkilib's harness never
imports this module)."""

from axon import AxonArray


def kernel_softmax_matmul(x: AxonArray, w: AxonArray) -> AxonArray:
    ex = x.exp()
    den = ex.sum(axis=1, keep_dims=True)
    probs = ex / den
    return probs @ w
