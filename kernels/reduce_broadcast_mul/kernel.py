"""The Axon math for `reduce_broadcast_mul` (axon-side only; nkilib's harness
never imports this module)."""

from axon import AxonArray


def kernel_reduce_broadcast_mul(x: AxonArray, y: AxonArray, w: AxonArray) -> AxonArray:
    rec = y.sum(axis=1, keep_dims=True)
    rec_b = rec.broadcast_like(x)
    z = x * rec_b
    return z @ w
