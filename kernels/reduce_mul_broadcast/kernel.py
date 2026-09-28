"""The Axon math for `reduce_mul_broadcast` (axon-side only; nkilib's harness
never imports this module)."""

from axon import AxonArray


def kernel_reduce_mul_broadcast(x: AxonArray, y: AxonArray, w: AxonArray) -> AxonArray:
    rec = y.sum(axis=1, keep_dims=True)
    z = x * rec
    z_b = z.broadcast_like(x)
    return z_b @ w
