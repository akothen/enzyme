"""The Axon math for `broadcast_row_bias_add` (axon-side only; nkilib's harness
never imports this module)."""

from axon import AxonArray


def kernel_broadcast_row_bias_add(
    x: AxonArray, y: AxonArray, w: AxonArray
) -> AxonArray:
    bias = y.sum(axis=1, keep_dims=True)
    bias_b = bias.broadcast_like(x)
    z = x + bias_b
    return z @ w
