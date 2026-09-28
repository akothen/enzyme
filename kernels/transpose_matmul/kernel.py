"""The Axon math for `transpose_matmul` (axon-side only; nkilib's harness never
imports this module)."""

from axon import AxonArray


def kernel_transpose_matmul(x: AxonArray, w: AxonArray) -> AxonArray:
    xt = x.transpose()
    return xt @ w
