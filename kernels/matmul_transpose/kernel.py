"""The Axon math for `matmul_transpose` (axon-side only; nkilib's harness never
imports this module)."""

from axon import AxonArray


def kernel_matmul_transpose(x: AxonArray, w: AxonArray) -> AxonArray:
    z = x @ w
    return z.transpose()
