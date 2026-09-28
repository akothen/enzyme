"""The Axon math for `matmul` (axon-side only; nkilib's harness never imports
this module)."""

from axon import AxonArray


def kernel_matmul(x: AxonArray, w: AxonArray) -> AxonArray:
    return x @ w
