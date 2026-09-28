"""The Axon math for `relu_matmul` (axon-side only; nkilib's harness never
imports this module)."""

from axon import AxonArray


def kernel_relu_matmul(x: AxonArray, w: AxonArray) -> AxonArray:
    return x.relu() @ w
