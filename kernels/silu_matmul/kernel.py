"""The Axon math for `silu_matmul` (axon-side only; nkilib's harness never
imports this module)."""

from axon import AxonArray


def kernel_silu_matmul(x: AxonArray, w: AxonArray) -> AxonArray:
    return x.silu() @ w
