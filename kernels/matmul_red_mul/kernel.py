"""The Axon math for `matmul_red_mul` (axon-side only; nkilib's harness never
imports this module)."""

from axon import AxonArray


def kernel_matmul_red_mul(x: AxonArray, y: AxonArray, w: AxonArray) -> AxonArray:
    rec = y.sum(axis=1, keep_dims=True)
    return (x * rec) @ w
