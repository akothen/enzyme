"""The Axon math for `part_silu_mlp` (axon-side only; nkilib's harness never
imports this module)."""

from axon import AxonArray


def kernel_silu_mlp_part(x: AxonArray, w1: AxonArray, w2: AxonArray) -> AxonArray:
    h1 = x @ w1
    h2 = x @ w2
    a = h1.silu()
    return a * h2
