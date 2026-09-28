"""The Axon math for `relu_mlp` (axon-side only; nkilib's harness never imports
this module)."""

from axon import AxonArray


def kernel_relu_mlp_full(
    x: AxonArray, w1: AxonArray, w2: AxonArray, w3: AxonArray
) -> AxonArray:
    h1 = x @ w1
    h2 = x @ w2
    a = h1.relu()
    h3 = a * h2
    return h3 @ w3
