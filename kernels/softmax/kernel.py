"""The Axon math for `softmax` (axon-side only; nkilib's harness never imports
this module)."""

import os

# See silu.py: trimmed env block to the XLA debug toggles.
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

from axon import AxonArray


def kernel_softmax(x: AxonArray) -> AxonArray:
    ex = x.exp()
    den = ex.sum(axis=1, keep_dims=True)
    return ex / den
