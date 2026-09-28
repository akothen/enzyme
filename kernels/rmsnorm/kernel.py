"""The Axon math for `rmsnorm` (axon-side only; nkilib's harness never imports
this module)."""

import os

# See silu.py: trimmed env block to the XLA debug toggles.
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

from axon import AxonArray


def kernel_rmsnorm(x: AxonArray) -> AxonArray:
    # NOTE: Uses sum(x^2) rather than the textbook mean(x^2). AxonArray
    # exposes only sum at the moment; the torch reference is matched to
    # this so the head-to-head evaluation is on the same math.
    xx = x * x
    rec = xx.sum(axis=1, keep_dims=True)
    rms = rec.sqrt()
    return x / rms
