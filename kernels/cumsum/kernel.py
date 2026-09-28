"""The Axon math for `cumsum` (axon-side only; nkilib's harness never imports
this module)."""

import os

os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

from axon import AxonArray


def kernel_cumsum(x: AxonArray) -> AxonArray:
    return x.cumsum(axis=-1)
