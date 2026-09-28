"""The Axon math for `relu` (axon-side only; nkilib's harness never imports
this module)."""

import os

# See silu.py: trimmed env block to the XLA debug toggles. The legacy
# NEURON_CC_FLAGS=--internal-tensorizer-opt-level=2 collided with the =nki
# opt-level nki.baremetal already passes (NCC_ILSX902 LowerShardAxis on TRN2).
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

from axon import AxonArray


def kernel_relu(x: AxonArray) -> AxonArray:
    return x.relu()
