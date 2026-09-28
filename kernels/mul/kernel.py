"""The Axon math for `mul` (axon-side only; nkilib's harness never imports
this module)."""

import os

# See silu.py for why most of the legacy NEURON_CC_FLAGS env block is gone:
# the --internal-tensorizer-opt-level=2 setting collided with the =nki
# opt-level nki.baremetal already passes, manifesting on TRN2 as
# [NCC_ILSX902] LowerShardAxis errors. Keep only the XLA debug toggles.
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

from axon import AxonArray


def kernel_tensor_mul(x: AxonArray, y: AxonArray) -> AxonArray:
    return x * y
