"""The Axon math for `silu` (axon-side only; nkilib's harness never imports
this module)."""

import os

# NCC_ILSX902 LowerShardAxis triage: the legacy NEURON_CC_FLAGS block forced
# --internal-tensorizer-opt-level=2, which collides with the =nki opt-level
# nki.baremetal already passes. Most of the other vars are PyTorch-XLA
# framework debug flags that don't apply to the nki.baremetal direct path.
# Keeping only XLA_IR_DEBUG / XLA_HLO_DEBUG, which the compiler error message
# itself recommends for extra diagnostics.
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

from axon import AxonArray


def kernel_silu(x: AxonArray) -> AxonArray:
    return x.silu()
