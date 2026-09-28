"""The Axon math for `fused_adam` (axon-side only; nkilib's harness never
imports this module).

Fused Adam (L2-regularized variant, no AMSGrad). The three scalar inputs come
in at `(P, 1)` and broadcast across the free dim, so the synthesizer can lower
the multiplications to `tensor_scalar`. beta1 / beta2 / eps are baked in as
module-level constants (the `KernelSpec` has no slot for kernel hyperparameters
and the math is invariant under any fixed choice for synthesis purposes).
"""

import os

# Match the env block used by other kernels.
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

from axon import AxonArray

# Adam hyperparameters — fixed at the spec level. See module docstring.
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
ONE_MINUS_BETA1 = 1.0 - BETA1
ONE_MINUS_BETA2 = 1.0 - BETA2


def kernel_fused_adam(
    param: AxonArray,
    grad: AxonArray,
    exp_avg: AxonArray,
    exp_avg_sq: AxonArray,
    step_size: AxonArray,
    inv_bc2_sqrt: AxonArray,
    wd: AxonArray,
) -> tuple[AxonArray, AxonArray, AxonArray]:
    # Adam L2: grad' = grad + wd * param
    grad_p = grad + wd * param
    # First/second moment updates.
    exp_avg_new = BETA1 * exp_avg + ONE_MINUS_BETA1 * grad_p
    exp_avg_sq_new = BETA2 * exp_avg_sq + ONE_MINUS_BETA2 * (grad_p * grad_p)
    # denom = sqrt(v) * inv_bc2_sqrt + eps; update = m / denom * step_size.
    denom = exp_avg_sq_new.sqrt() * inv_bc2_sqrt + EPS
    update = exp_avg_new / denom * step_size
    param_new = param - update
    return (param_new, exp_avg_new, exp_avg_sq_new)
