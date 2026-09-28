"""References + input generation for `fused_adam`. Axon-free: importable with
numpy + torch + ml_dtypes only, so nkilib's harness loads this file directly
(the generated `_case.py` reads torch_ref/baseline_op/make_inputs as module
attrs) with no axon/nkipy import shims. It is loaded standalone by path, so it
must not use relative imports.

`torch_ref` is pure math: the generated harness coerces every input to an
fp32 torch tensor (case_gen's `_to_torch_f32`) before calling it, and casts
the outputs to the case dtype after.

`bfloat16` comes from ml_dtypes rather than nkipy.core.language: nkipy's
bfloat16 is the same numpy bf16 dtype (np.dtype equality verified on-device),
and ml_dtypes is what the generated harness already imports.
"""

import numpy as np
import torch
from ml_dtypes import bfloat16

# Adam hyperparameters — fixed at the spec level (see kernel.py).
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
ONE_MINUS_BETA1 = 1.0 - BETA1
ONE_MINUS_BETA2 = 1.0 - BETA2


def baseline_op(param, grad, exp_avg, exp_avg_sq, step_size, inv_bc2_sqrt, wd):
    grad_p = grad + wd * param
    exp_avg_new = BETA1 * exp_avg + ONE_MINUS_BETA1 * grad_p
    exp_avg_sq_new = BETA2 * exp_avg_sq + ONE_MINUS_BETA2 * (grad_p * grad_p)
    denom = np.sqrt(exp_avg_sq_new) * inv_bc2_sqrt + EPS
    update = exp_avg_new / denom * step_size
    param_new = param - update
    return param_new, exp_avg_new, exp_avg_sq_new


def torch_ref(param, grad, exp_avg, exp_avg_sq, step_size, inv_bc2_sqrt, wd):
    # fp32-accumulate reference; the wrapper casts the outputs to the case dtype.
    grad_p = grad + wd * param
    exp_avg_new = BETA1 * exp_avg + ONE_MINUS_BETA1 * grad_p
    exp_avg_sq_new = BETA2 * exp_avg_sq + ONE_MINUS_BETA2 * (grad_p * grad_p)
    denom = torch.sqrt(exp_avg_sq_new) * inv_bc2_sqrt + EPS
    update = exp_avg_new / denom * step_size
    param_new = param - update
    return (param_new, exp_avg_new, exp_avg_sq_new)


def make_inputs(*, p, f, dtype=bfloat16, rng):
    # The four full-shape tensors are (p, f); step_size / inv_bc2_sqrt / wd
    # are (p, 1) per-row scalars that the kernel broadcasts across the free
    # dim. Numpy broadcasts naturally so `baseline_op` gets the same answer.
    # `exp_avg_sq` is abs() so its sqrt is real; the three (p, 1) scalars are
    # constant fills (step_size=1e-3, inv_bc2_sqrt=1.0, wd=0.01). Draw order
    # (param, grad, exp_avg, exp_avg_sq, then the constant scalars) matches the
    # old eval/torch_refs/fused_adam.py make_inputs exactly.
    full = (p, f)
    scalar = (p, 1)
    return (
        rng.standard_normal(full).astype(dtype),
        rng.standard_normal(full).astype(dtype),
        rng.standard_normal(full).astype(dtype),
        np.abs(rng.standard_normal(full)).astype(dtype),
        np.full(scalar, 1e-3, dtype=dtype),
        np.full(scalar, 1.0, dtype=dtype),
        np.full(scalar, 0.01, dtype=dtype),
    )
