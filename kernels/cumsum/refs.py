"""References + input generation for `cumsum`. Axon-free: importable with
numpy + torch + ml_dtypes only, so nkilib's harness loads this file directly
(the generated `_case.py` reads torch_ref/baseline_op/make_inputs as module
attrs) with no axon/nkipy import shims. It is loaded standalone by path, so
it must not use relative imports.

`torch_ref` is pure math: the generated harness coerces every input to an
fp32 torch tensor (case_gen's `_to_torch_f32`) before calling it, and casts
the output to the case dtype after.
"""

import numpy as np
import torch


def baseline_op(a):
    return np.cumsum(a, axis=-1)


def torch_ref(x):
    # fp32-accumulate reference; the wrapper casts the output to the case dtype.
    return torch.cumsum(x, dim=-1)


def make_inputs(*, m, n, dtype=np.float32, rng):
    return (rng.standard_normal((m, n)).astype(dtype),)
