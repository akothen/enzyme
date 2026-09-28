"""References + input generation for `mul`. Axon-free: importable with
numpy + torch + ml_dtypes only, so nkilib's harness loads this file directly
(the generated `_case.py` reads torch_ref/baseline_op/make_inputs as module
attrs) with no axon/nkipy import shims. It is loaded standalone by path, so
it must not use relative imports.

`torch_ref` is pure math: the generated harness coerces every input to an
fp32 torch tensor (case_gen's `_to_torch_f32`) before calling it, and casts
the output to the case dtype after.

`bfloat16` comes from ml_dtypes rather than nkipy.core.language: nkipy's
bfloat16 is the same numpy bf16 dtype (np.dtype equality verified on-device),
and ml_dtypes is what the generated harness already imports.
"""

import numpy as np
from ml_dtypes import bfloat16


def baseline_op(x, y):
    return np.multiply(x, y)


def torch_ref(x, y):
    return x * y


def make_inputs(*, m, n, dtype=bfloat16, rng):
    return (
        rng.standard_normal((m, n)).astype(dtype),
        rng.standard_normal((m, n)).astype(dtype),
    )
