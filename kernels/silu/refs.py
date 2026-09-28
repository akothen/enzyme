"""References + input generation for `silu`. Axon-free: importable with
numpy + torch + ml_dtypes only, so nkilib's harness loads this file directly
(the generated `_case.py` reads torch_ref/baseline_op/make_inputs as module
attrs) with no axon/nkipy import shims. It is loaded standalone by path, so
it must not use relative imports.
"""

import numpy as np


def baseline_op(a):
    return a / (1 + np.exp(-a))


def make_inputs(*, m, n, dtype=np.float32, rng):
    return (rng.standard_normal((m, n)).astype(dtype),)
