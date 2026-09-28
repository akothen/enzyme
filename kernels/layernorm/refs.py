"""References + input generation for `layernorm`. Axon-free: importable with
numpy + torch + ml_dtypes only, so nkilib's harness loads this file directly
(the generated `_case.py` reads torch_ref/baseline_op/make_inputs as module
attrs) with no axon/nkipy import shims. It is loaded standalone by path, so it
must not use relative imports.

`torch_ref` is pure math: the generated harness coerces every input to an fp32
torch tensor before calling it, and casts the output to the case dtype after.
The EPS here must match kernel.py.
"""

import numpy as np
import torch
from ml_dtypes import bfloat16

EPS = 1e-6


def baseline_op(x):
    mean = np.mean(x, axis=1, keepdims=True)
    centered = x - mean
    var = np.mean(np.square(centered), axis=1, keepdims=True)
    return centered / np.sqrt(var + EPS)


def torch_ref(x):
    # Bare LayerNorm (no affine), mean-based, fp32-accumulate; the wrapper casts
    # the output to the case dtype. Matches kernel.py's math and EPS.
    mean = torch.mean(x, dim=1, keepdim=True)
    centered = x - mean
    var = torch.mean(centered * centered, dim=1, keepdim=True)
    return centered / torch.sqrt(var + EPS)


def make_inputs(*, m, n, dtype=bfloat16, rng):
    # Output is normalized (~unit scale), so unit-gaussian inputs need no scaling.
    return (rng.standard_normal((m, n)).astype(dtype),)
