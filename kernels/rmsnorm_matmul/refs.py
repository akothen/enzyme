"""References + input generation for `rmsnorm_matmul`. Axon-free: importable
with numpy + torch + ml_dtypes only, so nkilib's harness loads this file
directly (the generated `_case.py` reads baseline_op/make_inputs as module
attrs) with no axon/nkipy import shims. It is loaded standalone by path, so
it must not use relative imports.

`bfloat16` comes from ml_dtypes rather than nkipy.core.language: nkipy's
bfloat16 is the same numpy bf16 dtype (np.dtype equality verified on-device),
and ml_dtypes is what the generated harness already imports.
"""

import numpy as np
from ml_dtypes import bfloat16


def baseline_op(x, w):
    # SUM-based RMS (matches the kernel, which uses sum(x^2) — AxonArray has
    # no mean; see the NOTE in kernel.py). A mean-based reference here would
    # disagree with the kernel by sqrt(k) uniformly (~32x at k=1024), which
    # is exactly the err=170.3 every synthesized variant showed on-device.
    rms = np.sqrt(np.sum(np.square(x), axis=1, keepdims=True))
    norm = x / rms
    return np.matmul(norm, w)


def make_inputs(*, m, n, k, dtype=bfloat16, rng):
    return (
        rng.standard_normal((m, k)).astype(dtype),
        rng.standard_normal((k, n)).astype(dtype),
    )
