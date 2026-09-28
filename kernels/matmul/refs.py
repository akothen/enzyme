"""References + input generation for `matmul`. Axon-free: importable with
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


def baseline_op(x, w):
    return np.matmul(x, w)


def torch_ref(x, w):
    return x @ w


def make_inputs(*, m, n, k, dtype=bfloat16, rng):
    # Scale x by 1/sqrt(k) so x @ w is O(1) (w stays N(0,1)). Without this an
    # unscaled matmul output grows ~sqrt(k); in bf16 that yields absolute errors
    # ~0.4 at k=384, which swamp any reasonable atol at the near-zero output
    # elements (np.allclose's tolerance collapses to atol where |ref| ~ 0). The
    # scale only affects the correctness comparison -- latency is value-independent.
    scale = 1.0 / np.sqrt(k)
    return (
        (rng.standard_normal((m, k)) * scale).astype(dtype),
        rng.standard_normal((k, n)).astype(dtype),
    )
