"""References + input generation for `qkv_cte`. Axon-free: importable with
numpy + torch + ml_dtypes only, so nkilib's harness loads this file directly
(the generated `_case.py` reads torch_ref/baseline_op/make_inputs as module
attrs) with no axon/nkipy import shims. It is loaded standalone by path, so it
must not use relative imports.

`torch_ref` is pure math: the generated harness coerces every input to an
fp32 torch tensor (case_gen's `_to_torch_f32`) before calling it, and casts
the output to the case dtype after.

`bfloat16` comes from ml_dtypes rather than nkipy.core.language: nkipy's
bfloat16 is the same numpy bf16 dtype (np.dtype equality verified on-device),
and ml_dtypes is what the generated harness already imports.
"""

import numpy as np
import torch
from ml_dtypes import bfloat16

EPS = 1e-6


def baseline_op(x, mlp_prev, attention_prev, w):
    h = x + mlp_prev + attention_prev
    rms = np.sqrt(np.mean(np.square(h), axis=1, keepdims=True) + EPS)
    norm = h / rms
    return np.matmul(norm, w)


def torch_ref(x, mlp_prev, attention_prev, w):
    # MEAN-based RMS (matches the kernel's `* (1.0 / x.shape[1])`),
    # fp32-accumulate; the wrapper casts the output to the case dtype.
    h = x + mlp_prev + attention_prev
    mean = torch.mean(h * h, dim=1, keepdim=True)
    norm = h / torch.sqrt(mean + EPS)
    return torch.matmul(norm, w)


def make_inputs(*, m, n, k, dtype=bfloat16, rng):
    # Draw order (x, mlp_prev, attention_prev, w) matches the old
    # eval/torch_refs/qkv_cte.py make_inputs. The weight is scaled by 1/sqrt(k)
    # so the projection output stays in a sane range.
    return (
        rng.standard_normal((m, k)).astype(dtype),  # x          (B*S, H)
        rng.standard_normal((m, k)).astype(dtype),  # mlp_prev   (B*S, H)
        rng.standard_normal((m, k)).astype(dtype),  # attention_prev (B*S, H)
        (rng.standard_normal((k, n)) / np.sqrt(k)).astype(dtype),  # weights (H, I)
    )
