"""References + input generation for `linear_attention`. Axon-free: importable
with numpy + torch + ml_dtypes only, so the harness loads this file directly
with no axon/nkipy import shims. Loaded standalone by path, so no relative
imports.

`make_inputs` draws uniform [0,1) (matching attention_nkilib and nkilib's own
`np_random_sample`). There is no softmax / exp (linear attention), so unlike
attention_nkilib there is no fp32-exp overflow bound and no per-d softmax scale.

Cross-attention shape: q [sq, dk], k [sk, dk], v [sk, dv]. out = Q (Kᵀ V).
"""

import numpy as np
import torch
from ml_dtypes import bfloat16


def baseline_op(q, key, v):
    state = np.matmul(key.T, v)  # [dk, dv]
    return np.matmul(q, state)  # [sq, dv]


def torch_ref(q, key, v):
    state = torch.matmul(key.transpose(-1, -2), v)  # [dk, dv]
    return torch.matmul(q, state)  # [sq, dv]


def make_inputs(*, sq, sk, dk, dv, dtype=bfloat16, rng):
    # Uniform [0,1); q [sq, dk], key [sk, dk], v [sk, dv].
    return (
        rng.random((sq, dk)).astype(dtype),
        rng.random((sk, dk)).astype(dtype),
        rng.random((sk, dv)).astype(dtype),
    )
