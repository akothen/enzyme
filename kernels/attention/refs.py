"""References + input generation for `attention`. Axon-free: importable with
numpy + torch + ml_dtypes only, so nkilib's harness loads this file directly
(the generated `_case.py` reads baseline_op/make_inputs as module attrs) with
no axon/nkipy import shims. It is loaded standalone by path, so it must not use
relative imports.

`bfloat16` comes from ml_dtypes rather than nkipy.core.language: nkipy's
bfloat16 is the same numpy bf16 dtype (np.dtype equality verified on-device),
and ml_dtypes is what the generated harness already imports.
"""

import numpy as np
import torch
from ml_dtypes import bfloat16


def baseline_op(x, w_q, w_k, w_v):
    q = np.matmul(x, w_q)
    k = np.matmul(x, w_k)
    v = np.matmul(x, w_v)
    k_t = k.T
    qk = np.matmul(q, k_t)
    exp_qk = np.exp(qk)
    probs = exp_qk / np.sum(exp_qk, axis=1, keepdims=True)
    return np.matmul(probs, v)


def torch_ref(x, w_q, w_k, w_v):
    # Bare exp/sum softmax, matching the kernel (no max subtraction); the
    # wrapper casts the output to the case dtype.
    q = torch.matmul(x, w_q)
    k = torch.matmul(x, w_k)
    v = torch.matmul(x, w_v)
    exp_qk = torch.exp(torch.matmul(q, k.transpose(0, 1)))
    probs = exp_qk / torch.sum(exp_qk, dim=1, keepdim=True)
    return torch.matmul(probs, v)


def make_inputs(*, m, n, k, dtype=bfloat16, rng):
    # The kernel's softmax is a bare exp/sum with no max subtraction, so the
    # qk logits must arrive O(1) or exp overflows to an all-NaN reference.
    qk_scale = 1.0 / (np.sqrt(k) * n**0.25)  # keeps std(q @ k.T) ~= 1
    v_scale = 1.0 / np.sqrt(k)  # keeps std(v) ~= 1
    return (
        rng.standard_normal((m, k)).astype(dtype),
        (rng.standard_normal((k, n)) * qk_scale).astype(dtype),
        (rng.standard_normal((k, n)) * qk_scale).astype(dtype),
        (rng.standard_normal((k, n)) * v_scale).astype(dtype),
    )
