"""References + input generation for `attention_nkilib`. Axon-free: importable
with numpy + torch + ml_dtypes only, so nkilib's harness loads this file
directly (the generated `_case.py` reads torch_ref/baseline_op/make_inputs as
module attrs) with no axon/nkipy import shims. It is loaded standalone by path,
so it must not use relative imports.

`make_inputs` draws uniform [0,1), matching nkilib's own `np_random_sample`
(test/integration/nkilib/utils/tensor_generators.py) rather than the normal
draw the other Axon specs use. Both sides of the head-to-head then see the
same input distribution, so the comparison isolates the kernel.

q is pre-scaled by a `softmax_scale` chosen from d (`_softmax_scale`): 1.0 for
d<=256 and 0.5 for d>=512, matching the `softmax_scale` nkilib's own
`attention_cte` parametrize rows pass at each head dim. Folding the scale into q
keeps the Axon graph at two matmuls (attention_cte applies its scale inside the
MM1 epilogue for free), so both sides still do the same arithmetic. The scale is
a free scalar and does not change the compute pattern, only the score magnitude
the bare-exp softmax sees.

CAUTION: the kernel's softmax has no row-max subtraction, so this spec is
numerically safe only while max(q @ k_t) stays under the fp32 exp limit of
88.72. With unscaled uniform [0,1) inputs, max(q @ k_t) grows about linearly
in d: ~26 at d=64, ~44 at d=128, ~82 at d=256, and ~152 at d=512, which
overflows. So d<=256 runs unscaled and d>=512 takes the 0.5 scale (~76 at
d=512).

s barely moves that bound (max over more samples adds ~2 at s=8192), and the
softmax denominator stays well under fp32 max: s * exp(76) ~ 1e37 at s=8192.
"""

import numpy as np
import torch
from ml_dtypes import bfloat16


def baseline_op(q, k_t, v):
    qk = np.matmul(q, k_t)
    exp_qk = np.exp(qk)
    probs = exp_qk / np.sum(exp_qk, axis=1, keepdims=True)
    return np.matmul(probs, v)


def torch_ref(q, k_t, v):
    # Bare exp/sum softmax, matching the kernel (no max subtraction); the
    # wrapper casts the output to the case dtype.
    exp_qk = torch.exp(torch.matmul(q, k_t))
    probs = exp_qk / torch.sum(exp_qk, dim=1, keepdim=True)
    return torch.matmul(probs, v)


def _softmax_scale(d):
    # 1.0 keeps max(q@k_t) ~44 at d=128 (safe for the no-row-max softmax);
    # d>=512 needs 0.5 to stay under the fp32 exp limit. See module docstring.
    return 0.5 if d >= 512 else 1.0


def make_inputs(*, s, d, dtype=bfloat16, rng):
    # Uniform [0,1), matching nkilib's np_random_sample. q is (s, d), k arrives
    # pre-transposed as (d, s) per attention_cte's tp_k=False layout, v is (s, d).
    # q carries the softmax scale so the graph stays at two matmuls.
    return (
        (rng.random((s, d)) * _softmax_scale(d)).astype(dtype),
        rng.random((d, s)).astype(dtype),
        rng.random((s, d)).astype(dtype),
    )
