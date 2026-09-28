"""References + input generation for `rope`. Axon-free: importable with
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


def baseline_op(x_even, x_odd, cos, sin):
    out_even = x_even * cos - x_odd * sin
    out_odd = x_odd * cos + x_even * sin
    return out_even, out_odd


def torch_ref(x_even, x_odd, cos, sin):
    # fp32-accumulate reference; the wrapper casts the outputs to the case dtype.
    out_even = x_even * cos - x_odd * sin
    out_odd = x_odd * cos + x_even * sin
    return (out_even, out_odd)


def make_inputs(*, half_d, free, dtype=bfloat16, rng):
    # x_even/x_odd are gaussian activations; cos/sin are REAL rotation values
    # (cos=cos(theta), sin=sin(theta)) rather than independent gaussians. This
    # matches what the production kernel sees and keeps |cos|,|sin| <= 1, so the
    # bf16 products stay bounded and the output avoids the heavy catastrophic-
    # cancellation tail that independent-gaussian cos/sin manufacture (which no
    # reasonable bf16 tolerance can absorb under per-element np.allclose).
    shape = (half_d, free)
    x_even = rng.standard_normal(shape).astype(dtype)
    x_odd = rng.standard_normal(shape).astype(dtype)
    theta = rng.uniform(-np.pi, np.pi, shape)
    cos = np.cos(theta).astype(dtype)
    sin = np.sin(theta).astype(dtype)
    return (x_even, x_odd, cos, sin)
