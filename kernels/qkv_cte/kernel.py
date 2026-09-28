"""The Axon math for `qkv_cte` (axon-side only; nkilib's harness never imports
this module).

QKV Context-Encoding (CTE) projection core, single-core (LNC=1) path:
residual add -> RMSNorm over the hidden dim -> fused QKV projection.

    h    = x + mlp_prev + attention_prev
    out  = (h / sqrt(mean(h^2) + eps)) @ w

Dimensions (flattened): M = B*S tokens, K = H hidden, N = I fused-QKV dim.
"""

from axon import AxonArray

EPS = 1e-6


def kernel_qkv_cte(
    x: AxonArray,
    mlp_prev: AxonArray,
    attention_prev: AxonArray,
    w: AxonArray,
) -> AxonArray:
    h = x + mlp_prev + attention_prev
    mean = (h * h).sum(axis=1, keep_dims=True) * (1.0 / x.shape[1])
    norm = h / (mean + EPS).sqrt()
    return norm @ w
