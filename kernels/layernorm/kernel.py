"""The Axon math for `layernorm` (axon-side only; nkilib's harness never imports
this module).

Bare LayerNorm over the last (hidden) axis: subtract the row mean, divide by the
row std. Affine gamma/beta are omitted to keep the head-to-head on the core
normalization (mirrors how rmsnorm's spec is scoped to the bare normalization).

    mean = sum(x) / n
    out  = (x - mean) / sqrt( sum((x-mean)^2)/n + eps )

The `1/n` mean (via x.shape[1]) is the same pattern qkv_cte uses for its RMS
denominator, so it stays inside the AxonArray op vocabulary.
"""

from axon import AxonArray

EPS = 1e-6


def kernel_layernorm(x: AxonArray) -> AxonArray:
    n = x.shape[1]
    mean = x.sum(axis=1, keep_dims=True) * (1.0 / n)
    centered = x - mean
    var = (centered * centered).sum(axis=1, keep_dims=True) * (1.0 / n)
    return centered / (var + EPS).sqrt()
