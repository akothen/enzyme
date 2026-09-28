"""The Axon math for `rmsnorm_matmul` (axon-side only; nkilib's harness never
imports this module)."""

from axon import AxonArray


def kernel_rmsnorm_matmul(x: AxonArray, w: AxonArray) -> AxonArray:
    # NOTE: Uses sum(x^2) rather than the textbook mean(x^2), same as
    # kernels/rmsnorm (AxonArray exposes only sum). baseline_op in refs.py is
    # matched to this — keep the two in sync or the on-device correctness
    # check fails uniformly by sqrt(k).
    xx = x * x
    rec = xx.sum(axis=1, keep_dims=True)
    rms = rec.sqrt()
    norm = x / rms
    return norm @ w
