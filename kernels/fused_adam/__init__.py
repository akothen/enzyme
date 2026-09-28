"""Kernel package for `fused_adam` (L2-regularized variant, no AMSGrad).

Modeled after `nkilib.experimental.optimizer.fused_adam.adam_kernel` in
`KaenaNeuronKernelLibrary`. The reference treats `step_size`, `inv_bc2_sqrt`,
and `wd_factor` as `[P_MAX, 1]` scalar broadcasts. This spec follows the same
contract: the three scalar inputs come in at `(P, 1)` and broadcast across
the free dim, so the synthesizer can lower the multiplications to
`tensor_scalar`.

Inputs default to bfloat16. The fp32 head-to-head (nkilib's `F32_PARAMS` /
`qwen_f32` roofline point) is produced from this same spec with
`axon run kernels/fused_adam --dtype fp32` — dtype is a pure `make_inputs`
concern here (no `nc_matmul`/psum path), so there is no separate fp32 spec.

Layout (the standard directory-kernel split):
  kernel.py — the Axon math (imports axon; axon-runtime only)
  refs.py   — baseline_op / torch_ref / make_inputs (numpy/torch/ml_dtypes
              only; loaded directly by nkilib's harness, no import shims)
  __init__  — assembles SPEC (what `axon kernels/fused_adam` loads)
"""

from axon.kernel_spec import KernelSpec

from .kernel import kernel_fused_adam
from .refs import baseline_op, make_inputs, torch_ref

SPEC = KernelSpec(
    name="fused_adam",
    axon_kernel=kernel_fused_adam,
    dim_vars=("p", "f"),
    # `1` (int) in a sym_shape is treated as a concrete dim by
    # `build_graph_from_kernel`; the equivalence checker sees IntVal(1) and
    # the codegen detects (P, 1) inputs to lower per-row scalars as
    # `tensor_scalar` with a (P_MAX, 1) SBUF tile.
    input_specs=(
        ("param", ("p", "f")),
        ("grad", ("p", "f")),
        ("exp_avg", ("p", "f")),
        ("exp_avg_sq", ("p", "f")),
        ("step_size", ("p", 1)),
        ("inv_bc2_sqrt", ("p", 1)),
        ("wd", ("p", 1)),
    ),
    baseline_op=baseline_op,
    torch_ref=torch_ref,
    make_inputs=make_inputs,
    tile_args=("TILES_IN_BLOCK_M", "TILES_IN_BLOCK_N"),
)
