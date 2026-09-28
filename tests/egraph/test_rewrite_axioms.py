"""Rewrite obligations the semantics must admit, and ones it must reject.

Each case is one graph rewrite, stated as the graph before and the graph after,
and checked through the same strict API the e-graph uses
(``check_valid_and_equivalent``). Every case prints both graphs so a failing or
surprising verdict can be read directly from ``pytest -s`` output.

The cases cover the changes that borrow from TensorRight:

* broadcast reads clamp their own index instead of a global singleton axiom
  (the old axiom made an ``iota`` context contradictory, so anything proved);
* softmax is a row reduction, not an elementwise function (the old encoding
  proved that softmax commutes with transpose);
* symbolic evaluation with reduction elements proves transposes of matmuls,
  matmul associativity, summation exchange and scale lifting quantifier-free;
* max/min/product reductions, public reductions, and softmax get fold
  semantics, so a public op and its ISA lowering can be proved equal;
* the exp, silu, and rsqrt identities;
* ``dma_transpose`` rejects non-permutations and carries matmul folds.
"""

from __future__ import annotations

import pytest
import z3

import axon.isa_semantics as isa
import axon.lang_semantics as lang
from axon.egraph.rewrite_trace import format_sym_expr
from axon.isa_semantics import (
    EquivalenceVerdict,
    SymTensor,
    _public_binary,
    activation,
    check_valid_and_equivalent,
    dma_transpose,
    iota,
    memset,
    nc_matmul,
    nc_transpose,
    nl,
    reciprocal,
    tensor_partition_reduce,
    tensor_reduce,
    tensor_scalar,
    tensor_tensor,
)

# Rejections run the quantified fallback to its budget, so keep it short.
_PROVE_MS = 10000
_REJECT_MS = 1500

K, M, N, P = 8, 4, 6, 5


def _rewrite(
    before: SymTensor, after: SymTensor, timeout: int = _PROVE_MS
) -> EquivalenceVerdict:
    verdict = check_valid_and_equivalent(before, after, timeout=timeout)
    print("\n--- graph before rewrite")
    print(format_sym_expr(before.expr))
    print("+++ graph after rewrite")
    print(format_sym_expr(after.expr))
    print(
        f"=== verdict: proved={verdict.proved} stage={verdict.stage} {verdict.detail}"
    )
    return verdict


def _proved(before: SymTensor, after: SymTensor) -> EquivalenceVerdict:
    verdict = _rewrite(before, after)
    assert verdict.proved, verdict
    return verdict


def _rejected(before: SymTensor, after: SymTensor) -> EquivalenceVerdict:
    verdict = _rewrite(before, after, timeout=_REJECT_MS)
    assert not verdict.proved, verdict
    return verdict


def _t(name: str, *shape: int) -> SymTensor:
    return SymTensor(name, shape=shape)


# Soundness regressions ---------------------------------------------------------


def test_iota_with_singleton_axis_is_not_every_tensor() -> None:
    """A size-1 axis used to make the context unsat: iota equalled memset(7)."""
    out = _t("iota_out", 4, 1)
    before = iota(dst=out, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    after = memset(dst=out, value=7.0)
    _rejected(before, after)


def test_broadcast_of_symbolic_singleton_still_proves() -> None:
    """Clamping at the read keeps broadcasting of a maybe-1 dim provable."""
    m, n, one = z3.Ints("bc_m bc_n bc_one")
    x = SymTensor("bc_x", shape=(m, n))
    col = SymTensor("bc_col", shape=(m, one))
    before = _public_binary("add", x, col)
    after = _public_binary("add", col, x)
    _proved(before, after)


def test_softmax_does_not_commute_with_transpose() -> None:
    """Elementwise softmax proved T(softmax(x)) == softmax(T(x)); it is false."""
    x = _t("sm_sq", M, M)
    _rejected(lang.transpose(lang.softmax(x)), lang.softmax(lang.transpose(x)))


def test_softmax_is_not_a_column_softmax() -> None:
    x = _t("sm_col", M, N)
    row = lang.softmax(x)
    col_max = tensor_reduce(dst=None, op=nl.maximum, data=x, axis=0, keepdims=True)
    shifted = _public_binary("subtract", x, col_max)
    e = lang.exp(shifted)
    col_sum = tensor_reduce(dst=None, op=nl.add, data=e, axis=0, keepdims=True)
    _rejected(row, _public_binary("divide", e, col_sum))


def test_is_transpose_matmul_is_not_a_plain_matmul() -> None:
    a, b = _t("ist_a", K, M), _t("ist_b", K, N)
    plain = nc_matmul(dst=None, stationary=a, moving=b)
    flagged = nc_matmul(dst=None, stationary=a, moving=b, is_transpose=True)
    _rejected(plain, flagged)


# Transposes --------------------------------------------------------------------


def test_dma_transpose_of_matmul_swaps_operands() -> None:
    a, b = _t("dt_a", K, M), _t("dt_b", K, N)
    before = dma_transpose(
        dst=None, src=nc_matmul(dst=None, stationary=a, moving=b), axes=(1, 0)
    )
    after = nc_matmul(dst=None, stationary=b, moving=a)
    _proved(before, after)


def test_nc_transpose_and_dma_transpose_agree() -> None:
    a, b = _t("tt_a", K, M), _t("tt_b", K, N)
    mm = nc_matmul(dst=None, stationary=a, moving=b)
    _proved(nc_transpose(dst=None, data=mm), dma_transpose(dst=None, src=mm))


def test_dma_transpose_of_matmul_rejects_wrong_operand() -> None:
    a, b, c = _t("dw_a", K, M), _t("dw_b", K, N), _t("dw_c", K, M)
    before = dma_transpose(dst=None, src=nc_matmul(dst=None, stationary=a, moving=b))
    _rejected(before, nc_matmul(dst=None, stationary=b, moving=c))


def test_rank3_transpose_round_trip_is_identity() -> None:
    x = _t("r3", 2, 3, 4)
    once = dma_transpose(dst=None, src=x, axes=(2, 0, 1))
    back = dma_transpose(dst=None, src=once, axes=(1, 2, 0))
    _proved(back, isa.tensor_copy(dst=None, src=x))


def test_dma_transpose_rejects_non_permutation() -> None:
    x = _t("np_x", 3, 3)
    bad = dma_transpose(dst=None, src=x, axes=(0, 0))
    verdict = _rewrite(isa.tensor_copy(dst=None, src=x), bad, timeout=_REJECT_MS)
    assert not verdict.proved
    assert verdict.stage == "validity"


def test_transpose_builder_keeps_explicit_dst_shape() -> None:
    x = _t("dst_x", 3, 5)
    dst = _t("dst_out", 5, 3)
    assert [d.as_long() for d in nc_transpose(dst=dst, data=x).shape] == [5, 3]
    assert [d.as_long() for d in dma_transpose(dst=dst, src=x).shape] == [5, 3]


# Reductions --------------------------------------------------------------------


def test_matmul_associativity() -> None:
    x, y, z = _t("as_x", M, K), _t("as_y", K, N), _t("as_z", N, P)
    before = lang.matmul(lang.matmul(x, y), z)
    after = lang.matmul(x, lang.matmul(y, z))
    verdict = _proved(before, after)
    assert "symbolic_evaluation" in verdict.detail


def test_matmul_associativity_rejects_wrong_factor() -> None:
    x, y, z, w = _t("aw_x", M, K), _t("aw_y", K, N), _t("aw_z", N, P), _t("aw_w", N, P)
    before = lang.matmul(lang.matmul(x, y), z)
    _rejected(before, lang.matmul(x, lang.matmul(y, w)))


def test_summation_exchange() -> None:
    s = _t("ex_s", M, N)
    rows_then_parts = tensor_partition_reduce(
        dst=None,
        op=nl.add,
        data=tensor_reduce(dst=None, op=nl.add, data=s, axis=1, keepdims=True),
    )
    parts_then_rows = tensor_reduce(
        dst=None,
        op=nl.add,
        data=tensor_partition_reduce(dst=None, op=nl.add, data=s),
        axis=1,
        keepdims=True,
    )
    _proved(rows_then_parts, parts_then_rows)


def test_summation_exchange_rejects_mixed_combine_ops() -> None:
    s = _t("mx_s", M, N)
    sum_sum = tensor_partition_reduce(
        dst=None,
        op=nl.add,
        data=tensor_reduce(dst=None, op=nl.add, data=s, axis=1, keepdims=True),
    )
    max_sum = tensor_reduce(
        dst=None,
        op=nl.maximum,
        data=tensor_partition_reduce(dst=None, op=nl.add, data=s),
        axis=1,
        keepdims=True,
    )
    _rejected(sum_sum, max_sum)


def test_row_scale_lifts_into_matmul_stationary() -> None:
    a, b, s = _t("sc_a", K, M), _t("sc_b", K, N), _t("sc_s", M, 1)
    scaled_stationary = _public_binary("mul", a, nc_transpose(dst=None, data=s))
    before = nc_matmul(dst=None, stationary=scaled_stationary, moving=b)
    after = tensor_scalar(
        dst=None,
        data=nc_matmul(dst=None, stationary=a, moving=b),
        op0=nl.multiply,
        operand0=s,
    )
    _proved(before, after)


def test_max_reduction_has_fold_semantics() -> None:
    x = _t("mr_x", M, N)
    _proved(lang.max(x, axis=1), tensor_reduce(dst=None, op=nl.maximum, data=x, axis=1))
    _rejected(lang.max(x, axis=1), lang.min(x, axis=1))


def test_public_sum_equals_tensor_reduce() -> None:
    x = _t("ps_x", M, N)
    _proved(lang.sum(x, axis=1), tensor_reduce(dst=None, op=nl.add, data=x, axis=1))


def test_public_mean_is_sum_over_extent() -> None:
    x = _t("pm_x", M, 4)
    mean = lang.mean(x, axis=1, keepdims=True)
    scaled = tensor_scalar(
        dst=None,
        data=tensor_reduce(dst=None, op=nl.add, data=x, axis=1, keepdims=True),
        op0=nl.multiply,
        operand0=0.25,
    )
    _proved(mean, scaled)


def test_softmax_equals_its_isa_lowering() -> None:
    x = _t("sl_x", M, N)
    row_max = tensor_reduce(dst=None, op=nl.maximum, data=x, axis=1, keepdims=True)
    shifted = tensor_scalar(dst=None, data=x, op0=nl.subtract, operand0=row_max)
    e = activation(dst=None, op=nl.exp, data=shifted)
    total = tensor_reduce(dst=None, op=nl.add, data=e, axis=1, keepdims=True)
    lowered = tensor_scalar(
        dst=None, data=e, op0=nl.multiply, operand0=reciprocal(dst=None, data=total)
    )
    _proved(lang.softmax(x), lowered)


# Elementwise identities ----------------------------------------------------------


def test_silu_is_x_times_sigmoid() -> None:
    x = _t("si_x", M, N)
    silu = activation(dst=None, op=nl.silu, data=x)
    sig = activation(dst=None, op=nl.sigmoid, data=x)
    _proved(silu, tensor_tensor(dst=None, data1=x, data2=sig, op=nl.multiply))
    relu = activation(dst=None, op=nl.relu, data=x)
    _rejected(silu, tensor_tensor(dst=None, data1=x, data2=relu, op=nl.multiply))


def test_rsqrt_is_reciprocal_of_sqrt() -> None:
    x = _t("rs_x", M, N)
    _proved(lang.rsqrt(x), lang.reciprocal(lang.sqrt(x)))


def test_power_two_is_square() -> None:
    x = _t("pw_x", M, N)
    _proved(lang.power(x, 2.0), lang.square(x))


def test_exp_of_sum_is_product_of_exps() -> None:
    x, y = _t("ex_x", M, N), _t("ex_y", M, N)
    before = lang.exp(lang.add(x, y))
    _proved(before, _public_binary("mul", lang.exp(x), lang.exp(y)))
    _rejected(before, lang.add(lang.exp(x), lang.exp(y)))


@pytest.mark.parametrize("enabled", [True, False])
def test_quantified_path_still_decides_without_evaluation(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    """The evaluator is an accelerator: simple rewrites prove either way."""
    monkeypatch.setattr(isa, "SYMBOLIC_EVALUATION_ENABLED", enabled)
    x = _t(f"qp_x_{enabled}", M, N)
    before = nc_transpose(dst=None, data=nc_transpose(dst=None, data=x))
    verdict = _proved(before, isa.tensor_copy(dst=None, src=x))
    assert ("symbolic_evaluation" in verdict.detail) == enabled
