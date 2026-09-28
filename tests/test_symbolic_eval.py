"""The symbolic evaluator must mean exactly what the compile rules mean.

``axon.symbolic_eval`` re-implements each supported operation so it can
evaluate a graph at one symbolic index. A mismatch with the compile rule would
let the evaluator prove (or refute) something the quantified encoding does not
agree with. For every non-reduction operation, these tests evaluate an
expression with the evaluator and ask the solver whether the compiled
semantics can disagree with that value at an in-bounds index; it must not.

The reduction-element rules are covered by unit tests on ``Evaluator``.
"""

from __future__ import annotations

import pytest
import z3

import axon.isa_semantics as isa
import axon.lang_semantics as lang
from axon.isa_semantics import (
    SymTensor,
    _public_binary,
    activation,
    compile_expr,
    dma_transpose,
    exponential,
    memset,
    nc_matmul,
    nc_transpose,
    nl,
    scalar_tensor_tensor,
    tensor_copy,
    tensor_reduce,
    tensor_scalar,
    tensor_tensor,
)
from axon.symbolic_eval import Evaluator, Outcome, Red, prove_by_evaluation

M, N = 3, 4


def _t(name: str, *shape: int) -> SymTensor:
    return SymTensor(name, shape=shape)


def _x() -> SymTensor:
    return _t("se_x", M, N)


def _y() -> SymTensor:
    return _t("se_y", M, N)


def _col() -> SymTensor:
    return _t("se_col", M, 1)


_CASES = {
    "tensor_copy": lambda: tensor_copy(dst=None, src=_x()),
    "nc_transpose": lambda: nc_transpose(dst=None, data=_x()),
    "dma_transpose_rank3": lambda: dma_transpose(
        dst=None, src=_t("se_r3", 2, 3, 4), axes=(1, 2, 0)
    ),
    "tensor_tensor_sub": lambda: tensor_tensor(
        dst=None, data1=_x(), data2=_y(), op=nl.subtract
    ),
    "tensor_tensor_div": lambda: tensor_tensor(
        dst=None, data1=_x(), data2=_y(), op=nl.divide
    ),
    "tensor_scalar_two_ops": lambda: tensor_scalar(
        dst=None,
        data=_x(),
        op0=nl.multiply,
        operand0=_col(),
        op1=nl.subtract,
        operand1=2.0,
        reverse1=True,
    ),
    "scalar_tensor_tensor": lambda: scalar_tensor_tensor(
        dst=None,
        data=_x(),
        op0=nl.add,
        operand0=1.5,
        op1=nl.maximum,
        operand1=_y(),
    ),
    "activation_scale_bias": lambda: activation(
        dst=None, op=nl.relu, data=_x(), bias=_col(), scale=3.0
    ),
    "activation_silu": lambda: activation(dst=None, op=nl.silu, data=_x()),
    "exponential": lambda: exponential(dst=None, src=_x(), max_value=_col()),
    "public_broadcast_add": lambda: _public_binary("add", _x(), _col()),
    "public_scalar_reverse": lambda: _public_binary("subtract", 2.0, _x()),
    "public_rsqrt": lambda: lang.rsqrt(_x()),
    "public_power": lambda: lang.power(_x(), 0.5),
    "where": lambda: lang.where(_x(), _y(), _col()),
    "expand_dims": lambda: lang.expand_dims(_x(), 1),
    "memset": lambda: memset(dst=_x(), value=4.0),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_evaluator_matches_compile_rule(name: str) -> None:
    tensor = _CASES[name]()
    sem = compile_expr(tensor.expr, {})
    ev = Evaluator()
    witness = [z3.Int(f"xc_w{i}") for i in range(sem.shape.rank)]
    value = ev.eval(tensor.expr, witness)
    assert not isinstance(value, Red)
    bounds = [
        z3.And(w >= 0, w < dim) for w, dim in zip(witness, sem.shape.dims, strict=True)
    ]
    assertions = [
        sem.ctx.as_formula(),
        sem.validity.as_formula(),
        *bounds,
        sem.fn(*witness) != value,
    ]
    assert isa._check_deterministic(assertions, 10000) == z3.unsat, name


def test_reduction_nests_flatten_and_scale_moves_inside() -> None:
    ev = Evaluator()
    inner = ev.reduction("add", [(z3.IntVal(0), z3.IntVal(3))], lambda ks: ks[0] * 1.0)
    outer = ev.reduction("add", [(z3.IntVal(0), z3.IntVal(2))], lambda ks: inner)
    assert isinstance(outer, Red) and len(outer.vars) == 2
    scaled = ev.mul(z3.Real("scale"), outer)
    assert isinstance(scaled, Red) and len(scaled.vars) == 2
    product = ev.mul(outer, inner)
    assert isinstance(product, Red) and len(product.vars) == 3


def test_equal_reductions_share_one_atom() -> None:
    ev = Evaluator()
    x = isa._tensor_function("V_atom_x", 2)

    def row_sum() -> Red:
        return ev.reduction(
            "add", [(z3.IntVal(0), z3.IntVal(4))], lambda ks: x(z3.IntVal(1), ks[0])
        )

    first, second = ev.scalar(row_sum()), ev.scalar(row_sum())
    assert z3.eq(first, second)
    other = ev.scalar(
        ev.reduction("max", [(z3.IntVal(0), z3.IntVal(4))], lambda ks: x(0, ks[0]))
    )
    assert not z3.eq(first, other)


def test_evaluation_refutes_only_exact_fragments() -> None:
    x = _x()
    lhs = tensor_tensor(dst=None, data1=x, data2=x, op=nl.add).expr
    rhs = tensor_tensor(dst=None, data1=x, data2=x, op=nl.multiply).expr
    shape = isa.ShapeExpr([z3.IntVal(M), z3.IntVal(N)])
    assert prove_by_evaluation(lhs, rhs, [], shape, 5000).outcome is Outcome.REFUTED

    a, b = _t("rf_a", N, M), _t("rf_b", N, M)
    mm = nc_matmul(dst=None, stationary=a, moving=b).expr
    wrong = nc_matmul(dst=None, stationary=b, moving=b).expr
    shape = isa.ShapeExpr([z3.IntVal(M), z3.IntVal(M)])
    # A reduction was abstracted, so a failed proof is never a refutation.
    assert prove_by_evaluation(mm, wrong, [], shape, 5000).outcome is Outcome.UNKNOWN


def test_unsupported_ops_fall_back() -> None:
    x = _x()
    lhs = lang.rms_norm(x, _t("rn_w", M, N), axis=1, n=N).expr
    shape = isa.ShapeExpr([z3.IntVal(M), z3.IntVal(N)])
    result = prove_by_evaluation(lhs, lhs, [], shape, 5000)
    assert result.outcome is Outcome.UNKNOWN
    assert "unsupported" in result.detail


def test_max_reduce_tensor_reduce_body_uses_fold_family() -> None:
    x = _x()
    reduced = tensor_reduce(dst=None, op=nl.maximum, data=x, axis=1)
    sem = compile_expr(reduced.expr, {})
    assert sem.reduction is not None
    assert sem.reduction.combine_op == "max"
