"""Define typed egglog constructors for tensor expressions.

The language is generic: one input constructor plus one constructor per
operand arity. The operation name and its canonical attribute payload are
plain strings, so e-node identity is (op, attrs, ordered children).
"""

from __future__ import annotations

from egglog import Expr, function

from egglog import StringLike  # isort: skip

from axon.egraph.values import Shape


class TensorExpr(Expr):
    """The single result sort of the tensor-level language."""


@function(egg_fn="axTInput")
def t_input(source_id: StringLike, shape: Shape) -> TensorExpr: ...


@function(egg_fn="axTOp1")
def t_op1(op: StringLike, attrs: StringLike, x: TensorExpr) -> TensorExpr: ...


@function(egg_fn="axTOp2")
def t_op2(
    op: StringLike, attrs: StringLike, x: TensorExpr, y: TensorExpr
) -> TensorExpr: ...
