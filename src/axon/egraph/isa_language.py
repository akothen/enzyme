"""Define typed egglog constructors for ISA expressions.

The language is generic: one input constructor plus one constructor per
operand arity. The operation name and its canonical attribute payload are
plain strings, so e-node identity is (op, attrs, ordered children).
"""

from __future__ import annotations

from egglog import Expr, function

from egglog import StringLike  # isort: skip

from axon.egraph.values import Shape


class IsaExpr(Expr):
    """The single result sort of the ISA-level language."""


@function(egg_fn="axIInput")
def i_input(source_id: StringLike, shape: Shape) -> IsaExpr: ...


@function(egg_fn="axIOp1")
def i_op1(op: StringLike, attrs: StringLike, x: IsaExpr) -> IsaExpr: ...


@function(egg_fn="axIOp2")
def i_op2(op: StringLike, attrs: StringLike, x: IsaExpr, y: IsaExpr) -> IsaExpr: ...


@function(egg_fn="axIOp3")
def i_op3(
    op: StringLike, attrs: StringLike, x: IsaExpr, y: IsaExpr, z: IsaExpr
) -> IsaExpr: ...
