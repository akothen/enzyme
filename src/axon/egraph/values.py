"""Define egglog value sorts shared by tensor and ISA input constructors."""

from __future__ import annotations

from egglog import Expr, i64Like, method

from egglog import StringLike  # isort: skip


class Dim(Expr):
    """One tensor dimension: a concrete literal or a named symbolic size."""

    @method(egg_fn="axDimLit")
    @classmethod
    def lit(cls, value: i64Like) -> Dim: ...

    @method(egg_fn="axDimSym")
    @classmethod
    def sym(cls, name: StringLike) -> Dim: ...


class Shape(Expr):
    """A dimension list built outermost-first with ``cons`` appending."""

    @method(egg_fn="axShapeNil")
    @classmethod
    def nil(cls) -> Shape: ...

    @method(egg_fn="axShapeCons")
    def cons(self, dim: Dim) -> Shape: ...
