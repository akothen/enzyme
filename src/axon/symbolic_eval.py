"""Prove tensor equalities by symbolic evaluation at one symbolic access.

This is the proof strategy TensorRight uses, specialised to Axon's concrete
ranks. Instead of compiling every node to a quantified definition
(``forall i. V_node(i) == ...``) and leaving Z3 to instantiate the chain, both
expressions are *evaluated* at one symbolic output index. Index remapping
(transposes, broadcasts, copies) and elementwise arithmetic are composed in
Python, so the solver sees one quantifier-free formula over the input functions.

Reductions follow TensorRight's "reduction element": a reduction evaluates to a
``Red`` that holds its combine op, fresh reduction variables with their ranges,
and its body evaluated at those variables. Two reductions are equal when their
ranges agree and their bodies agree under some bijection of their reduction
variables. TensorRight takes that bijection from a user hint; here the prover
tries each permutation of the variables, which covers summation reordering and
matmul associativity without hints. The arithmetic rules on reduction elements
are TensorRight's (Fig. 9) plus additivity:

* ``v * Sum_X f  ->  Sum_X (v * f)`` and ``Sum_X f / v -> Sum_X (f / v)``
* ``Sum_X f * Sum_Y g  ->  Sum_{X,Y} (f * g)``
* ``Op_X (Op_Y f)  ->  Op_{X,Y} f`` for one combine op
* ``Sum_X f + Sum_X g  ->  Sum_X (f + g)`` when the ranges are identical

A reduction used any other way (``exp(Sum f)``, ``x - max f``) becomes an
opaque atom keyed by its canonical form, so two occurrences of the same
reduction are still the same term.

The prover covers a fragment of the registered operations. An expression that
leaves the fragment makes the prover return ``UNKNOWN``, and the caller falls
back to the quantified encoding, so the fragment can only add proofs. Every
supported operation mirrors its compile rule in ``isa_semantics`` or
``lang_semantics`` exactly; ``tests/test_symbolic_eval.py`` cross-checks them.
"""

from __future__ import annotations

import itertools
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import z3

from axon import isa_semantics as isa

_RED_PREFIX = "__red_"
_MAX_PERMUTED_VARS = 4


class Outcome(Enum):
    PROVED = "proved"
    REFUTED = "refuted"
    UNKNOWN = "unknown"


class Unsupported(Exception):
    """The expression leaves the evaluator's fragment."""


@dataclass(frozen=True)
class Red:
    """A reduction element: ``op`` over ``vars`` in ``[lows, highs)`` of ``body``."""

    op: str
    vars: tuple[z3.ArithRef, ...]
    lows: tuple[z3.ArithRef, ...]
    highs: tuple[z3.ArithRef, ...]
    body: z3.ArithRef


Value = Any  # z3.ArithRef | Red


def _is_red_var(node: z3.ExprRef) -> bool:
    return (
        z3.is_const(node)
        and node.decl().kind() == z3.Z3_OP_UNINTERPRETED
        and node.decl().name().startswith(_RED_PREFIX)
    )


def _red_vars_in(terms: list[z3.ExprRef]) -> list[z3.ArithRef]:
    """Reduction variables in first-appearance order of a fixed traversal."""
    found: list[z3.ArithRef] = []
    names: set[str] = set()
    seen: set[int] = set()
    stack = list(reversed(terms))
    while stack:
        node = stack.pop()
        key = node.get_id()
        if key in seen:
            continue
        seen.add(key)
        if _is_red_var(node):
            if node.decl().name() not in names:
                names.add(node.decl().name())
                found.append(node)
            continue
        if z3.is_app(node):
            stack.extend(reversed(node.children()))
    return found


def _normal(term: z3.ExprRef) -> z3.ExprRef:
    return z3.simplify(term, som=True)


class Evaluator:
    """Evaluate ``SymExpr`` trees at symbolic indices."""

    def __init__(self) -> None:
        self.facts: list[z3.BoolRef] = []
        self.exact = True
        self._next_var = 0
        self._atoms: dict[Any, z3.FuncDeclRef] = {}
        self._shape_cache: dict[int, isa.ShapeSemantics] = {}
        self._key_cache: dict[int, Any] = {}
        self._memo: dict[tuple[Any, tuple[str, ...]], Value] = {}

    # Reduction elements -------------------------------------------------

    def fresh_var(self) -> z3.ArithRef:
        self._next_var += 1
        return z3.Int(f"{_RED_PREFIX}{self._next_var}")

    def reduction(
        self,
        op: str,
        ranges: list[tuple[z3.ArithRef, z3.ArithRef]],
        body_at: Callable[[list[z3.ArithRef]], Value],
    ) -> Red:
        self.exact = False
        variables = [self.fresh_var() for _ in ranges]
        body = body_at(variables)
        lows = tuple(low for low, _ in ranges)
        highs = tuple(high for _, high in ranges)
        if isinstance(body, Red):
            if body.op == op:
                return Red(
                    op,
                    (*variables, *body.vars),
                    (*lows, *body.lows),
                    (*highs, *body.highs),
                    body.body,
                )
            body = self.scalar(body)
        return Red(op, tuple(variables), lows, highs, body)

    def scalar(self, value: Value) -> z3.ArithRef:
        """A plain term for ``value``; a reduction becomes a canonical atom."""
        if not isinstance(value, Red):
            return value
        own = list(value.vars)
        own_names = {var.decl().name() for var in own}
        bound_terms = [*value.lows, *value.highs, value.body]
        free = [
            var
            for var in _red_vars_in(bound_terms)
            if var.decl().name() not in own_names
        ]
        substitution = [
            (var, z3.Int(f"__canon_own{index}")) for index, var in enumerate(own)
        ] + [(var, z3.Int(f"__canon_free{index}")) for index, var in enumerate(free)]
        canonical = [
            _normal(z3.substitute(term, *substitution))
            if substitution
            else _normal(term)
            for term in bound_terms
        ]
        key = (value.op, len(own), tuple(term.sexpr() for term in canonical))
        atom = self._atoms.get(key)
        if atom is None:
            atom = z3.Function(
                f"__redatom_{len(self._atoms)}",
                *([z3.IntSort()] * len(free)),
                z3.RealSort(),
            )
            self._atoms[key] = atom
        return atom(*free) if free else atom()

    # Arithmetic on values -------------------------------------------------

    def add(self, lhs: Value, rhs: Value) -> Value:
        if (
            isinstance(lhs, Red)
            and isinstance(rhs, Red)
            and lhs.op == rhs.op == "add"
            and len(lhs.vars) == len(rhs.vars)
            and all(
                z3.eq(_normal(a), _normal(b))
                for a, b in zip(lhs.lows + lhs.highs, rhs.lows + rhs.highs, strict=True)
            )
        ):
            renamed = z3.substitute(rhs.body, *zip(rhs.vars, lhs.vars, strict=True))
            return Red("add", lhs.vars, lhs.lows, lhs.highs, lhs.body + renamed)
        return self.scalar(lhs) + self.scalar(rhs)

    def neg(self, value: Value) -> Value:
        if isinstance(value, Red) and value.op == "add":
            return Red("add", value.vars, value.lows, value.highs, -value.body)
        return -self.scalar(value)

    def mul(self, lhs: Value, rhs: Value) -> Value:
        lhs_sum = isinstance(lhs, Red) and lhs.op == "add"
        rhs_sum = isinstance(rhs, Red) and rhs.op == "add"
        if lhs_sum and rhs_sum:
            return Red(
                "add",
                lhs.vars + rhs.vars,
                lhs.lows + rhs.lows,
                lhs.highs + rhs.highs,
                lhs.body * rhs.body,
            )
        if lhs_sum:
            return Red(
                "add", lhs.vars, lhs.lows, lhs.highs, lhs.body * self.scalar(rhs)
            )
        if rhs_sum:
            return Red(
                "add", rhs.vars, rhs.lows, rhs.highs, self.scalar(lhs) * rhs.body
            )
        return self.scalar(lhs) * self.scalar(rhs)

    def div(self, lhs: Value, rhs: Value) -> Value:
        divisor = self.scalar(rhs)
        if isinstance(lhs, Red) and lhs.op == "add":
            # If(d == 0, 0, S / d) == Sum If(d == 0, 0, f / d): d is free of X.
            return Red(
                "add",
                lhs.vars,
                lhs.lows,
                lhs.highs,
                isa._safe_divide(lhs.body, divisor),
            )
        return isa._safe_divide(self.scalar(lhs), divisor)

    def binary(self, op: Any, lhs: Value, rhs: Value) -> Value:
        """``isa._apply_binary`` with reduction-aware ring operations."""
        name = isa._operand_to_expr(op)
        if name in ("add", "plus"):
            return self.add(lhs, rhs)
        if name in ("subtract", "sub"):
            return self.add(lhs, self.neg(rhs))
        if name in ("multiply", "mul"):
            return self.mul(lhs, rhs)
        if name in ("divide", "div"):
            return self.div(lhs, rhs)
        return isa._apply_binary(op, self.scalar(lhs), self.scalar(rhs))

    def activation(self, op: Any, value: Value) -> z3.ArithRef:
        result, facts = isa._apply_activation(op, self.scalar(value))
        self.facts.extend(facts)
        return result

    # Tree walking ------------------------------------------------------------

    def shape(self, expr: isa.SymExpr) -> isa.ShapeExpr:
        return isa.compile_shape_expr(expr, self._shape_cache).shape

    def read(
        self, expr: isa.SymExpr, out_shape: isa.ShapeExpr, idx: list[z3.ArithRef]
    ) -> Value:
        """``_call_broadcasted``: read ``expr`` at the broadcast of ``idx``."""
        holder = isa.ShapeSemantics(shape=self.shape(expr))
        mapped = isa._broadcast_indices(holder, out_shape, idx)  # type: ignore[arg-type]
        if not mapped:
            raise Unsupported("rank-0 operand")
        return self.eval(expr, mapped)

    def operand(
        self,
        expr: isa.SymExpr,
        const_key: str,
        index_key: str,
        out_shape: isa.ShapeExpr,
        idx: list[z3.ArithRef],
    ) -> Value:
        """``_operand_value``: a tensor operand or a scalar immediate."""
        if index_key in expr.attrs:
            return self.read(expr.inputs[int(expr.attrs[index_key])], out_shape, idx)
        return isa._as_scalar(expr.attrs.get(const_key))

    def eval(self, expr: isa.SymExpr, idx: list[z3.ArithRef]) -> Value:
        key = (
            isa._expr_structural_key(expr, self._key_cache),
            tuple(term.sexpr() for term in idx),
        )
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        value = self._eval(expr, list(idx))
        self._memo[key] = value
        return value

    def _eval(self, expr: isa.SymExpr, idx: list[z3.ArithRef]) -> Value:
        if expr.op == "input":
            rank = len(expr.shape)
            if rank == 0 or len(idx) != rank:
                raise Unsupported("input rank")
            return isa._tensor_function(f"V_{expr.name}", rank)(*idx)
        entry = isa._SEMANTICS.get(expr.op)
        if entry is None:
            raise Unsupported(f"no semantics for {expr.op}")
        handler = _handler_for(entry.compile_rule, expr.op)
        if handler is None:
            raise Unsupported(f"{expr.op} is outside the evaluator fragment")
        out_shape = self.shape(expr)
        if out_shape.rank == 0 or len(idx) != out_shape.rank:
            raise Unsupported("rank-0 or misaligned output")
        return handler(self, expr, out_shape, idx)


# Operation handlers. Each mirrors one compile rule.


def _h_copy(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    return ev.read(expr.inputs[0], out, idx)


def _h_nc_transpose(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    if out.rank < 2:
        raise Unsupported("transpose rank")
    return ev.eval(expr.inputs[0], [idx[1], idx[0], *idx[2:]])


def _h_dma_transpose(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    src = expr.inputs[0]
    axes = isa._dma_transpose_axes(expr.attrs.get("axes"), ev.shape(src).rank)
    if axes is None or len(axes) != out.rank:
        raise Unsupported("invalid dma_transpose axes")
    return ev.eval(src, [idx[axes.index(axis)] for axis in range(len(axes))])


def _h_tensor_tensor(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    lhs = ev.eval(expr.inputs[0], idx)
    rhs = ev.eval(expr.inputs[1], idx)
    return ev.binary(expr.attrs.get("op"), lhs, rhs)


def _h_tensor_scalar(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    data = ev.read(expr.inputs[0], out, idx)
    op0 = ev.operand(expr, "operand0_const", "operand0_input_index", out, idx)
    lhs, rhs = (op0, data) if expr.attrs.get("reverse0", False) else (data, op0)
    value = ev.binary(expr.attrs.get("op0"), lhs, rhs)
    if expr.attrs.get("op1") is not None:
        op1 = ev.operand(expr, "operand1_const", "operand1_input_index", out, idx)
        lhs, rhs = (op1, value) if expr.attrs.get("reverse1", False) else (value, op1)
        value = ev.binary(expr.attrs.get("op1"), lhs, rhs)
    return value


def _h_scalar_tensor_tensor(
    ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list
) -> Value:
    if "operand1_input_index" not in expr.attrs:
        raise Unsupported("scalar_tensor_tensor without operand1")
    data = ev.read(expr.inputs[0], out, idx)
    op0 = ev.operand(expr, "operand0_const", "operand0_input_index", out, idx)
    lhs, rhs = (op0, data) if expr.attrs.get("reverse0", False) else (data, op0)
    tmp = ev.binary(expr.attrs.get("op0"), lhs, rhs)
    op1 = ev.read(expr.inputs[int(expr.attrs["operand1_input_index"])], out, idx)
    lhs, rhs = (op1, tmp) if expr.attrs.get("reverse1", False) else (tmp, op1)
    return ev.binary(expr.attrs.get("op1"), lhs, rhs)


def _activation_pre(ev: Evaluator, expr: Any, shape: isa.ShapeExpr, idx: list) -> Value:
    scale = ev.operand(expr, "scale", "scale_input_index", shape, idx)
    bias = ev.operand(expr, "bias_const", "bias_input_index", shape, idx)
    data = ev.read(expr.inputs[0], shape, idx)
    return ev.add(ev.mul(data, scale), bias)


def _h_activation(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    return ev.activation(expr.attrs.get("op"), _activation_pre(ev, expr, out, idx))


def _h_exponential(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    src = ev.read(expr.inputs[0], out, idx)
    max_value = ev.operand(expr, "max_value", "max_input_index", out, idx)
    return ev.activation("exp", ev.add(src, ev.neg(max_value)))


def _h_reciprocal(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    return ev.activation("reciprocal", ev.read(expr.inputs[0], out, idx))


def _h_public_unary(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    value = ev.read(expr.inputs[0], out, idx)
    return ev.activation(expr.attrs.get("op", expr.op), value)


def _h_public_binary(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    op = expr.attrs.get("op", expr.op)
    if len(expr.inputs) >= 2:
        lhs = ev.read(expr.inputs[0], out, idx)
        rhs = ev.read(expr.inputs[1], out, idx)
    elif len(expr.inputs) == 1:
        lhs = ev.read(expr.inputs[0], out, idx)
        attrs = expr.attrs
        rhs = isa._as_scalar(
            attrs.get(
                "scalar",
                attrs.get("rhs", attrs.get("value", attrs.get("operand0_const", 0))),
            )
        )
        if attrs.get("reverse", False):
            lhs, rhs = rhs, lhs
    else:
        raise Unsupported("nullary public binary")
    return ev.binary(op, lhs, rhs)


def _h_where(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    cond = ev.scalar(ev.read(expr.inputs[0], out, idx))
    on_true = ev.scalar(ev.read(expr.inputs[1], out, idx))
    on_false = ev.scalar(ev.read(expr.inputs[2], out, idx))
    return z3.If(cond != 0, on_true, on_false)


def _h_select_reduce(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    pred = ev.scalar(ev.read(expr.inputs[0], out, idx)) != 0
    if expr.attrs.get("reverse_pred", False):
        pred = z3.Not(pred)
    on_true = ev.scalar(ev.read(expr.inputs[1], out, idx))
    on_false = ev.scalar(
        ev.operand(expr, "on_false_const", "on_false_input_index", out, idx)
    )
    return z3.If(pred, on_true, on_false)


def _h_expand_dims(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    axis = int(expr.attrs.get("axis", 0))
    norm_axis = axis + out.rank if axis < 0 else axis
    return ev.eval(expr.inputs[0], [dim for i, dim in enumerate(idx) if i != norm_axis])


def _h_memset(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    return isa._as_scalar(expr.attrs.get("value"))


def _h_iota(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    if out.rank != 2:
        raise Unsupported("iota rank")
    offset = int(expr.attrs.get("offset", 0))
    multiplier = int(expr.attrs.get("channel_multiplier", 0))
    step = isa._extract_iota_step(expr.attrs.get("pattern", []))
    return z3.ToReal(offset + idx[0] * multiplier + idx[1] * int(step))


def _h_nc_matmul(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    stationary, moving = expr.inputs
    s_shape, m_shape = ev.shape(stationary), ev.shape(moving)
    if (
        s_shape.rank != 2
        or m_shape.rank != 2
        or out.rank != 2
        or expr.attrs.get("is_transpose", False) is True
        or expr.attrs.get("accumulate") is True
    ):
        raise Unsupported("nc_matmul form")
    m, n = idx

    def body(ks: list[z3.ArithRef]) -> Value:
        return ev.mul(ev.eval(stationary, [ks[0], m]), ev.eval(moving, [ks[0], n]))

    return ev.reduction("add", [(z3.IntVal(0), s_shape.dims[0])], body)


def _h_public_matmul(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    a, b = expr.inputs
    a_shape = ev.shape(a)
    if a_shape.rank != 2 or ev.shape(b).rank != 2 or out.rank != 2:
        raise Unsupported("matmul rank")
    m, n = idx
    if bool(expr.attrs.get("transpose_x", False)):
        extent = a_shape.dims[0]

        def body(ks: list[z3.ArithRef]) -> Value:
            return ev.mul(ev.eval(a, [ks[0], m]), ev.eval(b, [ks[0], n]))

    else:
        extent = a_shape.dims[1]

        def body(ks: list[z3.ArithRef]) -> Value:
            return ev.mul(ev.eval(a, [m, ks[0]]), ev.eval(b, [ks[0], n]))

    return ev.reduction("add", [(z3.IntVal(0), extent)], body)


def _fold_reduce(
    ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list, combine: str
) -> Value:
    """``compile_rank2_fold_reduce``."""
    src = expr.inputs[0]
    shape = ev.shape(src)
    if shape.rank != 2 or isa.fold_family(1, isa.ReductionKind.REDUCE, combine) is None:
        raise Unsupported("reduce form")
    axis = isa._normalize_reduce_axis(expr.attrs.get("axis", 1), shape.rank)
    keep = bool(expr.attrs.get("keepdims", expr.attrs.get("keep_dims", False)))
    if axis == 1:
        row = idx[0]
        value = ev.reduction(
            combine,
            [(z3.IntVal(0), shape.dims[1])],
            lambda ks: ev.eval(src, [row, ks[0]]),
        )
    elif axis == 0:
        col = idx[1] if keep else idx[0]
        value = ev.reduction(
            combine,
            [(z3.IntVal(0), shape.dims[0])],
            lambda ks: ev.eval(src, [ks[0], col]),
        )
    else:
        raise Unsupported("reduce axis")
    if bool(expr.attrs.get("negate", False)):
        return ev.neg(value)
    return value


def _h_tensor_reduce(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    return _fold_reduce(
        ev, expr, out, idx, isa._normalize_combine_op(expr.attrs.get("op"))
    )


def _h_public_reduce(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    lang = sys.modules.get("axon.lang_semantics")
    if lang is None:
        raise Unsupported("lang semantics not loaded")
    shape = ev.shape(expr.inputs[0])
    axis = lang.public_reduce_single_axis(expr, shape.rank)
    if shape.rank != 2 or axis is None:
        raise Unsupported("public reduce form")
    combine = lang._PUBLIC_REDUCE_COMBINE.get(expr.op)
    if combine is not None:
        return _fold_reduce(ev, expr, out, idx, combine)
    if expr.op == "mean":
        total = _fold_reduce(ev, expr, out, idx, "add")
        return ev.div(total, z3.ToReal(shape.dims[axis]))
    raise Unsupported(f"public reduce {expr.op}")


def _h_activation_reduce(
    ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list
) -> Value:
    src = expr.inputs[0]
    shape = ev.shape(src)
    combine = isa._normalize_combine_op(expr.attrs.get("reduce_op"))
    if shape.rank != 2 or isa.fold_family(1, isa.ReductionKind.REDUCE, combine) is None:
        raise Unsupported("activation_reduce form")
    row = idx[0]

    def body(ks: list[z3.ArithRef]) -> Value:
        return ev.activation(
            expr.attrs.get("op"), _activation_pre(ev, expr, shape, [row, ks[0]])
        )

    return ev.reduction(combine, [(z3.IntVal(0), shape.dims[1])], body)


def _h_partition_reduce(
    ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list
) -> Value:
    src = expr.inputs[0]
    shape = ev.shape(src)
    combine = isa._normalize_combine_op(expr.attrs.get("op"))
    if isa.fold_family(1, isa.ReductionKind.REDUCE, combine) is None:
        raise Unsupported("partition reduce op")
    if shape.rank == 1:
        return ev.reduction(
            combine, [(z3.IntVal(0), shape.dims[0])], lambda ks: ev.eval(src, [ks[0]])
        )
    tail = idx[1:]
    return ev.reduction(
        combine,
        [(z3.IntVal(0), shape.dims[0])],
        lambda ks: ev.eval(src, [ks[0], *tail]),
    )


def _h_cumsum(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    src = expr.inputs[0]
    if out.rank != 2:
        raise Unsupported("cumsum rank")
    axis = expr.attrs.get("axis", -1)
    if isinstance(axis, int) and axis < 0:
        axis = out.rank + axis
    if axis != out.rank - 1:
        raise Unsupported("cumsum axis")
    row, col = idx
    return ev.reduction(
        "add", [(z3.IntVal(0), col + 1)], lambda ks: ev.eval(src, [row, ks[0]])
    )


def _h_tensor_scalar_cumulative(
    ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list
) -> Value:
    if out.rank != 2:
        raise Unsupported("cumulative rank")
    scan_eligible = (
        isa._operand_to_expr(expr.attrs.get("op1")) in ("add", "plus")
        and expr.attrs.get("reduce_cmd") == isa.reduce_cmd.reset_reduce
        and not expr.attrs.get("reverse1", False)
    )
    if not scan_eligible:
        raise Unsupported("cumulative recurrence")
    src = expr.inputs[0]
    row, col = idx

    def body(ks: list[z3.ArithRef]) -> Value:
        imm0 = ev.operand(expr, "imm0_const", "imm0_input_index", out, [row, ks[0]])
        data = ev.eval(src, [row, ks[0]])
        lhs, rhs = (imm0, data) if expr.attrs.get("reverse0", False) else (data, imm0)
        return ev.binary(expr.attrs.get("op0"), lhs, rhs)

    return ev.reduction("add", [(z3.IntVal(0), col + 1)], body)


def _h_public_softmax(ev: Evaluator, expr: Any, out: isa.ShapeExpr, idx: list) -> Value:
    lang = sys.modules.get("axon.lang_semantics")
    src = expr.inputs[0]
    shape = ev.shape(src)
    if (
        lang is None
        or out.rank != 2
        or not lang.public_softmax_axis_ok(expr, shape.rank)
    ):
        raise Unsupported("softmax form")
    row, col = idx
    n = shape.dims[1]
    row_max = ev.scalar(
        ev.reduction("max", [(z3.IntVal(0), n)], lambda ks: ev.eval(src, [row, ks[0]]))
    )

    def shifted(column: z3.ArithRef) -> Value:
        return ev.activation("exp", ev.add(ev.eval(src, [row, column]), -row_max))

    total = ev.reduction("add", [(z3.IntVal(0), n)], lambda ks: shifted(ks[0]))
    return ev.div(shifted(col), total)


_NAMED_HANDLERS: dict[str, Callable[..., Value]] = {
    "memset": _h_memset,
    "iota": _h_iota,
    "select_reduce": _h_select_reduce,
    "reciprocal": _h_reciprocal,
}


def _handler_for(compile_rule: Any, op: str) -> Callable[..., Value] | None:
    table: dict[Any, Callable[..., Value]] = {
        isa._compile_copy: _h_copy,
        isa._compile_nc_transpose: _h_nc_transpose,
        isa._compile_dma_transpose: _h_dma_transpose,
        isa._compile_tensor_tensor: _h_tensor_tensor,
        isa._compile_tensor_scalar: _h_tensor_scalar,
        isa._compile_scalar_tensor_tensor: _h_scalar_tensor_tensor,
        isa._compile_activation: _h_activation,
        isa._compile_exponential: _h_exponential,
        isa._compile_reciprocal: _h_reciprocal,
        isa._compile_public_unary: _h_public_unary,
        isa._compile_public_binary: _h_public_binary,
        isa._compile_select_reduce: _h_select_reduce,
        isa._compile_nc_matmul: _h_nc_matmul,
        isa._compile_tensor_reduce: _h_tensor_reduce,
        isa._compile_activation_reduce: _h_activation_reduce,
        isa._compile_tensor_partition_reduce: _h_partition_reduce,
        isa._compile_tensor_scalar_cumulative: _h_tensor_scalar_cumulative,
    }
    lang = sys.modules.get("axon.lang_semantics")
    if lang is not None:
        table.update(
            {
                lang._compile_public_where: _h_where,
                lang._compile_public_expand_dims: _h_expand_dims,
                lang._compile_public_matmul: _h_public_matmul,
                lang._compile_public_reduce: _h_public_reduce,
                lang._compile_public_cumsum: _h_cumsum,
                lang._compile_public_softmax: _h_public_softmax,
            }
        )
    handler = table.get(compile_rule)
    if handler is not None:
        return handler
    # Rules defined inside their builders are closures; match them by op name.
    if getattr(compile_rule, "__qualname__", "").endswith("value_rule"):
        return _NAMED_HANDLERS.get(op)
    return None


# The proof -------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationResult:
    outcome: Outcome
    detail: str = ""


def _remaining_ms(deadline: float) -> int:
    return math.ceil((deadline - time.monotonic()) * 1000)


def _obligation(
    ev: Evaluator,
    assumptions: list[z3.BoolRef],
    extra: list[z3.BoolRef],
    goal: z3.BoolRef,
) -> list[z3.BoolRef]:
    formulas = [*assumptions, *ev.facts, *extra]
    mentioned = isa._decl_names([*formulas, goal])
    lemmas = isa.elementwise_algebra_context(mentioned).facts
    return [*formulas, *lemmas, z3.Not(goal)]


def prove_by_evaluation(
    lhs: isa.SymExpr,
    rhs: isa.SymExpr,
    assumptions: list[z3.BoolRef],
    out_shape: isa.ShapeExpr,
    timeout: int,
) -> EvaluationResult:
    """Decide ``lhs == rhs`` at every in-bounds index by symbolic evaluation.

    ``assumptions`` must already hold the source validity, preconditions and
    shape equality. PROVED is sound. REFUTED is returned only when evaluation
    abstracted nothing (no reduction element), so the counterexample is a real
    one for the same semantics the quantified encoding uses."""
    if out_shape.rank == 0 or timeout <= 0:
        return EvaluationResult(Outcome.UNKNOWN, "rank 0")
    deadline = time.monotonic() + timeout / 1000.0
    ev = Evaluator()
    witness = [z3.Int(f"witness_{index}") for index in range(out_shape.rank)]
    bounds = [
        z3.And(witness[i] >= 0, witness[i] < out_shape.dims[i])
        for i in range(out_shape.rank)
    ]
    try:
        with isa._Z3_LOCK:
            lhs_value = ev.eval(lhs, witness)
            rhs_value = ev.eval(rhs, witness)
    except (Unsupported, z3.Z3Exception, KeyError, IndexError, ValueError) as exc:
        return EvaluationResult(Outcome.UNKNOWN, f"unsupported: {exc}")

    base = [*assumptions, *bounds]
    if (
        isinstance(lhs_value, Red)
        and isinstance(rhs_value, Red)
        and lhs_value.op == rhs_value.op
        and len(lhs_value.vars) == len(rhs_value.vars)
        and len(lhs_value.vars) <= _MAX_PERMUTED_VARS
        and _prove_reductions_equal(ev, lhs_value, rhs_value, base, deadline)
    ):
        return EvaluationResult(Outcome.PROVED, "reduction bodies")
    remaining = _remaining_ms(deadline)
    if remaining <= 0:
        return EvaluationResult(Outcome.UNKNOWN, "deadline")
    with isa._Z3_LOCK:
        goal = ev.scalar(lhs_value) == ev.scalar(rhs_value)
        query = _obligation(ev, base, [], goal)
    result = isa._check_deterministic(query, remaining)
    if result == z3.unsat:
        return EvaluationResult(Outcome.PROVED, "pointwise")
    if result == z3.sat and ev.exact:
        return EvaluationResult(Outcome.REFUTED, "pointwise counterexample")
    return EvaluationResult(Outcome.UNKNOWN, str(result))


def _prove_reductions_equal(
    ev: Evaluator,
    lhs: Red,
    rhs: Red,
    base: list[z3.BoolRef],
    deadline: float,
) -> bool:
    """Search for a bijection of reduction variables that equates the bodies.

    This is TensorRight's reduction-index relation, found by search instead of
    supplied as a hint. Identity is tried first; the other permutations handle
    reordered nested reductions (matmul associativity, summation exchange)."""
    count = len(lhs.vars)
    red_bounds = [
        z3.And(var >= low, var < high)
        for var, low, high in zip(lhs.vars, lhs.lows, lhs.highs, strict=True)
    ]
    for perm in itertools.permutations(range(count)):
        remaining = _remaining_ms(deadline)
        if remaining <= 0:
            return False
        substitution = [(rhs.vars[perm[i]], lhs.vars[i]) for i in range(count)]
        # Reject a permutation whose literal extents already disagree.
        literal_clash = False
        for i in range(count):
            left = _normal(lhs.highs[i])
            right = _normal(rhs.highs[perm[i]])
            if (
                z3.is_int_value(left)
                and z3.is_int_value(right)
                and not z3.eq(left, right)
            ):
                literal_clash = True
        if literal_clash:
            continue
        with isa._Z3_LOCK:
            ranges_equal = z3.And(
                *[
                    z3.And(
                        lhs.lows[i] == z3.substitute(rhs.lows[perm[i]], *substitution),
                        lhs.highs[i]
                        == z3.substitute(rhs.highs[perm[i]], *substitution),
                    )
                    for i in range(count)
                ]
            )
            rhs_body = z3.substitute(rhs.body, *substitution)
            renamed_facts = [z3.substitute(fact, *substitution) for fact in ev.facts]
            # Range equality is proved outside the range assumption: an empty
            # left range would otherwise make it hold vacuously.
            goal = z3.And(
                ranges_equal,
                z3.Implies(z3.And(*red_bounds), lhs.body == rhs_body),
            )
            query = _obligation(ev, base, renamed_facts, goal)
        # Per permutation budget: never let one permutation eat the rest.
        budget = max(1, min(remaining, remaining // max(1, count) + 1))
        if isa._check_deterministic(query, budget) == z3.unsat:
            return True
    return False
