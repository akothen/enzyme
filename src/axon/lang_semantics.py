from __future__ import annotations

import builtins
from typing import Any

import z3

from axon.isa_semantics import (
    _BODY_IDS,
    _EXP_FN,
    REDUCE1_FAM,
    REDUCE1_MAX_FAM,
    REDUCE2_FAM,
    SCAN2_FAM,
    Context,
    ReductionDesc,
    Semantics,
    ShapeExpr,
    ShapeResult,
    SymExpr,
    _broadcast_shape,
    _call_broadcasted,
    _compile_copy,
    _compile_nc_transpose,
    _compile_public_binary,
    _compile_public_opaque,
    _compile_public_unary,
    _compile_tensor_reduce,
    _ensure_semantics,
    _index_vars,
    _is_sym_tensor,
    _new_sym_tensor,
    _public_binary,
    _public_broadcast_shape_tuple,
    _public_dynamic_slice,
    _public_gather_flattened,
    _public_pointwise_binary,
    _public_reduce,
    _public_rms_norm,
    _public_store,
    _public_transpose_out_shape,
    _public_unary,
    _safe_divide,
    _shape_ctx,
    _shape_from_out,
    _shape_public_attr_or_first,
    _shape_public_binary,
    _shape_public_unary,
    _shape_same_as_first,
    _tensor_function,
    activation,
    compile_rank2_fold_reduce,
    dma_copy,
    dma_transpose,
    nc_matmul,
    nc_transpose,
    nl,
    semantics,
)


@semantics()
def abs(x, dtype=None):
    return _public_unary("abs", x)


@semantics()
def add(x, y, dtype=None):
    return _public_binary("add", x, y)


@semantics()
def all(x, axis, dtype=None):
    return _public_reduce("all", x, axis)


@semantics()
def arctan(x, dtype=None):
    return _public_unary("arctan", x)


@semantics()
def bitwise_and(x, y, dtype=None):
    return _public_binary("bitwise_and", x, y)


@semantics()
def bitwise_or(x, y, dtype=None):
    return _public_binary("bitwise_or", x, y)


@semantics()
def bitwise_xor(x, y, dtype=None):
    return _public_binary("bitwise_xor", x, y)


@semantics()
def broadcast_to(x, shape, dtype=None):
    assert _is_sym_tensor(x)
    return _new_sym_tensor("broadcast_to", [x], tuple(shape), out_shape=tuple(shape))


@semantics()
def ceil(x, dtype=None):
    return _public_unary("ceil", x)


@semantics()
def copy(x, dtype=None):
    return _public_unary("copy", x)


@semantics()
def cos(x, dtype=None):
    return _public_unary("cos", x)


@semantics()
def divide(x, y, dtype=None):
    return _public_binary("divide", x, y)


@semantics()
def dropout(x, rate, dtype=None):
    assert _is_sym_tensor(x)
    inputs = [x]
    if _is_sym_tensor(rate):
        inputs.append(rate)
    return _new_sym_tensor("dropout", inputs, {"rate": rate}, x.shape)


@semantics()
def ds(start, size):
    return _public_dynamic_slice(start, size)


@semantics()
def equal(x, y, dtype=None):
    return _public_binary("equal", x, y)


@semantics()
def erf(x, dtype=None):
    return _public_unary("erf", x)


@semantics()
def erf_dx(x, dtype=None):
    return _public_unary("erf_dx", x)


@semantics()
def exp(x, dtype=None):
    return _public_unary("exp", x)


@semantics()
def expand_dims(x, axis):
    return (
        _new_sym_tensor(
            "expand_dims",
            [x],
            {"axis": axis},
            _public_expand_dims_out_shape(x.shape, int(axis)),
        )
        if _is_sym_tensor(x)
        else ...
    )


@semantics()
def floor(x, dtype=None):
    return _public_unary("floor", x)


@semantics()
def fmod(x, y, dtype=None):
    return _public_binary("fmod", x, y)


@semantics()
def gather_flattened(data, indices, axis=0, dtype=None):
    return _public_gather_flattened(data, indices, axis)


@semantics()
def gelu(x, dtype=None):
    return _public_unary("gelu", x)


@semantics()
def gelu_apprx_sigmoid(x, dtype=None):
    return _public_unary("gelu_apprx_sigmoid", x)


@semantics()
def gelu_apprx_sigmoid_dx(x, dtype=None):
    return _public_unary("gelu_apprx_sigmoid_dx", x)


@semantics()
def gelu_apprx_tanh(x, dtype=None):
    return _public_unary("gelu_apprx_tanh", x)


@semantics()
def gelu_dx(x, dtype=None):
    return _public_unary("gelu_dx", x)


@semantics()
def greater(x, y, dtype=None):
    return _public_binary("greater", x, y)


@semantics()
def greater_equal(x, y, dtype=None):
    return _public_binary("greater_equal", x, y)


@semantics()
def invert(x, dtype=None):
    return _public_unary("invert", x)


@semantics()
def left_shift(x, y, dtype=None):
    return _public_pointwise_binary(x, y, nl.left_shift)


@semantics()
def less(x, y, dtype=None):
    return _public_binary("less", x, y)


@semantics()
def less_equal(x, y, dtype=None):
    return _public_binary("less_equal", x, y)


@semantics()
def load(src, dtype=None):
    return dma_copy(dst=None, src=src)


@semantics()
def load_transpose2d(src, dtype=None):
    return dma_transpose(dst=None, src=src, axes=(1, 0))


@semantics()
def log(x, dtype=None):
    return _public_unary("log", x)


@semantics()
def logical_and(x, y, dtype=None):
    return _public_binary("logical_and", x, y)


@semantics()
def logical_not(x, dtype=None):
    return _public_unary("logical_not", x)


@semantics()
def logical_or(x, y, dtype=None):
    return _public_binary("logical_or", x, y)


@semantics()
def logical_xor(x, y, dtype=None):
    return _public_binary("logical_xor", x, y)


@semantics()
def matmul(x, y, transpose_x=False):
    stationary = x if transpose_x else nc_transpose(dst=None, data=x)
    assert _is_sym_tensor(stationary) and _is_sym_tensor(y)
    return nc_matmul(dst=None, stationary=stationary, moving=y)


@semantics()
def max(x, axis, dtype=None, keepdims=False):
    return _public_reduce("max", x, axis, keepdims=keepdims)


@semantics()
def maximum(x, y, dtype=None):
    return _public_binary("maximum", x, y)


@semantics()
def mean(x, axis, dtype=None, keepdims=False):
    return _public_reduce("mean", x, axis, keepdims=keepdims)


@semantics()
def min(x, axis, dtype=None, keepdims=False):
    return _public_reduce("min", x, axis, keepdims=keepdims)


@semantics()
def minimum(x, y, dtype=None):
    return _public_binary("minimum", x, y)


@semantics()
def mish(x, dtype=None):
    return activation(dst=None, op=nl.mish, data=x)


@semantics()
def mod(x, y, dtype=None):
    return _public_pointwise_binary(x, y, nl.mod)


@semantics()
def multiply(x, y, dtype=None):
    return _public_pointwise_binary(x, y, nl.multiply)


@semantics()
def negative(x, dtype=None):
    return _public_unary("negative", x)


@semantics()
def not_equal(x, y, dtype=None):
    return _public_binary("not_equal", x, y)


@semantics()
def power(x, y, dtype=None):
    return _public_binary("power", x, y)


@semantics()
def prod(x, axis, dtype=None, keepdims=False):
    return _public_reduce("prod", x, axis, keepdims=keepdims)


@semantics()
def reciprocal(x, dtype=None):
    return _public_unary("reciprocal", x)


@semantics()
def relu(x, dtype=None):
    return _public_unary("relu", x)


@semantics()
def right_shift(x, y, dtype=None):
    return _public_pointwise_binary(x, y, nl.right_shift)


@semantics()
def rms_norm(x, w, axis, n, epsilon=1e-06, dtype=None, compute_dtype=None):
    return _public_rms_norm(x, w)


@semantics()
def rsqrt(x, dtype=None):
    return _public_unary("rsqrt", x)


@semantics()
def sigmoid(x, dtype=None):
    return _public_unary("sigmoid", x)


@semantics()
def sign(x, dtype=None):
    return _public_unary("sign", x)


@semantics()
def silu(x, dtype=None):
    return _public_unary("silu", x)


@semantics()
def silu_dx(x, dtype=None):
    return _public_unary("silu_dx", x)


@semantics()
def sin(x, dtype=None):
    return _public_unary("sin", x)


@semantics()
def softmax(x, axis=-1, dtype=None):
    return _public_unary("softmax", x, axis=axis)


@semantics()
def cumsum(x, axis=-1, dtype=None):
    return _public_unary("cumsum", x, axis=axis)


@semantics()
def softplus(x, dtype=None):
    return _public_unary("softplus", x)


@semantics()
def sqrt(x, dtype=None):
    return _public_unary("sqrt", x)


@semantics()
def square(x, dtype=None):
    return _public_unary("square", x)


@semantics()
def store(dst, value):
    return _public_store(dst, value)


@semantics()
def subtract(x, y, dtype=None):
    return _public_binary("subtract", x, y)


@semantics()
def sum(x, axis, dtype=None, keepdims=False):
    return _public_reduce("sum", x, axis, keepdims=keepdims)


@semantics()
def tan(x, dtype=None):
    return _public_unary("tan", x)


@semantics()
def tanh(x, dtype=None):
    return _public_unary("tanh", x)


@semantics()
def transpose(x, dtype=None):
    return (
        _new_sym_tensor("transpose", [x], {}, _public_transpose_out_shape(x))
        if _is_sym_tensor(x)
        else ...
    )


@semantics()
def trunc(x, dtype=None):
    return _public_unary("trunc", x)


@semantics()
def var(x, axis, dtype=None, keepdims=False):
    return _public_reduce("var", x, axis, keepdims=keepdims)


@semantics()
def where(condition, x, y, dtype=None):
    assert _is_sym_tensor(condition) and _is_sym_tensor(x) and _is_sym_tensor(y)
    xy_shape = _public_broadcast_shape_tuple(x.shape, y.shape)
    out_shape = _public_broadcast_shape_tuple(condition.shape, xy_shape)
    return _new_sym_tensor("where", [condition, x, y], {}, out_shape)


def _shape_public_reduce(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    a = ins[0]
    ctx = Context([d > 0 for d in a.dims])
    axis_attr = attrs.get("axis")
    if axis_attr is None:
        ctx.add(z3.BoolVal(False))
        return ShapeResult(ShapeExpr(list(a.dims)), ctx)
    axes = [axis_attr] if isinstance(axis_attr, int) else list(axis_attr)
    norm_axes: list[int] = []
    for axis in axes:
        norm_axis = axis + a.rank if axis < 0 else axis
        if norm_axis < 0 or norm_axis >= a.rank:
            ctx.add(z3.BoolVal(False))
            return ShapeResult(ShapeExpr(list(a.dims)), ctx)
        norm_axes.append(norm_axis)
    keepdims = bool(attrs.get("keepdims", False))
    out_dims: list[z3.ArithRef] = []
    for dim_index, dim in enumerate(a.dims):
        if dim_index in norm_axes:
            if keepdims:
                out_dims.append(z3.IntVal(1))
        else:
            out_dims.append(dim)
    return ShapeResult(ShapeExpr(out_dims if out_dims else [z3.IntVal(1)]), ctx)


def _shape_graph_reduce_sum(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    return _shape_public_reduce(
        ins,
        {
            "axis": attrs.get("axis"),
            "keepdims": bool(attrs.get("keep_dims", attrs.get("keepdims", False))),
        },
    )


_PUBLIC_REDUCE_COMBINE = {"sum": "add", "max": "max", "min": "min", "prod": "mul"}


def public_reduce_single_axis(expr: SymExpr, rank: int) -> int | None:
    """The one normalized axis a public reduction folds, when it folds one."""
    axis_attr = expr.attrs.get("axis")
    if isinstance(axis_attr, (list, tuple)):
        if len(axis_attr) != 1:
            return None
        axis_attr = axis_attr[0]
    if not isinstance(axis_attr, int) or isinstance(axis_attr, bool):
        return None
    axis = axis_attr + rank if axis_attr < 0 else axis_attr
    return axis if 0 <= axis < rank else None


def _compile_public_reduce(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    """Fold semantics for single-axis rank-2 public reductions.

    ``sum``/``max``/``min``/``prod`` used to compile to an opaque function of the
    output index alone, so ``nl.sum(x, axis=1)`` could never be proved equal to
    its own ``tensor_reduce`` lowering. They now share the ISA fold families;
    ``mean`` is the sum divided by the reduced extent. Other shapes and ops
    (``var``, ``all``, multi-axis) stay opaque."""
    a = ins[0]
    axis = public_reduce_single_axis(expr, a.shape.rank)
    if a.shape.rank == 2 and axis is not None:
        combine = _PUBLIC_REDUCE_COMBINE.get(expr.op)
        if combine is not None:
            return compile_rank2_fold_reduce(expr, ins, out_shape, combine)
        if expr.op == "mean":
            sum_expr = SymExpr(
                "sum", expr.inputs, expr.shape, dict(expr.attrs), f"{expr.name}_sum"
            )
            total = compile_rank2_fold_reduce(sum_expr, ins, out_shape, "add")
            out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
            idx = _index_vars(expr.name, out_shape.rank)
            extent = z3.ToReal(a.shape.dims[axis])
            ctx = total.ctx.merged()
            ctx.add(
                z3.ForAll(idx, out_fn(*idx) == _safe_divide(total.fn(*idx), extent))
            )
            return Semantics(expr.name, out_shape, out_fn, ctx)
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = _index_vars(expr.name, out_shape.rank)
    ctx = a.ctx.merged()
    opaque = z3.Function(
        f"PUBLIC_REDUCE_{expr.name}",
        *([z3.IntSort()] * builtins.max(1, out_shape.rank)),
        z3.RealSort(),
    )
    index_args = idx if idx else [z3.IntVal(0)]
    ctx.add(z3.ForAll(idx, out_fn(*idx) == opaque(*index_args)))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def public_softmax_axis_ok(expr: SymExpr, rank: int) -> bool:
    axis = expr.attrs.get("axis", -1)
    if not isinstance(axis, int) or isinstance(axis, bool):
        return False
    return rank == 2 and (axis + rank if axis < 0 else axis) == rank - 1


def _compile_public_softmax(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    """Row softmax: ``exp(x - max_row) / sum_row(exp(x - max_row))``.

    Softmax used to be compiled as an elementwise uninterpreted function of one
    element, which claims it commutes with any index permutation: the checker
    proved ``transpose(softmax(x)) == softmax(transpose(x))``. It is now defined
    through a max fold and a sum fold over the last axis, the two reductions its
    ISA lowering performs. Other ranks and axes get a per-node opaque function of
    the whole input, which proves nothing about them."""
    a = ins[0]
    if out_shape.rank != 2 or not public_softmax_axis_ok(expr, a.shape.rank):
        return _compile_public_opaque(expr, ins, out_shape)
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = a.ctx.merged()
    i = z3.Int(f"{expr.name}_si")
    j = z3.Int(f"{expr.name}_sj")
    k = z3.Int(f"{expr.name}_sk")
    n = a.shape.dims[1]
    max_body = _BODY_IDS.next()
    ctx.add(
        z3.ForAll([i, k], REDUCE1_MAX_FAM.step(z3.IntVal(max_body), i, k) == a.fn(i, k))
    )
    row_max = REDUCE1_MAX_FAM.fold(z3.IntVal(max_body), i, n)

    def shifted_exp(col: z3.ArithRef) -> z3.ArithRef:
        return _EXP_FN(a.fn(i, col) - row_max)

    sum_body = _BODY_IDS.next()
    ctx.add(
        z3.ForAll([i, k], REDUCE1_FAM.step(z3.IntVal(sum_body), i, k) == shifted_exp(k))
    )
    ctx.add(z3.ForAll([i, k], shifted_exp(k) > 0))
    row_sum = REDUCE1_FAM.fold(z3.IntVal(sum_body), i, n)
    ctx.add(z3.ForAll([i, j], out_fn(i, j) == _safe_divide(shifted_exp(j), row_sum)))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_public_cumsum(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    return _shape_same_as_first(ins, attrs)


def _validity_public_cumsum(ins: list[ShapeExpr], attrs: dict[str, Any]) -> Context:
    ctx = Context()
    if not ins or ins[0].rank != 2:
        ctx.add(z3.BoolVal(False))
        return ctx
    axis = attrs.get("axis", -1)
    if isinstance(axis, int) and axis < 0:
        axis = ins[0].rank + axis
    if axis != ins[0].rank - 1:
        ctx.add(z3.BoolVal(False))
    return ctx


def _compile_public_cumsum(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    a = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = a.ctx.merged()
    if out_shape.rank != 2:
        # Invalid rank or axis: the validity rule marks these unsatisfiable.
        return Semantics(expr.name, out_shape, out_fn, ctx)
    axis = expr.attrs.get("axis", -1)
    if isinstance(axis, int) and axis < 0:
        axis = out_shape.rank + axis
    if axis != out_shape.rank - 1:
        return Semantics(expr.name, out_shape, out_fn, ctx)
    # Emit the scan via the SCAN2 fold family. Equivalence to
    # tensor_scalar_cumulative falls out of SCAN2's extensionality axiom.
    body_id = _BODY_IDS.next()
    target_body = z3.IntVal(body_id)
    fam = SCAN2_FAM
    i = z3.Int(f"{expr.name}_si")
    j = z3.Int(f"{expr.name}_sj")
    k = z3.Int(f"{expr.name}_sk")
    ctx.add(z3.ForAll([i, j, k], fam.step(target_body, i, j, k) == a.fn(i, k)))
    ctx.add(z3.ForAll([i, j], out_fn(i, j) == fam.fold(target_body, i, j)))
    extent = out_shape.dims[1]
    return Semantics(
        expr.name,
        out_shape,
        out_fn,
        ctx,
        reduction=ReductionDesc(
            body_id,
            extent,
            outer_rank=fam.outer_arity,
            kind=fam.kind,
            combine_op=fam.combine_op,
            outer_dims=tuple(out_shape.dims),
            output_transform="identity",
        ),
    )


def _shape_public_broadcast_to(
    ins: list[ShapeExpr], attrs: dict[str, Any]
) -> ShapeResult:
    target_shape = attrs.get("shape") or attrs.get("out_shape")
    if target_shape is None:
        return _shape_same_as_first(ins, attrs)
    out = _shape_from_out(tuple(target_shape))
    if not ins:
        return out
    compat = _broadcast_shape(ins[0], out.out)
    return ShapeResult(out.out, compat.ctx)


def _shape_graph_broadcast(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    target_shape = tuple(ins[1].dims) if len(ins) >= 2 else attrs.get("shape")
    return _shape_public_broadcast_to(ins[:1], {"shape": target_shape})


def _shape_public_expand_dims(
    ins: list[ShapeExpr], attrs: dict[str, Any]
) -> ShapeResult:
    a = ins[0]
    ctx = Context([d > 0 for d in a.dims])
    axis = int(attrs.get("axis", 0))
    rank = a.rank + 1
    norm_axis = axis + rank if axis < 0 else axis
    if norm_axis < 0 or norm_axis > a.rank:
        ctx.add(z3.BoolVal(False))
        return ShapeResult(ShapeExpr(list(a.dims)), ctx)
    dims = list(a.dims)
    dims.insert(norm_axis, z3.IntVal(1))
    return ShapeResult(ShapeExpr(dims), ctx)


def _compile_public_expand_dims(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    src = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = _index_vars(expr.name, out_shape.rank)
    ctx = src.ctx.merged()
    axis = int(expr.attrs.get("axis", 0))
    norm_axis = axis + out_shape.rank if axis < 0 else axis
    src_idx = [dim for i, dim in enumerate(idx) if i != norm_axis]
    ctx.add(z3.ForAll(idx, out_fn(*idx) == src.fn(*src_idx)))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_public_transpose2d(
    ins: list[ShapeExpr], attrs: dict[str, Any]
) -> ShapeResult:
    a = ins[0]
    ctx = Context([d > 0 for d in a.dims])
    if a.rank != 2:
        ctx.add(z3.BoolVal(False))
        return ShapeResult(ShapeExpr(list(a.dims)), ctx)
    return ShapeResult(ShapeExpr([a.dims[1], a.dims[0]]), ctx)


def _shape_public_where(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    if len(ins) != 3:
        return ShapeResult(ShapeExpr([z3.IntVal(1)]), Context([z3.BoolVal(False)]))
    xy = _broadcast_shape(ins[1], ins[2])
    cxy = _broadcast_shape(ins[0], xy.out)
    return ShapeResult(cxy.out, xy.ctx.merged(cxy.ctx))


def _compile_public_where(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    cond_sem, x_sem, y_sem = ins
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = _index_vars(expr.name, out_shape.rank)
    ctx = cond_sem.ctx.merged(x_sem.ctx, y_sem.ctx)
    cond_value = _call_broadcasted(cond_sem, out_shape, idx) != 0
    x_value = _call_broadcasted(x_sem, out_shape, idx)
    y_value = _call_broadcasted(y_sem, out_shape, idx)
    ctx.add(z3.ForAll(idx, out_fn(*idx) == z3.If(cond_value, x_value, y_value)))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_public_matmul(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    a, b = ins
    ctx = Context([d > 0 for d in a.dims + b.dims])
    if a.rank != 2 or b.rank != 2:
        ctx.add(z3.BoolVal(False))
        return ShapeResult(ShapeExpr(list(a.dims)), ctx)
    if bool(attrs.get("transpose_x", False)):
        ctx.add(a.dims[0] == b.dims[0])
        return ShapeResult(ShapeExpr([a.dims[1], b.dims[1]]), ctx)
    ctx.add(a.dims[1] == b.dims[0])
    return ShapeResult(ShapeExpr([a.dims[0], b.dims[1]]), ctx)


def _compile_public_matmul(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    a, b = ins
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = a.ctx.merged(b.ctx)
    if a.shape.rank != 2 or b.shape.rank != 2 or out_shape.rank != 2:
        # Invalid rank: the shape rule marks validity as unsatisfiable.
        return Semantics(expr.name, out_shape, out_fn, ctx)
    m = z3.Int(f"{expr.name}_m")
    n = z3.Int(f"{expr.name}_n")
    k = z3.Int(f"{expr.name}_k")
    body_id = _BODY_IDS.next()
    fam = REDUCE2_FAM  # public matmul is intrinsically sum-of-products
    if bool(expr.attrs.get("transpose_x", False)):
        ctx.add(
            z3.ForAll(
                [m, n, k],
                fam.step(z3.IntVal(body_id), m, n, k) == a.fn(k, m) * b.fn(k, n),
            )
        )
        extent = a.shape.dims[0]
    else:
        ctx.add(
            z3.ForAll(
                [m, n, k],
                fam.step(z3.IntVal(body_id), m, n, k) == a.fn(m, k) * b.fn(k, n),
            )
        )
        extent = a.shape.dims[1]
    ctx.add(
        z3.ForAll([m, n], out_fn(m, n) == fam.fold(z3.IntVal(body_id), m, n, extent))
    )
    return Semantics(
        expr.name,
        out_shape,
        out_fn,
        ctx,
        reduction=ReductionDesc(
            body_id,
            extent,
            outer_rank=fam.outer_arity,
            kind=fam.kind,
            combine_op=fam.combine_op,
            outer_dims=tuple(out_shape.dims),
            output_transform="identity",
        ),
    )


def _shape_public_gather_flattened(
    ins: list[ShapeExpr], attrs: dict[str, Any]
) -> ShapeResult:
    if len(ins) != 2:
        return ShapeResult(ShapeExpr([z3.IntVal(1)]), Context([z3.BoolVal(False)]))
    data_shape, idx_shape = ins
    dims = (
        [data_shape.dims[0], *idx_shape.dims[1:]]
        if idx_shape.rank >= 1
        else list(data_shape.dims)
    )
    return ShapeResult(ShapeExpr(dims), _shape_ctx(*data_shape.dims, *idx_shape.dims))


def _public_expand_dims_out_shape(shape: tuple[Any, ...], axis: int) -> tuple[Any, ...]:
    dims = list(shape)
    rank = len(dims) + 1
    norm_axis = axis + rank if axis < 0 else axis
    if norm_axis < 0 or norm_axis > len(dims):
        return tuple(dims)
    dims.insert(norm_axis, 1)
    return tuple(dims)


def loop_reduce(
    data: Any, op: Any, loop_indices: list[Any], mask=None, dtype=None, name=None
):
    if not _is_sym_tensor(data):
        return data
    return _new_sym_tensor("loop_reduce", [data], {"op": op, "name": name}, data.shape)


def _register_public_semantics() -> None:
    unary_ops = {
        "abs",
        "arctan",
        "ceil",
        "copy",
        "cos",
        "erf",
        "erf_dx",
        "exp",
        "floor",
        "gelu",
        "gelu_apprx_sigmoid",
        "gelu_apprx_sigmoid_dx",
        "gelu_apprx_tanh",
        "gelu_dx",
        "invert",
        "log",
        "logical_not",
        "mish",
        "negative",
        "reciprocal",
        "relu",
        "rsqrt",
        "sigmoid",
        "sign",
        "silu",
        "silu_dx",
        "sin",
        "softplus",
        "sqrt",
        "square",
        "tan",
        "tanh",
        "trunc",
    }
    for op_name in unary_ops:
        _ensure_semantics(op_name, _shape_public_unary, _compile_public_unary)

    binary_ops = {
        "add",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "divide",
        "equal",
        "fmod",
        "greater",
        "greater_equal",
        "left_shift",
        "less",
        "less_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
        "maximum",
        "minimum",
        "mod",
        "multiply",
        "not_equal",
        "power",
        "right_shift",
        "subtract",
    }
    for op_name in binary_ops:
        _ensure_semantics(op_name, _shape_public_binary, _compile_public_binary)

    for op_name in {"all", "max", "mean", "min", "prod", "sum", "var"}:
        _ensure_semantics(op_name, _shape_public_reduce, _compile_public_reduce)

    for op_name in {
        "dropout",
        "ds",
        "gather_flattened",
        "rms_norm",
        "loop_reduce",
        "store",
    }:
        shape_rule = (
            _shape_public_gather_flattened
            if op_name == "gather_flattened"
            else _shape_public_attr_or_first
        )
        compile_rule = _compile_copy if op_name == "store" else _compile_public_opaque
        _ensure_semantics(op_name, shape_rule, compile_rule)

    _ensure_semantics("broadcast", _shape_graph_broadcast, _compile_copy)
    _ensure_semantics("broadcast_to", _shape_public_broadcast_to, _compile_copy)
    _ensure_semantics(
        "cumsum", _shape_public_cumsum, _compile_public_cumsum, _validity_public_cumsum
    )
    _ensure_semantics("softmax", _shape_public_unary, _compile_public_softmax)
    _ensure_semantics("div", _shape_public_binary, _compile_public_binary)
    _ensure_semantics(
        "expand_dims", _shape_public_expand_dims, _compile_public_expand_dims
    )
    _ensure_semantics("load", _shape_same_as_first, _compile_copy)
    _ensure_semantics(
        "load_transpose2d", _shape_public_transpose2d, _compile_nc_transpose
    )
    _ensure_semantics("matmul", _shape_public_matmul, _compile_public_matmul)
    _ensure_semantics("mul", _shape_public_binary, _compile_public_binary)
    _ensure_semantics("reduce_sum", _shape_graph_reduce_sum, _compile_tensor_reduce)
    _ensure_semantics("transpose", _shape_public_transpose2d, _compile_nc_transpose)
    _ensure_semantics("where", _shape_public_where, _compile_public_where)


_register_public_semantics()
