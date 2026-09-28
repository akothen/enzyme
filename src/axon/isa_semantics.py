from __future__ import annotations

import builtins
import math
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from enum import Enum
from threading import Lock, RLock
from typing import Any

import z3


class StaleIdentityError(Exception):
    """A graph contains node ids issued before the latest counter reset."""


class GlobalCounter:
    """Thread-safe counter that tracks which reset generation issued each name."""

    def __init__(self, origin: int) -> None:
        self._origin = origin
        self._lock = Lock()
        self._value = origin
        self._generation = 0
        self._issued: dict[str, int] = {}

    @property
    def origin(self) -> int:
        return self._origin

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def next(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    def next_name(self, prefix: str) -> str:
        with self._lock:
            self._value += 1
            name = f"{prefix}_{self._value}"
            self._issued[name] = self._generation
            return name

    def reset(self) -> None:
        with self._lock:
            self._value = self._origin
            self._generation += 1

    def assert_current(self, names: Iterable[str], context: str) -> None:
        """Raise when a name was last issued before the current generation."""
        with self._lock:
            generation = self._generation
            stale = sorted(
                {
                    name
                    for name in names
                    if self._issued.get(name, generation) < generation
                }
            )
        if stale:
            raise StaleIdentityError(
                f"{context}: {len(stale)} id(s) predate the last identity reset "
                f"(generation {generation}): {stale[:5]}"
            )


class _OpRef:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"nl.{self.name}"


class _NLNamespace:
    def __getattr__(self, name: str) -> _OpRef:
        op = _OpRef(name)
        setattr(self, name, op)
        return op


nl = _NLNamespace()

_NODE_IDS = GlobalCounter(1000)
_BODY_IDS = GlobalCounter(2000)
_LARGE_POSITIVE_SENTINEL = 1e30
# NLSAT ignores the wall timeout, so use a deterministic resource limit.
# Recalibrate this ratio after Z3 upgrades.
_RLIMIT_PER_MS = 400
_BINARY_UFS: dict[str, z3.FuncDeclRef] = {}
_UNARY_UFS: dict[str, z3.FuncDeclRef] = {}
_COMPARE_UFS: dict[str, z3.FuncDeclRef] = {}
_UF_LOCK = Lock()
_SEMANTICS_LOCK = Lock()
_VERBOSE_LOCK = Lock()  # serializes verbose prints from concurrent synthesis threads
_Z3_LOCK = (
    RLock()
)  # serializes z3 formula construction (z3.main_ctx is not thread-safe)

_NormSketch = Any


def _reset_identity_counters() -> None:
    """Reset node and reduction identities at a kernel boundary."""
    _NODE_IDS.reset()
    _BODY_IDS.reset()


def _start_kernel_synthesis_cache(
    kernel_name: str | None = None,
    verbose: bool = False,
) -> None:
    _reset_identity_counters()
    if verbose:
        if kernel_name is None:
            print("[synthesis-cache] starting with an empty cache")
        else:
            print(
                f"[synthesis-cache] starting kernel '{kernel_name}' with an empty cache"
            )


def _effective_max_workers(max_workers: int | None, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        return builtins.max(1, builtins.min(cpu_count, task_count))
    return builtins.max(1, builtins.min(max_workers, task_count))


_SYNTHESIS_STATS_LOCK = Lock()


def _tensor_function(name: str, rank: int) -> z3.FuncDeclRef:
    return z3.Function(name, *([z3.IntSort()] * builtins.max(1, rank)), z3.RealSort())


class ReductionKind(Enum):
    REDUCE = "reduce"
    SCAN = "scan"


@dataclass(frozen=True)
class FoldFamily:
    name: str
    outer_arity: int
    kind: ReductionKind
    combine_op: str
    step: z3.FuncDeclRef
    fold: z3.FuncDeclRef
    fold_takes_extent: bool


_FOLD_FAMILIES: dict[tuple[int, ReductionKind, str], FoldFamily] = {}
_FOLD_EXTENSIONALITY_FACTS: list[z3.BoolRef] = []
# Additive-fold lemmas apply only to sum reductions in directional proofs.
_FOLD_ADDITIVITY_FACTS: list[z3.BoolRef] = []
# Operand-swap lemmas, sound for a reduce family but never for a scan, whose
# outer index is the summation bound.
_FOLD_SWAP_FACTS: list[z3.BoolRef] = []


def _make_extensionality_axiom(fam: FoldFamily) -> z3.BoolRef:
    b1 = z3.Int(f"_{fam.name}_ext_b1")
    b2 = z3.Int(f"_{fam.name}_ext_b2")
    outer = [z3.Int(f"_{fam.name}_ext_o{n}") for n in range(fam.outer_arity)]
    k = z3.Int(f"_{fam.name}_ext_k")
    same_step = z3.ForAll([k], fam.step(b1, *outer, k) == fam.step(b2, *outer, k))
    if fam.fold_takes_extent:
        e1 = z3.Int(f"_{fam.name}_ext_e1")
        e2 = z3.Int(f"_{fam.name}_ext_e2")
        return z3.ForAll(
            [b1, b2, *outer, e1, e2],
            z3.Implies(
                z3.And(e1 == e2, same_step),
                fam.fold(b1, *outer, e1) == fam.fold(b2, *outer, e2),
            ),
        )
    return z3.ForAll(
        [b1, b2, *outer],
        z3.Implies(same_step, fam.fold(b1, *outer) == fam.fold(b2, *outer)),
    )


def _make_swap_axiom(fam: FoldFamily) -> z3.BoolRef:
    """Swapping a rank-2 sum fold's outer indices transposes its result, so
    ``nc_matmul(A, B)`` equals the transpose of ``nc_matmul(B, A)``."""
    b1 = z3.Int(f"_{fam.name}_swap_b1")
    b2 = z3.Int(f"_{fam.name}_swap_b2")
    m = z3.Int(f"_{fam.name}_swap_m")
    n = z3.Int(f"_{fam.name}_swap_n")
    k = z3.Int(f"_{fam.name}_swap_k")
    swapped_step = z3.ForAll([k], fam.step(b1, m, n, k) == fam.step(b2, n, m, k))
    if fam.fold_takes_extent:
        e1 = z3.Int(f"_{fam.name}_swap_e1")
        e2 = z3.Int(f"_{fam.name}_swap_e2")
        return z3.ForAll(
            [b1, b2, m, n, e1, e2],
            z3.Implies(
                z3.And(e1 == e2, swapped_step),
                fam.fold(b1, m, n, e1) == fam.fold(b2, n, m, e2),
            ),
        )
    return z3.ForAll(
        [b1, b2, m, n],
        z3.Implies(swapped_step, fam.fold(b1, m, n) == fam.fold(b2, n, m)),
    )


def _make_additivity_axiom(fam: FoldFamily) -> z3.BoolRef:
    """Return the pointwise additivity axiom for a sum fold."""
    b0 = z3.Int(f"_{fam.name}_add_b0")
    b1 = z3.Int(f"_{fam.name}_add_b1")
    b2 = z3.Int(f"_{fam.name}_add_b2")
    outer = [z3.Int(f"_{fam.name}_add_o{n}") for n in range(fam.outer_arity)]
    k = z3.Int(f"_{fam.name}_add_k")
    split_step = z3.ForAll(
        [k],
        fam.step(b0, *outer, k) == fam.step(b1, *outer, k) + fam.step(b2, *outer, k),
    )
    if fam.fold_takes_extent:
        e = z3.Int(f"_{fam.name}_add_e")
        return z3.ForAll(
            [b0, b1, b2, *outer, e],
            z3.Implies(
                split_step,
                fam.fold(b0, *outer, e)
                == fam.fold(b1, *outer, e) + fam.fold(b2, *outer, e),
            ),
        )
    return z3.ForAll(
        [b0, b1, b2, *outer],
        z3.Implies(
            split_step,
            fam.fold(b0, *outer) == fam.fold(b1, *outer) + fam.fold(b2, *outer),
        ),
    )


def register_fold_family(
    name: str,
    *,
    outer_arity: int,
    kind: ReductionKind,
    combine_op: str,
    takes_extent: bool,
) -> FoldFamily:
    key = (outer_arity, kind, combine_op)
    if key in _FOLD_FAMILIES:
        raise ValueError(f"fold family already registered for {key}")
    step_arity = 1 + outer_arity + 1
    fold_arity = 1 + outer_arity + (1 if takes_extent else 0)
    step = z3.Function(f"STEP_{name}", *([z3.IntSort()] * step_arity), z3.RealSort())
    fold = z3.Function(f"FOLD_{name}", *([z3.IntSort()] * fold_arity), z3.RealSort())
    fam = FoldFamily(
        name=name,
        outer_arity=outer_arity,
        kind=kind,
        combine_op=combine_op,
        step=step,
        fold=fold,
        fold_takes_extent=takes_extent,
    )
    _FOLD_FAMILIES[key] = fam
    _FOLD_EXTENSIONALITY_FACTS.append(_make_extensionality_axiom(fam))
    if combine_op == "add":
        _FOLD_ADDITIVITY_FACTS.append(_make_additivity_axiom(fam))
    if outer_arity == 2 and combine_op == "add" and kind == ReductionKind.REDUCE:
        _FOLD_SWAP_FACTS.append(_make_swap_axiom(fam))
    return fam


REDUCE1_FAM = register_fold_family(
    "REDUCE1",
    outer_arity=1,
    kind=ReductionKind.REDUCE,
    combine_op="add",
    takes_extent=True,
)
REDUCE2_FAM = register_fold_family(
    "REDUCE2",
    outer_arity=2,
    kind=ReductionKind.REDUCE,
    combine_op="add",
    takes_extent=True,
)
# SCAN2: outer_arity=2 because the fold output is rank-2 in (i, j); the
# extensionality axiom doesn't quantify a fixed extent — `j` is the cutoff.
SCAN2_FAM = register_fold_family(
    "SCAN2",
    outer_arity=2,
    kind=ReductionKind.SCAN,
    combine_op="add",
    takes_extent=False,
)
# Non-additive rank-1 reductions. Extensionality (equal steps give equal folds)
# holds for any combine op, so a max/min/product reduction gets a fold instead
# of an opaque output and can be proved equal by body. Additivity, operand swap,
# and scale lifting stay restricted to "add" (``register_fold_family`` and
# ``_lift_scale_through_fold`` already guard on ``combine_op``).
REDUCE1_MAX_FAM = register_fold_family(
    "REDUCE1_MAX",
    outer_arity=1,
    kind=ReductionKind.REDUCE,
    combine_op="max",
    takes_extent=True,
)
REDUCE1_MIN_FAM = register_fold_family(
    "REDUCE1_MIN",
    outer_arity=1,
    kind=ReductionKind.REDUCE,
    combine_op="min",
    takes_extent=True,
)
REDUCE1_MUL_FAM = register_fold_family(
    "REDUCE1_MUL",
    outer_arity=1,
    kind=ReductionKind.REDUCE,
    combine_op="mul",
    takes_extent=True,
)


def fold_family(
    outer_arity: int, kind: ReductionKind, combine_op: str
) -> FoldFamily | None:
    return _FOLD_FAMILIES.get((outer_arity, kind, combine_op))


# Back-compat aliases — old names continue to work; prefer the family objects.
BODY1, REDUCE1 = REDUCE1_FAM.step, REDUCE1_FAM.fold
BODY2, REDUCE2 = REDUCE2_FAM.step, REDUCE2_FAM.fold
BODY2_SCAN, SCAN2 = SCAN2_FAM.step, SCAN2_FAM.fold
STEP1, STEP2, STEP2_SCAN = REDUCE1_FAM.step, REDUCE2_FAM.step, SCAN2_FAM.step

_POW_FN = z3.Function("NKI_POW", z3.RealSort(), z3.RealSort(), z3.RealSort())
_EXP_FN = z3.Function("NKI_EXP", z3.RealSort(), z3.RealSort())


@dataclass
class Context:
    facts: list[z3.BoolRef] = field(default_factory=list)

    def add(self, *facts: z3.BoolRef) -> None:
        self.facts.extend(facts)

    def extend(self, facts: list[z3.BoolRef]) -> None:
        self.facts.extend(facts)

    def merged(self, *others: Context) -> Context:
        out = Context(list(self.facts))
        for o in others:
            out.extend(o.facts)
        return out

    def as_formula(self) -> z3.BoolRef:
        if not self.facts:
            return z3.BoolVal(True)
        return z3.And(*self.facts)


@dataclass
class ShapeExpr:
    dims: list[z3.ArithRef]

    @property
    def rank(self) -> int:
        return len(self.dims)


@dataclass
class ShapeResult:
    out: ShapeExpr
    ctx: Context


@dataclass
class ReductionDesc:
    body_id: int
    extent: z3.ArithRef
    outer_rank: int
    kind: ReductionKind = ReductionKind.REDUCE
    combine_op: str = "add"
    outer_dims: tuple[z3.ArithRef, ...] | None = None
    output_transform: str | None = None


@dataclass
class Semantics:
    name: str
    shape: ShapeExpr
    fn: z3.FuncDeclRef
    ctx: Context = field(default_factory=Context)
    reduction: ReductionDesc | None = None
    validity: Context = field(default_factory=Context)
    """Facts that must hold for this expression to be well-defined: input
    positivity, shape compatibility, and operation legality. These are proof
    goals for a candidate, never assumptions. Value-defining facts stay in
    ``ctx``."""


@dataclass
class ShapeSemantics:
    """Shape and validity facts compiled without value definitions."""

    shape: ShapeExpr
    validity: Context = field(default_factory=Context)


@dataclass
class Precondition:
    description: str
    constraint: z3.BoolRef


@dataclass
class SymExpr:
    op: str
    inputs: list[SymExpr]
    shape: tuple[z3.ArithRef, ...]
    attrs: dict[str, Any]
    name: str


class SymTensor:
    def __init__(
        self,
        id: str,
        shape: tuple[Any, ...] | None = None,
        expr: SymExpr | None = None,
        rank: int | None = None,
    ):
        self.id = id
        if expr is not None:
            self.expr = expr
            self.shape = expr.shape
            return
        if shape is not None:
            self.shape = tuple(z3.IntVal(d) if isinstance(d, int) else d for d in shape)
        else:
            if rank is None:
                raise ValueError("Either shape or rank must be provided")
            self.shape = tuple(z3.Int(f"{id}_d{k}") for k in range(rank))
        self.expr = SymExpr("input", [], self.shape, {"shape": self.shape}, id)

    @property
    def rank(self) -> int:
        return len(self.shape)


class _ShapeProvedSymTensor(SymTensor):
    """Internal tensor wrapper carrying one exact proved-shape identity."""

    def __init__(self, tensor: SymTensor, proof_key: Any):
        super().__init__(tensor.id, expr=tensor.expr)
        self.proof_key = proof_key


class _ShapeOnlySymTensor(SymTensor):
    """Internal tensor wrapper requesting only the shape proof phase."""

    def __init__(self, tensor: SymTensor, proof_key: Any):
        super().__init__(tensor.id, expr=tensor.expr)
        self.proof_key = proof_key


def _with_shape_only(
    current: SymTensor,
    candidate: SymTensor,
    proof_key: Any,
) -> tuple[SymTensor, SymTensor]:
    return (
        _ShapeOnlySymTensor(current, proof_key),
        _ShapeOnlySymTensor(candidate, proof_key),
    )


def _with_proved_shape(
    current: SymTensor,
    candidate: SymTensor,
    proof_key: Any,
) -> tuple[SymTensor, SymTensor]:
    return (
        _ShapeProvedSymTensor(current, proof_key),
        _ShapeProvedSymTensor(candidate, proof_key),
    )


ShapeRuleFn = Callable[[list[ShapeExpr], dict[str, Any]], ShapeResult]
CompileRuleFn = Callable[[SymExpr, list[Semantics], ShapeExpr], Semantics]
ValidityRuleFn = Callable[[list[ShapeExpr], dict[str, Any]], Context]


@dataclass(frozen=True)
class SemanticsEntry:
    """The shape, compile, and validity rules for one operation."""

    shape_rule: ShapeRuleFn
    compile_rule: CompileRuleFn
    validity_rule: ValidityRuleFn


def _no_extra_validity(ins: list[ShapeExpr], attrs: dict[str, Any]) -> Context:
    return Context()


_SEMANTICS: dict[str, SemanticsEntry | None] = {}


def semantics():
    def _decorator(fn):
        _SEMANTICS.setdefault(fn.__name__, None)
        return fn

    return _decorator


def semantics_hw():
    def _decorator(fn):
        _SEMANTICS.setdefault(fn.__name__, None)
        return fn

    return _decorator


def register_semantics(
    op_name: str,
    shape_rule_fn: ShapeRuleFn,
    compile_rule_fn: CompileRuleFn,
    validity_rule_fn: ValidityRuleFn | None = None,
) -> None:
    _SEMANTICS[op_name] = SemanticsEntry(
        shape_rule=shape_rule_fn,
        compile_rule=compile_rule_fn,
        validity_rule=(
            validity_rule_fn if validity_rule_fn is not None else _no_extra_validity
        ),
    )


def lookup_semantics(op_name: str) -> SemanticsEntry:
    """Return the registered semantics entry for one operation.
    Raises ``KeyError`` for an operation without registered semantics."""
    entry = _SEMANTICS.get(op_name)
    if entry is None:
        raise KeyError(f"no semantics registered for operation '{op_name}'")
    return entry


def _ensure_semantics(
    op_name: str,
    shape_rule_fn: ShapeRuleFn,
    compile_rule_fn: CompileRuleFn,
    validity_rule_fn: ValidityRuleFn | None = None,
) -> None:
    with _SEMANTICS_LOCK:
        if _SEMANTICS.get(op_name) is None:
            register_semantics(
                op_name, shape_rule_fn, compile_rule_fn, validity_rule_fn
            )


def _to_dim(d: Any) -> z3.ArithRef:
    return z3.IntVal(d) if isinstance(d, int) else d


def _is_sym_tensor(x: Any) -> bool:
    return isinstance(x, SymTensor)


def _new_sym_tensor(
    op: str, inputs: list[SymTensor], attrs: dict[str, Any], out_shape: tuple[Any, ...]
) -> SymTensor:
    out_id = attrs.get("name") or _NODE_IDS.next_name(op)
    expr = SymExpr(
        op,
        [i.expr for i in inputs],
        tuple(_to_dim(d) for d in out_shape),
        attrs,
        out_id,
    )
    return SymTensor(out_id, expr=expr)


def _public_broadcast_shape_tuple(
    a_shape: tuple[Any, ...], b_shape: tuple[Any, ...]
) -> tuple[Any, ...]:
    rank = builtins.max(len(a_shape), len(b_shape))
    a_dims = [1] * (rank - len(a_shape)) + list(a_shape)
    b_dims = [1] * (rank - len(b_shape)) + list(b_shape)
    out: list[Any] = []
    for a_dim, b_dim in zip(a_dims, b_dims, strict=True):
        if isinstance(a_dim, int) and isinstance(b_dim, int):
            if a_dim == 1:
                out.append(b_dim)
                continue
            if b_dim == 1 or a_dim == b_dim:
                out.append(a_dim)
                continue
        out.append(
            z3.If(_to_dim(a_dim) == z3.IntVal(1), _to_dim(b_dim), _to_dim(a_dim))
        )
    return tuple(out)


def _public_reduce_out_shape(
    shape: tuple[Any, ...], axis: Any, keepdims: bool
) -> tuple[Any, ...]:
    if axis is None:
        return shape
    rank = len(shape)
    axes = [axis] if isinstance(axis, int) else list(axis)
    norm_axes: set[int] = set()
    for ax in axes:
        if isinstance(ax, int):
            norm_axes.add(ax + rank if ax < 0 else ax)
    out: list[Any] = []
    for i, dim in enumerate(shape):
        if i in norm_axes:
            if keepdims:
                out.append(1)
        else:
            out.append(dim)
    return tuple(out) if out else (1,)


def _public_pointwise_binary(x: Any, y: Any, op: _OpRef) -> Any:
    assert _is_sym_tensor(x) or _is_sym_tensor(y)
    if _is_sym_tensor(x) and _is_sym_tensor(y):
        return tensor_tensor(dst=None, data1=x, data2=y, op=op)
    if _is_sym_tensor(x):
        return tensor_scalar(dst=None, data=x, op0=op, operand0=y)
    return tensor_scalar(dst=None, data=y, op0=op, operand0=x, reverse0=True)


def _public_unary(op_name: str, x: Any, **attrs: Any) -> Any:
    assert _is_sym_tensor(x)
    return _new_sym_tensor(op_name, [x], attrs, x.shape)


def _public_binary(op_name: str, x: Any, y: Any, **attrs: Any) -> Any:
    assert _is_sym_tensor(x) or _is_sym_tensor(y)
    if _is_sym_tensor(x) and _is_sym_tensor(y):
        return _new_sym_tensor(
            op_name, [x, y], attrs, _public_broadcast_shape_tuple(x.shape, y.shape)
        )
    if _is_sym_tensor(x):
        return _new_sym_tensor(op_name, [x], {"scalar": y, **attrs}, x.shape)
    return _new_sym_tensor(
        op_name, [y], {"scalar": x, "reverse": True, **attrs}, y.shape
    )


def _public_reduce(
    op_name: str, x: Any, axis: Any, *, keepdims: bool = False, **attrs: Any
) -> Any:
    assert _is_sym_tensor(x)
    out_shape = _public_reduce_out_shape(x.shape, axis, keepdims)
    return _new_sym_tensor(
        op_name, [x], {"axis": axis, "keepdims": keepdims, **attrs}, out_shape
    )


def _public_dynamic_slice(start: Any, size: Any) -> Any:
    assert isinstance(start, int) and isinstance(size, int)
    return slice(start, start + size)


def _public_gather_flattened(data: Any, indices: Any, axis: Any) -> Any:
    assert _is_sym_tensor(data) and _is_sym_tensor(indices)
    out_shape = (data.shape[0], *indices.shape[1:])
    return _new_sym_tensor(
        "gather_flattened", [data, indices], {"axis": axis}, out_shape
    )


def _public_rms_norm(x: Any, w: Any) -> Any:
    assert _is_sym_tensor(x)
    inputs = [value for value in (x, w) if _is_sym_tensor(value)]
    return _new_sym_tensor("rms_norm", inputs, {}, x.shape)


def _public_store(dst: Any, value: Any) -> Any:
    assert _is_sym_tensor(value)
    out_shape = _default_out_shape(dst, value)
    return _new_sym_tensor("store", [value], {"out_shape": out_shape}, out_shape)


def _public_transpose_out_shape(x: SymTensor) -> tuple[Any, ...]:
    return (x.shape[1], x.shape[0]) if len(x.shape) == 2 else x.shape


def _arith_equal(lhs: Any, rhs: Any) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    lhs_expr = _to_dim(lhs) if isinstance(lhs, (int, z3.ArithRef)) else lhs
    rhs_expr = _to_dim(rhs) if isinstance(rhs, (int, z3.ArithRef)) else rhs
    if isinstance(lhs_expr, z3.ArithRef) and isinstance(rhs_expr, z3.ArithRef):
        return z3.is_true(z3.simplify(lhs_expr == rhs_expr))
    return lhs_expr == rhs_expr


def _broadcast_shape(a: ShapeExpr, b: ShapeExpr) -> ShapeResult:
    rank = builtins.max(a.rank, b.rank)
    ad = [z3.IntVal(1)] * (rank - a.rank) + list(a.dims)
    bd = [z3.IntVal(1)] * (rank - b.rank) + list(b.dims)
    out: list[z3.ArithRef] = []
    ctx = Context([d > 0 for d in ad + bd])
    for da, db in zip(ad, bd, strict=True):
        ctx.add(z3.Or(da == db, da == 1, db == 1))
        if _arith_equal(da, 1):
            out.append(db)
        elif _arith_equal(db, 1):
            out.append(da)
        else:
            out.append(z3.If(da == 1, db, da))
    return ShapeResult(ShapeExpr(out), ctx)


def _broadcast_to_shape(src: ShapeExpr, target: ShapeExpr) -> ShapeResult:
    rank = builtins.max(src.rank, target.rank)
    sd = [z3.IntVal(1)] * (rank - src.rank) + list(src.dims)
    td = [z3.IntVal(1)] * (rank - target.rank) + list(target.dims)
    ctx = Context([d > 0 for d in sd + td])
    for sdim, tdim in zip(sd, td, strict=True):
        ctx.add(z3.Or(sdim == tdim, sdim == 1))
    return ShapeResult(ShapeExpr(list(td)), ctx)


def _broadcast_indices(
    src: Semantics, out_shape: ShapeExpr, indices: list[z3.ArithRef]
) -> list[z3.ArithRef]:
    src_dims = list(src.shape.dims)
    if len(indices) < out_shape.rank:
        aligned_indices = [z3.IntVal(0)] * (out_shape.rank - len(indices)) + list(
            indices
        )
    elif len(indices) > out_shape.rank:
        aligned_indices = list(indices[len(indices) - out_shape.rank :])
    else:
        aligned_indices = list(indices)
    src_to_out: list[int | None] = [None] * len(src_dims)
    out_pos = out_shape.rank - 1
    for src_pos in range(len(src_dims) - 1, -1, -1):
        src_dim = src_dims[src_pos]
        if out_pos >= 0:
            src_to_out[src_pos] = out_pos
            out_pos -= 1
            continue
        src_to_out[src_pos] = None
    out: list[z3.ArithRef] = []
    for src_pos, src_dim in enumerate(src_dims):
        mapped_pos = src_to_out[src_pos]
        if mapped_pos is None or _arith_equal(src_dim, 1):
            out.append(z3.IntVal(0))
            continue
        index = aligned_indices[mapped_pos]
        out_dim = out_shape.dims[mapped_pos]
        if z3.is_int_value(src_dim) or _arith_equal(src_dim, out_dim):
            # A literal non-one dim, or the output's own dim, never broadcasts.
            out.append(index)
        else:
            # A symbolic dim may be 1 at runtime. Clamp the read here, the way
            # TensorRight composes accesses, instead of asserting a global
            # "dim == 1 implies fn(i) == fn(0)" axiom on every node: that axiom
            # contradicts any definition whose value depends on an index outside
            # the tensor (iota, affine_select), which made the context unsat.
            out.append(z3.If(src_dim == 1, z3.IntVal(0), index))
    return out


def _operand_to_expr(op: Any) -> Any:
    return op.name if isinstance(op, _OpRef) else op


def _safe_divide(lhs: z3.ArithRef, rhs: z3.ArithRef) -> z3.ArithRef:
    return z3.If(rhs == 0, z3.RealVal(0), lhs / rhs)


def _normalize_combine_op(op: Any) -> str:
    """Map a reduction-op argument to the canonical combine_op string used in
    fold-family keys. None / missing op → "add" (sum is the historical default
    for reduce-shaped ops without an explicit op attr, e.g. reduce_sum)."""
    if op is None:
        return "add"
    opn = _operand_to_expr(op)
    if opn is None:
        return "add"
    if opn in ("add", "plus"):
        return "add"
    if opn in ("multiply", "mul"):
        return "mul"
    if opn in ("maximum", "max"):
        return "max"
    if opn in ("minimum", "min"):
        return "min"
    return str(opn)


def _apply_binary(op: Any, lhs: z3.ArithRef, rhs: z3.ArithRef) -> z3.ArithRef:
    opn = _operand_to_expr(op)
    if opn in ("add", "plus", "maximum", "max"):
        return lhs + rhs if opn in ("add", "plus") else z3.If(lhs >= rhs, lhs, rhs)
    if opn in ("subtract", "sub"):
        return lhs - rhs
    if opn in ("multiply", "mul"):
        return lhs * rhs
    if opn in ("divide", "div"):
        return _safe_divide(lhs, rhs)
    if opn in ("minimum", "min"):
        return z3.If(lhs <= rhs, lhs, rhs)
    if opn in ("equal",):
        return z3.If(lhs == rhs, z3.RealVal(1), z3.RealVal(0))
    if opn in ("less",):
        return z3.If(lhs < rhs, z3.RealVal(1), z3.RealVal(0))
    if opn in ("less_equal",):
        return z3.If(lhs <= rhs, z3.RealVal(1), z3.RealVal(0))
    if opn in ("greater",):
        return z3.If(lhs > rhs, z3.RealVal(1), z3.RealVal(0))
    if opn in ("greater_equal",):
        return z3.If(lhs >= rhs, z3.RealVal(1), z3.RealVal(0))
    if opn in ("power",):
        exponent = z3.simplify(rhs)
        if z3.is_rational_value(exponent):
            value = exponent.as_fraction()
            if value == 1:
                return lhs
            if value == 2:
                return lhs * lhs
            if value == 0.5:
                sqrt_value, _ = _apply_activation("sqrt", lhs)
                return sqrt_value
            if value == -1:
                return _safe_divide(z3.RealVal(1), lhs)
        return _POW_FN(lhs, rhs)
    key = str(opn)
    with _UF_LOCK:
        if key not in _BINARY_UFS:
            _BINARY_UFS[key] = z3.Function(
                f"NKI_BIN_{len(_BINARY_UFS)}",
                z3.RealSort(),
                z3.RealSort(),
                z3.RealSort(),
            )
        fn = _BINARY_UFS[key]
    return fn(lhs, rhs)


def _unary_uf(key: str) -> z3.FuncDeclRef:
    """The shared uninterpreted function for one named elementwise op."""
    with _UF_LOCK:
        if key not in _UNARY_UFS:
            _UNARY_UFS[key] = z3.Function(
                f"NKI_UN_{len(_UNARY_UFS)}", z3.RealSort(), z3.RealSort()
            )
        return _UNARY_UFS[key]


def _apply_activation(op: Any, x: z3.ArithRef) -> tuple[z3.ArithRef, list[z3.BoolRef]]:
    """Value of one elementwise activation plus facts about that value.

    Functions with no closed form stay uninterpreted, but ops that are
    compositions of others are defined through them, so two spellings of the
    same function meet at the same term: ``silu(x) = x * sigmoid(x)`` and
    ``rsqrt(x) = reciprocal(sqrt(x))``. The returned facts are quantifier-free
    and mention only ``x``; the caller closes them over its index variables."""
    opn = _operand_to_expr(op)
    if opn in ("copy", "identity", None):
        return x, []
    if opn in ("relu",):
        return z3.If(x >= 0, x, z3.RealVal(0)), []
    if opn in ("exp", "exponential"):
        y = _EXP_FN(x)
        return y, [y > 0, z3.Implies(x == 0, y == 1)]
    if opn in ("reciprocal",):
        return z3.If(x == 0, z3.RealVal(0), z3.RealVal(1) / x), []
    if opn in ("square",):
        return x * x, []
    if opn in ("negative", "neg"):
        return -x, []
    if opn in ("abs",):
        return z3.If(x >= 0, x, -x), []
    if opn in ("sqrt",):
        y = _unary_uf("sqrt")(x)
        return y, [y >= 0, z3.Implies(x >= 0, y * y == x), z3.Implies(x == 0, y == 0)]
    if opn in ("rsqrt",):
        root, facts = _apply_activation("sqrt", x)
        return z3.If(root == 0, z3.RealVal(0), z3.RealVal(1) / root), facts
    if opn in ("sigmoid",):
        y = _unary_uf("sigmoid")(x)
        return y, [y > 0, y < 1]
    if opn in ("silu",):
        sig, facts = _apply_activation("sigmoid", x)
        return x * sig, facts
    return _unary_uf(str(opn))(x), []


def singleton_dimension_extensionality(sem: Semantics) -> Context:
    ctx = Context()
    if sem.shape.rank == 0:
        return ctx
    vars = _index_vars(f"{sem.name}_singleton", sem.shape.rank, "s")
    for axis, dim in enumerate(sem.shape.dims):
        zero_idx = list(vars)
        zero_idx[axis] = z3.IntVal(0)
        ctx.add(
            z3.ForAll(vars, z3.Implies(dim == 1, sem.fn(*vars) == sem.fn(*zero_idx)))
        )
    return ctx


def _shape_eq(a: ShapeExpr, b: ShapeExpr) -> z3.BoolRef:
    if a.rank != b.rank:
        return z3.BoolVal(False)
    out = z3.BoolVal(True)
    for da, db in zip(a.dims, b.dims, strict=True):
        out = z3.And(out, da == db)
    return out


def reduction_extensionality_context() -> Context:
    ctx = Context()
    ctx.extend(list(_FOLD_EXTENSIONALITY_FACTS))
    return ctx


def reduction_swap_context() -> Context:
    """Operand-swap lemmas for rank-2 sum folds. A nested matmul's step goal
    holds inner fold terms, so the body fallback consumes these too."""
    ctx = Context()
    ctx.extend(list(_FOLD_SWAP_FACTS))
    return ctx


def reduction_additivity_context() -> Context:
    """Return distribution lemmas for sum folds."""
    ctx = Context()
    ctx.extend(list(_FOLD_ADDITIVITY_FACTS))
    return ctx


def _decl_names(formulas: Iterable[z3.ExprRef]) -> set[str]:
    """Names of every function symbol applied anywhere in ``formulas``."""
    names: set[str] = set()
    seen: set[int] = set()
    stack = list(formulas)
    while stack:
        node = stack.pop()
        key = node.get_id()
        if key in seen:
            continue
        seen.add(key)
        if z3.is_app(node):
            names.add(node.decl().name())
            stack.extend(node.children())
        elif z3.is_quantifier(node):
            stack.append(node.body())
    return names


def _relevant_facts(facts: list[z3.BoolRef], mentioned: set[str]) -> list[z3.BoolRef]:
    """Keep the lemmas whose fold or function symbols occur in the goal.

    A lemma about a symbol the goal never mentions cannot contribute to a proof;
    it only adds quantifiers for Z3 to instantiate. Every registered family's
    axioms used to ride along on every value query."""
    kept: list[z3.BoolRef] = []
    for fact in facts:
        owners = {
            name
            for name in _decl_names([fact])
            if name.startswith(("FOLD_", "NKI_EXP", "NKI_POW"))
        }
        # A lemma with no recognised owner symbol is kept: filtering is only
        # an optimisation and must never drop something it cannot classify.
        if not owners or owners & mentioned:
            kept.append(fact)
    return kept


# Exponential algebra. ``exp`` is an uninterpreted function, so without these
# lemmas ``exp(a + b)`` and ``exp(a) * exp(b)`` are unrelated terms and the
# softmax / online-softmax rescaling rewrites can never be admitted. Each lemma
# has an explicit trigger on the ``exp`` of a sum or difference, so instantiation
# only fires on terms the goal already contains and cannot loop.
def _make_exp_algebra_facts() -> list[z3.BoolRef]:
    a, b = z3.Reals("_exp_alg_a _exp_alg_b")
    add_rule = z3.ForAll(
        [a, b],
        _EXP_FN(a + b) == _EXP_FN(a) * _EXP_FN(b),
        patterns=[_EXP_FN(a + b)],
    )
    sub_rule = z3.ForAll(
        [a, b],
        _EXP_FN(a - b) * _EXP_FN(b) == _EXP_FN(a),
        patterns=[_EXP_FN(a - b)],
    )
    positive = z3.ForAll([a], _EXP_FN(a) > 0, patterns=[_EXP_FN(a)])
    return [add_rule, sub_rule, positive, _EXP_FN(z3.RealVal(0)) == 1]


_EXP_ALGEBRA_FACTS: list[z3.BoolRef] = _make_exp_algebra_facts()


def elementwise_algebra_context(mentioned: set[str] | None = None) -> Context:
    """Lemmas for interpreted-but-uninterpreted elementwise functions.

    Pass the symbol names the goal mentions to keep only the relevant lemmas."""
    facts = list(_EXP_ALGEBRA_FACTS)
    if mentioned is not None:
        facts = _relevant_facts(facts, mentioned)
    return Context(facts)


# Goals at or above this many AST nodes go to Z3's tuned tactic instead of the
# `auto_config` off fast path. Calibrated on the tensor-propagation obligations,
# where the decided ones hold 32 to 34 nodes and the ones that exhaust their
# resource limit hold 76. Recalibrate together with `_RLIMIT_PER_MS`.
_TUNED_TACTIC_MIN_NODES = 64


def _ast_size_at_least(assertions: list[z3.BoolRef], cap: int) -> bool:
    """Whether the goal holds at least ``cap`` distinct AST nodes.

    Stops at the cap, so the walk costs O(cap) rather than O(goal)."""
    seen: set[int] = set()
    stack = list(assertions)
    while stack:
        node = stack.pop()
        key = node.get_id()
        if key in seen:
            continue
        seen.add(key)
        if len(seen) >= cap:
            return True
        if z3.is_app(node):
            stack.extend(node.children())
        elif z3.is_quantifier(node):
            stack.append(node.body())
    return False


def _check_deterministic(
    assertions: list[z3.BoolRef], timeout: int
) -> z3.CheckSatResult:
    """Run a bounded solver check and retry ``unknown`` with MBQI.

    Each call solves in a private Z3 context so process-global solver state cannot
    make a byte-identical query path-dependent. The wall timeout and the resource
    limit are both fixed, and the resource limit is what makes a verdict
    reproducible: NLSAT ignores wall time.

    `auto_config` off is a speed choice, not a capability one. It keeps Z3 off a
    tactic that roughly doubles the e-graph suite's time, but a large goal only
    closes on that tuned tactic; below, a goal of 76 nodes exhausts its resource
    limit and returns `unknown`, while every caller reads `unknown` as "not
    equivalent". The rewrite is then dropped rather than reported, which is how the
    distributivity-through-matmul rewrite went missing. So the size of the goal, not
    a retry, picks the tactic: retrying instead spends the round's time budget and
    loses the rewrite a second way.
    """
    ctx = z3.Context()
    local_assertions = [assertion.translate(ctx) for assertion in assertions]
    deadline = time.monotonic() + timeout / 1000.0
    rlimit = timeout * _RLIMIT_PER_MS
    solver = z3.Solver(ctx=ctx)
    if not _ast_size_at_least(local_assertions, _TUNED_TACTIC_MIN_NODES):
        solver.set("auto_config", False)
    solver.set("timeout", timeout)
    solver.set("rlimit", rlimit)
    for assertion in local_assertions:
        solver.add(assertion)
    res = solver.check()
    if res != z3.unknown:
        return res
    retry_timeout = math.ceil((deadline - time.monotonic()) * 1000)
    if retry_timeout <= 0:
        return z3.unknown
    retry_timeout = min(timeout, retry_timeout)
    solver = z3.Solver(ctx=ctx)
    solver.set("timeout", retry_timeout)
    solver.set("rlimit", retry_timeout * _RLIMIT_PER_MS)
    solver.set("smt.ematching", False)
    solver.set("smt.mbqi", True)
    solver.set("random_seed", 0)
    for assertion in local_assertions:
        solver.add(assertion)
    return solver.check()


def _check_reduction_equivalent_by_body(
    lhs: Semantics,
    rhs: Semantics,
    shape_eq: z3.BoolRef,
    timeout: int,
    full_ctx: Context,
    *,
    shape_proved: bool = False,
) -> bool:
    """Compare fold bodies under the caller's sound assumptions."""
    if lhs.reduction is None or rhs.reduction is None:
        return False
    if lhs.reduction.outer_rank != rhs.reduction.outer_rank:
        return False
    if lhs.reduction.kind != rhs.reduction.kind:
        return False
    if lhs.reduction.combine_op != rhs.reduction.combine_op:
        return False
    supported_transforms = {"identity", "negate"}
    if (
        lhs.reduction.output_transform not in supported_transforms
        or rhs.reduction.output_transform not in supported_transforms
        or lhs.reduction.output_transform != rhs.reduction.output_transform
        or lhs.reduction.outer_dims is None
        or rhs.reduction.outer_dims is None
    ):
        return False
    fam = fold_family(
        lhs.reduction.outer_rank, lhs.reduction.kind, lhs.reduction.combine_op
    )
    if (
        fam is None
        or len(lhs.reduction.outer_dims) != fam.outer_arity
        or len(rhs.reduction.outer_dims) != fam.outer_arity
    ):
        return False
    deadline = time.monotonic() + timeout / 1000.0

    def remaining_timeout() -> int | None:
        remaining = math.ceil((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            return None
        return min(timeout, remaining)

    if not shape_proved:
        shape_timeout = remaining_timeout()
        if shape_timeout is None or (
            _check_deterministic(
                [full_ctx.as_formula(), z3.Not(shape_eq)],
                shape_timeout,
            )
            != z3.unsat
        ):
            return False

    extent_timeout = remaining_timeout()
    if extent_timeout is None or (
        _check_deterministic(
            [full_ctx.as_formula(), lhs.reduction.extent != rhs.reduction.extent],
            extent_timeout,
        )
        != z3.unsat
    ):
        return False
    outer_dims_eq = z3.And(
        *[
            lhs_dim == rhs_dim
            for lhs_dim, rhs_dim in zip(
                lhs.reduction.outer_dims, rhs.reduction.outer_dims, strict=True
            )
        ]
    )
    outer_timeout = remaining_timeout()
    if outer_timeout is None or (
        _check_deterministic(
            [full_ctx.as_formula(), z3.Not(outer_dims_eq)], outer_timeout
        )
        != z3.unsat
    ):
        return False

    outer_vars = [z3.Int(f"body_eq_{fam.name}_o{n}") for n in range(fam.outer_arity)]
    k = z3.Int(f"body_eq_{fam.name}_k")
    outer_bounds = [
        z3.And(o >= 0, o < dim)
        for o, dim in zip(outer_vars, lhs.reduction.outer_dims, strict=True)
    ]
    if fam.kind == ReductionKind.SCAN:
        # k bounded by the cutoff `j`, which is the last outer index.
        k_bound = z3.And(k >= 0, k <= outer_vars[-1])
    else:
        k_bound = z3.And(k >= 0, k < lhs.reduction.extent)
    counterexample_formula = z3.And(
        shape_eq,
        lhs.reduction.extent == rhs.reduction.extent,
        z3.And(*outer_bounds, k_bound),
        fam.step(z3.IntVal(lhs.reduction.body_id), *outer_vars, k)
        != fam.step(z3.IntVal(rhs.reduction.body_id), *outer_vars, k),
    )
    body_timeout = remaining_timeout()
    return body_timeout is not None and (
        _check_deterministic(
            [full_ctx.as_formula(), counterexample_formula], body_timeout
        )
        == z3.unsat
    )


def compile_expr(expr: SymExpr, cache: dict[int, Semantics]) -> Semantics:
    key = id(expr)
    if key in cache:
        return cache[key]

    if expr.op == "input":
        shape = ShapeExpr(list(expr.shape))
        sem = Semantics(
            name=expr.name,
            shape=shape,
            fn=_tensor_function(f"V_{expr.name}", shape.rank),
            validity=Context([d > 0 for d in shape.dims]),
        )
        sem.ctx = sem.ctx.merged(singleton_dimension_extensionality(sem))
        cache[key] = sem
        return sem

    compiled_inputs = [compile_expr(inp, cache) for inp in expr.inputs]
    input_shapes = [c.shape for c in compiled_inputs]
    entry = _SEMANTICS.get(expr.op)
    if entry is None:
        raise KeyError(f"No semantics registered for op '{expr.op}'")
    shape_res = entry.shape_rule(input_shapes, expr.attrs)
    sem = entry.compile_rule(expr, compiled_inputs, shape_res.out)
    # No singleton axiom here. Broadcast reads clamp their own index
    # (``_broadcast_indices``), so a derived tensor needs no extra fact, and the
    # axiom is unsound for a definition that depends on its raw index: with a
    # size-1 axis it forced ``iota(i, j) == iota(i, 0)`` for every ``j``, a
    # contradiction that let any candidate "prove" equal. Inputs keep the axiom
    # because they are unconstrained functions, so it cannot conflict.
    validity = Context()
    for compiled in compiled_inputs:
        validity.extend(compiled.validity.facts)
    validity.extend(shape_res.ctx.facts)
    validity.extend(entry.validity_rule(input_shapes, expr.attrs).facts)
    sem.validity = sem.validity.merged(validity)
    cache[key] = sem
    return sem


def compile_shape_expr(
    expr: SymExpr, cache: dict[int, ShapeSemantics]
) -> ShapeSemantics:
    """Compile only rank, shape, and validity constraints for ``expr``."""
    key = id(expr)
    if key in cache:
        return cache[key]

    if expr.op == "input":
        shape = ShapeExpr(list(expr.shape))
        sem = ShapeSemantics(shape=shape, validity=Context([d > 0 for d in shape.dims]))
        cache[key] = sem
        return sem

    compiled_inputs = [compile_shape_expr(inp, cache) for inp in expr.inputs]
    input_shapes = [compiled.shape for compiled in compiled_inputs]
    entry = _SEMANTICS.get(expr.op)
    if entry is None:
        raise KeyError(f"No semantics registered for op '{expr.op}'")
    shape_res = entry.shape_rule(input_shapes, expr.attrs)
    validity = Context()
    for compiled in compiled_inputs:
        validity.extend(compiled.validity.facts)
    validity.extend(shape_res.ctx.facts)
    validity.extend(entry.validity_rule(input_shapes, expr.attrs).facts)
    sem = ShapeSemantics(shape=shape_res.out, validity=validity)
    cache[key] = sem
    return sem


def _encode_shape_dim(dim: z3.ExprRef) -> Any:
    """Encode one shape dimension from its Z3 AST without pretty-printing."""
    if z3.is_int_value(dim):
        return ("int", dim.as_long())
    children = dim.children()
    if not children:
        return ("const", dim.decl().name())
    return (dim.decl().name(), tuple(_encode_shape_dim(c) for c in children))


def _expr_structural_key(expr: SymExpr, cache: dict[int, Any] | None = None) -> Any:
    cache = {} if cache is None else cache
    key = id(expr)
    if key in cache:
        return cache[key]
    if expr.op == "input":
        sig = ("input", expr.name, tuple(_encode_shape_dim(d) for d in expr.shape))
    else:
        sig = (
            expr.op,
            tuple(_encode_shape_dim(d) for d in expr.shape),
            tuple(sorted((k, repr(v)) for k, v in expr.attrs.items())),
            tuple(_expr_structural_key(inp, cache) for inp in expr.inputs),
        )
    cache[key] = sig
    return sig


_IndexedExpr = tuple[Any, ...]


def _indexed_dim_key(dim: z3.ArithRef) -> Any:
    return _encode_shape_dim(z3.simplify(dim))


def _indexed_shape_key(shape: tuple[z3.ArithRef, ...]) -> tuple[Any, ...]:
    return tuple(_indexed_dim_key(dim) for dim in shape)


def _indexed_same_shape(
    lhs: tuple[z3.ArithRef, ...], rhs: tuple[z3.ArithRef, ...]
) -> bool:
    return _indexed_shape_key(lhs) == _indexed_shape_key(rhs)


def _indexed_default_attrs(attrs: dict[str, Any], defaults: dict[str, Any]) -> bool:
    return set(attrs).issubset(defaults) and all(
        type(attrs[key]) is type(defaults[key]) and attrs[key] == defaults[key]
        for key in attrs
    )


def _indexed_mul(*values: _IndexedExpr) -> _IndexedExpr:
    factors: list[_IndexedExpr] = []
    for value in values:
        if value[0] == "mul":
            factors.extend(value[1])
        else:
            factors.append(value)
    return ("mul", tuple(sorted(factors, key=repr)))


def _indexed_expr_normal_form(
    expr: SymExpr,
    indices: tuple[_IndexedExpr, _IndexedExpr],
    binder_depth: int = 0,
) -> _IndexedExpr | None:
    """Normalize the small rank-2 matmul/ReLU/multiply extraction fragment."""
    if len(expr.shape) != 2:
        return None

    if expr.op == "input":
        if expr.inputs or set(expr.attrs) - {"shape"}:
            return None
        attr_shape = expr.attrs.get("shape")
        if attr_shape is not None and (
            not isinstance(attr_shape, tuple)
            or not _indexed_same_shape(expr.shape, attr_shape)
        ):
            return None
        return ("input", expr.name, indices)

    if expr.op == "matmul":
        if len(expr.inputs) != 2 or not _indexed_default_attrs(
            expr.attrs, {"transpose_x": False}
        ):
            return None
        lhs, rhs = expr.inputs
        if (
            len(lhs.shape) != 2
            or len(rhs.shape) != 2
            or not _indexed_same_shape((lhs.shape[1],), (rhs.shape[0],))
            or not _indexed_same_shape(expr.shape, (lhs.shape[0], rhs.shape[1]))
        ):
            return None
        reduction_index: _IndexedExpr = ("bound", binder_depth)
        lhs_value = _indexed_expr_normal_form(
            lhs, (indices[0], reduction_index), binder_depth + 1
        )
        rhs_value = _indexed_expr_normal_form(
            rhs, (reduction_index, indices[1]), binder_depth + 1
        )
        if lhs_value is None or rhs_value is None:
            return None
        return (
            "sum",
            _indexed_dim_key(lhs.shape[1]),
            _indexed_mul(lhs_value, rhs_value),
        )

    if expr.op == "nc_matmul":
        allowed_attrs = {
            "is_stationary_onezero",
            "is_moving_onezero",
            "is_transpose",
            "accumulate",
            "tile_position",
            "tile_size",
            "perf_mode",
            "out_shape",
            "name",
        }
        if len(expr.inputs) != 2 or set(expr.attrs) - allowed_attrs:
            return None
        if (
            expr.attrs.get("is_stationary_onezero", False) is not False
            or expr.attrs.get("is_moving_onezero", False) is not False
            or expr.attrs.get("is_transpose", False) is not False
            or (
                expr.attrs.get("accumulate") is not None
                and expr.attrs.get("accumulate") is not False
            )
            or expr.attrs.get("tile_position", ()) != ()
            or expr.attrs.get("tile_size", ()) != ()
            or expr.attrs.get("perf_mode", matmul_perf_mode.none)
            != matmul_perf_mode.none
            or expr.attrs.get("name") is not None
        ):
            return None
        out_shape = expr.attrs.get("out_shape")
        if out_shape is not None and (
            not isinstance(out_shape, tuple)
            or not _indexed_same_shape(expr.shape, out_shape)
        ):
            return None
        stationary, moving = expr.inputs
        if (
            len(stationary.shape) != 2
            or len(moving.shape) != 2
            or not _indexed_same_shape((stationary.shape[0],), (moving.shape[0],))
            or not _indexed_same_shape(
                expr.shape, (stationary.shape[1], moving.shape[1])
            )
        ):
            return None
        reduction_index = ("bound", binder_depth)
        stationary_value = _indexed_expr_normal_form(
            stationary, (reduction_index, indices[0]), binder_depth + 1
        )
        moving_value = _indexed_expr_normal_form(
            moving, (reduction_index, indices[1]), binder_depth + 1
        )
        if stationary_value is None or moving_value is None:
            return None
        return (
            "sum",
            _indexed_dim_key(stationary.shape[0]),
            _indexed_mul(stationary_value, moving_value),
        )

    if expr.op == "nc_transpose":
        if (
            len(expr.inputs) != 1
            or not _indexed_default_attrs(
                expr.attrs, {"engine": engine.unknown, "name": None}
            )
            or len(expr.inputs[0].shape) != 2
            or not _indexed_same_shape(
                expr.shape, (expr.inputs[0].shape[1], expr.inputs[0].shape[0])
            )
        ):
            return None
        return _indexed_expr_normal_form(
            expr.inputs[0], (indices[1], indices[0]), binder_depth
        )

    if expr.op == "relu":
        if (
            len(expr.inputs) != 1
            or not _indexed_default_attrs(expr.attrs, {"op": "relu"})
            or not _indexed_same_shape(expr.shape, expr.inputs[0].shape)
        ):
            return None
        value = _indexed_expr_normal_form(expr.inputs[0], indices, binder_depth)
        return None if value is None else ("relu", value)

    if expr.op == "activation":
        allowed_attrs = {
            "op",
            "scale",
            "reduce_op",
            "reduce_cmd",
            "name",
            "with_reduce",
            "bias_const",
        }
        scale = expr.attrs.get("scale")
        if (
            len(expr.inputs) != 1
            or set(expr.attrs) - allowed_attrs
            or _operand_to_expr(expr.attrs.get("op")) != "relu"
            or "scale" not in expr.attrs
            or type(scale) not in (int, float)
            or float(scale) != 1.0
            or expr.attrs.get("reduce_op") is not None
            or expr.attrs.get("reduce_cmd", reduce_cmd.idle) != reduce_cmd.idle
            or expr.attrs.get("name") is not None
            or expr.attrs.get("with_reduce", False) is not False
            or expr.attrs.get("bias_const") is not None
            or not _indexed_same_shape(expr.shape, expr.inputs[0].shape)
        ):
            return None
        value = _indexed_expr_normal_form(expr.inputs[0], indices, binder_depth)
        return None if value is None else ("relu", value)

    if expr.op in {"mul", "multiply"}:
        expected_op = expr.op
        if (
            len(expr.inputs) != 2
            or not _indexed_default_attrs(expr.attrs, {"op": expected_op})
            or any(
                len(inp.shape) != 2 or not _indexed_same_shape(expr.shape, inp.shape)
                for inp in expr.inputs
            )
        ):
            return None
        lhs = _indexed_expr_normal_form(expr.inputs[0], indices, binder_depth)
        rhs = _indexed_expr_normal_form(expr.inputs[1], indices, binder_depth)
        return None if lhs is None or rhs is None else _indexed_mul(lhs, rhs)

    if expr.op == "tensor_tensor":
        if (
            len(expr.inputs) != 2
            or set(expr.attrs) - {"op", "engine", "name"}
            or _operand_to_expr(expr.attrs.get("op")) != "multiply"
            or expr.attrs.get("engine", engine.unknown) != engine.unknown
            or expr.attrs.get("name") is not None
            or any(
                len(inp.shape) != 2 or not _indexed_same_shape(expr.shape, inp.shape)
                for inp in expr.inputs
            )
        ):
            return None
        lhs = _indexed_expr_normal_form(expr.inputs[0], indices, binder_depth)
        rhs = _indexed_expr_normal_form(expr.inputs[1], indices, binder_depth)
        return None if lhs is None or rhs is None else _indexed_mul(lhs, rhs)

    return None


def _indexed_exprs_equivalent(lhs: SymExpr, rhs: SymExpr) -> bool:
    if (
        len(lhs.shape) != 2
        or len(rhs.shape) != 2
        or not _indexed_same_shape(lhs.shape, rhs.shape)
    ):
        return False
    output_indices: tuple[_IndexedExpr, _IndexedExpr] = (
        ("free", 0),
        ("free", 1),
    )
    lhs_normal = _indexed_expr_normal_form(lhs, output_indices)
    if lhs_normal is None:
        return False
    rhs_normal = _indexed_expr_normal_form(rhs, output_indices)
    return rhs_normal is not None and lhs_normal == rhs_normal


def _rename_expr_tree_for_equivalence(
    expr: SymExpr,
    side: str,
    shared_names: dict[Any, str],
    used_names: set[str],
    name_counter: list[int],
    cache: dict[int, SymExpr] | None = None,
    sig_cache: dict[int, Any] | None = None,
) -> SymExpr:
    cache = {} if cache is None else cache
    sig_cache = {} if sig_cache is None else sig_cache
    key = id(expr)
    if key in cache:
        return cache[key]

    if expr.op == "input":
        renamed = SymExpr(expr.op, [], expr.shape, dict(expr.attrs), expr.name)
        cache[key] = renamed
        return renamed

    sig = _expr_structural_key(expr, sig_cache)
    if sig in shared_names:
        name = shared_names[sig]
    else:
        candidate_names = [expr.name, f"{expr.name}_{side}"]
        fallback_name = f"{expr.name}_{side}_{name_counter[0]}"
        name = next(
            (candidate for candidate in candidate_names if candidate not in used_names),
            fallback_name,
        )
        if name == fallback_name:
            name_counter[0] += 1
        shared_names[sig] = name
        used_names.add(name)

    renamed = SymExpr(
        expr.op,
        [
            _rename_expr_tree_for_equivalence(
                inp, side, shared_names, used_names, name_counter, cache, sig_cache
            )
            for inp in expr.inputs
        ],
        expr.shape,
        dict(expr.attrs),
        name,
    )
    cache[key] = renamed
    return renamed


def _compile_pair_for_equivalence(
    lhs: SymTensor, rhs: SymTensor
) -> tuple[Semantics, Semantics]:
    shared_names: dict[Any, str] = {}
    used_names: set[str] = set()
    name_counter = [0]
    lhs_expr = _rename_expr_tree_for_equivalence(
        lhs.expr, "lhs", shared_names, used_names, name_counter
    )
    rhs_expr = _rename_expr_tree_for_equivalence(
        rhs.expr, "rhs", shared_names, used_names, name_counter
    )
    return compile_expr(lhs_expr, {}), compile_expr(rhs_expr, {})


def _compile_shape_pair_for_equivalence(
    lhs: SymTensor, rhs: SymTensor
) -> tuple[ShapeSemantics, ShapeSemantics]:
    shared_names: dict[Any, str] = {}
    used_names: set[str] = set()
    name_counter = [0]
    lhs_expr = _rename_expr_tree_for_equivalence(
        lhs.expr, "lhs", shared_names, used_names, name_counter
    )
    rhs_expr = _rename_expr_tree_for_equivalence(
        rhs.expr, "rhs", shared_names, used_names, name_counter
    )
    return compile_shape_expr(lhs_expr, {}), compile_shape_expr(rhs_expr, {})


def check_candidate_validity(
    source: Semantics | ShapeSemantics,
    candidate: Semantics | ShapeSemantics,
    assumptions: Context | None = None,
    timeout: int = 10000,
) -> bool:
    """Prove candidate validity from source validity and allowed assumptions."""
    assertions = [source.validity.as_formula()]
    if assumptions is not None:
        assertions.append(assumptions.as_formula())
    assertions.append(z3.Not(candidate.validity.as_formula()))
    return _check_deterministic(assertions, timeout) == z3.unsat


@dataclass(frozen=True)
class EquivalenceVerdict:
    """Structured result of one strict equivalence check."""

    proved: bool
    stage: str
    """``"proved"`` on success, else the failing stage: ``"rank"``,
    ``"validity"``, ``"shape"``, or ``"value"``."""
    detail: str = ""
    used_reduction_fallback: bool = False
    elapsed_ms: int = 0

    def __bool__(self) -> bool:
        return self.proved


def _remaining_timeout(deadline: float, maximum: int) -> int | None:
    remaining = math.ceil((deadline - time.monotonic()) * 1000)
    if remaining <= 0:
        return None
    return min(maximum, remaining)


def check_shape_valid_and_equivalent(
    current: SymTensor,
    candidate: SymTensor,
    timeout: int = 10000,
    preconditions: list[Precondition] | None = None,
    rule_name: str | None = None,
) -> EquivalenceVerdict:
    """Prove rank, candidate validity, and output shape without value rules."""
    del rule_name
    deadline = time.monotonic() + timeout / 1000.0
    lsem, rsem = _compile_shape_pair_for_equivalence(current, candidate)
    if lsem.shape.rank != rsem.shape.rank:
        return EquivalenceVerdict(
            proved=False,
            stage="rank",
            detail=f"rank {lsem.shape.rank} != {rsem.shape.rank}",
        )

    assumptions = Context([p.constraint for p in (preconditions or [])])
    validity_timeout = _remaining_timeout(deadline, timeout)
    if validity_timeout is None or not check_candidate_validity(
        lsem, rsem, assumptions, validity_timeout
    ):
        return EquivalenceVerdict(
            proved=False,
            stage="validity",
            detail="candidate validity not implied by source validity",
        )

    shape_eq = _shape_eq(lsem.shape, rsem.shape)
    shape_timeout = _remaining_timeout(deadline, timeout)
    if shape_timeout is None:
        return EquivalenceVerdict(
            proved=False, stage="shape", detail="candidate deadline exhausted"
        )
    shape_ctx = lsem.validity.merged(assumptions)
    shape_res = _check_deterministic(
        [shape_ctx.as_formula(), z3.Not(shape_eq)], shape_timeout
    )
    if shape_res != z3.unsat:
        return EquivalenceVerdict(proved=False, stage="shape", detail=str(shape_res))
    return EquivalenceVerdict(proved=True, stage="shape")


def _check_value_equivalent_after_shape(
    current: SymTensor,
    candidate: SymTensor,
    timeout: int = 10000,
    preconditions: list[Precondition] | None = None,
    rule_name: str | None = None,
) -> EquivalenceVerdict:
    """Prove values after the identical directional shape obligation proved."""
    del rule_name
    if not (
        isinstance(current, _ShapeProvedSymTensor)
        and isinstance(candidate, _ShapeProvedSymTensor)
        and current.proof_key == candidate.proof_key
    ):
        return EquivalenceVerdict(
            proved=False,
            stage="shape",
            detail="value proof requires an identity-bound shape proof",
        )
    deadline = time.monotonic() + timeout / 1000.0
    lsem, rsem = _compile_pair_for_equivalence(current, candidate)
    assumptions = Context([p.constraint for p in (preconditions or [])])
    semantic_ctx = lsem.validity.merged(assumptions, lsem.ctx, rsem.ctx)
    # Only lemmas about symbols the goal mentions: an unused family's axioms
    # cannot help, they only give the solver more quantifiers to instantiate.
    mentioned = _decl_names(semantic_ctx.facts)
    fallback_ctx = semantic_ctx.merged(
        Context(_relevant_facts(reduction_extensionality_context().facts, mentioned))
    )
    value_ctx = fallback_ctx.merged(
        Context(_relevant_facts(reduction_additivity_context().facts, mentioned)),
        Context(_relevant_facts(reduction_swap_context().facts, mentioned)),
        elementwise_algebra_context(mentioned),
    )
    shape_eq = _shape_eq(lsem.shape, rsem.shape)
    if _indexed_exprs_equivalent(current.expr, candidate.expr):
        return EquivalenceVerdict(
            proved=True, stage="proved", used_reduction_fallback=True
        )
    evaluation = _evaluation_verdict(
        current,
        candidate,
        [*lsem.validity.facts, *assumptions.facts, shape_eq],
        lsem.shape,
        deadline,
        timeout,
    )
    if evaluation is not None:
        return evaluation
    value_assertions = [value_ctx.as_formula(), shape_eq]
    if lsem.shape.rank == 0:
        value_assertions.append(lsem.fn() != rsem.fn())
    else:
        vars = [z3.Int(f"witness_{idx}") for idx in range(lsem.shape.rank)]
        bounds = z3.And(
            *[
                z3.And(vars[i] >= 0, vars[i] < lsem.shape.dims[i])
                for i in range(lsem.shape.rank)
            ]
        )
        value_assertions.append(bounds)
        value_assertions.append(lsem.fn(*vars) != rsem.fn(*vars))
    value_timeout = _remaining_timeout(deadline, timeout)
    if value_timeout is None:
        return EquivalenceVerdict(
            proved=False, stage="value", detail="candidate deadline exhausted"
        )
    value_res = _check_deterministic(value_assertions, value_timeout)
    if value_res == z3.unsat:
        return EquivalenceVerdict(proved=True, stage="proved")
    fallback_timeout = _remaining_timeout(deadline, timeout)
    if fallback_timeout is not None and _check_reduction_equivalent_by_body(
        lsem,
        rsem,
        shape_eq,
        fallback_timeout,
        fallback_ctx,
        shape_proved=True,
    ):
        return EquivalenceVerdict(
            proved=True, stage="proved", used_reduction_fallback=True
        )
    return EquivalenceVerdict(proved=False, stage="value", detail=str(value_res))


# Share of a value proof's budget for the quantifier-free evaluation attempt.
# Its queries are small and usually decide in a few milliseconds; the cap keeps
# a hard evaluation query from starving the quantified encoding behind it.
_EVALUATION_BUDGET_FRACTION = 3
SYMBOLIC_EVALUATION_ENABLED = os.environ.get("AXON_SYMBOLIC_EVAL", "1") != "0"


def _evaluation_verdict(
    current: SymTensor,
    candidate: SymTensor,
    assumptions: list[z3.BoolRef],
    out_shape: ShapeExpr,
    deadline: float,
    timeout: int,
) -> EquivalenceVerdict | None:
    """Try the TensorRight-style symbolic-evaluation proof first.

    Returns a verdict when evaluation decides the obligation, else None so the
    quantified encoding runs. A refutation is only reported when evaluation
    abstracted nothing, so it is a counterexample in the same semantics."""
    if not SYMBOLIC_EVALUATION_ENABLED:
        return None
    from axon.symbolic_eval import Outcome, prove_by_evaluation

    remaining = _remaining_timeout(deadline, timeout)
    if remaining is None:
        return None
    budget = max(1, remaining // _EVALUATION_BUDGET_FRACTION)
    result = prove_by_evaluation(
        current.expr, candidate.expr, assumptions, out_shape, budget
    )
    if result.outcome is Outcome.PROVED:
        return EquivalenceVerdict(
            proved=True,
            stage="proved",
            detail=f"symbolic_evaluation: {result.detail}",
        )
    if result.outcome is Outcome.REFUTED:
        # Same detail the quantified query reports for a counterexample.
        return EquivalenceVerdict(proved=False, stage="value", detail="sat")
    return None


def _fallback_timeout(timeout: int) -> int:
    """Split the caller timeout so the cheap reduction proof cannot starve the rest."""
    return max(1, min(timeout // 2, max(400, timeout // 4)))


def check_value_with_early_reduction_fallback(
    current: SymTensor,
    candidate: SymTensor,
    *,
    timeout: int,
    preconditions: tuple[Precondition, ...],
) -> EquivalenceVerdict:
    """Try the bounded reduction proof before a potentially expensive query."""
    if not (
        isinstance(current, _ShapeProvedSymTensor)
        and isinstance(candidate, _ShapeProvedSymTensor)
        and current.proof_key == candidate.proof_key
    ):
        return _check_value_equivalent_after_shape(
            current,
            candidate,
            timeout=timeout,
            preconditions=list(preconditions),
        )

    started_at = time.monotonic()
    deadline = started_at + timeout / 1000.0

    def elapsed_ms() -> int:
        return max(1, math.ceil((time.monotonic() - started_at) * 1000))

    def deadline_verdict() -> EquivalenceVerdict:
        return EquivalenceVerdict(
            proved=False,
            stage="value",
            detail="candidate deadline exhausted",
            elapsed_ms=elapsed_ms(),
        )

    lhs, rhs = _compile_pair_for_equivalence(current, candidate)
    assumptions = Context([item.constraint for item in preconditions])
    fallback_context = lhs.validity.merged(
        assumptions,
        lhs.ctx,
        rhs.ctx,
        reduction_extensionality_context(),
    )
    fallback_proved = _check_reduction_equivalent_by_body(
        lhs,
        rhs,
        _shape_eq(lhs.shape, rhs.shape),
        _fallback_timeout(timeout),
        fallback_context,
        shape_proved=True,
    )
    if time.monotonic() >= deadline:
        return deadline_verdict()
    if fallback_proved:
        return EquivalenceVerdict(
            proved=True,
            stage="proved",
            used_reduction_fallback=True,
            elapsed_ms=elapsed_ms(),
        )

    remaining = math.ceil((deadline - time.monotonic()) * 1000)
    if remaining <= 0:
        return deadline_verdict()
    verdict = _check_value_equivalent_after_shape(
        current,
        candidate,
        timeout=min(timeout, remaining),
        preconditions=list(preconditions),
    )
    if time.monotonic() >= deadline:
        return deadline_verdict()
    return replace(
        verdict,
        elapsed_ms=elapsed_ms(),
    )


def check_valid_and_equivalent(
    current: SymTensor,
    candidate: SymTensor,
    timeout: int = 10000,
    preconditions: list[Precondition] | None = None,
    rule_name: str | None = None,
) -> EquivalenceVerdict:
    """Strict directional shape-then-value check for e-graph admission."""
    if (
        isinstance(current, _ShapeOnlySymTensor)
        and isinstance(candidate, _ShapeOnlySymTensor)
        and current.proof_key == candidate.proof_key
    ):
        started_at = time.monotonic()
        verdict = check_shape_valid_and_equivalent(
            current,
            candidate,
            timeout=timeout,
            preconditions=preconditions,
            rule_name=rule_name,
        )
        return replace(
            verdict,
            elapsed_ms=max(1, math.ceil((time.monotonic() - started_at) * 1000)),
        )
    if (
        isinstance(current, _ShapeProvedSymTensor)
        and isinstance(candidate, _ShapeProvedSymTensor)
        and current.proof_key == candidate.proof_key
    ):
        started_at = time.monotonic()
        verdict = check_value_with_early_reduction_fallback(
            current,
            candidate,
            timeout=timeout,
            preconditions=tuple(preconditions or ()),
        )
        return replace(
            verdict,
            elapsed_ms=max(1, math.ceil((time.monotonic() - started_at) * 1000)),
        )
    deadline = time.monotonic() + timeout / 1000.0
    shape_verdict = check_shape_valid_and_equivalent(
        current,
        candidate,
        timeout=timeout,
        preconditions=preconditions,
        rule_name=rule_name,
    )
    if not shape_verdict.proved:
        return shape_verdict
    value_timeout = _remaining_timeout(deadline, timeout)
    if value_timeout is None:
        return EquivalenceVerdict(
            proved=False, stage="value", detail="candidate deadline exhausted"
        )
    proved_current, proved_candidate = _with_proved_shape(
        current,
        candidate,
        (
            _expr_structural_key(current.expr),
            _expr_structural_key(candidate.expr),
            tuple(p.constraint.sexpr() for p in (preconditions or [])),
        ),
    )
    return _check_value_equivalent_after_shape(
        proved_current,
        proved_candidate,
        timeout=value_timeout,
        preconditions=preconditions,
        rule_name=rule_name,
    )


def _default_out_shape(dst: Any, *srcs: Any) -> tuple[Any, ...]:
    if isinstance(dst, SymTensor):
        return dst.shape
    for s in srcs:
        if isinstance(s, SymTensor):
            return s.shape
    return (z3.IntVal(1),)


def _as_scalar(v: Any) -> z3.ArithRef:
    if isinstance(v, (int, float)):
        return z3.RealVal(v)
    if isinstance(v, z3.ArithRef):
        return v
    return z3.RealVal(0)


def _index_vars(name: str, rank: int, prefix: str = "i") -> list[z3.ArithRef]:
    return [z3.Int(f"{name}_{prefix}{k}") for k in range(rank)]


def _shape_ctx(*dims: z3.ArithRef) -> Context:
    return Context([d > 0 for d in dims])


def _shape_product(dims: list[z3.ArithRef]) -> z3.ArithRef:
    out: z3.ArithRef = z3.IntVal(1)
    for d in dims:
        out = out * d
    return out


def _int_floor_div(value: z3.ArithRef, divisor: int) -> z3.ArithRef:
    return value / z3.IntVal(divisor)


def sym_ceil_div(value: Any, divisor: Any) -> Any:
    if isinstance(value, int) and isinstance(divisor, int) and divisor > 0:
        return math.ceil(value / divisor)
    v = _to_dim(value)
    d = _to_dim(divisor)
    return (v + d - z3.IntVal(1)) / d


def _linear_index(dims: list[z3.ArithRef], indices: list[z3.ArithRef]) -> z3.ArithRef:
    out: z3.ArithRef = z3.IntVal(0)
    for d, idx in zip(dims, indices, strict=True):
        out = out * d + idx
    return out


def _shape_from_out(out_shape: tuple[Any, ...]) -> ShapeResult:
    dims = [_to_dim(d) for d in out_shape]
    return ShapeResult(ShapeExpr(dims), _shape_ctx(*dims))


def _permute_dims(dims: list[z3.ArithRef], axes: list[int]) -> list[z3.ArithRef]:
    return [dims[a] for a in axes]


def _flattened_free_index(shape: ShapeExpr, indices: list[z3.ArithRef]) -> z3.ArithRef:
    if shape.rank <= 1:
        return z3.IntVal(0)
    return _linear_index(list(shape.dims[1:]), list(indices[1:]))


def _opaque_pointwise_semantics(
    expr: SymExpr,
    ins: list[Semantics],
    out_shape: ShapeExpr,
    arg_exprs: list[z3.ArithRef],
    *,
    extra_ctx: Context | None = None,
    reduction: ReductionDesc | None = None,
) -> Semantics:
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = _index_vars(expr.name, out_shape.rank)
    ctx = Context()
    for sem in ins:
        ctx = ctx.merged(sem.ctx)
    if extra_ctx is not None:
        ctx = ctx.merged(extra_ctx)
    opaque = z3.Function(
        f"OPAQUE_{expr.name}",
        *([z3.RealSort()] * len(arg_exprs)),
        *([z3.IntSort()] * builtins.max(1, out_shape.rank)),
        z3.RealSort(),
    )
    opaque_args = [*arg_exprs, *(idx if idx else [z3.IntVal(0)])]
    ctx.add(z3.ForAll(idx, out_fn(*idx) == opaque(*opaque_args)))
    return Semantics(expr.name, out_shape, out_fn, ctx, reduction=reduction)


def _pattern_counts(pattern: Any) -> list[int]:
    counts: list[int] = []
    if pattern is None:
        return counts
    for pos, entry in enumerate(pattern):
        if not (isinstance(entry, (list, tuple)) and len(entry) >= 2):
            raise ValueError(
                f"pattern entries must be [step, count] pairs, got {entry!r} at position {pos} in {pattern!r}"
            )
        counts.append(int(entry[1]))
    return counts


class dge_mode(Enum):
    unknown = 0
    """Unknown DGE mode, i.e., let compiler decide the DGE mode"""
    swdge = 1
    """Software DGE"""
    hwdge = 2
    """Hardware DGE"""
    none = 3
    """Not using DGE"""


class dma_engine(Enum):
    dma = 1
    """Shared DMA with CoreBarrier synchronization (default). Can be triggered from any engine."""
    gpsimd_dma = 2
    """GPSIMD's internal DMA engine for low-latency SB-to-SB swaps in LNC=2.
        Implies GPSIMD as the trigger engine."""


class engine(Enum):
    tensor = 1
    """Tensor Engine"""
    vector = 5
    """Vector Engine"""
    scalar = 2
    """Scalar Engine"""
    gpsimd = 3
    """GpSIMD Engine"""
    dma = 4
    """DMA Engine"""
    sync = 6
    """Sync Engine"""
    unknown = 0
    """Unknown Engine"""


class matmul_perf_mode(Enum):
    none = "none"
    """Default mode, no performance optimization"""
    double_row = "double_row"
    """Double FP8 mode, 2x matmul throughput by packing two FP8 weight/ifmap element pairs"""


class oob_mode(Enum):
    error = 0
    """Raise a runtime error when an out-of-bounds access is detected."""
    skip = 1
    """Silently skip the runtime out-of-bounds access."""


class reduce_cmd(Enum):
    idle = 0
    """Not using the accumulator registers"""
    reset = 1
    """Resets the accumulator registers to its initial state"""
    reduce = 2
    """Keeps accumulating over the current value of the accumulator registers"""
    reset_reduce = 3
    """Resets the accumulator registers then immediately accumulate the results of the current instruction into the accumulators"""
    load_reduce = 4
    """Loads a value into the accumulator registers, then accumulate the results of the current instruction into the accumulators"""


class tile_size:
    bn_stats_fmax = ...
    """Maximum free dimension of BN_STATS"""
    gemm_moving_fmax = ...
    """Maximum free dimension of the moving operand of General Matrix Multiplication on Tensor Engine"""
    gemm_stationary_fmax = ...
    """Maximum free dimension of the stationary operand of General Matrix Multiplication on Tensor Engine"""
    pmax = ...
    """Maximum partition dimension of a tile"""
    psum_fmax = ...
    """Maximum free dimension of a tile on PSUM buffer"""
    psum_min_align = ...
    """Minimum byte alignment requirement for PSUM free dimension address"""
    sbuf_min_align = ...
    """Minimum byte alignment requirement for SBUF free dimension address"""
    total_available_sbuf_size = ...
    """Usable SBUF size per partition (total minus reserved bytes)."""


@semantics_hw()
def activation(
    dst,
    op,
    data,
    bias=None,
    scale=1.0,
    reduce_op=None,
    reduce_res=None,
    reduce_cmd=reduce_cmd.idle,
    name=None,
):
    assert _is_sym_tensor(data)
    inputs = [data]
    attrs: dict[str, Any] = {
        "op": op,
        "scale": scale,
        "reduce_op": reduce_op,
        "reduce_cmd": reduce_cmd,
        "name": name,
        "with_reduce": reduce_res is not None or reduce_op is not None,
    }
    if _is_sym_tensor(bias):
        attrs["bias_input_index"] = len(inputs)
        inputs.append(bias)
    else:
        attrs["bias_const"] = bias
    if _is_sym_tensor(scale):
        attrs["scale_input_index"] = len(inputs)
        inputs.append(scale)
    return _new_sym_tensor("activation", inputs, attrs, _default_out_shape(dst, data))


@semantics_hw()
def activation_reduce(
    dst, op, data, reduce_op, reduce_res, bias=None, scale=1.0, name=None
):
    assert _is_sym_tensor(data)
    inputs = [data]
    attrs: dict[str, Any] = {
        "op": op,
        "reduce_op": reduce_op,
        "scale": scale,
        "name": name,
    }
    if _is_sym_tensor(bias):
        attrs["bias_input_index"] = len(inputs)
        inputs.append(bias)
    else:
        attrs["bias_const"] = bias
    if _is_sym_tensor(scale):
        attrs["scale_input_index"] = len(inputs)
        inputs.append(scale)
    # Use the reduce result shape for the e-class value.
    out_shape = tuple(
        _shape_activation_reduce([ShapeExpr(list(data.shape))], attrs).out.dims
    )
    return _new_sym_tensor("activation_reduce", inputs, attrs, out_shape)


@semantics_hw()
def affine_select(
    dst,
    pattern,
    channel_multiplier,
    on_true_tile,
    on_false_value,
    cmp_op=nl.equal,
    offset=0,
    name=None,
):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        ctx = _shape_ctx(*data_shape.dims)
        pattern_counts = _pattern_counts(attrs.get("pattern", []))
        if data_shape.rank > 1 and pattern_counts:
            ctx.add(
                _shape_product(list(data_shape.dims[1:]))
                == _shape_product([z3.IntVal(n) for n in pattern_counts])
            )
        return ShapeResult(ShapeExpr(list(data_shape.dims)), ctx)

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        on_true_sem = ins[0]
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        idx = _index_vars(expr.name, out_shape.rank)
        ctx = on_true_sem.ctx.merged()
        flat_idx = _flattened_free_index(out_shape, idx)
        affine_val = z3.ToReal(z3.IntVal(int(expr.attrs.get("offset", 0)))) + z3.ToReal(
            idx[0] * z3.IntVal(int(expr.attrs.get("channel_multiplier", 0))) + flat_idx
        )
        pred = _compare_bool(expr.attrs.get("cmp_op"), affine_val, z3.RealVal(0))
        true_val = _call_broadcasted(on_true_sem, out_shape, idx)
        false_val = _as_scalar(expr.attrs.get("on_false_value"))
        ctx.add(z3.ForAll(idx, out_fn(*idx) == z3.If(pred, true_val, false_val)))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("affine_select", shape_rule, value_rule)

    assert _is_sym_tensor(on_true_tile)
    return _new_sym_tensor(
        "affine_select",
        [on_true_tile],
        {
            "pattern": pattern,
            "channel_multiplier": channel_multiplier,
            "on_false_value": on_false_value,
            "cmp_op": cmp_op,
            "offset": offset,
            "name": name,
        },
        _default_out_shape(dst, on_true_tile),
    )


@semantics_hw()
def bn_aggr(dst, data, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        dims = [data_shape.dims[0], z3.IntVal(2)]
        ctx = _shape_ctx(*data_shape.dims)
        if data_shape.rank >= 2:
            ctx.add(data_shape.dims[1] % 3 == 0)
        return ShapeResult(ShapeExpr(dims), ctx)

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        return _opaque_pointwise_semantics(expr, ins, out_shape, [])

    _ensure_semantics("bn_aggr", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    out_shape = _default_out_shape(dst, data)
    if out_shape == data.shape:
        out_shape = (data.shape[0], z3.IntVal(2))
    return _new_sym_tensor("bn_aggr", [data], {"name": name}, out_shape)


@semantics_hw()
def bn_stats(dst, data, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        return ShapeResult(
            ShapeExpr([data_shape.dims[0], z3.IntVal(6)]), _shape_ctx(*data_shape.dims)
        )

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        return _opaque_pointwise_semantics(expr, ins, out_shape, [])

    _ensure_semantics("bn_stats", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    out_shape = _default_out_shape(dst, data)
    if out_shape == data.shape:
        out_shape = (data.shape[0], z3.IntVal(6))
    return _new_sym_tensor("bn_stats", [data], {"name": name}, out_shape)


@semantics_hw()
def dma_compute(dst, srcs, reduce_op, scales=None, unique_indices=True, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        out_shape = attrs.get("out_shape")
        if out_shape is not None:
            return _shape_from_out(out_shape)
        if ins:
            first = ins[0]
            return ShapeResult(ShapeExpr(list(first.dims)), _shape_ctx(*first.dims))
        return _shape_from_out((z3.IntVal(1),))

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        idx = _index_vars(expr.name, out_shape.rank)
        ctx = Context()
        for sem in ins:
            ctx = ctx.merged(sem.ctx)
        values = [_call_broadcasted(sem, out_shape, idx) for sem in ins]
        tmp = z3.RealVal(0)
        for value, scale in zip(values, expr.attrs.get("scales", []), strict=False):
            tmp = _apply_binary(
                expr.attrs.get("reduce_op"), tmp, value * _as_scalar(scale)
            )
        ctx.add(z3.ForAll(idx, out_fn(*idx) == tmp))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("dma_compute", shape_rule, value_rule)

    sym_srcs = [src for src in srcs if _is_sym_tensor(src)]
    assert sym_srcs
    resolved_scales = list(scales) if scales is not None else [1.0] * len(srcs)
    return _new_sym_tensor(
        "dma_compute",
        sym_srcs,
        {
            "reduce_op": reduce_op,
            "scales": resolved_scales,
            "unique_indices": unique_indices,
            "out_shape": _default_out_shape(dst, *sym_srcs),
            "name": name,
        },
        _default_out_shape(dst, *sym_srcs),
    )


@semantics_hw()
def dma_copy(
    dst,
    src,
    oob_mode=oob_mode.error,
    dge_mode=dge_mode.unknown,
    engine=engine.unknown,
    name=None,
):
    assert _is_sym_tensor(src)
    return _new_sym_tensor(
        "dma_copy",
        [src],
        {"oob_mode": oob_mode, "dge_mode": dge_mode, "engine": engine, "name": name},
        _default_out_shape(dst, src),
    )


_DMA_TRANSPOSE_DEFAULT_AXES: dict[int, tuple[int, ...]] = {
    2: (1, 0),
    3: (2, 1, 0),
    4: (3, 1, 2, 0),
}


def _dma_transpose_axes(axes_attr: Any, rank: int) -> tuple[int, ...] | None:
    """The permutation a ``dma_transpose`` applies, or None when it is invalid.

    TensorRight's ``transpose`` asserts that its map is a permutation before it
    relabels anything. Mirror that: an ``axes`` that is not a permutation of
    ``range(rank)`` used to reach ``_permute_dims`` and either crash or read the
    wrong axis."""
    if axes_attr is None:
        return _DMA_TRANSPOSE_DEFAULT_AXES.get(rank)
    try:
        axes = tuple(int(a) for a in axes_attr)
    except (TypeError, ValueError):
        return None
    if len(axes) != rank or sorted(axes) != list(range(rank)):
        return None
    return axes


def _shape_dma_transpose(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    src_shape = ins[0]
    axes = _dma_transpose_axes(attrs.get("axes"), src_shape.rank)
    if axes is None:
        ctx = _shape_ctx(*src_shape.dims)
        ctx.add(z3.BoolVal(False))
        return ShapeResult(ShapeExpr(list(src_shape.dims)), ctx)
    dims = _permute_dims(list(src_shape.dims), list(axes))
    return ShapeResult(ShapeExpr(dims), _shape_ctx(*src_shape.dims, *dims))


def _validity_dma_transpose(ins: list[ShapeExpr], attrs: dict[str, Any]) -> Context:
    ctx = Context()
    if not ins or _dma_transpose_axes(attrs.get("axes"), ins[0].rank) is None:
        ctx.add(z3.BoolVal(False))
    return ctx


def _compile_dma_transpose(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    src_sem = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = src_sem.ctx.merged()
    axes = _dma_transpose_axes(expr.attrs.get("axes"), src_sem.shape.rank)
    if axes is None or len(axes) != out_shape.rank:
        # Invalid permutation: validity is unsatisfiable, so leave it opaque.
        return Semantics(expr.name, out_shape, out_fn, ctx)
    idx = _index_vars(expr.name, out_shape.rank)
    src_idx = [idx[axes.index(axis)] for axis in range(len(axes))]
    ctx.add(z3.ForAll(idx, out_fn(*idx) == src_sem.fn(*src_idx)))
    reduction = None
    if axes == (1, 0):
        reduction = _lift_fold_through_transpose(
            expr.name, src_sem.reduction, out_fn, out_shape, ctx
        )
    return Semantics(expr.name, out_shape, out_fn, ctx, reduction=reduction)


def _builder_out_shape(dst: Any, src: SymTensor, permuted: tuple[Any, ...]) -> Any:
    """Output shape for a shape-changing builder.

    A ``dst`` tile already has the output's shape, so it is used as is. The
    transpose builders used to permute ``dst.shape`` a second time, which gave
    the untransposed shape whenever a caller passed a real destination."""
    if isinstance(dst, SymTensor):
        return dst.shape
    del src
    return permuted


@semantics_hw()
def dma_transpose(
    dst, src, axes=None, dge_mode=dge_mode.unknown, oob_mode=oob_mode.error, name=None
):
    assert _is_sym_tensor(src)
    resolved_axes = tuple(axes) if axes is not None else None
    perm = _dma_transpose_axes(resolved_axes, len(src.shape))
    permuted = (
        tuple(src.shape[a] for a in perm) if perm is not None else tuple(src.shape)
    )
    return _new_sym_tensor(
        "dma_transpose",
        [src],
        {
            "axes": resolved_axes,
            "dge_mode": dge_mode,
            "oob_mode": oob_mode,
            "name": name,
        },
        _builder_out_shape(dst, src, permuted),
    )


@semantics_hw()
def dropout(dst, data, prob, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        ctx = _shape_ctx(*data_shape.dims)
        if len(ins) > 1:
            ctx.extend(_broadcast_shape(data_shape, ins[1]).ctx.facts)
        return ShapeResult(ShapeExpr(list(data_shape.dims)), ctx)

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        data_sem = ins[0]
        idx = _index_vars(expr.name, out_shape.rank)
        prob_value = _operand_value(
            expr.attrs, "prob_const", "prob_input_index", ins, out_shape, idx
        )
        data_value = _call_broadcasted(data_sem, out_shape, idx)
        drop_mask = z3.Function(
            f"DROP_MASK_{expr.name}",
            z3.RealSort(),
            *([z3.IntSort()] * builtins.max(1, out_shape.rank)),
            z3.BoolSort(),
        )
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        ctx = data_sem.ctx.merged(*[s.ctx for s in ins[1:]])
        mask_args = [prob_value, *(idx if idx else [z3.IntVal(0)])]
        value = z3.If(
            prob_value <= 0,
            data_value,
            z3.If(
                prob_value >= 1,
                z3.RealVal(0),
                z3.If(drop_mask(*mask_args), z3.RealVal(0), data_value),
            ),
        )
        ctx.add(z3.ForAll(idx, out_fn(*idx) == value))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("dropout", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    inputs = [data]
    attrs: dict[str, Any] = {"name": name}
    if _is_sym_tensor(prob):
        attrs["prob_input_index"] = len(inputs)
        inputs.append(prob)
    else:
        attrs["prob_const"] = prob
    return _new_sym_tensor("dropout", inputs, attrs, _default_out_shape(dst, data))


@semantics_hw()
def exponential(
    dst,
    src,
    max_value=0.0,
    reduce_res=None,
    reduce_cmd=reduce_cmd.idle,
    reduce_init=0.0,
    name=None,
):
    assert _is_sym_tensor(src)
    inputs = [src]
    attrs: dict[str, Any] = {
        "max_value": max_value,
        "reduce_cmd": reduce_cmd,
        "reduce_init": reduce_init,
        "with_reduce": reduce_res is not None,
        "name": name,
    }
    if _is_sym_tensor(max_value):
        attrs["max_input_index"] = len(inputs)
        inputs.append(max_value)
    return _new_sym_tensor("exponential", inputs, attrs, _default_out_shape(dst, src))


@semantics_hw()
def iota(dst, pattern, offset=0, channel_multiplier=0, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        out_shape = attrs.get("out_shape", ())
        dims = [_to_dim(d) for d in out_shape]
        return ShapeResult(ShapeExpr(dims), Context([d > 0 for d in dims]))

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        ctx = Context()
        if out_shape.rank != 2:
            return Semantics(expr.name, out_shape, out_fn, ctx)
        i = z3.Int(f"{expr.name}_i")
        j = z3.Int(f"{expr.name}_j")
        offset = int(expr.attrs.get("offset", 0))
        channel_multiplier = int(expr.attrs.get("channel_multiplier", 0))
        step = _extract_iota_step(expr.attrs.get("pattern", []))
        val = z3.ToReal(offset + i * channel_multiplier + j * int(step))
        ctx.add(z3.ForAll([i, j], out_fn(i, j) == val))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    def validity_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> Context:
        ctx = Context()
        if len(attrs.get("out_shape", ())) != 2:
            ctx.add(z3.BoolVal(False))
        return ctx

    _ensure_semantics(
        "iota",
        shape_rule,
        value_rule,
        validity_rule,
    )

    out_shape = _default_out_shape(dst)
    assert out_shape
    return _new_sym_tensor(
        "iota",
        [],
        {
            "pattern": pattern,
            "offset": offset,
            "channel_multiplier": channel_multiplier,
            "out_shape": out_shape,
            "name": name,
        },
        out_shape,
    )


@semantics_hw()
def local_gather(
    dst, src_buffer, index, num_elem_per_idx=1, num_valid_indices=None, name=None
):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_from_out(attrs.get("out_shape", tuple(ins[0].dims)))

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        return _opaque_pointwise_semantics(expr, ins, out_shape, [])

    _ensure_semantics("local_gather", shape_rule, value_rule)

    assert _is_sym_tensor(src_buffer) and _is_sym_tensor(index)
    return _new_sym_tensor(
        "local_gather",
        [src_buffer, index],
        {
            "num_elem_per_idx": num_elem_per_idx,
            "num_valid_indices": num_valid_indices,
            "out_shape": _default_out_shape(dst, src_buffer),
            "name": name,
        },
        _default_out_shape(dst, src_buffer),
    )


@semantics_hw()
def max8(dst, src, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        src_shape = ins[0]
        return ShapeResult(
            ShapeExpr([src_shape.dims[0], z3.IntVal(8)]), _shape_ctx(*src_shape.dims)
        )

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        return _opaque_pointwise_semantics(expr, ins, out_shape, [])

    _ensure_semantics("max8", shape_rule, value_rule)

    assert _is_sym_tensor(src)
    out_shape = _default_out_shape(dst, src)
    if out_shape == src.shape:
        out_shape = (src.shape[0], z3.IntVal(8))
    return _new_sym_tensor("max8", [src], {"name": name}, out_shape)


@semantics_hw()
def memset(dst, value, engine=engine.unknown, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_from_out(attrs.get("out_shape", (z3.IntVal(1),)))

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        idx = _index_vars(expr.name, out_shape.rank)
        ctx = Context()
        ctx.add(z3.ForAll(idx, out_fn(*idx) == _as_scalar(expr.attrs.get("value"))))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("memset", shape_rule, value_rule)

    out_shape = _default_out_shape(dst)
    assert out_shape
    return _new_sym_tensor(
        "memset",
        [],
        {"value": value, "engine": engine, "out_shape": out_shape, "name": name},
        out_shape,
    )


@semantics_hw()
def nc_find_index8(dst, data, vals, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        ctx = _shape_ctx(*data_shape.dims, *ins[1].dims)
        return ShapeResult(ShapeExpr([data_shape.dims[0], z3.IntVal(8)]), ctx)

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        return _opaque_pointwise_semantics(expr, ins, out_shape, [])

    _ensure_semantics("nc_find_index8", shape_rule, value_rule)

    assert _is_sym_tensor(data) and _is_sym_tensor(vals)
    out_shape = _default_out_shape(dst, data)
    if out_shape == data.shape:
        out_shape = (data.shape[0], z3.IntVal(8))
    return _new_sym_tensor("nc_find_index8", [data, vals], {"name": name}, out_shape)


@semantics_hw()
def nc_match_replace8(dst, data, vals, imm, dst_idx=None, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        ctx = _shape_ctx(*data_shape.dims, *ins[1].dims)
        return ShapeResult(ShapeExpr(list(data_shape.dims)), ctx)

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        return _opaque_pointwise_semantics(
            expr,
            ins,
            out_shape,
            [_as_scalar(expr.attrs.get("imm"))],
        )

    _ensure_semantics("nc_match_replace8", shape_rule, value_rule)

    assert _is_sym_tensor(data) and _is_sym_tensor(vals)
    return _new_sym_tensor(
        "nc_match_replace8",
        [data, vals],
        {"imm": imm, "name": name},
        _default_out_shape(dst, data),
    )


def _shape_nc_matmul(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    lhs, rhs = ins
    out_shape = attrs.get("out_shape")
    if out_shape is not None:
        return _shape_from_out(out_shape)
    if lhs.rank >= 2 and rhs.rank >= 2:
        dims = [lhs.dims[-1], rhs.dims[-1]]
        ctx = _shape_ctx(*lhs.dims, *rhs.dims, *dims)
        ctx.add(lhs.dims[0] == rhs.dims[0])
        return ShapeResult(ShapeExpr(dims), ctx)
    return ShapeResult(
        ShapeExpr([lhs.dims[0], rhs.dims[-1] if rhs.rank else z3.IntVal(1)]),
        _shape_ctx(*lhs.dims, *rhs.dims),
    )


def _compile_nc_matmul(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    stationary_sem, moving_sem = ins
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = stationary_sem.ctx.merged(moving_sem.ctx)
    if (
        stationary_sem.shape.rank == 2
        and moving_sem.shape.rank == 2
        and out_shape.rank == 2
        # ``is_transpose`` switches the Tensor Engine into its transpose mode,
        # which is not the sum of products below. Leave it opaque rather than
        # let it prove equal to an ordinary matmul. The one-zero and perf-mode
        # flags are hints that do not change the value.
        and expr.attrs.get("is_transpose", False) is not True
    ):
        m = z3.Int(f"{expr.name}_m")
        n = z3.Int(f"{expr.name}_n")
        k = z3.Int(f"{expr.name}_k")
        body_id = _BODY_IDS.next()
        fam = REDUCE2_FAM  # nc_matmul is intrinsically sum-of-products
        ctx.add(
            z3.ForAll(
                [m, n, k],
                fam.step(z3.IntVal(body_id), m, n, k)
                == stationary_sem.fn(k, m) * moving_sem.fn(k, n),
            )
        )
        extent = stationary_sem.shape.dims[0]
        val = fam.fold(z3.IntVal(body_id), m, n, extent)
        if expr.attrs.get("accumulate") is True:
            acc = z3.Function(
                f"ACC_{expr.name}", z3.IntSort(), z3.IntSort(), z3.RealSort()
            )
            val = acc(m, n) + val
        ctx.add(z3.ForAll([m, n], out_fn(m, n) == val))
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
                output_transform=(
                    "accumulate" if expr.attrs.get("accumulate") is True else "identity"
                ),
            ),
        )
    return _opaque_pointwise_semantics(expr, ins, out_shape, [], extra_ctx=ctx)


def _validity_nc_matmul(ins: list[ShapeExpr], attrs: dict[str, Any]) -> Context:
    ctx = Context()
    if len(ins) != 2 or ins[0].rank != 2 or ins[1].rank != 2:
        ctx.add(z3.BoolVal(False))
        return ctx
    # State operand constraints that ``out_shape`` bypasses in the shape rule.
    stationary, moving = ins
    ctx.add(stationary.dims[0] == moving.dims[0])
    out_shape = attrs.get("out_shape")
    if out_shape is not None and len(out_shape) == 2:
        ctx.add(_to_dim(out_shape[0]) == stationary.dims[1])
        ctx.add(_to_dim(out_shape[1]) == moving.dims[1])
    return ctx


@semantics_hw()
def nc_matmul(
    dst,
    stationary,
    moving,
    is_stationary_onezero=False,
    is_moving_onezero=False,
    is_transpose=False,
    accumulate=None,
    tile_position=(),
    tile_size=(),
    perf_mode=matmul_perf_mode.none,
    name=None,
):
    assert _is_sym_tensor(stationary) and _is_sym_tensor(moving)
    out_shape = _default_out_shape(dst, stationary)
    if out_shape == stationary.shape and stationary.rank >= 2 and moving.rank >= 2:
        out_shape = (stationary.shape[-1], moving.shape[-1])
    return _new_sym_tensor(
        "nc_matmul",
        [stationary, moving],
        {
            "is_stationary_onezero": is_stationary_onezero,
            "is_moving_onezero": is_moving_onezero,
            "is_transpose": is_transpose,
            "accumulate": accumulate,
            "tile_position": tile_position,
            "tile_size": tile_size,
            "perf_mode": perf_mode,
            "out_shape": out_shape,
            "name": name,
        },
        out_shape,
    )


@semantics_hw()
def nc_n_gather(dst, data, indices, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape, idx_shape = ins
        dims = [data_shape.dims[0], *idx_shape.dims[1:]]
        return ShapeResult(
            ShapeExpr(dims), _shape_ctx(*data_shape.dims, *idx_shape.dims)
        )

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        data_sem, idx_sem = ins
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        idx = _index_vars(expr.name, out_shape.rank)
        ctx = data_sem.ctx.merged(idx_sem.ctx)
        flat = (
            z3.ToInt(idx_sem.fn(idx[0], *idx[1:]))
            if out_shape.rank > 1
            else z3.ToInt(idx_sem.fn(idx[0]))
        )
        free_extent = (
            _shape_product(list(data_sem.shape.dims[1:]))
            if data_sem.shape.rank > 1
            else z3.IntVal(1)
        )
        src_free = flat % free_extent
        if data_sem.shape.rank == 2:
            ctx.add(z3.ForAll(idx, out_fn(*idx) == data_sem.fn(idx[0], src_free)))
        else:
            opaque_src = z3.Function(
                f"GATHER_{expr.name}", z3.IntSort(), z3.IntSort(), z3.RealSort()
            )
            ctx.add(z3.ForAll(idx, out_fn(*idx) == opaque_src(idx[0], src_free)))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("nc_n_gather", shape_rule, value_rule)

    assert _is_sym_tensor(data) and _is_sym_tensor(indices)
    out_shape = _default_out_shape(dst, indices)
    if out_shape == indices.shape:
        out_shape = (data.shape[0], *indices.shape[1:])
    return _new_sym_tensor("nc_n_gather", [data, indices], {"name": name}, out_shape)


@semantics_hw()
def nc_stream_shuffle(dst, src, shuffle_mask, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        src_shape = ins[0]
        return ShapeResult(ShapeExpr(list(src_shape.dims)), _shape_ctx(*src_shape.dims))

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        src_sem = ins[0]
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        idx = _index_vars(expr.name, out_shape.rank)
        ctx = src_sem.ctx.merged()
        mask = expr.attrs.get("shuffle_mask", [])
        if len(mask) != 32 or out_shape.rank < 1:
            return _opaque_pointwise_semantics(expr, ins, out_shape, [], extra_ctx=ctx)
        part = idx[0]
        quadrant_base = part - (part % z3.IntVal(32))
        source_part = z3.IntVal(0)
        value = z3.RealVal(0)
        for pos, source in enumerate(mask):
            src_pos = (
                part if int(source) == 255 else quadrant_base + z3.IntVal(int(source))
            )
            source_part = z3.If((part % z3.IntVal(32)) == pos, src_pos, source_part)
        src_idx = [source_part, *idx[1:]]
        value = src_sem.fn(*src_idx) if out_shape.rank > 1 else src_sem.fn(source_part)
        ctx.add(z3.ForAll(idx, out_fn(*idx) == value))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("nc_stream_shuffle", shape_rule, value_rule)

    assert _is_sym_tensor(src)
    return _new_sym_tensor(
        "nc_stream_shuffle",
        [src],
        {"shuffle_mask": list(shuffle_mask), "name": name},
        _default_out_shape(dst, src),
    )


@semantics_hw()
def nc_transpose(dst, data, engine=engine.unknown, name=None):
    assert _is_sym_tensor(data)
    permuted = tuple(data.shape)
    if len(permuted) >= 2:
        permuted = (permuted[1], permuted[0], *permuted[2:])
    return _new_sym_tensor(
        "nc_transpose",
        [data],
        {"engine": engine, "name": name},
        _builder_out_shape(dst, data, permuted),
    )


@semantics_hw()
def reciprocal(dst, data, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_unary_same(ins, attrs)

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        return _compile_reciprocal(expr, ins, out_shape)

    _ensure_semantics("reciprocal", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    return _new_sym_tensor(
        "reciprocal", [data], {"name": name}, _default_out_shape(dst, data)
    )


def _shape_scalar_tensor_tensor(
    ins: list[ShapeExpr], attrs: dict[str, Any]
) -> ShapeResult:
    data_shape = ins[0]
    ctx = _shape_ctx(*data_shape.dims)
    if len(ins) > 1:
        ctx.extend(_broadcast_shape(data_shape, ins[1]).ctx.facts)
    if len(ins) > 2:
        ctx.extend(_broadcast_shape(data_shape, ins[2]).ctx.facts)
    return ShapeResult(ShapeExpr(list(data_shape.dims)), ctx)


def _compile_scalar_tensor_tensor(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    data_sem = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = _index_vars(expr.name, out_shape.rank)
    ctx = data_sem.ctx.merged(*[s.ctx for s in ins[1:]])
    data_value = _call_broadcasted(data_sem, out_shape, idx)
    op0_value = _operand_value(
        expr.attrs, "operand0_const", "operand0_input_index", ins, out_shape, idx
    )
    lhs0, rhs0 = (
        (op0_value, data_value)
        if expr.attrs.get("reverse0", False)
        else (data_value, op0_value)
    )
    tmp = _apply_binary(expr.attrs.get("op0"), lhs0, rhs0)
    op1_value = _call_broadcasted(
        ins[int(expr.attrs["operand1_input_index"])], out_shape, idx
    )
    lhs1, rhs1 = (
        (op1_value, tmp) if expr.attrs.get("reverse1", False) else (tmp, op1_value)
    )
    ctx.add(
        z3.ForAll(idx, out_fn(*idx) == _apply_binary(expr.attrs.get("op1"), lhs1, rhs1))
    )
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _validity_scalar_tensor_tensor(
    ins: list[ShapeExpr], attrs: dict[str, Any]
) -> Context:
    # A tensor ``operand0`` must have a free dimension of one.
    ctx = Context()
    if "operand1_input_index" not in attrs:
        ctx.add(z3.BoolVal(False))
        return ctx
    operand0_index = attrs.get("operand0_input_index")
    if operand0_index is not None:
        operand0_index = int(operand0_index)
        if 0 <= operand0_index < len(ins):
            operand0_shape = ins[operand0_index]
            if operand0_shape.dims:
                ctx.add(operand0_shape.dims[-1] == z3.IntVal(1))
        else:
            ctx.add(z3.BoolVal(False))
    return ctx


@semantics_hw()
def scalar_tensor_tensor(
    dst, data, op0, operand0, op1, operand1, reverse0=False, reverse1=False, name=None
):
    assert _is_sym_tensor(data) and _is_sym_tensor(operand1)
    inputs = [data]
    attrs: dict[str, Any] = {
        "op0": op0,
        "op1": op1,
        "reverse0": reverse0,
        "reverse1": reverse1,
        "name": name,
    }
    if _is_sym_tensor(operand0):
        attrs["operand0_input_index"] = len(inputs)
        inputs.append(operand0)
    else:
        attrs["operand0_const"] = operand0
    attrs["operand1_input_index"] = len(inputs)
    inputs.append(operand1)
    return _new_sym_tensor(
        "scalar_tensor_tensor", inputs, attrs, _default_out_shape(dst, data)
    )


@semantics_hw()
def select_reduce(
    dst,
    predicate,
    on_true,
    on_false,
    reduce_res=None,
    reduce_cmd=reduce_cmd.idle,
    reduce_op=nl.maximum,
    reverse_pred=False,
    name=None,
):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_tensor_scalar(ins, attrs)

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        return _compile_select_reduce(expr, ins, out_shape)

    _ensure_semantics("select_reduce", shape_rule, value_rule)

    assert _is_sym_tensor(predicate) and _is_sym_tensor(on_true)
    inputs = [predicate, on_true]
    attrs: dict[str, Any] = {
        "reduce_cmd": reduce_cmd,
        "reduce_op": reduce_op,
        "reverse_pred": reverse_pred,
        "with_reduce": reduce_res is not None,
        "name": name,
    }
    if _is_sym_tensor(on_false):
        attrs["on_false_input_index"] = len(inputs)
        inputs.append(on_false)
    else:
        attrs["on_false_const"] = on_false
    return _new_sym_tensor(
        "select_reduce", inputs, attrs, _default_out_shape(dst, on_true)
    )


@semantics_hw()
def tensor_copy(dst, src, engine=engine.unknown, name=None):
    assert _is_sym_tensor(src)
    return _new_sym_tensor(
        "tensor_copy",
        [src],
        {"engine": engine, "name": name},
        _default_out_shape(dst, src),
    )


@semantics_hw()
def tensor_copy_predicated(dst, src, predicate, reverse_pred=False, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        src_shape = ins[0]
        pred_shape = ins[1]
        ctx = _shape_ctx(*src_shape.dims)
        ctx.extend(_broadcast_shape(src_shape, pred_shape).ctx.facts)
        if len(ins) > 2:
            ctx.extend(_broadcast_shape(src_shape, ins[2]).ctx.facts)
        return ShapeResult(ShapeExpr(list(src_shape.dims)), ctx)

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        src_sem, pred_sem = ins[0], ins[1]
        prior_dst = ins[2] if len(ins) > 2 else None
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        idx = _index_vars(expr.name, out_shape.rank)
        ctx = src_sem.ctx.merged(pred_sem.ctx, *(s.ctx for s in ins[2:]))
        pred = _call_broadcasted(pred_sem, out_shape, idx) != 0
        pred = z3.Not(pred) if expr.attrs.get("reverse_pred", False) else pred
        src_value = _call_broadcasted(src_sem, out_shape, idx)
        old_value = (
            _call_broadcasted(prior_dst, out_shape, idx)
            if prior_dst is not None
            else z3.RealVal(0)
        )
        ctx.add(z3.ForAll(idx, out_fn(*idx) == z3.If(pred, src_value, old_value)))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("tensor_copy_predicated", shape_rule, value_rule)

    assert _is_sym_tensor(src) and _is_sym_tensor(predicate)
    inputs = [src, predicate]
    if _is_sym_tensor(dst):
        inputs.append(dst)
    return _new_sym_tensor(
        "tensor_copy_predicated",
        inputs,
        {"reverse_pred": reverse_pred, "name": name},
        _default_out_shape(dst, src),
    )


def _shape_tensor_partition_reduce(
    ins: list[ShapeExpr], attrs: dict[str, Any]
) -> ShapeResult:
    data_shape = ins[0]
    if data_shape.rank == 0:
        return ShapeResult(ShapeExpr([]), Context([z3.BoolVal(False)]))
    out_dims = (
        [z3.IntVal(1), *data_shape.dims[1:]] if data_shape.rank > 1 else [z3.IntVal(1)]
    )
    return ShapeResult(ShapeExpr(out_dims), _shape_ctx(*data_shape.dims, *out_dims))


def _compile_tensor_partition_reduce(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    data_sem = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = data_sem.ctx.merged()
    combine_op = _normalize_combine_op(expr.attrs.get("op"))
    fam = fold_family(1, ReductionKind.REDUCE, combine_op)
    if data_sem.shape.rank == 1:
        n = data_sem.shape.dims[0]
        if fam is None:
            # Combine op has no registered family — leave the output opaque
            # so equivalence-by-body cannot fire (sound but conservative).
            return Semantics(expr.name, out_shape, out_fn, ctx)
        j = z3.Int(f"{expr.name}_j")
        body_id = _BODY_IDS.next()
        ctx.add(
            z3.ForAll(
                [j],
                fam.step(z3.IntVal(body_id), z3.IntVal(0), j) == data_sem.fn(j),
            )
        )
        ctx.add(
            z3.ForAll(
                [j],
                out_fn(j) == fam.fold(z3.IntVal(body_id), z3.IntVal(0), n),
            )
        )
        return Semantics(
            expr.name,
            out_shape,
            out_fn,
            ctx,
            reduction=ReductionDesc(
                body_id,
                n,
                outer_rank=fam.outer_arity,
                kind=fam.kind,
                combine_op=fam.combine_op,
                outer_dims=(z3.IntVal(1),),
                output_transform="identity",
            ),
        )
    if fam is None:
        return Semantics(expr.name, out_shape, out_fn, ctx)
    tail_idx = _index_vars(expr.name, out_shape.rank - 1, "k")
    p = z3.Int(f"{expr.name}_p")
    body_id = _BODY_IDS.next()
    partition_extent = data_sem.shape.dims[0]
    ctx.add(
        z3.ForAll(
            [p, *tail_idx],
            fam.step(
                z3.IntVal(body_id),
                _linear_index(list(data_sem.shape.dims[1:]), tail_idx),
                p,
            )
            == data_sem.fn(p, *tail_idx),
        )
    )
    for_all_idx = [z3.Int(f"{expr.name}_o{k}") for k in range(out_shape.rank)]
    flat_tail = (
        _linear_index(list(out_shape.dims[1:]), for_all_idx[1:])
        if out_shape.rank > 1
        else z3.IntVal(0)
    )
    ctx.add(
        z3.ForAll(
            for_all_idx,
            z3.Implies(
                for_all_idx[0] == 0,
                out_fn(*for_all_idx)
                == fam.fold(z3.IntVal(body_id), flat_tail, partition_extent),
            ),
        )
    )
    return Semantics(
        expr.name,
        out_shape,
        out_fn,
        ctx,
        reduction=ReductionDesc(
            body_id,
            partition_extent,
            outer_rank=builtins.max(1, out_shape.rank - 1),
            kind=fam.kind,
            combine_op=fam.combine_op,
            outer_dims=(
                z3.Product(*out_shape.dims[1:]) if out_shape.rank > 1 else z3.IntVal(1),
            ),
            output_transform="identity",
        ),
    )


@semantics_hw()
def tensor_partition_reduce(dst, op, data, name=None):
    assert _is_sym_tensor(data)
    out_shape = _default_out_shape(dst, data)
    if out_shape == data.shape:
        out_shape = (
            (z3.IntVal(1), *data.shape[1:]) if data.rank > 1 else (z3.IntVal(1),)
        )
    return _new_sym_tensor(
        "tensor_partition_reduce", [data], {"op": op, "name": name}, out_shape
    )


@semantics_hw()
def tensor_reduce(dst, op, data, axis, negate=False, keepdims=False, name=None):
    assert _is_sym_tensor(data)
    out_shape = tuple(
        _shape_tensor_reduce(
            [ShapeExpr(list(data.shape))],
            {"axis": axis, "keepdims": keepdims},
        ).out.dims
    )
    return _new_sym_tensor(
        "tensor_reduce",
        [data],
        {"op": op, "axis": axis, "negate": negate, "keepdims": keepdims, "name": name},
        out_shape,
    )


@semantics_hw()
def tensor_scalar(
    dst,
    data,
    op0,
    operand0,
    reverse0=False,
    op1=None,
    operand1=None,
    reverse1=False,
    engine=engine.unknown,
    name=None,
):
    assert _is_sym_tensor(data)
    inputs = [data]
    attrs: dict[str, Any] = {
        "op0": op0,
        "reverse0": reverse0,
        "op1": op1,
        "reverse1": reverse1,
        "engine": engine,
        "name": name,
    }
    if _is_sym_tensor(operand0):
        attrs["operand0_input_index"] = len(inputs)
        inputs.append(operand0)
    else:
        attrs["operand0_const"] = operand0
    if _is_sym_tensor(operand1):
        attrs["operand1_input_index"] = len(inputs)
        inputs.append(operand1)
    elif operand1 is not None:
        attrs["operand1_const"] = operand1
    return _new_sym_tensor(
        "tensor_scalar",
        inputs,
        attrs,
        _tensor_scalar_out_shape(dst, data, operand0, operand1),
    )


def _shape_tensor_scalar_cumulative(
    ins: list[ShapeExpr], attrs: dict[str, Any]
) -> ShapeResult:
    data_shape = ins[0]
    ctx = _shape_ctx(*data_shape.dims)
    if len(ins) > 1:
        ctx.extend(_broadcast_shape(data_shape, ins[1]).ctx.facts)
    if len(ins) > 2:
        ctx.extend(_broadcast_shape(data_shape, ins[2]).ctx.facts)
    return ShapeResult(ShapeExpr(list(data_shape.dims)), ctx)


def _compile_tensor_scalar_cumulative(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    src_sem = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = src_sem.ctx.merged(*[s.ctx for s in ins[1:]])
    if out_shape.rank != 2:
        return Semantics(expr.name, out_shape, out_fn, ctx)
    i = z3.Int(f"{expr.name}_i")
    j = z3.Int(f"{expr.name}_j")
    k = z3.Int(f"{expr.name}_k")

    def _step0_at(jj: z3.ArithRef) -> z3.ArithRef:
        imm0_at = _operand_value(
            expr.attrs, "imm0_const", "imm0_input_index", ins, out_shape, [i, jj]
        )
        data_at = src_sem.fn(i, jj)
        l0, r0 = (
            (imm0_at, data_at)
            if expr.attrs.get("reverse0", False)
            else (data_at, imm0_at)
        )
        return _apply_binary(expr.attrs.get("op0"), l0, r0)

    op1_name = _operand_to_expr(expr.attrs.get("op1"))
    reduce_cmd_attr = expr.attrs.get("reduce_cmd")
    # Emit the body-axiom SCAN2 form when configured as a sum-scan with
    # reset-reduce; this lets the equivalence checks prove equality to other
    # scans (e.g. the public cumsum op) via SCAN2 extensionality.
    scan_eligible = (
        op1_name in ("add", "plus")
        and reduce_cmd_attr == reduce_cmd.reset_reduce
        and not expr.attrs.get("reverse1", False)
    )
    if scan_eligible:
        body_id = _BODY_IDS.next()
        target_body = z3.IntVal(body_id)
        fam = SCAN2_FAM
        ctx.add(z3.ForAll([i, j, k], fam.step(target_body, i, j, k) == _step0_at(k)))
        ctx.add(z3.ForAll([i, j], out_fn(i, j) == fam.fold(target_body, i, j)))
        return Semantics(
            expr.name,
            out_shape,
            out_fn,
            ctx,
            reduction=ReductionDesc(
                body_id,
                out_shape.dims[1],
                outer_rank=fam.outer_arity,
                kind=fam.kind,
                combine_op=fam.combine_op,
                outer_dims=tuple(out_shape.dims),
                output_transform="identity",
            ),
        )

    # Fallback: explicit recurrence (used for op1=mul/min/max, load_reduce, etc.).
    prev = z3.Function(f"SCAN_{expr.name}", z3.IntSort(), z3.IntSort(), z3.RealSort())
    init_value = _operand_value(
        expr.attrs,
        "imm1_const",
        "imm1_input_index",
        ins,
        ShapeExpr([out_shape.dims[0], z3.IntVal(1)]),
        [i, z3.IntVal(0)],
    )
    seed = z3.If(
        reduce_cmd_attr == reduce_cmd.load_reduce,
        init_value,
        z3.If(
            op1_name in ("multiply", "mul"),
            z3.RealVal(1),
            z3.If(
                op1_name in ("minimum", "min"),
                z3.RealVal(_LARGE_POSITIVE_SENTINEL),
                z3.RealVal(0),
            ),
        ),
    )
    step0_first = _step0_at(z3.IntVal(0))
    first = (
        _apply_binary(expr.attrs.get("op1"), step0_first, seed)
        if not expr.attrs.get("reverse1", False)
        else _apply_binary(expr.attrs.get("op1"), seed, step0_first)
    )
    ctx.add(z3.ForAll([i], prev(i, z3.IntVal(0)) == first))
    step0_j = _step0_at(j)
    stepj = (
        _apply_binary(expr.attrs.get("op1"), step0_j, prev(i, j - 1))
        if not expr.attrs.get("reverse1", False)
        else _apply_binary(expr.attrs.get("op1"), prev(i, j - 1), step0_j)
    )
    ctx.add(z3.ForAll([i, j], z3.Implies(j > 0, prev(i, j) == stepj)))
    ctx.add(z3.ForAll([i, j], out_fn(i, j) == prev(i, j)))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _validity_tensor_scalar_cumulative(
    ins: list[ShapeExpr], attrs: dict[str, Any]
) -> Context:
    ctx = Context()
    if not ins or ins[0].rank != 2:
        ctx.add(z3.BoolVal(False))
        return ctx
    # A tensor immediate must have a free dimension of one.
    for index_key in ("imm0_input_index", "imm1_input_index"):
        operand_index = attrs.get(index_key)
        if operand_index is None:
            continue
        operand_index = int(operand_index)
        if 0 <= operand_index < len(ins):
            operand_shape = ins[operand_index]
            if operand_shape.dims:
                ctx.add(operand_shape.dims[-1] == z3.IntVal(1))
        else:
            ctx.add(z3.BoolVal(False))
    return ctx


@semantics_hw()
def tensor_scalar_cumulative(
    dst, src, op0, op1, imm0, imm1=None, reduce_cmd=reduce_cmd.reset_reduce, name=None
):
    assert _is_sym_tensor(src)
    inputs = [src]
    attrs: dict[str, Any] = {
        "op0": op0,
        "op1": op1,
        "reverse0": False,
        "reverse1": False,
        "reduce_cmd": reduce_cmd,
        "name": name,
    }
    if _is_sym_tensor(imm0):
        attrs["imm0_input_index"] = len(inputs)
        inputs.append(imm0)
    else:
        attrs["imm0_const"] = imm0
    if _is_sym_tensor(imm1):
        attrs["imm1_input_index"] = len(inputs)
        inputs.append(imm1)
    elif imm1 is not None:
        attrs["imm1_const"] = imm1
    return _new_sym_tensor(
        "tensor_scalar_cumulative", inputs, attrs, _default_out_shape(dst, src)
    )


@semantics_hw()
def tensor_scalar_reduce(
    dst, data, op0, operand0, reduce_op, reduce_res, reverse0=False, name=None
):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        ctx = _shape_ctx(*data_shape.dims)
        if len(ins) > 1:
            ctx.extend(_broadcast_shape(data_shape, ins[1]).ctx.facts)
        return ShapeResult(ShapeExpr(list(data_shape.dims)), ctx)

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        data_sem = ins[0]
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        idx = _index_vars(expr.name, out_shape.rank)
        ctx = data_sem.ctx.merged(*[s.ctx for s in ins[1:]])
        data_value = _call_broadcasted(data_sem, out_shape, idx)
        operand_value = _operand_value(
            expr.attrs, "operand0_const", "operand0_input_index", ins, out_shape, idx
        )
        lhs, rhs = (
            (operand_value, data_value)
            if expr.attrs.get("reverse0", False)
            else (data_value, operand_value)
        )
        tmp = _apply_binary(expr.attrs.get("op0"), lhs, rhs)
        ctx.add(z3.ForAll(idx, out_fn(*idx) == tmp))
        reduction = None
        if out_shape.rank == 2:
            combine_op = _normalize_combine_op(expr.attrs.get("reduce_op"))
            fam = fold_family(1, ReductionKind.REDUCE, combine_op)
            if fam is not None:
                body_id = _BODY_IDS.next()
                i = z3.Int(f"{expr.name}_ri")
                k = z3.Int(f"{expr.name}_rk")
                n = out_shape.dims[1]
                ctx.add(
                    z3.ForAll(
                        [i, k],
                        fam.step(z3.IntVal(body_id), i, k) == out_fn(i, k),
                    )
                )
                reduction = ReductionDesc(
                    body_id,
                    n,
                    outer_rank=fam.outer_arity,
                    kind=fam.kind,
                    combine_op=fam.combine_op,
                )
        return Semantics(expr.name, out_shape, out_fn, ctx, reduction=reduction)

    _ensure_semantics("tensor_scalar_reduce", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    inputs = [data]
    attrs: dict[str, Any] = {
        "op0": op0,
        "reduce_op": reduce_op,
        "reverse0": reverse0,
        "name": name,
    }
    if _is_sym_tensor(operand0):
        attrs["operand0_input_index"] = len(inputs)
        inputs.append(operand0)
    else:
        attrs["operand0_const"] = operand0
    return _new_sym_tensor(
        "tensor_scalar_reduce", inputs, attrs, _default_out_shape(dst, data)
    )


@semantics_hw()
def tensor_tensor(dst, data1, data2, op, engine=engine.unknown, name=None):
    assert _is_sym_tensor(data1) and _is_sym_tensor(data2)
    return _new_sym_tensor(
        "tensor_tensor",
        [data1, data2],
        {"op": op, "engine": engine, "name": name},
        _default_out_shape(dst, data1),
    )


@semantics_hw()
def tensor_tensor_scan(
    dst, data0, data1, initial, op0, op1, reverse0=False, reverse1=False, name=None
):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        ctx = _shape_ctx(*data_shape.dims)
        ctx.extend(_broadcast_shape(data_shape, ins[1]).ctx.facts)
        if len(ins) > 2:
            ctx.extend(
                _broadcast_shape(
                    ShapeExpr([data_shape.dims[0], z3.IntVal(1)]), ins[2]
                ).ctx.facts
            )
        return ShapeResult(ShapeExpr(list(data_shape.dims)), ctx)

    def value_rule(
        expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
    ) -> Semantics:
        a, b = ins[0], ins[1]
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        ctx = a.ctx.merged(b.ctx, *(s.ctx for s in ins[2:]))
        if out_shape.rank != 2:
            return Semantics(expr.name, out_shape, out_fn, ctx)
        i = z3.Int(f"{expr.name}_i")
        j = z3.Int(f"{expr.name}_j")
        prev = z3.Function(
            f"SCAN_{expr.name}", z3.IntSort(), z3.IntSort(), z3.RealSort()
        )
        init = _operand_value(
            expr.attrs,
            "initial_const",
            "initial_input_index",
            ins,
            ShapeExpr([out_shape.dims[0], z3.IntVal(1)]),
            [i, z3.IntVal(0)],
        )
        first_a = a.fn(i, z3.IntVal(0))
        lhs0_first, rhs0_first = (
            (init, first_a) if expr.attrs.get("reverse0", False) else (first_a, init)
        )
        first_tmp = _apply_binary(expr.attrs.get("op0"), lhs0_first, rhs0_first)
        first_b = b.fn(i, z3.IntVal(0))
        lhs1_first, rhs1_first = (
            (first_b, first_tmp)
            if expr.attrs.get("reverse1", False)
            else (first_tmp, first_b)
        )
        first_step = _apply_binary(expr.attrs.get("op1"), lhs1_first, rhs1_first)
        prev_or_init = prev(i, j - 1)
        lhs0, rhs0 = (
            (prev_or_init, a.fn(i, j))
            if expr.attrs.get("reverse0", False)
            else (a.fn(i, j), prev_or_init)
        )
        tmp = _apply_binary(expr.attrs.get("op0"), lhs0, rhs0)
        lhs1, rhs1 = (
            (b.fn(i, j), tmp)
            if expr.attrs.get("reverse1", False)
            else (tmp, b.fn(i, j))
        )
        step = _apply_binary(expr.attrs.get("op1"), lhs1, rhs1)
        ctx.add(z3.ForAll([i], prev(i, z3.IntVal(0)) == first_step))
        ctx.add(z3.ForAll([i, j], z3.Implies(j > 0, prev(i, j) == step)))
        ctx.add(z3.ForAll([i, j], out_fn(i, j) == prev(i, j)))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    def validity_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> Context:
        ctx = Context()
        if not ins or ins[0].rank != 2:
            ctx.add(z3.BoolVal(False))
        return ctx

    _ensure_semantics("tensor_tensor_scan", shape_rule, value_rule, validity_rule)

    assert _is_sym_tensor(data0) and _is_sym_tensor(data1)
    inputs = [data0, data1]
    attrs: dict[str, Any] = {
        "op0": op0,
        "op1": op1,
        "reverse0": reverse0,
        "reverse1": reverse1,
        "name": name,
    }
    if _is_sym_tensor(initial):
        attrs["initial_input_index"] = len(inputs)
        inputs.append(initial)
    else:
        attrs["initial_const"] = initial
    return _new_sym_tensor(
        "tensor_tensor_scan", inputs, attrs, _default_out_shape(dst, data0)
    )


def _call_broadcasted(
    sem: Semantics, out_shape: ShapeExpr, indices: list[z3.ArithRef]
) -> z3.ArithRef:
    mapped = _broadcast_indices(sem, out_shape, indices)
    return sem.fn(*mapped) if mapped else sem.fn(z3.IntVal(0))


def _shape_same_as_first(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    a = ins[0]
    return ShapeResult(ShapeExpr(list(a.dims)), Context([d > 0 for d in a.dims]))


def _compile_copy(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    src = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = src.ctx.merged()
    idx = [z3.Int(f"{expr.name}_i{k}") for k in range(out_shape.rank)]
    ctx.add(z3.ForAll(idx, out_fn(*idx) == _call_broadcasted(src, out_shape, idx)))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_tensor_tensor(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    a, b = ins
    ctx = _shape_ctx(*a.dims, *b.dims)
    if a.rank != b.rank:
        ctx.add(z3.BoolVal(False))
        return ShapeResult(ShapeExpr(list(a.dims)), ctx)
    for adim, bdim in zip(a.dims, b.dims, strict=True):
        ctx.add(adim == bdim)
    return ShapeResult(ShapeExpr(list(a.dims)), ctx)


def _compile_tensor_tensor(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    a, b = ins
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = a.ctx.merged(b.ctx)
    idx = [z3.Int(f"{expr.name}_i{k}") for k in range(out_shape.rank)]
    av = a.fn(*idx)
    bv = b.fn(*idx)
    ctx.add(z3.ForAll(idx, out_fn(*idx) == _apply_binary(expr.attrs.get("op"), av, bv)))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_tensor_scalar(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    data = ins[0]
    ctx = Context([d > 0 for d in data.dims])
    if len(ins) > 1:
        ctx.extend(_broadcast_to_shape(ins[1], data).ctx.facts)
        if ins[1].dims:
            ctx.add(ins[1].dims[-1] == 1)
    if len(ins) > 2:
        ctx.extend(_broadcast_to_shape(ins[2], data).ctx.facts)
        if ins[2].dims:
            ctx.add(ins[2].dims[-1] == 1)
    return ShapeResult(ShapeExpr(list(data.dims)), ctx)


def _tensor_scalar_out_shape(
    dst: Any,
    data: SymTensor,
    operand0: Any,
    operand1: Any = None,
) -> tuple[Any, ...]:
    if isinstance(dst, SymTensor):
        return dst.shape
    return tuple(data.shape)


def _operand_value(
    attrs: dict[str, Any],
    constant_attr_key: str,
    input_index_attr_key: str,
    ins: list[Semantics],
    out_shape: ShapeExpr,
    indices: list[z3.ArithRef],
) -> z3.ArithRef:
    if input_index_attr_key in attrs:
        return _call_broadcasted(
            ins[int(attrs[input_index_attr_key])], out_shape, indices
        )
    return _as_scalar(attrs.get(constant_attr_key))


def _lift_scale_through_fold(
    name: str,
    reduction: ReductionDesc | None,
    op: Any,
    factor_at: Callable[[z3.ArithRef, z3.ArithRef], z3.ArithRef],
    reduction_on_left: bool,
    out_fn: Any,
    out_shape: ShapeExpr,
    ctx: Context,
) -> ReductionDesc | None:
    """Lift a reduction-independent scale factor into a fold body."""
    opn = _operand_to_expr(op)
    if opn not in ("mul", "multiply", "div", "divide"):
        return None
    if (
        reduction is None
        or reduction.outer_rank != 2
        or reduction.output_transform != "identity"
        or reduction.outer_dims is None
        or out_shape.rank != 2
    ):
        return None
    if opn in ("div", "divide") and not reduction_on_left:
        return None
    # A scale distributes through the fold body only for an additive fold; it
    # does not commute with max/min or a product fold.
    if reduction.combine_op != "add":
        return None
    fam = fold_family(2, reduction.kind, reduction.combine_op)
    if fam is None or not fam.fold_takes_extent:
        return None

    body_id = _BODY_IDS.next()
    target_body = z3.IntVal(body_id)
    source_body = z3.IntVal(reduction.body_id)
    i = z3.Int(f"{name}_ri")
    j = z3.Int(f"{name}_rj")
    k = z3.Int(f"{name}_rk")
    factor = factor_at(i, j)
    step = fam.step(source_body, i, j, k)
    if opn in ("mul", "multiply"):
        lifted_body = step * factor if reduction_on_left else factor * step
    else:
        lifted_body = _safe_divide(step, factor)

    ctx.add(z3.ForAll([i, j, k], fam.step(target_body, i, j, k) == lifted_body))
    ctx.add(
        z3.ForAll([i, j], out_fn(i, j) == fam.fold(target_body, i, j, reduction.extent))
    )
    return ReductionDesc(
        body_id,
        reduction.extent,
        outer_rank=fam.outer_arity,
        kind=fam.kind,
        combine_op=fam.combine_op,
        outer_dims=tuple(out_shape.dims),
        output_transform="identity",
    )


def _compile_tensor_scalar(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    data = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = data.ctx.merged(*[s.ctx for s in ins[1:]])
    idx = [z3.Int(f"{expr.name}_i{k}") for k in range(out_shape.rank)]
    dv = _call_broadcasted(data, out_shape, idx)
    op0 = _operand_value(
        expr.attrs, "operand0_const", "operand0_input_index", ins, out_shape, idx
    )
    reverse0 = expr.attrs.get("reverse0", False)
    lhs0, rhs0 = (op0, dv) if reverse0 else (dv, op0)
    tmp = _apply_binary(expr.attrs.get("op0"), lhs0, rhs0)
    if expr.attrs.get("op1") is not None:
        op1 = _operand_value(
            expr.attrs, "operand1_const", "operand1_input_index", ins, out_shape, idx
        )
        lhs1, rhs1 = (op1, tmp) if expr.attrs.get("reverse1", False) else (tmp, op1)
        tmp = _apply_binary(expr.attrs.get("op1"), lhs1, rhs1)
    ctx.add(z3.ForAll(idx, out_fn(*idx) == tmp))

    # A single per-row scale (op0 only) lifts the operand's fold, so a scaled
    # matmul keeps its descriptor; a second op composes and is not lifted.
    reduction: ReductionDesc | None = None
    if expr.attrs.get("op1") is None:
        reduction = _lift_scale_through_fold(
            expr.name,
            data.reduction,
            expr.attrs.get("op0"),
            lambda i, j: _operand_value(
                expr.attrs,
                "operand0_const",
                "operand0_input_index",
                ins,
                out_shape,
                [i, j],
            ),
            reduction_on_left=not reverse0,
            out_fn=out_fn,
            out_shape=out_shape,
            ctx=ctx,
        )
    return Semantics(expr.name, out_shape, out_fn, ctx, reduction=reduction)


def _shape_nc_transpose(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    a = ins[0]
    ctx = Context([d > 0 for d in a.dims])
    if a.rank < 2:
        ctx.add(z3.BoolVal(False))
        return ShapeResult(ShapeExpr(list(a.dims)), ctx)
    return ShapeResult(ShapeExpr([a.dims[1], a.dims[0], *a.dims[2:]]), ctx)


def _compile_nc_transpose(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    a = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = a.ctx.merged()
    reduction = None
    if out_shape.rank < 2:
        # Invalid rank: the shape rule marks validity as unsatisfiable, so
        # leave the value undefined instead of poisoning the definitions.
        return Semantics(expr.name, out_shape, out_fn, ctx)
    i = z3.Int(f"{expr.name}_i")
    j = z3.Int(f"{expr.name}_j")
    if out_shape.rank == 2:
        ctx.add(z3.ForAll([i, j], out_fn(i, j) == a.fn(j, i)))
    else:
        rest = [z3.Int(f"{expr.name}_k{t}") for t in range(out_shape.rank - 2)]
        out_idx = [i, j, *rest]
        src_idx = [j, i, *rest]
        ctx.add(z3.ForAll(out_idx, out_fn(*out_idx) == a.fn(*src_idx)))
    if out_shape.rank == 2:
        reduction = _lift_fold_through_transpose(
            expr.name, a.reduction, out_fn, out_shape, ctx
        )
    return Semantics(expr.name, out_shape, out_fn, ctx, reduction=reduction)


def _lift_fold_through_transpose(
    name: str,
    reduction: ReductionDesc | None,
    out_fn: z3.FuncDeclRef,
    out_shape: ShapeExpr,
    ctx: Context,
) -> ReductionDesc | None:
    """Carry a rank-2 fold descriptor through a 2-D transpose.

    TensorRight gets this for free, because a transpose only renames the index
    before the inner access. Here a fold is an opaque term, so each transpose
    must restate it: the new step reads the old step with the outer indices
    swapped. Every transpose-like op must go through this helper; before it,
    only ``nc_transpose`` did, so ``dma_transpose(matmul(...))`` lost its
    descriptor and fell back to quantifier instantiation."""
    if (
        reduction is None
        or reduction.outer_rank != 2
        or reduction.output_transform != "identity"
        or reduction.outer_dims is None
        or out_shape.rank != 2
    ):
        return None
    fam = fold_family(2, reduction.kind, reduction.combine_op)
    if fam is None or not fam.fold_takes_extent:
        return None
    i = z3.Int(f"{name}_ti")
    j = z3.Int(f"{name}_tj")
    k = z3.Int(f"{name}_rk")
    body_id = _BODY_IDS.next()
    source_body = z3.IntVal(reduction.body_id)
    target_body = z3.IntVal(body_id)
    ctx.add(
        z3.ForAll(
            [i, j, k],
            fam.step(target_body, i, j, k) == fam.step(source_body, j, i, k),
        )
    )
    ctx.add(
        z3.ForAll(
            [i, j],
            out_fn(i, j) == fam.fold(target_body, i, j, reduction.extent),
        )
    )
    return ReductionDesc(
        body_id,
        reduction.extent,
        outer_rank=fam.outer_arity,
        kind=fam.kind,
        combine_op=fam.combine_op,
        outer_dims=tuple(out_shape.dims),
        output_transform="identity",
    )


def _normalize_reduce_axis(axis_attr: Any, rank: int) -> Any:
    axis = axis_attr[0] if isinstance(axis_attr, (list, tuple)) else axis_attr
    if isinstance(axis, int) and axis < 0:
        return axis + rank
    return axis


def _shape_tensor_reduce(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    a = ins[0]
    ctx = Context([d > 0 for d in a.dims])
    if a.rank != 2:
        ctx.add(z3.BoolVal(False))
        return ShapeResult(ShapeExpr(list(a.dims)), ctx)
    axis = _normalize_reduce_axis(attrs.get("axis", 1), a.rank)
    keep = bool(attrs.get("keepdims", False))
    if axis == 1:
        return ShapeResult(
            ShapeExpr([a.dims[0], z3.IntVal(1)] if keep else [a.dims[0]]), ctx
        )
    if axis == 0:
        return ShapeResult(
            ShapeExpr([z3.IntVal(1), a.dims[1]] if keep else [a.dims[1]]), ctx
        )
    ctx.add(z3.BoolVal(False))
    return ShapeResult(ShapeExpr(list(a.dims)), ctx)


def _compile_tensor_reduce(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    return compile_rank2_fold_reduce(
        expr, ins, out_shape, _normalize_combine_op(expr.attrs.get("op"))
    )


def compile_rank2_fold_reduce(
    expr: SymExpr,
    ins: list[Semantics],
    out_shape: ShapeExpr,
    combine_op: str,
) -> Semantics:
    """Fold semantics for a single-axis reduction of a rank-2 tensor.

    Shared by ``tensor_reduce``, the graph ``reduce_sum``, and the public
    ``sum``/``max``/``min``/``prod`` reductions, so a public reduction and its
    ISA lowering produce the same fold family and can be proved equal by body."""
    a = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = a.ctx.merged()
    if a.shape.rank != 2:
        # Only rank-2 reductions have fold semantics.
        return Semantics(expr.name, out_shape, out_fn, ctx)
    axis = _normalize_reduce_axis(expr.attrs.get("axis", 1), a.shape.rank)
    keep = bool(expr.attrs.get("keepdims", expr.attrs.get("keep_dims", False)))
    negate = bool(expr.attrs.get("negate", False))
    i = z3.Int(f"{expr.name}_i")
    j = z3.Int(f"{expr.name}_j")
    k = z3.Int(f"{expr.name}_k")
    fam = fold_family(1, ReductionKind.REDUCE, combine_op)
    if fam is None:
        # No registered family for this combine op — leave output opaque.
        return Semantics(expr.name, out_shape, out_fn, ctx)
    body_id = _BODY_IDS.next()
    if axis == 1:
        n = a.shape.dims[1]
        ctx.add(z3.ForAll([i, k], fam.step(z3.IntVal(body_id), i, k) == a.fn(i, k)))
        val = fam.fold(z3.IntVal(body_id), i, n)
        val = z3.If(negate, -val, val)
        if keep:
            ctx.add(z3.ForAll([i, j], z3.Implies(j == 0, out_fn(i, j) == val)))
        else:
            ctx.add(z3.ForAll([i], out_fn(i) == val))
        return Semantics(
            expr.name,
            out_shape,
            out_fn,
            ctx,
            reduction=ReductionDesc(
                body_id,
                n,
                outer_rank=fam.outer_arity,
                kind=fam.kind,
                combine_op=fam.combine_op,
                outer_dims=(a.shape.dims[0],),
                output_transform="negate" if negate else "identity",
            ),
        )
    if axis == 0:
        m = a.shape.dims[0]
        ctx.add(z3.ForAll([j, k], fam.step(z3.IntVal(body_id), j, k) == a.fn(k, j)))
        val = fam.fold(z3.IntVal(body_id), j, m)
        val = z3.If(negate, -val, val)
        if keep:
            ctx.add(z3.ForAll([i, j], z3.Implies(i == 0, out_fn(i, j) == val)))
        else:
            ctx.add(z3.ForAll([j], out_fn(j) == val))
        return Semantics(
            expr.name,
            out_shape,
            out_fn,
            ctx,
            reduction=ReductionDesc(
                body_id,
                m,
                outer_rank=fam.outer_arity,
                kind=fam.kind,
                combine_op=fam.combine_op,
                outer_dims=(a.shape.dims[1],),
                output_transform="negate" if negate else "identity",
            ),
        )
    # Unsupported axis: the shape rule marks validity as unsatisfiable.
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_unary_same(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    return _shape_same_as_first(ins, attrs)


def _compile_reciprocal(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    a = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = [z3.Int(f"{expr.name}_i{k}") for k in range(out_shape.rank)]
    ctx = a.ctx.merged()
    x = _call_broadcasted(a, out_shape, idx)
    ctx.add(
        z3.ForAll(idx, out_fn(*idx) == z3.If(x == 0, z3.RealVal(0), z3.RealVal(1) / x))
    )
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _compile_activation(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    data = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = [z3.Int(f"{expr.name}_i{k}") for k in range(out_shape.rank)]
    ctx = data.ctx.merged(*[s.ctx for s in ins[1:]])
    scale = _operand_value(
        expr.attrs, "scale", "scale_input_index", ins, out_shape, idx
    )
    bias = _operand_value(
        expr.attrs, "bias_const", "bias_input_index", ins, out_shape, idx
    )
    pre = _call_broadcasted(data, out_shape, idx) * scale + bias
    act, extra = _apply_activation(expr.attrs.get("op"), pre)
    ctx.add(z3.ForAll(idx, out_fn(*idx) == act))
    for fact in extra:
        ctx.add(z3.ForAll(idx, fact))
    reduction = None
    if expr.attrs.get("with_reduce", False) and out_shape.rank == 2:
        combine_op = _normalize_combine_op(expr.attrs.get("reduce_op"))
        fam = fold_family(1, ReductionKind.REDUCE, combine_op)
        if fam is not None:
            body_id = _BODY_IDS.next()
            i = z3.Int(f"{expr.name}_ri")
            k = z3.Int(f"{expr.name}_rk")
            n = out_shape.dims[1]
            ctx.add(
                z3.ForAll([i, k], fam.step(z3.IntVal(body_id), i, k) == out_fn(i, k))
            )
            reduction = ReductionDesc(
                body_id,
                n,
                outer_rank=fam.outer_arity,
                kind=fam.kind,
                combine_op=fam.combine_op,
            )
    return Semantics(expr.name, out_shape, out_fn, ctx, reduction=reduction)


def _shape_activation_reduce(
    ins: list[ShapeExpr], attrs: dict[str, Any]
) -> ShapeResult:
    # The e-class value is the per-partition reduce result.
    data = ins[0]
    ctx = Context([d > 0 for d in data.dims])
    if data.rank != 2:
        ctx.add(z3.BoolVal(False))
        return ShapeResult(ShapeExpr(list(data.dims)), ctx)
    return ShapeResult(ShapeExpr([data.dims[0], z3.IntVal(1)]), ctx)


def _compile_activation_reduce(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    # Fold the elementwise activation over its free dimension.
    data = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = data.ctx.merged(*[s.ctx for s in ins[1:]])
    if data.shape.rank != 2:
        # Keep invalid non-rank-2 values opaque.
        return Semantics(expr.name, out_shape, out_fn, ctx)
    combine_op = _normalize_combine_op(expr.attrs.get("reduce_op"))
    fam = fold_family(1, ReductionKind.REDUCE, combine_op)
    if fam is None:
        # No registered family for this combine op; leave output opaque.
        return Semantics(expr.name, out_shape, out_fn, ctx)
    i = z3.Int(f"{expr.name}_i")
    j = z3.Int(f"{expr.name}_j")
    k = z3.Int(f"{expr.name}_k")
    idx = [i, k]
    scale = _operand_value(
        expr.attrs, "scale", "scale_input_index", ins, data.shape, idx
    )
    bias = _operand_value(
        expr.attrs, "bias_const", "bias_input_index", ins, data.shape, idx
    )
    pre = _call_broadcasted(data, data.shape, idx) * scale + bias
    act, extra = _apply_activation(expr.attrs.get("op"), pre)
    body_id = _BODY_IDS.next()
    ctx.add(z3.ForAll([i, k], fam.step(z3.IntVal(body_id), i, k) == act))
    for fact in extra:
        ctx.add(z3.ForAll([i, k], fact))
    n = data.shape.dims[1]
    ctx.add(
        z3.ForAll(
            [i, j],
            z3.Implies(j == 0, out_fn(i, j) == fam.fold(z3.IntVal(body_id), i, n)),
        )
    )
    return Semantics(
        expr.name,
        out_shape,
        out_fn,
        ctx,
        reduction=ReductionDesc(
            body_id,
            n,
            outer_rank=fam.outer_arity,
            kind=fam.kind,
            combine_op=fam.combine_op,
            outer_dims=(data.shape.dims[0],),
            output_transform="identity",
        ),
    )


def _shape_activation(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    data = ins[0]
    ctx = Context([d > 0 for d in data.dims])
    if len(ins) > 1:
        ctx.extend(_broadcast_shape(data, ins[1]).ctx.facts)
    if len(ins) > 2:
        ctx.extend(_broadcast_shape(data, ins[2]).ctx.facts)
    return ShapeResult(ShapeExpr(list(data.dims)), ctx)


def _compile_exponential(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    src = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = [z3.Int(f"{expr.name}_i{k}") for k in range(out_shape.rank)]
    ctx = src.ctx.merged(*[s.ctx for s in ins[1:]])
    maxv = _operand_value(
        expr.attrs, "max_value", "max_input_index", ins, out_shape, idx
    )
    pre = _call_broadcasted(src, out_shape, idx) - maxv
    act, extra = _apply_activation("exp", pre)
    ctx.add(z3.ForAll(idx, out_fn(*idx) == act))
    for fact in extra:
        ctx.add(z3.ForAll(idx, fact))
    reduction = None
    if expr.attrs.get("with_reduce", False) and out_shape.rank == 2:
        combine_op = _normalize_combine_op(expr.attrs.get("reduce_op"))
        fam = fold_family(1, ReductionKind.REDUCE, combine_op)
        if fam is not None:
            body_id = _BODY_IDS.next()
            i = z3.Int(f"{expr.name}_ri")
            k = z3.Int(f"{expr.name}_rk")
            n = out_shape.dims[1]
            ctx.add(
                z3.ForAll([i, k], fam.step(z3.IntVal(body_id), i, k) == out_fn(i, k))
            )
            reduction = ReductionDesc(
                body_id,
                n,
                outer_rank=fam.outer_arity,
                kind=fam.kind,
                combine_op=fam.combine_op,
            )
    return Semantics(expr.name, out_shape, out_fn, ctx, reduction=reduction)


def _compare_bool(op: Any, lhs: z3.ArithRef, rhs: z3.ArithRef) -> z3.BoolRef:
    opn = _operand_to_expr(op)
    if opn == "equal":
        return lhs == rhs
    if opn == "less":
        return lhs < rhs
    if opn == "less_equal":
        return lhs <= rhs
    if opn == "greater":
        return lhs > rhs
    if opn == "greater_equal":
        return lhs >= rhs
    key = str(opn)
    with _UF_LOCK:
        if key not in _COMPARE_UFS:
            _COMPARE_UFS[key] = z3.Function(
                f"NKI_CMP_{len(_COMPARE_UFS)}",
                z3.RealSort(),
                z3.RealSort(),
                z3.BoolSort(),
            )
        fn = _COMPARE_UFS[key]
    return fn(lhs, rhs)


def _compile_select_reduce(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    pred, on_true = ins[0], ins[1]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = [z3.Int(f"{expr.name}_i{k}") for k in range(out_shape.rank)]
    ctx = pred.ctx.merged(on_true.ctx, *[s.ctx for s in ins[2:]])
    p = _call_broadcasted(pred, out_shape, idx) != 0
    p = z3.Not(p) if expr.attrs.get("reverse_pred", False) else p
    t = _call_broadcasted(on_true, out_shape, idx)
    f = _operand_value(
        expr.attrs, "on_false_const", "on_false_input_index", ins, out_shape, idx
    )
    ctx.add(z3.ForAll(idx, out_fn(*idx) == z3.If(p, t, f)))
    reduction = None
    if expr.attrs.get("with_reduce", False) and out_shape.rank == 2:
        combine_op = _normalize_combine_op(expr.attrs.get("reduce_op"))
        fam = fold_family(1, ReductionKind.REDUCE, combine_op)
        if fam is not None:
            body_id = _BODY_IDS.next()
            i = z3.Int(f"{expr.name}_ri")
            k = z3.Int(f"{expr.name}_rk")
            n = out_shape.dims[1]
            ctx.add(
                z3.ForAll([i, k], fam.step(z3.IntVal(body_id), i, k) == out_fn(i, k))
            )
            reduction = ReductionDesc(
                body_id,
                n,
                outer_rank=fam.outer_arity,
                kind=fam.kind,
                combine_op=fam.combine_op,
            )
    return Semantics(expr.name, out_shape, out_fn, ctx, reduction=reduction)


def _extract_iota_step(pattern: Any) -> int:
    if isinstance(pattern, (list, tuple)) and pattern:
        tail_pos = len(pattern) - 1
        tail = pattern[tail_pos]
        if isinstance(tail, (list, tuple)) and len(tail) >= 1:
            try:
                return int(tail[0])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"iota pattern step values must be integers, got {tail[0]!r} in entry {tail_pos} of {pattern!r}"
                ) from exc
    return 1


def _shape_public_attr_or_first(
    ins: list[ShapeExpr], attrs: dict[str, Any]
) -> ShapeResult:
    out_shape = attrs.get("out_shape")
    if out_shape is not None:
        return _shape_from_out(tuple(out_shape))
    shape = attrs.get("shape")
    if shape is not None:
        return _shape_from_out(tuple(shape))
    if ins:
        return _shape_same_as_first(ins, attrs)
    return _shape_from_out((z3.IntVal(1),))


def _compile_public_opaque(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = _index_vars(expr.name, out_shape.rank)
    ctx = Context()
    for sem in ins:
        ctx = ctx.merged(sem.ctx)
    opaque = z3.Function(
        f"PUBLIC_{expr.name}",
        *([z3.RealSort()] * len(ins)),
        *([z3.IntSort()] * builtins.max(1, out_shape.rank)),
        z3.RealSort(),
    )
    input_values = [_call_broadcasted(sem, out_shape, idx) for sem in ins]
    index_args = idx if idx else [z3.IntVal(0)]
    ctx.add(z3.ForAll(idx, out_fn(*idx) == opaque(*(input_values + index_args))))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_public_unary(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    return _shape_same_as_first(ins, attrs)


def _compile_public_unary(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    a = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = _index_vars(expr.name, out_shape.rank)
    ctx = a.ctx.merged()
    value, facts = _apply_activation(
        expr.attrs.get("op", expr.op), _call_broadcasted(a, out_shape, idx)
    )
    # Close each fact over the index variables. A bare fact asserted the value
    # fact only at one arbitrary index, which no proof could use.
    for fact in facts:
        ctx.add(z3.ForAll(idx, fact) if idx else fact)
    ctx.add(z3.ForAll(idx, out_fn(*idx) == value))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_public_binary(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    if len(ins) >= 2:
        return _broadcast_shape(ins[0], ins[1])
    if len(ins) == 1:
        return _shape_same_as_first(ins, attrs)
    return _shape_from_out(attrs.get("out_shape", (z3.IntVal(1),)))


def _compile_public_binary(
    expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr
) -> Semantics:
    def lift_binary_reduction(op_name: Any) -> ReductionDesc | None:
        if len(ins) != 2:
            return None
        # Whichever operand carries the fold is the one scaled through; a
        # division keeps the fold in the numerator only.
        reduction_index = next(
            (idx for idx, sem in enumerate(ins) if sem.reduction is not None), None
        )
        if reduction_index is None:
            return None
        factor_sem = ins[1 - reduction_index]
        return _lift_scale_through_fold(
            expr.name,
            ins[reduction_index].reduction,
            op_name,
            lambda i, j: _call_broadcasted(factor_sem, out_shape, [i, j]),
            reduction_on_left=reduction_index == 0,
            out_fn=out_fn,
            out_shape=out_shape,
            ctx=ctx,
        )

    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = _index_vars(expr.name, out_shape.rank)
    ctx = Context()
    for sem in ins:
        ctx = ctx.merged(sem.ctx)
    if len(ins) >= 2:
        lhs = _call_broadcasted(ins[0], out_shape, idx)
        rhs = _call_broadcasted(ins[1], out_shape, idx)
    elif len(ins) == 1:
        lhs = _call_broadcasted(ins[0], out_shape, idx)
        rhs = _as_scalar(
            expr.attrs.get(
                "scalar",
                expr.attrs.get(
                    "rhs",
                    expr.attrs.get("value", expr.attrs.get("operand0_const", 0)),
                ),
            )
        )
        if expr.attrs.get("reverse", False):
            lhs, rhs = rhs, lhs
    else:
        return _compile_public_opaque(expr, ins, out_shape)
    ctx.add(
        z3.ForAll(
            idx, out_fn(*idx) == _apply_binary(expr.attrs.get("op", expr.op), lhs, rhs)
        )
    )
    return Semantics(
        expr.name,
        out_shape,
        out_fn,
        ctx,
        reduction=lift_binary_reduction(expr.attrs.get("op", expr.op)),
    )


def register_hw_semantics() -> None:
    """Register the hardware-operation semantics that synthesis evaluates.
    ``reciprocal`` stays lazy because ``lang_semantics`` owns that entry."""
    _ensure_semantics("activation", _shape_activation, _compile_activation)
    _ensure_semantics(
        "activation_reduce", _shape_activation_reduce, _compile_activation_reduce
    )
    _ensure_semantics("dma_copy", _shape_same_as_first, _compile_copy)
    _ensure_semantics(
        "dma_transpose",
        _shape_dma_transpose,
        _compile_dma_transpose,
        _validity_dma_transpose,
    )
    _ensure_semantics("exponential", _shape_activation, _compile_exponential)
    _ensure_semantics(
        "nc_matmul", _shape_nc_matmul, _compile_nc_matmul, _validity_nc_matmul
    )
    _ensure_semantics("nc_transpose", _shape_nc_transpose, _compile_nc_transpose)
    _ensure_semantics(
        "scalar_tensor_tensor",
        _shape_scalar_tensor_tensor,
        _compile_scalar_tensor_tensor,
        _validity_scalar_tensor_tensor,
    )
    _ensure_semantics("tensor_copy", _shape_same_as_first, _compile_copy)
    _ensure_semantics(
        "tensor_partition_reduce",
        _shape_tensor_partition_reduce,
        _compile_tensor_partition_reduce,
    )
    _ensure_semantics("tensor_reduce", _shape_tensor_reduce, _compile_tensor_reduce)
    _ensure_semantics("tensor_scalar", _shape_tensor_scalar, _compile_tensor_scalar)
    _ensure_semantics(
        "tensor_scalar_cumulative",
        _shape_tensor_scalar_cumulative,
        _compile_tensor_scalar_cumulative,
        _validity_tensor_scalar_cumulative,
    )
    _ensure_semantics("tensor_tensor", _shape_tensor_tensor, _compile_tensor_tensor)


register_hw_semantics()
