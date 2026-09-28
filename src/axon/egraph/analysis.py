"""Compute symbolic shapes and legality facts for e-classes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import InitVar, dataclass
from typing import cast

import z3

import axon.lang_semantics  # noqa: F401  (registers public semantics on import)
from axon.egraph.adapter import EClassRef, ENodeRef, Snapshot
from axon.egraph.codec import (
    DecodedIsaENode,
    DecodedTensorENode,
    decode_isa_enode,
    decode_tensor_enode,
)
from axon.isa_semantics import (
    SemanticsEntry,
    ShapeExpr,
    SymTensor,
    _to_dim,
    activation,
    activation_reduce,
    dma_copy,
    dma_transpose,
    exponential,
    lookup_semantics,
    nc_matmul,
    nc_transpose,
    nl,
    reciprocal,
    scalar_tensor_tensor,
    tensor_copy,
    tensor_partition_reduce,
    tensor_reduce,
    tensor_scalar,
    tensor_scalar_cumulative,
    tensor_tensor,
)


class AnalysisError(RuntimeError):
    """Raised when an e-class violates the analysis invariants."""


_ISA_SEMANTICS_SEEDED = False


def ensure_isa_semantics_registered() -> None:
    """Register each lazy ISA semantic rule in this process."""
    global _ISA_SEMANTICS_SEEDED
    if _ISA_SEMANTICS_SEEDED:
        return
    x = SymTensor("__egraph_seed_x", rank=2)
    y = SymTensor("__egraph_seed_y", rank=2)
    activation(dst=None, op=nl.copy, data=x)
    activation_reduce(dst=None, op=nl.copy, data=x, reduce_op=nl.add, reduce_res=True)
    dma_copy(dst=None, src=x)
    dma_transpose(dst=None, src=x, axes=(1, 0))
    exponential(dst=None, src=x, max_value=0.0)
    nc_matmul(dst=None, stationary=x, moving=y)
    nc_transpose(dst=None, data=x)
    reciprocal(dst=None, data=x)
    scalar_tensor_tensor(
        dst=None, data=x, op0=nl.multiply, operand0=2.0, op1=nl.add, operand1=y
    )
    tensor_copy(dst=None, src=x)
    tensor_partition_reduce(dst=None, op=nl.add, data=x)
    tensor_reduce(dst=None, op=nl.add, data=x, axis=1, keepdims=True)
    tensor_scalar(dst=None, data=x, op0=nl.multiply, operand0=2.0)
    tensor_scalar_cumulative(dst=None, src=x, op0=nl.multiply, op1=nl.add, imm0=1.0)
    tensor_tensor(dst=None, data1=x, data2=y, op=nl.add)
    _ISA_SEMANTICS_SEEDED = True


@dataclass(frozen=True)
class EClassAnalysis:
    """Symbolic shape and guaranteed facts of one expression e-class."""

    dims: tuple[z3.ArithRef, ...]
    facts: tuple[z3.BoolRef, ...]
    _normalized: InitVar[bool] = False

    def __post_init__(self, _normalized: bool) -> None:
        if _normalized:
            return
        dims = tuple(cast(z3.ArithRef, z3.simplify(dim)) for dim in self.dims)
        facts = _merge_normalized_facts((), self.facts)
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "facts", facts)

    @property
    def rank(self) -> int:
        return len(self.dims)


def _merge_normalized_facts(
    inherited: tuple[z3.BoolRef, ...],
    generated: tuple[z3.BoolRef, ...],
) -> tuple[z3.BoolRef, ...]:
    """Merge normalized inherited facts while simplifying new facts once."""
    facts: list[z3.BoolRef] = []
    by_hash: dict[int, list[z3.BoolRef]] = {}

    def append(fact: z3.BoolRef) -> None:
        if z3.is_true(fact):
            return
        bucket = by_hash.setdefault(fact.hash(), [])
        if any(z3.eq(fact, existing) for existing in bucket):
            return
        bucket.append(fact)
        facts.append(fact)

    for fact in inherited:
        append(fact)
    for fact in generated:
        append(cast(z3.BoolRef, z3.simplify(fact)))
    return tuple(facts)


DecodeFn = Callable[[Snapshot, ENodeRef], DecodedTensorENode | DecodedIsaENode]

_EXPRESSION_SORTS = frozenset({"TensorExpr", "IsaExpr"})


def _registry_entry(op: str) -> SemanticsEntry:
    try:
        return lookup_semantics(op)
    except KeyError as exc:
        raise AnalysisError(str(exc)) from exc


def _input_dims(shape: tuple[int | str, ...]) -> tuple[z3.ArithRef, ...]:
    return tuple(
        z3.IntVal(dim) if isinstance(dim, int) else z3.Int(str(dim)) for dim in shape
    )


def analyze_enode(
    snapshot: Snapshot,
    enode: ENodeRef,
    child_analyses: list[EClassAnalysis],
    decode: DecodeFn,
) -> EClassAnalysis:
    """Apply the registered shape and validity rules to one decoded e-node."""
    decoded = decode(snapshot, enode)
    if decoded.op == "input":
        assert decoded.input_shape is not None
        dims = _input_dims(decoded.input_shape)
        return EClassAnalysis(dims=dims, facts=tuple(d > 0 for d in dims))
    entry = _registry_entry(decoded.op)
    input_shapes = [ShapeExpr(list(analysis.dims)) for analysis in child_analyses]
    attrs = dict(decoded.attrs)
    shape_res = entry.shape_rule(input_shapes, attrs)
    inherited_facts: list[z3.BoolRef] = []
    for analysis in child_analyses:
        inherited_facts.extend(analysis.facts)
    generated_facts = [
        *shape_res.ctx.facts,
        *entry.validity_rule(input_shapes, attrs).facts,
    ]
    return EClassAnalysis(
        dims=tuple(
            cast(z3.ArithRef, z3.simplify(_to_dim(d))) for d in shape_res.out.dims
        ),
        facts=_merge_normalized_facts(
            tuple(inherited_facts),
            tuple(generated_facts),
        ),
        _normalized=True,
    )


def _decoded_child_classes(
    snapshot: Snapshot, enode: ENodeRef, decode: DecodeFn
) -> tuple[EClassRef, ...] | None:
    """Return semantic children in operand order, or ``None`` for an unknown row."""
    try:
        decoded = decode(snapshot, enode)
    except Exception:
        return None
    return decoded.child_classes


def analyze_snapshot(
    snapshot: Snapshot,
    decode: DecodeFn,
    classes: list[EClassRef] | None = None,
) -> dict[EClassRef, EClassAnalysis]:
    """Compute shape and legality facts to a fixed point."""
    if classes is None:
        classes = [ref for ref in snapshot.classes if ref.sort in _EXPRESSION_SORTS]
    pending = [ref for ref in classes if ref.sort in _EXPRESSION_SORTS]
    analyses: dict[EClassRef, EClassAnalysis] = {}
    changed = True
    while changed and pending:
        changed = False
        still_pending: list[EClassRef] = []
        for ref in pending:
            analysis = None
            for row in snapshot.members(ref):
                child_refs = _decoded_child_classes(snapshot, row, decode)
                if child_refs is None:
                    continue
                if any(child not in analyses for child in child_refs):
                    continue
                analysis = analyze_enode(
                    snapshot, row, [analyses[c] for c in child_refs], decode
                )
                break
            if analysis is None:
                still_pending.append(ref)
            else:
                analyses[ref] = analysis
                changed = True
        pending = still_pending
    return analyses


def decode_tensor(snapshot: Snapshot, enode: ENodeRef) -> DecodedTensorENode:
    """Decode dispatch for tensor-level analyses."""
    return decode_tensor_enode(snapshot, enode)


def decode_isa(snapshot: Snapshot, enode: ENodeRef) -> DecodedIsaENode:
    """Decode dispatch for ISA-level analyses (semantics must be seeded)."""
    ensure_isa_semantics_registered()
    return decode_isa_enode(snapshot, enode)


__all__ = [
    "AnalysisError",
    "EClassAnalysis",
    "analyze_enode",
    "analyze_snapshot",
    "decode_isa",
    "decode_tensor",
    "ensure_isa_semantics_registered",
]
