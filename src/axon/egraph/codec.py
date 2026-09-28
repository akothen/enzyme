"""Encode and decode generic tensor and ISA e-nodes.

Each language is a dedicated input constructor plus generic arity constructors
carrying (op, attrs payload, ordered children). Canonicalization happens before
insertion; encoders reject an operation without registered semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from axon.egraph import isa_language as il
from axon.egraph import tensor_language as tl
from axon.egraph.adapter import EClassRef, ENodeRef, LitArg, Snapshot
from axon.egraph.payload import PayloadError, decode_attrs, encode_attrs
from axon.egraph.values import Dim, Shape
from axon.isa_semantics import lookup_semantics


class CodecError(ValueError):
    """Raised when an operation, attribute, or value has no declared encoding."""


# Shared value coding


def encode_dim(dim: int | str) -> Any:
    if isinstance(dim, bool):
        raise CodecError(f"boolean is not a tensor dimension: {dim!r}")
    if isinstance(dim, int):
        return Dim.lit(dim)
    if isinstance(dim, str):
        return Dim.sym(dim)
    raise CodecError(f"unsupported dimension value {dim!r}")


def encode_shape(dims: tuple[int | str, ...]) -> Any:
    shape = Shape.nil()
    for dim in dims:
        shape = shape.cons(encode_dim(dim))
    return shape


def _single_row(snapshot: Snapshot, ref: EClassRef, context: str) -> ENodeRef:
    rows = snapshot.members(ref)
    if len(rows) != 1:
        raise CodecError(
            f"{context}: value e-class {ref!r} has {len(rows)} rows; "
            "expected exactly one"
        )
    return rows[0]


def _lit(arg: LitArg | EClassRef, context: str) -> Any:
    if not isinstance(arg, LitArg):
        raise CodecError(f"{context}: expected a literal argument, got {arg!r}")
    return arg.value


def _cls(arg: LitArg | EClassRef, context: str) -> EClassRef:
    if not isinstance(arg, EClassRef):
        raise CodecError(f"{context}: expected an e-class argument, got {arg!r}")
    return arg


def decode_dim(snapshot: Snapshot, ref: EClassRef) -> int | str:
    row = _single_row(snapshot, ref, "decode_dim")
    if row.egg_fn == "axDimLit":
        return int(_lit(row.args[0], "decode_dim"))
    if row.egg_fn == "axDimSym":
        return str(_lit(row.args[0], "decode_dim"))
    raise CodecError(f"decode_dim: unexpected constructor {row.egg_fn!r}")


def decode_shape(snapshot: Snapshot, ref: EClassRef) -> tuple[int | str, ...]:
    dims: list[int | str] = []
    current = ref
    while True:
        row = _single_row(snapshot, current, "decode_shape")
        if row.egg_fn == "axShapeNil":
            break
        if row.egg_fn != "axShapeCons":
            raise CodecError(f"decode_shape: unexpected constructor {row.egg_fn!r}")
        prefix = next(
            a for a in row.args if isinstance(a, EClassRef) and a.sort == "Shape"
        )
        dim_ref = next(
            a for a in row.args if isinstance(a, EClassRef) and a.sort == "Dim"
        )
        dims.append(decode_dim(snapshot, dim_ref))
        current = prefix
    return tuple(reversed(dims))


# Generic encoding and decoding shared by both expression languages


def _encode_generic_enode(
    constructors: dict[int, Any],
    op: str,
    attrs: dict[str, Any],
    children: list[Any],
    language: str,
) -> Any:
    try:
        lookup_semantics(op)
    except KeyError as exc:
        raise CodecError(str(exc)) from exc
    try:
        payload = encode_attrs(attrs)
    except PayloadError as exc:
        raise CodecError(f"operation '{op}': {exc}") from exc
    constructor = constructors.get(len(children))
    if constructor is None:
        raise CodecError(
            f"operation '{op}' has {len(children)} children; the {language} "
            f"language declares arities {sorted(constructors)}"
        )
    return constructor(op, payload, *children)


def _decode_input(enode: ENodeRef, snapshot: Snapshot, decoded_cls: type) -> Any:
    ctx = f"decode[{enode.egg_fn}]"
    return decoded_cls(
        op="input",
        child_classes=(),
        attrs={},
        source_id=str(_lit(enode.args[0], ctx)),
        input_shape=decode_shape(snapshot, _cls(enode.args[1], ctx)),
    )


def _decode_generic_enode(enode: ENodeRef, decoded_cls: type) -> Any:
    ctx = f"decode[{enode.egg_fn}]"
    op = str(_lit(enode.args[0], ctx))
    try:
        attrs = decode_attrs(str(_lit(enode.args[1], ctx)))
    except PayloadError as exc:
        raise CodecError(f"operation '{op}': {exc}") from exc
    return decoded_cls(
        op=op,
        child_classes=tuple(_cls(arg, ctx) for arg in enode.args[2:]),
        attrs=attrs,
    )


# Tensor codec

_GENERIC_TENSOR_CONSTRUCTORS = {1: tl.t_op1, 2: tl.t_op2}


@dataclass
class DecodedTensorENode:
    """Canonical operation, ordered child classes, and semantics attributes."""

    op: str
    child_classes: tuple[EClassRef, ...]
    attrs: dict[str, Any]
    source_id: str | None = None
    input_shape: tuple[int | str, ...] | None = None


def encode_tensor_enode(op: str, attrs: dict[str, Any], children: list[Any]) -> Any:
    """Encode one tensor operation with children in ``Node.inputs`` order.
    ``op`` and ``attrs`` must be canonical; see ``tensor.canonical_tensor_parts``."""
    return _encode_generic_enode(
        _GENERIC_TENSOR_CONSTRUCTORS, op, attrs, children, "tensor"
    )


def decode_tensor_enode(snapshot: Snapshot, enode: ENodeRef) -> DecodedTensorENode:
    """Recover the canonical operation, children, and attributes of one row."""
    if enode.egg_fn == "axTInput":
        return _decode_input(enode, snapshot, DecodedTensorENode)
    if enode.egg_fn not in ("axTOp1", "axTOp2"):
        raise CodecError(
            f"no tensor decoding declared for constructor {enode.egg_fn!r}"
        )
    return _decode_generic_enode(enode, DecodedTensorENode)


def encode_tensor_input(source_id: str, shape: tuple[int | str, ...]) -> Any:
    return tl.t_input(source_id, encode_shape(shape))


# ISA codec

_GENERIC_ISA_CONSTRUCTORS = {1: il.i_op1, 2: il.i_op2, 3: il.i_op3}


@dataclass
class DecodedIsaENode:
    """The exact Node operation, ordered child classes, and codegen attrs."""

    op: str
    child_classes: tuple[EClassRef, ...]
    attrs: dict[str, Any]
    source_id: str | None = None
    input_shape: tuple[int | str, ...] | None = None


def encode_isa_enode(op: str, attrs: dict[str, Any], children: list[Any]) -> Any:
    """Encode one ISA operation with children in ``Node.inputs`` order.
    ``op`` must be canonical; alias conversion happens before insertion."""
    return _encode_generic_enode(_GENERIC_ISA_CONSTRUCTORS, op, attrs, children, "ISA")


def decode_isa_enode(snapshot: Snapshot, enode: ENodeRef) -> DecodedIsaENode:
    """Recover the exact Node operation, children, and codegen attributes."""
    if enode.egg_fn == "axIInput":
        return _decode_input(enode, snapshot, DecodedIsaENode)
    if enode.egg_fn not in ("axIOp1", "axIOp2", "axIOp3"):
        raise CodecError(f"no ISA decoding declared for constructor {enode.egg_fn!r}")
    return _decode_generic_enode(enode, DecodedIsaENode)


def encode_isa_input(source_id: str, shape: tuple[int | str, ...]) -> Any:
    return il.i_input(source_id, encode_shape(shape))
