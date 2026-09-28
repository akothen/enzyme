"""Deterministic tagged codec for e-node attribute payloads.

Encodes exactly the value types that ``Node.attrs`` uses; every other value
is rejected. Payload equality preserves current e-node attribute identity:
values are type-exact (1, 1.0, and True are distinct), floats compare
bit-exactly except that -0.0 normalizes to 0.0 (egglog f64 literals unify
the two), lists and tuples both normalize to tuples (the typed encoders
tuple-normalize every sequence), mappings compare by sorted string keys and
equal values, enums by registered class and member name, and operation
values (``nl.<name>``) by name. NaN has no encoding.
"""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any

from axon.isa_semantics import (
    _OpRef,
    dge_mode,
    engine,
    matmul_perf_mode,
    nl,
    oob_mode,
    reduce_cmd,
)


class PayloadError(ValueError):
    """Raised for a value with no declared encoding or a malformed payload."""


# The closed set of payload type tags; compared against attribute-types.json.
SUPPORTED_TYPE_TAGS: frozenset[str] = frozenset(
    {"none", "bool", "int", "float", "str", "sequence", "mapping", "enum", "op"}
)

# Enum classes that Node.attrs carries, keyed by the class name used in payloads.
SUPPORTED_ENUM_TYPES: dict[str, type[Enum]] = {
    cls.__name__: cls
    for cls in (dge_mode, engine, matmul_perf_mode, oob_mode, reduce_cmd)
}

_CANONICAL_INT = re.compile(r"0|-?[1-9][0-9]*")


# Encoding


def _float_text(value: float) -> str:
    if math.isnan(value):
        raise PayloadError("NaN has no attribute encoding")
    if value == 0.0:
        value = 0.0  # egglog f64 identity unifies 0.0 and -0.0
    return value.hex()


def _frame(text: str) -> str:
    return f"{len(text)}:{text}"


def _encode_sequence(value: list[Any] | tuple[Any, ...]) -> str:
    items = [_frame(encode_value(item)) for item in value]
    return f"T{len(items)}:" + "".join(items)


def _encode_mapping(value: dict[Any, Any]) -> str:
    parts: list[str] = []
    for key in sorted(value):
        if not isinstance(key, str):
            raise PayloadError(f"mapping keys must be strings, got {key!r}")
        parts.append(_frame(key))
        parts.append(_frame(encode_value(value[key])))
    return f"M{len(value)}:" + "".join(parts)


def _encode_enum(value: Enum) -> str:
    cls = type(value)
    if SUPPORTED_ENUM_TYPES.get(cls.__name__) is not cls:
        raise PayloadError(f"unsupported enum type {cls.__qualname__}")
    return f"E{cls.__name__}:{value.name}"


def encode_value(value: Any) -> str:
    """Encode one attribute value as a deterministic tagged payload."""
    if value is None:
        return "N"
    if isinstance(value, bool):
        return "B1" if value else "B0"
    if isinstance(value, Enum):
        return _encode_enum(value)
    if isinstance(value, int):
        return f"I{value}"
    if isinstance(value, float):
        return "F" + _float_text(value)
    if isinstance(value, str):
        return "S" + value
    if isinstance(value, _OpRef):
        return "O" + value.name
    if isinstance(value, (list, tuple)):
        return _encode_sequence(value)
    if isinstance(value, dict):
        return _encode_mapping(value)
    raise PayloadError(
        f"unsupported attribute value {value!r} of type {type(value).__qualname__}"
    )


def encode_attrs(attrs: dict[str, Any]) -> str:
    """Encode a full attribute mapping; insertion order never affects bytes."""
    if not isinstance(attrs, dict):
        raise PayloadError(f"attrs must be a mapping, got {type(attrs).__qualname__}")
    return _encode_mapping(attrs)


# Decoding


def _take_frame(text: str, pos: int, context: str) -> tuple[str, int]:
    sep = text.find(":", pos)
    if sep < 0:
        raise PayloadError(f"{context}: missing frame length at offset {pos}")
    length_text = text[pos:sep]
    if not _CANONICAL_INT.fullmatch(length_text) or length_text.startswith("-"):
        raise PayloadError(f"{context}: malformed frame length {length_text!r}")
    end = sep + 1 + int(length_text)
    if end > len(text):
        raise PayloadError(f"{context}: frame overruns the payload")
    return text[sep + 1 : end], end


def _split_count(body: str, context: str) -> tuple[int, str]:
    count_text, sep, items_text = body.partition(":")
    if not sep or not _CANONICAL_INT.fullmatch(count_text) or "-" in count_text:
        raise PayloadError(f"{context}: malformed element count in {body!r}")
    return int(count_text), items_text


def _decode_float(body: str) -> float:
    try:
        value = float.fromhex(body)
    except ValueError:
        raise PayloadError(f"malformed float payload {body!r}") from None
    if math.isnan(value) or _float_text(value) != body:
        raise PayloadError(f"non-canonical float payload {body!r}")
    return value


def _decode_op(body: str) -> Any:
    value = getattr(nl, body) if body else None
    if not isinstance(value, _OpRef):
        raise PayloadError(f"malformed operation payload {body!r}")
    return value


def _decode_enum(body: str) -> Enum:
    cls_name, sep, member = body.partition(":")
    cls = SUPPORTED_ENUM_TYPES.get(cls_name)
    if not sep or cls is None:
        raise PayloadError(f"malformed enum payload {body!r}")
    try:
        return cls[member]
    except KeyError:
        raise PayloadError(f"unknown member {member!r} for enum {cls_name}") from None


def _decode_sequence(body: str) -> tuple[Any, ...]:
    count, items_text = _split_count(body, "sequence")
    items: list[Any] = []
    pos = 0
    for _ in range(count):
        item_text, pos = _take_frame(items_text, pos, "sequence")
        items.append(decode_value(item_text))
    if pos != len(items_text):
        raise PayloadError("sequence payload has trailing data")
    return tuple(items)


def _decode_mapping(body: str) -> dict[str, Any]:
    count, items_text = _split_count(body, "mapping")
    out: dict[str, Any] = {}
    pos = 0
    previous: str | None = None
    for _ in range(count):
        key, pos = _take_frame(items_text, pos, "mapping key")
        if previous is not None and key <= previous:
            raise PayloadError(f"mapping keys not sorted and distinct at {key!r}")
        value_text, pos = _take_frame(items_text, pos, "mapping value")
        out[key] = decode_value(value_text)
        previous = key
    if pos != len(items_text):
        raise PayloadError("mapping payload has trailing data")
    return out


_SCALAR_DECODERS = {
    "F": _decode_float,
    "S": lambda body: body,
    "O": _decode_op,
    "E": _decode_enum,
    "T": _decode_sequence,
    "M": _decode_mapping,
}


def decode_value(payload: str) -> Any:
    """Decode one payload back to the exact encoded value."""
    if not isinstance(payload, str) or not payload:
        raise PayloadError(f"payload must be a non-empty string, got {payload!r}")
    tag, body = payload[0], payload[1:]
    if tag == "N":
        if body:
            raise PayloadError(f"malformed None payload {payload!r}")
        return None
    if tag == "B":
        if body not in ("0", "1"):
            raise PayloadError(f"malformed boolean payload {payload!r}")
        return body == "1"
    if tag == "I":
        if not _CANONICAL_INT.fullmatch(body):
            raise PayloadError(f"non-canonical integer payload {body!r}")
        return int(body)
    decoder = _SCALAR_DECODERS.get(tag)
    if decoder is None:
        raise PayloadError(f"unknown payload tag {tag!r}")
    return decoder(body)


def decode_attrs(payload: str) -> dict[str, Any]:
    """Decode a full attribute mapping payload produced by ``encode_attrs``."""
    value = decode_value(payload)
    if not isinstance(value, dict):
        raise PayloadError(f"payload {payload!r} does not encode an attribute mapping")
    return value
