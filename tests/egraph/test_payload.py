"""Payload codec tests: round-trip, determinism, injectivity, fail-closed."""

from __future__ import annotations

import enum
import math

import pytest

from axon.egraph.payload import (
    SUPPORTED_ENUM_TYPES,
    SUPPORTED_TYPE_TAGS,
    PayloadError,
    decode_attrs,
    decode_value,
    encode_attrs,
    encode_value,
)
from axon.isa_semantics import (
    dge_mode,
    engine,
    matmul_perf_mode,
    nl,
    oob_mode,
    reduce_cmd,
)

ROUND_TRIP_VALUES = [
    None,
    True,
    False,
    0,
    1,
    -1,
    2**80,
    -(2**80),
    0.0,
    1.5,
    -2.25,
    0.1,
    math.inf,
    -math.inf,
    5e-324,
    1e308,
    "",
    "hello",
    "12:34",
    "N",
    "unicode é中",
    (),
    (1, 2, 3),
    ("m", 128, ("k", 4)),
    (0, "x", None, True, 2.5),
    {},
    {"axis": (0, 1), "keep_dims": False},
    {"nested": {"shape": (128, "n")}, "scale": 1.0},
    dge_mode.swdge,
    engine.tensor,
    matmul_perf_mode.double_row,
    oob_mode.skip,
    reduce_cmd.reset_reduce,
    nl.add,
    nl.multiply,
    nl.exp,
]


class TestRoundTrip:
    @pytest.mark.parametrize("value", ROUND_TRIP_VALUES, ids=repr)
    def test_exact_round_trip(self, value: object) -> None:
        decoded = decode_value(encode_value(value))
        assert decoded == value
        assert type(decoded) is type(value)

    def test_float_round_trip_is_bit_exact(self) -> None:
        for value in (0.1, 1e308, 5e-324, -2.25, math.inf, -math.inf):
            assert math.copysign(1.0, decode_value(encode_value(value))) == (
                math.copysign(1.0, value)
            )
            assert decode_value(encode_value(value)).hex() == value.hex()

    def test_negative_zero_normalizes_to_positive_zero(self) -> None:
        # egglog f64 identity unifies 0.0 and -0.0, so the payloads must too.
        assert encode_value(-0.0) == encode_value(0.0)
        decoded = decode_value(encode_value(-0.0))
        assert decoded == 0.0
        assert math.copysign(1.0, decoded) == 1.0

    def test_list_normalizes_to_tuple(self) -> None:
        # The typed encoders tuple-normalize sequences, so [0, 1] == (0, 1).
        assert encode_value([0, 1]) == encode_value((0, 1))
        assert decode_value(encode_value([0, 1])) == (0, 1)

    def test_operation_values_decode_to_nl_singletons(self) -> None:
        assert decode_value(encode_value(nl.add)) is nl.add
        assert decode_value(encode_value(nl.subtract)) is nl.subtract

    def test_enum_members_decode_to_identical_members(self) -> None:
        for cls in SUPPORTED_ENUM_TYPES.values():
            for member in cls:
                assert decode_value(encode_value(member)) is member

    def test_attrs_round_trip(self) -> None:
        attrs = {
            "op": nl.add,
            "reduce_cmd": reduce_cmd.idle,
            "bias_const": None,
            "scale": 1.0,
            "axes": (1, 0),
            "with_reduce": False,
            "sym_shape": ("m", 128),
        }
        assert decode_attrs(encode_attrs(attrs)) == attrs


class TestDeterminism:
    def test_mapping_insertion_order_is_byte_equal(self) -> None:
        forward = {"a": 1, "b": (2.0,), "c": None}
        reverse = {"c": None, "b": (2.0,), "a": 1}
        assert encode_attrs(forward).encode() == encode_attrs(reverse).encode()

    def test_nested_mapping_insertion_order_is_byte_equal(self) -> None:
        forward = {"outer": {"x": 1, "y": 2}, "z": 3}
        reverse = {"z": 3, "outer": {"y": 2, "x": 1}}
        assert encode_value(forward).encode() == encode_value(reverse).encode()

    def test_repeated_encoding_is_stable(self) -> None:
        value = {"op": nl.add, "shape": (128, "n"), "engine": engine.vector}
        assert encode_value(value) == encode_value(value)


class TestInjectivity:
    def test_numeric_lookalikes_are_distinct(self) -> None:
        payloads = {encode_value(v) for v in (1, 1.0, True, "1")}
        assert len(payloads) == 4
        assert encode_value(0) != encode_value(False)

    def test_empty_lookalikes_are_distinct(self) -> None:
        payloads = {encode_value(v) for v in (None, "", (), {}, 0, False)}
        assert len(payloads) == 6

    def test_operation_value_differs_from_its_name(self) -> None:
        assert encode_value(nl.add) != encode_value("add")

    def test_enum_member_differs_from_name_and_other_classes(self) -> None:
        assert encode_value(engine.unknown) != encode_value("unknown")
        assert encode_value(engine.unknown) != encode_value(dge_mode.unknown)

    def test_sequence_framing_is_unambiguous(self) -> None:
        payloads = {
            encode_value(v)
            for v in (("ab", ""), ("a", "b"), ("", "ab"), ("a:b",), ("a", "b", ""))
        }
        assert len(payloads) == 5

    def test_mapping_framing_is_unambiguous(self) -> None:
        assert encode_value({"a": "b"}) != encode_value({"ab": ""})
        assert encode_value({"a": ("b",)}) != encode_value({"a": "b"})


class TestRejection:
    @pytest.mark.parametrize(
        "value",
        [object(), {1, 2}, frozenset(), b"bytes", 1 + 2j, math.nan, Ellipsis],
        ids=lambda v: type(v).__name__,
    )
    def test_unsupported_values_are_rejected(self, value: object) -> None:
        with pytest.raises(PayloadError):
            encode_value(value)

    def test_nan_inside_container_is_rejected(self) -> None:
        with pytest.raises(PayloadError):
            encode_value({"scale": math.nan})

    def test_unregistered_enum_is_rejected(self) -> None:
        class engine(enum.Enum):  # noqa: N801 - shadows the registered name
            tensor = 1

        with pytest.raises(PayloadError):
            encode_value(engine.tensor)

    def test_non_string_mapping_key_is_rejected(self) -> None:
        with pytest.raises(PayloadError):
            encode_value({1: "a"})

    def test_encode_attrs_requires_a_mapping(self) -> None:
        with pytest.raises(PayloadError):
            encode_attrs([("a", 1)])  # type: ignore[arg-type]


class TestMalformedPayloads:
    @pytest.mark.parametrize(
        "payload",
        [
            "",
            "X1",
            "N1",
            "B2",
            "B",
            "I",
            "I01",
            "I-0",
            "I1.5",
            "F",
            "Fnan",
            "F-0x0.0p+0",
            "F 0x1.0p+0",
            "T",
            "T-1:",
            "T01:1:N",
            "T2:1:N",
            "T1:1:NX",
            "T1:9:N",
            "T1:x:N",
            "M",
            "M1:1:a",
            "M1:01:a1:N",
            "M2:1:b1:N1:a1:N",
            "M2:1:a1:N1:a1:N",
            "M1:1:a1:N1:b1:N",
            "Eengine",
            "Eengine:warp",
            "Enosuch:tensor",
            "O",
            "O__class__",
        ],
    )
    def test_malformed_payloads_are_rejected(self, payload: str) -> None:
        with pytest.raises(PayloadError):
            decode_value(payload)

    def test_decode_attrs_rejects_non_mapping_payloads(self) -> None:
        for payload in ("N", "I3", encode_value((1, 2))):
            with pytest.raises(PayloadError):
                decode_attrs(payload)

    def test_decode_rejects_non_string_payloads(self) -> None:
        with pytest.raises(PayloadError):
            decode_value(b"N")  # type: ignore[arg-type]


class TestRegistry:
    def test_supported_type_tags_are_the_planned_set(self) -> None:
        assert (
            frozenset(
                {
                    "none",
                    "bool",
                    "int",
                    "float",
                    "str",
                    "sequence",
                    "mapping",
                    "enum",
                    "op",
                }
            )
            == SUPPORTED_TYPE_TAGS
        )

    def test_registered_enum_classes_match_attr_usage(self) -> None:
        assert set(SUPPORTED_ENUM_TYPES) == {
            "dge_mode",
            "engine",
            "matmul_perf_mode",
            "oob_mode",
            "reduce_cmd",
        }
