"""Adapter tests: public freeze decoding, handles, unions, and schema checks."""

from __future__ import annotations

import pytest

from axon.egraph import adapter as adapter_mod
from axon.egraph.adapter import (
    EClassRef,
    EGraphAdapter,
    EGraphAdapterError,
    LitArg,
    compute_minimum_heights,
    minimum_height_witness,
    reachable_classes,
)
from axon.egraph.codec import encode_shape
from axon.egraph.payload import encode_attrs
from axon.egraph.tensor_language import t_input, t_op1, t_op2

_NO_ATTRS = encode_attrs({})


def t_relu(x: object) -> object:
    return t_op1("relu", _NO_ATTRS, x)


def t_add(x: object, y: object) -> object:
    return t_op2("add", _NO_ATTRS, x, y)


def t_mul_scalar(x: object, scalar: float, reverse: bool) -> object:
    return t_op1("mul", encode_attrs({"scalar": scalar, "reverse": reverse}), x)


def _adapter_with_inputs() -> tuple[EGraphAdapter, object, object]:
    adapter = EGraphAdapter("t")
    x = adapter.intern_expr(t_input("x", encode_shape(("m", 4))), "x").handle
    y = adapter.intern_expr(t_input("y", encode_shape(("m", 4))), "y").handle
    return adapter, x, y


class TestFreezeDecoding:
    def test_rows_literals_and_child_classes(self) -> None:
        adapter, x, _ = _adapter_with_inputs()
        scaled = adapter.intern_expr(t_mul_scalar(x, 2.0, False), "scaled").handle
        snap = adapter.freeze_snapshot()

        ref = adapter.resolve_handle(snap, scaled)
        assert ref.sort == "TensorExpr"
        rows = snap.members(ref)
        assert len(rows) == 1
        row = rows[0]
        assert row.egg_fn == "axTOp1"
        assert row.args[0] == LitArg("mul")
        assert row.args[1] == LitArg(encode_attrs({"scalar": 2.0, "reverse": False}))
        assert isinstance(row.args[2], EClassRef)

    def test_auxiliary_value_sorts_are_decoded(self) -> None:
        adapter, x, _ = _adapter_with_inputs()
        snap = adapter.freeze_snapshot()
        ref = adapter.resolve_handle(snap, x)
        (row,) = snap.members(ref)
        assert row.egg_fn == "axTInput"
        assert row.args[0] == LitArg("x")
        shape_ref = row.args[1]
        assert isinstance(shape_ref, EClassRef)
        assert shape_ref.sort == "Shape"
        (shape_row,) = snap.members(shape_ref)
        assert shape_row.egg_fn == "axShapeCons"

    def test_literal_types_are_exact(self) -> None:
        adapter, x, _ = _adapter_with_inputs()
        _ = adapter.intern_expr(t_mul_scalar(x, 1.0, False), "one").handle
        snap = adapter.freeze_snapshot()
        lits = [
            arg
            for rows in snap.classes.values()
            for row in rows
            for arg in row.args
            if isinstance(arg, LitArg)
        ]
        int_four = [a for a in lits if a.value == 4]
        assert int_four, "the i64 literal 4 must decode as an int"
        assert isinstance(int_four[0].value, int)
        assert not isinstance(int_four[0].value, bool)
        strings = [a for a in lits if isinstance(a.value, str)]
        assert any(a.value == "m" for a in strings)


class TestHandlesAndUnions:
    def test_handle_reresolution_after_union(self) -> None:
        adapter, x, y = _adapter_with_inputs()
        before = adapter.freeze_snapshot()
        assert adapter.resolve_handle(before, x) != adapter.resolve_handle(before, y)

        assert adapter.union_if_distinct(x, y) is True
        assert adapter.union_if_distinct(x, y) is False

        after = adapter.freeze_snapshot()
        assert adapter.resolve_handle(after, x) == adapter.resolve_handle(after, y)

    def test_congruence_merges_equal_applications(self) -> None:
        adapter, x, y = _adapter_with_inputs()
        rx = adapter.intern_expr(t_relu(x), "rx").handle
        ry = adapter.intern_expr(t_relu(y), "ry").handle
        adapter.union_if_distinct(x, y)
        snap = adapter.freeze_snapshot()
        assert adapter.resolve_handle(snap, rx) == adapter.resolve_handle(snap, ry)

    def test_duplicate_intern_adds_no_second_row(self) -> None:
        adapter, x, y = _adapter_with_inputs()
        assert adapter.enode_exists(adapter.freeze_snapshot(), t_add(x, y)) is False
        first = adapter.intern_expr(t_add(x, y), "sum")
        assert adapter.enode_exists(adapter.freeze_snapshot(), t_add(x, y)) is True
        second = adapter.intern_expr(t_add(x, y), "sum_again")
        snap = adapter.freeze_snapshot()
        assert len(snap.members(adapter.resolve_handle(snap, first.handle))) == 1
        assert adapter.resolve_handle(snap, first.handle) == adapter.resolve_handle(
            snap, second.handle
        )

    def test_enode_exists_tracks_unions(self) -> None:
        adapter, x, y = _adapter_with_inputs()
        adapter.intern_expr(t_relu(x), "rx")
        snap = adapter.freeze_snapshot()
        assert adapter.enode_exists(snap, t_relu(x)) is True
        assert adapter.enode_exists(snap, t_relu(y)) is False
        # After the union, relu over y's class is the same row as relu over x.
        adapter.union_if_distinct(x, y)
        snap = adapter.freeze_snapshot()
        assert adapter.enode_exists(snap, t_relu(y)) is True

    def test_resolve_handle_requires_let_reference(self) -> None:
        adapter, x, _ = _adapter_with_inputs()
        snap = adapter.freeze_snapshot()
        with pytest.raises(EGraphAdapterError):
            adapter.resolve_handle(snap, t_relu(x))

    def test_cross_snapshot_refs_are_rejected(self) -> None:
        adapter, x, _ = _adapter_with_inputs()
        first = adapter.freeze_snapshot()
        ref = adapter.resolve_handle(first, x)
        second = adapter.freeze_snapshot()
        with pytest.raises(EGraphAdapterError):
            second.members(ref)

    def test_cross_adapter_refs_and_snapshots_are_rejected(self) -> None:
        first_adapter, first_x, _ = _adapter_with_inputs()
        second_adapter, second_x, _ = _adapter_with_inputs()
        first = first_adapter.freeze_snapshot()
        second = second_adapter.freeze_snapshot()
        first_ref = first_adapter.resolve_handle(first, first_x)
        second_ref = second_adapter.resolve_handle(second, second_x)

        assert first_ref != second_ref
        with pytest.raises(EGraphAdapterError):
            second.members(first_ref)
        with pytest.raises(EGraphAdapterError):
            first_adapter.resolve_handle(second, first_x)
        with pytest.raises(EGraphAdapterError):
            first_adapter.resolve_handle(first, second_x)
        with pytest.raises(EGraphAdapterError):
            first_adapter.union_if_distinct(first_x, second_x)
        with pytest.raises(EGraphAdapterError):
            first_adapter.intern_expr(t_relu(second_x), "foreign")


class TestTraversalAndWitnesses:
    def test_reachable_classes_ignores_unreachable_rows(self) -> None:
        adapter, x, y = _adapter_with_inputs()
        reached = adapter.intern_expr(t_relu(x), "rx").handle
        adapter.intern_expr(t_relu(y), "ry")
        snap = adapter.freeze_snapshot()
        root = adapter.resolve_handle(snap, reached)
        reach = reachable_classes(snap, [root])
        y_ref = adapter.resolve_handle(snap, y)
        assert root in reach
        assert adapter.resolve_handle(snap, x) in reach
        assert y_ref not in reach

    def test_minimum_height_witness_handles_cycles(self) -> None:
        adapter, x, _ = _adapter_with_inputs()
        wrapped = adapter.intern_expr(t_relu(x), "rx").handle
        # relu(x) == x makes x's class self-reachable: relu(cls) is a member
        # of cls itself.
        adapter.union_if_distinct(wrapped, x)
        snap = adapter.freeze_snapshot()
        ref = adapter.resolve_handle(snap, x)
        rows = {row.egg_fn for row in snap.members(ref)}
        assert rows == {"axTInput", "axTOp1"}

        heights = compute_minimum_heights(snap)
        witness = minimum_height_witness(snap, ref, heights)
        assert witness.enode.egg_fn == "axTInput"

    def test_uninhabited_class_has_no_witness(self) -> None:
        adapter, x, _ = _adapter_with_inputs()
        wrapped = adapter.intern_expr(t_relu(x), "rx").handle
        adapter.union_if_distinct(wrapped, x)
        snap = adapter.freeze_snapshot()
        heights = compute_minimum_heights(snap)
        # Every class in this snapshot is inhabited; fabricate an absent one.
        ghost = EClassRef(snap.snapshot_id, object(), "TensorExpr")
        with pytest.raises(EGraphAdapterError):
            minimum_height_witness(snap, ghost, heights)


class TestSchemaRejection:
    def test_missing_decl_fields_are_rejected_with_version(self) -> None:
        original = adapter_mod._REQUIRED_DECL_FIELDS
        adapter_mod._REQUIRED_DECL_FIELDS = frozenset({"definitely_not_a_field"})
        try:
            with pytest.raises(EGraphAdapterError) as excinfo:
                adapter_mod._import_decl_types()
        finally:
            adapter_mod._REQUIRED_DECL_FIELDS = original
        message = str(excinfo.value)
        assert "unsupported egglog" in message
        assert "installed egglog version" in message
