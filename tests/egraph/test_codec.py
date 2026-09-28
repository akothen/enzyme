"""Codec tests: aliases, generic languages, exact round trips, fail-closed."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
from egglog import expr_parts

from axon.egraph import isa_language as il
from axon.egraph import tensor_language as tl
from axon.egraph.adapter import EGraphAdapter
from axon.egraph.analysis import ensure_isa_semantics_registered
from axon.egraph.codec import (
    CodecError,
    decode_isa_enode,
    decode_tensor_enode,
    encode_isa_enode,
    encode_isa_input,
)
from axon.egraph.tensor import (
    canonical_tensor_parts,
    encode_tensor_node,
    ingest_tensor_graph,
    normalize_axes,
)
from axon.ir import Node, build_graph_from_kernel
from axon.isa_semantics import (
    dge_mode,
    engine,
    matmul_perf_mode,
    nl,
    oob_mode,
    reduce_cmd,
)

_CODEC_SOURCE = Path(inspect.getsourcefile(encode_isa_enode)).read_text()


def _node(op: str, inputs: list[str], attrs: dict | None = None) -> Node:
    return Node(id=f"{op}_0", op=op, inputs=inputs, attrs=dict(attrs or {}))


def _tensor_adapter() -> tuple[EGraphAdapter, dict[str, object]]:
    adapter = EGraphAdapter("t")
    handles: dict[str, object] = {}
    for name in ("a", "b"):
        node = _node("input", [], {"shape": (4, 8), "sym_shape": ("m", "k")})
        node.id = name
        handles[name] = adapter.intern_expr(
            encode_tensor_node(node, [], []), name
        ).handle
    return adapter, handles


class TestAliasNormalization:
    def test_multiply_and_divide_normalize(self) -> None:
        _, handles = _tensor_adapter()
        a, b = handles["a"], handles["b"]
        assert expr_parts(
            encode_tensor_node(_node("multiply", ["a", "b"]), [a, b], [2, 2])
        ) == expr_parts(encode_tensor_node(_node("mul", ["a", "b"]), [a, b], [2, 2]))
        assert expr_parts(
            encode_tensor_node(_node("divide", ["a", "b"]), [a, b], [2, 2])
        ) == expr_parts(encode_tensor_node(_node("div", ["a", "b"]), [a, b], [2, 2]))

    def test_keep_dims_spellings_normalize(self) -> None:
        _, handles = _tensor_adapter()
        a = handles["a"]
        left = encode_tensor_node(
            _node("reduce_sum", ["a"], {"axis": 1, "keep_dims": True}), [a], [2]
        )
        right = encode_tensor_node(
            _node("reduce_sum", ["a"], {"axis": 1, "keepdims": True}), [a], [2]
        )
        assert expr_parts(left) == expr_parts(right)

    def test_negative_axes_normalize(self) -> None:
        _, handles = _tensor_adapter()
        a = handles["a"]
        left = encode_tensor_node(_node("cumsum", ["a"], {"axis": -1}), [a], [2])
        right = encode_tensor_node(_node("cumsum", ["a"], {"axis": 1}), [a], [2])
        assert expr_parts(left) == expr_parts(right)
        assert normalize_axes(-2, 2) == (0,)
        with pytest.raises(CodecError):
            normalize_axes((0, -2), 2)  # duplicates after normalization

    def test_scalar_and_tensor_forms_are_distinct(self) -> None:
        _, handles = _tensor_adapter()
        a, b = handles["a"], handles["b"]
        tensor_form = encode_tensor_node(_node("mul", ["a", "b"]), [a, b], [2, 2])
        scalar_form = encode_tensor_node(_node("mul", ["a"], {"scalar": 2.0}), [a], [2])
        assert expr_parts(tensor_form) != expr_parts(scalar_form)


class TestTensorRoundTrip:
    def test_every_tensor_constructor_round_trips(self) -> None:
        def kernel(x, w):
            a = x + w
            b = (x - w) * 2.0
            c = (2.0 - x) / (w + 1.0)
            d = (a * b) @ (c / w).transpose()
            e = d.sum(axis=1, keep_dims=True)
            f = e.broadcast_like(d)
            g = f.sqrt().exp().relu().silu()
            return g.softmax(axis=-1).cumsum(axis=-1)

        G = build_graph_from_kernel(
            kernel, ("x", ("m", "m")), ("w", ("m", "m")), dim_sizes={"m": 8}
        )
        adapter = EGraphAdapter("t")
        ingest = ingest_tensor_graph(adapter, G)
        snap = adapter.freeze_snapshot()

        ranks = {
            node.id: len(node.shape or node.attrs.get("shape", ())) for node in G.nodes
        }
        decoded_ops: set[str] = set()
        for node in G.nodes:
            ref = adapter.resolve_handle(snap, ingest.node_handles[node.id])
            (row,) = snap.members(ref)
            decoded = decode_tensor_enode(snap, row)
            child_ranks = [ranks[inp] for inp in node.inputs]
            op, _attrs = canonical_tensor_parts(node, child_ranks)
            assert decoded.op == op
            decoded_ops.add(decoded.op)
            if node.op == "input":
                assert decoded.source_id == node.id
                assert decoded.input_shape == ("m", "m")
            else:
                assert len(decoded.child_classes) == len(node.inputs)

        assert {
            "input",
            "add",
            "subtract",
            "mul",
            "div",
            "matmul",
            "reduce_sum",
            "broadcast",
            "sqrt",
            "exp",
            "relu",
            "silu",
            "transpose",
            "softmax",
            "cumsum",
        } <= decoded_ops

    def test_exact_attributes_round_trip(self) -> None:
        adapter, handles = _tensor_adapter()
        a = handles["a"]
        cases = [
            (
                _node("mul", ["a"], {"scalar": 3, "reverse": True}),
                {"scalar": 3.0, "reverse": True},
            ),
            (
                _node("reduce_sum", ["a"], {"axis": -1, "keep_dims": False}),
                {"axis": (1,), "keep_dims": False},
            ),
            (_node("softmax", ["a"], {"axis": 0}), {"axis": 0}),
        ]
        for node, expected in cases:
            handle = adapter.intern_expr(
                encode_tensor_node(node, [a], [2]), node.op
            ).handle
            snap = adapter.freeze_snapshot()
            ref = adapter.resolve_handle(snap, handle)
            (row,) = snap.members(ref)
            assert decode_tensor_enode(snap, row).attrs == expected

    def test_scalar_subtract_and_divide_round_trip(self) -> None:
        adapter, handles = _tensor_adapter()
        a = handles["a"]
        for op, expected_reverse in [
            ("subtract", False),
            ("subtract", True),
            ("div", False),
            ("div", True),
        ]:
            node = _node(op, ["a"], {"scalar": 5.0, "reverse": expected_reverse})
            handle = adapter.intern_expr(
                encode_tensor_node(node, [a], [2]), f"{op}_r{expected_reverse}"
            ).handle
            snap = adapter.freeze_snapshot()
            ref = adapter.resolve_handle(snap, handle)
            (row,) = snap.members(ref)
            decoded = decode_tensor_enode(snap, row)
            assert decoded.op == op
            assert decoded.attrs["scalar"] == 5.0
            assert decoded.attrs["reverse"] == expected_reverse
            assert len(decoded.child_classes) == 1

    def test_source_identity_distinguishes_same_shape_inputs(self) -> None:
        adapter, handles = _tensor_adapter()
        snap = adapter.freeze_snapshot()
        assert adapter.resolve_handle(snap, handles["a"]) != adapter.resolve_handle(
            snap, handles["b"]
        )


class TestGenericLanguages:
    """Both languages are generic: dedicated inputs plus arity constructors."""

    def test_generic_language_has_only_generic_constructors_and_inputs(self) -> None:
        isa_constructors = {
            name
            for name, value in vars(il).items()
            if not name.startswith("_") and callable(value) and name.startswith("i_")
        }
        assert isa_constructors == {"i_input", "i_op1", "i_op2", "i_op3"}
        tensor_constructors = {
            name
            for name, value in vars(tl).items()
            if not name.startswith("_") and callable(value) and name.startswith("t_")
        }
        assert tensor_constructors == {"t_input", "t_op1", "t_op2"}
        # The only egglog sort each language declares is its result sort.
        for module, expected in ((il, {"IsaExpr"}), (tl, {"TensorExpr"})):
            sorts = {
                name
                for name, value in vars(module).items()
                if inspect.isclass(value) and value.__module__ == module.__name__
            }
            assert sorts == expected

    def test_no_operation_metadata_in_codec(self) -> None:
        # The plan's success criterion: no per-operation schema machinery or
        # operation names remain in codec.py.
        assert not re.search(
            r"class IsaCodecEntry|ISA_CODEC|_ISA_LANGUAGE_EGG_FNS|_ISA_SYM_EVAL",
            _CODEC_SOURCE,
        )
        assert not re.search(
            r'"(activation|activation_reduce|dma_copy|dma_transpose|exponential'
            r"|nc_matmul|nc_transpose|reciprocal|scalar_tensor_tensor|tensor_copy"
            r"|tensor_partition_reduce|tensor_reduce|tensor_scalar"
            r"|tensor_scalar_cumulative|tensor_tensor|broadcast|broadcast_to"
            r'|load|store)"',
            _CODEC_SOURCE,
        )
        assert not re.search(
            r"ISA_HW_OPS|ISA_POOL_OP_NAMES|ISA_PASSTHROUGH_OPS", _CODEC_SOURCE
        )


class TestIsaRoundTrip:
    def _isa_children(self, adapter: EGraphAdapter, count: int) -> list[object]:
        return [
            adapter.intern_expr(encode_isa_input(f"in{i}", ("m", "k")), f"in{i}").handle
            for i in range(count)
        ]

    # Canonical decoded-form attribute sets per instruction, as produced by
    # ``sketch_node_attrs``. Child count is the number of tensor inputs.
    _CASES: list[tuple[str, dict, int]] = [
        (
            "activation",
            {
                "op": nl.relu,
                "bias_const": None,
                "scale": 2.0,
                "reduce_op": None,
                "reduce_cmd": reduce_cmd.idle,
                "with_reduce": False,
            },
            1,
        ),
        (
            "activation",
            {
                "op": nl.copy,
                "bias_input_index": 1,
                "scale_input_index": 2,
                "reduce_op": nl.add,
                "reduce_cmd": reduce_cmd.reset_reduce,
                "with_reduce": True,
            },
            3,
        ),
        (
            "activation_reduce",
            {"op": nl.exp, "bias_const": None, "scale": 1.0, "reduce_op": nl.add},
            1,
        ),
        (
            "dma_copy",
            {
                "oob_mode": oob_mode.skip,
                "dge_mode": dge_mode.hwdge,
                "engine": engine.dma,
            },
            1,
        ),
        (
            "dma_transpose",
            {
                "axes": (1, 0),
                "dge_mode": dge_mode.unknown,
                "oob_mode": oob_mode.error,
            },
            1,
        ),
        (
            "exponential",
            {
                "max_value": 5.0,
                "reduce_cmd": reduce_cmd.idle,
                "reduce_init": 0.0,
                "with_reduce": False,
            },
            1,
        ),
        (
            "exponential",
            {
                "max_input_index": 1,
                "reduce_cmd": reduce_cmd.load_reduce,
                "reduce_init": 1.5,
                "with_reduce": True,
            },
            2,
        ),
        (
            "nc_matmul",
            {
                "is_stationary_onezero": True,
                "is_moving_onezero": False,
                "is_transpose": False,
                "accumulate": False,
                "perf_mode": matmul_perf_mode.double_row,
            },
            2,
        ),
        ("nc_transpose", {"engine": engine.vector}, 1),
        ("reciprocal", {}, 1),
        (
            "scalar_tensor_tensor",
            {
                "op0": nl.multiply,
                "op1": nl.add,
                "operand0_const": 0.5,
                "operand1_input_index": 1,
                "reverse0": True,
                "reverse1": False,
            },
            2,
        ),
        (
            "scalar_tensor_tensor",
            {
                "op0": nl.subtract,
                "op1": nl.maximum,
                "operand0_input_index": 1,
                "operand1_input_index": 2,
                "reverse0": False,
                "reverse1": True,
            },
            3,
        ),
        ("tensor_copy", {"engine": engine.gpsimd}, 1),
        ("tensor_partition_reduce", {"op": nl.add}, 1),
        (
            "tensor_reduce",
            {"op": nl.maximum, "axis": 1, "negate": True, "keepdims": True},
            1,
        ),
        (
            "tensor_scalar",
            {
                "op0": nl.multiply,
                "operand0_const": 2.0,
                "op1": None,
                "reverse0": False,
                "reverse1": False,
                "engine": engine.unknown,
            },
            1,
        ),
        (
            "tensor_scalar",
            {
                "op0": nl.add,
                "operand0_input_index": 1,
                "op1": nl.minimum,
                "operand1_const": 7.0,
                "reverse0": True,
                "reverse1": True,
                "engine": engine.scalar,
            },
            2,
        ),
        (
            "tensor_scalar_cumulative",
            {
                "op0": nl.multiply,
                "op1": nl.add,
                "imm0_const": 1.0,
                "reduce_cmd": reduce_cmd.reset_reduce,
                "reverse0": False,
                "reverse1": False,
            },
            1,
        ),
        ("tensor_tensor", {"op": nl.divide, "engine": engine.tensor}, 2),
        ("broadcast", {}, 2),
        ("broadcast_to", {"shape": ("m", 4)}, 1),
        ("load", {}, 1),
        ("store", {"out_shape": ("m", "k")}, 1),
    ]

    def test_every_arity_round_trips_exactly(self) -> None:
        ensure_isa_semantics_registered()
        adapter = EGraphAdapter("i")
        arities: set[int] = set()
        for op, attrs, child_count in self._CASES:
            children = self._isa_children(adapter, child_count)
            handle = adapter.intern_expr(
                encode_isa_enode(op, dict(attrs), children), f"{op}_case"
            ).handle
            snap = adapter.freeze_snapshot()
            ref = adapter.resolve_handle(snap, handle)
            rows = [r for r in snap.members(ref) if r.egg_fn != "axIInput"]
            (row,) = rows
            decoded = decode_isa_enode(snap, row)
            assert decoded.op == op
            arities.add(child_count)
            assert len(decoded.child_classes) == child_count
            # Distinct ISA input children must stay distinct and ordered, so
            # operand-index attributes keep addressing the right child.
            assert len(set(decoded.child_classes)) == child_count, (
                f"{op}: children collapsed {decoded.child_classes}"
            )
            assert decoded.attrs == attrs, f"{op}: {decoded.attrs} != {attrs}"
        assert arities == {1, 2, 3}

    def test_distinct_attrs_are_distinct_enodes(self) -> None:
        ensure_isa_semantics_registered()
        adapter = EGraphAdapter("i")
        (child,) = self._isa_children(adapter, 1)
        left = encode_isa_enode("nc_transpose", {"engine": engine.vector}, [child])
        right = encode_isa_enode("nc_transpose", {"engine": engine.unknown}, [child])
        assert expr_parts(left) != expr_parts(right)
        same = encode_isa_enode("nc_transpose", {"engine": engine.vector}, [child])
        assert expr_parts(left) == expr_parts(same)

    def test_attr_insertion_order_never_affects_identity(self) -> None:
        ensure_isa_semantics_registered()
        adapter = EGraphAdapter("i")
        children = self._isa_children(adapter, 2)
        forward = encode_isa_enode(
            "tensor_tensor", {"op": nl.add, "engine": engine.unknown}, children
        )
        reversed_attrs = encode_isa_enode(
            "tensor_tensor", {"engine": engine.unknown, "op": nl.add}, children
        )
        assert expr_parts(forward) == expr_parts(reversed_attrs)

    def test_isa_input_round_trips(self) -> None:
        adapter = EGraphAdapter("i")
        handle = adapter.intern_expr(encode_isa_input("x", ("m", 4)), "x").handle
        snap = adapter.freeze_snapshot()
        (row,) = snap.members(adapter.resolve_handle(snap, handle))
        decoded = decode_isa_enode(snap, row)
        assert decoded.op == "input"
        assert decoded.source_id == "x"
        assert decoded.input_shape == ("m", 4)


class TestFailClosed:
    def test_unknown_tensor_operation_rejected(self) -> None:
        with pytest.raises(CodecError):
            canonical_tensor_parts(_node("rms_norm", ["a"]), [2])

    def test_undeclared_attribute_rejected(self) -> None:
        with pytest.raises(CodecError):
            canonical_tensor_parts(
                _node("mul", ["a", "b"], {"mystery_attribute": 1}), [2, 2]
            )

    def test_two_input_scalar_form_rejected(self) -> None:
        with pytest.raises(CodecError):
            canonical_tensor_parts(_node("mul", ["a", "b"], {"scalar": 2.0}), [2, 2])

    def test_unknown_isa_operation_rejected(self) -> None:
        with pytest.raises(CodecError):
            encode_isa_enode("definitely_not_an_op", {}, [object()])

    def test_unregistered_semantics_rejected(self) -> None:
        # Decorated but never registered operations stay outside the language.
        with pytest.raises(CodecError):
            encode_isa_enode("iota", {}, [object()])

    def test_unsupported_attribute_value_rejected(self) -> None:
        ensure_isa_semantics_registered()
        adapter = EGraphAdapter("i")
        child = adapter.intern_expr(encode_isa_input("x", ("m", 4)), "x").handle
        with pytest.raises(CodecError):
            encode_isa_enode("tensor_tensor", {"op": object()}, [child, child])

    def test_undeclared_arity_rejected(self) -> None:
        ensure_isa_semantics_registered()
        adapter = EGraphAdapter("i")
        child = adapter.intern_expr(encode_isa_input("x", ("m", 4)), "x").handle
        with pytest.raises(CodecError):
            encode_isa_enode("tensor_tensor", {"op": nl.add}, [child] * 4)
        with pytest.raises(CodecError):
            encode_isa_enode("tensor_tensor", {"op": nl.add}, [])
