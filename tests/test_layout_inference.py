"""Layout inference for the matmul-family emitter (codegen/layout.py).

Layouts state which MATH dim of a value sits on SBUF partitions vs the free
axis. Dims are union-find variables: ops impose equalities (nc_matmul unifies
its operands' partition dims = the contraction), so two axes are 'the same
dim' iff inference united them. This is the typed replacement for the old
body's single hardcoded namespace."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import pytest

import axon.codegen.layout as layout_module
from axon.codegen.layout import (
    DimVar,
    Layout,
    LayoutError,
    analyze_dims,
    infer_layouts,
    topo_order,
)
from axon.ir import Node


def test_dimvar_union_find_unites_representatives():
    a, b, c = DimVar("a"), DimVar("b"), DimVar("c")
    a.unite(b)
    assert a.find() is b.find()
    assert a.find() is not c.find()
    b.unite(c)
    assert a.find() is c.find()


def test_dimvar_fork_shares_extent_without_sharing_schedule_coordinate():
    source = DimVar("source")
    forked = source.fork("forked")
    assert source.find() is not forked.find()
    assert source.same_extent(forked)


def test_square_layout_differs_from_its_swapped_orientation():
    part = DimVar("part")
    free = part.fork("free", fresh_role=True)
    layout = Layout(part=part, free=free)
    assert part.same_extent(free)
    assert not part.same_role(free)
    assert not layout.same(layout.swapped())


def test_layout_equality_is_by_representative():
    a, b = DimVar("a"), DimVar("b")
    l1, l2 = Layout(part=a, free=None), Layout(part=b, free=None)
    assert not l1.same(l2)
    a.unite(b)
    assert l1.same(l2)


def test_layout_swap():
    a, b = DimVar("a"), DimVar("b")
    layout = Layout(part=a, free=b)
    s = layout.swapped()
    assert s.part.find() is b.find() and s.free.find() is a.find()


def test_swapping_scalar_layout_refuses():
    a = DimVar("a")
    with pytest.raises(LayoutError):
        Layout(part=a, free=None).swapped()


def _graph(nodes):
    id_to_node = {n.id: n for n in nodes}
    inputs = {n.id for n in nodes if n.op == "input"}
    compute = [n for n in nodes if n.op != "input"]
    return compute, id_to_node, inputs


def test_infer_plain_matmul():
    # nc_matmul(stat=x^T, mov=w): x^T part=k, w part=k -> unified; out (m, n)
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node("w", "input", []),
            Node("xt", "nc_transpose", ["x"]),
            Node("mm", "nc_matmul", ["xt", "w"]),
        ]
    )
    L = infer_layouts(compute, id2n, inputs)
    # Contraction extents align without merging the producer coordinates.
    assert L["xt"].part.find() is not L["w"].part.find()
    assert L["xt"].part.same_extent(L["w"].part)
    # The output owns fresh coordinates with the operand extents.
    assert L["mm"].part.find() is not L["x"].part.find()
    assert L["mm"].part.same_extent(L["x"].part)
    assert L["mm"].free.find() is not L["w"].free.find()
    assert L["mm"].free.same_extent(L["w"].free)


def test_infer_elementwise_unifies_operand_axes():
    compute, id2n, inputs = _graph(
        [
            Node("a", "input", [], {"shape": (8, 8)}),
            Node("b", "input", [], {"shape": (8, 8)}),
            Node("tt", "tensor_tensor", ["a", "b"], {"op": "multiply"}),
        ]
    )
    L = infer_layouts(compute, id2n, inputs)
    assert L["a"].same(L["b"]) and L["tt"].same(L["a"])


def test_matching_symbolic_and_concrete_metadata_is_accepted():
    compute, id2n, inputs = _graph(
        [
            Node(
                "a",
                "input",
                [],
                {"sym_shape": ("m", "k"), "shape": (8, 16)},
                shape=(8, 16),
            ),
            Node(
                "b",
                "input",
                [],
                {"sym_shape": ("m", "k"), "shape": (8, 16)},
            ),
            Node("tt", "tensor_tensor", ["a", "b"], {"op": "multiply"}),
        ]
    )
    layouts = infer_layouts(compute, id2n, inputs)
    assert layouts["a"].part.same_extent(layouts["b"].part)
    assert layouts["a"].free.same_extent(layouts["b"].free)


def test_matmul_accepts_distinct_symbols_bound_to_equal_concrete_extents():
    compute, id2n, inputs = _graph(
        [
            Node(
                "stationary",
                "input",
                [],
                {"sym_shape": ("k_left", "m"), "shape": (64, 32)},
            ),
            Node(
                "moving",
                "input",
                [],
                {"sym_shape": ("k_right", "n"), "shape": (64, 16)},
            ),
            Node("mm", "nc_matmul", ["stationary", "moving"], shape=(32, 16)),
        ]
    )

    layouts = infer_layouts(compute, id2n, inputs)

    assert layouts["stationary"].part.same_extent(layouts["moving"].part)


def test_extent_identity_is_normalized_integer_or_symbol_equality():
    """The public extent identity: equal ints, or equal symbol names."""
    assert layout_module._extent_key(8) == layout_module._extent_key(8)
    assert layout_module._extent_key(8) != layout_module._extent_key(16)
    assert layout_module._extent_key("k") == layout_module._extent_key("k")
    assert layout_module._extent_key("k") != layout_module._extent_key("m")
    assert layout_module._extent_key("8") != layout_module._extent_key(8)


def test_infer_refuses_unequal_concrete_contraction_extents():
    compute, id2n, inputs = _graph(
        [
            Node("stationary", "input", [], {"shape": (64, 32)}),
            Node("moving", "input", [], {"shape": (63, 16)}),
            Node("mm", "nc_matmul", ["stationary", "moving"]),
        ]
    )
    with pytest.raises(LayoutError, match="operand extents disagree"):
        infer_layouts(compute, id2n, inputs)


def test_infer_square_tensor_plus_its_transpose_refuses_role_collapse():
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", [], {"shape": (8, 8)}),
            Node("xt", "nc_transpose", ["x"]),
            Node("tt", "tensor_tensor", ["x", "xt"], {"op": "add"}),
        ]
    )
    with pytest.raises(LayoutError, match="roles"):
        infer_layouts(compute, id2n, inputs)


def test_infer_reduce_gives_scalar_layout_and_scalar_broadcasts():
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node("red", "tensor_reduce", ["x"], {"op": "add", "keepdims": True}),
            Node(
                "ts",
                "tensor_scalar",
                ["x", "red"],
                {"op0": "multiply", "operand0_input_index": 1},
            ),
        ]
    )
    L = infer_layouts(compute, id2n, inputs)
    assert L["red"].free is None
    assert L["red"].part.find() is L["x"].part.find()
    assert L["ts"].same(L["x"])  # scalar operand broadcasts, no unification of free


def test_infer_mixed_mlp_v1_graph():
    # The v1 graph: mm1 straight, mm2 transposed, combine after nc_transpose.
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node("w1", "input", []),
            Node("w2", "input", []),
            Node("w3", "input", []),
            Node("xt1", "nc_transpose", ["x"]),
            Node("mm1", "nc_matmul", ["xt1", "w1"]),
            Node("xt2", "nc_transpose", ["x"]),
            Node("mm2", "nc_matmul", ["w2", "xt2"]),
            Node("act", "activation", ["mm1"], {"op": "relu"}),
            Node("tr", "nc_transpose", ["act"]),
            Node("tt", "tensor_tensor", ["tr", "mm2"], {"op": "multiply"}),
            Node("mm3", "nc_matmul", ["tt", "w3"]),
        ]
    )
    L = infer_layouts(compute, id2n, inputs)
    # act is m-major, mm2 is n-major, and the graph's transpose reconciles them
    assert not L["act"].same(L["mm2"])
    assert L["tr"].same(L["mm2"])
    # chained mm consumes tt with part = contraction dim of mm3 = w3's rows
    assert L["mm3"].part.same_extent(L["act"].part)


def test_infer_refuses_elementwise_layout_mismatch():
    # combine WITHOUT the reconciling transpose: must refuse, not guess
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node("w1", "input", []),
            Node("w2", "input", []),
            Node("xt1", "nc_transpose", ["x"]),
            Node("mm1", "nc_matmul", ["xt1", "w1"]),
            Node("xt2", "nc_transpose", ["x"]),
            Node("mm2", "nc_matmul", ["w2", "xt2"]),
            Node("tt", "tensor_tensor", ["mm1", "mm2"], {"op": "multiply"}),
        ]
    )
    with pytest.raises(LayoutError, match="tt"):
        infer_layouts(compute, id2n, inputs)


def test_infer_exponential_inherits_layout():
    # exponential is single-input elementwise — output layout equals input layout.
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node("exp", "exponential", ["x"]),
        ]
    )
    L = infer_layouts(compute, id2n, inputs)
    assert L["exp"].part.find() is L["x"].part.find()
    assert L["exp"].free is not None
    assert L["exp"].free.find() is L["x"].free.find()


def test_infer_reciprocal_on_partition_scalar_stays_scalar():
    # per-partition scalar input (free=None) → reciprocal output is also scalar.
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node("red", "tensor_reduce", ["x"], {"op": "add", "keepdims": True}),
            Node("rcp", "reciprocal", ["red"]),
        ]
    )
    L = infer_layouts(compute, id2n, inputs)
    assert L["red"].free is None
    assert L["rcp"].free is None
    assert L["rcp"].part.find() is L["red"].part.find()


def test_infer_constant_tensor_scalar_reduce_post_chain_stays_scalar():
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", [], shape=(8, 16)),
            Node("w", "input", [], shape=(16, 32)),
            Node(
                "red",
                "tensor_reduce",
                ["x"],
                {"op": "add", "axis": 1, "keepdims": True},
                shape=(8, 1),
            ),
            Node(
                "scaled",
                "tensor_scalar",
                ["red"],
                {"op0": "multiply", "operand0_const": 1.0 / 16},
                shape=(8, 1),
            ),
            Node(
                "shifted",
                "tensor_scalar",
                ["scaled"],
                {"op0": "add", "operand0_const": 1e-6},
                shape=(8, 1),
            ),
            Node("root", "activation", ["shifted"], {"op": "sqrt"}, shape=(8, 1)),
            Node("inv", "reciprocal", ["root"], shape=(8, 1)),
            Node(
                "normalized",
                "tensor_scalar",
                ["x", "inv"],
                {"op0": "multiply", "operand0_input_index": 1},
                shape=(8, 16),
            ),
            Node("normalized_t", "nc_transpose", ["normalized"], shape=(16, 8)),
            Node("out", "nc_matmul", ["normalized_t", "w"], shape=(8, 32)),
        ]
    )

    layouts = infer_layouts(compute, id2n, inputs)

    for node_id in ("red", "scaled", "shifted", "root", "inv"):
        assert layouts[node_id].free is None
        assert layouts[node_id].part.find() is layouts["x"].part.find()
    assert layouts["normalized"].same(layouts["x"])


@pytest.mark.parametrize(
    "attrs",
    [
        {"op0": "multiply", "operand0_input_index": 1},
        {
            "op0": "multiply",
            "operand0_const": 0.5,
            "op1": "add",
            "operand1_const": 1.0,
        },
        {"op0": None, "operand0_const": 0.5},
        {"op0": "multiply", "operand0_const": None},
    ],
)
def test_infer_refuses_malformed_tensor_scalar_on_reduce_scalar(attrs):
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", [], shape=(8, 16)),
            Node(
                "red",
                "tensor_reduce",
                ["x"],
                {"op": "add", "axis": 1, "keepdims": True},
                shape=(8, 1),
            ),
            Node("bad", "tensor_scalar", ["red"], attrs, shape=(8, 1)),
        ]
    )

    with pytest.raises(LayoutError, match="no full-tile operand"):
        infer_layouts(compute, id2n, inputs)


def test_infer_refuses_constant_tensor_scalar_with_full_tile_output_shape():
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", [], shape=(8, 16)),
            Node(
                "red",
                "tensor_reduce",
                ["x"],
                {"op": "add", "axis": 1, "keepdims": True},
                shape=(8, 1),
            ),
            Node(
                "bad",
                "tensor_scalar",
                ["red"],
                {"op0": "multiply", "operand0_const": 0.5},
                shape=(8, 16),
            ),
        ]
    )

    with pytest.raises(LayoutError, match="singleton free dimension"):
        infer_layouts(compute, id2n, inputs)


def test_infer_refuses_tensor_scalar_fan_in_of_two_reduce_scalars():
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", [], shape=(8, 16)),
            Node(
                "left",
                "tensor_reduce",
                ["x"],
                {"op": "add", "axis": 1, "keepdims": True},
                shape=(8, 1),
            ),
            Node(
                "right",
                "tensor_reduce",
                ["x"],
                {"op": "maximum", "axis": 1, "keepdims": True},
                shape=(8, 1),
            ),
            Node(
                "bad",
                "tensor_scalar",
                ["left", "right"],
                {"op0": "multiply", "operand0_input_index": 1},
                shape=(8, 1),
            ),
        ]
    )

    with pytest.raises(LayoutError, match="no full-tile operand"):
        infer_layouts(compute, id2n, inputs)


def test_infer_activation_reduce_gives_scalar_layout():
    # activation_reduce reduces the free axis → output is a per-partition scalar.
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node("ar", "activation_reduce", ["x"], {"op": "exp", "reduce_op": "add"}),
        ]
    )
    L = infer_layouts(compute, id2n, inputs)
    assert L["ar"].free is None
    assert L["ar"].part.find() is L["x"].part.find()
    assert L["ar"].part.same_extent(L["x"].part)


@pytest.mark.parametrize("aux_shape", [(8, 1), (1, 16), (1, 1)])
def test_activation_reduce_aligns_broadcast_auxiliary_roles_without_extents(
    aux_shape,
):
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", [], {"shape": (8, 16)}),
            Node("aux", "input", [], {"shape": aux_shape}),
            Node("ar", "activation_reduce", ["x", "aux"]),
        ]
    )
    layouts = infer_layouts(compute, id2n, inputs)
    for data_axis, auxiliary_axis, auxiliary_extent in zip(
        (layouts["x"].part, layouts["x"].free),
        (layouts["aux"].part, layouts["aux"].free),
        aux_shape,
        strict=True,
    ):
        assert data_axis.same_role(auxiliary_axis)
        assert data_axis.same_extent(auxiliary_axis) is (auxiliary_extent != 1)


def test_activation_reduce_rejects_non_broadcast_auxiliary_extent():
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", [], {"shape": (8, 16)}),
            Node("aux", "input", [], {"shape": (8, 7)}),
            Node("ar", "activation_reduce", ["x", "aux"]),
        ]
    )
    with pytest.raises(LayoutError, match="operand extents disagree"):
        infer_layouts(compute, id2n, inputs)


def test_activation_reduce_rejects_square_transposed_auxiliary():
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", [], {"shape": (8, 8)}),
            Node("xt", "nc_transpose", ["x"]),
            Node("ar", "activation_reduce", ["x", "xt"]),
        ]
    )
    with pytest.raises(LayoutError, match="roles"):
        infer_layouts(compute, id2n, inputs)


def _self_attention_nodes():
    return [
        Node("x", "input", [], {"sym_shape": ("m", "k")}),
        Node("wq", "input", [], {"sym_shape": ("k", "n")}),
        Node("wk", "input", [], {"sym_shape": ("k", "n")}),
        Node("wv", "input", [], {"sym_shape": ("k", "n")}),
        Node("xt", "nc_transpose", ["x"]),
        Node("q", "nc_matmul", ["xt", "wq"]),
        Node("k", "nc_matmul", ["xt", "wk"]),
        Node("v", "nc_matmul", ["xt", "wv"]),
        Node("qt", "nc_transpose", ["q"]),
        Node("kt", "nc_transpose", ["k"]),
        Node("scores", "nc_matmul", ["qt", "kt"]),
        Node("ex", "exponential", ["scores"]),
        Node(
            "sum",
            "tensor_reduce",
            ["ex"],
            {"op": "add", "axis": 1, "keepdims": True},
        ),
        Node("inv", "reciprocal", ["sum"]),
        Node(
            "probs",
            "tensor_scalar",
            ["ex", "inv"],
            {"op0": "multiply", "operand0_input_index": 1},
        ),
        Node("probst", "nc_transpose", ["probs"]),
        Node("out", "nc_matmul", ["probst", "v"]),
    ]


def test_self_attention_scores_own_two_occurrence_coordinates():
    compute, id2n, inputs = _graph(_self_attention_nodes())
    layouts = infer_layouts(compute, id2n, inputs)
    assert layouts["q"].part.find() is not layouts["k"].part.find()
    assert layouts["q"].part.same_extent(layouts["k"].part)
    assert layouts["scores"].part.find() is not layouts["scores"].free.find()
    assert layouts["scores"].part.same_extent(layouts["scores"].free)
    assert not layouts["scores"].part.same_role(layouts["scores"].free)
    assert not layouts["scores"].same(layouts["scores"].swapped())


def test_output_contraction_does_not_rewrite_score_dimensions():
    compute, id2n, inputs = _graph(_self_attention_nodes())
    info = analyze_dims(compute, id2n, inputs, output_id="out")
    score_dims = info.dims["scores"]
    assert len(score_dims) == 2
    assert len({info.canon[dim] for dim in score_dims}) == 2
    output_contraction = next(dim for mm_id, dim in info.contractions if mm_id == "out")
    assert output_contraction.find() in score_dims


def test_rectangular_partition_scalar_still_rejects_wrong_axis():
    compute, id2n, inputs = _graph(
        [
            Node("full", "input", [], {"sym_shape": ("m", "n")}),
            Node("source", "input", [], {"sym_shape": ("n", "k")}),
            Node(
                "scalar",
                "tensor_reduce",
                ["source"],
                {"op": "add", "axis": 1, "keepdims": True},
            ),
            Node(
                "scaled",
                "tensor_scalar",
                ["full", "scalar"],
                {"op0": "multiply", "operand0_input_index": 1},
            ),
        ]
    )
    with pytest.raises(LayoutError, match="wrong axis"):
        infer_layouts(compute, id2n, inputs)


def test_infer_tensor_copy_aliases_layout():
    # tensor_copy is a layout alias of its input.
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node("cp", "tensor_copy", ["x"]),
        ]
    )
    L = infer_layouts(compute, id2n, inputs)
    assert L["cp"].part.find() is L["x"].part.find()
    assert L["cp"].free is not None
    assert L["cp"].free.find() is L["x"].free.find()


def test_infer_broadcast_is_a_copy_not_alias():
    # broadcast should produce a Layout copy, not an alias of the same object.
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node("bc", "broadcast", ["x"]),
        ]
    )
    L = infer_layouts(compute, id2n, inputs)
    assert L["bc"] is not L["x"]
    assert L["bc"].part.find() is L["x"].part.find()
    assert L["bc"].free is not None
    assert L["bc"].free.find() is L["x"].free.find()


def test_dims_and_canon_for_rmsnorm_shape():
    # x:(m,k); red=sum(x^2) over k -> dims {m}; scale -> {m,k}; mm -> {m,n}
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node("w", "input", []),
            Node("sq", "tensor_tensor", ["x", "x"], {"op": "multiply"}),
            Node("red", "tensor_reduce", ["sq"], {"op": "add", "keepdims": True}),
            Node("rs", "activation", ["red"], {"op": "rsqrt"}),
            Node(
                "sc",
                "tensor_scalar",
                ["x", "rs"],
                {"op0": "multiply", "operand0_input_index": 1},
            ),
            Node("sct", "nc_transpose", ["sc"]),
            Node("mm", "nc_matmul", ["sct", "w"]),
        ]
    )
    info = analyze_dims(compute, id2n, inputs, output_id="mm")

    def names(nid):
        return {info.canon[d] for d in info.dims[nid]}

    assert names("red") == {"m"}
    assert names("sc") == {"m", "k"}
    assert names("mm") == {"m", "n"}
    assert [info.canon[c] for _, c in info.contractions] == ["k"]


def test_dims_for_chained_mlp_adds_p():
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node("w1", "input", []),
            Node("w2", "input", []),
            Node("w3", "input", []),
            Node("xt1", "nc_transpose", ["x"]),
            Node("mm1", "nc_matmul", ["xt1", "w1"]),
            Node("xt2", "nc_transpose", ["x"]),
            Node("mm2", "nc_matmul", ["w2", "xt2"]),
            Node("act", "activation", ["mm1"], {"op": "relu"}),
            Node("tr", "nc_transpose", ["act"]),
            Node("tt", "tensor_tensor", ["tr", "mm2"], {"op": "multiply"}),
            Node("mm3", "nc_matmul", ["tt", "w3"]),
        ]
    )
    info = analyze_dims(compute, id2n, inputs, output_id="mm3")

    def names(nid):
        return {info.canon[d] for d in info.dims[nid]}

    assert names("mm3") == {"m", "p"}
    assert names("tt") == {"m", "n"}
    # two contractions: k (mm1, mm2) then n (mm3)
    cnames = [info.canon[c] for _, c in info.contractions]
    assert cnames.count("k") == 2 and cnames.count("n") == 1


def _swapped_operand_mlp_nodes():
    # nc_matmul(w1, x^T): the weight is stationary, so the output rows (m) land
    # on the FIRST matmul's free axis instead of its partition axis.
    return [
        Node("x", "input", []),
        Node("w1", "input", []),
        Node("w2", "input", []),
        Node("w3", "input", []),
        Node("xt", "nc_transpose", ["x"]),
        Node("mm1", "nc_matmul", ["w1", "xt"]),
        Node("act", "activation", ["mm1"], {"op": "relu"}),
        Node("mm2", "nc_matmul", ["w2", "xt"]),
        Node("tt", "tensor_tensor", ["act", "mm2"], {"op": "multiply"}),
        Node("mm3", "nc_matmul", ["tt", "w3"]),
    ]


def test_swapped_operand_mlp_stays_inside_the_mnkp_namespace():
    # The emitted signature offers only TILES_IN_BLOCK_{M,N,K,P}, so a dim named
    # outside that set reads an absent parameter. Every present dim must be in it.
    compute, id2n, inputs = _graph(_swapped_operand_mlp_nodes())
    info = analyze_dims(compute, id2n, inputs, output_id="mm3")
    present = {info.canon[d] for ds in info.dims.values() for d in ds}
    assert present == {"m", "n", "k", "p"}
    cnames = [info.canon[c] for _, c in info.contractions]
    assert cnames.count("k") == 2 and cnames.count("n") == 1


def test_swapped_operand_first_matmul_free_axis_reuses_the_output_row_name():
    # The first matmul's free axis is the very occurrence the output matmul forks
    # into its partition dim, so it IS the m axis and is named "m". Naming it "n"
    # would park m under the n tile role and push the hidden dim out of the
    # namespace.
    compute, id2n, inputs = _graph(_swapped_operand_mlp_nodes())
    info = analyze_dims(compute, id2n, inputs, output_id="mm3")
    first_free = info.layouts["mm1"].free
    output_row_source = info.layouts["tt"].free
    assert first_free is not None and output_row_source is not None
    assert output_row_source.find() is first_free.find()
    assert info.canon[first_free.find()] == "m"


def test_first_matmul_free_axis_keeps_n_when_it_is_a_distinct_occurrence():
    # Control: an attention-shaped graph's first projection free axis is a
    # DIFFERENT occurrence from the one the output matmul forks into "m", even
    # though the two share a role. It stays "n", so no two distinct axes collapse
    # onto one tile role. Shared roles and equal extents never license the reuse.
    compute, id2n, inputs = _graph(_self_attention_nodes())
    info = analyze_dims(compute, id2n, inputs, output_id="out")
    first_free = info.layouts["q"].free
    output_row_source = info.layouts["probst"].free
    assert first_free is not None and output_row_source is not None
    assert output_row_source.find() is not first_free.find()
    assert info.canon[first_free.find()] == "n"


# ---------------------------------------------------------------------------
# FIX 1: tensor_reduce axis validation
# Layout inference must accept free-axis (axis=1) reductions and refuse
# partition-axis (axis=0) reductions with a LayoutError.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [1, -1, (1,), (-1,), [1], [-1]])
def test_tensor_reduce_free_axis_accepted(axis):
    """Equivalent rank-2 free-axis spellings infer a scalar layout."""
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node(
                "red",
                "tensor_reduce",
                ["x"],
                {"op": "add", "axis": axis, "keepdims": True},
            ),
        ]
    )
    L = infer_layouts(compute, id2n, inputs)
    assert L["red"].free is None
    assert L["red"].part.find() is L["x"].part.find()
    assert L["red"].part.same_extent(L["x"].part)


@pytest.mark.parametrize(
    "axis",
    [True, 1.5, "1", (True,), (1.5,), ("1",), [True], [1.5], ["1"]],
)
def test_tensor_reduce_coercible_non_integer_axis_rejected(axis):
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node(
                "red",
                "tensor_reduce",
                ["x"],
                {"op": "add", "axis": axis, "keepdims": True},
            ),
        ]
    )

    with pytest.raises(LayoutError, match="axis must be an integer"):
        infer_layouts(compute, id2n, inputs)


def test_tensor_reduce_axis0_raises_layout_error():
    """tensor_reduce with axis=0 (partition axis) raises LayoutError naming the axis."""
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node(
                "red",
                "tensor_reduce",
                ["x"],
                {"op": "add", "axis": 0, "keepdims": True},
            ),
        ]
    )
    with pytest.raises(LayoutError, match="axis=0"):
        infer_layouts(compute, id2n, inputs)


def test_tensor_reduce_axis0_error_names_node():
    """LayoutError for axis=0 reduce includes the node id."""
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node("my_red_node", "tensor_reduce", ["x"], {"op": "add", "axis": 0}),
        ]
    )
    with pytest.raises(LayoutError, match="my_red_node"):
        infer_layouts(compute, id2n, inputs)


def test_tensor_reduce_missing_axis_defaults_to_1():
    """tensor_reduce with no axis attr defaults to axis=1 (accepted)."""
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node("red", "tensor_reduce", ["x"], {"op": "add", "keepdims": True}),
        ]
    )
    L = infer_layouts(compute, id2n, inputs)
    assert L["red"].free is None
    assert L["red"].part.find() is L["x"].part.find()
    assert L["red"].part.same_extent(L["x"].part)


def test_tensor_reduce_axis_as_list1_accepted():
    """tensor_reduce with axis=[1] (list form) is accepted as free-axis reduce."""
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node(
                "red",
                "tensor_reduce",
                ["x"],
                {"op": "add", "axis": [1], "keepdims": True},
            ),
        ]
    )
    L = infer_layouts(compute, id2n, inputs)
    assert L["red"].free is None


def test_tensor_reduce_axis_as_list0_raises_layout_error():
    """tensor_reduce with axis=[0] (list form) raises LayoutError."""
    compute, id2n, inputs = _graph(
        [
            Node("x", "input", []),
            Node("red", "tensor_reduce", ["x"], {"op": "add", "axis": [0]}),
        ]
    )
    with pytest.raises(LayoutError, match="axis=0"):
        infer_layouts(compute, id2n, inputs)


# ---------------------------------------------------------------------------
# topo_order: a graphlib.TopologicalSorter order over the caller's list.
# Ties break on list position, so the result is deterministic across runs and
# invariant under any renaming of node ids.
# ---------------------------------------------------------------------------


def _relu_graph(
    ids: list[str], deps: dict[int, list[int]] | None = None
) -> tuple[list[Node], dict[str, Node]]:
    """Compute nodes over one input, node `k` also consuming `deps[k]` by index."""
    deps = deps or {}
    nodes = [
        Node(
            node_id,
            "activation",
            ["x"] + [ids[p] for p in deps.get(k, [])],
            {"op": "relu"},
        )
        for k, node_id in enumerate(ids)
    ]
    id_to_node = {node.id: node for node in nodes}
    id_to_node["x"] = Node("x", "input", [])
    return nodes, id_to_node


_TANGLED_DEPS = {2: [0], 3: [1], 4: [0, 3], 5: [2], 6: [4, 5], 7: [1, 6]}


def _positions(ids: list[str], deps: dict[int, list[int]]) -> list[int]:
    """The output order as indices into `ids`, so renamings stay comparable."""
    nodes, id_to_node = _relu_graph(ids, deps)
    return [ids.index(node.id) for node in topo_order(nodes, id_to_node)]


def test_topo_order_is_deterministic_across_repeated_calls():
    """The same input list yields the identical order every time."""
    ids = [f"activation_{i}" for i in range(8)]
    first = _positions(ids, _TANGLED_DEPS)

    assert all(_positions(ids, _TANGLED_DEPS) == first for _ in range(200))


def test_topo_order_is_deterministic_across_processes():
    """Separate interpreters with randomized hash seeds agree on the order."""
    script = (
        "from axon.codegen.layout import topo_order\n"
        "from axon.ir import Node\n"
        "ids = [f'activation_{i}' for i in range(8)]\n"
        f"deps = {_TANGLED_DEPS!r}\n"
        "nodes = [Node(i, 'activation', ['x'] + [ids[p] for p in deps.get(k, [])],"
        " {'op': 'relu'}) for k, i in enumerate(ids)]\n"
        "m = {n.id: n for n in nodes}\n"
        "m['x'] = Node('x', 'input', [])\n"
        "print(','.join(str(ids.index(n.id)) for n in topo_order(nodes, m)))\n"
    )
    env = {**os.environ, "PYTHONHASHSEED": "random"}
    digests = {
        hashlib.sha256(
            subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            ).stdout.encode()
        ).hexdigest()
        for _ in range(4)
    }

    assert len(digests) == 1
    assert digests == {
        hashlib.sha256(
            (
                ",".join(
                    str(p)
                    for p in _positions(
                        [f"activation_{i}" for i in range(8)], _TANGLED_DEPS
                    )
                )
                + "\n"
            ).encode()
        ).hexdigest()
    }


@pytest.mark.parametrize(
    "rename",
    [
        lambda i: f"zzz_activation_{i}",
        lambda i: f"activation_{i + 996}",
        lambda i: f"activation_{7 - i}",
        lambda i: "activation_" + "abcdefgh"[i] * (i + 1),
    ],
    ids=["prefixed", "digit_width", "reversed_numbering", "unordered_names"],
)
def test_topo_order_is_invariant_under_renaming_ids(rename):
    """Renaming every node, structure preserved, preserves the output positions.

    A lexicographic tiebreak would move `activation_1000` before `activation_999`.
    """
    base = [f"activation_{i}" for i in range(8)]
    renamed = [rename(i) for i in range(8)]

    assert _positions(renamed, _TANGLED_DEPS) == _positions(base, _TANGLED_DEPS)


def test_topo_order_emits_every_node_after_its_in_graph_inputs():
    """The output is a real topological order of the compute-node edges."""
    ids = [f"activation_{i}" for i in range(8)]
    nodes, id_to_node = _relu_graph(ids, _TANGLED_DEPS)

    ordered = [node.id for node in topo_order(nodes, id_to_node)]

    assert sorted(ordered) == sorted(ids)
    seen: set[str] = set()
    for node_id in ordered:
        inputs = set(id_to_node[node_id].inputs or [])
        assert inputs & set(ids) <= seen, node_id
        seen.add(node_id)


def test_topo_order_keeps_independent_nodes_in_list_order():
    """With no edge to constrain them, nodes come out in the caller's order."""
    ids = ["activation_999", "activation_1000", "activation_1001"]
    nodes, id_to_node = _relu_graph(ids)

    ordered = [node.id for node in topo_order(nodes, id_to_node)]

    assert ordered == ids
    assert ordered != sorted(ids)


def test_topo_order_reorders_a_consumer_that_precedes_its_producer():
    """A list out of dependency order is sorted, not accepted as given."""
    producer = Node("activation_1000", "activation", ["x"], {"op": "relu"})
    consumer = Node("activation_999", "activation", ["activation_1000"], {"op": "relu"})
    nodes = [consumer, producer]
    id_to_node = {n.id: n for n in nodes}
    id_to_node["x"] = Node("x", "input", [])

    ordered = [node.id for node in topo_order(nodes, id_to_node)]

    assert ordered == ["activation_1000", "activation_999"]


def test_topo_order_accepts_a_dependency_ordered_chain():
    """A producer-before-consumer chain passes through unchanged."""
    producer = Node("activation_1000", "activation", ["x"], {"op": "relu"})
    consumer = Node("activation_999", "activation", ["activation_1000"], {"op": "relu"})
    nodes = [producer, consumer]
    id_to_node = {n.id: n for n in nodes}
    id_to_node["x"] = Node("x", "input", [])

    ordered = [node.id for node in topo_order(nodes, id_to_node)]

    assert ordered == ["activation_1000", "activation_999"]


def test_topo_order_refuses_a_cycle():
    """A dependency cycle among compute nodes is a LayoutError, not a short order."""
    a = Node("act_a", "activation", ["act_b"], {"op": "relu"})
    b = Node("act_b", "activation", ["act_a"], {"op": "relu"})
    nodes = [a, b]
    id_to_node = {n.id: n for n in nodes}

    with pytest.raises(LayoutError, match="cycle in hw graph"):
        topo_order(nodes, id_to_node)
