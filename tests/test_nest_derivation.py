"""Loop-nest derivation: block loops from the output's dims, node placement
at the shallowest binding level (loop-invariant placement), materialization
for cross-level values. The derived nests must reproduce today's structure
for the existing kernel classes — asserted here on hand-built graphs shaped
like rmsnorm_matmul and the mixed MLP."""

from __future__ import annotations

from axon.codegen.nest import derive_nest
from axon.ir import Node


def _graph(nodes):
    id_to_node = {n.id: n for n in nodes}
    inputs = {n.id for n in nodes if n.op == "input"}
    compute = [n for n in nodes if n.op != "input"]
    return compute, id_to_node, inputs


def _rmsnorm_nodes():
    return [
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


def _mlp_v1_nodes():
    return [
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


def test_rmsnorm_nest_matches_todays_structure():
    nest = derive_nest(*_graph(_rmsnorm_nodes()), output_id="mm")
    assert nest.block_loops == ["m", "n"]
    assert nest.placement["red"] == ("m",)  # reduce preamble at m level
    assert nest.placement["rs"] == ("m",)
    assert nest.placement["sc"] == ("m", "n")  # inside the k-loop compute
    assert nest.placement["mm"] == ("m", "n")
    assert nest.accum_loops["mm"] == "k"
    assert "red" not in nest.materialized or nest.materialized["red"] == ()


def test_mlp_v1_nest_matches_template_structure():
    nest = derive_nest(*_graph(_mlp_v1_nodes()), output_id="mm3")
    assert nest.block_loops == ["m", "p"]
    # both first-stage matmuls and the combine live at (m, n): n is a
    # contraction of mm3, so it appears as mm3's accumulation loop, and the
    # first stage is placed inside an n-block loop hoisted to m level
    assert nest.placement["mm1"] == ("m", "n")
    assert nest.placement["mm2"] == ("m", "n")
    assert nest.placement["tt"] == ("m", "n")
    assert nest.placement["mm3"] == ("m", "p")
    assert nest.accum_loops["mm1"] == "k"
    assert nest.accum_loops["mm3"] == "n"
    # tt is consumed at (m,p) across all n blocks -> materialized over n
    assert nest.materialized["tt"] == ("n",)


def test_self_attention_nest_keeps_both_score_loops():
    nest = derive_nest(*_graph(_self_attention_nodes()), output_id="out")
    score_names = {nest.info.canon[dim] for dim in nest.info.dims["scores"]}
    assert len(nest.info.dims["scores"]) == 2
    assert len(score_names) == 2
    assert set(nest.placement["scores"]) == score_names
    assert set(nest.placement["probs"]) == score_names
    assert nest.block_loops == ["m", "p"]


def test_cross_stage_reduce_scalar_has_no_loop_span_in_base_nest():
    nodes = _self_attention_nodes()
    next(node for node in nodes if node.id == "sum").inputs[0] = "q"

    nest = derive_nest(*_graph(nodes), output_id="out")

    assert nest.placement["inv"] == ("m",)
    assert "inv" not in nest.materialized
