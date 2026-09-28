"""Unit tests on the EmissionPlan builder (codegen/plan.py).

The plan is the single interpretation of the hardware graph: analysis walks the
edges once and freezes exact node ids, roles, layouts, and lifetimes; the
renderer only translates it. These tests assert the plan carries the exact
expected ids/roles for each kernel shape, that the resolved per-dim TILE_<D> is
the configured base (or a typed refusal), and that the full-coverage keystone
refuses an unclassifiable node (an orphan) and a layout-mismatch combine at
PLAN time — before any emission. Input nodes carry concrete shapes, which the
plan pairs with the inferred layouts to derive the extent each tile is guarded
against."""

from __future__ import annotations

import pytest

from axon.codegen.constants import MOVING_FMAX, PARTITION_FMAX
from axon.codegen.ops import UnsupportedEmission
from axon.codegen.plan import build_emission_plan
from axon.ir import Node, nuGraph


def _matmul_site_count(code):
    """How many distinct matmuls the emitted source realizes.

    NOT `code.count("nisa.nc_matmul(")`: the drain protocol peels the first
    contraction sub-tile, so every matmul emits TWO `nc_matmul` calls (the peeled
    `accumulate=False` one and the `accumulate=True` loop body). Counting calls
    conflates a graph with N matmuls against one with 2N. Each matmul's PSUM
    accumulator is uniquified per matmul, so counting those is exact."""
    import ast as _ast

    names = set()
    for node in _ast.walk(_ast.parse(code)):
        if not isinstance(node, _ast.Call):
            continue
        if "nc_matmul" not in _ast.unparse(node.func):
            continue
        if not node.args:
            continue
        target = _ast.unparse(node.args[0])
        names.add(target.split("[")[0])
    return len(names)


# The rectangular extents the fixtures below are shaped at: every dim divides
# its default base (tile_m/tile_k = 128 on partition dims, tile_n = 512 wide).
M, N, K, P = 256, 1024, 128, 512


def _inp(node_id, shape):
    """A graph input node with a concrete 2D shape (what dim extents derive from)."""
    return Node(node_id, "input", [], {"shape": shape})


def _graph(nodes):
    id_to_node = {n.id: n for n in nodes}
    inputs = {n.id for n in nodes if n.op == "input"}
    compute = [n for n in nodes if n.op != "input"]
    return compute, id_to_node, inputs


# --- plain matmul (class A) ------------------------------------------------- #
def _plain_mm_nodes(x_shape=(M, K), w_shape=(K, N)):
    # mm = xᵀ@w, so x is (m, k) and w is (k, n).
    return [
        _inp("x", x_shape),
        _inp("w", w_shape),
        Node("xt", "nc_transpose", ["x"]),
        Node("mm", "nc_matmul", ["xt", "w"]),
    ]


def test_plain_matmul_plan_roles_and_ids():
    plan = build_emission_plan(*_graph(_plain_mm_nodes()), output_id="mm")
    assert len(plan.matmuls) == 1
    mp = plan.matmuls[0]
    assert mp.mm_id == "mm"
    assert not mp.is_chained
    # stationary derives from x (through the transpose), moving from w directly.
    assert mp.stat.root_id == "x"
    assert mp.mov.root_id == "w"
    assert [s.node_id for s in mp.stat.steps] == ["xt"]
    assert mp.stat.transpose_index == 0
    assert mp.mov.steps == ()
    assert mp.mov.transpose_index is None
    assert mp.accum_dim == "k"
    assert plan.post_mm == ()
    assert plan.reduces == ()
    assert not plan.is_multi


def test_plain_matmul_tile_roles_single_role_per_dim():
    # A plain matmul plays exactly one physical role per dim: n is the directly-
    # loaded moving free dim (wide only), m/k are partition/contraction (part
    # only). No dim carries BOTH granularities, so the moving slice stays at the
    # dim's single base tile (mov_wide is False).
    plan = build_emission_plan(*_graph(_plain_mm_nodes()), output_id="mm")
    assert plan.tile_roles["n"] == ("wide",)
    assert plan.tile_roles["m"] == ("part",)
    assert plan.tile_roles["k"] == ("part",)
    # no dim has the ("part", "wide") dual role
    assert not any(r == ("part", "wide") for r in plan.tile_roles.values())
    assert plan.matmuls[0].mov_wide is False


# --- resolved per-dim TILE_<D> ---------------------------------------------- #
# The renderer must execute the tile configuration it is handed, so the plan
# resolves TILE_<D> to the configured base or refuses; never a substitute.
def test_resolved_tiles_are_the_configured_bases():
    plan = build_emission_plan(*_graph(_plain_mm_nodes()), output_id="mm")
    # Bases by ROLE: tile_n for the wide dim (n), tile_k for the contraction (k),
    # tile_m for the rest (m). All three divide their extents here.
    assert plan.tiles == {"m": 128, "k": 128, "n": 512}
    assert plan.wide_dim == "n"


def test_resolved_tiles_honor_a_non_default_config():
    # A config the sweep (or a hardware annotation) resolved to smaller bases is
    # emitted verbatim, provided each still divides its extent.
    plan = build_emission_plan(
        *_graph(_plain_mm_nodes()),
        output_id="mm",
        tile_config={"tile_m": 64, "tile_k": 64, "tile_n": 256},
    )
    assert plan.tiles == {"m": 64, "k": 64, "n": 256}


def test_tile_refuses_when_base_does_not_divide_extent():
    # m=578 at base 128: no power-of-two divisor exists, so the old renderer
    # substituted TILE_M=2 (1% partition utilization) and misreported it.
    nodes = _plain_mm_nodes(x_shape=(578, K), w_shape=(K, N))
    with pytest.raises(UnsupportedEmission) as excinfo:
        build_emission_plan(*_graph(nodes), output_id="mm")
    msg = str(excinfo.value)
    assert "'m'" in msg and "578" in msg and "128" in msg
    assert str(PARTITION_FMAX) in msg


def test_tile_refuses_when_wide_base_does_not_divide_extent():
    # The wide dim is guarded the same way, at its own MOVING_FMAX cap: n=264 is
    # not divisible by the tile_n base of 512.
    nodes = _plain_mm_nodes(x_shape=(M, K), w_shape=(K, 264))
    with pytest.raises(UnsupportedEmission) as excinfo:
        build_emission_plan(*_graph(nodes), output_id="mm")
    msg = str(excinfo.value)
    assert "'n'" in msg and "264" in msg and "512" in msg
    assert str(MOVING_FMAX) in msg


def test_tile_refuses_when_base_exceeds_its_role_cap():
    # A partition-role dim caps at PARTITION_FMAX (128), so an annotated base of
    # 512 is refused even though it divides the extent (1024 % 512 == 0).
    nodes = _plain_mm_nodes(x_shape=(1024, 1024), w_shape=(1024, N))
    with pytest.raises(UnsupportedEmission) as excinfo:
        build_emission_plan(
            *_graph(nodes),
            output_id="mm",
            tile_config={"tile_m": 512, "tile_k": 128, "tile_n": 512},
        )
    msg = str(excinfo.value)
    assert "'m'" in msg and "512" in msg and str(PARTITION_FMAX) in msg
    assert "cap" in msg


def test_dual_role_dim_resolves_at_the_partition_base():
    # A dual-role ("part","wide") dim is NOT the wide_dim, so it bases off tile_m
    # and caps at PARTITION_FMAX; MATMUL_TILE_<D> subdivides the block it yields.
    plan = build_emission_plan(*_graph(_mlp_v1_nodes()), output_id="mm3")
    assert plan.tile_roles["n"] == ("part", "wide")
    assert plan.tiles["n"] == 128
    assert plan.tiles == {"m": 128, "n": 128, "k": 128, "p": 512}


def test_wide_base_wider_than_extent_clamps_to_one_full_tile():
    # attention_nkilib at head dim d=128: the probs@v wide dim (p) bases at
    # tile_n=512, which no longer refuses; 512 % 128 == 0, so it clamps to a
    # single 128-wide tile rather than refusing (contrast the n=264 refusal).
    plan = build_emission_plan(
        *_graph(_attention_core_nodes(s=512, d=128)), output_id="out"
    )
    assert plan.wide_dim == "p"
    assert plan.tiles["p"] == 128
    assert plan.tiles == {"m": 128, "n": 128, "k": 128, "p": 128}


def test_wide_clamp_renders_the_clamped_tile_not_the_base():
    # The renderer must emit TILE_P at the plan's clamped value (128), not the
    # tile_n base (512): a 128-wide P tiled at 512 would emit an impossible
    # `P % BLOCK_P == 0` assert (128 % 512) and die at trace.
    from axon.codegen.bodies.matmul_generic import emit_matmul_body_generic
    from axon.codegen.context import EmitCtx

    nodes = _attention_core_nodes(s=512, d=128)
    compute, id_to_node, _ = _graph(nodes)
    input_nodes = [node for node in nodes if node.op == "input"]
    ctx = EmitCtx(tile_config={"tile_m": 128, "tile_k": 128, "tile_n": 512})
    lines, _ = emit_matmul_body_generic(
        ctx,
        input_nodes,
        compute,
        id_to_node,
        [node.id for node in input_nodes],
        output_ids={"out"},
    )
    code = "\n".join(lines)
    assert "TILE_P = 128" in code
    assert "TILE_P = 512" not in code
    # The head dim's block guards on the clamped tile, satisfiable at
    # TILES_IN_BLOCK_P == 1 (128 % (128 * 1) == 0).
    assert 'assert P % BLOCK_P == 0, "P must be divisible by BLOCK_P"' in code
    assert ctx.tile_constraints["TILES_IN_BLOCK_P"].extent == 128
    assert ctx.tile_constraints["TILES_IN_BLOCK_P"].tile_size == 128
    assert ctx.tile_constraints["TILES_IN_BLOCK_P"].require_divisible


# --- rmsnorm-shaped (class B.2 reduce preamble + operand chain) ------------- #
def _rmsnorm_nodes():
    return [
        _inp("x", (M, K)),
        _inp("w", (K, N)),
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


def test_rmsnorm_plan_reduce_and_operand_chain():
    plan = build_emission_plan(*_graph(_rmsnorm_nodes()), output_id="mm")
    mp = plan.matmuls[0]
    # stationary operand chain: sc (single-scalar combiner) -> sct (transpose)
    assert [s.node_id for s in mp.stat.steps] == ["sc", "sct"]
    assert mp.stat.transpose_index == 1
    # the combiner reads the (m,) scalar `rs`
    sc_step = mp.stat.steps[0]
    assert sc_step.scalar_id == "rs"
    assert sc_step.data_pred_id == "x"
    # exactly one reduce plan feeding it
    assert len(plan.reduces) == 1
    rp = plan.reduces[0]
    assert rp.reduce_id == "red"
    assert rp.chain_ids == ("sq",)
    assert rp.hbm_ids == ("x",)
    assert rp.post_scalar_ids == ("rs",)
    # rmsnorm shares nothing between reduce chain and operand chain
    assert plan.staged_ids == frozenset()


def test_reduce_plan_tracks_constant_tensor_scalar_post_chain():
    nodes = [
        Node("x", "input", [], shape=(128, 512)),
        Node("w", "input", [], shape=(512, 1024)),
        Node(
            "red",
            "tensor_reduce",
            ["x"],
            {"op": "add", "axis": 1, "keepdims": True},
            shape=(128, 1),
        ),
        Node(
            "mean",
            "tensor_scalar",
            ["red"],
            {"op0": "multiply", "operand0_const": 1.0 / 512},
            shape=(128, 1),
        ),
        Node(
            "eps",
            "tensor_scalar",
            ["mean"],
            {"op0": "add", "operand0_const": 1e-6},
            shape=(128, 1),
        ),
        Node("root", "activation", ["eps"], {"op": "sqrt"}, shape=(128, 1)),
        Node("inv", "reciprocal", ["root"], shape=(128, 1)),
        Node(
            "norm",
            "tensor_scalar",
            ["x", "inv"],
            {"op0": "multiply", "operand0_input_index": 1},
            shape=(128, 512),
        ),
        Node("norm_t", "nc_transpose", ["norm"], shape=(512, 128)),
        Node("mm", "nc_matmul", ["norm_t", "w"], shape=(128, 1024)),
    ]

    plan = build_emission_plan(*_graph(nodes), output_id="mm")

    assert plan.scalar_ids >= {"red", "mean", "eps", "root", "inv"}
    assert plan.reduces[0].post_scalar_ids == ("mean", "eps", "root", "inv")
    norm_step = plan.matmuls[0].stat.steps[0]
    assert norm_step.node_id == "norm"
    assert norm_step.data_pred_id == "x"
    assert norm_step.scalar_id == "inv"


def _qkv_residual_nodes():
    return [
        Node("x", "input", [], shape=(128, 512)),
        Node("mlp", "input", [], shape=(128, 512)),
        Node("attention", "input", [], shape=(128, 512)),
        Node("w", "input", [], shape=(512, 1024)),
        Node(
            "residual_1",
            "tensor_tensor",
            ["x", "mlp"],
            {"op": "add"},
            shape=(128, 512),
        ),
        Node(
            "residual",
            "tensor_tensor",
            ["residual_1", "attention"],
            {"op": "add"},
            shape=(128, 512),
        ),
        Node(
            "square",
            "tensor_tensor",
            ["residual", "residual"],
            {"op": "multiply"},
            shape=(128, 512),
        ),
        Node(
            "sum",
            "tensor_reduce",
            ["square"],
            {"op": "add", "axis": 1, "keepdims": True},
            shape=(128, 1),
        ),
        Node(
            "mean",
            "tensor_scalar",
            ["sum"],
            {"op0": "multiply", "operand0_const": 1.0 / 512},
            shape=(128, 1),
        ),
        Node(
            "eps",
            "tensor_scalar",
            ["mean"],
            {"op0": "add", "operand0_const": 1e-6},
            shape=(128, 1),
        ),
        Node("root", "activation", ["eps"], {"op": "sqrt"}, shape=(128, 1)),
        Node("inv", "reciprocal", ["root"], shape=(128, 1)),
        Node(
            "norm",
            "tensor_scalar",
            ["residual", "inv"],
            {"op0": "multiply", "operand0_input_index": 1},
            shape=(128, 512),
        ),
        Node("norm_t", "nc_transpose", ["norm"], shape=(512, 128)),
        Node("out", "nc_matmul", ["norm_t", "w"], shape=(128, 1024)),
    ]


def test_qkv_residual_fanin_is_a_reduce_owned_staged_root():
    plan = build_emission_plan(*_graph(_qkv_residual_nodes()), output_id="out")

    stat = plan.matmuls[0].stat
    assert stat.root_id == "residual"
    assert [step.node_id for step in stat.steps] == ["norm", "norm_t"]
    assert plan.reduces[0].chain_ids == ("residual_1", "residual", "square")
    assert plan.staged_ids == frozenset({"residual"})
    assert plan.staged_dtype_sources == (("residual", "x"),)


def test_shared_linear_prefix_stages_only_maximal_endpoint():
    nodes = [
        Node("x", "input", [], shape=(128, 512)),
        Node("w", "input", [], shape=(512, 1024)),
        Node("first", "activation", ["x"], {"op": "relu"}, shape=(128, 512)),
        Node(
            "second",
            "activation",
            ["first"],
            {"op": "relu"},
            shape=(128, 512),
        ),
        Node(
            "square",
            "tensor_tensor",
            ["second", "second"],
            {"op": "multiply"},
            shape=(128, 512),
        ),
        Node(
            "sum",
            "tensor_reduce",
            ["square"],
            {"op": "add", "axis": 1, "keepdims": True},
            shape=(128, 1),
        ),
        Node("inv", "reciprocal", ["sum"], shape=(128, 1)),
        Node(
            "norm",
            "tensor_scalar",
            ["second", "inv"],
            {"op0": "multiply", "operand0_input_index": 1},
            shape=(128, 512),
        ),
        Node("norm_t", "nc_transpose", ["norm"], shape=(512, 128)),
        Node("out", "nc_matmul", ["norm_t", "w"], shape=(128, 1024)),
    ]

    plan = build_emission_plan(*_graph(nodes), output_id="out")

    assert plan.staged_ids == frozenset({"second"})
    assert plan.staged_dtype_sources == (("second", "x"),)


def test_unowned_full_tile_fanin_still_refuses():
    nodes = [
        Node("x", "input", [], shape=(128, 512)),
        Node("y", "input", [], shape=(128, 512)),
        Node("w", "input", [], shape=(512, 1024)),
        Node(
            "residual",
            "tensor_tensor",
            ["x", "y"],
            {"op": "add"},
            shape=(128, 512),
        ),
        Node("residual_t", "nc_transpose", ["residual"], shape=(512, 128)),
        Node("out", "nc_matmul", ["residual_t", "w"], shape=(128, 1024)),
    ]

    with pytest.raises(UnsupportedEmission, match="residual.*single-scalar"):
        build_emission_plan(*_graph(nodes), output_id="out")


def test_reduce_plan_accepts_negative_rank2_free_axis():
    nodes = _rmsnorm_nodes()
    next(node for node in nodes if node.id == "red").attrs["axis"] = (-1,)

    plan = build_emission_plan(*_graph(nodes), output_id="mm")

    assert plan.reduces[0].reduce_id == "red"
    assert plan.reduces[0].axis == 1


def test_reduce_plan_rejects_unknown_tensor_reduce_attribute():
    nodes = _rmsnorm_nodes()
    next(node for node in nodes if node.id == "red").attrs["unknown"] = True

    with pytest.raises(
        UnsupportedEmission,
        match=r"tensor_reduce red: unsupported attributes \['unknown'\]",
    ):
        build_emission_plan(*_graph(nodes), output_id="mm")


def test_reduce_broadcast_is_a_scalar_alias_for_operand_combiner():
    nodes = [
        Node("x", "input", [], shape=(128, 512)),
        Node("y", "input", [], shape=(128, 512)),
        Node("w", "input", [], shape=(512, 1024)),
        Node(
            "red",
            "tensor_reduce",
            ["y"],
            {"op": "add", "axis": 1, "keepdims": True},
            shape=(128, 1),
        ),
        Node(
            "red_broadcast",
            "broadcast",
            ["red", "x"],
            shape=(128, 512),
        ),
        Node(
            "scaled",
            "tensor_tensor",
            ["red_broadcast", "x"],
            {"op": "multiply"},
            shape=(128, 512),
        ),
        Node("scaled_t", "nc_transpose", ["scaled"], shape=(512, 128)),
        Node("mm", "nc_matmul", ["scaled_t", "w"], shape=(128, 1024)),
    ]

    plan = build_emission_plan(*_graph(nodes), output_id="mm")

    assert "red_broadcast" in plan.scalar_ids
    assert [step.node_id for step in plan.matmuls[0].stat.steps] == [
        "scaled",
        "scaled_t",
    ]
    assert plan.matmuls[0].stat.steps[0].scalar_id == "red_broadcast"
    assert plan.reduces[0].post_scalar_ids == ("red_broadcast",)


def test_full_tile_broadcast_is_a_transparent_operand_step():
    nodes = [
        Node("x", "input", [], shape=(128, 512)),
        Node("y", "input", [], shape=(128, 512)),
        Node("w", "input", [], shape=(512, 1024)),
        Node(
            "red",
            "tensor_reduce",
            ["y"],
            {"op": "add", "axis": 1, "keepdims": True},
            shape=(128, 1),
        ),
        Node(
            "scaled",
            "tensor_scalar",
            ["x", "red"],
            {"op0": "multiply", "operand0_input_index": 1},
            shape=(128, 512),
        ),
        Node(
            "scaled_broadcast",
            "broadcast",
            ["scaled", "x"],
            shape=(128, 512),
        ),
        Node(
            "scaled_t",
            "nc_transpose",
            ["scaled_broadcast"],
            shape=(512, 128),
        ),
        Node("mm", "nc_matmul", ["scaled_t", "w"], shape=(128, 1024)),
    ]

    plan = build_emission_plan(*_graph(nodes), output_id="mm")

    steps = plan.matmuls[0].stat.steps
    assert [step.node_id for step in steps] == [
        "scaled",
        "scaled_broadcast",
        "scaled_t",
    ]
    assert steps[1].data_pred_id == "scaled"
    assert steps[1].scalar_id is None


def test_activation_reduce_plan_keeps_auxiliary_hbm_inputs():
    nodes = [
        _inp("x", (M, K)),
        _inp("bias", (M, 1)),
        _inp("w", (K, N)),
        Node(
            "red",
            "activation_reduce",
            ["x", "bias"],
            {
                "op": "square",
                "reduce_op": "add",
                "bias_input_index": 1,
            },
        ),
        Node("rs", "reciprocal", ["red"]),
        Node(
            "sc",
            "tensor_scalar",
            ["x", "rs"],
            {"op0": "multiply", "operand0_input_index": 1},
        ),
        Node("sct", "nc_transpose", ["sc"]),
        Node("mm", "nc_matmul", ["sct", "w"]),
    ]
    plan = build_emission_plan(*_graph(nodes), output_id="mm")
    assert plan.reduces[0].hbm_ids == ("x", "bias")


# --- softmax-shaped (staged shared exp) ------------------------------------- #
def _softmax_nodes():
    # exp feeds BOTH the sum-reduce and the matmul operand (staged once).
    return [
        _inp("x", (M, K)),
        _inp("w", (K, N)),
        Node("ex", "exponential", ["x"]),
        Node("red", "tensor_reduce", ["ex"], {"op": "add", "keepdims": True}),
        Node("rc", "reciprocal", ["red"]),
        Node(
            "sc",
            "tensor_scalar",
            ["ex", "rc"],
            {"op0": "multiply", "operand0_input_index": 1},
        ),
        Node("sct", "nc_transpose", ["sc"]),
        Node("mm", "nc_matmul", ["sct", "w"]),
    ]


def test_softmax_plan_stages_shared_exp():
    plan = build_emission_plan(*_graph(_softmax_nodes()), output_id="mm")
    # ex is on both the reduce chain and the operand chain -> staged.
    assert "ex" in plan.staged_ids
    rp = plan.reduces[0]
    assert rp.reduce_id == "red"
    assert "ex" in rp.chain_ids
    assert rp.post_scalar_ids == ("rc",)
    mp = plan.matmuls[0]
    assert "ex" in {s.node_id for s in mp.stat.steps}


# --- mixed-MLP-shaped (chained matmul, class F) ----------------------------- #
def _mlp_v1_nodes():
    # mm1 = xᵀ@w1 (m,k)x(k,n); mm2 = w2ᵀ@xᵀ; mm3 = ttᵀ@w3 contracts over n.
    return [
        _inp("x", (M, K)),
        _inp("w1", (K, N)),
        _inp("w2", (K, N)),
        _inp("w3", (N, P)),
        Node("xt1", "nc_transpose", ["x"]),
        Node("mm1", "nc_matmul", ["xt1", "w1"]),
        Node("xt2", "nc_transpose", ["x"]),
        Node("mm2", "nc_matmul", ["w2", "xt2"]),
        Node("act", "activation", ["mm1"], {"op": "relu"}),
        Node("tr", "nc_transpose", ["act"]),
        Node("tt", "tensor_tensor", ["tr", "mm2"], {"op": "multiply"}),
        Node("mm3", "nc_matmul", ["tt", "w3"]),
    ]


def test_mlp_v1_plan_is_chained_with_materialized_combine():
    plan = build_emission_plan(*_graph(_mlp_v1_nodes()), output_id="mm3")
    assert plan.is_multi
    assert plan.stages == ()
    assert plan.chained_mm_id == "mm3"
    # the materialized inter node feeding mm3's stationary operand is the combine
    assert plan.materialized_inter_id == "tt"
    # three matmuls planned; mm3 is chained (materialized stationary operand)
    ids = {mp.mm_id for mp in plan.matmuls}
    assert ids == {"mm1", "mm2", "mm3"}
    chained = plan.matmul("mm3")
    assert chained.is_chained
    assert chained.stat.is_materialized
    assert chained.stat.op_id == "tt"
    assert chained.mov.root_id == "w3"
    # inter nodes cover act, tr, tt (not the matmuls, not the operand loads)
    assert set(plan.inter_ids) == {"act", "tr", "tt"}


def test_mlp_v1_chained_dim_carries_both_tile_roles():
    # The keystone of finding 3: in a chained schedule the same logical n dim
    # plays TWO physical roles — the wide inner-matmul moving tile in stage 1
    # (mm1: xᵀ@w1, whose moving free dim is n, directly loaded) and the
    # partitioned contraction in stage 2 (mm3 contracts over n). So n carries
    # BOTH roles; m/k stay partition-only; p is the stage-2 moving free dim
    # (wide only). Only stage-1's mm1 slices its moving operand wide.
    plan = build_emission_plan(*_graph(_mlp_v1_nodes()), output_id="mm3")
    assert plan.tile_roles["n"] == ("part", "wide")
    assert plan.tile_roles["k"] == ("part",)
    assert plan.tile_roles["p"] == ("wide",)
    # `m` gained the wide role at plan step 7b. `mm2 = nc_matmul(w2, xt2)` has a
    # TRANSPOSED moving operand, and the old guard read that transpose as evidence
    # the operand was not block-spanning. It is: every load path in
    # `_emit_operand_load` gathers into the same
    # `(TILE_<k>, TILES_IN_BLOCK_<k>, BLOCK_<free>)` shape, so a wide slice out of
    # it is legal, and `proto_wide_transposed_mov.py` measures 1.12e-07 at both
    # widths. This assertion read `("part",)` before that change.
    assert plan.tile_roles["m"] == ("part", "wide")
    mov_wide = {mp.mm_id: mp.mov_wide for mp in plan.matmuls}
    # mm1's moving operand (w1, free dim n, directly loaded) tiles wide; mm2's now
    # does too, despite reaching its slot through a transpose; mm3's moving free
    # dim p is wide-only so it already tiles wide at its base (not "both").
    #
    # Read the emitted diff, not just this dict: mm2's loop bound, PSUM shape,
    # operand slice and drain slice all have to move to MATMUL_TILE_M together, or
    # the kernel indexes past its accumulator.
    assert mov_wide == {"mm1": True, "mm2": True, "mm3": False}


def test_chained_materialized_id_is_the_graph_edge():
    # Value-identity authority: the chained matmul's stationary operand is the
    # graph edge mm3.inputs[0], and the plan's materialized_inter_id is set from
    # THAT edge (not an independent scan of nest.materialized).
    nodes = _mlp_v1_nodes()
    plan = build_emission_plan(*_graph(nodes), output_id="mm3")
    mm3 = next(n for n in nodes if n.id == "mm3")
    assert plan.materialized_inter_id == mm3.inputs[0]
    # and the chained matmul's stationary OperandPlan binds the same id.
    assert plan.matmul("mm3").stat.op_id == mm3.inputs[0]


def test_chained_refuses_when_materialized_lacks_the_graph_edge(monkeypatch):
    # If liveness (nest.materialized) ever disagrees with the graph edge — the
    # edge's id is NOT materialized — the plan must refuse (typed) rather than
    # silently read some other buffer. Strip the edge's id from nest.materialized
    # and assert the refusal names both the matmul and the operand.
    import axon.codegen.plan as plan_mod
    from axon.codegen.nest import Nest

    real_derive = plan_mod.derive_nest

    def _strip_materialized(*args, **kwargs):
        nest = real_derive(*args, **kwargs)
        # Drop "tt" (mm3's stationary graph edge) from materialized.
        stripped = {k: v for k, v in nest.materialized.items() if k != "tt"}
        return Nest(
            info=nest.info,
            block_loops=nest.block_loops,
            placement=nest.placement,
            accum_loops=nest.accum_loops,
            materialized=stripped,
        )

    monkeypatch.setattr(plan_mod, "derive_nest", _strip_materialized)
    compute, id_to_node, inputs = _graph(_mlp_v1_nodes())
    with pytest.raises(UnsupportedEmission, match="tt"):
        build_emission_plan(compute, id_to_node, inputs, output_id="mm3")


# --- multi-stage attention ------------------------------------------------- #
# The staged-attention extents. Each must divide the base its ROLE resolves to
# (`_resolve_tiles`): AM/AK at the 128 partition base, AN/AP at the 512 wide one.
AM, AK, AN, AP = 512, 128, 512, 512


def _self_attention_nodes():
    return [
        Node("x", "input", [], shape=(AM, AK)),
        Node("wq", "input", [], shape=(AK, AN)),
        Node("wk", "input", [], shape=(AK, AN)),
        Node("wv", "input", [], shape=(AK, AP)),
        Node("xtq", "nc_transpose", ["x"], shape=(AK, AM)),
        Node("q", "nc_matmul", ["xtq", "wq"], shape=(AM, AN)),
        Node("xtk", "nc_transpose", ["x"], shape=(AK, AM)),
        Node("k", "nc_matmul", ["xtk", "wk"], shape=(AM, AN)),
        Node("xtv", "nc_transpose", ["x"], shape=(AK, AM)),
        Node("v", "nc_matmul", ["xtv", "wv"], shape=(AM, AP)),
        Node("qt", "nc_transpose", ["q"], shape=(AN, AM)),
        Node("kt", "nc_transpose", ["k"], shape=(AN, AM)),
        Node("scores", "nc_matmul", ["qt", "kt"], shape=(AM, AM)),
        Node("ex", "exponential", ["scores"], shape=(AM, AM)),
        Node(
            "sum",
            "tensor_reduce",
            ["ex"],
            {"op": "add", "axis": 1, "keepdims": True},
            shape=(AM, 1),
        ),
        Node("inv", "reciprocal", ["sum"], shape=(AM, 1)),
        Node(
            "probs",
            "tensor_scalar",
            ["ex", "inv"],
            {"op0": "multiply", "operand0_input_index": 1},
            shape=(AM, AM),
        ),
        Node("probst", "nc_transpose", ["probs"], shape=(AM, AM)),
        Node("out", "nc_matmul", ["probst", "v"], shape=(AM, AP)),
    ]


def _attention_core_nodes(s: int = AM, d: int = AP):
    """`kernels/attention_nkilib`'s graph: ``softmax(q @ k_t) @ v`` with q/k_t/v
    as inputs, so it plans as two stages and its raw dims fill the namespace."""
    return [
        Node("q", "input", [], shape=(s, d)),
        Node("k_t", "input", [], shape=(d, s)),
        Node("v", "input", [], shape=(s, d)),
        Node("qt", "nc_transpose", ["q"], shape=(d, s)),
        Node("qk", "nc_matmul", ["qt", "k_t"], shape=(s, s)),
        Node("ex", "exponential", ["qk"], shape=(s, s)),
        Node(
            "sum",
            "tensor_reduce",
            ["ex"],
            {"op": "add", "axis": 1, "keepdims": True},
            shape=(s, 1),
        ),
        Node("inv", "reciprocal", ["sum"], shape=(s, 1)),
        Node(
            "probs",
            "tensor_scalar",
            ["ex", "inv"],
            {"op0": "multiply", "operand0_input_index": 1},
            shape=(s, s),
        ),
        Node("probst", "nc_transpose", ["probs"], shape=(s, s)),
        Node("out", "nc_matmul", ["probst", "v"], shape=(s, d)),
    ]


def test_attention_plan_has_ordered_projection_normalization_output_stages():
    plan = build_emission_plan(*_graph(_self_attention_nodes()), output_id="out")

    assert plan.is_multi
    assert len(plan.stages) == 3
    projections, normalization, output = plan.stages

    assert set(projections.matmul_ids) == {"q", "k", "v"}
    assert set(projections.inter_ids) == {"qt", "kt"}

    assert normalization.matmul_ids == ("scores",)
    assert [reduce.reduce_id for reduce in normalization.reduces] == ["sum"]
    assert normalization.inter_ids == ("probs", "probst")

    assert output.matmul_ids == ("out",)
    assert output.inter_ids == ()
    assert {node_id for stage in plan.stages for node_id in stage.node_ids} == {
        node.id for node in _self_attention_nodes() if node.op != "input"
    }

    assert {matmul.mm_id for matmul in plan.matmuls} == {
        "q",
        "k",
        "v",
        "scores",
        "out",
    }
    assert plan.matmul("scores").stat.op_id == "qt"
    assert plan.matmul("scores").mov.op_id == "kt"
    assert plan.matmul("out").stat.op_id == "probst"
    assert plan.matmul("out").mov.op_id == "v"
    assert {plan.matmul(mm_id).stat.root_id for mm_id in ("q", "k", "v")} == {"x"}
    score_layout = plan.layouts["scores"]
    assert score_layout.free is not None
    assert score_layout.part.find() is not score_layout.free.find()
    assert all(
        inter.op not in {"tensor_reduce", "activation_reduce"}
        for stage in plan.stages
        for inter in stage.inters
    )


def test_attention_plan_reuses_reduce_and_liveness_analysis():
    plan = build_emission_plan(*_graph(_self_attention_nodes()), output_id="out")
    projections, normalization, output = plan.stages
    reduce = normalization.reduces[0]

    assert reduce.input_id == "ex"
    assert reduce.chain_ids == ("ex",)
    assert reduce.post_scalar_ids == ("inv",)
    assert reduce.source_ids == ("scores",)
    assert reduce.hbm_inputs == ()
    assert plan.staged_ids == frozenset({"ex"})

    assert projections.live_out_ids == ("v", "qt", "kt")
    assert normalization.live_in_ids == ("v", "qt", "kt")
    assert normalization.live_out_ids == ("v", "probst")
    assert output.live_in_ids == ("v", "probst")
    assert output.live_out_ids == ()
    assert set(plan.nest.materialized) >= {"kt", "qt", "probst", "v"}


def _cross_stage_reduce_scalar_nodes(scalar_op):
    nodes = _self_attention_nodes()
    by_id = {node.id: node for node in nodes}
    by_id["sum"].inputs[0] = "q"
    by_id["inv"].op = scalar_op
    by_id["inv"].attrs = {"op": "reciprocal"} if scalar_op == "activation" else {}
    return nodes


@pytest.mark.parametrize("scalar_op", ["activation", "reciprocal"])
def test_cross_stage_reduce_scalar_reaches_the_role_extent_gate(scalar_op):
    """A crossing PER-PARTITION SCALAR needs no `nest.materialized` lifetime.

    `nest.materialized` records one only for a consumer that is an `nc_matmul`
    accumulating over a dim the producer is placed at, which a scalar never is.
    So the plan excluded exactly the one crossing shape that needs no HBM buffer
    at all: the reducer already keeps such a value in a
    `(TILE_<part>, TILES_IN_BLOCK_<part>, 1)` SBUF buffer registered in
    `G.scalar_bufs`.

    Step 7a lifted that exclusion, so this fixture now reaches ONE later gate and
    refuses there: reading `sum` from `q` leaves its two `n`-role dims with
    different extents. The exact message is asserted, so any OTHER refusal
    (the materialization check included) fails the test."""
    nodes = _cross_stage_reduce_scalar_nodes(scalar_op)

    with pytest.raises(
        UnsupportedEmission,
        match=(
            r"^generic matmul: staged emission requires more than the fixed "
            r"m/n/k/p extents \(role 'n' has unequal extents\)$"
        ),
    ):
        build_emission_plan(*_graph(nodes), output_id="out")


def test_crossing_scalars_are_recorded_for_the_hbm_sites_to_exclude():
    """`sbuf_scalar_crossings` is the set every HBM site reads.

    Four sites act on a crossing value independently (the `shared_hbm` alloc, the
    `crossing_in` reload, the store, and the `unstored` accounting), so the set is
    defined once in the plan rather than re-derived at each one. This is the
    two-matmul core, whose reduce scalar does not cross a stage, plus the same
    graph normalized after the output, whose scalar does."""
    plan = build_emission_plan(*_graph(_self_attention_nodes()), output_id="out")

    # Every recorded crossing is genuinely a per-partition scalar: no free axis.
    for node_id in plan.sbuf_scalar_crossings:
        assert plan.layouts[node_id].free is None, (
            f"{node_id!r} was recorded as an SBUF scalar crossing but has a free "
            "axis, so it needs a real HBM buffer"
        )
    # And every recorded crossing is a scalar the plan already knows about.
    assert plan.sbuf_scalar_crossings <= plan.scalar_ids


@pytest.mark.parametrize("scalar_op", ["activation", "reciprocal"])
def test_attention_emit_supports_same_stage_reduce_scalar(scalar_op):
    from axon.codegen import emit

    nodes = _self_attention_nodes()
    by_id = {node.id: node for node in nodes}
    by_id["wv"].shape = (AK, AN)
    by_id["v"].shape = (AM, AN)
    by_id["out"].shape = (AM, AN)
    by_id["k"].inputs[0] = "xtq"
    by_id["v"].inputs[0] = "xtq"
    nodes = [node for node in nodes if node.id not in {"xtk", "xtv"}]
    inv = next(node for node in nodes if node.id == "inv")
    inv.op = scalar_op
    inv.attrs = {"op": "reciprocal"} if scalar_op == "activation" else {}
    graph = nuGraph(
        nodes=nodes,
        input_ids=("x", "wq", "wk", "wv"),
        output_ids=("out",),
    )

    code = emit(graph, kernel_name=f"same_stage_{scalar_op}_attention")

    compile(code, f"<same_stage_{scalar_op}_attention>", "exec")
    assert _matmul_site_count(code) == 5


def test_attention_plan_refuses_unowned_cross_stage_activation():
    nodes = _self_attention_nodes()
    by_id = {node.id: node for node in nodes}
    for node_id in ("wq", "wk"):
        by_id[node_id].shape = (AK, AM)
    for node_id in ("q", "k", "qt", "kt"):
        by_id[node_id].shape = (AM, AM)
    nodes.extend(
        [
            Node(
                "unowned",
                "activation",
                ["q"],
                {"op": "relu"},
                shape=(AM, AM),
            ),
            Node(
                "mixed",
                "tensor_tensor",
                ["ex", "unowned"],
                {"op": "add"},
                shape=(AM, AM),
            ),
        ]
    )
    by_id["probs"].inputs[0] = "mixed"

    with pytest.raises(
        UnsupportedEmission,
        match=(
            "stage-crossing values have no nest.materialized lifetime: "
            r"\['unowned'\]"
        ),
    ):
        build_emission_plan(*_graph(nodes), output_id="out")


@pytest.mark.parametrize("reduce_op", ["tensor_reduce", "activation_reduce"])
def test_attention_plan_refuses_unowned_terminal_reduction(reduce_op):
    nodes = _self_attention_nodes()
    attrs = (
        {"op": "add", "axis": 1, "keepdims": True}
        if reduce_op == "tensor_reduce"
        else {"op": "square", "reduce_op": "add"}
    )
    nodes.append(
        Node(
            "terminal_reduce",
            reduce_op,
            ["ex"],
            attrs,
            shape=(AM, 1),
        )
    )

    with pytest.raises(UnsupportedEmission, match="terminal_reduce"):
        build_emission_plan(*_graph(nodes), output_id="out")


# --- full-coverage keystone: an orphan node refuses ------------------------- #
def test_orphan_node_refuses_at_plan_time():
    nodes = _plain_mm_nodes() + [
        # a stray elementwise on w that nothing on the matmul path consumes
        Node("orphan", "activation", ["w"], {"op": "relu"}),
    ]
    # add a second sink guard: make orphan not the output; output stays mm.
    compute, id_to_node, inputs = _graph(nodes)
    with pytest.raises(UnsupportedEmission, match="orphan"):
        build_emission_plan(compute, id_to_node, inputs, output_id="mm")


# --- layout-mismatch combine refuses at plan time --------------------------- #
def test_layout_mismatch_combine_refuses_at_plan_time():
    # combine of mm1 (m-major) and mm2 (n-major) WITHOUT the reconciling
    # transpose: two full-tile operands with disagreeing layouts.
    nodes = [
        _inp("x", (M, K)),
        _inp("w1", (K, N)),
        _inp("w2", (K, N)),
        _inp("w3", (N, P)),
        Node("xt1", "nc_transpose", ["x"]),
        Node("mm1", "nc_matmul", ["xt1", "w1"]),
        Node("xt2", "nc_transpose", ["x"]),
        Node("mm2", "nc_matmul", ["w2", "xt2"]),
        Node("tt", "tensor_tensor", ["mm1", "mm2"], {"op": "multiply"}),
        Node("mm3", "nc_matmul", ["tt", "w3"]),
    ]
    compute, id_to_node, inputs = _graph(nodes)
    with pytest.raises(UnsupportedEmission):
        build_emission_plan(compute, id_to_node, inputs, output_id="mm3")


# --- post-mm chain (class C) ------------------------------------------------ #
def test_post_mm_chain_plan():
    # matmul then a relu on the result (post-mm elementwise).
    nodes = [
        _inp("x", (M, K)),
        _inp("w", (K, N)),
        Node("xt", "nc_transpose", ["x"]),
        Node("mm", "nc_matmul", ["xt", "w"]),
        Node("act", "activation", ["mm"], {"op": "relu"}),
    ]
    plan = build_emission_plan(*_graph(nodes), output_id="act")
    assert [s.node_id for s in plan.post_mm] == ["act"]
    assert plan.post_mm[0].data_pred_id == "mm"


# --- tile-role resolution --------------------------------------------------- #
def test_resolve_tile_roles_maps():
    """The five-matmul QKV nest folds ``k2`` onto ``p`` and ``p`` onto ``n``;
    the two-matmul core takes the identity. A drift here retiles silently."""
    plan = build_emission_plan(*_graph(_self_attention_nodes()), output_id="out")
    assert plan.dim_aliases == {"m": "m", "k2": "p", "n": "n", "p": "n", "k": "k"}
    assert set(plan.role_dims) == {"m", "n", "p", "k"}

    core = build_emission_plan(*_graph(_attention_core_nodes()), output_id="out")
    assert core.dim_aliases == {"m": "m", "n": "n", "k": "k", "p": "p"}
    assert set(core.role_dims) == {"m", "n", "k", "p"}


def test_resolve_tile_roles_binds_direct_key_projection():
    """A projection reaching its consumer with no transpose has a free dim that
    is neither a contraction nor bound, so it takes its consumer axis's role."""
    import test_generic_emitter as emitter_tests

    graph = emitter_tests._constructed_attention_graph()
    emitter_tests._use_direct_attention_key_projection(graph)
    id_to_node = {node.id: node for node in graph.nodes}
    inputs = {node.id for node in graph.nodes if node.op == "input"}
    compute = [node for node in graph.nodes if node.op != "input"]

    plan = build_emission_plan(compute, id_to_node, inputs, output_id="out")

    assert set(plan.dim_aliases.values()) == {"m", "n", "k", "p"}
    assert plan.dim_aliases == {"m": "m", "p": "n", "k2": "p", "n": "n", "k": "k"}
