"""End-to-end structural tests for the generic matmul emitter: emit hand-built
graphs via the generic path and run them under nki.simulate on RECTANGULAR fp32
shapes. fp32 kills accumulation-rounding noise, so any surviving error is a
mis-derived loop or index; rectangular dims make namespace mixups misindex
instead of cancel."""

from __future__ import annotations

import ast
import importlib.util
import itertools
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_dead_sbuf_buffer import find_dead_sbuf_buffers  # noqa: E402


def _activation_reduce_node():
    from axon.ir import Node
    from axon.isa_semantics import nl

    return Node(
        "activation_reduce_1",
        "activation_reduce",
        ["x"],
        {"op": nl.exp, "reduce_op": nl.add},
        shape=(128, 1),
    )


def test_staged_activation_reduce_emits_one_statement_writing_the_staged_dst():
    """The staged destination is a parameter, so the emitter writes the final call
    in one pass: no scratch-tile line and no `{IND}` continuation to rewrite."""
    from axon.codegen.context import EmitCtx
    from axon.codegen.ops import emit_activation_reduce_to

    call = emit_activation_reduce_to(
        EmitCtx(),
        _activation_reduce_node(),
        {"x": "data"},
        act_dst="exponential_0_staged[0:M, 0:K]",
    )

    assert call.startswith("nisa.activation_reduce(exponential_0_staged[")
    assert "reduce_res={DST}" in call
    assert "\n" not in call
    assert "{IND}" not in call
    assert "_act" not in call


def test_unstaged_activation_reduce_keeps_its_throwaway_scratch_tile():
    from axon.codegen.context import EmitCtx
    from axon.codegen.ops import emit_node_call

    call = emit_node_call(EmitCtx(), _activation_reduce_node(), {"x": "data"})

    scratch, statement = call.split("\n")
    assert scratch.startswith("activation_reduce_1_act = nl.ndarray(data.shape,")
    assert statement.startswith("{IND}nisa.activation_reduce(activation_reduce_1_act,")


def test_activation_reduce_to_refuses_a_non_activation_reduce_node():
    from axon.codegen.context import EmitCtx
    from axon.codegen.ops import UnsupportedEmission, emit_activation_reduce_to
    from axon.ir import Node

    node = Node("t", "tensor_copy", ["x"], {}, shape=(128, 1))
    with pytest.raises(UnsupportedEmission, match="not activation_reduce"):
        emit_activation_reduce_to(EmitCtx(), node, {"x": "data"}, act_dst="dst")


def test_tv_refuses_a_schedule_dim_outside_the_signature_tile_namespace():
    """The relocated tile-name guard, at its cause.

    `_GenericMM.tv` is the sole mint site for the emitted tile names, so a dim
    whose `TILES_IN_BLOCK_<D>` the signature does not bind (a fallback
    `analyze_dims` name such as `d4` landing on a scheduled axis) must refuse
    there rather than emit a module that raises `NameError` at trace time."""
    from axon.codegen.bodies.matmul_generic import _GenericMM
    from axon.codegen.ops import UnsupportedEmission

    G = _generic_mm_for_plain_matmul()
    # A dim the signature does bind mints its names.
    assert G.tv("k").tiles_in_block == "TILES_IN_BLOCK_K"
    with pytest.raises(UnsupportedEmission, match=r"TILES_IN_BLOCK_D4"):
        G.tv("d4")
    assert isinstance(G, _GenericMM)


def test_tv_refuses_every_dim_outside_the_bound_tile_dims():
    from axon.codegen.ops import UnsupportedEmission

    G = _generic_mm_for_plain_matmul(tile_dims=frozenset({"m", "n", "k"}))
    for dim in ("m", "n", "k"):
        assert G.tv(dim).tile == f"TILE_{dim.upper()}"
    for dim in ("p", "d5", "k2"):
        with pytest.raises(UnsupportedEmission, match="outside the tile namespace"):
            G.tv(dim)


def _generic_mm_for_plain_matmul(tile_dims=None):
    """A `_GenericMM` over the simplest real plan, for mint-site tests."""
    from axon.codegen.bodies.matmul_generic import _GenericMM
    from axon.codegen.context import EmitCtx
    from axon.codegen.plan import NodeAttrs, build_emission_plan
    from axon.ir import Node

    m, k, n = 128, 128, 512
    nodes = [
        Node("x", "input", [], {"shape": (m, k)}, shape=(m, k)),
        Node("w", "input", [], {"shape": (k, n)}, shape=(k, n)),
        Node("xt", "nc_transpose", ["x"], {}, shape=(k, m)),
        Node("mm", "nc_matmul", ["xt", "w"], {}, shape=(m, n)),
    ]
    id_to_node = {node.id: node for node in nodes}
    compute = [node for node in nodes if node.op != "input"]
    tile_config = {"tile_m": m, "tile_k": k, "tile_n": n}
    plan = build_emission_plan(
        compute, id_to_node, {"x", "w"}, "mm", tile_config=tile_config
    )
    ctx = EmitCtx(tile_config=tile_config)
    kwargs = {} if tile_dims is None else {"tile_dims": tile_dims}
    return _GenericMM(ctx, plan, NodeAttrs(id_to_node), **kwargs)


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


def _sim(code, fn_name, args, tiles):
    import nki

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    spec_l = importlib.util.spec_from_file_location(
        f"gsim_{fn_name}_{abs(hash(code))}", path
    )
    mod = importlib.util.module_from_spec(spec_l)
    sys.modules[spec_l.name] = mod
    spec_l.loader.exec_module(mod)
    return np.asarray(
        nki.simulate(getattr(mod, fn_name))(*args, *tiles), dtype=np.float32
    )


def _constructed_attention_graph(
    *,
    m: int = 256,
    k: int = 128,
    n: int = 512,
    value_width: str = "n",
    normalize_op: str = "multiply",
    separate_query_input: bool = False,
    shared_projection_weight: bool = False,
):
    from axon.ir import Node, nuGraph

    value_extent = n if value_width == "n" else m
    nodes = [
        Node(
            "x",
            "input",
            [],
            {"sym_shape": ("m", "k")},
            shape=(m, k),
        ),
        Node(
            "wq",
            "input",
            [],
            {"sym_shape": ("k", "n")},
            shape=(k, n),
        ),
        Node(
            "wk",
            "input",
            [],
            {"sym_shape": ("k", "n")},
            shape=(k, n),
        ),
        Node(
            "wv",
            "input",
            [],
            {"sym_shape": ("k", value_width)},
            shape=(k, value_extent),
        ),
        Node("xt", "nc_transpose", ["x"], shape=(k, m)),
        Node("q", "nc_matmul", ["xt", "wq"], shape=(m, n)),
        Node("kproj", "nc_matmul", ["xt", "wk"], shape=(m, n)),
        Node(
            "v",
            "nc_matmul",
            ["xt", "wv"],
            shape=(m, value_extent),
        ),
        Node("qt", "nc_transpose", ["q"], shape=(n, m)),
        Node("kt", "nc_transpose", ["kproj"], shape=(n, m)),
        Node("scores", "nc_matmul", ["qt", "kt"], shape=(m, m)),
        Node("ex", "exponential", ["scores"], shape=(m, m)),
        Node(
            "sum",
            "tensor_reduce",
            ["ex"],
            {"op": "add", "axis": 1, "keepdims": True},
            shape=(m, 1),
        ),
        Node("inv", "reciprocal", ["sum"], shape=(m, 1)),
        Node(
            "probs",
            "tensor_scalar",
            ["ex", "inv"],
            {"op0": normalize_op, "operand0_input_index": 1},
            shape=(m, m),
        ),
        Node("probst", "nc_transpose", ["probs"], shape=(m, m)),
        Node(
            "out",
            "nc_matmul",
            ["probst", "v"],
            shape=(m, value_extent),
        ),
    ]
    input_ids = ["x", "wq", "wk", "wv"]
    if separate_query_input:
        nodes.extend(
            [
                Node(
                    "xq",
                    "input",
                    [],
                    {"sym_shape": ("m", "k")},
                    shape=(m, k),
                ),
                Node("xtq", "nc_transpose", ["xq"], shape=(k, m)),
            ]
        )
        next(node for node in nodes if node.id == "q").inputs[0] = "xtq"
        input_ids.insert(1, "xq")
    if shared_projection_weight:
        next(node for node in nodes if node.id == "kproj").inputs[1] = "wq"
        next(node for node in nodes if node.id == "v").inputs[1] = "wq"
        nodes = [node for node in nodes if node.id not in {"wk", "wv"}]
        input_ids = [node_id for node_id in input_ids if node_id not in {"wk", "wv"}]
    return nuGraph(
        nodes=nodes,
        input_ids=tuple(input_ids),
        output_ids=("out",),
    )


def _assert_attention_variant_matches_reference(
    graph,
    *,
    seed: int,
    combine=lambda ex, scalar: ex * scalar,
    scalar_op=lambda total: 1.0 / total,
):
    """Simulate an emitted attention variant against its numpy reference.

    The input order is the emitted signature's, which is ``graph.input_ids``;
    each input's shape comes from its node so a variant that adds or drops one
    still feeds the right arrays."""
    from axon.codegen.assemble import NKIEmitter

    code = NKIEmitter(
        kernel_name="attention_variant",
        tile_config={"tile_m": 64, "tile_k": 64, "tile_n": 128},
    ).emit(graph)

    rng = np.random.default_rng(seed)
    by_id = {node.id: node for node in graph.nodes}
    args = {
        input_id: rng.normal(0.0, 0.03, by_id[input_id].shape).astype(np.float32)
        for input_id in graph.input_ids
    }

    def operand(node_id):
        """One matmul operand's array: a raw input, or a transpose of one."""
        if node_id in args:
            return args[node_id]
        return args[by_id[node_id].inputs[0]].T

    def projection(node_id):
        """``nc_matmul(a, b)`` is ``aᵀ @ b``. A direct variant swaps the two
        operands, so its output is the transpose; orient it back to ``(m, n)``."""
        left, right = by_id[node_id].inputs
        value = operand(left).T @ operand(right)
        return value if value.shape[0] == by_id[node_id].shape[0] else value.T

    query = projection("q")
    keys = projection("kproj")
    values = projection("v")
    tokens = values.shape[0]
    query = query if query.shape[0] == tokens else query.T
    keys = keys if keys.shape[0] == tokens else keys.T
    exp_scores = np.exp(query @ keys.T)
    expected = (
        combine(exp_scores, scalar_op(exp_scores.sum(axis=1, keepdims=True))) @ values
    )

    actual = _sim(
        code,
        "attention_variant",
        tuple(args[input_id] for input_id in graph.input_ids),
        (1, 1, 1, 1),
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def _use_direct_attention_key_projection(graph):
    key = next(node for node in graph.nodes if node.id == "kproj")
    key_transpose = next(node for node in graph.nodes if node.id == "kt")
    scores = next(node for node in graph.nodes if node.id == "scores")
    key.inputs = list(reversed(key.inputs))
    assert key.shape is not None
    key.shape = (key.shape[1], key.shape[0])
    scores.inputs = [
        key.id if input_id == key_transpose.id else input_id
        for input_id in scores.inputs
    ]
    graph.nodes.remove(key_transpose)


def _use_direct_attention_query_projection(graph):
    query = next(node for node in graph.nodes if node.id == "q")
    query_transpose = next(node for node in graph.nodes if node.id == "qt")
    scores = next(node for node in graph.nodes if node.id == "scores")
    query.inputs = list(reversed(query.inputs))
    assert query.shape is not None
    query.shape = (query.shape[1], query.shape[0])
    scores.inputs = [
        query.id if input_id == query_transpose.id else input_id
        for input_id in scores.inputs
    ]
    graph.nodes.remove(query_transpose)


def _use_attention_activation_reduce(graph):
    reduce_node = next(node for node in graph.nodes if node.id == "sum")
    reduce_node.op = "activation_reduce"
    reduce_node.inputs = ["scores"]
    reduce_node.attrs = {
        "op": "exp",
        "reduce_op": "add",
        "bias_const": None,
        "scale": 1.0,
    }


def _use_attention_activation_reciprocal(graph):
    reciprocal = next(node for node in graph.nodes if node.id == "inv")
    reciprocal.op = "activation"
    reciprocal.attrs = {
        "op": "reciprocal",
        "bias_const": None,
        "scale": 1.0,
        "reduce_op": None,
        "reduce_cmd": "idle",
        "with_reduce": False,
    }


def _use_post_output_attention_normalization(graph):
    from axon.ir import Node

    _use_attention_activation_reduce(graph)
    _use_attention_activation_reciprocal(graph)
    by_id = {node.id: node for node in graph.nodes}
    score_transpose = Node(
        "score_t",
        "nc_transpose",
        ["scores"],
        shape=by_id["scores"].shape,
    )
    by_id["ex"].inputs = [score_transpose.id]
    by_id["out"].inputs = ["ex", "v"]
    by_id["probs"].inputs = ["out", "inv"]
    by_id["probs"].shape = by_id["out"].shape
    prefix = graph.nodes[: graph.nodes.index(by_id["scores"]) + 1]
    graph.nodes = prefix + [
        score_transpose,
        by_id["ex"],
        by_id["sum"],
        by_id["inv"],
        by_id["out"],
        by_id["probs"],
    ]
    graph.output_ids = ("probs",)


def _constructed_reduce_matmul_graph(axis):
    from axon.ir import Node, nuGraph

    nodes = [
        Node(
            "x",
            "input",
            [],
            {"sym_shape": ("m", "k")},
            shape=(256, 128),
        ),
        Node(
            "w",
            "input",
            [],
            {"sym_shape": ("k", "n")},
            shape=(128, 512),
        ),
        Node(
            "sq",
            "tensor_tensor",
            ["x", "x"],
            {"op": "multiply"},
            shape=(256, 128),
        ),
        Node(
            "sum",
            "tensor_reduce",
            ["sq"],
            {"op": "add", "axis": axis, "keepdims": True},
            shape=(256, 1),
        ),
        Node("inv", "reciprocal", ["sum"], shape=(256, 1)),
        Node(
            "scaled",
            "tensor_scalar",
            ["x", "inv"],
            {"op0": "multiply", "operand0_input_index": 1},
            shape=(256, 128),
        ),
        Node("scaled_t", "nc_transpose", ["scaled"], shape=(128, 256)),
        Node("out", "nc_matmul", ["scaled_t", "w"], shape=(256, 512)),
    ]
    return nuGraph(
        nodes=nodes,
        input_ids=("x", "w"),
        output_ids=("out",),
    )


def _constructed_qkv_scalar_post_chain_graph(m=256, k=128, n=512):
    from axon.ir import Node, nuGraph

    nodes = [
        Node(
            "x",
            "input",
            [],
            {"sym_shape": ("m", "k")},
            shape=(m, k),
        ),
        Node(
            "mlp_prev",
            "input",
            [],
            {"sym_shape": ("m", "k")},
            shape=(m, k),
        ),
        Node(
            "attention_prev",
            "input",
            [],
            {"sym_shape": ("m", "k")},
            shape=(m, k),
        ),
        Node(
            "w",
            "input",
            [],
            {"sym_shape": ("k", "n")},
            shape=(k, n),
        ),
        Node(
            "residual_1",
            "tensor_tensor",
            ["x", "mlp_prev"],
            {"op": "add"},
            shape=(m, k),
        ),
        Node(
            "residual",
            "tensor_tensor",
            ["residual_1", "attention_prev"],
            {"op": "add"},
            shape=(m, k),
        ),
        Node(
            "square",
            "tensor_tensor",
            ["residual", "residual"],
            {"op": "multiply"},
            shape=(m, k),
        ),
        Node(
            "sum",
            "tensor_reduce",
            ["square"],
            {"op": "add", "axis": 1, "keepdims": True},
            shape=(m, 1),
        ),
        Node(
            "mean",
            "tensor_scalar",
            ["sum"],
            {"op0": "multiply", "operand0_const": 1.0 / k},
            shape=(m, 1),
        ),
        Node(
            "eps",
            "tensor_scalar",
            ["mean"],
            {"op0": "add", "operand0_const": 1e-6},
            shape=(m, 1),
        ),
        Node("root", "activation", ["eps"], {"op": "sqrt"}, shape=(m, 1)),
        Node("inv", "reciprocal", ["root"], shape=(m, 1)),
        Node(
            "norm",
            "tensor_scalar",
            ["residual", "inv"],
            {"op0": "multiply", "operand0_input_index": 1},
            shape=(m, k),
        ),
        Node("norm_t", "nc_transpose", ["norm"], shape=(k, m)),
        Node("out", "nc_matmul", ["norm_t", "w"], shape=(m, n)),
    ]
    return nuGraph(
        nodes=nodes,
        input_ids=("x", "mlp_prev", "attention_prev", "w"),
        output_ids=("out",),
    )


def test_constructed_qkv_scalar_post_chain_emits_and_simulates():
    from axon.codegen import emit

    m, k, n = 256, 128, 512
    code = emit(
        _constructed_qkv_scalar_post_chain_graph(m=m, k=k, n=n),
        kernel_name="constructed_qkv_scalar_post_chain",
    )

    compile(code, "<constructed_qkv_scalar_post_chain>", "exec")
    mean_at = code.index("operand0=0.0078125")
    eps_at = code.index("operand0=1e-06")
    sqrt_at = code.index("nl.sqrt")
    reciprocal_at = code.index("nisa.reciprocal(")
    normalize_at = code.rindex("nisa.tensor_scalar(")
    matmul_at = code.index("nisa.nc_matmul(")
    assert mean_at < eps_at < sqrt_at < reciprocal_at < normalize_at < matmul_at
    assert "residual_staged = nl.ndarray(" in code
    assert "residual_1_staged" not in code
    assert not find_dead_sbuf_buffers(code)

    rng = np.random.default_rng(31)
    x = rng.normal(0.0, 0.05, (m, k)).astype(np.float32)
    mlp_prev = rng.normal(0.0, 0.05, (m, k)).astype(np.float32)
    attention_prev = rng.normal(0.0, 0.05, (m, k)).astype(np.float32)
    w = rng.normal(0.0, 0.05, (k, n)).astype(np.float32)
    residual = x + mlp_prev + attention_prev
    expected = (
        residual
        / np.sqrt(np.sum(residual * residual, axis=1, keepdims=True) / k + 1e-6)
        @ w
    )

    actual = _sim(
        code,
        "constructed_qkv_scalar_post_chain",
        (x, mlp_prev, attention_prev, w),
        (1, 1, 1),
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_staged_broadcast_keeps_four_dimensional_source_indexing():
    import re

    from axon.codegen import emit
    from axon.ir import Node, nuGraph

    m, k, n = 128, 128, 512
    graph = nuGraph(
        nodes=[
            Node("x", "input", [], shape=(m, k)),
            Node("y", "input", [], shape=(m, k)),
            Node("w", "input", [], shape=(k, n)),
            Node(
                "residual",
                "tensor_tensor",
                ["x", "y"],
                {"op": "add"},
                shape=(m, k),
            ),
            Node(
                "square",
                "tensor_tensor",
                ["residual", "residual"],
                {"op": "multiply"},
                shape=(m, k),
            ),
            Node(
                "sum",
                "tensor_reduce",
                ["square"],
                {"op": "add", "axis": 1, "keepdims": True},
                shape=(m, 1),
            ),
            Node("inv", "reciprocal", ["sum"], shape=(m, 1)),
            Node(
                "residual_alias",
                "broadcast",
                ["residual", "x"],
                shape=(m, k),
            ),
            Node("residual_t", "nc_transpose", ["residual_alias"], shape=(k, m)),
            Node("mm", "nc_matmul", ["residual_t", "w"], shape=(m, n)),
            Node(
                "out",
                "tensor_scalar",
                ["mm", "inv"],
                {"op0": "multiply", "operand0_input_index": 1},
                shape=(m, n),
            ),
        ],
        input_ids=("x", "y", "w"),
        output_ids=("out",),
    )
    code = emit(graph, kernel_name="staged_broadcast")

    assert "residual_staged = nl.ndarray(" in code
    assert re.search(
        r"residual_staged\[0:TILE_M, k, b_m,"
        r"\s+\(bk_t \* TILE_K\):\(bk_t \* TILE_K\) \+ TILE_K\]",
        code,
    )
    assert not re.search(
        r"residual_staged\[0:TILE_M,\s+\(bk_t \* TILE_K\):",
        code,
    )
    assert not find_dead_sbuf_buffers(code)

    rng = np.random.default_rng(41)
    x = rng.normal(0.0, 0.05, (m, k)).astype(np.float32)
    y = rng.normal(0.0, 0.05, (m, k)).astype(np.float32)
    w = rng.normal(0.0, 0.05, (k, n)).astype(np.float32)
    residual = x + y
    expected = (residual @ w) / np.sum(
        residual * residual,
        axis=1,
        keepdims=True,
    )
    actual = _sim(code, "staged_broadcast", (x, y, w), (1, 1, 1))

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def _find_ast_inplace_dma_mutations(code: str) -> tuple[set[str], list[int]]:
    """Find NISA calls that mutate a DMA destination while reading it."""

    def buffer_root(expr: ast.expr | None) -> str | None:
        while isinstance(expr, ast.Subscript):
            expr = expr.value
        return expr.id if isinstance(expr, ast.Name) else None

    calls = sorted(
        (
            node
            for node in ast.walk(ast.parse(code))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "nisa"
        ),
        key=lambda call: call.lineno,
    )
    dma_destinations: set[str] = set()
    loaded: set[str] = set()
    violations: list[int] = []
    for call in calls:
        dst_keyword = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "dst"),
            None,
        )
        dst_expr = dst_keyword or (call.args[0] if call.args else None)
        dst = buffer_root(dst_expr)
        if dst is None:
            continue
        if call.func.attr == "dma_copy":
            dma_destinations.add(dst)
            loaded.add(dst)
            continue
        if dst not in dma_destinations:
            continue
        source_exprs = list(call.args if dst_keyword is not None else call.args[1:])
        source_exprs.extend(
            keyword.value for keyword in call.keywords if keyword.arg != "dst"
        )
        source_roots = {
            root
            for expr in source_exprs
            for subscript in ast.walk(expr)
            if isinstance(subscript, ast.Subscript)
            if (root := buffer_root(subscript)) is not None
        }
        if dst in source_roots:
            violations.append(call.lineno)
        dma_destinations.discard(dst)
    return loaded, violations


def test_ast_dma_immutability_check_handles_multiline_calls():
    code = """
nisa.dma_copy(
    dst=loaded[0:M, 0:N],
    src=source[0:M, 0:N])
nisa.tensor_scalar(
    loaded[0:M, 0:N],
    loaded[0:M, 0:N],
    nl.multiply,
    operand0=scale)
"""

    loaded, violations = _find_ast_inplace_dma_mutations(code)

    assert loaded == {"loaded"}
    assert violations


def test_constructed_self_attention_spills_only_across_stages():
    import re

    from axon.codegen import emit

    code = emit(
        _constructed_attention_graph(),
        kernel_name="constructed_attention",
    )

    compile(code, "<constructed_attention>", "exec")
    assert _matmul_site_count(code) == 5
    assert len(re.findall(r"\bnl\.exp\b", code)) == 1
    assert re.search(r"^    P = M$", code, re.MULTILINE)
    assert "NUM_BLOCK_P = P // BLOCK_P" in code
    assert "K2" not in code
    assert "D4" not in code

    # Values a LATER stage reads still cross through HBM: this is the
    # five-matmul QKV shape, whose stage 0 holds `part=p` beside `part=m`, so
    # pattern 1b's same-part fusion legitimately refuses it.
    expected_hbm = (
        "qt_hbm",
        "kt_hbm",
        "v_hbm",
        "probst_hbm",
    )
    for hbm in expected_hbm:
        assert f"{hbm} = nl.ndarray(" in code
        assert f"dst={hbm}[" in code
        assert f"src={hbm}[" in code

    # The reducer's two values do NOT: `scores` and `ex` live entirely inside the
    # reduce stage, so pattern 1a keeps them in spanning SBUF buffers. They took
    # 26.69 us of the 62.13 us gap as HBM spills, storing and reloading a buffer
    # inside one loop iteration at the winner's tile config.
    for resident in ("scores", "ex"):
        assert f"{resident}_hbm" not in code, (
            f"{resident} still spills to HBM; pattern 1a did not land"
        )
        # The allocator can be a conditional expression: a resident reduce source
        # is also a drain target, so it takes the accumulator's
        # `(nl.ndarray if NUM_BLOCK_<D> == 1 else nl.zeros)` form.
        assert re.search(rf"^\s+{resident}_sb = \(?nl\.", code, re.MULTILINE), (
            f"{resident} has no resident SBUF buffer"
        )

    reduce_at = code.index("reduce_op=nl.add")
    reciprocal_at = code.index("nisa.reciprocal(")
    normalize_at = code.index("nisa.tensor_scalar(")
    output_at = code.rindex("nisa.nc_matmul(")
    assert reduce_at < reciprocal_at < normalize_at < output_at
    assert re.search(
        r"nisa\.tensor_scalar\([^\n]+, ex_stage_tiles\[[^\n]+,"
        r" nl\.multiply, operand0=inv_tiles\[",
        code,
    )
    assert not find_dead_sbuf_buffers(code)
    loaded, mutations = _find_ast_inplace_dma_mutations(code)
    assert loaded
    assert not mutations


def test_constructed_self_attention_simulates_multiple_blocks_on_every_axis():
    from axon.codegen.assemble import NKIEmitter

    m, k, n = 256, 256, 256
    # Every extent is 256, so tile_n=512 (the default) would refuse; 128 divides
    # each one and leaves two blocks per axis, which is what this test checks.
    code = NKIEmitter(
        kernel_name="constructed_attention_numeric",
        tile_config={"tile_m": 128, "tile_k": 128, "tile_n": 128},
    ).emit(_constructed_attention_graph(m=m, k=k, n=n))
    rng = np.random.default_rng(19)
    x = rng.normal(0.0, 0.05, (m, k)).astype(np.float32)
    wq = rng.normal(0.0, 0.05, (k, n)).astype(np.float32)
    wk = rng.normal(0.0, 0.05, (k, n)).astype(np.float32)
    wv = rng.normal(0.0, 0.05, (k, n)).astype(np.float32)

    q = x @ wq
    keys = x @ wk
    values = x @ wv
    exp_scores = np.exp(q @ keys.T)
    expected = (exp_scores / exp_scores.sum(axis=1, keepdims=True)) @ values
    actual = _sim(
        code,
        "constructed_attention_numeric",
        (x, wq, wk, wv),
        (1, 1, 1, 1),
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    ("direct_key", "fused_reduce", "activation_reciprocal"),
    itertools.product((False, True), repeat=3),
)
def test_staged_attention_numerical_pre_output_cross_product(
    direct_key,
    fused_reduce,
    activation_reciprocal,
):
    from axon.codegen.assemble import NKIEmitter

    m, k, n = 128, 64, 256
    graph = _constructed_attention_graph(m=m, k=k, n=n)
    if direct_key:
        _use_direct_attention_key_projection(graph)
    if fused_reduce:
        _use_attention_activation_reduce(graph)
    if activation_reciprocal:
        _use_attention_activation_reciprocal(graph)

    # 64 divides m and k, 128 divides n: the defaults (128/128/512) refuse here.
    code = NKIEmitter(
        kernel_name="attention_pre_output_cross_product",
        tile_config={"tile_m": 64, "tile_k": 64, "tile_n": 128},
    ).emit(graph)
    seed = 29 + direct_key + 2 * fused_reduce + 4 * activation_reciprocal
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.03, (m, k)).astype(np.float32)
    wq = rng.normal(0.0, 0.03, (k, n)).astype(np.float32)
    wk = rng.normal(0.0, 0.03, (k, n)).astype(np.float32)
    wv = rng.normal(0.0, 0.03, (k, n)).astype(np.float32)
    q = x @ wq
    keys = x @ wk
    values = x @ wv
    exp_scores = np.exp(q @ keys.T)
    expected = (exp_scores / exp_scores.sum(axis=1, keepdims=True)) @ values

    actual = _sim(
        code,
        "attention_pre_output_cross_product",
        (x, wq, wk, wv),
        (1, 1, 1, 1),
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def _attention_core_graph(s: int = 128, d: int = 128):
    """`kernels/attention_nkilib`'s two-matmul core: q, k_t, and v are inputs,
    so it plans as two stages with its reduce sharing stage 0 with a matmul."""
    from axon.ir import Node, nuGraph

    return nuGraph(
        nodes=[
            Node("q", "input", [], {"sym_shape": ("s", "d")}, shape=(s, d)),
            Node("k_t", "input", [], {"sym_shape": ("d", "s")}, shape=(d, s)),
            Node("v", "input", [], {"sym_shape": ("s", "d")}, shape=(s, d)),
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
        ],
        input_ids=("q", "k_t", "v"),
        output_ids=("out",),
    )


def test_two_stage_attention_core_numerical():
    """Its reduce sits in stage 0 beside that stage's matmul, which the old
    three-phase body never had to schedule."""
    from axon.codegen.assemble import NKIEmitter

    s, d = 128, 128
    graph = _attention_core_graph(s=s, d=d)
    # d holds role `wide`, so at the defaults its base is tile_n=512; 64/64/128
    # is the smallest config every extent here divides.
    code = NKIEmitter(
        kernel_name="attention_core",
        tile_config={"tile_m": 64, "tile_k": 64, "tile_n": 128},
    ).emit(graph)

    rng = np.random.default_rng(41)
    q = rng.normal(0.0, 0.05, (s, d)).astype(np.float32)
    k_t = rng.normal(0.0, 0.05, (d, s)).astype(np.float32)
    v = rng.normal(0.0, 0.05, (s, d)).astype(np.float32)
    exp_scores = np.exp(q @ k_t)
    expected = (exp_scores / exp_scores.sum(axis=1, keepdims=True)) @ v

    actual = _sim(code, "attention_core", (q, k_t, v), (1, 1, 1, 1))

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_staged_direct_query_projection_is_numerically_correct():
    """A query projection reaching the scores matmul with no transpose emits and
    is correct. The deleted attention pattern refused this shape."""
    graph = _constructed_attention_graph(m=128, k=64, n=256)
    _use_direct_attention_query_projection(graph)

    _assert_attention_variant_matches_reference(graph, seed=51)


def test_staged_attention_refuses_reused_direct_projection_weight():
    from axon.codegen import emit
    from axon.codegen.ops import UnsupportedEmission

    graph = _constructed_attention_graph()
    _use_direct_attention_key_projection(graph)
    next(node for node in graph.nodes if node.id == "kproj").inputs[0] = "wq"

    # Reusing wq as the direct key weight leaves two raw dims that no schedule
    # role can carry, so role resolution refuses before any emission.
    with pytest.raises(
        UnsupportedEmission,
        match="outside the fixed m/n/k/p namespace",
    ):
        emit(graph, kernel_name="unsupported_shared_attention_weight")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("op", "square"),
        ("reduce_op", "maximum"),
        ("scale", 2.0),
        ("bias_const", 1.0),
    ],
)
def test_staged_attention_refuses_mutated_activation_reduce(key, value):
    from axon.codegen import emit
    from axon.codegen.ops import UnsupportedEmission

    graph = _constructed_attention_graph()
    _use_attention_activation_reduce(graph)
    next(node for node in graph.nodes if node.id == "sum").attrs[key] = value

    with pytest.raises(
        UnsupportedEmission,
        match="exact unscaled exponential sum",
    ):
        emit(graph, kernel_name="unsupported_attention_activation_reduce")


@pytest.mark.parametrize(
    ("attrs", "scalar_op"),
    [
        ({"op": "sqrt"}, np.sqrt),
        ({"scale": 2.0}, lambda total: 1.0 / (2.0 * total)),
    ],
)
def test_staged_rendered_scalar_attributes_are_numerically_correct(attrs, scalar_op):
    """`_emit_activation` renders the scalar chain's op and scale, so a graph
    naming either emits and must compute it, rather than refusing."""
    graph = _constructed_attention_graph(m=128, k=64, n=256)
    _use_attention_activation_reciprocal(graph)
    next(node for node in graph.nodes if node.id == "inv").attrs.update(attrs)

    _assert_attention_variant_matches_reference(graph, seed=55, scalar_op=scalar_op)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("reduce_op", "add"),
        ("with_reduce", True),
        ("reduce_cmd", "not_idle"),
        ("max_value", 1.0),
    ],
)
def test_staged_attention_refuses_unrendered_scalar_attributes(key, value):
    from axon.codegen import emit
    from axon.codegen.ops import UnsupportedEmission

    graph = _constructed_attention_graph()
    _use_attention_activation_reciprocal(graph)
    next(node for node in graph.nodes if node.id == "inv").attrs[key] = value

    with pytest.raises(
        UnsupportedEmission,
        match="cannot preserve non-default ISA attributes",
    ):
        emit(graph, kernel_name="unsupported_attention_activation_reciprocal")


def test_staged_attention_post_output_normalization_reaches_the_numerator_gate():
    """Plan step 7a. Post-output normalization refused with a dedicated message
    about the crossing being a per-partition scalar. That gate is gone, so this
    QKV-shaped fixture now reaches ONE later gate and refuses there: its fused
    `activation_reduce` reads `scores`, which no independent exponential reads
    (`_use_post_output_attention_normalization` reroutes `ex` through `score_t`).

    The exact message is asserted, so any OTHER refusal fails the test. Emission
    of the orientation itself is no longer covered here."""
    from axon.codegen import emit
    from axon.codegen.ops import UnsupportedEmission

    graph = _constructed_attention_graph()
    _use_post_output_attention_normalization(graph)

    with pytest.raises(
        UnsupportedEmission,
        match=(
            r"^generic matmul: fused staged reduction requires one independent "
            r"exponential over 'scores', got \[\]$"
        ),
    ):
        emit(graph, kernel_name="post_output_attention")


@pytest.mark.parametrize(
    ("node_id", "attrs"),
    [
        ("q", {"is_transpose": True}),
        ("scores", {"perf_mode": "double_row"}),
        ("scores", {"perf_mode": "not_none"}),
        ("scores", {"perf_mode": "not.none"}),
        ("out", {"accumulate": True}),
        ("ex", {"max_value": 1.0}),
        ("ex", {"reduce_cmd": "not_idle"}),
        ("ex", {"reduce_cmd": "not.idle"}),
        ("ex", {"with_reduce": True}),
    ],
)
def test_staged_attention_refuses_unrendered_isa_attributes(node_id, attrs):
    from axon.codegen import emit
    from axon.codegen.ops import UnsupportedEmission

    graph = _constructed_attention_graph()
    next(node for node in graph.nodes if node.id == node_id).attrs.update(attrs)

    with pytest.raises(
        UnsupportedEmission,
        match="cannot preserve non-default ISA attributes",
    ):
        emit(graph, kernel_name="unsupported_attention_attributes")


def test_staged_default_enum_matching_is_exact():
    """One spelling per default enum value: the member itself, which recorded
    graphs carry, or its bare member name, which hand-built graphs carry."""
    from enum import Enum

    from axon.codegen.bodies.matmul_generic import _has_only_default_isa_attrs
    from axon.isa_semantics import matmul_perf_mode, reduce_cmd

    class OtherDefault(Enum):
        none = "none"
        idle = "idle"

    def matmul_ok(key, value):
        return not _has_only_default_isa_attrs("nc_matmul", {key: value})

    def activation_ok(key, value):
        return not _has_only_default_isa_attrs("exponential", {key: value})

    for value in (matmul_perf_mode.none, "none"):
        assert matmul_ok("perf_mode", value)
    for value in (reduce_cmd.idle, "idle"):
        assert activation_ok("reduce_cmd", value)

    # A same-named member of another enum is a different value, and so is any
    # other spelling of the default's name.
    for value in (None, "not.none", "matmul_perf_mode.none", OtherDefault.none):
        assert not matmul_ok("perf_mode", value)
    for value in (None, "not.idle", "reduce_cmd.idle", OtherDefault.idle):
        assert not activation_ok("reduce_cmd", value)


def test_staged_concrete_defaults_reject_explicit_none():
    from axon.codegen.bodies.matmul_generic import _has_only_default_isa_attrs

    for key in (
        "is_moving_onezero",
        "is_stationary_onezero",
        "is_transpose",
        "tile_position",
        "tile_size",
        "perf_mode",
    ):
        assert _has_only_default_isa_attrs("nc_matmul", {key: None}) == {key: None}
    for key in ("max_value", "reduce_init", "reduce_cmd", "with_reduce"):
        assert _has_only_default_isa_attrs("exponential", {key: None}) == {key: None}

    assert not _has_only_default_isa_attrs("nc_matmul", {"accumulate": None})
    assert not _has_only_default_isa_attrs("nc_matmul", {"name": None})
    assert not _has_only_default_isa_attrs("exponential", {"name": None})

    # An attribute the op's emitter renders is not this check's business; one it
    # drops with no known default refuses rather than passing unexamined.
    assert not _has_only_default_isa_attrs("tensor_scalar", {"op0": "multiply"})
    assert _has_only_default_isa_attrs("tensor_scalar", {"anything": 7}) == {
        "anything": 7
    }
    # `engine` is dropped by every combiner emitter, so a non-default refuses.
    for combiner in ("tensor_scalar", "tensor_tensor", "tensor_copy"):
        assert _has_only_default_isa_attrs(combiner, {"engine": "vector"}) == {
            "engine": "vector"
        }
        assert not _has_only_default_isa_attrs(combiner, {"engine": "unknown"})


@pytest.mark.parametrize("axis", [-1, (-1,), [-1]])
def test_negative_free_axis_emits_from_normalized_reduce_plan(axis):
    from axon.codegen import emit

    code = emit(
        _constructed_reduce_matmul_graph(axis),
        kernel_name="normalized_negative_reduce_axis",
    )

    assert "nisa.tensor_reduce(" in code
    assert "axis=[1]" in code


def test_keep_dims_alias_emits_validated_keepdims_true():
    from axon.codegen import emit

    graph = _constructed_reduce_matmul_graph(1)
    reduce_node = next(node for node in graph.nodes if node.id == "sum")
    reduce_node.attrs["keep_dims"] = reduce_node.attrs.pop("keepdims")

    code = emit(graph, kernel_name="keep_dims_reduce")

    assert "nisa.tensor_reduce(" in code
    assert "keepdims=True" in code


@pytest.mark.parametrize(
    "attrs",
    [
        {"keepdims": False},
        {"negate": True},
    ],
)
def test_staged_attention_refuses_unrendered_reduction_attributes(attrs):
    from axon.codegen import emit
    from axon.codegen.ops import UnsupportedEmission

    graph = _constructed_attention_graph()
    next(node for node in graph.nodes if node.id == "sum").attrs.update(attrs)

    with pytest.raises(
        UnsupportedEmission,
        match="keepdims=True|negate=False",
    ):
        emit(graph, kernel_name="unsupported_attention_reduction")


def test_staged_attention_refuses_fifth_extent():
    from axon.codegen.assemble import NKIEmitter
    from axon.codegen.ops import UnsupportedEmission

    # The value width is m (256) while n is 512, so tile_n must divide 256 for
    # tile resolution to pass and the FIFTH-EXTENT refusal to be the one raised.
    with pytest.raises(
        UnsupportedEmission,
        match="fixed m/n/k/p|extra schedule axis",
    ):
        NKIEmitter(
            kernel_name="unsupported_cross_attention",
            tile_config={"tile_m": 128, "tile_k": 128, "tile_n": 256},
        ).emit(_constructed_attention_graph(value_width="d"))


def test_staged_additive_normalization_is_numerically_correct():
    """The driver emits whatever combiner the graph names, not just multiply:
    nothing in the schedule cares which, so an additive combine is correct."""
    graph = _constructed_attention_graph(m=128, k=64, n=256, normalize_op="add")

    _assert_attention_variant_matches_reference(
        graph, seed=52, combine=lambda ex, scalar: ex + scalar
    )


def test_staged_distinct_query_source_is_numerically_correct():
    """Cross-attention: the query projection reads its own activation input,
    rather than one token input shared across Q, K, and V."""
    graph = _constructed_attention_graph(m=128, k=64, n=256, separate_query_input=True)

    _assert_attention_variant_matches_reference(graph, seed=53)


def test_staged_attention_temp_hbm_names_do_not_shadow_inputs():
    from axon.codegen import emit

    graph = _constructed_attention_graph()
    weight = next(node for node in graph.nodes if node.id == "wq")
    weight.id = "qt_hbm"
    next(node for node in graph.nodes if node.id == "q").inputs[1] = weight.id
    graph.input_ids = ("x", weight.id, "wk", "wv")

    code = emit(graph, kernel_name="attention_name_collision")

    assert "def attention_name_collision(x, qt_hbm, wk, wv," in code
    assert "qt_hbm_2 = nl.ndarray(" in code
    assert "src=qt_hbm[" in code


def test_staged_shared_projection_weight_is_numerically_correct():
    """One weight feeding all three projections emits, with and without a
    separate query input; the three weights need not be distinct."""
    for separate_query in (False, True):
        graph = _constructed_attention_graph(
            m=128,
            k=64,
            n=256,
            separate_query_input=separate_query,
            shared_projection_weight=True,
        )
        _assert_attention_variant_matches_reference(graph, seed=54)


def test_activation_reduce_matmul_preamble_preserves_fused_op():
    from axon.codegen import emit
    from axon.ir import Node, nuGraph
    from axon.isa_semantics import nl

    nodes = [
        Node("x", "input", [], shape=(128, 512)),
        Node("w", "input", [], shape=(512, 1024)),
        Node(
            "red",
            "activation_reduce",
            ["x"],
            {"op": nl.square, "reduce_op": nl.add, "bias_const": None},
            shape=(128, 1),
        ),
        Node(
            "scaled",
            "tensor_scalar",
            ["x", "red"],
            {"op0": nl.multiply, "operand0_input_index": 1},
            shape=(128, 512),
        ),
        Node("transposed", "nc_transpose", ["scaled"], shape=(512, 128)),
        Node("mm", "nc_matmul", ["transposed", "w"], shape=(128, 1024)),
    ]
    graph = nuGraph(
        nodes=nodes,
        input_ids=("x", "w"),
        output_ids=("mm",),
    )
    code = emit(graph, kernel_name="activation_reduce_matmul")
    assert "nisa.activation_reduce(" in code
    assert "nl.square" in code
    assert "nl.add" in code
    assert "reduce_res=partial_red[0:TILE_M, 0:1]" in code


def _shared_hbm_allocs(code):
    """Every `nl.shared_hbm` allocation's target name, parsed not grepped."""
    names = []
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not any(
            kw.arg == "buffer"
            and isinstance(kw.value, ast.Attribute)
            and kw.value.attr == "shared_hbm"
            for kw in node.value.keywords
        ):
            continue
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            names.append(node.targets[0].id)
    return names


# --- same-part stage fusion for a crossing value (plan step 5) --------------- #


def test_qkv_shape_refuses_fusion_and_stays_correct():
    """The five-matmul QKV plan does NOT qualify: its stage 0 holds `part=p`
    beside `part=m`, so no single `part` loop spans the lifetime. It keeps its
    HBM crossing, which is correct and costs a microsecond rather than a wrong
    answer."""
    from axon.codegen import emit

    code = emit(_constructed_attention_graph(), kernel_name="qkv_no_fusion")
    compile(code, "<qkv_no_fusion>", "exec")
    hbm = _shared_hbm_allocs(code)
    assert "probst_hbm" in hbm, (
        "the QKV shape must keep its cross-stage HBM buffer; fusion was applied "
        f"where the part dims differ. shared_hbm: {hbm}"
    )
    # The reducer's two same-stage values are still resident (pattern 1a holds
    # regardless of whether 1b's fusion applies).
    assert "scores_hbm" not in hbm
    assert "ex_hbm" not in hbm
    assert not find_dead_sbuf_buffers(code)


def test_fusion_rule_reads_every_stage_not_just_the_endpoints():
    """The lifetime case the rule exists for. A crossing value can be live across
    a stage that neither produces nor consumes it, and that stage's loops sit
    inside the hoisted one too, so `_fusible_crossing_values` must check EVERY
    stage's part dims.

    Driven directly rather than through a graph, because the discriminating input
    is a middle stage that disagrees while the endpoints agree: a rule reading
    only the endpoints would accept it and emit a buffer that is dead by the time
    the consumer runs."""
    from axon.codegen.bodies.matmul_generic import _fusible_crossing_values

    class _FakeGroupsGM:
        """Just enough of `_GenericMM` for the prepass: per-stage part dims and
        one crossing value with a free axis."""

        def __init__(self, part_dims_by_stage):
            self._parts = part_dims_by_stage

        def layout(self, node_id):
            class _L:
                part = "m"
                free = "n"

            return _L()

        def dim_name(self, dv):
            return dv

    def _run(part_dims_by_stage):
        import axon.codegen.bodies.matmul_generic as mg

        stages = tuple(
            type("S", (), {"index": i, "matmul_ids": ()})()
            for i in range(len(part_dims_by_stage))
        )
        G = _FakeGroupsGM(part_dims_by_stage)
        original = mg._stage_groups
        mg._stage_groups = lambda _G, stage: {
            (part, "n", "k"): [] for part in part_dims_by_stage[stage.index]
        }
        try:
            return _fusible_crossing_values(
                G, stages, {"xv": 2}, {"xv": 0}, already_resident=[]
            )
        finally:
            mg._stage_groups = original

    # Endpoints agree AND the middle stage agrees: fusible.
    part_dim, ids = _run([["m"], ["m"], ["m"]])
    assert part_dim == "m" and ids == ["xv"]

    # Endpoints agree but the MIDDLE stage does not. A rule reading only the
    # producer and the last consumer would wrongly accept this.
    part_dim, ids = _run([["m"], ["p"], ["m"]])
    assert part_dim is None and ids == [], (
        "fusion was accepted despite an intermediate stage on a different part "
        "dim; the hoisted loop would not span that stage's nest"
    )

    # One stage holding two part dims at once is the QKV shape's refusal.
    part_dim, ids = _run([["m", "p"], ["m"], ["m"]])
    assert part_dim is None and ids == []


# --- the wide-role guard (plan step 7b) ------------------------------------- #


def test_reducer_is_correct_across_several_psum_subtiles_per_block():
    """The reducer walks the whole `BLOCK_<D>`, which can span several PSUM
    sub-tiles of width `MATMUL_TILE_<D>`.

    At `TILES_IN_BLOCK_N=4` with `MATMUL_TILE_N=128` the drain fills four PSUM
    sub-tiles per block, so a reducer summing only one of them sums a quarter of
    each row. No other test simulates the two-matmul core at that count."""
    from axon.codegen.assemble import NKIEmitter

    s, d = 512, 128
    tiles = (1, 4, 1, 1)
    code = NKIEmitter(
        kernel_name="attention_core",
        tile_config={"tile_m": 128, "tile_k": 128, "tile_n": 128},
    ).emit(_attention_core_graph(s=s, d=d))

    rng = np.random.default_rng(3)
    q = rng.normal(0.0, 0.05, (s, d)).astype(np.float32)
    k_t = rng.normal(0.0, 0.05, (d, s)).astype(np.float32)
    v = rng.normal(0.0, 0.05, (s, d)).astype(np.float32)
    scores = np.exp(q @ k_t)
    expected = (scores / scores.sum(axis=1, keepdims=True)) @ v

    actual = _sim(code, "attention_core", (q, k_t, v), tiles)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
