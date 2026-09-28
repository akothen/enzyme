"""Focused generic-codegen tests for activation_reduce."""

from __future__ import annotations

import re

import numpy as np
import pytest

from axon.codegen import emit
from axon.codegen.ops import UnsupportedEmission
from axon.codegen.structure import classify_kernel
from axon.ir import Node, nuGraph
from axon.isa_semantics import nl

_M = 128
_K = 512
_N = 1024


def _activation_reduce_graph(
    *,
    reduce_op,
    auxiliary: str | None = None,
    auxiliary_shape: tuple[int, int] | None = None,
) -> nuGraph:
    x = Node("x", "input", [], {"shape": (_M, _K)}, shape=(_M, _K))
    w = Node("w", "input", [], {"shape": (_K, _N)}, shape=(_K, _N))
    nodes = [x]
    input_ids = ["x"]
    reduce_inputs = ["x"]
    reduce_attrs = {
        "op": nl.square,
        "reduce_op": reduce_op,
        "bias_const": None,
        "scale": 1.0,
    }
    if auxiliary is not None:
        assert auxiliary_shape is not None
        aux = Node(
            "aux",
            "input",
            [],
            {"shape": auxiliary_shape},
            shape=auxiliary_shape,
        )
        nodes.append(aux)
        input_ids.append("aux")
        reduce_inputs.append("aux")
        reduce_attrs[f"{auxiliary}_input_index"] = 1
        if auxiliary == "bias":
            reduce_attrs.pop("bias_const")
        elif auxiliary == "scale":
            reduce_attrs.pop("scale")
        else:
            raise AssertionError(f"unexpected auxiliary kind {auxiliary!r}")
    nodes.extend(
        [
            w,
            Node(
                "red",
                "activation_reduce",
                reduce_inputs,
                reduce_attrs,
                shape=(_M, 1),
            ),
            Node(
                "scaled",
                "tensor_scalar",
                ["x", "red"],
                {"op0": nl.multiply, "operand0_input_index": 1},
                shape=(_M, _K),
            ),
            Node("transposed", "nc_transpose", ["scaled"], shape=(_K, _M)),
            Node("mm", "nc_matmul", ["transposed", "w"], shape=(_M, _N)),
        ]
    )
    input_ids.append("w")
    return nuGraph(nodes=nodes, input_ids=tuple(input_ids), output_ids=("mm",))


def _emit_activation_reduce(**kwargs) -> str:
    code = emit(
        _activation_reduce_graph(**kwargs),
        kernel_name="activation_reduce_codegen",
    )
    compile(code, "<activation_reduce_codegen>", "exec")
    return code


def test_activation_reduce_uses_additive_cross_block_monoid():
    code = _emit_activation_reduce(reduce_op=nl.add)
    assert "red_tiles = nl.zeros(" in code
    assert "nl.square, x_red_chain_tile_k" in code
    assert ", nl.add, reduce_res=partial_red" in code
    assert re.search(
        r"nisa\.tensor_tensor\(red_tiles.*?op=nl\.add\)",
        code,
        re.DOTALL,
    )


def test_generic_matmul_refuses_non_add_activation_reduce():
    with pytest.raises(UnsupportedEmission, match="unsupported reduction op"):
        _emit_activation_reduce(reduce_op=nl.maximum)


def _reduce_body_graph(reduce_op) -> nuGraph:
    """A reduce-body (non-matmul) graph: activation_reduce then a scalar apply."""
    return nuGraph(
        nodes=[
            Node("x", "input", [], {"shape": (_M, _K)}, shape=(_M, _K)),
            Node(
                "red",
                "activation_reduce",
                ["x"],
                {
                    "op": nl.square,
                    "reduce_op": reduce_op,
                    "bias_const": None,
                    "scale": 1.0,
                },
                shape=(_M, 1),
            ),
            Node(
                "scaled",
                "tensor_scalar",
                ["x", "red"],
                {"op0": nl.multiply, "operand0_input_index": 1},
                shape=(_M, _K),
            ),
        ],
        input_ids=("x",),
        output_ids=("scaled",),
    )


def test_reduce_body_refuses_non_add_activation_reduce():
    assert classify_kernel(list(_reduce_body_graph(nl.maximum).nodes)[1:]) == "reduce"
    with pytest.raises(UnsupportedEmission, match="zeros identity"):
        emit(_reduce_body_graph(nl.maximum), kernel_name="reduce_body_non_add")


def test_reduce_body_emits_additive_activation_reduce():
    code = emit(_reduce_body_graph(nl.add), kernel_name="reduce_body_add")
    compile(code, "<reduce_body_add>", "exec")
    assert "accum = nl.zeros(" in code
    assert "op=nl.add)" in code


# LayerNorm hidden dim for the two-reduce test. 1024 > TILE_N (512), so the
# emitted reduce body must accumulate across NUM_BLOCK_N >= 2 free-axis blocks
# -- exactly where a body that only handles one reduction gets the variance
# wrong (it would sum centered^2 per block instead of over the whole row).
_LN_N = 1024
_LN_EPS = 1e-6


def _layernorm_graph(n: int = _LN_N) -> nuGraph:
    """A faithful bare-LayerNorm reduce-body graph with TWO free-axis reductions:
    the mean's sum, then the variance's (centered-square) sum, which depends on
    the mean, then the full normalize (var, +eps, sqrt, reciprocal, scale). This
    is the shape ``analyze_reduce_structure`` handles wrong today -- see
    celeste_docs/layernorm_two_reduce_bug.md. The constants below bake in 1/n, so
    keep the simulated input's hidden dim equal to ``n``."""
    return nuGraph(
        nodes=[
            Node("x", "input", [], {"shape": (_M, n)}, shape=(_M, n)),
            # reduce #1: row sum, then * (1/n) -> mean
            Node(
                "sum1",
                "tensor_reduce",
                ["x"],
                {"op": nl.add, "axis": 1, "keepdims": True},
                shape=(_M, 1),
            ),
            Node(
                "mean",
                "tensor_scalar",
                ["sum1"],
                {"op0": nl.multiply, "operand0_const": 1.0 / n},
                shape=(_M, 1),
            ),
            # centered = x - mean (broadcast the (M, 1) mean over the free axis)
            Node(
                "centered",
                "tensor_scalar",
                ["x", "mean"],
                {"op0": nl.subtract, "operand0_input_index": 1},
                shape=(_M, n),
            ),
            # reduce #2: sum(centered^2) -> variance's sum (depends on reduce #1)
            Node(
                "sum2",
                "activation_reduce",
                ["centered"],
                {
                    "op": nl.square,
                    "reduce_op": nl.add,
                    "bias_const": None,
                    "scale": 1.0,
                },
                shape=(_M, 1),
            ),
            # var = sum2 * (1/n); +eps; sqrt; reciprocal -> 1/std
            Node(
                "var",
                "tensor_scalar",
                ["sum2"],
                {"op0": nl.multiply, "operand0_const": 1.0 / n},
                shape=(_M, 1),
            ),
            Node(
                "vareps",
                "tensor_scalar",
                ["var"],
                {"op0": nl.add, "operand0_const": _LN_EPS},
                shape=(_M, 1),
            ),
            Node("std", "activation", ["vareps"], {"op": nl.sqrt}, shape=(_M, 1)),
            Node("rstd", "activation", ["std"], {"op": nl.reciprocal}, shape=(_M, 1)),
            # out = centered * (1/std)
            Node(
                "out",
                "tensor_scalar",
                ["centered", "rstd"],
                {"op0": nl.multiply, "operand0_input_index": 1},
                shape=(_M, n),
            ),
        ],
        input_ids=("x",),
        output_ids=("out",),
    )


def _layernorm_ref(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    centered = x - mean
    var = (centered * centered).mean(axis=1, keepdims=True)
    return centered / np.sqrt(var + _LN_EPS)


def test_reduce_body_computes_layernorm_with_two_reductions():
    """The n-reduce body computes LayerNorm correctly. We emit a two-reduce
    (LayerNorm) graph and *run the emitted NKI on the CPU simulator*, comparing
    to a NumPy reference. This is a real correctness gate: a string check that
    only asserts "not the flat body" is fooled by naively relaxing the reduce
    guard (that emits a tiled-looking body that computes the variance wrong), but
    the simulated numbers are not.

    The reduce body's lowering is hand-written and the SMT equivalence check
    never sees it, so only numerics catch a wrong body -- hence simulation rather
    than a source grep. See celeste_docs/layernorm_two_reduce_bug.md."""
    nki_sim = pytest.importorskip("nki").simulate

    graph = _layernorm_graph()
    assert classify_kernel(list(graph.nodes)[1:]) == "reduce"

    code = emit(graph, kernel_name="layernorm_two_reduce")
    compile(code, "<layernorm_two_reduce>", "exec")
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102 - emitted kernel under test
    kernel = namespace["layernorm_two_reduce"]

    rng = np.random.default_rng(0)
    x = rng.standard_normal((_M, _LN_N)).astype(np.float32)
    expected = _layernorm_ref(x)

    # tile params (TILES_IN_BLOCK_M, TILES_IN_BLOCK_N) = (1, 1) -> BLOCK_N=512,
    # NUM_BLOCK_N=2, so the variance reduction must span two free-axis blocks.
    try:
        got = np.asarray(nki_sim(kernel)(x, 1, 1), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - a broken lowering may not simulate
        raise AssertionError(
            f"emitted LayerNorm kernel failed to simulate: {type(exc).__name__}: {exc}"
        ) from exc

    assert got.shape == expected.shape, (
        f"emitted kernel returned {got.shape}, expected {expected.shape} "
        "(the flat fallback returns a (128, 1) tile)"
    )
    assert np.allclose(got, expected, rtol=2e-2, atol=2e-2), (
        f"emitted kernel does not compute LayerNorm; max abs err "
        f"{np.abs(got - expected).max():.4g}"
    )


def test_reduce_body_refuses_unrecognized_reduce_instead_of_flat():
    """A reduce graph that neither the single-reduce nor the n-reduce body
    recognizes (here a bare ``sum(x)`` with no apply) must raise, not silently
    emit the degenerate flat (128, 1) body that computes the wrong result."""
    graph = nuGraph(
        nodes=[
            Node("x", "input", [], {"shape": (_M, _K)}, shape=(_M, _K)),
            Node(
                "s",
                "tensor_reduce",
                ["x"],
                {"op": nl.add, "axis": 1, "keepdims": True},
                shape=(_M, 1),
            ),
        ],
        input_ids=("x",),
        output_ids=("s",),
    )
    assert classify_kernel(list(graph.nodes)[1:]) == "reduce"
    with pytest.raises(UnsupportedEmission, match="unrecognized reduction structure"):
        emit(graph, kernel_name="bare_reduce")


def test_reduce_body_refuses_multiple_inputs_instead_of_aliasing():
    """A reduce over more than one input (here ``sum(x * y)``) must raise: the
    reduce bodies load a single (M, N) tensor and would silently alias the extra
    inputs onto it, producing a wrong kernel with no error."""
    graph = nuGraph(
        nodes=[
            Node("x", "input", [], {"shape": (_M, _K)}, shape=(_M, _K)),
            Node("y", "input", [], {"shape": (_M, _K)}, shape=(_M, _K)),
            Node(
                "xy", "tensor_tensor", ["x", "y"], {"op": nl.multiply}, shape=(_M, _K)
            ),
            Node(
                "s",
                "tensor_reduce",
                ["xy"],
                {"op": nl.add, "axis": 1, "keepdims": True},
                shape=(_M, 1),
            ),
            Node(
                "out",
                "tensor_scalar",
                ["x", "s"],
                {"op0": nl.multiply, "operand0_input_index": 1},
                shape=(_M, _K),
            ),
        ],
        input_ids=("x", "y"),
        output_ids=("out",),
    )
    assert classify_kernel(list(graph.nodes)[2:]) == "reduce"
    with pytest.raises(UnsupportedEmission, match="has 2 inputs"):
        emit(graph, kernel_name="two_input_reduce")


@pytest.mark.parametrize("auxiliary", ["bias", "scale"])
@pytest.mark.parametrize(
    "shape,allocation",
    [
        ((_M, _K), "(TILE_M, TILES_IN_BLOCK_M, BLOCK_K)"),
        ((_M, 1), "(TILE_M, TILES_IN_BLOCK_M, 1)"),
        ((1, _K), "(1, BLOCK_K)"),
        ((1, 1), "(1, 1)"),
    ],
)
def test_activation_reduce_loads_tensor_auxiliary_at_its_broadcast_shape(
    auxiliary,
    shape,
    allocation,
):
    code = _emit_activation_reduce(
        reduce_op=nl.add,
        auxiliary=auxiliary,
        auxiliary_shape=shape,
    )
    assert re.search(
        rf"aux_red_chain_tile_k = nl\.ndarray\(\s*{re.escape(allocation)},",
        code,
    )
    source_match = re.search(r"src=aux\[(.*?)\]\)", code, re.DOTALL)
    assert source_match is not None
    source = source_match.group(1)
    if shape[1] == 1:
        assert "BLOCK_K" not in source
        assert "0:1" in source
    else:
        assert "BLOCK_K * k_side" in source
    if shape[0] == 1:
        assert "bm_red" not in source
        auxiliary_slice = "0:1"
    else:
        assert "bm_red" in source
        auxiliary_slice = "0:TILE_M, bm_red"
    call_match = re.search(
        rf"{auxiliary}=aux_red_chain_tile_k\[(.*?)\]",
        code,
    )
    assert call_match is not None
    assert auxiliary_slice in call_match.group(1)
    assert ("0:1" if shape[1] == 1 else "0:BLOCK_K") in call_match.group(1)
