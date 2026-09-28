"""Focused generic matmul codegen tests for transparent broadcast nodes."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from axon.codegen import emit
from axon.codegen.assemble import NKIEmitter
from axon.codegen.ops import UnsupportedEmission
from axon.ir import Node, nuGraph
from axon.isa_semantics import nl

_M = 32
_K = 64
_N = 128
# The plan resolves each TILE_<D> to the configured base and refuses a base that
# does not divide the extent, so these small extents need their own config.
_TILE_CONFIG = {"tile_m": _M, "tile_k": _K, "tile_n": _N}
_STRATEGIES = ("same_loop", "separate_loop", "load_transpose2d")


def _emit_tiled(
    graph: nuGraph, kernel_name: str, strategy: str = "separate_loop"
) -> str:
    return NKIEmitter(
        kernel_name=kernel_name,
        tile_config=_TILE_CONFIG,
        rhs_transpose_strategy=strategy,
    ).emit(graph)


def _input(node_id: str, shape: tuple[int, int]) -> Node:
    return Node(node_id, "input", [], {"shape": shape}, shape=shape)


def _graph(kind: str, *, reverse0: bool = False) -> nuGraph:
    x = _input("x", (_M, _K))
    y = _input("y", (_M, _K))
    w = _input("w", (_K, _N))
    red = Node(
        "red",
        "tensor_reduce",
        ["y"],
        {"op": nl.add, "axis": 1, "keepdims": True},
        shape=(_M, 1),
    )

    if kind == "scalar":
        nodes = [
            x,
            y,
            w,
            red,
            Node(
                "red_broadcast",
                "broadcast",
                ["red", "x"],
                shape=(_M, _K),
            ),
            Node(
                "scaled",
                "tensor_tensor",
                ["red_broadcast", "x"],
                {"op": nl.multiply},
                shape=(_M, _K),
            ),
            Node("scaled_t", "nc_transpose", ["scaled"], shape=(_K, _M)),
            Node("mm", "nc_matmul", ["scaled_t", "w"], shape=(_M, _N)),
        ]
        output_id = "mm"
    elif kind == "operand":
        nodes = [
            x,
            y,
            w,
            red,
            Node(
                "scaled",
                "tensor_scalar",
                ["x", "red"],
                {"op0": nl.multiply, "operand0_input_index": 1},
                shape=(_M, _K),
            ),
            Node(
                "operand_broadcast",
                "broadcast",
                ["scaled", "x"],
                shape=(_M, _K),
            ),
            Node(
                "scaled_t",
                "nc_transpose",
                ["operand_broadcast"],
                shape=(_K, _M),
            ),
            Node("mm", "nc_matmul", ["scaled_t", "w"], shape=(_M, _N)),
        ]
        output_id = "mm"
    elif kind == "post":
        nodes = [
            x,
            y,
            w,
            Node("x_t", "nc_transpose", ["x"], shape=(_K, _M)),
            Node("mm", "nc_matmul", ["x_t", "w"], shape=(_M, _N)),
            red,
            Node(
                "scaled",
                "tensor_scalar",
                ["mm", "red"],
                {"op0": nl.multiply, "operand0_input_index": 1},
                shape=(_M, _N),
            ),
            Node(
                "post_broadcast",
                "broadcast",
                ["scaled", "mm"],
                shape=(_M, _N),
            ),
        ]
        output_id = "post_broadcast"
    elif kind == "reversed":
        nodes = [
            x,
            y,
            w,
            red,
            Node(
                "scaled",
                "tensor_scalar",
                ["red", "x"],
                {
                    "op0": nl.divide,
                    "operand0_input_index": 1,
                    "reverse0": reverse0,
                },
                shape=(_M, _K),
            ),
            Node("scaled_t", "nc_transpose", ["scaled"], shape=(_K, _M)),
            Node("mm", "nc_matmul", ["scaled_t", "w"], shape=(_M, _N)),
        ]
        output_id = "mm"
    else:
        raise AssertionError(f"unexpected graph kind {kind!r}")

    return nuGraph(
        nodes=nodes,
        input_ids=("x", "y", "w"),
        output_ids=(output_id,),
    )


@pytest.mark.parametrize(
    "kind,forbidden_buffer",
    [
        ("scalar", "red_broadcast_tiles"),
        ("operand", "operand_broadcast_tiles"),
        ("post", "post_broadcast_post"),
    ],
)
def test_generic_matmul_aliases_broadcast_without_materializing_it(
    kind: str,
    forbidden_buffer: str,
) -> None:
    code = _emit_tiled(_graph(kind), f"matmul_broadcast_{kind}")

    compile(code, f"<matmul_broadcast_{kind}>", "exec")
    assert forbidden_buffer not in code
    assert "red_tiles" in code
    assert "nisa.nc_matmul" in code


def test_full_tile_broadcast_rejects_mismatched_shape() -> None:
    graph = _graph("operand")
    broadcast = next(n for n in graph.nodes if n.id == "operand_broadcast")
    broadcast.shape = (_M, _K + 1)

    with pytest.raises(UnsupportedEmission, match="does not exactly equal data shape"):
        emit(graph, kernel_name="mismatched_full_tile_broadcast")


def test_scalar_broadcast_rejects_mismatched_combiner_shape() -> None:
    graph = _graph("scalar")
    broadcast = next(n for n in graph.nodes if n.id == "red_broadcast")
    broadcast.shape = (_M, _K + 1)

    with pytest.raises(
        UnsupportedEmission,
        match="does not exactly equal full-tile shape",
    ):
        emit(graph, kernel_name="mismatched_scalar_broadcast")


@pytest.mark.parametrize("source_shape", [(_M,), (_M, 2)])
def test_scalar_broadcast_rejects_non_partition_scalar_source(
    source_shape: tuple[int, ...],
) -> None:
    graph = _graph("scalar")
    source = next(n for n in graph.nodes if n.id == "red")
    source.shape = source_shape

    with pytest.raises(
        UnsupportedEmission,
        match="rank-2 shape with singleton free dimension",
    ):
        emit(graph, kernel_name="invalid_scalar_broadcast_source")


@pytest.mark.parametrize(
    "kind,kernel_name",
    [
        ("scalar", "reduce_broadcast_mul"),
        ("operand", "reduce_mul_broadcast"),
    ],
)
def test_reduction_broadcast_kernel_simulates_numerically(
    kind: str,
    kernel_name: str,
    tmp_path,
) -> None:
    import nki

    rng = np.random.default_rng(11)
    x = rng.standard_normal((_M, _K)).astype(np.float32)
    y = rng.standard_normal((_M, _K)).astype(np.float32)
    w = rng.standard_normal((_K, _N)).astype(np.float32)
    expected = (x * y.sum(axis=1, keepdims=True)) @ w
    codes = [
        _emit_tiled(_graph(kind), kernel_name, strategy) for strategy in _STRATEGIES
    ]
    assert len(codes) == 3

    for schedule_index, code in enumerate(codes):
        module_name = f"{kernel_name}_{schedule_index}"
        module_path = tmp_path / f"{module_name}.py"
        module_path.write_text(code)
        module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        result = nki.simulate(getattr(module, kernel_name))(x, y, w, 1, 1, 1)

        np.testing.assert_allclose(
            np.asarray(result, dtype=np.float32),
            expected,
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.parametrize("reverse0", [False, True])
def test_reversed_tensor_scalar_simulates_numerically(
    reverse0: bool,
    tmp_path,
) -> None:
    import nki

    kernel_name = f"reversed_tensor_scalar_{int(reverse0)}"
    code = _emit_tiled(_graph("reversed", reverse0=reverse0), kernel_name)
    assert ("reverse0=True" in code) is not reverse0

    module_path = tmp_path / f"{kernel_name}.py"
    module_path.write_text(code)
    module_spec = importlib.util.spec_from_file_location(kernel_name, module_path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)

    rng = np.random.default_rng(19)
    x = rng.uniform(0.5, 1.5, size=(_M, _K)).astype(np.float32)
    y = rng.uniform(0.5, 1.5, size=(_M, _K)).astype(np.float32)
    w = rng.standard_normal((_K, _N)).astype(np.float32)
    reduced = y.sum(axis=1, keepdims=True)
    scaled = x / reduced if reverse0 else reduced / x
    expected = scaled @ w
    result = nki.simulate(getattr(module, kernel_name))(x, y, w, 1, 1, 1)

    np.testing.assert_allclose(
        np.asarray(result, dtype=np.float32),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )
