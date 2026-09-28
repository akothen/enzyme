"""Host tests for the matmul operand-role invariant (codegen soundness class).

The old tiled matmul emitter once re-derived each `nc_matmul`'s
stationary/moving operands from structural heuristics that could silently
disagree with the graph node's operand roles, emitting a wrong computation
(`part_relu_mlp`/`part_silu_mlp`: `w2ᵀ@w1` instead of `x@w2`; `matmul_transpose`:
`x@w` instead of `(x@w)ᵀ`). These tests pin the fix: the operand-role invariant
helper trips on a swapped emit and passes on a faithful one.

All host-only (no device): emission is pure. Correctness *values* are checked on
device separately; here we assert the emit-time invariant that guards the class.
"""

from __future__ import annotations

import pytest

from axon.codegen.bodies.matmul_generic import (
    _assert_mm_operand_roles,
    _input_ancestors,
)
from axon.codegen.ops import UnsupportedEmission
from axon.ir import Node


# ---- unit: the invariant helper itself -----------------------------------
def _toy_graph():
    # x @ w2 with x transposed:  nc_matmul(stat=nc_transpose(x), mov=w2)
    x = Node("x", "input", [])
    w1 = Node("w1", "input", [])
    w2 = Node("w2", "input", [])
    xt = Node("xt", "nc_transpose", ["x"])
    mm = Node("mm", "nc_matmul", ["xt", "w2"])
    id_to_node = {n.id: n for n in (x, w1, w2, xt, mm)}
    input_ids = {"x", "w1", "w2"}
    return mm, id_to_node, input_ids


def test_input_ancestors_follows_all_inputs():
    _, id_to_node, input_ids = _toy_graph()
    # a fused add of two inputs has both as ancestors
    tt = Node("tt", "tensor_tensor", ["x", "w1"])
    id_to_node["tt"] = tt
    assert _input_ancestors("tt", id_to_node, input_ids) == {"x", "w1"}
    assert _input_ancestors("xt", id_to_node, input_ids) == {"x"}


def test_operand_role_invariant_passes_on_faithful_emit():
    mm, id_to_node, input_ids = _toy_graph()
    # faithful: stationary sources x (via xt), moving sources w2
    _assert_mm_operand_roles(mm, "x", "w2", id_to_node, input_ids)  # no raise


def test_operand_role_invariant_trips_on_swapped_emit():
    mm, id_to_node, input_ids = _toy_graph()
    # Bug-A-shaped swap: emit stationary<-w2, moving<-w1 (neither matches roles)
    with pytest.raises(UnsupportedEmission, match="disagree with"):
        _assert_mm_operand_roles(mm, "w2", "w1", id_to_node, input_ids)
    # Bug-B-shaped swap: stationary/moving flipped
    with pytest.raises(UnsupportedEmission, match="disagree with"):
        _assert_mm_operand_roles(mm, "w2", "x", id_to_node, input_ids)
