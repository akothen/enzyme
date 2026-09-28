from __future__ import annotations

import builtins
import heapq
import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import product as _iproduct
from typing import Any

import z3

from axon.egraph.codec import CodecError
from axon.ir import (
    _FIRST_INPUT_SHAPE_OPS,
    Node,
    _format_shape,
    _graph_symbolic_tensors,
    _sym_expr_from_graph_node,
    nuGraph,
)
from axon.isa_semantics import (
    _NODE_IDS,
    _SEMANTICS,
    _SYNTHESIS_STATS_LOCK,
    _Z3_LOCK,
    ShapeExpr,
    SymExpr,
    SymTensor,
    _NormSketch,
    _operand_to_expr,
    _to_dim,
    check_valid_and_equivalent,
    engine,
    matmul_perf_mode,
    nl,
    reduce_cmd,
)

_SKETCH_OP_ALIASES = {"transpose": "nc_transpose"}


def canonical_isa_op(op: str) -> str:
    """Resolve a sketch-level op name to its canonical ISA op name."""
    return _SKETCH_OP_ALIASES.get(op, op)


_LAYOUT_TRANSFORM_OPS: frozenset[str] = frozenset(
    {
        "transpose",
    }
)

_OP_CONSTITUENTS: dict[str, frozenset[str]] = {
    "add": frozenset({"add"}),
    "div": frozenset({"divide", "multiply"}),
    "divide": frozenset({"divide", "multiply"}),
    "exp": frozenset({"exp"}),
    "matmul": frozenset({"multiply", "add"}),
    "mul": frozenset({"multiply"}),
    "multiply": frozenset({"multiply"}),
    "reduce_sum": frozenset({"add"}),
    "relu": frozenset({"relu", "maximum"}),
    "silu": frozenset({"silu"}),
    "softmax": frozenset({"exp", "add", "divide"}),
    "cumsum": frozenset({"add", "scan"}),
    "tensor_scalar_cumulative": frozenset({"add", "scan"}),
    "sqrt": frozenset({"sqrt"}),
    "subtract": frozenset({"subtract"}),
    "transpose": frozenset({"transpose"}),
    "rms_norm": frozenset({"multiply", "add", "sqrt", "divide"}),
    "activation": frozenset(
        {
            "relu",
            "silu",
            "gelu",
            "tanh",
            "sigmoid",
            "sqrt",
            "copy",
            "exp",
            "divide",
        }
    ),
    "activation_reduce": frozenset(
        {
            "relu",
            "silu",
            "gelu",
            "tanh",
            "sigmoid",
            "sqrt",
            "copy",
            "exp",
            "add",
            "multiply",
        }
    ),
    "dma_copy": frozenset({"copy"}),
    "dma_transpose": frozenset({"transpose", "copy"}),
    "exponential": frozenset({"exp"}),
    "nc_matmul": frozenset({"multiply", "add"}),
    "nc_transpose": frozenset({"transpose", "copy"}),
    "reciprocal": frozenset({"divide"}),
    "scalar_tensor_tensor": frozenset(
        {"add", "multiply", "subtract", "maximum", "minimum"}
    ),
    "tensor_copy": frozenset({"copy"}),
    "tensor_partition_reduce": frozenset({"add"}),
    "tensor_reduce": frozenset({"add"}),
    "tensor_scalar": frozenset({"add", "multiply", "subtract", "maximum"}),
    "tensor_tensor": frozenset(
        {
            "add",
            "multiply",
            "divide",
            "subtract",
            "maximum",
            "minimum",
        }
    ),
}


def _op_constituents(op_name: str) -> frozenset[str]:
    return _OP_CONSTITUENTS.get(op_name, frozenset({op_name}))


# Map activation names to the arithmetic operations that they perform.
_ACT_OP_CONSTITUENTS: dict[str, frozenset[str]] = {
    "reciprocal": frozenset({"divide"}),
    "square": frozenset({"multiply"}),
}


def _act_op_constituents(name: str) -> frozenset[str]:
    return _ACT_OP_CONSTITUENTS.get(name, frozenset({name}))


def _node_actual_constituents(node: Node) -> frozenset[str]:
    """The constituents *actually* used by this node, narrowed by attrs.

    `_op_constituents` returns the set of operators an op *can* perform; this
    walks the node's `op`/`op0`/`op1` attrs to pick out the operators it
    *does* perform. Used by the simplifier to compute a tight required-set
    (e.g., a `tensor_tensor` node with `op=add` only requires `add`, not the
    full {add, mul, divide, ...} envelope).
    """
    if node.op == "tensor_tensor":
        return frozenset({_operand_to_expr(node.attrs.get("op", nl.add))})
    if node.op == "tensor_scalar":
        constituents = {_operand_to_expr(node.attrs.get("op0", nl.multiply))}
        op1 = node.attrs.get("op1")
        if op1 is not None:
            constituents.add(_operand_to_expr(op1))
        return frozenset(constituents)
    if node.op == "scalar_tensor_tensor":
        return frozenset(
            {
                _operand_to_expr(node.attrs.get("op0", nl.multiply)),
                _operand_to_expr(node.attrs.get("op1", nl.add)),
            }
        )
    if node.op in ("tensor_reduce", "tensor_partition_reduce"):
        return frozenset({_operand_to_expr(node.attrs.get("op", nl.add))})
    if node.op == "activation":
        return _act_op_constituents(_operand_to_expr(node.attrs.get("op", nl.copy)))
    if node.op == "activation_reduce":
        return _act_op_constituents(_operand_to_expr(node.attrs.get("op", nl.copy))) | {
            _operand_to_expr(node.attrs.get("reduce_op", nl.add))
        }
    if node.op == "tensor_scalar_cumulative":
        return frozenset(
            {
                _operand_to_expr(node.attrs.get("op0", nl.add)),
                _operand_to_expr(node.attrs.get("op1", nl.add)),
                "scan",
            }
        )
    return _op_constituents(node.op)


def _shares_constituents(target_op: str, hw_op: str) -> bool:
    return bool(_op_constituents(target_op) & _op_constituents(hw_op))


_HOLE_SENTINEL = object()


@dataclass
class SketchNode:
    hole: bool
    op: str  # "INPUT" | hw op name
    children: list[SketchNode]  # operands (may contain holes)
    attrs: dict[str, Any]  # fixed op attributes
    sym: SymTensor | None  # evaluated SymTensor (None if unresolved)

    def has_hole(self) -> bool:
        if self.hole:
            return True
        return any(c.has_hole() for c in self.children)

    def hw_size(self) -> int:
        if self.hole:
            return 0
        if self.op == "INPUT":
            return 0
        return 1 + builtins.sum(c.hw_size() for c in self.children)

    def _key(self) -> Any:
        if self.hole:
            return ("HOLE",)
        if self.op == "INPUT":
            return ("INPUT", id(self.sym))
        return (
            self.op,
            tuple(sorted((k, repr(v)) for k, v in self.attrs.items() if k != "name")),
            tuple(c._key() for c in self.children),
        )

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SketchNode) and self._key() == other._key()

    @staticmethod
    def make_hole() -> SketchNode:
        return SketchNode(hole=True, op="HOLE", children=[], attrs={}, sym=None)

    @staticmethod
    def make_input(sym: SymTensor) -> SketchNode:
        return SketchNode(hole=False, op="INPUT", children=[], attrs={}, sym=sym)

    @staticmethod
    def make_op(
        op: str, children: list[SketchNode], attrs: dict[str, Any]
    ) -> SketchNode:
        return SketchNode(
            hole=False, op=op, children=list(children), attrs=dict(attrs), sym=None
        )


def _fill_first_hole(sketch: SketchNode, replacement: SketchNode) -> SketchNode | None:
    if sketch.hole:
        return replacement
    if sketch.op == "INPUT":
        return None
    new_children: list[SketchNode] = []
    filled = False
    for child in sketch.children:
        if not filled:
            result = _fill_first_hole(child, replacement)
            if result is not None:
                new_children.append(result)
                filled = True
                new_children.extend(sketch.children[len(new_children) :])
                break
        new_children.append(child)
    if not filled:
        return None
    return SketchNode(
        hole=False,
        op=sketch.op,
        children=new_children,
        attrs=dict(sketch.attrs),
        sym=None,
    )


# Generic symbolic evaluation of ISA operations via registered semantics.


class SymEvalRejection:
    """The singleton result for an invalid sketch candidate."""

    _INSTANCE: SymEvalRejection | None = None

    def __new__(cls) -> SymEvalRejection:
        if cls._INSTANCE is None:
            cls._INSTANCE = super().__new__(cls)
        return cls._INSTANCE

    def __repr__(self) -> str:
        return "SymEvalRejection()"


SYM_EVAL_REJECTED = SymEvalRejection()

# The evaluable ISA inventory with each operation's minimum operand count.
# Membership gates sketch evaluation; do not derive it from ``@semantics_hw``.
_SYM_EVAL_MIN_OPERANDS: dict[str, int] = {
    "activation": 1,
    "activation_reduce": 1,
    "broadcast": 2,
    "broadcast_to": 1,
    "dma_copy": 1,
    "dma_transpose": 1,
    "exponential": 1,
    "load": 1,
    "nc_matmul": 2,
    "nc_transpose": 1,
    "reciprocal": 1,
    "scalar_tensor_tensor": 2,
    "store": 1,
    "tensor_copy": 1,
    "tensor_partition_reduce": 1,
    "tensor_reduce": 1,
    "tensor_scalar": 1,
    "tensor_scalar_cumulative": 1,
    "tensor_tensor": 2,
}

# Lowering passthrough operations evaluate over their raw decoded attributes
# and a fixed name, keeping the exact-identity passthrough verdict unchanged.
_RAW_ATTR_SYM_EVAL_OPS: frozenset[str] = frozenset(
    {"broadcast", "broadcast_to", "load", "store"}
)

# Every sketch op with a symbolic evaluator: the evaluable inventory plus
# the sketch-level aliases (currently ``transpose`` -> ``nc_transpose``).
_CODEC_EVALUABLE_OPS: frozenset[str] = frozenset(_SYM_EVAL_MIN_OPERANDS) | {"transpose"}


def _required_operand_count(op: str, attrs: dict[str, Any]) -> int:
    """Minimum operand count; a tensor operand0 needs one more child."""
    if op == "tensor_scalar" and attrs.get("operand0_const") is None:
        return 2
    if op == "scalar_tensor_tensor" and "operand0_const" not in attrs:
        return 3
    return _SYM_EVAL_MIN_OPERANDS[op]


def isa_symbolic_eval(op: str, input_syms: list[Any], attrs: dict[str, Any]) -> Any:
    """Evaluate one ISA operation, or return ``SYM_EVAL_REJECTED``."""
    op = canonical_isa_op(op)
    if op not in _SYM_EVAL_MIN_OPERANDS:
        raise CodecError(f"operation '{op}' has no symbolic evaluator")
    ins = list(input_syms)
    if len(ins) < _required_operand_count(op, attrs):
        return SYM_EVAL_REJECTED
    if op in _RAW_ATTR_SYM_EVAL_OPS:
        node_attrs = dict(attrs)
        name = f"_isaeval_{op}"
    else:
        node_attrs = sketch_node_attrs(op, len(ins), attrs)
        name = attrs.get("name") or _NODE_IDS.next_name(op)
        if op in _FIRST_INPUT_SHAPE_OPS and op in _SEMANTICS:
            # Same SymTensor _sym_expr_from_graph_node builds, minus the
            # Node round-trip; these ops' out dims are the data operand's.
            expr = SymExpr(
                op, [i.expr for i in ins], tuple(ins[0].shape), node_attrs, name
            )
            return SymTensor(name, expr=expr)
    node = Node(
        id=name,
        op=op,
        inputs=[f"_c{i}" for i in range(len(ins))],
        attrs=node_attrs,
    )
    return _sym_expr_from_graph_node(node, ins)


def _invoke_hw_op(
    op: str, input_syms: list[SymTensor], attrs: dict[str, Any]
) -> SymTensor | None:
    """Evaluate one sketch operation, or return ``None`` for a rejected candidate."""
    if op not in _CODEC_EVALUABLE_OPS:
        return None
    result = isa_symbolic_eval(op, list(input_syms), attrs)
    if isinstance(result, SymEvalRejection):
        return None
    return result


def _eval_sketch(sketch: SketchNode) -> SymTensor | None:
    if sketch.hole:
        return None
    if sketch.op == "INPUT":
        return sketch.sym
    child_syms: list[SymTensor] = []
    for child in sketch.children:
        s = _eval_sketch(child)
        if s is None:
            return None
        child_syms.append(s)
    return _invoke_hw_op(sketch.op, child_syms, sketch.attrs)


def _template_constituents(sketch: SketchNode) -> frozenset[str]:
    if sketch.hole or sketch.op == "INPUT":
        return frozenset()
    if sketch.op == "activation":
        return _act_op_constituents(_operand_to_expr(sketch.attrs.get("op", nl.copy)))
    if sketch.op == "activation_reduce":
        return _act_op_constituents(
            _operand_to_expr(sketch.attrs.get("op", nl.copy))
        ) | {_operand_to_expr(sketch.attrs.get("reduce_op", nl.add))}
    if sketch.op in ("dma_copy", "tensor_copy"):
        return frozenset({"copy"})
    if sketch.op in ("dma_transpose", "nc_transpose"):
        return frozenset({"transpose", "copy"})
    if sketch.op == "transpose":
        return frozenset({"transpose"})
    if sketch.op == "exponential":
        return frozenset({"exp"})
    if sketch.op == "nc_matmul":
        return frozenset({"multiply", "add"})
    if sketch.op == "reciprocal":
        return frozenset({"divide"})
    if sketch.op in ("tensor_partition_reduce", "tensor_reduce"):
        return frozenset({_operand_to_expr(sketch.attrs.get("op", nl.add))})
    if sketch.op == "tensor_scalar":
        constituents = {_operand_to_expr(sketch.attrs.get("op0", nl.multiply))}
        op1 = sketch.attrs.get("op1")
        if op1 is not None:
            constituents.add(_operand_to_expr(op1))
        return frozenset(constituents)
    if sketch.op == "scalar_tensor_tensor":
        return frozenset(
            {
                _operand_to_expr(sketch.attrs.get("op0", nl.multiply)),
                _operand_to_expr(sketch.attrs.get("op1", nl.add)),
            }
        )
    if sketch.op == "tensor_tensor":
        return frozenset({_operand_to_expr(sketch.attrs.get("op", nl.add))})
    if sketch.op == "tensor_scalar_cumulative":
        return frozenset(
            {
                _operand_to_expr(sketch.attrs.get("op0", nl.add)),
                _operand_to_expr(sketch.attrs.get("op1", nl.add)),
                "scan",
            }
        )
    return _op_constituents(sketch.op)


_PoolTemplateFn = Callable[[list["SketchNode"], dict[str, Any]], list["SketchNode"]]


def _unary_pool_templates(
    hw_op: str, concrete: list[SketchNode], attrs: dict[str, Any]
) -> list[SketchNode]:
    H = SketchNode.make_hole
    templates = [SketchNode.make_op(hw_op, [H()], attrs)]
    for n1 in concrete:
        templates.append(SketchNode.make_op(hw_op, [n1], attrs))
    return templates


def _binary_pool_templates(
    hw_op: str, concrete: list[SketchNode], attrs: dict[str, Any]
) -> list[SketchNode]:
    H = SketchNode.make_hole
    templates: list[SketchNode] = []
    ordered_pairs = list(_iproduct(concrete, concrete))
    ordered_pairs.sort(key=lambda pair: pair[0] == pair[1])
    for n1, n2 in ordered_pairs:
        templates.append(
            SketchNode.make_op(
                hw_op,
                [SketchNode.make_op("transpose", [n1], {}), n2],
                attrs,
            )
        )
        templates.append(
            SketchNode.make_op(
                hw_op,
                [n1, SketchNode.make_op("transpose", [n2], {})],
                attrs,
            )
        )
    templates.append(SketchNode.make_op(hw_op, [H(), H()], attrs))
    for n1 in concrete:
        templates.append(SketchNode.make_op(hw_op, [n1, H()], attrs))
        templates.append(SketchNode.make_op(hw_op, [H(), n1], attrs))
        for n2 in concrete:
            templates.append(SketchNode.make_op(hw_op, [n1, n2], attrs))
    return templates


def _activation_pool_templates(
    concrete: list[SketchNode], target_attrs: dict[str, Any]
) -> list[SketchNode]:
    templates: list[SketchNode] = []
    ta_constituents = _op_constituents(target_attrs.get("op_name", ""))
    # `nl.reciprocal` is deliberately absent. The Activation engine's reciprocal
    # has a limited domain: measured on trn2 it is accurate to ~1e-7 up to 1e12,
    # then returns exactly 0.0 (1e14 and above). `nisa.reciprocal` on the Vector
    # engine is correct over the whole range, so 1/x must lower to that op.
    #
    # The solver cannot reject the activation form on its own: `isa_semantics`
    # models this function as exact 1/x over the reals with no side condition, and
    # Axon proves over reals without value ranges, so the equality is provable and
    # the hardware limit is invisible. Nor can the simulator catch it -- it returns
    # the exact value, so only a device run disagrees.
    #
    # It broke `attention_nkilib`: a bare-exp softmax puts denominators at
    # 1e15..1e20, the reciprocal returned 0, and every variant carrying it produced
    # zeros (4 of 8 hardware variants, all wrong at max_abs_err=0.792 against a
    # 5e-2 tolerance). `sqrt` and `exp` were probed on the same range and have no
    # such cliff, so this is specific to reciprocal, not to the activation table.
    #
    # An agent post-process may still try this substitution, because it validates
    # empirically -- but only with a probe over the operand's real range, since the
    # failure is data-dependent and nominal test inputs can hide it.
    act_ops = [
        nl.copy,
        nl.relu,
        nl.silu,
        nl.gelu,
        nl.tanh,
        nl.sigmoid,
        nl.sqrt,
    ]
    for act_op in act_ops:
        op_str = _operand_to_expr(act_op)
        if not ta_constituents or (_act_op_constituents(op_str) & ta_constituents):
            templates.extend(
                _unary_pool_templates("activation", concrete, {"op": act_op})
            )
    return templates


def _activation_reduce_pool_templates(
    concrete: list[SketchNode], target_attrs: dict[str, Any]
) -> list[SketchNode]:
    templates: list[SketchNode] = []
    for act_op in [nl.copy, nl.relu, nl.exp, nl.square]:
        templates.extend(
            _unary_pool_templates(
                "activation_reduce", concrete, {"op": act_op, "reduce_op": nl.add}
            )
        )
    return templates


def _exponential_pool_templates(
    concrete: list[SketchNode], target_attrs: dict[str, Any]
) -> list[SketchNode]:
    return _unary_pool_templates("exponential", concrete, {})


def _nc_matmul_pool_templates(
    concrete: list[SketchNode], target_attrs: dict[str, Any]
) -> list[SketchNode]:
    return _binary_pool_templates("nc_matmul", concrete, {})


def _reciprocal_pool_templates(
    concrete: list[SketchNode], target_attrs: dict[str, Any]
) -> list[SketchNode]:
    return _unary_pool_templates("reciprocal", concrete, {})


def _scalar_tensor_tensor_pool_templates(
    concrete: list[SketchNode], target_attrs: dict[str, Any]
) -> list[SketchNode]:
    # Fuses tensor_scalar(data, op0, operand0) -> tensor_tensor(prev, op1, operand1)
    # into one Vector-Engine pass. `data` is a (P,F) tile; `operand0` is
    # either a compile-time constant or a (P,1) per-row scalar tensor;
    # `operand1` is a (P,F) tile. This template fires from the post-lowering
    # simplifier, where it collapses two adjacent hw ops into one.
    #
    # The simplifier surfaces the producer/consumer ops via
    # target_attrs.{producer_op0, consumer_op}; we only enumerate that
    # one (op0, op1) pair to keep the pool tight (this op participates
    # in `_build_general_simplification_pool`'s cross-product over all
    # candidates). If the producer/consumer aren't tensor_scalar /
    # tensor_tensor, the simplifier won't surface those attrs and we
    # fall back to (multiply, add) -- the dominant fusion pattern in
    # practice; widening costs synthesis-time and rarely wins (e.g.,
    # `relu` -> `+y` would need a separate `activation+tensor_tensor`
    # fusion target, which is a follow-up).
    hw_op = "scalar_tensor_tensor"
    H = SketchNode.make_hole
    templates: list[SketchNode] = []
    producer_op0 = target_attrs.get("producer_op0")
    consumer_op = target_attrs.get("consumer_op")
    op0_choices = (producer_op0,) if producer_op0 is not None else (nl.multiply,)
    op1_choices = (consumer_op,) if consumer_op is not None else (nl.add,)
    # Constant-scalar form: data is a hole or a concrete tile, operand1 is a
    # hole or another concrete tile. The constant comes from the upstream
    # tensor_scalar producer's `operand0_const` attr, which the simplifier
    # surfaces via target_attrs.
    scalar_const = target_attrs.get("operand0_const")
    if scalar_const is not None and isinstance(scalar_const, (int, float)):
        for op0 in op0_choices:
            for op1 in op1_choices:
                base_attrs: dict[str, Any] = {
                    "op0": op0,
                    "op1": op1,
                    "operand0_const": float(scalar_const),
                }
                templates.append(SketchNode.make_op(hw_op, [H(), H()], base_attrs))
                for n1 in concrete:
                    templates.append(SketchNode.make_op(hw_op, [n1, H()], base_attrs))
                    templates.append(SketchNode.make_op(hw_op, [H(), n1], base_attrs))
                    for n2 in concrete:
                        templates.append(
                            SketchNode.make_op(hw_op, [n1, n2], base_attrs)
                        )
    # Tensor-scalar form: operand0 is a per-row scalar tensor (e.g., (P,1)
    # tile). The synthesizer can pick any concrete input as operand0; the
    # broadcast shape rule in `scalar_tensor_tensor`'s semantics already
    # admits (P,1) -> (P,F) broadcasts.
    for op0 in op0_choices:
        for op1 in op1_choices:
            base_attrs2: dict[str, Any] = {"op0": op0, "op1": op1}
            templates.append(SketchNode.make_op(hw_op, [H(), H(), H()], base_attrs2))
            for n0 in concrete:
                templates.append(SketchNode.make_op(hw_op, [n0, H(), H()], base_attrs2))
                for n1 in concrete:
                    templates.append(
                        SketchNode.make_op(hw_op, [n0, n1, H()], base_attrs2)
                    )
                    templates.append(
                        SketchNode.make_op(hw_op, [n0, H(), n1], base_attrs2)
                    )
                    for n2 in concrete:
                        templates.append(
                            SketchNode.make_op(hw_op, [n0, n1, n2], base_attrs2)
                        )
    return templates


def _tensor_partition_reduce_pool_templates(
    concrete: list[SketchNode], target_attrs: dict[str, Any]
) -> list[SketchNode]:
    return _unary_pool_templates("tensor_partition_reduce", concrete, {"op": nl.add})


def _tensor_reduce_pool_templates(
    concrete: list[SketchNode], target_attrs: dict[str, Any]
) -> list[SketchNode]:
    ta_axis = target_attrs.get("axis", target_attrs.get("keep_dims_axis"))
    ta_kd = bool(target_attrs.get("keepdims", target_attrs.get("keep_dims", False)))
    axes_to_try = [ta_axis] if ta_axis is not None else [0, 1]
    kd_to_try = [ta_kd] if ta_axis is not None else [False, True]
    templates: list[SketchNode] = []
    for ax in axes_to_try:
        for kd in kd_to_try:
            templates.extend(
                _unary_pool_templates(
                    "tensor_reduce",
                    concrete,
                    {"op": nl.add, "axis": ax, "keepdims": kd},
                )
            )
    return templates


def _tensor_scalar_pool_templates(
    concrete: list[SketchNode], target_attrs: dict[str, Any]
) -> list[SketchNode]:
    hw_op = "tensor_scalar"
    H = SketchNode.make_hole
    templates: list[SketchNode] = []
    for op0 in [nl.add, nl.multiply, nl.subtract]:
        for n1 in concrete:
            templates.append(SketchNode.make_op(hw_op, [n1, H()], {"op0": op0}))
            templates.append(SketchNode.make_op(hw_op, [H(), n1], {"op0": op0}))
            for n2 in concrete:
                templates.append(SketchNode.make_op(hw_op, [n1, n2], {"op0": op0}))
        templates.append(SketchNode.make_op(hw_op, [H(), H()], {"op0": op0}))
    # Compile-time-constant scalar fold: when the source op (e.g. graph
    # `mul`/`add`/`sub`) carries a scalar attr, lower it to
    # tensor_scalar(data, op0, operand0_const=<scalar>) so kernels with
    # baked-in hyperparams (Adam beta/eps, etc.) synthesize.
    scalar_const = target_attrs.get("scalar", target_attrs.get("operand0_const"))
    if scalar_const is not None and isinstance(scalar_const, (int, float)):
        target_op_name = target_attrs.get("op_name", "")
        const_op_pairs = [
            ("add", nl.add),
            ("subtract", nl.subtract),
            ("mul", nl.multiply),
            ("multiply", nl.multiply),
        ]
        scalar_attrs = {
            op0: {"op0": op0, "operand0_const": float(scalar_const)}
            for name, op0 in const_op_pairs
            if name == target_op_name
        }
        for attrs_const in scalar_attrs.values():
            if target_attrs.get("reverse"):
                attrs_const = {**attrs_const, "reverse0": True}
            for n1 in concrete:
                templates.append(SketchNode.make_op(hw_op, [n1], attrs_const))
            templates.append(SketchNode.make_op(hw_op, [H()], attrs_const))
    for n1 in concrete:
        templates.append(
            SketchNode.make_op(hw_op, [n1], {"op0": nl.maximum, "operand0_const": 0})
        )
    templates.append(
        SketchNode.make_op(hw_op, [H()], {"op0": nl.maximum, "operand0_const": 0})
    )
    # Dual-op tensor_scalar: collapses two adjacent tensor_scalar calls
    # into a single one (`(data <op0> operand0) <op1> operand1`). The
    # simplifier surfaces producer/consumer ops + (when applicable)
    # constants via target_attrs.{producer_op0, producer_operand0_const,
    # consumer_op, consumer_operand0_const}. operand0/operand1 may each
    # be a compile-time constant or a (P,1) per-row scalar tile -- NKI's
    # tensor_scalar accepts both. Enumerate the four combinations the
    # simplifier might see; skip if the (op0, op1) pair isn't surfaced.
    prod_op0 = target_attrs.get("producer_op0")
    cons_op = target_attrs.get("consumer_op")
    if prod_op0 is not None and cons_op is not None:
        prod_const = target_attrs.get("producer_operand0_const")
        cons_const = target_attrs.get("consumer_operand0_const")
        base = {"op0": prod_op0, "op1": cons_op}
        # Both constants -- folds Adam's `denom = sqrt*c + eps` style chain.
        if isinstance(prod_const, (int, float)) and isinstance(
            cons_const, (int, float)
        ):
            a = {
                **base,
                "operand0_const": float(prod_const),
                "operand1_const": float(cons_const),
            }
            templates.append(SketchNode.make_op(hw_op, [H()], a))
            for n1 in concrete:
                templates.append(SketchNode.make_op(hw_op, [n1], a))
        # Producer-tile, consumer-const -- folds `(data * tile) + c`.
        if isinstance(cons_const, (int, float)):
            a = {**base, "operand1_const": float(cons_const)}
            for n0 in concrete:
                templates.append(SketchNode.make_op(hw_op, [H(), n0], a))
                for n1 in concrete:
                    templates.append(SketchNode.make_op(hw_op, [n1, n0], a))
        # Producer-const, consumer-tile -- folds `(data * c) + tile`.
        if isinstance(prod_const, (int, float)):
            a = {**base, "operand0_const": float(prod_const)}
            for n1 in concrete:
                templates.append(SketchNode.make_op(hw_op, [H(), n1], a))
                for n2 in concrete:
                    templates.append(SketchNode.make_op(hw_op, [n2, n1], a))
        # Both tiles -- folds `(data * tile0) + tile1`. Two children; both
        # used as operand0 / operand1 (not data). Indexes set in
        # `sketch_node_attrs`.
        if not isinstance(prod_const, (int, float)) and not isinstance(
            cons_const, (int, float)
        ):
            a = dict(base)
            for n0 in concrete:
                for n1 in concrete:
                    templates.append(SketchNode.make_op(hw_op, [H(), n0, n1], a))
                    for n2 in concrete:
                        templates.append(SketchNode.make_op(hw_op, [n2, n0, n1], a))
    return templates


def _tensor_scalar_cumulative_pool_templates(
    concrete: list[SketchNode], target_attrs: dict[str, Any]
) -> list[SketchNode]:
    # cumsum lowering: data + 0, then op1=add accumulates along free dim.
    return _unary_pool_templates(
        "tensor_scalar_cumulative",
        concrete,
        {"op0": nl.add, "op1": nl.add, "imm0_const": 0},
    )


def _tensor_tensor_pool_templates(
    concrete: list[SketchNode], target_attrs: dict[str, Any]
) -> list[SketchNode]:
    # nisa.tensor_tensor does NOT actually support op=divide on TRN; the
    # codegen lowers the divide case via reciprocal + multiply (see
    # `_emit_tensor_tensor` in axon.codegen). Keeping it in the synthesis
    # pool lets `div` graph nodes pick a single tensor_tensor sketch.
    templates: list[SketchNode] = []
    for op in [nl.add, nl.multiply, nl.divide, nl.subtract, nl.maximum, nl.minimum]:
        templates.extend(_binary_pool_templates("tensor_tensor", concrete, {"op": op}))
    return templates


def _transpose_pool_templates(
    concrete: list[SketchNode], target_attrs: dict[str, Any]
) -> list[SketchNode]:
    # default (swap first two dims)
    return _unary_pool_templates("transpose", concrete, {})


# The ordered synthesis-candidate registry: each pool operation is defined
# beside its template generator. Iteration order IS the pool and fusion
# enumeration order and must stay stable.
_POOL_TEMPLATE_REGISTRY: dict[str, _PoolTemplateFn] = {
    "activation": _activation_pool_templates,
    "activation_reduce": _activation_reduce_pool_templates,
    "exponential": _exponential_pool_templates,
    "nc_matmul": _nc_matmul_pool_templates,
    "reciprocal": _reciprocal_pool_templates,
    "scalar_tensor_tensor": _scalar_tensor_tensor_pool_templates,
    "tensor_partition_reduce": _tensor_partition_reduce_pool_templates,
    "tensor_reduce": _tensor_reduce_pool_templates,
    "tensor_scalar": _tensor_scalar_pool_templates,
    "tensor_scalar_cumulative": _tensor_scalar_cumulative_pool_templates,
    "tensor_tensor": _tensor_tensor_pool_templates,
    "transpose": _transpose_pool_templates,
}

# Pool names and fusion enumeration order derive from the registry keys.
ISA_POOL_OP_NAMES: tuple[str, ...] = tuple(_POOL_TEMPLATE_REGISTRY)
ISA_POOL_OP_NAMES_SET: frozenset[str] = frozenset(ISA_POOL_OP_NAMES)


def _pool_templates_for_hw_op(
    hw_op: str,
    concrete: list[SketchNode],
    target_attrs: dict[str, Any],
) -> list[SketchNode]:
    generator = _POOL_TEMPLATE_REGISTRY.get(hw_op)
    return generator(concrete, target_attrs) if generator is not None else []


def _build_synthesis_pool(
    target_op: str,
    target_attrs: dict[str, Any],
    input_syms: list[SymTensor],
) -> list[SketchNode]:
    concrete = [SketchNode.make_input(sym) for sym in input_syms]
    pool: list[SketchNode] = list(concrete)  # include raw inputs
    target_constituents = _op_constituents(target_op)

    augmented_attrs = dict(target_attrs)
    augmented_attrs["op_name"] = target_op

    for hw_op in ISA_POOL_OP_NAMES:
        is_layout = hw_op in _LAYOUT_TRANSFORM_OPS
        if not is_layout and not _shares_constituents(target_op, hw_op):
            continue
        templates = _pool_templates_for_hw_op(hw_op, concrete, augmented_attrs)
        if not is_layout:
            templates = [
                template
                for template in templates
                if _template_constituents(template).issubset(target_constituents)
            ]
        pool.extend(templates)

    return pool


def _check_equivalent_quiet(
    lhs: SymTensor,
    rhs: SymTensor,
    timeout: int = 3000,
) -> bool:
    try:
        with _Z3_LOCK:
            return check_valid_and_equivalent(lhs, rhs, timeout=timeout).proved
    except Exception:
        return False


def _shapes_match_exactly(lhs: SymTensor, rhs: SymTensor) -> bool:
    if lhs.rank != rhs.rank:
        return False
    with _Z3_LOCK:
        return builtins.all(
            z3.eq(ldim, rdim) for ldim, rdim in zip(lhs.shape, rhs.shape, strict=True)
        )


def _shapes_incompatible_symbolically(lhs: SymTensor, rhs: SymTensor) -> bool:
    if lhs.rank != rhs.rank:
        return True
    with _Z3_LOCK:
        for ldim, rdim in zip(lhs.shape, rhs.shape, strict=True):
            if (
                z3.is_int_value(ldim)
                and z3.is_int_value(rdim)
                and ldim.as_long() != rdim.as_long()
            ):
                return True
    return False


def _sketch_shape_constraints_violated(candidate_sym: SymTensor) -> bool:
    visited: set[int] = set()

    def check_node(expr: SymExpr) -> bool:
        key = id(expr)
        if key in visited:
            return False
        visited.add(key)

        if expr.op == "input":
            return False

        for inp in expr.inputs:
            if check_node(inp):
                return True

        entry = _SEMANTICS.get(expr.op)
        if entry is None:
            return False

        input_shapes = [ShapeExpr(list(inp.shape)) for inp in expr.inputs]
        attrs_for_check = {k: v for k, v in expr.attrs.items() if k != "out_shape"}
        try:
            constraints = list(
                entry.shape_rule(input_shapes, attrs_for_check).ctx.facts
            )
            constraints.extend(entry.validity_rule(input_shapes, attrs_for_check).facts)
            for constraint in constraints:
                if z3.is_false(z3.simplify(constraint)):
                    return True
        except Exception:
            pass

        return False

    with _Z3_LOCK:
        return check_node(candidate_sym.expr)


def _scalar_tensor_tensor_operand0_illegal(candidate_sym: SymTensor) -> bool:
    """True if any scalar_tensor_tensor node feeds a non-vector operand0.

    NKI's ``nisa.scalar_tensor_tensor`` requires ``operand0`` to be the
    "scalar" broadcast operand — either a compile-time constant or a
    per-partition ``(P, 1)`` vector (free dim 1). The symbolic broadcast
    shape rule is looser (it admits a full ``(P, F)`` operand0 too), so Z3
    happily proves a full-tensor operand0 equivalent — but it fails NKI
    codegen with "operand0's free dimension should be 1 (a vector)".
    Reject those candidates here so synthesis picks a tensor_tensor lowering
    instead. (rope's rotation is the canonical case: cos/sin are full tiles.)
    """
    visited: set[int] = set()

    def check_node(expr: SymExpr) -> bool:
        key = id(expr)
        if key in visited:
            return False
        visited.add(key)
        for inp in expr.inputs:
            if check_node(inp):
                return True
        if expr.op == "scalar_tensor_tensor":
            idx = expr.attrs.get("operand0_input_index")
            if idx is not None and 0 <= idx < len(expr.inputs):
                operand0_shape = expr.inputs[idx].shape
                # Reject unless the free dim is *provably* 1. A literal
                # IntVal(1) is the common case (leaf (P,1) inputs like
                # fused_adam's per-row scalars); a derived dim (e.g. from a
                # broadcast `If(a==1, b, a)`) may also be provably 1, so fall
                # back to the solver before rejecting.
                if len(operand0_shape) >= 2 and not _free_dim_provably_one(
                    operand0_shape[-1]
                ):
                    return True
        return False

    with _Z3_LOCK:
        return check_node(candidate_sym.expr)


def _free_dim_provably_one(free_dim: Any) -> bool:
    if z3.is_int_value(free_dim):
        return free_dim.as_long() == 1
    # Non-literal z3 expr: provably 1 iff `free_dim != 1` is unsatisfiable.
    solver = z3.Solver()
    solver.set("timeout", 200)
    solver.add(free_dim != z3.IntVal(1))
    return solver.check() == z3.unsat


def _shapes_not_provably_equivalent(
    target_sym: SymTensor,
    candidate_sym: SymTensor,
    timeout: int = 500,
    input_syms: list[SymTensor] | None = None,
) -> bool:
    input_expr_ids: frozenset[int] = frozenset(
        id(sym.expr) for sym in (input_syms or []) if sym.expr is not None
    )

    input_positivity: list[z3.BoolRef] = []
    visited_inputs: set[int] = set()

    def _collect_input_positivity(expr: SymExpr) -> None:
        key = id(expr)
        if key in visited_inputs:
            return
        visited_inputs.add(key)
        if expr.op == "input":
            for dim in expr.shape:
                positivity = dim > z3.IntVal(0)
                if not z3.is_true(z3.simplify(positivity)):
                    input_positivity.append(positivity)
            return
        for inp in expr.inputs:
            _collect_input_positivity(inp)

    def _collect_top_level_shape_constraints(expr: SymExpr) -> list[z3.BoolRef]:
        if expr.op == "input":
            return []
        entry = _SEMANTICS.get(expr.op)
        if entry is None:
            return []
        input_shapes = [ShapeExpr(list(inp.shape)) for inp in expr.inputs]
        attrs_for_check = {k: v for k, v in expr.attrs.items() if k != "out_shape"}
        try:
            result = entry.shape_rule(input_shapes, attrs_for_check)
        except Exception:
            return []
        out = []
        for constraint in result.ctx.facts:
            simplified = z3.simplify(constraint)
            if not z3.is_true(simplified) and not z3.is_false(simplified):
                out.append(constraint)
        return out

    candidate_shape_constraints: list[z3.BoolRef] = []
    visited_ops: set[int] = set()

    def _collect_candidate_shape_constraints(expr: SymExpr) -> None:
        key = id(expr)
        if key in visited_ops:
            return
        visited_ops.add(key)
        if expr.op == "input" or key in input_expr_ids:
            return
        for inp in expr.inputs:
            _collect_candidate_shape_constraints(inp)
        entry = _SEMANTICS.get(expr.op)
        if entry is None:
            return
        input_shapes = [ShapeExpr(list(inp.shape)) for inp in expr.inputs]
        attrs_for_check = {k: v for k, v in expr.attrs.items() if k != "out_shape"}
        try:
            facts = list(entry.shape_rule(input_shapes, attrs_for_check).ctx.facts)
            facts.extend(entry.validity_rule(input_shapes, attrs_for_check).facts)
            for constraint in facts:
                simplified = z3.simplify(constraint)
                if not z3.is_true(simplified) and not z3.is_false(simplified):
                    candidate_shape_constraints.append(constraint)
        except Exception:
            pass

    with _Z3_LOCK:
        _collect_input_positivity(candidate_sym.expr)
        target_expr = target_sym.expr if target_sym is not None else None
        target_shape_constraints = (
            _collect_top_level_shape_constraints(target_expr)
            if target_expr is not None
            else []
        )
        _collect_candidate_shape_constraints(candidate_sym.expr)

        if not candidate_shape_constraints:
            return False

        all_constraints = (
            z3.And(*candidate_shape_constraints)
            if len(candidate_shape_constraints) > 1
            else candidate_shape_constraints[0]
        )
        solver = z3.Solver()
        solver.set("timeout", timeout)
        valid_assumptions = input_positivity + target_shape_constraints
        if valid_assumptions:
            solver.add(z3.And(*valid_assumptions))
        solver.add(z3.Not(all_constraints))
        result = solver.check()
        return result == z3.sat


def _shape_rejection_reason(
    target_sym: SymTensor,
    candidate_sym: SymTensor,
    input_syms: list[SymTensor] | None = None,
) -> str | None:
    if _shapes_incompatible_symbolically(target_sym, candidate_sym):
        return "symbolic shape mismatch"
    if _sketch_shape_constraints_violated(candidate_sym):
        return "sketch shape constraints violated"
    if _scalar_tensor_tensor_operand0_illegal(candidate_sym):
        return "scalar_tensor_tensor operand0 not a vector"
    if _shapes_not_provably_equivalent(
        target_sym, candidate_sym, input_syms=input_syms
    ):
        return "output shape not provably equivalent to target"
    return None


_last_synthesis_stats: dict[str, Any] = {}


def _reset_synthesis_stats(**stats: Any) -> None:
    with _SYNTHESIS_STATS_LOCK:
        _last_synthesis_stats.clear()
        _last_synthesis_stats.update(stats)


def _update_synthesis_stats(**stats: Any) -> None:
    with _SYNTHESIS_STATS_LOCK:
        _last_synthesis_stats.update(stats)


def _format_sketch(sketch: SketchNode) -> str:
    if sketch.hole:
        return "□"
    if sketch.op == "INPUT":
        name = "?"
        if sketch.sym is not None:
            try:
                name = (
                    sketch.sym.expr.name
                    if sketch.sym.expr is not None
                    else repr(sketch.sym)
                )
            except Exception:
                name = repr(sketch.sym)
        return f"IN:{name}"
    children_str = ", ".join(_format_sketch(c) for c in sketch.children)
    attrs_parts = []
    for k, v in sketch.attrs.items():
        if k == "name":
            continue
        attrs_parts.append(f"{k}={v!r}")
    attrs_str = ("[" + ", ".join(attrs_parts) + "]") if attrs_parts else ""
    return f"{sketch.op}{attrs_str}({children_str})"


def iter_complete_sketches(
    target_sym: SymTensor,
    pool: list[SketchNode],
    max_hw_size: int = 2,
    input_syms: list[SymTensor] | None = None,
) -> Iterator[tuple[SketchNode, SymTensor]]:
    """Yield complete, bounded candidates that pass cheap rejection checks."""
    initial = SketchNode.make_hole()
    worklist: list[SketchNode] = [initial]
    seen: set[SketchNode] = {initial}

    while worklist:
        sketch = worklist.pop()

        if not sketch.has_hole():
            if sketch.hw_size() > max_hw_size:
                continue
            candidate_sym = _eval_sketch(sketch)
            if candidate_sym is None:
                continue
            rejection_reason = _shape_rejection_reason(
                target_sym,
                candidate_sym,
                input_syms=input_syms,
            )
            if rejection_reason is not None:
                continue
            yield sketch, candidate_sym
            continue

        can_add_hw_op = sketch.hw_size() < max_hw_size
        for pool_entry in reversed(pool):
            if pool_entry.op not in ("INPUT", "HOLE") and not can_add_hw_op:
                continue
            filled = _fill_first_hole(sketch, pool_entry)
            if filled is None or filled in seen:
                continue
            seen.add(filled)
            worklist.append(filled)


def iter_proved_sketches(
    target_sym: SymTensor,
    pool: list[SketchNode],
    max_hw_size: int = 2,
    input_syms: list[SymTensor] | None = None,
    prove: Callable[[SketchNode, SymTensor], Any] | None = None,
) -> Iterator[tuple[SketchNode, Any]]:
    """Yield proved sketches in deterministic search order."""
    if prove is None:
        raise ValueError("iter_proved_sketches requires a prove callback")
    for sketch, candidate_sym in iter_complete_sketches(
        target_sym,
        pool,
        max_hw_size=max_hw_size,
        input_syms=input_syms,
    ):
        verdict = prove(sketch, candidate_sym)
        if verdict:
            yield sketch, verdict


def _float_scalar_attr(attrs: dict[str, Any], key: str) -> None:
    value = attrs.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        attrs[key] = float(value)


def sketch_node_attrs(
    op: str, child_count: int, attrs: dict[str, Any]
) -> dict[str, Any]:
    """Return canonical codegen attributes for one sketch node.

    This is the normalization boundary for synthesized sketches: the result
    matches the decoded form exactly, so the generic ISA codec encodes it
    without further operation dispatch.
    """
    op = canonical_isa_op(op)
    clean_attrs = {k: v for k, v in attrs.items() if k != "name"}
    if op == "activation":
        clean_attrs.setdefault("scale", 1.0)
        clean_attrs.setdefault("bias_const", None)
        clean_attrs.setdefault("with_reduce", False)
        clean_attrs.setdefault("reduce_op", None)
        clean_attrs.setdefault("reduce_cmd", reduce_cmd.idle)
        _float_scalar_attr(clean_attrs, "scale")
        _float_scalar_attr(clean_attrs, "bias_const")
    if op == "activation_reduce":
        clean_attrs.setdefault("scale", 1.0)
        clean_attrs.setdefault("bias_const", None)
        _float_scalar_attr(clean_attrs, "scale")
        _float_scalar_attr(clean_attrs, "bias_const")
    if op == "exponential":
        if "max_input_index" not in clean_attrs:
            clean_attrs.setdefault("max_value", 0.0)
        clean_attrs.setdefault("reduce_cmd", reduce_cmd.idle)
        clean_attrs.setdefault("reduce_init", 0.0)
        clean_attrs.setdefault("with_reduce", False)
        _float_scalar_attr(clean_attrs, "max_value")
        _float_scalar_attr(clean_attrs, "reduce_init")
    if op == "nc_matmul":
        clean_attrs.setdefault("is_stationary_onezero", False)
        clean_attrs.setdefault("is_moving_onezero", False)
        clean_attrs.setdefault("is_transpose", False)
        clean_attrs["accumulate"] = bool(clean_attrs.get("accumulate") or False)
        clean_attrs.setdefault("perf_mode", matmul_perf_mode.none)
    if op in ("nc_transpose", "tensor_tensor", "tensor_copy"):
        clean_attrs.setdefault("engine", engine.unknown)
    if op == "tensor_reduce":
        axis_attr = clean_attrs.get("axis", 1)
        axis = axis_attr[0] if isinstance(axis_attr, (list, tuple)) else axis_attr
        if isinstance(axis, int) and not isinstance(axis, bool) and axis < 0:
            axis += 2
        clean_attrs["axis"] = axis
        clean_attrs["negate"] = bool(clean_attrs.get("negate", False))
        clean_attrs["keepdims"] = bool(
            clean_attrs.get("keepdims", clean_attrs.get("keep_dims", False))
        )
        clean_attrs.pop("keep_dims", None)
    if op == "tensor_scalar_cumulative":
        # Add the defaults required by the SCAN2 value rule.
        clean_attrs.setdefault("reverse0", False)
        clean_attrs.setdefault("reverse1", False)
        clean_attrs.setdefault("reduce_cmd", reduce_cmd.reset_reduce)
        _float_scalar_attr(clean_attrs, "imm0_const")
        _float_scalar_attr(clean_attrs, "imm1_const")
    if op == "tensor_scalar":
        # Index assignment depends on which operands are constants. Children
        # are emitted in `[data, operand0?, operand1?]` order; operand slots
        # carrying compile-time constants don't show up as children.
        op0_is_const = "operand0_const" in clean_attrs
        op1_is_const = "operand1_const" in clean_attrs
        next_child_idx = 1
        if not op0_is_const and child_count > next_child_idx:
            clean_attrs["operand0_input_index"] = next_child_idx
            next_child_idx += 1
        if (
            clean_attrs.get("op1") is not None
            and not op1_is_const
            and child_count > next_child_idx
        ):
            clean_attrs["operand1_input_index"] = next_child_idx
        clean_attrs.setdefault("op1", None)
        clean_attrs.setdefault("reverse0", False)
        clean_attrs.setdefault("reverse1", False)
        clean_attrs.setdefault("engine", engine.unknown)
        _float_scalar_attr(clean_attrs, "operand0_const")
        _float_scalar_attr(clean_attrs, "operand1_const")
    if op == "scalar_tensor_tensor":
        # Children: [data, (operand0|skip-if-const), operand1]. The constant-
        # scalar form has 2 children (data + operand1); the tensor-scalar
        # form has 3 (data + operand0 + operand1).
        if "operand0_const" in clean_attrs:
            if child_count >= 2:
                clean_attrs["operand1_input_index"] = 1
        else:
            if child_count >= 2:
                clean_attrs["operand0_input_index"] = 1
            if child_count >= 3:
                clean_attrs["operand1_input_index"] = 2
        clean_attrs.setdefault("reverse0", False)
        clean_attrs.setdefault("reverse1", False)
        _float_scalar_attr(clean_attrs, "operand0_const")
    return clean_attrs


def sketch_materialized_op(op: str) -> str:
    """The concrete Node op a sketch op materializes to (resolves aliases)."""
    return canonical_isa_op(op)


def _distinct_formals(input_syms: list[SymTensor]) -> list[SymTensor]:
    """Return distinct formals in first-occurrence identity order."""
    distinct: list[SymTensor] = []
    seen: set[int] = set()
    for sym in input_syms:
        if id(sym) not in seen:
            seen.add(id(sym))
            distinct.append(sym)
    return distinct


def _make_lowering_cache_key(
    op: str,
    attrs: dict[str, Any],
    input_syms: list[SymTensor],
    max_hw_size: int,
) -> tuple:
    """Return the cache key for one bounded lowering target."""
    distinct = _distinct_formals(input_syms)
    index_of: dict[int, int] = {id(sym): i for i, sym in enumerate(distinct)}
    position_pattern = tuple(index_of[id(sym)] for sym in input_syms)

    formal_shape_keys = _alpha_normalized_shape_keys(distinct)
    frozen_attrs = tuple(
        sorted((key, _recipe_value_key(value)) for key, value in attrs.items())
    )
    return (
        op,
        frozen_attrs,
        position_pattern,
        formal_shape_keys,
        max_hw_size,
        tuple(ISA_POOL_OP_NAMES),
    )


def _recipe_value_key(value: Any) -> Any:
    if isinstance(value, z3.AstRef):
        return ("z3", value.sexpr())
    if isinstance(value, dict):
        items = (
            (_recipe_value_key(key), _recipe_value_key(item))
            for key, item in value.items()
        )
        return ("dict", tuple(sorted(items, key=repr)))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_recipe_value_key(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return (
            type(value).__name__,
            tuple(sorted((_recipe_value_key(item) for item in value), key=repr)),
        )
    value_type = type(value)
    type_id = (value_type.__module__, value_type.__qualname__)
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return ("named", type_id, name)
    qualname = getattr(value, "__qualname__", None)
    if isinstance(qualname, str):
        return ("callable", type_id, qualname)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return ("literal", type_id, value)
    raise TypeError(
        "unsupported recipe attribute type "
        f"{value_type.__module__}.{value_type.__qualname__}"
    )


def _alpha_normalized_shape_keys(
    input_syms: list[SymTensor],
) -> tuple[tuple[Any, ...], ...]:
    symbol_indices: dict[int, int] = {}
    symbols: dict[int, z3.ExprRef] = {}

    def collect(expr: z3.ExprRef) -> None:
        if (
            z3.is_const(expr)
            and expr.num_args() == 0
            and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED
        ):
            expr_id = expr.get_id()
            if expr_id not in symbol_indices:
                symbol_indices[expr_id] = len(symbol_indices)
                symbols[expr_id] = expr
            return
        for child in expr.children():
            collect(child)

    dimensions = [[_to_dim(dim) for dim in sym.shape] for sym in input_syms]
    for shape in dimensions:
        for dim in shape:
            collect(dim)

    substitutions = tuple(
        (
            symbols[expr_id],
            z3.Const(f"__axon_recipe_dim_{index}", symbols[expr_id].sort()),
        )
        for expr_id, index in symbol_indices.items()
    )

    def dimension_key(dim: z3.ArithRef) -> Any:
        if z3.is_int_value(dim):
            return ("const", dim.as_long())
        normalized = z3.substitute(dim, *substitutions) if substitutions else dim
        return ("expr", z3.simplify(normalized).sexpr())

    return tuple(tuple(dimension_key(dim) for dim in shape) for shape in dimensions)


def _normalize_sketch(
    sketch: SketchNode,
    input_syms: list[SymTensor],
) -> _NormSketch | None:
    """Normalize a sketch with identity-based formal indexes."""
    distinct = _distinct_formals(input_syms)
    index_of: dict[int, int] = {id(sym): i for i, sym in enumerate(distinct)}
    return _normalize_sketch_indexed(sketch, index_of)


def _normalize_sketch_indexed(
    sketch: SketchNode,
    index_of: dict[int, int],
) -> _NormSketch | None:
    if sketch.op == "INPUT":
        if sketch.sym is None:
            return None
        idx = index_of.get(id(sketch.sym))
        if idx is None:
            return None
        return ("INPUT", idx)
    children: list[_NormSketch] = []
    for child in sketch.children:
        n = _normalize_sketch_indexed(child, index_of)
        if n is None:
            return None
        children.append(n)
    clean_attrs = {k: v for k, v in sketch.attrs.items() if k != "name"}
    return ("OP", sketch.op, clean_attrs, tuple(children))


def _denormalize_sketch(
    norm: _NormSketch,
    input_syms: list[SymTensor],
) -> SketchNode | None:
    """Rebuild a sketch from identity-based formal indexes."""
    distinct = _distinct_formals(input_syms)
    return _denormalize_sketch_indexed(norm, distinct)


def _denormalize_sketch_indexed(
    norm: _NormSketch,
    distinct: list[SymTensor],
) -> SketchNode | None:
    if norm[0] == "INPUT":
        idx = norm[1]
        if idx < 0 or idx >= len(distinct):
            return None
        return SketchNode.make_input(distinct[idx])
    _, op, attrs, children_norm = norm
    children: list[SketchNode] = []
    for cn in children_norm:
        child = _denormalize_sketch_indexed(cn, distinct)
        if child is None:
            return None
        children.append(child)
    return SketchNode.make_op(op, children, dict(attrs))


def _normalized_recipe_key(norm: _NormSketch) -> Any:
    if norm[0] == "INPUT":
        return ("INPUT", int(norm[1]))
    _, op, attrs, children = norm
    return (
        "OP",
        canonical_isa_op(op),
        tuple(
            sorted(
                ((key, _recipe_value_key(value)) for key, value in attrs.items()),
                key=repr,
            )
        ),
        tuple(_normalized_recipe_key(child) for child in children),
    )


def _syntactic_recipe_key(
    sketch: SketchNode,
    formal_indices: dict[int, int],
) -> Any:
    if sketch.hole:
        return ("HOLE",)
    if sketch.op == "INPUT":
        if sketch.sym is None or id(sketch.sym) not in formal_indices:
            raise ValueError("generated recipe references an unknown formal input")
        return ("INPUT", formal_indices[id(sketch.sym)])
    return (
        "OP",
        canonical_isa_op(sketch.op),
        tuple(
            sorted(
                (
                    (key, _recipe_value_key(value))
                    for key, value in sketch.attrs.items()
                    if key != "name"
                ),
                key=repr,
            )
        ),
        tuple(
            _syntactic_recipe_key(child, formal_indices) for child in sketch.children
        ),
    )


def _sketch_derivation_depth(sketch: SketchNode) -> int:
    if sketch.hole or sketch.op == "INPUT":
        return 0
    return 1 + max(
        (_sketch_derivation_depth(child) for child in sketch.children),
        default=0,
    )


def _sketch_formal_pattern(
    sketch: SketchNode,
    input_syms: list[SymTensor],
) -> tuple[int, ...] | None:
    distinct = _distinct_formals(input_syms)
    index_of = {id(sym): index for index, sym in enumerate(distinct)}
    pattern: list[int] = []

    def walk(node: SketchNode) -> bool:
        if node.op == "INPUT":
            if node.sym is None or id(node.sym) not in index_of:
                return False
            pattern.append(index_of[id(node.sym)])
            return True
        return all(walk(child) for child in node.children)

    return tuple(pattern) if walk(sketch) else None


def _sketch_constituents_allowed(sketch: SketchNode, target_op: str) -> bool:
    if sketch.op in ("INPUT", "HOLE"):
        return True
    target_constituents = _op_constituents(target_op)
    if sketch.op not in _LAYOUT_TRANSFORM_OPS and not _template_constituents(
        sketch
    ).issubset(target_constituents):
        return False
    return all(
        _sketch_constituents_allowed(child, target_op) for child in sketch.children
    )


def _canonical_direct_recipe_key(
    pool: list[SketchNode],
    input_syms: list[SymTensor],
    target_op: str | None,
    target_attrs: dict[str, Any] | None,
    max_hw_size: int,
) -> Any | None:
    recipe = _canonical_direct_recipe(
        pool,
        input_syms,
        target_op,
        target_attrs,
        max_hw_size,
    )
    return _normalized_recipe_key(recipe) if recipe is not None else None


def _canonical_direct_recipe(
    pool: list[SketchNode],
    input_syms: list[SymTensor],
    target_op: str | None,
    target_attrs: dict[str, Any] | None,
    max_hw_size: int,
) -> _NormSketch | None:
    target_sym: SymTensor | None = None
    if target_op is None:
        target_constituents = None
    else:
        target_node = Node(
            id="__recipe_target",
            op=target_op,
            inputs=[str(index) for index in range(len(input_syms))],
            attrs=dict(target_attrs or {}),
        )
        target_constituents = _node_actual_constituents(target_node)
        target_sym = _sym_expr_from_graph_node(target_node, input_syms)
    formal_ids = {
        id(sym): index for index, sym in enumerate(_distinct_formals(input_syms))
    }
    target_pattern = tuple(formal_ids[id(sym)] for sym in input_syms)

    for entry in pool:
        if (
            entry.op in ("INPUT", "HOLE")
            or entry.has_hole()
            or entry.hw_size() > max_hw_size
        ):
            continue
        if target_constituents is not None and (
            not _template_constituents(entry).issubset(target_constituents)
        ):
            continue
        if _sketch_formal_pattern(entry, input_syms) != target_pattern:
            continue
        candidate_sym = _eval_sketch(entry)
        if target_sym is not None and (
            candidate_sym is None
            or target_sym.rank != candidate_sym.rank
            or not all(
                z3.is_true(z3.simplify(target_dim == candidate_dim))
                for target_dim, candidate_dim in zip(
                    target_sym.shape,
                    candidate_sym.shape,
                    strict=True,
                )
            )
        ):
            continue
        norm = _normalize_sketch(entry, input_syms)
        if norm is not None:
            return norm
    return None


def _iter_complete_syntactic_sketches(
    pool: list[SketchNode],
    input_syms: list[SymTensor],
    max_hw_size: int,
) -> Iterator[SketchNode]:
    """Yield complete sketches in bounded hardware-size and depth waves."""
    initial = SketchNode.make_hole()
    seen: set[SketchNode] = {initial}
    sequence = itertools.count()
    worklist: list[tuple[int, int, str, int, SketchNode]] = []
    formal_indices = {
        id(sym): index for index, sym in enumerate(_distinct_formals(input_syms))
    }

    def push(sketch: SketchNode) -> None:
        heapq.heappush(
            worklist,
            (
                sketch.hw_size(),
                0 if sketch.has_hole() else 1,
                repr(_syntactic_recipe_key(sketch, formal_indices)),
                next(sequence),
                sketch,
            ),
        )

    push(initial)
    for wave_size in range(max_hw_size + 1):
        wave: list[SketchNode] = []
        while worklist and worklist[0][0] <= wave_size:
            _, _, _, _, sketch = heapq.heappop(worklist)
            if not sketch.has_hole():
                wave.append(sketch)
                continue
            can_add_hw_op = sketch.hw_size() < max_hw_size
            for pool_entry in pool:
                if pool_entry.op not in ("INPUT", "HOLE") and not can_add_hw_op:
                    continue
                filled = _fill_first_hole(sketch, pool_entry)
                if filled is None or filled.hw_size() > max_hw_size or filled in seen:
                    continue
                seen.add(filled)
                push(filled)
        wave.sort(
            key=lambda sketch: (
                _sketch_derivation_depth(sketch),
                repr(_syntactic_recipe_key(sketch, formal_indices)),
            )
        )
        yield from wave


def _iter_complete_syntactic_recipes(
    pool: list[SketchNode],
    input_syms: list[SymTensor],
    max_hw_size: int,
    *,
    target_op: str | None = None,
    target_attrs: dict[str, Any] | None = None,
) -> Iterator[_NormSketch]:
    canonical = _canonical_direct_recipe(
        pool,
        input_syms,
        target_op,
        target_attrs,
        max_hw_size,
    )
    canonical_key = _normalized_recipe_key(canonical) if canonical is not None else None
    if canonical is not None:
        yield canonical

    for sketch in _iter_complete_syntactic_sketches(pool, input_syms, max_hw_size):
        norm = _normalize_sketch(sketch, input_syms)
        if norm is None:
            raise ValueError("generated recipe cannot be rebound to its formal inputs")
        if canonical_key is not None and _normalized_recipe_key(norm) == canonical_key:
            continue
        yield norm


def _complete_syntactic_recipes(
    pool: list[SketchNode],
    input_syms: list[SymTensor],
    max_hw_size: int,
    *,
    target_op: str | None = None,
    target_attrs: dict[str, Any] | None = None,
) -> tuple[_NormSketch, ...]:
    return tuple(
        _iter_complete_syntactic_recipes(
            pool,
            input_syms,
            max_hw_size,
            target_op=target_op,
            target_attrs=target_attrs,
        )
    )


def _iter_complete_recipe_cache_entry(
    target_op: str,
    target_attrs: dict[str, Any],
    input_syms: list[SymTensor],
    max_hw_size: int,
) -> Iterator[_NormSketch]:
    pool = _build_synthesis_pool(target_op, target_attrs, input_syms)
    yield from _iter_complete_syntactic_recipes(
        pool,
        input_syms,
        max_hw_size,
        target_op=target_op,
        target_attrs=target_attrs,
    )


def _build_complete_recipe_cache_entry(
    target_op: str,
    target_attrs: dict[str, Any],
    input_syms: list[SymTensor],
    max_hw_size: int,
) -> tuple[_NormSketch, ...]:
    return tuple(
        _iter_complete_recipe_cache_entry(
            target_op,
            target_attrs,
            input_syms,
            max_hw_size,
        )
    )


def _iter_rebound_recipes(
    target_sym: SymTensor,
    recipes: tuple[_NormSketch, ...],
    input_syms: list[SymTensor],
    *,
    target_op: str | None = None,
    max_hw_size: int | None = None,
) -> Iterator[tuple[SketchNode, SymTensor]]:
    for norm in recipes:
        sketch = _denormalize_sketch(norm, input_syms)
        if sketch is None:
            raise ValueError("cached recipe cannot be rebound to its formal inputs")
        if max_hw_size is not None and sketch.hw_size() > max_hw_size:
            continue
        if target_op is not None and not _sketch_constituents_allowed(
            sketch, target_op
        ):
            continue
        candidate_sym = _eval_sketch(sketch)
        if candidate_sym is None:
            continue
        rejection_reason = _shape_rejection_reason(
            target_sym,
            candidate_sym,
            input_syms=input_syms,
        )
        if rejection_reason is None:
            yield sketch, candidate_sym


def _build_dag_levels(G: nuGraph) -> list[list[Node]]:
    if not G.nodes:
        return []
    level_of: dict[str, int] = {}
    for node in G.nodes:
        if not node.inputs or node.op == "input":
            level_of[node.id] = 0
        else:
            level_of[node.id] = (
                builtins.max(level_of.get(inp, 0) for inp in node.inputs) + 1
            )

    max_level = builtins.max(level_of.values())
    levels: list[list[Node]] = [[] for _ in range(max_level + 1)]
    for node in G.nodes:
        levels[level_of[node.id]].append(node)
    return levels


def synthesize_hw_graph(
    G: nuGraph,
    max_hw_size: int = 2,
    timeout: int = 3000,
    verbose: bool = False,
    max_workers: int | None = None,
    kernel_name: str | None = None,
) -> list[nuGraph]:
    """Synthesize every hardware graph for ``G`` through the e-graph pipeline."""
    del verbose, kernel_name
    from axon.egraph.pipeline import iter_synthesized_hw_graphs

    return list(
        iter_synthesized_hw_graphs(
            G,
            max_hw_size=max_hw_size,
            timeout=timeout,
            workers=max_workers,
        )
    )


def print_graph(G: nuGraph) -> None:
    symbolic_shapes: dict[str, tuple] = {}
    sym_shape_fallback = "None"
    try:
        symbolic_shapes = {
            node_id: tensor.shape
            for node_id, tensor in _graph_symbolic_tensors(G).items()
        }
    except (KeyError, z3.Z3Exception):
        symbolic_shapes = {}
        sym_shape_fallback = "unavailable"
    topologically_ordered_nodes = [
        node for level in _build_dag_levels(G) for node in level
    ]
    for i, n in enumerate(topologically_ordered_nodes):
        sym_shape = symbolic_shapes.get(n.id)
        sym_shape_str = (
            _format_shape(sym_shape) if sym_shape is not None else sym_shape_fallback
        )
        print(
            f"[{i}] id={n.id:12s} op={n.op:10s} inputs={n.inputs} "
            f"sym_shape={sym_shape_str} attrs={n.attrs}"
        )


def _unique_node_input_ids(node: Node) -> list[str]:
    input_ids: list[str] = []
    seen: set[str] = set()
    for inp_id in node.inputs:
        if inp_id in seen:
            continue
        seen.add(inp_id)
        input_ids.append(inp_id)
    return input_ids


def _equivalent_replacement_id(
    target_sym: SymTensor,
    candidate_ids: list[str],
    all_syms: dict[str, SymTensor],
    timeout: int,
) -> str | None:
    for candidate_id in candidate_ids:
        candidate_sym = all_syms.get(candidate_id)
        if candidate_sym is None:
            continue
        if not _shapes_match_exactly(target_sym, candidate_sym):
            continue
        if _check_equivalent_quiet(target_sym, candidate_sym, timeout=timeout):
            return candidate_id
    return None


def _consumer_input_replacement_is_equivalent(
    consumer: Node,
    original_input_id: str,
    replacement_id: str,
    all_syms: dict[str, SymTensor],
    timeout: int,
) -> bool:
    target_sym = all_syms.get(consumer.id)
    replacement_sym = all_syms.get(replacement_id)
    if target_sym is None or replacement_sym is None:
        return False

    rewritten_input_syms: list[SymTensor] = []
    for inp_id in consumer.inputs:
        sym = replacement_sym if inp_id == original_input_id else all_syms.get(inp_id)
        if sym is None:
            return False
        rewritten_input_syms.append(sym)

    try:
        rewritten_sym = _sym_expr_from_graph_node(consumer, rewritten_input_syms)
    except Exception:
        return False
    if rewritten_sym is None or not _shapes_match_exactly(target_sym, rewritten_sym):
        return False
    return _check_equivalent_quiet(target_sym, rewritten_sym, timeout=timeout)


def _allowed_single_op_simplification_hw_ops(
    producer: Node,
    consumer: Node,
    all_syms: dict[str, SymTensor],
    timeout: int,
) -> list[str]:
    transparent_node_ids: set[str] = set()

    consumer_target_sym = all_syms.get(consumer.id)
    if consumer_target_sym is not None and (
        _equivalent_replacement_id(
            consumer_target_sym,
            _unique_node_input_ids(consumer),
            all_syms,
            timeout,
        )
        is not None
    ):
        transparent_node_ids.add(consumer.id)

    for producer_input_id in _unique_node_input_ids(producer):
        if _consumer_input_replacement_is_equivalent(
            consumer,
            producer.id,
            producer_input_id,
            all_syms,
            timeout,
        ):
            transparent_node_ids.add(producer.id)
            break

    required_constituents: set[str] = set()
    for node in (producer, consumer):
        if node.id in transparent_node_ids:
            continue
        required_constituents.update(_node_actual_constituents(node))

    if not required_constituents:
        return []

    pool_ops: set[str] = set()
    if producer.op in ISA_POOL_OP_NAMES_SET:
        pool_ops.add(producer.op)
    pool_ops.update(
        hw_op
        for hw_op in ISA_POOL_OP_NAMES
        if required_constituents.issubset(_op_constituents(hw_op))
    )
    return list(pool_ops)


def _build_general_simplification_pool(
    input_syms: list[SymTensor],
    allowed_hw_ops: list[str] | None = None,
    extra_attrs: dict[str, Any] | None = None,
) -> list[SketchNode]:
    concrete = [SketchNode.make_input(sym) for sym in input_syms]
    pool: list[SketchNode] = list(concrete)
    hw_ops = allowed_hw_ops if allowed_hw_ops is not None else ISA_POOL_OP_NAMES
    for hw_op in hw_ops:
        hw_op_attrs: dict[str, Any] = {"op_name": hw_op}
        if extra_attrs:
            hw_op_attrs.update(extra_attrs)
        templates = _pool_templates_for_hw_op(hw_op, concrete, hw_op_attrs)
        pool.extend(templates)
    return pool
