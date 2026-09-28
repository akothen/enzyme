from __future__ import annotations

import builtins
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import z3

import axon.lang_semantics  # noqa: F401  -- registers public op semantics
from axon.isa_semantics import (
    _NODE_IDS,
    _SEMANTICS,
    _Z3_LOCK,
    ShapeExpr,
    SymExpr,
    SymTensor,
    _arith_equal,
    _public_broadcast_shape_tuple,
    _public_reduce_out_shape,
    _to_dim,
    sym_ceil_div,
)


class AxonArray:
    def __init__(
        self,
        node_id: str,
        shape: tuple[int, ...],
        nodes: list[Any] | None = None,
        sym_shape: tuple[str, ...] | None = None,
    ):
        self.node_id = node_id
        self.shape = shape
        self.nodes = (
            list(nodes)
            if nodes is not None
            else [_make_input_node(node_id, shape, sym_shape=sym_shape)]
        )

    @staticmethod
    def _merge_nodes(inputs: list[AxonArray]) -> list[Any]:
        merged: list[Any] = []
        seen: set[str] = set()
        for inp in inputs:
            for node in inp.nodes:
                if node.id in seen:
                    continue
                merged.append(node)
                seen.add(node.id)
        return merged

    @staticmethod
    def _from_op(
        op: str,
        inputs: list[AxonArray],
        out_shape: tuple[int, ...],
        attrs: dict[str, Any] | None = None,
    ) -> AxonArray:
        node_id = _NODE_IDS.next_name(op)
        nodes = AxonArray._merge_nodes(inputs)
        nodes.append(
            Node(
                id=node_id,
                op=op,
                inputs=[inp.node_id for inp in inputs],
                attrs=dict(attrs or {}),
            )
        )
        return AxonArray(node_id, out_shape, nodes)

    def _binary_op(self, other: Any, op: str) -> AxonArray:
        if op == "matmul":
            if not isinstance(other, AxonArray):
                raise TypeError("matmul expects AxonArray operand")
            if len(self.shape) != 2 or len(other.shape) != 2:
                raise ValueError("matmul expects rank-2 inputs")
            if not _dims_equal(self.shape[1], other.shape[0]):
                raise ValueError("matmul expects compatible inner dimensions")
            return AxonArray._from_op(
                op, [self, other], (self.shape[0], other.shape[1])
            )

        if isinstance(other, AxonArray):
            out_shape = tuple(
                _normalize_dim(d)
                for d in _public_broadcast_shape_tuple(self.shape, other.shape)
            )
            return AxonArray._from_op(op, [self, other], out_shape)
        else:
            out_shape = self.shape
            return AxonArray._from_op(op, [self], out_shape, {"scalar": other})

    def __add__(self, other: Any) -> AxonArray:
        return self._binary_op(other, "add")

    def __radd__(self, other: Any) -> AxonArray:
        return self._binary_op(other, "add")

    def __sub__(self, other: Any) -> AxonArray:
        return self._binary_op(other, "subtract")

    def __rsub__(self, other: Any) -> AxonArray:
        # other - self: not expressible as a single scalar-attr "subtract" with
        # reversed operands without extending the binary attr surface, so build
        # it as -1*self + other.
        return (self * -1) + other

    def __mul__(self, other: Any) -> AxonArray:
        return self._binary_op(other, "mul")

    def __rmul__(self, other: Any) -> AxonArray:
        return self._binary_op(other, "mul")

    def __truediv__(self, other: Any) -> AxonArray:
        return self._binary_op(other, "div")

    def __matmul__(self, other: Any) -> AxonArray:
        return self._binary_op(other, "matmul")

    def sum(self, axis: Any = None, keep_dims: bool = False) -> AxonArray:
        out_shape = _public_reduce_out_shape(self.shape, axis, keep_dims)
        return AxonArray._from_op(
            "reduce_sum",
            [self],
            tuple(_normalize_dim(d) for d in out_shape),
            {"axis": axis, "keep_dims": keep_dims},
        )

    def broadcast_like(self, other: AxonArray) -> AxonArray:
        return AxonArray._from_op("broadcast", [self, other], other.shape)

    def sqrt(self) -> AxonArray:
        return AxonArray._from_op("sqrt", [self], self.shape)

    def exp(self) -> AxonArray:
        return AxonArray._from_op("exp", [self], self.shape)

    def transpose(self) -> AxonArray:
        if len(self.shape) == 2:
            return AxonArray._from_op(
                "transpose", [self], (self.shape[1], self.shape[0])
            )
        return AxonArray._from_op("transpose", [self], self.shape)

    def relu(self) -> AxonArray:
        return AxonArray._from_op("relu", [self], self.shape)

    def silu(self) -> AxonArray:
        return AxonArray._from_op("silu", [self], self.shape)

    def softmax(self, axis: int = -1) -> AxonArray:
        return AxonArray._from_op("softmax", [self], self.shape, {"axis": axis})

    def cumsum(self, axis: int = -1) -> AxonArray:
        return AxonArray._from_op("cumsum", [self], self.shape, {"axis": axis})


@dataclass(eq=True, frozen=True)
class NodeSig:
    id: str
    op: str
    inputs: tuple[str, ...]
    attrs: tuple[tuple[str, Any], ...]


@dataclass
class Node:
    id: str
    op: str
    inputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    shape: tuple[int, ...] | None = None

    def sig(self) -> NodeSig:
        return NodeSig(
            self.id, self.op, tuple(self.inputs), tuple(sorted(self.attrs.items()))
        )


def _make_input_node(
    node_id: str,
    shape: tuple[int, ...],
    sym_shape: tuple[str | int, ...] | None = None,
) -> Node:
    attrs: dict[str, Any] = {"shape": shape}
    if sym_shape is not None:
        attrs["sym_shape"] = tuple(sym_shape)
    return Node(id=node_id, op="input", inputs=[], attrs=attrs)


@dataclass
class nuGraph:
    nodes: list[Node]
    # Ordered IDs for the kernel's output tensors. Empty/None means "infer
    # from sinks" (single-output legacy path). Set by `_graph_from_axon_array`
    # for kernels that return tuple[AxonArray, ...] so multi-output kernels
    # survive synthesis and codegen.
    output_ids: tuple[str, ...] = field(default_factory=tuple)
    # Ordered IDs for the kernel's inputs in user-declared (kernel-spec)
    # order. The codegen uses this to fix the parameter list of the emitted
    # NKI function — without it the input order is whatever topo sort the
    # graph builder lands on, which silently scrambles positional kernel
    # calls when input shapes differ (e.g. Option-B fused_adam: (P, F)
    # vs (P, 1)).
    input_ids: tuple[str, ...] = field(default_factory=tuple)

    def position(self, node: Node) -> int:
        for i, n in enumerate(self.nodes):
            if n.id == node.id:
                return i
        raise ValueError("node not found")

    def node_at(self, pos: int) -> Node:
        return self.nodes[pos]

    def successors(self, node: Node) -> list[Node]:
        return [n for n in self.nodes if node.id in n.inputs]

    def clone(self) -> nuGraph:
        return nuGraph(
            [
                Node(n.id, n.op, list(n.inputs), dict(n.attrs), n.shape)
                for n in self.nodes
            ],
            output_ids=tuple(self.output_ids),
            input_ids=tuple(self.input_ids),
        )

    def identity(self) -> str:
        """Returns the SHA-256 identity of this materialized graph.
        The identity includes attributes and ordered inputs and outputs."""
        node_indices = {node.id: index for index, node in enumerate(self.nodes)}
        payload = {
            "nodes": [
                [
                    node.op,
                    [node_indices[value] for value in node.inputs],
                    [[key, repr(value)] for key, value in sorted(node.attrs.items())],
                ]
                for node in self.nodes
            ],
            "outputs": [node_indices[value] for value in self.output_ids],
            "inputs": [node_indices[value] for value in self.input_ids],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def __hash__(self):
        return hash(tuple(n.sig() for n in self.nodes))

    def __eq__(self, other):
        return isinstance(other, nuGraph) and tuple(
            n.sig() for n in self.nodes
        ) == tuple(n.sig() for n in other.nodes)


@dataclass
class TileAnnotation:
    tile_dims: tuple[Any, ...]
    block_dims: tuple[Any, ...]
    strip_dims: tuple[Any, ...]
    reduction_axes: frozenset[int]
    assumptions: tuple[z3.BoolRef, ...] = ()
    hardware_metadata: TileHardwareMetadata | None = None
    operand_tilings: tuple[tuple[str, TensorTileAnnotation], ...] = ()


@dataclass(frozen=True)
class TensorTileAnnotation:
    tile_dims: tuple[Any, ...]
    block_dims: tuple[Any, ...]
    strip_dims: tuple[Any, ...]
    shape: tuple[Any, ...] = ()
    assumptions: tuple[z3.BoolRef, ...] = ()


@dataclass(frozen=True)
class TileHardwareMetadata:
    concrete_tile_candidates: tuple[tuple[int, ...], ...] = ()
    operand_tile_shapes: tuple[tuple[str, tuple[int, ...]], ...] = ()
    tensor_tile_candidates: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...] = ()


_SYMBOLIC_TILE_VAR_NAMES = ("tile_partition_size", "tile_free_size")
_DEFAULT_SYMBOLIC_TILE_DIMS: tuple[z3.ArithRef, ...] = tuple(
    z3.Int(name) for name in _SYMBOLIC_TILE_VAR_NAMES
)
_SYMBOLIC_BLOCK_VAR_NAMES = ("block_partition_count", "block_free_count")
_DEFAULT_BLOCK_DIMS: tuple[z3.ArithRef, ...] = tuple(
    z3.Int(name) for name in _SYMBOLIC_BLOCK_VAR_NAMES
)
_SYMBOLIC_TILE_POSITIVITY: tuple[z3.BoolRef, ...] = tuple(
    v > z3.IntVal(0) for v in _DEFAULT_SYMBOLIC_TILE_DIMS
)
_SYMBOLIC_BLOCK_POSITIVITY: tuple[z3.BoolRef, ...] = tuple(
    v > z3.IntVal(0) for v in _DEFAULT_BLOCK_DIMS
)
_DEFAULT_TILING_POSITIVITY: tuple[z3.BoolRef, ...] = (
    _SYMBOLIC_TILE_POSITIVITY + _SYMBOLIC_BLOCK_POSITIVITY
)
_DEFAULT_PARTITION_TILE_SIZES = (128,)
_DEFAULT_FREE_TILE_SIZE_CANDIDATES = (512,)
_NC_TRANSPOSE_TILE_CANDIDATES = (128, 128)


def graph_signature(G: nuGraph) -> str:
    return " | ".join([f"{n.id}:{n.op}({','.join(n.inputs)})" for n in G.nodes])


def graph_structure_signature(G: nuGraph) -> str:
    canonical_ids: dict[str, str] = {}
    parts: list[str] = []
    for idx, node in enumerate(G.nodes):
        canonical_id = f"n{idx}"
        canonical_ids[node.id] = canonical_id
        canonical_inputs = ",".join(
            canonical_ids.get(input_id, input_id) for input_id in node.inputs
        )
        parts.append(f"{canonical_id}:{node.op}({canonical_inputs})")
    return " | ".join(parts)


def _normalize_dim(d: Any) -> Any:
    if isinstance(d, int):
        return d
    if isinstance(d, z3.ArithRef):
        d = z3.simplify(d)
    if isinstance(d, z3.IntNumRef):
        return d.as_long()
    return d


def _sanitize_symbol_fragment(text: str) -> str:
    raw = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    if not raw:
        return "dim"
    collapsed: list[str] = []
    prev_underscore = False
    for ch in raw:
        if ch == "_":
            if prev_underscore:
                continue
            prev_underscore = True
        else:
            prev_underscore = False
        collapsed.append(ch)
    return "".join(collapsed)


def _shape_dim_symbol_name(dim: Any) -> str:
    normalized = _normalize_dim(dim)
    if isinstance(normalized, int):
        return str(normalized)
    if isinstance(normalized, z3.ArithRef):
        simplified = z3.simplify(normalized)
        if z3.is_const(simplified) and simplified.num_args() == 0:
            return simplified.decl().name()
        return _sanitize_symbol_fragment(str(simplified))
    return _sanitize_symbol_fragment(str(normalized))


def _shape_specific_block_dim(dim: Any) -> Any:
    normalized = _normalize_dim(dim)
    if isinstance(normalized, int) and normalized == 1:
        return 1
    return z3.Int(f"block_count_{_shape_dim_symbol_name(normalized)}")


def _assumptions_imply(fact: z3.BoolRef, assumptions: tuple[z3.BoolRef, ...]) -> bool:
    if not assumptions:
        return False
    with _Z3_LOCK:
        solver = z3.Solver()
        solver.add(*assumptions)
        solver.add(z3.Not(fact))
        return solver.check() == z3.unsat


def _simplify_dim_with_assumptions(
    d: Any,
    assumptions: tuple[z3.BoolRef, ...] = (),
) -> Any:
    d = _normalize_dim(d)
    if not isinstance(d, z3.ArithRef) or not assumptions:
        return d

    simplified = z3.simplify(d)
    if z3.is_app_of(simplified, z3.Z3_OP_ITE):
        cond, on_true, on_false = simplified.children()
        if _assumptions_imply(cond, assumptions):
            return _simplify_dim_with_assumptions(on_true, assumptions)
        if _assumptions_imply(z3.Not(cond), assumptions):
            return _simplify_dim_with_assumptions(on_false, assumptions)
    return simplified


def _format_dim(d: Any, assumptions: tuple[z3.BoolRef, ...] = ()) -> str:
    d = _simplify_dim_with_assumptions(d, assumptions)
    if isinstance(d, z3.ArithRef):
        return str(d)
    return str(d)


def _format_shape(shape: tuple[Any, ...] | None) -> str:
    if shape is None:
        return "None"
    dims = ", ".join(_format_dim(d) for d in shape)
    if len(shape) == 1:
        dims += ","
    return f"({dims})"


def _format_dims_tuple(dims: tuple[Any, ...]) -> str:
    if not dims:
        return "()"
    parts = ", ".join(_format_dim(d) for d in dims)
    if len(dims) == 1:
        parts += ","
    return f"({parts})"


def _dims_equal(a: Any, b: Any) -> bool:
    lhs = _to_dim(a) if isinstance(a, int) else a
    rhs = _to_dim(b) if isinstance(b, int) else b
    return _arith_equal(lhs, rhs)


def _shape_expr_from_dims(shape: tuple[Any, ...]) -> ShapeExpr:
    return ShapeExpr([_to_dim(dim) for dim in shape])


def annotate_shapes_concrete(G: nuGraph) -> nuGraph:
    shapes: dict[str, tuple[Any, ...]] = {}
    for n in G.nodes:
        if n.op == "input":
            shape = tuple(
                _normalize_dim(d) for d in n.attrs.get("shape", n.shape or ())
            )
        else:
            entry = _SEMANTICS.get(n.op)
            if entry is None:
                raise KeyError(f"No shape rule registered for graph op '{n.op}'")
            shape_res = entry.shape_rule(
                [_shape_expr_from_dims(shapes[inp]) for inp in n.inputs], dict(n.attrs)
            )
            shape = tuple(_normalize_dim(d) for d in shape_res.out.dims)
        n.shape = shape
        shapes[n.id] = shape
    return G


def _node_by_id(G: nuGraph, node_id: str) -> Node | None:
    for n in G.nodes:
        if n.id == node_id:
            return n
    return None


def _position_by_id(G: nuGraph, node_id: str) -> int | None:
    for i, n in enumerate(G.nodes):
        if n.id == node_id:
            return i
    return None


def _nodes_by_op(G: nuGraph, op: str) -> list[Node]:
    return [n for n in G.nodes if n.op == op]


def _immediate_successor_positions(G: nuGraph, pos: int) -> list[int]:
    node_id = G.node_at(pos).id
    return [i for i, n in enumerate(G.nodes) if node_id in n.inputs]


def _effective_input_ids(G: nuGraph, pos: int) -> list[str]:
    return list(dict.fromkeys(G.node_at(pos).inputs))


def _fresh_graph_node_id(G: nuGraph, base: str) -> str:
    existing = {n.id for n in G.nodes}
    if base not in existing:
        return base
    idx = 1
    while f"{base}_{idx}" in existing:
        idx += 1
    return f"{base}_{idx}"


def _graph_output_nodes(G: nuGraph) -> list[Node]:
    if G.output_ids:
        by_id = {n.id: n for n in G.nodes}
        out: list[Node] = []
        for oid in G.output_ids:
            node = by_id.get(oid)
            if node is not None:
                out.append(node)
        if out:
            return out
    used = {inp for n in G.nodes for inp in n.inputs}
    return [n for n in G.nodes if n.id not in used]


# Shape rules are pure in (op, input dims, attrs); z3 hash-consing makes
# term ids stable, so identical queries can reuse the computed dims.
_SHAPE_RULE_DIMS_CACHE: dict[tuple[Any, ...], tuple[z3.ArithRef, ...]] = {}

# Every path of these ops' shape rules returns the first input's dims verbatim.
_FIRST_INPUT_SHAPE_OPS = frozenset(
    {
        "activation",
        "dma_copy",
        "exponential",
        "reciprocal",
        "scalar_tensor_tensor",
        "tensor_copy",
        "tensor_scalar",
        "tensor_scalar_cumulative",
        "tensor_tensor",
    }
)


def _shape_rule_dims(
    entry: Any, node: Node, inputs: list[SymTensor]
) -> tuple[z3.ArithRef, ...]:
    if node.op in _FIRST_INPUT_SHAPE_OPS and inputs:
        return tuple(inputs[0].shape)
    try:
        key = (
            node.op,
            tuple(tuple(d.get_id() for d in inp.shape) for inp in inputs),
            tuple(
                (k, type(v).__name__, str(v))
                for k, v in sorted(node.attrs.items())
                if k != "name"
            ),
        )
    except Exception:
        key = None
    if key is not None:
        cached = _SHAPE_RULE_DIMS_CACHE.get(key)
        if cached is not None:
            return cached
    shape_res = entry.shape_rule(
        [ShapeExpr(list(inp.shape)) for inp in inputs], dict(node.attrs)
    )
    dims = tuple(shape_res.out.dims)
    if key is not None:
        _SHAPE_RULE_DIMS_CACHE[key] = dims
    return dims


def _sym_expr_from_graph_node(
    node: Node,
    inputs: list[SymTensor],
    symbolic_shape: tuple[z3.ArithRef, ...] | None = None,
) -> SymTensor:
    if node.op == "input":
        sym_shape = node.attrs.get("sym_shape")
        if sym_shape is not None:
            # Integer literals (e.g., 1) in sym_shape are concrete dims and must
            # carry through as IntVal so the equivalence checker can prove
            # broadcast-shape constraints (e.g., tensor_scalar requires
            # operand0.dims[-1] == 1, which is only provable when the 1 is
            # literal, not a symbolic name).
            shape = tuple(
                z3.IntVal(int(dim)) if isinstance(dim, int) else z3.Int(str(dim))
                for dim in sym_shape
            )
        else:
            rank = len(node.attrs.get("shape", node.shape or ()))
            shape = tuple(z3.Int(f"{node.id}_d{k}") for k in range(rank))
        return SymTensor(node.id, shape=shape)

    if node.op not in _SEMANTICS:
        raise KeyError(f"No symbolic conversion registered for graph op '{node.op}'")

    entry = _SEMANTICS.get(node.op)
    if entry is None:
        raise KeyError(f"No shape rule registered for symbolic op '{node.op}'")
    out_shape = symbolic_shape or _shape_rule_dims(entry, node, inputs)
    expr = SymExpr(
        node.op, [inp.expr for inp in inputs], out_shape, dict(node.attrs), node.id
    )
    return SymTensor(node.id, expr=expr)


def _graph_symbolic_tensors(G: nuGraph) -> dict[str, SymTensor]:
    out: dict[str, SymTensor] = {}
    for node in G.nodes:
        inputs = [out[inp] for inp in node.inputs]
        out[node.id] = _sym_expr_from_graph_node(node, inputs)
    return out


def _swap_composition_tensor(G: nuGraph, result_id: str) -> SymTensor | None:
    tensors = _graph_symbolic_tensors(G)
    return tensors.get(result_id)


_TENSOR_OPS: frozenset[str] = frozenset(
    {
        "add",
        "cumsum",
        "div",
        "divide",
        "exp",
        "matmul",
        "mul",
        "multiply",
        "reduce_sum",
        "relu",
        "silu",
        "sqrt",
        "subtract",
        "transpose",
        "rms_norm",
        "softmax",
        "activation",
        "activation_reduce",
        "dma_copy",
        "dma_transpose",
        "exponential",
        "nc_matmul",
        "nc_transpose",
        "reciprocal",
        "tensor_copy",
        "tensor_partition_reduce",
        "tensor_reduce",
        "tensor_scalar",
        "tensor_scalar_cumulative",
        "tensor_scalar_reduce",
        "tensor_tensor",
    }
)


def _node_output_reduction_axes(node: Node) -> frozenset[int]:
    op = node.op
    if op == "reduce_sum":
        axis = node.attrs.get("axis")
        if axis is None:
            if "keep_dims" in node.attrs:
                keep_dims = bool(node.attrs["keep_dims"])
            else:
                keep_dims = bool(node.attrs.get("keepdims", False))
            if not keep_dims:
                return frozenset()
            output_rank = len(node.shape or ())
            return frozenset(range(output_rank))
        axes = [axis] if isinstance(axis, int) else list(axis)
        return frozenset(int(a) for a in axes)
    if op == "tensor_reduce":
        axis_attr = node.attrs.get("axis", 1)
        axes = (
            list(axis_attr)
            if isinstance(axis_attr, (list, tuple))
            else [int(axis_attr)]
        )
        return frozenset(int(a) for a in axes)
    if op == "tensor_partition_reduce":
        return frozenset({0})
    return frozenset()


def _node_operand_reduction_axes(
    node: Node,
    operand_index: int,
    input_shape: tuple[Any, ...],
) -> frozenset[int]:
    del operand_index  # reserved for future multi-operand reduction ops
    op = node.op
    if op == "reduce_sum":
        axis = node.attrs.get("axis")
        if axis is None:
            return frozenset(range(len(input_shape)))
        axes = [axis] if isinstance(axis, int) else list(axis)
        return frozenset(int(a) for a in axes)
    if op == "tensor_reduce":
        return frozenset()
    if op == "tensor_partition_reduce":
        return frozenset({0})
    return frozenset()


def _symbolic_tile_dims_for_rank(rank: int) -> tuple[Any, ...]:
    return tuple(
        _DEFAULT_SYMBOLIC_TILE_DIMS[
            : builtins.min(rank, len(_DEFAULT_SYMBOLIC_TILE_DIMS))
        ]
    )


def _uses_default_symbolic_block_dims(block_dims: tuple[Any, ...], rank: int) -> bool:
    active = tuple(block_dims[: builtins.min(rank, len(block_dims))])
    return _symbolic_dim_names(active) == _SYMBOLIC_BLOCK_VAR_NAMES[: len(active)]


def _resolved_block_dims_for_shape(
    shape: tuple[Any, ...],
    block_dims: tuple[Any, ...],
    reduction_axes: frozenset[int] = frozenset(),
) -> tuple[Any, ...]:
    rank = builtins.min(len(shape), 2)
    if rank == 0:
        return ()
    use_shape_specific_defaults = _uses_default_symbolic_block_dims(block_dims, rank)
    resolved: list[Any] = []
    for axis in range(rank):
        dim = shape[axis]
        normalized_dim = _normalize_dim(dim)
        if axis in reduction_axes or normalized_dim == 1:
            resolved.append(1)
        elif use_shape_specific_defaults:
            resolved.append(_shape_specific_block_dim(normalized_dim))
        else:
            resolved.append(block_dims[axis])
    return tuple(resolved)


def _tiling_positive_assumptions(
    tile_dims: tuple[Any, ...],
    block_dims: tuple[Any, ...],
) -> tuple[z3.BoolRef, ...]:
    assumptions: list[z3.BoolRef] = []
    labeled_dims = [
        ("tile_dims", "tile dimension", i, dim) for i, dim in enumerate(tile_dims)
    ] + [("block_dims", "block dimension", i, dim) for i, dim in enumerate(block_dims)]
    for source, label, idx, dim in labeled_dims:
        if isinstance(dim, int):
            if dim <= 0:
                raise ValueError(
                    f"{source}[{idx}] ({label}) must be > 0 in the tiling model, got {dim}"
                )
            continue
        assumptions.append(_to_dim(dim) > z3.IntVal(0))
    return tuple(assumptions)


def _tile_candidate_dims_for_shape(
    shape: tuple[Any, ...],
) -> tuple[tuple[int, ...], ...]:
    n_dims = builtins.min(len(shape), 2)
    default_sizes = (
        _DEFAULT_PARTITION_TILE_SIZES,
        _DEFAULT_FREE_TILE_SIZE_CANDIDATES,
    )[:n_dims]
    candidates: list[tuple[int, ...]] = []
    for axis in range(n_dims):
        dim = _normalize_dim(shape[axis])
        if isinstance(dim, int) and dim == 1:
            candidates.append((1,))
        else:
            candidates.append(default_sizes[axis])
    return tuple(candidates)


def _tile_candidate_dims_for_exact_tile(
    tile_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple((int(dim),) for dim in tile_shape[:2])


def _tile_hardware_metadata_for_node(
    node: Node,
    output_shape: tuple[Any, ...],
    operand_shapes: tuple[tuple[str, tuple[Any, ...]], ...],
) -> TileHardwareMetadata:
    output_candidates = _tile_candidate_dims_for_shape(output_shape)
    tensor_tile_candidates: list[tuple[str, tuple[tuple[int, ...], ...]]] = [
        ("output", output_candidates)
    ]
    if node.op == "nc_transpose":
        fixed_candidates = _NC_TRANSPOSE_TILE_CANDIDATES[
            : builtins.min(len(output_shape), 2)
        ]
        tensor_tile_candidates = [("output", fixed_candidates)]
        tensor_tile_candidates.extend(
            (input_id, fixed_candidates) for input_id, _ in operand_shapes
        )
        return TileHardwareMetadata(
            concrete_tile_candidates=fixed_candidates,
            tensor_tile_candidates=tuple(tensor_tile_candidates),
        )
    if node.op == "nc_matmul":
        output_tile = (128, 512)
        stationary_tile = (128, 128)
        moving_tile = (128, 512)
        tensor_tile_candidates = [
            ("output", _tile_candidate_dims_for_exact_tile(output_tile))
        ]
        if operand_shapes:
            tensor_tile_candidates.append(
                (
                    operand_shapes[0][0],
                    _tile_candidate_dims_for_exact_tile(stationary_tile),
                )
            )
        if len(operand_shapes) > 1:
            tensor_tile_candidates.append(
                (operand_shapes[1][0], _tile_candidate_dims_for_exact_tile(moving_tile))
            )
        return TileHardwareMetadata(
            concrete_tile_candidates=_tile_candidate_dims_for_exact_tile(output_tile),
            operand_tile_shapes=(
                ("stationary", stationary_tile),
                ("moving", moving_tile),
            ),
            tensor_tile_candidates=tuple(tensor_tile_candidates),
        )
    tensor_tile_candidates.extend(
        (input_id, _tile_candidate_dims_for_shape(shape))
        for input_id, shape in operand_shapes
    )
    return TileHardwareMetadata(
        concrete_tile_candidates=output_candidates,
        tensor_tile_candidates=tuple(tensor_tile_candidates),
    )


def _tensor_tile_annotation(
    shape: tuple[Any, ...],
    tile_dims: tuple[Any, ...] | None,
    block_dims: tuple[Any, ...],
    reduction_axes: frozenset[int] = frozenset(),
) -> TensorTileAnnotation:
    rank = len(shape)
    n_dims = builtins.min(rank, 2)
    active_tile_dims = (
        tile_dims if tile_dims is not None else _symbolic_tile_dims_for_rank(n_dims)
    )
    t_dims: tuple[Any, ...] = tuple(active_tile_dims[:n_dims])
    b_dims: tuple[Any, ...] = _resolved_block_dims_for_shape(
        shape, block_dims, reduction_axes
    )
    assumptions = _tiling_positive_assumptions(t_dims, b_dims)

    strip: list[Any] = []
    for i in range(n_dims):
        b_i = b_dims[i]
        t_i = t_dims[i]
        if isinstance(b_i, int) and isinstance(t_i, int):
            block_size: Any = b_i * t_i
        else:
            block_size = _to_dim(b_i) * _to_dim(t_i)
        strip.append(sym_ceil_div(shape[i], block_size))

    return TensorTileAnnotation(
        tile_dims=t_dims,
        block_dims=b_dims,
        strip_dims=tuple(strip),
        shape=tuple(shape[:n_dims]),
        assumptions=assumptions,
    )


def _adjust_output_tiling_for_node(
    node: Node,
    output_tiling: TensorTileAnnotation,
    operand_tilings: tuple[tuple[str, TensorTileAnnotation], ...],
) -> TensorTileAnnotation:
    if node.op == "nc_transpose" and operand_tilings:
        input_tiling = operand_tilings[0][1]
        return TensorTileAnnotation(
            tile_dims=output_tiling.tile_dims,
            block_dims=output_tiling.block_dims,
            strip_dims=input_tiling.strip_dims,
            shape=output_tiling.shape,
            assumptions=output_tiling.assumptions,
        )
    return output_tiling


def tile_graph_variant(
    G: nuGraph,
    tile_dims: tuple[Any, ...] | None = None,
    block_dims: tuple[Any, ...] = _DEFAULT_BLOCK_DIMS,
) -> dict[str, TileAnnotation]:
    # The pipeline registers symbolic semantics before extraction.
    sym_tensors = _graph_symbolic_tensors(G)
    annotations: dict[str, TileAnnotation] = {}

    for node in G.nodes:
        if node.op == "input":
            continue
        if node.op not in _TENSOR_OPS:
            continue
        sym = sym_tensors.get(node.id)
        if sym is None:
            continue

        reduction_axes = _node_output_reduction_axes(node)
        output_tiling = _tensor_tile_annotation(
            sym.shape,
            tile_dims,
            block_dims,
            reduction_axes=reduction_axes,
        )
        operand_tilings: list[tuple[str, TensorTileAnnotation]] = []
        for operand_index, input_id in enumerate(node.inputs):
            input_sym = sym_tensors.get(input_id)
            if input_sym is None:
                continue
            operand_tilings.append(
                (
                    input_id,
                    _tensor_tile_annotation(
                        input_sym.shape,
                        tile_dims,
                        block_dims,
                        reduction_axes=_node_operand_reduction_axes(
                            node, operand_index, input_sym.shape
                        ),
                    ),
                )
            )
        output_tiling = _adjust_output_tiling_for_node(
            node,
            output_tiling,
            tuple(operand_tilings),
        )
        operand_shapes = tuple(
            (input_id, sym_tensors[input_id].shape)
            for input_id, _ in operand_tilings
            if input_id in sym_tensors
        )

        annotations[node.id] = TileAnnotation(
            tile_dims=output_tiling.tile_dims,
            block_dims=output_tiling.block_dims,
            strip_dims=output_tiling.strip_dims,
            reduction_axes=reduction_axes,
            assumptions=output_tiling.assumptions,
            hardware_metadata=_tile_hardware_metadata_for_node(
                node,
                sym.shape,
                operand_shapes,
            ),
            operand_tilings=tuple(operand_tilings),
        )

    return annotations


def format_tile_annotation(
    node_id: str,
    ann: TileAnnotation,
    include_node_id: bool = True,
) -> str:
    def _format_tensor_tiling(tiling: TensorTileAnnotation) -> str:
        n = ", ".join(_format_dim(d, tiling.assumptions) for d in tiling.strip_dims)
        b = ", ".join(_format_dim(d, tiling.assumptions) for d in tiling.block_dims)
        t = ", ".join(_format_dim(d, tiling.assumptions) for d in tiling.tile_dims)
        return f"[{n}, {b}] [{t}]"

    output_tiling = _format_tensor_tiling(
        TensorTileAnnotation(
            tile_dims=ann.tile_dims,
            block_dims=ann.block_dims,
            strip_dims=ann.strip_dims,
            assumptions=ann.assumptions,
        )
    )
    operand_str = ""
    if ann.operand_tilings:
        operand_parts = [
            f"{operand_id}={_format_tensor_tiling(tiling)}"
            for operand_id, tiling in ann.operand_tilings
        ]
        operand_str = f" operands=({', '.join(operand_parts)})"
    if include_node_id:
        return f"output={output_tiling} {node_id}{operand_str}"
    return f"output={output_tiling}{operand_str}"


def format_tile_hardware_metadata(metadata: TileHardwareMetadata | None) -> str:
    if metadata is None:
        return ""
    parts: list[str] = []
    if metadata.tensor_tile_candidates:
        tensor_candidates = ", ".join(
            f"{tensor_id}={candidates}"
            for tensor_id, candidates in metadata.tensor_tile_candidates
        )
        parts.append(f"hw_tile_candidates=({tensor_candidates})")
    elif metadata.concrete_tile_candidates:
        parts.append(f"hw_tile_candidates={metadata.concrete_tile_candidates}")
    if metadata.operand_tile_shapes:
        parts.append(f"operand_tiles={metadata.operand_tile_shapes}")
    return " ".join(parts)


def _graph_from_axon_array(out: AxonArray | tuple[AxonArray, ...]) -> nuGraph:
    outs = (out,) if isinstance(out, AxonArray) else tuple(out)
    if not outs:
        raise ValueError("kernel returned no outputs")
    # Merge nodes from all outputs (deduped by id) and record the output
    # ordering on the graph so codegen / bench can emit a tuple return.
    merged: list[Any] = []
    seen: set[str] = set()
    for o in outs:
        if not isinstance(o, AxonArray):
            raise TypeError(
                "kernel must return AxonArray or tuple[AxonArray, ...]; "
                f"got element of type {type(o).__name__}"
            )
        for node in o.nodes:
            if node.id in seen:
                continue
            merged.append(node)
            seen.add(node.id)
    G = nuGraph(
        [
            Node(id=n.id, op=n.op, inputs=list(n.inputs), attrs=dict(n.attrs))
            for n in merged
        ]
    )
    G.output_ids = tuple(o.node_id for o in outs)
    annotate_shapes_concrete(G)
    return G


def build_graph_from_kernel(
    kernel,
    *inputs: tuple[str, tuple[str | int, ...]],
    dim_sizes: dict[str, int],
) -> nuGraph:
    """Lift `kernel` into a concrete-shape nuGraph, resolving each input's
    sym_shape (int literals kept as-is, dim names looked up in the required
    `dim_sizes`; a missing name raises rather than inventing a size)."""

    def _resolve_dim(d: str | int, name: str, dim_sizes: dict[str, int]) -> int:
        if isinstance(d, int):
            return int(d)
        if d in dim_sizes:
            return dim_sizes[d]
        raise ValueError(
            f"build_graph_from_kernel: symbolic dim {d!r} of input "
            f"{name!r} has no binding in dim_sizes (keys: "
            f"{sorted(dim_sizes)}); dim_sizes must cover every symbolic "
            f"dim the spec declares"
        )

    def _make_arg(spec: tuple[str, tuple[str | int, ...]]) -> AxonArray:

        name, sym_shape = spec
        concrete_shape = tuple(_resolve_dim(d, name, dim_sizes) for d in sym_shape)
        return AxonArray(name, concrete_shape, sym_shape=sym_shape)

    args = list(map(_make_arg, inputs))
    out = kernel(*args)
    G = _graph_from_axon_array(out)
    G.input_ids = tuple(arg.node_id for arg in args)
    return G


def _symbolic_dim_names(dims: tuple[Any, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for dim in dims:
        if isinstance(dim, z3.ArithRef) and z3.is_const(dim) and dim.num_args() == 0:
            names.append(dim.decl().name())
        else:
            names.append(_format_dim(dim))
    return tuple(names)
