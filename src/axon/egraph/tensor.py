"""Tensor e-graph construction and saturation from one traced ``nuGraph``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import z3

from axon.egraph.adapter import EGraphAdapter, Snapshot
from axon.egraph.analysis import analyze_snapshot, decode_tensor
from axon.egraph.codec import CodecError, encode_tensor_enode, encode_tensor_input
from axon.egraph.proof import ProofStore
from axon.egraph.propagation import (
    PropagationWorklist,
    RoundResult,
    run_propagation_round,
)
from axon.egraph.saturation import RoundOutcome, run_saturation_loop
from axon.egraph.saturation import SaturationStatus as BaseSaturationStatus
from axon.egraph.workers import resolve_worker_count
from axon.ir import Node, nuGraph
from axon.isa_semantics import ShapeExpr, lookup_semantics

# Traced-node canonicalization: the tensor analogue of ``sketch_node_attrs``.
# This allowlist is the exact set of insertable tensor operations.

_TENSOR_OP_ALIASES = {"multiply": "mul", "divide": "div"}

_TENSOR_BINARY = frozenset({"add", "subtract", "mul", "div"})

_TENSOR_AXIS_UNARY = frozenset({"softmax", "cumsum"})

_TENSOR_NO_ATTR_OPS = ("broadcast", "sqrt", "exp", "relu", "silu", "transpose")

_TENSOR_ALLOWED_ATTRS: dict[str, frozenset[str]] = {
    "input": frozenset({"shape", "sym_shape"}),
    "matmul": frozenset({"transpose_x"}),
    "reduce_sum": frozenset({"axis", "keep_dims", "keepdims"}),
    **{op: frozenset({"scalar", "reverse"}) for op in _TENSOR_BINARY},
    **{op: frozenset({"axis"}) for op in _TENSOR_AXIS_UNARY},
    **dict.fromkeys(_TENSOR_NO_ATTR_OPS, frozenset()),
}

TENSOR_OPS = frozenset(_TENSOR_ALLOWED_ATTRS) - {"input"}


def normalize_axes(axis_attr: Any, rank: int) -> tuple[int, ...]:
    """Normalize an axis attribute to non-negative, ascending, distinct axes."""
    if axis_attr is None:
        raise CodecError("axis attribute is required and may not be None")
    raw = [axis_attr] if isinstance(axis_attr, int) else list(axis_attr)
    out: list[int] = []
    for axis in raw:
        if not isinstance(axis, int) or isinstance(axis, bool):
            raise CodecError(f"non-integer axis {axis!r}")
        normalized = axis + rank if axis < 0 else axis
        if normalized < 0 or normalized >= rank:
            raise CodecError(f"axis {axis} out of range for rank {rank}")
        out.append(normalized)
    ordered = tuple(sorted(out))
    if len(set(ordered)) != len(ordered):
        raise CodecError(f"duplicate axes in {axis_attr!r}")
    return ordered


def _encode_scalar(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CodecError(f"unsupported scalar attribute value {value!r}")
    return float(value)


def canonical_tensor_parts(
    node: Node, child_ranks: list[int]
) -> tuple[str, dict[str, Any]]:
    """Canonicalize one traced node to its exact (op, attrs) e-node identity."""
    op = _TENSOR_OP_ALIASES.get(node.op, node.op)
    allowed = _TENSOR_ALLOWED_ATTRS.get(op)
    if allowed is None:
        raise CodecError(f"no tensor encoding declared for operation '{node.op}'")
    attrs = {k: v for k, v in node.attrs.items() if k != "name"}
    unknown = set(attrs) - set(allowed)
    if unknown:
        raise CodecError(
            f"tensor operation '{op}' has undeclared attributes {sorted(unknown)}"
        )

    canonical: dict[str, Any] = {}
    if op == "input":
        canonical["shape"] = tuple(attrs.get("shape", ()))
        if "sym_shape" in attrs:
            canonical["sym_shape"] = tuple(attrs["sym_shape"])
    elif op in _TENSOR_BINARY:
        if len(node.inputs) == 2:
            if "scalar" in attrs or "reverse" in attrs:
                raise CodecError(
                    f"two-input '{op}' must not carry scalar-form attributes"
                )
        elif len(node.inputs) == 1:
            if "scalar" not in attrs:
                raise CodecError(f"one-input '{op}' requires a scalar attribute")
            canonical["scalar"] = _encode_scalar(attrs["scalar"])
            canonical["reverse"] = bool(attrs.get("reverse", False))
        else:
            raise CodecError(f"'{op}' with {len(node.inputs)} inputs is unsupported")
    elif op == "matmul":
        canonical["transpose_x"] = bool(attrs.get("transpose_x", False))
    elif op == "reduce_sum":
        canonical["axis"] = normalize_axes(attrs.get("axis"), child_ranks[0])
        canonical["keep_dims"] = bool(
            attrs.get("keep_dims", attrs.get("keepdims", False))
        )
    elif op in _TENSOR_AXIS_UNARY:
        (canonical["axis"],) = normalize_axes(attrs.get("axis", -1), child_ranks[0])
    return op, canonical


def encode_tensor_node(
    node: Node, child_handles: list[Any], child_ranks: list[int]
) -> Any:
    """Canonicalize one traced node and encode it in the generic language."""
    op, attrs = canonical_tensor_parts(node, child_ranks)
    if op == "input":
        dims = attrs.get("sym_shape") or attrs["shape"]
        return encode_tensor_input(node.id, tuple(dims))
    return encode_tensor_enode(op, attrs, child_handles)


@dataclass
class TensorIngest:
    """The handles and metadata from tensor graph ingestion."""

    adapter: EGraphAdapter
    node_handles: dict[str, Any] = field(default_factory=dict)
    input_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    output_ids: tuple[str, ...] = ()
    output_handles: list[Any] = field(default_factory=list)


def _node_rank(node: Any) -> int:
    if node.shape is not None:
        return len(node.shape)
    shape = node.attrs.get("shape")
    if shape is not None:
        return len(shape)
    raise CodecError(
        f"node '{node.id}' has no annotated shape; run annotate_shapes_concrete "
        "before ingest"
    )


def _declared_output_ids(G: nuGraph) -> tuple[str, ...]:
    if G.output_ids:
        return tuple(G.output_ids)
    consumed = {inp for node in G.nodes for inp in node.inputs}
    return tuple(node.id for node in G.nodes if node.id not in consumed)


def ingest_tensor_graph(adapter: EGraphAdapter, G: nuGraph) -> TensorIngest:
    """Encode one topologically ordered graph in the tensor e-graph."""
    ingest = TensorIngest(adapter=adapter)
    ranks: dict[str, int] = {}
    for node in G.nodes:
        if node.op == "input":
            expr = encode_tensor_node(node, [], [])
            ingest.input_metadata[node.id] = {
                "shape": tuple(node.attrs.get("shape", node.shape or ())),
                "sym_shape": (
                    tuple(node.attrs["sym_shape"])
                    if "sym_shape" in node.attrs
                    else None
                ),
            }
        else:
            missing = [inp for inp in node.inputs if inp not in ingest.node_handles]
            if missing:
                raise CodecError(
                    f"node '{node.id}' references unencoded inputs {missing}; "
                    "the graph is not topologically ordered"
                )
            children = [ingest.node_handles[inp] for inp in node.inputs]
            child_ranks = [ranks[inp] for inp in node.inputs]
            expr = encode_tensor_node(node, children, child_ranks)
        result = adapter.intern_expr(expr, provenance=node.id)
        ingest.node_handles[node.id] = result.handle
        ranks[node.id] = _node_rank(node)

    ingest.output_ids = _declared_output_ids(G)
    missing_outputs = [
        oid for oid in ingest.output_ids if oid not in ingest.node_handles
    ]
    if missing_outputs:
        raise CodecError(f"declared outputs {missing_outputs} are not graph nodes")
    ingest.output_handles = [ingest.node_handles[oid] for oid in ingest.output_ids]
    return ingest


def is_eligible_tensor_op(op: str) -> bool:
    """True for a non-input tensor operation with a registered encoding."""
    return op in TENSOR_OPS


def _tensor_output_rank(op: str, attrs: dict[str, Any], child_ranks: list[int]) -> int:
    """Result rank of one tensor operation from its registered shape rule."""
    try:
        entry = lookup_semantics(op)
    except KeyError as exc:
        raise CodecError(str(exc)) from exc
    input_shapes = [
        ShapeExpr([z3.Int(f"_r{i}_{d}") for d in range(rank)])
        for i, rank in enumerate(child_ranks)
    ]
    return len(entry.shape_rule(input_shapes, dict(attrs)).out.dims)


def encode_tensor_candidate(
    op: str,
    attrs: dict[str, Any],
    child_exprs: list[Any],
    child_ranks: list[int],
) -> tuple[Any, int]:
    """Encode one tensor application and return its expression and rank."""
    node = Node(
        id="prop_candidate",
        op=op,
        inputs=[f"_c{i}" for i in range(len(child_exprs))],
        attrs=dict(attrs),
    )
    expr = encode_tensor_node(node, list(child_exprs), list(child_ranks))
    return expr, _tensor_output_rank(op, attrs, child_ranks)


def _analyze_tensor(snapshot: Snapshot, decode: Any) -> dict[Any, Any]:
    return analyze_snapshot(snapshot, decode)


@dataclass(frozen=True)
class SaturationStatus(BaseSaturationStatus):
    """Reports whether tensor saturation reached a fixed point or a limit.
    ``pass_limit`` is separate from resource truncation."""

    stage: str = "tensor_propagation"


def saturate_tensor(
    adapter: EGraphAdapter,
    store: ProofStore | None = None,
    *,
    max_rounds: int = 50,
    max_passes: int | None = None,
    wall_clock_seconds: float = 300.0,
    timeout: int = 10000,
    deadline: float | None = None,
    semi_naive: bool = True,
    workers: int | None = None,
    guard_shared_producers: bool = True,
) -> SaturationStatus:
    """Runs tensor propagation until a fixed point or a limit.
    ``max_passes`` counts rounds that change the graph.

    ``guard_shared_producers`` is threaded to ``eligible_occurrences``: it is on by
    default, and passing ``False`` runs the unguarded enumeration, which is how an
    eval measures what the guards cost in reachable rewrites."""
    workers = resolve_worker_count(workers)
    if store is None:
        store = ProofStore()
    worklist = PropagationWorklist() if semi_naive else None

    def run_round(round_deadline: float | None) -> RoundOutcome:
        result: RoundResult = run_propagation_round(
            adapter,
            decode_tensor,
            encode_tensor_candidate,
            is_eligible_tensor_op,
            store,
            analyze=_analyze_tensor,
            stage="tensor_propagation",
            timeout=timeout,
            worklist=worklist,
            deadline=round_deadline,
            workers=workers,
            guard_shared_producers=guard_shared_producers,
        )
        return RoundOutcome(
            enodes_added=result.enodes_added,
            equalities_added=result.equalities_added,
            truncated_reason=result.truncated_reason,
            truncated_stage=result.truncated_stage,
        )

    return run_saturation_loop(
        run_round,
        status_cls=SaturationStatus,
        stage="tensor_propagation",
        store=store,
        max_rounds=max_rounds,
        max_passes=max_passes,
        wall_clock_seconds=wall_clock_seconds,
        deadline=deadline,
    )
