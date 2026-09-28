"""The single graph-interpretation layer for the matmul-family emitter.

``build_emission_plan`` walks the hardware graph ONCE, classifies every compute
node into at least one role, and freezes the result into an ``EmissionPlan``.
The renderer (``bodies/matmul_generic.py``) translates that plan without
re-walking edges; unclassified nodes refuse with a typed ``UnsupportedEmission``
before any emission. Values shared between a reduce chain and an operand chain
become staged values — the only deliberate overlap.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from axon.codegen.constants import (
    DEFAULT_MATMUL_TILE,
    MOVING_FMAX,
    PARTITION_FMAX,
)
from axon.codegen.layout import (
    DimVar,
    Layout,
    _extent_key,
    _is_unary_constant_tensor_scalar,
    _normalize_rank2_reduce_axis,
    topo_order,
)
from axon.codegen.nest import Nest, derive_nest
from axon.codegen.ops import UnsupportedEmission
from axon.codegen.structure import collect_reduce_input_chain
from axon.ir import Node

# Unary elementwise ops that may sit on a matmul operand chain (each reads its
# data from input 0; relu/silu are `activation`, softmax's numerator is
# `exponential`, a reciprocal denominator is `reciprocal`). Anything else on the
# chain (reduce, tensor_tensor, side/chained matmul) is out of scope here.
_ELEMENTWISE_OPS = frozenset({"activation", "exponential", "reciprocal"})

# Shape-only operations that preserve the emitted tile value. A broadcast on a
# reduce scalar remains a (P, 1) tile because its consumer performs the
# hardware broadcast.
_PASSTHROUGH_OPS = frozenset({"broadcast"})

# Reduce ops whose output is an (m,) per-partition scalar produced by the
# k_side preamble (class B.2). They are never on an operand chain themselves;
# their input chain and post-reduce scalar chain are emitted as a preamble.
_REDUCE_OPS = frozenset({"tensor_reduce", "activation_reduce"})
_TENSOR_REDUCE_ATTRS = frozenset(
    {"op", "axis", "negate", "keepdims", "keep_dims", "name"}
)

# Combiner ops valid on an operand chain when exactly one operand is an (m,)
# reduce-preamble scalar; two full-tile operands (fan-in) are out of scope.
_COMBINER_OPS = frozenset({"tensor_scalar", "tensor_tensor"})

# Ops an inter-matmul node (between the earlier matmuls and the output, in a
# side- or chained-matmul graph) may be. Anything else is refused at plan time.
_INTER_OPS = frozenset(
    {
        "activation",
        "exponential",
        "reciprocal",
        "nc_transpose",
        "tensor_tensor",
        "tensor_scalar",
        "scalar_tensor_tensor",
        "tensor_copy",
        "broadcast",
    }
)


# --------------------------------------------------------------------------- #
# Graph-walking primitives (all edge-following confined here).
# --------------------------------------------------------------------------- #
def _input_ancestors(
    nid: str, id_to_node: dict[str, Node], input_node_ids: set[str]
) -> set[str]:
    """The graph *input* ids reachable upstream from `nid` (or `nid` itself if
    it is an input): the operand-role identity of a matmul input."""
    roots: set[str] = set()
    stack = [nid]
    seen: set[str] = set()
    while stack:
        curr = stack.pop()
        if curr in seen:
            continue
        seen.add(curr)
        if curr in input_node_ids:
            roots.add(curr)
            continue
        n = id_to_node.get(curr)
        if n is None:
            continue
        for inp in n.inputs or []:
            stack.append(inp)
    return roots


def _node_ancestors(
    nid: str, id_to_node: dict[str, Node], input_node_ids: set[str]
) -> set[str]:
    """Every value id on the path ending at ``nid``, including ``nid``."""
    ancestors: set[str] = set()
    stack = [nid]
    while stack:
        curr = stack.pop()
        if curr in ancestors:
            continue
        ancestors.add(curr)
        if curr in input_node_ids:
            continue
        node = id_to_node.get(curr)
        if node is not None:
            stack.extend(node.inputs)
    return ancestors


def scalar_node_ids(compute_nodes: list[Node], id_to_node: dict[str, Node]) -> set[str]:
    """Node ids whose emitted tile is an ``(m,)`` per-partition scalar: a reduce
    node (free axis consumed) or a unary elementwise op applied to a scalar
    (rsqrt / reciprocal on a row-reduce)."""
    scalar_ids: set[str] = set()
    for node in topo_order(compute_nodes, id_to_node):
        if node.op in _REDUCE_OPS or (
            (
                node.op in (_ELEMENTWISE_OPS | _PASSTHROUGH_OPS)
                or _is_unary_constant_tensor_scalar(node)
            )
            and node.inputs
            and node.inputs[0] in scalar_ids
        ):
            scalar_ids.add(node.id)
    return scalar_ids


def _node_shape(node: Node) -> tuple[Any, ...] | None:
    shape = node.shape
    if shape is None:
        shape = node.attrs.get("shape")
    return None if shape is None else tuple(shape)


def _dims_equal_exactly(lhs: Any, rhs: Any) -> bool:
    """Extents reach codegen concrete, so identical normalized keys is equality."""
    return _extent_key(lhs) == _extent_key(rhs)


def _shapes_equal_exactly(
    lhs: tuple[Any, ...] | None,
    rhs: tuple[Any, ...] | None,
) -> bool:
    return (
        lhs is not None
        and rhs is not None
        and len(lhs) == len(rhs)
        and all(_dims_equal_exactly(a, b) for a, b in zip(lhs, rhs, strict=True))
    )


def _combiner_scalar_input(node: Node, scalar_ids: set[str]) -> str | None:
    """The lone scalar-operand id of a combiner node, or ``None`` if it is not a
    single-scalar combiner (raw fan-in of two full tiles, or not a combiner)."""
    if node.op not in _COMBINER_OPS or len(node.inputs) != 2:
        return None
    scal = [i for i in node.inputs if i in scalar_ids]
    return scal[0] if len(scal) == 1 else None


def _validate_broadcast_aliases(
    compute_nodes: list[Node],
    id_to_node: dict[str, Node],
    layouts: dict[str, Layout],
    scalar_ids: set[str],
) -> None:
    """Prove every broadcast discarded by the renderer is an exact alias."""
    consumers: dict[str, list[Node]] = {}
    for consumer in compute_nodes:
        for input_id in consumer.inputs:
            consumers.setdefault(input_id, []).append(consumer)

    for node in compute_nodes:
        if node.op != "broadcast":
            continue
        if not node.inputs or node.inputs[0] not in layouts:
            raise UnsupportedEmission(
                f"generic matmul: broadcast {node.id!r} has no data input"
            )

        data_id = node.inputs[0]
        output_shape = _node_shape(node)
        data_shape = _node_shape(id_to_node[data_id])

        if data_id not in scalar_ids:
            if not _shapes_equal_exactly(output_shape, data_shape):
                raise UnsupportedEmission(
                    f"generic matmul: full-tile broadcast {node.id!r} cannot "
                    f"be elided because output shape {output_shape!r} does not "
                    f"exactly equal data shape {data_shape!r}"
                )
            if not layouts[node.id].same(layouts[data_id]):
                raise UnsupportedEmission(
                    f"generic matmul: full-tile broadcast {node.id!r} cannot "
                    "be elided because its inferred layout differs from its "
                    "data input"
                )
            continue

        if (
            data_shape is None
            or len(data_shape) != 2
            or not _dims_equal_exactly(data_shape[1], 1)
        ):
            raise UnsupportedEmission(
                f"generic matmul: scalar broadcast {node.id!r} cannot be "
                f"normalized because source {data_id!r} must have exact "
                f"rank-2 shape with singleton free dimension, got "
                f"{data_shape!r}"
            )
        uses = consumers.get(node.id, [])
        if not uses:
            raise UnsupportedEmission(
                f"generic matmul: scalar broadcast {node.id!r} is not consumed "
                "by a supported scalar combiner"
            )
        for consumer in uses:
            if _combiner_scalar_input(consumer, scalar_ids) != node.id:
                raise UnsupportedEmission(
                    f"generic matmul: scalar broadcast {node.id!r} is consumed "
                    f"by unsupported node {consumer.id!r} (op={consumer.op})"
                )
            full_tile_id = next(
                input_id for input_id in consumer.inputs if input_id != node.id
            )
            full_tile_shape = _node_shape(id_to_node[full_tile_id])
            if not _shapes_equal_exactly(output_shape, full_tile_shape):
                raise UnsupportedEmission(
                    f"generic matmul: scalar broadcast {node.id!r} cannot be "
                    f"normalized for combiner {consumer.id!r} because output "
                    f"shape {output_shape!r} does not exactly equal full-tile "
                    f"shape {full_tile_shape!r}"
                )
            if not layouts[consumer.id].same(layouts[full_tile_id]):
                raise UnsupportedEmission(
                    f"generic matmul: scalar broadcast {node.id!r} cannot be "
                    f"normalized for combiner {consumer.id!r} because the "
                    "combiner and full-tile layouts differ"
                )


def _data_pred_input(node: Node, scalar_ids: set[str]) -> str:
    """The input a matmul-operand chain continues through: input 0 for a unary
    elementwise / transpose, the non-scalar operand of a single-scalar combiner.
    Raises for anything else (fan-in, non-unary)."""
    sid = _combiner_scalar_input(node, scalar_ids)
    if sid is not None:
        others = [i for i in node.inputs if i != sid]
        return others[0]
    if _is_unary_constant_tensor_scalar(node) and node.inputs[0] in scalar_ids:
        return node.inputs[0]
    if node.op in _PASSTHROUGH_OPS:
        if not node.inputs:
            raise UnsupportedEmission(
                f"generic matmul: operand chain passthrough {node.id!r} "
                f"(op={node.op}) has no data input"
            )
        return node.inputs[0]
    if node.op in _ELEMENTWISE_OPS or node.op == "nc_transpose":
        if len(node.inputs) != 1:
            raise UnsupportedEmission(
                f"generic matmul: operand chain node {node.id!r} (op={node.op}) "
                f"is not unary (inputs={node.inputs})"
            )
        return node.inputs[0]
    raise UnsupportedEmission(
        f"generic matmul: operand chain node {node.id!r} (op={node.op}) is not "
        f"a unary elementwise op, nc_transpose, or single-scalar combiner — "
        f"reduces / fan-ins / side matmuls are handled elsewhere"
    )


# --------------------------------------------------------------------------- #
# Frozen plan dataclasses — exact node ids, roles, layouts everywhere.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChainStep:
    """One node on an operand / post-matmul / staged chain, with the exact edge
    the renderer must bind at line-emission time: the full-tile data predecessor
    it reads (a raw input, an upstream chain buffer, or a matmul result) and the
    optional ``(m,)`` reduce-preamble scalar operand of a single-scalar combiner.
    ``op`` is copied for convenience; ``node_id`` indexes ``id_to_node`` for the
    op emitter's attrs."""

    node_id: str
    op: str
    data_pred_id: str
    scalar_id: str | None


@dataclass(frozen=True)
class OperandPlan:
    """The ordered chain from an HBM input (or an upstream mm/combine value) to a
    matmul operand, walked ONCE here. ``steps`` are in input→operand order; each
    entry is tagged (op kind + edges). ``root_id`` is the raw input the chain is
    rooted at (or the operand itself when the chain is empty / the operand is a
    materialized value). ``transpose_index`` is the index of the sole
    ``nc_transpose`` in ``steps`` (or ``None``); the layout of the operand comes
    from ``layouts[op_id]``."""

    op_id: str
    steps: tuple[ChainStep, ...]
    root_id: str
    transpose_index: int | None
    is_materialized: bool = False  # operand IS an inter node (chained mm stat)

    @property
    def node_ids(self) -> frozenset[str]:
        return frozenset(s.node_id for s in self.steps)


@dataclass(frozen=True)
class MatmulPlan:
    """One ``nc_matmul`` in topo order. ``nc_matmul`` computes ``inputs[0]ᵀ @
    inputs[1]``: the stationary operand is asserted (at plan build) to derive
    from ``inputs[0]`` and the moving operand from ``inputs[1]`` — no
    re-derivation downstream. ``accum_dim`` is the contraction canonical name;
    ``is_chained`` is set when the stationary operand is another matmul's
    materialized combine (not a raw-input chain)."""

    mm_id: str
    stat: OperandPlan
    mov: OperandPlan
    accum_dim: str
    is_chained: bool
    # True when the moving free dim is dual-role ("part","wide"): slice at
    # MATMUL_TILE_<D> rather than the partition-sized TILE_<D>.
    mov_wide: bool = False


@dataclass(frozen=True)
class ReduceInputPlan:
    """One HBM source used by a reduce preamble.

    ``part_broadcast`` and ``free_broadcast`` record rank-2 unit dimensions so
    the renderer loads the source at its actual shape instead of slicing every
    source as a full ``(M, K)`` tile.
    """

    input_id: str
    part_broadcast: bool
    free_broadcast: bool


@dataclass(frozen=True)
class ReducePlan:
    """One reduce preamble (class B.2): the ``(m,)`` reduce node, the full-tile
    elementwise chain feeding it (over HBM inputs, topo input→reduce order) with
    its ordered HBM source ids, the free-axis it reduces (validated to axis=1),
    and the post-reduce per-partition-scalar chain (rsqrt / reciprocal on the
    (m,) scalar). ``input_id`` is the reduce node's direct data predecessor (the
    graph edge ``reduce.inputs[0]``) the renderer reduces over, carried here so
    the renderer never reads ``Node.inputs`` to find it. ``source_ids`` names
    stage-local matmul or materialized values where the reduce input chain
    starts instead of HBM. All exact ids."""

    reduce_id: str
    input_id: str
    axis: int | None
    chain_ids: tuple[str, ...]
    hbm_inputs: tuple[ReduceInputPlan, ...]
    post_scalar_ids: tuple[str, ...]
    source_ids: tuple[str, ...] = ()

    @property
    def hbm_ids(self) -> tuple[str, ...]:
        return tuple(item.input_id for item in self.hbm_inputs)

    @property
    def all_ids(self) -> frozenset[str]:
        return frozenset({self.reduce_id, *self.chain_ids, *self.post_scalar_ids})


@dataclass(frozen=True)
class InterNodePlan:
    """One inter-matmul node (between the earlier matmuls and the output in a
    side-/chained-matmul graph): its id, its op (for the transpose-vs-combine
    render branch), and the EXACT ordered input edges (``Node.inputs``) the
    renderer binds to producer buffers. Carrying ``input_ids`` here is what lets
    the renderer emit the combine/transpose without ever reading ``Node.inputs``:
    structure (which producers feed this node) is decided at plan time."""

    node_id: str
    op: str
    input_ids: tuple[str, ...]


@dataclass(frozen=True)
class EmissionStagePlan:
    """One dependency-ordered multi-matmul execution stage.

    ``node_ids`` is the exact topological order for the stage. ``live_in_ids``
    and ``live_out_ids`` name values crossing stage boundaries, including values
    carried across an unrelated intermediate stage."""

    index: int
    node_ids: tuple[str, ...]
    matmul_ids: tuple[str, ...]
    inters: tuple[InterNodePlan, ...] = ()
    reduces: tuple[ReducePlan, ...] = ()
    live_in_ids: tuple[str, ...] = ()
    live_out_ids: tuple[str, ...] = ()

    @property
    def inter_ids(self) -> tuple[str, ...]:
        return tuple(inter.node_id for inter in self.inters)


@dataclass(frozen=True)
class EmissionPlan:
    """Frozen graph interpretation handed to the renderer.

    Carries all structural facts (chains, reduce groups, staged values,
    inter-node edges, materialized combine id). The renderer receives only a
    ``NodeAttrs`` (op + attrs, no edges) alongside this plan."""

    layouts: dict[str, Layout]
    nest: Nest
    scalar_ids: frozenset[str]
    output_id: str
    # The dim whose BASE tile is the wide moving granularity (a purely-free
    # directly-loaded moving operand dim): the one ``_tile_base`` bases off
    # ``tile_n``.
    wide_dim: str | None
    # Per-dim tile roles: "part" (partition axis / contraction) or "wide" (free
    # moving dim). Dual-role dims get both TILE_<D> and MATMUL_TILE_<D>.
    tile_roles: dict[str, tuple[str, ...]]
    # Resolved TILE_<D> per present dim (see _resolve_tiles); the renderer emits
    # each as a flat literal.
    tiles: dict[str, int]

    matmuls: tuple[MatmulPlan, ...]
    # single-matmul (class A-D) fields
    post_mm: tuple[ChainStep, ...] = ()
    reduces: tuple[ReducePlan, ...] = ()
    staged_ids: frozenset[str] = frozenset()
    staged_dtype_sources: tuple[tuple[str, str], ...] = ()
    # multi-matmul (class E, F) fields
    is_multi: bool = False
    chained_mm_id: str | None = None
    inters: tuple[InterNodePlan, ...] = ()
    materialized_inter_id: str | None = None
    stages: tuple[EmissionStagePlan, ...] = ()
    # Crossing values that are per-partition scalars. They cross in SBUF, through
    # the reducer's existing `(TILE_<part>, TILES_IN_BLOCK_<part>, 1)` buffer, so
    # they take no HBM buffer and every HBM site must exclude them. Defined once
    # here because four sites act on a crossing value independently.
    sbuf_scalar_crossings: frozenset[str] = frozenset()
    # Raw canon dim -> schedule role, collapsing a staged nest's dims onto the
    # four the emitted signature carries (see ``resolve_tile_roles``); the
    # identity when no collapse is needed, and empty on a non-staged plan.
    dim_aliases: dict[str, str] = field(default_factory=dict)
    # One representative DimVar per role, for the preamble's extent aliasing.
    role_dims: dict[str, DimVar] = field(default_factory=dict)

    @property
    def inter_ids(self) -> tuple[str, ...]:
        return tuple(ip.node_id for ip in self.inters)

    def matmul(self, mm_id: str) -> MatmulPlan:
        for mp in self.matmuls:
            if mp.mm_id == mm_id:
                return mp
        raise KeyError(mm_id)


@dataclass(frozen=True)
class _NodeView:
    """A node's ``op`` and ``attrs`` only — no ``inputs`` field, so edge access
    raises ``AttributeError`` (structural, not by convention)."""

    op: str
    attrs: Mapping[str, Any]


class NodeAttrs:
    """Restricted id-to-node view: maps a node id to its ``_NodeView`` (op + attrs).
    ``emit`` fetches the real ``Node`` internally for the op emitter."""

    __slots__ = ("_nodes",)

    def __init__(self, id_to_node: dict[str, Node]) -> None:
        self._nodes = id_to_node

    def __getitem__(self, node_id: str) -> _NodeView:
        n = self._nodes[node_id]
        return _NodeView(op=n.op, attrs=n.attrs)

    def emit(self, ctx, node_id: str, id_to_var: dict[str, str]) -> str | None:
        """Emit one node's NKI call string. The real ``Node`` (with edges) is
        fetched here and handed to the op emitter, which reads ``Node.inputs``
        only to index the renderer-supplied ``id_to_var``."""
        from axon.codegen.ops import emit_or_raise

        return emit_or_raise(ctx, self._nodes[node_id], id_to_var)

    def emit_activation_reduce_to(
        self, ctx, node_id: str, id_to_var: dict[str, str], act_dst: str
    ) -> str:
        """``emit`` for an ``activation_reduce`` whose (P, F) activation output has
        a real destination (a staged buffer slice) instead of a throwaway tile."""
        from axon.codegen.ops import emit_activation_reduce_to

        return emit_activation_reduce_to(
            ctx, self._nodes[node_id], id_to_var, act_dst=act_dst
        )


def find_output_sink(compute_nodes: list[Node]) -> str:
    """The sole non-consumed compute node id (the kernel output)."""
    consumed: set[str] = set()
    for n in compute_nodes:
        consumed.update(n.inputs or [])
    sinks = [n for n in compute_nodes if n.id not in consumed]
    if len(sinks) != 1:
        raise UnsupportedEmission(
            f"generic matmul (class A): expected one output sink, got "
            f"{[n.id for n in sinks]}"
        )
    return sinks[0].id


# --------------------------------------------------------------------------- #
# Chain builders.
# --------------------------------------------------------------------------- #
def _build_chain_steps(
    op_id: str,
    id_to_node: dict[str, Node],
    input_node_ids: set[str],
    scalar_ids: set[str],
    stop_ids: set[str] | None = None,
) -> tuple[list[ChainStep], str]:
    """Walk from ``op_id`` back toward a raw input (or a ``stop_ids`` boundary),
    tagging each compute node with its data predecessor and scalar operand.
    Returns ``(steps input→op order, root_id)``. Raises for any node that is not
    a unary elementwise / transpose / single-scalar combiner, or a broken
    chain."""
    stop_ids = stop_ids or set()
    steps: list[ChainStep] = []
    curr = op_id
    seen: set[str] = set()
    while curr not in input_node_ids and curr not in stop_ids:
        if curr in seen:
            raise UnsupportedEmission(
                f"generic matmul: operand chain cycle at {curr!r}"
            )
        seen.add(curr)
        node = id_to_node.get(curr)
        if node is None:
            raise UnsupportedEmission(
                f"generic matmul: operand {op_id!r} chain hits unknown node {curr!r}"
            )
        data_pred = _data_pred_input(node, scalar_ids)
        scal = _combiner_scalar_input(node, scalar_ids)
        steps.append(ChainStep(node.id, node.op, data_pred, scal))
        curr = data_pred
    steps.reverse()
    return steps, curr


def _build_operand_plan(
    op_id: str,
    id_to_node: dict[str, Node],
    input_node_ids: set[str],
    scalar_ids: set[str],
    stop_ids: set[str] | None = None,
) -> OperandPlan:
    steps, root = _build_chain_steps(
        op_id, id_to_node, input_node_ids, scalar_ids, stop_ids
    )
    tpos = next((idx for idx, s in enumerate(steps) if s.op == "nc_transpose"), None)
    if any(
        s.op == "nc_transpose" for s in steps[(tpos + 1 if tpos is not None else 0) :]
    ):
        raise UnsupportedEmission(
            f"generic matmul: operand {op_id!r} chain has more than one "
            f"nc_transpose — unsupported orientation"
        )
    is_mat = root in (stop_ids or set())
    return OperandPlan(
        op_id=op_id,
        steps=tuple(steps),
        root_id=root,
        transpose_index=tpos,
        is_materialized=is_mat,
    )


def _build_post_mm(
    output_id: str,
    mm_id: str,
    id_to_node: dict[str, Node],
    scalar_ids: set[str],
) -> tuple[ChainStep, ...]:
    """The post-matmul elementwise chain between the matmul and the output, in
    topo (mm→output) order. Empty when the output IS the matmul. Refuses an
    output-side ``nc_transpose`` (out of scope)."""
    if output_id == mm_id:
        return ()
    steps: list[ChainStep] = []
    curr = output_id
    seen: set[str] = set()
    while curr != mm_id:
        if curr in seen:
            raise UnsupportedEmission(
                f"generic matmul: post-mm chain cycle at {curr!r}"
            )
        seen.add(curr)
        node = id_to_node.get(curr)
        if node is None:
            raise UnsupportedEmission(
                f"generic matmul: post-mm output {output_id!r} chain hits "
                f"unknown node {curr!r}"
            )
        if node.op == "nc_transpose":
            raise UnsupportedEmission(
                f"generic matmul: post-mm chain to output {output_id!r} "
                f"contains an nc_transpose (output-transpose is out of scope)"
            )
        data_pred = _data_pred_input(node, scalar_ids)
        scal = _combiner_scalar_input(node, scalar_ids)
        steps.append(ChainStep(node.id, node.op, data_pred, scal))
        curr = data_pred
    steps.reverse()
    return tuple(steps)


def _scalar_producer_chain(
    scalar_id: str, id_to_node: dict[str, Node], reduce_ids: set[str]
) -> tuple[str, list[str]]:
    """Walk a combiner's scalar-operand id back to its reduce node through the
    post-reduce unary scalar chain. Returns ``(reduce_id, post_ids)`` with
    ``post_ids`` in reduce→consumer order."""
    post: list[str] = []
    curr = scalar_id
    seen: set[str] = set()
    while curr not in reduce_ids:
        if curr in seen:
            raise UnsupportedEmission(f"generic matmul: scalar chain cycle at {curr!r}")
        seen.add(curr)
        node = id_to_node.get(curr)
        if node is None or (
            node.op not in (_ELEMENTWISE_OPS | _PASSTHROUGH_OPS)
            and not _is_unary_constant_tensor_scalar(node)
        ):
            raise UnsupportedEmission(
                f"generic matmul: scalar operand {scalar_id!r} chain node "
                f"{curr!r} is not an elementwise or passthrough op on the "
                f"reduce scalar"
            )
        if (
            node.op in _ELEMENTWISE_OPS or _is_unary_constant_tensor_scalar(node)
        ) and len(node.inputs) != 1:
            raise UnsupportedEmission(
                f"generic matmul: scalar operand {scalar_id!r} chain node "
                f"{curr!r} is not unary (inputs={node.inputs})"
            )
        if not node.inputs:
            raise UnsupportedEmission(
                f"generic matmul: scalar operand {scalar_id!r} chain node "
                f"{curr!r} has no data input"
            )
        post.append(node.id)
        curr = node.inputs[0]
    post.reverse()
    return curr, post


def _build_reduces(
    operand_plans: list[OperandPlan],
    post_mm: tuple[ChainStep, ...],
    id_to_node: dict[str, Node],
    input_node_ids: set[str],
    scalar_ids: set[str],
    stop_ids: set[str] | None = None,
) -> list[ReducePlan]:
    """The reduce-preamble plans feeding combiners on any operand chain OR on the
    post-matmul chain, one per distinct reduce node (deterministic order)."""
    reduce_ids = {i for i in scalar_ids if id_to_node[i].op in _REDUCE_OPS}
    groups: dict[str, ReducePlan] = {}

    def _is_unit_dim(dim: Any) -> bool:
        return _dims_equal_exactly(dim, 1)

    def _reduce_input_plan(input_id: str) -> ReduceInputPlan:
        node = id_to_node[input_id]
        shape = tuple(node.attrs.get("shape", node.shape or ()))
        if shape and len(shape) != 2:
            raise UnsupportedEmission(
                f"generic matmul: reduce HBM input {input_id!r} must be rank 2, "
                f"got shape={shape!r}"
            )
        return ReduceInputPlan(
            input_id=input_id,
            part_broadcast=bool(shape and _is_unit_dim(shape[0])),
            free_broadcast=bool(shape and _is_unit_dim(shape[1])),
        )

    def _scan(steps) -> None:
        for step in steps:
            if step.scalar_id is None:
                continue
            rid, post = _scalar_producer_chain(step.scalar_id, id_to_node, reduce_ids)
            if rid in groups:
                continue
            rnode = id_to_node[rid]
            chain_nodes, hbm_ids = collect_reduce_input_chain(
                rnode.inputs[0], input_node_ids, id_to_node
            )
            source_ids: list[str] = []
            if stop_ids:
                chain_by_id = {node.id: node for node in chain_nodes}
                needed: set[str] = set()
                bounded_hbm: dict[str, None] = {}
                stack = [rnode.inputs[0]]
                while stack:
                    node_id = stack.pop()
                    if node_id in stop_ids:
                        if node_id not in source_ids:
                            source_ids.append(node_id)
                        continue
                    if node_id in input_node_ids:
                        bounded_hbm[node_id] = None
                        continue
                    node = chain_by_id.get(node_id)
                    if node is None or node_id in needed:
                        continue
                    needed.add(node_id)
                    stack.extend(reversed(node.inputs))
                chain_nodes = [node for node in chain_nodes if node.id in needed]
                hbm_ids = list(bounded_hbm)
            if rnode.op == "activation_reduce":
                auxiliary_ids = rnode.inputs[1:]
                unsupported = [
                    input_id
                    for input_id in auxiliary_ids
                    if input_id not in input_node_ids
                ]
                if unsupported:
                    raise UnsupportedEmission(
                        f"activation_reduce {rid}: computed bias or scale inputs "
                        f"are not supported by the generic matmul preamble "
                        f"({unsupported})"
                    )
                hbm_ids.extend(
                    input_id for input_id in auxiliary_ids if input_id not in hbm_ids
                )
            # Validate the free-axis reduce (belt-and-braces; layout inference
            # already refused axis=0, but the plan is the single authority).
            axis_val: int | None = None
            if rnode.op == "tensor_reduce":
                unsupported_attrs = set(rnode.attrs) - _TENSOR_REDUCE_ATTRS
                if unsupported_attrs:
                    raise UnsupportedEmission(
                        f"tensor_reduce {rid}: unsupported attributes "
                        f"{sorted(unsupported_attrs)}"
                    )
                axis_attr = rnode.attrs.get("axis", 1)
                try:
                    axis_val = _normalize_rank2_reduce_axis(axis_attr)
                except (TypeError, ValueError) as exc:
                    raise UnsupportedEmission(
                        f"tensor_reduce {rid}: invalid reduction axis "
                        f"{axis_attr!r}: {exc}"
                    ) from exc
                if axis_val != 1:
                    raise UnsupportedEmission(
                        f"tensor_reduce {rid}: only free-axis (axis=1) reductions "
                        f"supported; got axis={axis_val}"
                    )
                keepdims = rnode.attrs.get(
                    "keepdims",
                    rnode.attrs.get("keep_dims", False),
                )
                if keepdims is not True:
                    raise UnsupportedEmission(
                        f"tensor_reduce {rid}: generic matmul emission requires "
                        "keepdims=True"
                    )
                if rnode.attrs.get("negate", False) not in (None, False):
                    raise UnsupportedEmission(
                        f"tensor_reduce {rid}: generic matmul emission requires "
                        "negate=False"
                    )
            groups[rid] = ReducePlan(
                reduce_id=rid,
                input_id=rnode.inputs[0],
                axis=axis_val,
                chain_ids=tuple(n.id for n in chain_nodes),
                hbm_inputs=tuple(_reduce_input_plan(input_id) for input_id in hbm_ids),
                post_scalar_ids=tuple(post),
                source_ids=tuple(source_ids),
            )

    for op in operand_plans:
        _scan(op.steps)
    _scan(post_mm)
    return list(groups.values())


def _shared_full_tile_fanin_ids(
    compute_nodes: list[Node],
    id_to_node: dict[str, Node],
    input_node_ids: set[str],
    layouts: dict[str, Layout],
    scalar_ids: set[str],
) -> set[str]:
    """Binary full-tile values that an owned reduce preamble already computes."""
    reduce_chain_ids: set[str] = set()
    for node in topo_order(compute_nodes, id_to_node):
        if node.op not in _REDUCE_OPS or not node.inputs:
            continue
        chain, _hbm_ids = collect_reduce_input_chain(
            node.inputs[0],
            input_node_ids,
            id_to_node,
        )
        reduce_chain_ids.update(item.id for item in chain)

    boundaries: set[str] = set()
    for node_id in reduce_chain_ids:
        node = id_to_node[node_id]
        if (
            node.op != "tensor_tensor"
            or len(node.inputs) != 2
            or _combiner_scalar_input(node, scalar_ids) is not None
        ):
            continue
        output_layout = layouts[node_id]
        input_layouts = [layouts[input_id] for input_id in node.inputs]
        if output_layout.free is None or any(
            layout.free is None or not output_layout.same(layout)
            for layout in input_layouts
        ):
            raise UnsupportedEmission(
                f"generic matmul: shared full-tile fan-in {node_id!r} has "
                "incompatible operand layouts"
            )
        boundaries.add(node_id)
    return boundaries


def _single_matmul_staging(
    operand_plans: tuple[OperandPlan, ...],
    reduces: list[ReducePlan],
    nest: Nest,
    input_node_ids: set[str],
) -> tuple[frozenset[str], tuple[tuple[str, str], ...]]:
    """Select one maximal consumed endpoint from each shared operand prefix."""
    owners: dict[str, list[ReducePlan]] = {}
    for reduce in reduces:
        for node_id in reduce.chain_ids:
            owners.setdefault(node_id, []).append(reduce)

    selected: list[str] = []
    for operand in operand_plans:
        endpoint = operand.root_id if operand.root_id in owners else None
        if operand.transpose_index is not None:
            for step in operand.steps[: operand.transpose_index]:
                if step.node_id not in owners:
                    break
                endpoint = step.node_id

        if endpoint is None:
            if operand.is_materialized and operand.root_id not in input_node_ids:
                raise UnsupportedEmission(
                    f"generic matmul: materialized operand root "
                    f"{operand.root_id!r} has no reduce-preamble producer"
                )
            continue
        if operand.transpose_index is None:
            raise UnsupportedEmission(
                f"generic matmul: staged operand endpoint {endpoint!r} "
                "requires an nc_transpose consumer"
            )
        operand_free = nest.info.layouts[operand.op_id].free
        if operand_free is None or _dim_name(nest, operand_free) != "m":
            raise UnsupportedEmission(
                f"generic matmul: staged operand endpoint {endpoint!r} "
                "is not consumed on the m partition axis"
            )
        if endpoint not in selected:
            selected.append(endpoint)

    dtype_sources: list[tuple[str, str]] = []
    for node_id in selected:
        node_owners = owners.get(node_id, [])
        if len(node_owners) != 1:
            raise UnsupportedEmission(
                f"generic matmul: staged endpoint {node_id!r} must have exactly "
                f"one reduce-preamble producer, got {len(node_owners)}"
            )
        hbm_ids = node_owners[0].hbm_ids
        if not hbm_ids:
            raise UnsupportedEmission(
                f"generic matmul: staged endpoint {node_id!r} has no HBM dtype source"
            )
        dtype_sources.append((node_id, hbm_ids[0]))
    return frozenset(selected), tuple(dtype_sources)


def _assert_mm_operand_roles(
    mm_node: Node,
    stat_src_id: str,
    mov_src_id: str,
    id_to_node: dict[str, Node],
    input_node_ids: set[str],
) -> None:
    """Assert the stationary tile derives from ``inputs[0]`` and the moving from
    ``inputs[1]`` (``nc_matmul`` computes ``inputs[0]ᵀ @ inputs[1]``). Membership
    in the ancestor set allows legitimate multi-input pre-matmul chains; an
    operand swap lands outside the set and refuses at PLAN time."""
    stat_ok = _node_ancestors(mm_node.inputs[0], id_to_node, input_node_ids)
    mov_ok = _node_ancestors(mm_node.inputs[1], id_to_node, input_node_ids)
    if stat_src_id not in stat_ok or mov_src_id not in mov_ok:
        raise UnsupportedEmission(
            f"nc_matmul {mm_node.id}: emitted operands "
            f"(stationary<-{stat_src_id}, moving<-{mov_src_id}) disagree with "
            f"graph roles (stationary from inputs[0]<-{sorted(stat_ok)}, "
            f"moving from inputs[1]<-{sorted(mov_ok)}); refusing to emit a "
            f"matmul whose operand order does not match the synthesized node."
        )


def _staged_output_anchor(
    output_id: str,
    layouts: dict[str, Layout],
    mm_by_id: dict[str, MatmulPlan],
    nest: Nest,
    id_to_node: dict[str, Node],
) -> str:
    """The matmul whose layout the role collapse anchors on, when the graph's sink
    is not itself a matmul.

    The collapse binds ``(part, free, contraction)`` off the anchor, and only
    `nc_matmul` has a contraction to bind. So walk the sink's data ancestry to the
    nearest matmul: that is the matmul whose result the sink transforms, and an
    elementwise transform preserves the layout, so its axes are the sink's.

    The nearest ancestor is the right anchor, not "any matmul with a matching
    layout": two matmuls can share one layout (the QKV shape's projections do),
    and picking the wrong one binds the roles off an unrelated contraction."""
    sink_layout = layouts.get(output_id)
    if sink_layout is None or sink_layout.free is None:
        raise UnsupportedEmission(
            f"generic matmul: staged output {output_id!r} is not a matmul and has "
            "no full-tile layout to anchor the role collapse on"
        )
    seen: set[str] = set()
    frontier = [output_id]
    while frontier:
        node_id = frontier.pop(0)
        if node_id in seen:
            continue
        seen.add(node_id)
        node = id_to_node.get(node_id)
        if node is None:
            continue
        for input_id in node.inputs or ():
            if input_id in mm_by_id and input_id in nest.accum_loops:
                # The anchor must carry the sink's own axes, or the collapse binds
                # roles the emitted store would then index differently.
                if layouts[input_id].same(sink_layout):
                    return input_id
                continue
            frontier.append(input_id)
    raise UnsupportedEmission(
        f"generic matmul: staged output {output_id!r} is not a matmul and no "
        "matmul ancestor carries its layout, so nothing anchors the role collapse"
    )


def resolve_tile_roles(
    layouts: dict[str, Layout],
    nest: Nest,
    mm_plans: list[MatmulPlan],
    reduces: list[ReducePlan],
    stages: tuple[EmissionStagePlan, ...],
    output_id: str,
    id_to_node: dict[str, Node] | None = None,
) -> tuple[dict[str, str], dict[str, DimVar]]:
    """Collapse the nest's raw dims onto the four schedule roles ``m/n/k/p``.

    The emitted signature carries only ``TILES_IN_BLOCK_{M,N,K,P}``, but a
    reduce-bearing multi-matmul nest can hold more raw dims than that (the
    five-matmul QKV graph holds ``{m, n, k, p, k2}``). Assign every raw dim a
    role so each matmul's ``(output.part, output.free, contraction)`` gets a
    consistent triple, no raw dim takes two roles, dims sharing a role have
    equal extents, and every present raw dim is assigned. Refuse otherwise.

    Returns ``(dim_aliases, role_dims)``: the raw-dim to role map the renderer
    reads through ``alias_dim``, and one representative ``DimVar`` per role
    (the dims preamble emits its ``P = M`` aliasing lines from it)."""
    info = nest.info
    contractions = dict(info.contractions)
    raw_dims = {info.canon[dv.find()] for dv in info.canon}

    mm_by_id = {mp.mm_id: mp for mp in mm_plans}
    # The role collapse anchors on the LAST matmul, which is usually the output.
    # It is not when the graph normalizes after the output: the search's
    # transposed orientation applies the reciprocal to the second matmul's result,
    # so the sink is a `tensor_scalar` that shares the anchor's layout. Anchoring
    # on the last matmul rather than the sink is what admits that shape, and the
    # sink's own layout is unchanged by it (an elementwise op preserves layout).
    anchor_id = output_id
    if anchor_id not in mm_by_id:
        if id_to_node is None:
            raise UnsupportedEmission(
                f"generic matmul: staged output {output_id!r} is not a matmul; "
                "a staged plan must end in the matmul whose contraction binds a "
                "role"
            )
        anchor_id = _staged_output_anchor(
            output_id, layouts, mm_by_id, nest, id_to_node
        )
    output_id = anchor_id
    # A softmax-shaped graph has one reduce; a bare matmul→matmul chain (linear
    # attention `Q (Kᵀ V)`) has none — its 4th role is fixed by the earlier
    # matmul's contraction, not a reduce (see the reduce-source binding below).
    if len(reduces) > 1:
        raise UnsupportedEmission(
            "generic matmul: staged emission supports at most one reduction, "
            f"got {len(reduces)}"
        )

    aliases: dict[str, str] = {}
    role_dims: dict[str, DimVar] = {}

    def bind(dim: DimVar | None, role: str) -> None:
        if dim is None:
            raise UnsupportedEmission(
                "generic matmul: staged emission requires full-tile matmul layouts"
            )
        raw = info.canon[dim.find()]
        prior = aliases.get(raw)
        if prior is not None and prior != role:
            raise UnsupportedEmission(
                "generic matmul: staged emission requires an unsupported extra "
                f"schedule axis ({raw!r} is both {prior!r} and {role!r})"
            )
        aliases[raw] = role
        role_dim = role_dims.get(role)
        if role_dim is not None and not role_dim.same_extent(dim):
            raise UnsupportedEmission(
                "generic matmul: staged emission requires more than the fixed "
                f"m/n/k/p extents (role {role!r} has unequal extents)"
            )
        role_dims.setdefault(role, dim)

    # Four distinct raw dims already fill the namespace, so the collapse rule
    # below has nothing to do: running it would permute them, merging two raw
    # dims onto one role and leaving 'k' unassigned against a live tile arg.
    if raw_dims == {"m", "n", "k", "p"}:
        for dv, raw in info.canon.items():
            bind(dv, raw)
        return aliases, role_dims

    # The output matmul fixes three roles; the reduce's source matmul fixes the
    # fourth by sharing 'm' with it and putting its free axis on the output's
    # contraction.
    output_layout = layouts[output_id]
    bind(output_layout.part, "m")
    bind(output_layout.free, "n")
    bind(contractions[output_id], "p")

    reduce_source_id: str | None = None
    if reduces:
        reduce_sources = [
            source_id for source_id in reduces[0].source_ids if source_id in mm_by_id
        ]
        if len(reduce_sources) != 1:
            raise UnsupportedEmission(
                "generic matmul: staged reduction must read exactly one matmul, got "
                f"{reduce_sources}"
            )
        reduce_source_id = reduce_sources[0]
        source_layout = layouts[reduce_source_id]
        bind(source_layout.part, "m")
        bind(source_layout.free, "p")
        bind(contractions[reduce_source_id], "n")

    # Every earlier matmul contracts on 'k'. Its output axes usually already
    # carry roles; one that does not takes the role of the consumer axis it
    # serves. A matmul's stationary operand is laid out (part=contraction,
    # free=output.part) and its moving operand (part=contraction,
    # free=output.free), so a producer feeding one of those slots inherits those
    # two axes -- swapped when it reaches the slot through an nc_transpose.
    transpose_producer: dict[str, str] = {
        inter.node_id: inter.input_ids[0]
        for stage in stages
        for inter in stage.inters
        if inter.op == "nc_transpose" and len(inter.input_ids) == 1
    }
    axis_sources: dict[str, list[tuple[DimVar | None, DimVar | None]]] = {}
    for mp in mm_plans:
        consumer_layout = layouts[mp.mm_id]
        contraction = contractions[mp.mm_id]
        for operand, free_source in (
            (mp.stat, consumer_layout.part),
            (mp.mov, consumer_layout.free),
        ):
            axis_sources.setdefault(operand.op_id, []).append(
                (contraction, free_source)
            )
            producer_id = transpose_producer.get(operand.op_id)
            if producer_id is not None:
                axis_sources.setdefault(producer_id, []).append(
                    (free_source, contraction)
                )

    for mp in mm_plans:
        if mp.mm_id in (output_id, reduce_source_id):
            continue
        # An anchor may already have claimed this contraction; that binding is
        # authoritative, so do not overwrite it with 'k'.
        if info.canon[contractions[mp.mm_id].find()] not in aliases:
            bind(contractions[mp.mm_id], "k")
        mm_layout = layouts[mp.mm_id]
        for index, axis in enumerate((mm_layout.part, mm_layout.free)):
            if axis is None or info.canon[axis.find()] in aliases:
                continue
            # Every consumer this axis serves must agree on its role; two
            # consumers wanting different roles is an extra schedule axis.
            candidates = {
                aliases[info.canon[sources[index].find()]]
                for sources in axis_sources.get(mp.mm_id, [])
                if sources[index] is not None
                and info.canon[sources[index].find()] in aliases
            }
            if len(candidates) != 1:
                raise UnsupportedEmission(
                    "generic matmul: staged emission cannot place raw dim "
                    f"{info.canon[axis.find()]!r} of matmul {mp.mm_id!r} in the "
                    f"fixed m/n/k/p namespace (consumer roles {sorted(candidates)})"
                )
            bind(axis, candidates.pop())

    used_raw = {
        info.canon[dim.find()]
        for layout in layouts.values()
        for dim in (layout.part, layout.free)
        if dim is not None
    }
    used_raw.update(mp.accum_dim for mp in mm_plans)
    unmapped = sorted(used_raw - set(aliases))
    if unmapped:
        raise UnsupportedEmission(
            "generic matmul: staged emission has dimensions outside the fixed "
            f"m/n/k/p namespace: {unmapped}"
        )
    # A role no raw dim carries leaves its TILES_IN_BLOCK_<D> dead against the
    # spec's tile_args, so the collapse did not yield a usable namespace.
    unfilled = sorted({"m", "n", "k", "p"} - set(role_dims))
    if unfilled:
        raise UnsupportedEmission(
            "generic matmul: staged emission leaves schedule roles "
            f"{unfilled} unassigned; no raw dim carries them"
        )
    return aliases, role_dims


def _dim_name(nest: Nest, dv) -> str:
    return nest.info.canon[dv.find()]


def _ordered_dims(dims: set[str]) -> list[str]:
    order = ["m", "n", "k", "p"]
    out = [d for d in order if d in dims]
    out += sorted(d for d in dims if d not in order)
    return out


def _partition_accum_dims(nest: Nest) -> set[str]:
    """Canonical dims on any partition axis or as a matmul contraction dim
    (must tile at the partition base)."""
    part_dims: set[str] = set()
    for lay in nest.info.layouts.values():
        part_dims.add(_dim_name(nest, lay.part))
    for _mm_id, c in nest.info.contractions:
        part_dims.add(_dim_name(nest, c))
    return part_dims


def _mov_free_dim(nest: Nest, mm_id: str) -> str | None:
    """The canonical name of a matmul's moving operand free dim (the matmul
    output's free axis) — the dim a directly-loaded moving operand slices over."""
    mov_free = nest.info.layouts[mm_id].free
    return _dim_name(nest, mov_free) if mov_free is not None else None


# EVERY moving operand slices at `MATMUL_TILE_<D>`. It is gathered into a
# block-spanning `(TILE_<k>, TILES_IN_BLOCK_<k>, BLOCK_<free>)` buffer on every
# `_emit_operand_load` path, the transposed one included, so the width is always
# available. The earlier guard skipped an operand with `transpose_index` set and
# cost it 4x the PE instructions while staying correct, which no host check
# catches. Divisibility is asserted at trace time by `_emit_matmul_accumulate`.
def _compute_wide_dim(nest: Nest, matmul_plans: list[MatmulPlan]) -> str | None:
    """The moving operand's free dim that never appears on any partition axis
    (purely free, tiles at ``tile_n`` base). Dual-role dims are handled separately
    via ``tile_roles`` / ``mov_wide`` rather than being promoted here."""
    part_dims: set[str] = set()
    for lay in nest.info.layouts.values():
        part_dims.add(_dim_name(nest, lay.part))
    candidates: set[str] = set()
    for mp in matmul_plans:
        d = _mov_free_dim(nest, mp.mm_id)
        if d is not None:
            candidates.add(d)
    for d in _ordered_dims(candidates - part_dims):
        return d
    return None


def _compute_tile_roles(
    nest: Nest, matmul_plans: list[MatmulPlan]
) -> dict[str, tuple[str, ...]]:
    """Map each canonical dim to its tile roles across the schedule.

    ``"part"``: buffer partition axis or contraction dim (tile at ``TILE_<D>``).
    ``"wide"``: free dim of a directly-loaded moving operand (tile at
    ``MATMUL_TILE_<D>``). A dim with both roles gets both tiling granularities."""
    part_dims = _partition_accum_dims(nest)
    wide_dims: set[str] = set()
    for mp in matmul_plans:
        d = _mov_free_dim(nest, mp.mm_id)
        if d is not None:
            wide_dims.add(d)
    roles: dict[str, tuple[str, ...]] = {}
    for d in part_dims | wide_dims:
        r: list[str] = []
        if d in part_dims:
            r.append("part")
        if d in wide_dims:
            r.append("wide")
        roles[d] = tuple(r)
    return roles


def _mov_wide(nest: Nest, mm_id: str, part_dims: set[str]) -> bool:
    """True when the moving free dim is dual-role, i.e. also a partition axis, so
    moving slices use ``MATMUL_TILE_<D>`` rather than ``TILE_<D>``."""
    d = _mov_free_dim(nest, mm_id)
    return d is not None and d in part_dims


# --------------------------------------------------------------------------- #
# Per-dim TILE_<D> resolution. The plan is the SOLE owner of this step, so the
# tiling that runs is the tiling the sweep configured and the CSV reports.
# --------------------------------------------------------------------------- #
def present_dims(nest: Nest) -> set[str]:
    """Every canonical dim the nest touches (placements union accum loops) —
    exactly the dims the preamble mints ``TILE_<D>`` for."""
    dims: set[str] = set()
    for placement in nest.placement.values():
        dims.update(placement)
    dims.update(nest.accum_loops.values())
    return dims


def _dim_extents(
    nest: Nest, id_to_node: dict[str, Node], input_node_ids: set[str]
) -> dict[str, int]:
    """The concrete extent of each canonical dim, from each input's ``(part,
    free)`` layout paired with its ``(rows, cols)`` shape. Two inputs disagreeing
    on a unified dim is an inference bug, not a shape a tile could fit: refuse."""
    extents: dict[str, int] = {}
    for iid in sorted(input_node_ids):
        node = id_to_node.get(iid)
        lay = nest.info.layouts.get(iid)
        if node is None or lay is None:
            continue
        shape = tuple(node.attrs.get("shape") or node.shape or ())
        # The preamble unpacks every input as `ROW, COL = var.shape`, so a
        # non-rank-2 input names no extent (and refuses below if it was the only
        # source for a present dim).
        if len(shape) != 2 or lay.free is None:
            continue
        for dv, size in ((lay.part, shape[0]), (lay.free, shape[1])):
            dim = nest.info.canon[dv.find()]
            prev = extents.get(dim)
            if prev is not None and prev != int(size):
                raise UnsupportedEmission(
                    f"generic matmul: dim {dim!r} has disagreeing extents "
                    f"({prev} vs {int(size)}) across the inputs its axes unify; "
                    f"layout inference cannot name one extent for it."
                )
            extents[dim] = int(size)
    return extents


def _tile_base(dim: str, wide_dim: str | None, tile_config: dict[str, int]) -> int:
    """The configured base tile for ``dim``, selected by ROLE not name: ``tile_n``
    for the wide dim, ``tile_k`` for the contraction, ``tile_m`` for the rest."""
    if dim == wide_dim:
        key = "tile_n"
    elif dim == "k":
        key = "tile_k"
    else:
        key = "tile_m"
    return int(tile_config[key])


def _resolve_tiles(
    nest: Nest,
    wide_dim: str | None,
    tile_roles: dict[str, tuple[str, ...]],
    tile_config: dict[str, int],
    id_to_node: dict[str, Node],
    input_node_ids: set[str],
) -> dict[str, int]:
    """Resolve ``TILE_<D>`` for every present dim whose extent is derivable, or
    refuse: it is the configured base ``B`` when ``B`` is within its role cap
    (``PARTITION_FMAX`` on a partition axis or contraction, else ``MOVING_FMAX``)
    and divides the extent, and a typed refusal otherwise. Never a substituted
    value, so ``TILES_IN_BLOCK_<D>`` plus the base identifies the geometry."""
    extents = _dim_extents(nest, id_to_node, input_node_ids)

    tiles: dict[str, int] = {}
    for dim in _ordered_dims(present_dims(nest)):
        roles = tuple(sorted(tile_roles.get(dim, ())))
        extent = extents.get(dim)
        if extent is None:
            # A present dim with no derivable extent is a staged intermediate
            # that aliases onto an extent-equal peer (e.g. the score contraction
            # onto P): the renderer resolves its tile through `alias_dim`, so it
            # needs no tile of its own. A dim the renderer really needed but that
            # got no tile KeyErrors loudly at render rather than miscompiling.
            continue
        base = _tile_base(dim, wide_dim, tile_config)
        cap = PARTITION_FMAX if "part" in roles else MOVING_FMAX
        if base > cap:
            raise UnsupportedEmission(
                f"generic matmul: dim {dim!r} tile base {base} exceeds its role "
                f"cap {cap} (roles={roles or ('none',)}, extent={extent}); the "
                f"engine cannot run a tile that wide, so the plan refuses."
            )
        if extent < base and base % extent == 0:
            # A base wider than the extent still emits when it divides down cleanly
            # to the extent: clamp to one full tile. Guarded, so an indivisible
            # small extent (e.g. 264 vs 512) still refuses below.
            tiles[dim] = extent
            continue
        if extent % base != 0:
            raise UnsupportedEmission(
                f"generic matmul: dim {dim!r} extent {extent} is not divisible by "
                f"its tile base {base} (roles={roles or ('none',)}, cap={cap}); "
                f"the renderer must run the configured tiling, and no legal tile "
                f"count divides this extent at that base, so the plan refuses."
            )
        tiles[dim] = base
    return tiles


# --------------------------------------------------------------------------- #
# The single builder.
# --------------------------------------------------------------------------- #
def build_emission_plan(
    compute_nodes: list[Node],
    id_to_node: dict[str, Node],
    input_node_ids: set[str],
    output_id: str,
    tile_config: dict[str, int] | None = None,
) -> EmissionPlan:
    """Interpret the hardware graph ONCE into a complete EmissionPlan. Every
    compute node is covered by at least one role; an unclassified node refuses
    at PLAN time with a typed ``UnsupportedEmission`` naming it. Values shared
    between a reduce chain and an operand chain are the deliberate overlap and
    become staged values. This is the soundness keystone: one unambiguous
    interpretation, or a refusal.

    ``tile_config`` is the resolved matmul tile config (``tile_m`` / ``tile_k`` /
    ``tile_n``, as published on ``EmitCtx.tile_config``) the per-dim ``TILE_<D>``
    resolves from; it defaults to the hardware defaults for a caller inspecting
    pure graph structure."""
    cfg = dict(DEFAULT_MATMUL_TILE) if tile_config is None else dict(tile_config)
    nest = derive_nest(compute_nodes, id_to_node, input_node_ids, output_id)
    layouts = nest.info.layouts
    scalar_ids = scalar_node_ids(compute_nodes, id_to_node)
    _validate_broadcast_aliases(compute_nodes, id_to_node, layouts, scalar_ids)

    matmuls = [n for n in compute_nodes if n.op == "nc_matmul"]
    if not matmuls:
        raise UnsupportedEmission("generic matmul: no nc_matmul in graph")

    if len(matmuls) > 1:
        return _build_multi(
            compute_nodes,
            id_to_node,
            input_node_ids,
            output_id,
            nest,
            layouts,
            scalar_ids,
            matmuls,
            cfg,
        )

    # --- single matmul (class A-D) ------------------------------------------ #
    mm = matmuls[0]
    shared_fanin_ids = _shared_full_tile_fanin_ids(
        compute_nodes,
        id_to_node,
        input_node_ids,
        layouts,
        scalar_ids,
    )
    stat_op = _build_operand_plan(
        mm.inputs[0],
        id_to_node,
        input_node_ids,
        scalar_ids,
        shared_fanin_ids,
    )
    mov_op = _build_operand_plan(
        mm.inputs[1],
        id_to_node,
        input_node_ids,
        scalar_ids,
        shared_fanin_ids,
    )
    _assert_mm_operand_roles(
        mm, stat_op.root_id, mov_op.root_id, id_to_node, input_node_ids
    )
    accum_dim = nest.accum_loops[mm.id]
    mm_plan = MatmulPlan(
        mm_id=mm.id,
        stat=stat_op,
        mov=mov_op,
        accum_dim=accum_dim,
        is_chained=False,
        mov_wide=_mov_wide(nest, mm.id, _partition_accum_dims(nest)),
    )

    post_mm = _build_post_mm(output_id, mm.id, id_to_node, scalar_ids)
    reduces = _build_reduces(
        [stat_op, mov_op], post_mm, id_to_node, input_node_ids, scalar_ids
    )

    staged_ids, staged_dtype_sources = _single_matmul_staging(
        (stat_op, mov_op),
        reduces,
        nest,
        input_node_ids,
    )

    wide_dim = _compute_wide_dim(nest, [mm_plan])
    tile_roles = _compute_tile_roles(nest, [mm_plan])

    # --- full-coverage keystone -------------------------------------------- #
    accounted: set[str] = {mm.id}
    accounted |= set(stat_op.node_ids) | set(mov_op.node_ids)
    accounted |= {s.node_id for s in post_mm}
    for r in reduces:
        accounted |= set(r.all_ids)
    for n in compute_nodes:
        if n.id not in accounted:
            raise UnsupportedEmission(
                f"generic matmul: unexpected compute node {n.id!r} (op={n.op}) "
                f"— not on an operand chain, the post-mm chain, or a reduce "
                f"preamble; analysis cannot classify it, so the plan refuses."
            )

    # Resolved after the coverage keystone: a structural refusal is the more
    # informative diagnosis, so it wins over a tiling that does not fit.
    tiles = _resolve_tiles(nest, wide_dim, tile_roles, cfg, id_to_node, input_node_ids)

    return EmissionPlan(
        layouts=layouts,
        nest=nest,
        scalar_ids=frozenset(scalar_ids),
        output_id=output_id,
        wide_dim=wide_dim,
        tile_roles=tile_roles,
        tiles=tiles,
        matmuls=(mm_plan,),
        post_mm=post_mm,
        reduces=tuple(reduces),
        staged_ids=staged_ids,
        staged_dtype_sources=staged_dtype_sources,
    )


def _build_stage_plans(
    compute_nodes: list[Node],
    id_to_node: dict[str, Node],
    input_node_ids: set[str],
    nest: Nest,
    matmul_plans: list[MatmulPlan],
    reduces: list[ReducePlan],
    layouts: dict[str, Layout],
    scalar_ids: set[str],
) -> tuple[tuple[EmissionStagePlan, ...], frozenset[str]]:
    """Project a multi-matmul graph onto dependency-depth execution stages.

    Returns the stages plus the SBUF scalar crossings: the crossing values that
    are per-partition scalars and therefore never take an HBM buffer."""
    order = topo_order(compute_nodes, id_to_node)
    order_index = {node.id: index for index, node in enumerate(order)}
    mm_ids = {plan.mm_id for plan in matmul_plans}
    mm_depth: dict[str, int] = {}
    upstream_cache: dict[str, frozenset[str]] = {}

    def _upstream_mms(node_id: str) -> frozenset[str]:
        cached = upstream_cache.get(node_id)
        if cached is not None:
            return cached
        node = id_to_node[node_id]
        found: set[str] = set()
        for input_id in node.inputs:
            if input_id in mm_ids:
                found.add(input_id)
            elif input_id not in input_node_ids:
                found.update(_upstream_mms(input_id))
        result = frozenset(found)
        upstream_cache[node_id] = result
        return result

    for node in order:
        if node.id not in mm_ids:
            continue
        dependencies = _upstream_mms(node.id)
        mm_depth[node.id] = (
            1 + max(mm_depth[dependency] for dependency in dependencies)
            if dependencies
            else 0
        )

    consumers: dict[str, list[str]] = {}
    for node in order:
        for input_id in node.inputs:
            consumers.setdefault(input_id, []).append(node.id)

    downstream_cache: dict[str, frozenset[str]] = {}

    def _downstream_mms(node_id: str) -> frozenset[str]:
        cached = downstream_cache.get(node_id)
        if cached is not None:
            return cached
        found: set[str] = set()
        for consumer_id in consumers.get(node_id, []):
            if consumer_id in mm_ids:
                found.add(consumer_id)
            else:
                found.update(_downstream_mms(consumer_id))
        result = frozenset(found)
        downstream_cache[node_id] = result
        return result

    node_stage: dict[str, int] = {}
    for node in order:
        if node.id in mm_depth:
            node_stage[node.id] = mm_depth[node.id]
            continue
        upstream = _upstream_mms(node.id)
        if upstream:
            node_stage[node.id] = max(mm_depth[mm_id] for mm_id in upstream)
            continue
        downstream = _downstream_mms(node.id)
        if not downstream:
            raise UnsupportedEmission(
                f"generic matmul: compute node {node.id!r} is not connected "
                "to a matmul execution stage"
            )
        node_stage[node.id] = min(mm_depth[mm_id] for mm_id in downstream)

    stage_count = max(mm_depth.values()) + 1
    last_use: dict[str, int] = {}
    for node in order:
        consumer_stage = node_stage[node.id]
        for input_id in node.inputs:
            if input_id in input_node_ids:
                continue
            if node_stage[input_id] >= consumer_stage:
                continue
            last_use[input_id] = max(
                consumer_stage, last_use.get(input_id, consumer_stage)
            )

    crossing_ids = set(last_use)

    # A per-partition scalar crosses in SBUF, through the reducer's existing
    # `G.scalar_bufs` buffer, so it needs no HBM buffer and never gets a
    # `nest.materialized` lifetime. Defined here, and excluded at every renderer
    # HBM site, so the two cannot disagree about the set.
    sbuf_scalar_crossings = frozenset(
        node_id
        for node_id in crossing_ids
        if node_id in scalar_ids and layouts[node_id].free is None
    )

    missing_materialization = (
        crossing_ids - set(nest.materialized) - sbuf_scalar_crossings
    )
    if missing_materialization:
        raise UnsupportedEmission(
            "generic matmul: stage-crossing values have no nest.materialized "
            f"lifetime: {sorted(missing_materialization)}"
        )

    live_in: list[set[str]] = [set() for _ in range(stage_count)]
    live_out: list[set[str]] = [set() for _ in range(stage_count)]
    for value_id, final_stage in last_use.items():
        producer_stage = node_stage[value_id]
        for stage_index in range(producer_stage, final_stage):
            live_out[stage_index].add(value_id)
        for stage_index in range(producer_stage + 1, final_stage + 1):
            live_in[stage_index].add(value_id)

    reduce_by_node: dict[str, ReducePlan] = {}
    for reduce in reduces:
        for node_id in reduce.all_ids:
            reduce_by_node[node_id] = reduce

    stages: list[EmissionStagePlan] = []
    for stage_index in range(stage_count):
        stage_nodes = [node for node in order if node_stage[node.id] == stage_index]
        stage_reduces = tuple(
            reduce for reduce in reduces if node_stage[reduce.reduce_id] == stage_index
        )
        stage_inters = tuple(
            InterNodePlan(
                node_id=node.id,
                op=node.op,
                input_ids=tuple(node.inputs),
            )
            for node in stage_nodes
            if (
                node.id not in mm_ids
                and node.id not in reduce_by_node
                and node.op not in _REDUCE_OPS
                and _upstream_mms(node.id)
            )
        )
        stages.append(
            EmissionStagePlan(
                index=stage_index,
                node_ids=tuple(node.id for node in stage_nodes),
                matmul_ids=tuple(node.id for node in stage_nodes if node.id in mm_ids),
                inters=stage_inters,
                reduces=stage_reduces,
                live_in_ids=tuple(
                    sorted(live_in[stage_index], key=order_index.__getitem__)
                ),
                live_out_ids=tuple(
                    sorted(live_out[stage_index], key=order_index.__getitem__)
                ),
            )
        )
    return tuple(stages), sbuf_scalar_crossings


def _build_multi_stage(
    compute_nodes: list[Node],
    id_to_node: dict[str, Node],
    input_node_ids: set[str],
    output_id: str,
    nest: Nest,
    layouts: dict[str, Layout],
    scalar_ids: set[str],
    matmuls: list[Node],
    tile_config: dict[str, int],
) -> EmissionPlan:
    """Plan a general dependency-ordered multi-matmul graph with reductions."""
    ordered_matmuls = [
        node for node in topo_order(compute_nodes, id_to_node) if node.op == "nc_matmul"
    ]

    _from_matmul_cache: dict[str, bool] = {}

    def _from_matmul(node_id: str) -> bool:
        if node_id in input_node_ids:
            return False
        node = id_to_node.get(node_id)
        if node is None:
            return False
        if node.op == "nc_matmul":
            return True
        cached = _from_matmul_cache.get(node_id)
        if cached is not None:
            return cached
        result = any(_from_matmul(input_id) for input_id in node.inputs)
        _from_matmul_cache[node_id] = result
        return result

    # Boundary nodes (1) are materialized (i.e., must persist across a
    # consumer matmul's accum blocks), AND (2) come from a prior matmul
    # operation (i.e., shared HBM will be allocated for them in the prior
    # matmul, and read from in the subsequent matmul)
    boundary_ids = {
        node_id
        for node_id in nest.materialized
        if node_id in input_node_ids or _from_matmul(node_id)
    }
    part_dims = _partition_accum_dims(nest)
    mm_plans: list[MatmulPlan] = []
    operand_plans: list[OperandPlan] = []
    prior_mm_ids: set[str] = set()
    for mm in ordered_matmuls:
        stop_ids = boundary_ids | prior_mm_ids
        stat = _build_operand_plan(
            mm.inputs[0],
            id_to_node,
            input_node_ids,
            scalar_ids,
            stop_ids,
        )
        mov = _build_operand_plan(
            mm.inputs[1],
            id_to_node,
            input_node_ids,
            scalar_ids,
            stop_ids,
        )
        _assert_mm_operand_roles(
            mm, stat.root_id, mov.root_id, id_to_node, input_node_ids
        )
        operand_plans.extend((stat, mov))
        mm_plans.append(
            MatmulPlan(
                mm_id=mm.id,
                stat=stat,
                mov=mov,
                accum_dim=nest.accum_loops[mm.id],
                is_chained=stat.is_materialized,
                mov_wide=_mov_wide(nest, mm.id, part_dims),
            )
        )
        prior_mm_ids.add(mm.id)

    scalar_consumers = tuple(
        ChainStep(
            node_id=node.id,
            op=node.op,
            data_pred_id=_data_pred_input(node, scalar_ids),
            scalar_id=_combiner_scalar_input(node, scalar_ids),
        )
        for node in compute_nodes
        if _combiner_scalar_input(node, scalar_ids) is not None
    )
    reduces = _build_reduces(
        operand_plans,
        scalar_consumers,
        id_to_node,
        input_node_ids,
        scalar_ids,
        {plan.mm_id for plan in mm_plans} | boundary_ids,
    )
    graph_reduce_ids = [node.id for node in compute_nodes if node.op in _REDUCE_OPS]
    planned_reduce_ids = [reduce.reduce_id for reduce in reduces]
    invalid_reduce_ids = sorted(
        reduce_id
        for reduce_id in graph_reduce_ids
        if planned_reduce_ids.count(reduce_id) != 1
    )
    if invalid_reduce_ids:
        raise UnsupportedEmission(
            "generic matmul: every multi-stage reduction must belong to "
            "exactly one ReducePlan; unowned or multiply owned reductions: "
            f"{invalid_reduce_ids}"
        )

    reduce_chain_ids = {node_id for reduce in reduces for node_id in reduce.chain_ids}
    reduce_ids = {reduce.reduce_id for reduce in reduces}
    externally_consumed = {
        input_id
        for node in compute_nodes
        if node.id not in reduce_ids
        for input_id in node.inputs
    }
    staged_ids = frozenset(reduce_chain_ids & externally_consumed)

    for node in compute_nodes:
        if node.op in _REDUCE_OPS or node.op == "nc_matmul":
            continue
        if node.op not in _INTER_OPS:
            raise UnsupportedEmission(
                f"generic matmul: multi-stage node {node.id!r} "
                f"(op={node.op}) has no supported renderer role"
            )
        if node.op == "tensor_tensor" and len(node.inputs) == 2:
            full = [
                layouts[input_id]
                for input_id in node.inputs
                if layouts[input_id].free is not None
            ]
            if len(full) == 2 and not full[0].same(full[1]):
                raise UnsupportedEmission(
                    f"generic matmul: combine {node.id!r} has two full-tile "
                    "operands with disagreeing layouts"
                )

    stages, sbuf_scalar_crossings = _build_stage_plans(
        compute_nodes,
        id_to_node,
        input_node_ids,
        nest,
        mm_plans,
        reduces,
        layouts,
        scalar_ids,
    )
    dim_aliases, role_dims = resolve_tile_roles(
        layouts,
        nest,
        mm_plans,
        reduces,
        stages,
        output_id,
        id_to_node,
    )
    stage_inters = tuple(inter for stage in stages for inter in stage.inters)
    chained = id_to_node[output_id] if id_to_node[output_id].op == "nc_matmul" else None
    materialized_inter_id = None
    if chained is not None and chained.inputs[0] in boundary_ids:
        materialized_inter_id = chained.inputs[0]

    wide_dim = _compute_wide_dim(nest, mm_plans)
    tile_roles = _compute_tile_roles(nest, mm_plans)
    tiles = _resolve_tiles(
        nest, wide_dim, tile_roles, tile_config, id_to_node, input_node_ids
    )

    return EmissionPlan(
        layouts=layouts,
        nest=nest,
        scalar_ids=frozenset(scalar_ids),
        output_id=output_id,
        wide_dim=wide_dim,
        tile_roles=tile_roles,
        tiles=tiles,
        matmuls=tuple(mm_plans),
        reduces=tuple(reduces),
        staged_ids=staged_ids,
        is_multi=True,
        chained_mm_id=chained.id if chained is not None else None,
        inters=stage_inters,
        materialized_inter_id=materialized_inter_id,
        stages=stages,
        sbuf_scalar_crossings=sbuf_scalar_crossings,
        dim_aliases=dim_aliases,
        role_dims=role_dims,
    )


def _build_multi(
    compute_nodes: list[Node],
    id_to_node: dict[str, Node],
    input_node_ids: set[str],
    output_id: str,
    nest: Nest,
    layouts: dict[str, Layout],
    scalar_ids: set[str],
    matmuls: list[Node],
    tile_config: dict[str, int],
) -> EmissionPlan:
    """Plan a side-matmul (part_*_mlp) or chained-matmul (relu/silu_mlp) graph.

    A reduce anywhere in the graph routes to ``_build_multi_stage`` instead. So
    does a bare matmul→matmul chain (a matmul output that is directly an operand
    of another matmul, with no combine between them, as in linear attention
    ``Q (Kᵀ V)``): this graph's simpler chained path only materializes a
    *non-matmul* inter combine as the downstream matmul's stationary operand,
    whereas the stage path already stages a prior matmul's result as an operand
    via ``stop_ids`` (``boundary_ids | prior_mm_ids``). The inter-matmul nodes are
    every compute node not a matmul and not on a matmul operand load chain; each
    must be a known inter op (else refused). A chained output matmul reads exactly
    one materialized combine as its stationary operand."""
    mm_id_set = {m.id for m in matmuls}
    mm_feeds_mm = any(inp in mm_id_set for m in matmuls for inp in m.inputs)
    if mm_feeds_mm or any(node.op in _REDUCE_OPS for node in compute_nodes):
        return _build_multi_stage(
            compute_nodes,
            id_to_node,
            input_node_ids,
            output_id,
            nest,
            layouts,
            scalar_ids,
            matmuls,
            tile_config,
        )

    out_node = id_to_node[output_id]
    chained = out_node if out_node.op == "nc_matmul" else None
    earlier = [m for m in matmuls if m is not chained]

    # Earlier matmuls: plan both operands. Chained stationary is a materialized
    # inter node, planned below; only its moving operand is a raw-input chain.
    part_dims = _partition_accum_dims(nest)
    mm_plans: list[MatmulPlan] = []
    load_ids: set[str] = set()
    for mm in earlier:
        stat = _build_operand_plan(mm.inputs[0], id_to_node, input_node_ids, scalar_ids)
        mov = _build_operand_plan(mm.inputs[1], id_to_node, input_node_ids, scalar_ids)
        _assert_mm_operand_roles(
            mm, stat.root_id, mov.root_id, id_to_node, input_node_ids
        )
        load_ids |= set(stat.node_ids) | set(mov.node_ids)
        mm_plans.append(
            MatmulPlan(
                mm_id=mm.id,
                stat=stat,
                mov=mov,
                accum_dim=nest.accum_loops[mm.id],
                is_chained=False,
                mov_wide=_mov_wide(nest, mm.id, part_dims),
            )
        )

    if chained is not None:
        mov = _build_operand_plan(
            chained.inputs[1], id_to_node, input_node_ids, scalar_ids
        )
        # Stationary is the materialized combine (structural); only mov is ancestor-checked.
        mov_ok = _input_ancestors(chained.inputs[1], id_to_node, input_node_ids)
        if mov.root_id not in mov_ok:
            raise UnsupportedEmission(
                f"nc_matmul {chained.id}: moving operand source {mov.root_id!r} "
                f"not an ancestor of inputs[1] {sorted(mov_ok)}"
            )
        load_ids |= set(mov.node_ids)
        stat_mat = OperandPlan(
            op_id=chained.inputs[0],
            steps=(),
            root_id=chained.inputs[0],
            transpose_index=None,
            is_materialized=True,
        )
        mm_plans.append(
            MatmulPlan(
                mm_id=chained.id,
                stat=stat_mat,
                mov=mov,
                accum_dim=nest.accum_loops[chained.id],
                is_chained=True,
                mov_wide=_mov_wide(nest, chained.id, part_dims),
            )
        )

    mm_ids = {m.id for m in matmuls}
    inter_nodes = [
        n
        for n in topo_order(compute_nodes, id_to_node)
        if n.id not in mm_ids and n.id not in load_ids
    ]

    # Refuse unknown inter ops and layout-mismatched combines (missing nc_transpose).
    for n in inter_nodes:
        if n.op not in _INTER_OPS:
            raise UnsupportedEmission(
                f"generic matmul: inter-matmul node {n.id!r} (op={n.op}) is not "
                f"a known combine/elementwise/transpose op; analysis cannot "
                f"classify it, so the plan refuses."
            )
        if n.op == "tensor_tensor" and len(n.inputs) == 2:
            full = [layouts[i] for i in n.inputs if layouts[i].free is not None]
            if len(full) == 2 and not full[0].same(full[1]):
                raise UnsupportedEmission(
                    f"generic matmul: combine {n.id!r} has two full-tile "
                    f"operands with disagreeing layouts (missing nc_transpose)"
                )

    materialized = set(nest.materialized)
    mat_inter_id: str | None = None
    if chained is not None:
        # The graph edge inputs[0] is the sole authority for the materialized
        # combine id; nest.materialized validates, never re-selects it.
        mat_inter_id = chained.inputs[0]
        inter_id_set = {n.id for n in inter_nodes}
        if mat_inter_id not in inter_id_set:
            raise UnsupportedEmission(
                f"nc_matmul {chained.id}: chained stationary operand "
                f"{mat_inter_id!r} (graph edge inputs[0]) is not an inter-matmul "
                f"node ({sorted(inter_id_set)}); refusing to read a buffer the "
                f"graph does not name as the materialized combine."
            )
        if mat_inter_id not in materialized:
            raise UnsupportedEmission(
                f"nc_matmul {chained.id}: chained stationary operand "
                f"{mat_inter_id!r} (graph edge inputs[0]) has no nest.materialized "
                f"entry ({sorted(materialized)}); the combine it names is not "
                f"materialized across the contraction blocks, so the chained "
                f"matmul cannot read it as its stationary operand."
            )

    wide_dim = _compute_wide_dim(nest, mm_plans)
    tile_roles = _compute_tile_roles(nest, mm_plans)
    tiles = _resolve_tiles(
        nest, wide_dim, tile_roles, tile_config, id_to_node, input_node_ids
    )

    inters = tuple(
        InterNodePlan(node_id=n.id, op=n.op, input_ids=tuple(n.inputs or []))
        for n in inter_nodes
    )
    return EmissionPlan(
        layouts=layouts,
        nest=nest,
        scalar_ids=frozenset(scalar_ids),
        output_id=output_id,
        wide_dim=wide_dim,
        tile_roles=tile_roles,
        tiles=tiles,
        matmuls=tuple(mm_plans),
        is_multi=True,
        chained_mm_id=(chained.id if chained is not None else None),
        inters=inters,
        materialized_inter_id=mat_inter_id,
    )
