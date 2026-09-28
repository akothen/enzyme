"""The layout algebra for LNC=2 SPMD sharding (pure host code).

A sharding is a complete edge labeling of the high-level graph `G0`, determined
by a *coloring* (one active axis class per graph region). Cross-core combines are
derived at edges where producer and consumer labels disagree, never authored.

The generator enumerates *colorings*, not edge labels. A coloring fixes one
active class per monochromatic segment; PROPAGATE forces every edge label from
that single choice. Labels are computed, not searched, so the plan space is
duplicate-free by construction.

2-color limitation: when the graph is split into two segments with different
active classes, diamond-shaped subgraphs can create side-edges (non-boundary
edges from a segment-A producer to a segment-B consumer via a second path). These
side-edges require a mid-kernel combine (e.g. AllGather S(k)->R) that the
monolithic body emitter cannot splice today (it only has prelude/epilogue splice
points). Plans requiring unsupported internal combines are rejected at build time.
To lift this: split the body at the side-edge, emit a sendrecv+DMA between the
two halves, and continue. This is a future body-splitting pass, not an algebra
change.

No dependency on the synthesizer, the emitter, or the device; this is graph +
algebra, unit-testable on host.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from axon.ir import (
    Node,
    _node_operand_reduction_axes,
    _node_output_reduction_axes,
    nuGraph,
)
from axon.sharding_monoids import MONOIDS, MonoidId

# An axis class is identified by the smallest (node_id, dim) pair in its
# union-find component, so class identity is stable across runs.
ClassId = str
# An edge is the (producer_id, consumer_id) pair; producer feeds consumer.
EdgeId = tuple[str, str]


# --- Labels -----------------------------------------------------------------


@dataclass(frozen=True)
class Replicated:
    """`R`: every core holds a full identical copy."""


@dataclass(frozen=True)
class Sharded:
    """`S(a)`: axis class `a` is cut into two contiguous halves, one per core."""

    axis_class: ClassId


@dataclass(frozen=True)
class Partial:
    """`P(M)`: every core holds a partial under monoid `M`, unfinished until the
    monoid all-reduce."""

    monoid: MonoidId


Label = Replicated | Sharded | Partial


# --- Combines ---------------------------------------------------------------


@dataclass(frozen=True)
class Identity:
    """`σ -> σ` match. A barrier when the agreement crosses shared HBM (P)."""

    barrier: bool = False


@dataclass(frozen=True)
class Slice:
    """`R -> S(a)`: a local DMA offset, no cross-core traffic."""

    axis_class: ClassId


@dataclass(frozen=True)
class AllGather:
    """`S(a) -> R`: a pure `sendrecv` permutation, no arithmetic (X-gather)."""

    axis_class: ClassId


@dataclass(frozen=True)
class AllReduce:
    """`P(M) -> R`: a `sendrecv` plus the monoid merge."""

    monoid: MonoidId


@dataclass(frozen=True)
class AllToAll:
    """`S(a) -> S(b)`, a!=b: a `sendrecv` permutation. Non-canonical: `orientation`
    names which core keeps which slice (one combine per orientation)."""

    src_class: ClassId
    dst_class: ClassId
    orientation: int


@dataclass(frozen=True)
class ReduceScatter:
    """`P(M) -> S(b)`, b surviving: a `sendrecv` + monoid merge keeping one slice.
    Non-canonical, one combine per orientation."""

    monoid: MonoidId
    dst_class: ClassId
    orientation: int


Combine = Identity | Slice | AllGather | AllReduce | AllToAll | ReduceScatter


# --- Axis classes -----------------------------------------------------------


class _DisjointSet:
    """Union-find over `(node_id, dim)` pairs, keyed by the minimal member so a
    class id is stable and human-readable."""

    def __init__(self) -> None:
        self._parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(self, x: tuple[str, int]) -> tuple[str, int]:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: tuple[str, int], b: tuple[str, int]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Keep the lexicographically smaller root so ids are deterministic.
        lo, hi = (ra, rb) if ra <= rb else (rb, ra)
        self._parent[hi] = lo

    def components(self) -> dict[tuple[str, int], list[tuple[str, int]]]:
        comps: dict[tuple[str, int], list[tuple[str, int]]] = {}
        for x in self._parent:
            comps.setdefault(self.find(x), []).append(x)
        return comps


def same_index_pairs(
    node: Node, id_to_node: Mapping[str, Node]
) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    """The `(tensor, dim)` pairs this op forces to split together.

    matmul `(m,k)@(k,n)->(m,n)` ties `(lhs.k, rhs.k)`, `(lhs.m, out.m)`,
    `(rhs.n, out.n)`; the contracted `k` is tied to nothing downstream, so its
    class ends at the matmul output. Elementwise ties aligned (non-broadcast)
    operand+output dims. Transpose ties each input dim to its moved output slot.
    A reduced axis is tied to nothing downstream.
    """
    out_id = node.id
    out_shape = node.shape or ()
    pairs: list[tuple[tuple[str, int], tuple[str, int]]] = []

    if node.op in ("matmul", "nc_matmul"):
        lhs_id, rhs_id = node.inputs[0], node.inputs[1]
        lhs_m, lhs_k = (1, 0) if node.attrs.get("transpose_x", False) else (0, 1)
        pairs.append(((lhs_id, lhs_k), (rhs_id, 0)))  # lhs.k ~ rhs.k
        pairs.append(((lhs_id, lhs_m), (out_id, 0)))  # lhs.m ~ out.m
        pairs.append(((rhs_id, 1), (out_id, 1)))  # rhs.n ~ out.n
        return pairs

    if node.op in ("transpose", "nc_transpose", "dma_transpose"):
        src_id = node.inputs[0]
        src = id_to_node.get(src_id)
        src_shape = (src.shape or ()) if src is not None else ()
        if len(src_shape) == 2 and len(out_shape) == 2:
            pairs.append(((src_id, 0), (out_id, 1)))
            pairs.append(((src_id, 1), (out_id, 0)))
        else:
            for d in range(len(out_shape)):
                pairs.append(((src_id, d), (out_id, d)))
        return pairs

    reduced_out = _node_output_reduction_axes(node)
    if reduced_out or node.op in ("reduce_sum", "tensor_reduce", "softmax", "rms_norm"):
        src_id = node.inputs[0]
        src = id_to_node.get(src_id)
        src_shape = (src.shape or ()) if src is not None else ()
        reduced_in = _node_operand_reduction_axes(node, 0, tuple(src_shape))
        _tie_aligned(pairs, src_id, src_shape, out_id, out_shape, skip_in=reduced_in)
        return pairs

    for inp_id in node.inputs:
        src = id_to_node.get(inp_id)
        if src is None:
            continue
        _tie_aligned(pairs, inp_id, src.shape or (), out_id, out_shape)
    return pairs


def _tie_aligned(
    pairs: list[tuple[tuple[str, int], tuple[str, int]]],
    src_id: str,
    src_shape: tuple[int, ...],
    out_id: str,
    out_shape: tuple[int, ...],
    *,
    skip_in: frozenset[int] = frozenset(),
) -> None:
    """Tie right-aligned operand dims to output dims, skipping size-1
    (broadcast) and reduced axes."""
    sr, orank = len(src_shape), len(out_shape)
    for si in range(sr):
        if si in skip_in:
            continue
        oi = si + (orank - sr)
        if oi < 0 or oi >= orank:
            continue
        if src_shape[si] == 1 or out_shape[oi] == 1:
            continue
        pairs.append(((src_id, si), (out_id, oi)))


def axis_classes(G: nuGraph) -> dict[tuple[str, int], ClassId]:
    """Partition all `(node_id, dim)` pairs into shardable units: the connected
    components of "tied by some op" over `G.nodes`."""
    id_to_node = {n.id: n for n in G.nodes}
    uf = _DisjointSet()
    for n in G.nodes:
        for d in range(len(n.shape or ())):
            uf.find((n.id, d))
    for n in G.nodes:
        for a, b in same_index_pairs(n, id_to_node):
            uf.union(a, b)
    classes: dict[tuple[str, int], ClassId] = {}
    for root, members in uf.components().items():
        cid = f"{root[0]}#{root[1]}"
        for m in members:
            classes[m] = cid
    return classes


# --- Combine table ----------------------------------------------------------

LNC = 2


def _orientations() -> range:
    """The slice->core orientations a non-canonical reshard forks over."""
    return range(LNC)


def combine(sigma_p: Label, sigma_c: Label) -> frozenset[Combine]:
    """The set of combines bridging producer label `sigma_p` to consumer label
    `sigma_c`. A finite, total, static lookup."""
    match (sigma_p, sigma_c):
        case (Sharded(a), Sharded(b)) if a == b:
            return frozenset({Identity()})
        case (Partial(m), Partial(n)) if m == n:
            return frozenset({Identity()})
        case (Replicated(), Replicated()):
            return frozenset({Identity()})
        case (Replicated(), Sharded(a)):
            return frozenset({Slice(a)})
        case (Sharded(a), Replicated()):
            return frozenset({AllGather(a)})
        case (Sharded(a), Sharded(b)) if a != b:
            return frozenset(AllToAll(a, b, orientation=o) for o in _orientations())
        case (Partial(m), Replicated()):
            return frozenset({AllReduce(m)})
        case (Partial(m), Sharded(b)):
            return frozenset(
                ReduceScatter(m, b, orientation=o) for o in _orientations()
            )
        case _:
            return frozenset()


# --- ShardingPlan -----------------------------------------------------------

OUTPUT_SINK = "@output"


@dataclass(frozen=True)
class ShardingPlan:
    """The output of the sharding stage: a complete edge labeling with derived
    combines."""

    labels: dict[EdgeId, Label]
    combines: dict[EdgeId, Combine]
    shard_class: ClassId | None

    def tag(self) -> str:
        """A short tag for CSVs / filenames."""
        if self.shard_class is None and not any(
            isinstance(c, (AllReduce, AllGather)) for c in self.combines.values()
        ):
            return "lnc1"
        cls = self.shard_class or "?"
        terminal = any(consumer == OUTPUT_SINK for (_p, consumer) in self.combines)
        fin = "f" if terminal else ""
        monoids = sorted(
            {c.monoid for c in self.combines.values() if isinstance(c, AllReduce)}
        )
        if monoids:
            return f"shard={cls}:P{'+'.join(monoids)}{fin}"
        if any(isinstance(c, AllGather) for c in self.combines.values()):
            return f"shard={cls}:X{fin}"
        return f"shard={cls}:P{fin}"


def _inferred_sinks(nodes: list[Node]) -> tuple[str, ...]:
    consumed = {pid for n in nodes for pid in n.inputs}
    return tuple(n.id for n in nodes if n.id not in consumed)


# --- Coloring generator (replaces the old per-edge enumerator) ---------------


def _reduce_monoid(node: Node) -> MonoidId:
    """The monoid a reduce/norm/softmax op carries."""
    if node.op == "softmax":
        return "flash"
    if node.op == "rms_norm":
        return "var_naive"
    return node.attrs.get("reduce_monoid", "+")


def _node_has_class(
    node: Node, classes: dict[tuple[str, int], ClassId], cid: ClassId
) -> bool:
    """Whether any of the node's output dims belongs to class `cid`."""
    return any(classes.get((node.id, d)) == cid for d in range(len(node.shape or ())))


def _propagate(
    G: nuGraph,
    active: ClassId | None,
    classes: dict[tuple[str, int], ClassId],
    id_to_node: dict[str, Node],
    monoids: Mapping[MonoidId, object],
) -> dict[str, Label] | None:
    """PROPAGATE: given an active class, force every node's output label.

    Returns a dict mapping node_id -> output label, or None if the active class
    is inadmissible (a reduction over it has an unsupported monoid, or it has no
    shardable extent).
    """
    if active is None:
        return {n.id: Replicated() for n in G.nodes}

    out_label: dict[str, Label] = {}

    for node in G.nodes:
        if node.op == "input":
            # Inputs are always R: they physically reside as full copies in HBM.
            # A consumer that wants S(a) will see the R->S(a) mismatch and get a
            # Slice combine (the DMA offset operation).
            out_label[node.id] = Replicated()
            continue

        if node.op in ("matmul", "nc_matmul"):
            lhs_id, rhs_id = node.inputs[0], node.inputs[1]
            lhs_m, lhs_k = (1, 0) if node.attrs.get("transpose_x", False) else (0, 1)
            m_class = classes.get((lhs_id, lhs_m))
            k_class = classes.get((lhs_id, lhs_k))
            n_class = classes.get((rhs_id, 1))

            if active in (m_class, n_class):
                out_label[node.id] = Sharded(active)
            elif active == k_class:
                out_label[node.id] = Partial("+")
            else:
                out_label[node.id] = Replicated()
            continue

        if node.op in ("transpose", "nc_transpose", "dma_transpose"):
            if _node_has_class(node, classes, active):
                out_label[node.id] = Sharded(active)
            else:
                out_label[node.id] = Replicated()
            continue

        if node.op in ("reduce_sum", "tensor_reduce", "softmax", "rms_norm"):
            src_id = node.inputs[0]
            src_node = id_to_node.get(src_id)
            src_shape = (src_node.shape or ()) if src_node is not None else ()
            reduced_in = _node_operand_reduction_axes(node, 0, tuple(src_shape))
            reduced_classes = {
                classes.get((src_id, d)) for d in reduced_in if (src_id, d) in classes
            }
            if active in reduced_classes:
                monoid = _reduce_monoid(node)
                if monoid not in monoids:
                    return None
                out_label[node.id] = Partial(monoid)
            elif _node_has_class(node, classes, active):
                out_label[node.id] = Sharded(active)
            else:
                out_label[node.id] = Replicated()
            continue

        # Elementwise / broadcast
        if _node_has_class(node, classes, active):
            out_label[node.id] = Sharded(active)
        else:
            out_label[node.id] = Replicated()

    return out_label


def _assemble_plan(
    G: nuGraph,
    active: ClassId | None,
    out_label: dict[str, Label],
    classes: dict[tuple[str, int], ClassId],
    id_to_node: dict[str, Node],
) -> ShardingPlan:
    """ASSEMBLE: read labels and derive edge combines from mismatches."""
    nodes = list(G.nodes)
    labels: dict[EdgeId, Label] = {}
    combines: dict[EdgeId, Combine] = {}

    for node in nodes:
        for producer_id in node.inputs:
            edge: EdgeId = (producer_id, node.id)
            sigma_p = out_label[producer_id]
            labels[edge] = sigma_p

            # Determine what the consumer wants for this operand
            sigma_c = _consumer_wants(node, producer_id, active, classes, id_to_node)
            if sigma_p != sigma_c:
                cset = combine(sigma_p, sigma_c)
                if cset:
                    c = next(iter(sorted(cset, key=repr)))
                    combines[edge] = c

    output_ids = G.output_ids or _inferred_sinks(nodes)
    for oid in output_ids:
        sigma = out_label[oid]
        labels[(oid, OUTPUT_SINK)] = sigma
        tc = _terminal_combine_label(sigma)
        if tc is not None:
            combines[(oid, OUTPUT_SINK)] = tc

    return ShardingPlan(labels=labels, combines=combines, shard_class=active)


def _consumer_wants(
    node: Node,
    producer_id: str,
    active: ClassId | None,
    classes: dict[tuple[str, int], ClassId],
    id_to_node: dict[str, Node],
) -> Label:
    """What label a consumer node wants on a specific input edge."""
    if active is None:
        return Replicated()

    if node.op in ("matmul", "nc_matmul"):
        lhs_id, rhs_id = node.inputs[0], node.inputs[1]
        lhs_m, lhs_k = (1, 0) if node.attrs.get("transpose_x", False) else (0, 1)
        m_class = classes.get((lhs_id, lhs_m))
        k_class = classes.get((lhs_id, lhs_k))
        n_class = classes.get((rhs_id, 1))

        if producer_id == lhs_id:
            if active in (m_class, k_class):
                return Sharded(active)
            return Replicated()
        if producer_id == rhs_id:
            if active in (n_class, k_class):
                return Sharded(active)
            return Replicated()
        return Replicated()

    if node.op in ("reduce_sum", "tensor_reduce", "softmax", "rms_norm"):
        src_id = node.inputs[0]
        if producer_id == src_id:
            src_node = id_to_node.get(src_id)
            src_shape = (src_node.shape or ()) if src_node is not None else ()
            reduced_in = _node_operand_reduction_axes(node, 0, tuple(src_shape))
            reduced_classes = {
                classes.get((src_id, d)) for d in reduced_in if (src_id, d) in classes
            }
            if active in reduced_classes:
                return Sharded(active)
            producer_node = id_to_node.get(producer_id)
            if producer_node is not None and _node_has_class(
                producer_node, classes, active
            ):
                return Sharded(active)
            return Replicated()
        return Replicated()

    # Elementwise / transpose / other: wants S(active) if the producer carries it
    producer_node = id_to_node.get(producer_id)
    if producer_node is not None and _node_has_class(producer_node, classes, active):
        return Sharded(active)
    return Replicated()


def _terminal_combine_label(sigma: Label) -> Combine | None:
    """The combine that finishes the kernel boundary given the output label."""
    match sigma:
        case Partial(m):
            return AllReduce(m)
        case Sharded(_):
            return Identity(barrier=True)
        case _:
            return None


def _distinct_classes(
    G: nuGraph, classes: dict[tuple[str, int], ClassId]
) -> list[ClassId]:
    """All distinct axis classes in the graph, sorted for determinism."""
    seen: set[ClassId] = set()
    result: list[ClassId] = []
    for n in G.nodes:
        for d in range(len(n.shape or ())):
            cid = classes.get((n.id, d))
            if cid is not None and cid not in seen:
                seen.add(cid)
                result.append(cid)
    return sorted(result)


def _is_admissible(
    G: nuGraph,
    cid: ClassId,
    classes: dict[tuple[str, int], ClassId],
    id_to_node: dict[str, Node],
    monoids: Mapping[MonoidId, object],
) -> bool:
    """A class is admissible if PROPAGATE succeeds on it (monoid-legal)."""
    return _propagate(G, cid, classes, id_to_node, monoids) is not None


def _plan_canonical_key(plan: ShardingPlan) -> tuple:
    """A canonical key for dedup: the frozen set of (edge, combine) pairs plus
    the shard_class. Two plans with identical combines and shard_class are
    physically identical regardless of label-map differences."""
    return (
        plan.shard_class,
        frozenset(plan.combines.items()),
    )


def shardings(
    G: nuGraph,
    monoids: Mapping[MonoidId, object] = MONOIDS,
    *,
    max_colors: int = 2,
) -> Iterator[ShardingPlan]:
    """Enumerate every physically distinct sharding of G.

    The generator enumerates *colorings* (one active axis class per segment), not
    per-edge labels. For each coloring, PROPAGATE forces every label; ASSEMBLE
    reads combines off the combine table. The result is duplicate-free by
    construction for monochromatic (single-color) plans.

    `max_colors` caps the number of distinct colors in a coloring:
      - 1: monochromatic only (single-segment, K=0 in the doc)
      - 2: up to one reshard boundary (K=1)
      - etc.
    Default is 2. The replicate plan (active=None) is always included.
    """
    classes = axis_classes(G)
    id_to_node: dict[str, Node] = {n.id: n for n in G.nodes}
    all_classes = _distinct_classes(G, classes)

    seen: set[tuple] = set()

    def _emit(plan: ShardingPlan) -> Iterator[ShardingPlan]:
        if plan.shard_class is not None and not any(
            consumer == OUTPUT_SINK for (_p, consumer) in plan.combines
        ):
            return
        key = _plan_canonical_key(plan)
        if key not in seen:
            seen.add(key)
            yield plan

    # The replicate plan (active = None)
    rep_labels = _propagate(G, None, classes, id_to_node, monoids)
    if rep_labels is not None:
        plan = _assemble_plan(G, None, rep_labels, classes, id_to_node)
        yield from _emit(plan)

    # Monochromatic colorings: one active class for the whole graph
    admissible: list[ClassId] = []
    for cid in all_classes:
        labels = _propagate(G, cid, classes, id_to_node, monoids)
        if labels is None:
            continue
        admissible.append(cid)
        plan = _assemble_plan(G, cid, labels, classes, id_to_node)
        yield from _emit(plan)

    if max_colors < 2:
        return

    # 2-color colorings: enumerate pairs (a, b) where a != b, both admissible,
    # and a reshard boundary exists between them. A reshard boundary is an edge
    # where color changes from a to b. We place the boundary at every eligible
    # edge (one where the producer's label under color a can legally transition
    # to what color b wants at the consumer, and the transition is non-trivial).
    # Cache propagation results.
    prop_cache: dict[ClassId, dict[str, Label]] = {}
    for cid in admissible:
        labels = _propagate(G, cid, classes, id_to_node, monoids)
        if labels is not None:
            prop_cache[cid] = labels

    for a in admissible:
        labels_a = prop_cache.get(a)
        if labels_a is None:
            continue
        for b in admissible:
            if b == a:
                continue
            labels_b = prop_cache.get(b)
            if labels_b is None:
                continue
            for node in G.nodes:
                if node.op == "input":
                    continue
                for producer_id in node.inputs:
                    sigma_p_a = labels_a[producer_id]
                    sigma_c_b = _consumer_wants(
                        node, producer_id, b, classes, id_to_node
                    )
                    # Skip no-op boundaries (R->R or S(a)->S(a) at the boundary
                    # means no real color change, producing a ghost plan)
                    if sigma_p_a == sigma_c_b:
                        continue
                    cset = combine(sigma_p_a, sigma_c_b)
                    if not cset:
                        continue
                    plan = _build_two_color_plan(
                        G,
                        a,
                        b,
                        producer_id,
                        node.id,
                        labels_a,
                        labels_b,
                        classes,
                        id_to_node,
                        cset,
                    )
                    if plan is not None:
                        yield from _emit(plan)


def _build_two_color_plan(
    G: nuGraph,
    color_a: ClassId,
    color_b: ClassId,
    boundary_producer: str,
    boundary_consumer: str,
    labels_a: dict[str, Label],
    labels_b: dict[str, Label],
    classes: dict[tuple[str, int], ClassId],
    id_to_node: dict[str, Node],
    boundary_combines: frozenset[Combine],
) -> ShardingPlan | None:
    """Build a 2-color plan with one reshard boundary at a specific edge.

    Nodes that can reach boundary_consumer (inclusive) use color_b; all others
    use color_a. The boundary edge carries one of the reshard combines.
    """
    nodes = list(G.nodes)
    node_ids = [n.id for n in nodes]

    # Determine which nodes are in segment B (consumer and all its downstream).
    # Use forward reachability from boundary_consumer.
    successors: dict[str, set[str]] = {n.id: set() for n in nodes}
    for n in nodes:
        for inp in n.inputs:
            if inp in successors:
                successors[inp].add(n.id)

    in_b: set[str] = set()
    queue = [boundary_consumer]
    while queue:
        nid = queue.pop()
        if nid in in_b:
            continue
        in_b.add(nid)
        for succ in successors.get(nid, ()):
            queue.append(succ)

    # Build the merged label map: segment A nodes use labels_a, B nodes use labels_b.
    out_label: dict[str, Label] = {}
    for nid in node_ids:
        if nid in in_b:
            out_label[nid] = labels_b[nid]
        else:
            out_label[nid] = labels_a[nid]

    # Build edge labels and combines
    labels: dict[EdgeId, Label] = {}
    combines: dict[EdgeId, Combine] = {}

    for node in nodes:
        for producer_id in node.inputs:
            edge: EdgeId = (producer_id, node.id)
            sigma_p = out_label[producer_id]
            labels[edge] = sigma_p

            # Determine the boundary edge
            if producer_id == boundary_producer and node.id == boundary_consumer:
                # This is the reshard boundary: use the boundary combine
                for c in sorted(boundary_combines, key=repr):
                    combines[edge] = c
                    break
            else:
                # Normal intra-segment edge: compute what consumer wants
                if node.id in in_b:
                    sigma_c = _consumer_wants(
                        node, producer_id, color_b, classes, id_to_node
                    )
                else:
                    sigma_c = _consumer_wants(
                        node, producer_id, color_a, classes, id_to_node
                    )
                if sigma_p != sigma_c:
                    cset = combine(sigma_p, sigma_c)
                    if not cset:
                        return None
                    combines[edge] = next(iter(sorted(cset, key=repr)))

    # Reject plans with internal combines the emitter cannot lower. Today only
    # Slice (a local DMA offset) and AllReduce (a single sendrecv+merge) are
    # supported as non-terminal combines. An edge requiring e.g. AllGather or
    # AllToAll mid-kernel would need a body split the emitter doesn't do yet.
    for (_prod, cons), c in combines.items():
        if cons == OUTPUT_SINK:
            continue
        if not isinstance(c, (Slice, AllReduce)):
            return None

    # Reject plans where an input has Slice combines on multiple distinct classes
    # (physically impossible: one tensor can only be sliced one way).
    input_ids = (
        set(G.input_ids) if G.input_ids else {n.id for n in nodes if n.op == "input"}
    )
    input_slice_classes: dict[str, set[ClassId]] = {}
    for (prod, _cons), c in combines.items():
        if isinstance(c, Slice) and prod in input_ids:
            input_slice_classes.setdefault(prod, set()).add(c.axis_class)
    for _inp, cls_set in input_slice_classes.items():
        if len(cls_set) > 1:
            return None

    # Terminal combine at output
    output_ids = G.output_ids or _inferred_sinks(nodes)
    for oid in output_ids:
        sigma = out_label[oid]
        labels[(oid, OUTPUT_SINK)] = sigma
        tc = _terminal_combine_label(sigma)
        if tc is not None:
            combines[(oid, OUTPUT_SINK)] = tc

    # Determine the shard_class: the class active at the output.
    active_at_output = color_b if any(oid in in_b for oid in output_ids) else color_a
    shard_class: ClassId | None = active_at_output
    out_sigma = out_label[output_ids[0]] if output_ids else Replicated()
    if isinstance(out_sigma, Replicated):
        sliced = sorted(c.axis_class for c in combines.values() if isinstance(c, Slice))
        shard_class = sliced[0] if sliced else active_at_output

    return ShardingPlan(labels=labels, combines=combines, shard_class=shard_class)
