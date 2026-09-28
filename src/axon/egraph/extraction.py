"""Extract acyclic programs from a frozen ISA e-graph."""

from __future__ import annotations

import gc
import itertools
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from axon.egraph.adapter import (
    EClassRef,
    ENodeRef,
    Snapshot,
    compute_minimum_heights,
)
from axon.egraph.analysis import decode_isa
from axon.egraph.codec import DecodedIsaENode
from axon.egraph.proof import WallClockExceeded
from axon.egraph.proof_parallel import _disable_worker_gc, _fork_context
from axon.egraph.workers import resolve_worker_count
from axon.ir import (
    Node,
    annotate_shapes_concrete,
    nuGraph,
)

# A selection maps each reachable canonical e-class to its selected e-node.
Selection = dict[EClassRef, ENodeRef]


class ExtractionError(RuntimeError):
    """Raised when extraction cannot produce any selection for its roots."""


# Inhabited classes and cycle detection


def compute_inhabited_classes(
    snapshot: Snapshot,
) -> tuple[set[EClassRef], dict[EClassRef, int]]:
    """Return inhabited classes and their minimum finite heights."""
    heights = compute_minimum_heights(snapshot)
    return set(heights), heights


def selection_would_cycle(
    selection: Mapping[EClassRef, ENodeRef],
    new_class: EClassRef,
    new_enode: ENodeRef,
) -> bool:
    """Return true if the new selection creates a cycle."""
    stack = list(new_enode.child_classes())
    seen: set[EClassRef] = set()
    while stack:
        current = stack.pop()
        if current == new_class:
            return True
        if current in seen:
            continue
        seen.add(current)
        chosen = selection.get(current)
        if chosen is not None:
            stack.extend(chosen.child_classes())
    return False


# Generic joint selection enumeration


def iter_program_selections(
    snapshot: Snapshot,
    roots: list[EClassRef],
    inhabited: set[EClassRef],
) -> Iterator[Selection]:
    """Yield each consistent acyclic selection over the roots."""

    def candidates(cls: EClassRef) -> Iterator[ENodeRef]:
        # snapshot.members is already sorted by canonical row key.
        for enode in snapshot.members(cls):
            if all(child in inhabited for child in enode.child_classes()):
                yield enode

    def rec(frontier: list[EClassRef], selection: Selection) -> Iterator[Selection]:
        index = 0
        while index < len(frontier) and frontier[index] in selection:
            index += 1
        if index == len(frontier):
            yield dict(selection)
            return
        cls = frontier[index]
        rest = frontier[index + 1 :]
        for enode in candidates(cls):
            if selection_would_cycle(selection, cls, enode):
                continue
            selection[cls] = enode
            yield from rec(rest + list(enode.child_classes()), selection)
            del selection[cls]

    seen: set[EClassRef] = set()
    distinct_roots: list[EClassRef] = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            distinct_roots.append(root)
    yield from rec(distinct_roots, {})


def iter_isa_selections(
    snapshot: Snapshot,
    isa_output_roots: list[EClassRef],
    inhabited: set[EClassRef] | None = None,
) -> Iterator[Selection]:
    """Yield ISA selections for the ordered output roots."""
    if inhabited is None:
        inhabited, _heights = compute_inhabited_classes(snapshot)
    uninhabited = [root for root in isa_output_roots if root not in inhabited]
    if uninhabited:
        raise ExtractionError(
            f"declared ISA output classes are uninhabited: {uninhabited!r}"
        )
    yield from iter_program_selections(
        snapshot,
        list(isa_output_roots),
        inhabited,
    )


# Materialization


def materialize_isa_graph(
    snapshot: Snapshot,
    selection: Mapping[EClassRef, ENodeRef],
    isa_output_roots: list[EClassRef],
    input_metadata: Mapping[str, Mapping[str, Any]],
    decode_isa_enode: Callable[[Snapshot, ENodeRef], DecodedIsaENode] = decode_isa,
    declared_input_ids: tuple[str, ...] | None = None,
) -> nuGraph:
    """Materialize one selection as a concrete ISA ``nuGraph``."""
    node_by_class: dict[EClassRef, Node] = {}
    nodes: list[Node] = []
    counter = 0

    def build(cls: EClassRef) -> Node:
        nonlocal counter
        existing = node_by_class.get(cls)
        if existing is not None:
            return existing
        enode = selection[cls]
        decoded = decode_isa_enode(snapshot, enode)
        child_nodes = [build(child) for child in decoded.child_classes]
        if decoded.op == "input":
            assert decoded.source_id is not None
            meta = input_metadata.get(decoded.source_id, {})
            shape = meta.get("shape") or decoded.input_shape or ()
            attrs: dict[str, Any] = {"shape": tuple(shape)}
            sym_shape = meta.get("sym_shape")
            if sym_shape is not None:
                attrs["sym_shape"] = tuple(sym_shape)
            node = Node(id=decoded.source_id, op="input", inputs=[], attrs=attrs)
        else:
            node = Node(
                id=f"{decoded.op}_{counter}",
                op=decoded.op,
                inputs=[child.id for child in child_nodes],
                attrs=dict(decoded.attrs),
            )
            counter += 1
        node_by_class[cls] = node
        nodes.append(node)
        return node

    for root in isa_output_roots:
        build(root)

    output_ids = tuple(node_by_class[root].id for root in isa_output_roots)
    reachable_input_ids = tuple(node.id for node in nodes if node.op == "input")
    if declared_input_ids is None:
        input_ids = reachable_input_ids
    else:
        reachable = set(reachable_input_ids)
        input_ids = tuple(
            node_id for node_id in declared_input_ids if node_id in reachable
        )
    return nuGraph(nodes=nodes, output_ids=output_ids, input_ids=input_ids)


# Validation


def validate_materialized_graph(
    G: nuGraph,
) -> str | None:
    """Return the first validation error, or ``None`` for a valid ISA graph.
    Every materialized node comes from a controlled ISA insertion path."""
    try:
        annotate_shapes_concrete(G)
    except Exception as exc:
        return f"shape annotation failed: {exc}"

    return None


# Streaming production contract


@dataclass(frozen=True)
class _ForkExtraction:
    """Per-stream state fork workers inherit through process memory."""

    snapshot: Snapshot
    isa_output_roots: list[EClassRef]
    input_metadata: Mapping[str, Mapping[str, Any]]
    decode_isa_enode: Callable[[Snapshot, ENodeRef], DecodedIsaENode]
    declared_input_ids: tuple[str, ...] | None
    class_order: tuple[EClassRef, ...]


# Registered before each stream's pool is created, so forked children see it.
_FORK_EXTRACTIONS: dict[int, _ForkExtraction] = {}
_FORK_TOKENS = itertools.count()

# Selections carry unpicklable egglog values, so they cross the process
# boundary as (class position, member position) index pairs instead.
_EncodedSelection = tuple[tuple[int, int], ...]


def _materialize_fork_selection(
    token: int, encoded: _EncodedSelection
) -> nuGraph | None:
    """Materialize and validate one index-encoded selection in a fork worker."""
    state = _FORK_EXTRACTIONS[token]
    selection = {
        state.class_order[class_pos]: state.snapshot.classes[
            state.class_order[class_pos]
        ][member_pos]
        for class_pos, member_pos in encoded
    }
    graph = materialize_isa_graph(
        state.snapshot,
        selection,
        state.isa_output_roots,
        state.input_metadata,
        state.decode_isa_enode,
        declared_input_ids=state.declared_input_ids,
    )
    return graph if validate_materialized_graph(graph) is None else None


def _iter_materialized_parallel(
    snapshot: Snapshot,
    isa_output_roots: list[EClassRef],
    input_metadata: Mapping[str, Mapping[str, Any]],
    decode_isa_enode: Callable[[Snapshot, ENodeRef], DecodedIsaENode],
    deadline: float | None,
    declared_input_ids: tuple[str, ...] | None,
    workers: int,
) -> Iterator[nuGraph]:
    """Farm materialize+validate to fork workers, yielding in selection order."""
    context = _fork_context()
    assert context is not None
    class_order = tuple(snapshot.classes)
    class_pos = {cls: pos for pos, cls in enumerate(class_order)}
    member_pos: dict[EClassRef, dict[int, int]] = {}

    def encode(selection: Mapping[EClassRef, ENodeRef]) -> _EncodedSelection:
        pairs: list[tuple[int, int]] = []
        for cls, enode in selection.items():
            positions = member_pos.get(cls)
            if positions is None:
                positions = {
                    id(row): pos for pos, row in enumerate(snapshot.classes[cls])
                }
                member_pos[cls] = positions
            pairs.append((class_pos[cls], positions[id(enode)]))
        return tuple(pairs)

    selections = iter_isa_selections(snapshot, isa_output_roots)
    window = 2 * workers
    token = next(_FORK_TOKENS)
    _FORK_EXTRACTIONS[token] = _ForkExtraction(
        snapshot,
        isa_output_roots,
        input_metadata,
        decode_isa_enode,
        declared_input_ids,
        class_order,
    )
    # Match proof_parallel: the pool's manager thread must not gc-finalize
    # egglog objects created on the main thread while the pool is alive.
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    pool: ProcessPoolExecutor | None = None
    deadline_hit = False
    try:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_disable_worker_gc,
        )
        pending: dict[Future[nuGraph | None], int] = {}
        buffered: dict[int, tuple[nuGraph | None, BaseException | None]] = {}
        next_submit = 0
        next_yield = 0
        exhausted = False
        while True:
            # Keep the bounded in-flight window full while selections remain.
            while (
                not exhausted
                and not deadline_hit
                and len(pending) + len(buffered) < window
            ):
                try:
                    selection = next(selections)
                except StopIteration:
                    exhausted = True
                    break
                # Same per-selection deadline check order as the serial path.
                if deadline is not None and time.monotonic() >= deadline:
                    deadline_hit = True
                    break
                future = pool.submit(
                    _materialize_fork_selection, token, encode(selection)
                )
                pending[future] = next_submit
                next_submit += 1
            # Drain the contiguous completed prefix in enumeration order.
            while next_yield in buffered:
                graph, error = buffered.pop(next_yield)
                next_yield += 1
                if error is not None:
                    raise error
                if graph is not None:
                    yield graph
            if deadline_hit:
                raise WallClockExceeded("extraction")
            if not pending:
                if exhausted:
                    return
                continue
            # Selections already in flight passed their deadline check, so a
            # drained enumerator waits without a timeout, like the serial path.
            wait_timeout: float | None = None
            if deadline is not None and not exhausted:
                wait_timeout = max(0.0, deadline - time.monotonic())
            done, _ = wait(pending, timeout=wait_timeout, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                try:
                    buffered[index] = (future.result(), None)
                except BaseException as exc:  # re-raised at its stream position
                    buffered[index] = (None, exc)
            if not done and not exhausted:
                deadline_hit = True
    finally:
        del _FORK_EXTRACTIONS[token]
        if pool is not None:
            # Materializations run in milliseconds, so let in-flight tasks
            # finish; terminating workers can hang the executor manager.
            pool.shutdown(wait=True, cancel_futures=True)
        if gc_was_enabled:
            gc.enable()


def iter_materialized_isa_graphs(
    snapshot: Snapshot,
    isa_output_roots: list[EClassRef],
    input_metadata: Mapping[str, Mapping[str, Any]],
    decode_isa_enode: Callable[[Snapshot, ENodeRef], DecodedIsaENode] = decode_isa,
    deadline: float | None = None,
    declared_input_ids: tuple[str, ...] | None = None,
    workers: int | None = 1,
) -> Iterator[nuGraph]:
    """Yield one valid materialized ISA graph at a time, in enumeration order.
    A lapsed wall-clock ``deadline`` ends the stream between graphs."""
    resolved_workers = resolve_worker_count(workers)
    if resolved_workers > 1 and _fork_context() is not None:
        yield from _iter_materialized_parallel(
            snapshot,
            isa_output_roots,
            input_metadata,
            decode_isa_enode,
            deadline,
            declared_input_ids,
            resolved_workers,
        )
        return
    for selection in iter_isa_selections(
        snapshot,
        isa_output_roots,
    ):
        if deadline is not None and time.monotonic() >= deadline:
            raise WallClockExceeded("extraction")
        graph = materialize_isa_graph(
            snapshot,
            selection,
            isa_output_roots,
            input_metadata,
            decode_isa_enode,
            declared_input_ids=declared_input_ids,
        )
        if validate_materialized_graph(graph) is None:
            yield graph


__all__ = [
    "ExtractionError",
    "Selection",
    "compute_inhabited_classes",
    "iter_isa_selections",
    "iter_materialized_isa_graphs",
    "iter_program_selections",
    "materialize_isa_graph",
    "selection_would_cycle",
    "validate_materialized_graph",
]
