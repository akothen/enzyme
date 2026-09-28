"""Propagate proved operators over tensor or ISA e-graphs."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Any

from axon.egraph.adapter import (
    EClassRef,
    EGraphAdapter,
    EGraphAdapterError,
    ENodeRef,
    Snapshot,
)
from axon.egraph.context import SemanticContext
from axon.egraph.proof import (
    ProofStore,
    Term,
    TermApp,
    TermRef,
    WallClockExceeded,
    prove_candidate_batch,
)
from axon.egraph.workers import resolve_worker_count
from axon.isa_semantics import EquivalenceVerdict

# ``decode(snapshot, enode)`` returns a decoded e-node exposing ``op``,
# ``child_classes``, and ``attrs``.
DecodeFn = Callable[[Snapshot, ENodeRef], Any]

# Encode one operation and return its expression and result rank.
EncodeFn = Callable[[str, dict[str, Any], list[Any], list[int]], tuple[Any, int]]

# ``is_eligible_op(op)`` is True for a non-input operation with registered
# semantics at this level.
EligibleFn = Callable[[str], bool]

# A reject-only candidate filter. Returning True skips the proof obligation;
# returning False never admits the candidate without the normal proof requirement.
CandidatePrefilterFn = Callable[[Snapshot, dict[EClassRef, Any], Term, Term], bool]
_ADMISSION_CHUNK_SIZE = 8
_PROOF_CHUNK_SIZE = 64


@dataclass(frozen=True)
class Occurrence:
    """One producer, consumer, and operand position eligible for propagation."""

    snapshot: Snapshot
    analyses: dict[EClassRef, Any]
    consumer_class: EClassRef
    q: int
    producer_op: str
    producer_attrs: dict[str, Any]
    producer_children: tuple[EClassRef, ...]
    consumer_op: str
    consumer_attrs: dict[str, Any]
    consumer_children: tuple[EClassRef, ...]


@dataclass(frozen=True)
class Candidate:
    """A temporary local candidate: the current term and one swapped term."""

    degree: int
    current: Term
    new: Term


@dataclass(frozen=True)
class RoundResult:
    """Separate change counts from one propagation round."""

    enodes_added: int
    equalities_added: int
    occurrences: int
    candidates: int = 0
    represented: int = 0
    truncated_reason: str | None = None
    truncated_stage: str | None = None
    complete: bool = True
    queued_continuations: int = 0


@dataclass
class PropagationWorklist:
    """Stable occurrence identities whose required search completed."""

    processed: set[str] = field(default_factory=set)

    def select(
        self, context: SemanticContext, occurrences: list[Occurrence]
    ) -> list[Occurrence]:
        selected: list[Occurrence] = []
        for occurrence in occurrences:
            identity = occurrence_identity(context, occurrence)
            if identity in self.processed:
                continue
            selected.append(occurrence)
        return selected

    def complete(
        self,
        context: SemanticContext,
        occurrence: Occurrence,
    ) -> None:
        self.processed.add(occurrence_identity(context, occurrence))


def occurrence_identity(
    context: SemanticContext,
    occurrence: Occurrence,
) -> str:
    return context.occurrence_identity(
        consumer_class=occurrence.consumer_class,
        q=occurrence.q,
        producer_op=occurrence.producer_op,
        producer_attrs=occurrence.producer_attrs,
        producer_children=occurrence.producer_children,
        consumer_op=occurrence.consumer_op,
        consumer_attrs=occurrence.consumer_attrs,
        consumer_children=occurrence.consumer_children,
    )


def _decode(snapshot: Snapshot, enode: ENodeRef, decode: DecodeFn) -> Any | None:
    try:
        return decode(snapshot, enode)
    except Exception:
        return None


def eligible_occurrences(
    snapshot: Snapshot,
    analyses: dict[EClassRef, Any],
    decode: DecodeFn,
    is_eligible_op: EligibleFn,
    *,
    guard_shared_producers: bool = True,
) -> list[Occurrence]:
    """Return each eligible producer, consumer, and operand position.

    When ``guard_shared_producers`` is set, two structural guards (from the
    non-egraph Axon variant's ``_swap_with_successor_variants``) stop propagation
    from expanding shared subexpressions: a producer is eligible only when it
    feeds the consumer at exactly one operand slot and has exactly one successor
    e-node. The guards are on by default; pass ``False`` to disable them."""
    # successor count: distinct e-nodes referencing each e-class as a child.
    successors: dict[EClassRef, int] = {}
    if guard_shared_producers:
        for cls in snapshot.classes:
            for enode in snapshot.members(cls):
                decoded = _decode(snapshot, enode, decode)
                if decoded is None:
                    continue
                for child in set(decoded.child_classes):
                    successors[child] = successors.get(child, 0) + 1

    occurrences: list[Occurrence] = []
    for consumer_class in snapshot.classes:
        for consumer in snapshot.members(consumer_class):
            c_decoded = _decode(snapshot, consumer, decode)
            if c_decoded is None or not is_eligible_op(c_decoded.op):
                continue
            child_classes = c_decoded.child_classes
            for q, child_class in enumerate(child_classes):
                if guard_shared_producers:
                    # guard 1: producer consumed at exactly one operand slot.
                    if child_classes.count(child_class) != 1:
                        continue
                    # guard 2: producer has exactly one successor e-node.
                    if successors.get(child_class, 0) != 1:
                        continue
                for producer in snapshot.members(child_class):
                    p_decoded = _decode(snapshot, producer, decode)
                    if p_decoded is None or not is_eligible_op(p_decoded.op):
                        continue
                    occurrences.append(
                        Occurrence(
                            snapshot=snapshot,
                            analyses=analyses,
                            consumer_class=consumer_class,
                            q=q,
                            producer_op=p_decoded.op,
                            producer_attrs=dict(p_decoded.attrs),
                            producer_children=tuple(p_decoded.child_classes),
                            consumer_op=c_decoded.op,
                            consumer_attrs=dict(c_decoded.attrs),
                            consumer_children=tuple(child_classes),
                        )
                    )
    return occurrences


def _term_refs(term: Term) -> set[EClassRef]:
    if isinstance(term, TermRef):
        return {term.eclass}
    refs: set[EClassRef] = set()
    for child in term.children:
        refs |= _term_refs(child)
    return refs


def _op2_clone(occurrence: Occurrence, replacement: Term) -> TermApp:
    """One ``op2`` application with only its ``q``th child replaced."""
    children: list[Term] = []
    for j, b_class in enumerate(occurrence.consumer_children):
        if j == occurrence.q:
            children.append(replacement)
        else:
            children.append(TermRef(b_class))
    return TermApp.make(
        occurrence.consumer_op, occurrence.consumer_attrs, tuple(children)
    )


def swap_with_successor(
    occurrence: Occurrence, subset_indices: frozenset[int]
) -> list[Candidate]:
    """Build distinct swaps for one subset of producer operands."""
    return list(_iter_swap_with_successor(occurrence, subset_indices))


def _iter_swap_with_successor(
    occurrence: Occurrence,
    subset_indices: frozenset[int],
) -> Iterator[Candidate]:
    n = len(occurrence.producer_children)
    a_classes = occurrence.producer_children

    producer_app = TermApp.make(
        occurrence.producer_op,
        occurrence.producer_attrs,
        tuple(TermRef(a) for a in a_classes),
    )
    t_cur = _op2_clone(occurrence, producer_app)

    selected = sorted(subset_indices)
    # u[i] is the operand contributed to op1 at original index i.
    u: list[Term] = []
    for i in range(n):
        if i in subset_indices:
            u.append(_op2_clone(occurrence, TermRef(a_classes[i])))
        else:
            u.append(TermRef(a_classes[i]))

    degree = len(selected)
    seen: set[Term] = set()
    for perm in permutations(selected):
        mapping = {selected[k]: perm[k] for k in range(degree)}
        children = tuple(u[mapping.get(j, j)] for j in range(n))
        t_new = TermApp.make(
            occurrence.producer_op, occurrence.producer_attrs, children
        )
        if occurrence.consumer_class in _term_refs(t_new):
            continue
        if t_new in seen:
            continue
        seen.add(t_new)
        yield Candidate(degree=degree, current=t_cur, new=t_new)


def iter_occurrence_candidates(
    occurrence: Occurrence,
) -> Iterator[Candidate]:
    """Yield distinct candidates by ascending clone degree."""
    n = len(occurrence.producer_children)
    emitted: set[Term] = set()
    for k in range(1, n + 1):
        for subset in combinations(range(n), k):
            for candidate in _iter_swap_with_successor(
                occurrence,
                frozenset(subset),
            ):
                if candidate.new in emitted:
                    continue
                emitted.add(candidate.new)
                yield candidate


def _candidate_degree_groups(
    occurrence: Occurrence,
) -> Iterator[list[Candidate]]:
    group: list[Candidate] = []
    degree: int | None = None
    for candidate in iter_occurrence_candidates(occurrence):
        if degree is not None and candidate.degree != degree:
            yield group
            group = []
        degree = candidate.degree
        group.append(candidate)
    if group:
        yield group


@dataclass(frozen=True)
class _ProvedCandidate:
    sequence: tuple[int, int]
    occurrence: Occurrence
    candidate: Candidate


class AdmissionBatch:
    """Intern candidate terms once, then commit unions in sequence order."""

    def __init__(
        self,
        adapter: EGraphAdapter,
        context: SemanticContext,
        encode: EncodeFn,
    ) -> None:
        self.adapter = adapter
        self.context = context
        self.encode = encode
        self.term_cache: dict[Term, tuple[Any, int]] = {}
        self.enodes_added = 0

    def build(self, term: Term) -> tuple[Any, int]:
        cached = self.term_cache.get(term)
        if cached is not None:
            return cached
        if isinstance(term, TermRef):
            handle = self.context.class_handles.get(term.eclass)
            if handle is None:
                raise EGraphAdapterError(
                    f"no retained handle resolves to argument class {term.eclass!r}"
                )
            result = (handle, self.context.analyses[term.eclass].rank)
            self.term_cache[term] = result
            return result

        owner = self.context.represented_owner(term)
        if owner is not None:
            handle = self.context.class_handles.get(owner)
            if handle is not None:
                result = (handle, self.context.analyses[owner].rank)
                self.term_cache[term] = result
                return result

        child_results = [self.build(child) for child in term.children]
        expr, rank = self.encode(
            term.op,
            term.attrs_dict(),
            [result[0] for result in child_results],
            [result[1] for result in child_results],
        )
        interned = self.adapter.intern_expr(expr, provenance="batch")
        self.enodes_added += 1
        result = (interned.handle, rank)
        self.term_cache[term] = result
        return result


def _admit_propagation_batch(
    adapter: EGraphAdapter,
    context: SemanticContext,
    encode: EncodeFn,
    proved: list[_ProvedCandidate],
) -> tuple[int, int]:
    if not proved:
        return 0, 0
    before = adapter.freeze_snapshot()
    before_rows = sum(len(rows) for rows in before.classes.values())
    batch = AdmissionBatch(adapter, context, encode)
    roots: list[tuple[_ProvedCandidate, Any, Any]] = []
    for item in sorted(proved, key=lambda value: value.sequence):
        root_handle, _ = batch.build(item.candidate.new)
        consumer_handle = context.class_handles.get(item.occurrence.consumer_class)
        if consumer_handle is None:
            raise EGraphAdapterError(
                "no retained handle resolves to propagation consumer class "
                f"{item.occurrence.consumer_class!r}"
            )
        roots.append((item, root_handle, consumer_handle))

    equalities_added = 0
    for _item, root_handle, consumer_handle in roots:
        unioned = adapter.union_if_distinct(root_handle, consumer_handle)
        equalities_added += int(unioned)
    after = adapter.freeze_snapshot()
    after_rows = sum(len(rows) for rows in after.classes.values())
    return max(0, after_rows - before_rows), equalities_added


def run_propagation_round(
    adapter: EGraphAdapter,
    decode: DecodeFn,
    encode: EncodeFn,
    is_eligible_op: EligibleFn,
    store: ProofStore,
    *,
    analyze: Callable[[Snapshot, DecodeFn], dict[EClassRef, Any]],
    stage: str = "propagation",
    timeout: int = 10000,
    context: SemanticContext | None = None,
    worklist: PropagationWorklist | None = None,
    deadline: float | None = None,
    workers: int | None = None,
    candidate_prefilter: CandidatePrefilterFn | None = None,
    proof_chunk_size: int | None = _PROOF_CHUNK_SIZE,
    guard_shared_producers: bool = True,
) -> RoundResult:
    """Run one propagation round over a frozen snapshot."""
    if proof_chunk_size is not None and proof_chunk_size <= 0:
        raise ValueError("proof_chunk_size must be positive or None")
    workers = resolve_worker_count(workers)
    candidate_count = 0
    represented = 0
    enodes_added = 0
    equalities_added = 0
    truncated_reason: str | None = None
    truncated_stage: str | None = None
    occurrences: list[Occurrence] = []
    unfinished: set[int] = set()
    try:
        if context is None:
            snapshot = adapter.freeze_snapshot()
            context = SemanticContext.build(adapter, snapshot, decode, analyze)
        else:
            snapshot = context.snapshot
        analyses = context.analyses
        all_occurrences = eligible_occurrences(
            snapshot,
            analyses,
            decode,
            is_eligible_op,
            guard_shared_producers=guard_shared_producers,
        )
        occurrence_order = {
            id(occurrence): index for index, occurrence in enumerate(all_occurrences)
        }
        occurrences = all_occurrences
        if worklist is not None:
            occurrences = worklist.select(context, occurrences)
        unfinished = set(range(len(occurrences)))

        group_iters = [
            iter(_candidate_degree_groups(occurrence)) for occurrence in occurrences
        ]
        current_groups: dict[int, list[Candidate]] = {}
        for index, groups in enumerate(group_iters):
            group = next(groups, None)
            if group:
                current_groups[index] = group
            else:
                unfinished.discard(index)
                if worklist is not None:
                    worklist.complete(context, occurrences[index])

        active = list(current_groups)
        occurrence_candidate_indices = [0] * len(occurrences)
        while active:
            proof_chunk: list[tuple[tuple[int, int], int, Occurrence, Candidate]] = []
            successful: set[int] = set()

            def dispatch_chunk(
                chunk: list[
                    tuple[tuple[int, int], int, Occurrence, Candidate]
                ] = proof_chunk,
                successful_occurrences: set[int] = successful,
            ) -> None:
                nonlocal enodes_added, equalities_added
                if not chunk:
                    return
                proved: list[_ProvedCandidate] = []

                def verdict_completed(
                    position: int, verdict: EquivalenceVerdict
                ) -> None:
                    sequence, index, occurrence, candidate = chunk[position]
                    if not verdict.proved:
                        return
                    successful_occurrences.add(index)
                    proved.append(
                        _ProvedCandidate(
                            sequence=sequence,
                            occurrence=occurrence,
                            candidate=candidate,
                        )
                    )

                truncated: WallClockExceeded | None = None
                try:
                    prove_candidate_batch(
                        store,
                        stage,
                        snapshot,
                        analyses,
                        [
                            (candidate.current, candidate.new)
                            for (_sequence, _index, _occurrence, candidate) in chunk
                        ],
                        timeout=timeout,
                        deadline=deadline,
                        workers=workers,
                        on_verdict=verdict_completed,
                    )
                except WallClockExceeded as exc:
                    truncated = exc
                proved.sort(key=lambda item: item.sequence)
                admitted = 0
                while admitted < len(proved):
                    admission_chunk = proved[
                        admitted : admitted + _ADMISSION_CHUNK_SIZE
                    ]
                    chunk_enodes, chunk_equalities = _admit_propagation_batch(
                        adapter,
                        context,
                        encode,
                        admission_chunk,
                    )
                    enodes_added += chunk_enodes
                    equalities_added += chunk_equalities
                    admitted += len(admission_chunk)
                chunk.clear()
                if truncated is not None:
                    raise truncated

            for index in active:
                occurrence = occurrences[index]
                for candidate in current_groups[index]:
                    sequence = (
                        occurrence_order[id(occurrence)],
                        occurrence_candidate_indices[index],
                    )
                    occurrence_candidate_indices[index] += 1
                    candidate_count += 1
                    if candidate_prefilter is not None and candidate_prefilter(
                        snapshot,
                        analyses,
                        candidate.current,
                        candidate.new,
                    ):
                        continue
                    if context.candidate_is_represented(
                        occurrence.consumer_class, candidate.new
                    ):
                        represented += 1
                        successful.add(index)
                        continue
                    proof_chunk.append((sequence, index, occurrence, candidate))
                    if (
                        proof_chunk_size is not None
                        and len(proof_chunk) >= proof_chunk_size
                    ):
                        dispatch_chunk()
            dispatch_chunk()

            next_active: list[int] = []
            for index in active:
                if index in successful:
                    unfinished.discard(index)
                    if worklist is not None:
                        worklist.complete(context, occurrences[index])
                    continue
                group = next(group_iters[index], None)
                if group:
                    current_groups[index] = group
                    next_active.append(index)
                else:
                    unfinished.discard(index)
                    if worklist is not None:
                        worklist.complete(context, occurrences[index])
            active = next_active
    except WallClockExceeded as exc:
        truncated_reason = exc.reason
        truncated_stage = exc.stage

    return RoundResult(
        enodes_added=enodes_added,
        equalities_added=equalities_added,
        occurrences=len(occurrences),
        candidates=candidate_count,
        represented=represented,
        truncated_reason=truncated_reason,
        truncated_stage=truncated_stage,
        complete=truncated_reason is None and not unfinished,
        queued_continuations=len(unfinished),
    )
