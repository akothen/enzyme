"""Add proved instruction fusions to the ISA e-graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import z3

from axon.egraph.adapter import (
    EClassRef,
    EGraphAdapter,
    EGraphAdapterError,
    Snapshot,
)
from axon.egraph.analysis import AnalysisError
from axon.egraph.codec import (
    CodecError,
    encode_isa_enode,
)
from axon.egraph.context import SemanticContext
from axon.egraph.lowering import ISA_PASSTHROUGH_OPS, _sketch_to_term
from axon.egraph.proof import (
    ProofError,
    ProofStore,
    Term,
    TermApp,
    TermRef,
    WallClockExceeded,
    _collect_leaves,
    _term_to_sym,
    build_local_proof_terms,
    local_proof_key,
    prove_candidate_batch,
)
from axon.egraph.propagation import (
    AdmissionBatch,
    Occurrence,
    RoundResult,
    _op2_clone,
    _term_refs,
    eligible_occurrences,
    occurrence_identity,
)
from axon.egraph.workers import resolve_worker_count
from axon.isa_semantics import (
    _NODE_IDS,
    EquivalenceVerdict,
    ShapeExpr,
    SymTensor,
    lookup_semantics,
)
from axon.synthesizer import (
    ISA_POOL_OP_NAMES,
    ISA_POOL_OP_NAMES_SET,
    _build_general_simplification_pool,
    _node_actual_constituents,
    _op_constituents,
    _shapes_incompatible_symbolically,
    _sketch_shape_constraints_violated,
    iter_complete_sketches,
)

_FUSION_STAGE = "isa_fusion"
OPTIONAL_FUSION_TIMEOUT_MS = 100


# ISA eligibility and encoding


def is_eligible_isa_op(op: str) -> bool:
    """Return true if an ISA operation is eligible for fusion."""
    return op != "input" and op not in ISA_PASSTHROUGH_OPS


def _single_enode_zero_fusion_children(
    op: str,
    child_classes: tuple[EClassRef, ...],
) -> tuple[EClassRef, ...]:
    if op == "broadcast":
        return child_classes[:1]
    if is_eligible_isa_op(op):
        return child_classes
    return ()


def _isa_output_rank(op: str, attrs: dict[str, Any], child_ranks: list[int]) -> int:
    """Result rank of one ISA operation from its registered shape rule."""
    try:
        entry = lookup_semantics(op)
    except KeyError as exc:
        raise CodecError(str(exc)) from exc
    input_shapes = [
        ShapeExpr([z3.Int(f"_r{i}_{d}") for d in range(rank)])
        for i, rank in enumerate(child_ranks)
    ]
    return len(entry.shape_rule(input_shapes, dict(attrs)).out.dims)


def encode_isa_candidate(
    op: str,
    attrs: dict[str, Any],
    child_exprs: list[Any],
    child_ranks: list[int],
) -> tuple[Any, int]:
    """Encode one ISA application and return its expression and rank."""
    expr = encode_isa_enode(op, attrs, list(child_exprs))
    return expr, _isa_output_rank(op, attrs, child_ranks)


# Proof helpers


def isa_candidate_prefilter(
    snapshot: Snapshot,
    analyses: dict[EClassRef, Any],
    current: Term,
    candidate: Term,
) -> bool:
    """Return true for a definite rank, shape, or hardware legality failure."""
    try:
        terms = build_local_proof_terms(snapshot, analyses, current, candidate)
        return any(
            (
                _shapes_incompatible_symbolically(
                    terms.current,
                    terms.candidate,
                ),
                _sketch_shape_constraints_violated(terms.candidate),
                _literal_hardware_constraint_violated(terms.candidate),
            )
        )
    except (ProofError, CodecError, AnalysisError, KeyError, z3.Z3Exception):
        return True


def _literal_hardware_constraint_violated(candidate: SymTensor) -> bool:
    """Reject hardware constraints contradicted by concrete dimensions."""
    visited: set[int] = set()

    def visit(expr: Any) -> bool:
        identity = id(expr)
        if identity in visited:
            return False
        visited.add(identity)
        if any(visit(child) for child in expr.inputs):
            return True
        if expr.op != "scalar_tensor_tensor":
            return False
        index = expr.attrs.get("operand0_input_index")
        if index is None or not 0 <= index < len(expr.inputs):
            return False
        shape = expr.inputs[index].shape
        if len(shape) < 2 or not z3.is_int_value(shape[-1]):
            return False
        return shape[-1].as_long() != 1

    return candidate.expr is not None and visit(candidate.expr)


def _occurrence_current_term(occurrence: Occurrence) -> TermApp:
    """Build the current local term for a producer and consumer occurrence."""
    producer_app = TermApp.make(
        occurrence.producer_op,
        occurrence.producer_attrs,
        tuple(TermRef(a) for a in occurrence.producer_children),
    )
    return _op2_clone(occurrence, producer_app)


def _external_arg_classes(occurrence: Occurrence) -> list[EClassRef]:
    """Return the distinct external argument classes of an occurrence."""
    seen: dict[EClassRef, None] = {}
    for a in occurrence.producer_children:
        seen.setdefault(a, None)
    for j, b in enumerate(occurrence.consumer_children):
        if j == occurrence.q:
            continue
        seen.setdefault(b, None)
    return [c for c in seen if c != occurrence.consumer_class]


def _decoded_constituents(op: str, attrs: dict[str, Any]) -> frozenset[str]:
    """The operators one ISA op actually performs, narrowed by its attrs."""
    from axon.ir import Node

    return _node_actual_constituents(
        Node(id="_fuse", op=op, inputs=[], attrs=dict(attrs))
    )


def _allowed_fusion_hw_ops(occurrence: Occurrence) -> list[str]:
    """Return hardware operations that cover all required constituents."""
    required = _decoded_constituents(
        occurrence.producer_op, occurrence.producer_attrs
    ) | _decoded_constituents(occurrence.consumer_op, occurrence.consumer_attrs)
    if not required:
        return []
    pool_ops: set[str] = set()
    if occurrence.producer_op in ISA_POOL_OP_NAMES_SET:
        pool_ops.add(occurrence.producer_op)
    pool_ops.update(
        hw_op
        for hw_op in ISA_POOL_OP_NAMES
        if required.issubset(_op_constituents(hw_op))
    )
    return sorted(pool_ops)


# Fusion producers


def eligible_fusion_occurrences(
    snapshot: Snapshot,
    analyses: dict[EClassRef, Any],
    decode: Callable[..., Any],
    active: set[EClassRef],
) -> list[Occurrence]:
    """Return fusion occurrences with an active consumer class."""
    # Fusion has no pre-egraph analog, so it keeps the unguarded enumeration.
    occurrences = eligible_occurrences(
        snapshot,
        analyses,
        decode,
        is_eligible_isa_op,
        guard_shared_producers=False,
    )
    return [occ for occ in occurrences if occ.consumer_class in active]


# One fusion round


@dataclass
class FusionWorklist:
    """Stable single-row and occurrence identities processed by fusion."""

    single_rows: set[str] = field(default_factory=set)
    occurrences: set[str] = field(default_factory=set)

    def should_process_row(
        self, context: SemanticContext, owner: EClassRef, row: Any
    ) -> bool:
        identity = context.row_state_identity(owner, row)
        if identity is None or identity in self.single_rows:
            return False
        self.single_rows.add(identity)
        return True

    def select_occurrences(
        self, context: SemanticContext, values: list[Occurrence]
    ) -> list[Occurrence]:
        selected: list[Occurrence] = []
        for occurrence in values:
            identity = occurrence_identity(context, occurrence)
            if identity in self.occurrences:
                continue
            self.occurrences.add(identity)
            selected.append(occurrence)
        return selected


@dataclass(frozen=True)
class _FusionAdmission:
    sequence: int
    target_class: EClassRef
    candidate: Term


@dataclass(frozen=True)
class _FusionObligation:
    sequence: int
    target_class: EClassRef
    current: Term
    candidate: Term


def _admit_fusion_batch(
    adapter: EGraphAdapter,
    context: SemanticContext,
    admissions: list[_FusionAdmission],
) -> tuple[int, int]:
    if not admissions:
        return 0, 0
    before = adapter.freeze_snapshot()
    before_rows = sum(len(rows) for rows in before.classes.values())
    batch = AdmissionBatch(adapter, context, encode_isa_candidate)
    roots: list[tuple[_FusionAdmission, Any, Any]] = []
    for item in sorted(admissions, key=lambda value: value.sequence):
        root_handle, _ = batch.build(item.candidate)
        target_handle = context.class_handles.get(item.target_class)
        if target_handle is None:
            raise EGraphAdapterError(
                f"no retained handle resolves to target class {item.target_class!r}"
            )
        roots.append((item, root_handle, target_handle))

    equalities_added = 0
    for _item, root_handle, target_handle in roots:
        unioned = adapter.union_if_distinct(root_handle, target_handle)
        equalities_added += int(unioned)
    after = adapter.freeze_snapshot()
    after_rows = sum(len(rows) for rows in after.classes.values())
    return max(0, after_rows - before_rows), equalities_added


def run_fusion_round(
    adapter: EGraphAdapter,
    decode: Callable[..., Any],
    store: ProofStore,
    *,
    analyze: Callable[[Snapshot, Callable[..., Any]], dict[EClassRef, Any]],
    active_classes: Callable[[Snapshot], list[EClassRef]],
    stage: str = _FUSION_STAGE,
    timeout: int = OPTIONAL_FUSION_TIMEOUT_MS,
    context: SemanticContext | None = None,
    worklist: FusionWorklist | None = None,
    deadline: float | None = None,
    workers: int | None = None,
) -> RoundResult:
    """Process each active fusion occurrence in one frozen snapshot."""
    workers = resolve_worker_count(workers)
    if context is None:
        snapshot = adapter.freeze_snapshot()
        context = SemanticContext.build(adapter, snapshot, decode, analyze)
    else:
        snapshot = context.snapshot
    analyses = context.analyses

    active_list = active_classes(snapshot)
    active_set = set(active_list)

    admissions: list[_FusionAdmission] = []
    candidate_count = 0
    truncated_reason: str | None = None
    truncated_stage: str | None = None
    next_sequence = 0

    def should_process(owner: EClassRef, row: Any) -> bool:
        return worklist is None or worklist.should_process_row(context, owner, row)

    def prove_waves(
        candidates: list[tuple[EClassRef, Term, Term]],
    ) -> None:
        nonlocal candidate_count, next_sequence
        for start in range(0, len(candidates), workers):
            specs = candidates[start : start + workers]
            candidate_count += len(specs)
            obligations: list[_FusionObligation] = []
            for target_class, current, candidate in specs:
                try:
                    # Reject candidates whose proof key cannot be constructed.
                    local_proof_key(analyses, stage, current, candidate)
                except (
                    ProofError,
                    CodecError,
                    AnalysisError,
                    KeyError,
                    z3.Z3Exception,
                ):
                    continue
                if context.candidate_is_represented(target_class, candidate):
                    continue
                if isa_candidate_prefilter(
                    snapshot,
                    analyses,
                    current,
                    candidate,
                ):
                    continue
                obligations.append(
                    _FusionObligation(
                        sequence=next_sequence,
                        target_class=target_class,
                        current=current,
                        candidate=candidate,
                    )
                )
                next_sequence += 1

            pending = tuple(obligations)

            def verdict_completed(
                position: int,
                verdict: EquivalenceVerdict,
                _pending: tuple[_FusionObligation, ...] = pending,
            ) -> None:
                if not verdict.proved:
                    return
                obligation = _pending[position]
                admissions.append(
                    _FusionAdmission(
                        sequence=obligation.sequence,
                        target_class=obligation.target_class,
                        candidate=obligation.candidate,
                    )
                )

            prove_candidate_batch(
                store,
                stage,
                snapshot,
                analyses,
                [(obligation.current, obligation.candidate) for obligation in pending],
                timeout=timeout,
                deadline=deadline,
                workers=workers,
                on_verdict=verdict_completed,
            )

    occurrences: list[Occurrence] = []
    try:
        single_candidates: list[tuple[EClassRef, Term, Term]] = []
        for cls in active_list:
            for enode in snapshot.members(cls):
                if not should_process(cls, enode):
                    continue
                try:
                    decoded = decode(snapshot, enode)
                except Exception:
                    continue
                candidate_children = _single_enode_zero_fusion_children(
                    decoded.op,
                    decoded.child_classes,
                )
                if not candidate_children:
                    continue
                if any(child not in analyses for child in decoded.child_classes):
                    continue
                current = TermApp.make(
                    decoded.op,
                    dict(decoded.attrs),
                    tuple(TermRef(child) for child in decoded.child_classes),
                )
                seen: set[EClassRef] = set()
                for child in candidate_children:
                    if child == cls or child in seen:
                        continue
                    seen.add(child)
                    single_candidates.append((cls, current, TermRef(child)))
        prove_waves(single_candidates)

        occurrences = eligible_fusion_occurrences(
            snapshot,
            analyses,
            decode,
            active_set,
        )
        if worklist is not None:
            occurrences = worklist.select_occurrences(context, occurrences)
        for occurrence in occurrences:
            current = _occurrence_current_term(occurrence)
            leaves = _collect_leaves((current,))
            if occurrence.consumer_class in leaves:
                continue

            prove_waves(
                [
                    (
                        occurrence.consumer_class,
                        current,
                        TermRef(external),
                    )
                    for external in _external_arg_classes(occurrence)
                ]
            )

            if any(leaf not in occurrence.analyses for leaf in leaves):
                continue
            syms = {
                leaf: SymTensor(
                    _NODE_IDS.next_name("fusearg"),
                    shape=tuple(occurrence.analyses[leaf].dims),
                )
                for leaf in leaves
            }
            target_sym = _term_to_sym(current, syms)
            input_syms = [syms[leaf] for leaf in leaves]
            allowed = _allowed_fusion_hw_ops(occurrence)
            if not allowed:
                continue
            pool = _build_general_simplification_pool(
                input_syms, allowed_hw_ops=allowed
            )
            formal_to_class = {id(sym): leaf for leaf, sym in syms.items()}
            # The producer-expanded current erases the fact that the expansion
            # inhabits the operand class, so child-only sketches also prove
            # against the consumer's own unexpanded e-node term.
            consumer_term = TermApp.make(
                occurrence.consumer_op,
                occurrence.consumer_attrs,
                tuple(TermRef(child) for child in occurrence.consumer_children),
            )
            consumer_child_set = set(occurrence.consumer_children)
            one_node_candidates: list[tuple[EClassRef, Term, Term]] = []
            sketches = iter_complete_sketches(
                target_sym,
                pool,
                max_hw_size=1,
                input_syms=input_syms,
            )
            while True:
                try:
                    sketch, _candidate_sym = next(sketches)
                except StopIteration:
                    break
                candidate_term = _sketch_to_term(sketch, formal_to_class)
                one_node_candidates.append(
                    (occurrence.consumer_class, current, candidate_term)
                )
                if _term_refs(candidate_term) <= consumer_child_set:
                    one_node_candidates.append(
                        (occurrence.consumer_class, consumer_term, candidate_term)
                    )
                if len(one_node_candidates) >= workers:
                    prove_waves(one_node_candidates)
                    one_node_candidates.clear()
            prove_waves(one_node_candidates)
    except WallClockExceeded as exc:
        truncated_reason = exc.reason
        truncated_stage = exc.stage

    enodes_added, equalities_added = _admit_fusion_batch(adapter, context, admissions)

    return RoundResult(
        enodes_added=enodes_added,
        equalities_added=equalities_added,
        occurrences=len(occurrences),
        candidates=candidate_count,
        truncated_reason=truncated_reason,
        truncated_stage=truncated_stage,
    )
