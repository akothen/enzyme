"""Build local proof obligations for e-graph equality candidates.

Every solver query is dispatched through the shared strict proof API,
``axon.isa_semantics.check_valid_and_equivalent``, by ``proof_parallel``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import z3

from axon.egraph.adapter import EClassRef, Snapshot
from axon.egraph.proof_parallel import (
    LocalProofObligation,
    LocalProofResult,
    ProofBatchDeadlineExceeded,
    run_local_proof_batch,
)
from axon.ir import Node, _sym_expr_from_graph_node
from axon.isa_semantics import (
    _NODE_IDS,
    EquivalenceVerdict,
    Precondition,
    SymTensor,
    _with_proved_shape,
    _with_shape_only,
)


class ProofError(RuntimeError):
    """Raised when proof terms cannot be constructed for an occurrence."""


class WallClockExceeded(RuntimeError):
    """Raised when the owning stage's wall-clock deadline lapses mid-work."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.reason = "wall_clock_seconds"
        super().__init__(f"{stage}: wall_clock_seconds")


_CANDIDATE_LOCAL_ERRORS = (ProofError, KeyError, z3.Z3Exception)


@dataclass(frozen=True)
class TermRef:
    """A leaf of a local candidate term: one argument e-class."""

    eclass: EClassRef


def _hashable_attr(value: Any) -> Any:
    """Convert list attributes to hashable tuples."""
    if isinstance(value, list):
        return tuple(_hashable_attr(v) for v in value)
    return value


@dataclass(frozen=True)
class TermApp:
    """An operation applied to argument e-classes and nested applications."""

    op: str
    attrs: tuple[tuple[str, Any], ...]
    children: tuple[TermApp | TermRef, ...]

    @staticmethod
    def make(
        op: str, attrs: dict[str, Any], children: tuple[TermApp | TermRef, ...]
    ) -> TermApp:
        return TermApp(
            op=op,
            attrs=tuple(sorted((k, _hashable_attr(v)) for k, v in attrs.items())),
            children=children,
        )

    def attrs_dict(self) -> dict[str, Any]:
        return dict(self.attrs)


Term = TermApp | TermRef

CheckpointFn = Callable[[], None]


def _run_checkpoint(checkpoint: CheckpointFn | None) -> None:
    if checkpoint is not None:
        checkpoint()


def _collect_leaves(
    terms: tuple[Term, ...],
    checkpoint: CheckpointFn | None = None,
) -> list[EClassRef]:
    """Distinct argument e-classes in deterministic first-occurrence order."""
    seen: dict[EClassRef, None] = {}

    def walk(term: Term) -> None:
        _run_checkpoint(checkpoint)
        if isinstance(term, TermRef):
            seen.setdefault(term.eclass, None)
            return
        for child in term.children:
            walk(child)

    for term in terms:
        _run_checkpoint(checkpoint)
        walk(term)
    return list(seen)


def _term_to_sym(
    term: Term,
    leaf_syms: dict[EClassRef, SymTensor],
    checkpoint: CheckpointFn | None = None,
) -> SymTensor:
    _run_checkpoint(checkpoint)
    if isinstance(term, TermRef):
        return leaf_syms[term.eclass]
    child_syms = [_term_to_sym(child, leaf_syms, checkpoint) for child in term.children]
    _run_checkpoint(checkpoint)
    node = Node(
        id=_NODE_IDS.next_name(f"proof_{term.op}"),
        op=term.op,
        inputs=[sym.id for sym in child_syms],
        attrs=term.attrs_dict(),
    )
    sym = _sym_expr_from_graph_node(node, child_syms)
    _run_checkpoint(checkpoint)
    return sym


@dataclass(frozen=True)
class ProofTerms:
    """Compiled current and candidate terms plus their allowed assumptions."""

    current: SymTensor
    candidate: SymTensor
    preconditions: tuple[Precondition, ...]
    argument_classes: tuple[EClassRef, ...]


def build_local_proof_terms(
    snapshot: Snapshot,
    analyses: dict[EClassRef, Any],
    current: Term,
    candidate: Term,
    checkpoint: CheckpointFn | None = None,
) -> ProofTerms:
    """Build proof terms with one symbol per distinct argument class."""
    _run_checkpoint(checkpoint)
    leaves = _collect_leaves((current, candidate), checkpoint)
    leaf_syms: dict[EClassRef, SymTensor] = {}
    preconditions: list[Precondition] = []
    for index, leaf in enumerate(leaves):
        _run_checkpoint(checkpoint)
        analysis = analyses.get(leaf)
        if analysis is None:
            raise ProofError(f"argument e-class {leaf!r} has no analysis")
        sym = SymTensor(_NODE_IDS.next_name(f"parg{index}"), shape=tuple(analysis.dims))
        leaf_syms[leaf] = sym
        for fact in analysis.facts:
            _run_checkpoint(checkpoint)
            preconditions.append(
                Precondition(description=f"arg{index} guaranteed fact", constraint=fact)
            )
    return ProofTerms(
        current=_term_to_sym(current, leaf_syms, checkpoint),
        candidate=_term_to_sym(candidate, leaf_syms, checkpoint),
        preconditions=tuple(preconditions),
        argument_classes=tuple(leaves),
    )


def _term_pattern(
    term: Term,
    leaf_index: dict[EClassRef, int],
    checkpoint: CheckpointFn | None = None,
) -> Any:
    _run_checkpoint(checkpoint)
    if isinstance(term, TermRef):
        return ("arg", leaf_index[term.eclass])
    attrs: list[tuple[str, str]] = []
    for key, value in term.attrs:
        _run_checkpoint(checkpoint)
        attrs.append((key, repr(value)))
    return (
        term.op,
        tuple(sorted(attrs)),
        tuple(_term_pattern(child, leaf_index, checkpoint) for child in term.children),
    )


def _leaf_fact_signature(
    analyses: dict[EClassRef, Any],
    leaf: EClassRef,
    checkpoint: CheckpointFn | None = None,
) -> Any:
    _run_checkpoint(checkpoint)
    analysis = analyses.get(leaf)
    if analysis is None:
        raise ProofError(f"argument e-class {leaf!r} has no analysis")
    dims: list[str] = []
    for dim in analysis.dims:
        _run_checkpoint(checkpoint)
        dims.append(str(dim))
    facts: list[str] = []
    for fact in analysis.facts:
        _run_checkpoint(checkpoint)
        facts.append(fact.sexpr())
    return tuple(dims), tuple(sorted(facts))


def local_proof_key(
    analyses: dict[EClassRef, Any],
    stage: str,
    current: Term,
    candidate: Term,
    checkpoint: CheckpointFn | None = None,
) -> Any:
    """Return the cache key for a reusable local identity."""
    _run_checkpoint(checkpoint)
    leaves = _collect_leaves((current, candidate), checkpoint)
    leaf_index: dict[EClassRef, int] = {}
    for index, leaf in enumerate(leaves):
        _run_checkpoint(checkpoint)
        leaf_index[leaf] = index
    return (
        "local",
        stage,
        _term_pattern(current, leaf_index, checkpoint),
        _term_pattern(candidate, leaf_index, checkpoint),
        tuple(_leaf_fact_signature(analyses, leaf, checkpoint) for leaf in leaves),
    )


@dataclass
class ProofStore:
    """Final local verdict caches and dispatch counters, not a proof journal."""

    local_cache: dict[Any, EquivalenceVerdict] = field(default_factory=dict)
    local_shape_cache: dict[Any, EquivalenceVerdict] = field(default_factory=dict)
    dispatch_count: int = 0
    shape_dispatch_count: int = 0
    value_dispatch_count: int = 0


def _cached_verdict(
    cache: dict[Any, EquivalenceVerdict],
    key: Any,
    timeout: int,
) -> EquivalenceVerdict | None:
    proved = cache.get(key)
    if proved is not None and proved.proved:
        return proved
    exact = cache.get((key, "timeout", timeout))
    if exact is not None:
        return exact
    larger = sorted(
        (
            cached_key[2],
            verdict,
        )
        for cached_key, verdict in cache.items()
        if isinstance(cached_key, tuple)
        and len(cached_key) == 3
        and cached_key[0] == key
        and cached_key[1] == "timeout"
        and isinstance(cached_key[2], int)
        and cached_key[2] >= timeout
        and not verdict.proved
    )
    if larger:
        return larger[0][1]
    return None


def _cache_verdict(
    cache: dict[Any, EquivalenceVerdict],
    key: Any,
    timeout: int,
    verdict: EquivalenceVerdict,
) -> None:
    cache_key = key if verdict.proved else (key, "timeout", timeout)
    cache[cache_key] = verdict


def _deadline_verdict(stage: str) -> EquivalenceVerdict:
    return EquivalenceVerdict(
        proved=False,
        stage=stage,
        detail="candidate aggregate deadline exhausted",
    )


def _candidate_error_verdict(error: Exception) -> EquivalenceVerdict:
    return EquivalenceVerdict(
        proved=False,
        stage="candidate_error",
        detail=f"{type(error).__name__}: {error}",
    )


def _run_proof_batch_isolating_candidate_errors(
    obligations: list[LocalProofObligation],
    *,
    workers: int,
    deadline: float | None,
    on_result: Callable[[LocalProofResult], None],
    reserve_retry: Callable[[LocalProofObligation], LocalProofObligation],
    charge_failed_attempt: Callable[[tuple[LocalProofObligation, ...], int], None],
) -> tuple[tuple[LocalProofResult, ...], dict[Any, Exception]]:
    reported: set[Any] = set()

    def report(result: LocalProofResult) -> None:
        if result.proof_key in reported:
            return
        reported.add(result.proof_key)
        on_result(result)

    batch_started_at = time.monotonic()
    try:
        results = run_local_proof_batch(
            obligations,
            workers=workers,
            deadline=deadline,
            on_result=report,
        )
    except _CANDIDATE_LOCAL_ERRORS as batch_error:
        unreported = tuple(
            obligation
            for obligation in obligations
            if obligation.proof_key not in reported
        )
        charge_failed_attempt(
            unreported,
            max(1, math.ceil((time.monotonic() - batch_started_at) * 1000)),
        )
        if len(obligations) == 1:
            return (), {obligations[0].proof_key: batch_error}
        isolated_results: list[LocalProofResult] = []
        candidate_errors: dict[Any, Exception] = {}
        for obligation in obligations:
            if obligation.proof_key in reported:
                continue
            retry = reserve_retry(obligation)
            retry_started_at = time.monotonic()
            try:
                retry_results = run_local_proof_batch(
                    [retry],
                    workers=1,
                    deadline=deadline,
                    on_result=report,
                )
            except _CANDIDATE_LOCAL_ERRORS as candidate_error:
                if obligation.proof_key not in reported:
                    charge_failed_attempt(
                        (obligation,),
                        max(
                            1,
                            math.ceil((time.monotonic() - retry_started_at) * 1000),
                        ),
                    )
                candidate_errors[obligation.proof_key] = candidate_error
                continue
            for result in retry_results:
                report(result)
            isolated_results.extend(retry_results)
        return tuple(isolated_results), candidate_errors
    for result in results:
        report(result)
    return results, {}


def prove_candidate_batch(
    store: ProofStore,
    stage: str,
    snapshot: Snapshot,
    analyses: dict[EClassRef, Any],
    candidates: list[tuple[Term, Term]],
    *,
    timeout: int = 10000,
    deadline: float | None = None,
    workers: int = 1,
    on_verdict: Callable[[int, EquivalenceVerdict], None] | None = None,
) -> list[EquivalenceVerdict]:
    """Prove one ordered candidate batch with parallel local cache misses."""
    batch_deadline = deadline

    verdicts: list[EquivalenceVerdict | None] = [None] * len(candidates)

    def lookup_timeout(remaining: int) -> int | None:
        if remaining <= 0:
            return None
        if batch_deadline is None:
            return remaining
        wall_remaining = math.ceil((batch_deadline - time.monotonic()) * 1000)
        if wall_remaining <= 0:
            return None
        return min(remaining, wall_remaining)

    def check_preparation_deadline() -> None:
        if batch_deadline is not None and time.monotonic() >= batch_deadline:
            raise WallClockExceeded(stage)

    def preparation_elapsed_ms(started_at: float) -> int:
        return max(0, math.ceil((time.monotonic() - started_at) * 1000))

    def complete(index: int, verdict: EquivalenceVerdict) -> None:
        verdicts[index] = verdict
        if on_verdict is not None:
            on_verdict(index, verdict)

    local_keys: dict[int, Any] = {}
    local_indices: dict[Any, list[int]] = {}
    local_states: dict[Any, EquivalenceVerdict] = {}
    local_shape_proved: set[Any] = set()
    local_remaining: dict[Any, int] = {}
    representatives: dict[Any, int] = {}
    for index, (current, candidate) in enumerate(candidates):
        check_preparation_deadline()
        preparation_started_at = time.monotonic()
        try:
            key = local_proof_key(
                analyses,
                stage,
                current,
                candidate,
                checkpoint=check_preparation_deadline,
            )
        except _CANDIDATE_LOCAL_ERRORS as exc:
            complete(index, _candidate_error_verdict(exc))
            continue
        finally:
            key_elapsed_ms = preparation_elapsed_ms(preparation_started_at)
            check_preparation_deadline()
        local_keys[index] = key
        local_indices.setdefault(key, []).append(index)
        candidate_remaining = max(0, timeout - key_elapsed_ms)
        local_remaining[key] = min(
            local_remaining.get(key, candidate_remaining),
            candidate_remaining,
        )
        effective_lookup_timeout = lookup_timeout(local_remaining[key])
        if effective_lookup_timeout is None:
            local_states[key] = _deadline_verdict("shape")
            continue
        cached = _cached_verdict(store.local_cache, key, effective_lookup_timeout)
        if cached is not None:
            local_states[key] = cached
            if cached.proved:
                complete(index, cached)
        else:
            shape_verdict = _cached_verdict(
                store.local_shape_cache,
                ("shape", key),
                effective_lookup_timeout,
            )
            if shape_verdict is not None and not shape_verdict.proved:
                local_states[key] = shape_verdict
            else:
                if shape_verdict is not None:
                    local_shape_proved.add(key)
                representatives.setdefault(key, index)

    local_terms: dict[Any, ProofTerms] = {}
    obligations: list[LocalProofObligation] = []
    local_phase_by_key: dict[Any, str] = {}
    pending_local_values: set[Any] = set()
    for key, index in representatives.items():
        current, candidate = candidates[index]
        check_preparation_deadline()
        preparation_started_at = time.monotonic()
        try:
            terms = build_local_proof_terms(
                snapshot,
                analyses,
                current,
                candidate,
                checkpoint=check_preparation_deadline,
            )
        except _CANDIDATE_LOCAL_ERRORS as exc:
            for candidate_index in local_indices[key]:
                complete(candidate_index, _candidate_error_verdict(exc))
            continue
        finally:
            local_remaining[key] = max(
                0,
                local_remaining[key] - preparation_elapsed_ms(preparation_started_at),
            )
            check_preparation_deadline()
        effective_timeout = lookup_timeout(local_remaining[key])
        if effective_timeout is None:
            local_states[key] = _deadline_verdict("shape")
            continue
        local_terms[key] = terms
        if key in local_shape_proved:
            obligation_current, obligation_candidate = _with_proved_shape(
                terms.current, terms.candidate, ("shape", key)
            )
            local_phase_by_key[key] = "value"
            store.value_dispatch_count += 1
        else:
            obligation_current, obligation_candidate = _with_shape_only(
                terms.current, terms.candidate, ("shape", key)
            )
            local_phase_by_key[key] = "shape"
            store.shape_dispatch_count += 1
        store.dispatch_count += 1
        obligations.append(
            LocalProofObligation(
                proof_key=key,
                current=obligation_current,
                candidate=obligation_candidate,
                preconditions=terms.preconditions,
                timeout=effective_timeout,
            )
        )

    handled_local_keys: set[Any] = set()

    def finish_local_result(result: LocalProofResult) -> None:
        if result.proof_key in handled_local_keys:
            return
        handled_local_keys.add(result.proof_key)
        key = result.proof_key
        local_remaining[key] = max(0, local_remaining[key] - result.verdict.elapsed_ms)
        if local_phase_by_key[key] == "shape":
            _cache_verdict(
                store.local_shape_cache,
                ("shape", key),
                result.timeout,
                result.verdict,
            )
            if result.verdict.proved:
                local_shape_proved.add(key)
                pending_local_values.add(key)
            else:
                local_states[key] = result.verdict
            return
        _cache_verdict(
            store.local_cache,
            key,
            result.timeout,
            result.verdict,
        )
        local_states[key] = result.verdict
        if not result.verdict.proved:
            return
        for index in local_indices[key]:
            if verdicts[index] is not None:
                continue
            complete(index, result.verdict)

    def flush_unresolved_local_verdicts() -> None:
        for key, verdict in local_states.items():
            for index in local_indices[key]:
                if verdicts[index] is not None:
                    continue
                complete(index, verdict)

    def execute_local_phase(
        phase_obligations: list[LocalProofObligation],
    ) -> None:
        if not phase_obligations:
            return

        def charge_failed_local_attempt(
            failed_obligations: tuple[LocalProofObligation, ...],
            elapsed_ms: int,
        ) -> None:
            for obligation in failed_obligations:
                key = obligation.proof_key
                local_remaining[key] = max(
                    0,
                    local_remaining[key] - elapsed_ms,
                )

        def reserve_local_retry(
            obligation: LocalProofObligation,
        ) -> LocalProofObligation:
            key = obligation.proof_key
            if batch_deadline is not None and time.monotonic() >= batch_deadline:
                raise ProofBatchDeadlineExceeded(())
            effective_timeout = lookup_timeout(local_remaining[key])
            if effective_timeout is None:
                raise ProofBatchDeadlineExceeded(())
            store.dispatch_count += 1
            if local_phase_by_key[key] == "shape":
                store.shape_dispatch_count += 1
            else:
                store.value_dispatch_count += 1
            return replace(obligation, timeout=effective_timeout)

        try:
            results, candidate_errors = _run_proof_batch_isolating_candidate_errors(
                phase_obligations,
                workers=workers,
                deadline=batch_deadline,
                on_result=finish_local_result,
                reserve_retry=reserve_local_retry,
                charge_failed_attempt=charge_failed_local_attempt,
            )
        except ProofBatchDeadlineExceeded as exc:
            for result in exc.completed:
                finish_local_result(result)
            flush_unresolved_local_verdicts()
            if batch_deadline is not None and time.monotonic() >= batch_deadline:
                raise WallClockExceeded(stage) from exc
            raise
        for result in results:
            finish_local_result(result)
        for key, error in candidate_errors.items():
            for index in local_indices[key]:
                if verdicts[index] is not None:
                    continue
                complete(index, _candidate_error_verdict(error))

    execute_local_phase(obligations)

    value_obligations: list[LocalProofObligation] = []
    handled_local_keys.clear()
    for key in sorted(pending_local_values, key=representatives.__getitem__):
        effective_timeout = lookup_timeout(local_remaining[key])
        if effective_timeout is None:
            local_states[key] = _deadline_verdict("value")
            continue
        terms = local_terms[key]
        proved_current, proved_candidate = _with_proved_shape(
            terms.current, terms.candidate, ("shape", key)
        )
        local_phase_by_key[key] = "value"
        store.dispatch_count += 1
        store.value_dispatch_count += 1
        value_obligations.append(
            LocalProofObligation(
                proof_key=key,
                current=proved_current,
                candidate=proved_candidate,
                preconditions=terms.preconditions,
                timeout=effective_timeout,
            )
        )
    execute_local_phase(value_obligations)

    flush_unresolved_local_verdicts()

    return [verdict for verdict in verdicts if verdict is not None]
