"""Run prepared local proof obligations through a standard process pool."""

from __future__ import annotations

import gc
import multiprocessing
import os
import time
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from threading import Lock

from axon.isa_semantics import (
    _Z3_LOCK,
    EquivalenceVerdict,
    Precondition,
    SymTensor,
    _ShapeProvedSymTensor,
    check_valid_and_equivalent,
)


@dataclass(frozen=True)
class LocalProofObligation:
    """One immutable, reusable local proof request prepared by the caller."""

    proof_key: Hashable
    current: SymTensor
    candidate: SymTensor
    preconditions: tuple[Precondition, ...] = ()
    timeout: int = 10000

    def __post_init__(self) -> None:
        hash(self.proof_key)
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


@dataclass(frozen=True)
class LocalProofResult:
    """The local verdict for one input obligation."""

    sequence: int
    proof_key: Hashable
    verdict: EquivalenceVerdict
    cache_hit: bool
    deduplicated: bool
    timeout: int = 10000


class ProofBatchDeadlineExceeded(TimeoutError):
    """Deadline expiration carrying every result completed before truncation."""

    def __init__(self, completed: tuple[LocalProofResult, ...]) -> None:
        self.completed = completed
        super().__init__("local proof batch deadline exceeded")


@dataclass(frozen=True)
class _ProofCheck:
    verdict: EquivalenceVerdict
    timeout: int


class _ChecksDeadlineReached(TimeoutError):
    def __init__(self, completed: dict[int, _ProofCheck]) -> None:
        self.completed = completed
        super().__init__("local proof checks deadline exceeded")


# Fork workers inherit the batch through process memory, so z3 terms need no
# pickling. The lock serializes batches over the shared module state.
_FORK_OBLIGATIONS: tuple[LocalProofObligation, ...] = ()
_FORK_DEADLINE: float | None = None
_FORK_BATCH_LOCK = Lock()


def _remaining_timeout(obligation: LocalProofObligation) -> int | None:
    """Cap the obligation timeout by the batch deadline; None when lapsed."""
    if _FORK_DEADLINE is None:
        return obligation.timeout
    remaining_ms = int((_FORK_DEADLINE - time.monotonic()) * 1000)
    if remaining_ms < 1:
        return None
    return min(obligation.timeout, remaining_ms)


def _check_obligation(obligation: LocalProofObligation) -> _ProofCheck:
    """Prove one obligation through the shared strict proof API."""
    timeout = _remaining_timeout(obligation)
    if timeout is None:
        return _ProofCheck(
            verdict=EquivalenceVerdict(
                proved=False,
                stage="value",
                detail="candidate deadline exhausted",
                elapsed_ms=1,
            ),
            timeout=1,
        )
    started_at = time.monotonic()
    verdict = check_valid_and_equivalent(
        obligation.current,
        obligation.candidate,
        timeout=timeout,
        preconditions=list(obligation.preconditions),
    )
    # Discard any verdict arriving past the obligation's own wall so late
    # solver returns cannot admit candidates the deadline already rejected.
    if time.monotonic() - started_at > obligation.timeout / 1000.0:
        stage = (
            "value"
            if isinstance(obligation.current, _ShapeProvedSymTensor)
            else "shape"
        )
        return _ProofCheck(
            verdict=EquivalenceVerdict(
                proved=False,
                stage=stage,
                detail="unknown",
                elapsed_ms=obligation.timeout,
            ),
            timeout=obligation.timeout,
        )
    return _ProofCheck(verdict=verdict, timeout=timeout)


def _check_fork_obligation(index: int) -> _ProofCheck:
    return _check_obligation(_FORK_OBLIGATIONS[index])


def _disable_worker_gc() -> None:
    # Forked workers must not finalize inherited Z3 objects during a proof.
    gc.disable()


def _fork_context() -> multiprocessing.context.BaseContext | None:
    if "fork" not in multiprocessing.get_all_start_methods():
        return None
    return multiprocessing.get_context("fork")


def _serial_checks(
    obligations: tuple[LocalProofObligation, ...],
    deadline: float | None,
    on_completed: Callable[[int, _ProofCheck], None],
) -> None:
    completed: dict[int, _ProofCheck] = {}
    for index, obligation in enumerate(obligations):
        if deadline is not None and time.monotonic() >= deadline:
            raise _ChecksDeadlineReached(completed)
        with _Z3_LOCK:
            check = _check_obligation(obligation)
        completed[index] = check
        on_completed(index, check)


def _parallel_checks(
    obligations: tuple[LocalProofObligation, ...],
    context: multiprocessing.context.BaseContext,
    workers: int,
    deadline: float | None,
    on_completed: Callable[[int, _ProofCheck], None],
) -> None:
    completed: dict[int, _ProofCheck] = {}
    deadline_reached = False
    # The pool's manager thread must not gc-finalize egglog or Z3 objects
    # created on the main thread; suspend collection for the pool phase.
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    pool: ProcessPoolExecutor | None = None
    try:
        pool = ProcessPoolExecutor(
            max_workers=min(workers, len(obligations)),
            mp_context=context,
            initializer=_disable_worker_gc,
        )
        indices: dict[Future[_ProofCheck], int] = {
            pool.submit(_check_fork_obligation, index): index
            for index in range(len(obligations))
        }
        pending = set(indices)
        while pending:
            wait_timeout: float | None = None
            if deadline is not None:
                wait_timeout = deadline - time.monotonic()
                if wait_timeout <= 0.0:
                    deadline_reached = True
                    raise _ChecksDeadlineReached(completed)
            done, pending = wait(
                pending, timeout=wait_timeout, return_when=FIRST_COMPLETED
            )
            for future in sorted(done, key=indices.__getitem__):
                index = indices[future]
                check = future.result()
                completed[index] = check
                on_completed(index, check)
    finally:
        if pool is not None:
            # A lapsed deadline must not wait for in-flight solver queries,
            # so terminate the workers before the waiting shutdown.
            if deadline_reached:
                for process in (pool._processes or {}).values():
                    process.terminate()
            pool.shutdown(wait=True, cancel_futures=True)
        if gc_was_enabled:
            gc.enable()


def run_local_proof_batch(
    obligations: Sequence[LocalProofObligation],
    *,
    workers: int | None = 1,
    deadline: float | None = None,
    on_result: Callable[[LocalProofResult], None] | None = None,
) -> tuple[LocalProofResult, ...]:
    """Prove each stable key once and return results in input order."""
    prepared = tuple(obligations)
    if workers is not None and workers < 1:
        raise ValueError("workers must be positive or None")
    if not prepared:
        return ()
    if deadline is not None and time.monotonic() >= deadline:
        raise ProofBatchDeadlineExceeded(())

    miss_index: dict[tuple[Hashable, int], int] = {}
    misses: list[LocalProofObligation] = []
    origins: list[str] = []
    dispatch_keys: list[tuple[Hashable, int]] = []
    for obligation in prepared:
        dispatch_key = (obligation.proof_key, obligation.timeout)
        if dispatch_key in miss_index:
            origins.append("deduplicated")
            dispatch_keys.append(dispatch_key)
        else:
            miss_index[dispatch_key] = len(misses)
            misses.append(obligation)
            origins.append("dispatch")
            dispatch_keys.append(dispatch_key)

    miss_tuple = tuple(misses)
    effective_workers = (
        min(os.cpu_count() or 1, len(miss_tuple)) if workers is None else workers
    )
    completed_results: dict[int, LocalProofResult] = {}

    def complete_result(result: LocalProofResult) -> None:
        completed_results[result.sequence] = result
        if on_result is not None:
            on_result(result)

    def complete_check(miss_sequence: int, check: _ProofCheck) -> None:
        obligation = miss_tuple[miss_sequence]
        dispatch_key = (obligation.proof_key, obligation.timeout)
        for sequence, candidate_key in enumerate(dispatch_keys):
            if candidate_key != dispatch_key:
                continue
            complete_result(
                LocalProofResult(
                    sequence=sequence,
                    proof_key=prepared[sequence].proof_key,
                    verdict=check.verdict,
                    cache_hit=False,
                    deduplicated=origins[sequence] == "deduplicated",
                    timeout=check.timeout,
                )
            )

    deadline_error: _ChecksDeadlineReached | None = None
    context = _fork_context()
    global _FORK_DEADLINE, _FORK_OBLIGATIONS
    with _FORK_BATCH_LOCK:
        _FORK_OBLIGATIONS = miss_tuple
        _FORK_DEADLINE = deadline
        try:
            if not miss_tuple:
                pass
            elif effective_workers <= 1 or len(miss_tuple) == 1 or context is None:
                _serial_checks(miss_tuple, deadline, complete_check)
            else:
                _parallel_checks(
                    miss_tuple,
                    context,
                    effective_workers,
                    deadline,
                    complete_check,
                )
        except _ChecksDeadlineReached as exc:
            deadline_error = exc
        finally:
            _FORK_OBLIGATIONS = ()
            _FORK_DEADLINE = None

    results = tuple(completed_results[index] for index in sorted(completed_results))
    if deadline_error is not None:
        raise ProofBatchDeadlineExceeded(results) from None
    return results
