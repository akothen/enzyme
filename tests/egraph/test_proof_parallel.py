"""Tests for deterministic process-parallel local proof batches.

The batch API contract: multi-core process execution, results returned in
input order regardless of completion order, worker failure propagation,
and serial fallback.
"""

from __future__ import annotations

import gc
import inspect
import multiprocessing
import os
import time
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import pytest

import axon.egraph.fusion as fusion
import axon.egraph.isa as isa
import axon.egraph.lowering as lowering
import axon.egraph.pipeline as pipeline
import axon.egraph.proof_parallel as proof_parallel
import axon.egraph.propagation as propagation
import axon.egraph.tensor as tensor
import axon.egraph.workers as worker_defaults
from axon.egraph.adapter import EGraphAdapter
from axon.egraph.codec import encode_isa_input
from axon.egraph.lowering import LoweringStatus
from axon.egraph.proof_parallel import LocalProofObligation, run_local_proof_batch
from axon.ir import Node, nuGraph
from axon.isa_semantics import EquivalenceVerdict, SymTensor, _fallback_timeout

_HAS_FORK = "fork" in multiprocessing.get_all_start_methods()

requires_fork = pytest.mark.skipif(
    not _HAS_FORK,
    reason="process-parallel local proofs require fork",
)


def _obligation(key: str, name: str, timeout: int = 100) -> LocalProofObligation:
    term = SymTensor(name, rank=1)
    return LocalProofObligation(key, term, term, timeout=timeout)


def _fake_checker(
    current: SymTensor,
    candidate: SymTensor,
    *,
    timeout: int,
    preconditions: list[Any],
) -> EquivalenceVerdict:
    del candidate, timeout, preconditions
    delay = {"slow": 0.04, "medium": 0.02}.get(current.id, 0)
    time.sleep(delay)
    return EquivalenceVerdict(
        proved=current.id != "reject",
        stage="proved" if current.id != "reject" else "value",
        detail=f"{current.id}:{os.getpid()}",
    )


def _input_graph() -> nuGraph:
    return nuGraph(
        nodes=[Node("x", "input", [], {"shape": (1,)})],
        output_ids=("x",),
        input_ids=("x",),
    )


def test_all_public_entry_points_share_automatic_worker_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for entry_point in (
        pipeline.build_egraph_search,
        pipeline.iter_synthesized_hw_graphs,
        tensor.saturate_tensor,
        isa.saturate_isa,
        lowering.lower_tensor_egraph,
        propagation.run_propagation_round,
        fusion.run_fusion_round,
    ):
        assert inspect.signature(entry_point).parameters["workers"].default is None

    resolutions = 0

    def automatic_workers() -> int:
        nonlocal resolutions
        resolutions += 1
        return 9

    monkeypatch.setattr(
        worker_defaults,
        "available_cpu_count",
        automatic_workers,
    )
    observed: list[tuple[str, int]] = []

    def saturate_tensor(*_args: Any, workers: int, **_kwargs: Any) -> Any:
        observed.append(("tensor", workers))
        return tensor.SaturationStatus("fixed_point", 1, 0, 0, 0, 0.0)

    def lower(
        _snapshot: Any,
        outputs: list[Any],
        adapter: EGraphAdapter,
        _store: Any,
        _max_hw_size: int,
        *,
        workers: int,
        **_kwargs: Any,
    ) -> Any:
        observed.append(("lowering", workers))
        handle = adapter.intern_expr(
            encode_isa_input("x", (1,)),
            provenance="worker-test",
        ).handle
        return (
            adapter,
            dict.fromkeys(outputs, handle),
            LoweringStatus(status="lowered", realized=len(outputs)),
        )

    def saturate_isa(*_args: Any, workers: int, **_kwargs: Any) -> Any:
        observed.append(("isa", workers))
        return isa.SaturationStatus("fixed_point", 1, 0, 0, 0, 0.0)

    monkeypatch.setattr(pipeline, "saturate_tensor", saturate_tensor)
    monkeypatch.setattr(pipeline, "lower_tensor_egraph", lower)
    monkeypatch.setattr(pipeline, "saturate_isa", saturate_isa)

    search = pipeline.build_egraph_search(_input_graph())

    assert search.terminal_status.status == "completed"
    assert resolutions == 1
    # Lowering runs once after tensor saturation.
    assert observed == [
        ("tensor", 9),
        ("lowering", 9),
        ("isa", 9),
    ]


def test_explicit_worker_count_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker_defaults,
        "available_cpu_count",
        lambda: pytest.fail("explicit workers must not resolve the default"),
    )

    iterator = pipeline.iter_synthesized_hw_graphs(_input_graph(), workers=7)

    assert iterator._options["workers"] == 7
    assert worker_defaults.resolve_worker_count(11) == 11


def test_single_core_host_uses_one_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_defaults.os, "cpu_count", lambda: 1)

    assert worker_defaults.available_cpu_count() == 1
    assert worker_defaults.resolve_worker_count(None) == 1


def test_single_worker_batch_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proof_parallel, "check_valid_and_equivalent", _fake_checker)
    obligations = [
        _obligation("repeat", "first"),
        _obligation("other-key", "other"),
        _obligation("repeat", "duplicate-never-called"),
        _obligation("reject", "reject"),
    ]

    results = run_local_proof_batch(obligations, workers=1)

    assert [result.sequence for result in results] == [0, 1, 2, 3]
    assert results[0].verdict is results[2].verdict
    assert [result.deduplicated for result in results] == [False, False, True, False]
    assert results[0].verdict.proved and results[1].verdict.proved
    assert not results[3].verdict.proved


def test_missing_fork_falls_back_to_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proof_parallel, "_fork_context", lambda: None)
    monkeypatch.setattr(proof_parallel, "check_valid_and_equivalent", _fake_checker)

    results = run_local_proof_batch(
        [_obligation("first-key", "first"), _obligation("second-key", "second")],
        workers=4,
    )

    assert [result.sequence for result in results] == [0, 1]
    assert all(result.verdict.proved for result in results)
    # Without fork, both proofs run in this process.
    assert {result.verdict.detail.split(":")[1] for result in results} == {
        str(os.getpid())
    }


@requires_fork
def test_parallel_matches_serial_order_and_dedup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proof_parallel, "check_valid_and_equivalent", _fake_checker)
    obligations = [
        _obligation("slow-key", "slow"),
        _obligation("fast-key", "fast"),
        _obligation("slow-key", "duplicate-never-called"),
        _obligation("medium-key", "medium"),
    ]

    serial = run_local_proof_batch(obligations, workers=1)
    parallel = run_local_proof_batch(obligations, workers=3)

    assert [result.sequence for result in parallel] == [0, 1, 2, 3]
    assert [result.verdict.proved for result in parallel] == [
        result.verdict.proved for result in serial
    ]
    assert [result.verdict.detail.split(":")[0] for result in parallel] == [
        result.verdict.detail.split(":")[0] for result in serial
    ]
    assert parallel[0].verdict == parallel[2].verdict
    assert parallel[2].deduplicated
    assert any(
        result.verdict.detail.split(":")[1] != str(os.getpid()) for result in parallel
    )


@requires_fork
def test_parallel_batch_uses_multiple_worker_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def checker(
        current: SymTensor,
        candidate: SymTensor,
        *,
        timeout: int,
        preconditions: list[Any],
    ) -> EquivalenceVerdict:
        del candidate, timeout, preconditions
        # Hold the worker briefly so the pool must fan out.
        time.sleep(0.1)
        return EquivalenceVerdict(
            proved=True,
            stage="proved",
            detail=f"{current.id}:{os.getpid()}:{gc.isenabled()}",
        )

    monkeypatch.setattr(proof_parallel, "check_valid_and_equivalent", checker)

    results = run_local_proof_batch(
        [
            _obligation(f"key-{index}", f"proof-{index}", timeout=10000)
            for index in range(4)
        ],
        workers=4,
    )

    details = [result.verdict.detail.rsplit(":", 2) for result in results]
    worker_pids = {int(parts[1]) for parts in details}
    assert os.getpid() not in worker_pids
    assert len(worker_pids) > 1
    # Fork workers must not finalize inherited Z3 objects during a proof.
    assert {parts[2] for parts in details} == {"False"}


@requires_fork
def test_parallel_checker_exception_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def checker(
        current: SymTensor,
        candidate: SymTensor,
        *,
        timeout: int,
        preconditions: list[Any],
    ) -> EquivalenceVerdict:
        del current, candidate, timeout, preconditions
        raise RuntimeError("proof checker failed")

    monkeypatch.setattr(proof_parallel, "check_valid_and_equivalent", checker)

    with pytest.raises(RuntimeError, match="proof checker failed"):
        run_local_proof_batch(
            [_obligation("fail-key", "fail"), _obligation("other-key", "other")],
            workers=2,
        )


@requires_fork
def test_worker_process_death_raises_broken_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def dying_checker(
        current: SymTensor,
        candidate: SymTensor,
        *,
        timeout: int,
        preconditions: list[Any],
    ) -> EquivalenceVerdict:
        del current, candidate, timeout, preconditions
        os._exit(9)

    monkeypatch.setattr(proof_parallel, "check_valid_and_equivalent", dying_checker)

    with pytest.raises(BrokenProcessPool):
        run_local_proof_batch(
            [_obligation("die-key", "die"), _obligation("other-key", "other")],
            workers=2,
        )


@requires_fork
def test_parallel_batch_stops_promptly_at_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def slow_checker(
        current: SymTensor,
        candidate: SymTensor,
        *,
        timeout: int,
        preconditions: list[Any],
    ) -> EquivalenceVerdict:
        del current, candidate, timeout, preconditions
        time.sleep(2.0)
        return EquivalenceVerdict(proved=True, stage="proved", detail="")

    monkeypatch.setattr(proof_parallel, "check_valid_and_equivalent", slow_checker)
    obligations = [_obligation(f"key-{index}", "slow") for index in range(8)]
    started = time.monotonic()
    with pytest.raises(proof_parallel.ProofBatchDeadlineExceeded):
        run_local_proof_batch(
            obligations,
            workers=2,
            deadline=time.monotonic() + 0.1,
        )
    assert time.monotonic() - started < 1.0


def test_rejected_duplicate_preserves_sequence_and_dedup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def checker(
        current: SymTensor,
        candidate: SymTensor,
        *,
        timeout: int,
        preconditions: list[Any],
    ) -> EquivalenceVerdict:
        del current, candidate, timeout, preconditions
        return EquivalenceVerdict(proved=False, stage="value", detail="unknown")

    monkeypatch.setattr(proof_parallel, "check_valid_and_equivalent", checker)
    term = SymTensor("same", rank=1)
    results = run_local_proof_batch(
        [
            LocalProofObligation("same-key", term, term, timeout=50),
            LocalProofObligation("same-key", term, term, timeout=50),
        ],
        workers=1,
    )

    assert [result.sequence for result in results] == [0, 1]
    assert not any(result.verdict.proved for result in results)
    assert results[0].verdict.detail == results[1].verdict.detail == "unknown"
    assert not results[0].deduplicated
    assert results[1].deduplicated


@requires_fork
def test_deadline_returns_completed_results_with_effective_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def checker(
        current: SymTensor,
        candidate: SymTensor,
        *,
        timeout: int,
        preconditions: list[Any],
    ) -> EquivalenceVerdict:
        del candidate, preconditions
        if current.id == "slow":
            time.sleep(2.0)
        return EquivalenceVerdict(
            proved=True,
            stage="proved",
            detail=str(timeout),
        )

    monkeypatch.setattr(proof_parallel, "check_valid_and_equivalent", checker)
    fast = LocalProofObligation(
        "fast-key",
        SymTensor("fast", rank=1),
        SymTensor("fast", rank=1),
        timeout=10000,
    )
    slow = LocalProofObligation(
        "slow-key",
        SymTensor("slow", rank=1),
        SymTensor("slow", rank=1),
        timeout=10000,
    )
    with pytest.raises(proof_parallel.ProofBatchDeadlineExceeded) as raised:
        run_local_proof_batch(
            [fast, slow],
            workers=2,
            deadline=time.monotonic() + 0.2,
        )
    assert [result.proof_key for result in raised.value.completed] == ["fast-key"]
    # The worker caps the solver timeout by the remaining batch deadline.
    assert 1 <= raised.value.completed[0].timeout <= 200
    assert raised.value.completed[0].verdict.detail == str(
        raised.value.completed[0].timeout
    )


def test_mixed_timeout_batch_deduplication_is_timeout_qualified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def checker(
        current: SymTensor,
        candidate: SymTensor,
        *,
        timeout: int,
        preconditions: list[Any],
    ) -> EquivalenceVerdict:
        del current, candidate, preconditions
        return EquivalenceVerdict(proved=False, stage="value", detail=str(timeout))

    monkeypatch.setattr(proof_parallel, "check_valid_and_equivalent", checker)
    term = SymTensor("same", rank=1)
    results = run_local_proof_batch(
        [
            LocalProofObligation("same-key", term, term, timeout=100),
            LocalProofObligation("same-key", term, term, timeout=200),
        ],
        workers=1,
    )

    assert [result.verdict.detail for result in results] == ["100", "200"]
    assert not any(result.deduplicated for result in results)


def test_serial_deadline_stops_between_obligations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def slow_checker(
        current: SymTensor,
        candidate: SymTensor,
        *,
        timeout: int,
        preconditions: list[Any],
    ) -> EquivalenceVerdict:
        del current, candidate, timeout, preconditions
        time.sleep(0.2)
        return EquivalenceVerdict(proved=True, stage="proved", detail="")

    monkeypatch.setattr(proof_parallel, "check_valid_and_equivalent", slow_checker)
    obligations = [_obligation(f"key-{index}", f"slow-{index}") for index in range(4)]
    started = time.monotonic()
    with pytest.raises(proof_parallel.ProofBatchDeadlineExceeded) as raised:
        run_local_proof_batch(
            obligations,
            workers=1,
            deadline=time.monotonic() + 0.1,
        )
    assert time.monotonic() - started < 1.0
    assert len(raised.value.completed) < len(obligations)


@requires_fork
def test_completion_iterator_reports_fast_result_before_slow_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def checker(
        current: SymTensor,
        candidate: SymTensor,
        *,
        timeout: int,
        preconditions: list[Any],
    ) -> EquivalenceVerdict:
        del candidate, timeout, preconditions
        if current.id == "slow":
            time.sleep(0.4)
        return EquivalenceVerdict(proved=True, stage="proved", detail=current.id)

    monkeypatch.setattr(proof_parallel, "check_valid_and_equivalent", checker)
    started = time.monotonic()
    completions: list[tuple[int, float]] = []
    results = run_local_proof_batch(
        [_obligation("slow-key", "slow"), _obligation("fast-key", "fast")],
        workers=2,
        on_result=lambda result: completions.append(
            (result.sequence, time.monotonic() - started)
        ),
    )

    assert [result.sequence for result in results] == [0, 1]
    assert completions[0][0] == 1
    assert completions[0][1] < 0.2


@pytest.mark.parametrize("timeout,expected", [(100, 50), (500, 250), (3000, 750)])
def test_fallback_budget_splits_exactly(timeout: int, expected: int) -> None:
    assert _fallback_timeout(timeout) == expected
