"""Synthesis identity allocation must not depend on preceding work."""

from __future__ import annotations

import threading

import pytest

import axon.isa_semantics as isa
from axon.ir import build_graph_from_kernel, graph_signature
from axon.isa_semantics import (
    _BODY_IDS,
    _NODE_IDS,
    GlobalCounter,
    StaleIdentityError,
    _start_kernel_synthesis_cache,
)


def _kernel(x, w):
    return (x * x) @ w


_SPECS = (("x", ("m", "kk")), ("w", ("kk", "n")))
_DIMS = {"m": 128, "kk": 64, "n": 32}


def _lift_ids() -> list[str]:
    graph = build_graph_from_kernel(_kernel, *_SPECS, dim_sizes=_DIMS)
    return [node.id for node in graph.nodes]


def test_kernel_boundary_resets_generated_ids():
    """The same lift after a kernel boundary allocates the same ids.

    The defect: a second lift produced mul_1003/matmul_1004, not mul_1001/matmul_1002.
    """
    _start_kernel_synthesis_cache()
    first = _lift_ids()

    _start_kernel_synthesis_cache()
    second = _lift_ids()

    assert first == second


def test_unrelated_work_between_boundaries_does_not_shift_ids():
    """Intervening synthesis of a different kernel cannot shift a later kernel."""
    _start_kernel_synthesis_cache()
    baseline = _lift_ids()

    _start_kernel_synthesis_cache()

    def _other(a, b):
        return (a + a + a) @ b

    build_graph_from_kernel(_other, *_SPECS, dim_sizes=_DIMS)
    for _ in range(37):
        _NODE_IDS.next_name("filler")

    _start_kernel_synthesis_cache()
    assert _lift_ids() == baseline


def test_graph_signature_is_stable_across_the_boundary():
    """Variant order sorts on `graph_signature`, which embeds raw node ids.

    An unstable signature permutes the variant_index in every emitted module name.
    """
    _start_kernel_synthesis_cache()
    first = graph_signature(build_graph_from_kernel(_kernel, *_SPECS, dim_sizes=_DIMS))

    _start_kernel_synthesis_cache()
    second = graph_signature(build_graph_from_kernel(_kernel, *_SPECS, dim_sizes=_DIMS))

    assert first == second


def test_reduction_body_ids_reset_at_the_boundary():
    """Reduction body ids are identity too, and reset with the node ids."""
    _start_kernel_synthesis_cache()
    first = [_BODY_IDS.next() for _ in range(4)]

    _start_kernel_synthesis_cache()
    second = [_BODY_IDS.next() for _ in range(4)]

    assert first == second


def test_ids_stay_unique_within_a_run_after_reset():
    """Resetting must not let one synthesis run reuse an id it already issued."""
    _start_kernel_synthesis_cache()
    issued = [_NODE_IDS.next_name("op") for _ in range(200)]
    issued += [str(_BODY_IDS.next()) for _ in range(200)]
    assert len(set(issued)) == len(issued)


def test_concurrent_allocation_issues_no_duplicates():
    """Allocation is called from synthesis worker threads, so it stays atomic."""
    _start_kernel_synthesis_cache()
    collected: list[list[str]] = []
    lock = threading.Lock()

    def worker() -> None:
        mine = [_NODE_IDS.next_name("op") for _ in range(250)]
        with lock:
            collected.append(mine)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    flat = [node_id for chunk in collected for node_id in chunk]
    assert len(flat) == 2000
    assert len(set(flat)) == 2000


def test_reset_returns_counters_to_their_declared_origins():
    """The reset targets the module's declared origins, not an arbitrary value."""
    for _ in range(5):
        _NODE_IDS.next_name("op")
        _BODY_IDS.next()

    _start_kernel_synthesis_cache()

    assert isa._NODE_IDS.value == isa._NODE_IDS.origin
    assert isa._BODY_IDS.value == isa._BODY_IDS.origin


# ---------------------------------------------------------------------------
# GlobalCounter.assert_current: catches ids mixed across a reset
# ---------------------------------------------------------------------------


def test_assert_current_accepts_ids_from_the_live_generation():
    counter = GlobalCounter(1000)
    names = [counter.next_name("op") for _ in range(3)]

    counter.assert_current(names, "ctx")


def test_assert_current_rejects_ids_the_live_generation_never_issued():
    """Guards the invariant that no pass allocates into an existing graph.

    Only ids past the live counter are detectable; a reissued name is identical.
    """
    counter = GlobalCounter(1000)
    counter.next_name("op")
    beyond_next_generation = counter.next_name("op")
    counter.reset()
    counter.next_name("op")

    with pytest.raises(StaleIdentityError, match="predate the last identity reset"):
        counter.assert_current([beyond_next_generation], "ctx")


def test_reset_bumps_the_generation():
    counter = GlobalCounter(1000)
    before = counter.generation

    counter.reset()

    assert counter.generation == before + 1


def test_pool_registry_order_is_pinned():
    """The template registry reproduces the recorded pool and fusion order."""
    from axon.synthesizer import ISA_POOL_OP_NAMES

    assert ISA_POOL_OP_NAMES == (
        "activation",
        "activation_reduce",
        "exponential",
        "nc_matmul",
        "reciprocal",
        "scalar_tensor_tensor",
        "tensor_partition_reduce",
        "tensor_reduce",
        "tensor_scalar",
        "tensor_scalar_cumulative",
        "tensor_tensor",
        "transpose",
    )
