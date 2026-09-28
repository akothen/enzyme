"""Host-only unit tests for the pure sharding layer.

No device, no codegen. Exercises axis_classes, op_rows, set-valued combine,
single-segment shardings(G), and the offline per-monoid law discharge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from axon.ir import build_graph_from_kernel
from axon.sharding import (
    AllGather,
    AllReduce,
    Identity,
    Partial,
    Replicated,
    Sharded,
    ShardingPlan,
    Slice,
    axis_classes,
    combine,
    shardings,
)

_KERNELS = Path(__file__).resolve().parents[1] / "kernels"


def _spec(name: str):
    # Load through the real CLI loader (kernels are packages: kernels/<name>/).
    from axon.cli import _load_spec

    return _load_spec(str(_KERNELS / name))


def _graph(name: str, **sizes):
    spec = _spec(name)
    return build_graph_from_kernel(spec.axon_kernel, *spec.input_specs, dim_sizes=sizes)


# --- axis classes -----------------------------------------------------------


def test_axis_classes_matmul_contraction_ends_at_output():
    G = _graph("matmul", m=64, n=32, k=128)
    classes = axis_classes(G)
    # x:(m,k), w:(k,n), out:(m,n). m unifies x.0~out.0; n unifies w.1~out.1;
    # k unifies x.1~w.0 and ends at the matmul output (not in out).
    out_id = G.output_ids[0]
    assert classes[("x", 0)] == classes[(out_id, 0)]  # m
    assert classes[("w", 1)] == classes[(out_id, 1)]  # n
    assert classes[("x", 1)] == classes[("w", 0)]  # k contraction
    # k's class is NOT among the output's dim classes.
    out_classes = {classes[(out_id, d)] for d in range(2)}
    assert classes[("x", 1)] not in out_classes


def test_axis_classes_rmsnorm_reduced_axis_distinct():
    G = _graph("rmsnorm", m=64, n=128)
    classes = axis_classes(G)
    # x:(m,n); the reduced n (axis 1) does not survive into the (m,1) reduce out.
    rid = next(n.id for n in G.nodes if n.op == "reduce_sum")
    # reduce out is (m,1): axis-1 is size-1, not tied to x.1.
    assert classes[("x", 1)] != classes.get((rid, 1), object())


# --- combine table (set-valued, §3.4) ---------------------------------------


def test_combine_canonical_singletons():
    assert combine(Replicated(), Replicated()) == frozenset({Identity()})
    assert combine(Replicated(), Sharded("a")) != frozenset()
    assert len(combine(Sharded("a"), Replicated())) == 1  # all-gather, canonical
    assert combine(Partial("+"), Replicated()) == frozenset({AllReduce("+")})
    assert combine(Sharded("a"), Sharded("a")) == frozenset({Identity()})


def test_combine_illegal_is_empty():
    # P(M)→S(reduced) and any R→P / S→P / P(M)→P(M') is deny-by-default.
    assert combine(Replicated(), Partial("+")) == frozenset()
    assert combine(Sharded("a"), Partial("+")) == frozenset()
    assert combine(Partial("+"), Partial("max")) == frozenset()


def test_combine_noncanonical_reshard_forks():
    # S(a)→S(b), a≠b is non-canonical: one combine per orientation (2 at LNC=2).
    s = combine(Sharded("a"), Sharded("b"))
    assert len(s) == 2
    # P(M)→S(b) reduce-scatter likewise.
    assert len(combine(Partial("+"), Sharded("b"))) == 2


# --- shardings(G) single-segment --------------------------------------------


def _plans(name, **sizes) -> list[ShardingPlan]:
    return list(shardings(_graph(name, **sizes)))


def test_matmul_shardings_cover_p_c_and_replicate():
    plans = _plans("matmul", m=64, n=32, k=128)
    tags = {p.tag() for p in plans}
    # Expect: replicate (lnc1), P on m, P on n (both barrier), C on k (all-reduce+).
    # The trailing `f` marks a finalized (complete-kernel) plan.
    assert "lnc1" in tags
    assert any(":P" in t and "P+" not in t for t in tags)  # free-axis shard (P)
    assert any("P+" in t for t in tags)  # contraction shard (C, all-reduce +)


def test_matmul_C_plan_has_allreduce_at_output():
    plans = _plans("matmul", m=64, n=32, k=128)
    c_plans = [
        p for p in plans if any(isinstance(c, AllReduce) for c in p.combines.values())
    ]
    assert c_plans, "expected at least one C (all-reduce) plan"
    for p in c_plans:
        # The output edge carries P(+) and the combine is AllReduce(+).
        ar = [c for c in p.combines.values() if isinstance(c, AllReduce)]
        assert all(c.monoid == "+" for c in ar)


def test_matmul_P_plan_has_barrier_only():
    plans = _plans("matmul", m=64, n=32, k=128)
    p_plans = [
        p
        for p in plans
        if p.shard_class is not None
        and not any(isinstance(c, AllReduce) for c in p.combines.values())
    ]
    assert p_plans
    for p in p_plans:
        # A P-plan may carry R→S Slice (input DMA offset) and an S→S barrier /
        # S→R gather, but never an arithmetic all-reduce.
        assert all(
            isinstance(c, (Identity, AllGather, Slice)) for c in p.combines.values()
        )


def test_monochromatic_plans_have_at_most_one_active_class():
    for name, sizes in [
        ("matmul", {"m": 64, "n": 32, "k": 128}),
        ("qkv_cte", {"m": 64, "n": 32, "k": 128}),
        ("rmsnorm", {"m": 64, "n": 128}),
    ]:
        G = _graph(name, **sizes)
        for p in shardings(G, max_colors=1):
            classes = {
                lbl.axis_class for lbl in p.labels.values() if isinstance(lbl, Sharded)
            }
            assert len(classes) <= 1, f"{name}: {classes} active in one plan"


def test_replicate_plan_always_present():
    for name, sizes in [
        ("matmul", {"m": 64, "n": 32, "k": 128}),
        ("qkv_cte", {"m": 64, "n": 32, "k": 128}),
    ]:
        assert any(p.shard_class is None for p in _plans(name, **sizes))


def test_qkv_cte_has_P_on_token_axis():
    # qkv_cte is naturally P-on-m: m (tokens) is the matmul free/output axis,
    # so each core runs the whole residual→RMSNorm→projection on half the rows
    # with only a barrier.
    plans = _plans("qkv_cte", m=64, n=32, k=128)
    assert any(
        p.shard_class is not None
        and not any(isinstance(c, AllReduce) for c in p.combines.values())
        for p in plans
    )


# --- duplicate-free assertion (§3.8 regression guard) -----------------------


def test_monochromatic_plans_are_duplicate_free():
    from axon.sharding import _plan_canonical_key

    for name, sizes in [
        ("matmul", {"m": 1024, "n": 1024, "k": 16384}),
        ("rmsnorm", {"m": 128, "n": 8192}),
    ]:
        plans = list(shardings(_graph(name, **sizes), max_colors=1))
        keys = [_plan_canonical_key(p) for p in plans]
        assert len(keys) == len(set(keys)), (
            f"{name}: monochromatic plans have duplicates "
            f"({len(plans)} plans, {len(set(keys))} distinct)"
        )


def test_matmul_monochromatic_count_equals_admissible_plus_one():
    plans = list(shardings(_graph("matmul", m=64, n=32, k=128), max_colors=1))
    # matmul has 3 classes (m, n, k), all admissible, so 3+1=4 plans.
    assert len(plans) == 4, f"expected 4, got {len(plans)}"


# Offline monoid-law discharge lives in tests/test_sharding_laws.py (it is a
# test obligation, not a pipeline stage).


# --- cost model ordering (M5 prune) -----------------------------------------


def test_cost_ranks_C_over_P_for_large_k_and_P_over_C_for_large_m():
    from axon.sharding import AllReduce
    from axon.sharding_cost import prune_plans

    def _first(name, **sizes):
        G = _graph(name, **sizes)
        kept = prune_plans(list(shardings(G)), G, sizes)
        assert kept, "prune kept nothing"
        return kept[0], G

    # large-k: the contraction (k) shard, an all-reduce plan, ranks first.
    top_k, _ = _first("matmul", m=1024, n=1024, k=16384)
    assert any(isinstance(c, AllReduce) for c in top_k.combines.values()), (
        f"large-k top plan should be a C/all-reduce, got {top_k.tag()}"
    )
    # large-m: the free-axis (m) shard, barrier only, ranks first.
    top_m, _ = _first("matmul", m=16384, n=1024, k=1024)
    assert not any(isinstance(c, AllReduce) for c in top_m.combines.values()), (
        f"large-m top plan should be a P/barrier shard, got {top_m.tag()}"
    )


# --- flash seam (M7, stubbed) -----------------------------------------------


def test_flash_excluded_from_search_visible_monoids():
    # flash must NOT be in the search-visible library, so PROPAGATE's `if M in
    # MONOIDS` check never admits a P(flash) label (a softmax over a sharded axis
    # is not shardable yet).
    from axon.sharding_monoids import MONOIDS

    assert "flash" not in MONOIDS


def test_flash_body_synthesis_and_combine_are_stubbed():
    from axon.sharding_monoids import FLASH

    assert FLASH.combine_kind == "flash"
    assert FLASH.body_synthesis is not None
    with pytest.raises(NotImplementedError):
        FLASH.body_synthesis()
