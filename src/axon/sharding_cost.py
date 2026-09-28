"""Analytic prune: rank shardings by a cheap host-only cost model, keep top-K.

`cost(plan, G, dim_sizes)` ranks a plan by combine traffic vs sharded extent.
`prune_plans` drops the replicate plan (benched separately), drops tiny shard
axes, ranks by cost, and keeps the top-K for device bench."""

from __future__ import annotations

import math

from axon.ir import nuGraph
from axon.sharding import (
    AllGather,
    AllReduce,
    Identity,
    ReduceScatter,
    ShardingPlan,
    axis_classes,
)

_MIN_SHARD_EXTENT = 256
_DEFAULT_TOP_K = 10


def _class_extent(
    plan: ShardingPlan, G: nuGraph, dim_sizes: dict[str, int]
) -> int | None:
    """Concrete extent of the plan's active shard class (max over its members'
    concrete dims)."""
    if plan.shard_class is None:
        return None
    classes = axis_classes(G)
    id_to_node = {n.id: n for n in G.nodes}
    extent = 0
    for (nid, dim), cid in classes.items():
        if cid != plan.shard_class:
            continue
        node = id_to_node.get(nid)
        shape = (node.shape or ()) if node is not None else ()
        if dim < len(shape):
            extent = max(extent, int(shape[dim]))
    return extent or None


def _combine_traffic(plan: ShardingPlan, G: nuGraph) -> int:
    """Boundary elements moved by the plan's combines."""
    id_to_node = {n.id: n for n in G.nodes}
    traffic = 0
    for (producer_id, _consumer), c in plan.combines.items():
        node = id_to_node.get(producer_id)
        n_elems = math.prod(node.shape) if node is not None and node.shape else 0
        if isinstance(c, (AllReduce, AllGather, ReduceScatter)):
            traffic += n_elems
        elif isinstance(c, Identity) and c.barrier:
            pass
    return traffic


_EXTENT_WEIGHT = 1_000_000.0


def cost(plan: ShardingPlan, G: nuGraph, dim_sizes: dict[str, int]) -> float:
    """Lower is better. Rewards sharding the largest class; penalizes combine
    traffic as a tiebreaker."""
    if plan.shard_class is None:
        return 0.0
    extent = _class_extent(plan, G, dim_sizes) or 1
    traffic = _combine_traffic(plan, G)
    return traffic - _EXTENT_WEIGHT * extent


def prune_plans(
    plans: list[ShardingPlan],
    G: nuGraph,
    dim_sizes: dict[str, int],
    *,
    top_k: int = _DEFAULT_TOP_K,
) -> list[ShardingPlan]:
    """Keep top-K shard plans, dropping replicate and tiny-extent plans."""
    viable = [
        p
        for p in plans
        if p.shard_class is not None
        and (_class_extent(p, G, dim_sizes) or 0) >= _MIN_SHARD_EXTENT
    ]
    viable.sort(key=lambda p: cost(p, G, dim_sizes))
    return viable[:top_k]
