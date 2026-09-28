"""Plan/graph analysis that *chooses* which combine a `ShardingPlan` lowers to.

This is not codegen: it reads the `ShardingPlan`/`nuGraph` to decide which output
axis a shard maps to, which terminal combine the plan carries, and which operands
get sliced. It feeds `_shard_prelude` / `_combine_for` in the assembler, which
then drive the line-templates in `axon.codegen.combine_emit`. Reused verbatim
from the original SPMD assembler (`spmd.py`)."""

from __future__ import annotations

from axon.codegen.combine_emit import SpmdEmitError, _ShardAxis
from axon.ir import nuGraph
from axon.sharding import (
    OUTPUT_SINK,
    Combine,
    ShardingPlan,
    Slice,
    _inferred_sinks,
    axis_classes,
)

__all__ = [
    "SpmdEmitError",
    "_AXIS_LOOPS",
    "_operand_slices",
    "_output_id",
    "_output_ids",
    "_sharded_output_axis",
    "_terminal_combine",
]


# Output dim 0 (the partition/row axis) is the *outermost* block loop
# (`m`/`NUM_BLOCK_M`) in every body, so sharding it splits the whole per-row
# compute across programs. Dim 1 (`n`/`NUM_BLOCK_N`) is a loop *nested inside*
# `m` in the matmul body, so sharding it would leave the `m`-direction work
# fully replicated on both cores (no compute speedup); a real `n`-shard needs
# the body to hoist `n` outermost, which is deferred. So only axis 0 is a P
# free-axis shard here.
_AXIS_LOOPS = {0: _ShardAxis("m", "NUM_BLOCK_M")}


def _output_ids(G: nuGraph) -> tuple[str, ...]:
    return G.output_ids or _inferred_sinks(G.nodes)


def _output_id(G: nuGraph) -> str:
    return _output_ids(G)[0]


def _sharded_output_axis(plan: ShardingPlan, G: nuGraph) -> int | None:
    """Which output-tensor dim carries the plan's shard class (the free axis for
    a P shard), or None if the shard class is not present in the output (a
    contraction — its class ended at the matmul, so the combine is an
    all-reduce, not a block-loop shard)."""
    if plan.shard_class is None:
        return None
    classes = axis_classes(G)
    out_id = _output_id(G)
    out_node = next(n for n in G.nodes if n.id == out_id)
    for d in range(len(out_node.shape or ())):
        if classes.get((out_id, d)) == plan.shard_class:
            return d
    return None


def _terminal_combine(plan: ShardingPlan, G: nuGraph) -> Combine | None:
    return plan.combines.get((_output_id(G), OUTPUT_SINK))


def _operand_slices(plan: ShardingPlan, G: nuGraph) -> dict[str, int]:
    """For each kernel input the plan `Slice`s on the shard class, the input's
    sharded dim index. The body then runs on the `SHARD_ID`-offset half of that
    dim and computes the per-core partial."""
    classes = axis_classes(G)
    input_ids = set(G.input_ids)
    id_to_node = {n.id: n for n in G.nodes}
    sliced: dict[str, int] = {}
    for (producer_id, _consumer), c in plan.combines.items():
        if not isinstance(c, Slice) or producer_id not in input_ids:
            continue
        node = id_to_node[producer_id]
        for d in range(len(node.shape or ())):
            if classes.get((producer_id, d)) == c.axis_class:
                sliced[producer_id] = d
                break
    return sliced
