"""Loop-nest derivation for the generic matmul emitter.

Block loops come from the output's dims (m outermost, then the final free
dim). Each node is placed at the shallowest block level binding all its
dims, EXCEPT dims that are some matmul's contraction: a node varying over a
contraction dim c of matmul M is placed inside M's accumulation loop
structure — concretely, its placement is its non-contraction block dims, and
c appears via M's accum loop. A value produced at one level and consumed at
another (the chained matmul reading the first stage's combine across all n
blocks) is materialized over the intervening block dims."""

from __future__ import annotations

from dataclasses import dataclass

from axon.codegen.layout import DimInfo, analyze_dims, topo_order


@dataclass(frozen=True)
class Nest:
    info: DimInfo
    block_loops: list[str]
    placement: dict[str, tuple[str, ...]]
    accum_loops: dict[str, str]
    materialized: dict[str, tuple[str, ...]]


def derive_nest(compute_nodes, id_to_node, input_node_ids, output_id) -> Nest:
    info = analyze_dims(compute_nodes, id_to_node, input_node_ids, output_id)

    def name(d) -> str:
        return info.canon[d.find()]

    out_dims = {name(d) for d in info.dims[output_id]}

    # Block loops: m outermost, then the output's other dims (p or n).
    block_loops = ["m"] + sorted(out_dims - {"m"})

    accum_loops = {mm_id: name(c) for mm_id, c in info.contractions}

    order = topo_order(compute_nodes, id_to_node)

    # A contraction dim is a *block-style* loop when it is ALSO a matmul's
    # output dim (the MLP's n = mm1's free dim, later contracted by mm3): the
    # first stage produces one result per block of it, so it iterates as a
    # hoisted block loop. A contraction that is only ever some matmul's own
    # accumulation (plain k) is NOT a block loop — it never appears in a
    # placement; a node feeding that accumulation is emitted at the matmul's
    # own block level instead.
    matmul_out_names: set[str] = set()
    for mm_id, _ in info.contractions:
        lay = info.layouts[mm_id]
        matmul_out_names.add(name(lay.part))
        if lay.free is not None:
            matmul_out_names.add(name(lay.free))

    block_style_dims = set(out_dims)
    for _, c in info.contractions:
        if name(c) in matmul_out_names:
            block_style_dims.add(name(c))

    # Global loop nesting order used to order every placement tuple: the
    # output's block loops first (m, then p/n), then any hoisted block-style
    # contraction dims (the MLP's n).
    loop_order = block_loops + sorted(block_style_dims - set(block_loops))

    # For each plain-accumulation dim, the block level of the matmul that owns
    # it: a node varying over that dim is emitted inside that matmul's accum
    # loop, so its open block loops are the matmul's own block dims.
    accum_to_block: dict[str, set[str]] = {}
    for mm_id, c in info.contractions:
        cn = name(c)
        if cn not in block_style_dims:  # a plain accumulation dim
            mm_dims = {name(d) for d in info.dims[mm_id]}
            accum_to_block[cn] = block_style_dims & mm_dims

    placement: dict[str, tuple[str, ...]] = {}
    for node in order:
        nd = {name(d) for d in info.dims[node.id]}
        place = set(block_style_dims & nd)
        for cn in nd:
            if cn in accum_to_block:
                # feeds a matmul's accumulation -> emitted at its block level
                place |= accum_to_block[cn]
        placement[node.id] = tuple(d for d in loop_order if d in place)

    # Materialization: node consumed by a consumer whose placement differs in
    # a dim the producer's placement contains but the consumer iterates as an
    # accumulation loop -> span that dim.
    consumers: dict[str, list[str]] = {}
    for n in order:
        for i in n.inputs or []:
            consumers.setdefault(i, []).append(n.id)
    materialized: dict[str, tuple[str, ...]] = {}
    for node in order:
        span: set[str] = set()
        for c_id in consumers.get(node.id, []):
            c_node = id_to_node[c_id]
            if c_node.op == "nc_matmul" and accum_loops[c_id] in placement[node.id]:
                # consumer accumulates over a dim the producer is placed at:
                # the producer's value for EVERY block of that dim must be live
                span.add(accum_loops[c_id])
        if span:
            materialized[node.id] = tuple(sorted(span))
    return Nest(
        info=info,
        block_loops=block_loops,
        placement=placement,
        accum_loops=accum_loops,
        materialized=materialized,
    )
