"""Pure graph-structure analysis and tile-config extraction for codegen."""

from __future__ import annotations

import builtins

from axon.codegen.constants import (
    DEFAULT_ELEMENTWISE_TILE,
    DEFAULT_MATMUL_TILE,
    DEFAULT_TILE_K,
)
from axon.ir import Node, TileAnnotation


def extract_elementwise_tile_configs(
    compute_nodes: list[Node],
    tile_annotations: dict[str, TileAnnotation],
) -> list[dict[str, int]]:
    for node in compute_nodes:
        ann = tile_annotations.get(node.id)
        if ann is None or ann.hardware_metadata is None:
            continue
        cands = ann.hardware_metadata.concrete_tile_candidates
        if len(cands) < 2:
            continue
        tile_m_candidates = cands[0]
        tile_n_candidates = cands[1]
        configs: list[dict[str, int]] = []
        for tm in tile_m_candidates:
            for tn in tile_n_candidates:
                configs.append({"tile_m": int(tm), "tile_n": int(tn)})
        if configs:
            return configs
    return [dict(DEFAULT_ELEMENTWISE_TILE)]


def extract_matmul_tile_config(
    compute_nodes: list[Node],
    tile_annotations: dict[str, TileAnnotation],
) -> dict[str, int]:
    default = dict(DEFAULT_MATMUL_TILE)
    mm_node = next((n for n in compute_nodes if n.op == "nc_matmul"), None)
    if mm_node is None:
        return default
    ann = tile_annotations.get(mm_node.id)
    if ann is None or ann.hardware_metadata is None:
        return default
    meta = ann.hardware_metadata
    cands = meta.concrete_tile_candidates
    if len(cands) < 2 or not cands[0] or not cands[1]:
        return default
    tile_m = int(cands[0][0])
    tile_n = int(cands[1][0])
    tile_k = DEFAULT_TILE_K  # hardware constant for nc_matmul partition dim
    if meta.operand_tile_shapes:
        stationary_shape = meta.operand_tile_shapes[0][1]
        if stationary_shape:
            tile_k = int(stationary_shape[0])
    return {"tile_m": tile_m, "tile_k": tile_k, "tile_n": tile_n}


def classify_kernel(compute_nodes: list[Node]) -> str:
    ops = {n.op for n in compute_nodes}
    if "nc_matmul" in ops:
        return "matmul"
    # A cumulative scan threads a carry along the free axis (sequential loop),
    # a structurally different body from elementwise.
    if "tensor_scalar_cumulative" in ops:
        return "scan"
    # A free-axis reduce (to width 1) routes to the reduce body. tensor_reduce
    # is the incumbent; activation_reduce is a fused (pre-activation)+(free-axis
    # reduce) with the same (P, 1) reduce-output shape, so it uses the same body.
    # Either op alone is sufficient; tensor_partition_reduce (partition-axis
    # reduce) is a distinct body and stays "other".
    if "tensor_reduce" in ops or "activation_reduce" in ops:
        return "reduce"
    if "tensor_partition_reduce" in ops:
        return "other"
    return "elementwise"


def has_rhs_nc_transpose(
    compute_nodes: list[Node],
    id_to_node: dict[str, Node],
) -> bool:
    for node in compute_nodes:
        if node.op == "nc_matmul" and len(node.inputs) >= 2:
            moving_node = id_to_node.get(node.inputs[1])
            if moving_node is not None and moving_node.op == "nc_transpose":
                return True
    return False


def has_lhs_nc_transpose(
    compute_nodes: list[Node],
    id_to_node: dict[str, Node],
) -> bool:
    for node in compute_nodes:
        if node.op == "nc_matmul" and len(node.inputs) >= 1:
            stationary_node = id_to_node.get(node.inputs[0])
            if stationary_node is not None and stationary_node.op == "nc_transpose":
                return True
    return False


def find_post_matmul_nodes(
    compute_nodes: list[Node],
    mm_node: Node,
    pre_transpose_node_ids: set[str],
    input_node_ids: set[str],
) -> list[Node]:
    downstream_ids: set[str] = {mm_node.id}
    post_nodes: list[Node] = []
    for n in compute_nodes:
        if n.id in {mm_node.id} | pre_transpose_node_ids | input_node_ids:
            continue
        if any(inp_id in downstream_ids for inp_id in n.inputs):
            downstream_ids.add(n.id)
            post_nodes.append(n)
    return post_nodes


def collect_reduce_input_chain(
    inp_id: str,
    input_node_ids: set[str],
    id_to_node: dict[str, Node],
) -> tuple[list[Node], list[str]]:
    chain: list[Node] = []
    hbm_seen: dict[str, None] = {}  # ordered-set via dict
    visited: set[str] = set()

    def _visit(nid: str) -> None:
        if nid in visited:
            return
        visited.add(nid)
        if nid in input_node_ids:
            hbm_seen[nid] = None
            return
        n = id_to_node.get(nid)
        if n is None:
            return
        for child_id in n.inputs:
            _visit(child_id)
        chain.append(n)

    _visit(inp_id)
    return chain, list(hbm_seen)


def matmul_has_chained(
    compute_nodes: list[Node],
    id_to_node: dict[str, Node],
    input_node_ids: set[str],
) -> bool:
    """Whether the matmul graph carries a chained second matmul (so the emitted
    signature gains a ``TILES_IN_BLOCK_P`` tile param). A non-matmul or malformed
    graph (the only raise surface is ``IndexError``) reports ``False``."""
    try:
        (_lhs, _rhs, pre_trans, post_trans, _nc, mm_node) = find_matmul_structure(
            compute_nodes, id_to_node, input_node_ids
        )
        pre_post_ids = {n.id for n in pre_trans} | {n.id for n in post_trans}
        post_mm = find_post_matmul_nodes(
            compute_nodes, mm_node, pre_post_ids, input_node_ids
        )
        return find_chained_mm_in_post(post_mm, id_to_node, input_node_ids) is not None
    except IndexError:
        return False


def find_chained_mm_in_post(
    post_matmul_nodes: list[Node],
    id_to_node: dict[str, Node],
    input_node_ids: set[str],
) -> tuple[list[Node], Node, str, bool] | None:
    chained_mm: Node | None = None
    for n in post_matmul_nodes:
        if n.op == "nc_matmul":
            chained_mm = n
            break
    if chained_mm is None:
        return None

    stat_id = chained_mm.inputs[0]
    chained_nc_trans = id_to_node.get(stat_id)

    excluded_ids: set[str] = {chained_mm.id}
    chained_has_lhs_nc_trans = (
        chained_nc_trans is not None and chained_nc_trans.op == "nc_transpose"
    )
    if chained_has_lhs_nc_trans and chained_nc_trans is not None:
        excluded_ids.add(chained_nc_trans.id)

    truncated = [n for n in post_matmul_nodes if n.id not in excluded_ids]

    rhs_id = chained_mm.inputs[1]
    curr = rhs_id
    while curr not in input_node_ids:
        n2 = id_to_node.get(curr)
        if n2 is None or not n2.inputs:
            break
        curr = n2.inputs[0]
    chained_rhs_input_id = curr

    return truncated, chained_mm, chained_rhs_input_id, chained_has_lhs_nc_trans


def find_matmul_structure(
    compute_nodes: list[Node],
    id_to_node: dict[str, Node],
    input_node_ids: set[str],
) -> tuple[str, str, list[Node], list[Node], Node | None, Node]:
    mm_candidates = [n for n in compute_nodes if n.op == "nc_matmul"]

    def _mm_score(mm: Node) -> tuple[int, int]:
        downstream: set[str] = {mm.id}
        for n in compute_nodes:
            if n.id in downstream:
                continue
            if any(inp in downstream for inp in n.inputs):
                downstream.add(n.id)
        n_downstream = len(downstream) - 1  # exclude mm itself
        depth = 0
        curr = mm.inputs[0]
        seen: set[str] = set()
        while curr not in input_node_ids and curr not in seen:
            seen.add(curr)
            nd = id_to_node.get(curr)
            if nd is None or not nd.inputs:
                break
            depth += 1
            curr = nd.inputs[0]
        return (n_downstream, depth)

    mm_node = sorted(mm_candidates, key=_mm_score, reverse=True)[0]
    stationary_id = mm_node.inputs[0]
    moving_id = mm_node.inputs[1]

    lhs_chain: list[Node] = []
    curr_id = stationary_id
    visited: set[str] = set()
    while curr_id not in input_node_ids and curr_id not in visited:
        visited.add(curr_id)
        n = id_to_node.get(curr_id)
        if n is None or not n.inputs:
            break
        lhs_chain.append(n)
        curr_id = n.inputs[0]
    lhs_chain.reverse()  # restore topological order (oldest first)
    lhs_input_id = curr_id

    for i_sc, sc_n in enumerate(lhs_chain):
        if sc_n.op != "tensor_scalar":
            continue
        op0_idx = sc_n.attrs.get("operand0_input_index")
        if op0_idx is None or op0_idx >= len(sc_n.inputs):
            continue
        data_id = sc_n.inputs[0]
        operand0_id = sc_n.inputs[op0_idx]
        if operand0_id not in input_node_ids:
            continue
        if data_id in input_node_ids:
            continue
        data_node = id_to_node.get(data_id)
        if data_node is None or data_node.op != "tensor_reduce":
            continue
        lhs_input_id = operand0_id
        lhs_chain = lhs_chain[i_sc:]  # keep tensor_scalar and anything after
        break

    nc_transpose_idx: int | None = None
    for i, n in enumerate(lhs_chain):
        if n.op == "nc_transpose":
            nc_transpose_idx = i
    if nc_transpose_idx is not None:
        pre_transpose_nodes = lhs_chain[:nc_transpose_idx]
        nc_transpose_node: Node | None = lhs_chain[nc_transpose_idx]
        post_transpose_nodes = lhs_chain[nc_transpose_idx + 1 :]
    else:
        pre_transpose_nodes = []
        nc_transpose_node = None
        post_transpose_nodes = []

    rhs_input_id = moving_id
    curr_id = moving_id
    visited_rhs: set[str] = set()
    while curr_id not in input_node_ids and curr_id not in visited_rhs:
        visited_rhs.add(curr_id)
        n = id_to_node.get(curr_id)
        if n is None or not n.inputs:
            break
        curr_id = n.inputs[0]
    rhs_input_id = curr_id

    return (
        lhs_input_id,
        rhs_input_id,
        pre_transpose_nodes,
        post_transpose_nodes,
        nc_transpose_node,
        mm_node,
    )


def analyze_reduce_structure(
    compute_nodes: list[Node],
    input_nodes: list[Node],
    id_to_node: dict[str, Node],
) -> dict | None:
    input_ids = {n.id for n in input_nodes}

    # The reducing node is the single free-axis reduce (to width 1): the
    # incumbent tensor_reduce, or a fused activation_reduce (its square/etc. is
    # folded in, so its pre-reduce chain is empty). Both share the (P, 1) shape
    # and downstream broadcast, so the body treats them identically. Two such
    # nodes are ambiguous and fall back to the flat body.
    reduce_nodes = [
        n for n in compute_nodes if n.op in ("tensor_reduce", "activation_reduce")
    ]
    if len(reduce_nodes) != 1:
        return None
    reduce_node = reduce_nodes[0]

    pre_reduce_ancestor_ids: set[str] = set()

    def _collect_ancestors(nid: str) -> None:
        if nid in input_ids or nid in pre_reduce_ancestor_ids:
            return
        node = id_to_node.get(nid)
        if node is None or node.op == "input":
            return
        pre_reduce_ancestor_ids.add(nid)
        for inp in node.inputs or []:
            _collect_ancestors(inp)

    for inp_id in reduce_node.inputs or []:
        _collect_ancestors(inp_id)

    pre_reduce_chain = [n for n in compute_nodes if n.id in pre_reduce_ancestor_ids]

    reduce_descendant_ids: set[str] = {reduce_node.id}
    finalize_chain: list[Node] = []
    apply_node: Node | None = None
    post_apply_chain: list[Node] = []

    after_reduce = False
    for node in compute_nodes:
        if node.id == reduce_node.id:
            after_reduce = True
            continue
        if not after_reduce:
            continue

        if apply_node is not None:
            post_apply_chain.append(node)
        else:
            if builtins.all(i in reduce_descendant_ids for i in (node.inputs or [])):
                finalize_chain.append(node)
                reduce_descendant_ids.add(node.id)
            else:
                apply_node = node

    if apply_node is None:
        return None

    apply_tile_input_id: str | None = None
    apply_scalar_input_id: str | None = None
    for inp_id in apply_node.inputs or []:
        if inp_id in reduce_descendant_ids:
            apply_scalar_input_id = inp_id
        else:
            apply_tile_input_id = inp_id

    if apply_tile_input_id is None:
        return None

    needs_pre_buf = apply_tile_input_id in pre_reduce_ancestor_ids

    return {
        "pre_reduce_chain": pre_reduce_chain,
        "reduce_node": reduce_node,
        "finalize_chain": finalize_chain,
        "apply_node": apply_node,
        "apply_tile_input_id": apply_tile_input_id,
        "apply_scalar_input_id": apply_scalar_input_id,
        "post_apply_chain": post_apply_chain,
        "needs_pre_buf": needs_pre_buf,
    }


def _free_dim_is_one(node: Node) -> bool:
    """True when a node produces a per-partition ``(P, 1)`` scalar (its free axis
    is width 1), e.g. a reduction result or a scalar derived only from one."""
    shp = node.attrs.get("shape", node.shape) or ()
    return len(shp) == 2 and shp[1] == 1


def analyze_nreduce_structure(
    compute_nodes: list[Node],
    input_nodes: list[Node],
    id_to_node: dict[str, Node],
) -> dict | None:
    """Recognize a dependent chain of N>=2 free-axis reductions (e.g. LayerNorm:
    the mean's sum, then the variance's sum, which needs the mean) and return a
    per-stage plan the n-reduce body lowers as one hidden-axis sweep per
    reduction plus a final apply sweep. Returns None if the graph is not this
    shape, so the caller can fall back.

    Vocabulary:
      * *scalar* node -- produces a ``(P, 1)`` per-row value (a reduce, or an op
        derived only from scalars). *wide* node -- everything else (full free
        axis), computed from inputs + earlier scalars.
      * each *stage* is one reduction: the wide ``pre_reduce_chain`` that builds
        its reduce input from the input tiles + earlier stages' finalized
        scalars, the reduce itself, and a scalar ``finalize_chain`` (e.g.
        ``* 1/n``, ``+ eps``, ``sqrt``, ``reciprocal``) ending in the stage's
        scalar. A final ``output_chain`` builds the result from the input tiles +
        all finalized scalars.
    """
    input_ids = {n.id for n in input_nodes}
    reduce_ops = ("tensor_reduce", "activation_reduce")
    reduce_nodes = [n for n in compute_nodes if n.op in reduce_ops]
    if len(reduce_nodes) < 2:
        return None

    def _is_scalar(nid: str) -> bool:
        n = id_to_node.get(nid)
        if n is None:
            return False
        return n.op in reduce_ops or _free_dim_is_one(n)

    # Wide ancestor closure of `targets`, in topological order: the wide compute
    # nodes needed to build those tensors from inputs + scalars. Recursion stops
    # at inputs and at scalar nodes (those are read from persistent buffers); the
    # scalar ids reached are returned as dependencies to bind to stage scalars.
    def _wide_closure(targets: list[str]) -> tuple[list[Node], set[str]]:
        ordered: list[Node] = []
        seen: set[str] = set()
        scalar_deps: set[str] = set()

        def _visit(nid: str) -> None:
            if nid in input_ids:
                return
            if _is_scalar(nid):
                scalar_deps.add(nid)
                return
            if nid in seen:
                return
            seen.add(nid)
            n = id_to_node.get(nid)
            if n is None:
                return
            for inp in n.inputs or []:
                _visit(inp)
            ordered.append(n)

        for t in targets:
            _visit(t)
        return ordered, scalar_deps

    accounted: set[str] = set()
    stages: list[dict] = []
    scalar_stage: dict[str, int] = {}  # stage-endpoint scalar id -> stage index
    referenced_scalars: set[str] = set()

    for k, reduce_node in enumerate(reduce_nodes):
        pre_reduce_chain, pre_deps = _wide_closure(list(reduce_node.inputs or []))
        # A stage may only depend on the finalized scalars of *earlier* stages.
        for dep in pre_deps:
            if scalar_stage.get(dep, k) >= k:
                return None
        referenced_scalars |= pre_deps

        # finalize: the maximal chain of scalar nodes rooted at this reduce whose
        # inputs are all already inside the chain (so it is pure post-processing
        # of this reduction's accumulator).
        fin_set = {reduce_node.id}
        finalize_chain: list[Node] = []
        seen_reduce = False
        for n in compute_nodes:
            if n.id == reduce_node.id:
                seen_reduce = True
                continue
            if not seen_reduce or n.op in reduce_ops:
                continue
            if not _free_dim_is_one(n):
                continue
            if n.inputs and all(inp in fin_set for inp in n.inputs):
                finalize_chain.append(n)
                fin_set.add(n.id)

        scalar_id = finalize_chain[-1].id if finalize_chain else reduce_node.id
        scalar_stage[scalar_id] = k
        stages.append(
            {
                "reduce_node": reduce_node,
                "pre_reduce_chain": pre_reduce_chain,
                "finalize_chain": finalize_chain,
                "scalar_id": scalar_id,
            }
        )
        accounted.add(reduce_node.id)
        accounted.update(n.id for n in pre_reduce_chain)
        accounted.update(n.id for n in finalize_chain)

    if not compute_nodes:
        return None
    output_id = compute_nodes[-1].id
    output_chain, out_deps = _wide_closure([output_id])
    referenced_scalars |= out_deps
    accounted.update(n.id for n in output_chain)

    # Every scalar a wide node consumes must be some stage's finalized endpoint,
    # and every compute node must be accounted for -- otherwise the plan would
    # silently drop or misplace computation.
    if any(dep not in scalar_stage for dep in referenced_scalars):
        return None
    if accounted != {n.id for n in compute_nodes}:
        return None

    return {
        "stages": stages,
        "output_chain": output_chain,
        "output_id": output_id,
        "scalar_stage": scalar_stage,
    }
