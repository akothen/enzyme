"""The reduce body builder (mean / rmsnorm / softmax-shaped reductions).

Recognizes a single tensor_reduce with an optional pre-reduce chain, a finalize
chain, an apply node, and a post-apply chain, or (via the n-reduce body) a
dependent chain of reductions such as LayerNorm's mean-then-variance. A reduce
graph that matches neither is refused with ``UnsupportedEmission`` rather than
emitted as the degenerate flat body, which would be numerically wrong.
"""

from __future__ import annotations

from axon.codegen.blocks import substitute_dst
from axon.codegen.context import EmitCtx
from axon.codegen.ops import (
    UnsupportedEmission,
    _nki_op_ref,
    emit_or_raise,
    nki_safe_var,
)
from axon.codegen.structure import (
    _free_dim_is_one,
    analyze_nreduce_structure,
    analyze_reduce_structure,
)
from axon.ir import Node


def emit_reduce_body(
    ctx: EmitCtx,
    input_nodes: list[Node],
    compute_nodes: list[Node],
    id_to_node: dict[str, Node],
    params: list[str],
) -> tuple[list[str], list[str]]:
    """Reduce-then-apply body (e.g. rmsnorm / softmax). Each intermediate value
    is staged into its own SBUF scratch tile via the ``{DST}`` placeholder.
    Reduction accumulators are fp32 regardless of input dtype (a bf16
    ``operand0`` is rejected by ``nisa.tensor_scalar``)."""
    ind = ctx.indent
    tile_m = ctx.tile("tile_m")
    tile_n = ctx.tile("tile_n")

    # Both reduce bodies load a single (M, N) input tensor (params[0]) and treat
    # every input node as that tile.
    if len(params) != 1:
        raise UnsupportedEmission(
            f"reduce kernel '{ctx.kernel_name}' has {len(params)} inputs; the "
            f"reduce and n-reduce bodies load a single (M, N) input tensor and "
            f"would silently alias the rest onto it."
        )

    struct = analyze_reduce_structure(compute_nodes, input_nodes, id_to_node)
    if struct is None:
        # Not a single-reduce shape: try the general n-reduce body.
        nstruct = analyze_nreduce_structure(compute_nodes, input_nodes, id_to_node)
        if nstruct is not None:
            return emit_nreduce_body(
                ctx, input_nodes, compute_nodes, id_to_node, params, nstruct
            )
        n_reduce = sum(
            1 for n in compute_nodes if n.op in ("tensor_reduce", "activation_reduce")
        )
        raise UnsupportedEmission(
            f"reduce kernel '{ctx.kernel_name}' has an unrecognized reduction "
            f"structure ({n_reduce} free-axis reduce node(s)): neither the "
            f"single-reduce nor the n-reduce body can lower it."
        )

    pre_reduce_chain = struct["pre_reduce_chain"]
    reduce_node = struct["reduce_node"]
    finalize_chain = struct["finalize_chain"]
    apply_node = struct["apply_node"]
    apply_tile_input_id = struct["apply_tile_input_id"]
    post_apply_chain = struct["post_apply_chain"]
    needs_pre_buf = struct["needs_pre_buf"]

    # The cross-block accumulator below is a zeros identity combined with nl.add,
    # so any other monoid (maximum, multiply) would compute a wrong result.
    if reduce_node.op == "activation_reduce":
        reduce_op = _nki_op_ref(reduce_node.attrs.get("reduce_op"))
        if reduce_op != "nl.add":
            raise UnsupportedEmission(
                f"activation_reduce {reduce_node.id}: the reduce body accumulates "
                f"across free-axis blocks with nl.add from a zeros identity, so "
                f"reduce_op={reduce_op} would emit a wrong result"
            )

    lhs_var = params[0]

    # We need 4 indent levels
    i1 = ind
    i2 = ind + "    "
    i3 = ind + "        "
    i4 = ind + "            "

    lines: list[str] = []

    lines += [
        f"{i1}M, N = {lhs_var}.shape",
        f"{i1}output = nl.ndarray((M, N), dtype={lhs_var}.dtype, buffer=nl.shared_hbm)",
        "",
        f"{i1}TILE_M = {tile_m}",
        f"{i1}TILE_N = {tile_n}",
        f"{i1}BLOCK_M = TILE_M * TILES_IN_BLOCK_M",
        f"{i1}BLOCK_N = TILE_N * TILES_IN_BLOCK_N",
        f"{i1}NUM_BLOCK_M = M // BLOCK_M",
        f"{i1}NUM_BLOCK_N = N // BLOCK_N",
        "",
        f"{i1}assert NUM_BLOCK_M > 0 and NUM_BLOCK_N > 0, \\",
        f'{i1}    "Input size too small for the given tile configuration"',
        "",
        f"{i1}for m in nl.affine_range(NUM_BLOCK_M):",
    ]

    lines += [
        f"{i2}accum = nl.zeros(",
        f"{i2}    (TILE_M, TILES_IN_BLOCK_M, 1),",
        f"{i2}    dtype=nl.float32, buffer=nl.sbuf)",
        "",
    ]

    if needs_pre_buf:
        lines += [
            f"{i2}pre_buf = nl.ndarray(",
            f"{i2}    (TILE_M, NUM_BLOCK_N, TILES_IN_BLOCK_M, BLOCK_N),",
            f"{i2}    dtype={lhs_var}.dtype, buffer=nl.sbuf)",
            "",
        ]

    # Stage an ISA op's dst-first call into a fresh SBUF scratch tile of the
    # given (P, F), substituting the op emitter's {DST}/{IND} placeholders.
    def _stage(name: str, call: str, p: str, f: str, indent: str) -> str:
        dst = f"{name}[0:{p}, 0:{f}]"
        stmt = substitute_dst(
            call,
            dst=dst,
            dst_slice=f"0:{p}, 0:{f}",
            dst_shape=f"{p}, {f}",
            ind=indent,
        )
        lines.append(
            f"{indent}{name} = nl.ndarray(({p}, {f}), dtype=nl.float32, buffer=nl.sbuf)"
        )
        lines.append(f"{indent}{stmt}")
        return dst

    lines += [
        f"{i2}for n in nl.sequential_range(NUM_BLOCK_N):",
        f"{i3}in_tiles = nl.ndarray(",
        f"{i3}    (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),",
        f"{i3}    dtype={lhs_var}.dtype, buffer=nl.sbuf)",
        f"{i3}for tile_m in nl.affine_range(TILES_IN_BLOCK_M):",
        f"{i4}nisa.dma_copy(",
        f"{i4}    dst=in_tiles[0:TILE_M, tile_m, 0:BLOCK_N],",
        f"{i4}    src={lhs_var}[((TILES_IN_BLOCK_M * m + tile_m) * TILE_M):"
        f"((TILES_IN_BLOCK_M * m + tile_m) * TILE_M) + TILE_M,",
        f"{i4}       (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N])",
        "",
        f"{i3}for tile_m in nl.affine_range(TILES_IN_BLOCK_M):",
    ]

    in_tiles_expr = "in_tiles[0:TILE_M, tile_m, 0:BLOCK_N]"
    id_to_var_p1: dict[str, str] = {inp.id: in_tiles_expr for inp in input_nodes}
    for node in pre_reduce_chain:
        var = f"pre_{nki_safe_var(node.id)}"
        call = emit_or_raise(ctx, node, id_to_var_p1)
        id_to_var_p1[node.id] = _stage(var, call, "TILE_M", "BLOCK_N", i4)

    reduce_call = emit_or_raise(ctx, reduce_node, id_to_var_p1)
    temp_accum_expr = _stage("temp_accum", reduce_call, "TILE_M", "1", i4)

    if needs_pre_buf:
        pre_source = id_to_var_p1.get(apply_tile_input_id, in_tiles_expr)
        lines.append(
            f"{i4}nisa.tensor_copy(pre_buf[0:TILE_M, n, tile_m, 0:BLOCK_N],"
            f" {pre_source})"
        )

    lines += [
        f"{i4}nisa.tensor_tensor(accum[0:TILE_M, tile_m, 0],",
        f"{i4}    accum[0:TILE_M, tile_m, 0], {temp_accum_expr}, op=nl.add)",
        "",
    ]

    if finalize_chain:
        lines.append(f"{i2}for tile_m in nl.affine_range(TILES_IN_BLOCK_M):")
        id_to_var_fin: dict[str, str] = {reduce_node.id: "accum[0:TILE_M, tile_m, 0]"}
        for node in finalize_chain:
            var = f"fin_{nki_safe_var(node.id)}"
            call = emit_or_raise(ctx, node, id_to_var_fin)
            id_to_var_fin[node.id] = _stage(var, call, "TILE_M", "1", i3)
        fin_last_expr = id_to_var_fin[finalize_chain[-1].id]
        lines += [
            f"{i3}nisa.tensor_copy(accum[0:TILE_M, tile_m, 0], {fin_last_expr})",
            "",
        ]

    scalar_id = finalize_chain[-1].id if finalize_chain else reduce_node.id

    lines += [
        f"{i2}for n in nl.affine_range(NUM_BLOCK_N):",
    ]

    if not needs_pre_buf:
        lines += [
            f"{i3}in_tiles = nl.ndarray(",
            f"{i3}    (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),",
            f"{i3}    dtype={lhs_var}.dtype, buffer=nl.sbuf)",
            f"{i3}for tile_m in nl.affine_range(TILES_IN_BLOCK_M):",
            f"{i4}nisa.dma_copy(",
            f"{i4}    dst=in_tiles[0:TILE_M, tile_m, 0:BLOCK_N],",
            f"{i4}    src={lhs_var}[((TILES_IN_BLOCK_M * m + tile_m) * TILE_M):"
            f"((TILES_IN_BLOCK_M * m + tile_m) * TILE_M) + TILE_M,",
            f"{i4}       (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N])",
            "",
        ]

    lines.append(f"{i3}for tile_m in nl.affine_range(TILES_IN_BLOCK_M):")

    tile_data_expr = (
        "pre_buf[0:TILE_M, n, tile_m, 0:BLOCK_N]"
        if needs_pre_buf
        else "in_tiles[0:TILE_M, tile_m, 0:BLOCK_N]"
    )

    id_to_var_p2: dict[str, str] = {
        apply_tile_input_id: tile_data_expr,
        scalar_id: "accum[0:TILE_M, tile_m, 0]",
    }
    if not needs_pre_buf:
        for inp in input_nodes:
            id_to_var_p2[inp.id] = tile_data_expr

    apply_call = emit_or_raise(ctx, apply_node, id_to_var_p2)
    apply_var = nki_safe_var(apply_node.id)
    id_to_var_p2[apply_node.id] = _stage(apply_var, apply_call, "TILE_M", "BLOCK_N", i4)

    for node in post_apply_chain:
        var = nki_safe_var(node.id)
        call = emit_or_raise(ctx, node, id_to_var_p2)
        id_to_var_p2[node.id] = _stage(var, call, "TILE_M", "BLOCK_N", i4)

    final_expr = id_to_var_p2[
        post_apply_chain[-1].id if post_apply_chain else apply_node.id
    ]
    lines += _emit_reduce_store(ctx, i4, final_expr)

    return lines, ["output"]


def _emit_reduce_store(ctx: EmitCtx, i4: str, final_expr: str) -> list[str]:
    """The reduce result-store tail (SPMD result-tail seam): copy the final
    per-tile result to HBM."""
    return [
        f"{i4}nisa.dma_copy(",
        f"{i4}    dst=output[((TILES_IN_BLOCK_M * m + tile_m) * TILE_M):"
        f"((TILES_IN_BLOCK_M * m + tile_m) * TILE_M) + TILE_M,",
        f"{i4}           (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N],",
        f"{i4}    src={final_expr})",
        "",
    ]


def emit_nreduce_body(
    ctx: EmitCtx,
    input_nodes: list[Node],
    compute_nodes: list[Node],
    id_to_node: dict[str, Node],
    params: list[str],
    struct: dict,
) -> tuple[list[str], list[str]]:
    """General n-reduce body: a dependent chain of reductions (e.g. LayerNorm's
    mean then variance), lowered as **one hidden-axis sweep per reduction**
    followed by a final apply sweep.

    Each stage keeps a persistent ``(TILE_M, TILES_IN_BLOCK_M, 1)`` buffer that
    first accumulates the reduction across the ``NUM_BLOCK_N`` free-axis blocks,
    then (after the sweep) holds the finalized scalar (mean, 1/std, ...). Later
    stages and the final apply read those scalar buffers to rebuild wide
    intermediates (like ``x - mean``) from the reloaded input tiles. Wide tiles
    are recomputed per pass rather than cached, which keeps the body general for
    any number of stages. Reduction accumulators and staged tiles are fp32."""
    ind = ctx.indent
    tile_m = ctx.tile("tile_m")
    tile_n = ctx.tile("tile_n")
    lhs_var = params[0]

    stages: list[dict] = struct["stages"]
    output_chain: list[Node] = struct["output_chain"]
    output_id: str = struct["output_id"]

    # The tensor_scalar operand-order swap keys off this set, so mark every
    # per-partition scalar node (reduce outputs + their scalar finalize chains).
    ctx.partition_scalar_ids = {n.id for n in compute_nodes if _free_dim_is_one(n)}

    i1 = ind
    i2 = ind + "    "
    i3 = ind + "        "
    i4 = ind + "            "

    def _scalar_buf(k: int) -> str:
        return f"scalar_{k}"

    lines: list[str] = []

    # Stage a dst-first ISA call into a fresh fp32 SBUF scratch tile of (p, f),
    # substituting the op emitter's {DST}/{IND} placeholders.
    def _stage(name: str, call: str, p: str, f: str, indent: str) -> str:
        dst = f"{name}[0:{p}, 0:{f}]"
        stmt = substitute_dst(
            call, dst=dst, dst_slice=f"0:{p}, 0:{f}", dst_shape=f"{p}, {f}", ind=indent
        )
        lines.append(
            f"{indent}{name} = nl.ndarray(({p}, {f}), dtype=nl.float32, buffer=nl.sbuf)"
        )
        lines.append(f"{indent}{stmt}")
        return dst

    # Load the current (m, n) block of the input into `in_tiles`.
    def _emit_load(alloc_ind: str, copy_ind: str) -> None:
        lines.extend(
            [
                f"{alloc_ind}in_tiles = nl.ndarray(",
                f"{alloc_ind}    (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),",
                f"{alloc_ind}    dtype={lhs_var}.dtype, buffer=nl.sbuf)",
                f"{alloc_ind}for tile_m in nl.affine_range(TILES_IN_BLOCK_M):",
                f"{copy_ind}nisa.dma_copy(",
                f"{copy_ind}    dst=in_tiles[0:TILE_M, tile_m, 0:BLOCK_N],",
                f"{copy_ind}    src={lhs_var}[((TILES_IN_BLOCK_M * m + tile_m) * TILE_M):"
                f"((TILES_IN_BLOCK_M * m + tile_m) * TILE_M) + TILE_M,",
                f"{copy_ind}       (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N])",
                "",
            ]
        )

    in_tiles_expr = "in_tiles[0:TILE_M, tile_m, 0:BLOCK_N]"

    def _scalar_expr(k: int) -> str:
        return f"{_scalar_buf(k)}[0:TILE_M, tile_m, 0]"

    lines += [
        f"{i1}M, N = {lhs_var}.shape",
        f"{i1}output = nl.ndarray((M, N), dtype={lhs_var}.dtype, buffer=nl.shared_hbm)",
        "",
        f"{i1}TILE_M = {tile_m}",
        f"{i1}TILE_N = {tile_n}",
        f"{i1}BLOCK_M = TILE_M * TILES_IN_BLOCK_M",
        f"{i1}BLOCK_N = TILE_N * TILES_IN_BLOCK_N",
        f"{i1}NUM_BLOCK_M = M // BLOCK_M",
        f"{i1}NUM_BLOCK_N = N // BLOCK_N",
        "",
        f"{i1}assert NUM_BLOCK_M > 0 and NUM_BLOCK_N > 0, \\",
        f'{i1}    "Input size too small for the given tile configuration"',
        "",
        f"{i1}for m in nl.affine_range(NUM_BLOCK_M):",
    ]

    # One persistent accumulator-then-scalar buffer per reduction stage.
    for k in range(len(stages)):
        lines += [
            f"{i2}{_scalar_buf(k)} = nl.zeros(",
            f"{i2}    (TILE_M, TILES_IN_BLOCK_M, 1),",
            f"{i2}    dtype=nl.float32, buffer=nl.sbuf)",
        ]
    lines.append("")

    # --- one reduction sweep per stage, in dependency order ---
    for k, stage in enumerate(stages):
        reduce_node = stage["reduce_node"]
        pre_reduce_chain = stage["pre_reduce_chain"]
        finalize_chain = stage["finalize_chain"]
        buf = _scalar_buf(k)

        lines.append(f"{i2}for n in nl.sequential_range(NUM_BLOCK_N):")
        _emit_load(i3, i4)
        lines.append(f"{i3}for tile_m in nl.affine_range(TILES_IN_BLOCK_M):")

        id_to_var: dict[str, str] = {inp.id: in_tiles_expr for inp in input_nodes}
        for j in range(k):  # earlier stages' finalized scalars are available
            id_to_var[stages[j]["scalar_id"]] = _scalar_expr(j)

        for node in pre_reduce_chain:
            call = emit_or_raise(ctx, node, id_to_var)
            var = f"pre{k}_{nki_safe_var(node.id)}"
            id_to_var[node.id] = _stage(var, call, "TILE_M", "BLOCK_N", i4)

        reduce_call = emit_or_raise(ctx, reduce_node, id_to_var)
        temp_expr = _stage(f"temp_accum_{k}", reduce_call, "TILE_M", "1", i4)
        lines += [
            f"{i4}nisa.tensor_tensor({buf}[0:TILE_M, tile_m, 0],",
            f"{i4}    {buf}[0:TILE_M, tile_m, 0], {temp_expr}, op=nl.add)",
            "",
        ]

        # Finalize the accumulator into this stage's scalar (stored back in buf).
        if finalize_chain:
            lines.append(f"{i2}for tile_m in nl.affine_range(TILES_IN_BLOCK_M):")
            id_to_var_fin: dict[str, str] = {
                reduce_node.id: f"{buf}[0:TILE_M, tile_m, 0]"
            }
            for node in finalize_chain:
                call = emit_or_raise(ctx, node, id_to_var_fin)
                var = f"fin{k}_{nki_safe_var(node.id)}"
                id_to_var_fin[node.id] = _stage(var, call, "TILE_M", "1", i3)
            fin_last = id_to_var_fin[finalize_chain[-1].id]
            lines += [
                f"{i3}nisa.tensor_copy({buf}[0:TILE_M, tile_m, 0], {fin_last})",
                "",
            ]

    # --- final apply sweep: rebuild the output from inputs + all scalars ---
    lines.append(f"{i2}for n in nl.affine_range(NUM_BLOCK_N):")
    _emit_load(i3, i4)
    lines.append(f"{i3}for tile_m in nl.affine_range(TILES_IN_BLOCK_M):")

    id_to_var_out: dict[str, str] = {inp.id: in_tiles_expr for inp in input_nodes}
    for k, stage in enumerate(stages):
        id_to_var_out[stage["scalar_id"]] = _scalar_expr(k)
    for node in output_chain:
        call = emit_or_raise(ctx, node, id_to_var_out)
        var = f"out_{nki_safe_var(node.id)}"
        id_to_var_out[node.id] = _stage(var, call, "TILE_M", "BLOCK_N", i4)

    lines += _emit_reduce_store(ctx, i4, id_to_var_out[output_id])

    return lines, ["output"]
