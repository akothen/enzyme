"""The degenerate flat fallback body.

A best-effort path the matmul / reduce analyses route to when they cannot
recognize the graph; not exercised by any emitted kernel today. The reduce and
matmul bodies both fall back to it.
"""

from __future__ import annotations

from axon.codegen.blocks import substitute_dst
from axon.codegen.context import EmitCtx
from axon.codegen.ops import emit_or_raise, nki_safe_var
from axon.ir import Node


def emit_flat_body(
    ctx: EmitCtx,
    input_nodes: list[Node],
    compute_nodes: list[Node],
    params: list[str],
) -> tuple[list[str], list[str]]:
    """Degenerate flat fallback. Inputs and each compute result are staged into
    a per-node SBUF tile via the op emitters' ``{DST}`` placeholder. Carries no
    tiling, so a placeholder ``(128, 1)`` tile shape stands in for the unknown
    operand extent."""
    ind = ctx.indent
    lines: list[str] = []

    lines += [
        f"{ind}TILE_M = nl.tile_size.gemm_stationary_fmax  # 128",
        f"{ind}TILE_N = nl.tile_size.gemm_moving_fmax  # 512",
        "",
    ]

    default_dtype = _flat_default_dtype(input_nodes, compute_nodes, params)
    # node id -> (slice expr referring to the staged SBUF tile, dtype expr)
    staged: dict[str, tuple[str, str]] = {}
    sl = "[0:128, 0:1]"

    lines.append(f"{ind}# Load inputs from HBM to SBUF")
    for n in input_nodes:
        v = nki_safe_var(n.id)
        lines += [
            f"{ind}{v}_tile = nl.ndarray((128, 1), dtype={v}.dtype, buffer=nl.sbuf)",
            f"{ind}nisa.dma_copy(dst={v}_tile{sl}, src={v})",
        ]
        staged[n.id] = (f"{v}_tile{sl}", f"{v}.dtype")
    lines.append("")

    id_to_var: dict[str, str] = {n.id: staged[n.id][0] for n in input_nodes}

    last_var: str | None = None
    for node in compute_nodes:
        var = f"{nki_safe_var(node.id)}_tile"
        call = emit_or_raise(ctx, node, id_to_var)
        if call is None:
            continue
        dst = f"{var}{sl}"
        stmt = substitute_dst(
            call, dst=dst, dst_slice="0:128, 0:1", dst_shape="128, 1", ind=ind
        )
        lines += [
            f"{ind}{var} = nl.ndarray((128, 1), dtype={default_dtype}, buffer=nl.sbuf)",
            f"{ind}{stmt}",
        ]
        id_to_var[node.id] = dst
        staged[node.id] = (dst, default_dtype)
        last_var = dst
    lines.append("")

    if last_var:
        out_dtype = staged[compute_nodes[-1].id][1] if compute_nodes else default_dtype
        lines += _emit_flat_store(ctx, ind, last_var, out_dtype)
        return lines, ["output"]

    return lines, []


def _emit_flat_store(
    ctx: EmitCtx, ind: str, last_var: str, out_dtype: str
) -> list[str]:
    """The flat result-store tail (SPMD result-tail seam): allocate a named
    shared-HBM output buffer and copy the final staged tile into it, so emit()
    owns the ``return`` (matching the other bodies). The placeholder ``(128, 1)``
    shape mirrors the degenerate staged-tile extent."""
    return [
        f"{ind}# Store output to HBM",
        f"{ind}output = nl.ndarray((128, 1), dtype={out_dtype}, buffer=nl.shared_hbm)",
        f"{ind}nisa.dma_copy(dst=output, src={last_var})",
    ]


def _flat_default_dtype(
    input_nodes: list[Node], compute_nodes: list[Node], params: list[str]
) -> str:
    if params:
        return f"{params[0]}.dtype"
    if input_nodes:
        return f"{nki_safe_var(input_nodes[0].id)}.dtype"
    return "nl.float32"
