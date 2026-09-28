"""The cumulative-scan (cumsum) body builder.

The free-block loop runs sequentially and threads a running-sum carry along the
scanned axis, structurally distinct from elementwise.
"""

from __future__ import annotations

from axon.codegen.context import EmitCtx
from axon.codegen.ops import _nki_op_ref
from axon.ir import Node
from axon.isa_semantics import _operand_to_expr


def emit_scan_body(
    ctx: EmitCtx,
    input_nodes: list[Node],
    compute_nodes: list[Node],
    params: list[str],
) -> tuple[list[str], list[str]]:
    """Emit a cumulative scan (cumsum) that carries the running sum across
    free-axis blocks.

    No result-tail seam is extracted here (unlike the other bodies): the scan's
    two output stores are interleaved with the carry update inside the
    sequential loop, so there is no single contiguous store region to intercept.
    """
    ind = ctx.indent
    tile_m = ctx.tile("tile_m")
    tile_n = ctx.tile("tile_n")
    scan_nodes = [n for n in compute_nodes if n.op == "tensor_scalar_cumulative"]
    if (
        len(scan_nodes) != 1
        or len(compute_nodes) != 1
        or len(input_nodes) != 1
        or scan_nodes[0].inputs[:1] != [input_nodes[0].id]
    ):
        raise ValueError(
            "NKIEmitter: scan body only supports a single "
            "tensor_scalar_cumulative over a single input; refusing to "
            "fall back to the carry-less elementwise scan (wrong for "
            "NUM_BLOCK_N > 1)."
        )

    scan = scan_nodes[0]
    attrs = scan.attrs
    op0_ref = _nki_op_ref(attrs.get("op0"))
    op1_ref = _nki_op_ref(attrs.get("op1"))
    # A zero-initialized carry is the identity only for an additive scan, so
    # refuse product/min/max (which need a different seed) rather than miscompute.
    op1_name = _operand_to_expr(attrs.get("op1"))
    if op1_name not in ("add", "plus"):
        raise ValueError(
            "NKIEmitter: scan body only supports an additive scan "
            f"(op1=add); got op1={op1_name!r}. A zero-initialized carry is "
            "the wrong identity for product/min/max scans."
        )
    if "imm0_input_index" in attrs:
        # A tensor imm0 would need a per-element load we don't emit here.
        raise ValueError(
            "NKIEmitter: scan body does not support a tensor imm0; "
            "refusing to fall back to the carry-less elementwise scan."
        )
    imm0_const = attrs.get("imm0_const", 0)
    # Emit a bare numeric literal (a np.float32(...) wrapper fails under the
    # NKI tracer).
    imm0 = (
        f"{float(imm0_const)!r}"
        if isinstance(imm0_const, (int, float))
        else repr(imm0_const)
    )

    x = params[0] if params else "input"
    i2 = ind * 2
    i3 = ind * 3
    i4 = ind * 4

    lines: list[str] = [
        f"{ind}TILE_M = {tile_m}  # partition dimension tile size",
        f"{ind}TILE_N = {tile_n}  # free dimension tile size",
        f"{ind}BLOCK_M = TILES_IN_BLOCK_M * TILE_M",
        f"{ind}BLOCK_N = TILES_IN_BLOCK_N * TILE_N",
        "",
        f"{ind}NUM_BLOCK_M = {x}.shape[0] // BLOCK_M",
        f"{ind}NUM_BLOCK_N = {x}.shape[1] // BLOCK_N",
        "",
        f"{ind}assert NUM_BLOCK_M > 0 and NUM_BLOCK_N > 0, \\",
        f'{ind}    "Input size too small for the given tile configuration"',
        "",
        f"{ind}output = nl.ndarray(",
        f"{ind}    {x}.shape, dtype={x}.dtype, buffer=nl.shared_hbm)",
        "",
        f"{ind}for m in nl.affine_range(NUM_BLOCK_M):",
        f"{i2}for tile_m in nl.affine_range(TILES_IN_BLOCK_M):",
        # Per-partition running sum, fp32 (imm1 must be fp32 per ISA, and a
        # wider carry keeps the cross-block prefix accurate).
        f"{i3}carry = nl.zeros(",
        f"{i3}    (TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)",
        f"{i3}for n in nl.sequential_range(NUM_BLOCK_N):",
        f"{i4}x_tile = nl.ndarray(",
        f"{i4}    (TILE_M, BLOCK_N), dtype={x}.dtype, buffer=nl.sbuf)",
        f"{i4}nisa.dma_copy(",
        f"{i4}    dst=x_tile[0:TILE_M, 0:BLOCK_N],",
        f"{i4}    src={x}[((TILES_IN_BLOCK_M * m + tile_m) * TILE_M):"
        f"((TILES_IN_BLOCK_M * m + tile_m) * TILE_M) + TILE_M,",
        f"{i4}       (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N])",
        f"{i4}out_tile = nl.ndarray(",
        f"{i4}    (TILE_M, BLOCK_N), dtype={x}.dtype, buffer=nl.sbuf)",
        f"{i4}nisa.tensor_scalar_cumulative(",
        f"{i4}    dst=out_tile[0:TILE_M, 0:BLOCK_N], src=x_tile[0:TILE_M, 0:BLOCK_N],",
        f"{i4}    op0={op0_ref}, op1={op1_ref}, imm0={imm0},",
        f"{i4}    imm1=carry[0:TILE_M, 0:1],",
        f"{i4}    reduce_cmd=nisa.reduce_cmd.load_reduce)",
        f"{i4}nisa.dma_copy(",
        f"{i4}    dst=output[((TILES_IN_BLOCK_M * m + tile_m) * TILE_M):"
        f"((TILES_IN_BLOCK_M * m + tile_m) * TILE_M) + TILE_M,",
        f"{i4}           (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N],",
        f"{i4}    src=out_tile[0:TILE_M, 0:BLOCK_N])",
        # Seed the next block's carry from this block's last column.
        f"{i4}nisa.tensor_copy(",
        f"{i4}    carry[0:TILE_M, 0:1],"
        f" out_tile[0:TILE_M, (BLOCK_N - 1):(BLOCK_N - 1) + 1])",
        # Ragged trailing block: the full-block loop floor-divides the free
        # axis, leaving the last shape[1] % BLOCK_N columns. REM_N is concrete
        # at trace time so this `if` folds away when the axis tiles exactly. The
        # tail reuses the carry from the last full block (no update after).
        f"{i3}REM_N = {x}.shape[1] % BLOCK_N",
        f"{i3}if REM_N:",
        f"{i4}x_tail = nl.ndarray(",
        f"{i4}    (TILE_M, REM_N), dtype={x}.dtype, buffer=nl.sbuf)",
        f"{i4}nisa.dma_copy(",
        f"{i4}    dst=x_tail[0:TILE_M, 0:REM_N],",
        f"{i4}    src={x}[((TILES_IN_BLOCK_M * m + tile_m) * TILE_M):"
        f"((TILES_IN_BLOCK_M * m + tile_m) * TILE_M) + TILE_M,",
        f"{i4}       (BLOCK_N * NUM_BLOCK_N):(BLOCK_N * NUM_BLOCK_N) + REM_N])",
        f"{i4}out_tail = nl.ndarray(",
        f"{i4}    (TILE_M, REM_N), dtype={x}.dtype, buffer=nl.sbuf)",
        f"{i4}nisa.tensor_scalar_cumulative(",
        f"{i4}    dst=out_tail[0:TILE_M, 0:REM_N], src=x_tail[0:TILE_M, 0:REM_N],",
        f"{i4}    op0={op0_ref}, op1={op1_ref}, imm0={imm0},",
        f"{i4}    imm1=carry[0:TILE_M, 0:1],",
        f"{i4}    reduce_cmd=nisa.reduce_cmd.load_reduce)",
        f"{i4}nisa.dma_copy(",
        f"{i4}    dst=output[((TILES_IN_BLOCK_M * m + tile_m) * TILE_M):"
        f"((TILES_IN_BLOCK_M * m + tile_m) * TILE_M) + TILE_M,",
        f"{i4}           (BLOCK_N * NUM_BLOCK_N):(BLOCK_N * NUM_BLOCK_N) + REM_N],",
        f"{i4}    src=out_tail[0:TILE_M, 0:REM_N])",
        "",
    ]
    return lines, ["output"]
