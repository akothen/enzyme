"""The elementwise body builder (tiled elementwise + multi-input fold)."""

from __future__ import annotations

from axon.codegen.blocks import substitute_dst
from axon.codegen.context import EmitCtx
from axon.codegen.ops import emit_or_raise, nki_safe_var
from axon.ir import Node


def emit_elementwise_body(
    ctx: EmitCtx,
    input_nodes: list[Node],
    compute_nodes: list[Node],
    params: list[str],
    output_ids: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Tiled elementwise body."""
    ind = ctx.indent
    tile_m = ctx.tile("tile_m")
    tile_n = ctx.tile("tile_n")
    fuse_loads = ctx.fuse_loads
    lines: list[str] = []
    i2 = ind * 2
    i3 = ind * 3
    i4 = ind * 4

    first_input = params[0] if params else "input"

    if not output_ids:
        output_ids = [compute_nodes[-1].id] if compute_nodes else []
    output_buf_names = (
        ["output"]
        if len(output_ids) == 1
        else [f"out_{i}" for i in range(len(output_ids))]
    )

    lines += [
        f"{ind}TILE_M = {tile_m}  # partition dimension tile size",
        f"{ind}TILE_N = {tile_n}  # free dimension tile size",
        f"{ind}BLOCK_M = TILES_IN_BLOCK_M * TILE_M",
        f"{ind}BLOCK_N = TILES_IN_BLOCK_N * TILE_N",
        "",
        f"{ind}NUM_BLOCK_M = {first_input}.shape[0] // BLOCK_M",
        f"{ind}NUM_BLOCK_N = {first_input}.shape[1] // BLOCK_N",
        f"{ind}REM_N = {first_input}.shape[1] % BLOCK_N",
        "",
        f"{ind}assert NUM_BLOCK_M > 0, \\",
        f'{ind}    "Input partition size too small for the given tile config"',
        "",
    ]
    for buf_name in output_buf_names:
        lines += [
            f"{ind}{buf_name} = nl.ndarray(",
            f"{ind}    {first_input}.shape, dtype={first_input}.dtype,"
            f" buffer=nl.shared_hbm)",
        ]
    lines.append("")

    # (P, 1) per-partition scalars feed tensor_scalar / scalar_tensor_tensor.
    def _is_partition_scalar(n: Node) -> bool:
        shp = n.attrs.get("shape", n.shape or ())
        return len(shp) == 2 and shp[1] == 1

    # Record them so the op emitters' operand-order swap fires on the set.
    ctx.partition_scalar_ids = {n.id for n in input_nodes if _is_partition_scalar(n)}

    use_fused = fuse_loads and len(input_nodes) > 1
    row_stmt = "row = (TILES_IN_BLOCK_M * m + tile_m) * TILE_M"

    def _alloc_lines(n: Node, indent_str: str, width: str) -> list[str]:
        v = nki_safe_var(n.id)
        tiles_var = f"{v}_tiles"
        if _is_partition_scalar(n):
            # operand0 must be float32, so stage the (P, 1) tile as fp32
            # regardless of the HBM input dtype; the load below casts.
            return [
                f"{indent_str}{tiles_var} = nl.ndarray(",
                f"{indent_str}    (TILE_M, TILES_IN_BLOCK_M, 1),",
                f"{indent_str}    dtype=nl.float32, buffer=nl.sbuf)",
            ]
        return [
            f"{indent_str}{tiles_var} = nl.ndarray(",
            f"{indent_str}    (TILE_M, TILES_IN_BLOCK_M, {width}),",
            f"{indent_str}    dtype={v}.dtype, buffer=nl.sbuf)",
        ]

    def _load_lines(n: Node, indent_str: str, width: str, col0: str) -> list[str]:
        v = nki_safe_var(n.id)
        tiles_var = f"{v}_tiles"
        if _is_partition_scalar(n):
            # dma_copy cannot cast dtypes, so a narrower HBM input is staged
            # same-dtype then tensor_copy'd up to fp32; fp32 dmas directly.
            # A (P, 1) scalar has no free-axis offset, so col0/width don't apply.
            same = [
                f"{indent_str}nisa.dma_copy(",
                f"{indent_str}    dst={tiles_var}[0:TILE_M, tile_m, 0:1],",
                f"{indent_str}    src={v}[row:row + TILE_M, 0:1])",
            ]
            cast = [
                f"{indent_str}{tiles_var}_in = nl.ndarray(",
                f"{indent_str}    (TILE_M, 1), dtype={v}.dtype, buffer=nl.sbuf)",
                f"{indent_str}nisa.dma_copy(",
                f"{indent_str}    dst={tiles_var}_in[0:TILE_M, 0:1],",
                f"{indent_str}    src={v}[row:row + TILE_M, 0:1])",
                f"{indent_str}nisa.tensor_copy(",
                f"{indent_str}    {tiles_var}[0:TILE_M, tile_m, 0:1],"
                f" {tiles_var}_in[0:TILE_M, 0:1])",
            ]
            return [
                f"{indent_str}if str({v}.dtype) == 'float32':",
                *[f"    {ln}" for ln in same],
                f"{indent_str}else:",
                *[f"    {ln}" for ln in cast],
            ]
        return [
            f"{indent_str}nisa.dma_copy(",
            f"{indent_str}    dst={tiles_var}[0:TILE_M, tile_m, 0:{width}],",
            f"{indent_str}    src={v}[row:row + TILE_M, {col0}:{col0} + {width}])",
        ]

    def _emit_n_block(width: str, col0: str) -> list[str]:
        """Alloc + load + compute + store for one free-axis block of the given
        column ``width`` at column base ``col0``. Emitted at i3/i4, so it slots
        under either the ``for n`` loop or the ``if REM_N`` tail. Builds its own
        ``id_to_tile_expr`` so the two passes stay independent."""
        blk: list[str] = []

        id_to_tile_expr: dict[str, str] = {}
        for n in input_nodes:
            v = nki_safe_var(n.id)
            if _is_partition_scalar(n):
                id_to_tile_expr[n.id] = f"{v}_tiles[0:TILE_M, tile_m, 0:1]"
            else:
                id_to_tile_expr[n.id] = f"{v}_tiles[0:TILE_M, tile_m, 0:{width}]"

        if use_fused:
            alloc_lines: list[str] = []
            for n in input_nodes:
                alloc_lines += _alloc_lines(n, i3, width)
            blk += alloc_lines + [
                "",
                f"{i3}for tile_m in nl.affine_range(TILES_IN_BLOCK_M):",
                f"{i4}{row_stmt}",
            ]
            for n in input_nodes:
                blk += _load_lines(n, i4, width, col0)
            blk.append("")
        else:
            for n in input_nodes:
                v = nki_safe_var(n.id)
                blk += [f"{i3}# Load {v} from HBM into SBUF tiles"]
                blk += _alloc_lines(n, i3, width)
                blk += [
                    f"{i3}for tile_m in nl.affine_range(TILES_IN_BLOCK_M):",
                    f"{i4}{row_stmt}",
                ]
                blk += _load_lines(n, i4, width, col0)
                blk.append("")

        for node in compute_nodes:
            nv = nki_safe_var(node.id)
            out_tiles_var = f"{nv}_tiles"

            call_expr = emit_or_raise(ctx, node, id_to_tile_expr)
            if call_expr is None:
                continue

            dst_expr = f"{out_tiles_var}[0:TILE_M, tile_m, 0:{width}]"
            dst_slice = f"0:TILE_M, 0:{width}"
            dst_shape = f"TILE_M, {width}"
            stmt = substitute_dst(
                call_expr,
                dst=dst_expr,
                dst_slice=dst_slice,
                dst_shape=dst_shape,
                ind=i4,
            )
            blk += [
                f"{i3}# {node.op}",
                f"{i3}{out_tiles_var} = nl.ndarray(",
                f"{i3}    (TILE_M, TILES_IN_BLOCK_M, {width}),",
                f"{i3}    dtype={first_input}.dtype, buffer=nl.sbuf)",
                f"{i3}for tile_m in nl.affine_range(TILES_IN_BLOCK_M):",
                f"{i4}{stmt}",
                "",
            ]
            id_to_tile_expr[node.id] = dst_expr

        output_tile_vars: list[str] = []
        for oid in output_ids:
            tv = id_to_tile_expr.get(oid)
            if tv is None:
                raise ValueError(
                    f"NKIEmitter: declared output {oid!r} has no tile expr; "
                    "synthesis may have dropped a dependent node. Refusing to "
                    "alias outputs to the last computed tile."
                )
            tile_var = tv.split("[", 1)[0]
            output_tile_vars.append(tile_var)

        if output_tile_vars and any(output_tile_vars):
            blk += _emit_elementwise_store(
                i3, i4, row_stmt, output_buf_names, output_tile_vars, width, col0
            )
        return blk

    lines += [
        f"{ind}for m in nl.affine_range(NUM_BLOCK_M):",
        f"{i2}for n in nl.affine_range(NUM_BLOCK_N):",
        "",
    ]
    lines += _emit_n_block("BLOCK_N", "BLOCK_N * n")
    # Free-axis remainder: the last shape[1] % BLOCK_N columns. REM_N is concrete
    # at trace, so `if REM_N:` is a compile-time branch (matches the scan body).
    lines += [
        f"{i2}if REM_N:",
        "",
    ]
    lines += _emit_n_block("REM_N", "BLOCK_N * NUM_BLOCK_N")

    return lines, list(output_buf_names)


def _emit_elementwise_store(
    i3: str,
    i4: str,
    row_stmt: str,
    output_buf_names: list[str],
    output_tile_vars: list[str],
    width: str,
    col0: str,
) -> list[str]:
    """The elementwise result-store tail (SPMD result-tail seam): copy each
    output's SBUF tiles back to HBM. The body still owns the ``return``."""
    lines: list[str] = [
        f"{i3}# Store result tiles back to HBM",
        f"{i3}for tile_m in nl.affine_range(TILES_IN_BLOCK_M):",
        f"{i4}{row_stmt}",
    ]
    for buf_name, tile_var in zip(output_buf_names, output_tile_vars, strict=True):
        if not tile_var:
            continue
        lines += [
            f"{i4}nisa.dma_copy(",
            f"{i4}    dst={buf_name}[row:row + TILE_M, {col0}:{col0} + {width}],",
            f"{i4}    src={tile_var}[0:TILE_M, tile_m, 0:{width}])",
        ]
    lines.append("")
    return lines
