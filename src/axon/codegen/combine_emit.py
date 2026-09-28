"""Monoid-adjacent combine emission: the NKI line-templates a `ShardingPlan`'s
cross-core combine lowers to, emitted directly through the `emit_lnc2` pipeline.

These are the (device-validated) sequences the prototype's §3 placed "next to the
monoid library's `combine_emit` hooks": a P free-axis barrier, a C tiled
all-reduce epilogue, and the S-moment accumulator all-reduce + column gather.
The sequences themselves are unchanged from the original source-patching
assembler (`spmd.py`); only their delivery moved from regex-over-emitted-text to
direct splice points `emit_lnc2` drives via `_combine_for(plan)` /
`_shard_prelude(plan)`.

Each function takes the assembled single-core body `lines` and returns the
transformed `lines`; the assembler joins them. Names are minted with disjoint
`_sm_` / `_ar_` / `_g_` prefixes so a combine never collides with body-emitted
names. The plan analysis that *chooses* which combine to emit (`_AXIS_LOOPS`,
`_sharded_output_axis`, `_terminal_combine`, `_operand_slices`) lives alongside
in `axon.codegen.combine_plan`; this module is purely the line emission.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from axon.codegen.ops import nki_safe_var


class SpmdEmitError(Exception):
    """A combine the emitter cannot lower (an unsupported monoid, or a shard
    axis that maps to no block loop)."""


@dataclass(frozen=True)
class _ShardAxis:
    """Which emitted block loop a sharded output axis maps to."""

    loop_var: str  # the loop induction variable, e.g. "m"
    num_block: str  # its bound, e.g. "NUM_BLOCK_M"


# The inlined `get_program_sharding_info()` body (kernel_helpers.py): grid_ndim,
# n_prgs on axis 0, this program's id. (0, 1, 0) when not SPMD.
_SHARD_PREAMBLE = [
    "_grid_ndim = nl.program_ndim()",
    "if _grid_ndim != 0:",
    "    N_PRGS, SHARD_ID = nl.num_programs(axes=0), nl.program_id(axis=0)",
    "else:",
    "    N_PRGS, SHARD_ID = 1, 0",
]


def _insert_preamble(lines: list[str], indent: str) -> list[str]:
    """Insert the shard preamble as the first statements of the function body."""
    def_idx = next(
        i
        for i, ln in enumerate(lines)
        if ln.startswith("def ") and ln.rstrip().endswith(":")
    )
    pre = [indent + ln for ln in _SHARD_PREAMBLE]
    return lines[: def_idx + 1] + pre + lines[def_idx + 1 :]


def _slice_operands(lines: list[str], sliced: dict[str, int], indent: str) -> list[str]:
    """Rebind each contraction-sharded input to its `SHARD_ID` half on the
    sharded dim, right after the shard preamble. The body reads its loop bounds
    from the (now halved) operand `.shape`, so it computes the per-core partial
    with no body edits. The sharded dim must be divisible by the program count —
    asserted at trace time (the input shape is a compile-time constant)."""
    # Insert after the whole preamble if/else (its last line is the else-branch
    # `N_PRGS, SHARD_ID = 1, 0`, indented one level deeper than the function
    # body — so the binds go at function-body indent, *after* that line).
    else_line = next(
        i for i, ln in enumerate(lines) if ln.strip() == "N_PRGS, SHARD_ID = 1, 0"
    )
    anchor = else_line
    # Function-body indent = the `if _grid_ndim` line's indent (one level out).
    if_line = next(
        i for i, ln in enumerate(lines) if ln.strip().startswith("if _grid_ndim")
    )
    body_ind = lines[if_line][: len(lines[if_line]) - len(lines[if_line].lstrip())]
    binds: list[str] = []
    for inp, dim in sorted(sliced.items()):
        v = nki_safe_var(inp)
        ext = f"{v}.shape[{dim}]"
        # Index the sharded dim with the half-range, leaving other dims full.
        index = ", ".join(
            f"_{v}_lo:_{v}_hi" if d == dim else ":" for d in range(dim + 1)
        )
        binds += [
            f"{body_ind}assert {ext} % N_PRGS == 0, "
            f'"{v} dim {dim} must be divisible by the program count to shard"',
            f"{body_ind}_{v}_per = {ext} // N_PRGS",
            f"{body_ind}_{v}_lo = SHARD_ID * _{v}_per",
            f"{body_ind}_{v}_hi = _{v}_lo + _{v}_per",
            f"{body_ind}{v} = {v}[{index}]",
        ]
    return lines[: anchor + 1] + binds + lines[anchor + 1 :]


def _shard_block_loop(lines: list[str], shard: _ShardAxis, indent: str) -> list[str]:
    """Rebind the first `for <loop_var> in nl.affine_range(<num_block>):` so each
    program iterates only its contiguous share of the blocks. The loop var keeps
    its global meaning (`<loop_var> = <loop_var>_shard + SHARD_ID * per_core`),
    so every body reference to it — input reads and output writes alike — is
    unchanged and addresses exactly the core's slice."""
    pat = re.compile(
        rf"^(\s*)for\s+{re.escape(shard.loop_var)}\s+in\s+"
        rf"nl\.affine_range\({re.escape(shard.num_block)}\):\s*$"
    )
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if not m:
            continue
        loop_ind = m.group(1)
        body_ind = loop_ind + indent
        v, nb = shard.loop_var, shard.num_block
        per = f"({nb} // N_PRGS)"
        # `NUM_BLOCK_*` (tile args baked into the NEFF) and `N_PRGS` (the NEFF's
        # fixed program count) are both compile-time constants, so this assert
        # fires at trace/compile time — a non-divisible block count fails the
        # build loudly, never silently drops a block on device.
        replacement = [
            f"{loop_ind}assert {nb} % N_PRGS == 0, "
            f'"{nb} must be divisible by the program count to shard"',
            f"{loop_ind}for {v}_shard in nl.affine_range({per}):",
            f"{body_ind}{v} = {v}_shard + SHARD_ID * {per}",
        ]
        return lines[:i] + replacement + lines[i + 1 :]
    raise SpmdEmitError(
        f"no block loop `for {shard.loop_var} in nl.affine_range({shard.num_block})` "
        f"found to shard"
    )


def _append_barrier(lines: list[str], indent: str) -> list[str]:
    """Insert `nisa.core_barrier(<ret>, (0, 1))` immediately before the kernel's
    `return <ret>` so both cores' disjoint writes are visible downstream."""
    for i in range(len(lines) - 1, -1, -1):
        m = re.match(r"^(\s*)return\s+(\w+)\s*$", lines[i])
        if not m:
            continue
        ret_ind, ret_var = m.group(1), m.group(2)
        barrier = f"{ret_ind}nisa.core_barrier({ret_var}, (0, 1))"
        return lines[:i] + [barrier, lines[i]]
    raise SpmdEmitError("no `return <var>` found to barrier")


def _result_var(lines: list[str]) -> str:
    """The name the body allocates its `shared_hbm` output under (`result` for
    matmul, `output` elsewhere)."""
    for ln in lines:
        m = re.match(r"^\s*(\w+)\s*=\s*nl\.ndarray\(.*nl\.shared_hbm\)", ln)
        if m:
            return m.group(1)
    raise SpmdEmitError("no shared_hbm result allocation found to privatize")


def _rebuffer_result(lines: list[str], res_var: str) -> list[str]:
    """Point the body's result alloc at `private_hbm` (a per-core partial)."""
    out = []
    for ln in lines:
        if re.match(rf"^\s*{re.escape(res_var)}\s*=\s*nl\.ndarray\(", ln):
            out.append(ln.replace("nl.shared_hbm", "nl.private_hbm"))
        else:
            out.append(ln)
    return out


def _emit_allreduce_epilogue(
    lines: list[str], res_var: str, nl_op: str, indent: str
) -> list[str]:
    """Swap the body's `return <res>` for: allocate a shared output, then
    DMA-in / sendrecv / merge / DMA-out the partial in `(128, _AR_NTILE)` SBUF
    tiles, and return the shared output. Tiles over BOTH the partition (row) dim
    in 128-row strips (the NeuronCore partition limit) AND the free (column) dim
    in `_AR_NTILE`-wide strips, so the SBUF tile stays bounded for any N (a full
    un-tiled `(128, N)` tile would overflow the per-partition SBUF budget at
    large N)."""
    for i in range(len(lines) - 1, -1, -1):
        m = re.match(rf"^(\s*)return\s+{re.escape(res_var)}\s*$", lines[i])
        if not m:
            continue
        ind = m.group(1)
        b1 = ind + indent
        b2 = b1 + indent
        rs = "_ar_r * _AR_TILE"
        cs = "_ar_c * _AR_NTILE"
        epi = [
            f"{ind}_ar_M, _ar_N = {res_var}.shape",
            f"{ind}_ar_out = nl.ndarray("
            f"(_ar_M, _ar_N), dtype={res_var}.dtype, buffer=nl.shared_hbm)",
            f"{ind}_AR_TILE = 128",
            f"{ind}_AR_NTILE = _ar_N if _ar_N <= 512 else (512 if _ar_N % 512 == 0 else 128)",
            f"{ind}assert _ar_M % _AR_TILE == 0, "
            '"all-reduce row dim must be a multiple of 128"',
            f"{ind}assert _ar_N % _AR_NTILE == 0, "
            '"all-reduce free dim must be a multiple of the column tile"',
            f"{ind}for _ar_r in nl.affine_range(_ar_M // _AR_TILE):",
            f"{b1}for _ar_c in nl.affine_range(_ar_N // _AR_NTILE):",
            f"{b2}_ar_loc = nl.ndarray("
            "(_AR_TILE, _AR_NTILE), dtype=_ar_out.dtype, buffer=nl.sbuf)",
            f"{b2}_ar_peer = nl.ndarray("
            "(_AR_TILE, _AR_NTILE), dtype=_ar_out.dtype, buffer=nl.sbuf)",
            f"{b2}nisa.dma_copy(dst=_ar_loc[0:_AR_TILE, 0:_AR_NTILE], "
            f"src={res_var}[{rs}:{rs} + _AR_TILE, {cs}:{cs} + _AR_NTILE])",
            f"{b2}nisa.sendrecv(dst=_ar_peer[0:_AR_TILE, 0:_AR_NTILE], "
            "src=_ar_loc[0:_AR_TILE, 0:_AR_NTILE], send_to_rank=1 - SHARD_ID, "
            "recv_from_rank=1 - SHARD_ID, pipe_id=0)",
            f"{b2}nisa.tensor_tensor(_ar_loc[0:_AR_TILE, 0:_AR_NTILE], "
            f"_ar_loc[0:_AR_TILE, 0:_AR_NTILE], _ar_peer[0:_AR_TILE, 0:_AR_NTILE], "
            f"nl.{nl_op})",
            f"{b2}nisa.dma_copy("
            f"dst=_ar_out[{rs}:{rs} + _AR_TILE, {cs}:{cs} + _AR_NTILE], "
            "src=_ar_loc[0:_AR_TILE, 0:_AR_NTILE])",
            f"{ind}return _ar_out",
        ]
        return lines[:i] + epi
    raise SpmdEmitError(f"no `return {res_var}` found for the all-reduce epilogue")


def _allreduce_accumulator(lines: list[str], nl_op: str, indent: str) -> list[str]:
    """Insert a per-row-tile `sendrecv` + merge of the reduce body's `accum`
    statistic, right before the finalize loop. `accum` is `(TILE_M,
    TILES_IN_BLOCK_M, 1)` in SBUF, so the all-reduce is one small swap+merge per
    tile.

    The finalize loop is anchored *structurally* as the first
    `for tile_m in nl.affine_range(TILES_IN_BLOCK_M):` that is a **sibling** of
    the reduce-phase accumulation loop (`for n in nl.sequential_range(...)`) —
    i.e. at the *same indentation*, so it sits *after* the n-loop closes, not a
    tile_m loop nested *inside* it (the load / accumulate loops). Anchoring by
    indentation (not the finalize op name) is robust to which activation the
    finalize uses (sqrt / rsqrt / reciprocal / …) and never lands mid-reduce."""
    reduce_phase = next(
        (
            i
            for i, ln in enumerate(lines)
            if re.match(r"^(\s*)for n in nl\.sequential_range\(NUM_BLOCK_N\):", ln)
        ),
        None,
    )
    if reduce_phase is None:
        raise SpmdEmitError(
            "S-moment: no reduce-phase `for n in nl.sequential_range(NUM_BLOCK_N)`"
        )
    n_indent = len(lines[reduce_phase]) - len(lines[reduce_phase].lstrip())
    # The finalize loop is the first `for tile_m` at the n-loop's own indent
    # (a sibling after it), not one nested deeper inside the reduce phase.
    loop_candidates = [
        i
        for i in range(reduce_phase + 1, len(lines))
        if re.match(
            r"^\s*for tile_m in nl\.affine_range\(TILES_IN_BLOCK_M\):", lines[i]
        )
        and (len(lines[i]) - len(lines[i].lstrip())) == n_indent
    ]
    if not loop_candidates:
        raise SpmdEmitError("S-moment: no finalize `for tile_m` loop after reduce")
    loop_i = loop_candidates[0]
    loop_ind = lines[loop_i][: len(lines[loop_i]) - len(lines[loop_i].lstrip())]
    body = loop_ind + indent
    swap = [
        f"{loop_ind}_sm_peer = nl.ndarray(accum.shape, dtype=accum.dtype, buffer=nl.sbuf)",
        f"{loop_ind}for _sm_t in nl.affine_range(TILES_IN_BLOCK_M):",
        f"{body}nisa.sendrecv(dst=_sm_peer[0:accum.shape[0], _sm_t, 0:1], "
        "src=accum[0:accum.shape[0], _sm_t, 0:1], send_to_rank=1 - SHARD_ID, "
        "recv_from_rank=1 - SHARD_ID, pipe_id=0)",
        f"{body}nisa.tensor_tensor(accum[0:accum.shape[0], _sm_t, 0:1], "
        f"accum[0:accum.shape[0], _sm_t, 0:1], _sm_peer[0:accum.shape[0], _sm_t, 0:1], "
        f"nl.{nl_op})",
    ]
    return lines[:loop_i] + swap + lines[loop_i:]


def _gather_output_columns(lines: list[str], res: str, indent: str) -> list[str]:
    """Gather each core's half-width `output` (its disjoint H-columns) into a
    full-width shared output. The body's `output` is allocated from the *sliced*
    input, so it is `(M, N/2)` in `private_hbm` (per core). Each core
    DMA-copies its own half to the corresponding column offset in a shared
    `(M, N)` output, tiled in `(128, _G_NTILE)` chunks so the SBUF tile stays
    bounded. No sendrecv: pure per-core placement, barriered at the end."""
    for i in range(len(lines) - 1, -1, -1):
        m = re.match(rf"^(\s*)return\s+{re.escape(res)}\s*$", lines[i])
        if not m:
            continue
        ind = m.group(1)
        b1 = ind + indent
        b2 = b1 + indent
        rs = "_g_r * _G_TILE"
        cs = "_g_c * _G_NTILE"
        epi = [
            f"{ind}_g_M, _g_half = {res}.shape",
            f"{ind}_g_full = nl.ndarray("
            f"(_g_M, _g_half * N_PRGS), dtype={res}.dtype, buffer=nl.shared_hbm)",
            f"{ind}_g_self = SHARD_ID * _g_half",
            f"{ind}_G_TILE = 128",
            f"{ind}_G_NTILE = _g_half if _g_half <= 512 else (512 if _g_half % 512 == 0 else 128)",
            f"{ind}assert _g_M % _G_TILE == 0, "
            '"gather row dim must be a multiple of 128"',
            f"{ind}assert _g_half % _G_NTILE == 0, "
            '"gather col dim must be a multiple of the column tile"',
            f"{ind}for _g_r in nl.affine_range(_g_M // _G_TILE):",
            f"{b1}for _g_c in nl.affine_range(_g_half // _G_NTILE):",
            f"{b2}_g_loc = nl.ndarray("
            "(_G_TILE, _G_NTILE), dtype=_g_full.dtype, buffer=nl.sbuf)",
            f"{b2}nisa.dma_copy(dst=_g_loc[0:_G_TILE, 0:_G_NTILE], "
            f"src={res}[{rs}:{rs} + _G_TILE, {cs}:{cs} + _G_NTILE])",
            f"{b2}nisa.dma_copy("
            f"dst=_g_full[{rs}:{rs} + _G_TILE, _g_self + {cs}:_g_self + {cs} + _G_NTILE], "
            "src=_g_loc[0:_G_TILE, 0:_G_NTILE])",
            f"{ind}nisa.core_barrier(_g_full, (0, 1))",
            f"{ind}return _g_full",
        ]
        return lines[:i] + epi
    raise SpmdEmitError(f"no `return {res}` found for the S-moment output gather")
