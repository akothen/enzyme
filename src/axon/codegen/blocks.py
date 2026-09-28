"""Shared line-emitting primitives for the body builders."""

from __future__ import annotations

from axon.codegen.context import EmitCtx


def substitute_dst(
    call: str, *, dst: str, dst_slice: str, dst_shape: str, ind: str
) -> str:
    """Substitute an op emitter's ``{DST}`` / ``{DST_SLICE}`` / ``{DST_SHAPE}`` /
    ``{IND}`` placeholders with the destination tile expression, slice, shape, and
    line indent."""
    return (
        call.replace("{DST_SHAPE}", dst_shape)
        .replace("{DST_SLICE}", dst_slice)
        .replace("{DST}", dst)
        .replace("{IND}", ind)
    )


def nc_transpose_psum(
    ctx: EmitCtx,
    dst_slice: str,
    src_slice: str,
    src_dtype: str,
    p: str,
    f: str,
    indent: str,
) -> list[str]:
    """Emit an ``nc_transpose`` as a PSUM-scratch transpose followed by a
    ``nisa.tensor_copy`` into the SBUF ``dst_slice`` (the PE array can only
    write its transpose result to PSUM). Mints ``tp_psum_<n>`` from
    ``ctx.tp_counter`` (mutated in place to keep the numbering continuous)."""
    tp = f"tp_psum_{ctx.tp_counter}"
    ctx.tp_counter += 1
    tp_sl = f"{tp}[0:{p}, 0:{f}]"
    return [
        f"{indent}{tp} = nl.ndarray(({p}, {f}), dtype={src_dtype}, buffer=nl.psum)",
        f"{indent}nisa.nc_transpose({tp_sl}, {src_slice})",
        f"{indent}nisa.tensor_copy({dst_slice}, {tp_sl})",
    ]
