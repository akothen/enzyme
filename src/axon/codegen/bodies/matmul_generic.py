"""Generic matmul-family body RENDERER (orientation-aware).

Pure translator of an ``EmissionPlan`` (``codegen/plan.py``): the plan owns
all graph interpretation; this module holds only a restricted ``NodeAttrs``
view (op + attrs, no edges) and translates plan fields into NKI lines.
Unsupported structure is refused at plan-build time before any emission.
Every subscript derives from ``Layout`` + ``Nest``; no literal index strings
live outside the two idiom helpers here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from axon.candidate_filter import TileConstraint
from axon.codegen.blocks import nc_transpose_psum, substitute_dst
from axon.codegen.context import EmitCtx
from axon.codegen.layout import DimVar, Layout
from axon.codegen.ops import (
    UnsupportedEmission,
    _nki_op_ref,
    nki_safe_var,
)
from axon.codegen.plan import (
    ChainStep,
    EmissionPlan,
    EmissionStagePlan,
    InterNodePlan,
    MatmulPlan,
    NodeAttrs,
    OperandPlan,
    ReducePlan,
    _assert_mm_operand_roles,
    _input_ancestors,
    _ordered_dims,
    build_emission_plan,
    find_output_sink,
)
from axon.ir import Node
from axon.isa_semantics import engine, matmul_perf_mode, reduce_cmd

# Re-exported for the operand-role regression test (the assertion now lives in
# the plan builder, where roles are decided; kept importable here for the tests
# that pin the class-guarding invariant).
__all__ = [
    "emit_matmul_body_generic",
    "_assert_mm_operand_roles",
    "_input_ancestors",
]

# Unary elementwise ops (shared with the plan; used only for the reduce-preamble
# activation-fusion test at line-emission time).
_ELEMENTWISE_OPS = frozenset({"activation", "exponential", "reciprocal"})

# The nc-transpose strategies swept by the assembler (the t-axis). All three
# are numerically identical schedule choices; unknown values refuse.
_TRANSPOSE_STRATEGIES = ("same_loop", "separate_loop", "load_transpose2d")

# The schedule dims whose `TILES_IN_BLOCK_<D>` the emitted matmul signature
# binds (`assemble.py` writes the params; `p` only when the graph is chained,
# and passing `p` unused is harmless). A dim outside this set has no bound tile
# parameter, so `_GenericMM.tv` refuses it rather than emit a NameError.
_SIGNATURE_TILE_DIMS = frozenset({"m", "n", "k", "p"})

# The one reduction monoid any recorded graph carries: (cross-block combiner,
# accumulator identity). A non-add reduce_op refuses rather than accumulate wrong.
_REDUCE_ACCUMULATORS = {"nl.add": ("nl.add", "0.0")}


@dataclass
class _TileVars:
    """Emitted tiling names for one canonical math dim: ``TILE_<D>``,
    ``TILES_IN_BLOCK_<D>``, ``BLOCK_<D>``, ``NUM_BLOCK_<D>``, extent, loop var."""

    dim: str

    @property
    def U(self) -> str:  # noqa: N802 - short accessor
        return self.dim.upper()

    @property
    def tile(self) -> str:
        return f"TILE_{self.U}"

    @property
    def matmul_tile(self) -> str:
        """Wide inner-matmul tile for dual-role dims (``tile_roles[D] == ("part","wide")``).
        Moving slices use ``MATMUL_TILE_<D>``; partition sites use ``TILE_<D>``."""
        return f"MATMUL_TILE_{self.U}"

    @property
    def tiles_in_matmul_block(self) -> str:
        return f"TILES_IN_MATMUL_BLOCK_{self.U}"

    @property
    def tiles_in_block(self) -> str:
        return f"TILES_IN_BLOCK_{self.U}"

    @property
    def block(self) -> str:
        return f"BLOCK_{self.U}"

    @property
    def num_block(self) -> str:
        return f"NUM_BLOCK_{self.U}"

    @property
    def extent(self) -> str:
        return self.U

    @property
    def loop(self) -> str:
        return self.dim


def _is_enum_default(value: object, default: Enum) -> bool:
    """``default``, as the member itself (what a recorded graph carries) or as its
    bare member name (what a hand-built graph carries). No other spelling."""
    return value is default or value == default.name


# `tensor_scalar` and `scalar_tensor_tensor` take the same operand attributes and
# `codegen/ops.py` renders both the same way.
_SCALAR_OPERAND_ATTRS = frozenset(
    {
        "op0",
        "op1",
        "reverse0",
        "reverse1",
        "operand0_input_index",
        "operand1_input_index",
        "operand0_const",
        "operand1_const",
    }
)

# The attributes each op emitter in ``codegen/ops.py`` actually renders. Any
# other attribute on a staged node is dropped, so `_has_only_default_isa_attrs`
# refuses it unless it sits at a known no-op default. An op absent from this
# table is not checked, so add an entry when the staged body starts rendering it.
_RENDERED_ISA_ATTRS: dict[str, frozenset[str]] = {
    "nc_matmul": frozenset(),
    "nc_transpose": frozenset(),
    "exponential": frozenset(),
    "reciprocal": frozenset(),
    "activation": frozenset(
        {"op", "scale", "bias_const", "bias_input_index", "scale_input_index"}
    ),
    "tensor_reduce": frozenset({"op", "axis", "negate", "keepdims", "keep_dims"}),
    "tensor_tensor": frozenset({"op"}),
    "tensor_copy": frozenset(),
    "tensor_scalar": _SCALAR_OPERAND_ATTRS,
    "scalar_tensor_tensor": _SCALAR_OPERAND_ATTRS,
}


def _has_only_default_isa_attrs(op: str, attrs) -> dict[str, object]:
    """The attributes of a node that its op emitter DROPS and that do not sit at
    their no-op default, empty when the node carries none.

    A dropped non-default attribute would silently change what the kernel
    computes, so the staged body refuses instead. Every attribute is checked:
    one the emitter renders is listed in ``_RENDERED_ISA_ATTRS`` and skipped, and
    one no default is known for refuses rather than passing unexamined. The
    defaults below are the ones the ISA signatures set and recorded graphs carry;
    an unlisted key refuses, which is the safe direction."""

    def is_no_op(key: str, value: object) -> bool:
        if key == "name":
            return value is None
        if key == "engine":
            return _is_enum_default(value, engine.unknown)
        if key == "out_shape":
            return True
        if key == "accumulate":
            return value is None or value is False
        if key in {"is_moving_onezero", "is_stationary_onezero", "is_transpose"}:
            return value is False
        if key in {"tile_position", "tile_size"}:
            return value == () or value == []
        if key == "perf_mode":
            return _is_enum_default(value, matmul_perf_mode.none)
        if key in {"max_value", "reduce_init"}:
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value == 0
            )
        if key in {"reduce_op", "bias_const"}:
            return value is None
        if key == "reduce_cmd":
            return _is_enum_default(value, reduce_cmd.idle)
        # An explicit None on a concretely-defaulted flag is not the default: it
        # says the graph meant to set something the emitter would drop.
        if key == "with_reduce":
            return value is False
        # No default is known for this attribute, so it cannot be dismissed.
        return False

    rendered = _RENDERED_ISA_ATTRS.get(op)
    if rendered is None:
        return {}
    return {
        key: value
        for key, value in attrs.items()
        if key not in rendered and not is_no_op(key, value)
    }


class _GenericMM:
    """Shared render state: the frozen ``EmissionPlan``, the derived nest, the
    layouts, and the tile-var namespace. Carries scalar/staged-buffer bookkeeping
    filled during emission (the preamble writes ``scalar_bufs`` / ``staged_bufs``,
    the operand chain reads them). All render helpers thread through this
    object; structure comes only from ``self.plan``."""

    def __init__(
        self,
        ctx: EmitCtx,
        plan: EmissionPlan,
        node_attrs: NodeAttrs,
        tile_dims: frozenset[str] = _SIGNATURE_TILE_DIMS,
    ) -> None:
        self.ctx = ctx
        self.plan = plan
        self.nest = plan.nest
        self.info = plan.nest.info
        self.node_attrs = node_attrs  # (op, attrs) only — no edges
        # The schedule dims whose TILES_IN_BLOCK_<D> the emitted signature binds.
        self.tile_dims = tile_dims
        self._tv: dict[str, _TileVars] = {}
        self._reserved_names = {nki_safe_var(node_id) for node_id in plan.layouts}
        self._generated_names: set[str] = set()
        self.scalar_ids: set[str] = set(plan.scalar_ids)
        self.staged_dtype_sources = dict(plan.staged_dtype_sources)
        if len(self.staged_dtype_sources) != len(plan.staged_dtype_sources):
            raise UnsupportedEmission(
                "generic matmul: staged endpoints have duplicate dtype sources"
            )
        # reduce scalar id -> (TILE_M, TILES_IN_BLOCK_M, 1) buffer, filled by preamble.
        self.scalar_bufs: dict[str, str] = {}
        # staged chain node id -> m-level spanning buffer, filled in k_side pass.
        self.staged_bufs: dict[str, str] = {}
        # Stage-boundary node id -> temporary shared-HBM variable.
        self.hbm_bufs: dict[str, str] = {}
        # Stage-crossing node id -> SBUF buffer spanning its whole lifetime.
        # Checked before `hbm_bufs` everywhere a crossing value is resolved: a
        # resident value has no HBM buffer at all, so the lookup would KeyError.
        self.resident_bufs: dict[str, str] = {}
        # Resident node ids whose buffer widens their PART axis rather than their
        # free axis (the default). Set by whichever caller hoists the loop.
        self.part_spanning: set[str] = set()
        self.dim_aliases: dict[str, str] = {}
        self.role_dims: dict[str, DimVar] = {}
        for dim, role in self.info.canon.items():
            prior = self.role_dims.get(role)
            if prior is not None and not prior.same_extent(dim):
                raise UnsupportedEmission(
                    "generic matmul: canonical schedule role "
                    f"{role!r} has unequal extents"
                )
            self.role_dims.setdefault(role, dim)
        # Every staged plan carries the role collapse (`_build_multi_stage`
        # populates dim_aliases/role_dims), whether or not it has a reduce — a
        # bare matmul→matmul chain stages with none. Keying on `plan.stages`
        # rather than `plan.reduces` keeps its `k2`→`p` aliases from being
        # dropped, which would leak a raw dim into `tv()`.
        if plan.is_multi and plan.stages:
            self.dim_aliases = dict(plan.dim_aliases)
            self.role_dims = dict(plan.role_dims)

    def accum_alloc(self, accum_dim: str) -> str:
        """The allocator for an accumulator drained by ``_emit_matmul_accumulate``:
        no memset at one contraction block, where the drain copies rather than
        adds. `NUM_BLOCK_<accum>` is trace-time, so the choice is an emitted
        conditional expression."""
        kt = self.tv(accum_dim)
        return f"(nl.ndarray if {kt.num_block} == 1 else nl.zeros)"

    # --- dim / layout helpers -------------------------------------------- #
    def dim_name(self, dv) -> str:
        """Canonical name (m/n/k/p) of a DimVar."""
        raw = self.info.canon[dv.find()]
        return self.dim_aliases.get(raw, raw)

    def alias_dim(self, dim: str) -> str:
        return self.dim_aliases.get(dim, dim)

    def tv(self, dim: str) -> _TileVars:
        """The tile names for one schedule dim, minted once per dim.

        This is the SOLE mint site for every ``TILE_<D>`` / ``TILES_IN_BLOCK_<D>``
        / ``BLOCK_<D>`` / ``NUM_BLOCK_<D>`` name the body emits, so it is also
        where a name that the emitted signature does not bind would be born.
        ``TILES_IN_BLOCK_<D>`` is a function parameter and nothing in the body
        assigns it, so a dim outside ``tile_dims`` (a schedule dim that escaped
        the m/n/k/p namespace and took a fallback ``analyze_dims`` name like
        ``d4``) would emit a module that dies with ``NameError`` at trace time.
        Refusing here drops that variant instead."""
        if dim not in self._tv:
            if dim not in self.tile_dims:
                raise UnsupportedEmission(
                    f"generic matmul: schedule dim {dim!r} is outside the tile "
                    f"namespace the emitted signature binds "
                    f"({', '.join(_ordered_dims(self.tile_dims))}), so "
                    f"TILES_IN_BLOCK_{dim.upper()} would never be bound"
                )
            self._tv[dim] = _TileVars(dim)
        return self._tv[dim]

    def layout(self, node_id: str) -> Layout:
        return self.info.layouts[node_id]

    def fresh_var(self, base: str) -> str:
        candidate = base
        suffix = 2
        while candidate in self._reserved_names or candidate in self._generated_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        self._generated_names.add(candidate)
        return candidate


def _chain_step_materializes(step: ChainStep) -> bool:
    """True when this chain step emits a call into a fresh buffer of its own.

    A ``broadcast`` is a pure shape reinterpretation: it emits nothing, so the
    value stays in the buffer its producer wrote."""
    return step.op != "broadcast"


def _resolved_tile(G: _GenericMM, aliased_dim: str, fallback: int) -> int:
    """The plan's resolved (possibly clamped) ``TILE_<D>`` for an aliased dim,
    keyed back through ``dim_aliases`` to the raw canon dim the plan tiled. The
    configured base when the plan named no tile, so extent >= base is unchanged."""
    raw = next((r for r, a in G.dim_aliases.items() if a == aliased_dim), aliased_dim)
    return G.plan.tiles.get(raw, fallback)


def _emit_dims_preamble(
    G: _GenericMM,
    ind: str,
    input_nodes: list[Node],
    out_id: str,
    present: set[str],
    out_dtype_var: str,
) -> list[str]:
    """Emit shape unpacks, result alloc, and per-dim TILE/BLOCK/NUM_BLOCK lines.
    Dual-role dims (``tile_roles[D] == ("part","wide")``) also get ``MATMUL_TILE_<D>``.
    ``input_nodes`` are read only for their ids (layout keys); no structural edges."""
    lines: list[str] = []
    bound: set[str] = set()
    bound_roles: set[str] = set()
    concrete_extents: dict[str, int] = {}

    def _axis_var(dv, is_free: bool) -> str:
        # None free only happens for per-partition scalars, never a raw input.
        name = G.dim_name(dv) if dv is not None else "_scalar"
        base = name.upper()
        if base in bound:
            return base + "_"
        bound.add(base)
        bound_roles.add(name)
        return base

    # Shape unpacks: one per input, in signature order.
    for inode in input_nodes:
        lay = G.layout(inode.id)
        var = nki_safe_var(inode.id)
        row = _axis_var(lay.part, is_free=False)
        col = _axis_var(lay.free, is_free=True)
        lines.append(f"{ind}{row}, {col} = {var}.shape")
        shape = tuple(inode.attrs.get("shape") or inode.shape or ())
        if len(shape) == 2 and lay.free is not None:
            for dim_var, size in ((lay.part, shape[0]), (lay.free, shape[1])):
                dim = G.dim_name(dim_var)
                prior = concrete_extents.get(dim)
                if prior is not None and prior != int(size):
                    raise UnsupportedEmission(
                        f"generic matmul: aliased dim {dim!r} has disagreeing "
                        f"extents ({prior} vs {int(size)})"
                    )
                concrete_extents[dim] = int(size)

    for dim in _ordered_dims(present):
        if dim in bound_roles:
            continue
        role_dim = G.role_dims.get(dim)
        source = next(
            (
                bound_dim
                for bound_dim in _ordered_dims(bound_roles)
                if role_dim is not None
                and G.role_dims.get(bound_dim) is not None
                and role_dim.same_extent(G.role_dims[bound_dim])
            ),
            None,
        )
        if source is None:
            raise UnsupportedEmission(
                f"generic matmul: no input extent binds schedule role {dim!r}"
            )
        lines.append(f"{ind}{dim.upper()} = {source.upper()}")
        bound_roles.add(dim)
        concrete_extents[dim] = concrete_extents[source]

    # Result HBM alloc off the OUTPUT layout: part extent x free extent.
    out_lay = G.layout(out_id)
    out_part = G.tv(G.dim_name(out_lay.part))
    out_free = G.tv(G.dim_name(out_lay.free))
    lines.append(
        f"{ind}result = nl.ndarray(({out_part.extent}, {out_free.extent}), "
        f"dtype={out_dtype_var}.dtype, buffer=nl.shared_hbm)"
    )
    lines.append("")

    # Per-dim tiling: each TILE_<D> is the plan's resolved value, emitted flat
    # (the plan owns the choice). The divisibility assert guards the floor of the
    # NUM_BLOCK division, which would silently drop the trailing rows.
    tile_roles: dict[str, set[str]] = {}
    for dim, roles in G.plan.tile_roles.items():
        tile_roles.setdefault(G.alias_dim(dim), set()).update(roles)
    # The plan keys tiles by RAW canon dim, but `present` is already aliased into
    # the m/n/k/p namespace, and which raw dim carries a role varies by graph
    # shape (a direct key projection carries `n`'s role on `k2`). So select each
    # aliased dim's base by its ROLE here, the same rule `_resolve_tiles` applies
    # to raw dims, and fall back to the plan's raw entry when the dim is unaliased.
    wide_dim = G.alias_dim(G.plan.wide_dim) if G.plan.wide_dim is not None else None
    tile_n = G.ctx.tile("tile_n")
    for dim in _ordered_dims(present):
        tv = G.tv(dim)
        roles = tile_roles.get(dim, set())
        # A dual-role dim's partition sites cap it at tile_m; MATMUL_TILE_<D>
        # carries the wide granularity. MLP plans hold the dual-role dim too but
        # carry no reduce, so the conjunction excludes them.
        if G.plan.reduces and {"part", "wide"}.issubset(roles):
            tile = G.ctx.tile("tile_m")
        elif dim == wide_dim:
            tile = _resolved_tile(G, dim, tile_n)
        elif dim == "k":
            tile = _resolved_tile(G, dim, G.ctx.tile("tile_k"))
        elif dim in G.plan.tiles:
            tile = G.plan.tiles[dim]
        else:
            tile = G.ctx.tile("tile_m")
        G.ctx.tile_constraints[tv.tiles_in_block] = TileConstraint(
            tile_arg=tv.tiles_in_block,
            extent=concrete_extents[dim],
            tile_size=tile,
            require_divisible=True,
        )
        lines += [
            f"{ind}{tv.tile} = {tile}",
            f"{ind}{tv.block} = {tv.tile} * {tv.tiles_in_block}",
            f"{ind}assert {tv.extent} % {tv.block} == 0, "
            f'"{tv.extent} must be divisible by {tv.block}"',
            f"{ind}{tv.num_block} = {tv.extent} // {tv.block}",
            f"{ind}assert {tv.num_block} > 0, "
            f'"{tv.extent} too small for the tile configuration"',
            "",
        ]
        # Dual-role dim: MATMUL_TILE_<D> (wide, tiles at tile_n, clamped to
        # divide BLOCK_<D>) serves moving slices; TILE_<D> serves partition sites.
        # The clamp stays renderer-local: it subdivides an already-chosen
        # BLOCK_<D> rather than rewriting the searched outer tiling.
        if {"part", "wide"}.issubset(roles):
            lines += [
                f"{ind}{tv.matmul_tile} = {tile_n}",
                f"{ind}{tv.tiles_in_matmul_block} = {tv.block} // {tv.matmul_tile}",
                f"{ind}while {tv.tiles_in_matmul_block} == 0:",
                f"{ind}    {tv.matmul_tile} = {tv.matmul_tile} // 2",
                f"{ind}    {tv.tiles_in_matmul_block} = {tv.block} // {tv.matmul_tile}",
                f"{ind}assert {tv.tiles_in_matmul_block} > 0, "
                f'"Tile configuration results in zero tiles for matmul"',
                "",
            ]
    return lines


def _present_dims(G: _GenericMM) -> set[str]:
    """Every canonical dim the nest touches (placements ∪ accum loops), aliased
    into the fixed m/n/k/p namespace."""
    dims: set[str] = set()
    for placement in G.nest.placement.values():
        dims.update(G.alias_dim(dim) for dim in placement)
    dims.update(G.alias_dim(dim) for dim in G.nest.accum_loops.values())
    return dims


def _emit_chain_step(
    G: _GenericMM,
    node_id: str,
    id_to_var: dict[str, str],
    dst_buf: str,
    dst_slice: str,
    dst_shape: str,
    dtype_expr: str,
    ind: str,
) -> list[str]:
    """Emit one chain node into its OWN fresh SBUF buffer (never in place on a
    DMA-loaded block — the aliasing analyzer enforces this). The op emitter
    reads its inputs via ``id_to_var`` (the data predecessor's expr, plus any
    reduce-preamble scalar buffer slice for a combiner) and writes ``{DST}``."""
    call = G.node_attrs.emit(G.ctx, node_id, id_to_var)
    assert call is not None
    stmt = substitute_dst(
        call, dst=dst_slice, dst_slice=dst_slice, dst_shape=dst_shape, ind=ind
    )
    return [
        f"{ind}{dst_buf} = nl.ndarray({dst_shape}, dtype={dtype_expr}, buffer=nl.sbuf)",
        f"{ind}{stmt}",
    ]


def _emit_simple_transpose_load(
    G: _GenericMM,
    strategy: str,
    tile_var: str,
    src_var: str,
    dtype_expr: str,
    kt: _TileVars,
    ft: _TileVars,
    i: str,
    i1: str,
    i2: str,
) -> list[str]:
    """Realize the nc_transpose for a plain HBM block load (no pre-transpose
    ops, no staged reduce prefix). All three strategies produce the same
    ``tile_var`` ``(TILE_k, TILES_IN_BLOCK_k, BLOCK_free)`` and are numerically
    identical (schedule choice only):

      - ``same_loop``: DMA ``(TILE_free, BLOCK_k)`` into scratch, PSUM-transpose
        each ``(TILE_free, TILE_k)`` sub-tile — load and transpose interleaved.
      - ``separate_loop``: DMA into a persistent block buffer, then transpose
        in a second loop.
      - ``load_transpose2d``: ``nl.load_transpose2d`` each sub-tile from HBM."""
    # Row slice (free-tile) and whole-k-block col slice into the raw HBM input.
    row = (
        f"(({ft.tiles_in_block} * {ft.loop} + b_{ft.loop}) * {ft.tile}):"
        f"(({ft.tiles_in_block} * {ft.loop} + b_{ft.loop}) * {ft.tile}) + {ft.tile}"
    )
    kblock_col = f"({kt.block} * {kt.loop}):({kt.block} * {kt.loop}) + {kt.block}"
    dst_sl = (
        f"{tile_var}[0:{kt.tile}, b{kt.loop}_t,"
        f" (b_{ft.loop} * {ft.tile}):(b_{ft.loop} * {ft.tile}) + {ft.tile}]"
    )
    alloc = [
        f"{i}{tile_var} = nl.ndarray(",
        f"{i}    ({kt.tile}, {kt.tiles_in_block}, {ft.block}),",
        f"{i}    dtype={dtype_expr}, buffer=nl.sbuf)",
    ]
    if strategy == "same_loop":
        return alloc + [
            f"{i}for b_{ft.loop} in nl.affine_range({ft.tiles_in_block}):",
            f"{i1}tmp_{tile_var} = nl.ndarray(({ft.tile}, {kt.block}),"
            f" dtype={dtype_expr}, buffer=nl.sbuf)",
            f"{i1}nisa.dma_copy(",
            f"{i1}    dst=tmp_{tile_var}[0:{ft.tile}, 0:{kt.block}],",
            f"{i1}    src={src_var}[{row},",
            f"{i1}              {kblock_col}])",
            f"{i1}for b{kt.loop}_t in nl.affine_range({kt.tiles_in_block}):",
            *nc_transpose_psum(
                G.ctx,
                dst_sl,
                f"tmp_{tile_var}[0:{ft.tile},"
                f" (b{kt.loop}_t * {kt.tile}):(b{kt.loop}_t * {kt.tile})"
                f" + {kt.tile}]",
                dtype_expr,
                kt.tile,
                ft.tile,
                i2,
            ),
            "",
        ]
    if strategy == "separate_loop":
        return alloc + [
            f"{i}temp_{tile_var} = nl.ndarray(",
            f"{i}    ({ft.tile}, {ft.tiles_in_block}, {kt.block}),",
            f"{i}    dtype={dtype_expr}, buffer=nl.sbuf)",
            f"{i}for b_{ft.loop} in nl.affine_range({ft.tiles_in_block}):",
            f"{i1}nisa.dma_copy(",
            f"{i1}    dst=temp_{tile_var}[0:{ft.tile}, b_{ft.loop}, 0:{kt.block}],",
            f"{i1}    src={src_var}[{row},",
            f"{i1}              {kblock_col}])",
            "",
            f"{i}for b_{ft.loop} in nl.affine_range({ft.tiles_in_block}):",
            f"{i1}for b{kt.loop}_t in nl.affine_range({kt.tiles_in_block}):",
            *nc_transpose_psum(
                G.ctx,
                dst_sl,
                f"temp_{tile_var}[0:{ft.tile}, b_{ft.loop},"
                f" (b{kt.loop}_t * {kt.tile}):(b{kt.loop}_t * {kt.tile})"
                f" + {kt.tile}]",
                dtype_expr,
                kt.tile,
                ft.tile,
                i2,
            ),
            "",
        ]
    # load_transpose2d: direct HBM -> transposed SBUF sub-tile, per (free, k)
    # tile. The col slice narrows to the single k-tile the sub-tile covers.
    ktile_col = (
        f"({kt.block} * {kt.loop} + b{kt.loop}_t * {kt.tile}):"
        f"({kt.block} * {kt.loop} + b{kt.loop}_t * {kt.tile}) + {kt.tile}"
    )
    return alloc + [
        f"{i}for b_{ft.loop} in nl.affine_range({ft.tiles_in_block}):",
        f"{i1}for b{kt.loop}_t in nl.affine_range({kt.tiles_in_block}):",
        f"{i2}{dst_sl} = nl.load_transpose2d(",
        f"{i2}    {src_var}[{row},",
        f"{i2}              {ktile_col}])",
        "",
    ]


def _emit_operand_load(
    G: _GenericMM,
    ind_base: int,
    ind: str,
    op: OperandPlan,
    accum_dim: str,
    free_dim: str,
) -> tuple[list[str], str]:
    """Materialize a matmul operand into an SBUF tile laid out
    ``(TILE_<k>, TILES_IN_BLOCK_<k>, BLOCK_<free>)`` — contraction on the
    partition axis, the operand's free dim on the free axis.

    The operand chain (its ordered steps, its transpose index, its root input)
    comes ENTIRELY from ``op`` (an ``OperandPlan`` the analysis froze); this
    renderer never re-walks the graph. Orientation is decided by whether the
    plan's chain contains a transpose:
      - no transpose (layout already part=k): a plain ``dma_copy`` block load,
        then each elementwise op applied to the whole 3D tile into a fresh
        buffer.
      - one ``nc_transpose`` (raw input layout part=free, free=k): load the src
        block, apply the pre-transpose elementwise ops to the 2D block, PSUM-
        transpose each (TILE_free, TILE_k) sub-tile, then apply the post-
        transpose elementwise ops to the 3D tile.
    The buffer name is keyed on the operand node id, so a node emits once."""
    op_id = op.op_id
    kt = G.tv(accum_dim)
    ft = G.tv(free_dim)
    tile_var = f"{nki_safe_var(op_id)}_tiles"
    i = ind * ind_base
    i1 = ind * (ind_base + 1)
    i2 = ind * (ind_base + 2)

    strategy = G.ctx.rhs_transpose_strategy
    if strategy not in _TRANSPOSE_STRATEGIES:
        raise UnsupportedEmission(
            f"unknown rhs_transpose_strategy {strategy!r}; expected one of "
            f"{' / '.join(_TRANSPOSE_STRATEGIES)}"
        )

    steps = list(op.steps)
    tpos = op.transpose_index
    src_id = op.root_id

    # A resident root is already in SBUF in exactly the shape a block load would
    # produce, so slice it and emit no DMA. Checked before `hbm_bufs`, which holds
    # no entry for it. Only a bare root qualifies: a chain step would need a
    # destination buffer, and the existing paths below allocate one.
    if src_id in G.resident_bufs and not steps and tpos is None:
        return [
            f"{i}{tile_var} = {_resident_block_slice(G, src_id)}",
            "",
        ], tile_var

    src_var = G.hbm_bufs.get(src_id, nki_safe_var(src_id))
    root_is_staged = src_id in G.staged_bufs
    if src_id in G.plan.staged_ids and not root_is_staged:
        raise UnsupportedEmission(
            f"generic matmul: staged operand root {src_id!r} has no emitted buffer"
        )
    dtype_source = G.staged_dtype_sources.get(src_id, src_var)
    dtype_expr = f"{nki_safe_var(dtype_source)}.dtype"
    mt = G.tv("m")

    def _apply_ops(
        chain_steps: list[ChainStep],
        src_expr: str,
        dst_slice_fmt,
        dst_shape: str,
        ind_s: str,
        scalar_index_var: str | None = None,
    ) -> tuple[list[str], str]:
        """Emit each step in ``chain_steps`` into a fresh buffer, threading the
        previous buffer's slice as the next op's data source. A single-scalar
        combiner also reads its reduce-preamble ``(m,)`` scalar buffer, indexed
        by ``scalar_index_var`` (the current partition/m tile loop var). The step
        carries the exact data-predecessor and scalar edges the plan froze."""
        out: list[str] = []
        cur = src_expr
        for j, step in enumerate(chain_steps):
            if not _chain_step_materializes(step):
                continue
            buf = f"{tile_var}_c{j}"
            dst_slice = dst_slice_fmt(buf)
            id_to_var = {step.data_pred_id: cur}
            if step.scalar_id is not None:
                scal_id = step.scalar_id
                if scalar_index_var is None or scal_id not in G.scalar_bufs:
                    raise UnsupportedEmission(
                        f"generic matmul: combiner {step.node_id!r} consumes "
                        f"scalar {scal_id!r} where no (m,) scalar tile is in scope "
                        f"(partition axis is not m here)"
                    )
                id_to_var[scal_id] = (
                    f"{G.scalar_bufs[scal_id]}[0:{mt.tile}, {scalar_index_var}, 0]"
                )
            out += _emit_chain_step(
                G,
                step.node_id,
                id_to_var,
                buf,
                dst_slice,
                dst_shape,
                dtype_expr,
                ind_s,
            )
            cur = dst_slice
        return out, cur

    if tpos is not None:
        pre_ops = steps[:tpos]
        post_ops = steps[tpos + 1 :]
        staged_steps = [
            (index, step)
            for index, step in enumerate(pre_ops)
            if step.node_id in G.staged_bufs
        ]
        if len(staged_steps) > 1:
            raise UnsupportedEmission(
                f"generic matmul: operand {op.op_id!r} has multiple staged "
                "prefix endpoints"
            )
        staged_step = staged_steps[0] if staged_steps else None
        has_staged_source = root_is_staged or staged_step is not None
        # Simple case (no pre-transpose ops, no staged prefix): dispatch to
        # the strategy-swept simple helper; complex paths use a single idiom.
        if not has_staged_source and not pre_ops:
            lines = _emit_simple_transpose_load(
                G, strategy, tile_var, src_var, dtype_expr, kt, ft, i, i1, i2
            )
            final_var = tile_var
            if post_ops:

                def slice_fmt(b: str) -> str:
                    return f"{b}[0:{kt.tile}, 0:{kt.tiles_in_block}, 0:{ft.block}]"

                post_lines, last_expr = _apply_ops(
                    post_ops,
                    slice_fmt(tile_var),
                    slice_fmt,
                    f"({kt.tile}, {kt.tiles_in_block}, {ft.block})",
                    i,
                )
                lines += post_lines + [""]
                final_var = last_expr.split("[", 1)[0]
            return lines, final_var
        # Has pre-transpose ops or staged prefix: load per free-tile then
        # apply pre-ops, then PSUM-transpose each (TILE_free, TILE_k) sub-tile.
        lines = [
            f"{i}{tile_var} = nl.ndarray(",
            f"{i}    ({kt.tile}, {kt.tiles_in_block}, {ft.block}),",
            f"{i}    dtype={dtype_expr}, buffer=nl.sbuf)",
            f"{i}for b_{ft.loop} in nl.affine_range({ft.tiles_in_block}):",
        ]
        # The staged node id whose 4D spanning buffer `block_expr` currently
        # names, or None when `block_expr` is a plain 2D block. Structural: it
        # comes from the plan's staged registration, never from the emitted name.
        block_staged_id: str | None = None
        if staged_step is not None:
            staged_index, endpoint = staged_step
            staged_buf = G.staged_bufs.get(endpoint.node_id)
            if staged_buf is None:
                raise UnsupportedEmission(
                    f"generic matmul: staged endpoint {endpoint.node_id!r} "
                    "has no emitted buffer"
                )
            block_staged_id = endpoint.node_id
            block_expr = (
                f"{staged_buf}[0:{ft.tile}, {kt.loop}, b_{ft.loop}, 0:{kt.block}]"
            )
            pre_ops = pre_ops[staged_index + 1 :]
        elif root_is_staged:
            staged_buf = G.staged_bufs[src_id]
            block_staged_id = src_id
            block_expr = (
                f"{staged_buf}[0:{ft.tile}, {kt.loop}, b_{ft.loop}, 0:{kt.block}]"
            )
        else:
            lines += [
                f"{i1}tmp_{tile_var} = nl.ndarray(({ft.tile}, {kt.block}),"
                f" dtype={dtype_expr}, buffer=nl.sbuf)",
                f"{i1}nisa.dma_copy(",
                f"{i1}    dst=tmp_{tile_var}[0:{ft.tile}, 0:{kt.block}],",
                f"{i1}    src={src_var}[(({ft.tiles_in_block} * {ft.loop}"
                f" + b_{ft.loop})"
                f" * {ft.tile}):(({ft.tiles_in_block} * {ft.loop} + b_{ft.loop})"
                f" * {ft.tile}) + {ft.tile},",
                f"{i1}              ({kt.block} * {kt.loop}):({kt.block} * {kt.loop})"
                f" + {kt.block}])",
            ]
            block_expr = f"tmp_{tile_var}[0:{ft.tile}, 0:{kt.block}]"
        if pre_ops:
            # Scalar combiner index is b_{ft.loop}; valid only when ft.loop == "m".
            pre_lines, block_expr = _apply_ops(
                pre_ops,
                block_expr,
                lambda b: f"{b}[0:{ft.tile}, 0:{kt.block}]",
                f"({ft.tile}, {kt.block})",
                i1,
                scalar_index_var=(f"b_{ft.loop}" if ft.loop == "m" else None),
            )
            lines += pre_lines
            # Only a materializing pre-op moves the value into a fresh 2D buffer;
            # transparent ones leave the original 4D staged expression standing.
            if any(_chain_step_materializes(step) for step in pre_ops):
                block_staged_id = None
        block_buf = block_expr.split("[", 1)[0]
        if block_staged_id is not None:
            trans_src = (
                f"{block_buf}[0:{ft.tile}, {kt.loop}, b_{ft.loop},"
                f" (b{kt.loop}_t * {kt.tile}):(b{kt.loop}_t * {kt.tile})"
                f" + {kt.tile}]"
            )
        else:
            trans_src = (
                f"{block_buf}[0:{ft.tile},"
                f" (b{kt.loop}_t * {kt.tile}):(b{kt.loop}_t * {kt.tile})"
                f" + {kt.tile}]"
            )
        lines += [
            f"{i1}for b{kt.loop}_t in nl.affine_range({kt.tiles_in_block}):",
            *nc_transpose_psum(
                G.ctx,
                f"{tile_var}[0:{kt.tile}, b{kt.loop}_t,"
                f" (b_{ft.loop} * {ft.tile}):(b_{ft.loop} * {ft.tile}) + {ft.tile}]",
                trans_src,
                dtype_expr,
                kt.tile,
                ft.tile,
                i2,
            ),
            "",
        ]
        final_var = tile_var
        if post_ops:

            def slice_fmt(b: str) -> str:
                return f"{b}[0:{kt.tile}, 0:{kt.tiles_in_block}, 0:{ft.block}]"

            src3d = slice_fmt(tile_var)
            post_lines, last_expr = _apply_ops(
                post_ops,
                src3d,
                slice_fmt,
                f"({kt.tile}, {kt.tiles_in_block}, {ft.block})",
                i,
            )
            lines += post_lines + [""]
            final_var = last_expr.split("[", 1)[0]
        return lines, final_var

    # No transpose on the chain: layout is already (part=k, free=free_dim).
    lines = [
        f"{i}{tile_var} = nl.ndarray(",
        f"{i}    ({kt.tile}, {kt.tiles_in_block}, {ft.block}),",
        f"{i}    dtype={dtype_expr}, buffer=nl.sbuf)",
        f"{i}for b{kt.loop}_t in nl.affine_range({kt.tiles_in_block}):",
        f"{i1}nisa.dma_copy(",
        f"{i1}    dst={tile_var}[0:{kt.tile}, b{kt.loop}_t, 0:{ft.block}],",
        f"{i1}    src={src_var}[(({kt.tiles_in_block} * {kt.loop} + b{kt.loop}_t)"
        f" * {kt.tile}):(({kt.tiles_in_block} * {kt.loop} + b{kt.loop}_t)"
        f" * {kt.tile}) + {kt.tile},",
        f"{i1}         ({ft.block} * {ft.loop}):({ft.block} * {ft.loop})"
        f" + {ft.block}])",
        "",
    ]
    final_var = tile_var
    if steps:

        def slice_fmt(b: str) -> str:
            return f"{b}[0:{kt.tile}, 0:{kt.tiles_in_block}, 0:{ft.block}]"

        chain_lines, last_expr = _apply_ops(
            steps,
            slice_fmt(tile_var),
            slice_fmt,
            f"({kt.tile}, {kt.tiles_in_block}, {ft.block})",
            i,
        )
        lines += chain_lines + [""]
        final_var = last_expr.split("[", 1)[0]
    return lines, final_var


def _emit_matmul_accumulate(
    G: _GenericMM,
    ind_base: int,
    ind: str,
    mm_id: str,
    stat_tile: str,
    mov_tile: str,
    accum_dim: str,
    stat_free: str,
    mov_free: str,
    result_tiles: str,
    psum_var: str = "res_tile",
    stat_block_index: str | None = None,
    mov_wide: bool = False,
) -> list[str]:
    """PSUM-accumulate idiom: for each (stat-free, mov-free) tile pair, accumulate
    ``nc_matmul`` contraction sub-tiles into a PSUM scratch and add into the SBUF
    block accumulator. ``psum_var`` is uniquified per matmul in multi-matmul graphs.

    ``stat_block_index``: extra block-index subscript for a materialized combine
    buffer (chained matmul's stationary operand indexed by the n-block loop var).
    ``mov_wide``: use ``MATMUL_TILE_<mov>`` for the moving free tile (dual-role dim).

    PSUM is allocated unzeroed with the first sub-tile at ``accumulate=False``, so
    no memset is emitted; the drain branches on `NUM_BLOCK_<accum>` at trace time.
    """
    kt = G.tv(accum_dim)
    st = G.tv(stat_free)
    mt = G.tv(mov_free)
    m_tile = mt.matmul_tile if mov_wide else mt.tile
    m_count = mt.tiles_in_matmul_block if mov_wide else mt.tiles_in_block
    i = ind * ind_base
    i1 = ind * (ind_base + 1)
    i2 = ind * (ind_base + 2)
    i3 = ind * (ind_base + 3)
    _stat_pre = f"{stat_block_index}, " if stat_block_index is not None else ""

    def stat_sl_at(k_index: str) -> str:
        return (
            f"{stat_tile}[0:{kt.tile}, {_stat_pre}{k_index}, "
            f"(b{st.loop} * {st.tile}):(b{st.loop} * {st.tile}) + {st.tile}]"
        )

    def mov_sl_at(k_index: str) -> str:
        return (
            f"{mov_tile}[0:{kt.tile}, {k_index}, "
            f"(b{mt.loop} * {m_tile}):(b{mt.loop} * {m_tile}) + {m_tile}]"
        )

    acc_sl = (
        f"{result_tiles}[0:{st.tile}, b{st.loop}, "
        f"(b{mt.loop} * {m_tile}):(b{mt.loop} * {m_tile}) + {m_tile}]"
    )
    ps_sl = f"{psum_var}[0:{st.tile}, 0:{m_tile}]"

    # At `TILES_IN_BLOCK_<accum> == 1` the peeled loop is `affine_range(0)` and
    # does not run, so the peel needs no trace-time branch of its own. It renames
    # the loop var, so both operand slices take the renamed index.
    peel_var = f"b{kt.loop}_peel"
    lines: list[str] = []
    if mov_wide:
        # The drain loops span `tiles_in_block x m_count` slices of width
        # `m_tile`, which covers the accumulator's `BLOCK_<mov>` exactly only when
        # `m_tile` divides it. `MATMUL_TILE_<D>`'s clamp only halves until it is
        # <= BLOCK, so divisibility is a real condition, not a tautology (2a).
        lines.append(
            f"{i}assert {mt.block} % {m_tile} == 0, "
            f'"{mt.block} must be divisible by {m_tile}"'
        )
    lines += [
        f"{i}for b{st.loop} in nl.affine_range({st.tiles_in_block}):",
        f"{i1}for b{mt.loop} in nl.affine_range({m_count}):",
        f"{i2}{psum_var} = nl.ndarray(",
        f"{i2}    ({st.tile}, {m_tile}), dtype=nl.float32, buffer=nl.psum)",
        f"{i2}nisa.nc_matmul({ps_sl},",
        f"{i2}    {stat_sl_at('0')},",
        f"{i2}    {mov_sl_at('0')}, accumulate=False)",
        f"{i2}for {peel_var} in nl.affine_range({kt.tiles_in_block} - 1):",
        f"{i3}b{kt.loop} = {peel_var} + 1",
        f"{i3}nisa.nc_matmul({ps_sl},",
        f"{i3}    {stat_sl_at(f'b{kt.loop}')},",
        f"{i3}    {mov_sl_at(f'b{kt.loop}')}, accumulate=True)",
        # 2a: at one contraction block the accumulator is still dead on entry, so
        # the drain is a copy rather than an add of zeros.
        f"{i2}if {kt.num_block} == 1:",
        f"{i3}nisa.tensor_copy({acc_sl}, {ps_sl})",
        f"{i2}else:",
        f"{i3}nisa.tensor_tensor({acc_sl},",
        f"{i3}    {acc_sl}, {ps_sl}, nl.add)",
        "",
    ]
    return lines


def _emit_result_store(
    G: _GenericMM, ind_base: int, ind: str, mm_id: str, result_tiles: str
) -> list[str]:
    """Copy the block's SBUF result tiles to the HBM ``result`` (partition dim
    rows, free dim cols), one partition-tile per row-tile."""
    out_lay = G.layout(mm_id)
    st = G.tv(G.dim_name(out_lay.part))
    mt = G.tv(G.dim_name(out_lay.free))
    i = ind * ind_base
    i1 = ind * (ind_base + 1)
    return [
        f"{i}for b{st.loop} in nl.affine_range({st.tiles_in_block}):",
        f"{i1}nisa.dma_copy(",
        f"{i1}    dst=result[(({st.tiles_in_block} * {st.loop} + b{st.loop})"
        f" * {st.tile}):(({st.tiles_in_block} * {st.loop} + b{st.loop})"
        f" * {st.tile}) + {st.tile},",
        f"{i1}           ({mt.block} * {mt.loop}):({mt.block} * {mt.loop})"
        f" + {mt.block}],",
        f"{i1}    src={result_tiles}[0:{st.tile}, b{st.loop}, 0:{mt.block}])",
        "",
    ]


def _emit_hbm_store(
    G: _GenericMM,
    ind_base: int,
    ind: str,
    node_id: str,
    source_buf: str,
) -> list[str]:
    """Store one complete scheduled block into its node's temporary HBM."""
    layout = G.layout(node_id)
    pt = G.tv(G.dim_name(layout.part))
    ft = G.tv(G.dim_name(layout.free))
    target = G.hbm_bufs[node_id]
    i = ind * ind_base
    i1 = ind * (ind_base + 1)
    return [
        f"{i}for b{pt.loop}_store in nl.affine_range({pt.tiles_in_block}):",
        f"{i1}nisa.dma_copy(",
        f"{i1}    dst={target}[(({pt.tiles_in_block} * {pt.loop} "
        f"+ b{pt.loop}_store) * {pt.tile}):",
        f"{i1}        (({pt.tiles_in_block} * {pt.loop} "
        f"+ b{pt.loop}_store) * {pt.tile}) + {pt.tile},",
        f"{i1}        ({ft.block} * {ft.loop}):({ft.block} * {ft.loop}) + {ft.block}],",
        f"{i1}    src={source_buf}[0:{pt.tile}, b{pt.loop}_store, 0:{ft.block}])",
        "",
    ]


def _emit_resident_alloc(
    G: _GenericMM,
    ind_base: int,
    ind: str,
    node_id: str,
    dtype_expr: str,
    alloc: str = "nl.ndarray",
) -> list[str]:
    """Allocate ``node_id``'s spanning SBUF buffer and register it.

    A block tile is ``(TILE_<part>, TILES_IN_BLOCK_<part>, BLOCK_<free>)``. The
    resident buffer widens whichever of those two axes the hoisted block loop
    walks, so a block slice of it has the block tile's exact shape:

    * free-spanning (the reducer's values, whose free-block loop is hoisted):
      ``(TILE_<part>, TILES_IN_BLOCK_<part>, <free extent>)``;
    * partition-spanning (a value produced per partition block and consumed by a
      later stage): ``(TILE_<part>, NUM_BLOCK_<part> * TILES_IN_BLOCK_<part>,
      BLOCK_<free>)``, which is `strip_r3_probst_resident_fused.py`'s shape.

    Which axis a value spans is fixed by its layout and by which loop the caller
    hoisted, so it is recorded here rather than re-derived at each use."""
    layout = G.layout(node_id)
    pt = G.tv(G.dim_name(layout.part))
    ft = G.tv(G.dim_name(layout.free))
    name = G.fresh_var(f"{nki_safe_var(node_id)}_sb")
    G.resident_bufs[node_id] = name
    i = ind * ind_base
    if node_id in G.part_spanning:
        shape = f"({pt.tile}, {pt.num_block} * {pt.tiles_in_block}, {ft.block})"
    else:
        shape = f"({pt.tile}, {pt.tiles_in_block}, {ft.extent})"
    return [
        f"{i}{name} = {alloc}(",
        f"{i}    {shape},",
        f"{i}    dtype={dtype_expr}, buffer=nl.sbuf)",
        "",
    ]


def _resident_block_slice(G: _GenericMM, node_id: str) -> str:
    """The current block of ``node_id``'s resident buffer.

    Shaped like an `_emit_hbm_block_load` tile, so it substitutes for one."""
    layout = G.layout(node_id)
    pt = G.tv(G.dim_name(layout.part))
    ft = G.tv(G.dim_name(layout.free))
    name = G.resident_bufs[node_id]
    if node_id in G.part_spanning:
        return (
            f"{name}[0:{pt.tile}, "
            f"({pt.tiles_in_block} * {pt.loop}):"
            f"({pt.tiles_in_block} * {pt.loop}) + {pt.tiles_in_block}, "
            f"0:{ft.block}]"
        )
    return (
        f"{name}[0:{pt.tile}, 0:{pt.tiles_in_block}, "
        f"({ft.block} * {ft.loop}):({ft.block} * {ft.loop}) + {ft.block}]"
    )


def _emit_resident_block_alias(
    G: _GenericMM, ind_base: int, ind: str, node_id: str, tile_name: str
) -> tuple[list[str], str]:
    """Name the current free block of a resident buffer, block-locally.

    The block-local name is what makes a spanning buffer a drop-in for a block
    tile: every consumer indexes it as ``name[0:TILE, b<part>, ...]``, so passing
    the spanning name directly would write block zero on every iteration."""
    return [f"{ind * ind_base}{tile_name} = {_resident_block_slice(G, node_id)}"], (
        tile_name
    )


def _emit_hbm_block_load(
    G: _GenericMM,
    ind_base: int,
    ind: str,
    node_id: str,
    tile_name: str,
    dtype_expr: str,
) -> tuple[list[str], str]:
    """Load one scheduled block from a temporary HBM into a fresh SBUF tile."""
    layout = G.layout(node_id)
    pt = G.tv(G.dim_name(layout.part))
    ft = G.tv(G.dim_name(layout.free))
    source = G.hbm_bufs[node_id]
    i = ind * ind_base
    i1 = ind * (ind_base + 1)
    return [
        f"{i}{tile_name} = nl.ndarray(",
        f"{i}    ({pt.tile}, {pt.tiles_in_block}, {ft.block}),",
        f"{i}    dtype={dtype_expr}, buffer=nl.sbuf)",
        f"{i}for b{pt.loop}_load in nl.affine_range({pt.tiles_in_block}):",
        f"{i1}nisa.dma_copy(",
        f"{i1}    dst={tile_name}[0:{pt.tile}, b{pt.loop}_load, 0:{ft.block}],",
        f"{i1}    src={source}[(({pt.tiles_in_block} * {pt.loop} "
        f"+ b{pt.loop}_load) * {pt.tile}):",
        f"{i1}        (({pt.tiles_in_block} * {pt.loop} "
        f"+ b{pt.loop}_load) * {pt.tile}) + {pt.tile},",
        f"{i1}        ({ft.block} * {ft.loop}):({ft.block} * {ft.loop}) + {ft.block}])",
        "",
    ], tile_name


def _stage_numerator_id(
    stage: EmissionStagePlan,
    rp: ReducePlan,
    node_attrs: NodeAttrs,
) -> str:
    """The full-tile value the reduce normalizes, which the reducer owns.

    Two forms: a separate exponential the reduce reads (``chain_ids``), or a
    fused ``activation_reduce`` whose numerator is an independent sibling."""
    if len(rp.source_ids) != 1:
        raise UnsupportedEmission(
            "generic matmul: staged reduction must read exactly one staged "
            f"value, got {list(rp.source_ids)}"
        )
    source_id = rp.source_ids[0]
    reduce_node = node_attrs[rp.reduce_id]

    if reduce_node.op == "tensor_reduce":
        if (
            len(rp.chain_ids) != 1
            or node_attrs[rp.chain_ids[0]].op != "exponential"
            or rp.input_id != rp.chain_ids[0]
        ):
            raise UnsupportedEmission(
                "generic matmul: staged tensor_reduce requires one exponential "
                f"chain node reading its source, got {list(rp.chain_ids)}"
            )
        return rp.chain_ids[0]

    if reduce_node.op != "activation_reduce":
        raise UnsupportedEmission(
            f"generic matmul: staged reduction {rp.reduce_id!r} is "
            f"{reduce_node.op!r}; only tensor_reduce or activation_reduce emit"
        )

    # Fused form: the reduce computes the activation itself, so validate that it
    # is the exact exponential-sum a mutated attribute would silently change.
    attrs = reduce_node.attrs
    scale = attrs.get("scale", 1.0)
    if (
        rp.chain_ids
        or not set(attrs) <= {"op", "reduce_op", "bias_const", "scale", "name"}
        or _nki_op_ref(attrs.get("op")) != "nl.exp"
        or _nki_op_ref(attrs.get("reduce_op")) != "nl.add"
        or attrs.get("bias_const") is not None
        or isinstance(scale, bool)
        or scale != 1.0
        or attrs.get("name") is not None
    ):
        raise UnsupportedEmission(
            "generic matmul: staged activation_reduce must be an exact "
            f"unscaled exponential sum, got attrs={dict(attrs)}"
        )
    candidates = [
        inter.node_id
        for inter in stage.inters
        if inter.op == "exponential" and inter.input_ids == (source_id,)
    ]
    if len(candidates) != 1:
        raise UnsupportedEmission(
            "generic matmul: fused staged reduction requires one independent "
            f"exponential over {source_id!r}, got {candidates}"
        )
    return candidates[0]


def _emit_stage_reduce(
    G: _GenericMM,
    ind_base: int,
    ind: str,
    stage: EmissionStagePlan,
    numerator_id: str,
) -> list[str]:
    """Reduce a staged value over its free axis into row scalars, writing the
    numerator into its resident SBUF buffer.

    Both the value being reduced and the numerator are resident (pattern 1a), so
    the reducer slices spanning buffers by the side-block loop var rather than
    round-tripping either through HBM. At the winner's tile config every
    ``NUM_BLOCK_*`` is 1, so the spills it replaces stored and reloaded a buffer
    inside one loop iteration."""
    rp = stage.reduces[0]
    source_id = rp.source_ids[0]
    reduce_node = G.node_attrs[rp.reduce_id]
    reduce_op = _nki_op_ref(
        reduce_node.attrs.get("reduce_op")
        if reduce_node.op == "activation_reduce"
        else reduce_node.attrs.get("op")
    )
    if reduce_op != "nl.add":
        raise UnsupportedEmission(
            "generic matmul: staged emission supports additive row reduction only"
        )
    # Additive only, per the refusal above, so the accumulator identity is 0.
    combine_op, _ = _REDUCE_ACCUMULATORS[reduce_op]

    # The reduce's own placement dim carries the rows; its source's trailing dim
    # is the axis being summed, which the reducer walks one block at a time.
    mt = G.tv(G.dim_name(G.layout(rp.reduce_id).part))
    pt = G.tv(G.dim_name(G.layout(source_id).free))
    rid = nki_safe_var(rp.reduce_id)
    red_buf = f"{rid}_tiles"
    source_resident = G.resident_bufs[source_id]
    numerator_resident = G.resident_bufs[numerator_id]
    partial = f"partial_{rid}"
    i = ind * ind_base
    i1 = ind * (ind_base + 1)
    i2 = ind * (ind_base + 2)
    side = f"{pt.loop}_side"
    row = f"b{mt.loop}_red"

    lines = [
        f"{i}{red_buf} = nl.zeros(",
        f"{i}    ({mt.tile}, {mt.tiles_in_block}, 1),",
        f"{i}    dtype=nl.float32, buffer=nl.sbuf)",
    ]
    # The side loop walks the axis being summed, and both resident buffers span
    # it, so each iteration slices its own block instead of reloading one.
    lines += [
        f"{i}for {side} in nl.sequential_range({pt.num_block}):",
        f"{i1}for {row} in nl.affine_range({mt.tiles_in_block}):",
        f"{i2}{partial} = nl.ndarray(({mt.tile}, 1), dtype=nl.float32, buffer=nl.sbuf)",
    ]
    source_expr = (
        f"{source_resident}[0:{mt.tile}, {row}, "
        f"({pt.block} * {side}):({pt.block} * {side}) + {pt.block}]"
    )
    numerator_expr = (
        f"{numerator_resident}[0:{mt.tile}, {row}, "
        f"({pt.block} * {side}):({pt.block} * {side}) + {pt.block}]"
    )
    # The numerator and its row reduction in ONE instruction. The fused reduce
    # walks the free axis of its own source tile, which is one block wide here, so
    # it yields one partial per (row, side) that the accumulate below combines.
    call = G.node_attrs.emit(G.ctx, numerator_id, {source_id: source_expr})
    assert call is not None and call.rstrip().endswith(")")
    fused_stmt = substitute_dst(
        call.rstrip()[:-1]
        + f", reduce_op={reduce_op}, reduce_res={partial}[0:{mt.tile}, 0:1],"
        " reduce_cmd=nisa.reduce_cmd.reset_reduce)",
        dst=numerator_expr,
        dst_slice=numerator_expr,
        dst_shape=f"({mt.tile}, {pt.block})",
        ind=i2,
    )
    lines += [
        f"{i2}{fused_stmt}",
        f"{i2}nisa.tensor_tensor({red_buf}[0:{mt.tile}, {row}, 0],",
        f"{i2}    {red_buf}[0:{mt.tile}, {row}, 0], "
        f"{partial}[0:{mt.tile}, 0:1], op={combine_op})",
        "",
    ]
    G.scalar_bufs[rp.reduce_id] = red_buf

    previous_id = rp.reduce_id
    previous_buf = red_buf
    for scalar_id in rp.post_scalar_ids:
        if G.node_attrs[scalar_id].op == "broadcast":
            G.scalar_bufs[scalar_id] = previous_buf
            previous_id = scalar_id
            continue
        scalar_buf = f"{nki_safe_var(scalar_id)}_tiles"
        lines += [
            f"{i}{scalar_buf} = nl.ndarray(",
            f"{i}    ({mt.tile}, {mt.tiles_in_block}, 1),",
            f"{i}    dtype=nl.float32, buffer=nl.sbuf)",
            f"{i}for b{mt.loop}_scalar in nl.affine_range({mt.tiles_in_block}):",
        ]
        source = f"{previous_buf}[0:{mt.tile}, b{mt.loop}_scalar, 0]"
        target = f"{scalar_buf}[0:{mt.tile}, b{mt.loop}_scalar, 0]"
        scalar_call = G.node_attrs.emit(G.ctx, scalar_id, {previous_id: source})
        assert scalar_call is not None
        scalar_stmt = substitute_dst(
            scalar_call,
            dst=target,
            dst_slice=target,
            dst_shape=f"({mt.tile}, 1)",
            ind=i1,
        )
        lines.append(f"{i1}{scalar_stmt}")
        G.scalar_bufs[scalar_id] = scalar_buf
        previous_id = scalar_id
        previous_buf = scalar_buf
    lines.append("")
    return lines


def _emit_post_mm_chain(
    G: _GenericMM,
    ind_base: int,
    ind: str,
    mm_id: str,
    post_steps: tuple[ChainStep, ...],
    result_tiles: str,
    dtype_expr: str,
) -> tuple[list[str], str]:
    """Emit the post-matmul elementwise chain over the block's result tiles,
    after the contraction loop and before the HBM store.

    Each step (a unary elementwise op or a single-scalar combiner) is applied
    per (m-tile) over the ``(TILE_<part>, TILES_IN_BLOCK_<part>, BLOCK_<free>)``
    result buffer into a fresh SBUF buffer; a combiner's ``(m,)`` reduce-preamble
    scalar is indexed by the m-tile loop var. The chain (and each step's data /
    scalar edges) comes from the plan. Returns ``(lines, final_buf)`` — the last
    buffer the HBM store must read (``result_tiles`` when the chain is empty)."""
    if not post_steps:
        return [], result_tiles
    out_lay = G.layout(mm_id)
    pt = G.tv(G.dim_name(out_lay.part))
    ft = G.tv(G.dim_name(out_lay.free))
    i = ind * ind_base
    i1 = ind * (ind_base + 1)
    lv = f"b{pt.loop}_post"
    shape = f"({pt.tile}, {pt.tiles_in_block}, {ft.block})"
    lines: list[str] = []
    cur_buf = result_tiles
    for step in post_steps:
        if step.op == "broadcast":
            continue
        buf = f"{nki_safe_var(step.node_id)}_post"
        dst_slice = f"{buf}[0:{pt.tile}, {lv}, 0:{ft.block}]"
        id_to_var = {step.data_pred_id: f"{cur_buf}[0:{pt.tile}, {lv}, 0:{ft.block}]"}
        if step.scalar_id is not None:
            scal_id = step.scalar_id
            if scal_id not in G.scalar_bufs:
                raise UnsupportedEmission(
                    f"generic matmul: post-mm combiner {step.node_id!r} consumes "
                    f"scalar {scal_id!r} with no (m,) scalar tile in scope"
                )
            id_to_var[scal_id] = f"{G.scalar_bufs[scal_id]}[0:{pt.tile}, {lv}, 0]"
        call = G.node_attrs.emit(G.ctx, step.node_id, id_to_var)
        assert call is not None
        stmt = substitute_dst(
            call,
            dst=dst_slice,
            dst_slice=dst_slice,
            dst_shape=f"({pt.tile}, {ft.block})",
            ind=i1,
        )
        lines += [
            f"{i}{buf} = nl.ndarray({shape}, dtype={dtype_expr}, buffer=nl.sbuf)",
            f"{i}for {lv} in nl.affine_range({pt.tiles_in_block}):",
            f"{i1}{stmt}",
            "",
        ]
        cur_buf = buf
    return lines, cur_buf


def _activation_reduce_stage_step(
    G: _GenericMM,
    rp: ReducePlan,
) -> ChainStep | None:
    """Find an operand activation exactly duplicated by activation_reduce."""
    reduce_node = G.node_attrs[rp.reduce_id]
    if reduce_node.op != "activation_reduce" or rp.chain_ids:
        return None
    reduce_attrs = reduce_node.attrs
    if any(key in reduce_attrs for key in ("bias_input_index", "scale_input_index")):
        return None

    reduce_key = (
        _nki_op_ref(reduce_attrs.get("op")),
        reduce_attrs.get("bias_const"),
        reduce_attrs.get("scale", 1.0),
    )
    candidates: dict[str, ChainStep] = {}
    for mp in G.plan.matmuls:
        for operand in (mp.stat, mp.mov):
            if operand.transpose_index is None:
                continue
            pre_steps = operand.steps[: operand.transpose_index]
            if not pre_steps:
                continue
            step = pre_steps[0]
            if step.data_pred_id != rp.input_id:
                continue
            step_node = G.node_attrs[step.node_id]
            if step_node.op == "exponential":
                step_key = ("nl.exp", None, 1.0)
            elif step_node.op == "activation":
                step_attrs = step_node.attrs
                if any(
                    key in step_attrs
                    for key in ("bias_input_index", "scale_input_index")
                ):
                    continue
                step_key = (
                    _nki_op_ref(step_attrs.get("op")),
                    step_attrs.get("bias_const"),
                    step_attrs.get("scale", 1.0),
                )
            else:
                continue
            if step_key != reduce_key:
                continue
            free = G.layout(operand.op_id).free
            if free is None or G.dim_name(free) != "m":
                continue
            candidates[step.node_id] = step
    if len(candidates) != 1:
        return None
    return next(iter(candidates.values()))


def _emit_reduce_preamble(
    G: _GenericMM,
    ind_base: int,
    ind: str,
    reduces: tuple[ReducePlan, ...],
    accum_dim: str,
    shared_ids: frozenset[str],
) -> list[str]:
    """Emit the k_side reduce preamble at the m level (before the block-n loop).

    Per reduce plan: a ``(TILE_M, TILES_IN_BLOCK_M, 1)`` fp32 zeros accumulator
    named by the reduce-node id; a ``sequential_range(NUM_BLOCK_K)`` loop that
    DMA-loads each HBM input of the reduce's input chain, runs the elementwise
    chain per m-tile into fresh fp32 scratch, ``nisa.tensor_reduce`` into a
    ``(TILE_M, 1)`` fp32 partial, and ``nisa.tensor_tensor``-accumulates into the
    block accumulator. Then the post-reduce per-partition-scalar chain
    (rsqrt/reciprocal on the (m,) scalar) into fresh ``(TILE_M,
    TILES_IN_BLOCK_M, 1)`` buffers. Records every scalar buffer in
    ``G.scalar_bufs`` for the operand-chain combiner to consume.

    A chain node in ``shared_ids`` (produced here AND re-consumed by a matmul
    operand chain — softmax's ``exponential``) is additionally STAGED: an
    m-level spanning buffer ``(TILE_M, NUM_BLOCK_K, TILES_IN_BLOCK_M, BLOCK_K)``
    (named in ``G.staged_bufs``) receives the node's value at ``[:, k_side,
    bm_red, :]`` via a ``tensor_copy`` from its fp32 scratch (the reduce still
    reads the accurate fp32 scratch); the operand chain then reads the staged
    slice instead of recomputing the shared prefix. The chain node ids, HBM
    source ids, and post-scalar ids all come from the plan."""
    mt = G.tv("m")
    kt = G.tv(accum_dim)
    i = ind * ind_base
    i1 = ind * (ind_base + 1)
    i2 = ind * (ind_base + 2)
    lines: list[str] = []
    for rp in reduces:
        reduce_node = G.node_attrs[rp.reduce_id]
        reduce_attrs = reduce_node.attrs
        chain_ops = {cid: G.node_attrs[cid].op for cid in rp.chain_ids}
        hbm_inputs = list(rp.hbm_inputs)
        hbm_ids = [item.input_id for item in hbm_inputs]
        rid = nki_safe_var(rp.reduce_id)
        red_buf = f"{rid}_tiles"
        reduce_op = (
            reduce_attrs.get("reduce_op")
            if reduce_node.op == "activation_reduce"
            else reduce_attrs.get("op")
        )
        op_ref = _nki_op_ref(reduce_op)
        accumulator = _REDUCE_ACCUMULATORS.get(op_ref)
        if accumulator is None:
            raise UnsupportedEmission(
                f"{reduce_node.op} {rp.reduce_id}: unsupported reduction op "
                f"{reduce_op!r}"
            )
        combine_op, identity = accumulator
        keepdims_attr = reduce_attrs.get(
            "keepdims",
            reduce_attrs.get("keep_dims", False),
        )
        keepdims = ", keepdims=True" if keepdims_attr else ""
        # Shared chain nodes get a spanning staged buffer (live across k blocks).
        grp_dtype = f"{nki_safe_var(hbm_ids[0])}.dtype" if hbm_ids else "nl.float32"
        grp_shared = [cid for cid in rp.chain_ids if cid in shared_ids]
        activation_stage = _activation_reduce_stage_step(G, rp)
        if activation_stage is not None:
            grp_shared.append(activation_stage.node_id)
        accumulator_shape = f"({mt.tile}, {mt.tiles_in_block}, 1)"
        assert identity == "0.0", f"reduce {rp.reduce_id}: unexpected identity"
        lines += [
            f"{i}{red_buf} = nl.zeros(",
            f"{i}    {accumulator_shape},",
            f"{i}    dtype=nl.float32, buffer=nl.sbuf)",
        ]
        for sid in grp_shared:
            sbuf = f"{nki_safe_var(sid)}_staged"
            G.staged_bufs[sid] = sbuf
            lines += [
                f"{i}{sbuf} = nl.ndarray(",
                f"{i}    ({mt.tile}, {kt.num_block}, {mt.tiles_in_block}, {kt.block}),",
                f"{i}    dtype={grp_dtype}, buffer=nl.sbuf)",
            ]
        lines += [
            f"{i}for k_side in nl.sequential_range({kt.num_block}):",
        ]
        # Load each HBM input of the reduce's input chain (deduped).
        hbm_expr: dict[str, str] = {}
        for hbm_input in hbm_inputs:
            hbm_id = hbm_input.input_id
            if hbm_id in hbm_expr:
                continue
            hbm_var = nki_safe_var(hbm_id)
            htile = f"{hbm_var}_{rid}_chain_tile_k"
            free_extent = "1" if hbm_input.free_broadcast else kt.block
            free_source = (
                "0:1"
                if hbm_input.free_broadcast
                else f"({kt.block} * k_side):({kt.block} * k_side) + {kt.block}"
            )
            if hbm_input.part_broadcast:
                lines += [
                    f"{i1}{htile} = nl.ndarray((1, {free_extent}),"
                    f" dtype={hbm_var}.dtype, buffer=nl.sbuf)",
                    f"{i1}nisa.dma_copy(",
                    f"{i1}    dst={htile}[0:1, 0:{free_extent}],",
                    f"{i1}    src={hbm_var}[0:1, {free_source}])",
                ]
                hbm_expr[hbm_id] = f"{htile}[0:1, 0:{free_extent}]"
            else:
                lines += [
                    f"{i1}{htile} = nl.ndarray(",
                    f"{i1}    ({mt.tile}, {mt.tiles_in_block}, {free_extent}),"
                    f" dtype={hbm_var}.dtype, buffer=nl.sbuf)",
                    f"{i1}for bm_red in nl.affine_range({mt.tiles_in_block}):",
                    f"{i2}nisa.dma_copy(",
                    f"{i2}    dst={htile}[0:{mt.tile}, bm_red, 0:{free_extent}],",
                    f"{i2}    src={hbm_var}[(({mt.tiles_in_block} * m + bm_red)"
                    f" * {mt.tile}):(({mt.tiles_in_block} * m + bm_red) * {mt.tile})"
                    f" + {mt.tile},",
                    f"{i2}           {free_source}])",
                ]
                hbm_expr[hbm_id] = f"{htile}[0:{mt.tile}, bm_red, 0:{free_extent}]"
        # Fuse when the last chain node is staged, is the reduce's direct input,
        # and is an activation-family op summed by nl.add (e.g. softmax exp→sum).
        red_input_id = rp.input_id
        fuse_node_id = rp.chain_ids[-1] if rp.chain_ids else None
        fuse = (
            reduce_node.op == "tensor_reduce"
            and fuse_node_id is not None
            and fuse_node_id == red_input_id
            and fuse_node_id in G.staged_bufs
            and chain_ops[fuse_node_id] in _ELEMENTWISE_OPS
            and op_ref == "nl.add"
        )
        lines += [f"{i1}for bm_red in nl.affine_range({mt.tiles_in_block}):"]
        cur_expr: dict[str, str] = dict(hbm_expr)
        partial = f"partial_{rid}"
        for cid in rp.chain_ids:
            call = G.node_attrs.emit(G.ctx, cid, dict(cur_expr))
            assert call is not None
            if fuse and cid == fuse_node_id:
                # Write the activation straight into the staged slot, reducing
                # in the same op into the fp32 partial.
                sbuf = G.staged_bufs[cid]
                dst = f"{sbuf}[0:{mt.tile}, k_side, bm_red, 0:{kt.block}]"
                assert call.rstrip().endswith(")")
                fused_call = (
                    call.rstrip()[:-1]
                    + f", reduce_op={op_ref}, reduce_res={partial}[0:{mt.tile}, 0:1])"
                )
                stmt = substitute_dst(
                    fused_call,
                    dst=dst,
                    dst_slice=dst,
                    dst_shape=f"({mt.tile}, {kt.block})",
                    ind=i2,
                )
                lines += [
                    f"{i2}{partial} = nl.ndarray(({mt.tile}, 1),"
                    f" dtype=nl.float32, buffer=nl.sbuf)",
                    f"{i2}{stmt}",
                    f"{i2}nisa.tensor_tensor({red_buf}[0:{mt.tile}, bm_red, 0],",
                    f"{i2}    {red_buf}[0:{mt.tile}, bm_red, 0],"
                    f" {partial}[0:{mt.tile}, 0:1], op={combine_op})",
                    "",
                ]
                cur_expr[cid] = dst
                break
            scratch = f"{nki_safe_var(cid)}_{rid}_scratch"
            dst = f"{scratch}[0:{mt.tile}, 0:{kt.block}]"
            stmt = substitute_dst(
                call,
                dst=dst,
                dst_slice=dst,
                dst_shape=f"({mt.tile}, {kt.block})",
                ind=i2,
            )
            lines += [
                f"{i2}{scratch} = nl.ndarray(({mt.tile}, {kt.block}),"
                f" dtype=nl.float32, buffer=nl.sbuf)",
                f"{i2}{stmt}",
            ]
            cur_expr[cid] = dst
            # Stage this shared node into its m-level spanning buffer at the
            # current (k_side, bm_red) slot; the operand chain reads it back.
            if cid in G.staged_bufs:
                sbuf = G.staged_bufs[cid]
                lines += [
                    f"{i2}nisa.tensor_copy({sbuf}[0:{mt.tile}, k_side, bm_red,"
                    f" 0:{kt.block}], {dst})",
                ]
        if not fuse:
            red_in = cur_expr[red_input_id]
            # The plan already refused axis!=1. Assert again so a
            # future refactor cannot silently emit axis=0.
            if reduce_node.op == "tensor_reduce":
                assert rp.axis == 1, (
                    f"tensor_reduce {rp.reduce_id}: emitter hardcodes axis=[1] "
                    f"but plan carries axis={rp.axis!r}; the plan should "
                    f"have refused this variant"
                )
                reduce_stmt = (
                    f"nisa.tensor_reduce({partial}[0:{mt.tile}, 0:1], "
                    f"{op_ref}, {red_in}, axis=[{rp.axis}]{keepdims})"
                )
            elif activation_stage is not None:
                # The operand chain wants this activation's (P, F) output, so the
                # reduce writes it straight into the staged slot in one pass.
                staged_buf = G.staged_bufs[activation_stage.node_id]
                call = G.node_attrs.emit_activation_reduce_to(
                    G.ctx,
                    rp.reduce_id,
                    dict(cur_expr),
                    act_dst=f"{staged_buf}[0:{mt.tile}, k_side, bm_red, 0:{kt.block}]",
                )
                reduce_stmt = substitute_dst(
                    call,
                    dst=f"{partial}[0:{mt.tile}, 0:1]",
                    dst_slice=f"{partial}[0:{mt.tile}, 0:1]",
                    dst_shape=f"({mt.tile}, 1)",
                    ind=i2,
                )
            else:
                call = G.node_attrs.emit(G.ctx, rp.reduce_id, dict(cur_expr))
                assert call is not None
                reduce_stmt = substitute_dst(
                    call,
                    dst=f"{partial}[0:{mt.tile}, 0:1]",
                    dst_slice=f"{partial}[0:{mt.tile}, 0:1]",
                    dst_shape=f"({mt.tile}, 1)",
                    ind=i2,
                )
            lines += [
                f"{i2}{partial} = nl.ndarray(({mt.tile}, 1),"
                f" dtype=nl.float32, buffer=nl.sbuf)",
                f"{i2}{reduce_stmt}",
                f"{i2}nisa.tensor_tensor({red_buf}[0:{mt.tile}, bm_red, 0],",
                f"{i2}    {red_buf}[0:{mt.tile}, bm_red, 0], {partial}[0:{mt.tile}, 0:1],"
                f" op={combine_op})",
                "",
            ]
        G.scalar_bufs[rp.reduce_id] = red_buf
        # Post-reduce unary scalar chain (rsqrt/reciprocal on (m,)).
        prev_buf = red_buf
        prev_id = rp.reduce_id
        for pid in rp.post_scalar_ids:
            if G.node_attrs[pid].op == "broadcast":
                G.scalar_bufs[pid] = prev_buf
                prev_id = pid
                continue
            pbuf = f"{nki_safe_var(pid)}_tiles"
            lines += [
                f"{i}{pbuf} = nl.ndarray(",
                f"{i}    ({mt.tile}, {mt.tiles_in_block}, 1),",
                f"{i}    dtype=nl.float32, buffer=nl.sbuf)",
                f"{i}for bm_side in nl.affine_range({mt.tiles_in_block}):",
            ]
            src = f"{prev_buf}[0:{mt.tile}, bm_side, 0]"
            dst = f"{pbuf}[0:{mt.tile}, bm_side, 0]"
            call = G.node_attrs.emit(G.ctx, pid, {prev_id: src})
            assert call is not None
            stmt = substitute_dst(
                call, dst=dst, dst_slice=dst, dst_shape=f"({mt.tile}, 1)", ind=i1
            )
            lines += [f"{i1}{stmt}"]
            G.scalar_bufs[pid] = pbuf
            prev_buf = pbuf
            prev_id = pid
        lines += [""]
    return lines


def _group_accum_dim(G: _GenericMM, mps: list[MatmulPlan]) -> str:
    """The one contraction dim a matmul group shares, aliased through the plan's
    role collapse. Raises if they disagree, exactly as `_emit_accum_stage` does."""
    accum_dims = {mp.accum_dim for mp in mps}
    if len(accum_dims) != 1:
        raise UnsupportedEmission(
            f"generic matmul: side matmuls do not share one contraction dim "
            f"({sorted(accum_dims)})"
        )
    return G.alias_dim(accum_dims.pop())


def _emit_accum_stage(
    G: _GenericMM,
    mps: list[MatmulPlan],
    ind_base: int,
    ind: str,
    dtype_expr: str,
    *,
    alias_accum_dim: bool = False,
    operand_free_from_output: bool = False,
    preallocated: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Emit a set of matmuls that share a block level and one contraction dim:
    one SBUF accumulator per matmul, then a single sequential contraction-block
    loop with deduped operand loads and a per-matmul PSUM accumulate. Returns
    ``(lines, {mm.id: accumulator_buf})``.

    Each accumulator is ``(TILE_<part>, TILES_IN_BLOCK_<part>, BLOCK_<free>)``
    off the matmul's OUTPUT layout; the shared operand loads dedup by operand id
    so a value feeding two matmuls emits once. Operand plans (and their asserted
    roles) come from the ``MatmulPlan``s.

    ``alias_accum_dim`` resolves the contraction through the plan's role
    collapse, which a staged nest needs so its tile name is one the preamble
    minted. ``operand_free_from_output`` takes each operand's free dim from the
    matmul's output layout rather than the operand's own, which the caller sets
    when the loops it opened are named after the output axes.

    ``preallocated`` names a drain target the caller already allocated (a resident
    reduce source, whose buffer is hoisted above the block loop), so this function
    must not allocate a block-local one over the top of it."""
    i = ind * ind_base
    accum_dims = {mp.accum_dim for mp in mps}
    if len(accum_dims) != 1:
        raise UnsupportedEmission(
            f"generic matmul: side matmuls do not share one contraction dim "
            f"({sorted(accum_dims)})"
        )
    accum_dim = accum_dims.pop()
    if alias_accum_dim:
        accum_dim = G.alias_dim(accum_dim)

    lines: list[str] = []
    accum_buf: dict[str, str] = {}
    alloc = G.accum_alloc(accum_dim)
    preallocated = preallocated or {}
    for mp in mps:
        if mp.mm_id in preallocated:
            accum_buf[mp.mm_id] = preallocated[mp.mm_id]
            continue
        out_lay = G.layout(mp.mm_id)
        pt = G.tv(G.dim_name(out_lay.part))
        ft = G.tv(G.dim_name(out_lay.free))
        buf = f"{nki_safe_var(mp.mm_id)}_tiles"
        accum_buf[mp.mm_id] = buf
        lines += [
            f"{i}{buf} = {alloc}(",
            f"{i}    ({pt.tile}, {pt.tiles_in_block}, {ft.block}),",
            f"{i}    dtype={dtype_expr}, buffer=nl.sbuf)",
        ]
    kt = G.tv(accum_dim)
    lines += ["", f"{i}for {kt.loop} in nl.sequential_range({kt.num_block}):"]

    loaded: dict[str, str] = {}
    stat_mov: dict[str, tuple[str, str]] = {}
    for mp in mps:
        if operand_free_from_output:
            output_layout = G.layout(mp.mm_id)
            stat_free = G.dim_name(output_layout.part)
            mov_free = G.dim_name(output_layout.free)
        else:
            stat_free = G.dim_name(G.layout(mp.stat.op_id).free)
            mov_free = G.dim_name(G.layout(mp.mov.op_id).free)
        for op, free_dim in ((mp.stat, stat_free), (mp.mov, mov_free)):
            if op.op_id not in loaded:
                load_lines, tv = _emit_operand_load(
                    G, ind_base + 1, ind, op, accum_dim, free_dim
                )
                lines += load_lines
                loaded[op.op_id] = tv
        stat_mov[mp.mm_id] = (loaded[mp.stat.op_id], loaded[mp.mov.op_id])

    for mp in mps:
        stat_tile, mov_tile = stat_mov[mp.mm_id]
        if operand_free_from_output:
            output_layout = G.layout(mp.mm_id)
            stat_free = G.dim_name(output_layout.part)
            mov_free = G.dim_name(output_layout.free)
        else:
            stat_free = G.dim_name(G.layout(mp.stat.op_id).free)
            mov_free = G.dim_name(G.layout(mp.mov.op_id).free)
        lines += _emit_matmul_accumulate(
            G,
            ind_base + 1,
            ind,
            mp.mm_id,
            stat_tile,
            mov_tile,
            accum_dim,
            stat_free,
            mov_free,
            accum_buf[mp.mm_id],
            psum_var=f"ps_{nki_safe_var(mp.mm_id)}",
            mov_wide=mp.mov_wide,
        )
    return lines, accum_buf


def _emit_inter_node(
    G: _GenericMM,
    ip: InterNodePlan,
    id_to_buf: dict[str, str],
    ind_base: int,
    ind: str,
    dtype_expr: str,
    mat_index: str | None,
    dest: str | None = None,
) -> tuple[list[str], str]:
    """Emit one inter-matmul node (activation / combine / nc_transpose) over the
    block, per-tile, into a fresh SBUF buffer. Reads each input's block buffer
    (a sibling inter buffer or a matmul accumulator) via its producer's layout.
    The op and the exact input edges come from the ``InterNodePlan`` (frozen at
    plan time) — the renderer never reads ``Node.inputs``.

    A materialized node (``mat_index`` set: the sequential n-block loop var)
    writes a pre-allocated 4D buffer ``(TILE_<part>, NUM_BLOCK_<n>,
    TILES_IN_BLOCK_<part>, BLOCK_<free>)`` at that n-block (its allocation is
    hoisted to the m-level by ``_mat_buf_alloc``); every other node allocates and
    writes a fresh 3D block buffer here. Returns ``(lines, buf_name)``.

    ``dest`` writes an already-allocated buffer instead of allocating one, which is
    how a resident value's producer writes its spanning buffer. It must be a
    block-local alias (see `_emit_resident_block_alias`), never the spanning
    name."""
    node_id = ip.node_id
    i = ind * ind_base
    i1 = ind * (ind_base + 1)
    i2 = ind * (ind_base + 2)
    lay = G.layout(node_id)
    pt = G.tv(G.dim_name(lay.part))
    ft = G.tv(G.dim_name(lay.free)) if lay.free is not None else None
    if ft is None:
        raise UnsupportedEmission(
            f"generic matmul: inter node {node_id!r} is a per-partition scalar "
            f"(unsupported in a multi-matmul combine)"
        )
    # No caller passes both: `_emit_inter_chain` passes only `mat_index`, and the
    # staged body's resident site hardcodes `mat_index=None`.
    assert dest is None or mat_index is None
    buf = dest if dest is not None else f"{nki_safe_var(node_id)}_inter"

    def _alloc(shape: str) -> str:
        # A materialized buffer is allocated once at the m-level, and a resident
        # destination by the caller that hoisted it, so neither allocates here.
        if mat_index is not None or dest is not None:
            return None
        return f"{i}{buf} = nl.ndarray({shape}, dtype={dtype_expr}, buffer=nl.sbuf)"

    def _dst(bp: str) -> str:
        if mat_index is not None:
            return f"{buf}[0:{pt.tile}, {mat_index}, {bp}, 0:{ft.block}]"
        return f"{buf}[0:{pt.tile}, {bp}, 0:{ft.block}]"

    if mat_index is not None:
        shape = f"({pt.tile}, {pt.num_block}, {pt.tiles_in_block}, {ft.block})"
    else:
        shape = f"({pt.tile}, {pt.tiles_in_block}, {ft.block})"

    if ip.op == "nc_transpose":
        # Output (part, free) is the swap of the input's (part, free): the input
        # buffer is (TILE_<free>, TILES_IN_BLOCK_<free>, BLOCK_<part>). Transpose
        # each (TILE_free, TILE_part) sub-tile into (TILE_part, TILE_free).
        src_id = ip.input_ids[0]
        src_buf = id_to_buf[src_id]
        lines = [a for a in [_alloc(shape)] if a is not None]
        lines += [
            f"{i}for b{pt.loop}_t in nl.affine_range({pt.tiles_in_block}):",
            f"{i1}for b{ft.loop}_t in nl.affine_range({ft.tiles_in_block}):",
        ]
        dst_sl = (
            f"{buf}[0:{pt.tile}, {mat_index + ', ' if mat_index else ''}"
            f"b{pt.loop}_t, "
            f"(b{ft.loop}_t * {ft.tile}):(b{ft.loop}_t * {ft.tile}) + {ft.tile}]"
        )
        src_sl = (
            f"{src_buf}[0:{ft.tile}, b{ft.loop}_t, "
            f"(b{pt.loop}_t * {pt.tile}):(b{pt.loop}_t * {pt.tile}) + {pt.tile}]"
        )
        lines += nc_transpose_psum(
            G.ctx, dst_sl, src_sl, dtype_expr, pt.tile, ft.tile, i2
        )
        lines += [""]
        return lines, buf

    # Full-tile elementwise / combine: every full-tile input shares this node's
    # (part, free) layout (verified by the plan's combine acceptance rule), so
    # slice each input identically over the part-tile loop.
    lines = [a for a in [_alloc(shape)] if a is not None]
    lv = f"b{pt.loop}_i"
    lines += [f"{i}for {lv} in nl.affine_range({pt.tiles_in_block}):"]
    id_to_var: dict[str, str] = {}
    for inp in ip.input_ids:
        ip_lay = G.layout(inp)
        ipt = G.tv(G.dim_name(ip_lay.part))
        if ip_lay.free is None:
            scalar_buf = G.scalar_bufs.get(inp)
            if scalar_buf is None:
                raise UnsupportedEmission(
                    f"generic matmul: inter node {node_id!r} consumes scalar "
                    f"{inp!r} with no scalar tile in scope"
                )
            id_to_var[inp] = f"{scalar_buf}[0:{pt.tile}, {lv}, 0]"
            continue
        ipf = G.tv(G.dim_name(ip_lay.free))
        id_to_var[inp] = f"{id_to_buf[inp]}[0:{ipt.tile}, {lv}, 0:{ipf.block}]"
    dst_sl = _dst(lv)
    call = G.node_attrs.emit(G.ctx, node_id, id_to_var)
    assert call is not None
    stmt = substitute_dst(
        call,
        dst=dst_sl,
        dst_slice=dst_sl,
        dst_shape=f"({pt.tile}, {ft.block})",
        ind=i1,
    )
    lines += [f"{i1}{stmt}", ""]
    return lines, buf


def _mat_buf_alloc(
    G: _GenericMM, node_id: str, ind: str, dtype_expr: str
) -> tuple[str, list[str]]:
    """The hoisted allocation of a materialized inter node's 4D buffer
    ``(TILE_<part>, NUM_BLOCK_<n>, TILES_IN_BLOCK_<part>, BLOCK_<free>)`` — one
    per m-block, live across every n-block that fills it. Returns
    ``(buf_name, lines)``."""
    lay = G.layout(node_id)
    pt = G.tv(G.dim_name(lay.part))
    ft = G.tv(G.dim_name(lay.free))
    nt = G.tv(G.nest.materialized[node_id][0])
    buf = f"{nki_safe_var(node_id)}_inter"
    return buf, [
        f"{ind}{buf} = nl.zeros(",
        f"{ind}    ({pt.tile}, {nt.num_block}, {pt.tiles_in_block}, {ft.block}),",
        f"{ind}    dtype={dtype_expr}, buffer=nl.sbuf)",
        "",
    ]


def _emit_inter_chain(
    G: _GenericMM,
    inters: list[InterNodePlan],
    accum_buf: dict[str, str],
    ind_base: int,
    ind: str,
    dtype_expr: str,
    materialized: set[str],
    mat_index: str | None,
) -> tuple[list[str], dict[str, str]]:
    """Emit the inter-matmul nodes in topo order, threading each into its block
    buffer. Returns ``(lines, id_to_buf)`` covering matmul accumulators plus
    every inter buffer."""
    id_to_buf: dict[str, str] = dict(accum_buf)
    lines: list[str] = []
    for ip in inters:
        node_lines, buf = _emit_inter_node(
            G,
            ip,
            id_to_buf,
            ind_base,
            ind,
            dtype_expr,
            mat_index if ip.node_id in materialized else None,
        )
        lines += node_lines
        id_to_buf[ip.node_id] = buf
    return lines, id_to_buf


def _stage_groups(
    G: _GenericMM, stage: EmissionStagePlan
) -> dict[tuple[str, str, str], list[MatmulPlan]]:
    """A stage's matmuls grouped by ``(output part, output free, contraction)``.

    One function, called both by the fusion prepass and by the emission loop, so
    the grouping the prepass reasons about is the grouping that emits."""
    groups: dict[tuple[str, str, str], list[MatmulPlan]] = {}
    for mm_id in stage.matmul_ids:
        mp = G.plan.matmul(mm_id)
        layout = G.layout(mm_id)
        key = (
            G.dim_name(layout.part),
            G.dim_name(layout.free),
            G.alias_dim(mp.accum_dim),
        )
        groups.setdefault(key, []).append(mp)
    return groups


def _fusible_crossing_values(
    G: _GenericMM,
    stages: tuple[EmissionStagePlan, ...],
    consumer_stages: dict[str, int],
    node_stage: dict[str, int],
    already_resident: list[str],
) -> tuple[str | None, list[str]]:
    """Pattern 1b's prepass: ``(the shared part dim, the values that may stay
    resident across a stage boundary)``.

    The driver emits stages as sibling loop nests, so a resident buffer spanning
    the producing nest is dead before the consuming nest opens. Making a crossing
    value resident therefore requires hoisting one shared block loop above both
    nests, and the rule is:

        Make a crossing value resident only when every group of every stage, from
        its producer through its last consumer, carries the same ``part`` dim.
        Then hoist that one ``part`` loop above all those stages.

    The condition covers the whole LIFETIME, not one stage pair: a crossing value
    can be live across a stage that neither produces nor consumes it, and that
    stage's loops sit inside the hoisted one too.

    Measured: the two-matmul core qualifies (stage 0 and stage 1 are both
    ``part=m``). The five-matmul QKV plan does not, because its stage 0 holds
    ``part=p`` beside ``part=m``. It keeps its HBM crossing, which costs a
    microsecond rather than correctness.

    Returns ``(None, [])`` when nothing qualifies, which is the unfused emission.
    """
    if len(stages) < 2:
        return None, []

    # Every stage's part dims, from the same grouping that will emit.
    stage_parts: dict[int, set[str]] = {}
    for stage in stages:
        stage_parts[stage.index] = {
            part for part, _free, _accum in _stage_groups(G, stage)
        }
    if any(len(parts) != 1 for parts in stage_parts.values()):
        return None, []
    parts = {next(iter(p)) for p in stage_parts.values()}
    if len(parts) != 1:
        return None, []
    part_dim = parts.pop()

    # The output store sits inside the group that produced it, so hoisting the
    # part loop is only sound when every stage in the span shares it. That is
    # exactly what the check above established, over ALL stages.
    fusible: list[str] = []
    for node_id, last in consumer_stages.items():
        if node_id in already_resident:
            continue
        produced = node_stage.get(node_id)
        if produced is None or last <= produced:
            continue
        layout = G.layout(node_id)
        if layout.free is None:
            # A per-partition scalar has no free axis to shape, and the reducer
            # already keeps its scalars in `G.scalar_bufs`. Step 7a handles it.
            continue
        fusible.append(node_id)
    if not fusible:
        return None, []
    return part_dim, fusible


def _emit_staged_body(
    G: _GenericMM,
    ind: str,
    input_nodes: list[Node],
    dtype_expr: str,
) -> tuple[list[str], list[str]]:
    """Emit a multi-stage matmul graph carrying a free-axis reduce, driving the
    loop structure from ``plan.stages`` alone.

    Each stage groups its matmuls by ``(output part, output free, contraction)``,
    opens the two block loops that grouping names, accumulates, then emits the
    stage's reduce and inter nodes inside the group that produced their input.
    Values a later stage reads cross through shared HBM."""
    plan = G.plan
    stages = plan.stages
    if len(plan.reduces) > 1:
        raise UnsupportedEmission(
            "generic matmul: staged emission supports at most one reduction, "
            f"got {len(plan.reduces)}"
        )

    node_stage = {
        node_id: stage.index for stage in stages for node_id in stage.node_ids
    }
    consumer_stages: dict[str, int] = {}
    for stage in stages:
        for node_id in stage.live_in_ids:
            consumer_stages[node_id] = max(
                stage.index, consumer_stages.get(node_id, stage.index)
            )

    # A reduce's two values live entirely inside its own stage: the value it
    # reduces and the numerator.
    reduce_stage = next((stage for stage in stages if stage.reduces), None)
    if reduce_stage is not None:
        rp = reduce_stage.reduces[0]
        numerator_id = _stage_numerator_id(reduce_stage, rp, G.node_attrs)
        resident_ids = [rp.source_ids[0], numerator_id]
    else:
        numerator_id = None
        resident_ids = []

    # Pattern 1b: a value that DOES cross a stage can also stay resident, but only
    # if the consuming stage is emitted inside the producer's outer block loop.
    # `_fusible_crossing_values` decides that before any stage emits, because
    # groups are built one stage at a time while that stage emits.
    fused_part_dim, fused_ids = _fusible_crossing_values(
        G, stages, consumer_stages, node_stage, resident_ids
    )
    resident_ids = [*resident_ids, *fused_ids]

    # A crossing per-partition scalar crosses in SBUF, through `G.scalar_bufs`,
    # never through HBM. The plan defines the set (`sbuf_scalar_crossings`) and
    # each of the four independent HBM sites below excludes it.
    scalar_crossings = set(plan.sbuf_scalar_crossings)
    buffered_ids = [
        node_id
        for node_id in consumer_stages
        if node_id not in resident_ids and node_id not in scalar_crossings
    ]

    unsupported_attrs: dict[str, dict[str, object]] = {}
    for stage in stages:
        for node_id in stage.node_ids:
            node = G.node_attrs[node_id]
            nondefault = _has_only_default_isa_attrs(node.op, node.attrs)
            if nondefault:
                unsupported_attrs[node_id] = nondefault
    if unsupported_attrs:
        raise UnsupportedEmission(
            "generic matmul: staged emission cannot preserve non-default ISA "
            f"attributes: {unsupported_attrs}"
        )

    lines = _emit_dims_preamble(
        G,
        ind,
        input_nodes,
        plan.output_id,
        _present_dims(G),
        dtype_expr.split(".")[0],
    )

    seen: set[str] = set()
    for node_id in buffered_ids:
        if node_id in seen:
            continue
        seen.add(node_id)
        layout = G.layout(node_id)
        if layout.free is None:
            raise UnsupportedEmission(
                f"generic matmul: staged HBM value {node_id!r} is a "
                "per-partition scalar, which no HBM helper can shape"
            )
        pt = G.tv(G.dim_name(layout.part))
        ft = G.tv(G.dim_name(layout.free))
        hbm_name = G.fresh_var(f"{nki_safe_var(node_id)}_hbm")
        G.hbm_bufs[node_id] = hbm_name
        lines += [
            f"{ind}{hbm_name} = nl.ndarray(",
            f"{ind}    ({pt.extent}, {ft.extent}), dtype={dtype_expr},",
            f"{ind}    buffer=nl.shared_hbm)",
        ]
    lines.append("")

    # Pattern 1b: one shared `part` block loop above every stage, so a resident
    # buffer allocated here is live across the consuming stage. Each group then
    # skips its own `part` loop, which sat at exactly this level.
    if fused_part_dim is not None:
        fused_part = G.tv(fused_part_dim)
        lines.append(
            f"{ind}for {fused_part.loop} in nl.affine_range({fused_part.num_block}):"
        )
        for node_id in fused_ids:
            # A fused value's resident buffer spans its OWN part axis: the loop
            # that walks it in the producer is the producing group's free loop,
            # which is this value's part dim. The hoisted loop is the other one.
            G.part_spanning.add(node_id)
            lines += _emit_resident_alloc(G, 2, ind, node_id, dtype_expr)

    for stage in stages:
        stage_reduce = stage.reduces[0] if stage.reduces else None
        # The numerator is emitted inside the reducer, not by the inter chain.
        stage_inters = [
            inter for inter in stage.inters if inter.node_id != numerator_id
        ]
        inter_by_id = {inter.node_id: inter for inter in stage_inters}
        inter_inputs = {
            input_id for inter in stage_inters for input_id in inter.input_ids
        }
        # A crossing scalar resolves through `G.scalar_bufs`, not through a block
        # buffer, so it must not reach the reload below: `_emit_hbm_block_load`
        # would index a `G.hbm_bufs` entry that does not exist.
        crossing_in = [
            node_id
            for node_id in stage.live_in_ids
            if node_id in inter_inputs and node_id not in scalar_crossings
        ]

        groups = _stage_groups(G, stage)

        # A node emits in exactly one of the stage's groups, and stores once. A
        # crossing scalar reaches no store, and correctly so: its buffer is live
        # across the consuming stage already, so the `unstored` refusal below must
        # not fire for it.
        emitted: set[str] = set()
        stored: set[str] = set()
        crossing_out = [
            node_id
            for node_id in stage.live_out_ids
            if node_stage.get(node_id) == stage.index
            and node_id not in scalar_crossings
        ]

        for (part_dim, free_dim, _accum), mps in groups.items():
            part = G.tv(part_dim)
            free = G.tv(free_dim)
            group_dims = {part_dim, free_dim}
            group_ids = {mp.mm_id for mp in mps}
            owns_reduce = (
                stage_reduce is not None and stage_reduce.source_ids[0] in group_ids
            )
            # The hoisted loop IS this group's part loop (the prepass established
            # every group shares one part dim), so opening a second one here would
            # shadow it and reset the resident buffer's index.
            if fused_part_dim is None:
                lines.append(
                    f"{ind}for {part.loop} in nl.affine_range({part.num_block}):"
                )
            else:
                assert part_dim == fused_part_dim

            # Both resident reducer values span the free axis, so their buffers are
            # hoisted above its loop. The reduce source is a matmul drain target,
            # hence `accum_alloc`; the numerator stays the graph's dtype, not fp32,
            # because it feeds a transpose of that dtype anyway (pattern 5).
            preallocated: dict[str, str] = {}
            if owns_reduce:
                source_id = stage_reduce.source_ids[0]
                lines += _emit_resident_alloc(
                    G,
                    2,
                    ind,
                    source_id,
                    dtype_expr,
                    alloc=G.accum_alloc(_group_accum_dim(G, mps)),
                )
                lines += _emit_resident_alloc(
                    G, 2, ind, numerator_id, dtype_expr, alloc="nl.ndarray"
                )

            lines.append(
                f"{ind * 2}for {free.loop} in nl.affine_range({free.num_block}):"
            )
            if owns_reduce:
                # Block-local alias, not the spanning name: see
                # `_emit_resident_block_alias`.
                alias_lines, alias = _emit_resident_block_alias(
                    G,
                    3,
                    ind,
                    stage_reduce.source_ids[0],
                    f"{nki_safe_var(stage_reduce.source_ids[0])}_block",
                )
                lines += alias_lines
                preallocated[stage_reduce.source_ids[0]] = alias
            stage_lines, accum_buffers = _emit_accum_stage(
                G,
                mps,
                3,
                ind,
                dtype_expr,
                alias_accum_dim=True,
                operand_free_from_output=True,
                preallocated=preallocated,
            )
            lines += stage_lines
            id_to_buf = dict(accum_buffers)

            # Store this group's crossing matmul results while their accumulators
            # are still live: the reducer below reopens the free loop, and a store
            # after that reopen would write one stale block into every block. A
            # resident value needs no store; it has no HBM buffer at all.
            for node_id in crossing_out:
                if node_id not in group_ids or node_id in G.resident_bufs:
                    continue
                lines += _emit_hbm_store(G, 3, ind, node_id, id_to_buf[node_id])
                stored.add(node_id)

            # A reduce whose source this group produced is hoisted to just inside
            # the part loop, after the free loop that filled the resident buffer.
            if owns_reduce:
                lines += _emit_stage_reduce(G, 2, ind, stage, numerator_id)
                lines.append(
                    f"{ind * 2}for {free.loop} in nl.affine_range({free.num_block}):"
                )
                alias_lines, numerator_tile = _emit_resident_block_alias(
                    G,
                    3,
                    ind,
                    numerator_id,
                    f"{nki_safe_var(numerator_id)}_stage_tiles",
                )
                lines += alias_lines
                # The accumulators above are dead past the reopen, so only the
                # resident numerator seeds the inter chain.
                id_to_buf = {numerator_id: numerator_tile}

            # An inter node reads block buffers, so load any crossing value one
            # of them consumes. A resident one is already in SBUF, so it needs an
            # alias rather than a DMA. A crossing matmul operand needs neither:
            # ``_emit_operand_load`` resolves it itself.
            for node_id in crossing_in:
                if node_id in id_to_buf:
                    continue
                if node_id in G.resident_bufs:
                    load_lines, tile = _emit_resident_block_alias(
                        G, 3, ind, node_id, f"{nki_safe_var(node_id)}_stage_tiles"
                    )
                else:
                    load_lines, tile = _emit_hbm_block_load(
                        G,
                        3,
                        ind,
                        node_id,
                        f"{nki_safe_var(node_id)}_stage_tiles",
                        dtype_expr,
                    )
                lines += load_lines
                id_to_buf[node_id] = tile

            # Each inter node sits in the one group whose block loops span its
            # layout and whose buffers cover its inputs. A per-partition scalar
            # comes from G.scalar_bufs, so it gates on that map instead.
            for node_id in stage.node_ids:
                inter = inter_by_id.get(node_id)
                if inter is None or node_id in emitted:
                    continue
                layout = G.layout(node_id)
                if (
                    layout.free is None
                    or not {
                        G.dim_name(layout.part),
                        G.dim_name(layout.free),
                    }
                    <= group_dims
                ):
                    continue
                if any(
                    input_id not in id_to_buf
                    if G.layout(input_id).free is not None
                    else input_id not in G.scalar_bufs
                    for input_id in inter.input_ids
                ):
                    continue
                # A resident node writes its spanning buffer through a block-local
                # alias, and emits no store afterwards.
                dest = None
                if node_id in G.resident_bufs:
                    alias_lines, dest = _emit_resident_block_alias(
                        G, 3, ind, node_id, f"{nki_safe_var(node_id)}_inter"
                    )
                    lines += alias_lines
                inter_lines, inter_buf = _emit_inter_node(
                    G, inter, id_to_buf, 3, ind, dtype_expr, None, dest=dest
                )
                lines += inter_lines
                id_to_buf[node_id] = inter_buf
                emitted.add(node_id)

            for node_id in crossing_out:
                if (
                    node_id in stored
                    or node_id not in emitted
                    or node_id in G.resident_bufs
                ):
                    continue
                lines += _emit_hbm_store(G, 3, ind, node_id, id_to_buf[node_id])
                stored.add(node_id)

            # The output is usually the group's matmul, but it is an inter node
            # when the graph normalizes AFTER the output matmul, whose sink is the
            # normalizing `tensor_scalar`. Either way the store goes in the group
            # that produced it, while its buffer is live.
            if plan.output_id in group_ids or (
                plan.output_id in emitted and plan.output_id in id_to_buf
            ):
                lines += _emit_result_store(
                    G, 3, ind, plan.output_id, id_to_buf[plan.output_id]
                )

        # Emitting a partial body would miscompile silently, so refuse instead.
        unplaced = [
            inter.node_id for inter in stage_inters if inter.node_id not in emitted
        ]
        if unplaced:
            raise UnsupportedEmission(
                f"generic matmul: stage {stage.index} nodes {unplaced} fit no "
                "matmul group's loop nest, so no scheduled place emits them"
            )
        # A resident value legitimately reaches no store: it never leaves SBUF,
        # so there is no HBM for a later stage to read unwritten.
        unstored = sorted(set(crossing_out) - stored - set(G.resident_bufs))
        if unstored:
            raise UnsupportedEmission(
                f"generic matmul: stage {stage.index} crossing values {unstored} "
                "reach no store, so a later stage would read unwritten HBM"
            )

    return lines, ["result"]


def _emit_multi_matmul_body(
    G: _GenericMM,
    ind: str,
    input_nodes: list[Node],
    dtype_expr: str,
) -> tuple[list[str], list[str]]:
    """Emit a multi-matmul graph: a reduce-bearing one through the stage driver,
    else a side-matmul (part_*_mlp) or chained-matmul (relu/silu_mlp) graph.

    Structure comes entirely from the plan: earlier matmuls sharing a block level
    and contraction dim accumulate together (``_emit_accum_stage``); the
    inter-matmul nodes (``plan.inter_ids``, in topo order) are emitted per-tile;
    a chained output matmul (``plan.chained_mm_id``) consumes the plan's single
    materialized combine (``plan.materialized_inter_id``) as its stationary
    operand."""
    plan = G.plan
    nest = G.nest
    output_id = plan.output_id

    # `_build_multi_stage` is the sole producer of `plan.stages`, and it drives
    # the staged renderer whether or not the graph carries a reduce (a bare
    # matmul→matmul chain, e.g. linear attention, stages with zero reduces). The
    # side-/chained-matmul renderer below handles only the stage-free plans
    # `_build_multi` produces.
    if plan.stages:
        return _emit_staged_body(G, ind, input_nodes, dtype_expr)

    chained_id = plan.chained_mm_id
    chained_mp = plan.matmul(chained_id) if chained_id is not None else None
    earlier_mps = [mp for mp in plan.matmuls if mp.mm_id != chained_id]
    inters = list(plan.inters)

    lines = _emit_dims_preamble(
        G,
        ind,
        input_nodes,
        output_id,
        _present_dims(G),
        dtype_expr.split(".")[0],
    )

    materialized = set(nest.materialized)

    if chained_mp is None:
        # FLAT (part_*_mlp): block loops [m, free]; the output combine IS the
        # last inter node, written straight into the HBM store.
        block_loops = nest.block_loops
        for lvl, dim in enumerate(block_loops):
            tv = G.tv(dim)
            lines.append(
                f"{ind * (lvl + 1)}for {tv.loop} in nl.affine_range({tv.num_block}):"
            )
        nblk = len(block_loops)
        stage_lines, accum_buf = _emit_accum_stage(
            G, earlier_mps, nblk + 1, ind, dtype_expr
        )
        lines += stage_lines
        inter_lines, id_to_buf = _emit_inter_chain(
            G, inters, accum_buf, nblk + 1, ind, dtype_expr, set(), None
        )
        lines += inter_lines
        store_buf = id_to_buf[output_id]
        lines += _emit_result_store(G, nblk + 1, ind, output_id, store_buf)
        return lines, ["result"]

    # CHAINED (relu/silu_mlp): m loop encloses a first stage (n-block loop with
    # the earlier matmuls + inter-chain materializing the combine over n) and a
    # second stage (p-block loop with the chained matmul accumulating over n).
    accum_dim = chained_mp.accum_dim  # the materialized dim (n)
    nt = G.tv(accum_dim)
    out_lay = G.layout(chained_id)
    mt = G.tv(G.dim_name(out_lay.part))  # m
    pt = G.tv(G.dim_name(out_lay.free))  # p

    mat_id = plan.materialized_inter_id

    lines.append(f"{ind}for {mt.loop} in nl.affine_range({mt.num_block}):")
    # Materialized combine buffer: allocated per m-block, live across both stages.
    _mat_buf, mat_alloc_lines = _mat_buf_alloc(G, mat_id, ind * 2, dtype_expr)
    lines += mat_alloc_lines
    # First stage: for n: accumulate earlier mms over k, materialize combine.
    lines.append(f"{ind * 2}for {nt.loop} in nl.affine_range({nt.num_block}):")
    stage1_lines, accum_buf = _emit_accum_stage(G, earlier_mps, 3, ind, dtype_expr)
    lines += stage1_lines
    inter_lines, id_to_buf = _emit_inter_chain(
        G, inters, accum_buf, 3, ind, dtype_expr, materialized, nt.loop
    )
    lines += inter_lines
    mat_buf = id_to_buf[mat_id]

    # Second stage: for p: accumulate the chained matmul over n, then store.
    lines.append(f"{ind * 2}for {pt.loop} in nl.affine_range({pt.num_block}):")
    out_tiles = "out_tiles"
    lines += [
        f"{ind * 3}{out_tiles} = {G.accum_alloc(accum_dim)}(",
        f"{ind * 3}    ({mt.tile}, {mt.tiles_in_block}, {pt.block}),",
        f"{ind * 3}    dtype={dtype_expr}, buffer=nl.sbuf)",
        "",
        f"{ind * 3}for {nt.loop} in nl.sequential_range({nt.num_block}):",
    ]
    # w3 (the chained moving operand) loads per (n-step, p-block).
    w3_free = G.dim_name(G.layout(chained_mp.mov.op_id).free)  # p
    load_lines, w3_tile = _emit_operand_load(
        G, 4, ind, chained_mp.mov, accum_dim, w3_free
    )
    lines += load_lines
    lines += _emit_matmul_accumulate(
        G,
        4,
        ind,
        chained_id,
        mat_buf,
        w3_tile,
        accum_dim,
        G.dim_name(out_lay.part),
        w3_free,
        out_tiles,
        psum_var=f"ps_{nki_safe_var(chained_id)}",
        stat_block_index=nt.loop,
        mov_wide=chained_mp.mov_wide,
    )
    lines += _emit_result_store(G, 3, ind, output_id, out_tiles)
    return lines, ["result"]


def emit_matmul_body_generic(
    ctx: EmitCtx,
    input_nodes: list[Node],
    compute_nodes: list[Node],
    id_to_node: dict[str, Node],
    params: list[str],
    output_ids: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Build an ``EmissionPlan`` and translate it into NKI lines; returns ``(lines, out_bufs)``."""
    ind = ctx.indent
    ctx.tp_counter = 0
    ctx.nest_counter = 0

    input_node_ids = {n.id for n in input_nodes}

    if output_ids:
        declared_outputs = [node_id for node_id in output_ids if node_id in id_to_node]
        if len(declared_outputs) != 1:
            raise UnsupportedEmission(
                "generic matmul: expected one declared compute output, got "
                f"{sorted(declared_outputs)}"
            )
        output_id = declared_outputs[0]
    else:
        output_id = find_output_sink(compute_nodes)
    # The plan resolves each TILE_<D> from the bases the assembler published
    # (defaults or hardware-annotation derived), refusing before any line emits.
    plan = build_emission_plan(
        compute_nodes,
        id_to_node,
        input_node_ids,
        output_id,
        tile_config={
            "tile_m": ctx.tile("tile_m"),
            "tile_k": ctx.tile("tile_k"),
            "tile_n": ctx.tile("tile_n"),
        },
    )
    ctx.partition_scalar_ids = set(plan.scalar_ids)
    G = _GenericMM(ctx, plan, NodeAttrs(id_to_node))
    dtype_expr = f"{nki_safe_var(input_nodes[0].id)}.dtype"

    if plan.is_multi:
        return _emit_multi_matmul_body(G, ind, input_nodes, dtype_expr)

    # --- single matmul (classes A-D) --------------------------------------- #
    mm_plan = plan.matmuls[0]
    mm_id = mm_plan.mm_id
    accum_dim = mm_plan.accum_dim
    stat_free = G.dim_name(G.layout(mm_plan.stat.op_id).free)
    mov_free = G.dim_name(G.layout(mm_plan.mov.op_id).free)

    # Result dtype follows the stationary operand's underlying input (its root).
    stat_src_id = mm_plan.stat.root_id
    stat_dtype_id = G.staged_dtype_sources.get(stat_src_id, stat_src_id)

    lines: list[str] = []
    lines += _emit_dims_preamble(
        G,
        ind,
        input_nodes,
        output_id,
        _present_dims(G),
        nki_safe_var(stat_dtype_id),
    )

    # Reduce preamble inserts between the m loop (lvl=0) and the n loop (lvl=1).
    block_loops = plan.nest.block_loops
    for lvl, dim in enumerate(block_loops):
        tv = G.tv(dim)
        if lvl == 1 and plan.reduces:
            lines += _emit_reduce_preamble(
                G, lvl + 1, ind, plan.reduces, accum_dim, plan.staged_ids
            )
        lines.append(
            f"{ind * (lvl + 1)}for {tv.loop} in nl.affine_range({tv.num_block}):"
        )

    nblk = len(block_loops)
    kt = G.tv(accum_dim)
    out_lay = G.layout(mm_id)
    out_part = G.tv(G.dim_name(out_lay.part))
    out_free = G.tv(G.dim_name(out_lay.free))
    result_tiles = "result_tiles"

    # Block accumulator (spans partition-tiles x free block), then the
    # sequential contraction-block loop.
    i_acc = ind * (nblk + 1)
    lines += [
        f"{i_acc}{result_tiles} = {G.accum_alloc(accum_dim)}(",
        f"{i_acc}    ({out_part.tile}, {out_part.tiles_in_block}, {out_free.block}),",
        f"{i_acc}    dtype={nki_safe_var(stat_dtype_id)}.dtype, buffer=nl.sbuf)",
        "",
        f"{i_acc}for {kt.loop} in nl.sequential_range({kt.num_block}):",
    ]

    # Operand loads (deduped by node id — n-invariant nodes still emit once).
    loaded: dict[str, str] = {}
    stat_tile: str | None = None
    mov_tile: str | None = None
    for op, free_dim, is_stat in (
        (mm_plan.stat, stat_free, True),
        (mm_plan.mov, mov_free, False),
    ):
        if op.op_id in loaded:
            tile_var = loaded[op.op_id]
        else:
            load_lines, tile_var = _emit_operand_load(
                G, nblk + 2, ind, op, accum_dim, free_dim
            )
            lines += load_lines
            loaded[op.op_id] = tile_var
        if is_stat:
            stat_tile = tile_var
        else:
            mov_tile = tile_var

    lines += _emit_matmul_accumulate(
        G,
        nblk + 2,
        ind,
        mm_id,
        stat_tile,
        mov_tile,
        accum_dim,
        stat_free,
        mov_free,
        result_tiles,
        mov_wide=mm_plan.mov_wide,
    )

    store_buf = result_tiles
    if plan.post_mm:
        post_lines, store_buf = _emit_post_mm_chain(
            G,
            nblk + 1,
            ind,
            mm_id,
            plan.post_mm,
            result_tiles,
            f"{nki_safe_var(stat_dtype_id)}.dtype",
        )
        lines += post_lines

    lines += _emit_result_store(G, nblk + 1, ind, mm_id, store_buf)

    return lines, ["result"]
