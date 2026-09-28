from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import z3

from axon.candidate_filter import (
    MANIFEST_NAME,
    TileConstraint,
    manifest_literal,
)
from axon.codegen.bodies.elementwise import emit_elementwise_body
from axon.codegen.bodies.flat import emit_flat_body
from axon.codegen.bodies.matmul_generic import emit_matmul_body_generic
from axon.codegen.bodies.reduce import emit_reduce_body
from axon.codegen.bodies.scan import emit_scan_body
from axon.codegen.combine_emit import (
    SpmdEmitError,
    _allreduce_accumulator,
    _append_barrier,
    _emit_allreduce_epilogue,
    _gather_output_columns,
    _insert_preamble,
    _rebuffer_result,
    _result_var,
    _shard_block_loop,
    _slice_operands,
)
from axon.codegen.combine_plan import (
    _AXIS_LOOPS,
    _operand_slices,
    _output_ids,
    _sharded_output_axis,
    _terminal_combine,
)
from axon.codegen.constants import DEFAULT_ELEMENTWISE_TILE
from axon.codegen.context import EmitCtx
from axon.codegen.ops import (
    _strip_emitted_comments,
    nki_safe_var,
)
from axon.codegen.structure import (
    classify_kernel,
    extract_elementwise_tile_configs,
    extract_matmul_tile_config,
    has_lhs_nc_transpose,
    has_rhs_nc_transpose,
    matmul_has_chained,
)
from axon.ir import (
    _DEFAULT_BLOCK_DIMS,
    Node,
    TileAnnotation,
    _format_shape,
    _graph_symbolic_tensors,
    format_tile_annotation,
    format_tile_hardware_metadata,
    nuGraph,
    tile_graph_variant,
)
from axon.sharding import (
    OUTPUT_SINK,
    AllGather,
    AllReduce,
    Combine,
    Identity,
    ShardingPlan,
    Slice,
)
from axon.sharding_monoids import MONOIDS
from axon.synthesizer import _build_dag_levels


def _clamp_tile_to_extent(base: int, extent: int) -> int:
    """Non-matmul body tile size, mirroring the matmul path's `_resolve_tiles`
    clamp (plan.py): a tile base wider than the extent still emits when it divides
    down cleanly, so clamp to one full tile."""
    if extent < base and base % extent == 0:
        return extent
    return base


def _apply_hooks(
    src: str,
    prelude: Callable[[str], str] | None,
    combine: Callable[[str], str] | None,
) -> str:
    """Splice the optional `prelude`/`combine` source transforms onto the
    single-core `src`. With both `None` (the `emit()` path) this returns `src`
    unchanged, so single-core emission is byte-identical."""
    if prelude is not None:
        src = prelude(src)
    if combine is not None:
        src = combine(src)
    return src


def _shard_prelude(
    plan: ShardingPlan, G: nuGraph, indent: str = "    "
) -> Callable[[str], str] | None:
    """Build the lnc=2 *prelude* source transform for `plan`: insert the shard
    preamble (program id / count) as the first statements of the function body,
    and rebind each contraction/reduction-sharded input to its `SHARD_ID` half.

    Returns `None` when the plan is a replicate / lnc=1 no-op (so `_assemble`
    leaves the single-core source untouched). The operand-slice and plan
    analysis (`_operand_slices`) lives in `axon.codegen.combine_plan`; the
    line-templates (`_insert_preamble`, `_slice_operands`) live in
    `axon.codegen.combine_emit`.

    Operand slicing applies only to the *contraction* combines (C all-reduce and
    S-moment): a P / X-gather free-axis shard leaves operands full (the body's
    global block index already addresses the core's slice), so the prelude
    inserts only the shard preamble there.
    """
    output_combine = _terminal_combine(plan, G)
    if output_combine is None or (
        isinstance(output_combine, Identity) and not output_combine.barrier
    ):
        # Replicate / lnc=1: no preamble, no slices.
        return None
    # Slice operands only for a contraction shard (C all-reduce / S-moment),
    # never for a P / X-gather free-axis shard.
    internal_ar = any(
        isinstance(c, AllReduce) and e[1] != OUTPUT_SINK
        for e, c in plan.combines.items()
    )
    is_contraction = internal_ar or isinstance(output_combine, AllReduce)
    sliced = _operand_slices(plan, G) if is_contraction else {}

    def _prelude(src: str) -> str:
        out = _insert_preamble(src.splitlines(), indent)
        if sliced:
            out = _slice_operands(out, sliced, indent)
        return "\n".join(out) + "\n"

    return _prelude


def _combine_for(
    plan: ShardingPlan, G: nuGraph, indent: str = "    "
) -> Callable[[str], str] | None:
    """Build the lnc=2 *combine* source transform for `plan`: the cross-core
    monoid combine spliced after the body's output buffer, before the final
    `return`.

    The combine line-templates are the monoid-adjacent emitters in
    `axon.codegen.combine_emit` (P barrier / C tiled all-reduce / S-moment
    accumulator all-reduce + column gather); the dispatch here classifies the
    plan's terminal/internal combines to pick which one to splice. Returns
    `None` for a replicate / lnc=1 plan.
    """
    output_combine = _terminal_combine(plan, G)
    # Soundness guard: every non-terminal combine must be one we realize (an
    # operand `Slice` or an internal S-moment `AllReduce`); anything else is an
    # input-side gather / reshard we cannot lower.
    unlowerable = [
        c
        for e, c in plan.combines.items()
        if e != (_output_ids(G)[0], OUTPUT_SINK)
        and not isinstance(c, (Slice, AllReduce))
    ]
    if unlowerable:
        raise SpmdEmitError(
            f"plan {plan.tag()} has non-terminal combine(s) "
            f"{[type(c).__name__ for c in unlowerable]} the assembler cannot "
            f"lower (input gather / reshard — multi-segment, deferred)"
        )
    # S-moment: an internal all-reduce (a sharded reduction's partial statistic)
    # plus a terminal barrier.
    internal_ar = [
        (e, c)
        for e, c in plan.combines.items()
        if isinstance(c, AllReduce) and e[1] != OUTPUT_SINK
    ]
    if internal_ar:
        return _combine_s_moment(plan, G, internal_ar, output_combine, indent)
    combine = output_combine
    if combine is None or (isinstance(combine, Identity) and not combine.barrier):
        return None
    if isinstance(combine, Identity) and combine.barrier:
        return _combine_p(plan, G, indent)
    if isinstance(combine, AllReduce):
        return _combine_c(plan, G, combine, indent)
    if isinstance(combine, AllGather):
        # X-gather on the kernel output lowers via the P free-axis barrier path
        # (both disjoint halves already land in the one shared-HBM output).
        if len(_output_ids(G)) != 1:
            raise SpmdEmitError("multi-output X-gather not supported")
        return _combine_p(plan, G, indent)
    raise SpmdEmitError(f"combine {type(combine).__name__} not yet lowered")


def _combine_p(plan: ShardingPlan, G: nuGraph, indent: str) -> Callable[[str], str]:
    """P (free-axis) combine: shard the output-axis block loop across cores and
    barrier the shared output. The preamble is already inserted by the
    prelude."""
    if len(_output_ids(G)) != 1:
        raise SpmdEmitError(
            "multi-output P combine not supported (one barrier target only)"
        )
    axis = _sharded_output_axis(plan, G)
    if axis is None or axis not in _AXIS_LOOPS:
        raise SpmdEmitError(
            f"P plan shard_class={plan.shard_class} maps to no shardable block "
            f"loop (output axis {axis}; only the row axis 0 is a P shard)"
        )
    shard = _AXIS_LOOPS[axis]

    def _combine(src: str) -> str:
        out = _shard_block_loop(src.splitlines(), shard, indent)
        out = _append_barrier(out, indent)
        return "\n".join(out) + "\n"

    return _combine


def _combine_c(
    plan: ShardingPlan, G: nuGraph, combine: AllReduce, indent: str
) -> Callable[[str], str]:
    """C (contraction shard) combine: privatize the per-core partial then append
    the tiled all-reduce epilogue. Operands are already sliced by the prelude."""
    if len(_output_ids(G)) != 1:
        raise SpmdEmitError("multi-output all-reduce not supported")
    monoid = MONOIDS.get(combine.monoid)
    if monoid is None or monoid.combine_kind != "tensor_tensor" or monoid.nl_op is None:
        raise SpmdEmitError(
            f"all-reduce monoid {combine.monoid!r} is not a single-op "
            f"tensor_tensor combine (moment/welford/flash not yet supported)"
        )
    if not _operand_slices(plan, G):
        raise SpmdEmitError("C plan has no sliced contraction input")
    nl_op = monoid.nl_op

    def _combine_fn(src: str) -> str:
        out = src.splitlines()
        res_var = _result_var(out)
        out = _rebuffer_result(out, res_var)
        out = _emit_allreduce_epilogue(out, res_var, nl_op, indent)
        return "\n".join(out) + "\n"

    return _combine_fn


def _combine_s_moment(
    plan: ShardingPlan,
    G: nuGraph,
    internal_ar: list,
    terminal: Combine | None,
    indent: str,
) -> Callable[[str], str]:
    """S-moment combine: all-reduce the partial-statistic accumulator before the
    finalize, privatize the half-width output, and gather the per-core columns
    into the full-width shared output. Operands are sliced by the prelude."""
    if len(_output_ids(G)) != 1:
        raise SpmdEmitError("multi-output S-moment not supported")
    if len(internal_ar) != 1:
        raise SpmdEmitError(
            f"S-moment expects one internal all-reduce, got {len(internal_ar)}"
        )
    monoid = MONOIDS.get(internal_ar[0][1].monoid)
    if monoid is None or monoid.combine_kind != "tensor_tensor" or monoid.nl_op is None:
        raise SpmdEmitError(
            f"S-moment all-reduce monoid {internal_ar[0][1].monoid!r} is not a "
            f"single-op tensor_tensor combine (welford/flash land later)"
        )
    if not isinstance(terminal, Identity) or not terminal.barrier:
        raise SpmdEmitError("S-moment expects a terminal barrier after finalize")
    if not _operand_slices(plan, G):
        raise SpmdEmitError("S-moment plan has no sliced reduced input")
    nl_op = monoid.nl_op

    def _combine_fn(src: str) -> str:
        out = _allreduce_accumulator(src.splitlines(), nl_op, indent)
        res = _result_var(out)
        out = _rebuffer_result(out, res)
        out = _gather_output_columns(out, res, indent)
        return "\n".join(out) + "\n"

    return _combine_fn


class NKIEmitter:
    def __init__(
        self,
        kernel_name: str,
        variant_index: int = 0,
        indent: str = "    ",
        tile_config: dict[str, int] | None = None,
        tile_config_tag: str = "",
        fuse_loads: bool = False,
        rhs_transpose_strategy: str = "separate_loop",
    ) -> None:
        # variant_index only keys EmitOk/EmitErr results; it never affects
        # emission, so it stays off EmitCtx.
        self.variant_index = variant_index
        # tile_config may be None (no explicit config); emit() resolves it onto
        # _ctx.tile_config after extraction.
        self._unresolved_tile_config = tile_config
        self._ctx = EmitCtx(
            kernel_name=kernel_name,
            indent=indent,
            tile_config=dict(tile_config) if tile_config else {},
            tile_config_tag=tile_config_tag,
            fuse_loads=fuse_loads,
            rhs_transpose_strategy=rhs_transpose_strategy,
        )

    @property
    def kernel_name(self) -> str:
        return self._ctx.kernel_name

    @property
    def indent(self) -> str:
        return self._ctx.indent

    @property
    def tile_config(self) -> dict[str, int] | None:
        return self._unresolved_tile_config

    @property
    def tile_config_tag(self) -> str:
        return self._ctx.tile_config_tag

    @property
    def fuse_loads(self) -> bool:
        return self._ctx.fuse_loads

    @property
    def rhs_transpose_strategy(self) -> str:
        return self._ctx.rhs_transpose_strategy

    @property
    def tile_constraints(self) -> tuple[TileConstraint, ...]:
        return tuple(self._ctx.tile_constraints.values())

    def emit(
        self,
        G: nuGraph,
        tile_annotations: dict[str, TileAnnotation] | None = None,
    ) -> str:
        """Single-core emission: provably sharding-free (no prelude, no combine).

        This is `_assemble` with both splice points `None`, so the emitted
        source is byte-identical to the pre-refactor `emit()`.
        """
        return self._assemble(G, tile_annotations, prelude=None, combine=None)

    def emit_lnc2(
        self,
        G: nuGraph,
        plan: ShardingPlan,
        tile_annotations: dict[str, TileAnnotation] | None = None,
        plan_graph: nuGraph | None = None,
    ) -> str:
        """SPMD (lnc=2) emission: `_assemble` with the plan's shard prelude and
        combine spliced in.

        `G` is the hardware graph `_assemble` emits the body from (its ops have
        registered emitters). `plan_graph` is the graph the `plan` was built
        against (the high-level graph whose node ids the plan's combine keys
        reference); it defaults to `G` for the case the two coincide. The
        prelude/combine hooks read `plan_graph` (plan analysis), never `G`.

        The prelude/combine hooks (`_shard_prelude`, `_combine_for`) drive the
        monoid-adjacent line-templates in `axon.codegen.combine_emit` over the
        shared `_assemble` core (no `emit_spmd` text-patching). `_assemble`
        builds the single-core source exactly as `emit()` does, then the prelude
        and combine transform it for `plan` — so an lnc=1 emit and the source an
        lnc=2 emit starts from are identical.
        """
        pg = plan_graph if plan_graph is not None else G
        return self._assemble(
            G,
            tile_annotations,
            prelude=_shard_prelude(plan, pg),
            combine=_combine_for(plan, pg),
        )

    def _assemble(
        self,
        G: nuGraph,
        tile_annotations: dict[str, TileAnnotation] | None = None,
        *,
        prelude: Callable[[str], str] | None,
        combine: Callable[[str], str] | None,
    ) -> str:
        """Shared assembler for both targets.

        Builds signature + classify + tile-config + body + final return into one
        single-core source string, then applies the optional `prelude` (shard
        preamble + operand slices) and `combine` (cross-core monoid combine)
        source transforms. With `prelude=None, combine=None` (the `emit()` path)
        the result is the unmodified single-core source — no sharding concept is
        constructed or read, so the lnc=1 guarantee is structural.
        """
        levels = _build_dag_levels(G)
        topo_nodes: list[Node] = [node for level in levels for node in level]

        input_nodes = [n for n in topo_nodes if n.op == "input"]
        # Honor the kernel-spec input order so the emitted signature matches
        # positional bench calls (topo order can scramble operand order).
        if G.input_ids:
            order = {nid: idx for idx, nid in enumerate(G.input_ids)}
            input_nodes.sort(key=lambda n: order.get(n.id, len(order)))
        compute_nodes = [n for n in topo_nodes if n.op != "input"]
        id_to_node: dict[str, Node] = {n.id: n for n in topo_nodes}

        params: list[str] = [nki_safe_var(n.id) for n in input_nodes]
        safe_kname = nki_safe_var(self.kernel_name)
        fn_name = safe_kname

        klass = classify_kernel(compute_nodes)
        ind = self.indent

        # scan has no contiguous store seam (its stores interleave with the
        # carry), so there is no point to splice a cross-core combine around.
        # Reject a non-trivial lnc=2 emission loudly rather than emit silent /
        # wrong sharded source. A replicate / lnc=1 plan yields prelude=None and
        # combine=None (see _shard_prelude / _combine_for), so emit() single-core
        # and replicate-plan emit_lnc2 still emit scan unchanged.
        if klass == "scan" and (prelude is not None or combine is not None):
            raise SpmdEmitError(
                "scan kernels have no contiguous store seam (stores interleave "
                "with the carry); emit_lnc2 cannot lower a scan body for lnc=2. "
                "Use emit() for single-core scan emission."
            )

        if tile_annotations is None:
            try:
                tile_annotations = tile_graph_variant(G)
            except Exception:
                tile_annotations = {}

        tile_cfg = self._unresolved_tile_config
        if tile_cfg is None:
            if klass in ("elementwise", "scan"):
                configs = extract_elementwise_tile_configs(
                    compute_nodes, tile_annotations
                )
                tile_cfg = configs[0] if configs else dict(DEFAULT_ELEMENTWISE_TILE)
            elif klass == "matmul":
                tile_cfg = extract_matmul_tile_config(compute_nodes, tile_annotations)
            elif klass == "reduce":
                tile_cfg = dict(DEFAULT_ELEMENTWISE_TILE)
            else:
                tile_cfg = {}
        # Publish the resolved tile config on the ctx the bodies read.
        self._ctx.tile_config = dict(tile_cfg)
        self._ctx.tile_constraints = {}

        if klass == "matmul":
            _input_node_ids = {n.id for n in input_nodes}
            _has_chained_p = matmul_has_chained(
                compute_nodes, id_to_node, _input_node_ids
            )
            if _has_chained_p:
                tile_params = (
                    "TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K,"
                    " TILES_IN_BLOCK_P"
                )
            else:
                tile_params = "TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K"
        elif klass in ("elementwise", "reduce", "scan"):
            tile_params = "TILES_IN_BLOCK_M, TILES_IN_BLOCK_N"
        else:
            tile_params = ""

        # All body classes now allocate and return their own output buffer(s),
        # so the signature never carries an `output` HBM param: tile-bearing
        # classes append their TILES_* params, and flat (no tile params) takes
        # only the inputs.
        sig_params = params + [tile_params] if tile_params else params

        import_lines = [
            "import numpy as np",
            "import nki",
            "import nki.language as nl",
            "import nki.isa as nisa",
        ]
        lines: list[str] = [
            *import_lines,
            "",
            "",
            "@nki.jit",
            f"def {fn_name}({', '.join(sig_params)}):",
        ]

        if not compute_nodes:
            lines.append(f"{ind}pass  # empty graph")
            return _apply_hooks("\n".join(lines) + "\n", prelude, combine)

        if klass == "elementwise":
            shape = tuple(
                input_nodes[0].attrs.get("shape") or input_nodes[0].shape or ()
            )
            if len(shape) == 2:
                tile_m = _clamp_tile_to_extent(int(tile_cfg["tile_m"]), int(shape[0]))
                tile_n = _clamp_tile_to_extent(int(tile_cfg["tile_n"]), int(shape[1]))
                self._ctx.tile_config["tile_m"] = tile_m
                self._ctx.tile_config["tile_n"] = tile_n
                self._ctx.tile_constraints = {
                    "TILES_IN_BLOCK_M": TileConstraint(
                        "TILES_IN_BLOCK_M", int(shape[0]), tile_m, True
                    ),
                    "TILES_IN_BLOCK_N": TileConstraint(
                        "TILES_IN_BLOCK_N", int(shape[1]), tile_n, False
                    ),
                }
            body, out_bufs = emit_elementwise_body(
                self._ctx,
                input_nodes,
                compute_nodes,
                params,
                output_ids=list(G.output_ids) if G.output_ids else None,
            )
        elif klass == "matmul":
            # The matmul class is emitted from the orientation-aware generic
            # layout+nest emitter (Task 9 cutover; the old pattern-matched body
            # is deleted).
            body, out_bufs = emit_matmul_body_generic(
                self._ctx,
                input_nodes,
                compute_nodes,
                id_to_node,
                params,
                output_ids=set(G.output_ids) if G.output_ids else None,
            )
        elif klass == "reduce":
            shape = tuple(
                input_nodes[0].attrs.get("shape") or input_nodes[0].shape or ()
            )
            if len(shape) == 2:
                # As for elementwise: `extent // BLOCK` with no remainder
                # pass, so a non-dividing block silently drops the tail. Clamp
                # each tile to its extent so a small reduction axis (decode's
                # d_head) tiles as one block; write back to ctx for the body.
                tile_m = _clamp_tile_to_extent(int(tile_cfg["tile_m"]), int(shape[0]))
                tile_n = _clamp_tile_to_extent(int(tile_cfg["tile_n"]), int(shape[1]))
                self._ctx.tile_config["tile_m"] = tile_m
                self._ctx.tile_config["tile_n"] = tile_n
                self._ctx.tile_constraints = {
                    "TILES_IN_BLOCK_M": TileConstraint(
                        "TILES_IN_BLOCK_M", int(shape[0]), tile_m, True
                    ),
                    "TILES_IN_BLOCK_N": TileConstraint(
                        "TILES_IN_BLOCK_N", int(shape[1]), tile_n, True
                    ),
                }
            body, out_bufs = emit_reduce_body(
                self._ctx,
                input_nodes,
                compute_nodes,
                id_to_node,
                params,
            )
        elif klass == "scan":
            shape = tuple(
                input_nodes[0].attrs.get("shape") or input_nodes[0].shape or ()
            )
            if len(shape) == 2:
                # The scan body carries an N remainder pass (`REM_N`), so N
                # need not divide the extent; M has no such tail. Clamp each tile
                # to its extent (only fires for a cleanly-dividing small extent,
                # leaving the remainder path untouched); write back to ctx.
                tile_m = _clamp_tile_to_extent(int(tile_cfg["tile_m"]), int(shape[0]))
                tile_n = _clamp_tile_to_extent(int(tile_cfg["tile_n"]), int(shape[1]))
                self._ctx.tile_config["tile_m"] = tile_m
                self._ctx.tile_config["tile_n"] = tile_n
                self._ctx.tile_constraints = {
                    "TILES_IN_BLOCK_M": TileConstraint(
                        "TILES_IN_BLOCK_M", int(shape[0]), tile_m, True
                    ),
                    "TILES_IN_BLOCK_N": TileConstraint(
                        "TILES_IN_BLOCK_N", int(shape[1]), tile_n, False
                    ),
                }
            body, out_bufs = emit_scan_body(
                self._ctx,
                input_nodes,
                compute_nodes,
                params,
            )
        else:
            body, out_bufs = emit_flat_body(
                self._ctx, input_nodes, compute_nodes, params
            )

        if self._ctx.tile_constraints:
            lines[4:4] = [
                f"{MANIFEST_NAME} = {manifest_literal(self.tile_constraints)}",
                "",
            ]
        lines.extend(_strip_emitted_comments(body))
        # The body builders emit "compute + store into a buffer" and hand back
        # the output buffer name(s); emit() is the sole emitter of the final
        # `return`. This leaves a splice point for a future lnc=2 combine between
        # the store and the return.
        if out_bufs:
            if len(out_bufs) == 1:
                lines.append(f"{ind}return {out_bufs[0]}")
            else:
                lines.append(f"{ind}return ({', '.join(out_bufs)})")
        return _apply_hooks("\n".join(lines) + "\n", prelude, combine)

    def emit_tile_variants(
        self,
        G: nuGraph,
        tile_annotations: dict[str, TileAnnotation] | None = None,
    ) -> list[str]:
        """Single-core: one source per tile-config / strategy sub-variant."""
        return [
            code
            for code, _constraints, _tag in self.emit_tile_variant_records(
                G, tile_annotations
            )
        ]

    def emit_tile_variant_records(
        self,
        G: nuGraph,
        tile_annotations: dict[str, TileAnnotation] | None = None,
    ) -> list[tuple[str, tuple[TileConstraint, ...], str]]:
        records: list[tuple[str, tuple[TileConstraint, ...], str]] = []
        for sub, ann in self._tile_sub_emitters(G, tile_annotations):
            code = sub.emit(G, ann)
            records.append((code, sub.tile_constraints, sub.tile_config_tag))
        return records

    def emit_lnc2_tile_variants(
        self,
        G: nuGraph,
        plan: ShardingPlan,
        tile_annotations: dict[str, TileAnnotation] | None = None,
        plan_graph: nuGraph | None = None,
    ) -> list[str]:
        """SPMD: the same per-tile-config / strategy sub-variant sweep as
        `emit_tile_variants`, each emitted through `emit_lnc2(G, plan)` so the
        lnc=2 sweep mirrors the lnc=1 one (same (variant, tile) indexing)."""
        return [
            sub.emit_lnc2(G, plan, ann, plan_graph=plan_graph)
            for sub, ann in self._tile_sub_emitters(G, tile_annotations)
        ]

    def _tile_sub_emitters(
        self,
        G: nuGraph,
        tile_annotations: dict[str, TileAnnotation] | None = None,
    ) -> list[tuple[NKIEmitter, dict[str, TileAnnotation]]]:
        """Enumerate the (sub-emitter, tile-annotations) pairs for the tile-config
        / nc-transpose-strategy / load-fusion sweep. Shared by the single-core and
        lnc=2 tile-variant entrypoints so both sweep the same configs in the same
        order."""
        if tile_annotations is None:
            try:
                tile_annotations = tile_graph_variant(G)
            except Exception:
                tile_annotations = {}

        levels = _build_dag_levels(G)
        topo_nodes = [node for level in levels for node in level]
        input_nodes = [n for n in topo_nodes if n.op == "input"]
        if G.input_ids:
            order = {nid: idx for idx, nid in enumerate(G.input_ids)}
            input_nodes.sort(key=lambda n: order.get(n.id, len(order)))
        compute_nodes = [n for n in topo_nodes if n.op != "input"]
        klass = classify_kernel(compute_nodes)

        if klass in ("elementwise", "scan"):
            configs = extract_elementwise_tile_configs(compute_nodes, tile_annotations)
        elif klass == "matmul":
            configs = [extract_matmul_tile_config(compute_nodes, tile_annotations)]
        elif klass == "reduce":
            configs = [dict(DEFAULT_ELEMENTWISE_TILE)]
        else:
            configs = [{}]

        has_multi_input = klass == "elementwise" and len(input_nodes) > 1

        id_to_node_ev: dict[str, Node] = {n.id: n for n in topo_nodes}
        has_rhs_trans = klass == "matmul" and has_rhs_nc_transpose(
            compute_nodes, id_to_node_ev
        )
        has_lhs_trans = (
            klass == "matmul"
            and not has_rhs_trans
            and has_lhs_nc_transpose(compute_nodes, id_to_node_ev)
        )
        has_nc_trans = has_rhs_trans or has_lhs_trans

        pairs: list[tuple[NKIEmitter, dict[str, TileAnnotation]]] = []
        for cfg in configs:
            if cfg:
                tag = "tm{tile_m}_tn{tile_n}".format(**cfg)
                if "tile_k" in cfg:
                    tag = "tm{tile_m}_tk{tile_k}_tn{tile_n}".format(**cfg)
            else:
                tag = ""

            if has_nc_trans:
                assert has_rhs_trans != has_lhs_trans, (
                    "has_rhs_trans and has_lhs_trans must be mutually exclusive"
                )
                side_prefix = "rhs" if has_rhs_trans else "lhs"
                nc_trans_variants: list[tuple[str, str]] = [
                    (
                        "same_loop",
                        f"{tag}_{side_prefix}_same_loop"
                        if tag
                        else f"{side_prefix}_same_loop",
                    ),
                    (
                        "separate_loop",
                        f"{tag}_{side_prefix}_separate_loop"
                        if tag
                        else f"{side_prefix}_separate_loop",
                    ),
                    (
                        "load_transpose2d",
                        f"{tag}_{side_prefix}_load_transpose2d"
                        if tag
                        else f"{side_prefix}_load_transpose2d",
                    ),
                ]
                for strategy, variant_tag in nc_trans_variants:
                    sub_emitter = NKIEmitter(
                        kernel_name=self.kernel_name,
                        variant_index=self.variant_index,
                        indent=self.indent,
                        tile_config=cfg if cfg else None,
                        tile_config_tag=variant_tag,
                        fuse_loads=False,
                        rhs_transpose_strategy=strategy,
                    )
                    pairs.append((sub_emitter, tile_annotations))
            elif has_multi_input:
                load_variants: list[tuple[bool, str]] = [
                    (False, f"{tag}_loads_unfused" if tag else "loads_unfused"),
                    (True, f"{tag}_loads_fused" if tag else "loads_fused"),
                ]
                for fuse, variant_tag in load_variants:
                    sub_emitter = NKIEmitter(
                        kernel_name=self.kernel_name,
                        variant_index=self.variant_index,
                        indent=self.indent,
                        tile_config=cfg if cfg else None,
                        tile_config_tag=variant_tag,
                        fuse_loads=fuse,
                    )
                    pairs.append((sub_emitter, tile_annotations))
            else:
                sub_emitter = NKIEmitter(
                    kernel_name=self.kernel_name,
                    variant_index=self.variant_index,
                    indent=self.indent,
                    tile_config=cfg if cfg else None,
                    tile_config_tag=tag,
                    fuse_loads=False,
                )
                pairs.append((sub_emitter, tile_annotations))
        return pairs


def emit(
    G: nuGraph,
    kernel_name: str = "",
    tile_annotations: dict[str, TileAnnotation] | None = None,
) -> str:
    """Module-level single-core entrypoint over the shared `_assemble` core.

    Sharding-free by construction (`prelude=None, combine=None`): byte-identical
    to `NKIEmitter.emit`."""
    return NKIEmitter(kernel_name=kernel_name).emit(
        G, tile_annotations=tile_annotations
    )


def emit_lnc2(
    G: nuGraph,
    plan: ShardingPlan,
    kernel_name: str = "",
    tile_annotations: dict[str, TileAnnotation] | None = None,
    plan_graph: nuGraph | None = None,
) -> str:
    """Module-level SPMD (lnc=2) entrypoint over the shared `_assemble` core.

    `G` is the hardware graph the body is emitted from; `plan_graph` (default
    `G`) is the graph the `plan` was built against (the high-level graph whose
    node ids the plan's combine keys reference)."""
    return NKIEmitter(kernel_name=kernel_name).emit_lnc2(
        G, plan, tile_annotations=tile_annotations, plan_graph=plan_graph
    )


def emit_nki_code(
    G: nuGraph,
    kernel_name: str,
    variant_index: int = 0,
    tile_annotations: dict[str, TileAnnotation] | None = None,
) -> str:
    emitter = NKIEmitter(
        kernel_name=kernel_name,
        variant_index=variant_index,
    )
    return emitter.emit(G, tile_annotations=tile_annotations)


def emit_nki_code_tile_variants(
    G: nuGraph,
    kernel_name: str,
    variant_index: int = 0,
    tile_annotations: dict[str, TileAnnotation] | None = None,
) -> list[str]:
    emitter = NKIEmitter(
        kernel_name=kernel_name,
        variant_index=variant_index,
    )
    return emitter.emit_tile_variants(G, tile_annotations=tile_annotations)


@dataclass(frozen=True)
class EmitOk:
    variant_index: int
    tile_index: int
    code: str
    tile_constraints: tuple[TileConstraint, ...] = ()
    tile_config_tag: str = ""


@dataclass(frozen=True)
class EmitErr:
    variant_index: int
    error: Exception


EmitResult = EmitOk | EmitErr


def emit_nki_code_variants(
    hw_variants: list[nuGraph],
    kernel_name: str,
) -> Iterator[EmitResult]:
    for i, g_hw in enumerate(hw_variants):
        try:
            tile_ann = tile_graph_variant(g_hw)
        except Exception:
            tile_ann = {}
        try:
            tile_codes = NKIEmitter(
                kernel_name=kernel_name,
                variant_index=i,
            ).emit_tile_variant_records(g_hw, tile_annotations=tile_ann)
        except Exception as exc:
            yield EmitErr(i, exc)
            continue
        for j, (code, constraints, tag) in enumerate(tile_codes):
            yield EmitOk(i, j, code, constraints, tag)


def emit_nki_code_lnc2_variants(
    hw_variants: list[nuGraph],
    kernel_name: str,
    plan: ShardingPlan,
    plan_graph: nuGraph,
) -> Iterator[EmitResult]:
    """The lnc=2 counterpart of `emit_nki_code_variants`: emit every
    (hw variant, tile config) under `plan` via `emit_lnc2`, keeping the same
    (variant_index, tile_index) indexing as the single-core sweep.

    `plan_graph` is the graph the `plan` was built against (the high-level graph
    whose node ids the plan's combine keys reference); each `g_hw` is the body
    graph emission runs on. A plan the assembler cannot lower for a variant
    surfaces as an `EmitErr(i, exc)` (e.g. `SpmdEmitError`), matching the
    single-core generator's failure contract."""
    for i, g_hw in enumerate(hw_variants):
        try:
            tile_ann = tile_graph_variant(g_hw)
        except Exception:
            tile_ann = {}
        try:
            emitter = NKIEmitter(
                kernel_name=kernel_name,
                variant_index=i,
            )
            pairs = emitter._tile_sub_emitters(g_hw, tile_ann)
            tile_codes = []
            for sub, ann in pairs:
                code = sub.emit_lnc2(g_hw, plan, ann, plan_graph=plan_graph)
                tile_codes.append((code, sub.tile_constraints, sub.tile_config_tag))
        except Exception as exc:
            yield EmitErr(i, exc)
            continue
        for j, (code, constraints, tag) in enumerate(tile_codes):
            yield EmitOk(i, j, code, constraints, tag)


def print_graph(
    G: nuGraph,
    tile_annotations: dict[str, TileAnnotation] | None = None,
) -> None:
    symbolic_shapes: dict[str, tuple[Any, ...]] = {}
    sym_shape_fallback = "None"
    try:
        symbolic_shapes = {
            node_id: tensor.shape
            for node_id, tensor in _graph_symbolic_tensors(G).items()
        }
    except (KeyError, z3.Z3Exception):
        symbolic_shapes = {}
        sym_shape_fallback = "unavailable"
    topologically_ordered_nodes = [
        node for level in _build_dag_levels(G) for node in level
    ]
    for i, n in enumerate(topologically_ordered_nodes):
        sym_shape = symbolic_shapes.get(n.id)
        sym_shape_str = (
            _format_shape(sym_shape) if sym_shape is not None else sym_shape_fallback
        )
        display_attrs = {k: v for k, v in n.attrs.items() if k != "shape"}
        line = (
            f"[{i}] id={n.id:12s} op={n.op:10s} inputs={n.inputs} "
            f"sym_shape={sym_shape_str} attrs={display_attrs}"
        )
        if tile_annotations is not None:
            ann = tile_annotations.get(n.id)
            if ann is not None:
                reduction_axes = sorted(ann.reduction_axes)
                line += (
                    f" tiling={format_tile_annotation(n.id, ann, include_node_id=False)} "
                    f"reduction_axes={reduction_axes}"
                )
                metadata_str = format_tile_hardware_metadata(ann.hardware_metadata)
                if metadata_str:
                    line += f" {metadata_str}"
        print(line)


def print_graph_tiling(
    G: nuGraph,
    tile_dims: tuple[Any, ...] | None = None,
    block_dims: tuple[Any, ...] = _DEFAULT_BLOCK_DIMS,
) -> None:
    annotations = tile_graph_variant(G, tile_dims=tile_dims, block_dims=block_dims)
    print_graph(G, tile_annotations=annotations)
