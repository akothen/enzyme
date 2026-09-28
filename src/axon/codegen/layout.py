"""Layout inference for the matmul-family emitter.

A Layout states which schedule coordinate of a value sits on SBUF partitions and which sits on the free axis. Each dimension tracks three separate identities: its occurrence, its shape extent, and its positional axis role. Matmul outputs always receive fresh occurrences and inherit operand roles, splitting an output role when a square score would otherwise use one role twice. Consumers validate extent equality without merging producer occurrences. Positional elementwise operations may align role classes, but never part and free roles. Inference mirrors the ISA semantics Z3 proves against (isa_semantics), so the emitter consumes the same facts the prover verified instead of re-deriving structure from patterns. A consumer needing a layout its producer does not supply must find an nc_transpose already in the graph; otherwise inference refuses with LayoutError (an UnsupportedEmission), dropping the variant as EmitErr."""

from __future__ import annotations

from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING

from axon.codegen.ops import UnsupportedEmission

if TYPE_CHECKING:
    from axon.ir import Node


class LayoutError(UnsupportedEmission):
    """A graph edge needs a layout its producer does not supply."""


def _extent_key(dim: object) -> tuple[str, str]:
    """A shape extent's identity: normalized integers, else the symbol name.

    Extents reach codegen concrete (`annotate_shapes_concrete`), so equal keys
    are exactly equal extents and equality needs no solver."""
    if isinstance(dim, int) and not isinstance(dim, bool):
        return ("int", str(dim))
    if isinstance(dim, str):
        return ("symbol", dim)
    return (type(dim).__qualname__, str(dim))


def _normalize_rank2_reduce_axis(axis_attr: object) -> int:
    """Normalize one rank-2 reduction axis using ISA semantics."""
    axis = axis_attr
    if isinstance(axis, (list, tuple)):
        if len(axis) != 1:
            raise ValueError(f"expected one reduction axis, got {axis!r}")
        axis = axis[0]
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise ValueError(f"reduction axis must be an integer, got {axis!r}")
    axis_value = axis
    return axis_value + 2 if axis_value < 0 else axis_value


class _ExtentVar:
    """A shape extent with union-find identity."""

    __slots__ = ("_is_unit", "_label", "_parent")

    def __init__(
        self,
        label: tuple[str, str] | None = None,
        *,
        is_unit: bool = False,
    ) -> None:
        self._parent: _ExtentVar = self
        self._label = label
        self._is_unit = is_unit

    def find(self) -> _ExtentVar:
        root = self
        while root._parent is not root:
            root = root._parent
        node = self
        while node._parent is not node:
            node._parent, node = root, node._parent
        return root

    def unite(self, other: _ExtentVar) -> bool:
        left = self.find()
        right = other.find()
        if left is right:
            return True
        if (
            left._label is not None
            and right._label is not None
            and left._label != right._label
        ):
            return False
        left._parent = right
        if right._label is None:
            right._label = left._label
        right._is_unit = left._is_unit or right._is_unit
        return True

    def is_unit(self) -> bool:
        return self.find()._is_unit


class _RoleVar:
    """A positional schedule-axis role with union-find identity."""

    __slots__ = ("_parent",)

    def __init__(self) -> None:
        self._parent: _RoleVar = self

    def find(self) -> _RoleVar:
        root = self
        while root._parent is not root:
            root = root._parent
        node = self
        while node._parent is not node:
            node._parent, node = root, node._parent
        return root

    def unite(self, other: _RoleVar) -> None:
        self.find()._parent = other.find()


class DimVar:
    """A coordinate occurrence with separate extent and axis-role classes."""

    __slots__ = ("_extent", "_name", "_parent", "_role")

    def __init__(
        self,
        name: str,
        *,
        extent: _ExtentVar | None = None,
        role: _RoleVar | None = None,
    ) -> None:
        self._name = name
        self._parent: DimVar = self
        self._extent = extent or _ExtentVar()
        self._role = role or _RoleVar()

    def find(self) -> DimVar:
        root = self
        while root._parent is not root:
            root = root._parent
        node = self
        while node._parent is not node:
            node._parent, node = root, node._parent
        return root

    def unite(self, other: DimVar) -> None:
        if not self._extent.unite(other._extent):
            raise ValueError("cannot unite schedule coordinates with unequal extents")
        self._role.unite(other._role)
        self.find()._parent = other.find()

    def fork(self, name: str, *, fresh_role: bool = False) -> DimVar:
        """Create a fresh occurrence, preserving extent and usually axis role."""
        role = None if fresh_role else self._role.find()
        return DimVar(name, extent=self._extent.find(), role=role)

    def same_extent(self, other: DimVar) -> bool:
        return self._extent.find() is other._extent.find()

    def has_unit_extent(self) -> bool:
        return self._extent.is_unit()

    def align_extent(self, other: DimVar) -> bool:
        """Record compatible extent equality without merging coordinates."""
        return self._extent.unite(other._extent)

    def same_role(self, other: DimVar) -> bool:
        return self._role.find() is other._role.find()

    def extent_label(self) -> tuple[str, str] | None:
        return self._extent.find()._label

    def align_role(self, other: DimVar) -> None:
        """Align positional roles without merging coordinate occurrences."""
        self._role.unite(other._role)

    @property
    def name(self) -> str:
        return self.find()._name

    def __repr__(self) -> str:  # debug aid only
        return f"DimVar({self._name}->{self.name})"


@dataclass(frozen=True)
class Layout:
    """part: dim on SBUF partitions; free: dim on the free axis, or None for
    a per-partition scalar (a (P, 1) value such as a row-reduce result)."""

    part: DimVar
    free: DimVar | None = None

    def same(self, other: Layout) -> bool:
        if (self.free is None) != (other.free is None):
            return False
        if not self.part.same_role(other.part):
            return False
        return self.free is None or self.free.same_role(other.free)

    def swapped(self) -> Layout:
        if self.free is None:
            raise LayoutError(
                "cannot transpose a per-partition scalar layout "
                f"(part={self.part.name}, free=None)"
            )
        return Layout(part=self.free, free=self.part)


def topo_order(compute_nodes: list[Node], id_to_node: dict[str, Node]) -> list[Node]:
    """Topological order of the compute nodes, ties broken by list position.

    Insertion order into the sorter is the caller's list order, which makes the
    result deterministic and invariant under any renaming of node ids."""
    compute_ids = {n.id for n in compute_nodes}
    sorter = TopologicalSorter(
        {n.id: [i for i in (n.inputs or []) if i in compute_ids] for n in compute_nodes}
    )
    try:
        order = list(sorter.static_order())
    except CycleError as exc:
        raise LayoutError(f"cycle in hw graph: {exc.args[1]}") from exc
    return [id_to_node[nid] for nid in order]


_ELEMENTWISE = {"tensor_tensor", "tensor_scalar", "scalar_tensor_tensor"}
# Unary elementwise ops: output layout is identical to the single input's layout
# (full-tile → full-tile with the same dims; per-partition scalar → scalar).
# activation is also unary and must handle scalar inputs (e.g. rsqrt after a
# tensor_reduce), so it lives here rather than in _ELEMENTWISE.
_UNARY_ELEMENTWISE = {"exponential", "reciprocal", "activation"}


def _is_unary_constant_tensor_scalar(node: Node) -> bool:
    """Return whether tensor_scalar has one tensor input and one constant op."""
    attrs = node.attrs
    return (
        node.op == "tensor_scalar"
        and len(node.inputs or []) == 1
        and attrs.get("op0") is not None
        and attrs.get("operand0_const") is not None
        and "operand0_input_index" not in attrs
        and attrs.get("op1") is None
        and attrs.get("operand1_const") is None
        and "operand1_input_index" not in attrs
        and attrs.get("reverse1", False) is False
    )


def infer_layouts(
    compute_nodes: list[Node],
    id_to_node: dict[str, Node],
    input_node_ids: set[str],
) -> dict[str, Layout]:
    """Assign every node a Layout and refuse incompatible extent alignment.

    Inputs get fresh per-axis DimVars (rows on partitions, the HBM
    convention every existing body uses). Rules mirror isa_semantics:
      nc_transpose: swap.
      nc_matmul(stat, mov): validate equal contraction extents and create
        occurrence-specific output coordinates.
      elementwise: full-tile operand extents align axis-wise; per-partition
        scalars (free=None) align to the partition extent;
        out inherits the first full-tile operand's layout.
      tensor_reduce / activation_reduce: out = Layout(in.part, None).
      broadcast / tensor_copy: copy of input Layout (not an alias).
      exponential / reciprocal: unary elementwise, output = Layout(in.part, in.free).
    """
    L: dict[str, Layout] = {}
    extent_vars: dict[tuple[str, str], _ExtentVar] = {}
    symbolic_bindings: dict[tuple[str, str], tuple[str, str]] = {}

    def extent_keys_equal(
        left: tuple[str, str],
        right: tuple[str, str],
    ) -> bool:
        """Equal keys, or keys whose symbols bind to the same concrete extent."""
        left_keys = {left, symbolic_bindings.get(left, left)}
        right_keys = {right, symbolic_bindings.get(right, right)}
        return not left_keys.isdisjoint(right_keys)

    input_shapes: dict[
        str,
        tuple[tuple[object, ...] | None, tuple[object, ...] | None],
    ] = {}
    for iid in sorted(input_node_ids):
        node = id_to_node[iid]
        symbolic_shape = node.attrs.get("sym_shape")
        attrs_shape = node.attrs.get("shape")
        concrete_shape = attrs_shape if attrs_shape is not None else node.shape
        symbolic_shape = tuple(symbolic_shape) if symbolic_shape is not None else None
        concrete_shape = tuple(concrete_shape) if concrete_shape is not None else None
        if symbolic_shape is not None and concrete_shape is not None:
            # Bind each symbolic extent to its concrete one, so two distinct
            # symbol names with equal concrete extents share one extent var.
            for symbolic_dim, concrete_dim in zip(
                symbolic_shape, concrete_shape, strict=False
            ):
                symbolic_bindings[_extent_key(symbolic_dim)] = _extent_key(concrete_dim)
        input_shapes[iid] = (symbolic_shape, concrete_shape)

    def input_dim(iid: str, axis: int, suffix: str) -> DimVar:
        symbolic_shape, concrete_shape = input_shapes[iid]
        shape = symbolic_shape if symbolic_shape is not None else concrete_shape
        if shape is None or axis >= len(shape):
            return DimVar(f"{iid}.{suffix}")
        key = _extent_key(shape[axis])
        extent = next(
            (
                known_extent
                for known_key, known_extent in extent_vars.items()
                if extent_keys_equal(key, known_key)
            ),
            None,
        )
        if extent is None:
            extent = _ExtentVar(
                key,
                is_unit=extent_keys_equal(key, _extent_key(1)),
            )
        extent_vars[key] = extent
        return DimVar(f"{iid}.{suffix}", extent=extent)

    def validate_partition_scalar_output_shape(node: Node, data: Layout) -> None:
        declared_shapes = (
            ("attrs shape", node.attrs.get("shape")),
            ("node shape", node.shape),
        )
        for shape_name, shape in declared_shapes:
            if shape is None:
                continue
            try:
                declared = tuple(shape)
            except TypeError as exc:
                raise LayoutError(
                    f"tensor_scalar {node.id}: {shape_name} must be a rank-2 "
                    f"shape sequence, got {shape!r}"
                ) from exc
            if len(declared) != 2:
                raise LayoutError(
                    f"tensor_scalar {node.id}: {shape_name} must have rank 2, "
                    f"got {len(declared)}"
                )
            part_extent = data.part.extent_label()
            if part_extent is not None and not extent_keys_equal(
                _extent_key(declared[0]),
                part_extent,
            ):
                raise LayoutError(
                    f"tensor_scalar {node.id}: {shape_name}[0]={declared[0]} "
                    f"disagrees with scalar partition extent {part_extent[1]}"
                )
            if not extent_keys_equal(
                _extent_key(declared[1]),
                _extent_key(1),
            ):
                raise LayoutError(
                    f"tensor_scalar {node.id}: {shape_name}[1]={declared[1]} "
                    "must be the singleton free dimension of a per-partition scalar"
                )

    def align_extent(
        left: DimVar,
        right: DimVar,
        *,
        operation: str,
    ) -> None:
        if not left.align_extent(right):
            raise LayoutError(
                f"{operation}: operand extents disagree "
                f"({left.name} versus {right.name})"
            )

    def align_full_layout(
        base: Layout,
        other: Layout,
        *,
        operation: str,
        broadcast_other: bool = False,
    ) -> None:
        assert base.free is not None
        assert other.free is not None
        if (
            base.part.same_role(other.free)
            or base.free.same_role(other.part)
            or base.part.same_role(base.free)
            or other.part.same_role(other.free)
        ):
            raise LayoutError(
                f"{operation}: operand layouts conflict "
                "(alignment collapses part and free roles); "
                "an nc_transpose is missing on one input edge"
            )
        if not (broadcast_other and other.part.has_unit_extent()):
            align_extent(base.part, other.part, operation=operation)
        if not (broadcast_other and other.free.has_unit_extent()):
            align_extent(base.free, other.free, operation=operation)
        base.part.align_role(other.part)
        base.free.align_role(other.free)

    for iid in sorted(input_node_ids):
        L[iid] = Layout(
            part=input_dim(iid, 0, "p"),
            free=input_dim(iid, 1, "f"),
        )

    for node in topo_order(compute_nodes, id_to_node):
        ins = [L[i] for i in (node.inputs or [])]
        if node.op == "nc_transpose":
            L[node.id] = ins[0].swapped()
        elif node.op == "nc_matmul":
            stat, mov = ins[0], ins[1]
            if stat.free is None or mov.free is None:
                raise LayoutError(
                    f"nc_matmul {node.id}: operand is a per-partition scalar"
                )
            align_extent(
                stat.part,
                mov.part,
                operation=f"nc_matmul {node.id}",
            )
            duplicate_output_role = stat.free.same_role(mov.free)
            L[node.id] = Layout(
                part=stat.free.fork(f"{node.id}.p"),
                free=mov.free.fork(
                    f"{node.id}.f",
                    fresh_role=duplicate_output_role,
                ),
            )
        elif node.op == "tensor_reduce":
            # Validate that this is a pure free-axis (axis=1) reduce.  A
            # partition-axis reduce (axis=0) would produce a different output
            # shape and cannot be faithfully lowered with the free=None scalar
            # layout the emitter assumes.  Refuse rather than silently misemit.
            axis_attr = node.attrs.get("axis", 1)
            try:
                axis_val = _normalize_rank2_reduce_axis(axis_attr)
            except (TypeError, ValueError) as exc:
                raise LayoutError(
                    f"tensor_reduce {node.id}: invalid reduction axis "
                    f"{axis_attr!r}: {exc}"
                ) from exc
            if axis_val != 1:
                raise LayoutError(
                    f"tensor_reduce {node.id}: only free-axis (axis=1) reductions "
                    f"are supported; got axis={axis_val}"
                )
            L[node.id] = Layout(part=ins[0].part, free=None)
        elif node.op == "activation_reduce":
            # activation_reduce always reduces the free axis by design (no axis
            # attribute).  Layout is a per-partition scalar, same as axis=1
            # tensor_reduce.
            data = ins[0]
            if data.free is None:
                raise LayoutError(
                    f"activation_reduce {node.id}: data is a per-partition scalar"
                )
            for auxiliary in ins[1:]:
                if auxiliary.free is None:
                    if auxiliary.part.same_role(data.free):
                        raise LayoutError(
                            f"activation_reduce {node.id}: auxiliary scalar "
                            "sits on the free-axis role"
                        )
                    align_extent(
                        auxiliary.part,
                        data.part,
                        operation=f"activation_reduce {node.id}",
                    )
                    auxiliary.part.align_role(data.part)
                    continue
                align_full_layout(
                    data,
                    auxiliary,
                    operation=f"activation_reduce {node.id}",
                    broadcast_other=True,
                )
            L[node.id] = Layout(part=ins[0].part, free=None)
        elif node.op in ("broadcast", "tensor_copy"):
            # Layout copy (not alias) of the input.
            L[node.id] = Layout(ins[0].part, ins[0].free)
        elif node.op in _UNARY_ELEMENTWISE:
            # Unary elementwise: output inherits the input's layout exactly.
            # Works for both full-tile (free is a DimVar) and per-partition
            # scalars (free is None) — no unification needed.
            L[node.id] = Layout(ins[0].part, ins[0].free)
        elif _is_unary_constant_tensor_scalar(node) and ins[0].free is None:
            # A constant ScalarE operation preserves a reduce scalar's (P, 1)
            # shape. Tensor operands still use the guarded combiner path below.
            validate_partition_scalar_output_shape(node, ins[0])
            L[node.id] = Layout(ins[0].part, free=None)
        elif node.op in _ELEMENTWISE:
            full = [lay for lay in ins if lay.free is not None]
            scalars = [lay for lay in ins if lay.free is None]
            if not full:
                raise LayoutError(f"{node.op} {node.id}: no full-tile operand")
            base = full[0]
            for other in full[1:]:
                align_full_layout(
                    base,
                    other,
                    operation=f"{node.op} {node.id}",
                )
            for s in scalars:
                if s.part.same_role(base.free):
                    raise LayoutError(
                        f"{node.op} {node.id}: per-partition scalar sits on "
                        "the free-axis role; an nc_transpose is missing"
                    )
                if not s.part.same_extent(base.part):
                    raise LayoutError(
                        f"{node.op} {node.id}: per-partition scalar sits on "
                        "the wrong axis; an nc_transpose is missing"
                    )
                s.part.align_role(base.part)
            L[node.id] = base
        else:
            raise LayoutError(
                f"layout inference: unhandled op {node.op!r} (id={node.id})"
            )
    return L


@dataclass(frozen=True)
class DimInfo:
    """Result of dim-dependence analysis.

    ``dims`` maps every node id to the frozenset of DimVar *representatives*
    (call ``.find()`` on each before looking up in ``canon``; they are stored
    as final reps after all unions, so direct indexing ``info.canon[d]`` is
    safe for ``d`` in ``info.dims[nid]``).

    ``canon`` keys are representatives (.find() results at freeze time).
    Lookups on reps obtained *outside* DimInfo should use
    ``info.canon[d.find()]`` to be safe.
    """

    layouts: dict[str, Layout]
    dims: dict[str, frozenset[DimVar]]
    canon: dict[DimVar, str]
    contractions: list[tuple[str, DimVar]]


def analyze_dims(
    compute_nodes: list[Node],
    id_to_node: dict[str, Node],
    input_node_ids: set[str],
    output_id: str,
) -> DimInfo:
    """Compute dims each node varies over and assign canonical names m/n/k/p.

    Every node varies over its output layout coordinates. A reduction has only
    its partition coordinate, and every full-tile output has both coordinates.
    """
    L = infer_layouts(compute_nodes, id_to_node, input_node_ids)
    dims: dict[str, frozenset[DimVar]] = {}

    for iid in input_node_ids:
        part_rep = L[iid].part.find()
        free = L[iid].free
        free_rep = free.find() if free is not None else None
        dims[iid] = frozenset(d for d in (part_rep, free_rep) if d is not None)

    contractions: list[tuple[str, DimVar]] = []
    order = topo_order(compute_nodes, id_to_node)
    for node in order:
        ins = node.inputs or []
        if node.op == "nc_matmul":
            stat = L[ins[0]]
            contractions.append((node.id, stat.part.find()))
        layout = L[node.id]
        if layout.free is None:
            dims[node.id] = frozenset({layout.part.find()})
        else:
            dims[node.id] = frozenset({layout.part.find(), layout.free.find()})

    # --- Assign canonical names ---
    canon: dict[DimVar, str] = {}

    out_l = L[output_id]
    # Output partition dim → "m"
    canon[out_l.part.find()] = "m"

    # The coordinate the output matmul forks into its partition dim, i.e. the
    # occurrence "m" above already names. nc_matmul builds its output partition
    # as stat.free.fork(...), so this IS the m axis, one union-find occurrence
    # earlier, not merely something that shares m's role or extent.
    out_node = id_to_node.get(output_id)
    out_row_source = None
    if out_node is not None and out_node.op == "nc_matmul":
        stat_free = L[out_node.inputs[0]].free
        out_row_source = stat_free.find() if stat_free is not None else None

    # First mm in topo order: its free dim → "n", unless that coordinate is the
    # one the output matmul forks into "m". A swapped-operand orientation
    # (nc_matmul(w1, xᵀ), weight stationary) leaves the output rows on the first
    # matmul's FREE axis and carries that same occurrence to the output matmul's
    # stationary operand, so naming it "n" parks the m axis under the n tile role
    # and pushes the real hidden dim out to a fresh "k2", outside the
    # {m, n, k, p} namespace the signature provides. Reusing "m" names one axis
    # once. Occurrence identity is the license: a shared role or equal extent is
    # not enough (attention's projections share m's role across genuinely
    # distinct coordinates, and reusing "m" there would alias two axes onto one
    # tile role); only the same union-find coordinate qualifies.
    first_mm = next((n for n in order if n.op == "nc_matmul"), None)
    if first_mm is not None:
        first_free = L[first_mm.id].free
        if first_free is not None:
            rep = first_free.find()
            if rep not in canon:
                canon[rep] = "m" if rep is out_row_source else "n"

    # Final output free dim
    n_mms = sum(1 for n in order if n.op == "nc_matmul")
    if out_l.free is not None:
        final_free = out_l.free.find()
        if final_free not in canon:
            canon[final_free] = "p" if n_mms > 1 else "n"

    # In a chained multi-matmul graph the OUTPUT matmul's contraction is the
    # hidden dim it shares with the earlier matmuls' outputs: canonically "n",
    # never a fresh "k2". Without this, an all-transposed orientation (whose
    # first matmul's free dim is m, so the "first mm free -> n" rule above does
    # not fire) names the hidden dim "k2" and falls out of the m/n/k/p tile
    # namespace the emitted signature provides. Claiming it as "n" here also
    # keeps the base TILE_N / BLOCK_N lines bound: the preamble emits tile
    # scaffolding per present dim, so a dim outside the namespace both
    # references an absent TILES_IN_BLOCK_* param and leaves "n" unbound.
    #
    # Claim "n" only when nothing else holds it yet. Where the "first mm free ->
    # n" rule already fired (an attention-shaped graph, whose first projection
    # is not transposed) that coordinate owns "n", and reusing the name would
    # alias two distinct schedule axes onto one tile role. Those graphs keep the
    # fresh contraction name and are refused downstream by the attention binder
    # rather than silently mis-emitted.
    if (
        n_mms > 1
        and not any(name == "n" for name in canon.values())
        and id_to_node.get(output_id) is not None
        and id_to_node[output_id].op == "nc_matmul"
    ):
        for mm_id, c in contractions:
            if mm_id != output_id:
                continue
            rep = c.find()
            if rep not in canon:
                canon[rep] = "n"
            break

    # A contraction role is shared only by that matmul's two operand
    # coordinates. Equal extent alone never aliases schedule roles.
    k_idx = 0
    for mm_id, c in contractions:
        rep = c.find()
        mm_node = id_to_node[mm_id]
        moving_part = L[mm_node.inputs[1]].part.find()
        contraction_name = canon.get(rep) or canon.get(moving_part)
        if contraction_name is None:
            contraction_name = "k" if k_idx == 0 else f"k{k_idx + 1}"
            k_idx += 1
        canon.setdefault(rep, contraction_name)
        canon.setdefault(moving_part, contraction_name)

    # Propagate each matmul result's assigned roles to its operand free axes.
    # Conflicts stop at the shared producer coordinate instead of merging two
    # occurrence-specific result coordinates.
    for node in reversed(order):
        layout = L[node.id]
        if node.op == "nc_matmul":
            stat, mov = (L[input_id] for input_id in node.inputs)
            if stat.free is None or mov.free is None:
                raise LayoutError(
                    f"nc_matmul {node.id}: operand is a per-partition scalar"
                )
            if layout.part.find() in canon:
                canon.setdefault(stat.free.find(), canon[layout.part.find()])
            if layout.free is not None and layout.free.find() in canon:
                canon.setdefault(mov.free.find(), canon[layout.free.find()])
            continue
        if node.op not in _ELEMENTWISE:
            continue
        full = [L[input_id] for input_id in node.inputs if L[input_id].free is not None]
        scalars = [L[input_id] for input_id in node.inputs if L[input_id].free is None]
        part_name = canon.get(layout.part.find())
        free_name = canon.get(layout.free.find()) if layout.free is not None else None
        for operand in full:
            if part_name is not None:
                canon.setdefault(operand.part.find(), part_name)
            operand_free = operand.free
            if free_name is not None and operand_free is not None:
                canon.setdefault(operand_free.find(), free_name)
        for scalar in scalars:
            if part_name is not None:
                canon.setdefault(scalar.part.find(), part_name)

    # Shared raw inputs need one shape-unpack role. Prefer the first matmul use
    # in topological order while keeping each matmul result role independent.
    input_axes = {
        axis.find()
        for input_id in input_node_ids
        for axis in (L[input_id].part, L[input_id].free)
        if axis is not None
    }
    input_candidates: dict[DimVar, set[str]] = {}
    contraction_names = {mm_id: canon[c.find()] for mm_id, c in contractions}
    for node in order:
        if node.op != "nc_matmul":
            continue
        layout = L[node.id]
        stat, mov = (L[input_id] for input_id in node.inputs)
        if stat.free is None or mov.free is None or layout.free is None:
            raise LayoutError(
                f"nc_matmul {node.id}: expected full-tile operand and output layouts"
            )
        desired = (
            (stat.free.find(), canon.get(layout.part.find())),
            (mov.free.find(), canon.get(layout.free.find())),
            (stat.part.find(), contraction_names[node.id]),
            (mov.part.find(), contraction_names[node.id]),
        )
        for axis, name in desired:
            if axis in input_axes and name is not None:
                input_candidates.setdefault(axis, set()).add(name)
    preferred_names = {"m": 0, "p": 1, "n": 2, "k": 3}
    input_names = {
        axis: min(
            names,
            key=lambda name: (preferred_names.get(name, 4), name),
        )
        for axis, names in input_candidates.items()
    }
    canon.update(input_names)

    # Any remaining dims (defensive fallback)
    extra = 4
    for ds in dims.values():
        for d in ds:
            rep = d.find()
            if rep not in canon:
                canon[rep] = f"d{extra}"
                extra += 1

    # Freeze: re-key canon by final representatives (path compression may have
    # changed reps since we inserted entries; also normalises the keys).
    canon = {k.find(): v for k, v in canon.items()}

    # Rebuild dims with final representatives so direct indexing into canon works.
    dims = {nid: frozenset(d.find() for d in ds) for nid, ds in dims.items()}

    return DimInfo(layouts=L, dims=dims, canon=canon, contractions=contractions)
