"""Public API adapter for one egglog ``EGraph``."""

from __future__ import annotations

import dataclasses
import importlib.metadata
from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import Any

from egglog import EGraph, eq, expr_parts, union

_REQUIRED_DECL_FIELDS = frozenset({"let_bindings", "e_classes"})
_ADAPTER_IDS = count(1)


class EGraphAdapterError(RuntimeError):
    """Raised for adapter misuse or unsupported egglog schemas."""


def _installed_egglog_version() -> str:
    try:
        return importlib.metadata.version("egglog")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def _schema_error(detail: str) -> EGraphAdapterError:
    return EGraphAdapterError(
        f"unsupported egglog frozen-snapshot schema ({detail}); "
        f"installed egglog version is {_installed_egglog_version()}"
    )


def _import_decl_types() -> dict[str, Any]:
    try:
        from egglog.declarations import (
            CallDecl,
            EGraphDecl,
            LetRefDecl,
            LitDecl,
            TypedExprDecl,
            ValueDecl,
        )
    except ImportError as exc:  # pragma: no cover
        raise _schema_error(f"declaration types missing: {exc}") from exc
    field_names = {f.name for f in dataclasses.fields(EGraphDecl)}
    missing = _REQUIRED_DECL_FIELDS - field_names
    if missing:
        raise _schema_error(f"EGraphDecl lacks fields {sorted(missing)}")
    return {
        "CallDecl": CallDecl,
        "LetRefDecl": LetRefDecl,
        "LitDecl": LitDecl,
        "TypedExprDecl": TypedExprDecl,
        "ValueDecl": ValueDecl,
    }


_DECL = _import_decl_types()


@dataclass(frozen=True)
class EClassRef:
    """An e-class reference that is valid for one snapshot."""

    snapshot_id: int
    value: Any
    sort: str
    adapter_id: int = 0

    def __repr__(self) -> str:
        return f"EClassRef({self.sort}@{self.snapshot_id}:{self.value})"


@dataclass(frozen=True)
class LitArg:
    """A literal argument of an e-node row (i64, f64, Bool, String, Unit)."""

    value: int | float | str | bool | None


@dataclass(frozen=True)
class ENodeRef:
    """One e-node row: constructor identity, literal fields, child classes."""

    callable: Any
    egg_fn: str | None
    sort: str
    args: tuple[LitArg | EClassRef, ...]

    def child_classes(self) -> tuple[EClassRef, ...]:
        return tuple(a for a in self.args if isinstance(a, EClassRef))

    def _sort_key(self) -> tuple[str, ...]:
        return (
            self.egg_fn or repr(self.callable),
            *[repr(a) for a in self.args],
        )


class Snapshot:
    """An immutable decoded view of one frozen e-graph."""

    def __init__(
        self,
        snapshot_id: int,
        classes: dict[EClassRef, tuple[ENodeRef, ...]],
        let_bindings: dict[str, EClassRef],
        *,
        adapter_id: int | None = None,
    ) -> None:
        self.snapshot_id = snapshot_id
        child_refs = tuple(
            arg
            for rows in classes.values()
            for row in rows
            for arg in row.args
            if isinstance(arg, EClassRef)
        )
        refs = (*classes, *let_bindings.values(), *child_refs)
        ref_adapter_ids = {ref.adapter_id for ref in refs}
        if adapter_id is None:
            if len(ref_adapter_ids) > 1:
                raise EGraphAdapterError(
                    "Snapshot cannot contain EClassRefs from multiple adapters"
                )
            adapter_id = next(iter(ref_adapter_ids), 0)
        if any(ref_adapter_id != adapter_id for ref_adapter_id in ref_adapter_ids):
            raise EGraphAdapterError(
                "Snapshot adapter identity does not match its EClassRefs"
            )
        if any(ref.snapshot_id != snapshot_id for ref in refs):
            raise EGraphAdapterError("Snapshot identity does not match its EClassRefs")
        self.adapter_id = adapter_id
        self.classes = classes
        self.let_bindings = let_bindings
        self._row_index: dict[
            tuple[Any, tuple[LitArg | EClassRef, ...]], EClassRef
        ] = {}
        for cref, rows in classes.items():
            for row in rows:
                self._row_index[(row.callable, row.args)] = cref

    def members(self, ref: EClassRef) -> tuple[ENodeRef, ...]:
        if ref.adapter_id != self.adapter_id or ref.snapshot_id != self.snapshot_id:
            raise EGraphAdapterError(
                "EClassRef from a different snapshot passed to Snapshot.members"
            )
        return self.classes.get(ref, ())

    def find_row(
        self, callable_ref: Any, args: tuple[LitArg | EClassRef, ...]
    ) -> EClassRef | None:
        if any(
            isinstance(arg, EClassRef)
            and (
                arg.adapter_id != self.adapter_id or arg.snapshot_id != self.snapshot_id
            )
            for arg in args
        ):
            raise EGraphAdapterError(
                "EClassRef from a different snapshot passed to Snapshot.find_row"
            )
        return self._row_index.get((callable_ref, args))


@dataclass(frozen=True)
class InternResult:
    handle: Any
    name: str


@dataclass(frozen=True)
class Witness:
    """One deterministic finite term represented by an inhabited e-class."""

    enode: ENodeRef
    children: tuple[Witness | LitArg, ...]

    def height(self) -> int:
        child_heights = [c.height() for c in self.children if isinstance(c, Witness)]
        return 1 + (max(child_heights) if child_heights else 0)


class EGraphAdapter:
    """Owns one egglog ``EGraph`` plus the retained expression handles."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.adapter_id = next(_ADAPTER_IDS)
        self.egraph = EGraph()
        self._let_counter = 0
        self._snapshot_counter = 0
        self.handles: dict[str, Any] = {}

    # -- snapshots ---------------------------------------------------------

    def freeze_snapshot(self) -> Snapshot:
        """Decode the current e-graph through public frozen declarations."""
        frozen = self.egraph.freeze()
        try:
            decl = frozen.decl
            decls = frozen.__egg_decls__
            e_classes = decl.e_classes
            let_bindings = decl.let_bindings
        except AttributeError as exc:
            raise _schema_error(str(exc)) from exc

        self._snapshot_counter += 1
        sid = self._snapshot_counter
        classes: dict[EClassRef, tuple[ENodeRef, ...]] = {}
        for value, (type_ref, calls) in e_classes.items():
            cref = EClassRef(sid, value, _sort_name(type_ref), self.adapter_id)
            rows = []
            for call in calls:
                rows.append(
                    _decode_call(
                        self.adapter_id,
                        sid,
                        call,
                        decls,
                        _sort_name(type_ref),
                    )
                )
            rows.sort(key=lambda r: r._sort_key())
            classes[cref] = tuple(rows)

        bindings: dict[str, EClassRef] = {}
        for name, typed in let_bindings.items():
            expr = typed.expr
            if not isinstance(expr, _DECL["ValueDecl"]):
                continue
            bindings[name.removeprefix("$")] = EClassRef(
                sid, expr.value, _sort_name(typed.tp), self.adapter_id
            )
        return Snapshot(sid, classes, bindings, adapter_id=self.adapter_id)

    def _require_owned_snapshot(self, snapshot: Snapshot) -> None:
        if snapshot.adapter_id != self.adapter_id:
            raise EGraphAdapterError(
                "Snapshot from a different adapter passed to EGraphAdapter"
            )

    def _require_owned_expr(self, expr: Any) -> None:
        def visit(decl: Any) -> None:
            if isinstance(decl, _DECL["LetRefDecl"]):
                if decl.name not in self.handles:
                    raise EGraphAdapterError(
                        "expression contains a handle from a different adapter"
                    )
                return
            if isinstance(decl, _DECL["CallDecl"]):
                for arg in decl.args:
                    visit(arg.expr)

        visit(expr_parts(expr).expr)

    # -- interning ---------------------------------------------------------

    def intern_expr(self, expr: Any, provenance: str) -> InternResult:
        """Intern one constructor application and return its handle.

        Whether the row already existed is not reported: answering that needs a
        full snapshot decode, so callers who care ask `enode_exists` directly."""
        self._require_owned_expr(expr)
        typed = expr_parts(expr)
        if not isinstance(typed.expr, _DECL["CallDecl"]):
            raise EGraphAdapterError(
                "intern_expr requires a constructor application, got "
                f"{type(typed.expr).__name__}"
            )
        name = f"{self.label}_a{self.adapter_id}_{provenance}_{self._let_counter}"
        self._let_counter += 1
        handle = self.egraph.let(name, expr)
        self.handles[name] = handle
        return InternResult(handle=handle, name=name)

    def enode_exists(self, snapshot: Snapshot, expr: Any) -> bool:
        """Return true if the exact root row of ``expr`` exists."""
        self._require_owned_snapshot(snapshot)
        self._require_owned_expr(expr)
        typed = expr_parts(expr)
        if not isinstance(typed.expr, _DECL["CallDecl"]):
            raise EGraphAdapterError("enode_exists requires a constructor application")
        return self._resolve_call(snapshot, typed.expr, {}) is not None

    def _resolve_call(
        self,
        snapshot: Snapshot,
        call: Any,
        memo: dict[int, EClassRef | None],
    ) -> EClassRef | None:
        key = id(call)
        if key in memo:
            return memo[key]
        args: list[LitArg | EClassRef] = []
        resolved: EClassRef | None = None
        for arg in call.args:
            arg_expr = arg.expr
            if isinstance(arg_expr, _DECL["LitDecl"]):
                args.append(LitArg(arg_expr.value))
            elif isinstance(arg_expr, _DECL["LetRefDecl"]):
                bound = snapshot.let_bindings.get(arg_expr.name)
                if bound is None:
                    memo[key] = None
                    return None
                args.append(bound)
            elif isinstance(arg_expr, _DECL["CallDecl"]):
                child = self._resolve_call(snapshot, arg_expr, memo)
                if child is None:
                    memo[key] = None
                    return None
                args.append(child)
            else:
                raise EGraphAdapterError(
                    f"unsupported argument declaration {type(arg_expr).__name__} "
                    "in enode_exists"
                )
        resolved = snapshot.find_row(call.callable, tuple(args))
        memo[key] = resolved
        return resolved

    # -- handles and unions --------------------------------------------------

    def resolve_handle(self, snapshot: Snapshot, handle: Any) -> EClassRef:
        """Map a retained let handle to its e-class in one snapshot."""
        self._require_owned_snapshot(snapshot)
        typed = expr_parts(handle)
        expr = typed.expr
        if not isinstance(expr, _DECL["LetRefDecl"]):
            raise EGraphAdapterError(
                f"resolve_handle requires a let handle, got {type(expr).__name__}"
            )
        if expr.name not in self.handles:
            raise EGraphAdapterError(
                "handle from a different adapter passed to resolve_handle"
            )
        bound = snapshot.let_bindings.get(expr.name)
        if bound is None:
            raise EGraphAdapterError(
                f"let binding '{expr.name}' is absent from the snapshot"
            )
        return bound

    def union_if_distinct(self, lhs: Any, rhs: Any) -> bool:
        """Union two handles unless already equal; True if a union was added."""
        self._require_owned_expr(lhs)
        self._require_owned_expr(rhs)
        if self.egraph.check_bool(eq(lhs).to(rhs)):
            return False
        self.egraph.register(union(lhs).with_(rhs))
        return True


def _sort_name(type_ref: Any) -> str:
    ident = getattr(type_ref, "ident", None)
    name = getattr(ident, "name", None)
    if name is None:
        raise _schema_error(f"type reference without ident name: {type_ref!r}")
    return str(name)


def _decode_call(
    adapter_id: int,
    sid: int,
    call: Any,
    decls: Any,
    sort: str,
) -> ENodeRef:
    args: list[LitArg | EClassRef] = []
    for arg in call.args:
        arg_expr = arg.expr
        if isinstance(arg_expr, _DECL["LitDecl"]):
            args.append(LitArg(arg_expr.value))
        elif isinstance(arg_expr, _DECL["ValueDecl"]):
            args.append(
                EClassRef(
                    sid,
                    arg_expr.value,
                    _sort_name(arg.tp),
                    adapter_id,
                )
            )
        else:
            raise _schema_error(
                f"e-class row argument decoded to {type(arg_expr).__name__}, "
                "expected LitDecl or ValueDecl"
            )
    egg_fn: str | None
    try:
        egg_fn = decls.get_callable_decl(call.callable).egg_name
    except Exception:
        egg_fn = None
    return ENodeRef(callable=call.callable, egg_fn=egg_fn, sort=sort, args=tuple(args))


def reachable_classes(snapshot: Snapshot, roots: list[EClassRef]) -> list[EClassRef]:
    """Return reachable classes in deterministic breadth-first order."""
    seen: dict[EClassRef, None] = {}
    queue: deque[EClassRef] = deque()
    for root in roots:
        if root not in seen:
            seen[root] = None
            queue.append(root)
    while queue:
        current = queue.popleft()
        for row in snapshot.members(current):
            for child in row.child_classes():
                if child not in seen:
                    seen[child] = None
                    queue.append(child)
    return list(seen)


def compute_minimum_heights(snapshot: Snapshot) -> dict[EClassRef, int]:
    """Compute the minimum finite height of each inhabited class."""
    heights: dict[EClassRef, int] = {}
    changed = True
    while changed:
        changed = False
        for cref, rows in snapshot.classes.items():
            for row in rows:
                children = row.child_classes()
                if any(c not in heights for c in children):
                    continue
                height = 1 + max((heights[c] for c in children), default=0)
                if cref not in heights or height < heights[cref]:
                    heights[cref] = height
                    changed = True
    return heights


def minimum_height_row(
    snapshot: Snapshot,
    ref: EClassRef,
    heights: dict[EClassRef, int],
) -> ENodeRef:
    """Select a deterministic member with the minimum finite height."""
    best: ENodeRef | None = None
    best_height: int | None = None
    for row in snapshot.members(ref):
        children = row.child_classes()
        if any(c not in heights for c in children):
            continue
        height = 1 + max((heights[c] for c in children), default=0)
        if best is None or (height, row._sort_key()) < (best_height, best._sort_key()):
            best = row
            best_height = height
    if best is None:
        raise EGraphAdapterError(f"e-class {ref!r} is uninhabited")
    return best


def minimum_height_witness(
    snapshot: Snapshot,
    ref: EClassRef,
    heights: dict[EClassRef, int] | None = None,
) -> Witness:
    """Build a deterministic finite witness for an inhabited class."""
    if heights is None:
        heights = compute_minimum_heights(snapshot)
    if ref not in heights:
        raise EGraphAdapterError(f"e-class {ref!r} is uninhabited")

    def build(cref: EClassRef) -> Witness:
        best = minimum_height_row(snapshot, cref, heights)
        children_out: list[Witness | LitArg] = []
        for arg in best.args:
            if isinstance(arg, EClassRef):
                children_out.append(build(arg))
            else:
                children_out.append(arg)
        return Witness(enode=best, children=tuple(children_out))

    return build(ref)
