"""Immutable host context shared by all work over one semantic snapshot."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from axon.egraph.adapter import (
    EClassRef,
    EGraphAdapter,
    ENodeRef,
    Snapshot,
    compute_minimum_heights,
    minimum_height_row,
)
from axon.egraph.proof import Term, TermApp, TermRef

DecodeFn = Callable[[Snapshot, ENodeRef], Any]
AnalyzeFn = Callable[[Snapshot, DecodeFn], dict[EClassRef, Any]]


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    qualname = getattr(value, "__qualname__", None)
    if isinstance(qualname, str):
        return [
            "callable",
            getattr(value, "__module__", None),
            qualname,
        ]
    return repr(value)


def _frozen_value(value: Any) -> Any:
    normalized = _json_value(value)
    if isinstance(normalized, list):
        return tuple(_frozen_value(item) for item in normalized)
    if isinstance(normalized, dict):
        return tuple(
            (key, _frozen_value(item)) for key, item in sorted(normalized.items())
        )
    return normalized


@dataclass(frozen=True)
class SemanticContext:
    """Decoded graph facts and stable host identities for one snapshot."""

    adapter: EGraphAdapter
    snapshot: Snapshot
    decode: DecodeFn
    analyses: dict[EClassRef, Any]
    heights: dict[EClassRef, int]
    class_handles: dict[EClassRef, Any]
    class_identities: dict[EClassRef, str]
    term_owners: dict[TermApp, tuple[EClassRef, ...]]
    row_identities: dict[tuple[EClassRef, ENodeRef], str]
    _term_handle_cache: dict[Term, tuple[Any, int]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _witness_identity_cache: dict[EClassRef, Any] = field(
        default_factory=dict, repr=False, compare=False
    )
    _represented_owner_cache: dict[Term, EClassRef | None] = field(
        default_factory=dict, repr=False, compare=False
    )

    @classmethod
    def build(
        cls,
        adapter: EGraphAdapter,
        snapshot: Snapshot,
        decode: DecodeFn,
        analyze: AnalyzeFn,
    ) -> SemanticContext:
        analyses = analyze(snapshot, decode)
        heights = compute_minimum_heights(snapshot)

        names_by_class: dict[EClassRef, list[str]] = {}
        class_handles: dict[EClassRef, Any] = {}
        for name, handle in adapter.handles.items():
            ref = snapshot.let_bindings.get(name)
            if ref is None:
                continue
            names_by_class.setdefault(ref, []).append(name)
            class_handles.setdefault(ref, handle)

        class_identities: dict[EClassRef, str] = {}
        for ref in snapshot.classes:
            names = names_by_class.get(ref)
            identity = (
                ["handle", names[0]] if names else ["value", ref.sort, repr(ref.value)]
            )
            class_identities[ref] = json.dumps(identity, separators=(",", ":"))

        term_owners_mut: dict[TermApp, list[EClassRef]] = {}
        row_identities: dict[tuple[EClassRef, ENodeRef], str] = {}
        for owner in snapshot.classes:
            for row in snapshot.members(owner):
                try:
                    decoded = decode(snapshot, row)
                except Exception:
                    continue
                children = tuple(TermRef(child) for child in decoded.child_classes)
                term = TermApp.make(decoded.op, dict(decoded.attrs), children)
                term_owners_mut.setdefault(term, []).append(owner)
                row_identity = [
                    class_identities[owner],
                    decoded.op,
                    _json_value(dict(decoded.attrs)),
                    [class_identities[child] for child in decoded.child_classes],
                ]
                row_identities[(owner, row)] = json.dumps(
                    row_identity, sort_keys=True, separators=(",", ":")
                )

        return cls(
            adapter=adapter,
            snapshot=snapshot,
            decode=decode,
            analyses=analyses,
            heights=heights,
            class_handles=class_handles,
            class_identities=class_identities,
            term_owners={
                term: tuple(owners) for term, owners in term_owners_mut.items()
            },
            row_identities=row_identities,
        )

    def class_identity(self, ref: EClassRef) -> str:
        return self.class_identities[ref]

    def class_state_identity(self, ref: EClassRef) -> str:
        """Identify proof-relevant facts that can change after a union."""
        analysis = self.analyses.get(ref)
        dims = tuple(str(dim) for dim in getattr(analysis, "dims", ()))
        facts = tuple(
            sorted(
                fact.sexpr() if hasattr(fact, "sexpr") else repr(fact)
                for fact in getattr(analysis, "facts", ())
            )
        )
        value = [
            self.class_identity(ref),
            dims,
            facts,
            _json_value(self.witness_identity(ref)),
        ]
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def row_state_identity(self, owner: EClassRef, row: ENodeRef) -> str | None:
        base = self.row_identities.get((owner, row))
        if base is None:
            return None
        decoded = self.decode(self.snapshot, row)
        value = [
            base,
            [
                self.class_state_identity(child)
                for child in dict.fromkeys(decoded.child_classes)
            ],
        ]
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def occurrence_identity(
        self,
        *,
        consumer_class: EClassRef,
        q: int,
        producer_op: str,
        producer_attrs: dict[str, Any],
        producer_children: tuple[EClassRef, ...],
        consumer_op: str,
        consumer_attrs: dict[str, Any],
        consumer_children: tuple[EClassRef, ...],
    ) -> str:
        value = [
            self.class_identity(consumer_class),
            q,
            producer_op,
            _json_value(producer_attrs),
            [self.class_identity(child) for child in producer_children],
            consumer_op,
            _json_value(consumer_attrs),
            [self.class_identity(child) for child in consumer_children],
            [
                self.class_state_identity(child)
                for child in dict.fromkeys((*producer_children, *consumer_children))
            ],
        ]
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def candidate_is_represented(
        self, consumer_class: EClassRef, candidate: Term
    ) -> bool:
        return self.represented_owner(candidate) == consumer_class

    def represented_owner(self, term: Term) -> EClassRef | None:
        """Resolve a nested term through represented child e-classes."""
        if isinstance(term, TermRef):
            return term.eclass
        if term in self._represented_owner_cache:
            return self._represented_owner_cache[term]
        child_owners: list[EClassRef] = []
        for child in term.children:
            owner = self.represented_owner(child)
            if owner is None:
                self._represented_owner_cache[term] = None
                return None
            child_owners.append(owner)
        shallow = TermApp.make(
            term.op,
            term.attrs_dict(),
            tuple(TermRef(owner) for owner in child_owners),
        )
        owners = self.term_owners.get(shallow, ())
        owner = owners[0] if owners else None
        self._represented_owner_cache[term] = owner
        return owner

    def witness_identity(self, ref: EClassRef) -> Any:
        """Stable structural identity of the deterministic finite witness."""
        cached = self._witness_identity_cache.get(ref)
        if cached is not None:
            return cached
        row = minimum_height_row(self.snapshot, ref, self.heights)
        decoded = self.decode(self.snapshot, row)
        if decoded.op == "input":
            identity: Any = (
                "input",
                decoded.source_id,
                _frozen_value(decoded.input_shape),
            )
        else:
            identity = (
                decoded.op,
                _frozen_value(dict(decoded.attrs)),
                tuple(self.witness_identity(child) for child in decoded.child_classes),
            )
        self._witness_identity_cache[ref] = identity
        return identity


__all__ = ["SemanticContext"]
