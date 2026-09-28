"""Record and print every rewrite the e-graph admits.

Each admitted equality (a propagation swap, a fusion, or a lowering) becomes a
``RewriteEvent`` holding the matched graph and the graph that replaced it, both
rendered as indented trees. Leaves that are e-classes are expanded through one
representative member each, so the printout is a whole graph down to the
inputs rather than a term over opaque class ids.

Enable it in one of three ways:

* set ``AXON_TRACE_REWRITES=1`` (print to stderr) or ``AXON_TRACE_REWRITES=<path>``
  (append to a file);
* pass ``--trace-rewrites [PATH]`` to ``axon``;
* use ``capture_rewrites()`` in code or tests, which collects the events.

Rendering happens only when tracing is on, so a normal run pays nothing.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

from axon.egraph.adapter import EClassRef, Snapshot

_ENV_VAR = "AXON_TRACE_REWRITES"
_MAX_DEPTH = 12

RewriteListener = Callable[["RewriteEvent"], None]

_LOCK = threading.Lock()
_LISTENERS: list[RewriteListener] = []
_SEQUENCE = [0]


@dataclass(frozen=True)
class RewriteEvent:
    """One admitted rewrite: the graph it matched and the graph it added."""

    sequence: int
    stage: str
    target: str
    before: str
    after: str

    def render(self) -> str:
        rule = "=" * 72
        return (
            f"{rule}\n"
            f"rewrite #{self.sequence} [{self.stage}] into {self.target}\n"
            f"--- matched graph\n{self.before}\n"
            f"+++ new graph\n{self.after}\n"
        )


# Listener registry ---------------------------------------------------------


def add_listener(listener: RewriteListener) -> None:
    with _LOCK:
        _LISTENERS.append(listener)


def remove_listener(listener: RewriteListener) -> None:
    with _LOCK:
        if listener in _LISTENERS:
            _LISTENERS.remove(listener)


@contextmanager
def capture_rewrites(echo: bool = False) -> Iterator[list[RewriteEvent]]:
    """Collect the rewrites admitted inside the block; optionally print them."""
    events: list[RewriteEvent] = []

    def listener(event: RewriteEvent) -> None:
        events.append(event)
        if echo:
            print(event.render(), flush=True)

    add_listener(listener)
    try:
        yield events
    finally:
        remove_listener(listener)


def enable_from_cli(target: str | None) -> None:
    """``--trace-rewrites [PATH]``: print to stderr, or append to ``PATH``."""
    os.environ[_ENV_VAR] = target or "1"


def tracing_enabled() -> bool:
    with _LOCK:
        if _LISTENERS:
            return True
    return bool(os.environ.get(_ENV_VAR))


def _env_sink(event: RewriteEvent) -> None:
    target = os.environ.get(_ENV_VAR, "")
    if not target:
        return
    text = event.render()
    if target in ("1", "true", "stderr"):
        print(text, file=sys.stderr, flush=True)
        return
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def _publish(stage: str, target: str, before: str, after: str) -> None:
    with _LOCK:
        _SEQUENCE[0] += 1
        event = RewriteEvent(_SEQUENCE[0], stage, target, before, after)
        listeners = list(_LISTENERS)
    _env_sink(event)
    for listener in listeners:
        listener(event)


# Rendering -----------------------------------------------------------------


def format_value(value: Any) -> str:
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    name = getattr(value, "name", None)
    if type(value).__name__ == "_OpRef" and isinstance(name, str):
        return f"nl.{name}"
    if isinstance(value, float) and value.is_integer():
        return repr(value)
    return str(value) if hasattr(value, "sexpr") else repr(value)


_DEFAULT_ENUM_NAMES = frozenset({"none", "unknown", "idle"})


def _is_default(value: Any) -> bool:
    """Attribute values that only restate a default and clutter the graph."""
    if value is None or value is False:
        return True
    if isinstance(value, (tuple, list)) and not value:
        return True
    return isinstance(value, Enum) and value.name in _DEFAULT_ENUM_NAMES


def format_attrs(attrs: dict[str, Any] | tuple[tuple[str, Any], ...]) -> str:
    items = attrs.items() if isinstance(attrs, dict) else attrs
    shown = [
        f"{key}={format_value(value)}"
        for key, value in sorted(items, key=lambda item: item[0])
        if not _is_default(value) and key not in ("name", "out_shape", "sym_shape")
    ]
    return "{" + ", ".join(shown) + "}" if shown else ""


class GraphRenderer:
    """Render terms over e-classes as trees expanded down to the inputs."""

    def __init__(
        self,
        snapshot: Snapshot,
        decode: Callable[[Snapshot, Any], Any],
        heights: dict[EClassRef, int] | None = None,
        analyses: dict[EClassRef, Any] | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.decode = decode
        self.heights = heights or {}
        self.analyses = analyses or {}
        self._labels: dict[EClassRef, str] = {}

    def label(self, ref: EClassRef) -> str:
        label = self._labels.get(ref)
        if label is None:
            label = f"%{len(self._labels)}"
            self._labels[ref] = label
        return label

    def _shape(self, ref: EClassRef) -> str:
        analysis = self.analyses.get(ref)
        dims = getattr(analysis, "dims", None)
        if dims is None:
            return ""
        return " : [" + ", ".join(str(dim) for dim in dims) + "]"

    def _representative(self, ref: EClassRef) -> Any | None:
        """The member whose children are shallowest, so expansion terminates."""
        best: tuple[int, Any] | None = None
        for row in self.snapshot.members(ref):
            try:
                decoded = self.decode(self.snapshot, row)
            except Exception:
                continue
            depth = max(
                (self.heights.get(child, 0) for child in decoded.child_classes),
                default=-1,
            )
            if best is None or depth < best[0]:
                best = (depth, decoded)
        return None if best is None else best[1]

    def render_class(
        self, ref: EClassRef, indent: int, path: frozenset[EClassRef]
    ) -> list[str]:
        pad = "  " * indent
        label = self.label(ref)
        decoded = self._representative(ref)
        if decoded is None or ref in path or indent > _MAX_DEPTH:
            return [f"{pad}{label}{self._shape(ref)}"]
        if decoded.op == "input":
            source = decoded.source_id or "?"
            return [f"{pad}input '{source}'{self._shape(ref)}"]
        lines = [f"{pad}{decoded.op}{format_attrs(decoded.attrs)}{self._shape(ref)}"]
        for child in decoded.child_classes:
            lines.extend(self.render_class(child, indent + 1, path | {ref}))
        return lines

    def render_term(self, term: Any, indent: int = 0) -> list[str]:
        if isinstance(term, EClassRef):
            return self.render_class(term, indent, frozenset())
        eclass = getattr(term, "eclass", None)
        if eclass is not None:
            return self.render_class(eclass, indent, frozenset())
        pad = "  " * indent
        lines = [f"{pad}{term.op}{format_attrs(term.attrs)}"]
        for child in term.children:
            lines.extend(self.render_term(child, indent + 1))
        return lines


def format_sketch(sketch: Any, indent: int = 0) -> list[str]:
    pad = "  " * indent
    if sketch.op == "INPUT":
        sym = getattr(sketch, "sym", None)
        name = getattr(sym, "id", "?")
        return [f"{pad}input '{name}'"]
    lines = [f"{pad}{sketch.op}{format_attrs(dict(sketch.attrs))}"]
    for child in sketch.children:
        lines.extend(format_sketch(child, indent + 1))
    return lines


# Hooks called by the admission paths ------------------------------------------


def record_term_rewrite(
    stage: str,
    context: Any,
    target: EClassRef,
    before: Any,
    after: Any,
) -> None:
    """Publish one admitted term rewrite; a no-op unless tracing is enabled."""
    if not tracing_enabled():
        return
    try:
        renderer = GraphRenderer(
            context.snapshot,
            context.decode,
            getattr(context, "heights", None),
            getattr(context, "analyses", None),
        )
        target_label = f"{renderer.label(target)} ({target!r})"
        before_lines = renderer.render_term(before)
        after_lines = renderer.render_term(after)
    except Exception as exc:  # tracing must never break saturation
        target_label, before_lines, after_lines = repr(target), [f"<{exc}>"], []
    _publish(stage, target_label, "\n".join(before_lines), "\n".join(after_lines))


def record_lowering(stage: str, target: Any, sketch: Any) -> None:
    """Publish one admitted tensor-to-ISA lowering."""
    if not tracing_enabled():
        return
    try:
        after = "\n".join(format_sketch(sketch))
    except Exception as exc:
        after = f"<{exc}>"
    _publish(stage, repr(target), f"tensor class {target!r}", after)


def format_sym_expr(expr: Any, indent: int = 0) -> str:
    """Render a ``SymExpr`` tree (the proof-level graph) as indented text."""
    pad = "  " * indent
    shape = "[" + ", ".join(str(dim) for dim in expr.shape) + "]"
    if expr.op == "input":
        return f"{pad}input '{expr.name}' : {shape}"
    attrs = format_attrs({k: v for k, v in expr.attrs.items() if k != "shape"})
    lines = [f"{pad}{expr.op}{attrs} : {shape}"]
    lines.extend(format_sym_expr(child, indent + 1) for child in expr.inputs)
    return "\n".join(lines)
