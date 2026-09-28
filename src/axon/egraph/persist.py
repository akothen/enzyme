"""Persist a finished e-graph search so emission can replay it without synthesis.

A ``Snapshot`` is already a decoded pure-Python view, so the only obstacle to
pickling one is ``EClassRef.value``: egglog hands out a Rust-backed
``builtins.Value`` that cannot pickle. Every e-class reference is rebuilt with a
``_CachedValue`` standing in for it, which prints as ``Value(N)`` so cached refs
repr identically to live ones and every ``ENodeRef._sort_key`` ordering the
extraction path depends on is preserved.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import os
import pickle
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axon.egraph.adapter import EClassRef, ENodeRef, LitArg, Snapshot

# Bump on any change to the persisted schema below.
FORMAT_VERSION = 1

_VALUE_TEXT = re.compile(r"^Value\((\d+)\)$")


class CacheError(RuntimeError):
    """Raised when a cache file is unreadable, stale, or does not match the run."""


@dataclass(frozen=True, repr=False)
class _CachedValue:
    """Picklable stand-in for egglog's ``builtins.Value`` in a cached snapshot."""

    ordinal: int

    def __str__(self) -> str:
        return f"Value({self.ordinal})"

    __repr__ = __str__


def _encode_value(value: Any) -> _CachedValue:
    if isinstance(value, _CachedValue):
        return value
    match = _VALUE_TEXT.match(str(value))
    if match is None:
        raise CacheError(
            f"cannot persist e-class value {value!r}: expected an egglog value "
            f"printing as 'Value(N)'. Installed egglog is {installed_egglog_version()}"
        )
    return _CachedValue(int(match.group(1)))


def _encode_ref(ref: EClassRef) -> EClassRef:
    """Rebuild one reference with a picklable value, preserving its identity."""
    return EClassRef(
        snapshot_id=ref.snapshot_id,
        value=_encode_value(ref.value),
        sort=ref.sort,
        adapter_id=ref.adapter_id,
    )


def _encode_arg(arg: LitArg | EClassRef) -> LitArg | EClassRef:
    return _encode_ref(arg) if isinstance(arg, EClassRef) else arg


def _encode_row(row: ENodeRef) -> ENodeRef:
    return ENodeRef(
        callable=row.callable,
        egg_fn=row.egg_fn,
        sort=row.sort,
        args=tuple(_encode_arg(arg) for arg in row.args),
    )


@dataclass(frozen=True)
class SnapshotData:
    """One frozen e-graph in picklable form, rebuilt through ``Snapshot``."""

    snapshot_id: int
    adapter_id: int
    # Row order within a class is the order the live snapshot sorted them into,
    # and extraction enumerates selections in exactly that order.
    classes: tuple[tuple[EClassRef, tuple[ENodeRef, ...]], ...]
    let_bindings: tuple[tuple[str, EClassRef], ...]

    def rebuild(self) -> Snapshot:
        return Snapshot(
            self.snapshot_id,
            dict(self.classes),
            dict(self.let_bindings),
            adapter_id=self.adapter_id,
        )


def encode_snapshot(snapshot: Snapshot) -> SnapshotData:
    """Remap one snapshot's e-class values so the whole structure pickles."""
    return SnapshotData(
        snapshot_id=snapshot.snapshot_id,
        adapter_id=snapshot.adapter_id,
        classes=tuple(
            (_encode_ref(cref), tuple(_encode_row(row) for row in rows))
            for cref, rows in snapshot.classes.items()
        ),
        let_bindings=tuple(
            (name, _encode_ref(ref)) for name, ref in snapshot.let_bindings.items()
        ),
    )


@dataclass(frozen=True)
class EGraphCacheFile:
    """One synthesis run's frozen e-graphs plus what validates replaying them."""

    format_version: int
    egglog_version: str
    kernel_name: str
    dim_sizes: dict[str, int]
    graph_identity: str
    run_stem: str
    options: dict[str, Any]
    tensor: SnapshotData
    tensor_output_roots: tuple[EClassRef, ...]
    isa: SnapshotData
    isa_output_roots: tuple[EClassRef, ...]
    input_metadata: dict[str, dict[str, Any]]
    declared_input_ids: tuple[str, ...]
    # Only `terminal_status` drives replay; the per-stage statuses are kept so a
    # cached run can say which stage stopped the search that produced it.
    tensor_status: Any
    lowering_status: Any
    isa_status: Any
    terminal_status: Any

    def require_matches(
        self,
        *,
        kernel_name: str,
        dim_sizes: dict[str, int],
        graph_identity: str,
        run_stem: str,
        path: Path,
    ) -> None:
        """Reject a cache that does not describe the run being replayed."""
        terminal = self.terminal_status
        if (
            terminal.status == "failed"
            or terminal.unrealized_outputs
            or not self.isa_output_roots
        ):
            raise CacheError(
                f"{path}: this cache holds a search that realized no output "
                f"({terminal.stop_reason or 'unknown'} at stage "
                f"{terminal.truncated_stage or 'unknown'}), so "
                "replaying it can emit nothing"
            )
        if self.kernel_name != kernel_name:
            raise CacheError(
                f"{path}: cache holds kernel '{self.kernel_name}', "
                f"this run is '{kernel_name}'"
            )
        if self.run_stem != run_stem:
            raise CacheError(
                f"{path}: cache was written by run '{self.run_stem}', "
                f"this run is '{run_stem}'"
            )
        if self.dim_sizes != dim_sizes:
            raise CacheError(
                f"{path}: cache holds sizes {self.dim_sizes}, this run has {dim_sizes}"
            )
        if self.graph_identity != graph_identity:
            raise CacheError(
                f"{path}: the kernel's traced graph changed since this cache was "
                "written (identity mismatch); delete it and re-run synthesis"
            )

    def describe_options(self) -> str:
        return ", ".join(f"{k}={self.options[k]}" for k in sorted(self.options))

    def describe_stages(self) -> str:
        """The per-stage outcome of the saved search, for the replay log line."""
        stages = (
            ("tensor", self.tensor_status),
            ("lowering", self.lowering_status),
            ("isa", self.isa_status),
        )
        return ", ".join(
            f"{name}={getattr(status, 'status', '?')}" for name, status in stages
        )


def installed_egglog_version() -> str:
    try:
        return importlib.metadata.version("egglog")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def cache_file_path(
    cache_dir: Path,
    kernel_name: str,
    dim_sizes: dict[str, int],
    run_stem: str | None = None,
) -> Path:
    """`cache/<run stem>.bin`, or `cache/<kernel>__<dim><size>_....bin` when the
    run has no stem. The stem is what makes two runs distinct: seven declared
    case pairs share a (kernel, sizes) and differ only by dtype, so keying on
    that alone lets concurrent cases overwrite and then replay each other."""
    if run_stem:
        return cache_dir / f"{run_stem}.bin"
    sizes = "_".join(f"{dim}{size}" for dim, size in dim_sizes.items())
    stem = f"{kernel_name}__{sizes}" if sizes else kernel_name
    return cache_dir / f"{stem}.bin"


def build_cache_file(
    search: Any,
    *,
    kernel_name: str,
    dim_sizes: dict[str, int],
    graph_identity: str,
    run_stem: str,
    options: dict[str, Any],
) -> EGraphCacheFile:
    """Encode one ``EGraphSearch`` into the persisted record."""
    terminal = search.terminal_status
    lowering = search.lowering_status
    return EGraphCacheFile(
        format_version=FORMAT_VERSION,
        egglog_version=installed_egglog_version(),
        kernel_name=kernel_name,
        dim_sizes=dict(dim_sizes),
        graph_identity=graph_identity,
        run_stem=run_stem,
        options=dict(options),
        tensor=encode_snapshot(search.tensor_snapshot),
        tensor_output_roots=tuple(
            _encode_ref(ref) for ref in search.tensor_output_roots
        ),
        isa=encode_snapshot(search.isa_snapshot),
        isa_output_roots=tuple(_encode_ref(ref) for ref in search.isa_output_roots),
        input_metadata={
            source_id: dict(meta) for source_id, meta in search.input_metadata.items()
        },
        declared_input_ids=tuple(search.declared_input_ids),
        tensor_status=search.tensor_status,
        # Both statuses carry tensor-snapshot refs that need the same remap.
        lowering_status=_replace_unrealized(lowering),
        isa_status=search.isa_status,
        terminal_status=_replace_unrealized(terminal),
    )


def _replace_unrealized(status: Any) -> Any:
    """Remap the e-class refs a status carries, leaving its other fields alone."""
    unrealized = getattr(status, "unrealized_outputs", ())
    if not unrealized:
        return status
    return dataclasses.replace(
        status,
        unrealized_outputs=tuple(_encode_ref(ref) for ref in unrealized),
    )


def save_cache(path: Path, cache: EGraphCacheFile) -> None:
    """Write the record atomically so an interrupted save leaves no torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp = Path(handle.name)
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except BaseException:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        raise


def load_cache(path: Path) -> EGraphCacheFile:
    """Read one cache file, rejecting a schema or egglog mismatch."""
    if not path.is_file():
        raise CacheError(
            f"no e-graph cache at {path}; run the same case without --from-cache "
            "to synthesize and write it"
        )
    try:
        cache = pickle.loads(path.read_bytes())
    except Exception as exc:
        raise CacheError(f"{path}: unreadable e-graph cache ({exc})") from exc
    if not isinstance(cache, EGraphCacheFile):
        raise CacheError(f"{path}: not an e-graph cache file")
    if cache.format_version != FORMAT_VERSION:
        raise CacheError(
            f"{path}: cache format version {cache.format_version}, this build "
            f"reads {FORMAT_VERSION}; delete it and re-run synthesis"
        )
    running = installed_egglog_version()
    if cache.egglog_version != running:
        raise CacheError(
            f"{path}: cache was written against egglog {cache.egglog_version}, "
            f"running {running}; delete it and re-run synthesis"
        )
    return cache


__all__ = [
    "FORMAT_VERSION",
    "CacheError",
    "EGraphCacheFile",
    "SnapshotData",
    "build_cache_file",
    "cache_file_path",
    "encode_snapshot",
    "installed_egglog_version",
    "load_cache",
    "save_cache",
]
