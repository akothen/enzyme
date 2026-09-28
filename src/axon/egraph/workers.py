"""Shared proof-worker defaults for public e-graph synthesis entry points."""

from __future__ import annotations

import os


def available_cpu_count() -> int:
    """Return the detected host CPU count, with a one-worker fallback."""
    return os.cpu_count() or 1


def resolve_worker_count(workers: int | None) -> int:
    """Resolve an automatic worker count and validate explicit values."""
    resolved = available_cpu_count() if workers is None else workers
    if resolved < 1:
        raise ValueError("workers must be positive or None")
    return resolved


__all__ = ["available_cpu_count", "resolve_worker_count"]
