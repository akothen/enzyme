"""Structured pre-compile filtering for emitted Axon tile candidates."""

from __future__ import annotations

import ast
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_NAME = "__axon_tile_constraints__"
MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class TileConstraint:
    """One generated ``TILES_IN_BLOCK_*`` feasibility condition."""

    tile_arg: str
    extent: int
    tile_size: int
    require_divisible: bool

    def rejection_reason(self, value: int) -> str | None:
        block = self.tile_size * value
        if value <= 0:
            return f"{self.tile_arg}={value} is not positive"
        if block > self.extent:
            return f"{self.tile_arg}={value} gives block {block} > extent {self.extent}"
        if self.require_divisible and self.extent % block != 0:
            return (
                f"{self.tile_arg}={value} gives block {block}, which does not "
                f"divide extent {self.extent}"
            )
        return None

    def as_record(self) -> dict[str, Any]:
        return {
            "tile_arg": self.tile_arg,
            "extent": self.extent,
            "tile_size": self.tile_size,
            "require_divisible": self.require_divisible,
        }


def manifest_literal(constraints: Sequence[TileConstraint]) -> str:
    records = tuple(
        constraint.as_record()
        for constraint in sorted(constraints, key=lambda item: item.tile_arg)
    )
    return repr({"version": MANIFEST_VERSION, "constraints": records})


def read_manifest(module_path: str | Path) -> tuple[TileConstraint, ...]:
    """Read only the literal Axon manifest from an emitted Python module."""
    path = Path(module_path)
    if not path.is_file():
        return ()
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
        value: Any | None = None
        for statement in tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == MANIFEST_NAME
                for target in statement.targets
            ):
                value = ast.literal_eval(statement.value)
                break
        if value is None:
            return ()
        if isinstance(value, dict):
            if value.get("version") != MANIFEST_VERSION:
                return ()
            records = value.get("constraints")
        else:
            # Compatibility with the initial unversioned validation artifacts.
            records = value
        constraints = tuple(TileConstraint(**record) for record in records)
    except (OSError, SyntaxError, ValueError, TypeError):
        return ()
    if len({item.tile_arg for item in constraints}) != len(constraints):
        return ()
    return constraints


def rejection_reason(
    tile_args: Sequence[str],
    tile_values: Sequence[int],
    constraints: Sequence[TileConstraint],
) -> str | None:
    values = dict(zip(tile_args, tile_values, strict=True))
    for constraint in constraints:
        value = values.get(constraint.tile_arg)
        if value is None:
            continue
        reason = constraint.rejection_reason(int(value))
        if reason is not None:
            return reason
    return None


def select_diverse(
    tile_values: Sequence[tuple[int, ...]],
    *,
    budget: int,
    seed: int,
) -> list[tuple[int, ...]]:
    """Select deterministic quality/diversity representatives from legal tiles."""
    candidates = sorted(set(tile_values))
    if budget <= 0 or len(candidates) <= budget:
        return candidates
    width = len(candidates[0])
    maxima = tuple(
        max(values[index] for values in candidates) for index in range(width)
    )
    minima = tuple(
        min(values[index] for values in candidates) for index in range(width)
    )

    def vector(values: tuple[int, ...]) -> tuple[float, ...]:
        return tuple(
            math.log2(value) / math.log2(maximum) if maximum > 1 else 0.0
            for value, maximum in zip(values, maxima, strict=True)
        )

    vectors = {values: vector(values) for values in candidates}

    def quality(values: tuple[int, ...]) -> float:
        point = vectors[values]
        return sum(point) / len(point) if point else 0.0

    def tie_break(values: tuple[int, ...]) -> int:
        payload = f"{seed}:{','.join(map(str, values))}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    anchors = {min(candidates), max(candidates)}
    for index in range(width):
        low = list(minima)
        low[index] = maxima[index]
        high = list(maxima)
        high[index] = minima[index]
        if tuple(low) in vectors:
            anchors.add(tuple(low))
        if tuple(high) in vectors:
            anchors.add(tuple(high))
    selected = sorted(
        anchors, key=lambda values: (-quality(values), tie_break(values))
    )[:budget]

    def distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
        return math.sqrt(
            sum(
                (a - b) ** 2 for a, b in zip(vectors[left], vectors[right], strict=True)
            )
        )

    remaining = [values for values in candidates if values not in selected]
    while remaining and len(selected) < budget:
        choice = max(
            remaining,
            key=lambda values: (
                0.65 * quality(values)
                + 0.35 * min(distance(values, prior) for prior in selected),
                -tie_break(values),
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
    return sorted(selected)
