"""Host-only tests for structured candidate filtering."""

from __future__ import annotations

from axon.candidate_filter import (
    MANIFEST_NAME,
    TileConstraint,
    manifest_literal,
    read_manifest,
    rejection_reason,
    select_diverse,
)


def test_positive_block_constraint() -> None:
    constraint = TileConstraint("TILES_IN_BLOCK_N", 512, 512, False)
    assert constraint.rejection_reason(1) is None
    assert "block 1024 > extent 512" in constraint.rejection_reason(2)


def test_divisibility_constraint() -> None:
    constraint = TileConstraint("TILES_IN_BLOCK_N", 768, 128, True)
    assert constraint.rejection_reason(2) is None
    assert "does not divide extent 768" in constraint.rejection_reason(4)


def test_manifest_round_trip(tmp_path) -> None:
    constraints = (
        TileConstraint("TILES_IN_BLOCK_M", 128, 128, False),
        TileConstraint("TILES_IN_BLOCK_N", 512, 512, False),
    )
    module = tmp_path / "kernel.py"
    module.write_text(f"{MANIFEST_NAME} = {manifest_literal(constraints)}\n")
    assert read_manifest(module) == constraints


def test_missing_or_nonliteral_manifest_fails_open(tmp_path) -> None:
    missing = tmp_path / "missing.py"
    assert read_manifest(missing) == ()
    dynamic = tmp_path / "dynamic.py"
    dynamic.write_text(f"{MANIFEST_NAME} = make_constraints()\n")
    assert read_manifest(dynamic) == ()
    future = tmp_path / "future.py"
    future.write_text(f"{MANIFEST_NAME} = {{'version': 999, 'constraints': ()}}\n")
    assert read_manifest(future) == ()


def test_rejection_reason_matches_tile_argument_by_name() -> None:
    constraints = (TileConstraint("TILES_IN_BLOCK_N", 512, 512, False),)
    assert (
        rejection_reason(("TILES_IN_BLOCK_M", "TILES_IN_BLOCK_N"), (32, 1), constraints)
        is None
    )
    assert (
        rejection_reason(("TILES_IN_BLOCK_M", "TILES_IN_BLOCK_N"), (1, 2), constraints)
        is not None
    )


def test_diverse_selection_is_bounded_deterministic_and_keeps_corners() -> None:
    candidates = [(m, n) for m in (1, 2, 4, 8) for n in (1, 2, 4, 8)]
    selected = select_diverse(candidates, budget=8, seed=42)
    assert len(selected) == 8
    assert selected == select_diverse(candidates, budget=8, seed=42)
    assert (1, 1) in selected
    assert (8, 8) in selected
    assert (1, 8) in selected
    assert (8, 1) in selected


def test_zero_budget_keeps_all_legal_candidates() -> None:
    candidates = [(1,), (2,), (4,)]
    assert select_diverse(candidates, budget=0, seed=42) == candidates
