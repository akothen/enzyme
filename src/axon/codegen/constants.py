"""Hardware tile/partition constants for codegen (Python-level only).

The literal ``128`` / ``512`` inside emitted strings are emitted source, not
Python values, and are deliberately not routed through here.
"""

from __future__ import annotations

# NeuronCore matmul tile maxima (nl.tile_size.gemm_stationary_fmax / _moving_fmax).
PARTITION_FMAX = 128
MOVING_FMAX = 512

# Default tile sizes used when no tile annotation resolves a concrete candidate.
DEFAULT_TILE_M = 128
DEFAULT_TILE_K = 128  # hardware constant for the nc_matmul partition dim
DEFAULT_TILE_N = 512

# Default per-klass tile-config dicts (the shapes the body builders consume).
DEFAULT_ELEMENTWISE_TILE = {"tile_m": DEFAULT_TILE_M, "tile_n": DEFAULT_TILE_N}
DEFAULT_MATMUL_TILE = {
    "tile_m": DEFAULT_TILE_M,
    "tile_k": DEFAULT_TILE_K,
    "tile_n": DEFAULT_TILE_N,
}
