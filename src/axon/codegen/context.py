"""Per-emit state, threaded explicitly through the codegen package."""

from __future__ import annotations

from dataclasses import dataclass, field

from axon.candidate_filter import TileConstraint
from axon.codegen.constants import DEFAULT_TILE_K, DEFAULT_TILE_M, DEFAULT_TILE_N


@dataclass
class EmitCtx:
    """Per-emit state: read-only config plus mutable emission scratch.

    ``tp_counter`` / ``nest_counter`` mint ``tp_psum_<n>`` / ``nest_scratch_<n>``
    names; they are reset once at the top of ``emit_matmul_body_generic`` and must
    keep climbing across ``nc_transpose_psum``, the inline post-matmul mint site,
    and all inner loops (reduce preambles, chained matmul stages), so the same ctx
    is threaded through the entire emit call in place (a copy would desync the
    numbering).

    ``partition_scalar_ids`` (node ids whose tile is a per-partition ``(P, 1)``
    scalar) is populated during graph analysis in ``emit_matmul_body_generic``
    and read by the op emitters to decide the operand-order swap.
    """

    # Read-only config.
    kernel_name: str = ""
    indent: str = "    "
    tile_config: dict[str, int] = field(default_factory=dict)
    tile_config_tag: str = ""
    fuse_loads: bool = False
    rhs_transpose_strategy: str = "separate_loop"
    tile_constraints: dict[str, TileConstraint] = field(default_factory=dict)

    # Per-emit mutable scratch.
    partition_scalar_ids: set[str] = field(default_factory=set)
    tp_counter: int = 0
    nest_counter: int = 0

    def tile(self, key: str) -> int:
        """Resolved tile size for ``key``, with a default fallback for a partial
        ``tile_config`` (``assemble.emit`` normally fills every key)."""
        defaults = {
            "tile_m": DEFAULT_TILE_M,
            "tile_k": DEFAULT_TILE_K,
            "tile_n": DEFAULT_TILE_N,
        }
        return self.tile_config.get(key, defaults[key])
