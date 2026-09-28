from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# Shared input-generation seed
SEED = 42


@dataclass(frozen=True, slots=True)
class KernelSpec:
    name: str
    axon_kernel: Callable[..., Any]
    # Ordered canonical dim namespace (make_inputs keyword order, not sorted):
    # each name is a make_inputs param, a --sizes position, and a CSV column.
    dim_vars: tuple[str, ...]
    input_specs: Sequence[tuple[str, tuple[str | int, ...]]]
    make_inputs: Callable[..., tuple[Any, ...]]
    tile_args: Sequence[str]
    # NeuronPy baseline (numpy): `--mode baseline` timing
    baseline_op: Callable[..., Any] | None = None
    # Torch reference (used by the head-to-head harness).
    torch_ref: Callable[..., Any] | None = None
    baseline_compiler_args: str = "--model-type=transformer"
    tile_options: Sequence[int] = (1, 2, 4, 8, 16, 32)

    def __post_init__(self) -> None:
        if len(set(self.dim_vars)) != len(self.dim_vars):
            raise ValueError(
                f"kernel '{self.name}': dim_vars has duplicate names: {self.dim_vars}"
            )
        spec_labels = {
            dim
            for _, sym_shape in self.input_specs
            for dim in sym_shape
            if isinstance(dim, str)
        }
        if set(self.dim_vars) != spec_labels:
            raise ValueError(
                f"kernel '{self.name}': dim_vars {set(self.dim_vars)} does not "
                f"match input_specs labels {spec_labels}"
            )
