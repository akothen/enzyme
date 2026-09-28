"""The commutative-monoid library.

A cross-core reduction combine is a commutative monoid `(M, ⊕, e)`. The pattern
category is determined by *which* monoid. This module is the extension seam:
adding a sharded reduction means adding a `Monoid` entry, not editing the search.

Tiers:
  - codegen-emittable: `+`, `id` (the trivial permutation monoid), `max`/`min`,
    and the variance carrier (`var_naive` naive sufficient statistics,
    `welford` parallel-merge). `lift`/`finalize`/`combine_emit` map onto codegen
    body builders, so codegen lowers them with NO second synthesis pass.
  - synthesis-requiring: `flash`. Its `lift` (`Q@Kᵀ` / softmax-partial / `P@V`)
    contains matmuls codegen cannot lower, so it carries a `body_synthesis` hook
    and ships stubbed (`combine_emit` raises until populated).

The records are pure data: the actual NKI emission lives in the SPMD assembler,
which reads `combine_emit` / `finalize` to drive codegen. The Python `lift` /
`finalize` callables here are the *reference* semantics — used by the offline
law-discharge tests (`tests/test_sharding_laws.py`) to machine-check
associativity, commutativity, identity, and (for stats) the sufficient-statistic
homomorphism.

Each monoid is built by its own factory function so its helper closures
(`lift`/`merge`/`finalize`) are scoped to one block rather than scattered across
the module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

MonoidId = str

# How the assembler lowers `⊕` on (local, peer) after one sendrecv:
#   "tensor_tensor": a single tensor_tensor(op) — `+`/`max`/`min`.
#   "componentwise": apply the elementwise op across each carrier component
#                    (naive stats `(Σx, Σx², n)`).
#   "welford":       the multi-op non-componentwise parallel-merge sequence.
#   "flash":         the 8-op rescale-and-add (deferred).
CombineKind = Literal["tensor_tensor", "componentwise", "welford", "flash"]


@dataclass(frozen=True)
class Monoid:
    """One commutative monoid available as a cross-core combine."""

    id: MonoidId
    is_permutation: bool  # True ⇒ no arithmetic (the trivial/identity monoid)
    # The elementwise op the combine lowers to (for the 1-op / componentwise
    # tiers); None for the permutation monoid and for welford/flash.
    nl_op: str | None
    combine_kind: CombineKind
    # Reference semantics for the offline law discharge. `lift` maps a value to
    # the carried partial; `finalize` maps the reduced carrier to the output;
    # `merge` is the binary ⊕ on carriers. `identity` is the empty-shard value.
    # For `+`/`max`/`min` the partial *is* the value, so lift/finalize are None.
    lift: Callable | None
    finalize: Callable | None
    merge: Callable | None
    identity: object | None
    # The two-tier discriminator: a hook only for a body-splitting monoid
    # (flash). None ⇒ codegen emits lift|combine|finalize directly.
    body_synthesis: Callable | None = None


def _scalar_monoid(
    id: MonoidId, nl_op: str, merge: Callable, identity: object
) -> Monoid:
    """A 1-op scalar monoid (`+`/`max`/`min`): the partial *is* the value, so it
    carries no statistics (`lift`/`finalize` are None) and the combine lowers to
    a single `tensor_tensor(nl_op)`."""
    return Monoid(
        id=id,
        is_permutation=False,
        nl_op=nl_op,
        combine_kind="tensor_tensor",
        lift=None,
        finalize=None,
        merge=merge,
        identity=identity,
    )


def _id_monoid() -> Monoid:
    """The trivial permutation monoid (`id`): no arithmetic, used by X-gather and
    the P-barrier where the combine only relocates finished data."""
    return Monoid(
        id="id",
        is_permutation=True,
        nl_op=None,
        combine_kind="tensor_tensor",
        lift=None,
        finalize=None,
        merge=None,
        identity=None,
    )


def _var_naive_monoid() -> Monoid:
    """Variance via the naive sufficient-statistic carrier `(Σx, Σx², n)`,
    combined under componentwise `+`. Same variance as `welford`, but merged
    additively rather than by Welford's parallel-merge — the two differ only in
    the merge, not the result."""

    def lift(x, axis):
        arr = np.asarray(x, dtype=np.float64)
        return (
            arr.sum(axis=axis),
            np.square(arr).sum(axis=axis),
            float(arr.shape[axis]),
        )

    def merge(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def finalize(carrier, eps=0.0):
        s, sq, n = carrier
        mean = s / n
        var = sq / n - np.square(mean)
        return 1.0 / np.sqrt(var + eps)

    return Monoid(
        id="var_naive",
        is_permutation=False,
        nl_op="add",
        combine_kind="componentwise",
        lift=lift,
        finalize=finalize,
        merge=merge,
        identity=(0.0, 0.0, 0.0),
    )


def _welford_monoid() -> Monoid:
    """Variance via Welford's parallel-merge carrier `(mean, M2, n)`. Same
    variance as `var_naive` under a different merge; exists to validate the
    multi-op, non-componentwise `combine_emit`."""

    def lift(x, axis):
        arr = np.asarray(x, dtype=np.float64)
        n = float(arr.shape[axis])
        mean = arr.mean(axis=axis)
        m2 = np.square(arr - np.expand_dims(mean, axis)).sum(axis=axis)
        return (mean, m2, n)

    def merge(a, b):
        mean_a, m2_a, n_a = a
        mean_b, m2_b, n_b = b
        n = n_a + n_b
        delta = mean_b - mean_a
        mean = mean_a + delta * (n_b / n)
        m2 = m2_a + m2_b + np.square(delta) * (n_a * n_b / n)
        return (mean, m2, n)

    def finalize(carrier, eps=0.0):
        _mean, m2, n = carrier
        return 1.0 / np.sqrt(m2 / n + eps)

    return Monoid(
        id="welford",
        is_permutation=False,
        nl_op=None,
        combine_kind="welford",
        lift=lift,
        finalize=finalize,
        merge=merge,
        identity=(0.0, 0.0, 0.0),
    )


def _flash_monoid() -> Monoid:
    """Return the stub for the deferred flash synthesis monoid."""

    def body_synthesis(*_args, **_kwargs):
        raise NotImplementedError(
            "flash body_synthesis: hand lift/finalize sub-bodies to the "
            "synthesis pipeline separately (the second, sharding-induced stage)."
        )

    return Monoid(
        id="flash",
        is_permutation=False,
        nl_op=None,
        combine_kind="flash",
        lift=None,
        finalize=None,
        merge=None,
        identity=None,
        body_synthesis=body_synthesis,
    )


MONOIDS: dict[MonoidId, Monoid] = {
    "+": _scalar_monoid("+", "add", lambda a, b: a + b, 0.0),
    "id": _id_monoid(),
    "max": _scalar_monoid("max", "maximum", max, float("-inf")),
    "min": _scalar_monoid("min", "minimum", min, float("inf")),
    "var_naive": _var_naive_monoid(),
    "welford": _welford_monoid(),
}

# Deferred, synthesis-requiring tier — ships stubbed. Kept out of the default
# search-visible MONOIDS map until populated, but its record pins the shape:
# body_synthesis raises NotImplementedError.
FLASH = _flash_monoid()
