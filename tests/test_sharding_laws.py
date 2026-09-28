"""Offline monoid-law discharge.

Edge legality is a static predicate (`sharding.combine`); the genuinely
verification-worthy obligation is that each monoid's `⊕` is actually
associative + commutative with identity `e` (and, for stats carriers, that the
sufficient statistics form a correct homomorphism). That obligation is a
**per-monoid lemma, proven once when the monoid enters the library**, never
per-edge, never in the search loop.

This obligation IS a test obligation ("offline, once per monoid, before this
code ships"), so it lives here under `tests/`, not in `src/axon/`. The scalar
monoids (`+`/`max`/`min`) are discharged with Z3, reusing Axon's solver
(`_Z3_LOCK`); this is the literal "we machine-check the combiner" claim. The
carrier monoids (`var_naive`/`welford`) carry float arithmetic Z3 cannot model in
closed form, so their laws are discharged by a seeded numeric oracle:
associativity/commutativity of the merge to tight tolerance, identity behavior,
and the sufficient-statistic homomorphism (the parallel merge of two slices
equals the single-pass reduce of their concatenation).
"""

from __future__ import annotations

import numpy as np
import pytest

from axon.sharding_monoids import MONOIDS, Monoid


@pytest.mark.parametrize("mid", ["+", "max", "min"])
def test_scalar_monoid_laws(mid: str):
    """Z3-prove assoc + comm + identity for a scalar monoid (`+`/`max`/`min`)."""
    import z3

    from axon.isa_semantics import _Z3_LOCK

    op = {
        "+": lambda a, b: a + b,
        "max": lambda a, b: z3.If(a >= b, a, b),
        "min": lambda a, b: z3.If(a <= b, a, b),
    }[mid]
    with _Z3_LOCK:
        a, b, c = z3.Reals("a b c")
        s = z3.Solver()
        # Associativity and commutativity must hold for ALL a,b,c: refute the
        # negation.
        s.push()
        s.add(z3.Or(op(op(a, b), c) != op(a, op(b, c)), op(a, b) != op(b, a)))
        assert s.check() == z3.unsat, f"{mid}: assoc/comm not proven"
        s.pop()
        # Identity: refute `e ⊕ a ≠ a`. Only `+` has a finite Z3-expressible
        # identity (0); the ±inf identities of max/min aren't Z3 Reals, so the
        # numeric oracle covers their identity instead.
        if mid == "+":
            s.push()
            s.add(op(z3.RealVal(0), a) != a)
            assert s.check() == z3.unsat, "+ identity not proven"
            s.pop()


@pytest.mark.parametrize("mid", ["var_naive", "welford"])
def test_carrier_monoid_laws(mid: str, seed: int = 7):
    """Numerically discharge merge assoc/comm and the sufficient-statistic
    homomorphism for a carrier monoid (`var_naive`/`welford`)."""
    monoid: Monoid = MONOIDS[mid]
    lift, merge, finalize = monoid.lift, monoid.merge, monoid.finalize
    assert lift is not None and merge is not None and finalize is not None, (
        f"{mid}: carrier monoid must define lift/merge/finalize"
    )
    rng = np.random.default_rng(seed)
    axis = 1
    # Three random slices of the same row-shape.
    rows, w = 4, 6
    xs = [rng.standard_normal((rows, w)) for _ in range(3)]
    ca, cb, cc = (lift(x, axis) for x in xs)

    def _close(p, q):
        return all(
            np.allclose(np.asarray(pi), np.asarray(qi))
            for pi, qi in zip(p, q, strict=True)
        )

    assert _close(merge(merge(ca, cb), cc), merge(ca, merge(cb, cc))), (
        f"{mid}: merge not associative"
    )
    assert _close(merge(ca, cb), merge(cb, ca)), f"{mid}: merge not commutative"

    # Sufficient-statistic homomorphism: merging two slices' carriers equals
    # reducing their concatenation in one pass.
    whole = np.concatenate([xs[0], xs[1]], axis=axis)
    merged = merge(lift(xs[0], axis), lift(xs[1], axis))
    direct = lift(whole, axis)
    assert np.allclose(finalize(merged), finalize(direct)), (
        f"{mid}: sufficient-statistic homomorphism violated"
    )
