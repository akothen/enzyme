"""Focused tests for e-class analysis normalization."""

from __future__ import annotations

import z3

from axon.egraph.analysis import EClassAnalysis


def test_nested_broadcast_dimensions_collapse() -> None:
    m = z3.Int("m")
    broadcast_once = z3.If(m == 1, m, m)
    broadcast_twice = z3.If(
        broadcast_once == 1,
        broadcast_once,
        broadcast_once,
    )

    analysis = EClassAnalysis((broadcast_twice,), ())

    assert z3.eq(analysis.dims[0], m)


def test_equivalent_normalized_facts_are_deduplicated_in_order() -> None:
    m = z3.Int("m")
    positive = m > 0
    bounded = m < 128

    analysis = EClassAnalysis(
        (),
        (
            z3.And(z3.BoolVal(True), positive),
            bounded,
            positive,
            z3.And(bounded, z3.BoolVal(True)),
        ),
    )

    assert [fact.sexpr() for fact in analysis.facts] == [
        z3.simplify(positive).sexpr(),
        z3.simplify(bounded).sexpr(),
    ]


def test_fact_deduplication_preserves_same_name_different_sort_formulas() -> None:
    first_sort = z3.DeclareSort("First")
    second_sort = z3.DeclareSort("Second")
    first_equality = z3.Const("x", first_sort) == z3.Const("y", first_sort)
    second_equality = z3.Const("x", second_sort) == z3.Const("y", second_sort)

    assert first_equality.sexpr() == second_equality.sexpr()

    analysis = EClassAnalysis((), (first_equality, second_equality))

    assert len(analysis.facts) == 2
    assert z3.eq(analysis.facts[0], first_equality)
    assert z3.eq(analysis.facts[1], second_equality)


def test_facts_simplified_to_true_are_removed() -> None:
    m = z3.Int("m")

    analysis = EClassAnalysis((), (z3.BoolVal(True), m == m))

    assert analysis.facts == ()


def test_fact_simplified_to_false_is_retained() -> None:
    m = z3.Int("m")

    analysis = EClassAnalysis((), (m != m,))

    assert len(analysis.facts) == 1
    assert z3.is_false(analysis.facts[0])
