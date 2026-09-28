"""Regressions for the strict directional proof API.

Every e-graph equality producer admits a union or lowering relation only
through ``check_valid_and_equivalent``. These tests pin its properties:
candidate validity is a proof goal rather than an assumption, unknown and
timeout solver results reject, and the reduction-body fallback runs under
the directional validity context.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pytest
import z3

import axon.lang_semantics as lang
from axon.ir import Node, _sym_expr_from_graph_node
from axon.isa_semantics import (
    _FOLD_SWAP_FACTS,
    Context,
    ReductionDesc,
    ReductionKind,
    Semantics,
    ShapeExpr,
    SymExpr,
    SymTensor,
    _check_deterministic,
    _check_reduction_equivalent_by_body,
    _compile_pair_for_equivalence,
    _encode_shape_dim,
    _expr_structural_key,
    _indexed_exprs_equivalent,
    _new_sym_tensor,
    _shape_eq,
    _tensor_function,
    activation,
    activation_reduce,
    check_candidate_validity,
    check_valid_and_equivalent,
    compile_expr,
    fold_family,
    memset,
    nc_matmul,
    nc_transpose,
    nl,
    reciprocal,
    reduction_extensionality_context,
    tensor_copy,
    tensor_reduce,
    tensor_scalar,
    tensor_scalar_cumulative,
    tensor_tensor,
)

assert lang is not None  # imported for its public-semantics registrations


def test_fallback_rejects_negated_reduce_mismatch() -> None:
    m, n = z3.Int("negate_m"), z3.Int("negate_n")
    x = SymTensor("negate_x", shape=(m, n))
    current = tensor_reduce(
        dst=None, op=nl.add, data=x, axis=1, negate=False, keepdims=True
    )
    candidate = tensor_reduce(
        dst=None, op=nl.add, data=x, axis=1, negate=True, keepdims=True
    )

    verdict = check_valid_and_equivalent(current, candidate, timeout=10000)

    assert not verdict.proved
    assert verdict.stage == "value"


def test_fallback_rejects_accumulate_mismatch() -> None:
    k, m, n = z3.Int("acc_k"), z3.Int("acc_m"), z3.Int("acc_n")
    stationary = SymTensor("acc_stationary", shape=(k, m))
    moving = SymTensor("acc_moving", shape=(k, n))
    current = nc_matmul(
        dst=None, stationary=stationary, moving=moving, accumulate=False
    )
    candidate = nc_matmul(
        dst=None, stationary=stationary, moving=moving, accumulate=True
    )

    verdict = check_valid_and_equivalent(current, candidate, timeout=10000)

    assert not verdict.proved
    assert verdict.stage == "value"


def _axis_zero_reduction_semantics(
    name: str,
    body_id: int,
    step_value: Callable[[z3.ArithRef, z3.ArithRef], z3.ArithRef],
    *,
    output_transform: str,
) -> Semantics:
    outer_dim = z3.IntVal(2)
    extent = z3.IntVal(2)
    shape = ShapeExpr([z3.IntVal(1), outer_dim])
    out_fn = z3.Function(f"{name}_out", z3.IntSort(), z3.IntSort(), z3.RealSort())
    fam = fold_family(1, ReductionKind.REDUCE, "add")
    assert fam is not None
    j, k = z3.Ints(f"{name}_j {name}_k")
    ctx = Context(
        [
            z3.ForAll(
                [j, k],
                fam.step(z3.IntVal(body_id), j, k) == step_value(j, k),
            ),
            z3.ForAll(
                [j],
                out_fn(z3.IntVal(0), j) == fam.fold(z3.IntVal(body_id), j, extent),
            ),
        ]
    )
    return Semantics(
        name,
        shape,
        out_fn,
        ctx,
        reduction=ReductionDesc(
            body_id,
            extent,
            outer_rank=fam.outer_arity,
            kind=fam.kind,
            combine_op=fam.combine_op,
            outer_dims=(outer_dim,),
            output_transform=output_transform,
        ),
    )


def test_axis_zero_keepdims_checks_all_columns() -> None:
    lhs = _axis_zero_reduction_semantics(
        "axis_zero_lhs",
        91001,
        lambda _j, _k: z3.RealVal(0),
        output_transform="identity",
    )
    rhs = _axis_zero_reduction_semantics(
        "axis_zero_rhs",
        91002,
        lambda j, _k: z3.If(j == 0, z3.RealVal(0), z3.RealVal(1)),
        output_transform="identity",
    )
    full_ctx = lhs.ctx.merged(rhs.ctx, reduction_extensionality_context())

    assert not _check_reduction_equivalent_by_body(
        lhs, rhs, _shape_eq(lhs.shape, rhs.shape), 10000, full_ctx
    )


def test_fallback_rejects_unsupported_output_transform() -> None:
    lhs = _axis_zero_reduction_semantics(
        "unsupported_lhs",
        92001,
        lambda _j, _k: z3.RealVal(0),
        output_transform="custom",
    )
    rhs = _axis_zero_reduction_semantics(
        "unsupported_rhs",
        92002,
        lambda _j, _k: z3.RealVal(0),
        output_transform="custom",
    )
    full_ctx = lhs.ctx.merged(rhs.ctx, reduction_extensionality_context())

    assert not _check_reduction_equivalent_by_body(
        lhs, rhs, _shape_eq(lhs.shape, rhs.shape), 10000, full_ctx
    )


def _public_add(x: SymTensor, y: SymTensor) -> SymTensor:
    out_shape = tuple(
        z3.If(a == 1, b, a) if not (_int_val(a) or _int_val(b)) else _pick(a, b)
        for a, b in zip(x.shape, y.shape, strict=True)
    )
    return _new_sym_tensor("add", [x, y], {"op": "add"}, out_shape)


def _int_val(d) -> bool:
    return z3.is_int_value(d)


def _pick(a, b):
    if _int_val(a) and a.as_long() == 1:
        return b
    return a


def _public_matmul(x: SymTensor, y: SymTensor) -> SymTensor:
    return _new_sym_tensor(
        "matmul",
        [x, y],
        {"transpose_x": False},
        (x.shape[0], y.shape[1]),
    )


def _relu_mlp_normal_form_graphs() -> tuple[
    SymTensor,
    SymTensor,
    SymTensor,
    SymTensor,
    SymTensor,
]:
    m, k, n, p = z3.Ints("relu_mlp_m relu_mlp_k relu_mlp_n relu_mlp_p")
    x = SymTensor("relu_mlp_x", shape=(m, k))
    w1 = SymTensor("relu_mlp_w1", shape=(k, n))
    w2 = SymTensor("relu_mlp_w2", shape=(k, n))
    w3 = SymTensor("relu_mlp_w3", shape=(n, p))

    public_w1 = _public_matmul(x, w1)
    public_w2 = _public_matmul(x, w2)
    public_relu = _new_sym_tensor("relu", [public_w1], {}, public_w1.shape)
    public_product = _sym_expr_from_graph_node(
        Node(
            id="relu_mlp_public_mul",
            op="mul",
            inputs=[public_relu.id, public_w2.id],
        ),
        [public_relu, public_w2],
    )
    reference = _public_matmul(public_product, w3)

    transposed_x = nc_transpose(dst=None, data=x)
    reversed_w1 = nc_matmul(dst=None, stationary=w1, moving=transposed_x)
    reversed_w2 = nc_matmul(dst=None, stationary=w2, moving=transposed_x)
    reversed_product = tensor_tensor(
        dst=None,
        data1=activation(dst=None, op=nl.relu, data=reversed_w1),
        data2=reversed_w2,
        op=nl.multiply,
    )
    reversed_orientation = nc_matmul(dst=None, stationary=reversed_product, moving=w3)

    canonical_w1 = nc_matmul(
        dst=None,
        stationary=nc_transpose(dst=None, data=x),
        moving=w1,
    )
    canonical_w2 = nc_matmul(
        dst=None,
        stationary=nc_transpose(dst=None, data=x),
        moving=w2,
    )
    canonical_product = tensor_tensor(
        dst=None,
        data1=activation(dst=None, op=nl.relu, data=canonical_w1),
        data2=canonical_w2,
        op=nl.multiply,
    )
    transposed_relu = nc_transpose(
        dst=None,
        data=activation(dst=None, op=nl.relu, data=canonical_w1),
    )
    mixed_product = tensor_tensor(
        dst=None,
        data1=transposed_relu,
        data2=reversed_w2,
        op=nl.multiply,
    )
    mixed_orientation = nc_matmul(dst=None, stationary=mixed_product, moving=w3)
    return (
        reference,
        reversed_orientation,
        mixed_orientation,
        public_product,
        canonical_product,
    )


@pytest.mark.parametrize("candidate_index", [1, 2], ids=["reversed", "mixed"])
def test_indexed_normal_form_proves_pinned_relu_mlp_orientations(
    candidate_index: int,
) -> None:
    graphs = _relu_mlp_normal_form_graphs()
    reference = graphs[0]
    candidate = graphs[candidate_index]

    assert _indexed_exprs_equivalent(reference.expr, candidate.expr)
    verdict = check_valid_and_equivalent(reference, candidate, timeout=10000)
    assert verdict.proved
    assert verdict.used_reduction_fallback


def test_indexed_normal_form_handles_actual_public_mul_broadcast_shape() -> None:
    _, _, _, public_product, canonical_product = _relu_mlp_normal_form_graphs()

    assert all(z3.is_app_of(dim, z3.Z3_OP_ITE) for dim in public_product.shape)
    assert _indexed_exprs_equivalent(public_product.expr, canonical_product.expr)
    verdict = check_valid_and_equivalent(
        public_product, canonical_product, timeout=10000
    )
    assert verdict.proved
    assert verdict.used_reduction_fallback


def test_indexed_normal_form_rejects_mutated_relu_mlp_operand() -> None:
    reference, reversed_orientation, *_ = _relu_mlp_normal_form_graphs()
    product = reversed_orientation.expr.inputs[0]
    mutated_product = SymExpr(
        product.op,
        [product.inputs[0], product.inputs[0]],
        product.shape,
        dict(product.attrs),
        product.name,
    )
    mutated = SymExpr(
        reversed_orientation.expr.op,
        [mutated_product, reversed_orientation.expr.inputs[1]],
        reversed_orientation.expr.shape,
        dict(reversed_orientation.expr.attrs),
        reversed_orientation.expr.name,
    )

    assert not _indexed_exprs_equivalent(reference.expr, mutated)


def test_indexed_normal_form_rejects_unsupported_attributes() -> None:
    reference, reversed_orientation, *_ = _relu_mlp_normal_form_graphs()
    accumulated = SymExpr(
        reversed_orientation.expr.op,
        list(reversed_orientation.expr.inputs),
        reversed_orientation.expr.shape,
        {**reversed_orientation.expr.attrs, "accumulate": True},
        reversed_orientation.expr.name,
    )
    product = reversed_orientation.expr.inputs[0]
    relu = product.inputs[0]
    scaled_relu = SymExpr(
        relu.op,
        list(relu.inputs),
        relu.shape,
        {**relu.attrs, "scale": 2.0},
        relu.name,
    )
    scaled_product = SymExpr(
        product.op,
        [scaled_relu, product.inputs[1]],
        product.shape,
        dict(product.attrs),
        product.name,
    )
    scaled = SymExpr(
        reversed_orientation.expr.op,
        [scaled_product, reversed_orientation.expr.inputs[1]],
        reversed_orientation.expr.shape,
        dict(reversed_orientation.expr.attrs),
        reversed_orientation.expr.name,
    )

    assert not _indexed_exprs_equivalent(reference.expr, accumulated)
    assert not _indexed_exprs_equivalent(reference.expr, scaled)


def test_indexed_normal_form_rejects_activation_without_scale() -> None:
    m, n = z3.Ints("missing_scale_m missing_scale_n")
    x = SymTensor("missing_scale_x", shape=(m, n))
    reference = _new_sym_tensor("relu", [x], {}, x.shape)
    registered_candidate = activation(dst=None, op=nl.relu, data=x)
    omitted_scale_attrs = dict(registered_candidate.expr.attrs)
    omitted_scale_attrs.pop("scale")
    candidate = SymTensor(
        "missing_scale_candidate",
        expr=SymExpr(
            registered_candidate.expr.op,
            list(registered_candidate.expr.inputs),
            registered_candidate.expr.shape,
            omitted_scale_attrs,
            registered_candidate.expr.name,
        ),
    )

    assert not _indexed_exprs_equivalent(reference.expr, candidate.expr)
    verdict = check_valid_and_equivalent(reference, candidate, timeout=10000)
    assert not verdict.proved
    assert verdict.stage == "value"
    assert verdict.detail == "sat"


class TestCandidateValidityIsAGoal:
    def test_structurally_invalid_candidate_is_rejected(self) -> None:
        """A candidate whose validity is unsatisfiable must be rejected.

        ``tensor_scalar_cumulative`` requires rank-2 data. On rank-3 data its
        validity contains ``False``; a check that merged validity into the
        assumptions would treat that as given and prove anything.
        """
        x = SymTensor("x", rank=3)
        current = tensor_copy(dst=None, src=x)
        candidate = tensor_scalar_cumulative(
            dst=None, src=x, op0=nl.multiply, op1=nl.add, imm0=1.0
        )

        verdict = check_valid_and_equivalent(current, candidate, timeout=10000)
        assert not verdict.proved
        assert verdict.stage == "validity"

    def test_unimplied_candidate_precondition_is_rejected(self) -> None:
        """A satisfiable but unproved candidate precondition must reject.

        ``tensor_scalar`` requires a tensor operand's free dimension to be 1.
        With a fully symbolic operand that constraint is satisfiable but not
        implied by the source validity, so the candidate is rejected instead
        of the constraint becoming an assumption.
        """
        a = SymTensor("a", rank=2)
        b = SymTensor("b", shape=(a.shape[0], z3.Int("b_free")))
        current = tensor_tensor(dst=None, data1=a, data2=b, op=nl.add)
        candidate = tensor_scalar(dst=None, data=a, op0=nl.add, operand0=b)

        verdict = check_valid_and_equivalent(current, candidate, timeout=10000)
        assert not verdict.proved
        assert verdict.stage == "validity"

    def test_nested_candidate_precondition_is_rejected(self) -> None:
        """An unproved precondition below the candidate root must still reject.

        ``compile_expr`` accumulates every input's validity into the root's
        validity. Deleting that accumulation would admit this candidate, whose
        only unproved precondition sits one level below the root.
        """
        a = SymTensor("a", rank=2)
        b = SymTensor("b", shape=(a.shape[0], z3.Int("b_free")))
        current = tensor_copy(
            dst=None, src=tensor_tensor(dst=None, data1=a, data2=b, op=nl.add)
        )
        candidate = tensor_copy(
            dst=None, src=tensor_scalar(dst=None, data=a, op0=nl.add, operand0=b)
        )

        verdict = check_valid_and_equivalent(current, candidate, timeout=10000)
        assert not verdict.proved
        assert verdict.stage == "validity"

    def test_implied_candidate_precondition_is_accepted(self) -> None:
        """The same candidate is proved once its precondition is implied."""
        m = z3.Int("m")
        n = z3.Int("n")
        a = SymTensor("a", shape=(m, n))
        b = SymTensor("b", shape=(m, 1))
        current = _public_add(a, b)
        candidate = tensor_scalar(dst=None, data=a, op0=nl.add, operand0=b)

        verdict = check_valid_and_equivalent(current, candidate, timeout=20000)
        assert verdict.proved
        assert verdict.stage == "proved"

    def test_nc_matmul_contraction_mismatch_is_rejected(self) -> None:
        """Operand legality must not be lost to the out_shape shortcut.

        Both moving operands are constant tensors, so the value proof
        succeeds; only the contraction-dimension constraint in the validity
        rule separates the hardware-invalid candidate from the valid one.
        """
        k, m, n = z3.Int("k"), z3.Int("m"), z3.Int("n")
        x = SymTensor("x", shape=(k, m))
        moving_ok = memset(dst=SymTensor("mv_ok", shape=(k, n)), value=3.0)
        moving_bad = memset(dst=SymTensor("mv_bad", shape=(2, n)), value=3.0)
        current = nc_matmul(dst=None, stationary=x, moving=moving_ok)

        candidate = nc_matmul(dst=None, stationary=x, moving=moving_bad)
        verdict = check_valid_and_equivalent(current, candidate, timeout=10000)
        assert not verdict.proved
        assert verdict.stage == "validity"

        control = nc_matmul(
            dst=None,
            stationary=x,
            moving=memset(dst=SymTensor("mv_ok2", shape=(k, n)), value=3.0),
        )
        assert check_valid_and_equivalent(current, control, timeout=20000).proved

    def test_cumulative_full_tile_immediate_is_rejected(self) -> None:
        """A full-tile immediate operand must reject at the validity stage.

        NKI requires tensor immediates to be per-partition vectors. Without
        the constraint, cumsum(x * w) was provable equivalent to a
        tensor_scalar_cumulative whose imm0 is the full tile w.
        """
        m = z3.Int("m")
        x = SymTensor("x", shape=(m, 4))
        w = SymTensor("w", shape=(m, 4))
        product = _new_sym_tensor("mul", [x, w], {"op": "mul"}, (m, 4))
        current = _new_sym_tensor("cumsum", [product], {"axis": -1}, (m, 4))
        candidate = tensor_scalar_cumulative(
            dst=None, src=x, op0=nl.multiply, op1=nl.add, imm0=w
        )

        verdict = check_valid_and_equivalent(current, candidate, timeout=10000)
        assert not verdict.proved
        assert verdict.stage == "validity"


class TestSolverUncertaintyRejects:
    def test_timeout_rejects_candidate(self) -> None:
        """A 1ms budget cannot complete this chained matmul proof.

        The single-matmul pair in the generous-timeout test below is provable
        in a few milliseconds, which would leave too little margin against a
        1ms timeout. Chaining three matmuls makes the value obligation involve
        three fold families and three transpose definitions; the full proof
        takes tens of milliseconds on current hardware, far outside a 1ms
        budget (measured 20 of 20 rejections at timeout=1 on trn2).
        """
        m, k, n, p, q = z3.Int("m"), z3.Int("k"), z3.Int("n"), z3.Int("p"), z3.Int("q")
        x = SymTensor("x", shape=(m, k))
        y = SymTensor("y", shape=(k, n))
        w = SymTensor("w", shape=(n, p))
        u = SymTensor("u", shape=(p, q))
        current = _public_matmul(_public_matmul(_public_matmul(x, y), w), u)
        inner1 = nc_matmul(
            dst=None, stationary=nc_transpose(dst=None, data=x), moving=y
        )
        inner2 = nc_matmul(
            dst=None, stationary=nc_transpose(dst=None, data=inner1), moving=w
        )
        candidate = nc_matmul(
            dst=None, stationary=nc_transpose(dst=None, data=inner2), moving=u
        )

        verdict = check_valid_and_equivalent(current, candidate, timeout=1)
        assert not verdict.proved

    def test_unknown_solver_result_rejects(self) -> None:
        """A solver ``unknown`` result must reject the candidate.

        The candidate's validity goal requires factoring the product of the
        Mersenne primes 2^31-1 and 2^61-1 under nonlinear integer arithmetic.
        z3 cannot decide it and reports ``unknown`` when its budget expires.
        ``check_candidate_validity`` must treat any non-``unsat`` result as a
        rejection, so a mutant accepting ``unknown`` fails this test.
        """
        small_prime = 2**31 - 1
        large_prime = 2**61 - 1
        x, y = z3.Int("factor_x"), z3.Int("factor_y")
        fn = _tensor_function("V_unknown_probe", 1)
        source = Semantics(
            name="unknown_src",
            shape=ShapeExpr([z3.IntVal(1)]),
            fn=fn,
            validity=Context(
                [x > 1, y > 1, x <= y, x * y == small_prime * large_prime]
            ),
        )
        candidate = Semantics(
            name="unknown_cand",
            shape=ShapeExpr([z3.IntVal(1)]),
            fn=fn,
            validity=Context([x == small_prime]),
        )

        # 250ms is enough: z3 grinds to any deadline on this goal and reports
        # unknown; unsat is computationally infeasible and sat is impossible.
        assert not check_candidate_validity(source, candidate, timeout=250)

    def test_generous_timeout_proves_the_same_candidate(self) -> None:
        m, k, n = z3.Int("m"), z3.Int("k"), z3.Int("n")
        x = SymTensor("x", shape=(m, k))
        y = SymTensor("y", shape=(k, n))
        current = _public_matmul(x, y)
        candidate = nc_matmul(
            dst=None, stationary=nc_transpose(dst=None, data=x), moving=y
        )

        verdict = check_valid_and_equivalent(current, candidate, timeout=30000)
        assert verdict.proved

    def test_hard_nonlinear_obligation_returns_within_budget(self) -> None:
        """A pathological nonlinear-real goal must return, not run unboundedly.

        The soft wall-clock timeout is not honored by z3's nonlinear engine
        (nlsat), so a quantifier-free nonlinear-real system like this one was
        observed to grind for minutes and return ``unknown`` only when the wall
        clock finally tripped (measured ~219s at timeout=200 with no rlimit on
        trn2). ``_check_deterministic`` additionally sets a deterministic
        ``rlimit`` derived from the timeout, which the nonlinear engine does
        honor, so the check returns near-instantly. A tiny timeout keeps the
        rlimit tiny, so this test cannot become flaky-slow: it must return a
        verdict (any verdict) well inside the wall-clock bound below.
        """
        xs = [z3.Real(f"nlsat_p{i}") for i in range(7)]
        cons = [
            xs[i] * xs[i] * xs[i] - xs[(i + 1) % len(xs)] == z3.Q(1, 3)
            for i in range(len(xs))
        ]
        cons.append(z3.Sum(*xs) == 0)
        cons.append(z3.Product(*xs) == 1)

        start = time.perf_counter()
        result = _check_deterministic([z3.And(*cons)], timeout=200)
        elapsed = time.perf_counter() - start

        assert result in (z3.sat, z3.unsat, z3.unknown)
        assert elapsed < 10.0, f"check took {elapsed:.1f}s; rlimit not enforced"


class TestReductionFallbackDirectionalValidity:
    def test_scan_equivalence_via_directional_context(self) -> None:
        """The fallback proves scan equivalence without candidate validity.

        The context handed to the fallback contains only the source validity
        and both semantic definitions. The proof must still go through, which
        shows the fallback does not depend on merging the candidate's own
        validity facts.
        """
        m, n = z3.Int("m"), z3.Int("n")
        x = SymTensor("x", shape=(m, n))
        current = _new_sym_tensor("cumsum", [x], {"axis": -1}, (m, n))
        candidate = tensor_scalar_cumulative(
            dst=None, src=x, op0=nl.multiply, op1=nl.add, imm0=1.0
        )

        lsem, rsem = _compile_pair_for_equivalence(current, candidate)
        directional_ctx = lsem.validity.merged(
            lsem.ctx, rsem.ctx, reduction_extensionality_context()
        )
        shape_eq = _shape_eq(lsem.shape, rsem.shape)
        assert _check_reduction_equivalent_by_body(
            lsem, rsem, shape_eq, 20000, directional_ctx
        )

        verdict = check_valid_and_equivalent(current, candidate, timeout=20000)
        assert verdict.proved

    def test_fallback_rejects_invalid_candidate_under_directional_context(
        self,
    ) -> None:
        """The fallback must not resurrect the candidate's own validity.

        The candidate scan carries a SCAN2 reduction descriptor, but its
        operand shapes are incompatible (its validity is unsatisfiable) and
        its step body differs from the current term's. Under the directional
        context, which excludes candidate validity, the fallback must find the
        body counterexample and reject. An implementation that merges the
        candidate's validity back in would see an inconsistent context and
        vacuously accept.
        """
        m = z3.Int("m")
        x = SymTensor("x", shape=(m, 4))
        bad_operand = SymTensor("bad_operand", shape=(m, 3))
        current = _new_sym_tensor("cumsum", [x], {"axis": -1}, (m, 4))
        candidate = tensor_scalar_cumulative(
            dst=None, src=x, op0=nl.multiply, op1=nl.add, imm0=bad_operand
        )

        lsem, rsem = _compile_pair_for_equivalence(current, candidate)
        assert rsem.reduction is not None
        directional_ctx = lsem.validity.merged(
            lsem.ctx, rsem.ctx, reduction_extensionality_context()
        )
        shape_eq = _shape_eq(lsem.shape, rsem.shape)
        assert not _check_reduction_equivalent_by_body(
            lsem, rsem, shape_eq, 10000, directional_ctx
        )

    def test_invalid_scan_candidate_rejected_before_fallback(self) -> None:
        """An invalid candidate never reaches the reduction fallback."""
        m, n = z3.Int("m"), z3.Int("n")
        x = SymTensor("x", shape=(m, n))
        current = tensor_scalar_cumulative(
            dst=None, src=x, op0=nl.multiply, op1=nl.add, imm0=1.0
        )
        # cumsum along axis 0 is not supported: its validity rule adds False.
        candidate = _new_sym_tensor("cumsum", [x], {"axis": 0}, (m, n))

        verdict = check_valid_and_equivalent(current, candidate, timeout=10000)
        assert not verdict.proved
        assert verdict.stage == "validity"


class TestMatmulOperandSwap:
    """``silu_mlp`` emits both matmul operand orders. Each states its step over
    a fresh body id, so only the swap axiom connects the two folds."""

    @staticmethod
    def _silu_mlp_matmul_pair() -> tuple[SymTensor, SymTensor]:
        m, k, n = z3.Int("swap_m"), z3.Int("swap_k"), z3.Int("swap_n")
        x = SymTensor("swap_x", shape=(m, k))
        w1 = SymTensor("swap_w1", shape=(k, n))
        x_t = nc_transpose(dst=None, data=x)
        # The pinned silu_mlp graphs carry this wrapper, and it also blocks the
        # syntactic normal form that discharges a bare matmul pair.
        lhs = activation(
            dst=None,
            op=nl.silu,
            scale=1.0,
            data=nc_matmul(dst=None, stationary=x_t, moving=w1),
        )
        rhs = nc_transpose(
            dst=None,
            data=activation(
                dst=None,
                op=nl.silu,
                scale=1.0,
                data=nc_matmul(dst=None, stationary=w1, moving=x_t),
            ),
        )
        return lhs, rhs

    def test_swapped_matmul_operands_prove_equivalent(self) -> None:
        lhs, rhs = self._silu_mlp_matmul_pair()

        assert check_valid_and_equivalent(lhs, rhs).proved is True


def test_swap_axiom_excludes_scan_family() -> None:
    dumped = [str(fact) for fact in _FOLD_SWAP_FACTS]
    assert len(dumped) == 1
    assert any("REDUCE2" in text for text in dumped)
    assert not any("SCAN2" in text for text in dumped)


class TestProofCacheScopes:
    """Local proofs are reusable universal identities; contextual are not."""

    @staticmethod
    def _tensor_setup():
        from axon.egraph.adapter import EGraphAdapter
        from axon.egraph.analysis import analyze_snapshot, decode_tensor
        from axon.egraph.codec import encode_shape
        from axon.egraph.payload import encode_attrs
        from axon.egraph.proof import ProofStore
        from axon.egraph.tensor_language import t_input, t_op1

        adapter = EGraphAdapter("t")
        x = adapter.intern_expr(t_input("x", encode_shape(("m", "k"))), "x").handle
        y = adapter.intern_expr(t_input("y", encode_shape(("m", "k"))), "y").handle
        scaled = t_op1("mul", encode_attrs({"scalar": 2.0, "reverse": False}), x)
        doubled = adapter.intern_expr(scaled, "x2").handle
        snap = adapter.freeze_snapshot()
        refs = {
            "x": adapter.resolve_handle(snap, x),
            "y": adapter.resolve_handle(snap, y),
            "x2": adapter.resolve_handle(snap, doubled),
        }
        analyses = analyze_snapshot(snap, decode_tensor)
        return snap, refs, analyses, ProofStore(), decode_tensor

    def test_local_identity_is_cached_and_reused(self) -> None:
        from axon.egraph.proof import TermApp, TermRef, prove_candidate_batch

        snap, refs, analyses, store, _decode = self._tensor_setup()
        current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
        candidate = TermApp.make("mul", {}, (TermRef(refs["y"]), TermRef(refs["x"])))
        (first,) = prove_candidate_batch(
            store, "tensor", snap, analyses, [(current, candidate)]
        )
        assert first.proved
        dispatches_after_first = store.dispatch_count
        assert dispatches_after_first > 0

        # The commuted identity over the swapped argument classes has the
        # same canonical key: heads, equality pattern, and guaranteed facts.
        (second,) = prove_candidate_batch(
            store,
            "tensor",
            snap,
            analyses,
            [
                (
                    TermApp.make("mul", {}, (TermRef(refs["y"]), TermRef(refs["x"]))),
                    TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"]))),
                )
            ],
        )
        assert second.proved
        assert store.dispatch_count == dispatches_after_first

    def test_failed_verdict_is_retried_with_a_different_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import axon.egraph.proof as proof
        from axon.egraph.proof import TermApp, TermRef, prove_candidate_batch
        from axon.egraph.proof_parallel import LocalProofResult
        from axon.isa_semantics import EquivalenceVerdict

        calls: list[int] = []

        def run_batch(obligations: Any, **kwargs: Any) -> tuple[Any, ...]:
            values = list(obligations)
            results = []
            for sequence, obligation in enumerate(values):
                calls.append(obligation.timeout)
                proved = obligation.timeout > 100
                result = LocalProofResult(
                    sequence=sequence,
                    proof_key=obligation.proof_key,
                    verdict=EquivalenceVerdict(
                        proved=proved,
                        stage="proved" if proved else "shape",
                        detail="" if proved else "unknown",
                        elapsed_ms=1,
                    ),
                    cache_hit=False,
                    deduplicated=False,
                    timeout=obligation.timeout,
                )
                kwargs["on_result"](result)
                results.append(result)
            return tuple(results)

        monkeypatch.setattr(proof, "run_local_proof_batch", run_batch)
        snap, refs, analyses, store, _decode = self._tensor_setup()
        current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
        candidate = TermApp.make("mul", {}, (TermRef(refs["y"]), TermRef(refs["x"])))

        (first,) = prove_candidate_batch(
            store,
            "tensor",
            snap,
            analyses,
            [(current, candidate)],
            timeout=100,
        )
        (second,) = prove_candidate_batch(
            store,
            "tensor",
            snap,
            analyses,
            [(current, candidate)],
            timeout=200,
        )

        assert not first.proved
        assert second.proved
        # The 100ms rejection is timeout-qualified, so the 200ms run
        # re-dispatches the shape proof and then the value proof.
        assert len(calls) == 3
        assert calls[0] <= 100
        assert all(call > 100 for call in calls[1:])

    def test_deadline_preserves_completed_batch_records_and_effective_cache_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import axon.egraph.proof as proof
        from axon.egraph.proof import (
            TermApp,
            TermRef,
            WallClockExceeded,
            prove_candidate_batch,
        )
        from axon.egraph.proof_parallel import (
            LocalProofResult,
            ProofBatchDeadlineExceeded,
        )
        from axon.isa_semantics import EquivalenceVerdict

        now = [0.0]
        monkeypatch.setattr(proof.time, "monotonic", lambda: now[0])
        deadline = 10.0

        def run_batch(obligations: Any, **_kwargs: object) -> tuple[Any, ...]:
            values = list(obligations)
            completed = tuple(
                LocalProofResult(
                    sequence=index,
                    proof_key=values[index].proof_key,
                    verdict=EquivalenceVerdict(
                        proved=False,
                        stage="shape",
                        detail="unknown",
                    ),
                    cache_hit=False,
                    deduplicated=False,
                    timeout=7,
                )
                for index in range(2)
            )
            now[0] += 20.0
            raise ProofBatchDeadlineExceeded(completed)

        monkeypatch.setattr(proof, "run_local_proof_batch", run_batch)
        snap, refs, analyses, store, _decode = self._tensor_setup()
        current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
        candidates = [
            (
                current,
                TermApp.make(
                    op,
                    {},
                    (TermRef(refs["x"]), TermRef(refs["y"])),
                ),
            )
            for op in ("add", "subtract", "div")
        ]
        completed_indices: list[int] = []

        with pytest.raises(WallClockExceeded, match="tensor: wall_clock_seconds"):
            prove_candidate_batch(
                store,
                "tensor",
                snap,
                analyses,
                candidates,
                timeout=10000,
                workers=2,
                deadline=deadline,
                on_verdict=lambda index, _verdict: completed_indices.append(index),
            )

        assert completed_indices == [0, 1]
        assert any(
            isinstance(key, tuple) and key[-2:] == ("timeout", 7)
            for key in store.local_shape_cache
        )
        assert not any(
            isinstance(key, tuple) and key[-2:] == ("timeout", 10000)
            for key in store.local_shape_cache
        )

    def test_serial_proof_timeout_is_capped_to_wall_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time

        import axon.egraph.proof as proof
        from axon.egraph.proof import TermApp, TermRef, prove_candidate_batch
        from axon.egraph.proof_parallel import LocalProofResult
        from axon.isa_semantics import EquivalenceVerdict

        observed: list[int] = []

        def run_batch(obligations: Any, **kwargs: Any) -> tuple[Any, ...]:
            values = list(obligations)
            results = []
            for sequence, obligation in enumerate(values):
                observed.append(obligation.timeout)
                result = LocalProofResult(
                    sequence=sequence,
                    proof_key=obligation.proof_key,
                    verdict=EquivalenceVerdict(
                        proved=True, stage="proved", elapsed_ms=1
                    ),
                    cache_hit=False,
                    deduplicated=False,
                    timeout=obligation.timeout,
                )
                kwargs["on_result"](result)
                results.append(result)
            return tuple(results)

        monkeypatch.setattr(proof, "run_local_proof_batch", run_batch)
        snap, refs, analyses, store, _decode = self._tensor_setup()
        current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
        candidate = TermApp.make("add", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
        prove_candidate_batch(
            store,
            "tensor",
            snap,
            analyses,
            [(current, candidate)],
            timeout=10000,
            deadline=time.monotonic() + 0.1,
        )
        assert observed
        assert 1 <= observed[0] <= 100

    def test_cache_keys_preserve_repeated_argument_patterns(self) -> None:
        """x - x == 0 * x holds; a - b == 0 * a must not reuse that verdict."""
        from axon.egraph.proof import (
            TermApp,
            TermRef,
            local_proof_key,
            prove_candidate_batch,
        )

        snap, refs, analyses, store, _decode = self._tensor_setup()
        repeated_current = TermApp.make(
            "subtract", {}, (TermRef(refs["x"]), TermRef(refs["x"]))
        )
        repeated_candidate = TermApp.make(
            "mul", {"scalar": 0.0, "reverse": False}, (TermRef(refs["x"]),)
        )
        (verdict,) = prove_candidate_batch(
            store,
            "tensor",
            snap,
            analyses,
            [(repeated_current, repeated_candidate)],
        )
        assert verdict.proved

        distinct_current = TermApp.make(
            "subtract", {}, (TermRef(refs["x"]), TermRef(refs["y"]))
        )
        distinct_candidate = TermApp.make(
            "mul", {"scalar": 0.0, "reverse": False}, (TermRef(refs["x"]),)
        )
        assert local_proof_key(
            analyses, "tensor", repeated_current, repeated_candidate
        ) != local_proof_key(analyses, "tensor", distinct_current, distinct_candidate)
        (distinct,) = prove_candidate_batch(
            store,
            "tensor",
            snap,
            analyses,
            [(distinct_current, distinct_candidate)],
        )
        assert not distinct.proved


def test_batch_candidate_deadlines_exclude_queue_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axon.egraph.proof as proof
    import axon.isa_semantics as isa
    from axon.egraph.proof import TermApp, TermRef, prove_candidate_batch
    from axon.egraph.proof_parallel import LocalProofResult
    from axon.isa_semantics import EquivalenceVerdict

    now = [100.0]
    phase_timeouts: list[dict[Any, int]] = []
    shape_elapsed: dict[Any, int] = {}

    monkeypatch.setattr(proof.time, "monotonic", lambda: now[0])

    def run_batch(obligations: Any, **kwargs: Any) -> tuple[LocalProofResult, ...]:
        values = list(obligations)
        phase_timeouts.append(
            {obligation.proof_key: obligation.timeout for obligation in values}
        )
        is_shape = isinstance(values[0].current, isa._ShapeOnlySymTensor)
        results = []
        for sequence, obligation in enumerate(values):
            elapsed_ms = 10 + sequence * 10 if is_shape else 1
            if is_shape:
                shape_elapsed[obligation.proof_key] = elapsed_ms
            result = LocalProofResult(
                sequence=sequence,
                proof_key=obligation.proof_key,
                verdict=EquivalenceVerdict(
                    proved=True,
                    stage="shape" if is_shape else "proved",
                    elapsed_ms=elapsed_ms,
                ),
                cache_hit=False,
                deduplicated=False,
                timeout=obligation.timeout,
            )
            now[0] += 3600.0
            kwargs["on_result"](result)
            results.append(result)
        return tuple(results)

    monkeypatch.setattr(proof, "run_local_proof_batch", run_batch)
    snap, refs, analyses, store, _decode = TestProofCacheScopes._tensor_setup()
    current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    candidates = [
        (current, TermApp.make(op, {}, (TermRef(refs["x"]), TermRef(refs["y"]))))
        for op in ("add", "subtract")
    ]

    verdicts = prove_candidate_batch(
        store,
        "tensor",
        snap,
        analyses,
        candidates,
        timeout=100,
        workers=2,
    )

    assert len(verdicts) == 2
    assert all(verdict.proved for verdict in verdicts)
    assert list(phase_timeouts[0].values()) == [100, 100]
    value_timeouts = phase_timeouts[1]
    for value_key, value_timeout in value_timeouts.items():
        local_key = value_key
        assert value_timeout == 100 - shape_elapsed[local_key]


def test_batch_proof_term_preparation_reduces_candidate_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axon.egraph.proof as proof
    from axon.egraph.proof import TermApp, TermRef, prove_candidate_batch
    from axon.egraph.proof_parallel import LocalProofResult
    from axon.isa_semantics import EquivalenceVerdict

    now = [0.0]
    phase_timeouts: list[list[int]] = []
    original_local_build = proof.build_local_proof_terms

    monkeypatch.setattr(proof.time, "monotonic", lambda: now[0])

    def build_local(*args: Any, **kwargs: Any):
        terms = original_local_build(*args, **kwargs)
        now[0] += 0.125
        return terms

    def run_batch(obligations: Any, **kwargs: Any) -> tuple[LocalProofResult, ...]:
        values = list(obligations)
        phase_timeouts.append([obligation.timeout for obligation in values])
        results = tuple(
            LocalProofResult(
                sequence=sequence,
                proof_key=obligation.proof_key,
                verdict=EquivalenceVerdict(
                    proved=False,
                    stage="shape",
                    detail="sat",
                    elapsed_ms=100,
                ),
                cache_hit=False,
                deduplicated=False,
                timeout=obligation.timeout,
            )
            for sequence, obligation in enumerate(values)
        )
        for result in results:
            kwargs["on_result"](result)
        return results

    monkeypatch.setattr(proof, "build_local_proof_terms", build_local)
    monkeypatch.setattr(proof, "run_local_proof_batch", run_batch)
    snap, refs, analyses, store, _decode = TestProofCacheScopes._tensor_setup()
    current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    candidates = [
        (current, TermApp.make(op, {}, (TermRef(refs["x"]), TermRef(refs["y"]))))
        for op in ("add", "subtract")
    ]

    verdicts = prove_candidate_batch(
        store,
        "tensor",
        snap,
        analyses,
        candidates,
        timeout=1000,
        workers=2,
    )

    assert len(verdicts) == 2
    assert phase_timeouts == [[875, 875]]


def test_batch_key_preparation_reduces_candidate_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axon.egraph.proof as proof
    from axon.egraph.proof import TermApp, TermRef, prove_candidate_batch
    from axon.egraph.proof_parallel import LocalProofResult
    from axon.isa_semantics import EquivalenceVerdict

    now = [0.0]
    phase_timeouts: list[list[int]] = []
    original_local_key = proof.local_proof_key

    monkeypatch.setattr(proof.time, "monotonic", lambda: now[0])

    def local_key(*args: Any, **kwargs: Any):
        key = original_local_key(*args, **kwargs)
        now[0] += 0.125
        return key

    def run_batch(obligations: Any, **kwargs: Any) -> tuple[LocalProofResult, ...]:
        values = list(obligations)
        phase_timeouts.append([obligation.timeout for obligation in values])
        results = tuple(
            LocalProofResult(
                sequence=sequence,
                proof_key=obligation.proof_key,
                verdict=EquivalenceVerdict(
                    proved=False,
                    stage="shape",
                    detail="sat",
                    elapsed_ms=100,
                ),
                cache_hit=False,
                deduplicated=False,
                timeout=obligation.timeout,
            )
            for sequence, obligation in enumerate(values)
        )
        for result in results:
            kwargs["on_result"](result)
        return results

    monkeypatch.setattr(proof, "local_proof_key", local_key)
    monkeypatch.setattr(proof, "run_local_proof_batch", run_batch)
    snap, refs, analyses, store, _decode = TestProofCacheScopes._tensor_setup()
    current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    candidates = [
        (current, TermApp.make(op, {}, (TermRef(refs["x"]), TermRef(refs["y"]))))
        for op in ("add", "subtract")
    ]

    verdicts = prove_candidate_batch(
        store,
        "tensor",
        snap,
        analyses,
        candidates,
        timeout=1000,
        workers=2,
    )

    assert len(verdicts) == 2
    assert phase_timeouts == [[875, 875]]


def test_duplicate_local_key_preparation_is_not_cumulative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axon.egraph.proof as proof
    from axon.egraph.proof import TermApp, TermRef, prove_candidate_batch
    from axon.egraph.proof_parallel import LocalProofResult
    from axon.isa_semantics import EquivalenceVerdict

    now = [0.0]
    dispatched_timeouts: list[list[int]] = []
    original_local_key = proof.local_proof_key

    monkeypatch.setattr(proof.time, "monotonic", lambda: now[0])

    def local_key(*args: Any, **kwargs: Any):
        key = original_local_key(*args, **kwargs)
        now[0] += 0.125
        return key

    def run_batch(obligations: Any, **kwargs: Any) -> tuple[LocalProofResult, ...]:
        values = list(obligations)
        dispatched_timeouts.append([obligation.timeout for obligation in values])
        result = LocalProofResult(
            sequence=0,
            proof_key=values[0].proof_key,
            verdict=EquivalenceVerdict(proved=True, stage="proved"),
            cache_hit=False,
            deduplicated=False,
            timeout=values[0].timeout,
        )
        kwargs["on_result"](result)
        return (result,)

    monkeypatch.setattr(proof, "local_proof_key", local_key)
    monkeypatch.setattr(proof, "run_local_proof_batch", run_batch)
    snap, refs, analyses, store, _decode = TestProofCacheScopes._tensor_setup()
    current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    candidate = TermApp.make("add", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    key = original_local_key(analyses, "tensor", current, candidate)
    store.local_shape_cache[("shape", key)] = EquivalenceVerdict(
        proved=True,
        stage="shape",
    )

    verdicts = prove_candidate_batch(
        store,
        "tensor",
        snap,
        analyses,
        [(current, candidate), (current, candidate)],
        timeout=1000,
        workers=2,
    )

    assert len(verdicts) == 2
    assert all(verdict.proved for verdict in verdicts)
    assert dispatched_timeouts == [[875]]
    assert store.dispatch_count == 1


@pytest.mark.parametrize(
    ("preparation_name", "expected_dispatches"),
    [
        ("local_proof_key", 0),
        ("build_local_proof_terms", 0),
    ],
)
def test_wall_deadline_checkpoint_cancels_recursive_preparation(
    monkeypatch: pytest.MonkeyPatch,
    preparation_name: str,
    expected_dispatches: int,
) -> None:
    import axon.egraph.proof as proof
    from axon.egraph.proof import (
        TermApp,
        TermRef,
        WallClockExceeded,
        prove_candidate_batch,
    )
    from axon.egraph.proof_parallel import LocalProofResult
    from axon.isa_semantics import EquivalenceVerdict

    now = [0.0]
    in_preparation = [False]
    preparation_completed = [False]
    dispatched: list[Any] = []
    original_preparation = getattr(proof, preparation_name)

    def monotonic() -> float:
        if in_preparation[0]:
            now[0] += 0.04
        return now[0]

    def preparation(*args: Any, **kwargs: Any):
        in_preparation[0] = True
        try:
            result = original_preparation(*args, **kwargs)
        finally:
            in_preparation[0] = False
        preparation_completed[0] = True
        return result

    def run_batch(obligations: Any, **kwargs: Any) -> tuple[LocalProofResult, ...]:
        values = list(obligations)
        dispatched.extend(obligation.proof_key for obligation in values)
        results = tuple(
            LocalProofResult(
                sequence=sequence,
                proof_key=obligation.proof_key,
                verdict=EquivalenceVerdict(
                    proved=False,
                    stage="shape",
                    detail="sat",
                ),
                cache_hit=False,
                deduplicated=False,
                timeout=obligation.timeout,
            )
            for sequence, obligation in enumerate(values)
        )
        for result in results:
            kwargs["on_result"](result)
        return results

    monkeypatch.setattr(proof.time, "monotonic", monotonic)
    monkeypatch.setattr(proof, preparation_name, preparation)
    monkeypatch.setattr(proof, "run_local_proof_batch", run_batch)
    snap, refs, analyses, store, _decode = TestProofCacheScopes._tensor_setup()
    current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    candidate = TermApp.make("add", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    deadline = now[0] + 0.1

    with pytest.raises(WallClockExceeded, match="tensor: wall_clock_seconds"):
        prove_candidate_batch(
            store,
            "tensor",
            snap,
            analyses,
            [(current, candidate)],
            timeout=1000,
            workers=2,
            deadline=deadline,
        )

    assert not preparation_completed[0]
    assert len(dispatched) == expected_dispatches
    assert store.dispatch_count == expected_dispatches


def test_batch_retry_after_deadline_stops_with_wall_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axon.egraph.proof as proof
    from axon.egraph.proof import (
        TermApp,
        TermRef,
        WallClockExceeded,
        prove_candidate_batch,
    )

    now = [0.0]
    monkeypatch.setattr(proof.time, "monotonic", lambda: now[0])
    batch_sizes: list[int] = []

    def run_batch(obligations: Any, **_kwargs: Any) -> tuple[Any, ...]:
        batch_sizes.append(len(list(obligations)))
        now[0] += 20.0
        raise z3.Z3Exception("batch candidate failure")

    monkeypatch.setattr(proof, "run_local_proof_batch", run_batch)
    snap, refs, analyses, store, _decode = TestProofCacheScopes._tensor_setup()
    current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    candidates = [
        (current, TermApp.make(op, {}, (TermRef(refs["x"]), TermRef(refs["y"]))))
        for op in ("add", "subtract")
    ]

    with pytest.raises(WallClockExceeded, match="tensor: wall_clock_seconds"):
        prove_candidate_batch(
            store,
            "tensor",
            snap,
            analyses,
            candidates,
            workers=2,
            deadline=10.0,
        )

    assert batch_sizes == [2]
    assert store.dispatch_count == 2


def test_failed_local_batch_elapsed_reduces_retry_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axon.egraph.proof as proof
    from axon.egraph.proof import TermApp, TermRef, prove_candidate_batch

    now = [0.0]
    retry_timeouts: list[int] = []

    monkeypatch.setattr(proof.time, "monotonic", lambda: now[0])

    def run_batch(obligations: Any, **_kwargs: Any) -> tuple[Any, ...]:
        values = list(obligations)
        if len(values) > 1:
            now[0] += 0.03
            raise z3.Z3Exception("delayed batch failure")
        retry_timeouts.append(values[0].timeout)
        now[0] += 0.02
        raise z3.Z3Exception("isolated candidate failure")

    monkeypatch.setattr(proof, "run_local_proof_batch", run_batch)
    snap, refs, analyses, store, _decode = TestProofCacheScopes._tensor_setup()
    current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    candidates = [
        (current, TermApp.make(op, {}, (TermRef(refs["x"]), TermRef(refs["y"]))))
        for op in ("add", "subtract")
    ]

    verdicts = prove_candidate_batch(
        store,
        "tensor",
        snap,
        analyses,
        candidates,
        timeout=100,
        workers=2,
    )

    assert retry_timeouts == [70, 70]
    assert all(verdict.stage == "candidate_error" for verdict in verdicts)
    assert store.dispatch_count == 4


def test_exhausted_deadline_does_not_lookup_full_timeout_negative_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axon.egraph.proof as proof
    from axon.egraph.proof import TermApp, TermRef, prove_candidate_batch
    from axon.egraph.proof_parallel import LocalProofResult
    from axon.isa_semantics import EquivalenceVerdict

    def run_batch(obligations: Any, **kwargs: Any) -> tuple[LocalProofResult, ...]:
        results = tuple(
            LocalProofResult(
                sequence=sequence,
                proof_key=obligation.proof_key,
                verdict=EquivalenceVerdict(
                    proved=True,
                    stage="shape",
                    elapsed_ms=100,
                ),
                cache_hit=False,
                deduplicated=False,
                timeout=obligation.timeout,
            )
            for sequence, obligation in enumerate(obligations)
        )
        for result in results:
            kwargs["on_result"](result)
        return results

    monkeypatch.setattr(proof, "run_local_proof_batch", run_batch)
    snap, refs, analyses, store, _decode = TestProofCacheScopes._tensor_setup()
    current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    candidates = [
        (current, TermApp.make(op, {}, (TermRef(refs["x"]), TermRef(refs["y"]))))
        for op in ("add", "subtract")
    ]

    verdicts = prove_candidate_batch(
        store,
        "tensor",
        snap,
        analyses,
        candidates,
        timeout=100,
        workers=2,
    )

    assert len(verdicts) == 2
    assert all(verdict.stage == "value" for verdict in verdicts)
    assert store.value_dispatch_count == 0


def test_shortened_timeout_result_uses_effective_cache_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axon.egraph.proof as proof
    from axon.egraph.proof import TermApp, TermRef, prove_candidate_batch
    from axon.egraph.proof_parallel import LocalProofResult
    from axon.isa_semantics import EquivalenceVerdict

    call_count = 0

    def run_batch(obligations: Any, **kwargs: Any) -> tuple[LocalProofResult, ...]:
        nonlocal call_count
        values = list(obligations)
        results = []
        for sequence, value in enumerate(values):
            proved = call_count > 0
            verdict = EquivalenceVerdict(
                proved=proved,
                stage="proved" if call_count > 1 else "shape",
                detail="" if proved else "unknown",
                elapsed_ms=1,
            )
            result = LocalProofResult(
                sequence=sequence,
                proof_key=value.proof_key,
                verdict=verdict,
                cache_hit=False,
                deduplicated=False,
                timeout=7 if call_count == 0 else value.timeout,
            )
            kwargs["on_result"](result)
            results.append(result)
        call_count += 1
        return tuple(results)

    monkeypatch.setattr(proof, "run_local_proof_batch", run_batch)
    snap, refs, analyses, store, _decode = TestProofCacheScopes._tensor_setup()
    current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    candidates = [
        (current, TermApp.make(op, {}, (TermRef(refs["x"]), TermRef(refs["y"]))))
        for op in ("add", "subtract")
    ]

    first = prove_candidate_batch(
        store,
        "tensor",
        snap,
        analyses,
        candidates,
        timeout=10000,
        workers=2,
    )
    assert not any(verdict.proved for verdict in first)
    # The rejection is cached under its effective 7ms timeout, not the
    # requested 10000ms, so a full-timeout rerun re-dispatches and proves.
    assert any(
        isinstance(key, tuple) and key[-2:] == ("timeout", 7)
        for key in store.local_shape_cache
    )
    assert not any(
        isinstance(key, tuple) and key[-2:] == ("timeout", 10000)
        for key in store.local_shape_cache
    )

    second = prove_candidate_batch(
        store,
        "tensor",
        snap,
        analyses,
        candidates,
        timeout=10000,
        workers=2,
    )
    assert all(verdict.proved for verdict in second)


def test_candidate_semantic_error_rejects_only_that_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axon.egraph.proof as proof
    from axon.egraph.proof import ProofError, TermApp, TermRef, prove_candidate_batch
    from axon.egraph.proof_parallel import LocalProofResult
    from axon.isa_semantics import EquivalenceVerdict

    original_build = proof.build_local_proof_terms

    def build_terms(*args: Any, **kwargs: Any):
        candidate = args[3]
        if candidate.op == "subtract":
            raise ProofError("candidate cannot be compiled")
        return original_build(*args, **kwargs)

    def run_batch(obligations: Any, **kwargs: Any) -> tuple[LocalProofResult, ...]:
        values = list(obligations)
        results = tuple(
            LocalProofResult(
                sequence=sequence,
                proof_key=value.proof_key,
                verdict=EquivalenceVerdict(proved=True, stage="proved", detail=""),
                cache_hit=False,
                deduplicated=False,
                timeout=value.timeout,
            )
            for sequence, value in enumerate(values)
        )
        for result in results:
            kwargs["on_result"](result)
        return results

    monkeypatch.setattr(proof, "build_local_proof_terms", build_terms)
    monkeypatch.setattr(proof, "run_local_proof_batch", run_batch)
    snap, refs, analyses, store, _decode = TestProofCacheScopes._tensor_setup()
    current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    candidates = [
        (
            current,
            TermApp.make("subtract", {}, (TermRef(refs["x"]), TermRef(refs["y"]))),
        ),
        (current, TermApp.make("add", {}, (TermRef(refs["x"]), TermRef(refs["y"])))),
    ]

    verdicts = prove_candidate_batch(
        store,
        "tensor",
        snap,
        analyses,
        candidates,
        workers=2,
    )

    assert [verdict.proved for verdict in verdicts] == [False, True]
    assert verdicts[0].stage == "candidate_error"
    assert "ProofError" in verdicts[0].detail

    bad_key = proof.local_proof_key(
        analyses, "tensor", candidates[0][0], candidates[0][1]
    )

    def solver_error_batch(
        obligations: Any, **kwargs: Any
    ) -> tuple[LocalProofResult, ...]:
        values = list(obligations)
        if len(values) > 1 or values[0].proof_key == bad_key:
            raise z3.Z3Exception("candidate solver failure")
        result = LocalProofResult(
            sequence=0,
            proof_key=values[0].proof_key,
            verdict=EquivalenceVerdict(proved=True, stage="proved", detail=""),
            cache_hit=False,
            deduplicated=False,
            timeout=values[0].timeout,
        )
        kwargs["on_result"](result)
        return (result,)

    monkeypatch.setattr(proof, "build_local_proof_terms", original_build)
    monkeypatch.setattr(proof, "run_local_proof_batch", solver_error_batch)
    solver_verdicts = prove_candidate_batch(
        proof.ProofStore(),
        "tensor",
        snap,
        analyses,
        candidates,
        workers=2,
    )

    assert [verdict.proved for verdict in solver_verdicts] == [False, True]
    assert "Z3Exception" in solver_verdicts[0].detail


def test_infrastructure_errors_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axon.egraph.proof as proof
    from axon.egraph.proof import TermApp, TermRef, prove_candidate_batch

    snap, refs, analyses, store, _decode = TestProofCacheScopes._tensor_setup()
    current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    candidates = [
        (current, TermApp.make(op, {}, (TermRef(refs["x"]), TermRef(refs["y"]))))
        for op in ("add", "subtract")
    ]

    def infrastructure_failure(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
        raise RuntimeError("worker serialization failed")

    monkeypatch.setattr(proof, "run_local_proof_batch", infrastructure_failure)
    with pytest.raises(RuntimeError, match="worker serialization failed"):
        prove_candidate_batch(
            proof.ProofStore(),
            "tensor",
            snap,
            analyses,
            candidates,
            workers=2,
        )


def test_shape_mismatch_does_not_compile_value_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axon.egraph.proof as proof
    import axon.isa_semantics as isa
    from axon.egraph.proof import TermApp, TermRef, prove_candidate_batch
    from axon.isa_semantics import EquivalenceVerdict

    m, n = z3.Ints("shape_only_m shape_only_n")
    x = SymTensor("shape_only_x", shape=(m, n))
    y = SymTensor("shape_only_y", shape=(m, 1))
    semantic_value_compilations = 0

    def compile_semantic_values(*_args: Any, **_kwargs: Any):
        nonlocal semantic_value_compilations
        semantic_value_compilations += 1
        raise AssertionError("shape mismatch compiled semantic values")

    monkeypatch.setattr(isa, "_compile_pair_for_equivalence", compile_semantic_values)
    semantic_verdict = isa.check_valid_and_equivalent(
        tensor_copy(dst=None, src=x),
        tensor_copy(dst=None, src=y),
        timeout=1000,
    )
    assert not semantic_verdict.proved
    assert semantic_verdict.stage == "shape"
    assert semantic_value_compilations == 0
    direct_value_verdict = isa._check_value_equivalent_after_shape(x, y, timeout=1000)
    assert not direct_value_verdict.proved
    assert direct_value_verdict.stage == "shape"

    snap, refs, analyses, store, _decode = TestProofCacheScopes._tensor_setup()
    current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    candidate = TermApp.make("add", {}, (TermRef(refs["x"]), TermRef(refs["y"])))

    def reject_shape_batch(obligations: Any, **kwargs: Any) -> tuple[Any, ...]:
        from axon.egraph.proof_parallel import LocalProofResult

        values = list(obligations)
        results = []
        for sequence, obligation in enumerate(values):
            result = LocalProofResult(
                sequence=sequence,
                proof_key=obligation.proof_key,
                verdict=EquivalenceVerdict(
                    proved=False, stage="shape", detail="sat", elapsed_ms=1
                ),
                cache_hit=False,
                deduplicated=False,
                timeout=obligation.timeout,
            )
            kwargs["on_result"](result)
            results.append(result)
        return tuple(results)

    monkeypatch.setattr(proof, "run_local_proof_batch", reject_shape_batch)

    (verdict,) = prove_candidate_batch(
        store, "tensor", snap, analyses, [(current, candidate)]
    )

    assert not verdict.proved
    assert verdict.stage == "shape"
    assert store.value_dispatch_count == 0


def test_cached_value_rejection_is_not_promoted_by_shape_proof() -> None:
    import axon.egraph.proof as proof
    from axon.egraph.proof import TermApp, TermRef, prove_candidate_batch
    from axon.isa_semantics import EquivalenceVerdict

    snap, refs, analyses, store, _decode = TestProofCacheScopes._tensor_setup()
    current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    candidate = TermApp.make("add", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    local_key = proof.local_proof_key(
        analyses,
        "tensor",
        current,
        candidate,
    )
    store.local_shape_cache[("shape", local_key)] = EquivalenceVerdict(
        proved=True,
        stage="shape",
    )
    store.local_cache[(local_key, "timeout", 100)] = EquivalenceVerdict(
        proved=False,
        stage="value",
        detail="unknown",
    )

    (verdict,) = prove_candidate_batch(
        store,
        "tensor",
        snap,
        analyses,
        [(current, candidate)],
        timeout=100,
    )

    assert not verdict.proved
    assert verdict.stage == "value"
    assert verdict.detail == "unknown"
    assert store.dispatch_count == 0


@pytest.mark.parametrize("late_stage", ["fallback", "abstract"])
def test_late_value_success_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    late_stage: str,
) -> None:
    import axon.isa_semantics as isa
    from axon.isa_semantics import EquivalenceVerdict, SymTensor, _with_proved_shape

    current, candidate = _with_proved_shape(
        SymTensor("late_value", shape=(4, 4)),
        SymTensor("late_value", shape=(4, 4)),
        "same-shape-proof",
    )
    now = [0.0]
    monkeypatch.setattr(isa.time, "monotonic", lambda: now[0])

    def fallback(*_args: Any, **_kwargs: Any) -> bool:
        if late_stage == "fallback":
            now[0] = 0.2
            return True
        return False

    def abstract(*_args: Any, **_kwargs: Any) -> EquivalenceVerdict:
        now[0] = 0.2
        return EquivalenceVerdict(proved=True, stage="proved")

    monkeypatch.setattr(
        isa,
        "_check_reduction_equivalent_by_body",
        fallback,
    )
    monkeypatch.setattr(
        isa,
        "_check_value_equivalent_after_shape",
        abstract,
    )

    verdict = isa.check_value_with_early_reduction_fallback(
        current,
        candidate,
        timeout=100,
        preconditions=(),
    )

    assert not verdict.proved
    assert verdict.stage == "value"
    assert verdict.detail == "candidate deadline exhausted"
    assert verdict.elapsed_ms == 200


def test_shape_cache_separates_guaranteed_facts() -> None:
    from dataclasses import replace

    import axon.egraph.proof as proof
    from axon.egraph.proof import TermApp, TermRef

    snap, refs, analyses, _store, _decode = TestProofCacheScopes._tensor_setup()
    del snap
    current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    candidate = TermApp.make("add", {}, (TermRef(refs["x"]), TermRef(refs["y"])))
    fact_analyses = dict(analyses)
    fact_analyses[refs["x"]] = replace(
        analyses[refs["x"]],
        facts=(*analyses[refs["x"]].facts, analyses[refs["x"]].dims[0] == 8),
    )

    local_a = ("shape", proof.local_proof_key(analyses, "tensor", current, candidate))
    local_b = (
        "shape",
        proof.local_proof_key(fact_analyses, "tensor", current, candidate),
    )

    assert local_a != local_b


def test_negative_shape_cache_is_timeout_qualified() -> None:
    import axon.egraph.proof as proof
    from axon.isa_semantics import EquivalenceVerdict

    cache: dict[Any, EquivalenceVerdict] = {}
    key = ("shape", "candidate")
    verdict = EquivalenceVerdict(proved=False, stage="shape", detail="unknown")

    proof._cache_verdict(cache, key, 7, verdict)

    assert cache == {(key, "timeout", 7): verdict}
    assert proof._cached_verdict(cache, key, 7) is verdict
    assert proof._cached_verdict(cache, key, 8) is None


def test_candidate_aggregate_deadline_bounds_all_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axon.egraph.proof as proof
    from axon.egraph.proof import TermApp, TermRef, prove_candidate_batch
    from axon.egraph.proof_parallel import LocalProofResult
    from axon.isa_semantics import EquivalenceVerdict

    snap, refs, analyses, store, _decode = TestProofCacheScopes._tensor_setup()
    current = TermApp.make("mul", {}, (TermRef(refs["x"]), TermRef(refs["x2"])))
    candidate = TermApp.make("add", {}, (TermRef(refs["x"]), TermRef(refs["x2"])))
    now = [100.0]
    observed: list[int] = []
    calls = 0

    monkeypatch.setattr(proof.time, "monotonic", lambda: now[0])

    def run_batch(obligations: Any, **kwargs: Any) -> tuple[Any, ...]:
        nonlocal calls
        values = list(obligations)
        results = []
        for sequence, obligation in enumerate(values):
            observed.append(obligation.timeout)
            now[0] += 0.03
            calls += 1
            proved = calls == 1
            result = LocalProofResult(
                sequence=sequence,
                proof_key=obligation.proof_key,
                verdict=EquivalenceVerdict(
                    proved=proved,
                    stage="shape" if proved else "value",
                    detail="" if proved else "sat",
                    elapsed_ms=30,
                ),
                cache_hit=False,
                deduplicated=False,
                timeout=obligation.timeout,
            )
            kwargs["on_result"](result)
            results.append(result)
        return tuple(results)

    monkeypatch.setattr(proof, "run_local_proof_batch", run_batch)

    (verdict,) = prove_candidate_batch(
        store,
        "tensor",
        snap,
        analyses,
        [(current, candidate)],
        timeout=100,
    )

    assert not verdict.proved
    # The candidate aggregate deadline shrinks the value-phase timeout by
    # the shape phase's elapsed time.
    assert observed == [100, 70]


@pytest.mark.parametrize("abstract_result", [z3.sat, z3.unknown])
def test_reduction_fallback_remains_available_after_abstract_sat_or_unknown(
    monkeypatch: pytest.MonkeyPatch,
    abstract_result: z3.CheckSatResult,
) -> None:
    import axon.isa_semantics as isa

    m, n = z3.Ints("fallback_phase_m fallback_phase_n")
    x = SymTensor("fallback_phase_x", shape=(m, n))
    current = tensor_copy(dst=None, src=x)
    candidate = tensor_copy(dst=None, src=x)
    additivity_marker = z3.Bool("fallback_additivity_marker")
    fallback_calls = 0

    monkeypatch.setattr(
        isa,
        "reduction_additivity_context",
        lambda: Context([additivity_marker]),
    )

    def abstract_check(assertions: list[z3.BoolRef], _timeout: int):
        assert any(
            additivity_marker.sexpr() in assertion.sexpr() for assertion in assertions
        )
        return abstract_result

    def fallback(
        _lhs: Semantics,
        _rhs: Semantics,
        _shape_eq: z3.BoolRef,
        _timeout: int,
        fallback_ctx: Context,
        *,
        shape_proved: bool,
    ) -> bool:
        nonlocal fallback_calls
        fallback_calls += 1
        assert shape_proved
        assert all(
            additivity_marker.sexpr() not in fact.sexpr() for fact in fallback_ctx.facts
        )
        return True

    monkeypatch.setattr(isa, "_check_deterministic", abstract_check)
    monkeypatch.setattr(isa, "_check_reduction_equivalent_by_body", fallback)

    proved_current, proved_candidate = isa._with_proved_shape(
        current, candidate, "fallback-test"
    )
    verdict = isa._check_value_equivalent_after_shape(
        proved_current, proved_candidate, timeout=1000
    )

    assert verdict.proved
    assert verdict.used_reduction_fallback
    assert fallback_calls == 1


class TestVerdictShape:
    def test_rank_mismatch_reports_rank_stage(self) -> None:
        x = SymTensor("x", rank=2)
        current = tensor_copy(dst=None, src=x)
        candidate = _new_sym_tensor(
            "reduce_sum", [x], {"axis": 1, "keep_dims": False}, (x.shape[0],)
        )

        verdict = check_valid_and_equivalent(current, candidate, timeout=10000)
        assert not verdict.proved
        assert verdict.stage == "rank"

    def test_rank1_reduce_negative_axis_returns_verdict(self) -> None:
        """A rank-1 reduce_sum with a negative axis must not crash.

        reduce_sum's shape rule accepts any rank, so the compile rule must
        guard its rank-2 fold construction instead of indexing out of bounds
        after axis normalization. The verdict is a rejection because the
        value is opaque, but it must be a verdict, not a Z3Exception.
        """
        n = z3.Int("n")
        x = SymTensor("x", shape=(n,))
        current = tensor_copy(dst=None, src=x)
        candidate = _new_sym_tensor(
            "reduce_sum", [x], {"axis": -1, "keep_dims": True}, (z3.IntVal(1),)
        )

        verdict = check_valid_and_equivalent(current, candidate, timeout=5000)
        assert not verdict.proved

    def test_identity_is_proved(self) -> None:
        x = SymTensor("x", rank=2)
        current = tensor_copy(dst=None, src=x)
        candidate = tensor_copy(dst=None, src=x)

        verdict = check_valid_and_equivalent(current, candidate, timeout=10000)
        assert verdict.proved
        assert bool(verdict) is True


class TestReciprocalActivationEquivalence:
    """The reciprocal arm makes the Scalar activation leg an equal alternative.

    These are the crux guards for the nonlinear reciprocal obligation, which
    historically flapped between unsat and unknown. Each verdict is asserted
    stable across repeated runs so a flaky arm cannot silently drop its variant.
    """

    @staticmethod
    def _shaped_pair(name_a: str, name_b: str) -> tuple[SymTensor, SymTensor]:
        m, n = z3.Int("m"), z3.Int("n")
        return SymTensor(name_a, shape=(m, n)), SymTensor(name_b, shape=(m, n))

    @staticmethod
    def _stable_verdict(current: SymTensor, candidate: SymTensor, timeout: int) -> bool:
        verdicts = [
            check_valid_and_equivalent(current, candidate, timeout=timeout).proved
            for _ in range(3)
        ]
        assert len(set(verdicts)) == 1, verdicts
        return verdicts[0]

    def test_divide_equals_multiply_by_activation_reciprocal(self) -> None:
        num, den = self._shaped_pair("num", "den")
        current = tensor_tensor(dst=None, data1=num, data2=den, op=nl.divide)
        candidate = tensor_tensor(
            dst=None,
            data1=num,
            data2=activation(dst=None, op=nl.reciprocal, data=den),
            op=nl.multiply,
        )

        assert self._stable_verdict(current, candidate, timeout=30000)

    def test_reciprocal_op_equals_activation_reciprocal(self) -> None:
        x = SymTensor("x", rank=2)
        current = reciprocal(dst=None, data=x)
        candidate = activation(dst=None, op=nl.reciprocal, data=x)

        assert self._stable_verdict(current, candidate, timeout=20000)

    def test_activation_square_does_not_prove_reciprocal(self) -> None:
        x = SymTensor("x", rank=2)
        current = reciprocal(dst=None, data=x)
        candidate = activation(dst=None, op=nl.square, data=x)

        assert not self._stable_verdict(current, candidate, timeout=20000)


class TestActivationReduceSemantics:
    """activation_reduce's e-class value is the per-partition reduce result."""

    def test_square_add_compiles_to_reduced_shape_with_reduction(self) -> None:
        m, n = z3.Int("m"), z3.Int("n")
        h = SymTensor("h", shape=(m, n))
        node = activation_reduce(
            dst=None, op=nl.square, data=h, reduce_op=nl.add, reduce_res=True
        )
        sem = compile_expr(node.expr, {})
        assert sem.shape.rank == 2
        assert z3.is_true(z3.simplify(sem.shape.dims[1] == 1))
        assert sem.reduction is not None
        assert sem.reduction.combine_op == "add"


class TestSumOfSquaresActivationReduceEquivalence:
    """activation_reduce(square, add) fuses mul(h, h) + reduce(add) on Scalar.

    The reduction obligation runs through the same fold-family machinery as
    the reciprocal crux, so each verdict is asserted stable across repeated
    runs to guard against a flaky variant drop at emit time.
    """

    @staticmethod
    def _stable_verdict(current: SymTensor, candidate: SymTensor, timeout: int) -> bool:
        verdicts = [
            check_valid_and_equivalent(current, candidate, timeout=timeout).proved
            for _ in range(3)
        ]
        assert len(set(verdicts)) == 1, verdicts
        return verdicts[0]

    def test_sum_of_squares_equals_activation_reduce_square(self) -> None:
        m, n = z3.Int("m"), z3.Int("n")
        h = SymTensor("h", shape=(m, n))
        current = tensor_reduce(
            dst=None,
            op=nl.add,
            data=tensor_tensor(dst=None, data1=h, data2=h, op=nl.multiply),
            axis=1,
            keepdims=True,
        )
        candidate = activation_reduce(
            dst=None, op=nl.square, data=h, reduce_op=nl.add, reduce_res=True
        )

        assert self._stable_verdict(current, candidate, timeout=30000)

    def test_relu_activation_reduce_does_not_prove_sum_of_squares(self) -> None:
        m, n = z3.Int("m"), z3.Int("n")
        h = SymTensor("h", shape=(m, n))
        current = tensor_reduce(
            dst=None,
            op=nl.add,
            data=tensor_tensor(dst=None, data1=h, data2=h, op=nl.multiply),
            axis=1,
            keepdims=True,
        )
        candidate = activation_reduce(
            dst=None, op=nl.relu, data=h, reduce_op=nl.add, reduce_res=True
        )

        assert not self._stable_verdict(current, candidate, timeout=30000)


class TestExprStructuralKey:
    """The structural key backs the equivalence rename and verdict caching.
    A key COLLISION between structurally distinct obligations would let a
    cached verdict be reused for a different obligation (a soundness bug), so
    these tests pin that the cheap dim encoding preserves exactly the equality
    the old ``str``-based key induced.
    """

    def test_dim_encoding_agrees_with_str_equivalence(self) -> None:
        """``_encode_shape_dim`` must induce the same equality on dims as the
        old ``str(dim)`` key: encode(a) == encode(b) iff str(a) == str(b)."""
        m, n = z3.Int("m"), z3.Int("n")
        dims = [
            z3.IntVal(1),
            z3.IntVal(1),
            z3.IntVal(8192),
            m,
            z3.Int("m"),  # same declaration name as m -> str-equal
            n,
            z3.If(m == 1, n, m),
            z3.If(m == 1, n, m),  # structurally identical compound
            z3.If(n == 1, m, n),  # different compound, same "if" decl name
            m + n,
            n + m,  # str-distinct: "m + n" vs "n + m"
        ]
        for a in dims:
            for b in dims:
                assert (_encode_shape_dim(a) == _encode_shape_dim(b)) == (
                    str(a) == str(b)
                ), (str(a), str(b))

    def test_compound_dim_not_collapsed_by_decl_name(self) -> None:
        """A naive ``decl().name()`` encoding would collide two different
        compound dims that share an operator symbol. The encoded-child tuple
        must keep them distinct."""
        m, n, p = z3.Int("m"), z3.Int("n"), z3.Int("p")
        # Both are "if" nodes: decl().name() == "if" for each, but operands
        # differ. Faithful encoding must distinguish them.
        d1 = z3.If(m == 1, n, m)
        d2 = z3.If(p == 1, n, p)
        assert d1.decl().name() == d2.decl().name()
        assert _encode_shape_dim(d1) != _encode_shape_dim(d2)
        # Two identical products must still encode equal.
        assert _encode_shape_dim(m * n) == _encode_shape_dim(m * n)
        # A concrete int and a same-printing symbolic const stay distinct.
        assert _encode_shape_dim(z3.IntVal(1)) != _encode_shape_dim(z3.Int("d"))

    def test_identical_exprs_share_key_distinct_exprs_do_not(self) -> None:
        """Structurally identical exprs get equal keys; differences in op,
        attrs, shape (incl. concrete-vs-symbolic), or children get distinct
        keys."""
        m, n = z3.Int("m"), z3.Int("n")
        a = SymTensor("a", shape=(m, n))
        b = SymTensor("b", shape=(m, n))

        e1 = tensor_tensor(dst=None, data1=a, data2=b, op=nl.add)
        e2 = tensor_tensor(dst=None, data1=a, data2=b, op=nl.add)
        assert _expr_structural_key(e1.expr) == _expr_structural_key(e2.expr)

        # Different op.
        e_mul = tensor_tensor(dst=None, data1=a, data2=b, op=nl.multiply)
        assert _expr_structural_key(e1.expr) != _expr_structural_key(e_mul.expr)

        # Different children (swap an input).
        c = SymTensor("c", shape=(m, n))
        e_swap = tensor_tensor(dst=None, data1=a, data2=c, op=nl.add)
        assert _expr_structural_key(e1.expr) != _expr_structural_key(e_swap.expr)

        # Concrete-vs-symbolic shape must not share a key.
        in_sym = SymExpr("input", [], (m,), {}, "x")
        in_concrete = SymExpr("input", [], (z3.IntVal(1),), {}, "x")
        assert _expr_structural_key(in_sym) != _expr_structural_key(in_concrete)

        # Different attrs.
        attr1 = SymExpr("op", [], (m,), {"axis": 0}, "z")
        attr2 = SymExpr("op", [], (m,), {"axis": 1}, "z")
        assert _expr_structural_key(attr1) != _expr_structural_key(attr2)
