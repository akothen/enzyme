"""Lower a frozen tensor e-graph into an ISA e-graph."""

from __future__ import annotations

import heapq
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import z3

from axon.egraph.adapter import (
    EClassRef,
    EGraphAdapter,
    Snapshot,
)
from axon.egraph.analysis import (
    EClassAnalysis,
    analyze_snapshot,
    decode_tensor,
    ensure_isa_semantics_registered,
)
from axon.egraph.codec import (
    DecodedIsaENode,
    DecodedTensorENode,
    decode_isa_enode,
    decode_tensor_enode,
    encode_isa_enode,
    encode_isa_input,
)
from axon.egraph.proof import (
    ProofStore,
    Term,
    TermApp,
    TermRef,
    local_proof_key,
    prove_candidate_batch,
)
from axon.egraph.proof_parallel import (
    LocalProofObligation,
    run_local_proof_batch,
)
from axon.egraph.workers import resolve_worker_count
from axon.ir import _sym_expr_from_graph_node
from axon.isa_semantics import (
    _NODE_IDS,
    EquivalenceVerdict,
    SymTensor,
    _expr_structural_key,
)
from axon.synthesizer import (
    SYM_EVAL_REJECTED,
    SketchNode,
    _iter_complete_recipe_cache_entry,
    _iter_rebound_recipes,
    _make_lowering_cache_key,
    isa_symbolic_eval,
    sketch_materialized_op,
    sketch_node_attrs,
)

# The ISA operations lowering inserts verbatim instead of synthesizing.
ISA_PASSTHROUGH_OPS = frozenset({"broadcast", "broadcast_to", "store", "load"})


class LoweringError(RuntimeError):
    """Raised when the lowering driver hits an unrecoverable inconsistency."""


def prove_candidate(
    store: ProofStore,
    stage: str,
    snapshot: Snapshot,
    analyses: dict[EClassRef, Any],
    current: Term,
    candidate: Term,
    *,
    timeout: int,
) -> EquivalenceVerdict:
    """Run one lowering proof through the batch-capable proof API."""
    verdicts = prove_candidate_batch(
        store,
        stage,
        snapshot,
        analyses,
        [(current, candidate)],
        timeout=timeout,
        workers=1,
    )
    if len(verdicts) != 1:
        raise LoweringError("single lowering proof returned no verdict")
    return verdicts[0]


# Synthesis target


def make_synthesis_target(
    decoded: DecodedTensorENode,
    child_analyses: dict[EClassRef, EClassAnalysis],
) -> tuple[SymTensor, dict[EClassRef, SymTensor]]:
    """Build the symbolic synthesis target for one tensor e-node."""
    if decoded.op == "input":
        raise LoweringError("make_synthesis_target does not apply to input e-nodes")
    formal_syms: dict[EClassRef, SymTensor] = {}
    for child in decoded.child_classes:
        if child in formal_syms:
            continue
        analysis = child_analyses.get(child)
        if analysis is None:
            raise LoweringError(f"child e-class {child!r} has no analysis")
        formal_syms[child] = SymTensor(
            _NODE_IDS.next_name("formal"), shape=tuple(analysis.dims)
        )
    ordered_inputs = [formal_syms[child] for child in decoded.child_classes]
    node = _target_node(decoded, ordered_inputs)
    target_sym = _sym_expr_from_graph_node(node, ordered_inputs)
    return target_sym, formal_syms


def _target_node(decoded: DecodedTensorENode, inputs: list[SymTensor]) -> Any:
    from axon.ir import Node

    return Node(
        id=_NODE_IDS.next_name(f"target_{decoded.op}"),
        op=decoded.op,
        inputs=[sym.id for sym in inputs],
        attrs=dict(decoded.attrs),
    )


def _target_term(decoded: DecodedTensorENode) -> TermApp:
    """The proof term for ``TARGET(t)`` over its child e-classes."""
    return TermApp.make(
        decoded.op,
        dict(decoded.attrs),
        tuple(TermRef(child) for child in decoded.child_classes),
    )


def _sketch_to_term(sketch: SketchNode, formal_to_class: dict[int, EClassRef]) -> Term:
    """Convert a proved sketch to a proof term over child e-classes."""
    if sketch.op == "INPUT":
        if sketch.sym is None:
            raise LoweringError("sketch INPUT leaf has no symbol")
        cls = formal_to_class.get(id(sketch.sym))
        if cls is None:
            raise LoweringError("sketch references an unknown formal argument")
        return TermRef(cls)
    child_terms = tuple(_sketch_to_term(c, formal_to_class) for c in sketch.children)
    attrs = sketch_node_attrs(sketch.op, len(child_terms), sketch.attrs)
    return TermApp.make(sketch_materialized_op(sketch.op), attrs, child_terms)


def _direct_passthrough_verdict(
    store: ProofStore,
    analyses: dict[EClassRef, EClassAnalysis],
    current: Term,
    candidate: Term,
    current_sym: SymTensor,
    candidate_sym: SymTensor,
) -> EquivalenceVerdict | None:
    """Admit only exact registered-codec identities under the normal proof key."""
    if current != candidate:
        return None
    if _expr_structural_key(current_sym.expr) != _expr_structural_key(
        candidate_sym.expr
    ):
        return None
    key = local_proof_key(analyses, "lowering", current, candidate)
    verdict = store.local_cache.get(key)
    if verdict is None or not verdict.proved:
        verdict = EquivalenceVerdict(
            proved=True,
            stage="proved",
            detail="exact registered passthrough codec identity",
        )
        store.local_cache[key] = verdict
    return verdict


# ISA e-graph seeding, program insertion, and admission


def _identity_value(value: Any) -> Any:
    if isinstance(value, dict):
        return (
            "dict",
            tuple((key, _identity_value(item)) for key, item in sorted(value.items())),
        )
    if isinstance(value, (list, tuple)):
        return (
            type(value).__name__,
            tuple(_identity_value(item) for item in value),
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return (type(value).__name__, value)
    qualname = getattr(value, "__qualname__", None)
    if isinstance(qualname, str):
        return ("callable", getattr(value, "__module__", None), qualname)
    try:
        hash(value)
    except TypeError:
        return (type(value).__module__, type(value).__qualname__, repr(value))
    return (type(value).__module__, type(value).__qualname__, value)


def _input_identity(
    decoded: DecodedTensorENode | DecodedIsaENode,
) -> tuple[Any, ...]:
    operand_ids: dict[EClassRef, int] = {}
    operand_pattern = tuple(
        operand_ids.setdefault(child, len(operand_ids))
        for child in decoded.child_classes
    )
    return (
        decoded.op,
        decoded.source_id,
        decoded.input_shape,
        tuple(
            (key, _identity_value(value))
            for key, value in sorted(decoded.attrs.items())
        ),
        operand_pattern,
    )


def _decode_registered_input(expr: Any) -> tuple[DecodedIsaENode, SymTensor]:
    codec_adapter = EGraphAdapter("input_codec")
    result = codec_adapter.intern_expr(expr, provenance="roundtrip")
    snapshot = codec_adapter.freeze_snapshot()
    root = codec_adapter.resolve_handle(snapshot, result.handle)
    decoded_cache: dict[EClassRef, DecodedIsaENode] = {}
    sym_cache: dict[EClassRef, SymTensor] = {}

    def expand(ref: EClassRef) -> SymTensor:
        if ref in sym_cache:
            return sym_cache[ref]
        decoded_rows: list[DecodedIsaENode] = []
        for row in snapshot.members(ref):
            try:
                decoded_rows.append(decode_isa_enode(snapshot, row))
            except Exception:
                continue
        if len(decoded_rows) != 1:
            raise LoweringError(
                "registered ISA input codec produced "
                f"{len(decoded_rows)} decodable rows for {ref!r}"
            )
        decoded = decoded_rows[0]
        decoded_cache[ref] = decoded
        if decoded.op == "input":
            sym = _input_sym(decoded)
        else:
            child_syms = [expand(child) for child in decoded.child_classes]
            node = _target_node(
                DecodedTensorENode(
                    op=decoded.op,
                    child_classes=decoded.child_classes,
                    attrs=decoded.attrs,
                ),
                child_syms,
            )
            sym = _sym_expr_from_graph_node(node, child_syms)
        sym_cache[ref] = sym
        return sym

    candidate_sym = expand(root)
    return decoded_cache[root], candidate_sym


def _mismatched_input_sym(
    decoded: DecodedTensorENode,
    error: Exception,
) -> SymTensor:
    return SymTensor(
        _NODE_IDS.next_name(f"lower_input_mismatch_{type(error).__name__}"),
        shape=tuple(
            z3.IntVal(dim) if isinstance(dim, int) else z3.Int(dim)
            for dim in decoded.input_shape or ()
        ),
    )


def _input_sym(decoded: DecodedTensorENode | DecodedIsaENode) -> SymTensor:
    source_id = decoded.source_id or _NODE_IDS.next_name("lower_input_mismatch")
    shape = decoded.input_shape or ()
    return SymTensor(
        source_id,
        shape=tuple(
            z3.IntVal(dim) if isinstance(dim, int) else z3.Int(dim) for dim in shape
        ),
    )


def seed_isa_inputs(
    isa_adapter: EGraphAdapter,
    tensor_snapshot: Snapshot,
    L: dict[EClassRef, Any],
    proof_store: ProofStore,
    timeout: int = 10000,
) -> None:
    """Seed one proved ISA input for each tensor input class."""
    for ref in tensor_snapshot.classes:
        if ref.sort != "TensorExpr":
            continue
        decoded = _decode_input(tensor_snapshot, ref)
        if decoded is None:
            continue
        assert decoded.source_id is not None and decoded.input_shape is not None
        expr = encode_isa_input(decoded.source_id, decoded.input_shape)
        current_identity = _input_identity(decoded)
        try:
            candidate_decoded, candidate_sym = _decode_registered_input(expr)
            candidate_identity: Any = _input_identity(candidate_decoded)
        except Exception as exc:
            candidate_sym = _mismatched_input_sym(decoded, exc)
            candidate_identity = (
                "codec_error",
                type(exc).__name__,
                str(exc),
            )

        direct_identity = candidate_identity == current_identity
        if direct_identity:
            verdict = EquivalenceVerdict(
                proved=True,
                stage="proved",
                detail="exact registered input codec identity",
            )
        else:
            obligation = LocalProofObligation(
                proof_key=(
                    "lowering_input",
                    current_identity,
                    candidate_identity,
                ),
                current=_input_sym(decoded),
                candidate=candidate_sym,
                timeout=timeout,
            )
            try:
                (result,) = run_local_proof_batch(
                    [obligation],
                    workers=1,
                )
            except z3.Z3Exception as exc:
                verdict = EquivalenceVerdict(
                    proved=False,
                    stage="candidate_error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            else:
                verdict = result.verdict
            proof_store.dispatch_count += 1
        if not verdict.proved:
            continue
        result = isa_adapter.intern_expr(expr, provenance=f"input_{decoded.source_id}")
        L[ref] = result.handle


def _decode_input(snapshot: Snapshot, ref: EClassRef) -> DecodedTensorENode | None:
    for row in snapshot.members(ref):
        try:
            decoded = decode_tensor_enode(snapshot, row)
        except Exception:
            continue
        if decoded.op == "input":
            return decoded
    return None


def add_program(
    isa_adapter: EGraphAdapter,
    sketch: SketchNode,
    L: dict[EClassRef, Any],
    formal_to_class: dict[int, EClassRef],
) -> Any:
    """Encode a proved sketch in the ISA e-graph and return its root."""
    if sketch.op == "INPUT":
        if sketch.sym is None:
            raise LoweringError("sketch INPUT leaf has no symbol")
        cls = formal_to_class.get(id(sketch.sym))
        if cls is None or cls not in L:
            raise LoweringError(
                "add_program leaf references an unrealized formal argument"
            )
        return L[cls]
    child_handles = [
        add_program(isa_adapter, child, L, formal_to_class) for child in sketch.children
    ]
    attrs = sketch_node_attrs(sketch.op, len(child_handles), sketch.attrs)
    expr = encode_isa_enode(sketch_materialized_op(sketch.op), attrs, child_handles)
    return isa_adapter.intern_expr(expr, provenance="lower").handle


def admit_lowering(
    isa_adapter: EGraphAdapter,
    sketch: SketchNode,
    tensor_class: EClassRef,
    L: dict[EClassRef, Any],
    formal_to_class: dict[int, EClassRef],
) -> Any:
    """Add a proved sketch to the realization of ``tensor_class``."""
    root = add_program(isa_adapter, sketch, L, formal_to_class)
    if tensor_class in L:
        isa_adapter.union_if_distinct(root, L[tensor_class])
    else:
        L[tensor_class] = root
    return root


# Driver


@dataclass
class LoweringStatus:
    """The result and counters from ``lower_tensor_egraph``."""

    status: str
    unrealized_outputs: tuple[EClassRef, ...] = ()
    attempted: int = 0
    realized: int = 0
    proved_programs: int = 0
    reason: str | None = None
    stage: str = "lowering"


@dataclass
class _ENodeTask:
    index: int
    owner: EClassRef
    decoded: DecodedTensorENode
    pending: set[EClassRef] = field(default_factory=set)
    continuation: _TaskContinuation | None = None


@dataclass
class _RecipeCacheEntry:
    recipes: list[tuple[Any, ...]]
    source: Iterator[tuple[Any, ...]]
    complete: bool = False

    def replay(self) -> Iterator[tuple[Any, ...]]:
        index = 0
        while True:
            if index < len(self.recipes):
                recipe = self.recipes[index]
            elif self.complete:
                return
            else:
                try:
                    recipe = next(self.source)
                except StopIteration:
                    self.complete = True
                    return
                self.recipes.append(recipe)
            index += 1
            yield recipe


@dataclass
class _TaskContinuation:
    target_sym: SymTensor
    formal_to_class: dict[int, EClassRef]
    target_term: TermApp
    candidates: Iterator[tuple[SketchNode, SymTensor | None]]
    deferred: deque[_CandidateWork] = field(default_factory=deque)
    wave: int = 0
    exhausted: bool = False

    def take_wave(self, size: int) -> list[_CandidateWork]:
        wave: list[_CandidateWork] = []
        while len(wave) < size:
            if self.deferred:
                wave.append(self.deferred.popleft())
                continue
            try:
                sketch, candidate_sym = next(self.candidates)
                wave.append(_CandidateWork(sketch, candidate_sym))
            except StopIteration:
                self.exhausted = True
                break
        return wave

    def defer_front(self, candidates: list[_CandidateWork]) -> None:
        self.deferred.extendleft(reversed(candidates))


@dataclass
class _CandidateWork:
    sketch: SketchNode
    candidate_sym: SymTensor | None
    term: Term | None = None


@dataclass
class _OrderedProofCompletions:
    """Buffer proof verdicts to preserve candidate admission order."""

    on_proved: Callable[[int, EquivalenceVerdict], None]
    completed: dict[int, EquivalenceVerdict] = field(default_factory=dict)
    next_index: int = 0

    def record(self, index: int, verdict: EquivalenceVerdict) -> None:
        self.completed[index] = verdict
        self.drain()

    def drain(self) -> None:
        while self.next_index in self.completed:
            verdict = self.completed.pop(self.next_index)
            if verdict.proved:
                self.on_proved(self.next_index, verdict)
            self.next_index += 1


_LOWERING_CANDIDATE_WAVE_SIZE = 8
_RecipeCache = dict[tuple[Any, ...], tuple[Any, ...] | _RecipeCacheEntry]

# Each proof uses the smaller of the caller timeout and this stage limit.
LOWERING_TIMEOUT_MS = 300


def lower_tensor_egraph(
    tensor_snapshot: Snapshot,
    tensor_outputs: list[EClassRef],
    isa_adapter: EGraphAdapter,
    proof_store: ProofStore,
    d: int,
    *,
    timeout: int = 10000,
    lowering_timeout: int = LOWERING_TIMEOUT_MS,
    workers: int | None = None,
    deadline: float | None = None,
    stop_when_outputs_realized: bool = False,
    recipe_cache: _RecipeCache | None = None,
) -> tuple[EGraphAdapter, dict[EClassRef, Any], LoweringStatus]:
    """Lowers tensor e-nodes into proved ISA realizations.
    ``stop_when_outputs_realized`` stops at the first full realization of every
    declared output, which is the cheapest extractable prefix."""
    ensure_isa_semantics_registered()
    workers = resolve_worker_count(workers)
    if lowering_timeout <= 0:
        raise ValueError("lowering proof timeout must be positive")
    timeout = min(timeout, lowering_timeout)
    tensor_classes = [
        ref for ref in tensor_snapshot.classes if ref.sort == "TensorExpr"
    ]
    L: dict[EClassRef, Any] = {}

    analyses = analyze_snapshot(tensor_snapshot, decode_tensor, classes=tensor_classes)

    seed_isa_inputs(
        isa_adapter,
        tensor_snapshot,
        L,
        proof_store,
        timeout,
    )

    # Build every non-input e-node task, its
    # pending child sets, and the reverse index from child class to consumers.
    tasks: list[_ENodeTask] = []
    reverse: dict[EClassRef, list[int]] = {}
    for ref in tensor_classes:
        for row in tensor_snapshot.members(ref):
            try:
                decoded = decode_tensor_enode(tensor_snapshot, row)
            except Exception:
                continue
            if decoded.op == "input":
                continue
            index = len(tasks)
            pending = {c for c in decoded.child_classes if c not in L}
            task = _ENodeTask(
                index=index, owner=ref, decoded=decoded, pending=set(pending)
            )
            tasks.append(task)
            for child in set(decoded.child_classes):
                reverse.setdefault(child, []).append(index)

    output_dependencies = set(tensor_outputs)
    dependency_work = list(tensor_outputs)
    tasks_by_owner: dict[EClassRef, list[_ENodeTask]] = {}
    for task in tasks:
        tasks_by_owner.setdefault(task.owner, []).append(task)
    while dependency_work:
        owner = dependency_work.pop()
        for task in tasks_by_owner.get(owner, ()):
            for child in task.decoded.child_classes:
                if child not in output_dependencies:
                    output_dependencies.add(child)
                    dependency_work.append(child)

    ready: list[tuple[int, int, int]] = []
    queued: set[int] = set()
    attempted: set[int] = set()
    proved_programs = 0
    if recipe_cache is None:
        recipe_cache = {}

    def schedule(index: int) -> None:
        if index in queued:
            return
        task = tasks[index]
        wave = task.continuation.wave if task.continuation is not None else 0
        heapq.heappush(
            ready,
            (
                wave,
                0 if task.owner in output_dependencies else 1,
                index,
            ),
        )
        queued.add(index)

    for task in tasks:
        if not task.pending:
            schedule(task.index)

    def realize(cls: EClassRef) -> None:
        # A class becoming realized for the first time activates the consumers
        # waiting only on it. Guarded so this runs at most once per class.
        for consumer_index in reverse.get(cls, ()):
            task = tasks[consumer_index]
            task.pending.discard(cls)
            if not task.pending and consumer_index not in queued:
                schedule(consumer_index)

    def prepare(task: _ENodeTask) -> _TaskContinuation:
        target_sym, formal_syms = make_synthesis_target(task.decoded, analyses)
        formal_to_class = {id(sym): cls for cls, sym in formal_syms.items()}
        continuation = _TaskContinuation(
            target_sym=target_sym,
            formal_to_class=formal_to_class,
            target_term=_target_term(task.decoded),
            candidates=_candidate_sketches_for(
                task,
                target_sym,
                formal_syms,
                d,
                recipe_cache=recipe_cache,
            ),
        )
        task.continuation = continuation
        return continuation

    def admit(
        task: _ENodeTask,
        continuation: _TaskContinuation,
        sketch: SketchNode,
    ) -> None:
        nonlocal proved_programs
        was_realized = task.owner in L
        proved_programs += 1
        admit_lowering(
            isa_adapter,
            sketch,
            task.owner,
            L,
            continuation.formal_to_class,
        )
        if not was_realized and task.owner in L:
            realize(task.owner)

    def prepare_one(
        task: _ENodeTask,
        continuation: _TaskContinuation,
        work: _CandidateWork,
    ) -> Term:
        if work.term is not None:
            return work.term
        candidate_term = _sketch_to_term(
            work.sketch,
            continuation.formal_to_class,
        )
        work.term = candidate_term
        return candidate_term

    def prove_one(
        task: _ENodeTask,
        continuation: _TaskContinuation,
        candidate_term: Term,
        candidate_sym: SymTensor | None,
    ) -> EquivalenceVerdict:
        if task.decoded.op in ISA_PASSTHROUGH_OPS and candidate_sym is not None:
            direct = _direct_passthrough_verdict(
                proof_store,
                analyses,
                continuation.target_term,
                candidate_term,
                continuation.target_sym,
                candidate_sym,
            )
            if direct is not None:
                return direct
        return prove_candidate(
            proof_store,
            "lowering",
            tensor_snapshot,
            analyses,
            continuation.target_term,
            candidate_term,
            timeout=timeout,
        )

    def run_serial_wave(
        task: _ENodeTask,
        continuation: _TaskContinuation,
        wave: list[_CandidateWork],
        terms: list[Term],
        owner_was_realized: bool,
    ) -> None:
        """Prove and admit one wave in candidate order."""
        for candidate_index, (work, candidate_term) in enumerate(
            zip(wave, terms, strict=True)
        ):
            verdict = prove_one(
                task,
                continuation,
                candidate_term,
                work.candidate_sym,
            )
            if verdict.proved:
                admit(task, continuation, work.sketch)
                if not owner_was_realized:
                    continuation.defer_front(wave[candidate_index + 1 :])
                    break

    def run_parallel_wave(
        task: _ENodeTask,
        continuation: _TaskContinuation,
        wave: list[_CandidateWork],
        terms: list[Term],
        owner_was_realized: bool,
    ) -> None:
        """Prove one wave as a batch and admit results in candidate order."""
        first_realization_index: int | None = None

        def admit_wave_verdict(verdict_index: int, verdict: EquivalenceVerdict) -> None:
            nonlocal first_realization_index
            if first_realization_index is not None:
                return
            admit(task, continuation, wave[verdict_index].sketch)
            if not owner_was_realized and verdict.proved and task.owner in L:
                first_realization_index = verdict_index

        admissions = _OrderedProofCompletions(admit_wave_verdict)

        wave_verdicts = prove_candidate_batch(
            proof_store,
            "lowering",
            tensor_snapshot,
            analyses,
            [(continuation.target_term, candidate_term) for candidate_term in terms],
            timeout=timeout,
            workers=workers,
            on_verdict=admissions.record,
        )
        for verdict_index, verdict in enumerate(wave_verdicts):
            if verdict_index >= admissions.next_index:
                admissions.completed.setdefault(verdict_index, verdict)
        admissions.drain()
        if first_realization_index is not None:
            continuation.defer_front(wave[first_realization_index + 1 :])

    wall_lapsed = False
    while ready:
        if deadline is not None and time.monotonic() >= deadline:
            # One direct wall check per wave, matching the baseline stop.
            wall_lapsed = True
            break
        _, _, index = heapq.heappop(ready)
        queued.discard(index)
        attempted.add(index)
        task = tasks[index]
        continuation = task.continuation or prepare(task)
        wave = continuation.take_wave(_LOWERING_CANDIDATE_WAVE_SIZE)
        if not wave:
            continue
        owner_was_realized = task.owner in L
        terms = [prepare_one(task, continuation, work) for work in wave]

        if workers <= 1 or len(wave) == 1:
            run_serial_wave(task, continuation, wave, terms, owner_was_realized)
        else:
            run_parallel_wave(task, continuation, wave, terms, owner_was_realized)

        continuation.wave += 1
        if stop_when_outputs_realized and all(out in L for out in tensor_outputs):
            break
        if not continuation.exhausted or continuation.deferred:
            schedule(index)

    unrealized = tuple(out for out in tensor_outputs if out not in L)
    # A lapsed wall that still realized every declared output lowered fully; the
    # wall only truncated the optional alternatives.
    truncated_reason = "wall_clock_seconds" if wall_lapsed and unrealized else None
    status = LoweringStatus(
        status=(
            "truncated"
            if truncated_reason is not None
            else ("failed" if unrealized else "lowered")
        ),
        unrealized_outputs=unrealized,
        attempted=len(attempted),
        realized=len(L),
        proved_programs=proved_programs,
        reason=truncated_reason,
        stage="lowering",
    )
    return isa_adapter, L, status


def _candidate_sketches_for(
    task: _ENodeTask,
    target_sym: SymTensor,
    formal_syms: dict[EClassRef, SymTensor],
    d: int,
    *,
    recipe_cache: dict[tuple[Any, ...], tuple[Any, ...] | _RecipeCacheEntry]
    | None = None,
) -> Iterator[tuple[SketchNode, SymTensor | None]]:
    """Yield bound candidates and reuse normalized recipes."""
    input_syms = [formal_syms[c] for c in task.decoded.child_classes]
    if task.decoded.op in ISA_PASSTHROUGH_OPS:
        direct = SketchNode.make_op(
            task.decoded.op,
            [SketchNode.make_input(sym) for sym in input_syms],
            dict(task.decoded.attrs),
        )
        candidate_sym = isa_symbolic_eval(
            task.decoded.op,
            input_syms,
            dict(task.decoded.attrs),
        )
        if candidate_sym is SYM_EVAL_REJECTED:
            candidate_sym = None
        yield direct, candidate_sym
        return
    cache_key = _make_lowering_cache_key(
        task.decoded.op,
        task.decoded.attrs,
        input_syms,
        d,
    )
    cached = recipe_cache.get(cache_key) if recipe_cache is not None else None
    if isinstance(cached, _RecipeCacheEntry):
        recipes: Iterator[tuple[Any, ...]] = cached.replay()
    elif cached is not None:
        recipes = iter(cached)
    else:
        source = iter(
            _iter_complete_recipe_cache_entry(
                task.decoded.op,
                task.decoded.attrs,
                input_syms,
                d,
            )
        )
        if recipe_cache is None:
            recipes = source
        else:
            entry = _RecipeCacheEntry([], source)
            recipe_cache[cache_key] = entry
            recipes = entry.replay()

    for recipe in recipes:
        yield from _iter_rebound_recipes(
            target_sym,
            (recipe,),
            input_syms,
            target_op=task.decoded.op,
            max_hw_size=d,
        )
