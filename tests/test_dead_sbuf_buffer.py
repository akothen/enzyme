"""Host tests for the dead-SBUF-buffer invariant (codegen soundness).

The chained-matmul emitter's `shares_lhs` fusion computes the elementwise
combine of the first matmul's activation and the fused side matmul (e.g.
relu_mlp's `relu(x@w1) * (x@w2)`) into a dedicated pre-transposed operand
buffer (`inter_*_lhs2`) for the second matmul. The chained loop receives that
buffer as `lhs2_precomputed_var` — but its no-lhs-nc-transpose branch ignored
it and consumed the bare activation buffer instead, so the fused multiply was
written to a buffer nothing ever read (relu_mlp/silu_mlp hw-v1: the device
computed `relu(x@w1) @ w3`, max_abs_err ~115k vs the ~1k bf16 noise floor of
the correct variants).

Invariant: every SBUF buffer a variant allocates and writes must also be read.
A written-never-read buffer means an emitted computation is dropped on the
floor — some consumer is reading the wrong source. `.dtype` mentions are type
plumbing, not data reads.

`find_dead_sbuf_buffers` here is the single canonical analyzer; the other test
modules import it rather than keeping their own variant. It parses the emitted
source and classifies each `nisa` argument by its ROLE in the op's real NKI
signature. Role classification is required for correctness, not tidiness: the
e-graph search emits `nisa.activation_reduce(dst, op, data, reduce_op,
reduce_res=...)`, whose destination is an unsubscripted whole-tile name and
whose `data` source is positional. A scan that assumed "first subscripted
reference is the destination" mistook `data` for the destination and reported
the live input as dead.
"""

from __future__ import annotations

import ast

import pytest

_SBUF_ALLOCATORS = ("zeros", "ndarray", "full")

# Positional parameter order and destination roles per `nisa` op, taken from the
# real NKI signatures (`inspect.signature(nki.isa.<op>)`). Classifying arguments
# by ROLE is what makes the analyzer correct: a bare `args[0]` heuristic breaks
# on `activation_reduce`, whose value-carrying output is the `reduce_res` kwarg
# while its positional `dst` is an API-mandated scratch tile, and on `sendrecv`,
# whose `src` comes FIRST and would be mistaken for a destination.
_NISA_SIGNATURES: dict[str, tuple[tuple[str, ...], frozenset[str]]] = {
    "activation": (
        ("dst", "op", "data", "bias", "scale", "reduce_op", "reduce_res"),
        frozenset({"dst", "reduce_res"}),
    ),
    "activation_reduce": (
        ("dst", "op", "data", "reduce_op", "reduce_res", "bias", "scale"),
        frozenset({"dst", "reduce_res"}),
    ),
    "core_barrier": (("data", "cores"), frozenset()),
    "dma_copy": (("dst", "src"), frozenset({"dst"})),
    "dma_transpose": (("dst", "src", "axes"), frozenset({"dst"})),
    "nc_matmul": (("dst", "stationary", "moving"), frozenset({"dst"})),
    "nc_transpose": (("dst", "data"), frozenset({"dst"})),
    "reciprocal": (("dst", "data"), frozenset({"dst"})),
    "scalar_tensor_tensor": (
        ("dst", "data", "op0", "operand0", "op1", "operand1"),
        frozenset({"dst"}),
    ),
    "sendrecv": (
        ("src", "dst", "send_to_rank", "recv_from_rank", "pipe_id"),
        frozenset({"dst"}),
    ),
    "tensor_copy": (("dst", "src"), frozenset({"dst"})),
    "tensor_partition_reduce": (("dst", "op", "data"), frozenset({"dst"})),
    "tensor_reduce": (("dst", "op", "data", "axis"), frozenset({"dst"})),
    "tensor_scalar": (
        ("dst", "data", "op0", "operand0", "reverse0", "op1", "operand1"),
        frozenset({"dst"}),
    ),
    "tensor_scalar_cumulative": (
        ("dst", "src", "op0", "op1", "imm0", "imm1"),
        frozenset({"dst"}),
    ),
    "tensor_tensor": (("dst", "data1", "data2", "op"), frozenset({"dst"})),
}


def _buffer_root(expr: ast.expr | None) -> str | None:
    """Name a buffer reference is rooted at, peeling any subscripts."""
    while isinstance(expr, ast.Subscript):
        expr = expr.value
    return expr.id if isinstance(expr, ast.Name) else None


def _read_roots(expr: ast.expr | None) -> set[str]:
    """Buffer roots an expression READS as data.

    Attribute accesses are NOT descended into, so `buf.dtype` / `buf[...].shape`
    stay type-and-shape plumbing rather than data reads, matching the invariant
    in this module's docstring.
    """
    if expr is None or isinstance(expr, ast.Attribute):
        return set()
    root = _buffer_root(expr)
    if root is not None:
        # A subscript's index expressions only mention loop and dim variables.
        return {root}
    return {
        root
        for child in ast.iter_child_nodes(expr)
        if isinstance(child, ast.expr)
        for root in _read_roots(child)
    }


def _sbuf_alloc_target(node: ast.stmt) -> tuple[str | None, bool]:
    """(assigned name, is_sbuf_alloc) for an `nl.zeros/ndarray/full` assignment.

    Returns (None, False) for statements that are not allocations. Allocation
    statements contribute no data reads at all: `nl.ndarray(other[...].shape,
    dtype=other.dtype, ...)` is shape and dtype plumbing.
    """
    if not (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "nl"
        and node.value.func.attr in _SBUF_ALLOCATORS
    ):
        return None, False
    is_sbuf = any(
        keyword.arg == "buffer" and ast.unparse(keyword.value) == "nl.sbuf"
        for keyword in node.value.keywords
    )
    return node.targets[0].id, is_sbuf


def _slice_alias(node: ast.stmt) -> tuple[str | None, str | None]:
    """(alias name, aliased buffer root) for a bare `name = buf[slice]` view.

    SBUF-resident values are allocated once as a spanning buffer and then viewed
    through such an alias, so a read of the alias is a read of the buffer.
    """
    if not (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Subscript)
    ):
        return None, None
    return node.targets[0].id, _buffer_root(node.value)


def _subscript_slices(expr: ast.expr) -> list[ast.expr]:
    """Index expressions of a (possibly chained) subscript, outermost first."""
    slices: list[ast.expr] = []
    while isinstance(expr, ast.Subscript):
        slices.append(expr.slice)
        expr = expr.value
    return slices


# Constructs that can bind or rebind a name in the ENCLOSING scope in a way the
# walk does not model, so tracking an alias across one could attribute a read or
# write to the wrong buffer. Refusing mirrors the `unclassified nisa ops` guard:
# a loud stop beats a silent wrong answer. An AST census over the 12 generated
# kernel modules counts 0 of each (against For 352 / While 12 / FunctionDef 12),
# so this is a guard against future emitter shapes, not a live restriction.
#
# Deliberately NOT listed: comprehensions and `lambda` bind only inside their own
# scope and so cannot corrupt an outer alias (their targets are still shadowed
# while walking them); `ClassDef` and nested `FunctionDef` likewise introduce a
# fresh scope. `Delete` is handled directly by dropping the name.
_UNMODELED_NODES: tuple[type[ast.AST], ...] = (
    ast.Try,
    ast.TryStar,
    ast.Match,
    ast.NamedExpr,
    ast.Global,
    ast.Nonlocal,
)

_COMPREHENSIONS: tuple[type[ast.AST], ...] = (
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _comprehension_targets(node: ast.expr) -> list[str]:
    """Names a comprehension binds in its OWN scope, which shadow outer aliases."""
    names: list[str] = []
    for generator in getattr(node, "generators", []):
        stack = [generator.target]
        while stack:
            target = stack.pop()
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                stack.extend(target.elts)
            elif isinstance(target, ast.Starred):
                stack.append(target.value)
    return names


def _rebound_names(node: ast.stmt) -> list[str]:
    """Bare names a statement REBINDS, so a stale alias entry can be cleared.

    Any rebinding invalidates an alias, not just a rebinding to another slice:
    after `w = some_plain_name`, `w` no longer views whatever it used to.
    """
    targets: list[ast.expr | None] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
        targets = [node.target]
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        targets = [item.optional_vars for item in node.items]
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        return [(a.asname or a.name).split(".")[0] for a in node.names]
    names: list[str] = []
    stack = [t for t in targets if t is not None]
    while stack:
        target = stack.pop()
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            stack.extend(target.elts)
        elif isinstance(target, ast.Starred):
            stack.append(target.value)
    return names


# An alias name maps to the set of buffers it MAY view. A single root cannot
# express a name bound differently on two paths, and collapsing that to "unknown"
# loses the read, which is what turns a live buffer into a false report.
AliasMap = dict[str, frozenset[str]]

# Each loop pass only ever grows the possible-root sets, and the lattice (names in
# the module x subsets of roots) is finite, so the fixpoint settles in a few
# passes. The cap is a tripwire against a non-monotone edit, not a real bound.
_FIXPOINT_ITERATION_LIMIT = 64


def _merge_alias_maps(outcomes: list[AliasMap]) -> AliasMap:
    """UNION the possible roots of every name across mutually exclusive paths.

    A name absent from a path's outcome is not an alias on that path, so it
    contributes only itself. Union, not intersection: a read through the merged
    alias must credit EVERY root it might view, or a buffer that is live down one
    path is reported dead.
    """
    merged: AliasMap = {}
    for name in set().union(*(set(outcome) for outcome in outcomes)):
        roots: set[str] = set()
        for outcome in outcomes:
            roots |= outcome.get(name, frozenset({name}))
        if roots != {name}:
            merged[name] = frozenset(roots)
    return merged


def _nisa_op(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "nisa"
    ):
        return node.func.attr
    return None


def find_dead_sbuf_buffers(code: str) -> list[str]:
    """SBUF buffers that are allocated and written but never read.

    Arguments of `nisa` calls are classified by their ROLE in the op's real NKI
    signature (see `_NISA_SIGNATURES`), not by bare positional order, so a
    destination is identified even when it is an unsubscripted whole-tile name
    and a source is never mistaken for one.

    `activation_reduce` needs that role information: NKI requires a positional
    `dst` for the (P, F) elementwise activation even when the caller only wants
    the (P, 1) reduction, which lands in `reduce_res`. When the emitter has no
    consumer for the activation it passes a freshly allocated whole-tile scratch
    buffer, so that buffer being unread is an API artifact and not a dropped
    computation. A SUBSCRIPTED `dst` means the emitter deliberately redirected
    the activation into a spanning buffer a consumer is meant to read, so it
    stays subject to the check.

    A bare `name = buf[slice]` binding is a slice ALIAS, not a read: the
    SBUF-resident idiom allocates a spanning buffer once and views a window of it
    under a new name (`nc_matmul_1_tiles = nc_matmul_1_sb[...]`). Reads and
    writes through an alias are attributed to the buffer it is rooted at, so a
    buffer read only through its aliases is live while one that is aliased and
    still never read stays dead.

    Alias tracking is conservative in the SAFE direction. An alias maps to the
    SET of buffers it may view, and a read through it credits every one of them,
    so a buffer that is live down any path is never reported dead. A write
    likewise marks every possible root written.

    Accepted input language: this analyzer runs on generated NKI source, a flat
    loop nest of `For`/`While`, straight-line assignments and `nisa.*` calls. An
    AST census over the 12 generated kernel modules counts `For` 352, `While` 12,
    `FunctionDef` 12 and ZERO of `Try`, `TryStar`, `Match`, walrus,
    comprehensions, `Lambda`, `With`, `AugAssign`, `Delete`, `global`/`nonlocal`,
    `ClassDef` and (today) `If`.

    Anything outside that language is either modeled or refused, never
    mis-tracked. `Try`, `TryStar`, `Match`, walrus and `global`/`nonlocal` can
    rebind a name in the enclosing scope in ways this walk does not follow, so
    they RAISE, mirroring the `unclassified nisa ops` guard. Comprehensions,
    `lambda` and nested `def`/`class` bind only inside their own scope and so are
    walked with those names shadowed; `del` drops the name; `with`, `import` and
    unpacking assignments invalidate what they rebind.

    `If` is zero today but will appear (the emitter gains a trace-time
    `if NUM_BLOCK_K == 1:`), which is why the merge rule is union-based rather
    than relying on both arms happening to agree.
    """
    tree = ast.parse(code)
    allocs: set[str] = set()
    written: set[str] = set()
    read: set[str] = set()
    unclassified: set[str] = set()
    unmodeled: set[str] = set()

    def visit(node: ast.AST, aliases: AliasMap) -> None:
        """Walk `node`, mutating `aliases` for whatever follows it.

        Values are fully resolved when stored and lookups take one hop, so
        rebinding a root later cannot retroactively redirect an older alias.
        """

        def roots_of(name: str) -> frozenset[str]:
            return aliases.get(name, frozenset({name}))

        if isinstance(node, _UNMODELED_NODES):
            unmodeled.add(type(node).__name__)
            return

        if isinstance(node, _COMPREHENSIONS + (ast.Lambda,)):
            # A fresh scope: its targets shadow outer aliases while walking it,
            # and nothing it binds escapes, so the outer map is left untouched.
            inner = dict(aliases)
            for name in _comprehension_targets(node):
                inner.pop(name, None)
            if isinstance(node, ast.Lambda):
                for arg in [*node.args.args, *node.args.kwonlyargs]:
                    inner.pop(arg.arg, None)
            for child in ast.iter_child_nodes(node):
                visit(child, inner)
            return

        if isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases.pop(target.id, None)
                else:
                    visit(target, aliases)
            return

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Nested scope. Emitted kernels are a single top-level function, so
            # this is the kernel body itself; parameters shadow outer aliases.
            inner = dict(aliases)
            args = getattr(node, "args", None)
            if args is not None:
                for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                    inner.pop(arg.arg, None)
            for child in ast.iter_child_nodes(node):
                visit(child, inner)
            aliases.pop(node.name, None)
            return

        if isinstance(node, ast.stmt):
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                # A loop is entry state, then body-or-not, then `orelse`, which
                # runs AFTER zero or more iterations rather than as an
                # alternative to the body.
                for child in ast.iter_child_nodes(node):
                    if not isinstance(child, ast.stmt):
                        visit(child, aliases)
                skipped = dict(aliases)
                first = dict(aliases)
                # The loop target only shadows an alias on the path that runs.
                for rebound in _rebound_names(node):
                    first.pop(rebound, None)
                # Iterate to a fixpoint so the BACK EDGE is modeled: on the second
                # and later iterations the top of the body sees bindings made at
                # the bottom of the previous one. Re-walking only ever adds to the
                # read/written sets, and every state walked is one that genuinely
                # reaches the top of the body, so extra passes cannot invent a
                # read. The lattice (names x subsets of roots) is finite, hence
                # this terminates.
                entering = dict(first)
                for _ in range(_FIXPOINT_ITERATION_LIMIT):
                    exiting = dict(entering)
                    for stmt in node.body:
                        visit(stmt, exiting)
                    widened = _merge_alias_maps([first, exiting])
                    if widened == entering:
                        break
                    entering = widened
                else:  # pragma: no cover - finite lattice, kept as a tripwire
                    raise AssertionError(
                        "find_dead_sbuf_buffers: loop alias fixpoint did not settle"
                    )
                aliases.clear()
                aliases.update(_merge_alias_maps([skipped, exiting]))
                for stmt in node.orelse:
                    visit(stmt, aliases)
                return

            if isinstance(node, ast.If):
                visit(node.test, aliases)
                arms = [node.body, node.orelse or []]
                outcomes: list[AliasMap] = []
                for arm in arms:
                    branch = dict(aliases)
                    for stmt in arm:
                        visit(stmt, branch)
                    outcomes.append(branch)
                merged = _merge_alias_maps(outcomes)
                aliases.clear()
                aliases.update(merged)
                return

            name, is_sbuf = _sbuf_alloc_target(node)
            if name is not None:
                # Shape and dtype plumbing only, but the call is still evaluated
                # before the target is bound.
                aliases.pop(name, None)
                if is_sbuf:
                    allocs.add(name)
                return

            alias, aliased = _slice_alias(node)
            if alias is not None and aliased is not None:
                # Python evaluates the right-hand side, including every index
                # expression, BEFORE rebinding the target, so both are walked
                # under the OLD bindings. This is also what makes a self-reslice
                # (`w = w[...]`) resolve back to w's own roots.
                resolved = roots_of(aliased)
                for index in _subscript_slices(node.value):
                    visit(index, aliases)
                for rebound in _rebound_names(node):
                    aliases.pop(rebound, None)
                aliases[alias] = resolved
                return

        op = _nisa_op(node)
        if op is not None:
            assert isinstance(node, ast.Call)
            if op not in _NISA_SIGNATURES:
                unclassified.add(op)
                return
            params, dst_roles = _NISA_SIGNATURES[op]
            bound: list[tuple[str, ast.expr]] = [
                (params[i], arg) for i, arg in enumerate(node.args) if i < len(params)
            ]
            bound += [
                (keyword.arg, keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            ]
            for role, arg in bound:
                if role not in dst_roles:
                    for r in _read_roots(arg):
                        read.update(roots_of(r))
                    continue
                root = _buffer_root(arg)
                if root is None:
                    continue
                byproduct = (
                    op == "activation_reduce"
                    and role == "dst"
                    and isinstance(arg, ast.Name)
                    # An alias is a window of a spanning buffer a consumer is
                    # meant to read, exactly like a subscripted dst.
                    and arg.id not in aliases
                )
                if not byproduct:
                    written.update(roots_of(root))
            return
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Subscript) for target in node.targets
        ):
            # value form: buf[...] = nl.something(src[...], ...)
            for target in node.targets:
                root = _buffer_root(target)
                if root is not None:
                    written.update(roots_of(root))
            for r in _read_roots(node.value):
                read.update(roots_of(r))
            return
        if isinstance(node, ast.stmt):
            # Any other statement: walk it, then invalidate what it rebinds.
            for child in ast.iter_child_nodes(node):
                visit(child, aliases)
            for rebound in _rebound_names(node):
                aliases.pop(rebound, None)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, aliases)

    visit(tree, {})
    assert not unmodeled, (
        f"find_dead_sbuf_buffers: unmodeled binding construct(s) "
        f"{sorted(unmodeled)}; alias tracking does not model these, so extend "
        "the walk rather than letting a name be tracked to the wrong buffer"
    )
    assert not unclassified, (
        f"find_dead_sbuf_buffers: unclassified nisa ops {sorted(unclassified)}; "
        "add their signature to _NISA_SIGNATURES so arguments keep being "
        "classified by role"
    )
    return sorted(b for b in allocs if b in written and b not in read)


# ---- unit: the analyzer itself --------------------------------------------
def test_analyzer_flags_written_never_read_buffer():
    code = (
        "dead = nl.zeros((TILE_K, BLOCK_M), dtype=x.dtype, buffer=nl.sbuf)\n"
        "live = nl.zeros((TILE_K, BLOCK_M), dtype=x.dtype, buffer=nl.sbuf)\n"
        "tp = nl.ndarray((TILE_K, TILE_M), dtype=dead.dtype, buffer=nl.psum)\n"
        "nisa.tensor_tensor(dead[0:K, 0:M], tp[0:K, 0:M], live[0:K, 0:M], nl.multiply)\n"
        "nisa.dma_copy(dst=result[0:K, 0:M], src=live[0:K, 0:M])\n"
    )
    # `dead` is written by tensor_tensor but only ever mentioned as .dtype or
    # dst; `live` is read twice.
    assert find_dead_sbuf_buffers(code) == ["dead"]


def test_analyzer_accepts_written_then_read_buffer():
    code = (
        "buf = nl.zeros((TILE_K, BLOCK_M), dtype=x.dtype, buffer=nl.sbuf)\n"
        "nisa.tensor_tensor(buf[0:K, 0:M], a[0:K, 0:M], b[0:K, 0:M], nl.multiply)\n"
        "nisa.nc_matmul(out[0:K, 0:M], buf[0:K, 0:M], w[0:K, 0:M], accumulate=True)\n"
    )
    assert find_dead_sbuf_buffers(code) == []


def test_analyzer_reads_activation_reduce_data_argument():
    # Regression: the `data` source is the 3rd positional argument, and the
    # positional `dst` here is an unsubscripted whole-tile name. A positional
    # `bufs[0]` scan of subscripted references misses that bare `dst` entirely
    # and mis-reads `data` as the destination, so `chain` looked written but
    # never read. Shapes copied from real softmax_mm output.
    code = (
        "chain = nl.ndarray((TILE_M, TILES_IN_BLOCK_M, BLOCK_K), dtype=x.dtype,"
        " buffer=nl.sbuf)\n"
        "nisa.dma_copy(dst=chain[0:TILE_M, bm_red, 0:BLOCK_K], src=x[0:TILE_M, 0:BLOCK_K])\n"
        "partial = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)\n"
        "act = nl.ndarray(chain[0:TILE_M, bm_red, 0:BLOCK_K].shape, dtype=nl.float32,"
        " buffer=nl.sbuf)\n"
        "nisa.activation_reduce(act, nl.exp, chain[0:TILE_M, bm_red, 0:BLOCK_K], nl.add,"
        " reduce_res=partial[0:TILE_M, 0:1])\n"
        "nisa.tensor_tensor(totals[0:TILE_M, bm_red, 0], totals[0:TILE_M, bm_red, 0],"
        " partial[0:TILE_M, 0:1], op=nl.add)\n"
    )
    # `chain` is READ as `data`; `act` is the API-mandated activation scratch
    # tile nothing consumes; `partial` carries the value and is read.
    assert find_dead_sbuf_buffers(code) == []


def test_analyzer_flags_redirected_activation_output_that_is_dropped():
    # The activation output is exempt only when it is the throwaway bare-name
    # scratch tile. Once the emitter REDIRECTS it into a spanning buffer (a
    # subscripted dst) that buffer is a real result and must be consumed, so
    # dropping it is still a defect the analyzer reports.
    code = (
        "staged = nl.ndarray((TILE_M, NUM_K, TILES_IN_BLOCK_M, BLOCK_K),"
        " dtype=x.dtype, buffer=nl.sbuf)\n"
        "partial = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)\n"
        "nisa.activation_reduce(staged[0:TILE_M, k_side, bm_red, 0:BLOCK_K], nl.exp,"
        " chain[0:TILE_M, bm_red, 0:BLOCK_K], nl.add, reduce_res=partial[0:TILE_M, 0:1])\n"
        "nisa.tensor_tensor(totals[0:TILE_M, bm_red, 0], totals[0:TILE_M, bm_red, 0],"
        " partial[0:TILE_M, 0:1], op=nl.add)\n"
    )
    assert find_dead_sbuf_buffers(code) == ["staged"]


def test_analyzer_reads_sources_of_ops_whose_dst_is_not_first():
    # `sendrecv(src, dst, ...)` puts the SOURCE first. Classifying by position
    # would call `outgoing` a destination and miss that it is read.
    code = (
        "outgoing = nl.ndarray((TILE, N), dtype=nl.float32, buffer=nl.sbuf)\n"
        "nisa.tensor_copy(outgoing[0:TILE, 0:N], acc[0:TILE, 0:N])\n"
        "nisa.sendrecv(outgoing[0:TILE, 0:N], peer[0:TILE, 0:N], 1, 1, 0)\n"
    )
    assert find_dead_sbuf_buffers(code) == []


def test_analyzer_rejects_unknown_nisa_op_rather_than_guessing():
    # A new op with no signature entry must fail loudly: silently guessing
    # `args[0]` is the destination is exactly the bug this table replaces.
    code = (
        "buf = nl.ndarray((TILE, N), dtype=nl.float32, buffer=nl.sbuf)\n"
        "nisa.some_future_op(buf[0:TILE, 0:N], src[0:TILE, 0:N])\n"
    )
    with pytest.raises(AssertionError, match="unclassified nisa ops"):
        find_dead_sbuf_buffers(code)


def test_analyzer_ignores_read_only_and_unwritten_allocs():
    # allocated but never written (e.g. loop-scoped scratch the variant
    # doesn't use on this path) is not flagged — only computed-then-dropped is.
    code = "scratch = nl.zeros((1, 1), dtype=x.dtype, buffer=nl.sbuf)\n"
    assert find_dead_sbuf_buffers(code) == []


# ---- unit: slice aliases of SBUF-resident spanning buffers -----------------
def test_analyzer_follows_slice_alias_of_resident_buffer():
    # The SBUF-residency idiom: allocate a spanning buffer once, then bind a
    # window of it to a bare name and use that name. `resident[...]` never
    # appears at the read site, so an analyzer that ignores the alias binding
    # calls the buffer dead even though the alias is consumed downstream.
    code = (
        "resident = nl.zeros((TILE_M, TILES_IN_BLOCK_M, N), dtype=q.dtype,"
        " buffer=nl.sbuf)\n"
        "window = resident[0:TILE_M, 0:TILES_IN_BLOCK_M, (BLOCK_N * n):(BLOCK_N * n)"
        " + BLOCK_N]\n"
        "nisa.tensor_copy(window[0:TILE_M, bm, 0:BLOCK_N], ps[0:TILE_M, 0:BLOCK_N])\n"
        "nisa.activation(out[0:TILE_M, bm, 0:BLOCK_N], nl.exp,"
        " window[0:TILE_M, bm, 0:BLOCK_N])\n"
    )
    assert find_dead_sbuf_buffers(code) == []


def test_analyzer_follows_transitive_slice_alias_chain():
    # `b = a[...]` where `a` is itself an alias must resolve back to the root.
    code = (
        "resident = nl.zeros((TILE_M, TILES_IN_BLOCK_M, N), dtype=q.dtype,"
        " buffer=nl.sbuf)\n"
        "level1 = resident[0:TILE_M, 0:TILES_IN_BLOCK_M, 0:BLOCK_N]\n"
        "level2 = level1[0:TILE_M, bm, 0:BLOCK_N]\n"
        "nisa.tensor_copy(level2[0:TILE_M, 0:BLOCK_N], ps[0:TILE_M, 0:BLOCK_N])\n"
        "nisa.nc_transpose(tp[0:TILE_M, 0:TILE_N], level2[0:TILE_M, 0:TILE_N])\n"
    )
    assert find_dead_sbuf_buffers(code) == []


def test_analyzer_still_flags_aliased_buffer_that_is_never_read():
    # The alias extension must not become a blanket amnesty: writing through an
    # alias is a write of the root, not a read of it. Neither `dropped` nor its
    # alias is ever consumed, so the computation is still dropped on the floor.
    code = (
        "dropped = nl.zeros((TILE_M, TILES_IN_BLOCK_M, N), dtype=q.dtype,"
        " buffer=nl.sbuf)\n"
        "window = dropped[0:TILE_M, 0:TILES_IN_BLOCK_M, 0:BLOCK_N]\n"
        "nisa.tensor_copy(window[0:TILE_M, bm, 0:BLOCK_N], ps[0:TILE_M, 0:BLOCK_N])\n"
        "nisa.dma_copy(dst=result[0:TILE_M, 0:BLOCK_P], src=other[0:TILE_M, 0:BLOCK_P])\n"
    )
    assert find_dead_sbuf_buffers(code) == ["dropped"]


def test_analyzer_credits_the_alias_target_current_at_the_read():
    # An alias name reused for a second buffer must not retroactively make the
    # first one live: only `second` is read through `window` here.
    code = (
        "first = nl.zeros((TILE_M, N), dtype=q.dtype, buffer=nl.sbuf)\n"
        "second = nl.zeros((TILE_M, N), dtype=q.dtype, buffer=nl.sbuf)\n"
        "window = first[0:TILE_M, 0:BLOCK_N]\n"
        "nisa.tensor_copy(window[0:TILE_M, 0:BLOCK_N], ps[0:TILE_M, 0:BLOCK_N])\n"
        "window = second[0:TILE_M, 0:BLOCK_N]\n"
        "nisa.tensor_copy(window[0:TILE_M, 0:BLOCK_N], ps[0:TILE_M, 0:BLOCK_N])\n"
        "nisa.dma_copy(dst=result[0:TILE_M, 0:BLOCK_N], src=window[0:TILE_M, 0:BLOCK_N])\n"
    )
    assert find_dead_sbuf_buffers(code) == ["first"]


def test_analyzer_treats_aliased_activation_dst_as_a_real_destination():
    # `activation_reduce`'s bare-Name `dst` exemption covers the throwaway
    # scratch tile only. An alias name is a window of a spanning buffer, so it
    # keeps the redirected-output semantics of a subscripted dst and the buffer
    # behind it must still be consumed.
    code = (
        "staged = nl.ndarray((TILE_M, TILES_IN_BLOCK_M, N), dtype=x.dtype,"
        " buffer=nl.sbuf)\n"
        "partial = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)\n"
        "window = staged[0:TILE_M, bm_red, 0:BLOCK_N]\n"
        "nisa.activation_reduce(window, nl.exp, chain[0:TILE_M, 0:BLOCK_N], nl.add,"
        " reduce_res=partial[0:TILE_M, 0:1])\n"
        "nisa.tensor_tensor(totals[0:TILE_M, 0], totals[0:TILE_M, 0],"
        " partial[0:TILE_M, 0:1], op=nl.add)\n"
    )
    assert find_dead_sbuf_buffers(code) == ["staged"]


def test_analyzer_alias_extension_keeps_the_unclassified_op_guard():
    # The alias walk returns early on alias bindings; an unknown nisa op inside
    # the same module must still trip the guard rather than be skipped.
    code = (
        "resident = nl.zeros((TILE_M, N), dtype=q.dtype, buffer=nl.sbuf)\n"
        "window = resident[0:TILE_M, 0:BLOCK_N]\n"
        "nisa.some_future_op(window[0:TILE_M, 0:BLOCK_N], src[0:TILE_M, 0:BLOCK_N])\n"
    )
    with pytest.raises(AssertionError, match="unclassified nisa ops"):
        find_dead_sbuf_buffers(code)


# ---- unit: alias tracking must not create false negatives -------------------
# Each case below is a way a genuinely dead buffer could slip past the
# write-through-alias capability, either because a stale or wrong alias entry
# credits the wrong root or because the alias binding shadowed part of the walk.
# All of them are `['dead']`: the alias map is an aid to attribution, never an
# excuse to stop reporting.
_ALIAS_FALSE_NEGATIVE_CASES: dict[str, str] = {
    # Rebinding an alias to a NON-slice must clear it, or the later read of `w`
    # is credited to `dead` and the dropped computation goes unreported.
    "ordinary_rebind": """
dead = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)
w = dead[0:T, 0:N]
nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])
w = replacement
nisa.dma_copy(dst=out[0:T, 0:N], src=w[0:T, 0:N])
""",
    # Rebinding the ROOT must not retroactively redirect an existing alias:
    # `w` still views the original `dead`, so the write through `w` lands there
    # and it is `dead`, not `other`, that is dropped.
    "root_rebind": """
dead = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)
other = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)
w = dead[0:T, 0:N]
dead = other[0:T, 0:N]
nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])
nisa.dma_copy(dst=out[0:T, 0:N], src=other[0:T, 0:N])
""",
    # `w = w[...]` must resolve the right-hand side before clearing the target,
    # or the entry becomes `w -> w` and the root is lost.
    "self_reslice": """
dead = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)
w = dead[0:T, 0:N]
w = w[0:T, 0:N]
nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])
""",
    # An alias binding must still be descended into: a write nested in the
    # subscript's index expression is a real write.
    "write_in_index_subtree": """
dead = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)
w = base[nisa.tensor_copy(dead[0:T, 0:N], src[0:T, 0:N])]
""",
    # A loop target shadows the alias name for the body.
    "loop_target_shadows_alias": """
dead = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)
w = dead[0:T, 0:N]
nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])
for w in nl.affine_range(4):
    nisa.dma_copy(dst=out[0:T, 0:N], src=w[0:T, 0:N])
""",
    # An alias bound in an `if` body must not be in effect in the `else` body:
    # the read there is of the PRE-EXISTING `w`, which views `other`.
    "sibling_branch_leak": """
dead = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)
other = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)
w = other[0:T, 0:N]
nisa.tensor_copy(dead[0:T, 0:N], src[0:T, 0:N])
if cond:
    w = dead[0:T, 0:N]
else:
    nisa.dma_copy(dst=out[0:T, 0:N], src=w[0:T, 0:N])
""",
    # Tuple unpacking is a rebinding too.
    "tuple_unpack_rebind": """
dead = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)
w = dead[0:T, 0:N]
nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])
w, z = shape
nisa.dma_copy(dst=out[0:T, 0:N], src=w[0:T, 0:N])
""",
}


@pytest.mark.parametrize("case", sorted(_ALIAS_FALSE_NEGATIVE_CASES), ids=lambda c: c)
def test_analyzer_alias_tracking_has_no_false_negatives(case):
    assert find_dead_sbuf_buffers(_ALIAS_FALSE_NEGATIVE_CASES[case]) == ["dead"]


def test_analyzer_finds_unclassified_op_hidden_in_an_alias_index():
    # The `unclassified nisa ops` guard is what forces a new op's signature to be
    # registered. An alias binding must not become a hole it can hide in: the
    # early return on the binding still has to walk the index subtree.
    code = (
        "resident = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "w = resident[nisa.some_future_op(x)]\n"
        "nisa.tensor_copy(out[0:T, 0:N], w[0:T, 0:N])\n"
    )
    with pytest.raises(AssertionError, match="unclassified nisa ops"):
        find_dead_sbuf_buffers(code)


def test_analyzer_finds_unclassified_op_inside_a_branch():
    # Branch bodies are walked with a saved/restored alias map; that must not
    # skip them.
    code = (
        "resident = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "if cond:\n"
        "    w = resident[0:T, 0:N]\n"
        "    nisa.some_future_op(w)\n"
    )
    with pytest.raises(AssertionError, match="unclassified nisa ops"):
        find_dead_sbuf_buffers(code)


def test_analyzer_unions_possible_roots_when_branches_disagree():
    # Both arms bind `w`, but to DIFFERENT buffers, so after the merge `w` may
    # view either. A write through it drops both, since neither is ever read.
    # Tracking a single root cannot express this, and dropping the name on
    # disagreement credits neither, which is how the write went unreported.
    code = (
        "a = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "b = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "if cond:\n"
        "    w = a[0:T, 0:N]\n"
        "else:\n"
        "    w = b[0:T, 0:N]\n"
        "nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])\n"
    )
    assert find_dead_sbuf_buffers(code) == ["a", "b"]


def test_analyzer_read_through_disagreeing_alias_credits_every_root():
    """A read must credit EVERY buffer the alias may view.

    This is the false-positive direction and the dangerous one: crediting only
    one arm (or neither) reports a genuinely live buffer as dead, which fails a
    codegen gate on correct output. `live` is written before the branch and read
    after it through an alias that views it on only ONE arm; it is still live.
    """
    code = (
        "live = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "nisa.tensor_copy(live[0:T, 0:N], src[0:T, 0:N])\n"
        "if cond:\n"
        "    w = live[0:T, 0:N]\n"
        "else:\n"
        "    w = external[0:T, 0:N]\n"
        "nisa.dma_copy(dst=out[0:T, 0:N], src=w[0:T, 0:N])\n"
    )
    assert find_dead_sbuf_buffers(code) == []


def test_analyzer_write_through_disagreeing_alias_credits_every_root():
    # The write counterpart: `dead` is written through an alias that views it on
    # one arm only, and never read, so it is still dropped.
    code = (
        "dead = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "if cond:\n"
        "    w = dead[0:T, 0:N]\n"
        "else:\n"
        "    w = external[0:T, 0:N]\n"
        "nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])\n"
    )
    assert find_dead_sbuf_buffers(code) == ["dead"]


def test_analyzer_reads_the_rhs_of_a_rebinding_under_the_old_binding():
    # Python evaluates the right-hand side and its indices BEFORE rebinding the
    # target, so the `w` inside the RHS still views `live`. Rebinding first loses
    # that read and reports a live buffer dead.
    code = (
        "live = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "w = live[0:T, 0:N]\n"
        "nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])\n"
        "w = external[nisa.dma_copy(dst=out[0:T, 0:N], src=w[0:T, 0:N])]\n"
    )
    assert find_dead_sbuf_buffers(code) == []


def test_analyzer_reads_the_rhs_of_a_non_slice_rebinding_under_the_old_binding():
    # Same ordering rule when the new value is not a slice at all, which takes
    # the generic statement path rather than the alias path.
    code = (
        "live = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "w = live[0:T, 0:N]\n"
        "nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])\n"
        "w = plain(nisa.dma_copy(dst=out[0:T, 0:N], src=w[0:T, 0:N]))\n"
    )
    assert find_dead_sbuf_buffers(code) == []


def test_analyzer_runs_loop_orelse_after_the_body_not_instead_of_it():
    # A loop's `orelse` runs after zero or more iterations, so it sees bindings
    # the body made. Treating it as a mutually exclusive sibling loses this read.
    code = (
        "resident = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "nisa.tensor_copy(resident[0:T, 0:N], src[0:T, 0:N])\n"
        "for i in nl.affine_range(4):\n"
        "    w = resident[0:T, 0:N]\n"
        "else:\n"
        "    nisa.dma_copy(dst=out[0:T, 0:N], src=w[0:T, 0:N])\n"
    )
    assert find_dead_sbuf_buffers(code) == []


def test_analyzer_models_the_loop_back_edge():
    # On the second and later iterations the top of the body sees the binding
    # made at the BOTTOM of the previous one, so this read of `w` really does
    # read `resident`. A single forward pass over the body misses it and reports a
    # live buffer dead, which is the false-positive direction.
    code = (
        "resident = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "for i in nl.affine_range(4):\n"
        "    nisa.dma_copy(dst=out[0:T, 0:N], src=w[0:T, 0:N])\n"
        "    w = resident[0:T, 0:N]\n"
        "    nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])\n"
    )
    assert find_dead_sbuf_buffers(code) == []


def test_analyzer_still_flags_a_dead_buffer_written_only_inside_a_loop():
    # The back-edge fixpoint must not hand out reads that do not exist: nothing
    # reads through `w` here, so the buffer is still dropped.
    code = (
        "dead = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "for i in nl.affine_range(4):\n"
        "    w = dead[0:T, 0:N]\n"
        "    nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])\n"
    )
    assert find_dead_sbuf_buffers(code) == ["dead"]


def test_analyzer_keeps_an_alias_across_a_loop_that_may_not_run():
    # A loop TARGET shadows the alias only on the path where the body runs. On
    # the zero-iteration path `w` still views `resident`, so the read after the
    # loop credits it and the buffer is live.
    code = (
        "resident = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "w = resident[0:T, 0:N]\n"
        "nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])\n"
        "for w in nl.affine_range(4):\n"
        "    pass\n"
        "nisa.dma_copy(dst=out[0:T, 0:N], src=w[0:T, 0:N])\n"
    )
    assert find_dead_sbuf_buffers(code) == []


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            "resident = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
            "try:\n"
            "    w = resident[0:T, 0:N]\n"
            "except E:\n"
            "    pass\n",
            id="try",
        ),
        pytest.param(
            "resident = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
            "w = (v := resident[0:T, 0:N])\n",
            id="walrus",
        ),
        pytest.param(
            "resident = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
            "global resident\n",
            id="global",
        ),
    ],
)
def test_analyzer_refuses_unmodeled_binding_constructs(code):
    # Generated NKI source is flat loop nests; an AST census over the generated
    # kernel modules counts zero of these. Rather than track a name to the wrong
    # buffer if one ever appears, the analyzer refuses, the same way it refuses an
    # unregistered nisa op.
    with pytest.raises(AssertionError, match="unmodeled binding construct"):
        find_dead_sbuf_buffers(code)


def test_analyzer_scopes_comprehension_and_lambda_bindings():
    """A comprehension or lambda binds only in its own scope.

    These are not refused, because they cannot corrupt an outer alias: the target
    shadows it while walking the body and nothing escapes. `strip_gen_rungs.py`
    (a rung GENERATOR, whose `nl.sbuf` text lives in string templates) contains a
    generator expression, so refusing outright would have regressed a file the
    committed analyzer handled fine.
    """
    # `i` shadows the outer alias inside the comprehension, so the read of `i`
    # there is not a read of `resident`, which stays dead.
    code = (
        "resident = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "i = resident[0:T, 0:N]\n"
        "nisa.tensor_copy(i[0:T, 0:N], src[0:T, 0:N])\n"
        "ws = [nisa.dma_copy(dst=out[0:T, 0:N], src=i[0:T, 0:N]) for i in range(N)]\n"
    )
    assert find_dead_sbuf_buffers(code) == ["resident"]

    # A read through the alias in the comprehension's ITERABLE is outside the
    # comprehension's own binding, so it does count.
    live = (
        "resident = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "w = resident[0:T, 0:N]\n"
        "nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])\n"
        "ws = [z for z in nisa.dma_copy(dst=out[0:T, 0:N], src=w[0:T, 0:N])]\n"
    )
    assert find_dead_sbuf_buffers(live) == []


def test_analyzer_drops_an_alias_on_del():
    # `del w` unbinds the alias, so a later `w` is not that buffer's window.
    code = (
        "dead = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "w = dead[0:T, 0:N]\n"
        "nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])\n"
        "del w\n"
        "nisa.dma_copy(dst=out[0:T, 0:N], src=w[0:T, 0:N])\n"
    )
    assert find_dead_sbuf_buffers(code) == ["dead"]


def test_analyzer_handles_the_generated_kernel_function_wrapper():
    # Emitted kernels wrap everything in a decorated `def`, and a parameter name
    # must shadow an outer alias rather than inherit it.
    code = (
        "@nki.jit\n"
        "def kernel(q, w):\n"
        "    resident = nl.zeros((T, N), dtype=q.dtype, buffer=nl.sbuf)\n"
        "    v = resident[0:T, 0:N]\n"
        "    nisa.tensor_copy(v[0:T, 0:N], src[0:T, 0:N])\n"
        "    nisa.dma_copy(dst=out[0:T, 0:N], src=v[0:T, 0:N])\n"
        "    return out\n"
    )
    assert find_dead_sbuf_buffers(code) == []


def test_analyzer_keeps_alias_bound_identically_on_every_branch():
    # Branch isolation must not throw away a binding the paths AGREE on, or the
    # resident idiom under a conditional would go back to reporting false
    # positives. Both arms root `w` at `resident`, so the read after the merge
    # credits it.
    code = (
        "resident = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "if cond:\n"
        "    w = resident[0:T, 0:N]\n"
        "else:\n"
        "    w = resident[0:T, 0:N]\n"
        "nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])\n"
        "nisa.dma_copy(dst=out[0:T, 0:N], src=w[0:T, 0:N])\n"
    )
    assert find_dead_sbuf_buffers(code) == []


def test_analyzer_credits_the_resident_idiom_bound_inside_a_loop_body():
    # This is the shape the real resident kernels take: the spanning buffer is
    # allocated outside the loop and re-aliased per iteration. Branch isolation
    # must still let a binding be used by later statements in the SAME body.
    code = (
        "resident = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "for n in nl.affine_range(4):\n"
        "    w = resident[0:T, 0:N]\n"
        "    nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])\n"
        "    nisa.dma_copy(dst=out[0:T, 0:N], src=w[0:T, 0:N])\n"
    )
    assert find_dead_sbuf_buffers(code) == []


def test_analyzer_credits_a_read_through_an_alias_inside_a_branch():
    # Isolating branch alias state must not lose reads made inside a branch:
    # the alias is bound in straight-line code and read under a conditional,
    # which is the shape a guarded consumer takes.
    code = (
        "resident = nl.zeros((T, N), dtype=x.dtype, buffer=nl.sbuf)\n"
        "w = resident[0:T, 0:N]\n"
        "nisa.tensor_copy(w[0:T, 0:N], src[0:T, 0:N])\n"
        "if cond:\n"
        "    nisa.dma_copy(dst=out[0:T, 0:N], src=w[0:T, 0:N])\n"
    )
    assert find_dead_sbuf_buffers(code) == []
