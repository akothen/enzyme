"""Host tests for the load-then-mutate aliasing invariant (codegen soundness).

The tiled matmul emitter's pre-transpose LHS chain once scaled the DMA-loaded
operand block *in place* (`nisa.tensor_scalar(temp_tiles, temp_tiles, ...)`).
The load and the scale are invariant to the enclosing `nl.affine_range(n)`
block loop, so the compiler may legally reuse the buffer across n iterations —
and an in-place mutation is not idempotent under re-execution, so the second n
block saw a double-scaled operand (rmsnorm_matmul v0 @ TILES 1/1/*: the n=1
column block matched `(x/rms**2) @ w`, bit-identical max_abs_err=4.8672). The
CPU simulator executes loop iterations sequentially with no buffer reuse, so
it cannot see this class; these tests pin it statically instead.

Invariant: no emitted statement may mutate a DMA-loaded SBUF block in place
(read and write the same buffer). In-place mutation of *computed* buffers
(e.g. the `result_tiles` accumulate + post-scale, whose contents depend on the
enclosing loop indices) stays legal; the analyzer only flags buffers whose
most recent writer is a `dma_copy`.
"""

from __future__ import annotations

import re

_DMA_DST = re.compile(r"nisa\.dma_copy\(\s*dst=(\w+)\[")
_NISA_CALL = re.compile(r"nisa\.(\w+)\((.*)$")
_BUF_REF = re.compile(r"(\w+)\[")


def find_inplace_dma_mutations(code: str) -> list[str]:
    """Lines that mutate a DMA-loaded buffer in place.

    Statement-level scan (emitted nisa calls are one line): track which buffer
    names were last written by `nisa.dma_copy`; flag any other nisa call whose
    dst buffer (first arg, or dst= kwarg) also appears among its source args
    while still in the dma-loaded state. Any non-dma write to a buffer clears
    its state (mutating a computed buffer is the accumulator pattern, legal).
    """
    dma_loaded: set[str] = set()
    violations: list[str] = []
    for raw in code.splitlines():
        line = raw.strip()
        m = _DMA_DST.search(line)
        if m:
            dma_loaded.add(m.group(1))
            continue
        m = _NISA_CALL.search(line)
        if not m:
            continue
        bufs = _BUF_REF.findall(m.group(2))
        if not bufs:
            continue
        dst, srcs = bufs[0], bufs[1:]
        if dst in dma_loaded and dst in srcs:
            violations.append(line)
        dma_loaded.discard(dst)
    return violations


# ---- unit: the analyzer itself --------------------------------------------
def test_analyzer_flags_inplace_mutation_of_loaded_block():
    code = (
        "nisa.dma_copy(dst=t[0:M, b, 0:K], src=x[0:M, 0:K])\n"
        "nisa.tensor_scalar(t[0:M, b, 0:K], t[0:M, b, 0:K], nl.multiply, "
        "operand0=r[0:M, b, 0])\n"
    )
    assert len(find_inplace_dma_mutations(code)) == 1


def test_analyzer_allows_fresh_dst_and_computed_accumulators():
    code = (
        # load -> scale into a DIFFERENT buffer: fine
        "nisa.dma_copy(dst=t[0:M, b, 0:K], src=x[0:M, 0:K])\n"
        "nisa.tensor_scalar(s[0:M, b, 0:K], t[0:M, b, 0:K], nl.multiply, "
        "operand0=r[0:M, b, 0])\n"
        # accumulate / post-scale a computed buffer in place: fine
        "nisa.tensor_tensor(acc[0:M, 0:N], acc[0:M, 0:N], p[0:M, 0:N], nl.add)\n"
        "nisa.tensor_scalar(acc[0:M, 0:N], acc[0:M, 0:N], nl.multiply, "
        "operand0=r[0:M, b, 0])\n"
    )
    assert find_inplace_dma_mutations(code) == []
