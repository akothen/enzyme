# How Axon works, and what it has (and hasn't) tried

Axon is a synthesizer that optimizes NKI kernels. It runs in four phases:

1. **Graph rewrites.** Axon applies rewrite rules (operator propagation) to the
   input kernel's data-flow graph and uses an SMT solver to check that each
   rewritten graph is semantically equivalent to the original. Equivalent graphs
   are kept as candidate rewrites.
2. **ISA lowering.** Each candidate is lowered to Trainium ISA by top-down
   enumerative search over ISA operators, followed by another round of
   operator propagation + SMT checking at the ISA level to expose
   hardware-specific rewrites.
3. **Kernel extraction.** Axon extracts NKI kernels from the ISA rewrites. Tile
   sizes come from graph-derived ISA constraints; the `TILES_IN_BLOCK_*` values
   come from a fixed grid; LNC sharding plans come from exhaustive enumeration of
   colorings followed by feasibility pruning.
4. **Compile + bench.** The candidates that survive pre-compile filtering are
   compiled and run on hardware, and the fastest correct one wins. Two caps
   apply by default: at most 256 hardware graphs, and at most 8 legal tile
   configurations per emitted schedule family.

**What Axon has explored for the kernel you are given — and its limits:**
- Axon applies a **fixed set of rewrite rules**. Operator-level rewrites have been
attempted, but are NOT exhausted.
- The **tile-configuration** space (`TILES_IN_BLOCK_*`) is **sampled, not
  exhausted**: by default at most 8 diverse legal configurations per emitted
  schedule family are benched, and the kernel you are given won that subset.
  Retuning tile sizes is therefore a legitimate lever, not a closed space.

**What Axon structurally CANNOT express:**
- **Choosing the layout / orientation in which an intermediate value is materialized** to avoid a downstream transpose. Operator propagation reorders
  operators; it does not decide what layout a value lives in.
- **Loop restructuring / fusion / residency decisions** — e.g. keeping a value
  resident in SBUF across two passes instead of recomputing or re-reading it
  from HBM, or fusing two sweeps into one.
- **Moving work to or from the DMA engines** — for example a DMA transpose in
  place of `nc_transpose`. Axon does choose between the Activation, Vector, and
  PE engines during ISA lowering, so that choice is not off-limits.

**One rewrite is valid only over part of the input range. Handle it with care:**
- **`1/x` on the Activation engine.** `nisa.activation(out, nl.reciprocal, in)`
  runs on the Activation engine and is faster than the Vector-engine
  `nisa.reciprocal`, but its domain is limited: measured on trn2 it is accurate
  to about 1e-7 up to 1e12, and it returns exactly 0.0 at and above 1e14.
  `nisa.reciprocal` is correct over the whole range. Axon proves equivalence over
  the reals with no value ranges, so it cannot tell the two apart, and the
  simulator returns the exact value either way — only a device run disagrees.
  Use the activation form **only** when the operand provably stays well below
  1e12 for this kernel's real inputs. A softmax denominator built from a bare
  `exp` does not: it reaches 1e15 to 1e20 and returns zeros. If the kernel you
  are given already contains the activation form, the same bound is a
  precondition to confirm, not a change to make. Say in your note which operand
  you bounded and why the bound holds; the harness records that note but cannot
  verify it, because the correctness check runs one input distribution. If you
  cannot bound the operand, use `nisa.reciprocal`.

The same rule applies to any rewrite that is only valid over part of the input
range: the correctness check runs one input distribution, so a rewrite that is
wrong for large values can still pass. State the precondition, or do not make
the change.

**Bottom line:** treat Axon's output as a strong starting point, not a ceiling.
The wins it misses come from two directions — operator rewrites its limited or
guarded rules didn't reach, *and* the scheduling / layout / residency decisions it
cannot express at all.
