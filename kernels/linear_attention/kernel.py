"""The Axon math for `linear_attention` (axon-side only; nkilib's harness never
imports this module).

The unnormalized core of global (non-causal) linear attention with an identity
feature map — the defining linear-attention reassociation. Instead of the softmax
path `softmax(Q Kᵀ) V` (an sq×sk score matrix), it computes the state `S = Kᵀ V`
(a dk×dv matrix) once and then `Q S`, turning O(sq·sk·d) attention into
O(sq·dk·dv): out = Q (Kᵀ V)
"""

from axon import AxonArray


def kernel_linear_attention(q: AxonArray, key: AxonArray, v: AxonArray) -> AxonArray:
    # q [sq, dk], key [sk, dk], v [sk, dv]. The key input is named `key`, not `k`:
    # the emitter uses `k` (and m/n/p) as block-loop variables, so a `k` input
    # would be shadowed inside the K-loop (dtype=k.dtype then fails to resolve).
    kt = key.transpose()  # [dk, sk]
    state = kt @ v  # [dk, dv]  = Σ_j k_jᵀ v_j  (the linear-attention state)
    return q @ state  # [sq, dv]  = Σ_j (q_t · k_j) v_j
