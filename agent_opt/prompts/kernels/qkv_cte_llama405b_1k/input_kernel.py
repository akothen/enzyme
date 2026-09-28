import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def qkv_cte(x, mlp_prev, attention_prev, w, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K):
    M, K = x.shape
    M_, K_ = mlp_prev.shape
    M_, K_ = attention_prev.shape
    K_, N = w.shape
    result = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)

    TILE_M = 128
    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    assert M % BLOCK_M == 0, "M must be divisible by BLOCK_M"
    NUM_BLOCK_M = M // BLOCK_M
    assert NUM_BLOCK_M > 0, "M too small for the tile configuration"

    TILE_N = 512
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    assert N % BLOCK_N == 0, "N must be divisible by BLOCK_N"
    NUM_BLOCK_N = N // BLOCK_N
    assert NUM_BLOCK_N > 0, "N too small for the tile configuration"

    TILE_K = 128
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K
    assert K % BLOCK_K == 0, "K must be divisible by BLOCK_K"
    NUM_BLOCK_K = K // BLOCK_K
    assert NUM_BLOCK_K > 0, "K too small for the tile configuration"

    for m in nl.affine_range(NUM_BLOCK_M):
        activation_reduce_4_tiles = nl.zeros(
            (TILE_M, TILES_IN_BLOCK_M, 1),
            dtype=nl.float32, buffer=nl.sbuf)
        tensor_tensor_1_staged = nl.ndarray(
            (TILE_M, NUM_BLOCK_K, TILES_IN_BLOCK_M, BLOCK_K),
            dtype=x.dtype, buffer=nl.sbuf)
        for k_side in nl.sequential_range(NUM_BLOCK_K):
            x_activation_reduce_4_chain_tile_k = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_K), dtype=x.dtype, buffer=nl.sbuf)
            for bm_red in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.dma_copy(
                    dst=x_activation_reduce_4_chain_tile_k[0:TILE_M, bm_red, 0:BLOCK_K],
                    src=x[((TILES_IN_BLOCK_M * m + bm_red) * TILE_M):((TILES_IN_BLOCK_M * m + bm_red) * TILE_M) + TILE_M,
                           (BLOCK_K * k_side):(BLOCK_K * k_side) + BLOCK_K])
            attention_prev_activation_reduce_4_chain_tile_k = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_K), dtype=attention_prev.dtype, buffer=nl.sbuf)
            for bm_red in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.dma_copy(
                    dst=attention_prev_activation_reduce_4_chain_tile_k[0:TILE_M, bm_red, 0:BLOCK_K],
                    src=attention_prev[((TILES_IN_BLOCK_M * m + bm_red) * TILE_M):((TILES_IN_BLOCK_M * m + bm_red) * TILE_M) + TILE_M,
                           (BLOCK_K * k_side):(BLOCK_K * k_side) + BLOCK_K])
            mlp_prev_activation_reduce_4_chain_tile_k = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_K), dtype=mlp_prev.dtype, buffer=nl.sbuf)
            for bm_red in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.dma_copy(
                    dst=mlp_prev_activation_reduce_4_chain_tile_k[0:TILE_M, bm_red, 0:BLOCK_K],
                    src=mlp_prev[((TILES_IN_BLOCK_M * m + bm_red) * TILE_M):((TILES_IN_BLOCK_M * m + bm_red) * TILE_M) + TILE_M,
                           (BLOCK_K * k_side):(BLOCK_K * k_side) + BLOCK_K])
            for bm_red in nl.affine_range(TILES_IN_BLOCK_M):
                tensor_tensor_0_activation_reduce_4_scratch = nl.ndarray((TILE_M, BLOCK_K), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(tensor_tensor_0_activation_reduce_4_scratch[0:TILE_M, 0:BLOCK_K], x_activation_reduce_4_chain_tile_k[0:TILE_M, bm_red, 0:BLOCK_K], attention_prev_activation_reduce_4_chain_tile_k[0:TILE_M, bm_red, 0:BLOCK_K], nl.add)
                tensor_tensor_1_activation_reduce_4_scratch = nl.ndarray((TILE_M, BLOCK_K), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(tensor_tensor_1_activation_reduce_4_scratch[0:TILE_M, 0:BLOCK_K], tensor_tensor_0_activation_reduce_4_scratch[0:TILE_M, 0:BLOCK_K], mlp_prev_activation_reduce_4_chain_tile_k[0:TILE_M, bm_red, 0:BLOCK_K], nl.add)
                nisa.tensor_copy(tensor_tensor_1_staged[0:TILE_M, k_side, bm_red, 0:BLOCK_K], tensor_tensor_1_activation_reduce_4_scratch[0:TILE_M, 0:BLOCK_K])
                partial_activation_reduce_4 = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
                activation_reduce_4_act = nl.ndarray(tensor_tensor_1_activation_reduce_4_scratch[0:TILE_M, 0:BLOCK_K].shape, dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation_reduce(activation_reduce_4_act, nl.square, tensor_tensor_1_activation_reduce_4_scratch[0:TILE_M, 0:BLOCK_K], nl.add, reduce_res=partial_activation_reduce_4[0:TILE_M, 0:1])
                nisa.tensor_tensor(activation_reduce_4_tiles[0:TILE_M, bm_red, 0],
                    activation_reduce_4_tiles[0:TILE_M, bm_red, 0], partial_activation_reduce_4[0:TILE_M, 0:1], op=nl.add)

        tensor_scalar_5_tiles = nl.ndarray(
            (TILE_M, TILES_IN_BLOCK_M, 1),
            dtype=nl.float32, buffer=nl.sbuf)
        for bm_side in nl.affine_range(TILES_IN_BLOCK_M):
            nisa.tensor_scalar(tensor_scalar_5_tiles[0:TILE_M, bm_side, 0], activation_reduce_4_tiles[0:TILE_M, bm_side, 0], nl.multiply, operand0=6.103515625e-05)
        tensor_scalar_6_tiles = nl.ndarray(
            (TILE_M, TILES_IN_BLOCK_M, 1),
            dtype=nl.float32, buffer=nl.sbuf)
        for bm_side in nl.affine_range(TILES_IN_BLOCK_M):
            nisa.tensor_scalar(tensor_scalar_6_tiles[0:TILE_M, bm_side, 0], tensor_scalar_5_tiles[0:TILE_M, bm_side, 0], nl.add, operand0=1e-06)
        activation_7_tiles = nl.ndarray(
            (TILE_M, TILES_IN_BLOCK_M, 1),
            dtype=nl.float32, buffer=nl.sbuf)
        for bm_side in nl.affine_range(TILES_IN_BLOCK_M):
            nisa.activation(activation_7_tiles[0:TILE_M, bm_side, 0], nl.sqrt, tensor_scalar_6_tiles[0:TILE_M, bm_side, 0])
        activation_8_tiles = nl.ndarray(
            (TILE_M, TILES_IN_BLOCK_M, 1),
            dtype=nl.float32, buffer=nl.sbuf)
        for bm_side in nl.affine_range(TILES_IN_BLOCK_M):
            nisa.activation(activation_8_tiles[0:TILE_M, bm_side, 0], nl.reciprocal, activation_7_tiles[0:TILE_M, bm_side, 0])

        for n in nl.affine_range(NUM_BLOCK_N):
            result_tiles = (nl.ndarray if NUM_BLOCK_K == 1 else nl.zeros)(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=x.dtype, buffer=nl.sbuf)

            for k in nl.sequential_range(NUM_BLOCK_K):
                nc_transpose_2_tiles = nl.ndarray(
                    (TILE_K, TILES_IN_BLOCK_K, BLOCK_M),
                    dtype=x.dtype, buffer=nl.sbuf)
                for b_m in nl.affine_range(TILES_IN_BLOCK_M):
                    for bk_t in nl.affine_range(TILES_IN_BLOCK_K):
                        tp_psum_0 = nl.ndarray((TILE_K, TILE_M), dtype=x.dtype, buffer=nl.psum)
                        nisa.nc_transpose(tp_psum_0[0:TILE_K, 0:TILE_M], tensor_tensor_1_staged[0:TILE_M, k, b_m, (bk_t * TILE_K):(bk_t * TILE_K) + TILE_K])
                        nisa.tensor_copy(nc_transpose_2_tiles[0:TILE_K, bk_t, (b_m * TILE_M):(b_m * TILE_M) + TILE_M], tp_psum_0[0:TILE_K, 0:TILE_M])

                w_tiles = nl.ndarray(
                    (TILE_K, TILES_IN_BLOCK_K, BLOCK_N),
                    dtype=w.dtype, buffer=nl.sbuf)
                for bk_t in nl.affine_range(TILES_IN_BLOCK_K):
                    nisa.dma_copy(
                        dst=w_tiles[0:TILE_K, bk_t, 0:BLOCK_N],
                        src=w[((TILES_IN_BLOCK_K * k + bk_t) * TILE_K):((TILES_IN_BLOCK_K * k + bk_t) * TILE_K) + TILE_K,
                             (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N])

                for bm in nl.affine_range(TILES_IN_BLOCK_M):
                    for bn in nl.affine_range(TILES_IN_BLOCK_N):
                        res_tile = nl.ndarray(
                            (TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(res_tile[0:TILE_M, 0:TILE_N],
                            nc_transpose_2_tiles[0:TILE_K, 0, (bm * TILE_M):(bm * TILE_M) + TILE_M],
                            w_tiles[0:TILE_K, 0, (bn * TILE_N):(bn * TILE_N) + TILE_N], accumulate=False)
                        for bk_peel in nl.affine_range(TILES_IN_BLOCK_K - 1):
                            bk = bk_peel + 1
                            nisa.nc_matmul(res_tile[0:TILE_M, 0:TILE_N],
                                nc_transpose_2_tiles[0:TILE_K, bk, (bm * TILE_M):(bm * TILE_M) + TILE_M],
                                w_tiles[0:TILE_K, bk, (bn * TILE_N):(bn * TILE_N) + TILE_N], accumulate=True)
                        if NUM_BLOCK_K == 1:
                            nisa.tensor_copy(result_tiles[0:TILE_M, bm, (bn * TILE_N):(bn * TILE_N) + TILE_N], res_tile[0:TILE_M, 0:TILE_N])
                        else:
                            nisa.tensor_tensor(result_tiles[0:TILE_M, bm, (bn * TILE_N):(bn * TILE_N) + TILE_N],
                                result_tiles[0:TILE_M, bm, (bn * TILE_N):(bn * TILE_N) + TILE_N], res_tile[0:TILE_M, 0:TILE_N], nl.add)

            tensor_scalar_9_post = nl.ndarray((TILE_M, TILES_IN_BLOCK_M, BLOCK_N), dtype=x.dtype, buffer=nl.sbuf)
            for bm_post in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_scalar(tensor_scalar_9_post[0:TILE_M, bm_post, 0:BLOCK_N], result_tiles[0:TILE_M, bm_post, 0:BLOCK_N], nl.multiply, operand0=activation_8_tiles[0:TILE_M, bm_post, 0])

            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.dma_copy(
                    dst=result[((TILES_IN_BLOCK_M * m + bm) * TILE_M):((TILES_IN_BLOCK_M * m + bm) * TILE_M) + TILE_M,
                           (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N],
                    src=tensor_scalar_9_post[0:TILE_M, bm, 0:BLOCK_N])

    return result
