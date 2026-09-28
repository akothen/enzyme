import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def softmax_mm(x, w, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K):
    M, K = x.shape
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
        tensor_reduce_1_tiles = nl.zeros(
            (TILE_M, TILES_IN_BLOCK_M, 1),
            dtype=nl.float32, buffer=nl.sbuf)
        exponential_0_staged = nl.ndarray(
            (TILE_M, NUM_BLOCK_K, TILES_IN_BLOCK_M, BLOCK_K),
            dtype=x.dtype, buffer=nl.sbuf)
        for k_side in nl.sequential_range(NUM_BLOCK_K):
            x_tensor_reduce_1_chain_tile_k = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_K), dtype=x.dtype, buffer=nl.sbuf)
            for bm_red in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.dma_copy(
                    dst=x_tensor_reduce_1_chain_tile_k[0:TILE_M, bm_red, 0:BLOCK_K],
                    src=x[((TILES_IN_BLOCK_M * m + bm_red) * TILE_M):((TILES_IN_BLOCK_M * m + bm_red) * TILE_M) + TILE_M,
                           (BLOCK_K * k_side):(BLOCK_K * k_side) + BLOCK_K])
            for bm_red in nl.affine_range(TILES_IN_BLOCK_M):
                partial_tensor_reduce_1 = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(exponential_0_staged[0:TILE_M, k_side, bm_red, 0:BLOCK_K], nl.exp, x_tensor_reduce_1_chain_tile_k[0:TILE_M, bm_red, 0:BLOCK_K], reduce_op=nl.add, reduce_res=partial_tensor_reduce_1[0:TILE_M, 0:1])
                nisa.tensor_tensor(tensor_reduce_1_tiles[0:TILE_M, bm_red, 0],
                    tensor_reduce_1_tiles[0:TILE_M, bm_red, 0], partial_tensor_reduce_1[0:TILE_M, 0:1], op=nl.add)

        reciprocal_2_tiles = nl.ndarray(
            (TILE_M, TILES_IN_BLOCK_M, 1),
            dtype=nl.float32, buffer=nl.sbuf)
        for bm_side in nl.affine_range(TILES_IN_BLOCK_M):
            nisa.reciprocal(reciprocal_2_tiles[0:TILE_M, bm_side, 0], tensor_reduce_1_tiles[0:TILE_M, bm_side, 0])

        for n in nl.affine_range(NUM_BLOCK_N):
            result_tiles = (nl.ndarray if NUM_BLOCK_K == 1 else nl.zeros)(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=x.dtype, buffer=nl.sbuf)

            for k in nl.sequential_range(NUM_BLOCK_K):
                nc_transpose_4_tiles = nl.ndarray(
                    (TILE_K, TILES_IN_BLOCK_K, BLOCK_M),
                    dtype=x.dtype, buffer=nl.sbuf)
                for b_m in nl.affine_range(TILES_IN_BLOCK_M):
                    nc_transpose_4_tiles_c0 = nl.ndarray((TILE_M, BLOCK_K), dtype=x.dtype, buffer=nl.sbuf)
                    nisa.tensor_scalar(nc_transpose_4_tiles_c0[0:TILE_M, 0:BLOCK_K], exponential_0_staged[0:TILE_M, k, b_m, 0:BLOCK_K], nl.multiply, operand0=reciprocal_2_tiles[0:TILE_M, b_m, 0])
                    for bk_t in nl.affine_range(TILES_IN_BLOCK_K):
                        tp_psum_0 = nl.ndarray((TILE_K, TILE_M), dtype=x.dtype, buffer=nl.psum)
                        nisa.nc_transpose(tp_psum_0[0:TILE_K, 0:TILE_M], nc_transpose_4_tiles_c0[0:TILE_M, (bk_t * TILE_K):(bk_t * TILE_K) + TILE_K])
                        nisa.tensor_copy(nc_transpose_4_tiles[0:TILE_K, bk_t, (b_m * TILE_M):(b_m * TILE_M) + TILE_M], tp_psum_0[0:TILE_K, 0:TILE_M])

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
                            nc_transpose_4_tiles[0:TILE_K, 0, (bm * TILE_M):(bm * TILE_M) + TILE_M],
                            w_tiles[0:TILE_K, 0, (bn * TILE_N):(bn * TILE_N) + TILE_N], accumulate=False)
                        for bk_peel in nl.affine_range(TILES_IN_BLOCK_K - 1):
                            bk = bk_peel + 1
                            nisa.nc_matmul(res_tile[0:TILE_M, 0:TILE_N],
                                nc_transpose_4_tiles[0:TILE_K, bk, (bm * TILE_M):(bm * TILE_M) + TILE_M],
                                w_tiles[0:TILE_K, bk, (bn * TILE_N):(bn * TILE_N) + TILE_N], accumulate=True)
                        if NUM_BLOCK_K == 1:
                            nisa.tensor_copy(result_tiles[0:TILE_M, bm, (bn * TILE_N):(bn * TILE_N) + TILE_N], res_tile[0:TILE_M, 0:TILE_N])
                        else:
                            nisa.tensor_tensor(result_tiles[0:TILE_M, bm, (bn * TILE_N):(bn * TILE_N) + TILE_N],
                                result_tiles[0:TILE_M, bm, (bn * TILE_N):(bn * TILE_N) + TILE_N], res_tile[0:TILE_M, 0:TILE_N], nl.add)

            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.dma_copy(
                    dst=result[((TILES_IN_BLOCK_M * m + bm) * TILE_M):((TILES_IN_BLOCK_M * m + bm) * TILE_M) + TILE_M,
                           (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N],
                    src=result_tiles[0:TILE_M, bm, 0:BLOCK_N])

    return result
