import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def silu_mlp(x, w1, w2, w3, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K, TILES_IN_BLOCK_P):
    M, K = x.shape
    K_, N = w1.shape
    K_, N_ = w2.shape
    N_, P = w3.shape
    result = nl.ndarray((M, P), dtype=x.dtype, buffer=nl.shared_hbm)

    TILE_M = 128
    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    assert M % BLOCK_M == 0, "M must be divisible by BLOCK_M"
    NUM_BLOCK_M = M // BLOCK_M
    assert NUM_BLOCK_M > 0, "M too small for the tile configuration"

    TILE_N = 128
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    assert N % BLOCK_N == 0, "N must be divisible by BLOCK_N"
    NUM_BLOCK_N = N // BLOCK_N
    assert NUM_BLOCK_N > 0, "N too small for the tile configuration"

    MATMUL_TILE_N = 512
    TILES_IN_MATMUL_BLOCK_N = BLOCK_N // MATMUL_TILE_N
    while TILES_IN_MATMUL_BLOCK_N == 0:
        MATMUL_TILE_N = MATMUL_TILE_N // 2
        TILES_IN_MATMUL_BLOCK_N = BLOCK_N // MATMUL_TILE_N
    assert TILES_IN_MATMUL_BLOCK_N > 0, "Tile configuration results in zero tiles for matmul"

    TILE_K = 128
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K
    assert K % BLOCK_K == 0, "K must be divisible by BLOCK_K"
    NUM_BLOCK_K = K // BLOCK_K
    assert NUM_BLOCK_K > 0, "K too small for the tile configuration"

    TILE_P = 512
    BLOCK_P = TILE_P * TILES_IN_BLOCK_P
    assert P % BLOCK_P == 0, "P must be divisible by BLOCK_P"
    NUM_BLOCK_P = P // BLOCK_P
    assert NUM_BLOCK_P > 0, "P too small for the tile configuration"

    for m in nl.affine_range(NUM_BLOCK_M):
        nc_transpose_5_inter = nl.zeros(
            (TILE_N, NUM_BLOCK_N, TILES_IN_BLOCK_N, BLOCK_M),
            dtype=x.dtype, buffer=nl.sbuf)

        for n in nl.affine_range(NUM_BLOCK_N):
            nc_matmul_1_tiles = (nl.ndarray if NUM_BLOCK_K == 1 else nl.zeros)(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=x.dtype, buffer=nl.sbuf)
            nc_matmul_3_tiles = (nl.ndarray if NUM_BLOCK_K == 1 else nl.zeros)(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=x.dtype, buffer=nl.sbuf)

            for k in nl.sequential_range(NUM_BLOCK_K):
                nc_transpose_0_tiles = nl.ndarray(
                    (TILE_K, TILES_IN_BLOCK_K, BLOCK_M),
                    dtype=x.dtype, buffer=nl.sbuf)
                for b_m in nl.affine_range(TILES_IN_BLOCK_M):
                    for bk_t in nl.affine_range(TILES_IN_BLOCK_K):
                        nc_transpose_0_tiles[0:TILE_K, bk_t, (b_m * TILE_M):(b_m * TILE_M) + TILE_M] = nl.load_transpose2d(
                            x[((TILES_IN_BLOCK_M * m + b_m) * TILE_M):((TILES_IN_BLOCK_M * m + b_m) * TILE_M) + TILE_M,
                                      (BLOCK_K * k + bk_t * TILE_K):(BLOCK_K * k + bk_t * TILE_K) + TILE_K])

                w1_tiles = nl.ndarray(
                    (TILE_K, TILES_IN_BLOCK_K, BLOCK_N),
                    dtype=w1.dtype, buffer=nl.sbuf)
                for bk_t in nl.affine_range(TILES_IN_BLOCK_K):
                    nisa.dma_copy(
                        dst=w1_tiles[0:TILE_K, bk_t, 0:BLOCK_N],
                        src=w1[((TILES_IN_BLOCK_K * k + bk_t) * TILE_K):((TILES_IN_BLOCK_K * k + bk_t) * TILE_K) + TILE_K,
                             (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N])

                w2_tiles = nl.ndarray(
                    (TILE_K, TILES_IN_BLOCK_K, BLOCK_N),
                    dtype=w2.dtype, buffer=nl.sbuf)
                for bk_t in nl.affine_range(TILES_IN_BLOCK_K):
                    nisa.dma_copy(
                        dst=w2_tiles[0:TILE_K, bk_t, 0:BLOCK_N],
                        src=w2[((TILES_IN_BLOCK_K * k + bk_t) * TILE_K):((TILES_IN_BLOCK_K * k + bk_t) * TILE_K) + TILE_K,
                             (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N])

                assert BLOCK_N % MATMUL_TILE_N == 0, "BLOCK_N must be divisible by MATMUL_TILE_N"
                for bm in nl.affine_range(TILES_IN_BLOCK_M):
                    for bn in nl.affine_range(TILES_IN_MATMUL_BLOCK_N):
                        ps_nc_matmul_1 = nl.ndarray(
                            (TILE_M, MATMUL_TILE_N), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(ps_nc_matmul_1[0:TILE_M, 0:MATMUL_TILE_N],
                            nc_transpose_0_tiles[0:TILE_K, 0, (bm * TILE_M):(bm * TILE_M) + TILE_M],
                            w1_tiles[0:TILE_K, 0, (bn * MATMUL_TILE_N):(bn * MATMUL_TILE_N) + MATMUL_TILE_N], accumulate=False)
                        for bk_peel in nl.affine_range(TILES_IN_BLOCK_K - 1):
                            bk = bk_peel + 1
                            nisa.nc_matmul(ps_nc_matmul_1[0:TILE_M, 0:MATMUL_TILE_N],
                                nc_transpose_0_tiles[0:TILE_K, bk, (bm * TILE_M):(bm * TILE_M) + TILE_M],
                                w1_tiles[0:TILE_K, bk, (bn * MATMUL_TILE_N):(bn * MATMUL_TILE_N) + MATMUL_TILE_N], accumulate=True)
                        if NUM_BLOCK_K == 1:
                            nisa.tensor_copy(nc_matmul_1_tiles[0:TILE_M, bm, (bn * MATMUL_TILE_N):(bn * MATMUL_TILE_N) + MATMUL_TILE_N], ps_nc_matmul_1[0:TILE_M, 0:MATMUL_TILE_N])
                        else:
                            nisa.tensor_tensor(nc_matmul_1_tiles[0:TILE_M, bm, (bn * MATMUL_TILE_N):(bn * MATMUL_TILE_N) + MATMUL_TILE_N],
                                nc_matmul_1_tiles[0:TILE_M, bm, (bn * MATMUL_TILE_N):(bn * MATMUL_TILE_N) + MATMUL_TILE_N], ps_nc_matmul_1[0:TILE_M, 0:MATMUL_TILE_N], nl.add)

                assert BLOCK_N % MATMUL_TILE_N == 0, "BLOCK_N must be divisible by MATMUL_TILE_N"
                for bm in nl.affine_range(TILES_IN_BLOCK_M):
                    for bn in nl.affine_range(TILES_IN_MATMUL_BLOCK_N):
                        ps_nc_matmul_3 = nl.ndarray(
                            (TILE_M, MATMUL_TILE_N), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(ps_nc_matmul_3[0:TILE_M, 0:MATMUL_TILE_N],
                            nc_transpose_0_tiles[0:TILE_K, 0, (bm * TILE_M):(bm * TILE_M) + TILE_M],
                            w2_tiles[0:TILE_K, 0, (bn * MATMUL_TILE_N):(bn * MATMUL_TILE_N) + MATMUL_TILE_N], accumulate=False)
                        for bk_peel in nl.affine_range(TILES_IN_BLOCK_K - 1):
                            bk = bk_peel + 1
                            nisa.nc_matmul(ps_nc_matmul_3[0:TILE_M, 0:MATMUL_TILE_N],
                                nc_transpose_0_tiles[0:TILE_K, bk, (bm * TILE_M):(bm * TILE_M) + TILE_M],
                                w2_tiles[0:TILE_K, bk, (bn * MATMUL_TILE_N):(bn * MATMUL_TILE_N) + MATMUL_TILE_N], accumulate=True)
                        if NUM_BLOCK_K == 1:
                            nisa.tensor_copy(nc_matmul_3_tiles[0:TILE_M, bm, (bn * MATMUL_TILE_N):(bn * MATMUL_TILE_N) + MATMUL_TILE_N], ps_nc_matmul_3[0:TILE_M, 0:MATMUL_TILE_N])
                        else:
                            nisa.tensor_tensor(nc_matmul_3_tiles[0:TILE_M, bm, (bn * MATMUL_TILE_N):(bn * MATMUL_TILE_N) + MATMUL_TILE_N],
                                nc_matmul_3_tiles[0:TILE_M, bm, (bn * MATMUL_TILE_N):(bn * MATMUL_TILE_N) + MATMUL_TILE_N], ps_nc_matmul_3[0:TILE_M, 0:MATMUL_TILE_N], nl.add)

            activation_2_inter = nl.ndarray((TILE_M, TILES_IN_BLOCK_M, BLOCK_N), dtype=x.dtype, buffer=nl.sbuf)
            for bm_i in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.activation(activation_2_inter[0:TILE_M, bm_i, 0:BLOCK_N], nl.silu, nc_matmul_1_tiles[0:TILE_M, bm_i, 0:BLOCK_N])

            tensor_tensor_4_inter = nl.ndarray((TILE_M, TILES_IN_BLOCK_M, BLOCK_N), dtype=x.dtype, buffer=nl.sbuf)
            for bm_i in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_tensor(tensor_tensor_4_inter[0:TILE_M, bm_i, 0:BLOCK_N], activation_2_inter[0:TILE_M, bm_i, 0:BLOCK_N], nc_matmul_3_tiles[0:TILE_M, bm_i, 0:BLOCK_N], nl.multiply)

            for bn_t in nl.affine_range(TILES_IN_BLOCK_N):
                for bm_t in nl.affine_range(TILES_IN_BLOCK_M):
                    tp_psum_0 = nl.ndarray((TILE_M, TILE_N), dtype=x.dtype, buffer=nl.psum)
                    nisa.nc_transpose(tp_psum_0[0:TILE_M, 0:TILE_N], tensor_tensor_4_inter[0:TILE_M, bm_t, (bn_t * TILE_N):(bn_t * TILE_N) + TILE_N])
                    nisa.tensor_copy(nc_transpose_5_inter[0:TILE_N, n, bn_t, (bm_t * TILE_M):(bm_t * TILE_M) + TILE_M], tp_psum_0[0:TILE_M, 0:TILE_N])

        for p in nl.affine_range(NUM_BLOCK_P):
            out_tiles = (nl.ndarray if NUM_BLOCK_N == 1 else nl.zeros)(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_P),
                dtype=x.dtype, buffer=nl.sbuf)

            for n in nl.sequential_range(NUM_BLOCK_N):
                w3_tiles = nl.ndarray(
                    (TILE_N, TILES_IN_BLOCK_N, BLOCK_P),
                    dtype=w3.dtype, buffer=nl.sbuf)
                for bn_t in nl.affine_range(TILES_IN_BLOCK_N):
                    nisa.dma_copy(
                        dst=w3_tiles[0:TILE_N, bn_t, 0:BLOCK_P],
                        src=w3[((TILES_IN_BLOCK_N * n + bn_t) * TILE_N):((TILES_IN_BLOCK_N * n + bn_t) * TILE_N) + TILE_N,
                             (BLOCK_P * p):(BLOCK_P * p) + BLOCK_P])

                for bm in nl.affine_range(TILES_IN_BLOCK_M):
                    for bp in nl.affine_range(TILES_IN_BLOCK_P):
                        ps_nc_matmul_6 = nl.ndarray(
                            (TILE_M, TILE_P), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(ps_nc_matmul_6[0:TILE_M, 0:TILE_P],
                            nc_transpose_5_inter[0:TILE_N, n, 0, (bm * TILE_M):(bm * TILE_M) + TILE_M],
                            w3_tiles[0:TILE_N, 0, (bp * TILE_P):(bp * TILE_P) + TILE_P], accumulate=False)
                        for bn_peel in nl.affine_range(TILES_IN_BLOCK_N - 1):
                            bn = bn_peel + 1
                            nisa.nc_matmul(ps_nc_matmul_6[0:TILE_M, 0:TILE_P],
                                nc_transpose_5_inter[0:TILE_N, n, bn, (bm * TILE_M):(bm * TILE_M) + TILE_M],
                                w3_tiles[0:TILE_N, bn, (bp * TILE_P):(bp * TILE_P) + TILE_P], accumulate=True)
                        if NUM_BLOCK_N == 1:
                            nisa.tensor_copy(out_tiles[0:TILE_M, bm, (bp * TILE_P):(bp * TILE_P) + TILE_P], ps_nc_matmul_6[0:TILE_M, 0:TILE_P])
                        else:
                            nisa.tensor_tensor(out_tiles[0:TILE_M, bm, (bp * TILE_P):(bp * TILE_P) + TILE_P],
                                out_tiles[0:TILE_M, bm, (bp * TILE_P):(bp * TILE_P) + TILE_P], ps_nc_matmul_6[0:TILE_M, 0:TILE_P], nl.add)

            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.dma_copy(
                    dst=result[((TILES_IN_BLOCK_M * m + bm) * TILE_M):((TILES_IN_BLOCK_M * m + bm) * TILE_M) + TILE_M,
                           (BLOCK_P * p):(BLOCK_P * p) + BLOCK_P],
                    src=out_tiles[0:TILE_M, bm, 0:BLOCK_P])

    return result
