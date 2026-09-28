import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def linear_attention(q, key, v, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K, TILES_IN_BLOCK_P):
    M, P = q.shape
    K, P_ = key.shape
    K_, N = v.shape
    result = nl.ndarray((M, N), dtype=q.dtype, buffer=nl.shared_hbm)

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

    TILE_P = 128
    BLOCK_P = TILE_P * TILES_IN_BLOCK_P
    assert P % BLOCK_P == 0, "P must be divisible by BLOCK_P"
    NUM_BLOCK_P = P // BLOCK_P
    assert NUM_BLOCK_P > 0, "P too small for the tile configuration"

    nc_matmul_1_hbm = nl.ndarray(
        (P, N), dtype=q.dtype,
        buffer=nl.shared_hbm)

    for p in nl.affine_range(NUM_BLOCK_P):
        for n in nl.affine_range(NUM_BLOCK_N):
            nc_matmul_1_tiles = (nl.ndarray if NUM_BLOCK_K == 1 else nl.zeros)(
                (TILE_P, TILES_IN_BLOCK_P, BLOCK_N),
                dtype=q.dtype, buffer=nl.sbuf)

            for k in nl.sequential_range(NUM_BLOCK_K):
                key_tiles = nl.ndarray(
                    (TILE_K, TILES_IN_BLOCK_K, BLOCK_P),
                    dtype=key.dtype, buffer=nl.sbuf)
                for bk_t in nl.affine_range(TILES_IN_BLOCK_K):
                    nisa.dma_copy(
                        dst=key_tiles[0:TILE_K, bk_t, 0:BLOCK_P],
                        src=key[((TILES_IN_BLOCK_K * k + bk_t) * TILE_K):((TILES_IN_BLOCK_K * k + bk_t) * TILE_K) + TILE_K,
                             (BLOCK_P * p):(BLOCK_P * p) + BLOCK_P])

                v_tiles = nl.ndarray(
                    (TILE_K, TILES_IN_BLOCK_K, BLOCK_N),
                    dtype=v.dtype, buffer=nl.sbuf)
                for bk_t in nl.affine_range(TILES_IN_BLOCK_K):
                    nisa.dma_copy(
                        dst=v_tiles[0:TILE_K, bk_t, 0:BLOCK_N],
                        src=v[((TILES_IN_BLOCK_K * k + bk_t) * TILE_K):((TILES_IN_BLOCK_K * k + bk_t) * TILE_K) + TILE_K,
                             (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N])

                for bp in nl.affine_range(TILES_IN_BLOCK_P):
                    for bn in nl.affine_range(TILES_IN_BLOCK_N):
                        ps_nc_matmul_1 = nl.ndarray(
                            (TILE_P, TILE_N), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(ps_nc_matmul_1[0:TILE_P, 0:TILE_N],
                            key_tiles[0:TILE_K, 0, (bp * TILE_P):(bp * TILE_P) + TILE_P],
                            v_tiles[0:TILE_K, 0, (bn * TILE_N):(bn * TILE_N) + TILE_N], accumulate=False)
                        for bk_peel in nl.affine_range(TILES_IN_BLOCK_K - 1):
                            bk = bk_peel + 1
                            nisa.nc_matmul(ps_nc_matmul_1[0:TILE_P, 0:TILE_N],
                                key_tiles[0:TILE_K, bk, (bp * TILE_P):(bp * TILE_P) + TILE_P],
                                v_tiles[0:TILE_K, bk, (bn * TILE_N):(bn * TILE_N) + TILE_N], accumulate=True)
                        if NUM_BLOCK_K == 1:
                            nisa.tensor_copy(nc_matmul_1_tiles[0:TILE_P, bp, (bn * TILE_N):(bn * TILE_N) + TILE_N], ps_nc_matmul_1[0:TILE_P, 0:TILE_N])
                        else:
                            nisa.tensor_tensor(nc_matmul_1_tiles[0:TILE_P, bp, (bn * TILE_N):(bn * TILE_N) + TILE_N],
                                nc_matmul_1_tiles[0:TILE_P, bp, (bn * TILE_N):(bn * TILE_N) + TILE_N], ps_nc_matmul_1[0:TILE_P, 0:TILE_N], nl.add)

            for bp_store in nl.affine_range(TILES_IN_BLOCK_P):
                nisa.dma_copy(
                    dst=nc_matmul_1_hbm[((TILES_IN_BLOCK_P * p + bp_store) * TILE_P):
                        ((TILES_IN_BLOCK_P * p + bp_store) * TILE_P) + TILE_P,
                        (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N],
                    src=nc_matmul_1_tiles[0:TILE_P, bp_store, 0:BLOCK_N])

    for m in nl.affine_range(NUM_BLOCK_M):
        for n in nl.affine_range(NUM_BLOCK_N):
            nc_matmul_2_tiles = (nl.ndarray if NUM_BLOCK_P == 1 else nl.zeros)(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=q.dtype, buffer=nl.sbuf)

            for p in nl.sequential_range(NUM_BLOCK_P):
                nc_transpose_0_tiles = nl.ndarray(
                    (TILE_P, TILES_IN_BLOCK_P, BLOCK_M),
                    dtype=q.dtype, buffer=nl.sbuf)
                for b_m in nl.affine_range(TILES_IN_BLOCK_M):
                    for bp_t in nl.affine_range(TILES_IN_BLOCK_P):
                        nc_transpose_0_tiles[0:TILE_P, bp_t, (b_m * TILE_M):(b_m * TILE_M) + TILE_M] = nl.load_transpose2d(
                            q[((TILES_IN_BLOCK_M * m + b_m) * TILE_M):((TILES_IN_BLOCK_M * m + b_m) * TILE_M) + TILE_M,
                                      (BLOCK_P * p + bp_t * TILE_P):(BLOCK_P * p + bp_t * TILE_P) + TILE_P])

                nc_matmul_1_tiles = nl.ndarray(
                    (TILE_P, TILES_IN_BLOCK_P, BLOCK_N),
                    dtype=nc_matmul_1_hbm.dtype, buffer=nl.sbuf)
                for bp_t in nl.affine_range(TILES_IN_BLOCK_P):
                    nisa.dma_copy(
                        dst=nc_matmul_1_tiles[0:TILE_P, bp_t, 0:BLOCK_N],
                        src=nc_matmul_1_hbm[((TILES_IN_BLOCK_P * p + bp_t) * TILE_P):((TILES_IN_BLOCK_P * p + bp_t) * TILE_P) + TILE_P,
                             (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N])

                for bm in nl.affine_range(TILES_IN_BLOCK_M):
                    for bn in nl.affine_range(TILES_IN_BLOCK_N):
                        ps_nc_matmul_2 = nl.ndarray(
                            (TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(ps_nc_matmul_2[0:TILE_M, 0:TILE_N],
                            nc_transpose_0_tiles[0:TILE_P, 0, (bm * TILE_M):(bm * TILE_M) + TILE_M],
                            nc_matmul_1_tiles[0:TILE_P, 0, (bn * TILE_N):(bn * TILE_N) + TILE_N], accumulate=False)
                        for bp_peel in nl.affine_range(TILES_IN_BLOCK_P - 1):
                            bp = bp_peel + 1
                            nisa.nc_matmul(ps_nc_matmul_2[0:TILE_M, 0:TILE_N],
                                nc_transpose_0_tiles[0:TILE_P, bp, (bm * TILE_M):(bm * TILE_M) + TILE_M],
                                nc_matmul_1_tiles[0:TILE_P, bp, (bn * TILE_N):(bn * TILE_N) + TILE_N], accumulate=True)
                        if NUM_BLOCK_P == 1:
                            nisa.tensor_copy(nc_matmul_2_tiles[0:TILE_M, bm, (bn * TILE_N):(bn * TILE_N) + TILE_N], ps_nc_matmul_2[0:TILE_M, 0:TILE_N])
                        else:
                            nisa.tensor_tensor(nc_matmul_2_tiles[0:TILE_M, bm, (bn * TILE_N):(bn * TILE_N) + TILE_N],
                                nc_matmul_2_tiles[0:TILE_M, bm, (bn * TILE_N):(bn * TILE_N) + TILE_N], ps_nc_matmul_2[0:TILE_M, 0:TILE_N], nl.add)

            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.dma_copy(
                    dst=result[((TILES_IN_BLOCK_M * m + bm) * TILE_M):((TILES_IN_BLOCK_M * m + bm) * TILE_M) + TILE_M,
                           (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N],
                    src=nc_matmul_2_tiles[0:TILE_M, bm, 0:BLOCK_N])

    return result
