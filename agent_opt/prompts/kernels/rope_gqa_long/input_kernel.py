import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def rope(x_even, x_odd, cos, sin, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N):
    TILE_M = 128
    TILE_N = 512
    BLOCK_M = TILES_IN_BLOCK_M * TILE_M
    BLOCK_N = TILES_IN_BLOCK_N * TILE_N

    NUM_BLOCK_M = x_even.shape[0] // BLOCK_M
    NUM_BLOCK_N = x_even.shape[1] // BLOCK_N

    assert NUM_BLOCK_M > 0 and NUM_BLOCK_N > 0, \
        "Input size too small for the given tile configuration"

    out_0 = nl.ndarray(
        x_even.shape, dtype=x_even.dtype, buffer=nl.shared_hbm)
    out_1 = nl.ndarray(
        x_even.shape, dtype=x_even.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(NUM_BLOCK_M):
        for n in nl.affine_range(NUM_BLOCK_N):

            x_even_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=x_even.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                row = (TILES_IN_BLOCK_M * m + tile_m) * TILE_M
                nisa.dma_copy(
                    dst=x_even_tiles[0:TILE_M, tile_m, 0:BLOCK_N],
                    src=x_even[row:row + TILE_M, BLOCK_N * n:BLOCK_N * n + BLOCK_N])

            x_odd_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=x_odd.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                row = (TILES_IN_BLOCK_M * m + tile_m) * TILE_M
                nisa.dma_copy(
                    dst=x_odd_tiles[0:TILE_M, tile_m, 0:BLOCK_N],
                    src=x_odd[row:row + TILE_M, BLOCK_N * n:BLOCK_N * n + BLOCK_N])

            cos_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=cos.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                row = (TILES_IN_BLOCK_M * m + tile_m) * TILE_M
                nisa.dma_copy(
                    dst=cos_tiles[0:TILE_M, tile_m, 0:BLOCK_N],
                    src=cos[row:row + TILE_M, BLOCK_N * n:BLOCK_N * n + BLOCK_N])

            sin_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=sin.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                row = (TILES_IN_BLOCK_M * m + tile_m) * TILE_M
                nisa.dma_copy(
                    dst=sin_tiles[0:TILE_M, tile_m, 0:BLOCK_N],
                    src=sin[row:row + TILE_M, BLOCK_N * n:BLOCK_N * n + BLOCK_N])

            tensor_tensor_0_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=x_even.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_tensor(tensor_tensor_0_tiles[0:TILE_M, tile_m, 0:BLOCK_N], cos_tiles[0:TILE_M, tile_m, 0:BLOCK_N], x_even_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.multiply)

            tensor_tensor_1_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=x_even.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_tensor(tensor_tensor_1_tiles[0:TILE_M, tile_m, 0:BLOCK_N], sin_tiles[0:TILE_M, tile_m, 0:BLOCK_N], x_odd_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.multiply)

            tensor_tensor_3_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=x_even.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_tensor(tensor_tensor_3_tiles[0:TILE_M, tile_m, 0:BLOCK_N], x_even_tiles[0:TILE_M, tile_m, 0:BLOCK_N], sin_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.multiply)

            tensor_tensor_4_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=x_even.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_tensor(tensor_tensor_4_tiles[0:TILE_M, tile_m, 0:BLOCK_N], cos_tiles[0:TILE_M, tile_m, 0:BLOCK_N], x_odd_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.multiply)

            tensor_tensor_2_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=x_even.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_tensor(tensor_tensor_2_tiles[0:TILE_M, tile_m, 0:BLOCK_N], tensor_tensor_0_tiles[0:TILE_M, tile_m, 0:BLOCK_N], tensor_tensor_1_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.subtract)

            tensor_tensor_5_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=x_even.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_tensor(tensor_tensor_5_tiles[0:TILE_M, tile_m, 0:BLOCK_N], tensor_tensor_3_tiles[0:TILE_M, tile_m, 0:BLOCK_N], tensor_tensor_4_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.add)

            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                row = (TILES_IN_BLOCK_M * m + tile_m) * TILE_M
                nisa.dma_copy(
                    dst=out_0[row:row + TILE_M, BLOCK_N * n:BLOCK_N * n + BLOCK_N],
                    src=tensor_tensor_2_tiles[0:TILE_M, tile_m, 0:BLOCK_N])
                nisa.dma_copy(
                    dst=out_1[row:row + TILE_M, BLOCK_N * n:BLOCK_N * n + BLOCK_N],
                    src=tensor_tensor_5_tiles[0:TILE_M, tile_m, 0:BLOCK_N])

    return (out_0, out_1)
