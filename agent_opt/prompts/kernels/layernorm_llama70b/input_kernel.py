import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def layernorm(x, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N):
    M, N = x.shape
    output = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)

    TILE_M = 128
    TILE_N = 512
    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N

    assert NUM_BLOCK_M > 0 and NUM_BLOCK_N > 0, \
        "Input size too small for the given tile configuration"

    for m in nl.affine_range(NUM_BLOCK_M):
        scalar_0 = nl.zeros(
            (TILE_M, TILES_IN_BLOCK_M, 1),
            dtype=nl.float32, buffer=nl.sbuf)
        scalar_1 = nl.zeros(
            (TILE_M, TILES_IN_BLOCK_M, 1),
            dtype=nl.float32, buffer=nl.sbuf)

        for n in nl.sequential_range(NUM_BLOCK_N):
            in_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=x.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.dma_copy(
                    dst=in_tiles[0:TILE_M, tile_m, 0:BLOCK_N],
                    src=x[((TILES_IN_BLOCK_M * m + tile_m) * TILE_M):((TILES_IN_BLOCK_M * m + tile_m) * TILE_M) + TILE_M,
                       (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N])

            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                temp_accum_0 = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(temp_accum_0[0:TILE_M, 0:1], nl.add, in_tiles[0:TILE_M, tile_m, 0:BLOCK_N], axis=1, keepdims=True)
                nisa.tensor_tensor(scalar_0[0:TILE_M, tile_m, 0],
                    scalar_0[0:TILE_M, tile_m, 0], temp_accum_0[0:TILE_M, 0:1], op=nl.add)

        for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
            fin0_tensor_scalar_1 = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(fin0_tensor_scalar_1[0:TILE_M, 0:1], scalar_0[0:TILE_M, tile_m, 0], nl.multiply, operand0=0.0001220703125)
            nisa.tensor_copy(scalar_0[0:TILE_M, tile_m, 0], fin0_tensor_scalar_1[0:TILE_M, 0:1])

        for n in nl.sequential_range(NUM_BLOCK_N):
            in_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=x.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.dma_copy(
                    dst=in_tiles[0:TILE_M, tile_m, 0:BLOCK_N],
                    src=x[((TILES_IN_BLOCK_M * m + tile_m) * TILE_M):((TILES_IN_BLOCK_M * m + tile_m) * TILE_M) + TILE_M,
                       (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N])

            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                pre1_tensor_scalar_2 = nl.ndarray((TILE_M, BLOCK_N), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(pre1_tensor_scalar_2[0:TILE_M, 0:BLOCK_N], in_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.subtract, operand0=scalar_0[0:TILE_M, tile_m, 0])
                temp_accum_1 = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
                activation_reduce_3_act = nl.ndarray(pre1_tensor_scalar_2[0:TILE_M, 0:BLOCK_N].shape, dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation_reduce(activation_reduce_3_act, nl.square, pre1_tensor_scalar_2[0:TILE_M, 0:BLOCK_N], nl.add, reduce_res=temp_accum_1[0:TILE_M, 0:1])
                nisa.tensor_tensor(scalar_1[0:TILE_M, tile_m, 0],
                    scalar_1[0:TILE_M, tile_m, 0], temp_accum_1[0:TILE_M, 0:1], op=nl.add)

        for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
            fin1_tensor_scalar_4 = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(fin1_tensor_scalar_4[0:TILE_M, 0:1], scalar_1[0:TILE_M, tile_m, 0], nl.multiply, operand0=0.0001220703125)
            fin1_tensor_scalar_5 = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(fin1_tensor_scalar_5[0:TILE_M, 0:1], fin1_tensor_scalar_4[0:TILE_M, 0:1], nl.add, operand0=1e-06)
            fin1_activation_6 = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(fin1_activation_6[0:TILE_M, 0:1], nl.sqrt, fin1_tensor_scalar_5[0:TILE_M, 0:1])
            fin1_reciprocal_7 = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(fin1_reciprocal_7[0:TILE_M, 0:1], fin1_activation_6[0:TILE_M, 0:1])
            nisa.tensor_copy(scalar_1[0:TILE_M, tile_m, 0], fin1_reciprocal_7[0:TILE_M, 0:1])

        for n in nl.affine_range(NUM_BLOCK_N):
            in_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=x.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.dma_copy(
                    dst=in_tiles[0:TILE_M, tile_m, 0:BLOCK_N],
                    src=x[((TILES_IN_BLOCK_M * m + tile_m) * TILE_M):((TILES_IN_BLOCK_M * m + tile_m) * TILE_M) + TILE_M,
                       (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N])

            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                out_tensor_scalar_2 = nl.ndarray((TILE_M, BLOCK_N), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(out_tensor_scalar_2[0:TILE_M, 0:BLOCK_N], in_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.subtract, operand0=scalar_0[0:TILE_M, tile_m, 0])
                out_tensor_scalar_8 = nl.ndarray((TILE_M, BLOCK_N), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(out_tensor_scalar_8[0:TILE_M, 0:BLOCK_N], out_tensor_scalar_2[0:TILE_M, 0:BLOCK_N], nl.multiply, operand0=scalar_1[0:TILE_M, tile_m, 0])
                nisa.dma_copy(
                    dst=output[((TILES_IN_BLOCK_M * m + tile_m) * TILE_M):((TILES_IN_BLOCK_M * m + tile_m) * TILE_M) + TILE_M,
                           (BLOCK_N * n):(BLOCK_N * n) + BLOCK_N],
                    src=out_tensor_scalar_8[0:TILE_M, 0:BLOCK_N])

    return output
