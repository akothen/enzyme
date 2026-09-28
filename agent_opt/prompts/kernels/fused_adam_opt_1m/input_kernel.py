import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def fused_adam(param, grad, exp_avg, exp_avg_sq, step_size, inv_bc2_sqrt, wd, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N):
    TILE_M = 128
    TILE_N = 512
    BLOCK_M = TILES_IN_BLOCK_M * TILE_M
    BLOCK_N = TILES_IN_BLOCK_N * TILE_N

    NUM_BLOCK_M = param.shape[0] // BLOCK_M
    NUM_BLOCK_N = param.shape[1] // BLOCK_N

    assert NUM_BLOCK_M > 0 and NUM_BLOCK_N > 0, \
        "Input size too small for the given tile configuration"

    out_0 = nl.ndarray(
        param.shape, dtype=param.dtype, buffer=nl.shared_hbm)
    out_1 = nl.ndarray(
        param.shape, dtype=param.dtype, buffer=nl.shared_hbm)
    out_2 = nl.ndarray(
        param.shape, dtype=param.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(NUM_BLOCK_M):
        for n in nl.affine_range(NUM_BLOCK_N):

            param_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            grad_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=grad.dtype, buffer=nl.sbuf)
            exp_avg_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=exp_avg.dtype, buffer=nl.sbuf)
            exp_avg_sq_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=exp_avg_sq.dtype, buffer=nl.sbuf)
            step_size_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, 1),
                dtype=nl.float32, buffer=nl.sbuf)
            inv_bc2_sqrt_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, 1),
                dtype=nl.float32, buffer=nl.sbuf)
            wd_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, 1),
                dtype=nl.float32, buffer=nl.sbuf)

            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                row = (TILES_IN_BLOCK_M * m + tile_m) * TILE_M
                nisa.dma_copy(
                    dst=param_tiles[0:TILE_M, tile_m, 0:BLOCK_N],
                    src=param[row:row + TILE_M, BLOCK_N * n:BLOCK_N * n + BLOCK_N])
                nisa.dma_copy(
                    dst=grad_tiles[0:TILE_M, tile_m, 0:BLOCK_N],
                    src=grad[row:row + TILE_M, BLOCK_N * n:BLOCK_N * n + BLOCK_N])
                nisa.dma_copy(
                    dst=exp_avg_tiles[0:TILE_M, tile_m, 0:BLOCK_N],
                    src=exp_avg[row:row + TILE_M, BLOCK_N * n:BLOCK_N * n + BLOCK_N])
                nisa.dma_copy(
                    dst=exp_avg_sq_tiles[0:TILE_M, tile_m, 0:BLOCK_N],
                    src=exp_avg_sq[row:row + TILE_M, BLOCK_N * n:BLOCK_N * n + BLOCK_N])
                if str(step_size.dtype) == 'float32':
                    nisa.dma_copy(
                        dst=step_size_tiles[0:TILE_M, tile_m, 0:1],
                        src=step_size[row:row + TILE_M, 0:1])
                else:
                    step_size_tiles_in = nl.ndarray(
                        (TILE_M, 1), dtype=step_size.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=step_size_tiles_in[0:TILE_M, 0:1],
                        src=step_size[row:row + TILE_M, 0:1])
                    nisa.tensor_copy(
                        step_size_tiles[0:TILE_M, tile_m, 0:1], step_size_tiles_in[0:TILE_M, 0:1])
                if str(inv_bc2_sqrt.dtype) == 'float32':
                    nisa.dma_copy(
                        dst=inv_bc2_sqrt_tiles[0:TILE_M, tile_m, 0:1],
                        src=inv_bc2_sqrt[row:row + TILE_M, 0:1])
                else:
                    inv_bc2_sqrt_tiles_in = nl.ndarray(
                        (TILE_M, 1), dtype=inv_bc2_sqrt.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=inv_bc2_sqrt_tiles_in[0:TILE_M, 0:1],
                        src=inv_bc2_sqrt[row:row + TILE_M, 0:1])
                    nisa.tensor_copy(
                        inv_bc2_sqrt_tiles[0:TILE_M, tile_m, 0:1], inv_bc2_sqrt_tiles_in[0:TILE_M, 0:1])
                if str(wd.dtype) == 'float32':
                    nisa.dma_copy(
                        dst=wd_tiles[0:TILE_M, tile_m, 0:1],
                        src=wd[row:row + TILE_M, 0:1])
                else:
                    wd_tiles_in = nl.ndarray(
                        (TILE_M, 1), dtype=wd.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=wd_tiles_in[0:TILE_M, 0:1],
                        src=wd[row:row + TILE_M, 0:1])
                    nisa.tensor_copy(
                        wd_tiles[0:TILE_M, tile_m, 0:1], wd_tiles_in[0:TILE_M, 0:1])

            tensor_scalar_0_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_scalar(tensor_scalar_0_tiles[0:TILE_M, tile_m, 0:BLOCK_N], exp_avg_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.multiply, operand0=0.9)

            tensor_scalar_1_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_scalar(tensor_scalar_1_tiles[0:TILE_M, tile_m, 0:BLOCK_N], exp_avg_sq_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.multiply, operand0=0.999)

            scalar_tensor_tensor_2_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.scalar_tensor_tensor(dst=scalar_tensor_tensor_2_tiles[0:TILE_M, tile_m, 0:BLOCK_N], data=param_tiles[0:TILE_M, tile_m, 0:BLOCK_N], op0=nl.multiply, operand0=wd_tiles[0:TILE_M, tile_m, 0:1], op1=nl.add, operand1=grad_tiles[0:TILE_M, tile_m, 0:BLOCK_N])

            tensor_tensor_3_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_tensor(tensor_tensor_3_tiles[0:TILE_M, tile_m, 0:BLOCK_N], scalar_tensor_tensor_2_tiles[0:TILE_M, tile_m, 0:BLOCK_N], scalar_tensor_tensor_2_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.multiply)

            tensor_scalar_11_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_scalar(tensor_scalar_11_tiles[0:TILE_M, tile_m, 0:BLOCK_N], scalar_tensor_tensor_2_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.multiply, operand0=0.09999999999999998)

            tensor_scalar_4_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_scalar(tensor_scalar_4_tiles[0:TILE_M, tile_m, 0:BLOCK_N], tensor_tensor_3_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.multiply, operand0=0.0010000000000000009)

            tensor_tensor_16_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_tensor(tensor_tensor_16_tiles[0:TILE_M, tile_m, 0:BLOCK_N], tensor_scalar_0_tiles[0:TILE_M, tile_m, 0:BLOCK_N], tensor_scalar_11_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.add)

            tensor_tensor_5_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_tensor(tensor_tensor_5_tiles[0:TILE_M, tile_m, 0:BLOCK_N], tensor_scalar_1_tiles[0:TILE_M, tile_m, 0:BLOCK_N], tensor_scalar_4_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.add)

            activation_6_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.activation(activation_6_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.sqrt, tensor_tensor_5_tiles[0:TILE_M, tile_m, 0:BLOCK_N])

            tensor_scalar_7_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_scalar(tensor_scalar_7_tiles[0:TILE_M, tile_m, 0:BLOCK_N], activation_6_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.multiply, operand0=inv_bc2_sqrt_tiles[0:TILE_M, tile_m, 0:1])

            tensor_scalar_8_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_scalar(tensor_scalar_8_tiles[0:TILE_M, tile_m, 0:BLOCK_N], tensor_scalar_7_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.add, operand0=1e-08)

            activation_9_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.activation(activation_9_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.reciprocal, tensor_scalar_8_tiles[0:TILE_M, tile_m, 0:BLOCK_N])

            tensor_tensor_10_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_tensor(tensor_tensor_10_tiles[0:TILE_M, tile_m, 0:BLOCK_N], tensor_scalar_0_tiles[0:TILE_M, tile_m, 0:BLOCK_N], activation_9_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.multiply)

            tensor_tensor_12_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_tensor(tensor_tensor_12_tiles[0:TILE_M, tile_m, 0:BLOCK_N], activation_9_tiles[0:TILE_M, tile_m, 0:BLOCK_N], tensor_scalar_11_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.multiply)

            tensor_tensor_13_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_tensor(tensor_tensor_13_tiles[0:TILE_M, tile_m, 0:BLOCK_N], tensor_tensor_10_tiles[0:TILE_M, tile_m, 0:BLOCK_N], tensor_tensor_12_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.add)

            tensor_scalar_14_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_scalar(tensor_scalar_14_tiles[0:TILE_M, tile_m, 0:BLOCK_N], tensor_tensor_13_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.multiply, operand0=step_size_tiles[0:TILE_M, tile_m, 0:1])

            tensor_tensor_15_tiles = nl.ndarray(
                (TILE_M, TILES_IN_BLOCK_M, BLOCK_N),
                dtype=param.dtype, buffer=nl.sbuf)
            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                nisa.tensor_tensor(tensor_tensor_15_tiles[0:TILE_M, tile_m, 0:BLOCK_N], param_tiles[0:TILE_M, tile_m, 0:BLOCK_N], tensor_scalar_14_tiles[0:TILE_M, tile_m, 0:BLOCK_N], nl.subtract)

            for tile_m in nl.affine_range(TILES_IN_BLOCK_M):
                row = (TILES_IN_BLOCK_M * m + tile_m) * TILE_M
                nisa.dma_copy(
                    dst=out_0[row:row + TILE_M, BLOCK_N * n:BLOCK_N * n + BLOCK_N],
                    src=tensor_tensor_15_tiles[0:TILE_M, tile_m, 0:BLOCK_N])
                nisa.dma_copy(
                    dst=out_1[row:row + TILE_M, BLOCK_N * n:BLOCK_N * n + BLOCK_N],
                    src=tensor_tensor_16_tiles[0:TILE_M, tile_m, 0:BLOCK_N])
                nisa.dma_copy(
                    dst=out_2[row:row + TILE_M, BLOCK_N * n:BLOCK_N * n + BLOCK_N],
                    src=tensor_tensor_5_tiles[0:TILE_M, tile_m, 0:BLOCK_N])

    return (out_0, out_1, out_2)
