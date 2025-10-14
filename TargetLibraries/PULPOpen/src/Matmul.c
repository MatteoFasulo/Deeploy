/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pmsis.h"

#include "DeeployPULPMath.h"

void PULP_MatMul_fp32_fp32_fp32_unroll1x7(const float32_t *__restrict__ pSrcA,
                                          const float32_t *__restrict__ pSrcB,
                                          float32_t *__restrict__ pDstY,
                                          uint32_t M, uint32_t N, uint32_t O) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = log2(NUM_CORES);

  uint32_t M_chunk = (M >> log2Core) + ((M & (NUM_CORES - 1)) != 0);
  uint32_t M_start = MIN(core_id * M_chunk, M);
  uint32_t M_end = MIN(M_start + M_chunk, M);
  uint32_t M_size = M_end - M_start;

  if (M_size == 0) {
    return;
  }

  const float32_t *local_pSrcA = pSrcA + M_start * N;
  float32_t *local_pDstY = pDstY + M_start * O;

  uint32_t O_block = O - (O % 7);

  for (uint32_t i = 0; i < M_size; i++) {

    for (uint32_t j = 0; j < O_block; j += 7) {
      float32_t sum0 = 0.0f;
      float32_t sum1 = 0.0f;
      float32_t sum2 = 0.0f;
      float32_t sum3 = 0.0f;
      float32_t sum4 = 0.0f;
      float32_t sum5 = 0.0f;
      float32_t sum6 = 0.0f;

      for (uint32_t k = 0; k < N; k++) {
        float32_t a0 = local_pSrcA[i * N + k];

        float32_t b0 = pSrcB[k * O + (j + 0)];
        float32_t b1 = pSrcB[k * O + (j + 1)];
        float32_t b2 = pSrcB[k * O + (j + 2)];
        float32_t b3 = pSrcB[k * O + (j + 3)];
        float32_t b4 = pSrcB[k * O + (j + 4)];
        float32_t b5 = pSrcB[k * O + (j + 5)];
        float32_t b6 = pSrcB[k * O + (j + 6)];

        sum0 += a0 * b0;
        sum1 += a0 * b1;
        sum2 += a0 * b2;
        sum3 += a0 * b3;
        sum4 += a0 * b4;
        sum5 += a0 * b5;
        sum6 += a0 * b6;
      }

      local_pDstY[i * O + (j + 0)] = sum0;
      local_pDstY[i * O + (j + 1)] = sum1;
      local_pDstY[i * O + (j + 2)] = sum2;
      local_pDstY[i * O + (j + 3)] = sum3;
      local_pDstY[i * O + (j + 4)] = sum4;
      local_pDstY[i * O + (j + 5)] = sum5;
      local_pDstY[i * O + (j + 6)] = sum6;
    }

    for (uint32_t j = O_block; j < O; j++) {
      float32_t sum = 0.0f;

      for (uint32_t k = 0; k < N; k++) {
        float32_t a_val = local_pSrcA[i * N + k];
        float32_t b_val = pSrcB[k * O + j];
        sum += a_val * b_val;
      }

      local_pDstY[i * O + j] = sum;
    }
  }
}

// Parameter structure for multi-core MatMul
typedef struct {
  const int8_t *pSrcA;
  const int8_t *pSrcB;
  int32_t *pDstY;
  uint32_t M, N, O;
  int32_t A_offset, B_offset, C_offset;
} DeeployPULPMatMulParams;

// Direct integer MatMul function matching FP32 performance
void PULP_MatMul_s8_s8_s32_unroll1x7(void *args) {
  // Extract parameters from the struct (same as FP32)
  struct {
    const int8_t *pSrcA;
    const int8_t *pSrcB;
    int32_t *pDstY;
    uint32_t M, N, O;
  } *params = (struct {
    const int8_t *pSrcA;
    const int8_t *pSrcB;
    int32_t *pDstY;
    uint32_t M, N, O;
  } *)args;

  const int8_t *pSrcA = params->pSrcA;
  const int8_t *pSrcB = params->pSrcB;
  int32_t *pDstY = params->pDstY;
  uint32_t M = params->M;
  uint32_t N = params->N;
  uint32_t O = params->O;
  int8_t core_id = pi_core_id();

  // Better work distribution: ensure all cores get work
  uint32_t M_chunk = (M + NUM_CORES - 1) / NUM_CORES; // Ceiling division
  uint32_t M_start = core_id * M_chunk;
  uint32_t M_end = MIN(M_start + M_chunk, M);
  uint32_t M_size = M_end - M_start;

  if (M_size == 0 || M_start >= M) {
    return;
  }

  const int8_t *local_pSrcA = pSrcA + M_start * N;
  int32_t *local_pDstY = pDstY + M_start * O;

  // Optimized row-wise parallelization across 8 cores
  uint32_t O_block = O - (O % 7);

  for (uint32_t i = 0; i < M_size; i++) {
    for (uint32_t j = 0; j < O_block; j += 7) {
      register int32_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
      register int32_t sum4 = 0, sum5 = 0, sum6 = 0;

      // Highly optimized inner loop
      const int8_t *a_row = local_pSrcA + i * N;
      for (uint32_t k = 0; k < N; k++) {
        register int32_t a_val = a_row[k];
        register const int8_t *b_col = pSrcB + k * O + j;

        sum0 += a_val * b_col[0];
        sum1 += a_val * b_col[1];
        sum2 += a_val * b_col[2];
        sum3 += a_val * b_col[3];
        sum4 += a_val * b_col[4];
        sum5 += a_val * b_col[5];
        sum6 += a_val * b_col[6];
      }

      local_pDstY[i * O + (j + 0)] = sum0;
      local_pDstY[i * O + (j + 1)] = sum1;
      local_pDstY[i * O + (j + 2)] = sum2;
      local_pDstY[i * O + (j + 3)] = sum3;
      local_pDstY[i * O + (j + 4)] = sum4;
      local_pDstY[i * O + (j + 5)] = sum5;
      local_pDstY[i * O + (j + 6)] = sum6;
    }

    // Handle remaining columns
    for (uint32_t j = O_block; j < O; j++) {
      register int32_t sum = 0;
      const int8_t *a_row = local_pSrcA + i * N;

      for (uint32_t k = 0; k < N; k++) {
        sum += a_row[k] * pSrcB[k * O + j];
      }

      local_pDstY[i * O + j] = sum;
    }
  }
}

// Multi-core optimized integer MatMul functions with SIMD vectorization
void PULP_MatMul_s8_s8_s32(void *args) {
  DeeployPULPMatMulParams *params = (DeeployPULPMatMulParams *)args;

  int8_t core_id = pi_core_id();

  // Optimized work distribution for 8 cores
  uint32_t M_chunk = (params->M + NUM_CORES - 1) / NUM_CORES;
  uint32_t M_start = core_id * M_chunk;
  uint32_t M_end = MIN(M_start + M_chunk, params->M);
  uint32_t M_size = M_end - M_start;

  if (M_size == 0 || M_start >= params->M) {
    return;
  }

  const int8_t *local_pSrcA = params->pSrcA + M_start * params->N;
  int32_t *local_pDstY = params->pDstY + M_start * params->O;

  // Optimize for common case where offsets are zero
  if (params->A_offset == 0 && params->B_offset == 0 && params->C_offset == 0) {
    // Zero-offset optimized path with 7-way unrolling (matching FP32
    // performance)
    uint32_t O_block = params->O - (params->O % 7);

    for (uint32_t i = 0; i < M_size; i++) {
      const int8_t *row_A = local_pSrcA + i * params->N;
      int32_t *row_Y = local_pDstY + i * params->O;

      // Process 7 outputs at once (matching FP32 unroll factor)
      for (uint32_t j = 0; j < O_block; j += 7) {
        int32_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        int32_t sum4 = 0, sum5 = 0, sum6 = 0;

        // Inner loop with aggressive optimization
        for (uint32_t k = 0; k < params->N; k++) {
          int32_t a_val = (int32_t)row_A[k];
          const int8_t *col_B = params->pSrcB + k * params->O + j;

          // Manual unrolling for better compiler optimization
          sum0 += a_val * (int32_t)col_B[0];
          sum1 += a_val * (int32_t)col_B[1];
          sum2 += a_val * (int32_t)col_B[2];
          sum3 += a_val * (int32_t)col_B[3];
          sum4 += a_val * (int32_t)col_B[4];
          sum5 += a_val * (int32_t)col_B[5];
          sum6 += a_val * (int32_t)col_B[6];
        }

        row_Y[j + 0] = sum0;
        row_Y[j + 1] = sum1;
        row_Y[j + 2] = sum2;
        row_Y[j + 3] = sum3;
        row_Y[j + 4] = sum4;
        row_Y[j + 5] = sum5;
        row_Y[j + 6] = sum6;
      }

      // Handle remaining outputs
      for (uint32_t j = O_block; j < params->O; j++) {
        int32_t sum = 0;
        for (uint32_t k = 0; k < params->N; k++) {
          sum += (int32_t)row_A[k] * (int32_t)params->pSrcB[k * params->O + j];
        }
        row_Y[j] = sum;
      }
    }
  } else {
    // Fallback path with offsets
    for (uint32_t i = 0; i < M_size; i++) {
      for (uint32_t j = 0; j < params->O; j++) {
        int32_t sum = 0;

        for (uint32_t k = 0; k < params->N; k++) {
          int32_t a_val =
              (int32_t)local_pSrcA[i * params->N + k] + params->A_offset;
          int32_t b_val =
              (int32_t)params->pSrcB[k * params->O + j] + params->B_offset;
          sum += a_val * b_val;
        }

        local_pDstY[i * params->O + j] = sum + params->C_offset;
      }
    }
  }
}