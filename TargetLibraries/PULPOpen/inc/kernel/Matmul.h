/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_MATMUL_KERNEL_HEADER_
#define __DEEPLOY_MATH_MATMUL_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_MatMul_fp32_fp32_fp32_unroll1x7(const float32_t *__restrict__ pSrcA,
                                          const float32_t *__restrict__ pSrcB,
                                          float32_t *__restrict__ pDstY,
                                          uint32_t M, uint32_t N, uint32_t O);

// Multi-core optimized integer MatMul
void PULP_MatMul_s8_s8_s32(void *args);

// Direct integer MatMul function matching FP32 performance
void PULP_MatMul_s8_s8_s32_unroll1x7(void *args);

#endif // __DEEPLOY_MATH_MATMUL_KERNEL_HEADER_