/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_CONV_KERNEL_HEADER_
#define __DEEPLOY_MATH_CONV_KERNEL_HEADER_

#include "DeeployPULPMath.h"

// Parameter structure for multi-core integer convolution
typedef struct {
  const int8_t *pSrcA;
  uint32_t H;
  uint32_t W;
  uint32_t C;
  const int8_t *pSrcB;
  uint32_t F;
  uint32_t P;
  uint32_t Q;
  uint32_t SP;
  uint32_t SQ;
  int32_t *pDstC;
  int32_t input_offset;
  int32_t output_offset;
  const int32_t *bias;
  uint32_t pad_top;
  uint32_t pad_bottom;
  uint32_t pad_left;
  uint32_t pad_right;
  int8_t *ctxtBuffer;
} DeeployPULPConvParams;

void PULP_Conv2d_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right);

void PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pContextBuffer);

// Integer convolution functions
void Conv2d_s8_s8_s32_HWC(
    const int8_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const int8_t *__restrict__ pSrcB, uint32_t F, uint32_t P, uint32_t Q,
    uint32_t SP, uint32_t SQ, int32_t *__restrict__ pDstC, int32_t input_offset,
    int32_t output_offset, const int32_t *__restrict__ bias, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right);

// Multi-core optimized integer convolution functions
void PULP_Conv2d_s8_s8_s32_HWC(void *args);
void PULP_pointwise_i8_i32_i8_HWC(void *args);

// Im2Col optimized integer convolution function
void PULP_Conv2d_Im2Col_s8_s8_s32_HWC(
    const int8_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const int8_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P, uint32_t Q,
    uint32_t SP, uint32_t SQ, int32_t *__restrict__ pDstC, int32_t input_offset,
    int32_t output_offset, const int32_t *__restrict__ bias, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right,
    int8_t *__restrict__ pContextBuffer);

#endif // __DEEPLOY_MATH_CONV_KERNEL_HEADER_