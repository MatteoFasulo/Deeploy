/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void Conv2d_s8_s8_s32_HWC(
    const int8_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const int8_t *__restrict__ pSrcB, uint32_t F, uint32_t P, uint32_t Q,
    uint32_t SP, uint32_t SQ, int32_t *__restrict__ pDstC, int32_t input_offset,
    int32_t output_offset, const int32_t *__restrict__ bias, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {

  const uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  const uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;

  for (uint32_t f_idx = 0; f_idx < F; f_idx++) {
    const int32_t bias_val = (bias != NULL) ? bias[f_idx] : 0;

    for (uint32_t h_out = 0; h_out < H_out; h_out++) {
      for (uint32_t w_out = 0; w_out < W_out; w_out++) {
        int32_t output_value = bias_val;

        const uint32_t out_idx = (h_out * W_out + w_out) * F + f_idx;

        for (uint32_t kernel_pos = 0; kernel_pos < P * Q; kernel_pos++) {
          const uint32_t p_idx = kernel_pos / Q;
          const uint32_t q_idx = kernel_pos % Q;

          const int32_t h_in = h_out * SP + p_idx - pad_top;
          const int32_t w_in = w_out * SQ + q_idx - pad_left;

          if (h_in >= 0 && h_in < (int32_t)H && w_in >= 0 &&
              w_in < (int32_t)W) {
            const uint32_t in_base = (h_in * W + w_in) * C;
            const uint32_t weight_base =
                (f_idx * P * Q + p_idx * Q + q_idx) * C;

            const int8_t *input_ptr = pSrcA + in_base;
            const int8_t *weight_ptr = pSrcB + weight_base;

            for (uint32_t c_idx = 0; c_idx < C; c_idx++) {
              const int32_t input_val =
                  (int32_t)input_ptr[c_idx] + input_offset;
              const int32_t weight_val = (int32_t)weight_ptr[c_idx];
              output_value += input_val * weight_val;
            }
          }
        }

        pDstC[out_idx] = output_value + output_offset;
      }
    }
  }
}