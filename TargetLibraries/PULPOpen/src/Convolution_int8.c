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

void PULP_Conv2d_s8_s8_s32_HWC(void *args) {
  DeeployPULPConvParams *params = (DeeployPULPConvParams *)args;

  int8_t core_id = pi_core_id();
  int8_t log2Core = 3;

  uint16_t ch_out_chunk =
      (params->F >> log2Core) + ((params->F & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, params->F);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, params->F);
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  if (ch_out_count == 0) {
    return;
  }

  for (uint32_t f = ch_out_start; f < ch_out_stop; ++f) {
    const int8_t *weight_slice =
        params->pSrcB + f * params->C * params->P * params->Q;
    const int32_t *bias_slice =
        (params->bias != NULL) ? &params->bias[f] : NULL;

    uint32_t H_out =
        (params->H + params->pad_top + params->pad_bottom - params->P) /
            params->SP +
        1;
    uint32_t W_out =
        (params->W + params->pad_left + params->pad_right - params->Q) /
            params->SQ +
        1;

    for (uint32_t h = 0; h < H_out; ++h) {
      for (uint32_t w = 0; w < W_out; ++w) {
        int32_t sum = 0;

        if (bias_slice != NULL) {
          sum = *bias_slice;
        }

        for (uint32_t p = 0; p < params->P; ++p) {
          for (uint32_t q = 0; q < params->Q; ++q) {
            int32_t h_in = h * params->SP + p - params->pad_top;
            int32_t w_in = w * params->SQ + q - params->pad_left;

            if (h_in >= 0 && h_in < (int32_t)params->H && w_in >= 0 &&
                w_in < (int32_t)params->W) {
              for (uint32_t c = 0; c < params->C; ++c) {
                uint32_t input_idx = (h_in * params->W + w_in) * params->C + c;
                uint32_t weight_idx = (p * params->Q + q) * params->C + c;

                int32_t input_val =
                    (int32_t)params->pSrcA[input_idx] + params->input_offset;
                int32_t weight_val = (int32_t)weight_slice[weight_idx];

                sum += input_val * weight_val;
              }
            }
          }
        }

        uint32_t output_idx = (h * W_out + w) * params->F + f;
        params->pDstC[output_idx] = sum + params->output_offset;
      }
    }
  }
}

void PULP_Conv2d_Im2Col_s8_s8_s32_HWC(
    const int8_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const int8_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P, uint32_t Q,
    uint32_t SP, uint32_t SQ, int32_t *__restrict__ pDstC, int32_t input_offset,
    int32_t output_offset, const int32_t *__restrict__ bias, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right,
    int8_t *__restrict__ pContextBuffer) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = 3;

  uint16_t ch_out_chunk =
      (F_total >> log2Core) + ((F_total & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  if (ch_out_count == 0) {
    return;
  }

  const int8_t *weight_ptr = pSrcB + ch_out_start * C * P * Q;

  uint32_t im2col_size_per_core = C * P * Q;
  int8_t *im2col_buffer = pContextBuffer + core_id * im2col_size_per_core;

  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
  uint32_t kernel_size = P * Q * C;

  for (uint32_t h_out = 0; h_out < H_out; h_out++) {
    for (uint32_t w_out = 0; w_out < W_out; w_out++) {
      int32_t h_in_start = h_out * SP - pad_top;
      int32_t w_in_start = w_out * SQ - pad_left;

      for (uint32_t p = 0; p < P; p++) {
        int32_t h_in = h_in_start + p;

        for (uint32_t q = 0; q < Q; q++) {
          int32_t w_in = w_in_start + q;

          for (uint32_t c = 0; c < C; c++) {
            if (h_in >= 0 && h_in < (int32_t)H && w_in >= 0 &&
                w_in < (int32_t)W) {
              uint32_t in_idx = (h_in * W + w_in) * C + c;
              im2col_buffer[p * Q * C + q * C + c] = pSrcA[in_idx];
            } else {
              im2col_buffer[p * Q * C + q * C + c] = 0;
            }
          }
        }
      }

      for (uint32_t f = 0; f < ch_out_count; f++) {
        int32_t sum = 0;

        if (bias != NULL) {
          sum = bias[ch_out_start + f];
        }

        const int8_t *local_weight_ptr = weight_ptr + f * kernel_size;

        for (uint32_t k = 0; k < kernel_size; k++) {
          int32_t input_val = (int32_t)im2col_buffer[k] + input_offset;
          int32_t weight_val = (int32_t)local_weight_ptr[k];
          sum += input_val * weight_val;
        }

        uint32_t out_idx =
            (h_out * W_out + w_out) * F_total + (ch_out_start + f);
        pDstC[out_idx] = sum + output_offset;
      }
    }
  }
}

void PULP_pointwise_i8_i32_i8_HWC(void *args) {
  DeeployPULPConvParams *params = (DeeployPULPConvParams *)args;

  int8_t core_id = pi_core_id();
  int8_t log2Core = 3;

  uint16_t ch_out_chunk =
      (params->F >> log2Core) + ((params->F & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, params->F);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, params->F);
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  if (ch_out_count == 0) {
    return;
  }

  const int8_t *weight_ptr = params->pSrcB + ch_out_start * params->C;

  uint32_t spatial_size = params->H * params->W;

  for (uint32_t spatial_idx = 0; spatial_idx < spatial_size; ++spatial_idx) {
    for (uint32_t f = 0; f < ch_out_count; ++f) {
      int32_t sum = 0;

      if (params->bias != NULL) {
        sum = params->bias[ch_out_start + f];
      }

      for (uint32_t c = 0; c < params->C; ++c) {
        uint32_t input_idx = spatial_idx * params->C + c;
        uint32_t weight_idx = f * params->C + c;

        int32_t input_val =
            (int32_t)params->pSrcA[input_idx] + params->input_offset;
        int32_t weight_val = (int32_t)weight_ptr[weight_idx];

        sum += input_val * weight_val;
      }

      uint32_t output_idx = spatial_idx * params->F + (ch_out_start + f);
      params->pDstC[output_idx] = sum + params->output_offset;
    }
  }
}