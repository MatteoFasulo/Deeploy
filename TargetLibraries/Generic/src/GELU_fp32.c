/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

#define M_PI 3.14159265358979323846

// Exact GELU using the error function
void GELU_fp32_fp32(float32_t *data_in, float32_t *data_out, int32_t dataSize) {
  for (int i = 0; i < dataSize; i++) {
    float32_t x = data_in[i];
    float32_t cdf =
        0.5f * (1.0f + erf(x / sqrtf(2.0f))); // erf() is the error function
    data_out[i] = x * cdf;
  }
}

// Approximation with tanh
void GELU_fp32_fp32_tanh(float32_t *data_in, float32_t *data_out,
                         int32_t dataSize) {
  for (int i = 0; i < dataSize; i++) {
    float32_t x = data_in[i];
    float32_t cdf = 0.5f * (1.0f + tanhf((sqrtf(2.0f / (float)M_PI) *
                                          (x + 0.044715f * powf(x, 3.0f)))));
    data_out[i] = x * cdf;
  }
}

// Approximation with sigmoid
void GELU_fp32_fp32_sigmoid(float32_t *data_in, float32_t *data_out,
                            int32_t dataSize) {

  const float32_t scale = 1.702f;
  for (int i = 0; i < dataSize; i++) {
    float32_t x = data_in[i];
    float32_t sigmoid_in = scale * x;
    // sigmoid(z) = 1 / (1 + exp(-z))
    float32_t sigmoid = 1.0f / (1.0f + expf(-sigmoid_in));
    data_out[i] = x * sigmoid;
  }
}
