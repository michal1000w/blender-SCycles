/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include <metal_stdlib>
using namespace metal;

float2 sum_pair(float a, float b)
{
  float s = a + b;
  float v = s - a;
  return float2(s, (a - (s - v)) + (b - v));
}

float2 accumulate_product(float2 sum, float a, float b)
{
  float product = a * b;
  float error = fma(a, b, -product);
  float2 next = sum_pair(sum.x, product);
  return sum_pair(next.x, sum.y + (next.y + error));
}

kernel void compensated_product(device const float2 *a [[buffer(0)]],
                                device const float2 *b [[buffer(1)]],
                                device float2 *out [[buffer(2)]],
                                constant uint &n [[buffer(3)]],
                                uint index [[thread_position_in_grid]])
{
  uint matrix = index / (n * n), row = (index / n) % n, col = index % n;
  uint base = matrix * n * n;
  float2 real_sum(0.0f), imag_sum(0.0f);
  for (uint k = 0; k < n; ++k) {
    float2 x = a[base + row * n + k], y = b[base + k * n + col];
    real_sum = accumulate_product(real_sum, x.x, y.x);
    real_sum = accumulate_product(real_sum, -x.y, y.y);
    imag_sum = accumulate_product(imag_sum, x.x, y.y);
    imag_sum = accumulate_product(imag_sum, x.y, y.x);
  }
  out[index] = float2(real_sum.x + real_sum.y, imag_sum.x + imag_sum.y);
}

kernel void ordinary_product(device const float2 *a [[buffer(0)]],
                             device const float2 *b [[buffer(1)]],
                             device float2 *out [[buffer(2)]],
                             constant uint &n [[buffer(3)]],
                             uint index [[thread_position_in_grid]])
{
  uint matrix = index / (n * n), row = (index / n) % n, col = index % n;
  uint base = matrix * n * n;
  float2 sum(0.0f);
  for (uint k = 0; k < n; ++k) {
    float2 x = a[base + row * n + k], y = b[base + k * n + col];
    sum += float2(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
  }
  out[index] = sum;
}
