/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "util/math.h"
CCL_NAMESPACE_BEGIN

/* Read packed real model data from the existing float2 cache buffer without
 * pointer type-punning or additional device allocations. */
struct DiffractionTensorFloatView {
  ccl_global const float2 *data;
  int offset;
  ccl_device_inline_method float operator[](const int index) const
  {
    const int i = offset + index;
    const float2 pair = data[i / 2];
    return (i & 1) ? pair.y : pair.x;
  }
  ccl_device_inline_method DiffractionTensorFloatView operator+(const int index) const
  {
    return {data, offset + index};
  }
};

/* Seven-node real barycentric basis. The nearest-distance scale cancels in
 * the quotient and avoids overflow adjacent to a node, including subnormals.
 * The host validates strictly increasing nodes and finite nonzero weights. */
template<typename Pointer>
ccl_device_inline bool diffraction_tensor_basis(const float coordinate,
                                                const Pointer nodes,
                                                const Pointer weights,
                                                ccl_private float *basis)
{
  if (!isfinite_safe(coordinate) || coordinate < 0.0f || coordinate > 1.0f)
    return false;
  int nearest = 0;
  for (int i = 0; i < 7; i++) {
    if (coordinate == nodes[i]) {
      for (int j = 0; j < 7; j++)
        basis[j] = float(i == j);
      return true;
    }
    if (fabsf(coordinate - nodes[i]) < fabsf(coordinate - nodes[nearest]))
      nearest = i;
  }
  const float distance = coordinate - nodes[nearest];
  float sum = 0.0f;
  for (int i = 0; i < 7; i++) {
    basis[i] = weights[i] * (distance / (coordinate - nodes[i]));
    sum += basis[i];
  }
  if (!isfinite_safe(sum) || sum == 0.0f)
    return false;
  for (int i = 0; i < 7; i++) {
    basis[i] /= sum;
    if (!isfinite_safe(basis[i]))
      return false;
  }
  return true;
}

/* Layout in real floats: nodes[7], weights[7], coefficients[7^3][rank],
 * directions[rank][N^2], mean[N^2]. Hermitian packing is diagonal real,
 * sqrt(2)*upper-triangle real, sqrt(2)*upper-triangle imaginary. Reconstruct
 * the skew-Hermitian chart Y=iH, for diffraction_reference_match_anchor.
 * The separate anchor has N*N complex entries in the same channel basis.
 * No material/channel truncation, renormalization or extrapolation occurs. */
template<int N, typename Pointer>
/* Explicit outlining bounds Metal compilation of the fixed-channel dispatch.
 * Apple's generic ccl_device_noinline deliberately permits inlining. */
#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device_inline
#endif
bool diffraction_tensor_chart(const Pointer model,
                                                const int model_floats,
                                                const int rank,
                                                const float3 coordinate,
                                                ccl_private float2 *chart)
{
  static_assert(N > 0 && N % 2 == 0);
  if (rank < 0 || rank > N * N || model_floats < 14 + 343 * rank + rank * N * N + N * N)
    return false;
  float bx[7], by[7], bz[7];
  if (!diffraction_tensor_basis(coordinate.x, model, model + 7, bx) ||
      !diffraction_tensor_basis(coordinate.y, model, model + 7, by) ||
      !diffraction_tensor_basis(coordinate.z, model, model + 7, bz))
    return false;
  float coefficients[N * N];
  for (int r = 0; r < rank; r++) {
    float value = 0.0f;
    for (int z = 0; z < 7; z++) {
      float zy = 0.0f;
      for (int y = 0; y < 7; y++) {
        float yx = 0.0f;
        for (int x = 0; x < 7; x++)
          yx += bx[x] * model[14 + ((z * 7 + y) * 7 + x) * rank + r];
        zy += by[y] * yx;
      }
      value += bz[z] * zy;
    }
    coefficients[r] = value;
  }
  const int directions = 14 + 343 * rank;
  const int mean = directions + rank * N * N;
  constexpr int upper_count = N * (N - 1) / 2;
  int upper = 0;
  for (int row = 0; row < N; row++) {
    float diagonal = model[mean + row];
    for (int r = 0; r < rank; r++)
      diagonal += coefficients[r] * model[directions + r * N * N + row];
    if (!isfinite_safe(diagonal))
      return false;
    chart[row * N + row] = make_float2(0.0f, diagonal);
    for (int col = row + 1; col < N; col++, upper++) {
      float re = model[mean + N + upper], im = model[mean + N + upper_count + upper];
      for (int r = 0; r < rank; r++) {
        re += coefficients[r] * model[directions + r * N * N + N + upper];
        im += coefficients[r] * model[directions + r * N * N + N + upper_count + upper];
      }
      re *= M_SQRT1_2F;
      im *= M_SQRT1_2F;
      if (!isfinite_safe(re) || !isfinite_safe(im))
        return false;
      chart[row * N + col] = make_float2(-im, re);
      chart[col * N + row] = make_float2(im, re);
    }
  }
  return true;
}
CCL_NAMESPACE_END
