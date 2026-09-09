/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/atomic.h"
#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Weighted local regression of unit incident direction on normalized scene position.
 * This fits a proposal; it is never used as a replacement for a rendered contribution. */
struct GuidingPositionFit {
  float3 mean_position;
  float3 mean_direction;
  float3 slope[3];
  float residual_variance;
  bool valid;

  ccl_device_inline_method float3 direction(const float3 position) const
  {
    const float3 delta = position - mean_position;
    const float3 prediction = mean_direction + make_float3(dot(slope[0], delta),
                                                           dot(slope[1], delta),
                                                           dot(slope[2], delta));
    const float length_squared = len_squared(prediction);
    return isfinite_safe(prediction) && isfinite_safe(length_squared) && length_squared > 1e-12f ?
               prediction / sqrtf(length_squared) :
               safe_normalize_fallback(mean_direction, make_float3(0, 0, 1));
  }
};

struct GuidingPositionMoments {
  /* W, W*d[3], W*p[3], W*pp^T[6], W*dp^T[9], W^2. */
  ccl_static_constexpr int storage_size = 23;

  ccl_device_inline void record(ccl_global float *moments,
                                const float weight,
                                const float3 position,
                                const float3 direction)
  {
    if (!(weight > 0.0f) || !isfinite_safe(weight) || !isfinite_safe(position) ||
        !isfinite_safe(direction))
    {
      return;
    }
    atomic_add_and_fetch_float(moments, weight);
    for (int i = 0; i < 3; ++i) {
      atomic_add_and_fetch_float(moments + 1 + i, weight * direction[i]);
      atomic_add_and_fetch_float(moments + 4 + i, weight * position[i]);
      for (int j = 0; j < 3; ++j) {
        atomic_add_and_fetch_float(moments + 13 + 3 * i + j, weight * direction[i] * position[j]);
      }
    }
    int entry = 7;
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        atomic_add_and_fetch_float(moments + entry++, weight * position[i] * position[j]);
      }
    }
    atomic_add_and_fetch_float(moments + 22, weight * weight);
  }

  ccl_device_inline GuidingPositionFit fit(const ccl_global float *moments,
                                           const float observations)
  {
    GuidingPositionFit result{};
    const float weight = moments[0];
    const float weight_squared_sum = moments[22];
    if (!(weight > 0.0f) || !isfinite_safe(weight) || !(weight_squared_sum > 0.0f) ||
        !isfinite_safe(weight_squared_sum) || observations < 64.0f)
    {
      return result;
    }
    /* Effective sample size avoids fitting a narrow positional model to a single firefly
     * surrounded by many negligible observations. The ratio form avoids squaring W. */
    const float effective_count_root = weight / sqrtf(weight_squared_sum);
    if (effective_count_root < sqrtf(32.0f)) {
      return result;
    }
    const float inverse_weight = 1.0f / weight;
    result.mean_direction = make_float3(moments[1], moments[2], moments[3]) * inverse_weight;
    result.mean_position = make_float3(moments[4], moments[5], moments[6]) * inverse_weight;
    if (!isfinite_safe(result.mean_direction) || !isfinite_safe(result.mean_position)) {
      return result;
    }
    const float3 p = result.mean_position;
    const float cxx = moments[7] * inverse_weight - p.x * p.x;
    const float cxy = moments[8] * inverse_weight - p.x * p.y;
    const float cxz = moments[9] * inverse_weight - p.x * p.z;
    const float cyy = moments[10] * inverse_weight - p.y * p.y;
    const float cyz = moments[11] * inverse_weight - p.y * p.z;
    const float czz = moments[12] * inverse_weight - p.z * p.z;
    /* Positions are in scene-relative coordinates. The absolute floor protects cancellation
     * in raw moments; the relative term regularizes planar and line-like sample supports. */
    const float ridge = max(1e-6f, 1e-3f * (max(cxx, 0.0f) + max(cyy, 0.0f) + max(czz, 0.0f)));
    const float a00 = cxx + ridge;
    if (!(a00 > 0.0f) || !isfinite_safe(a00)) {
      return result;
    }
    const float l00 = sqrtf(a00);
    const float l10 = cxy / l00, l20 = cxz / l00;
    const float a11 = cyy + ridge - l10 * l10;
    if (!(a11 > 0.0f) || !isfinite_safe(a11)) {
      return result;
    }
    const float l11 = sqrtf(a11), l21 = (cyz - l20 * l10) / l11;
    const float a22 = czz + ridge - l20 * l20 - l21 * l21;
    if (!(a22 > 0.0f) || !isfinite_safe(a22)) {
      return result;
    }
    const float l22 = sqrtf(a22);
    float explained_variance = 0.0f;
    for (int i = 0; i < 3; ++i) {
      const float3 cross = make_float3(
                               moments[13 + 3 * i], moments[14 + 3 * i], moments[15 + 3 * i]) *
                               inverse_weight -
                           result.mean_direction[i] * p;
      /* Solve the regularized symmetric system with Cholesky, without a matrix inverse. */
      const float y0 = cross.x / l00;
      const float y1 = (cross.y - l10 * y0) / l11;
      const float y2 = (cross.z - l20 * y0 - l21 * y1) / l22;
      const float x2 = y2 / l22;
      const float x1 = (y1 - l21 * x2) / l11;
      const float x0 = (y0 - l10 * x1 - l20 * x2) / l00;
      result.slope[i] = make_float3(x0, x1, x2);
      if (!isfinite_safe(result.slope[i])) {
        return result;
      }
      explained_variance += dot(result.slope[i], cross);
    }
    const float variance = 1.0f - len_squared(result.mean_direction) - explained_variance;
    if (!isfinite_safe(variance) || variance < -1e-4f || variance > 1.0f) {
      return result;
    }
    result.residual_variance = max(variance, 0.0f);
    result.valid = true;
    return result;
  }
};

CCL_NAMESPACE_END
