/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/atomic.h"
#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Distance to a virtual radiance source, propagated backwards through a path.
 * All distances use one isotropic scene scale. FLT_MAX denotes a distant or unknown endpoint.
 * This affects proposal fitting only; it is not an optical path length or an image weight. */
struct GuidingVirtualDistance {
  ccl_device_inline float scatter_scale(const bool singular,
                                        const float roughness,
                                        const float eta,
                                        const float cosine_in,
                                        const float cosine_out)
  {
    if (!(eta > 0.0f) || !isfinite_safe(eta) || !isfinite_safe(roughness) ||
        !isfinite_safe(cosine_in) || !isfinite_safe(cosine_out))
    {
      return FLT_MAX;
    }
    if (!singular && roughness >= 0.3f) {
      return 0.0f;
    }
    if (eta == 1.0f) {
      return 1.0f;
    }
    const float denominator = fabsf(cosine_out * eta);
    const float scale = denominator > 0.0f ? fabsf(cosine_in) / denominator : FLT_MAX;
    return isfinite_safe(scale) && scale >= 0.0f ? scale : FLT_MAX;
  }

  ccl_device_inline float propagate(const float segment,
                                    const float downstream,
                                    const float scatter_scale)
  {
    if (!(segment >= 0.0f) || !isfinite_safe(segment)) {
      return FLT_MAX;
    }
    /* A rough vertex ends the virtual-source chain, including a distant-source chain. */
    if (scatter_scale == 0.0f || downstream == 0.0f) {
      return segment;
    }
    if (!(downstream >= 0.0f) || !isfinite_safe(downstream) || downstream == FLT_MAX ||
        !(scatter_scale >= 0.0f) || !isfinite_safe(scatter_scale))
    {
      return FLT_MAX;
    }
    const float distance = segment + scatter_scale * downstream;
    return isfinite_safe(distance) ? min(distance, FLT_MAX) : FLT_MAX;
  }
};

struct GuidingParallaxFit {
  float3 target;
  float3 fallback_direction;
  /* Symmetric covariance of virtual source positions: xx, xy, xz, yy, yz, zz. */
  float covariance[6];
  bool valid;

  ccl_device_inline_method float3 direction(const float3 position) const
  {
    const float3 delta = target - position;
    const float squared_distance = len_squared(delta);
    const float3 fallback = isfinite_safe(fallback_direction) ? fallback_direction : zero_float3();
    return valid && isfinite_safe(delta) && isfinite_safe(squared_distance) &&
                   squared_distance > 1e-12f ?
               delta / sqrtf(squared_distance) :
               safe_normalize_fallback(fallback, make_float3(0, 0, 1));
  }

  ccl_device_inline_method float concentration(const float3 position) const
  {
    const float3 delta = target - position;
    const float distance_squared = len_squared(delta);
    if (!valid || !(distance_squared > 1e-12f) || !isfinite_safe(distance_squared)) {
      return 0.0f;
    }
    const float3 axis = direction(position);
    const float along_axis = covariance[0] * axis.x * axis.x +
                             2.0f * covariance[1] * axis.x * axis.y +
                             2.0f * covariance[2] * axis.x * axis.z +
                             covariance[3] * axis.y * axis.y +
                             2.0f * covariance[4] * axis.y * axis.z +
                             covariance[5] * axis.z * axis.z;
    const float variance = covariance[0] + covariance[3] + covariance[5] - along_axis;
    if (!isfinite_safe(variance)) {
      return 0.0f;
    }
    const float transverse_variance = max(variance, 0.0f);
    const float r = min(safe_sqrtf(1.0f - transverse_variance / distance_squared), 1.0f - 1e-6f);
    return min(r * (3.0f - r * r) / (1.0f - r * r), 16384.0f);
  }
};

/* Harmonic-distance-weighted virtual source moments. Every finite sample corresponds to
 * Q = P + distance * direction. Compact sources therefore retain one common target even
 * when observations originate at different positions. The covariance controls angular width.
 * This is an independent moment model, not OpenPGL's complete soft-assignment EM algorithm. */
struct GuidingParallaxMoments {
  /* W, sum(w^2), H=sum(w/d), sum((w/d)^2), H*Q[3], H*QQ^T[6]. */
  ccl_static_constexpr int storage_size = 13;

  ccl_device_inline bool squared_weight_entry(const int entry)
  {
    return entry == 1 || entry == 3;
  }

  ccl_device_inline void record(ccl_global float *moments,
                                const float weight,
                                const float3 position,
                                const float3 direction,
                                const float distance)
  {
    if (!(weight > 0.0f) || !isfinite_safe(weight) || !(distance > 0.0f) || distance == FLT_MAX ||
        !isfinite_safe(distance) || !isfinite_safe(position) || !isfinite_safe(direction))
    {
      return;
    }
    /* The harmonic effective-sample-size check rejects domination by a near endpoint. */
    const float harmonic_weight = weight / distance;
    record_aggregate(moments, weight, harmonic_weight, weight * distance, position, direction);
  }

  /* One path observation may contain multiple shadow/emission endpoints. Preserve its first
   * and second source moments, rather than inventing one endpoint at the mean distance. */
  ccl_device_inline void record_aggregate(ccl_global float *moments,
                                          const float weight,
                                          const float harmonic_weight,
                                          const float distance_weight,
                                          const float3 position,
                                          const float3 direction)
  {
    if (!(weight > 0.0f) || !(harmonic_weight > 0.0f) || !(distance_weight >= 0.0f) ||
        !isfinite_safe(weight) || !isfinite_safe(harmonic_weight) ||
        !isfinite_safe(distance_weight) || !isfinite_safe(position) || !isfinite_safe(direction))
    {
      return;
    }
    atomic_add_and_fetch_float(moments, weight);
    atomic_add_and_fetch_float(moments + 1, weight * weight);
    atomic_add_and_fetch_float(moments + 2, harmonic_weight);
    atomic_add_and_fetch_float(moments + 3, harmonic_weight * harmonic_weight);
    for (int i = 0; i < 3; ++i) {
      atomic_add_and_fetch_float(moments + 4 + i,
                                 harmonic_weight * position[i] + weight * direction[i]);
    }
    int entry = 7;
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        atomic_add_and_fetch_float(
            moments + entry++,
            harmonic_weight * position[i] * position[j] +
                weight * (position[i] * direction[j] + direction[i] * position[j]) +
                distance_weight * direction[i] * direction[j]);
      }
    }
  }

  ccl_device_inline GuidingParallaxFit fit(const ccl_global float *moments,
                                           const float total_weight,
                                           const float observations,
                                           const float3 fallback_direction)
  {
    GuidingParallaxFit result{};
    result.fallback_direction = fallback_direction;
    if (!(total_weight > 0.0f) || !isfinite_safe(total_weight) || observations < 64.0f ||
        !(moments[0] >= 0.9f * total_weight))
    {
      return result;
    }
    for (int i = 0; i < storage_size; ++i) {
      if (!isfinite_safe(moments[i])) {
        return result;
      }
    }
    /* Check support for both radiance and harmonic weights. The second check detects
     * domination by a single near endpoint even if the radiance weights are balanced. */
    if (!(moments[1] > 0.0f) || !(moments[3] > 0.0f) ||
        moments[0] / sqrtf(moments[1]) < sqrtf(32.0f) ||
        moments[2] / sqrtf(moments[3]) < sqrtf(32.0f))
    {
      return result;
    }
    const float inverse_harmonic_weight = 1.0f / moments[2];
    result.target = make_float3(moments[4], moments[5], moments[6]) * inverse_harmonic_weight;
    if (!isfinite_safe(result.target)) {
      return result;
    }
    int entry = 0;
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        const float covariance = moments[7 + entry] * inverse_harmonic_weight -
                                 result.target[i] * result.target[j];
        if (!isfinite_safe(covariance)) {
          return result;
        }
        if (i == j && covariance < -32.0f * FLT_EPSILON * max(1.0f, len_squared(result.target))) {
          return result;
        }
        result.covariance[entry++] = i == j ? max(covariance, 0.0f) : covariance;
      }
    }
    result.valid = true;
    return result;
  }
};

CCL_NAMESPACE_END
