/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/sample/guiding_spherical_gaussian.h"

CCL_NAMESPACE_BEGIN

/* Batch weighted EM on the unit sphere. Each float4 observation contains a unit
 * direction and a nonnegative importance weight. The caller owns observations;
 * fitting never changes the query model until explicitly exported.
 * Cooperative=true requires every lane of one complete Metal SIMD group to call
 * each fitting step with the same observations/count and its actual lane/width.
 * The private model is replicated across lanes; reductions synchronize statistics.
 * Only one lane may publish to a shared destination after fitting completes.
 *
 * This independent fitting primitive is shared by host and Metal tests. Renderer
 * history scheduling and spatial/parallax fitting are separate integrations. */
template<int Capacity> struct GuidingDirectionalMixtureFit {
  GuidingSphericalGaussian lobes[Capacity];
  float weights[Capacity];
  int active = 0;
  float maximum_weight = 0.0f;

  ccl_device_inline_method static bool valid(const float4 observation)
  {
    const float norm = len_squared(make_float3(observation));
    return isfinite_safe(observation.w) && observation.w > 0.0f && isfinite_safe(norm) &&
           fabsf(norm - 1.0f) < 1e-4f;
  }

  /* Invert the 3D vMF mean resultant A(k)=coth(k)-1/k. The concentration
   * cap bounds the maximum likelihood fit for coincident observations. */
  ccl_device_inline_method static float concentration(const float resultant)
  {
    return GuidingSphericalGaussian::concentration_from_resultant(resultant);
  }

  ccl_device_inline_method float log_component(const int component, const float3 direction) const
  {
    return weights[component] > 0.0f ?
               logf(weights[component]) + logf(lobes[component].normalization()) +
                   lobes[component].concentration *
                       (min(dot(lobes[component].axis, direction), 1.0f) - 1.0f) :
               -FLT_MAX;
  }

  template<bool Cooperative = false, typename Observations>
  ccl_device_inline_method bool initialize(const Observations observations,
                                           const uint count,
                                           const uint lane = 0,
                                           const uint width = 1)
  {
#ifndef __KERNEL_METAL__
    static_assert(!Cooperative);
#endif
    active = 0;
    maximum_weight = 0.0f;
    for (uint i = lane; i < count; i += width) {
      if (valid(observations[i])) {
        maximum_weight = max(maximum_weight, observations[i].w);
      }
    }
#ifdef __KERNEL_METAL__
    if constexpr (Cooperative) {
      maximum_weight = metal::simd_max(maximum_weight);
    }
#endif
    if (!(maximum_weight > 0.0f)) {
      return false;
    }
    /* Deterministic weighted farthest-point seeds do not inherit angular bins. */
    for (int component = 0; component < Capacity; ++component) {
      float best = -1.0f;
      uint best_index = ~0u;
      float3 axis = make_float3(0, 0, 1);
      for (uint i = lane; i < count; i += width) {
        const float4 observation = observations[i];
        if (!valid(observation)) {
          continue;
        }
        const float3 direction = normalize(make_float3(observation));
        float separation = 1.0f;
        for (int j = 0; j < active; ++j) {
          separation = min(separation, len_squared(direction - lobes[j].axis));
        }
        const float score = (observation.w / maximum_weight) * separation;
        if (score > best) {
          best = score;
          best_index = i;
          axis = direction;
        }
      }
#ifdef __KERNEL_METAL__
      if constexpr (Cooperative) {
        const float maximum = metal::simd_max(best);
        /* The earliest observation wins ties, matching sequential initialization. */
        best_index = metal::simd_min(best == maximum ? best_index : ~0u);
        best = maximum;
        if (best_index != ~0u) {
          axis = normalize(make_float3(observations[best_index]));
        }
      }
#endif
      if (!(best > 0.0f)) {
        break;
      }
      lobes[active++] = {axis, 16.0f};
    }
    for (int i = 0; i < Capacity; ++i) {
      weights[i] = i < active ? 1.0f / active : 0.0f;
    }
    return active > 0;
  }

  template<bool Cooperative = false, typename Observations>
  ccl_device_inline_method bool iterate(const Observations observations,
                                        const uint count,
                                        const uint lane = 0,
                                        const uint width = 1)
  {
#ifdef __clang__
#  pragma clang fp reassociate(off)
#endif
#ifndef __KERNEL_METAL__
    static_assert(!Cooperative);
#endif
    if (active == 0 || count == 0 || !(maximum_weight > 0.0f)) {
      return false;
    }
    float4 moments[Capacity];
    float4 compensation[Capacity];
    float squared_offsets[Capacity];
    float squared_compensation[Capacity];
    for (int j = 0; j < active; ++j) {
      moments[j] = make_float4(0.0f);
      compensation[j] = make_float4(0.0f);
      squared_offsets[j] = 0.0f;
      squared_compensation[j] = 0.0f;
    }
    for (uint i = lane; i < count; i += width) {
      const float4 observation = observations[i];
      if (!valid(observation)) {
        continue;
      }
      const float3 direction = normalize(make_float3(observation));
      float responsibilities[Capacity];
      float maximum = -FLT_MAX;
      for (int j = 0; j < active; ++j) {
        responsibilities[j] = log_component(j, direction);
        maximum = max(maximum, responsibilities[j]);
      }
      float sum = 0.0f;
      for (int j = 0; j < active; ++j) {
        responsibilities[j] = expf(responsibilities[j] - maximum);
        sum += responsibilities[j];
      }
      /* Common rescaling preserves the optimum and avoids bright-weight overflow. */
      const float weight = (observation.w / maximum_weight) / count;
      for (int j = 0; j < active; ++j) {
        const float assigned = weight * responsibilities[j] / sum;
        /* Narrow-lobe concentration is sensitive to errors in the resultant
         * length near one. Compensate both mass and directional sums. */
        const float3 offset = direction - lobes[j].axis;
        const float4 delta = make_float4(offset.x, offset.y, offset.z, 1.0f) * assigned -
                             compensation[j];
        const float4 updated = moments[j] + delta;
        compensation[j] = (updated - moments[j]) - delta;
        moments[j] = updated;
        const float squared_delta = assigned * len_squared(offset) - squared_compensation[j];
        const float squared_updated = squared_offsets[j] + squared_delta;
        squared_compensation[j] = (squared_updated - squared_offsets[j]) - squared_delta;
        squared_offsets[j] = squared_updated;
      }
    }
    float total = 0.0f;
    for (int j = 0; j < active; ++j) {
#ifdef __KERNEL_METAL__
      if constexpr (Cooperative) {
        moments[j] = metal::simd_sum(moments[j]);
        squared_offsets[j] = metal::simd_sum(squared_offsets[j]);
      }
#endif
      total += moments[j].w;
    }
    if (!(total > 0.0f)) {
      return false;
    }
    for (int j = 0; j < active; ++j) {
      weights[j] = moments[j].w / total;
      if (moments[j].w > 0.0f) {
        const float3 offset = make_float3(moments[j]) / moments[j].w;
        const float3 mean = lobes[j].axis + offset;
        const float length = len(mean);
        /* For unit directions, Var(D)=1-|E(D)|^2. Compute this from local
         * offsets so a narrow fit never subtracts its resultant from one.
         * Above k=~19 the omitted coth correction is below float precision. */
        const float variance = max(squared_offsets[j] / moments[j].w - len_squared(offset), 0.0f);
        const float resultant = safe_sqrtf(1.0f - variance);
        const float k = variance < .1f ?
                            (variance > 0.0f ? min((1.0f + resultant) / variance, 16384.0f) :
                                               16384.0f) :
                            concentration(resultant);
        lobes[j] = {length > 0.0f ? mean / length : make_float3(0, 0, 1), k};
      }
    }
    return true;
  }

  ccl_device_inline_method float pdf(const float3 direction) const
  {
    float result = 0.0f;
    for (int j = 0; j < active; ++j) {
      result += weights[j] * lobes[j].pdf(direction);
    }
    return result;
  }

  /* Publish into the existing query representation; the sampler retains its
   * explicit uniform exploration term. No observation or fitting state escapes. */
  ccl_device_inline_method void publish(ccl_global float *storage) const
  {
    static_assert(Capacity <= GuidingGaussianMixture::components);
    for (int i = 0; i < GuidingGaussianMixture::components; ++i) {
      ccl_global float *entry = storage + i * GuidingGaussianMixture::component_stride;
      for (int j = 0; j < GuidingGaussianMixture::component_stride; ++j) {
        entry[j] = 0.0f;
      }
      entry[4] = 1.0f;
      if (i < active) {
        entry[0] = weights[i];
        entry[1] = lobes[i].concentration;
        entry[2] = lobes[i].axis.x;
        entry[3] = lobes[i].axis.y;
        entry[4] = lobes[i].axis.z;
      }
      else if (active == 0 && i == 0) {
        entry[0] = 1.0f;
      }
    }
  }
};

CCL_NAMESPACE_END
