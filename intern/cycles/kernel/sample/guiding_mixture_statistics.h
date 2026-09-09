/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/sample/guiding_mixture_fit.h"

CCL_NAMESPACE_BEGIN

/* Source entries are finite radiance W, sum(w/d), sum(w*d), and unused padding.
 * Distances use the same isotropic scene metric as position * metric_extent.
 * A zero source tuple denotes an unknown/distant endpoint. Directional radiance
 * includes every endpoint, including those without finite source information. */
struct GuidingMixtureObservation {
  float4 direction_weight;
  float4 position;
  float4 source;
};

/* One component's sufficient statistics. A component owns its accumulator;
 * neither collection nor publication requires global floating-point atomics.
 * Radiance rescaling is common to all components. Fractional responsibilities
 * enter squared-weight statistics quadratically, preserving effective support. */
struct GuidingMixtureStatistics {
  ccl_static_constexpr int position_size = GuidingPositionMoments::storage_size;
  ccl_static_constexpr int directional_offset = position_size +
                                                GuidingParallaxMoments::storage_size;
  /* Sum(w^2 * responsibility), unlike entry 22's sum((w*r)^2).
   * Summing this entry over components recovers the global squared weight. */
  ccl_static_constexpr int global_squared_weight = directional_offset + 4;
  ccl_static_constexpr int storage_size = global_squared_weight + 1;
  ccl_static_constexpr int working_component_size = 2 * storage_size + 1;
  ccl_static_constexpr int working_size = GuidingGaussianMixture::components *
                                          working_component_size;
  float values[storage_size] = {};
  float errors[storage_size] = {};
  /* Set once before collection. All batches in an interval use the same frozen
   * component's reference. These four centered moments are linear in weight. */
  float3 direction_reference = make_float3(0, 0, 1);

  ccl_device_inline_method static bool squared_weight(const int entry)
  {
    return entry == 22 || entry == position_size + 1 || entry == position_size + 3 ||
           entry == global_squared_weight;
  }

  /* Merge disjoint batches collected against the SAME immutable input mixture.
   * Each batch uses its own maximum radiance as a scale. Raising the common scale
   * rescales linear and squared moments differently; treating all entries as linear
   * changes effective sample size and the finite-source fit. Keep compensation
   * across groups, including the low part of each incoming sum. This does not run
   * another E-step or count the incoming observations more than once. */
  ccl_device_inline_method bool merge(const ccl_private GuidingMixtureStatistics &batch,
                                      const float batch_scale,
                                      ccl_private float &scale)
  {
    if (!(batch_scale > 0.0f) || !isfinite_safe(batch_scale) || !(scale >= 0.0f) ||
        !isfinite_safe(scale))
    {
      return false;
    }
    if (!isfinite_safe(batch.direction_reference) ||
        (scale > 0.0f && (direction_reference.x != batch.direction_reference.x ||
                          direction_reference.y != batch.direction_reference.y ||
                          direction_reference.z != batch.direction_reference.z)))
    {
      return false;
    }
    const float target = max(scale, batch_scale);
    const float old_ratio = scale / target;
    const float batch_ratio = batch_scale / target;
    for (int i = 0; i < storage_size; ++i) {
      if (!isfinite_safe(batch.values[i]) || !isfinite_safe(batch.errors[i])) {
        return false;
      }
    }
    for (int i = 0; i < storage_size; ++i) {
      const float old_factor = squared_weight(i) ? old_ratio * old_ratio : old_ratio;
      const float batch_factor = squared_weight(i) ? batch_ratio * batch_ratio : batch_ratio;
      values[i] *= old_factor;
      errors[i] *= old_factor;
      add(i, batch.values[i] * batch_factor);
      add(i, -batch.errors[i] * batch_factor);
    }
    direction_reference = batch.direction_reference;
    scale = target;
    return true;
  }

  ccl_device_inline_method void add(const int entry, const float value)
  {
#ifdef __clang__
#  pragma clang fp reassociate(off)
#endif
    const float corrected = value - errors[entry];
    const float updated = values[entry] + corrected;
    errors[entry] = (updated - values[entry]) - corrected;
    values[entry] = updated;
  }

  ccl_device_inline_method void record(const GuidingMixtureObservation observation,
                                       const float responsibility,
                                       const float maximum_weight,
                                       const float3 metric_extent)
  {
    if (!GuidingDirectionalMixtureFit<1>::valid(observation.direction_weight) ||
        !(responsibility > 0.0f) || !isfinite_safe(responsibility) || !(maximum_weight > 0.0f) ||
        !isfinite_safe(maximum_weight) || !isfinite_safe(make_float3(observation.position)) ||
        !isfinite_safe(direction_reference))
    {
      return;
    }
    /* Observations already contain validated unit directions. Renormalizing here
     * introduces a backend-dependent offset into centered sums (Metal and CPU
     * normalize use different arithmetic), without adding directional information. */
    const float3 direction = make_float3(observation.direction_weight);
    const float3 position = make_float3(observation.position);
    const float w = (observation.direction_weight.w / maximum_weight) * responsibility;
    add(0, w);
    for (int i = 0; i < 3; ++i) {
      add(1 + i, w * direction[i]);
      add(4 + i, w * position[i]);
      for (int j = 0; j < 3; ++j) {
        add(13 + 3 * i + j, w * direction[i] * position[j]);
      }
    }
    int entry = 7;
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        add(entry++, w * position[i] * position[j]);
      }
    }
    add(22, w * w);
    add(global_squared_weight, w * (observation.direction_weight.w / maximum_weight));
    const float3 offset = direction - direction_reference;
    for (int i = 0; i < 3; ++i) {
      add(directional_offset + i, w * offset[i]);
    }
    add(directional_offset + 3, w * len_squared(offset));

    const float finite_weight = (observation.source.x / maximum_weight) * responsibility;
    const float harmonic_weight = (observation.source.y / maximum_weight) * responsibility;
    const float distance_weight = (observation.source.z / maximum_weight) * responsibility;
    const float3 p = position * metric_extent;
    if (!(finite_weight > 0.0f) || !(harmonic_weight > 0.0f) || !(distance_weight >= 0.0f) ||
        !isfinite_safe(finite_weight * finite_weight) ||
        !isfinite_safe(harmonic_weight * harmonic_weight) || !isfinite_safe(distance_weight) ||
        !isfinite_safe(p))
    {
      return;
    }
    add(position_size, finite_weight);
    add(position_size + 1, finite_weight * finite_weight);
    add(position_size + 2, harmonic_weight);
    add(position_size + 3, harmonic_weight * harmonic_weight);
    for (int i = 0; i < 3; ++i) {
      add(position_size + 4 + i, harmonic_weight * p[i] + finite_weight * direction[i]);
    }
    entry = position_size + 7;
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        add(entry++,
            harmonic_weight * p[i] * p[j] +
                finite_weight * (p[i] * direction[j] + direction[i] * p[j]) +
                distance_weight * direction[i] * direction[j]);
      }
    }
  }

  ccl_device_inline_method GuidingSphericalGaussian directional_fit() const
  {
    if (!(values[0] > 0.0f)) {
      return {make_float3(0, 0, 1), 0.0f};
    }
    const float3 offset = make_float3(values[directional_offset],
                                      values[directional_offset + 1],
                                      values[directional_offset + 2]) /
                          values[0];
    const float3 mean = direction_reference + offset;
    const float length = len(mean);
    const float variance = max(values[directional_offset + 3] / values[0] - len_squared(offset),
                               0.0f);
    const float resultant = safe_sqrtf(1.0f - variance);
    const float concentration = variance < .1f ?
                                    (variance > 0.0f ?
                                         min((1.0f + resultant) / variance, 16384.0f) :
                                         16384.0f) :
                                    GuidingDirectionalMixtureFit<1>::concentration(resultant);
    return {length > 0.0f ? mean / length : make_float3(0, 0, 1), concentration};
  }

  /* Change only the coordinate origin of the centered moments. This permits a
   * decayed prior to be carried into the next interval's frozen component frame.
   * The represented observations and their mass are unchanged. */
  ccl_device_inline_method bool recenter(const float3 reference)
  {
    if (!isfinite_safe(reference)) {
      return false;
    }
    const float3 shift = direction_reference - reference;
    const float3 sum = make_float3(values[directional_offset],
                                   values[directional_offset + 1],
                                   values[directional_offset + 2]);
    add(directional_offset + 3, 2.0f * dot(shift, sum));
    add(directional_offset + 3, values[0] * len_squared(shift));
    for (int i = 0; i < 3; ++i) {
      add(directional_offset + i, values[0] * shift[i]);
    }
    direction_reference = reference;
    return true;
  }

  ccl_device_inline_method void publish(ccl_global float *storage) const
  {
    for (int i = 0; i < storage_size; ++i) {
      storage[i] = values[i];
    }
  }
};

CCL_NAMESPACE_END
