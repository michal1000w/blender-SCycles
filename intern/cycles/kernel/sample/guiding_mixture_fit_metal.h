/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/sample/guiding_mixture_conditional.h"

CCL_NAMESPACE_BEGIN

#ifdef __KERNEL_METAL__
/* Collect against one immutable model. Low 16 lanes return complete component
 * statistics; high lanes only provide the second observation stream. */
template<typename Range>
ccl_device_inline void guiding_mixture_collect_cooperative(
    const ccl_global float *model,
    const ccl_private Range &range,
    const uint count,
    const uint lane,
    const float maximum_weight,
    const float3 metric_extent,
    ccl_private GuidingMixtureStatistics &batch)
{
  using Stats = GuidingMixtureStatistics;
  static_assert(GuidingGaussianMixture::components == 16);
  const uint component_index = lane % GuidingGaussianMixture::components;
  const uint base = component_index * GuidingGaussianMixture::component_stride;
  batch.direction_reference = make_float3(model[base + 2], model[base + 3], model[base + 4]);
  GuidingConditionalMixture conditional;
  /* Two independent 16-lane component reductions use every lane of the SIMD group.
   * Both halves retain compensated statistics at the same radiance scale. */
  for (uint i = 0; i < count; i += 2) {
    const uint observation_index = i + lane / GuidingGaussianMixture::components;
    const auto observation = range.observation(min(observation_index, count - 1));
    const float log_density = observation_index < count ?
                                  conditional.log_component(
                                      model,
                                      component_index,
                                      make_float3(observation.direction_weight),
                                      make_float3(observation.position)) :
                                  -FLT_MAX;
    float maximum = log_density;
    for (uint offset = 8; offset > 0; offset /= 2) {
      maximum = max(maximum, metal::simd_shuffle_xor(maximum, offset));
    }
    const float density = log_density > -FLT_MAX ? expf(log_density - maximum) : 0;
    float sum = density;
    for (uint offset = 8; offset > 0; offset /= 2) {
      sum += metal::simd_shuffle_xor(sum, offset);
    }
    if (observation_index < count && sum > 0) {
      batch.record(observation, density / sum, maximum_weight, metric_extent);
    }
  }
  for (int i = 0; i < Stats::storage_size; ++i) {
    const float other_value = metal::simd_shuffle_xor(batch.values[i], 16);
    const float other_error = metal::simd_shuffle_xor(batch.errors[i], 16);
    if (lane < GuidingGaussianMixture::components) {
      batch.add(i, other_value);
      batch.add(i, -other_error);
    }
  }
}
#endif

CCL_NAMESPACE_END
