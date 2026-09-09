/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/sample/guiding_mixture_statistics.h"

CCL_NAMESPACE_BEGIN

/* Publication from soft component statistics into the shared query layout.
 * Callers retain an immutable input distribution throughout responsibility
 * collection and write this result into a separate output distribution. */
struct GuidingConditionalMixture {
  ccl_device_inline float log_component(const ccl_global float *storage,
                                        const int component,
                                        const float3 direction,
                                        const float3 position)
  {
    const float mass = storage[component * GuidingGaussianMixture::component_stride];
    if (!(mass > 0.0f)) {
      return -FLT_MAX;
    }
    GuidingGaussianMixture mixture;
    const auto lobe = mixture.component(storage, component, position);
    return logf(mass) + logf(lobe.normalization()) +
           lobe.concentration * (min(dot(lobe.axis, direction), 1.0f) - 1.0f);
  }

  ccl_device_inline void publish_component(ccl_global float *output,
                                           const int component,
                                           const ccl_global float *statistics,
                                           const float total_weight,
                                           const uint observations,
                                           const GuidingSphericalGaussian fallback,
                                           const float3 metric_extent,
                                           const float effective_observations = 0.0f)
  {
    ccl_global float *entry = output + component * GuidingGaussianMixture::component_stride;
    for (int i = 0; i < GuidingGaussianMixture::component_stride; ++i) {
      entry[i] = 0.0f;
    }
    entry[4] = 1.0f;
    if (!(total_weight > 0.0f)) {
      entry[0] = component == 0 ? 1.0f : 0.0f;
      return;
    }
    entry[0] = statistics[0] / total_weight;
    if (effective_observations > 0.0f) {
      /* Dirichlet weight prior and zero-mean directional prior, in effective
       * sample units. Constants match OpenPGL's default prior strengths; using
       * weighted support prevents a single high-radiance path posing as many. */
      entry[0] = (entry[0] * effective_observations + 0.01f) /
                 (effective_observations + 0.01f * GuidingGaussianMixture::components);
      const float support = statistics[22] > 0.0f ? sqr(statistics[0]) / statistics[22] : 0.0f;
      entry[18] = support > 0.0f ? 0.2f / support : 0.0f;
    }
    entry[1] = fallback.concentration;
    entry[2] = fallback.axis.x;
    entry[3] = fallback.axis.y;
    entry[4] = fallback.axis.z;
    if (!(statistics[0] > 0.0f)) {
      return;
    }
    GuidingParallaxMoments physical;
    const auto source = physical.fit(statistics + GuidingMixtureStatistics::position_size,
                                     statistics[0],
                                     observations,
                                     fallback.axis);
    if (source.valid) {
      for (int i = 0; i < 3; ++i) {
        entry[5 + i] = source.target[i];
        entry[14 + i] = metric_extent[i];
      }
      for (int i = 0; i < 6; ++i) {
        entry[8 + i] = source.covariance[i];
      }
      entry[17] = 2.0f;
      return;
    }
    GuidingPositionMoments positional;
    const auto fit = positional.fit(statistics, observations);
    if (fit.valid) {
      const float r = min(safe_sqrtf(1.0f - fit.residual_variance), 1.0f - 1e-6f);
      entry[1] = min(r * (3.0f - r * r) / (1.0f - r * r), 16384.0f);
      for (int i = 0; i < 3; ++i) {
        entry[2 + i] = fit.mean_direction[i];
        entry[5 + i] = fit.mean_position[i];
        for (int j = 0; j < 3; ++j) {
          entry[8 + 3 * i + j] = fit.slope[i][j];
        }
      }
      entry[17] = 1.0f;
    }
    /* These concentrations are independent of the query position. Apply their
     * prior once; finite-source fits return above and retain it for each query. */
    entry[1] = GuidingSphericalGaussian{fallback.axis, entry[1]}
                   .regularized(1.0f, entry[18])
                   .concentration;
    entry[18] = 0.0f;
  }
};

CCL_NAMESPACE_END
