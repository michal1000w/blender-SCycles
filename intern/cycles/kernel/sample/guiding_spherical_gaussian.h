/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/sample/guiding_parallax.h"

#include "kernel/sample/guiding_distribution.h"
#include "kernel/sample/guiding_position.h"

CCL_NAMESPACE_BEGIN

/* A normalized von Mises-Fisher lobe on the sphere. The shifted exponential avoids overflow
 * for narrow glossy lobes. Products remain in this family and have an analytic integral. */
struct GuidingSphericalGaussian {
  float3 axis;
  float concentration;

  /* Mean cosine A_3(kappa) of the normalized spherical distribution. Avoid the
   * cancellation in coth(kappa) - 1/kappa near zero and overflow at large kappa. */
  ccl_device_inline_method static float mean_resultant(const float kappa)
  {
    if (kappa < 0.1f) {
      const float k2 = kappa * kappa;
      return kappa * (1.0f / 3.0f - k2 * (1.0f / 45.0f - k2 * (2.0f / 945.0f)));
    }
    if (kappa > 10.0f) {
      return 1.0f - 1.0f / kappa;
    }
    const float e = expf(-2.0f * kappa);
    return (1.0f + e) / (1.0f - e) - 1.0f / kappa;
  }

  /* Invert A_3 using a rational initial estimate and safeguarded Newton steps.
   * This is needed when regularizing an already fitted conditional lobe: its
   * concentration is not itself a mean cosine and cannot be averaged as one. */
  ccl_device_inline_method static float concentration_from_resultant(const float resultant)
  {
    const float r = clamp(resultant, 0.0f, 1.0f);
    if (r == 0.0f) {
      return 0.0f;
    }
    float lower = 0.0f, upper = 16384.0f;
    if (r >= 1.0f - 1.0f / upper) {
      return upper;
    }
    float k = min(r * (3.0f - r * r) / (1.0f - r * r), upper);
    for (int step = 0; step < 12; ++step) {
      float mean, derivative;
      if (k < .1f) {
        const float k2 = k * k;
        mean = k * (1.0f / 3.0f - k2 / 45.0f + 2.0f * k2 * k2 / 945.0f);
        derivative = 1.0f / 3.0f - k2 / 15.0f + 2.0f * k2 * k2 / 189.0f;
      }
      else {
        const float e = expf(-2.0f * k);
        const float denominator = GuidingSphericalGaussian::one_minus_exp_negative(2.0f * k);
        mean = 1.0f + 2.0f * e / denominator - 1.0f / k;
        derivative = 1.0f / (k * k) - 4.0f * e / (denominator * denominator);
      }
      if (fabsf(mean - r) < 2e-7f) {
        break;
      }
      if (mean < r) {
        lower = k;
      }
      else {
        upper = k;
      }
      const float next = k - (mean - r) / derivative;
      k = isfinite_safe(next) && next > lower && next < upper ? next : .5f * (lower + upper);
    }
    return k;
  }

  /* Zero-mean directional prior, expressed in effective-observation units.
   * A caller must apply this after conditioning (including source parallax),
   * otherwise a later conditional fit can undo the prior. A zero prior is an
   * exact identity so unregularized models preserve their existing queries. */
  ccl_device_inline_method GuidingSphericalGaussian regularized(const float effective_support,
                                                                const float prior_strength) const
  {
    if (!(prior_strength > 0.0f)) {
      return *this;
    }
    if (!(effective_support > 0.0f)) {
      return {axis, 0.0f};
    }
    const float r = mean_resultant(concentration) / (1.0f + prior_strength / effective_support);
    return {axis, min(concentration, concentration_from_resultant(r))};
  }

  ccl_device_inline_method static float one_minus_exp_negative(const float x)
  {
    if (x < 0.05f) {
      return x * (1.0f - x * (0.5f - x * (1.0f / 6.0f - x * (1.0f / 24.0f - x / 120.0f))));
    }
    return 1.0f - expf(-x);
  }

  ccl_device_inline_method static float log_one_minus(const float x)
  {
    if (x < 0.01f) {
      return -x * (1.0f + x * (0.5f + x * (1.0f / 3.0f + x * (0.25f + x * 0.2f))));
    }
    return logf(1.0f - x);
  }

  ccl_device_inline_method float normalization() const
  {
    return concentration > 0.0f ?
               concentration / (M_2PI_F * one_minus_exp_negative(2.0f * concentration)) :
               M_1_4PI_F;
  }

  ccl_device_inline_method float pdf(const float3 direction) const
  {
    return normalization() * expf(concentration * (min(dot(axis, direction), 1.0f) - 1.0f));
  }

  ccl_device_inline_method float3 sample(float2 random, ccl_private float *sample_pdf) const
  {
    random = clamp(random, zero_float2(), make_float2(0x1.fffffep-1f));
    const float cosine = concentration > 0.0f ?
                             clamp(1.0f + log_one_minus(random.x * one_minus_exp_negative(
                                                                       2.0f * concentration)) /
                                              concentration,
                                   -1.0f,
                                   1.0f) :
                             1.0f - 2.0f * random.x;
    const float sine = safe_sqrtf(1.0f - cosine * cosine);
    const float phi = M_2PI_F * random.y;
    float3 tangent, bitangent;
    make_orthonormals(axis, &tangent, &bitangent);
    const float3 direction = normalize(axis * cosine +
                                       sine * (tangent * cosf(phi) + bitangent * sinf(phi)));
    *sample_pdf = pdf(direction);
    return direction;
  }

  ccl_device_inline_method GuidingSphericalGaussian product(const GuidingSphericalGaussian other,
                                                            ccl_private float *integral) const
  {
    /* A uniform factor changes only the integral. Reconstructing the natural
     * parameter would needlessly renormalize the axis and perturb a narrow PDF. */
    if (other.concentration == 0.0f) {
      *integral = M_1_4PI_F;
      return *this;
    }
    if (concentration == 0.0f) {
      *integral = M_1_4PI_F;
      return other;
    }
    const float3 vector = axis * concentration + other.axis * other.concentration;
    const float length = len(vector);
    const GuidingSphericalGaussian result = {
        length > 0.0f ? vector / length : make_float3(0, 0, 1), length};
    *integral = normalization() * other.normalization() / result.normalization() *
                expf(min(length - concentration - other.concentration, 0.0f));
    return result;
  }
};

struct GuidingGaussianProduct {
  GuidingSphericalGaussian lobes[2];
  float weights[2];
};

/* A smooth companion to the histogram, used when a narrow scattering lobe needs more angular
 * resolution than a coarse piecewise-constant product can provide. Stored components contain
 * mass, concentration, direction, positional regression, and a conditioning flag. Fitting
 * is performed only at publication; queries never mutate the model. */
struct GuidingGaussianMixture {
  ccl_static_constexpr int components = 16;
  /* Entry 18 is prior strength / effective support, zero for unregularized fits. */
  ccl_static_constexpr int component_stride = 19;
  ccl_static_constexpr int storage_size = components * component_stride;
  ccl_static_constexpr float exploration = 0.05f;

  ccl_device_inline GuidingSphericalGaussian component(const ccl_global float *storage,
                                                       const int i,
                                                       const float3 position = zero_float3())
  {
    const ccl_global float *entry = storage + component_stride * i;
    const float3 direction = make_float3(entry[2], entry[3], entry[4]);
    if (entry[17] == 0.0f) {
      return GuidingSphericalGaussian{direction, entry[1]}.regularized(1.0f, entry[18]);
    }
    if (entry[17] == 2.0f) {
      GuidingParallaxFit source{};
      source.valid = true;
      source.fallback_direction = direction;
      source.target = make_float3(entry[5], entry[6], entry[7]);
      for (int j = 0; j < 6; ++j) {
        source.covariance[j] = entry[8 + j];
      }
      const float3 metric_position = position * make_float3(entry[14], entry[15], entry[16]);
      return GuidingSphericalGaussian{source.direction(metric_position),
                                      source.concentration(metric_position)}
          .regularized(1.0f, entry[18]);
    }
    GuidingPositionFit fit{};
    fit.mean_direction = direction;
    fit.mean_position = make_float3(entry[5], entry[6], entry[7]);
    for (int row = 0; row < 3; ++row) {
      fit.slope[row] = make_float3(entry[8 + 3 * row], entry[9 + 3 * row], entry[10 + 3 * row]);
    }
    return GuidingSphericalGaussian{fit.direction(position), entry[1]}.regularized(1.0f,
                                                                                   entry[18]);
  }

  ccl_device_inline void build(ccl_global float *storage,
                               const ccl_global float *tree,
                               const ccl_global float *observed_moments = nullptr,
                               const ccl_global float *counts = nullptr,
                               const ccl_global float *source_moments = nullptr,
                               const float3 metric_extent = one_float3())
  {
    using Tree = GuidingDirectionalTree<5>;
    Tree mapping;
#pragma unroll 1
    for (int i = 0; i < components; ++i) {
      for (int j = 0; j < component_stride; ++j) {
        storage[component_stride * i + j] = j == 4 ? 1.0f : 0.0f;
      }
      if (!(tree[5 + i] > 0.0f)) {
        continue;
      }
      float mass = 0.0f;
      float3 moment = zero_float3();
#pragma unroll 1
      for (int j = 0; j < Tree::leaf_count / components; ++j) {
        const int leaf = i * (Tree::leaf_count / components) + j;
        int x = 0, y = 0;
        for (int bit = 0; bit < 5; ++bit) {
          x |= ((leaf >> (2 * bit)) & 1) << bit;
          y |= ((leaf >> (2 * bit + 1)) & 1) << bit;
        }
        const float weight = tree[Tree::leaf_offset + leaf];
        mass += weight;
        moment += weight * mapping.square_to_direction(make_float2((x + 0.5f) / Tree::resolution,
                                                                   (y + 0.5f) / Tree::resolution));
      }
      float moment_mass = mass;
      float max_concentration = 512.0f;
      GuidingPositionFit fit{};
      if (observed_moments && counts && counts[5 + i] >= 32.0f) {
        const ccl_global float *observations = observed_moments +
                                               GuidingPositionMoments::storage_size * i;
        const float observed_mass = observations[0];
        const float3 observed = make_float3(observations[1], observations[2], observations[3]);
        if (observed_mass > 0.0f && isfinite_safe(observed_mass) && isfinite_safe(observed)) {
          /* Actual ray directions retain angular detail below the histogram's bin width.
           * Sparse regions keep the regularized histogram fit until observations support it. */
          moment = observed;
          moment_mass = observed_mass;
          max_concentration = 16384.0f;
          GuidingPositionMoments model;
          fit = model.fit(observations, counts[5 + i]);
        }
      }
      /* Normalize by mass before squaring: finite radiance moments can overflow
       * len() while their mean direction remains bounded by one. */
      const float3 mean = moment_mass > 0.0f ? moment / moment_mass : zero_float3();
      const float length = len(mean);
      const float3 axis = length > 0.0f ? mean / length : make_float3(0, 0, 1);
      float r = min(length, 1.0f - 1e-6f);
      const bool conditioned = fit.valid &&
                               1.0f - len_squared(fit.mean_direction) - fit.residual_variance >
                                   1e-6f;
      if (conditioned) {
        r = min(safe_sqrtf(1.0f - fit.residual_variance), 1.0f - 1e-6f);
      }
      /* Moment-matched vMF concentration. Histogram-only fits respect the bin resolution;
       * sufficiently observed actual ray directions can support narrower lobes. */
      storage[component_stride * i] = tree[0] > 0.0f ? mass / tree[0] : 0.0f;
      storage[component_stride * i + 1] = min(r * (3.0f - r * r) / (1.0f - r * r),
                                              max_concentration);
      storage[component_stride * i + 2] = axis.x;
      storage[component_stride * i + 3] = axis.y;
      storage[component_stride * i + 4] = axis.z;
      if (conditioned) {
        storage[component_stride * i + 2] = fit.mean_direction.x;
        storage[component_stride * i + 3] = fit.mean_direction.y;
        storage[component_stride * i + 4] = fit.mean_direction.z;
        for (int row = 0; row < 3; ++row) {
          storage[component_stride * i + 5 + row] = fit.mean_position[row];
          for (int col = 0; col < 3; ++col) {
            storage[component_stride * i + 8 + 3 * row + col] = fit.slope[row][col];
          }
        }
        storage[component_stride * i + 17] = 1.0f;
      }
      if (source_moments && observed_moments && counts) {
        GuidingParallaxMoments model;
        const auto source = model.fit(source_moments + i * model.storage_size,
                                      observed_moments[i * GuidingPositionMoments::storage_size],
                                      counts[5 + i],
                                      axis);
        if (source.valid) {
          for (int j = 0; j < 3; ++j) {
            storage[component_stride * i + 2 + j] = axis[j];
            storage[component_stride * i + 5 + j] = source.target[j];
            storage[component_stride * i + 14 + j] = metric_extent[j];
          }
          for (int j = 0; j < 6; ++j) {
            storage[component_stride * i + 8 + j] = source.covariance[j];
          }
          storage[component_stride * i + 17] = 2.0f;
        }
      }
    }
  }

  ccl_device_inline float pdf(const ccl_global float *storage,
                              const float3 direction,
                              const float3 position = zero_float3())
  {
    float density = 0.0f;
#pragma unroll 1
    for (int i = 0; i < components; ++i) {
      density += storage[component_stride * i] * component(storage, i, position).pdf(direction);
    }
    return (1.0f - exploration) * density + exploration * M_1_4PI_F;
  }

  ccl_device_inline float pdf_product(const ccl_global float *storage,
                                      const GuidingGaussianProduct profile,
                                      const float3 direction,
                                      const float3 position = zero_float3())
  {
    float density = 0.0f, mass = 0.0f;
#pragma unroll 1
    for (int i = 0; i < components; ++i) {
      const GuidingSphericalGaussian illumination = component(storage, i, position);
      for (int j = 0; j < 2; ++j) {
        if (!(profile.weights[j] > 0.0f) || !(storage[component_stride * i] > 0.0f)) {
          continue;
        }
        float integral;
        const GuidingSphericalGaussian product = illumination.product(profile.lobes[j], &integral);
        const float weight = storage[component_stride * i] * profile.weights[j] * integral;
        mass += weight;
        density += weight * product.pdf(direction);
      }
    }
    return mass > 0.0f ? (1.0f - exploration) * density / mass + exploration * M_1_4PI_F :
                         M_1_4PI_F;
  }

  ccl_device_inline float3 sample_product(const ccl_global float *storage,
                                          const GuidingGaussianProduct profile,
                                          float2 random,
                                          ccl_private float *sample_pdf,
                                          const float3 position = zero_float3())
  {
    random = clamp(random, zero_float2(), make_float2(0x1.fffffep-1f));
    float weights[2 * components];
    float mass = 0.0f;
    int last_nonzero = 0;
#pragma unroll 1
    for (int i = 0; i < components; ++i) {
      const GuidingSphericalGaussian illumination = component(storage, i, position);
      for (int j = 0; j < 2; ++j) {
        float integral = 0.0f;
        if (profile.weights[j] > 0.0f && storage[component_stride * i] > 0.0f) {
          illumination.product(profile.lobes[j], &integral);
        }
        weights[2 * i + j] = storage[component_stride * i] * profile.weights[j] * integral;
        mass += weights[2 * i + j];
        if (weights[2 * i + j] > 0.0f) {
          last_nonzero = 2 * i + j;
        }
      }
    }
    float3 direction;
    if (!(mass > 0.0f) || random.x < exploration) {
      if (mass > 0.0f) {
        random.x /= exploration;
      }
      GuidingDirectionalTree<5> mapping;
      direction = mapping.square_to_direction(random);
    }
    else {
      random.x = (random.x - exploration) / (1.0f - exploration);
      float target = min(random.x * mass, __uint_as_float(__float_as_uint(mass) - 1));
      int selected = 0;
      for (; selected < last_nonzero; ++selected) {
        if (target < weights[selected]) {
          break;
        }
        target -= weights[selected];
      }
      random.x = weights[selected] > 0.0f ? target / weights[selected] : 0.0f;
      float integral;
      const GuidingSphericalGaussian product = component(storage, selected / 2, position)
                                                   .product(profile.lobes[selected % 2],
                                                            &integral);
      float component_pdf;
      direction = product.sample(random, &component_pdf);
    }
    *sample_pdf = pdf_product(storage, profile, direction, position);
    return direction;
  }
};

CCL_NAMESPACE_END
