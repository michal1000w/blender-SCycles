/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/types.h"
#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* The two compact terms are sums of individual strategy ratios raised to an
 * exponent. Transform factors before summing; raising an accumulated sum adds
 * spurious cross terms. Exponent 1 is balance; exponent 2 is power weighting.
 * A different exponent requires matching endpoint/geometry conversions too. */
template<int Exponent> struct BDPTMISRecurrence {
  static_assert(Exponent == 1 || Exponent == 2);

  ccl_device_inline_method static float factor(const float ratio)
  {
    return Exponent == 1 ? ratio : ratio * ratio;
  }

  /* PDFs must already satisfy the integrator's nonzero forward-PDF contract.
   * x is dVCM, y is dVC. Reservoir contribution compensation is not a PDF. */
  ccl_device_inline_method static float2 scatter(const float2 previous,
                                                 const float cosine,
                                                 const float forward_pdf,
                                                 const float reverse_pdf,
                                                 const float selection_ratio = 1.0f)
  {
    return make_float2(factor(1.0f / forward_pdf),
                       factor(cosine / forward_pdf) * (previous.y * factor(reverse_pdf) +
                                                       previous.x * factor(selection_ratio)));
  }
};

/* Logarithms of the same two sums. Compose ratios from their numerator and
 * denominator before either division or exponentiation can overflow. PDFs are
 * nonnegative and forward densities are positive; delta exclusions use zero
 * terms explicitly, not artificial nonzero densities. */
template<int Exponent> struct BDPTMISLogRecurrence {
  static_assert(Exponent == 1 || Exponent == 2);

  ccl_device_inline_method static float factor(const float density)
  {
    return density > 0.0f ? Exponent * logf(density) : -INFINITY;
  }

  ccl_device_inline_method static float ratio(const float numerator, const float denominator)
  {
    return numerator > 0.0f ? Exponent * (logf(numerator) - logf(denominator)) : -INFINITY;
  }

  ccl_device_inline_method static float product(const float a, const float b)
  {
    return a == -INFINITY || b == -INFINITY ? -INFINITY : a + b;
  }

  ccl_device_inline_method static float sum(const float a, const float b)
  {
    if (a == -INFINITY) {
      return b;
    }
    if (b == -INFINITY) {
      return a;
    }
    const float largest = max(a, b);
    return largest == INFINITY ? largest : largest + logf(1.0f + expf(min(a, b) - largest));
  }

  ccl_device_inline_method static float2 scatter(const float2 previous,
                                                 const float cosine,
                                                 const float forward_pdf,
                                                 const float reverse_pdf,
                                                 const float selection_ratio = 1.0f)
  {
    return make_float2(ratio(1.0f, forward_pdf),
                       product(ratio(cosine, forward_pdf),
                               sum(product(previous.y, factor(reverse_pdf)),
                                   product(previous.x, factor(selection_ratio)))));
  }

  /* The selected strategy has unit relative density. Neither alternative sum
   * needs to be exponentiated before normalization. */
  ccl_device_inline_method static float weight(const float light_sum, const float camera_sum)
  {
    return expf(-sum(0.0f, sum(light_sum, camera_sum)));
  }
};

/* A typed logarithmic MIS term prevents stored logs from being used as linear
 * PDFs. Scalars are density factors; sums combine powered strategy terms.
 * Encoding/decoding at path/cache storage boundaries is always explicit. */
template<int Exponent> struct BDPTMISWeightT {
  using Log = BDPTMISLogRecurrence<Exponent>;
  float value;

  ccl_device_inline_method BDPTMISWeightT(const float density = 0.0f) : value(Log::factor(density))
  {
  }

  ccl_device_inline_method static BDPTMISWeightT from_encoded(const float value)
  {
    BDPTMISWeightT result;
    result.value = value;
    return result;
  }

  ccl_device_inline_method float encoded() const
  {
    return value;
  }
  ccl_device_inline_method float inverse() const
  {
    return expf(-value);
  }
  /* The input is an unpowered log density, not an encoded MIS term. This
   * keeps products of many null-event probabilities out of ordinary float space. */
  ccl_device_inline_method BDPTMISWeightT scaled_by_log_density(const float log_density) const
  {
    return from_encoded(Log::product(value, Exponent * log_density));
  }

  /* This term is the sum of competing strategy ratios before the selected
   * ratio-tracked connection removes its random-walk null-choice probability.
   * Its log probability is nonpositive. Zero denotes a vacuum connection. */
  ccl_device_inline_method float connection_weight(const float log_null_probability = 0.0f) const
  {
    return (BDPTMISWeightT(1.0f) + scaled_by_log_density(log_null_probability)).inverse();
  }

  ccl_device_inline_method bool operator==(const float density) const
  {
    return value == Log::factor(density);
  }
  ccl_device_inline_method BDPTMISWeightT operator+(const BDPTMISWeightT other) const
  {
    return from_encoded(Log::sum(value, other.value));
  }
  ccl_device_inline_method BDPTMISWeightT operator*(const BDPTMISWeightT other) const
  {
    return from_encoded(Log::product(value, other.value));
  }
  ccl_device_inline_method BDPTMISWeightT operator/(const float density) const
  {
    return from_encoded(value == -INFINITY ? value : value - Log::factor(density));
  }
  ccl_device_inline_method ccl_private BDPTMISWeightT &operator*=(const float density)
  {
    value = Log::product(value, Log::factor(density));
    return *this;
  }
  ccl_device_inline_method ccl_private BDPTMISWeightT &operator/=(const float density)
  {
    *this = *this / density;
    return *this;
  }
};

// Experimental power candidate; acceptance requires matched-work renderer checks.
using BDPTMISWeight = BDPTMISWeightT<2>;

/* Caustic controls concern sharp reflection/transmission. Diffuse scattering
 * and null transmission must not terminate a light path when those controls
 * are disabled. This classifies the current event; it does not infer the full
 * future transport class of a light subpath. */
ccl_device_inline bool bdpt_caustic_event_enabled(const int label,
                                                  const bool reflective,
                                                  const bool refractive)
{
  if ((label & LABEL_TRANSPARENT) || !(label & (LABEL_GLOSSY | LABEL_SINGULAR))) {
    return true;
  }
  return (!((label & LABEL_REFLECT) && !reflective) && !((label & LABEL_TRANSMIT) && !refractive));
}

/* Solid angle subtended by one pixel of the image/focus plane, conditional on
 * the sampled aperture point. The plane basis also supports shifted lenses. */
ccl_device_inline float bdpt_camera_plane_inverse_pdf(const float3 sensor_to_plane,
                                                      const float3 pixel_dx,
                                                      const float3 pixel_dy)
{
  const float distance_squared = len_squared(sensor_to_plane);
  return distance_squared > 0.0f ? fabsf(dot(cross(pixel_dx, pixel_dy), sensor_to_plane)) /
                                       (distance_squared * sqrtf(distance_squared)) :
                                   0.0f;
}

/* Intersection distances start at the near clipping plane, but a perspective/panoramic
 * sensor's solid-angle density originates at the camera. The following hit conversion
 * multiplies by the clipped distance squared, so undo that measure here. */
ccl_device_inline float bdpt_camera_clip_measure(const float inverse_direction_pdf,
                                                 const float sensor_distance_squared,
                                                 const float intersection_distance)
{
  return inverse_direction_pdf * sensor_distance_squared /
         sqr(max(intersection_distance, 1.0e-10f));
}

/* Transpose the camera scattering operator in geometric projected-area measure. The reciprocal
 * camera evaluation already contains its sampled direction's shading-normal cosine and the
 * correct dielectric transport factor; only the geometric measure ratio remains. */
ccl_device_inline Spectrum bdpt_transpose_surface_eval(const Spectrum reciprocal_eval,
                                                       const float3 geometric_normal,
                                                       const float3 incoming,
                                                       const float3 outgoing)
{
  const float cosine_in = fabsf(dot(geometric_normal, incoming));
  return cosine_in > 0.0f ?
             reciprocal_eval * (fabsf(dot(geometric_normal, outgoing)) / cosine_in) :
             zero_spectrum();
}

/* Dirac closures cannot be evaluated at a direction. Transpose their sampled weight using the
 * shading-normal correction and, for transmission, the dielectric measure conversion. */
ccl_device_inline Spectrum bdpt_transpose_delta_eval(const Spectrum forward_eval,
                                                     const float3 shading_normal,
                                                     const float3 geometric_normal,
                                                     const float3 incoming,
                                                     const float3 outgoing,
                                                     const float eta,
                                                     const bool transmission)
{
  const float denominator = fabsf(dot(geometric_normal, incoming) * dot(shading_normal, outgoing));
  if (!(denominator > 0.0f)) {
    return zero_spectrum();
  }
  const float numerator = fabsf(dot(shading_normal, incoming) * dot(geometric_normal, outgoing));
  return forward_eval * (numerator / denominator / (transmission ? sqr(eta) : 1.0f));
}

/* Roulette must retain signed transport, including non-bounding weighted null
 * events. Use the selected path's magnitude; rejected RIS candidates must not
 * affect this probability. Nonnegative paths retain the existing probability. */
ccl_device_inline float bdpt_light_path_continuation_probability(const Spectrum throughput,
                                                                 const Spectrum unguided_factor)
{
  return min(saturatef(reduce_max(fabs(throughput * unguided_factor))), 0.95f);
}

CCL_NAMESPACE_END
