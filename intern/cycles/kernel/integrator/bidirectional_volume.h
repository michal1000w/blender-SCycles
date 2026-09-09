/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/types.h"
#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Throughput-independent weighted null tracking. Coefficients must be finite
 * and nonnegative. A conservative proposal rate gives nonnegative weights;
 * underestimating extinction requires signed null weights, never density clamps.
 * Absorption is represented by the spectral weights, not a third roulette event.
 * The probabilities can therefore be evaluated on a reversed realized path
 * without knowing its incoming throughput or selected spectral channel.
 *
 * This primitive does not implement medium traversal or connection MIS. Callers
 * must retain null-choice probabilities and distinguish random-walk sampling
 * from deterministic-null ratio tracking on shadow connections. */
struct BDPTVolumeEventDistribution {
  Spectrum real_weight = zero_spectrum();
  Spectrum null_weight = zero_spectrum();
  float real_probability = 0.0f;
  float null_probability = 1.0f;
  bool valid = false;

  ccl_device_inline_method static BDPTVolumeEventDistribution make(const Spectrum sigma_t,
                                                                   const Spectrum sigma_s,
                                                                   const float majorant)
  {
#ifdef __clang__
#  pragma clang fp reassociate(off)
#endif
    BDPTVolumeEventDistribution result;
    if (!(majorant > 0.0f) || !isfinite_safe(majorant) || !isfinite_safe(sigma_t) ||
        !isfinite_safe(sigma_s) || reduce_min(sigma_s) < 0.0f ||
        reduce_min(sigma_t - sigma_s) < 0.0f)
    {
      return result;
    }
    if (is_zero(sigma_t)) {
      result.null_weight = one_spectrum();
      result.valid = true;
      return result;
    }
    if (is_zero(make_spectrum(majorant) - sigma_s) && is_zero(sigma_t - sigma_s)) {
      result.real_probability = 1.0f;
      result.null_probability = 0.0f;
      result.real_weight = one_spectrum();
      result.valid = true;
      return result;
    }
    const Spectrum real_fraction = sigma_s / majorant;
    /* Subtract before division: approximate GPU reciprocals must not turn an
     * exact majorant-boundary zero into a negative null coefficient. */
    const Spectrum null_fraction = (make_spectrum(majorant) - sigma_t) / majorant;
    const bool underestimated = reduce_max(sigma_t) > majorant;
    if (underestimated) {
      /* Weighted delta tracking with a positive event proposal proportional to
       * the magnitudes of scattering and null coefficients. Scale before taking
       * averages, so finite large coefficients do not overflow the score sum.
       * The signs belong to transport weights, never to sampling probabilities.
       * This remains an exact estimator for a non-bounding rate, but a poor rate
       * can have very high variance. Callers must preserve signed contributions. */
      const float scale = reduce_max(sigma_t);
      const float real_score = average(sigma_s / scale);
      const float null_score = average(fabs((make_spectrum(majorant) - sigma_t) / scale));
      result.real_probability = real_score / (real_score + null_score);
    }
    else {
      result.real_probability = average(real_fraction);
    }
    result.null_probability = 1.0f - result.real_probability;
    if (result.real_probability > 0.0f) {
      result.real_weight = real_fraction / result.real_probability;
    }
    if (result.null_probability > 0.0f) {
      result.null_weight = null_fraction / result.null_probability;
    }
    // Reject representational loss of support rather than silently replacing a
    // nonzero event with zero probability. Exact zero-support endpoints are valid.
    result.valid = result.real_probability >= 0.0f && result.real_probability <= 1.0f &&
                   (result.real_probability > 0.0f || is_zero(sigma_s)) &&
                   (result.null_probability > 0.0f || is_zero(null_fraction)) &&
                   isfinite_safe(result.real_weight) && isfinite_safe(result.null_weight) &&
                   reduce_min(result.real_weight) >= 0.0f &&
                   (underestimated || reduce_min(result.null_weight) >= 0.0f);
    return result;
  }

  /* random is in [0,1), and valid must be true. */
  ccl_device_inline_method bool sample_real(const float random) const
  {
    return random < real_probability;
  }
};

CCL_NAMESPACE_END
