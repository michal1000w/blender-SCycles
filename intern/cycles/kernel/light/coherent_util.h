/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Return r_a - r_b, where r_a = |receiver - source_a|.
 *
 * The factored expression avoids subtracting two nearly equal squared ranges.
 * It cannot recover coordinate precision already lost when positions were
 * quantized to float. */
ccl_device_inline float coherent_source_range_difference(const float3 receiver,
                                                         const float3 source_a,
                                                         const float3 source_b)
{
  const float3 to_a = receiver - source_a;
  const float3 to_b = receiver - source_b;
  const float numerator = dot(source_b - source_a, to_a + to_b);
  return numerator / (len(to_a) + len(to_b));
}

/* Return phase in cycles, reduced to [0, 1) before trigonometric evaluation.
 * Float division still limits phase accuracy when path_difference / wavelength
 * is very large; reduction cannot restore fractional bits already rounded off. */
ccl_device_inline float coherent_phase_cycles(const float path_difference,
                                              const float wavelength,
                                              const float phase_offset_cycles = 0.0f)
{
  /* Reduce the path before division so large whole-cycle quotients do not
   * discard additional fractional bits. Input path precision remains float. */
  const float path_cycles = fmodf(path_difference, wavelength) / wavelength;
  const float offset_cycles = fmodf(phase_offset_cycles, 1.0f);
  float cycles = fmodf(path_cycles + offset_cycles, 1.0f);
  if (cycles < 0.0f) {
    cycles += 1.0f;
  }
  return cycles;
}

/* Retain the low component of a compensated optical-path difference while
 * reducing a many-wavelength high component. Summing the pair as float before
 * reduction loses sub-wavelength phase at centimetre-scale arm differences. */
ccl_device_inline float coherent_phase_cycles_split(const float2 path_difference,
                                                    const float2 wavelength,
                                                    const float phase_offset_cycles = 0.0f)
{
  if (!(wavelength.x > 0.0f) || !isfinite_safe(wavelength.x + wavelength.y) ||
      !isfinite_safe(path_difference.x) || !isfinite_safe(path_difference.y))
  {
    return 0.0f;
  }
  /* q need only be a nearby integer. fmaf computes the high product's exact
   * binary difference in one rounding; the residual wavelength and optical
   * length then restore the fractional cycle lost by float division. */
  const float q = roundf(path_difference.x / wavelength.x);
  if (!isfinite_safe(q)) return 0.0f;
#ifdef __KERNEL_METAL__
  const float high_remainder = metal::fma(-q, wavelength.x, path_difference.x);
  const float corrected = metal::fma(-q, wavelength.y, high_remainder + path_difference.y);
#else
  const float high_remainder = fmaf(-q, wavelength.x, path_difference.x);
  const float corrected = fmaf(-q, wavelength.y, high_remainder + path_difference.y);
#endif
  const float reduced = fmodf(corrected, wavelength.x + wavelength.y);
  const float offset = fmodf(phase_offset_cycles, 1.0f);
  float cycles = fmodf(reduced / (wavelength.x + wavelength.y) + offset, 1.0f);
  if (cycles < 0.0f) cycles += 1.0f;
  return cycles;
}

ccl_device_inline float coherent_phase_cycles_split(const float2 path_difference,
                                                    const float wavelength,
                                                    const float phase_offset_cycles = 0.0f)
{
  return coherent_phase_cycles_split(
      path_difference, make_float2(wavelength, 0.0f), phase_offset_cycles);
}

/* Gaussian mutual coherence with coherence_length = 1 / sigma_k.
 * A zero length denotes the exact incoherent limit for distinct source paths. */
ccl_device_inline float coherent_gaussian_mutual_coherence(const float path_difference,
                                                           const float coherence_length)
{
  if (coherence_length == 0.0f) {
    return 0.0f;
  }
  const float normalized_difference = path_difference / coherence_length;
  return expf(-0.5f * normalized_difference * normalized_difference);
}

/* Scalar pair contribution for one spectral channel. Intensities must be
 * nonnegative and mutual_coherence must be in [0, 1]. */
ccl_device_inline float coherent_pair_intensity(const float intensity_a,
                                                const float intensity_b,
                                                const float mutual_coherence,
                                                const float phase_cycles)
{
  const float phase = M_2PI_F * phase_cycles;
  return intensity_a + intensity_b +
         2.0f * mutual_coherence * sqrtf(intensity_a * intensity_b) * cosf(phase);
}

CCL_NAMESPACE_END
