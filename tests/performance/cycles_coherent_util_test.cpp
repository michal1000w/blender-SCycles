/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/light/coherent_util.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

using namespace ccl;

static double reference_range_difference(const float3 receiver,
                                         const float3 source_a,
                                         const float3 source_b)
{
  const double ax = double(receiver.x) - double(source_a.x);
  const double ay = double(receiver.y) - double(source_a.y);
  const double az = double(receiver.z) - double(source_a.z);
  const double bx = double(receiver.x) - double(source_b.x);
  const double by = double(receiver.y) - double(source_b.y);
  const double bz = double(receiver.z) - double(source_b.z);
  const double numerator = (double(source_b.x) - source_a.x) * (ax + bx) +
                           (double(source_b.y) - source_a.y) * (ay + by) +
                           (double(source_b.z) - source_a.z) * (az + bz);
  const double range_a = std::sqrt(ax * ax + ay * ay + az * az);
  const double range_b = std::sqrt(bx * bx + by * by + bz * bz);
  return numerator / (range_a + range_b);
}

static double circular_cycle_error(const double a, const double b)
{
  double difference = std::fmod(a - b, 1.0);
  if (difference > 0.5) {
    difference -= 1.0;
  }
  if (difference < -0.5) {
    difference += 1.0;
  }
  return std::abs(difference);
}

int main()
{
  constexpr std::array<double, 3> wavelengths = {380e-9, 550e-9, 780e-9};
  constexpr std::array<double, 5> receiver_x = {-0.004, -0.001, 0.0, 0.001, 0.004};
  const float3 source_a = make_float3(0.0f, 0.0f, 0.0f);
  const float3 source_b = make_float3(0.018f, 0.0f, 0.003f);
  constexpr float z = 3.0f;
  double max_range_error = 0.0;
  double max_phase_error = 0.0;
  double max_intensity_error = 0.0;
  double max_translated_phase_error = 0.0;
  double max_translated_intensity_error = 0.0;
  double max_translated_phase_shift = 0.0;
  int failures = 0;
  int checked = 0;

  for (const double x : receiver_x) {
    const float3 receiver = make_float3(float(x), 0.0f, z);
    const double exact_range = reference_range_difference(receiver, source_a, source_b);
    const float range = coherent_source_range_difference(receiver, source_a, source_b);
    max_range_error = std::max(max_range_error, std::abs(double(range) - exact_range));

    const float intensity_a = 1.0f / sqr(len(receiver - source_a));
    const float intensity_b = 0.7f / sqr(len(receiver - source_b));
    for (const double wavelength : wavelengths) {
      const float phase = coherent_phase_cycles(range, float(wavelength), 0.137f);
      double reference_cycles = std::fmod(exact_range / wavelength + 0.137, 1.0);
      if (reference_cycles < 0.0) {
        reference_cycles += 1.0;
      }
      const double phase_error = circular_cycle_error(phase, reference_cycles);
      max_phase_error = std::max(max_phase_error, phase_error);
      const float gamma = coherent_gaussian_mutual_coherence(range, 0.002f);
      const float measured = coherent_pair_intensity(intensity_a, intensity_b, gamma, phase);
      const double reference_gamma = std::exp(-0.5 * std::pow(exact_range / 0.002, 2));
      const double reference_intensity = double(intensity_a) + double(intensity_b) +
          2.0 * reference_gamma * std::sqrt(double(intensity_a) * intensity_b) *
              std::cos(6.2831853071795864769 * reference_cycles);
      const double intensity_error = std::abs(double(measured) - reference_intensity);
      max_intensity_error = std::max(max_intensity_error, intensity_error);
      failures += phase_error > 0.002 || intensity_error > 3e-4;
      ++checked;

      const float3 translation = make_float3(1000.0f, -700.0f, 350.0f);
      const float3 moved_receiver = receiver + translation;
      const float3 moved_a = source_a + translation;
      const float3 moved_b = source_b + translation;
      const double moved_exact_range = reference_range_difference(moved_receiver, moved_a, moved_b);
      const float moved_range = coherent_source_range_difference(moved_receiver, moved_a, moved_b);
      const float translated = coherent_phase_cycles(moved_range, float(wavelength), 0.137f);
      double moved_reference_cycles = std::fmod(moved_exact_range / wavelength + 0.137, 1.0);
      if (moved_reference_cycles < 0.0) {
        moved_reference_cycles += 1.0;
      }
      max_translated_phase_error = std::max(
          max_translated_phase_error, circular_cycle_error(translated, moved_reference_cycles));
      max_translated_phase_shift = std::max(
          max_translated_phase_shift, circular_cycle_error(translated, phase));
      const float moved_intensity_a = 1.0f / sqr(len(moved_receiver - moved_a));
      const float moved_intensity_b = 0.7f / sqr(len(moved_receiver - moved_b));
      const float moved_gamma = coherent_gaussian_mutual_coherence(moved_range, 0.002f);
      const float moved_intensity = coherent_pair_intensity(
          moved_intensity_a, moved_intensity_b, moved_gamma, translated);
      const double moved_reference_gamma = std::exp(-0.5 * std::pow(moved_exact_range / 0.002, 2));
      const double moved_reference_intensity = double(moved_intensity_a) + double(moved_intensity_b) +
          2.0 * moved_reference_gamma * std::sqrt(double(moved_intensity_a) * moved_intensity_b) *
              std::cos(6.2831853071795864769 * moved_reference_cycles);
      max_translated_intensity_error = std::max(
          max_translated_intensity_error,
          std::abs(double(moved_intensity) - moved_reference_intensity));
      std::printf("sample,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                  double(receiver.x), wavelength, double(range), double(phase), double(gamma),
                  double(measured), double(moved_range), double(translated),
                  double(moved_gamma), double(moved_intensity));
    }
  }

  failures += coherent_gaussian_mutual_coherence(0.003f, 0.0f) != 0.0f;
  failures += std::abs(coherent_gaussian_mutual_coherence(0.0f, 0.001f) - 1.0f) > 1e-7f;
  failures += max_translated_phase_error > 0.002 || max_translated_intensity_error > 3e-4;
  /* The translation phase shift below is diagnostic: float position
   * quantization at a 1000 m origin can materially alter sub-millimeter OPD. */
  const float fully_coherent = coherent_pair_intensity(1.0f, 1.0f, 1.0f, 0.5f);
  failures += std::abs(fully_coherent) > 1e-6f;
  const float constructive = coherent_pair_intensity(1.0f, 1.0f, 1.0f, 0.0f);
  failures += std::abs(constructive - 4.0f) > 1e-6f;

  std::printf("checked=%d failures=%d max_range_error_m=%.9g "
              "max_phase_error_cycles=%.9g max_intensity_error=%.9g "
              "max_1000m_translated_phase_error_cycles=%.9g "
              "max_1000m_translated_intensity_error=%.9g "
              "max_1000m_translation_phase_shift_cycles=%.9g\n",
              checked, failures, max_range_error, max_phase_error, max_intensity_error,
              max_translated_phase_error, max_translated_intensity_error,
              max_translated_phase_shift);
  return failures == 0 ? 0 : 1;
}
