/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#include "kernel/light/coherent_field.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>

using namespace ccl;

static double power(const std::complex<double> z)
{
  return std::norm(z);
}

static double reference_pair(const std::complex<double> a,
                             const std::complex<double> b,
                             const double opd,
                             const double wavelength,
                             const double length,
                             const double phase_cycles = 0.0)
{
  const double gamma = length == 0.0 ? 0.0 : std::exp(-0.5 * std::pow(opd / length, 2));
  const std::complex<double> phase = std::polar(1.0, 2.0 * M_PI * (opd / wavelength + phase_cycles));
  return power(a) + power(b) + 2.0 * gamma * std::real(a * std::conj(b) * phase);
}

static double reference_range_difference(const double receiver_x,
                                         const double source_a_x,
                                         const double source_b_x,
                                         const double vertical_range)
{
  const double ax = receiver_x - source_a_x;
  const double bx = receiver_x - source_b_x;
  const double ra = std::hypot(ax, vertical_range);
  const double rb = std::hypot(bx, vertical_range);
  return ((source_b_x - source_a_x) * (ax + bx)) / (ra + rb);
}

static std::complex<double> reference_reflection(const double ni,
                                                 const double nt,
                                                 const double cosine,
                                                 const CoherentScalarPolarization polarization)
{
  const std::complex<double> ct = std::sqrt(std::complex<double>(
      1.0 - std::pow(ni / nt, 2) * (1.0 - cosine * cosine), 0.0));
  const std::complex<double> yi = polarization == COHERENT_SCALAR_S ?
                                      std::complex<double>(ni * cosine, 0.0) :
                                      std::complex<double>(ni / cosine, 0.0);
  const std::complex<double> yt = polarization == COHERENT_SCALAR_S ? nt * ct : nt / ct;
  return (yi - yt) / (yi + yt);
}

int main()
{
  int failures = 0;
  int checked = 0;
  double max_error = 0.0;
  auto check = [&](const double observed, const double expected, const double limit) {
    ++checked;
    const double error = std::abs(observed - expected);
    max_error = std::max(max_error, error);
    failures += !std::isfinite(observed) || error > limit;
  };

  constexpr double wavelength = 550e-9;
  constexpr double source_a_x = -50e-6;
  constexpr double source_b_x = 50e-6;
  for (int i = 0; i < 17; i++) {
    const double x = 0.85 + double(i) * 0.3 / 16.0;
    const double direct_opd = reference_range_difference(x, source_a_x, source_b_x, 1.0);
    const double mirror_opd = reference_range_difference(x, source_a_x, source_b_x, 3.0);
    const float direct = coherent_source_range_difference(make_float3(float(x), 0.0f, 1.0f),
                                                           make_float3(float(source_a_x), 0.0f, 0.0f),
                                                           make_float3(float(source_b_x), 0.0f, 0.0f));
    /* Unfolding one perfect planar mirror gives virtual point sources at z=-1. */
    const float mirror = coherent_source_range_difference(make_float3(float(x), 0.0f, 2.0f),
                                                           make_float3(float(source_a_x), 0.0f, -1.0f),
                                                           make_float3(float(source_b_x), 0.0f, -1.0f));
    const double ra = std::hypot(x - source_a_x, 3.0);
    const double rb = std::hypot(x - source_b_x, 3.0);
    const double cosine_a = 3.0 / ra;
    const double cosine_b = 3.0 / rb;
    /* Unit mirror and declared scalar detector sqrt(cos(theta)/pi). */
    const double ia = 10.0 * cosine_a / (4.0 * M_PI * M_PI * ra * ra);
    const double ib = 10.0 * cosine_b / (4.0 * M_PI * M_PI * rb * rb);
    const CoherentScalarField a{make_float2(float(-std::sqrt(ia)), 0.0f), 0.0f};
    const CoherentScalarField b{make_float2(float(-std::sqrt(ib)), 0.0f), 0.0f};
    for (const float phase : {0.0f, 0.5f}) {
      const CoherentScalarField phased_b{b.amplitude, phase};
      const float observed = coherent_field_pair_intensity(a, phased_b, mirror,
                                                            float(wavelength), 1.0f);
      const double expected = reference_pair(-std::sqrt(ia), -std::sqrt(ib),
                                             mirror_opd, wavelength, 1.0, -phase);
      check(observed, expected, 1.5e-4);
    }
    check(double(direct), direct_opd, 3e-8);
    check(double(mirror), mirror_opd, 3e-8);
    const double direct_ra = std::hypot(x - source_a_x, 1.0);
    const double direct_rb = std::hypot(x - source_b_x, 1.0);
    const double direct_ia = 1.0 / (direct_ra * direct_ra);
    const double direct_ib = 0.7 / (direct_rb * direct_rb);
    const CoherentScalarField direct_a{make_float2(float(std::sqrt(direct_ia)), 0.0f), 0.0f};
    const CoherentScalarField direct_b{make_float2(float(std::sqrt(direct_ib)), 0.0f), 0.0f};
    check(coherent_field_pair_intensity(direct_a, direct_b, direct, float(wavelength), 1.0f),
          reference_pair(std::sqrt(direct_ia), std::sqrt(direct_ib), direct_opd,
                         wavelength, 1.0), 2e-4);
  }

  for (const CoherentScalarPolarization polarization : {COHERENT_SCALAR_S, COHERENT_SCALAR_P}) {
    for (const float ni : {1.0f, 1.33f, 1.5f}) {
      for (const float nt : {1.0f, 1.33f, 1.5f}) {
        for (const float cosine : {0.3f, 0.6f, 1.0f}) {
          const CoherentScalarInterface f = coherent_field_dielectric(ni, nt, cosine, polarization);
          failures += !f.valid;
          const std::complex<double> r = reference_reflection(ni, nt, cosine, polarization);
          check(f.reflection.x, r.real(), 3e-6);
          check(f.reflection.y, r.imag(), 3e-6);
          check(coherent_field_power(f.reflection) + f.transmission * f.transmission, 1.0, 3e-6);
          if (!f.total_internal_reflection) {
            const double transmitted_cosine = std::sqrt(1.0 - std::pow(double(ni / nt), 2) *
                                                               (1.0 - double(cosine * cosine)));
            const auto reverse = coherent_field_dielectric(nt, ni, float(transmitted_cosine),
                                                           polarization);
            check(f.transmission, reverse.transmission, 3e-6);
            check(f.reflection.x, -reverse.reflection.x, 3e-6);
          }
          else {
            check(f.transmission, 0.0, 0.0);
          }
        }
      }
    }
  }

  for (const CoherentScalarPolarization polarization : {COHERENT_SCALAR_S, COHERENT_SCALAR_P}) {
    /* This float tuple gives sin^2(theta_t) exactly one in the kernel. */
    const auto critical = coherent_field_dielectric(1.25f, 1.0f, 0.6f, polarization);
    failures += !critical.valid;
    check(critical.reflection.x, polarization == COHERENT_SCALAR_S ? 1.0 : -1.0, 0.0);
    check(critical.reflection.y, 0.0, 0.0);
    check(critical.transmission, 0.0, 0.0);
    const auto grazing = coherent_field_dielectric(1.0f, 1.5f, 0.0f, polarization);
    failures += !grazing.valid;
    check(grazing.reflection.x, polarization == COHERENT_SCALAR_S ? -1.0 : 1.0, 0.0);
    check(grazing.transmission, 0.0, 0.0);
    const auto matched_grazing = coherent_field_dielectric(1.5f, 1.5f, 0.0f, polarization);
    failures += !matched_grazing.valid;
    check(matched_grazing.reflection.x, 0.0, 0.0);
    check(matched_grazing.transmission, 1.0, 0.0);
    const auto bad = coherent_field_dielectric(-1.0f, 1.5f, 0.5f, polarization);
    failures += bad.valid;
    failures += coherent_field_dielectric(0.0f, 1.5f, 0.5f, polarization).valid;
    failures += coherent_field_dielectric(1.0f, 1.5f, -0.1f, polarization).valid;
    failures += coherent_field_dielectric(1.0f, 1.5f, 1.1f, polarization).valid;
    failures += coherent_field_dielectric(1.0e30f, 1.0e-30f, 0.5f, polarization).valid;
    const auto tir_grazing = coherent_field_dielectric(1.5f, 1.0f, 0.0f, polarization);
    failures += !tir_grazing.valid || !tir_grazing.total_internal_reflection;
    check(coherent_field_power(tir_grazing.reflection), 1.0, 0.0);
  }

  /* Two distinct arms from the SAME source. Their source phase cancels. */
  const CoherentScalarField arm_a{make_float2(0.8f, 0.0f), 0.37f};
  const CoherentScalarField arm_b{make_float2(0.0f, 0.6f), 0.37f};
  const float opd = 2.25f * float(wavelength);
  check(coherent_field_pair_intensity(arm_a, arm_b, opd, float(wavelength), 1e-4f),
        reference_pair(0.8, {0.0, 0.6}, double(opd), wavelength, 1e-4), 2e-6);
  check(coherent_field_pair_intensity(arm_a, arm_b, opd, float(wavelength), 0.0f),
        1.0, 2e-6);
  check(coherent_field_pair_intensity(arm_a, arm_b, opd, float(wavelength), 1e-7f),
        reference_pair(0.8, {0.0, 0.6}, double(opd), wavelength, 1e-7), 2e-6);
  check(coherent_field_pair_intensity(arm_a, arm_a, 0.0f, float(wavelength), 1.0f),
        4.0 * 0.64, 2e-6);

  /* Interface amplitudes, including TIR's complex phase, participate in
   * interference rather than being reduced to scalar reflectance. */
  const auto glass = coherent_field_dielectric(1.0f, 1.5f, 0.6f, COHERENT_SCALAR_S);
  const CoherentScalarField reflected{glass.reflection, 0.0f};
  const CoherentScalarField transmitted{make_float2(glass.transmission, 0.0f), 0.0f};
  check(coherent_field_pair_intensity(reflected, transmitted, float(wavelength) * 0.25f,
                                      float(wavelength), 1.0f),
        reference_pair(reference_reflection(1.0, 1.5, 0.6, COHERENT_SCALAR_S),
                       std::sqrt(1.0 - power(reference_reflection(1.0, 1.5, 0.6,
                                                                  COHERENT_SCALAR_S))),
                       double(float(wavelength) * 0.25f), wavelength, 1.0),
        2e-6);
  const auto tir = coherent_field_dielectric(1.5f, 1.0f, 0.3f, COHERENT_SCALAR_P);
  const CoherentScalarField tir_arm{tir.reflection, 0.0f};
  const CoherentScalarField second_arm{arm_a.amplitude, 0.0f};
  check(coherent_field_pair_intensity(tir_arm, second_arm, 0.0f, float(wavelength), 1.0f),
        reference_pair(reference_reflection(1.5, 1.0, 0.3, COHERENT_SCALAR_P), 0.8,
                       0.0, wavelength, 1.0), 2e-6);

  /* Basis changes on a shared segment cannot alter field power or pair
   * interference. Two interface planes below are non-coplanar. */
  const float3 direction = normalize(make_float3(0.2f, 0.1f, -0.97f));
  const float3 normal_one = normalize(make_float3(0.0f, 1.0f, 0.2f));
  const float3 normal_two = normalize(make_float3(1.0f, 0.0f, 0.3f));
  const float3 s_one = normalize(cross(direction, normal_one));
  const float3 p_one = normalize(cross(s_one, direction));
  const float3 s_two = normalize(cross(direction, normal_two));
  const float3 p_two = normalize(cross(s_two, direction));
  const CoherentJonesField jones_a{{0.7f, 0.2f}, {0.1f, -0.4f}};
  const CoherentJonesField jones_b{{-0.3f, 0.5f}, {0.6f, 0.1f}};
  const auto rotated_a = coherent_field_rotate_frame(jones_a, s_one, p_one, s_two, p_two);
  const auto rotated_b = coherent_field_rotate_frame(jones_b, s_one, p_one, s_two, p_two);
  const auto restored_a = coherent_field_rotate_frame(rotated_a, s_two, p_two, s_one, p_one);
  check(coherent_field_jones_power(rotated_a), coherent_field_jones_power(jones_a), 2e-6);
  check(restored_a.s.x, jones_a.s.x, 2e-6);
  check(restored_a.s.y, jones_a.s.y, 2e-6);
  check(restored_a.p.x, jones_a.p.x, 2e-6);
  check(restored_a.p.y, jones_a.p.y, 2e-6);
  const float jones_opd = float(wavelength) * 0.37f;
  check(coherent_field_jones_pair_cross(jones_a, jones_b, 0.1f, -0.2f,
                                        jones_opd, float(wavelength), 1.0f),
        coherent_field_jones_pair_cross(rotated_a, rotated_b, 0.1f, -0.2f,
                                        jones_opd, float(wavelength), 1.0f), 2e-6);
  /* A non-coplanar chain: rotate to the second plane, apply independent
   * Fresnel factors, and compare each component to double-complex algebra. */
  const auto second_s = coherent_field_dielectric(1.0f, 1.5f, 0.6f, COHERENT_SCALAR_S);
  const auto second_p = coherent_field_dielectric(1.0f, 1.5f, 0.6f, COHERENT_SCALAR_P);
  const auto chain = coherent_field_interface_jones(rotated_a, second_s, second_p,
                                                     false, -1.0f);
  const std::complex<double> old_s(jones_a.s.x, jones_a.s.y);
  const std::complex<double> old_p(jones_a.p.x, jones_a.p.y);
  const std::complex<double> expected_s =
      (old_s * double(dot(s_one, s_two)) + old_p * double(dot(p_one, s_two))) *
      reference_reflection(1.0, 1.5, 0.6, COHERENT_SCALAR_S);
  const std::complex<double> expected_p =
      -(old_s * double(dot(s_one, p_two)) + old_p * double(dot(p_one, p_two))) *
      reference_reflection(1.0, 1.5, 0.6, COHERENT_SCALAR_P);
  check(chain.s.x, expected_s.real(), 2e-6);
  check(chain.s.y, expected_s.imag(), 2e-6);
  check(chain.p.x, expected_p.real(), 2e-6);
  check(chain.p.y, expected_p.imag(), 2e-6);
  const auto transmitted_jones = coherent_field_interface_jones(rotated_a, second_s,
                                                                  second_p, true, 1.0f);
  check(coherent_field_jones_power(chain) + coherent_field_jones_power(transmitted_jones),
        coherent_field_jones_power(rotated_a), 2e-6);

  /* Float subtraction of two completed 3 m routes destroys a quarter-wave
   * difference. A paired connector must provide the small OPD directly. */
  const double precise_opd = wavelength * 0.25;
  const float rounded_opd = float(3.0 + precise_opd) - float(3.0);
  const double naive_cycle_error = std::abs(double(rounded_opd) / wavelength - 0.25);
  failures += naive_cycle_error < 0.02;
  check(coherent_field_pair_intensity(arm_a, arm_a, float(precise_opd), float(wavelength), 1.0f),
        reference_pair(0.8, 0.8, precise_opd, wavelength, 1.0), 2e-6);

  /* A 1.8 cm arm difference spans over 32,000 wavelengths. The low part of
   * its split representation still changes phase by a fraction of a cycle.
   * Compare first against the same represented float wavelength to isolate
   * reduction arithmetic, then report the systematic UI-550 nm rounding. */
  double max_ui_wavelength_error_cycles = 0.0;
  double max_naive_large_opd_error_cycles = 0.0;
  for (const double quarter : {0.0, 0.25, 0.5, 0.75}) {
    const double exact_opd = 0.018 + quarter * wavelength;
    const float high = float(exact_opd);
    const float low = float(exact_opd - double(high));
    const float represented_wavelength = float(wavelength);
    const double represented_opd = double(high) + double(low);
    auto wrap = [](const double cycles) {
      const double value = std::fmod(cycles, 1.0);
      return value < 0.0 ? value + 1.0 : value;
    };
    auto circular_distance = [](const double a, const double b) {
      const double distance = std::abs(a - b);
      return std::min(distance, 1.0 - distance);
    };
    const double same_float_lambda = wrap(represented_opd / double(represented_wavelength));
    const double exact_ui_lambda = wrap(exact_opd / wavelength);
    const float observed = coherent_phase_cycles_split(
        make_float2(high, low), represented_wavelength);
    check(circular_distance(observed, same_float_lambda), 0.0, 3e-5);
    max_ui_wavelength_error_cycles = std::max(
        max_ui_wavelength_error_cycles,
        circular_distance(observed, exact_ui_lambda));
    max_naive_large_opd_error_cycles = std::max(
        max_naive_large_opd_error_cycles,
        circular_distance(coherent_phase_cycles(high, represented_wavelength),
                          same_float_lambda));
  }
  failures += max_naive_large_opd_error_cycles < 5e-4;
  double max_meter_split_phase_error_cycles = 0.0;
  for (const double scene_unit_scale : {0.01, 1.0, 100.0}) {
    const double exact_wavelength_scene = 550.0 * 1e-9 / scene_unit_scale;
    const float wavelength_high = float(exact_wavelength_scene);
    const float wavelength_low = float(exact_wavelength_scene - double(wavelength_high));
    for (const double quarter : {0.0, 0.25, 0.5, 0.75}) {
      const double physical_opd = 1.0 + quarter * wavelength;
      const double scene_opd = physical_opd / scene_unit_scale;
      const float high = float(scene_opd);
      const float low = float(scene_opd - double(high));
      const float observed = coherent_phase_cycles_split(
          make_float2(high, low), make_float2(wavelength_high, wavelength_low));
      double expected = std::fmod(scene_opd / exact_wavelength_scene, 1.0);
      if (expected < 0.0) expected += 1.0;
      const double distance = std::abs(double(observed) - expected);
      const double circular = std::min(distance, 1.0 - distance);
      max_meter_split_phase_error_cycles = std::max(max_meter_split_phase_error_cycles, circular);
      check(circular, 0.0, 1e-4);
    }
  }

  /* Blender scene units convert geometry and OPD to physical metres. */
  for (const double scale : {0.01, 1.0, 100.0}) {
    const float3 receiver = make_float3(float(0.05 / scale), 0.0f, float(0.1 / scale));
    const float3 first = make_float3(float(-5e-6 / scale), 0.0f, 0.0f);
    const float3 second = make_float3(float(5e-6 / scale), 0.0f, 0.0f);
    const float scene_difference = coherent_source_range_difference(receiver, first, second);
    const double physical_reference = reference_range_difference(0.05, -5e-6, 5e-6, 0.1);
    check(double(scene_difference) * scale, physical_reference, 1e-11);
    const float phase = coherent_phase_cycles(scene_difference * float(scale), float(wavelength));
    double expected_phase = std::fmod(physical_reference / wavelength, 1.0);
    if (expected_phase < 0.0) {
      expected_phase += 1.0;
    }
    const double circular_error = std::min(std::abs(double(phase) - expected_phase),
                                           1.0 - std::abs(double(phase) - expected_phase));
    check(circular_error, 0.0, 2e-5);
  }
  std::printf("coherent_field checks: checks=%d failures=%d max_absolute_error=%.9g "
              "naive_3m_quarter_wave_error_cycles=%.9g "
              "max_naive_18mm_error_cycles=%.9g "
              "max_float_wavelength_systematic_cycles=%.9g "
              "max_meter_split_phase_error_cycles=%.9g\n",
              checked, failures, max_error, naive_cycle_error,
              max_naive_large_opd_error_cycles, max_ui_wavelength_error_cycles,
              max_meter_split_phase_error_cycles);
  return failures == 0 ? 0 : 1;
}
