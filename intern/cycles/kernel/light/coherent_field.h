/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/light/coherent_util.h"

CCL_NAMESPACE_BEGIN

/* Scalar electric field in a common tangential-field gauge. The connector is
 * responsible for geometric spreading, detector response, and a stable optical
 * path DIFFERENCE. A radiometric BSDF weight is not a field amplitude. */
struct CoherentScalarField {
  float2 amplitude;
  float source_phase_cycles;
};

ccl_device_inline float2 coherent_field_mul(const float2 a, const float2 b)
{
  return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

ccl_device_inline float coherent_field_power(const float2 a)
{
  return a.x * a.x + a.y * a.y;
}

ccl_device_inline float2 coherent_field_div(const float2 a, const float2 b)
{
  const float denominator = coherent_field_power(b);
  return make_float2((a.x * b.x + a.y * b.y) / denominator,
                     (a.y * b.x - a.x * b.y) / denominator);
}

ccl_device_inline float2 coherent_field_scale(const float2 a, const float scale)
{
  return make_float2(a.x * scale, a.y * scale);
}

ccl_device_inline float2 coherent_field_add(const float2 a, const float2 b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}

/* Two transverse Jones components. Each frame is an orthonormal (s,p)
 * transverse basis for the same propagation direction during a rotation. */
struct CoherentJonesField {
  float2 s;
  float2 p;
};

ccl_device_inline CoherentJonesField coherent_field_rotate_frame(
    const CoherentJonesField field,
    const float3 old_s,
    const float3 old_p,
    const float3 new_s,
    const float3 new_p)
{
  return {coherent_field_add(coherent_field_scale(field.s, dot(old_s, new_s)),
                             coherent_field_scale(field.p, dot(old_p, new_s))),
          coherent_field_add(coherent_field_scale(field.s, dot(old_s, new_p)),
                             coherent_field_scale(field.p, dot(old_p, new_p)))};
}

ccl_device_inline float coherent_field_jones_power(const CoherentJonesField field)
{
  return coherent_field_power(field.s) + coherent_field_power(field.p);
}

/* Optical path difference must be computed from paired geometry before this
 * call: subtraction of two metre-scale float path sums loses visible cycles. */
ccl_device_inline float coherent_field_pair_cross(const CoherentScalarField a,
                                                  const CoherentScalarField b,
                                                  const float optical_path_difference_m,
                                                  const float wavelength_m,
                                                  const float coherence_length_m)
{
  const float gamma = coherent_gaussian_mutual_coherence(optical_path_difference_m,
                                                        coherence_length_m);
  if (gamma == 0.0f) {
    return 0.0f;
  }
  const float cycles = coherent_phase_cycles(optical_path_difference_m,
                                            wavelength_m,
                                            a.source_phase_cycles - b.source_phase_cycles);
  const float phase = M_2PI_F * cycles;
  const float real_product = a.amplitude.x * b.amplitude.x + a.amplitude.y * b.amplitude.y;
  const float imaginary_product = a.amplitude.y * b.amplitude.x -
                                  a.amplitude.x * b.amplitude.y;
  return 2.0f * gamma * (real_product * cosf(phase) - imaginary_product * sinf(phase));
}

ccl_device_inline float coherent_field_jones_pair_cross(
    const CoherentJonesField a,
    const CoherentJonesField b,
    const float phase_a_cycles,
    const float phase_b_cycles,
    const float optical_path_difference_m,
    const float wavelength_m,
    const float coherence_length_m)
{
  return coherent_field_pair_cross({a.s, phase_a_cycles}, {b.s, phase_b_cycles},
                                   optical_path_difference_m, wavelength_m,
                                   coherence_length_m) +
         coherent_field_pair_cross({a.p, phase_a_cycles}, {b.p, phase_b_cycles},
                                   optical_path_difference_m, wavelength_m,
                                   coherence_length_m);
}

/* An already completed set of paths at ONE detector sample. Both different
 * sources in a coherence group and distinct arms from one source are paired. */
ccl_device_inline float coherent_field_pair_intensity(const CoherentScalarField a,
                                                      const CoherentScalarField b,
                                                      const float optical_path_difference_m,
                                                      const float wavelength_m,
                                                      const float coherence_length_m)
{
  return coherent_field_power(a.amplitude) + coherent_field_power(b.amplitude) +
         coherent_field_pair_cross(a, b, optical_path_difference_m, wavelength_m,
                                   coherence_length_m);
}

enum CoherentScalarPolarization { COHERENT_SCALAR_S = 0, COHERENT_SCALAR_P = 1 };

struct CoherentScalarInterface {
  float2 reflection;
  float transmission; /* Flux-normalized propagating amplitude. */
  bool total_internal_reflection;
  bool valid;
};

/* Apply s/p interface factors after rotating the incoming field into the
 * local incidence basis. The p factor is defined for tangential E, while
 * p = cross(s,direction) changes tangential sign on reflection. The connector
 * supplies that sign and transports the returned outgoing basis to the next
 * interface. No s/p cross-polarization is modeled here. */
ccl_device_inline CoherentJonesField coherent_field_interface_jones(
    const CoherentJonesField incident,
    const CoherentScalarInterface s_factor,
    const CoherentScalarInterface p_factor,
    const bool transmission,
    const float p_tangential_sign)
{
  const float2 s = transmission ? coherent_field_scale(incident.s, s_factor.transmission) :
                                  coherent_field_mul(incident.s, s_factor.reflection);
  const float2 p = transmission ? coherent_field_scale(incident.p,
                                                       p_factor.transmission * p_tangential_sign) :
                                  coherent_field_scale(
                                      coherent_field_mul(incident.p, p_factor.reflection),
                                      p_tangential_sign);
  return {s, p};
}

/* Lossless, nonmagnetic, planar dielectric. s/p admittances are respectively
 * n cos(theta) and n/cos(theta), using tangential E on both sides. This gauge
 * makes r_s and r_p share the same normal-incidence sign. Transmission uses
 * sqrt(power), so |r|^2 + t^2 = 1. In TIR t=0 and r retains its phase. */
ccl_device_inline CoherentScalarInterface coherent_field_dielectric(
    const float incident_ior,
    const float transmitted_ior,
    const float incident_cosine,
    const CoherentScalarPolarization polarization)
{
  if (!(incident_ior > 0.0f) || !(transmitted_ior > 0.0f) ||
      !(incident_cosine >= 0.0f && incident_cosine <= 1.0f) ||
      !isfinite_safe(incident_ior) || !isfinite_safe(transmitted_ior) ||
      !isfinite_safe(incident_cosine))
  {
    return {zero_float2(), 0.0f, false, false};
  }
  if (incident_ior == transmitted_ior) {
    return {zero_float2(), 1.0f, false, true};
  }
  if (incident_cosine == 0.0f) {
    return {make_float2(polarization == COHERENT_SCALAR_S ? -1.0f : 1.0f, 0.0f),
            0.0f,
            incident_ior > transmitted_ior,
            true};
  }
  const float sin2_transmitted = (incident_ior / transmitted_ior) *
                                 (incident_ior / transmitted_ior) *
                                 (1.0f - incident_cosine * incident_cosine);
  const float incident_admittance = polarization == COHERENT_SCALAR_S ?
                                       incident_ior * incident_cosine :
                                       incident_ior / incident_cosine;
  if (!isfinite_safe(sin2_transmitted) || !isfinite_safe(incident_admittance)) {
    return {zero_float2(), 0.0f, false, false};
  }
  if (sin2_transmitted == 1.0f) {
    return {make_float2(polarization == COHERENT_SCALAR_S ? 1.0f : -1.0f, 0.0f),
            0.0f,
            false,
            true};
  }
  if (sin2_transmitted > 1.0f) {
    const float imaginary_cosine = sqrtf(sin2_transmitted - 1.0f);
    const float imaginary_admittance = polarization == COHERENT_SCALAR_S ?
                                           transmitted_ior * imaginary_cosine :
                                           -transmitted_ior / imaginary_cosine;
    if (!isfinite_safe(imaginary_admittance)) {
      return {zero_float2(), 0.0f, false, false};
    }
    const float2 yi = make_float2(incident_admittance, 0.0f);
    const float2 yt = make_float2(0.0f, imaginary_admittance);
    return {coherent_field_div(yi - yt, yi + yt), 0.0f, true, true};
  }
  const float transmitted_cosine = sqrtf(1.0f - sin2_transmitted);
  const float transmitted_admittance = polarization == COHERENT_SCALAR_S ?
                                           transmitted_ior * transmitted_cosine :
                                           transmitted_ior / transmitted_cosine;
  const float denominator = incident_admittance + transmitted_admittance;
  if (!isfinite_safe(transmitted_admittance) || !isfinite_safe(denominator) ||
      !(denominator > 0.0f))
  {
    return {zero_float2(), 0.0f, false, false};
  }
  return {make_float2((incident_admittance - transmitted_admittance) / denominator, 0.0f),
          2.0f * sqrtf(incident_admittance * transmitted_admittance) / denominator,
          false,
          true};
}

CCL_NAMESPACE_END
