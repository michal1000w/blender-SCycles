/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/light/coherent_field.h"
#include "kernel/light/coherent_polarizer.h"
#include "kernel/light/coherent_geometry.h"

CCL_NAMESPACE_BEGIN

/* This source ensemble has three mutually independent, equal world-axis
 * dipoles. Each is projected onto the emitted ray's transverse plane and
 * carries 1/sqrt(2) of its projected amplitude. The three powers then sum to one for
 * every emission direction. Corresponding dipoles of coherent sources share
 * the declared source phase; unlike dipoles have zero mutual coherence. */
#define COHERENT_PATH_WORLD_MODES 3

struct CoherentPathDetectorFrame {
  float3 u;
  float3 v;
  float3 normal;
};

struct CoherentCompletedPathField {
  /* Jones amplitudes after all interfaces, in the last ray's transverse basis. */
  CoherentJonesField world_mode[COHERENT_PATH_WORLD_MODES];
  /* Components of that transverse basis in the shared detector (u,v,n) frame. */
  float3 final_s_detector;
  float3 final_p_detector;
  float3 radiance_amplitude_rgb;
  float3 physical_diagonal_rgb;
  float3 native_scalar_diagonal_rgb;
  float2 optical_length_split;
  float source_phase_cycles;
};

ccl_device_inline float3 coherent_path_frame_components(const float3 world_vector,
                                                         const CoherentPathDetectorFrame frame)
{
  return make_float3(dot(world_vector, frame.u),
                     dot(world_vector, frame.v),
                     dot(world_vector, frame.normal));
}

ccl_device_inline bool coherent_path_frame_valid(const CoherentPathDetectorFrame frame)
{
  return isfinite_safe(frame.u) && isfinite_safe(frame.v) && isfinite_safe(frame.normal) &&
         fabsf(len_squared(frame.u) - 1.0f) < 1.0e-4f &&
         fabsf(len_squared(frame.v) - 1.0f) < 1.0e-4f &&
         fabsf(len_squared(frame.normal) - 1.0f) < 1.0e-4f &&
         fabsf(dot(frame.u, frame.v)) < 1.0e-4f &&
         fabsf(dot(frame.u, frame.normal)) < 1.0e-4f &&
         fabsf(dot(frame.v, frame.normal)) < 1.0e-4f;
}

ccl_device_inline bool coherent_path_incidence_basis(const float3 direction,
                                                     const float3 normal,
                                                     const float3 fallback_tangent,
                                                     ccl_private float3 *s,
                                                     ccl_private float3 *p)
{
  float3 transverse = cross(normal, direction);
  if (len_squared(transverse) < 1.0e-10f) {
    transverse = fallback_tangent - direction * dot(fallback_tangent, direction);
  }
  if (!(len_squared(transverse) > 1.0e-12f)) return false;
  *s = normalize(transverse);
  *p = normalize(cross(*s, direction));
  return isfinite_safe(*s) && isfinite_safe(*p);
}

ccl_device_inline CoherentJonesField coherent_path_source_mode(const int world_axis,
                                                                const float3 s,
                                                                const float3 p)
{
  const float3 axis = world_axis == 0 ? make_float3(1.0f, 0.0f, 0.0f) :
                      world_axis == 1 ? make_float3(0.0f, 1.0f, 0.0f) :
                                        make_float3(0.0f, 0.0f, 1.0f);
  const float normalization = 0.7071067811865475244f;
  return {make_float2(normalization * dot(axis, s), 0.0f),
          make_float2(normalization * dot(axis, p), 0.0f)};
}

/* Complete path only: geometry, event sides, medium sequence, material, and
 * visibility must already have been validated by the connector. ideal_mirror
 * is true solely for declared unit mirror reflection; false selects dielectric
 * Fresnel for reflection or transmission. The detector is a polarization-
 * neutral white/colored Lambertian: dOmega/dA includes its cosine, 1/pi is
 * its radiance conversion, and the full 3D field is compared in the common
 * orthonormal detector frame without discarding its normal component. */
ccl_device_inline bool coherent_path_field_transport(
    const float3 source,
    const float3 detector,
    const CoherentPathDetectorFrame detector_frame,
    const ccl_private CoherentGeometryPath *path,
    const ccl_private CoherentGeometryInterface patches[COHERENT_GEOMETRY_MAX_INTERFACES],
    const ccl_private bool ideal_mirror[COHERENT_GEOMETRY_MAX_INTERFACES],
    const float3 source_power_rgb,
    const float3 detector_albedo_rgb,
    const float source_phase_cycles,
    ccl_private CoherentCompletedPathField *output,
    const ccl_private float3 *polarizer_axes = nullptr,
    const ccl_private bool *polarizer_enabled = nullptr)
{
  if (path == nullptr || output == nullptr || path->count < 0 ||
      path->count > COHERENT_GEOMETRY_MAX_INTERFACES ||
      !(path->spreading > 0.0f) || !isfinite_safe(path->spreading) ||
      !coherent_path_frame_valid(detector_frame) ||
      !isfinite_safe(source_power_rgb) || !isfinite_safe(detector_albedo_rgb) ||
      !isfinite_safe(source_phase_cycles) ||
      source_power_rgb.x < 0.0f || source_power_rgb.y < 0.0f ||
      source_power_rgb.z < 0.0f || detector_albedo_rgb.x < 0.0f ||
      detector_albedo_rgb.y < 0.0f || detector_albedo_rgb.z < 0.0f)
  {
    return false;
  }
  const float3 first_target = path->count == 0 ? detector : path->point[0];
  const float3 first_segment = first_target - source;
  if (!(len_squared(first_segment) > 1.0e-14f)) return false;
  float3 current_direction = normalize(first_segment);
  float3 current_s, current_p;
  if (path->count > 0) {
    const float3 normal = normalize(cross(patches[0].tangent_u, patches[0].tangent_v));
    if (!coherent_path_incidence_basis(current_direction,
                                       normal,
                                       patches[0].tangent_u,
                                       &current_s,
                                       &current_p))
      return false;
  }
  else {
    make_orthonormals(current_direction, &current_s, &current_p);
  }
  CoherentJonesField modes[COHERENT_PATH_WORLD_MODES];
  for (int mode = 0; mode < COHERENT_PATH_WORLD_MODES; mode++) {
    modes[mode] = coherent_path_source_mode(mode, current_s, current_p);
  }
  float native_fraction = 1.0f;
  for (int index = 0; index < path->count; index++) {
    const ccl_private CoherentGeometryInterface &patch = patches[index];
    const float3 normal = normalize(cross(patch.tangent_u, patch.tangent_v));
    const float3 next_point = index + 1 == path->count ? detector : path->point[index + 1];
    const float3 next_segment = next_point - path->point[index];
    if (!(len_squared(next_segment) > 1.0e-14f)) return false;
    const float3 outgoing_direction = normalize(next_segment);
    float3 incidence_s, incidence_p;
    if (!coherent_path_incidence_basis(current_direction,
                                       normal,
                                       patch.tangent_u,
                                       &incidence_s,
                                       &incidence_p))
      return false;
    const float3 outgoing_p = normalize(cross(incidence_s, outgoing_direction));
    if (!isfinite_safe(outgoing_p)) return false;
    const bool transmission = patch.event == COHERENT_GEOMETRY_TRANSMIT;
    if (transmission && ideal_mirror[index]) return false;
    if (patch.event != COHERENT_GEOMETRY_REFLECT && !transmission) return false;
    const float incident_ior = patch.ior_before;
    const float opposite_ior = transmission ? patch.ior_after : patch.ior_opposite;
    const float incident_cosine = fabsf(dot(current_direction, normal));
    CoherentScalarInterface factor_s, factor_p;
    if (ideal_mirror[index]) {
      factor_s = {make_float2(1.0f, 0.0f), 0.0f, false, true};
      factor_p = factor_s;
    }
    else {
      factor_s = coherent_field_dielectric(
          incident_ior, opposite_ior, incident_cosine, COHERENT_SCALAR_S);
      factor_p = coherent_field_dielectric(
          incident_ior, opposite_ior, incident_cosine, COHERENT_SCALAR_P);
      if (!factor_s.valid || !factor_p.valid ||
          (transmission && (factor_s.total_internal_reflection ||
                            factor_p.total_internal_reflection)))
        return false;
    }
    const float s_power = transmission ? factor_s.transmission * factor_s.transmission :
                                         coherent_field_power(factor_s.reflection);
    const float p_power = transmission ? factor_p.transmission * factor_p.transmission :
                                         coherent_field_power(factor_p.reflection);
    native_fraction *= 0.5f * (s_power + p_power);
    const float p_tangential_sign = transmission ? 1.0f : -1.0f;
    for (int mode = 0; mode < COHERENT_PATH_WORLD_MODES; mode++) {
      const CoherentJonesField incoming = coherent_field_rotate_frame(
          modes[mode], current_s, current_p, incidence_s, incidence_p);
      if (transmission && polarizer_enabled && polarizer_enabled[index]) {
        float2 incident_axis, outgoing_axis;
        if (!polarizer_axes ||
            !coherent_polarizer_axis(polarizer_axes[index], incidence_s, incidence_p,
                                     &incident_axis) ||
            !coherent_polarizer_axis(polarizer_axes[index], incidence_s, outgoing_p,
                                     &outgoing_axis)) return false;
        modes[mode] = coherent_polarizer_transmit(
            incoming, incident_axis, outgoing_axis, factor_s, factor_p);
      }
      else {
        modes[mode] = coherent_field_interface_jones(
            incoming, factor_s, factor_p, transmission, p_tangential_sign);
      }
    }
    current_s = incidence_s;
    current_p = outgoing_p;
    current_direction = outgoing_direction;
  }
  if (!(dot(detector_frame.normal, -current_direction) > 0.0f)) return false;
  float physical_fraction = 0.0f;
  for (int mode = 0; mode < COHERENT_PATH_WORLD_MODES; mode++) {
    physical_fraction += coherent_field_jones_power(modes[mode]);
    output->world_mode[mode] = modes[mode];
  }
  const float3 base = source_power_rgb * detector_albedo_rgb *
                      (path->spreading * (0.25f * M_1_PI_F * M_1_PI_F));
  output->radiance_amplitude_rgb = make_float3(sqrtf(base.x), sqrtf(base.y), sqrtf(base.z));
  output->physical_diagonal_rgb = base * physical_fraction;
  output->native_scalar_diagonal_rgb = base * native_fraction;
  output->final_s_detector = coherent_path_frame_components(current_s, detector_frame);
  output->final_p_detector = coherent_path_frame_components(current_p, detector_frame);
  output->optical_length_split = path->optical_length_split;
  output->source_phase_cycles = source_phase_cycles;
  return isfinite_safe(output->physical_diagonal_rgb) &&
         isfinite_safe(output->native_scalar_diagonal_rgb);
}

ccl_device_inline void coherent_path_overlap_add(ccl_private float2 *overlap,
                                                 const float2 a,
                                                 const float2 b,
                                                 const float spatial_dot)
{
  const float2 product = coherent_field_mul(a, make_float2(b.x, -b.y));
  *overlap = coherent_field_add(*overlap, coherent_field_scale(product, spatial_dot));
}

/* Caller only pairs paths from one coherence group and one detector frame.
 * This returns the signed cross term per RGB channel; add the physical
 * diagonals separately. No arbitrary RGB BSDF or material phase is invented. */
ccl_device_inline float3 coherent_path_field_pair_cross(
    const ccl_private CoherentCompletedPathField *a,
    const ccl_private CoherentCompletedPathField *b,
    const float2 optical_path_difference_m,
    const float2 wavelength_m,
    const float coherence_length_m)
{
  if (coherence_length_m == 0.0f) return zero_float3();
  const float gamma = coherent_gaussian_mutual_coherence(optical_path_difference_m.x +
                                                            optical_path_difference_m.y,
                                                         coherence_length_m);
  if (gamma == 0.0f) return zero_float3();
  float2 overlap = zero_float2();
  const float ss = dot(a->final_s_detector, b->final_s_detector);
  const float sp = dot(a->final_s_detector, b->final_p_detector);
  const float ps = dot(a->final_p_detector, b->final_s_detector);
  const float pp = dot(a->final_p_detector, b->final_p_detector);
  for (int mode = 0; mode < COHERENT_PATH_WORLD_MODES; mode++) {
    const CoherentJonesField left = a->world_mode[mode];
    const CoherentJonesField right = b->world_mode[mode];
    coherent_path_overlap_add(&overlap, left.s, right.s, ss);
    coherent_path_overlap_add(&overlap, left.s, right.p, sp);
    coherent_path_overlap_add(&overlap, left.p, right.s, ps);
    coherent_path_overlap_add(&overlap, left.p, right.p, pp);
  }
  const float phase_cycles = coherent_phase_cycles_split(
      optical_path_difference_m,
      wavelength_m,
      a->source_phase_cycles - b->source_phase_cycles);
  const float phase = M_2PI_F * phase_cycles;
  const float cross = 2.0f * gamma *
                      (overlap.x * cosf(phase) - overlap.y * sinf(phase));
  return cross * a->radiance_amplitude_rgb * b->radiance_amplitude_rgb;
}

ccl_device_inline float3 coherent_path_field_pair_cross(
    const ccl_private CoherentCompletedPathField *a,
    const ccl_private CoherentCompletedPathField *b,
    const float optical_path_difference_m,
    const float wavelength_m,
    const float coherence_length_m)
{
  return coherent_path_field_pair_cross(a,
                                        b,
                                        make_float2(optical_path_difference_m, 0.0f),
                                        make_float2(wavelength_m, 0.0f),
                                        coherence_length_m);
}

ccl_device_inline float3 coherent_path_field_pair_cross(
    const ccl_private CoherentCompletedPathField *a,
    const ccl_private CoherentCompletedPathField *b,
    const float2 optical_path_difference_m,
    const float wavelength_m,
    const float coherence_length_m)
{
  return coherent_path_field_pair_cross(a,
                                        b,
                                        optical_path_difference_m,
                                        make_float2(wavelength_m, 0.0f),
                                        coherence_length_m);
}

CCL_NAMESPACE_END
