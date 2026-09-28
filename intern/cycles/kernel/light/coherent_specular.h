/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/bvh/bvh.h"
#include "kernel/bvh/util.h"
#include "kernel/light/coherent_field.h"
#include "kernel/light/coherent_detector.h"
#include "kernel/light/coherent_feasibility.h"
#include "kernel/light/coherent_geometry.h"
#include "kernel/light/coherent_unfold_internal_geometry.h"
#include "kernel/light/coherent_sphere_transmit_geometry.h"
#include "kernel/light/coherent_history.h"
#include "kernel/light/coherent_path_field.h"
#include "kernel/light/coherent_passes.h"
#include "kernel/light/coherent_patch_membership.h"
#include "kernel/light/sample.h"
#include "kernel/light/coherent_facet_integrator.h"

CCL_NAMESPACE_BEGIN

/* Bounded deterministic source-to-detector connector. Scalar mode preserves
 * the ideal-mirror approximation; explicit vector mode uses transported
 * Jones fields for planar mirror and dielectric reflection/transmission. */
#define COHERENT_SPECULAR_MAX_PATHS 64
ccl_device_inline bool coherent_specular_shared_triangle_edge(KernelGlobals kg,
                                                               const int first_prim,
                                                               const int second_prim)
{
  if (first_prim == PRIM_NONE || second_prim == PRIM_NONE) {
    return false;
  }
  /* Intersection and ShaderData primitive indices address the same global
   * tri_vindex array; its three indices are local to their already-matched
   * mesh object. */
  const uint3 first = kernel_data_fetch(tri_vindex, first_prim);
  const uint3 second = kernel_data_fetch(tri_vindex, second_prim);
  const bool x_shared = second.x == first.x || second.x == first.y || second.x == first.z;
  const bool y_shared = second.y == first.x || second.y == first.y || second.y == first.z;
  const bool z_shared = second.z == first.x || second.z == first.y || second.z == first.z;
  return int(x_shared) + int(y_shared) + int(z_shared) == 2;
}

ccl_device_inline bool coherent_specular_visible(
    KernelGlobals kg,
    const ccl_private ShaderData *sd,
    const float3 source,
    const ccl_private CoherentGeometryInterface
        patches[COHERENT_GEOMETRY_MAX_INTERFACES],
    const ccl_private int patch_indices[COHERENT_GEOMETRY_MAX_INTERFACES],
    const ccl_private CoherentGeometryPath *path)
{
  float3 begin = source;
  int previous_object = OBJECT_NONE;
  int previous_prim = PRIM_NONE;
  int previous_patch = -1;
  for (int segment = 0; segment <= path->count; segment++) {
    const bool final_segment = segment == path->count;
    const float3 end = final_segment ? sd->P : path->point[segment];
    const float3 delta = end - begin;
    const float distance = len(delta);
    if (!(distance > 1.0e-7f)) {
      return false;
    }
    Ray ray ccl_optional_struct_init;
    ray.P = begin;
    ray.D = delta / distance;
    ray.tmin = 0.0f;
    ray.tmax = distance * (1.0f + 1.0e-5f);
    ray.time = sd->time;
    ray.self.object = previous_object;
    ray.self.prim = previous_prim;
    ray.self.light_object = OBJECT_NONE;
    ray.self.light_prim = PRIM_NONE;
#ifdef __RAY_DIFFERENTIALS__
    ray.dP = differential_zero_compact();
    ray.dD = differential_zero_compact();
#endif
    Intersection hit ccl_optional_struct_init;
    const bool intersected = scene_intersect(kg, &ray, PATH_RAY_VISIBILITY_SHADOW_OPAQUE, &hit);
    if (!intersected) {
      return false;
    }
    if (final_segment) {
      if (hit.object != sd->object) return false;
    }
    else if (!coherent_patch_transition_valid(
        kg, previous_patch, patch_indices[segment], hit.object, hit.prim, hit.type))
    {
      return false;
    }
    /* A ray aimed at a detector triangle edge can resolve its neighboring
     * triangle in the BVH. Accept only an actual shared-edge neighbor, not
     * an unrelated triangle of the same object near the endpoint. */
    if (final_segment && hit.prim != sd->prim &&
        !coherent_specular_shared_triangle_edge(kg, hit.prim, sd->prim))
    {
      return false;
    }
    const float tolerance = max(1.0e-5f, distance * 1.0e-4f);
    if (hit.t < distance - tolerance) {
      return false;
    }
    if (hit.t > distance + tolerance) {
      return false;
    }
    if (final_segment) return true;
    /* Geometry already bounds the stationary point to this patch. The BVH
     * identifies its actual object and catches all nearer blockers. */
    previous_object = hit.object;
    previous_prim = hit.prim;
    previous_patch = patch_indices[segment];
    const float3 next_end = (segment + 1 == path->count) ? sd->P : path->point[segment + 1];
    const float3 next_direction = next_end - end;
    float3 patch_normal = normalize(
        cross(patches[segment].tangent_u, patches[segment].tangent_v));
    if (dot(patch_normal, next_direction) < 0.0f) {
      patch_normal = -patch_normal;
    }
    /* The exact stationary interface point can re-hit an adjacent triangle of
     * its own planar patch. Offset only the visibility ray toward the outgoing
     * medium, as for ordinary Cycles surface continuations. The path points
     * and optical length remain exact for field phase and spreading. */
    begin = ray_offset(end, patch_normal);
  }
  return false;
}

/* One exterior reflection or the complete isolated two-transmission sphere
 * inventory. Planar sequences keep their existing solver and side checks. */
ccl_device_inline bool coherent_specular_connect(
    KernelGlobals kg, const ccl_global KernelCoherentCandidate *candidate,
    const float3 source, const float3 receiver, const float3 normal,
    ccl_private CoherentGeometryInterface *patches,
    ccl_private CoherentGeometryPath *path,
    ccl_private float *geometric_phase_cycles)
{
  *geometric_phase_cycles = 0.0f;
  int sphere_first = -1, sphere_count = 0;
  for (int i = 0; i < candidate->count; i++) {
    if (kernel_data_fetch(coherent_patches, candidate->patch[i]).shape == 1) {
      if (sphere_first < 0) sphere_first = i;
      sphere_count++;
    }
  }
  if (sphere_first >= 0) {
    const ccl_global KernelCoherentPatch *sphere =
        &kernel_data_fetch(coherent_patches, candidate->patch[sphere_first]);
    const bool transmit = candidate->event[sphere_first] == COHERENT_GEOMETRY_TRANSMIT;
    if ((!transmit && sphere_count != 1) || (transmit && (sphere_count < 2 || sphere_count > 4))) return false;
    if (transmit) {
      if (sphere->mode != 2 || sphere_first + sphere_count > candidate->count) return false;
      for (int i=0;i<sphere_count;i++) {
        if (candidate->patch[sphere_first+i] != candidate->patch[sphere_first] ||
            candidate->event[sphere_first+i] !=
                (i==0 || i==sphere_count-1 ? COHERENT_GEOMETRY_TRANSMIT : COHERENT_GEOMETRY_REFLECT))
          return false;
      }
    }
    for (int i = 0; i < candidate->count; i++) {
      if (i >= sphere_first && i < sphere_first + sphere_count) continue;
      if (candidate->event[i] != COHERENT_GEOMETRY_REFLECT ||
          candidate->ior_before[i] != 1.0f || candidate->ior_after[i] != 1.0f) return false;
    }
    const CoherentSphereTTStatus status = transmit && sphere_count > 2 ?
        coherent_unfold_internal_connect(source, receiver, normal, patches, candidate->count,
            sphere_first, sphere_count-2, candidate->sphere_branch, sphere->center,
            sphere->radius, sphere->inside_ior, path, geometric_phase_cycles) :
        coherent_unfold_sphere_connect(source, receiver, normal, patches, candidate->count,
            sphere_first, transmit, candidate->sphere_branch, sphere->center,
            sphere->radius, sphere->inside_ior, path, geometric_phase_cycles);
    if (status != COHERENT_SPHERE_TT_OK && status != COHERENT_SPHERE_TT_EMPTY) {
#ifdef __KERNEL_METAL__
      atomic_fetch_and_or_uint32(&kernel_integrator_state.queue_counter->coherent_error, 1u);
#endif
    }
    return status == COHERENT_SPHERE_TT_OK;
  }
  return coherent_geometry_candidate_side_feasible(source, receiver, patches, candidate->count) &&
         coherent_geometry_connect(source, receiver, normal, patches, candidate->count, path);
}

/* The vector mode transports three globally shared, mutually independent
 * dipole launch modes through every interface. The scalar mode below remains
 * the declared ideal-mirror approximation used by the existing mirror oracle. */
#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device_noinline
#endif
Spectrum coherent_specular_vector_intensity(KernelGlobals kg,
                                            IntegratorState state,
                                            ccl_private ShaderData *sd,
                                            ccl_private Spectrum *primary_direct)
{
  const int candidate_count = kernel_data.integrator.coherent_candidate_count;
  if (candidate_count < 1 || candidate_count > COHERENT_SPECULAR_MAX_PATHS) {
    return zero_spectrum();
  }
  float3 detector_u, detector_v;
  const float3 detector_normal = safe_normalize(sd->N);
  make_orthonormals(detector_normal, &detector_u, &detector_v);
  const CoherentPathDetectorFrame detector_frame = {detector_u, detector_v, detector_normal};
  const float3 detector_albedo = spectrum_to_rgb(sd->closure[0].weight);
  CoherentCompletedPathField fields[COHERENT_SPECULAR_MAX_PATHS];
  int group[COHERENT_SPECULAR_MAX_PATHS];
  bool direct_route[COHERENT_SPECULAR_MAX_PATHS];
  float wavelength[COHERENT_SPECULAR_MAX_PATHS];
  float wavelength_low[COHERENT_SPECULAR_MAX_PATHS];
  float coherence_length[COHERENT_SPECULAR_MAX_PATHS];
  int paths = 0;
  for (int candidate_index = 0; candidate_index < candidate_count; candidate_index++) {
    const ccl_global KernelCoherentCandidate *candidate =
        &kernel_data_fetch(coherent_candidates, candidate_index);
    const ccl_global KernelLight *light = &kernel_data_fetch(lights, candidate->light);
    if (light->coherence_group <= 0 || light->coherence_length <= 0.0f ||
        !coherent_history_candidate_within_budget(
            candidate,
            INTEGRATOR_STATE(state, path, bounce),
            INTEGRATOR_STATE(state, path, glossy_bounce),
            INTEGRATOR_STATE(state, path, transmission_bounce),
            kernel_data.integrator.max_bounce,
            kernel_data.integrator.max_glossy_bounce,
            kernel_data.integrator.max_transmission_bounce,
            bool(kernel_data.integrator.use_bidirectional_path_tracing),
            kernel_data.integrator.bdpt_max_bounces,
            light->max_bounces))
    {
      continue;
    }
    CoherentGeometryInterface patches[COHERENT_GEOMETRY_MAX_INTERFACES];
    float3 polarizer_axes[COHERENT_GEOMETRY_MAX_INTERFACES];
    bool polarizer_enabled[COHERENT_GEOMETRY_MAX_INTERFACES];
    bool ideal_mirror[COHERENT_GEOMETRY_MAX_INTERFACES];
    int patch_indices[COHERENT_GEOMETRY_MAX_INTERFACES];
    bool supported = true;
    for (int index = 0; index < candidate->count; index++) {
      const ccl_global KernelCoherentPatch *patch =
          &kernel_data_fetch(coherent_patches, candidate->patch[index]);
      if ((patch->mode != 1 && patch->mode != 2) ||
          (patch->mode == 1 && candidate->event[index] != COHERENT_GEOMETRY_REFLECT))
      {
        supported = false;
        break;
      }
      patches[index] = {patch->center,
                        patch->tangent_u,
                        patch->tangent_v,
                        patch->half_u,
                        patch->half_v,
                        candidate->ior_before[index],
                        candidate->ior_after[index],
                        candidate->ior_opposite[index],
                        candidate->event[index],
                        candidate->expected_incident_side[index]};
      polarizer_enabled[index] = patch->polarizer != 0;
      polarizer_axes[index] = patch->polarizer_axis;
      ideal_mirror[index] = patch->mode == 1;
      patch_indices[index] = candidate->patch[index];
    }
    if (!supported)
    {
      continue;
    }
    CoherentGeometryPath path;
    float geometric_phase_cycles;
    if (!coherent_specular_connect(kg, candidate, light->co, sd->P, sd->N, patches, &path,
                                   &geometric_phase_cycles)) {
      continue;
    }
    if (!coherent_specular_visible(kg, sd, light->co, patches, patch_indices, &path)) {
      continue;
    }
    Spectrum emission;
    if (!light_sample_shader_eval_nee_constant(kg, light->shader_id, candidate->light,
                                               true, emission))
    {
      continue;
    }
    const float3 source_power = spectrum_to_rgb(emission) *
                                (light->spot.eval_fac * M_4PI_F);
    CoherentCompletedPathField field;
    if (!coherent_path_field_transport(light->co,
                                       sd->P,
                                       detector_frame,
                                       &path,
                                       patches,
                                       ideal_mirror,
                                       source_power,
                                       detector_albedo,
                                       light->coherence_phase * M_1_2PI_F + geometric_phase_cycles,
                                       &field, polarizer_axes, polarizer_enabled))
    {
      continue;
    }
    fields[paths] = field;
    group[paths] = light->coherence_group;
    wavelength[paths] = light->coherence_wavelength;
    wavelength_low[paths] = light->coherence_wavelength_low;
    coherence_length[paths] = light->coherence_length;
    direct_route[paths] = candidate->count == 0;
    paths++;
  }
  float3 intensity = zero_float3();
  float3 direct_intensity = zero_float3();
  for (int index = 0; index < paths; index++) {
    intensity += fields[index].physical_diagonal_rgb;
    if (primary_direct && direct_route[index]) {
      direct_intensity += fields[index].physical_diagonal_rgb;
    }
  }
  for (int a = 0; a < paths; a++) {
    for (int b = a + 1; b < paths; b++) {
      if (group[a] != group[b]) {
        continue;
      }
      const float2 opd_split = coherent_geometry_add_split(
          fields[a].optical_length_split,
          make_float2(-fields[b].optical_length_split.x, -fields[b].optical_length_split.y));
      const float3 cross = coherent_path_field_pair_cross(&fields[a],
                                                  &fields[b],
                                                  opd_split,
                                                  make_float2(wavelength[a], wavelength_low[a]),
                                                  coherence_length[a]);
      intensity += cross;
      if (primary_direct) {
        direct_intensity += coherent_pass_direct_pair_share(cross, direct_route[a], direct_route[b]);
      }
    }
  }
  if (primary_direct) *primary_direct = rgb_to_spectrum(direct_intensity);
  return rgb_to_spectrum(intensity);
}


#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device_noinline
#endif
Spectrum coherent_specular_complete_intensity(KernelGlobals kg,
                                              IntegratorState state,
                                              ccl_private ShaderData *sd,
                                              ccl_private Spectrum *primary_direct = nullptr)
{
  if (primary_direct) *primary_direct = zero_spectrum();
  if (!kernel_data.integrator.coherent_specular_enabled ||
      !coherent_detector_eligible(sd) || sd->num_closure == 0)
  {
    return zero_spectrum();
  }
  if (kernel_data.integrator.coherent_transport_mode == 1) {
    return coherent_facet_stream_intensity(kg, state, sd, primary_direct);
  }
  if (kernel_data.integrator.coherent_polarization_mode == 1) {
    return coherent_specular_vector_intensity(kg, state, sd, primary_direct);
  }
  const int candidate_count = kernel_data.integrator.coherent_candidate_count;
  if (candidate_count < 1 || candidate_count > COHERENT_SPECULAR_MAX_PATHS) {
    return zero_spectrum();
  }
  Spectrum base_power[COHERENT_SPECULAR_MAX_PATHS];
  float2 optical_path[COHERENT_SPECULAR_MAX_PATHS];
  float phase[COHERENT_SPECULAR_MAX_PATHS];
  int group[COHERENT_SPECULAR_MAX_PATHS];
  bool direct_route[COHERENT_SPECULAR_MAX_PATHS];
  float wavelength[COHERENT_SPECULAR_MAX_PATHS];
  float wavelength_low[COHERENT_SPECULAR_MAX_PATHS];
  float coherence_length[COHERENT_SPECULAR_MAX_PATHS];
  int paths = 0;
  const Spectrum detector_weight = sd->closure[0].weight;
  for (int candidate_index = 0; candidate_index < candidate_count; candidate_index++) {
    const ccl_global KernelCoherentCandidate *candidate =
        &kernel_data_fetch(coherent_candidates, candidate_index);
    const ccl_global KernelLight *light = &kernel_data_fetch(lights, candidate->light);
    if (light->coherence_group <= 0 || light->coherence_length <= 0.0f ||
        !coherent_history_candidate_within_budget(
            candidate,
            INTEGRATOR_STATE(state, path, bounce),
            INTEGRATOR_STATE(state, path, glossy_bounce),
            INTEGRATOR_STATE(state, path, transmission_bounce),
            kernel_data.integrator.max_bounce,
            kernel_data.integrator.max_glossy_bounce,
            kernel_data.integrator.max_transmission_bounce,
            bool(kernel_data.integrator.use_bidirectional_path_tracing),
            kernel_data.integrator.bdpt_max_bounces,
            light->max_bounces))
    {
      continue;
    }
    CoherentGeometryInterface patches[COHERENT_GEOMETRY_MAX_INTERFACES];
    int patch_indices[COHERENT_GEOMETRY_MAX_INTERFACES];
    bool supported = true;
    for (int index = 0; index < candidate->count; index++) {
      const ccl_global KernelCoherentPatch *patch =
          &kernel_data_fetch(coherent_patches, candidate->patch[index]);
      if (patch->mode != 1 || candidate->event[index] != COHERENT_GEOMETRY_REFLECT) {
        supported = false;
        break;
      }
      patches[index] = {patch->center,
                        patch->tangent_u,
                        patch->tangent_v,
                        patch->half_u,
                        patch->half_v,
                        candidate->ior_before[index],
                        candidate->ior_after[index],
                        candidate->ior_opposite[index],
                        candidate->event[index],
                        candidate->expected_incident_side[index]};
      patch_indices[index] = candidate->patch[index];
    }
    if (!supported)
    {
      continue; /* Host rejects unsupported candidates before upload. */
    }
    CoherentGeometryPath path;
    float geometric_phase_cycles;
    if (!coherent_specular_connect(kg, candidate, light->co, sd->P, sd->N, patches, &path,
                                   &geometric_phase_cycles)) {
      continue;
    }
    const float3 previous = candidate->count == 0 ? float3(light->co) :
                                                    path.point[candidate->count - 1];
    if (!(dot(sd->N, previous - sd->P) > 0.0f) ||
        !coherent_specular_visible(kg, sd, light->co, patches, patch_indices, &path))
    {
      continue;
    }
    Spectrum emission;
    if (!light_sample_shader_eval_nee_constant(kg, light->shader_id, candidate->light,
                                               true, emission))
    {
      continue;
    }
    /* Point-light eval_fac is radiant intensity per emitted shader strength.
     * Geometric spreading includes receiver cosine and inverse square range;
     * the declared coherent Lambertian detector contributes 1/pi. */
    const Spectrum power = emission * detector_weight *
                           (light->spot.eval_fac * path.spreading * M_1_PI_F);
    if (!isfinite_safe(power) || is_zero(power)) {
      continue;
    }
    base_power[paths] = max(power, zero_spectrum());
    optical_path[paths] = path.optical_length_split;
    phase[paths] = light->coherence_phase;
    group[paths] = light->coherence_group;
    wavelength[paths] = light->coherence_wavelength;
    wavelength_low[paths] = light->coherence_wavelength_low;
    coherence_length[paths] = light->coherence_length;
    direct_route[paths] = candidate->count == 0;
    paths++;
  }
  Spectrum result = zero_spectrum();
  for (int index = 0; index < paths; index++) {
    result += base_power[index];
    if (primary_direct && direct_route[index]) *primary_direct += base_power[index];
  }
  for (int a = 0; a < paths; a++) {
    for (int b = a + 1; b < paths; b++) {
      if (group[a] != group[b]) {
        continue;
      }
      const float2 difference = coherent_geometry_add_split(
          optical_path[a], make_float2(-optical_path[b].x, -optical_path[b].y));
      const float gamma = coherent_gaussian_mutual_coherence(
          difference.x + difference.y, coherence_length[a]);
      const float phase_cycles = coherent_phase_cycles_split(
          difference,
          make_float2(wavelength[a], wavelength_low[a]),
          (phase[a] - phase[b]) * M_1_2PI_F);
      const float pair = 2.0f * gamma * cosf(M_2PI_F * phase_cycles);
      const Spectrum cross = pair * sqrt(base_power[a] * base_power[b]);
      result += cross;
      if (primary_direct) {
        *primary_direct += coherent_pass_direct_pair_share(cross, direct_route[a], direct_route[b]);
      }
    }
  }
  return result;
}

CCL_NAMESPACE_END
