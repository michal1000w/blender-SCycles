/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Vertex merging, the second half of vertex connection and merging (Georgiev et al. 2012,
 * Hachisuka et al. 2012), for Metal and CPU devices.
 *
 * The bidirectional light pass keeps every surface vertex of its subpaths in the photon map,
 * together with the recursive MIS terms dVCM, dVC and dVM. A camera vertex gathers the light
 * vertices inside the merge radius and treats each as the end of a complete path, weighted against
 * every other strategy that samples the same path: the camera path that reaches the emitter, next
 * event estimation and, in bidirectional path tracing, the vertex and sensor connections. Without
 * connections the same weights combine path tracing with merging only. */

#include "kernel/film/light_passes.h"
#include "kernel/integrator/bidirectional.h"
#include "kernel/integrator/photon_mapping.h"
#include "kernel/integrator/surface_shader.h"

CCL_NAMESPACE_BEGIN

/* Shader of the emitter with its visibility flags, which restrict the closures that the first
 * vertex of a light subpath lights. */
ccl_device_inline uint vcm_emitter_shader_flags(KernelGlobals kg, const int emitter_distribution)
{
  if (emitter_distribution < 0) {
    return 0u;
  }
  const int prim = kernel_data_fetch(light_distribution, emitter_distribution).prim;
  return prim >= 0 ? uint(kernel_data_fetch(tri_shader, prim)) :
                     uint(kernel_data_fetch(lights, ~prim).shader_id);
}

/* Cheap predicate to count the vertices around a camera vertex before any is evaluated. */
ccl_device_inline bool vcm_vertex_matches(KernelGlobals kg,
                                          const ccl_private ShaderData *sd,
                                          const uint slot,
                                          const uint cache,
                                          const int time_bin,
                                          const float radius2,
                                          const uint max_path_length)
{
  if (slot / kernel_integrator_state.vcm_cache_slots != cache) {
    return false;
  }
  const ccl_global KernelPhoton *photon = &kernel_integrator_state.photons[slot];
  if (photon->receiver_object != sd->object ||
      photon_time_bin(kg, photon_unpack_time(photon->time_wavelength)) != time_bin)
  {
    return false;
  }
  if (len_squared(photon->P - sd->P) > radius2) {
    return false;
  }
  /* The other side of a thin object and the far side of an edge are different surfaces. A
   * transmitting surface is lit from behind, where the normal points the other way. */
  packed_normal photon_normal;
  photon_normal.value = photon->normal;
  if (fabsf(dot(sd->Ng, photon_normal.decode())) < kernel_data.integrator.photon_normal_threshold)
  {
    return false;
  }
  /* Bounces of the complete path, as the camera path counts them when it reaches the emitter. */
  return kernel_integrator_state.vcm_vertices[slot].path_length <= max_path_length;
}

/* Write the merged light of a camera vertex to the film, as direct light writes its own.
 * Index 0 is light that reaches the vertex straight from an emitter, index 1 any other. */
ccl_device_inline_transport void vcm_merge_write(KernelGlobals kg,
                                                 ConstIntegratorState state,
                                                 ccl_global float *ccl_restrict render_buffer,
                                                 const ccl_private Spectrum *L,
                                                 const ccl_private Spectrum *L_diffuse,
                                                 const ccl_private Spectrum *L_glossy)
{
  const Spectrum throughput = INTEGRATOR_STATE(state, path, throughput);
  const PathRayVisibility visibility = INTEGRATOR_STATE(state, path, visibility);
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  const int sample = INTEGRATOR_STATE(state, path, sample);
  const int bounce = INTEGRATOR_STATE(state, path, bounce);
  ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);

  for (int i = 0; i < 2; i++) {
    if (is_zero(L[i])) {
      continue;
    }
    const bool direct = (i == 0 && bounce == 0);
    Spectrum contribution = throughput * L[i];
    const float unclamped = reduce_add(fabs(contribution));
    film_clamp_light(kg, &contribution, direct ? 0 : 1);
    if (is_zero(contribution)) {
      continue;
    }
    film_write_combined_pass(kg, visibility, path_flag, sample, contribution, buffer);

#ifdef __PASSES__
    if (!(kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES) ||
        (path_flag & PATH_RAY_SHADOW_CATCHER_HIT))
    {
      continue;
    }
    Spectrum diffuse, glossy;
    if (path_flag & PATH_RAY_ANY_PASS) {
      /* Seen through an earlier bounce, whose closures decide the pass. */
      if (!(path_flag & PATH_RAY_SURFACE_PASS)) {
        if (kernel_data.film.pass_volume_indirect != PASS_UNUSED) {
          film_write_pass_spectrum(buffer + kernel_data.film.pass_volume_indirect, contribution);
        }
        continue;
      }
      diffuse = Spectrum(INTEGRATOR_STATE(state, path, pass_diffuse_weight)) * contribution;
      glossy = Spectrum(INTEGRATOR_STATE(state, path, pass_glossy_weight)) * contribution;
    }
    else {
      const float scale = unclamped > 0.0f ? reduce_add(fabs(contribution)) / unclamped : 0.0f;
      diffuse = ensure_finite(throughput * L_diffuse[i] * scale);
      glossy = ensure_finite(throughput * L_glossy[i] * scale);
    }
    const int pass_diffuse = direct ? kernel_data.film.pass_diffuse_direct :
                                      kernel_data.film.pass_diffuse_indirect;
    const int pass_glossy = direct ? kernel_data.film.pass_glossy_direct :
                                     kernel_data.film.pass_glossy_indirect;
    const int pass_transmission = direct ? kernel_data.film.pass_transmission_direct :
                                           kernel_data.film.pass_transmission_indirect;
    if (pass_diffuse != PASS_UNUSED) {
      film_write_pass_spectrum(buffer + pass_diffuse, diffuse);
    }
    if (pass_glossy != PASS_UNUSED) {
      film_write_pass_spectrum(buffer + pass_glossy, glossy);
    }
    if (pass_transmission != PASS_UNUSED) {
      film_write_pass_spectrum(buffer + pass_transmission, contribution - diffuse - glossy);
    }
#endif
  }
}

/* Merge the camera vertex with the light subpath vertices around it and write their light.
 * Returns false when a shader evaluation has to wait for an image tile: nothing was written
 * then, and the caller repeats the merge. */
ccl_device_inline_transport bool vcm_merge(KernelGlobals kg,
                                           IntegratorState state,
                                           ccl_private ShaderData *sd,
                                           ccl_global float *ccl_restrict render_buffer)
{
  if (!vcm_merging_enabled(kg) || !bdpt_enabled_for_surface_path(kg, state) ||
      !(sd->runtime_flag & SR_BSDF_HAS_EVAL) || !kernel_integrator_state.photons ||
      !kernel_integrator_state.vcm_vertices || kernel_integrator_state.photon_hash_size == 0)
  {
    return true;
  }

  const uint cache = kernel_integrator_state.bdpt_cache_count > 1 ?
                         uint(INTEGRATOR_STATE(state, path, sample)) -
                             kernel_integrator_state.bdpt_cache_start_sample :
                         0u;
  if (cache >= kernel_integrator_state.bdpt_cache_count) {
    atomic_fetch_and_or_uint32(&kernel_integrator_state.queue_counter->bdpt_error, 1u);
    return true;
  }

  const bool merge_only = vcm_merge_only(kg);
  if (merge_only && !vcm_camera_path_is_specular(state)) {
    return true;
  }
  const uint bounce = uint(INTEGRATOR_STATE(state, path, bounce));
  if (bounce + 1u > uint(kernel_data.integrator.max_bounce)) {
    /* Even the shortest light subpath completes a path that is too long. */
    return true;
  }
  /* A light subpath of n vertices completes a path of n + bounce - 1 bounces. */
  const uint max_path_length = uint(kernel_data.integrator.max_bounce) + 1u - bounce;

  const float radius = kernel_integrator_state.photon_radius;
  const float radius2 = sqr(radius);
  const int3 base = photon_cell(sd->P, radius);
  const float time = INTEGRATOR_STATE(state, ray, time);
  const int time_bin = photon_time_bin(kg, time);
  const uint capacity = kernel_integrator_state.photon_capacity;

  int num_valid = 0;
  for (int z = -1; z <= 1; z++) {
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        uint node = kernel_integrator_state.photon_hash[vcm_hash_bucket(
            kg, base + make_int3(x, y, z), time, cache)];
        uint traversed = 0;
        while (node != 0u && node <= capacity && traversed++ < capacity) {
          const uint slot = node - 1u;
          node = kernel_integrator_state.photons[slot].next;
          if (vcm_vertex_matches(kg, sd, slot, cache, time_bin, radius2, max_path_length)) {
            num_valid++;
          }
        }
      }
    }
  }
  if (num_valid == 0) {
    return true;
  }

  /* Evaluate a bounded subset of a dense neighborhood: randomized systematic sampling gives every
   * vertex the same inclusion probability, see photon_mapping_gather(). */
  const float selection_probability = min(
      1.0f, float(kernel_data.integrator.vcm_merge_max) / float(num_valid));
  const uint rng_pixel = INTEGRATOR_STATE(state, path, rng_pixel);
  const uint sample = uint(INTEGRATOR_STATE(state, path, sample));
  const float selection_offset = hash_uint3_to_float(rng_pixel, sample, bounce ^ 0x76636d00u);

  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  const bool camera_spectral = (path_flag & PATH_RAY_SPECTRAL) != 0u;
  const bool use_connections = bdpt_connections_enabled(kg);
  const BDPTMISWeight vc_factor = vcm_mis_vc_factor(kg);
  const BDPTMISWeight camera_d_vcm = use_connections ?
                                         BDPTMISWeight::from_encoded(
                                             INTEGRATOR_STATE(state, path, bdpt_d_vcm)) :
                                         BDPTMISWeight(0.0f);
  const BDPTMISWeight camera_d_vm = BDPTMISWeight::from_encoded(
      INTEGRATOR_STATE(state, path, bdpt_d_vm));
  /* Density estimate over the merge disk, of the subpaths that share the time bin. */
  const float normalization = float(kernel_data.integrator.photon_time_bins) /
                              (selection_probability * M_PI_F * radius2 *
                               float(max(kernel_integrator_state.vcm_light_path_count, 1u)));
  const bool write_lightgroups = kernel_data.film.pass_lightgroup != PASS_UNUSED &&
                                 !(path_flag & PATH_RAY_SHADOW_CATCHER_HIT);

  Spectrum L[2] = {zero_spectrum(), zero_spectrum()};
  Spectrum L_diffuse[2] = {zero_spectrum(), zero_spectrum()};
  Spectrum L_glossy[2] = {zero_spectrum(), zero_spectrum()};
  int valid_index = 0;
  bool written = false;

  for (int z = -1; z <= 1; z++) {
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        uint node = kernel_integrator_state.photon_hash[vcm_hash_bucket(
            kg, base + make_int3(x, y, z), time, cache)];
        uint traversed = 0;
        while (node != 0u && node <= capacity && traversed++ < capacity) {
          const uint slot = node - 1u;
          const ccl_global KernelPhoton *photon = &kernel_integrator_state.photons[slot];
          node = photon->next;
          if (!vcm_vertex_matches(kg, sd, slot, cache, time_bin, radius2, max_path_length)) {
            continue;
          }
          bool selected = true;
          if (selection_probability < 1.0f) {
            selected = floorf(float(valid_index + 1) * selection_probability + selection_offset) >
                       floorf(float(valid_index) * selection_probability + selection_offset);
          }
          valid_index++;
          if (!selected) {
            continue;
          }

          const KernelVCMVertex light_vertex = kernel_integrator_state.vcm_vertices[slot];
          const bool first_vertex = light_vertex.path_length == 2u;
          packed_normal photon_direction;
          photon_direction.value = photon->direction;
          const float3 light_direction = -photon_direction.decode();
          /* The light subpath measures flux through the geometric surface, the closures of
           * the camera vertex include the cosine of their shading normal. */
          const float cos_geometric = fabsf(dot(sd->Ng, light_direction));
          if (!(cos_geometric > 1.0e-4f)) {
            continue;
          }

          const uint shader_flags = first_vertex ?
                                        (vcm_emitter_shader_flags(
                                             kg, light_vertex.emitter_distribution) |
                                         SHADER_USE_MIS) :
                                        SHADER_USE_MIS;
          BsdfEval camera_eval;
          float roughness_squared = 0.0f;
          const float camera_pdf = surface_shader_bsdf_eval(
              kg, state, sd, light_direction, &camera_eval, shader_flags, roughness_squared);
          if (!(camera_pdf > 0.0f) || bsdf_eval_is_zero(&camera_eval)) {
            continue;
          }
          /* Density with which the light subpath continues from here toward the previous
           * camera vertex. */
          const float camera_reverse_pdf = bdpt_reverse_pdf(kg, state, sd, light_direction);
          if (sd->runtime_flag & SR_CACHE_MISS) {
            if (!written) {
              return false;
            }
            /* The tiles a surface point reads do not depend on the direction: no vertex after
             * the first is expected here. Its light is lost rather than written twice. */
            sd->runtime_flag &= ~SR_CACHE_MISS;
            continue;
          }

          /* The camera path samples this light vertex itself and reaches the rest of the subpath
           * with next event estimation at the first vertex, with a connection at any other. */
          const float selection_ratio = first_vertex ?
                                            bdpt_light_selection_ratio(kg,
                                                                       light_vertex.emitter_distribution,
                                                                       light_vertex.emitter_P,
                                                                       photon->P,
                                                                       sd->time,
                                                                       photon->P,
                                                                       sd->N,
                                                                       sd->runtime_flag,
                                                                       sd->object) :
                                            (use_connections ? 1.0f : 0.0f);
          const BDPTMISWeight w_light = BDPTMISWeight::from_encoded(light_vertex.d_vcm) *
                                            selection_ratio * vc_factor +
                                        BDPTMISWeight::from_encoded(light_vertex.d_vm) * camera_pdf;
          const BDPTMISWeight w_camera = camera_d_vcm * vc_factor +
                                         camera_d_vm * camera_reverse_pdf;
          const float mis_weight = merge_only ?
                                       1.0f :
                                       (BDPTMISWeight(1.0f) + w_light + w_camera).inverse();

          Spectrum weight = rgb_to_spectrum(make_float3(photon->power)) *
                            bdpt_light_vertex_spectral_weight(
                                kg, state, photon->time_wavelength, camera_spectral) *
                            (mis_weight * normalization / cos_geometric);
          if (polarization_enabled(kg)) {
            /* Merged light is taken as unpolarized, like direct light. */
            const auto sensitivity = polarization_surface_transport(kg,
                                                                    sd,
                                                                    light_direction,
                                                                    polarization_path_read(state),
                                                                    true,
                                                                    nullptr,
                                                                    bsdf_eval_sum(&camera_eval),
                                                                    false,
                                                                    shader_flags);
            weight *= sensitivity.value[0];
          }
          const Spectrum photon_L = weight * bsdf_eval_sum(&camera_eval);
          if (!isfinite_safe(photon_L)) {
            continue;
          }

          const int i = first_vertex ? 0 : 1;
          L[i] += photon_L;
          L_diffuse[i] += weight * camera_eval.diffuse;
          L_glossy[i] += weight * camera_eval.glossy;
          written = true;

#ifdef __PASSES__
          if (write_lightgroups) {
            const int lightgroup = object_lightgroup(kg, photon->emitter_object);
            if (lightgroup != LIGHTGROUP_NONE) {
              Spectrum group_contribution = INTEGRATOR_STATE(state, path, throughput) * photon_L;
              film_clamp_light(kg, &group_contribution, (first_vertex && bounce == 0u) ? 0 : 1);
              ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);
              film_write_pass_spectrum(buffer + kernel_data.film.pass_lightgroup + 3 * lightgroup,
                                       group_contribution);
            }
          }
#endif
        }
      }
    }
  }

  if (written) {
    vcm_merge_write(kg, state, render_buffer, L, L_diffuse, L_glossy);
  }
  return true;
}

CCL_NAMESPACE_END
