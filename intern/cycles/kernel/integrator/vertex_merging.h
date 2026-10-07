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

/* Largest number of light vertices that one camera vertex evaluates. */
#define VCM_MERGE_MAX 32

/* Cheap predicate that selects the vertices around a camera vertex before any is evaluated. */
ccl_device_inline bool vcm_vertex_matches(KernelGlobals kg,
                                          const ccl_private ShaderData *sd,
                                          const uint slot,
                                          const uint cache,
                                          const uint group,
                                          const int time_bin,
                                          const float radius2,
                                          const uint max_path_length)
{
  /* Buckets are shared with other cells, caches and groups. */
  if (slot / kernel_integrator_state.vcm_cache_slots != cache ||
      vcm_light_path_group(kg,
                           (slot % kernel_integrator_state.vcm_cache_slots) /
                               kernel_integrator_state.vcm_path_slots) != group)
  {
    return false;
  }
  const ccl_global KernelPhoton *photon = &kernel_integrator_state.photons[slot];
  if (len_squared(photon->P - sd->P) > radius2 || photon->receiver_object != sd->object ||
      photon_time_bin(kg, photon_unpack_time(photon->time_wavelength)) != time_bin)
  {
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

/* Merge the camera vertex with the light subpath vertices in the hash chains `heads` and write
 * their light, see vcm_merge(). Kept out of line: most camera vertices have no light vertex
 * near them and never need the working memory of this. */
ccl_device __attribute__((noinline)) bool vcm_merge_chains(
    KernelGlobals kg,
    IntegratorState state,
    ccl_private ShaderData *sd,
    ccl_global float *ccl_restrict render_buffer,
    const ccl_private uint *heads,
    const int num_heads,
    const uint cache,
    const uint group,
    const float radius)
{
  const bool merge_only = vcm_merge_only(kg);
  const uint bounce = uint(INTEGRATOR_STATE(state, path, bounce));
  /* A light subpath of n vertices completes a path of n + bounce - 1 bounces. */
  const uint max_path_length = uint(kernel_data.integrator.max_bounce) + 1u - bounce;
  const float radius2 = sqr(radius);
  const int time_bin = photon_time_bin(kg, INTEGRATOR_STATE(state, ray, time));
  const uint capacity = kernel_integrator_state.photon_capacity;

  /* One pass over the chains: keep a uniform random subset of a dense neighborhood in a
   * reservoir, so that the number of shader evaluations is bounded and every vertex has the
   * same probability to be one of them. */
  const int merge_max = clamp(kernel_data.integrator.vcm_merge_max, 1, VCM_MERGE_MAX);
  uint selected[VCM_MERGE_MAX];
  int num_valid = 0;
  uint reservoir_rng = lcg_init(hash_uint3(INTEGRATOR_STATE(state, path, rng_pixel),
                                           uint(INTEGRATOR_STATE(state, path, sample)),
                                           bounce ^ 0x76636d00u));
  for (int i = 0; i < num_heads; i++) {
    uint node = heads[i];
    uint traversed = 0;
    while (node != 0u && node <= capacity && traversed++ < capacity) {
      const uint slot = node - 1u;
      node = kernel_integrator_state.photons[slot].next;
      if (!vcm_vertex_matches(kg, sd, slot, cache, group, time_bin, radius2, max_path_length)) {
        continue;
      }
      if (num_valid < merge_max) {
        selected[num_valid] = slot;
      }
      else {
        const int replace = int(lcg_step_float(&reservoir_rng) * float(num_valid + 1));
        if (replace < merge_max) {
          selected[replace] = slot;
        }
      }
      num_valid++;
    }
  }
  if (num_valid == 0) {
    return true;
  }
  const int num_selected = min(num_valid, merge_max);

  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  const bool camera_spectral = (path_flag & PATH_RAY_SPECTRAL) != 0u;
  const bool use_connections = bdpt_connections_enabled(kg);
  const BDPTMISWeight inverse_eta = BDPTMISWeight::from_encoded(
      -vcm_mis_eta(kg, radius).encoded());
  const BDPTMISWeight camera_d_vcm = use_connections ?
                                         BDPTMISWeight::from_encoded(
                                             INTEGRATOR_STATE(state, path, bdpt_d_vcm)) :
                                         BDPTMISWeight(0.0f);
  const BDPTMISWeight camera_d_vc = BDPTMISWeight::from_encoded(
      INTEGRATOR_STATE(state, path, bdpt_d_vc));
  /* The light vertex continues the merge strategies of the earlier camera vertices across a
   * sharp event at this one. */
  const BDPTMISWeight camera_merges = vcm_surface_is_sharp(kg, sd) ?
                                          BDPTMISWeight::from_encoded(
                                              INTEGRATOR_STATE(state, path, bdpt_d_vm)) +
                                              BDPTMISWeight::from_encoded(
                                                  INTEGRATOR_STATE(state, path, bdpt_d_vp)) :
                                          BDPTMISWeight(0.0f);
  const bool needs_reverse_pdf = !merge_only &&
                                 (use_connections || !(camera_merges == 0.0f) ||
                                  !(camera_d_vc == 0.0f));
  /* Density estimate over the merge disk, of the subpaths that share the time bin. */
  const float normalization = float(kernel_data.integrator.photon_time_bins) * float(num_valid) /
                              (float(num_selected) * M_PI_F * radius2 *
                               float(max(kernel_integrator_state.vcm_light_path_count, 1u)));
  const bool write_lightgroups = kernel_data.film.pass_lightgroup != PASS_UNUSED &&
                                 !(path_flag & PATH_RAY_SHADOW_CATCHER_HIT);

  Spectrum L[2] = {zero_spectrum(), zero_spectrum()};
  Spectrum L_diffuse[2] = {zero_spectrum(), zero_spectrum()};
  Spectrum L_glossy[2] = {zero_spectrum(), zero_spectrum()};
  bool written = false;

  for (int index = 0; index < num_selected; index++) {
    const uint slot = selected[index];
    const ccl_global KernelPhoton *photon = &kernel_integrator_state.photons[slot];
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
                                  (vcm_emitter_shader_flags(kg, light_vertex.emitter_distribution) |
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
    const float camera_reverse_pdf = needs_reverse_pdf ?
                                         bdpt_reverse_pdf(kg, state, sd, light_direction) :
                                         0.0f;
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
     * with next event estimation at the first vertex, with a connection at any other. Each
     * of these strategies is weighted relative to the merge: dVM of the paper is the sum
     * without merges over eta plus the merges. */
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
    const BDPTMISWeight w_light =
        (BDPTMISWeight::from_encoded(light_vertex.d_vcm) * selection_ratio +
         BDPTMISWeight::from_encoded(light_vertex.d_vc) * camera_pdf) *
            inverse_eta +
        BDPTMISWeight::from_encoded(light_vertex.d_vm) * camera_pdf;
    const BDPTMISWeight w_camera = (camera_d_vcm + camera_d_vc * camera_reverse_pdf) *
                                       inverse_eta +
                                   camera_merges * camera_reverse_pdf;
    const float mis_weight = merge_only ? 1.0f :
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

  if (written) {
    vcm_merge_write(kg, state, render_buffer, L, L_diffuse, L_glossy);
  }
  return true;
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
      !kernel_integrator_state.vcm_vertices || kernel_integrator_state.photon_hash_size == 0 ||
      !kernel_integrator_state.photon_stored || *kernel_integrator_state.photon_stored == 0u)
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

  if (vcm_merge_only(kg) && !vcm_camera_path_is_specular(state)) {
    return true;
  }
  /* The shortest light subpath that is kept has two vertices, and three when only those of
   * caustics are: even that may complete a path that is too long. */
  const uint bounce = uint(INTEGRATOR_STATE(state, path, bounce));
  const uint min_path_length = vcm_keep_all(kg) ? 2u : 3u;
  if (bounce + min_path_length - 1u > uint(kernel_data.integrator.max_bounce)) {
    return true;
  }

  /* The chains of the cells that the merge disk reaches. Cells that share a bucket share its
   * chain. */
  const float radius = vcm_path_merge_radius(kg, state);
  const float time = INTEGRATOR_STATE(state, ray, time);
  const int3 cell_min = vcm_cell(kg, sd->P - make_float3(radius));
  const int3 cell_max = vcm_cell(kg, sd->P + make_float3(radius));
  const uint group = uint(INTEGRATOR_STATE(state, path, sample)) %
                     max(kernel_integrator_state.vcm_groups, 1u);
  uint heads[27];
  uint buckets[27];
  int num_heads = 0;
  for (int z = cell_min.z; z <= cell_max.z; z++) {
    for (int y = cell_min.y; y <= cell_max.y; y++) {
      for (int x = cell_min.x; x <= cell_max.x; x++) {
        const uint bucket = vcm_hash_bucket(kg, make_int3(x, y, z), time, cache, group);
        const uint head = kernel_integrator_state.photon_hash[bucket];
        if (head == 0u || num_heads == 27) {
          continue;
        }
        bool repeated = false;
        for (int i = 0; i < num_heads; i++) {
          repeated = repeated || (buckets[i] == bucket);
        }
        if (!repeated) {
          buckets[num_heads] = bucket;
          heads[num_heads++] = head;
        }
      }
    }
  }
  if (num_heads == 0) {
    return true;
  }
  return vcm_merge_chains(kg, state, sd, render_buffer, heads, num_heads, cache, group, radius);
}

CCL_NAMESPACE_END
