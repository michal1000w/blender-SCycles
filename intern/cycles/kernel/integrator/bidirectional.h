/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Light-vertex-cache bidirectional path tracing, for Metal and CPU devices.
 *
 * Light subpaths are generated in one pass and one connectible vertex per path is selected
 * into a compact global cache by reservoir sampling. Camera paths sample this cache at their
 * surface vertices. Recursive MIS terms follow Georgiev's balance-heuristic formulation, so
 * connecting a pair only requires local PDFs and the two cached partial weights. */

#include "kernel/bvh/bvh.h"
#include "kernel/camera/camera.h"
#include "kernel/film/adaptive_sampling.h"
#include "kernel/geom/shader_data.h"
#include "kernel/integrator/bidirectional_transport.h"
#include "kernel/integrator/bidirectional_volume.h"
#include "kernel/integrator/path_state.h"
#include "kernel/integrator/photon_mapping.h"
#include "kernel/integrator/state_flow.h"
#include "kernel/integrator/state_util.h"
#include "kernel/integrator/surface_shader.h"
#ifdef __VOLUME__
#  include "kernel/integrator/volume_shader.h"
#endif
#include "kernel/light/coherent_history_kernel.h"
#ifdef __MNEE__
#  include "kernel/integrator/mnee.h"
#endif
#include "kernel/sample/lcg.h"
#include "kernel/util/compact_indices.h"
#include "util/atomic.h"

CCL_NAMESPACE_BEGIN

#if defined(__VOLUME__) && !defined(__KERNEL_METAL__)
/* Defined in `shade_volume.h`, whose volume shading uses the recursive MIS below. Metal kernels
 * are members of one context class, where no prior declaration is needed. */
ccl_device bool bdpt_volume_connection_transmittance(KernelGlobals kg,
                                                     IntegratorState state,
                                                     const float3 start,
                                                     const float3 end,
                                                     const float time,
                                                     const float wavelength_rand,
                                                     const uint segment,
                                                     ccl_private Spectrum *throughput);
#endif

ccl_device_inline bool bdpt_camera_supported(KernelGlobals kg)
{
  const CameraType camera_type = CameraType(kernel_data.cam.type);
  if (kernel_data.cam.interocular_offset != 0.0f || camera_type == CAMERA_CUSTOM) {
    return false;
  }
  return camera_type == CAMERA_PERSPECTIVE ||
         ((camera_type == CAMERA_PANORAMA || camera_type == CAMERA_ORTHOGRAPHIC) &&
          kernel_data.cam.aperturesize == 0.0f && kernel_data.cam.num_motion_steps == 0);
}

/* Whether light subpaths are connected to camera vertices and to the sensor. Without, the
 * bidirectional pass only provides the vertices that camera paths merge with. */
ccl_device_inline bool bdpt_connections_enabled(KernelGlobals kg)
{
  return kernel_data.integrator.bdpt_use_connections && bdpt_camera_supported(kg);
}

/* Vertex merging: camera vertices gather the light subpath vertices around them. */
ccl_device_inline bool vcm_merging_enabled(KernelGlobals kg)
{
  return kernel_data.integrator.use_vertex_merging &&
         kernel_integrator_state.vcm_eta_scale > 0.0f;
}

/* Whether every light subpath vertex is kept for merging, not only those of caustics. */
ccl_device_inline bool vcm_keep_all(KernelGlobals kg)
{
  return !kernel_data.integrator.vcm_caustics_only;
}

/* Whether scattering at the surface concentrates light, as a caustic needs: every closure is a
 * mirror, a refraction or a narrow glossy lobe. A light vertex is kept for merging only when all
 * scattering events between the emitter and it were sharp. Next event estimation and the camera
 * path sample every other path well enough, and merging those too would only add the low
 * frequency noise of a sparse photon map to them.
 *
 * This is a property of the surface, the same for the light subpath and for the camera path
 * that weights its strategies against it. An event that samples a delta closure is sharp at any
 * surface: paths with and without it do not overlap. */
ccl_device_inline bool vcm_surface_is_sharp(KernelGlobals kg, const ccl_private ShaderData *sd)
{
  if (vcm_keep_all(kg)) {
    return true;
  }
  bool sharp = false;
  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];
    if (!CLOSURE_IS_BSDF_OR_BSSRDF(sc->type) || CLOSURE_IS_BSDF_TRANSPARENT(sc->type)) {
      continue;
    }
    if (!CLOSURE_IS_BSDF(sc->type) ||
        bsdf_get_specular_roughness_squared(sc) >
            kernel_data.integrator.vcm_sharp_roughness_squared)
    {
      return false;
    }
    sharp = true;
  }
  return sharp;
}

/* Distance that sets the merge radius of a path whose first vertex is at `P`. */
ccl_device_inline float vcm_camera_distance(KernelGlobals kg, const float3 P)
{
  const Transform camera_to_world = kernel_data.cam.cameratoworld;
  return len(P - make_float3(camera_to_world.x.w, camera_to_world.y.w, camera_to_world.z.w));
}

/* Merge radius of a path that travels `length` from the camera to its first vertex that is not
 * a delta event: the footprint of a few pixels there. It is the same for every strategy that
 * samples the path, which all share that vertex, so their weights still sum to one. */
ccl_device_inline float vcm_merge_radius(KernelGlobals kg, const float length)
{
  const float maximum = kernel_integrator_state.photon_radius;
  return clamp(kernel_integrator_state.vcm_radius_base +
                   kernel_integrator_state.vcm_radius_slope * length,
               1.0e-4f * maximum,
               maximum);
}

ccl_device_inline float vcm_path_merge_radius(KernelGlobals kg, ConstIntegratorState state)
{
  return vcm_merge_radius(kg, fabsf(INTEGRATOR_STATE(state, path, bdpt_merge_length)));
}

/* Merging does not need to invert the camera, so its recursive MIS runs for every camera. */
ccl_device_inline bool bdpt_recursion_supported(KernelGlobals kg)
{
  return bdpt_camera_supported(kg) || kernel_data.integrator.use_vertex_merging;
}

/* MIS factor of merging relative to a connection: eta is the number of light subpaths a camera
 * vertex merges with times the area of its merge disk. An empty sum when merging is off, which
 * leaves every weight as it is without. */
ccl_device_inline BDPTMISWeight vcm_mis_eta(KernelGlobals kg, const float radius)
{
  return vcm_merging_enabled(kg) ?
             BDPTMISWeight(kernel_integrator_state.vcm_eta_scale * sqr(radius)) :
             BDPTMISWeight(0.0f);
}

/* Diagnostic of the merging estimator on its own (CYCLES_VCM_MERGE_ONLY): every path is sampled
 * by merging at its first vertex from the camera that is not a delta event, with unit weight.
 * Other strategies only keep the paths without such a vertex. The result is progressive photon
 * mapping, which converges to the same image when the estimator is normalized correctly. */
ccl_device_inline bool vcm_merge_only(KernelGlobals kg)
{
  return kernel_data.integrator.vcm_merge_only && vcm_merging_enabled(kg);
}

/* Whether the camera path has not scattered at anything but delta events yet. */
ccl_device_inline bool vcm_camera_path_is_specular(ConstIntegratorState state)
{
  return !(INTEGRATOR_STATE(state, path, bdpt_merge_length) > 0.0f);
}

ccl_device_inline bool bdpt_enabled_for_surface_path(KernelGlobals kg, ConstIntegratorState state)
{
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  return kernel_data.integrator.use_bidirectional_path_tracing && bdpt_recursion_supported(kg) &&
         !(path_flag & (PATH_RAY_SHADOW_CATCHER_HIT | PATH_RAY_SHADOW_CATCHER_PASS |
                        PATH_RAY_BDPT_UNSUPPORTED));
}

/* The first camera-volume scatter competes with NEE and a light-to-sensor
 * connection even though later camera connections use the unsupported-path
 * fallback. Preserve that partition when its continuation hits an emitter.
 * Additional real events are deliberately not covered by this predicate. */
ccl_device_inline bool bdpt_enabled_for_emission(KernelGlobals kg, ConstIntegratorState state)
{
  if (bdpt_enabled_for_surface_path(kg, state)) {
    return true;
  }
  const uint32_t flag = INTEGRATOR_STATE(state, path, flag);
  return kernel_data.integrator.use_bidirectional_path_tracing && bdpt_recursion_supported(kg) &&
         !(flag & (PATH_RAY_SHADOW_CATCHER_HIT | PATH_RAY_SHADOW_CATCHER_PASS)) &&
         (flag & PATH_RAY_BDPT_UNSUPPORTED) && INTEGRATOR_STATE(state, path, bounce) == 1 &&
         INTEGRATOR_STATE(state, path, volume_bounce) == 1;
}

ccl_device_inline float bdpt_safe_pdf(const float pdf)
{
  return max(pdf, 1.0e-20f);
}

/* Selection probabilities have two distinct contexts: emitted paths use the global CDF,
 * whereas NEE uses the receiver's tree. Never replace an emission density by a tree density. */
ccl_device_inline uint bdpt_tree_emitter(KernelGlobals kg, const int object, const int prim)
{
  if (prim < 0) {
    return kernel_data_fetch(light_to_tree, object);
  }
  const uint lookup = kernel_data_fetch(object_lookup_offset, object);
  const uint offset = kernel_data_fetch(object_prim_offset, object);
  return kernel_data_fetch(triangle_to_tree, prim - offset + lookup);
}

ccl_device_inline float bdpt_forward_selection_pdf(KernelGlobals kg,
                                                   IntegratorState state,
                                                   const int object,
                                                   const int prim,
                                                   const float flat_pdf)
{
#ifdef __LIGHT_TREE__
  if (kernel_data.integrator.use_light_tree &&
      INTEGRATOR_STATE(state, path, bounce) != 0 &&
      !(INTEGRATOR_STATE(state, path, flag) & PATH_RAY_MIS_SKIP))
  {
    return light_tree_pdf(kg,
                          INTEGRATOR_STATE(state, ray, P),
                          INTEGRATOR_STATE(state, path, mis_origin_n),
                          INTEGRATOR_STATE(state, ray, previous_dt),
                          INTEGRATOR_STATE(state, path, visibility),
                          INTEGRATOR_STATE(state, path, flag),
                          object,
                          bdpt_tree_emitter(kg, object, prim),
                          light_link_receiver_forward(kg, state));
  }
#endif
  return flat_pdf;
}

/* Evaluate the NEE/emission selection ratio at the first scattering vertex. The caller supplies
 * the reciprocal shading normal (toward the camera-side predecessor), not the emission-facing
 * normal. Subsequent light vertices already contain this ratio in their recursive d_vc term. */
ccl_device_inline float bdpt_light_selection_ratio(KernelGlobals kg,
                                                   const int distribution_index,
                                                   const float3 emitter_P,
                                                   const float3 shading_P,
                                                   const float time,
                                                   const float3 P,
                                                   const float3 N,
                                                   const int runtime_flags,
                                                   const int receiver,
                                                   const bool volume = false,
                                                   const float dt = 0.0f)
{
  /* NEE resamples the light point from the scattering position, including
   * volume NEE. The tree may instead be selected from an entire ray segment.
   * Do not use that segment origin for this conditional shape density. */
  const ccl_global KernelLightDistribution *distribution =
      &kernel_data_fetch(light_distribution, distribution_index);
  float shape_ratio = 1.0f;
  if (distribution->prim >= 0) {
    float distance;
    ShaderData emitter_sd;
    emitter_sd.P = emitter_P;
    emitter_sd.wi = safe_normalize_len(shading_P - emitter_P, &distance);
    emitter_sd.object = distribution->object_id;
    emitter_sd.prim = distribution->prim;
    emitter_sd.time = time;
    float3 V[3];
    triangle_world_space_vertices(kg, emitter_sd.object, emitter_sd.prim, time, V);
    emitter_sd.Ng = safe_normalize(cross(V[1] - V[0], V[2] - V[0]));
    float flat_selection;
    const float emission_pdf_a = triangle_light_emission_pdf(
        kg, emitter_sd.object, emitter_sd.prim, time, &flat_selection);
    float nee_pdf_w = triangle_light_pdf(kg, &emitter_sd, distance);
    if (kernel_data.integrator.use_light_tree) {
      nee_pdf_w *= flat_selection;
    }
    shape_ratio = distance > 0.0f && emission_pdf_a > 0.0f ?
                      nee_pdf_w * fabsf(dot(emitter_sd.Ng, emitter_sd.wi)) /
                          (sqr(distance) * emission_pdf_a) :
                      0.0f;
  }
  else {
    const ccl_global KernelLight *light = &kernel_data_fetch(lights, ~distribution->prim);
    if (light->type == LIGHT_AREA) {
      float distance;
      const float3 direction = safe_normalize_len(emitter_P - shading_P, &distance);
      const float area = area_light_is_ellipse(&light->area) ?
                             M_PI_F * light->area.len_u * light->area.len_v * 0.25f :
                             light->area.len_u * light->area.len_v;
      const LightEval eval = area_light_eval_from_intersection(
          light, shading_P, direction, distance);
      shape_ratio = distance > 0.0f ?
                        eval.pdf * fabsf(dot(light->area.dir, direction)) * area / sqr(distance) :
                        0.0f;
    }
    else if ((light->type == LIGHT_POINT || light->type == LIGHT_SPOT) &&
             light->spot.is_sphere && light->spot.radius > 0.0f)
    {
      float distance;
      const float3 direction = safe_normalize_len(emitter_P - shading_P, &distance);
      const uint flag = (runtime_flags & SR_BSDF_HAS_TRANSMISSION) ?
                            PATH_RAY_MIS_HAD_TRANSMISSION : 0u;
      const LightEval eval = light->type == LIGHT_POINT ?
                                 point_light_eval_from_intersection(
                                     light, shading_P, direction, distance, N, flag) :
                                 spot_light_eval_from_intersection(
                                     kg, light, shading_P, direction, distance, N, flag);
      const float cosine = fabsf(dot(safe_normalize(emitter_P - light->co), direction));
      const float area = 4.0f * M_PI_F * sqr(light->spot.radius);
      shape_ratio = distance > 0.0f ? eval.pdf * cosine * area / sqr(distance) : 0.0f;
    }
  }

#ifdef __LIGHT_TREE__
  if (kernel_data.integrator.use_light_tree) {
    const ccl_global KernelLightDistribution *emitter =
        &kernel_data_fetch(light_distribution, distribution_index);
    const float flat_pdf = kernel_data_fetch(light_distribution, distribution_index + 1).totarea -
                           emitter->totarea;
    if (!(flat_pdf > 0.0f)) {
      return 0.0f;
    }
    const uint tree_id = bdpt_tree_emitter(kg, emitter->object_id, emitter->prim);
    const uint flag = (runtime_flags & SR_BSDF_HAS_TRANSMISSION) ?
                          PATH_RAY_MIS_HAD_TRANSMISSION : 0u;
    const float tree_pdf = volume ?
        light_tree_pdf<true>(kg, P, N, dt, flag, emitter->object_id, tree_id, receiver) :
        light_tree_pdf<false>(kg, P, N, 0.0f, flag, emitter->object_id, tree_id, receiver);
    return shape_ratio * tree_pdf / flat_pdf;
  }
#endif
  return shape_ratio;
}

ccl_device_inline uint bdpt_pack_vertex_support(const uint path_length,
                                                const uint selection_count,
                                                const uint transparent_bounce)
{
  return (path_length & 0xffu) | ((selection_count & 0xfffu) << 8u) |
         (min(transparent_bounce, 4095u) << 20u);
}

ccl_device_inline uint bdpt_vertex_path_length(const ccl_private KernelBDPTVertex *light_vertex)
{
  return light_vertex->path_length & 0xffu;
}

ccl_device_inline uint bdpt_vertex_selection_count(
    const ccl_private KernelBDPTVertex *light_vertex)
{
  return (light_vertex->path_length >> 8u) & 0xfffu;
}

ccl_device_inline Spectrum bdpt_light_vertex_spectral_weight(
    KernelGlobals kg,
    ConstIntegratorState state,
    const uint time_wavelength,
    const bool camera_spectral)
{
#ifdef __SPECTRAL__
  if (photon_is_spectral(time_wavelength)) {
    const float light_rand = photon_unpack_wavelength_rand(time_wavelength);
    if (camera_spectral) {
      const float camera_rand = path_rng_1D(kg,
                                            INTEGRATOR_STATE(state, path, rng_pixel),
                                            INTEGRATOR_STATE(state, path, sample),
                                            PRNG_BOUNCE_NUM + PRNG_WAVELENGTH);
      float light_pdf;
      const float light_wavelength = sample_wavelength(light_rand, &light_pdf);
      const float camera_wavelength = sample_wavelength(camera_rand);
      return make_spectrum(photon_spectral_kernel(light_wavelength, camera_wavelength) /
                           bdpt_safe_pdf(light_pdf));
    }
    return dispersion_throughput_weight(kg, light_rand);
  }
#else
  (void)kg;
  (void)state;
  (void)time_wavelength;
  (void)camera_spectral;
#endif
  return one_spectrum();
}

ccl_device_inline float bdpt_infinite_position_pdf(KernelGlobals kg,
                                                   const float3 P,
                                                   const float3 direction)
{
  const float3 scene_center = make_float3(kernel_data.integrator.photon_scene);
  const float scene_radius = kernel_data.integrator.photon_scene.w;
  const float3 target_center = make_float3(kernel_data.integrator.photon_target);
  const float target_radius = min(kernel_data.integrator.photon_target.w, scene_radius);
  const float target_probability = (target_radius > 0.0f &&
                                    (target_radius < 0.999f * scene_radius ||
                                     len_squared(target_center - scene_center) > 1.0e-10f)) ?
                                       0.9f :
                                       0.0f;

  const float3 scene_delta = P - scene_center;
  const float3 target_delta = P - target_center;
  const float scene_distance2 = len_squared(scene_delta) - sqr(dot(scene_delta, direction));
  const float target_distance2 = len_squared(target_delta) - sqr(dot(target_delta, direction));
  float pdf = 0.0f;
  if (scene_radius > 0.0f && scene_distance2 <= sqr(scene_radius)) {
    pdf += (1.0f - target_probability) /
           max(M_PI_F * sqr(scene_radius), 1.0e-20f);
  }
  if (target_probability > 0.0f && target_distance2 <= sqr(target_radius)) {
    pdf += target_probability / max(M_PI_F * sqr(target_radius), 1.0e-20f);
  }
  return pdf;
}

ccl_device_inline float bdpt_point_emission_direction_pdf(KernelGlobals kg,
                                                          const float3 light_P,
                                                          const float3 direction)
{
  constexpr float uniform_pdf = M_1_2PI_F * 0.5f;
  constexpr float target_probability = 0.9f;
  const float3 target_center = make_float3(kernel_data.integrator.photon_target);
  const float target_radius = kernel_data.integrator.photon_target.w;
  const float3 to_target = target_center - light_P;
  const float distance2 = len_squared(to_target);
  if (!(target_radius > 0.0f) || distance2 <= sqr(target_radius)) {
    return uniform_pdf;
  }

  const float inv_distance = inversesqrtf(distance2);
  const float3 target_direction = to_target * inv_distance;
  const float cos_half_angle = safe_sqrtf(1.0f - sqr(target_radius) / distance2);
  float pdf = (1.0f - target_probability) * uniform_pdf;
  if (dot(direction, target_direction) >= cos_half_angle) {
    pdf += target_probability / max(M_2PI_F * (1.0f - cos_half_angle), 1.0e-20f);
  }
  return pdf;
}

ccl_device_inline float bdpt_spot_emission_direction_pdf(
    KernelGlobals kg, const ccl_global KernelLight *klight, const float3 direction)
{
  const float one_minus_cos_angle = 1.0f - klight->spot.cos_half_spot_angle;
  if (!(one_minus_cos_angle > 0.0f)) {
    return 1.0f;
  }
  const float attenuation = spot_light_attenuation(
      &klight->spot, spot_light_to_local(kg, klight, direction));
  const float blend_width = isfinite_safe(klight->spot.spot_smooth) ?
                                1.0f / klight->spot.spot_smooth :
                                0.0f;
  const float integrated_cosine = max(one_minus_cos_angle - 0.5f * blend_width, 1.0e-20f);
  return attenuation / (M_2PI_F * integrated_cosine);
}

/* dVC of a camera path that reached an emitter: the strategies that sample its vertices from the
 * light side. The emitter follows the last vertex, and a first light vertex is only merged
 * with when all vertices are kept: the strategy of that vertex is still pending. */
ccl_device_inline_transport BDPTMISWeight bdpt_camera_d_vc(KernelGlobals kg,
                                                           ConstIntegratorState state)
{
  const BDPTMISWeight d_vc = BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vc));
  if (!vcm_merging_enabled(kg)) {
    return d_vc;
  }
  BDPTMISWeight merges = BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vm));
  if (vcm_keep_all(kg)) {
    merges = merges + BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vp));
  }
  return d_vc + merges * vcm_mis_eta(kg, vcm_path_merge_radius(kg, state));
}

/* A delta event removes its local NEE alternative (d_vcm = 0), but earlier
 * connectible vertices can retain light-path alternatives in d_vc. Pure delta
 * camera prefixes have both terms zero and still receive unit weight. */
ccl_device_inline_transport float bdpt_emission_mis_weight_infinite(KernelGlobals kg,
                                                                    IntegratorState state,
                                                                    const float direct_pdf_w,
                                                                    const float position_pdf,
                                                                    const float selection_ratio)
{
  if (INTEGRATOR_STATE(state, path, bounce) == 0)
  {
    return 1.0f;
  }
  if (vcm_merge_only(kg) && INTEGRATOR_STATE(state, path, volume_bounce) == 0) {
    return vcm_camera_path_is_specular(state) ? 1.0f : 0.0f;
  }
  const float emission_pdf_w = direct_pdf_w * position_pdf;
  const BDPTMISWeight w_camera =
      BDPTMISWeight(direct_pdf_w) * selection_ratio *
          BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vcm)) +
      BDPTMISWeight(emission_pdf_w) * bdpt_camera_d_vc(kg, state);
  return (BDPTMISWeight(1.0f) + w_camera).inverse();
}

ccl_device_inline_transport float bdpt_emission_mis_weight_surface(
    KernelGlobals kg, IntegratorState state, const ccl_private ShaderData *sd)
{
  if (INTEGRATOR_STATE(state, path, bounce) == 0)
  {
    return 1.0f;
  }
  if (vcm_merge_only(kg) && INTEGRATOR_STATE(state, path, volume_bounce) == 0) {
    return vcm_camera_path_is_specular(state) ? 1.0f : 0.0f;
  }

  const bool front = (sd->shader_flag & SD_MIS_FRONT) != 0;
  const bool back = (sd->shader_flag & SD_MIS_BACK) != 0;
  if ((!front && !back) || (!(sd->type & PRIMITIVE_TRIANGLE))) {
    return 1.0f;
  }

  /* Emission uses the fixed emitter CDF and uniform area at the shutter time.
   * NEE instead follows the conditional triangle sampler. A two-sided emitter
   * samples either hemisphere with equal probability. */
  const float side_pdf = (front && back) ? 0.5f : 1.0f;
  float flat_selection;
  const float direct_pdf_a = triangle_light_emission_pdf(
      kg, sd->object, sd->prim, sd->time, &flat_selection);
  const float cos_at_light = max(fabsf(dot(sd->Ng, sd->wi)), 1.0e-8f);
  const float emission_pdf_w = direct_pdf_a * side_pdf * cos_at_light * M_1_PI_F;
  float nee_pdf_w = triangle_light_pdf(kg, sd, sd->ray_length);
  if (kernel_data.integrator.use_light_tree) {
    nee_pdf_w *= flat_selection;
  }
  const float nee_pdf_a = nee_pdf_w * cos_at_light / sqr(sd->ray_length);
  const float selection_ratio = bdpt_forward_selection_pdf(
      kg, state, sd->object, sd->prim, flat_selection) / bdpt_safe_pdf(flat_selection);
  const BDPTMISWeight w_camera =
      BDPTMISWeight(nee_pdf_a) * selection_ratio *
          BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vcm)) +
      BDPTMISWeight(emission_pdf_w) * bdpt_camera_d_vc(kg, state);
  return (BDPTMISWeight(1.0f) + w_camera).inverse();
}

ccl_device_inline_transport float bdpt_emission_mis_weight_lamp(
    KernelGlobals kg,
    IntegratorState state,
    const ccl_global KernelLight *klight,
    const float3 ray_P,
    const float3 ray_D,
    const float distance,
    const float nee_pdf_w)
{
  if (INTEGRATOR_STATE(state, path, bounce) == 0)
  {
    return 1.0f;
  }
  if (vcm_merge_only(kg) && INTEGRATOR_STATE(state, path, volume_bounce) == 0) {
    return vcm_camera_path_is_specular(state) ? 1.0f : 0.0f;
  }

  float area = 0.0f;
  float cos_at_light = 0.0f;
  if (klight->type == LIGHT_AREA) {
    area = area_light_is_ellipse(&klight->area) ?
               M_PI_F * klight->area.len_u * klight->area.len_v * 0.25f :
               klight->area.len_u * klight->area.len_v;
    cos_at_light = fabsf(dot(klight->area.dir, -ray_D));
  }
  else if ((klight->type == LIGHT_POINT || klight->type == LIGHT_SPOT) &&
           klight->spot.is_sphere)
  {
    area = 4.0f * M_PI_F * sqr(klight->spot.radius);
    const float3 hit_P = ray_P + ray_D * distance;
    cos_at_light = fabsf(dot(safe_normalize(hit_P - klight->co), -ray_D));
  }
  else {
    /* Singular lights cannot be reached by an ordinary camera ray. */
    return 1.0f;
  }

  if (!(area > 0.0f) || !(cos_at_light > 0.0f)) {
    return 1.0f;
  }

  const float direct_pdf_a = kernel_data.integrator.distribution_pdf_lights / area;
  const float emission_pdf_w = direct_pdf_a * cos_at_light * M_1_PI_F;
  const BDPTMISWeight d_vcm = BDPTMISWeight::from_encoded(
                                  INTEGRATOR_STATE(state, path, bdpt_d_vcm)) *
                              sqr(distance) / cos_at_light;
  const BDPTMISWeight d_vc = bdpt_camera_d_vc(kg, state) / cos_at_light;
  const float selection_ratio = bdpt_forward_selection_pdf(
      kg, state, klight->object_id, -1, kernel_data.integrator.distribution_pdf_lights) /
      bdpt_safe_pdf(kernel_data.integrator.distribution_pdf_lights);
  const float nee_pdf_a = kernel_data.integrator.distribution_pdf_lights * nee_pdf_w *
                          cos_at_light / sqr(distance);
  return (BDPTMISWeight(1.0f) + BDPTMISWeight(nee_pdf_a) * selection_ratio * d_vcm +
          BDPTMISWeight(emission_pdf_w) * d_vc)
      .inverse();
}

/* Reconstruct the reciprocal shader before evaluating a reverse density. Cycles closures are
 * prepared for an incoming upper-hemisphere direction: swapping wi alone returns zero for
 * microfacet transmission and also retains the wrong relative IOR and layered mixture weights.
 * An inconsistent reverse PDF breaks the partition of the bidirectional MIS weights. */
/* Keep the full shader graph in one callable body. Replicating it at every MIS query makes
 * Metal compilation and the caller's live register set unnecessarily large. */
ccl_device __attribute__((noinline)) float bdpt_reverse_pdf(
    KernelGlobals kg,
    IntegratorState state,
    ccl_private ShaderData *sd,
    const float3 sampled_wo,
    const bool light_path = false,
    ccl_private Spectrum *reciprocal_eval = nullptr,
    const bool delta_event = false,
    const ccl_private PolarizationSpectrumState *forward_input = nullptr,
    ccl_private PolarizationSpectrumState *forward_output = nullptr)
{
  Ray reverse_ray ccl_optional_struct_init;
  reverse_ray.D = -normalize(sampled_wo);
  reverse_ray.P = sd->P - reverse_ray.D;
  reverse_ray.tmin = 0.0f;
  reverse_ray.tmax = 1.0f;
  reverse_ray.time = sd->time;
#ifdef __RAY_DIFFERENTIALS__
  reverse_ray.dP = differential_zero_compact();
  reverse_ray.dD = differential_zero_compact();
#endif
  Intersection reverse_isect;
  reverse_isect.t = 1.0f;
  reverse_isect.u = sd->u;
  reverse_isect.v = sd->v;
  reverse_isect.prim = sd->prim;
  reverse_isect.object = sd->object;
  reverse_isect.type = sd->type;
  ShaderData reverse_sd;
  shader_setup_from_ray(kg, &reverse_sd, &reverse_ray, &reverse_isect);
#ifdef __SPECTRAL__
  reverse_sd.rand_wavelength = sd->rand_wavelength;
#endif
#ifdef __NESTED_DIELECTRICS__
  /* The same interface seen from the other side. */
  reverse_sd.medium_ior = sd->medium_ior;
#endif
  const PathRayVisibility visibility = path_state_ray_visibility(state);
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE>(
      kg, state, &reverse_sd, nullptr, visibility, INTEGRATOR_STATE(state, path, flag));
  if (reverse_sd.runtime_flag & SR_CACHE_MISS) {
    sd->runtime_flag |= SR_CACHE_MISS;
    return 0.0f;
  }
  surface_shader_prepare_closures(kg, state, &reverse_sd, visibility);
  BsdfEval reverse_eval;
  float roughness_squared = 0.0f;
  const float pdf = delta_event ?
                       surface_shader_bsdf_eval_delta(kg, &reverse_sd, sd->wi, &reverse_eval) :
                       surface_shader_bsdf_eval(kg,
                                             state,
                                             &reverse_sd,
                                             sd->wi,
                                             &reverse_eval,
                                             SHADER_USE_MIS,
                                             roughness_squared,
                                             !light_path);
  if (reciprocal_eval) {
    *reciprocal_eval = bsdf_eval_sum(&reverse_eval);
  }
  if (forward_input && forward_output) {
    *forward_output = polarization_surface_transport(kg, &reverse_sd, sd->wi,
        *forward_input, true, nullptr, bsdf_eval_sum(&reverse_eval), delta_event,
        SHADER_USE_MIS, true);
  }
  return pdf;
}

/* Whether the weights of a camera vertex depend on its reverse density, which takes another
 * evaluation of its shader. With merging alone they only do at a sharp surface that carries the
 * merge strategies of earlier vertices on. */
ccl_device_inline_transport bool bdpt_camera_vertex_needs_reverse_pdf(
    KernelGlobals kg, ConstIntegratorState state, const ccl_private ShaderData *sd)
{
  if (bdpt_connections_enabled(kg)) {
    return true;
  }
  if (!vcm_merging_enabled(kg)) {
    return false;
  }
  return (INTEGRATOR_STATE(state, path, bdpt_d_vm) != -INFINITY ||
          INTEGRATOR_STATE(state, path, bdpt_d_vp) != -INFINITY) &&
         vcm_surface_is_sharp(kg, sd);
}

/* Strategies that reach the current camera vertex from the light side, relative to the density
 * of the light subpath that ends at it: merging at the vertex, a connection to it and, through
 * dVC and the reverse density of the vertex, everything further along the camera path.
 *
 * `light_side_sharp` is whether every scattering event of the light subpath is sharp: it then
 * extends to a vertex that is kept for merging here (`merge_here`, which also needs such an
 * event to exist) and, across a sharp event at this vertex, at the earlier vertices. */
ccl_device_inline_transport BDPTMISWeight bdpt_camera_vertex_alternatives(
    KernelGlobals kg,
    ConstIntegratorState state,
    const ccl_private ShaderData *sd,
    const float reverse_pdf,
    const bool light_side_sharp,
    const bool merge_here)
{
  BDPTMISWeight alternatives = BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vc));
  BDPTMISWeight here = 0.0f;
  if (vcm_merging_enabled(kg)) {
    const BDPTMISWeight eta = vcm_mis_eta(kg, vcm_path_merge_radius(kg, state));
    if (light_side_sharp && vcm_surface_is_sharp(kg, sd)) {
      alternatives = alternatives +
                     (BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vm)) +
                      BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vp))) *
                         eta;
    }
    if (merge_here) {
      here = eta;
    }
  }
  alternatives = alternatives * reverse_pdf;
  if (bdpt_connections_enabled(kg)) {
    alternatives = alternatives +
                   BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vcm));
  }
  return alternatives + here;
}

/* Balance-heuristic weight for the next-event strategy in the presence of light tracing and
 * vertex connections. This is Georgiev's recursive form of Veach MIS (SmallVCM eqs. 44-45).
 * The current implementation evaluates the exact emission density for triangle and area lights;
 * other analytic lights retain the regular two-strategy balance weight until their singular
 * measures are handled explicitly. */
ccl_device_inline_transport float bdpt_nee_mis_weight(KernelGlobals kg,
                                                      IntegratorState state,
                                                      ccl_private ShaderData *sd,
                                                      const ccl_private LightSample *ls,
                                                      const float bsdf_pdf)
{
  if (vcm_merge_only(kg)) {
    return 0.0f;
  }
  const float direct_pdf = bdpt_safe_pdf(ls->pdf);
  BDPTMISWeight w_light = BDPTMISWeight(bsdf_pdf) / direct_pdf;
  /* The light subpath of this path is its emitter alone: the vertex here is its first, which
   * is merged with only when all vertices are kept. */
  const bool merge_here = vcm_keep_all(kg);
  const bool needs_reverse_pdf = bdpt_camera_vertex_needs_reverse_pdf(kg, state, sd);
  if (!needs_reverse_pdf && !(merge_here && vcm_merging_enabled(kg)) &&
      (ls->type == LIGHT_TRIANGLE || ls->type == LIGHT_AREA || ls->type == LIGHT_BACKGROUND ||
       (ls->type == LIGHT_SUN && kernel_data_fetch(lights, ls->prim).sun.angle != 0.0f)))
  {
    /* No light subpath strategy samples this path: the emitter is reached by next event
     * estimation or by the camera path alone. */
    return (BDPTMISWeight(1.0f) + w_light).inverse();
  }

  float emission_position_pdf = 0.0f;
  float emission_side_pdf = 1.0f;
  BDPTMISWeight w_camera = 0.0f;
  if (ls->type == LIGHT_TRIANGLE) {
    float flat_selection;
    emission_position_pdf = triangle_light_emission_pdf(
        kg, ls->object, ls->prim, sd->time, &flat_selection);
    const int shader_flags = kernel_data_fetch(shaders, ls->shader & SHADER_MASK).flags;
    if ((shader_flags & SD_MIS_FRONT) && (shader_flags & SD_MIS_BACK)) {
      emission_side_pdf = 0.5f;
    }
  }
  else if (ls->type == LIGHT_AREA) {
    const ccl_global KernelLight *klight = &kernel_data_fetch(lights, ls->prim);
    const bool ellipse = area_light_is_ellipse(&klight->area);
    const float area = ellipse ? M_PI_F * klight->area.len_u * klight->area.len_v * 0.25f :
                                 klight->area.len_u * klight->area.len_v;
    emission_position_pdf = kernel_data.integrator.distribution_pdf_lights /
                            max(area, 1.0e-20f);
  }
  else if (ls->type == LIGHT_BACKGROUND || ls->type == LIGHT_SUN) {
    if (ls->type == LIGHT_SUN) {
      const ccl_global KernelLight *klight = &kernel_data_fetch(lights, ls->prim);
      if (klight->sun.angle == 0.0f) {
        w_light = 0.0f;
      }
    }
    const float position_pdf = bdpt_infinite_position_pdf(kg, sd->P, -ls->D);
    const float reverse_pdf = needs_reverse_pdf ? bdpt_reverse_pdf(kg, state, sd, ls->D) : 0.0f;
    const float cos_camera = max(fabsf(dot(sd->Ng, ls->D)), 1.0e-8f);
    const float emission_selection_ratio = kernel_data.integrator.distribution_pdf_lights /
                                           bdpt_safe_pdf(ls->pdf_selection);
    w_camera = BDPTMISWeight(emission_selection_ratio) * position_pdf * cos_camera *
               bdpt_camera_vertex_alternatives(kg, state, sd, reverse_pdf, true, merge_here);
  }
  else if (ls->type == LIGHT_POINT || ls->type == LIGHT_SPOT) {
    const ccl_global KernelLight *klight = &kernel_data_fetch(lights, ls->prim);
    if (klight->spot.is_sphere) {
      const float area = 4.0f * M_PI_F * sqr(klight->spot.radius);
      emission_position_pdf = kernel_data.integrator.distribution_pdf_lights /
                              max(area, 1.0e-20f);
    }
    else if (klight->spot.radius > 0.0f) {
      /* The receiver-facing disk has no reciprocal light-tracing strategy. Preserve the exact
       * regular two-strategy Cycles weight for this non-physical compatibility light. */
      return light_sample_mis_weight_nee(kg, ls->pdf, bsdf_pdf);
    }
    else {
      /* A point endpoint is singular in position, so an ordinary BSDF sample cannot hit it. */
      w_light = 0.0f;
      const float emission_pdf_w = kernel_data.integrator.distribution_pdf_lights *
                                   ((ls->type == LIGHT_SPOT) ?
                                        bdpt_spot_emission_direction_pdf(kg, klight, -ls->D) :
                                        bdpt_point_emission_direction_pdf(kg, klight->co, -ls->D));
      const float reverse_pdf = needs_reverse_pdf ? bdpt_reverse_pdf(kg, state, sd, ls->D) : 0.0f;
      const float cos_camera = max(fabsf(dot(sd->Ng, ls->D)), 1.0e-8f);
      w_camera = BDPTMISWeight(emission_pdf_w) * cos_camera / direct_pdf *
                 bdpt_camera_vertex_alternatives(kg, state, sd, reverse_pdf, true, merge_here);
    }
  }

  if (emission_position_pdf > 0.0f) {
    const float reverse_pdf = needs_reverse_pdf ? bdpt_reverse_pdf(kg, state, sd, ls->D) : 0.0f;
    const float cos_camera = max(fabsf(dot(sd->Ng, ls->D)), 1.0e-8f);
    const BDPTMISWeight emission_to_direct = BDPTMISWeight(emission_position_pdf) *
                                             emission_side_pdf * cos_camera / M_PI_F / direct_pdf;
    w_camera = emission_to_direct *
               bdpt_camera_vertex_alternatives(kg, state, sd, reverse_pdf, true, merge_here);
  }

  return (BDPTMISWeight(1.0f) + w_light + w_camera).inverse();
}

/* Match camera_sample_perspective()'s time interpolation in camera space. */
ccl_device_inline float3 bdpt_perspective_image_point(KernelGlobals kg,
                                                      const float3 raster,
                                                      const float time)
{
  const ProjectionTransform raster_to_camera = kernel_data.cam.rastertocamera;
  float3 image_P = transform_perspective(&raster_to_camera, raster);
  if (kernel_data.cam.have_perspective_motion) {
    if (time < 0.5f) {
      const ProjectionTransform raster_to_camera_pre = kernel_data.cam.perspective_pre;
      const float3 image_pre = transform_perspective(&raster_to_camera_pre, raster);
      image_P = interp(image_pre, image_P, time * 2.0f);
    }
    else {
      const ProjectionTransform raster_to_camera_post = kernel_data.cam.perspective_post;
      const float3 image_post = transform_perspective(&raster_to_camera_post, raster);
      image_P = interp(image_P, image_post, (time - 0.5f) * 2.0f);
    }
  }
  return image_P;
}

ccl_device_inline_transport void bdpt_recursive_mis_before_measure_conversion(
    KernelGlobals kg,
    IntegratorState state,
    const ccl_private ShaderData *sd,
    ccl_private BDPTMISWeight *d_vcm_out,
    ccl_private BDPTMISWeight *d_vc_out)
{
  BDPTMISWeight d_vcm = BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vcm));
  BDPTMISWeight d_vc = BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vc));

  if (INTEGRATOR_STATE(state, path, bounce) == 0 && !bdpt_connections_enabled(kg)) {
    /* No light subpath is connected to the sensor: nothing competes for the first vertex. */
    d_vcm = 0.0f;
    d_vc = 0.0f;
  }
  else if (INTEGRATOR_STATE(state, path, bounce) == 0 && d_vcm == 0.0f) {
    float inverse_camera_pdf_w = 0.0f;
    if (kernel_data.cam.type == CAMERA_PERSPECTIVE) {
      Transform camera_to_world = kernel_data.cam.cameratoworld;
      if (kernel_data.cam.num_motion_steps) {
        transform_motion_array_interpolate(&camera_to_world,
                                           kernel_data_array(camera_motion),
                                           kernel_data.cam.num_motion_steps,
                                           sd->time);
      }
      const Transform world_to_camera = transform_inverse(camera_to_world);
      const float3 camera_D = normalize(
          transform_direction(&world_to_camera, INTEGRATOR_STATE(state, ray, D)));
      const float3 clipped_P = transform_point(&world_to_camera, INTEGRATOR_STATE(state, ray, P));
      /* Recover the actual aperture sample from the ray, including its near-plane shift.
       * This is the conditional camera density for that lens sample, not a differential
       * footprint approximation. Camera and light strategies use the same aperture measure. */
      const float3 lens_P = clipped_P - camera_D * (clipped_P.z / camera_D.z);
      const float3 image_origin = bdpt_perspective_image_point(kg, zero_float3(), sd->time);
      const float3 image_dx = bdpt_perspective_image_point(kg, make_float3(1, 0, 0), sd->time) -
                              image_origin;
      const float3 image_dy = bdpt_perspective_image_point(kg, make_float3(0, 1, 0), sd->time) -
                              image_origin;
      const float plane_scale = kernel_data.cam.aperturesize > 0.0f ?
                                    kernel_data.cam.focaldistance / image_origin.z :
                                    1.0f;
      const float3 sensor_to_plane = camera_D *
                                     ((image_origin.z * plane_scale - lens_P.z) / camera_D.z);
      inverse_camera_pdf_w = bdpt_camera_plane_inverse_pdf(
          sensor_to_plane, image_dx * plane_scale, image_dy * plane_scale);
      const float3 sensor_P = transform_point(&camera_to_world, lens_P);
      inverse_camera_pdf_w = bdpt_camera_clip_measure(
          inverse_camera_pdf_w, len_squared(sd->P - sensor_P), sd->ray_length);
    }
    else if (kernel_data.cam.type == CAMERA_ORTHOGRAPHIC &&
             kernel_data.cam.aperturesize == 0.0f && kernel_data.cam.num_motion_steps == 0)
    {
      const ProjectionTransform raster_to_camera = kernel_data.cam.rastertocamera;
      const float3 image_origin = transform_perspective(
          &raster_to_camera, make_float3(0.0f, 0.0f, 0.0f));
      const float3 image_x = transform_perspective(
          &raster_to_camera, make_float3(1.0f, 0.0f, 0.0f));
      const float3 image_y = transform_perspective(
          &raster_to_camera, make_float3(0.0f, 1.0f, 0.0f));
      const float image_pixel_area = len(cross(image_x - image_origin, image_y - image_origin));
      inverse_camera_pdf_w = image_pixel_area /
                             sqr(max(sd->ray_length, 1.0e-10f));
    }
    else if (kernel_data.cam.type == CAMERA_PANORAMA &&
             kernel_data.cam.aperturesize == 0.0f && kernel_data.cam.num_motion_steps == 0)
    {
      const Transform world_to_camera = kernel_data.cam.worldtocamera;
      const float3 camera_direction = normalize(transform_point(&world_to_camera, sd->P));
      const float3 ndc = make_float3(direction_to_panorama(&kernel_data.cam, camera_direction));
      const float raster_x = ndc.x * kernel_data.cam.width;
      const float raster_y = ndc.y * kernel_data.cam.height;
      constexpr float h = 0.01f;
      const float3 dx0 = camera_panorama_direction(&kernel_data.cam, raster_x - h, raster_y);
      const float3 dx1 = camera_panorama_direction(&kernel_data.cam, raster_x + h, raster_y);
      const float3 dy0 = camera_panorama_direction(&kernel_data.cam, raster_x, raster_y - h);
      const float3 dy1 = camera_panorama_direction(&kernel_data.cam, raster_x, raster_y + h);
      inverse_camera_pdf_w = len(cross((dx1 - dx0) * (0.5f / h),
                                       (dy1 - dy0) * (0.5f / h)));
      const Transform camera_to_world = kernel_data.cam.cameratoworld;
      const float3 sensor_P = transform_point(&camera_to_world, zero_float3());
      inverse_camera_pdf_w = bdpt_camera_clip_measure(
          inverse_camera_pdf_w, len_squared(sd->P - sensor_P), sd->ray_length);
    }
    else {
      const float camera_footprint = max(INTEGRATOR_STATE(state, ray, dD), 1.0e-8f);
      inverse_camera_pdf_w = sqr(camera_footprint);
    }
    d_vcm = BDPTMISWeight(kernel_integrator_state.bdpt_light_path_sample_ratio) *
            inverse_camera_pdf_w;
    d_vc = 0.0f;
  }

  *d_vcm_out = d_vcm;
  *d_vc_out = d_vc;
}

#ifdef __VOLUME__
/* Volume analogue of bdpt_nee_mis_weight(). Medium vertices use volume rather than projected-area
 * measure, so neither the recursive density nor its reverse phase density carries a cosine. */
ccl_device_inline_transport float bdpt_volume_nee_mis_weight(
    KernelGlobals kg,
    IntegratorState state,
    ccl_private ShaderData *sd,
    const ccl_private ShaderVolumePhases *phases,
    const float3 P,
    const ccl_private LightSample *ls,
    const float phase_pdf)
{
  const float direct_pdf = bdpt_safe_pdf(ls->pdf);
  BDPTMISWeight w_light = BDPTMISWeight(phase_pdf) / direct_pdf;

  const float3 original_wi = sd->wi;
  sd->wi = ls->D;
  BsdfEval reverse_eval;
  const float reverse_pdf = volume_shader_phase_eval(
      kg, state, sd, phases, original_wi, &reverse_eval, SHADER_USE_MIS, true, &P);
  sd->wi = original_wi;

  const float3 original_P = sd->P;
  const float original_ray_length = sd->ray_length;
  sd->P = P;
  sd->ray_length = len(P - sd->ray_P);
  BDPTMISWeight d_vcm;
  BDPTMISWeight d_vc;
  bdpt_recursive_mis_before_measure_conversion(kg, state, sd, &d_vcm, &d_vc);
  d_vcm *= sqr(max(sd->ray_length, 1.0e-10f));
  if (kernel_data.integrator.use_vertex_merging && INTEGRATOR_STATE(state, path, bounce) != 0) {
    /* A light subpath ends at its first medium vertex and surface vertices are not connected
     * to medium vertices: after a surface, only the camera path samples a path through this
     * vertex. Counting the strategies of the surface recursion here loses their share.
     * Bidirectional path tracing without merging keeps its weights as they were. */
    d_vcm = 0.0f;
    d_vc = 0.0f;
  }
  else if (!bdpt_connections_enabled(kg)) {
    /* No light subpath is connected to the sensor. */
    d_vcm = 0.0f;
  }
  sd->P = original_P;
  sd->ray_length = original_ray_length;

  float emission_position_pdf = 0.0f;
  float emission_side_pdf = 1.0f;
  BDPTMISWeight w_camera = 0.0f;
  if (ls->type == LIGHT_TRIANGLE) {
    float flat_selection;
    emission_position_pdf = triangle_light_emission_pdf(
        kg, ls->object, ls->prim, sd->time, &flat_selection);
    const int shader_flags = kernel_data_fetch(shaders, ls->shader & SHADER_MASK).flags;
    if ((shader_flags & SD_MIS_FRONT) && (shader_flags & SD_MIS_BACK)) {
      emission_side_pdf = 0.5f;
    }
  }
  else if (ls->type == LIGHT_AREA) {
    const ccl_global KernelLight *klight = &kernel_data_fetch(lights, ls->prim);
    const float area = area_light_is_ellipse(&klight->area) ?
                           M_PI_F * klight->area.len_u * klight->area.len_v * 0.25f :
                           klight->area.len_u * klight->area.len_v;
    emission_position_pdf = kernel_data.integrator.distribution_pdf_lights /
                            max(area, 1.0e-20f);
  }
  else if (ls->type == LIGHT_BACKGROUND || ls->type == LIGHT_SUN) {
    if (ls->type == LIGHT_SUN) {
      const ccl_global KernelLight *klight = &kernel_data_fetch(lights, ls->prim);
      if (klight->sun.angle == 0.0f) {
        w_light = 0.0f;
      }
    }
    const float emission_selection_ratio = kernel_data.integrator.distribution_pdf_lights /
                                           bdpt_safe_pdf(ls->pdf_selection);
    w_camera = BDPTMISWeight(emission_selection_ratio) *
               bdpt_infinite_position_pdf(kg, P, -ls->D) * (d_vcm + d_vc * reverse_pdf);
  }
  else if (ls->type == LIGHT_POINT || ls->type == LIGHT_SPOT) {
    const ccl_global KernelLight *klight = &kernel_data_fetch(lights, ls->prim);
    if (klight->spot.is_sphere) {
      const float area = 4.0f * M_PI_F * sqr(klight->spot.radius);
      emission_position_pdf = kernel_data.integrator.distribution_pdf_lights /
                              max(area, 1.0e-20f);
    }
    else if (klight->spot.radius > 0.0f) {
      return light_sample_mis_weight_nee(kg, ls->pdf, phase_pdf);
    }
    else {
      w_light = 0.0f;
      const float emission_pdf_w = kernel_data.integrator.distribution_pdf_lights *
                                   ((ls->type == LIGHT_SPOT) ?
                                        bdpt_spot_emission_direction_pdf(kg, klight, -ls->D) :
                                        bdpt_point_emission_direction_pdf(kg, klight->co, -ls->D));
      w_camera = BDPTMISWeight(emission_pdf_w) / direct_pdf * (d_vcm + d_vc * reverse_pdf);
    }
  }

  if (emission_position_pdf > 0.0f) {
    const BDPTMISWeight emission_to_direct = BDPTMISWeight(emission_position_pdf) *
                                             emission_side_pdf / M_PI_F / direct_pdf;
    w_camera = emission_to_direct * (d_vcm + d_vc * reverse_pdf);
  }
  return (BDPTMISWeight(1.0f) + w_light + w_camera).inverse();
}
#endif

ccl_device_inline_transport void bdpt_recursive_mis_after_hit(KernelGlobals kg,
                                                              IntegratorState state,
                                                              const ccl_private ShaderData *sd)
{
  BDPTMISWeight d_vcm;
  BDPTMISWeight d_vc;
  bdpt_recursive_mis_before_measure_conversion(kg, state, sd, &d_vcm, &d_vc);

  /* Logarithms of the measure conversion, taken once and only for sums that are not empty. */
  const float cos_fixed = max(fabsf(dot(sd->Ng, sd->wi)), 1.0e-8f);
  const float d_vm = INTEGRATOR_STATE(state, path, bdpt_d_vm);
  const float d_vp = INTEGRATOR_STATE(state, path, bdpt_d_vp);
  if (d_vcm == 0.0f && d_vc == 0.0f && d_vm == -INFINITY && d_vp == -INFINITY) {
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vcm) = -INFINITY;
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vc) = -INFINITY;
  }
  else {
    const float log_cos = BDPTMISWeight::Log::factor(cos_fixed);
    const float log_distance2 = (d_vcm == 0.0f) ? 0.0f :
                                                  BDPTMISWeight::Log::factor(
                                                      sqr(max(sd->ray_length, 1.0e-10f)));
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vcm) = d_vcm.encoded() + log_distance2 - log_cos;
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vc) = d_vc.encoded() - log_cos;
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vm) = d_vm - log_cos;
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vp) = d_vp - log_cos;
  }

  /* The path up to its first vertex that is not a delta event sets the merge radius. */
  const float merge_length = INTEGRATOR_STATE(state, path, bdpt_merge_length);
  if (!(merge_length > 0.0f)) {
    INTEGRATOR_STATE_WRITE(state, path, bdpt_merge_length) =
        (INTEGRATOR_STATE(state, path, bounce) == 0) ? -vcm_camera_distance(kg, sd->P) :
                                                       merge_length - sd->ray_length;
  }
}

ccl_device_inline_transport void bdpt_recursive_mis_undo_transparent_hit(
    IntegratorState state, const ccl_private ShaderData *sd)
{
  /* A null event keeps the original ray origin and does not introduce a vertex
   * in the MIS path. Restore the directional measure before tracing the rest
   * of that same edge. A primary ray must recompute its camera/clip conversion
   * at the eventual scattering vertex. */
  if (INTEGRATOR_STATE(state, path, bounce) == 0) {
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vcm) = -INFINITY;
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vc) = -INFINITY;
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vm) = -INFINITY;
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vp) = -INFINITY;
    INTEGRATOR_STATE_WRITE(state, path, bdpt_merge_length) = 0.0f;
  }
  else {
    const float cosine = max(fabsf(dot(sd->Ng, sd->wi)), 1.0e-8f);
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vm) = (BDPTMISWeight::from_encoded(INTEGRATOR_STATE(
                                                          state, path, bdpt_d_vm)) *
                                                      BDPTMISWeight(cosine))
                                                         .encoded();
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vp) = (BDPTMISWeight::from_encoded(INTEGRATOR_STATE(
                                                          state, path, bdpt_d_vp)) *
                                                      BDPTMISWeight(cosine))
                                                         .encoded();
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vcm) =
        (BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vcm)) *
         (BDPTMISWeight(cosine) / sqr(max(sd->ray_length, 1.0e-10f))))
            .encoded();
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vc) = (BDPTMISWeight::from_encoded(INTEGRATOR_STATE(
                                                          state, path, bdpt_d_vc)) *
                                                      BDPTMISWeight(cosine))
                                                         .encoded();
    /* The next hit measures the edge from the same origin again. */
    const float merge_length = INTEGRATOR_STATE(state, path, bdpt_merge_length);
    if (!(merge_length > 0.0f)) {
      INTEGRATOR_STATE_WRITE(state, path, bdpt_merge_length) = min(merge_length + sd->ray_length,
                                                                   0.0f);
    }
  }
}

/* `mergeable` is whether a light subpath vertex can be merged with the vertex that scatters: not
 * in a medium. `sharp` is vcm_surface_is_sharp() of that vertex. */
ccl_device_inline_transport void bdpt_recursive_mis_after_scatter(KernelGlobals kg,
                                                                  IntegratorState state,
                                                                  const int label,
                                                                  const float cos_out,
                                                                  const float forward_pdf,
                                                                  const float reverse_pdf,
                                                                  const bool mergeable,
                                                                  const bool sharp)
{
  const float4 previous = make_float4(INTEGRATOR_STATE(state, path, bdpt_d_vcm),
                                      INTEGRATOR_STATE(state, path, bdpt_d_vc),
                                      INTEGRATOR_STATE(state, path, bdpt_d_vm),
                                      INTEGRATOR_STATE(state, path, bdpt_d_vp));
  float4 next;
  if (label & LABEL_SINGULAR) {
    next = BDPTMISWeight::Log::scatter_delta_camera_vcm(
        previous, cos_out, forward_pdf, reverse_pdf);
  }
  else {
    /* A camera vertex is sampled from the light side only by a connection. */
    next = BDPTMISWeight::Log::scatter_camera_vcm(previous,
                                                  cos_out,
                                                  bdpt_safe_pdf(forward_pdf),
                                                  reverse_pdf,
                                                  bdpt_connections_enabled(kg) ? 1.0f : 0.0f,
                                                  mergeable && vcm_merging_enabled(kg),
                                                  sharp);
    /* The merge radius of the path is the one of this vertex from here on. */
    const float merge_length = INTEGRATOR_STATE(state, path, bdpt_merge_length);
    if (!(merge_length > 0.0f)) {
      INTEGRATOR_STATE_WRITE(state, path, bdpt_merge_length) = max(-merge_length, 1.0e-30f);
    }
  }

  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vcm) = next.x;
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vc) = next.y;
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vm) = next.z;
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vp) = next.w;
}

/* A cached vertex on a nested dielectric keeps the index of refraction of the medium around it,
 * which the path that connects to it cannot know: 16 bit fixed point in the upper half of the
 * light group field, exact for the 1.0 of a surface outside any medium. */
ccl_device_inline int bdpt_pack_light_group_medium(const int light_group, const float medium_ior)
{
  const uint quantized = uint(clamp(medium_ior * 8192.0f + 0.5f, 0.0f, 65535.0f));
  return int((uint(light_group) & 0xffffu) | (quantized << 16u));
}

ccl_device_inline int bdpt_vertex_light_group(const int packed_light_group)
{
  const int group = int(uint(packed_light_group) & 0xffffu);
  return (group & 0x8000) ? group - 0x10000 : group;
}

ccl_device_inline float bdpt_vertex_medium_ior(const int packed_light_group)
{
  return float(uint(packed_light_group) >> 16u) * (1.0f / 8192.0f);
}

/* Shader data of a cached vertex is rebuilt from its primitive, restore what the light subpath
 * knew about the media around it. */
ccl_device_inline void bdpt_vertex_restore_medium(const int packed_light_group,
                                                  ccl_private ShaderData *light_sd)
{
#ifdef __NESTED_DIELECTRICS__
  light_sd->medium_ior = bdpt_vertex_medium_ior(packed_light_group);
#else
  (void)packed_light_group;
  (void)light_sd;
#endif
}

ccl_device_inline void bdpt_fill_light_vertex(ccl_private KernelBDPTVertex *stored_vertex,
                                              const ccl_private ShaderData *sd,
                                              const ccl_private Ray *ray,
                                              const Spectrum throughput,
                                              const int emitter_object,
                                              const int emitter_distribution,
                                              const int light_group,
                                              const BDPTMISWeight d_vcm,
                                              const BDPTMISWeight d_vc,
                                              const uint path_length,
                                              const uint selection_count,
                                              const uint transparent_bounce,
                                              const uint flag,
                                              const uint emitter_shader_flags,
                                              const float wavelength_rand)
{
  stored_vertex->P = sd->P;
  stored_vertex->throughput = PackedSpectrum(throughput);
  stored_vertex->u = sd->u;
  stored_vertex->v = sd->v;
  stored_vertex->time_wavelength = photon_pack_time_wavelength(
      ray->time, wavelength_rand, (flag & PATH_RAY_SPECTRAL) != 0u);
  stored_vertex->incoming = packed_normal(ray->D).value;
  stored_vertex->prim = sd->prim;
  stored_vertex->object = sd->object;
  stored_vertex->type = sd->type;
  stored_vertex->emitter_object = emitter_object;
  stored_vertex->emitter_distribution = emitter_distribution;
  stored_vertex->emitter_P = ray->P;
#ifdef __NESTED_DIELECTRICS__
  stored_vertex->light_group = bdpt_pack_light_group_medium(light_group, sd->medium_ior);
#else
  stored_vertex->light_group = bdpt_pack_light_group_medium(light_group, 1.0f);
#endif
  stored_vertex->d_vcm = d_vcm.encoded();
  stored_vertex->d_vc = d_vc.encoded();
  stored_vertex->path_length = bdpt_pack_vertex_support(
      path_length, selection_count, transparent_bounce);
  stored_vertex->flag = flag;
  stored_vertex->emitter_shader_flags = emitter_shader_flags;
  stored_vertex->sensor_complete = 0;
}

ccl_device_inline void bdpt_reservoir_store_light_vertex(KernelGlobals kg,
                                                         const ccl_private ShaderData *sd,
                                                         const ccl_private Ray *ray,
                                                         const Spectrum throughput,
                                                         const int emitter_object,
                                                         const int emitter_distribution,
                                                         const int light_group,
                                                         const BDPTMISWeight d_vcm,
                                                         const BDPTMISWeight d_vc,
                                                         const BDPTMISWeight d_vm,
                                                         const bool sharp_prefix,
                                                         const uint path_length,
                                                         const uint transparent_bounce,
                                                         const uint flag,
                                                         const uint emitter_shader_flags,
                                                         const float wavelength_rand,
                                                         const CoherentPathHistory coherent_history,
                                                         const ccl_private PolarizationSpectrumState &polarization,
                                                         ccl_private uint *reservoir_rng,
                                                         ccl_private uint *reservoir_slot,
                                                         ccl_private uint *selection_count,
                                                         const uint light_path_index)
{
  *selection_count += 1u;
  const bool replace = *selection_count == 1u ||
                       lcg_step_float(reservoir_rng) < 1.0f / float(*selection_count);

  if (*selection_count == 1u) {
    /* Each path owns exactly one reservoir. Its path-indexed slot avoids a
     * contended allocation counter; stable ordering later removes empty paths. */
    *reservoir_slot = light_path_index;
  }
  if (*reservoir_slot >=
      kernel_integrator_state.bdpt_vertex_capacity * kernel_integrator_state.bdpt_cache_count)
  {
    return;
  }

  kernel_integrator_state.bdpt_vertex_indices[light_path_index] = *reservoir_slot;
  ccl_global KernelBDPTVertex *stored_vertex =
      &kernel_integrator_state.bdpt_vertices[*reservoir_slot];
  if (replace) {
    KernelBDPTVertex local_vertex;
    bdpt_fill_light_vertex(&local_vertex,
                           sd,
                           ray,
                           throughput,
                           emitter_object,
                           emitter_distribution,
                           light_group,
                           d_vcm,
                           d_vc,
                           path_length,
                           *selection_count,
                           transparent_bounce,
                           flag,
                           emitter_shader_flags,
                           wavelength_rand);
    local_vertex.d_vm = d_vm.encoded();
    local_vertex.sharp_prefix = sharp_prefix ? 1u : 0u;
    local_vertex.pad0 = 0u;
    local_vertex.pad1 = 0u;
    if (kernel_integrator_state.bdpt_coherent_history) {
      kernel_integrator_state.bdpt_coherent_history[*reservoir_slot] = coherent_history;
    }
    if (kernel_integrator_state.bdpt_polarization) {
      packed_normal incoming; incoming.value = local_vertex.incoming;
      kernel_integrator_state.bdpt_polarization[*reservoir_slot] = polarization_pack(
          polarization_reframe(polarization, ray->D, incoming.decode()));
    }
    *stored_vertex = local_vertex;
  }
  else {
    stored_vertex->path_length = bdpt_pack_vertex_support(
        stored_vertex->path_length & 0xffu, *selection_count, stored_vertex->path_length >> 20u);
  }
}

/* Photon map slot of the `stored`-th vertex that a light subpath keeps for merging. */
ccl_device_inline uint vcm_vertex_slot(KernelGlobals kg,
                                       const uint cache,
                                       const uint light_path_index,
                                       const uint stored)
{
  return cache * kernel_integrator_state.vcm_cache_slots +
         light_path_index * kernel_integrator_state.vcm_path_slots + stored;
}

/* Cell of the merge grid. A cell is as wide as the largest merge radius, so that a merge disk
 * reaches three cells along each axis at most, and fewer where the footprint of a pixel is
 * small: the disk of most camera vertices is inside one or two. */
ccl_device_inline int3 vcm_cell(KernelGlobals kg, const float3 P)
{
  return photon_cell(P, kernel_integrator_state.photon_radius);
}

/* Group of a light subpath, see `vcm_groups`. */
ccl_device_inline uint vcm_light_path_group(KernelGlobals kg, const uint light_path_index)
{
  return light_path_index % max(kernel_integrator_state.vcm_groups, 1u);
}

/* Hash key of a merge vertex: the caches of a batch and the groups of a cache share one grid. */
ccl_device_inline uint vcm_hash_bucket(KernelGlobals kg,
                                       const int3 cell,
                                       const float time,
                                       const uint cache,
                                       const uint group)
{
  return photon_hash_cell(cell,
                          photon_time_bin(kg, time) + int(cache) * 8 + int(group) * 64,
                          kernel_integrator_state.photon_hash_size);
}

/* Keep a light subpath vertex for merging: camera vertices gather the records around them. The
 * subpath owns a fixed range of slots, so that no allocation is shared between paths and the map
 * of the CPU does not depend on thread scheduling. A subpath with more vertices to keep than
 * slots keeps a uniform random subset, see vcm_finish_light_path(). */
ccl_device_inline void vcm_store_light_vertex(KernelGlobals kg,
                                              const ccl_private ShaderData *sd,
                                              const ccl_private Ray *ray,
                                              const Spectrum throughput,
                                              const int emitter_object,
                                              const int emitter_distribution,
                                              const BDPTMISWeight d_vcm,
                                              const BDPTMISWeight d_vc,
                                              const BDPTMISWeight d_vm,
                                              const uint path_length,
                                              const uint flag,
                                              const float wavelength_rand,
                                              const uint cache,
                                              const uint light_path_index,
                                              ccl_private uint *stored,
                                              ccl_private uint *kept,
                                              ccl_private uint *rng)
{
  const uint path_slots = kernel_integrator_state.vcm_path_slots;
  if (light_path_index >= kernel_integrator_state.vcm_light_path_count *
                              max(kernel_integrator_state.vcm_groups, 1u) ||
      path_slots == 0u)
  {
    return;
  }
  *kept += 1u;
  uint index = *stored;
  if (index >= path_slots) {
    index = min(uint(lcg_step_float(rng) * float(*kept)), *kept - 1u);
    if (index >= path_slots) {
      return;
    }
  }
  const uint slot = vcm_vertex_slot(kg, cache, light_path_index, index);
  if (slot >= kernel_integrator_state.photon_capacity) {
    *kept -= 1u;
    return;
  }
  *stored = min(*stored + 1u, path_slots);

  ccl_global KernelPhoton *photon = &kernel_integrator_state.photons[slot];
  photon->P = sd->P;
  photon->power = throughput;
  photon->emitter_object = emitter_object;
  photon->direction = packed_normal(ray->D).value;
  photon->normal = packed_normal(sd->Ng).value;
  photon->time_wavelength = photon_pack_time_wavelength(
      ray->time, wavelength_rand, (flag & PATH_RAY_SPECTRAL) != 0u);
  photon->receiver_object = sd->object;
  photon->next = 0u;

  ccl_global KernelVCMVertex *merge_vertex = &kernel_integrator_state.vcm_vertices[slot];
  merge_vertex->d_vcm = d_vcm.encoded();
  merge_vertex->d_vc = d_vc.encoded();
  merge_vertex->d_vm = d_vm.encoded();
  merge_vertex->path_length = path_length;
  merge_vertex->emitter_P = ray->P;
  merge_vertex->emitter_distribution = emitter_distribution;
}

/* Publish the vertices that a finished light subpath keeps. Each of the `stored` slots holds
 * one of its `kept` vertices with the same probability, and stands for all of them. */
ccl_device_inline void vcm_finish_light_path(KernelGlobals kg,
                                             const uint cache,
                                             const uint light_path_index,
                                             const uint stored,
                                             const uint kept)
{
  for (uint i = 0; i < stored; i++) {
    const uint slot = vcm_vertex_slot(kg, cache, light_path_index, i);
    ccl_global KernelPhoton *photon = &kernel_integrator_state.photons[slot];
    if (kept > stored) {
      photon->power = make_float3(photon->power) * (float(kept) / float(stored));
    }
#ifdef __KERNEL_GPU__
    const uint bucket = vcm_hash_bucket(kg,
                                        vcm_cell(kg, photon->P),
                                        photon_unpack_time(photon->time_wavelength),
                                        cache,
                                        vcm_light_path_group(kg, light_path_index));
    photon->next = atomic_exchange_uint32(&kernel_integrator_state.photon_hash[bucket], slot + 1u);
    /* Camera vertices do not look for light vertices in an empty map. */
    atomic_fetch_and_add_uint32(kernel_integrator_state.photon_stored, 1u);
#else
    /* The host links the hash chains in slot order, see integrator_vcm_map_build(). */
    kernel_integrator_state.photon_valid[slot] = 1;
#endif
  }
}

#ifndef __KERNEL_GPU__
/* Link the merge vertices of all light subpaths into the hash chains, in slot order, and return
 * their number. The hash heads must be zero. */
ccl_device uint integrator_vcm_map_build(KernelGlobals kg)
{
  uint stored = 0;
  const uint slots = min(kernel_integrator_state.photon_capacity,
                         kernel_integrator_state.bdpt_cache_count *
                             kernel_integrator_state.vcm_cache_slots);
  for (uint slot = 0; slot < slots; slot++) {
    if (!kernel_integrator_state.photon_valid[slot]) {
      continue;
    }
    ccl_global KernelPhoton *photon = &kernel_integrator_state.photons[slot];
    const uint cache_slot = slot % kernel_integrator_state.vcm_cache_slots;
    const uint bucket = vcm_hash_bucket(
        kg,
        vcm_cell(kg, photon->P),
        photon_unpack_time(photon->time_wavelength),
        slot / kernel_integrator_state.vcm_cache_slots,
        vcm_light_path_group(kg, cache_slot / kernel_integrator_state.vcm_path_slots));
    photon->next = kernel_integrator_state.photon_hash[bucket];
    kernel_integrator_state.photon_hash[bucket] = slot + 1u;
    stored++;
  }
  return stored;
}
#endif

/* Reconstruct a cached light vertex. Keeping this in one helper ensures the cache connection and
 * the light-tracing sensor connection evaluate exactly the same Cycles shader closures. */
ccl_device_inline bool bdpt_setup_light_vertex(KernelGlobals kg,
                                               IntegratorState state,
                                               const ccl_private KernelBDPTVertex *light_vertex,
                                               ccl_private ShaderData *light_sd)
{
  packed_normal packed_incoming;
  packed_incoming.value = light_vertex->incoming;
  const float3 light_incoming = packed_incoming.decode();

  Ray light_ray ccl_optional_struct_init;
  light_ray.P = light_vertex->P - light_incoming;
  light_ray.D = light_incoming;
  light_ray.tmin = 0.0f;
  light_ray.tmax = 1.0f;
  light_ray.time = photon_unpack_time(light_vertex->time_wavelength);
#ifdef __RAY_DIFFERENTIALS__
  light_ray.dP = differential_zero_compact();
  light_ray.dD = differential_zero_compact();
#endif

  if (light_vertex->type == PRIMITIVE_VOLUME) {
#ifdef __VOLUME__
    light_ray.P = light_vertex->P;
    light_ray.tmin = 0.0f;
    shader_setup_from_volume(light_sd, &light_ray, light_vertex->object);
    light_sd->P = light_vertex->P;
#  ifdef __SPECTRAL__
    light_sd->rand_wavelength = photon_unpack_wavelength_rand(light_vertex->time_wavelength);
#  endif
    VolumeShaderCoefficients coeff;
    return volume_shader_sample(kg, state, light_sd, &coeff) &&
           (light_sd->runtime_flag & SR_SCATTER) &&
           !(light_sd->runtime_flag & SR_CACHE_MISS);
#else
    return false;
#endif
  }

  Intersection light_isect;
  light_isect.t = 1.0f;
  light_isect.u = light_vertex->u;
  light_isect.v = light_vertex->v;
  light_isect.prim = light_vertex->prim;
  light_isect.object = light_vertex->object;
  light_isect.type = light_vertex->type;

  shader_setup_from_ray(kg, light_sd, &light_ray, &light_isect);
#ifdef __SPECTRAL__
  light_sd->rand_wavelength = photon_unpack_wavelength_rand(light_vertex->time_wavelength);
#endif
  bdpt_vertex_restore_medium(light_vertex->light_group, light_sd);
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE>(
      kg, state, light_sd, nullptr, PATH_RAY_VISIBILITY_GLOSSY, light_vertex->flag);
  if (light_sd->runtime_flag & SR_CACHE_MISS) {
    return false;
  }
  surface_shader_prepare_closures(kg, state, light_sd, PATH_RAY_VISIBILITY_GLOSSY);
  return true;
}

/* Sample/invert a built-in camera endpoint and return the measurement Jacobian which multiplies
 * Cycles' f*cos BSDF value when splatting to one raster pixel. */
ccl_device_inline bool bdpt_sample_camera_endpoint(KernelGlobals kg,
                                                   const ccl_private KernelBDPTVertex *light_vertex,
                                                   ccl_private ShaderData *light_sd,
                                                   const float2 rand_lens,
                                                   ccl_private float3 *raster,
                                                   ccl_private float3 *sensor_P,
                                                   ccl_private float *connection_jacobian)
{
  const float vertex_time = photon_unpack_time(light_vertex->time_wavelength);
  const CameraType camera_type = CameraType(kernel_data.cam.type);
  if (kernel_data.cam.interocular_offset != 0.0f || camera_type == CAMERA_CUSTOM) {
    return false;
  }

  if (camera_type == CAMERA_PERSPECTIVE) {
    Transform camera_to_world = kernel_data.cam.cameratoworld;
    if (kernel_data.cam.num_motion_steps) {
      transform_motion_array_interpolate(&camera_to_world,
                                         kernel_data_array(camera_motion),
                                         kernel_data.cam.num_motion_steps,
                                         vertex_time);
    }
    const Transform world_to_camera = transform_inverse(camera_to_world);
    const float3 camera_space_P = transform_point(&world_to_camera, light_vertex->P);
    if (!(camera_space_P.z > 0.0f)) {
      return false;
    }

    const bool use_dof = kernel_data.cam.aperturesize > 0.0f;
    float3 lens_P = zero_float3();
    float3 projection_camera = camera_space_P;
    if (use_dof) {
      const float2 lens_uv = camera_sample_aperture(&kernel_data.cam, rand_lens) *
                             kernel_data.cam.aperturesize;
      lens_P = make_float3(lens_uv);
      const float focus_t = kernel_data.cam.focaldistance / camera_space_P.z;
      projection_camera = lens_P + (camera_space_P - lens_P) * focus_t;
    }

    /* Match the clipping interval of camera_sample_perspective(), including its
     * pinhole-projection scale for aperture rays. A sensor strategy cannot see
     * geometry that the corresponding camera ray clips away. */
    const float clip_scale = len(projection_camera) / projection_camera.z;
    const float camera_distance = len(camera_space_P - lens_P);
    if (camera_distance <= kernel_data.cam.nearclip * clip_scale ||
        camera_distance >= (kernel_data.cam.nearclip + kernel_data.cam.cliplength) * clip_scale)
    {
      return false;
    }

    const float3 image_origin = bdpt_perspective_image_point(kg, zero_float3(), vertex_time);
    const float3 image_dx = bdpt_perspective_image_point(
                                kg, make_float3(1.0f, 0.0f, 0.0f), vertex_time) -
                            image_origin;
    const float3 image_dy = bdpt_perspective_image_point(
                                kg, make_float3(0.0f, 1.0f, 0.0f), vertex_time) -
                            image_origin;
    const float3 projected_on_image = projection_camera *
                                      (image_origin.z / projection_camera.z);
    const float3 image_delta = projected_on_image - image_origin;
    const float xx = dot(image_dx, image_dx);
    const float xy = dot(image_dx, image_dy);
    const float yy = dot(image_dy, image_dy);
    const float determinant = xx * yy - xy * xy;
    if (!(determinant > 1.0e-20f)) {
      return false;
    }
    const float rx = dot(image_delta, image_dx);
    const float ry = dot(image_delta, image_dy);
    *raster = make_float3(
        (rx * yy - ry * xy) / determinant, (ry * xx - rx * xy) / determinant, 0.0f);
    *sensor_P = transform_point(&camera_to_world, lens_P);

    const float3 image_P = bdpt_perspective_image_point(kg, *raster, vertex_time);
    const float3 image_X = bdpt_perspective_image_point(
        kg, make_float3(raster->x + 1.0f, raster->y, raster->z), vertex_time);
    const float3 image_Y = bdpt_perspective_image_point(
        kg, make_float3(raster->x, raster->y + 1.0f, raster->z), vertex_time);
    const float3 sensor_plane_P = use_dof ?
                                      image_P * (kernel_data.cam.focaldistance / image_P.z) :
                                      image_P;
    const float3 sensor_plane_X = use_dof ?
                                      image_X * (kernel_data.cam.focaldistance / image_X.z) :
                                      image_X;
    const float3 sensor_plane_Y = use_dof ?
                                      image_Y * (kernel_data.cam.focaldistance / image_Y.z) :
                                      image_Y;
    const float image_pixel_area = len(
        cross(sensor_plane_X - sensor_plane_P, sensor_plane_Y - sensor_plane_P));
    const float3 sensor_to_plane = sensor_plane_P - lens_P;
    const float image_distance2 = len_squared(sensor_to_plane);
    const float cos_at_camera = fabsf(sensor_to_plane.z) /
                                sqrtf(max(image_distance2, 1.0e-20f));
    const float distance2 = len_squared(*sensor_P - light_vertex->P);
    if (!(image_pixel_area > 0.0f) || !(cos_at_camera > 0.0f) || !(distance2 > 1.0e-12f)) {
      return false;
    }
    *connection_jacobian = image_distance2 /
                           (cos_at_camera * image_pixel_area * distance2);
  }
  else {
    /* Panorama DOF uses a direction-dependent aperture plane, and built-in panorama/orthographic
     * motion uses a decomposed transform representation. Keep those uncommon combinations on the
     * regular path tracer until an exact inverse is available. */
    if (kernel_data.cam.aperturesize > 0.0f || kernel_data.cam.num_motion_steps != 0) {
      return false;
    }

    const float3 ndc = camera_world_to_ndc(kg, light_sd, light_vertex->P);
    *raster = make_float3(
        ndc.x * kernel_data.cam.width, ndc.y * kernel_data.cam.height, 0.0f);
    const Transform camera_to_world = kernel_data.cam.cameratoworld;

    if (camera_type == CAMERA_ORTHOGRAPHIC) {
      const ProjectionTransform raster_to_camera = kernel_data.cam.rastertocamera;
      const float3 image_P = transform_perspective(&raster_to_camera, *raster);
      const float3 image_X = transform_perspective(
          &raster_to_camera, make_float3(raster->x + 1.0f, raster->y, raster->z));
      const float3 image_Y = transform_perspective(
          &raster_to_camera, make_float3(raster->x, raster->y + 1.0f, raster->z));
      const float image_pixel_area = len(cross(image_X - image_P, image_Y - image_P));
      if (!(image_pixel_area > 0.0f)) {
        return false;
      }
      *sensor_P = transform_point(&camera_to_world, image_P);
      const Transform world_to_camera = kernel_data.cam.worldtocamera;
      const float camera_distance = transform_point(&world_to_camera, light_vertex->P).z -
                                    image_P.z;
      if (camera_distance <= kernel_data.cam.nearclip ||
          camera_distance >= kernel_data.cam.nearclip + kernel_data.cam.cliplength)
      {
        return false;
      }
      *connection_jacobian = 1.0f / image_pixel_area;
    }
    else if (camera_type == CAMERA_PANORAMA) {
      const float camera_distance = len(light_vertex->P -
                                        transform_point(&camera_to_world, zero_float3()));
      if (camera_distance <= kernel_data.cam.nearclip ||
          camera_distance >= kernel_data.cam.nearclip + kernel_data.cam.cliplength)
      {
        return false;
      }
      constexpr float h = 0.01f;
      const float3 D = camera_panorama_direction(&kernel_data.cam, raster->x, raster->y);
      const float3 Dx0 = camera_panorama_direction(
          &kernel_data.cam, raster->x - h, raster->y);
      const float3 Dx1 = camera_panorama_direction(
          &kernel_data.cam, raster->x + h, raster->y);
      const float3 Dy0 = camera_panorama_direction(
          &kernel_data.cam, raster->x, raster->y - h);
      const float3 Dy1 = camera_panorama_direction(
          &kernel_data.cam, raster->x, raster->y + h);
      if (is_zero(D) || is_zero(Dx0) || is_zero(Dx1) || is_zero(Dy0) || is_zero(Dy1)) {
        return false;
      }
      const float3 dDdx = (Dx1 - Dx0) * (0.5f / h);
      const float3 dDdy = (Dy1 - Dy0) * (0.5f / h);
      const float solid_angle_per_pixel = len(cross(dDdx, dDdy));
      const float distance2 = len_squared(camera_position(kg) - light_vertex->P);
      if (!(solid_angle_per_pixel > 0.0f) || !(distance2 > 1.0e-12f)) {
        return false;
      }
      *sensor_P = camera_position(kg);
      *connection_jacobian = 1.0f / (solid_angle_per_pixel * distance2);
    }
    else {
      return false;
    }
  }

  return raster->x >= 0.0f && raster->y >= 0.0f && raster->x < kernel_data.cam.width &&
         raster->y < kernel_data.cam.height && *connection_jacobian > 0.0f;
}

/* Invert a perspective camera ray after a manifold walk. The manifold contribution is expressed
 * per unit camera solid angle, so this returns only the sensor's solid-angle-to-pixel Jacobian
 * (the endpoint-to-surface geometry is already in the manifold transfer determinant). */
ccl_device_inline bool bdpt_perspective_ray_to_raster(KernelGlobals kg,
                                                      const float3 sensor_P,
                                                      const float3 camera_wo,
                                                      const float time,
                                                      ccl_private float3 *raster,
                                                      ccl_private float *sensor_jacobian)
{
  if (CameraType(kernel_data.cam.type) != CAMERA_PERSPECTIVE) {
    return false;
  }
  Transform camera_to_world = kernel_data.cam.cameratoworld;
  if (kernel_data.cam.num_motion_steps) {
    transform_motion_array_interpolate(
        &camera_to_world, kernel_data_array(camera_motion), kernel_data.cam.num_motion_steps, time);
  }
  const Transform world_to_camera = transform_inverse(camera_to_world);
  const float3 lens_P = transform_point(&world_to_camera, sensor_P);
  const float3 camera_D = transform_direction(&world_to_camera, camera_wo);
  if (!(camera_D.z > 1.0e-8f)) {
    return false;
  }

  float3 projection_camera = lens_P + camera_D;
  if (kernel_data.cam.aperturesize > 0.0f) {
    projection_camera = lens_P + camera_D * (kernel_data.cam.focaldistance / camera_D.z);
  }
  const float3 image_origin = bdpt_perspective_image_point(kg, zero_float3(), time);
  const float3 image_dx = bdpt_perspective_image_point(kg, make_float3(1.0f, 0.0f, 0.0f), time) -
                          image_origin;
  const float3 image_dy = bdpt_perspective_image_point(kg, make_float3(0.0f, 1.0f, 0.0f), time) -
                          image_origin;
  const float3 projected_on_image = projection_camera *
                                    (image_origin.z / projection_camera.z);
  const float3 image_delta = projected_on_image - image_origin;
  const float xx = dot(image_dx, image_dx);
  const float xy = dot(image_dx, image_dy);
  const float yy = dot(image_dy, image_dy);
  const float determinant = xx * yy - xy * xy;
  if (!(determinant > 1.0e-20f)) {
    return false;
  }
  const float rx = dot(image_delta, image_dx);
  const float ry = dot(image_delta, image_dy);
  *raster = make_float3(
      (rx * yy - ry * xy) / determinant, (ry * xx - rx * xy) / determinant, 0.0f);

  const float3 image_P = bdpt_perspective_image_point(kg, *raster, time);
  const float3 image_X = bdpt_perspective_image_point(
      kg, make_float3(raster->x + 1.0f, raster->y, raster->z), time);
  const float3 image_Y = bdpt_perspective_image_point(
      kg, make_float3(raster->x, raster->y + 1.0f, raster->z), time);
  const bool use_dof = kernel_data.cam.aperturesize > 0.0f;
  const float focus_scale = use_dof ? kernel_data.cam.focaldistance / image_P.z : 1.0f;
  const float3 sensor_plane_P = image_P * focus_scale;
  const float3 sensor_plane_X = use_dof ? image_X * (kernel_data.cam.focaldistance / image_X.z) :
                                         image_X;
  const float3 sensor_plane_Y = use_dof ? image_Y * (kernel_data.cam.focaldistance / image_Y.z) :
                                         image_Y;
  const float image_pixel_area = len(
      cross(sensor_plane_X - sensor_plane_P, sensor_plane_Y - sensor_plane_P));
  const float3 sensor_to_plane = sensor_plane_P - lens_P;
  const float image_distance2 = len_squared(sensor_to_plane);
  const float cos_at_camera = fabsf(sensor_to_plane.z) /
                              sqrtf(max(image_distance2, 1.0e-20f));
  if (!(image_pixel_area > 0.0f) || !(cos_at_camera > 0.0f)) {
    return false;
  }
  *sensor_jacobian = image_distance2 / (cos_at_camera * image_pixel_area);
  return raster->x >= 0.0f && raster->y >= 0.0f && raster->x < kernel_data.cam.width &&
         raster->y < kernel_data.cam.height;
}

/* A deterministic strategy partition avoids assigning ordinary straight-connection PDFs
 * to a refracted sensor connection. Only retire camera paths whose complete delta
 * prefix the same solver can reproduce. Failed/unsupported manifolds keep camera transport. */
ccl_device_inline_transport bool bdpt_volume_sensor_prefix(KernelGlobals kg,
                                                           IntegratorState state,
                                                           ccl_private ShaderData *sd,
                                                           const float3 P)
{
#if defined(__MNEE__) && defined(__VOLUME__)
  const uint flag = INTEGRATOR_STATE(state, path, flag);
  const int bounce = INTEGRATOR_STATE(state, path, bounce);
  if (!kernel_data.integrator.use_bidirectional_path_tracing ||
      !(flag & PATH_RAY_BDPT_VOLUME_SENSOR) || (flag & PATH_RAY_BDPT_UNSUPPORTED) ||
      INTEGRATOR_STATE(state, path, volume_bounce) != 0 || bounce == 0 ||
      CameraType(kernel_data.cam.type) != CAMERA_PERSPECTIVE)
  {
    return false;
  }
  if (kernel_data.kernel_features & KERNEL_FEATURE_SHADOW_LINKING) {
    return false;
  }
  KernelBDPTVertex target = {};
  target.P = P;
  target.type = PRIMITIVE_VOLUME;
  target.object = sd->object;
  target.time_wavelength = photon_pack_time_wavelength(sd->time, 0.0f, false);
  const float3 lens_rand = path_rng_3D(kg,
                                       INTEGRATOR_STATE(state, path, rng_pixel),
                                       INTEGRATOR_STATE(state, path, sample),
                                       PRNG_LENS_TIME);
  float3 raster, sensor_P;
  float jacobian;
  if (!bdpt_sample_camera_endpoint(
          kg, &target, sd, make_float2(lens_rand.y, lens_rand.z), &raster, &sensor_P, &jacobian))
  {
    return false;
  }
  ShaderDataTinyStorage camera_storage;
  ccl_private ShaderData *camera_sd = AS_SHADER_DATA(&camera_storage);
  camera_sd->P = sensor_P;
  camera_sd->N = camera_sd->Ng = normalize(P - sensor_P);
  camera_sd->object = OBJECT_NONE;
  camera_sd->prim = PRIM_NONE;
  camera_sd->time = sd->time;
#  ifdef __RAY_DIFFERENTIALS__
  camera_sd->dP = 0.0f;
#  endif
  LightSample ls ccl_optional_struct_init;
  ls.P = P;
  ls.Ng = sd->wi;
  ls.D = normalize_len(P - sensor_P, &ls.t);
  ls.pdf = ls.pdf_selection = ls.eval_fac = 1.0f;
  ls.object = sd->object;
  ls.prim = PRIM_NONE;
  ls.shader = sd->shader;
  ls.group = LIGHTGROUP_NONE;
  ls.type = LIGHT_TRIANGLE;
  ls.emitter_id = EMITTER_NONE;
  ShaderDataCausticsStorage manifold_storage;
  ccl_private ShaderData *manifold_sd = AS_SHADER_DATA(&manifold_storage);
  RNGState rng;
  path_state_rng_load(state, &rng);
  Spectrum throughput;
  float3 camera_wo, light_wo;
  float distance;
  int vertices = 0;
  const int transmission = INTEGRATOR_STATE(state, path, transmission_bounce);
  const int diffuse = INTEGRATOR_STATE(state, path, diffuse_bounce);
  const auto mnee = INTEGRATOR_STATE(state, path, mnee);
  INTEGRATOR_STATE_WRITE(state, path, bounce) = 0;
  INTEGRATOR_STATE_WRITE(state, path, transmission_bounce) = 0;
  INTEGRATOR_STATE_WRITE(state, path, diffuse_bounce) = 0;
#  ifdef __SPECTRAL__
  /* Volume shader setup does not initialize rand_wavelength. Let MNEE recover the
   * immutable camera-path wavelength for each interface that requires it. */
  const float wavelength_rand = -1.0f;
#  else
  const float wavelength_rand = -1.0f;
#  endif
  const ShaderEvalResult result = kernel_path_mnee_sample(kg,
                                                          state,
                                                          camera_sd,
                                                          manifold_sd,
                                                          &rng,
                                                          &ls,
                                                          &throughput,
                                                          &camera_wo,
                                                          vertices,
                                                          &light_wo,
                                                          true,
                                                          &distance,
                                                          wavelength_rand,
                                                          true);
  INTEGRATOR_STATE_WRITE(state, path, bounce) = bounce;
  INTEGRATOR_STATE_WRITE(state, path, transmission_bounce) = transmission;
  INTEGRATOR_STATE_WRITE(state, path, diffuse_bounce) = diffuse;
  INTEGRATOR_STATE_WRITE(state, path, mnee) = mnee;
  if (result == SHADER_EVAL_CACHE_MISS) {
    sd->runtime_flag |= SR_CACHE_MISS;
  }
  return result == SHADER_EVAL_OK && vertices == bounce && dot(light_wo, sd->wi) > 1.0f - 1.0e-6f;
#else
  return false;
#endif
}

ccl_device_inline bool bdpt_volume_sensor_owns_camera_path(KernelGlobals kg,
                                                           ConstIntegratorState state,
                                                           const int additional_bounces = 0,
                                                           const float max_light_bounces = FLT_MAX)
{
  if (!kernel_data.integrator.use_bidirectional_path_tracing ||
      !(INTEGRATOR_STATE(state, path, flag) & PATH_RAY_BDPT_VOLUME_SENSOR) ||
      INTEGRATOR_STATE(state, path, volume_bounce) != 1)
  {
    return false;
  }
  const int light_bounces = INTEGRATOR_STATE(state, path, bounce) -
                            INTEGRATOR_STATE(state, path, bdpt_volume_bounce) + additional_bounces;
  return light_bounces >= 0 && light_bounces < kernel_data.integrator.bdpt_max_bounces &&
         float(light_bounces) <= max_light_bounces;
}

ccl_device_inline bool bdpt_volume_sensor_supports_light(KernelGlobals kg,
                                                         const LightType type,
                                                         const int prim)
{
  if (type == LIGHT_TRIANGLE || type == LIGHT_BACKGROUND) {
    return true;
  }
  const ccl_global KernelLight *light = &kernel_data_fetch(lights, prim);
  return !((type == LIGHT_POINT || type == LIGHT_SPOT) && !light->spot.is_sphere &&
           light->spot.radius > 0.0f);
}

/* Splat one reservoir-selected light vertex onto a built-in sensor. */
ccl_device_inline void bdpt_connect_light_vertex_to_camera(
    KernelGlobals kg,
    IntegratorState state,
    const ccl_private KernelBDPTVertex *light_vertex,
    ccl_private ShaderData *light_sd,
    const ccl_private PolarizationSpectrumState &light_polarization,
    const uint candidate_count,
    const uint iteration,
    const uint batch_samples,
    ccl_global float *render_buffer,
    const float2 rand_lens)
{
  if (candidate_count == 0) {
    return;
  }
  float3 raster;
  float3 sensor_P;
  float connection_jacobian;
  if (!bdpt_sample_camera_endpoint(
          kg, light_vertex, light_sd, rand_lens, &raster, &sensor_P, &connection_jacobian))
  {
    return;
  }
  float3 delta = sensor_P - light_vertex->P;
  float distance2 = len_squared(delta);
  if (!(distance2 > 1.0e-12f)) {
    return;
  }
  float distance = sqrtf(distance2);
  float3 direction = delta / distance;
  const bool volume_vertex = light_vertex->type == PRIMITIVE_VOLUME;
  PolarizationMueller manifold_polarization{};
  for (int i=0;i<4;i++) manifold_polarization.value[i][i]=1;
  Spectrum manifold_throughput = one_spectrum();
  bool manifold_connection = false;

#ifdef __MNEE__
  /* A cached diffuse light vertex viewed through one or more specular interfaces is the SDS
   * transport class that ordinary BDPT sensor splats cannot connect. Walk the same exact
   * specular manifold used by Cycles MNEE, but in reverse: camera -> interfaces -> cached light
   * vertex. This yields both the physically valid endpoint direction and its transfer Jacobian. */
  if ((kernel_data.kernel_features & KERNEL_FEATURE_MNEE) &&
      CameraType(kernel_data.cam.type) == CAMERA_PERSPECTIVE &&
      (!volume_vertex || !(kernel_data.kernel_features & KERNEL_FEATURE_SHADOW_LINKING)))
  {
    ShaderDataTinyStorage camera_sd_storage;
    ccl_private ShaderData *camera_sd = AS_SHADER_DATA(&camera_sd_storage);
    camera_sd->P = sensor_P;
    camera_sd->N = -direction;
    camera_sd->Ng = -direction;
    camera_sd->object = OBJECT_NONE;
    camera_sd->prim = PRIM_NONE;
    camera_sd->time = photon_unpack_time(light_vertex->time_wavelength);
#  ifdef __RAY_DIFFERENTIALS__
    camera_sd->dP = 0.0f;
#  endif

    LightSample sensor_target ccl_optional_struct_init;
    sensor_target.P = light_vertex->P;
    sensor_target.Ng = volume_vertex ? direction : light_sd->N;
    sensor_target.t = distance;
    sensor_target.D = -direction;
    sensor_target.pdf = 1.0f;
    sensor_target.pdf_selection = 1.0f;
    sensor_target.eval_fac = 1.0f;
    sensor_target.object = light_vertex->object;
    sensor_target.prim = volume_vertex ? PRIM_NONE : light_vertex->prim;
    sensor_target.shader = light_sd->shader;
    sensor_target.group = bdpt_vertex_light_group(light_vertex->light_group);
    sensor_target.type = LIGHT_TRIANGLE;
    sensor_target.emitter_id = EMITTER_NONE;

#  ifdef __NESTED_DIELECTRICS__
    if (nested_dielectrics_enabled(kg)) {
      /* The chain starts at the camera, inside the nested dielectric media around it. This
       * state last held the media of another connection. */
      if (kernel_data.cam.is_inside_volume) {
        Ray camera_ray ccl_optional_struct_init;
        camera_ray.P = sensor_P;
        camera_ray.D = direction;
        camera_ray.tmin = 0.0f;
        camera_ray.tmax = FLT_MAX;
        camera_ray.time = camera_sd->time;
        camera_ray.self.object = OBJECT_NONE;
        camera_ray.self.prim = PRIM_NONE;
        camera_ray.self.light_object = OBJECT_NONE;
        camera_ray.self.light_prim = PRIM_NONE;
#    ifdef __RAY_DIFFERENTIALS__
        camera_ray.dP = differential_zero_compact();
        camera_ray.dD = differential_zero_compact();
#    endif
        integrator_state_write_ray(state, &camera_ray);
        integrator_volume_stack_init(kg, state, PATH_RAY_VISIBILITY_CAMERA);
      }
      else {
        integrator_state_medium_stack_clear(kg, state);
      }
    }
#  endif

    ShaderDataCausticsStorage manifold_sd_storage;
    ccl_private ShaderData *manifold_sd = AS_SHADER_DATA(&manifold_sd_storage);
    RNGState rng_state;
    path_state_rng_load(state, &rng_state);
    Spectrum candidate_throughput = zero_spectrum();
    float3 camera_wo = zero_float3();
    float3 light_wo = zero_float3();
    float light_distance = 0.0f;
    int manifold_vertex_count = 0;
    float3 manifold_vertices[MNEE_MAX_CAUSTIC_CASTERS];
    const ShaderEvalResult manifold_result = kernel_path_mnee_sample(kg,
                                                                     state,
                                                                     camera_sd,
                                                                     manifold_sd,
                                                                     &rng_state,
                                                                     &sensor_target,
                                                                     &candidate_throughput,
                                                                     &camera_wo,
                                                                     manifold_vertex_count,
                                                                     &light_wo,
                                                                     true,
                                                                     &light_distance,
        photon_unpack_wavelength_rand(light_vertex->time_wavelength),
        volume_vertex,
        volume_vertex ? manifold_vertices : nullptr,
        polarization_enabled(kg) ? &manifold_polarization : nullptr);
    if (manifold_result == SHADER_EVAL_CACHE_MISS) {
      light_sd->runtime_flag |= SR_CACHE_MISS;
      return;
    }
    float3 manifold_raster;
    float sensor_jacobian;
    if (manifold_vertex_count > 0 && isfinite_safe(candidate_throughput) &&
        bdpt_perspective_ray_to_raster(
            kg, sensor_P, camera_wo, camera_sd->time, &manifold_raster, &sensor_jacobian))
    {
      /* The manifold routine validates camera-to-interface segments. Check the final free segment
       * from the cached endpoint back to the last interface; the hit at its upper bound is the
       * intended manifold vertex, while anything earlier is a true blocker. */
      Ray verify_ray ccl_optional_struct_init;
      bool verify_skip_self = !volume_vertex;
      verify_ray.P = volume_vertex ? float3(light_vertex->P) :
                                     shadow_ray_offset(kg, light_sd, light_wo, &verify_skip_self);
      verify_ray.D = light_wo;
      verify_ray.tmin = 0.0f;
      verify_ray.tmax = light_distance;
      verify_ray.time = camera_sd->time;
      verify_ray.self.object = verify_skip_self ? light_sd->object : OBJECT_NONE;
      verify_ray.self.prim = verify_skip_self ? light_sd->prim : PRIM_NONE;
      verify_ray.self.light_object = OBJECT_NONE;
      verify_ray.self.light_prim = PRIM_NONE;
#  ifdef __RAY_DIFFERENTIALS__
      verify_ray.dP = differential_zero_compact();
      verify_ray.dD = differential_zero_compact();
#  endif
      Intersection verify_isect;
      const bool early_blocker =
          !volume_vertex &&
          scene_intersect(kg, &verify_ray, PATH_RAY_VISIBILITY_TRANSMIT, &verify_isect) &&
                                 verify_isect.t < light_distance - MNEE_MIN_DISTANCE;
      if (!early_blocker) {
#  ifdef __VOLUME__
        if (volume_vertex) {
          float3 segment_start = sensor_P;
          for (int segment = 0; segment <= manifold_vertex_count; segment++) {
            const float3 segment_end = segment == manifold_vertex_count ?
                                           float3(light_vertex->P) :
                                           manifold_vertices[segment];
            if (!bdpt_volume_connection_transmittance(
                    kg,
                    state,
                    segment_start,
                    segment_end,
                    camera_sd->time,
                    photon_unpack_wavelength_rand(light_vertex->time_wavelength),
                    uint(segment),
                    &candidate_throughput))
            {
              light_sd->runtime_flag |= SR_CACHE_MISS;
              return;
            }
            segment_start = segment_end;
          }
          /* The helper uses the dedicated sensor state's stack as scratch. Restore the
           * endpoint medium before evaluating its reciprocal phase function. */
          Ray endpoint_ray = verify_ray;
          endpoint_ray.P = light_vertex->P;
          integrator_state_write_ray(state, &endpoint_ray);
          integrator_volume_stack_init(kg, state, PATH_RAY_VISIBILITY_CAMERA);
        }
#  endif
        raster = manifold_raster;
        direction = light_wo;
        connection_jacobian = sensor_jacobian;
        manifold_throughput = candidate_throughput;
        manifold_connection = true;
      }
    }
  }
#endif

  const int pixel_x = int(raster.x);
  const int pixel_y = int(raster.y);
  const int buffer_min_x = kernel_integrator_state.bdpt_buffer_full_x;
  const int buffer_min_y = kernel_integrator_state.bdpt_buffer_full_y;
  if (pixel_x < buffer_min_x ||
      pixel_x >= buffer_min_x + kernel_integrator_state.bdpt_buffer_width ||
      pixel_y < buffer_min_y ||
      pixel_y >= buffer_min_y + kernel_integrator_state.bdpt_buffer_height)
  {
    return;
  }

  const uint render_pixel_index = uint(kernel_integrator_state.bdpt_buffer_offset + pixel_x +
                                       pixel_y * kernel_integrator_state.bdpt_buffer_stride);
  INTEGRATOR_STATE_WRITE(state, path, render_pixel_index) = render_pixel_index;
  /* A cache is generated before the camera work for its batch. Adaptive convergence from earlier
   * batches is already known at this point: do not splat B samples into a pixel that will schedule
   * none, otherwise film normalization by its smaller sample count produces a bright bias. */
  if (render_buffer != nullptr && !film_need_sample_pixel(kg, state, render_buffer)) {
    return;
  }

  /* Shader graphs and some layered closures build direction-dependent data from ShaderData::wi.
   * A light subpath prepared this shader with the direction toward the emitter. For a sensor
   * connection evaluate the reciprocal endpoint exactly as a camera path would: make the camera
   * direction fixed and evaluate toward the preceding light vertex. Merely swapping wi inside
   * bsdf_eval is insufficient for Principled and user graphs using the Incoming socket. The
   * caller restores the light-oriented closures before continuing the light subpath. */
  const float3 light_incoming = light_sd->wi;
  Ray camera_ray ccl_optional_struct_init;
  camera_ray.P = light_vertex->P + direction;
  camera_ray.D = -direction;
  camera_ray.tmin = 0.0f;
  camera_ray.tmax = 1.0f;
  camera_ray.time = photon_unpack_time(light_vertex->time_wavelength);
#ifdef __RAY_DIFFERENTIALS__
  camera_ray.dP = differential_zero_compact();
  camera_ray.dD = differential_zero_compact();
#endif
  BsdfEval light_eval;
  float light_pdf = 0.0f;
  if (volume_vertex) {
#ifdef __VOLUME__
    camera_ray.P = light_vertex->P;
    camera_ray.tmin = 0.0f;
    shader_setup_from_volume(light_sd, &camera_ray, light_vertex->object);
    light_sd->P = light_vertex->P;
#  ifdef __SPECTRAL__
    light_sd->rand_wavelength = photon_unpack_wavelength_rand(light_vertex->time_wavelength);
#  endif
    VolumeShaderCoefficients coeff;
    if (!volume_shader_sample(kg, state, light_sd, &coeff) ||
        !(light_sd->runtime_flag & SR_SCATTER) || (light_sd->runtime_flag & SR_CACHE_MISS))
    {
      return;
    }
    ShaderVolumePhases phases;
    volume_shader_copy_phases(&phases, light_sd);
    light_pdf = volume_shader_phase_eval(
        kg, state, light_sd, &phases, light_incoming, &light_eval, SHADER_USE_MIS);
#else
    return;
#endif
  }
  else {
    Intersection camera_isect;
    camera_isect.t = 1.0f;
    camera_isect.u = light_vertex->u;
    camera_isect.v = light_vertex->v;
    camera_isect.prim = light_vertex->prim;
    camera_isect.object = light_vertex->object;
    camera_isect.type = light_vertex->type;
    shader_setup_from_ray(kg, light_sd, &camera_ray, &camera_isect);
#ifdef __SPECTRAL__
    light_sd->rand_wavelength = photon_unpack_wavelength_rand(light_vertex->time_wavelength);
#endif
    bdpt_vertex_restore_medium(light_vertex->light_group, light_sd);
    surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE>(kg,
                                                          state,
                                                          light_sd,
                                                          nullptr,
                                                          PATH_RAY_VISIBILITY_CAMERA,
                                                          PATH_RAY_MIS_SKIP |
                                                              PATH_RAY_TRANSPARENT_BACKGROUND);
    if (light_sd->runtime_flag & SR_CACHE_MISS) {
      return;
    }
    surface_shader_prepare_closures(kg, state, light_sd, PATH_RAY_VISIBILITY_CAMERA);

    float roughness_squared = 0.0f;
    const uint emitter_shader_flags = (bdpt_vertex_path_length(light_vertex) == 2u) ?
                                          (light_vertex->emitter_shader_flags | SHADER_USE_MIS) :
                                          SHADER_USE_MIS;
    light_pdf = surface_shader_bsdf_eval(kg,
                                         state,
                                         light_sd,
                                         light_incoming,
                                         &light_eval,
                                         emitter_shader_flags,
                                         roughness_squared);
  }
  if (!(light_pdf > 0.0f) || bsdf_eval_is_zero(&light_eval)) {
    return;
  }
  /* In the reciprocal orientation this is exactly the camera-to-light forward density, which is
   * the reverse density required by the light-side recursive MIS term. */
  const float reverse_pdf = light_pdf;

  const float cos_at_surface = volume_vertex ? 1.0f :
                                               max(fabsf(dot(light_sd->Ng, direction)), 1.0e-8f);
  const float camera_pdf_area = connection_jacobian * cos_at_surface;

  const float light_path_count = float(kernel_integrator_state.bdpt_light_path_count);
  const float light_path_sample_ratio = max(
      kernel_integrator_state.bdpt_light_path_sample_ratio, 1.0e-20f);
  float3 selection_P = light_sd->P;
  float3 selection_N = volume_vertex ? zero_float3() : light_sd->N;
  float selection_dt = 0.0f;
#ifdef __VOLUME__
  if (volume_vertex && bdpt_vertex_path_length(light_vertex) == 2u &&
      kernel_data.integrator.use_light_tree)
  {
    /* Volume NEE selects an emitter over the entire camera ray segment, before drawing a
     * collision distance. Stopping that segment at the cached collision gives a different tree
     * PDF. Recover both boundaries, including transparent interfaces before this vertex. An
     * opaque interface makes the eventual shadow contribution zero, so it need not be shaded
     * here. The visibility pass still performs all material/transmittance evaluations. */
    Ray segment_ray ccl_optional_struct_init;
    segment_ray.P = sensor_P;
    segment_ray.D = -direction;
    segment_ray.tmin = 0.0f;
    segment_ray.tmax = FLT_MAX;
    segment_ray.time = photon_unpack_time(light_vertex->time_wavelength);
#  ifdef __RAY_DIFFERENTIALS__
    segment_ray.dP = differential_zero_compact();
    segment_ray.dD = differential_zero_compact();
#  endif
    segment_ray.self.object = OBJECT_NONE;
    segment_ray.self.prim = PRIM_NONE;
    segment_ray.self.light_object = OBJECT_NONE;
    segment_ray.self.light_prim = PRIM_NONE;
    float segment_start = 0.0f;
    float segment_end = FLT_MAX;
    for (int boundary = 0; boundary <= kernel_data.integrator.transparent_max_bounce; boundary++) {
      Intersection segment_isect;
      if (!scene_intersect(kg, &segment_ray, PATH_RAY_VISIBILITY_CAMERA, &segment_isect)) {
        break;
      }
      if (segment_isect.t >= distance) {
        segment_end = segment_isect.t;
        break;
      }
      segment_start = segment_isect.t;
      segment_ray.tmin = intersection_t_offset(segment_isect.t);
    }
    selection_P = sensor_P + segment_start * segment_ray.D;
    selection_N = segment_ray.D;
    selection_dt = segment_end - segment_start;
  }
#endif
  const float selection_ratio = bdpt_vertex_path_length(light_vertex) == 2u ?
      bdpt_light_selection_ratio(kg,
                                 light_vertex->emitter_distribution,
                                 light_vertex->emitter_P,
                                 light_sd->P,
                                 light_sd->time,
                                 selection_P,
                                 selection_N,
                                 volume_vertex ? SR_BSDF_HAS_TRANSMISSION : light_sd->runtime_flag,
                                 light_sd->object,
                                 volume_vertex,
                                 selection_dt) : 1.0f;
  /* A surface vertex that is kept for merging can also be merged with the camera vertex next
   * to it. The camera path that does has its first vertex here, which sets the merge radius. */
  const BDPTMISWeight eta = vcm_mis_eta(
      kg, vcm_merge_radius(kg, vcm_camera_distance(kg, light_sd->P)));
  const bool light_vertex_kept = !volume_vertex && vcm_merging_enabled(kg) &&
                                 (vcm_keep_all(kg) ||
                                  (light_vertex->sharp_prefix != 0u &&
                                   bdpt_vertex_path_length(light_vertex) > 2u));
  const BDPTMISWeight w_light = (BDPTMISWeight(camera_pdf_area) / light_path_sample_ratio) *
                                (BDPTMISWeight::from_encoded(light_vertex->d_vcm) *
                                     selection_ratio +
                                 (BDPTMISWeight::from_encoded(light_vertex->d_vc) +
                                  BDPTMISWeight::from_encoded(light_vertex->d_vm) * eta) *
                                     reverse_pdf +
                                 (light_vertex_kept ? eta : BDPTMISWeight(0.0f)));
  /* A refracted medium sensor connection replaces the matching camera strategy.
   * Its prefix is replayed at camera medium vertices before suppressing anything.
   * Ordinary straight sensor connections retain their existing MIS partition. */
  float mis_weight = (volume_vertex && manifold_connection) ?
                         1.0f :
                         (BDPTMISWeight(1.0f) + w_light).inverse();
  if (kernel_data.integrator.use_vertex_merging && volume_vertex && !manifold_connection &&
      bdpt_vertex_path_length(light_vertex) > 2u)
  {
    /* The camera path leaves the recursion at its first medium vertex and samples the rest of
     * such a path with the weights of the regular path tracer, which sum to one. The light
     * subpath only adds what that cannot sample: a delta emitter behind a delta event. Without
     * merging the weight stays as it was. */
    if (light_vertex->u != 0.0f) {
      return;
    }
    mis_weight = 1.0f;
  }

  /* Transpose the reciprocal camera evaluation in geometric projected-area measure.
   * One cached map serves a camera batch, so its splat represents every sample in that batch. */
  const Spectrum sensor_eval = volume_vertex ?
                                   bsdf_eval_sum(&light_eval) :
                                   bdpt_transpose_surface_eval(bsdf_eval_sum(&light_eval),
                                                               light_sd->Ng,
                                                               light_incoming,
                                                               direction);
  const float normalization = float(candidate_count) * float(batch_samples) /
                              max(light_path_count, 1.0f);
  const Spectrum spectral_weight = bdpt_light_vertex_spectral_weight(
      kg, state, light_vertex->time_wavelength, false);
  Spectrum contribution = Spectrum(light_vertex->throughput) * spectral_weight *
                                manifold_throughput * sensor_eval *
                                (mis_weight * normalization * connection_jacobian);
  if (polarization_enabled(kg)) {
    const auto endpoint_sensitivity = polarization_spectrum_apply(manifold_polarization,
        polarization_unpolarized(), true);
    const auto sensitivity = volume_vertex ? polarization_depolarized(endpoint_sensitivity) :
        polarization_surface_transport(kg, light_sd, light_incoming,
            endpoint_sensitivity, true, nullptr, bsdf_eval_sum(&light_eval), false);
    contribution *= polarization_spectrum_contract(sensitivity, light_polarization);
  }
  if (!isfinite_safe(contribution) || is_zero(contribution)) {
    return;
  }

  Ray shadow_ray ccl_optional_struct_init;
  bool skip_self = !volume_vertex;
  shadow_ray.P = volume_vertex ? float3(light_vertex->P) :
                                 shadow_ray_offset(kg, light_sd, direction, &skip_self);
  shadow_ray.D = direction;
  shadow_ray.tmin = 0.0f;
  /* Manifold visibility was validated explicitly above. Use a tiny terminal segment so the
   * ordinary shadow kernel performs film/pass accumulation without incorrectly treating the
   * refractive interfaces as opaque blockers. */
  shadow_ray.tmax = manifold_connection ? 1.0e-6f : distance;
  shadow_ray.time = photon_unpack_time(light_vertex->time_wavelength);
  shadow_ray.self.object = skip_self ? light_sd->object : OBJECT_NONE;
  shadow_ray.self.prim = skip_self ? light_sd->prim : PRIM_NONE;
  shadow_ray.self.light_object = light_vertex->emitter_object;
  shadow_ray.self.light_prim = PRIM_NONE;
#ifdef __RAY_DIFFERENTIALS__
  shadow_ray.dP = differential_zero_compact();
  shadow_ray.dD = differential_zero_compact();
#endif

#ifdef __VOLUME__
  if (!volume_vertex &&
      (kernel_data.integrator.use_volumes || kernel_data.integrator.use_nested_dielectrics))
  {
    /* Dedicated sensor work reuses path storage and does not inherit the cached
     * light path's medium stack. A surface can itself lie inside another object's
     * volume. Reconstruct at the outgoing, offset shadow origin so interfaces
     * select the correct side as well; never copy an unrelated path's stack. */
    integrator_state_write_ray(state, &shadow_ray);
    integrator_volume_stack_init(kg, state, PATH_RAY_VISIBILITY_CAMERA);
  }
#endif

  IntegratorShadowState shadow_state = integrator_shadow_path_init(
      kg, state, DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW, false);
#ifdef __VOLUME__
  integrator_state_copy_volume_stack_to_shadow(kg, shadow_state, state);
#endif
#ifdef __NESTED_DIELECTRICS__
  integrator_state_copy_medium_stack_to_shadow(kg, shadow_state, state);
#endif
  integrator_state_write_shadow_ray(shadow_state, &shadow_ray);
  integrator_state_write_shadow_ray_self(shadow_state, &shadow_ray);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, render_pixel_index) = render_pixel_index;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, sample) = iteration;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, rng_pixel) = hash_uint2(
      uint(pixel_x), uint(pixel_y));
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, rng_offset) = 0;
  INTEGRATOR_STATE_WRITE(
      shadow_state, shadow_path, transparent_bounce) = light_vertex->path_length >> 20u;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, volume_bounds_bounce) = 0;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, diffuse_bounce) = 1;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, glossy_bounce) = 0;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, transmission_bounce) = 0;
  /* path_length includes both endpoints. A first light-side scattering vertex is the camera
   * path's bounce zero and belongs in the direct pass. */
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, bounce) =
      bdpt_vertex_path_length(light_vertex) - 2u;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, throughput) = contribution;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, lightgroup) = bdpt_vertex_light_group(light_vertex->light_group) + 1;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, visibility) = PATH_RAY_VISIBILITY_CAMERA;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, flag) = volume_vertex ?
                                                                PATH_RAY_VOLUME_PASS :
                                                                PATH_RAY_SURFACE_PASS;
  if (!(kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_TREE)) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, bsdf_eval_average) = average(
        bsdf_eval_sum(&light_eval));
  }
  if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, pass_diffuse_weight) = PackedSpectrum(
        volume_vertex ? one_spectrum() : bsdf_eval_pass_diffuse_weight(&light_eval));
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, pass_glossy_weight) = PackedSpectrum(
        volume_vertex ? zero_spectrum() : bsdf_eval_pass_glossy_weight(&light_eval));
  }
}

/* Passes of the light subpath generation, see integrator_bdpt_light_generate(). */
#define BDPT_LIGHT_STAGE_ALL 0u
#define BDPT_LIGHT_STAGE_PROBE 1u
#define BDPT_LIGHT_STAGE_LISTED 2u

/* Generate one complete light subpath. The pass deliberately uses the same intersection, shader
 * evaluation, closure sampling, volume attenuation/boundary handling and Russian roulette
 * conventions as camera paths. */
ccl_device_inline void bdpt_light_generate_path(KernelGlobals kg,
                                                IntegratorState state,
                                                const uint storage_path_index,
                                                const uint start_iteration,
                                                const uint stage,
                                                ccl_private uint *vcm_stored,
                                                ccl_private uint *vcm_kept)
{
  const uint paths_per_cache = kernel_integrator_state.bdpt_light_path_count;
  const uint cache = storage_path_index / paths_per_cache;
  const uint light_path_index = storage_path_index % paths_per_cache;
  const uint iteration = start_iteration + cache;
  const bool use_connections = bdpt_connections_enabled(kg);
  const bool use_merging = vcm_merging_enabled(kg);
  if (stage == BDPT_LIGHT_STAGE_ALL) {
    kernel_integrator_state.bdpt_vertex_indices[storage_path_index] = ~0u;
  }
#ifndef __KERNEL_GPU__
  if (use_merging && light_path_index < kernel_integrator_state.vcm_light_path_count *
                                            max(kernel_integrator_state.vcm_groups, 1u))
  {
    /* Forget the vertices that the previous map had in the slots of this subpath. */
    for (uint i = 0; i < kernel_integrator_state.vcm_path_slots; i++) {
      const uint slot = vcm_vertex_slot(kg, cache, light_path_index, i);
      if (slot < kernel_integrator_state.photon_capacity) {
        kernel_integrator_state.photon_valid[slot] = 0;
      }
    }
  }
#endif
  if (!use_connections && !use_merging) {
    return;
  }
  uint rng = lcg_init(
      hash_uint3(light_path_index, iteration, uint(kernel_data.integrator.seed) ^ 0x62647074u));
  photon_state_init(kg, state, rng, iteration);
#ifdef __SPECTRAL__
  /* Keep one immutable wavelength sample for the complete light subpath. Cached vertices must
   * retain this identity: reconstructing them with the camera path wavelength collapses
   * dispersive caustics back to RGB. */
  const float light_wavelength_rand = path_rng_1D(
      kg, rng, iteration, PRNG_BOUNCE_NUM + PRNG_WAVELENGTH);
#else
  const float light_wavelength_rand = 0.0f;
#endif
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vcm) = 0.0f;
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vc) = -INFINITY;
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vm) = -INFINITY;

  Ray ray ccl_optional_struct_init;
  Spectrum throughput;
  PolarizationSpectrumState light_polarization = polarization_unpolarized();
  Spectrum unguided_factor = one_spectrum();
  int emitter_object = OBJECT_NONE;
  int emitter_distribution = -1;
  int light_group = LIGHTGROUP_NONE;
  float emission_pdf = 0.0f;
  float direct_pdf = 0.0f;
  float emission_cosine = 1.0f;
  bool is_delta_emitter = false;
  bool is_finite_emitter = true;
  uint emitter_shader_flags = 0u;
  float emitter_max_bounces = FLT_MAX;
  const float time = lcg_step_float(&rng);
  bool emitter_cache_miss = false;
  if (!photon_sample_emitter(kg,
                             state,
                             &rng,
                             time,
                             &ray,
                             &throughput,
                             &emitter_object,
                             &light_group,
                             &emission_pdf,
                             &direct_pdf,
                             &emission_cosine,
                             &is_delta_emitter,
                             &is_finite_emitter,
                             &emitter_shader_flags,
                             &emitter_max_bounces,
                             &emitter_distribution,
                             &emitter_cache_miss))
  {
    if (emitter_cache_miss) {
      kernel_integrator_state.queue_counter->cache_miss = true;
    }
    return;
  }
  CoherentPathHistory coherent_history = coherent_history_begin();

#ifdef __SPECTRAL__
  const int emitter_shader = int(emitter_shader_flags) & SHADER_MASK;
  if (kernel_data_fetch(shaders, emitter_shader).flags & SD_REQUIRES_WAVELENGTH) {
    /* Wavelength-dependent emitters begin a monochromatic path before the first surface. Keep
     * their raw sampled spectrum and defer the wavelength PDF/CMF weight to the connection. */
    INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_SPECTRAL;
  }
#endif

  BDPTMISWeight d_vcm = BDPTMISWeight(direct_pdf) / bdpt_safe_pdf(emission_pdf);
  BDPTMISWeight d_vc = is_delta_emitter ?
                           0.0f :
                           BDPTMISWeight(is_finite_emitter ? emission_cosine : 1.0f) /
                               bdpt_safe_pdf(emission_pdf);
  /* Merge strategies of the earlier vertices, see scatter_light_vcm(). */
  BDPTMISWeight d_vm = 0.0f;
  const bool keep_all = vcm_keep_all(kg);
  /* Whether every scattering event so far was sharp: the vertices behind them are kept for
   * merging, and without connections nothing else is done with a light subpath. */
  bool sharp_prefix = true;
  uint vcm_rng = lcg_init(
      hash_uint3(light_path_index, iteration, uint(kernel_data.integrator.seed) ^ 0x76636d73u));
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vcm) = d_vcm.encoded();
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vc) = d_vc.encoded();
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vm) = d_vm.encoded();

#ifdef __VOLUME__
  if (kernel_data.integrator.use_volumes || kernel_data.integrator.use_nested_dielectrics) {
    INTEGRATOR_STATE_WRITE(state, ray, P) = ray.P;
    INTEGRATOR_STATE_WRITE(state, ray, D) = ray.D;
    INTEGRATOR_STATE_WRITE(state, ray, tmin) = ray.tmin;
    INTEGRATOR_STATE_WRITE(state, ray, tmax) = ray.tmax;
    INTEGRATOR_STATE_WRITE(state, ray, time) = ray.time;
    integrator_volume_stack_init(kg, state, PATH_RAY_VISIBILITY_GLOSSY);
  }
#endif

  /* Select one of the vertices the path actually reaches. Sampling a fixed bounce from the
   * configured maximum wastes most light paths in short scenes and produces sparse, high-energy
   * splats that viewport denoisers turn into halos. Per-path reservoir sampling retains every
   * valid path and stores its exact selection support for normalization and MIS. */
  uint reservoir_rng = lcg_init(hash_uint3(
      light_path_index, iteration, uint(kernel_data.integrator.seed) ^ 0x72657376u));
  uint reservoir_slot = UINT_MAX;
  uint selection_count = 0u;
  /* Whether a camera path samples the paths of this subpath by itself: it reaches the emitter,
   * or next event estimation connects to it from the first vertex. See the sensor connection of
   * medium vertices. */
  bool camera_samples_subpath = !is_delta_emitter &&
                                (emitter_distribution < 0 ||
                                 kernel_data_fetch(light_distribution, emitter_distribution).prim >=
                                     0 ||
                                 (emitter_shader_flags & SHADER_USE_MIS));
  for (int bounce = 0; bounce < kernel_data.integrator.bdpt_max_bounces; bounce++) {
    Intersection isect;
    const PathRayVisibility path_visibility = path_state_ray_visibility(state);
    const bool hit_surface = scene_intersect(kg, &ray, path_visibility, &isect);

#ifdef __VOLUME__
    if (kernel_data.integrator.use_volumes && !integrator_state_volume_stack_is_empty(kg, state)) {
      ray.tmax = hit_surface ? isect.t : FLT_MAX;
      INTEGRATOR_STATE_WRITE(state, path, throughput) = throughput;
      float3 scatter_P;
      int receiver_object = OBJECT_NONE;
      const PhotonVolumeSampleEvent volume_event = photon_volume_sample_segment(
          kg, state, &ray, &throughput, &scatter_P, &receiver_object);
      if (volume_event == PHOTON_VOLUME_CACHE_MISS) {
        kernel_integrator_state.queue_counter->cache_miss = true;
        return;
      }
      if (volume_event == PHOTON_VOLUME_SCATTERED) {
        ShaderData volume_sd;
        shader_setup_from_volume(&volume_sd, &ray, receiver_object);
        volume_sd.P = scatter_P;
        volume_sd.ray_length = len(scatter_P - ray.P);

        /* Generalized path space uses volume measure for a medium vertex. Convert the preceding
         * directional density with 1/r^2, but unlike a surface vertex there is no projected-area
         * cosine. Infinite emitters already sample their launch disk in projected-area measure. */
        if (!(bounce == 0 && !is_finite_emitter)) {
          d_vcm *= max(len_squared(scatter_P - ray.P), 1.0e-20f);
        }

        /* This is the first light-side medium collision, but it may follow any number of surface
         * events. In particular, L-S+-V-E is the volumetric-caustic transport class produced by a
         * prism. Include it in the same per-path reservoir as surface vertices. We return
         * immediately below because repeated free-flight strategy densities are not represented
         * by the compact recursion; camera paths retain full multiple scattering after their
         * first collision. */
        /* A medium vertex has no texture coordinates: keep the flag in their place. */
        volume_sd.u = camera_samples_subpath ? 1.0f : 0.0f;
        if (use_connections) {
          bdpt_reservoir_store_light_vertex(kg,
                                          &volume_sd,
                                          &ray,
                                          throughput,
                                          emitter_object,
                                          emitter_distribution,
                                          light_group,
                                          d_vcm,
                                          d_vc,
                                          d_vm,
                                          false,
                                          uint(bounce + 2),
                                          INTEGRATOR_STATE(state, path, transparent_bounce),
                                          INTEGRATOR_STATE(state, path, flag),
                                          emitter_shader_flags,
                                          light_wavelength_rand,
                                          coherent_history_invalidate(coherent_history),
                                          light_polarization,
                                          &reservoir_rng,
                                          &reservoir_slot,
                                          &selection_count,
                                          storage_path_index);
        }

        return;
      }
    }
#endif
    if (!hit_surface) {
      break;
    }

    ShaderData sd;
    shader_setup_from_ray(kg, &sd, &ray, &isect);
#ifdef __SPECTRAL__
    shader_setup_wavelength(kg, &sd, state);
#endif
#ifdef __NESTED_DIELECTRICS__
    {
      const LightPathNestedHit nested_hit = light_path_nested_dielectric_hit(
          kg, state, &sd, &ray, path_visibility);
      if (nested_hit == LIGHT_PATH_NESTED_PASSED) {
        /* Not a vertex of the subpath: the edge continues to the next surface, and the
         * recurrence converts its measure with the distance from the unchanged ray origin. */
        bounce--;
        continue;
      }
      if (nested_hit == LIGHT_PATH_NESTED_CACHE_MISS) {
        kernel_integrator_state.queue_counter->cache_miss = true;
        return;
      }
      if (nested_hit != LIGHT_PATH_NESTED_TRUE) {
        break;
      }
    }
#endif
    if (bounce == 0 && !use_connections && !keep_all &&
        (sd.shader_flag & (SD_HAS_NO_SHARP_SURFACE | SD_HAS_ONLY_VOLUME)) ==
            SD_HAS_NO_SHARP_SURFACE)
    {
      /* Only the vertices of caustics are kept, and the subpath neither scatters at a sharp
       * closure here nor passes through: nothing of it would be. */
      break;
    }
    if (stage == BDPT_LIGHT_STAGE_PROBE) {
      /* The subpath may form a caustic: list it for the pass that traces all of it. */
      const uint listed = atomic_fetch_and_add_uint32(
          &kernel_integrator_state.bdpt_vertex_count[0], 1u);
      kernel_integrator_state.bdpt_vertex_indices[listed] = storage_path_index;
      return;
    }
    surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE>(
        kg, state, &sd, nullptr, path_visibility, INTEGRATOR_STATE(state, path, flag));
    if (sd.runtime_flag & SR_CACHE_MISS) {
      kernel_integrator_state.queue_counter->cache_miss = true;
      return;
    }
    if (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_TERMINATE) {
      return;
    }

#ifdef __SPECTRAL__
    if (sd.runtime_flag & (SR_BSDF_HAS_DISPERSION | SR_BSDF_HAS_SPECTRAL_TRANSMISSION)) {
      /* As with photon paths, retain raw monochromatic power. The sensor/camera connection adds
       * the wavelength PDF and color-matching weight exactly once. */
      INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_SPECTRAL;
    }
#endif

#ifdef __LIGHT_LINKING__
    if (bounce == 0 && (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_LINKING) &&
        !light_link_object_match(kg, sd.object, emitter_object))
    {
      return;
    }
#endif

#ifdef __VOLUME__
    if (sd.shader_flag & SD_HAS_ONLY_VOLUME) {
      if (!path_state_volume_next(state)) {
        break;
      }
      path_media_enter_exit<false>(kg, state, &sd);
      ray.tmin = intersection_t_offset(sd.ray_length);
      ray.tmax = FLT_MAX;
      ray.self.prim = sd.prim;
      ray.self.object = sd.object;
      bounce--;
      continue;
    }
#endif

    surface_shader_prepare_closures(kg, state, &sd, path_visibility);

    /* This source prefix is already evaluated as deterministic irradiance at
     * the detector. The eye integrator may follow any subsequent camera-side
     * suffix through that detector, so light tracing must stop here before
     * caching a vertex, sensor splatting, or scattering onward. */
    if (kernel_integrator_state.bdpt_coherent_history &&
        coherent_detector_eligible(&sd) &&
        coherent_history_owned_candidate(kg, coherent_history, emitter_object))
    {
      break;
    }

    const BDPTMISWeight d_vcm_before_hit = d_vcm;
    const BDPTMISWeight d_vc_before_hit = d_vc;
    const BDPTMISWeight d_vm_before_hit = d_vm;
    const float cos_fixed = max(fabsf(dot(sd.Ng, sd.wi)), 1.0e-8f);
    if (bounce == 0 && !is_finite_emitter) {
      /* Infinite emitters sample a launch disk perpendicular to the ray. Its position density is
       * already in projected-area measure, so the first surface conversion has no distance term
       * (SmallVCM eq. 49). All later edges connect ordinary finite surface vertices. */
      d_vcm /= cos_fixed;
    }
    else {
      d_vcm = d_vcm * sqr(max(sd.ray_length, 1.0e-10f)) / cos_fixed;
    }
    d_vc /= cos_fixed;
    d_vm /= cos_fixed;

    /* A vertex behind sharp events only is one of a caustic, which merging is for. */
    const bool merge_here = use_merging && (keep_all || (bounce > 0 && sharp_prefix));
    if (merge_here && (sd.runtime_flag & SR_BSDF_HAS_EVAL) &&
        !(sd.object_flag & SD_OBJECT_SHADOW_CATCHER))
    {
      vcm_store_light_vertex(kg,
                             &sd,
                             &ray,
                             throughput,
                             emitter_object,
                             emitter_distribution,
                             d_vcm,
                             d_vc,
                             d_vm,
                             uint(bounce + 2),
                             INTEGRATOR_STATE(state, path, flag),
                             light_wavelength_rand,
                             cache,
                             light_path_index,
                             vcm_stored,
                             vcm_kept,
                             &vcm_rng);
    }

    if (use_connections && (sd.runtime_flag & SR_BSDF_HAS_EVAL) &&
        !(sd.object_flag & SD_OBJECT_SHADOW_CATCHER))
    {
      bdpt_reservoir_store_light_vertex(kg,
                                        &sd,
                                        &ray,
                                        throughput,
                                        emitter_object,
                                        emitter_distribution,
                                        light_group,
                                        d_vcm,
                                        d_vc,
                                        d_vm,
                                        sharp_prefix,
                                        uint(bounce + 2),
                                        INTEGRATOR_STATE(state, path, transparent_bounce),
                                        INTEGRATOR_STATE(state, path, flag),
                                        emitter_shader_flags,
                                        light_wavelength_rand,
                                        coherent_detector_eligible(&sd) ?
                                            coherent_history :
                                            coherent_history_invalidate(coherent_history),
                                        light_polarization,
                                        &reservoir_rng,
                                        &reservoir_slot,
                                        &selection_count,
                                        storage_path_index);
    }

    if (float(bounce) >= emitter_max_bounces) {
      break;
    }

    /* A surface without scattering closures, an emitter for example, ends the subpath. Picking
     * from it would return a closure that an earlier shader evaluation left in this memory. */
    if (!(sd.runtime_flag & (SR_BSDF | SR_BSSRDF))) {
      break;
    }

    float3 rand_bsdf = lcg_step_float3(&rng);
    const ccl_private ShaderClosure *sc = surface_shader_bsdf_bssrdf_pick(&sd, &rand_bsdf);
    if (CLOSURE_IS_RAY_PORTAL(sc->type)) {
      if (polarization_enabled(kg)) light_polarization = polarization_depolarized(light_polarization);
      const ccl_private RayPortalClosure *pc = (const ccl_private RayPortalClosure *)sc;
      float sum_sample_weight = 0.0f;
      for (int i = 0; i < sd.num_closure; i++) {
        if (CLOSURE_IS_BSDF_OR_BSSRDF(sd.closure[i].type)) {
          sum_sample_weight += sd.closure[i].sample_weight;
        }
      }
      if (!(sum_sample_weight > 0.0f)) {
        break;
      }
      const float pick_pdf = pc->sample_weight / sum_sample_weight;
      throughput *= pc->weight / bdpt_safe_pdf(pick_pdf);
      if (!isfinite_safe(throughput)) {
        break;
      }

      const bool moved_origin = len_squared(sd.P - pc->P) > 1.0e-9f;
      ray.P = moved_origin ? pc->P : ray_offset(sd.P, dot(sd.Ng, pc->D) >= 0.0f ? sd.Ng : -sd.Ng);
      ray.D = pc->D;
      ray.tmin = 0.0f;
      ray.tmax = FLT_MAX;
      ray.self.prim = moved_origin ? PRIM_NONE : sd.prim;
      ray.self.object = moved_origin ? OBJECT_NONE : sd.object;
      ray.self.light_prim = PRIM_NONE;
      ray.self.light_object = OBJECT_NONE;
      d_vcm = 0.0f;
      path_state_next(kg, state, LABEL_TRANSMIT | LABEL_RAY_PORTAL, sd.runtime_flag);
      continue;
    }
    if (!CLOSURE_IS_BSDF(sc->type) ||
        (bounce == 0 && _surface_shader_exclude(sc->type, emitter_shader_flags)))
    {
      break;
    }

    BsdfEval eval;
    float3 wo;
    float pdf;
    float mis_pdf;
    float unguided_pdf;
    float2 sampled_roughness = one_float2();
    float eta = 1.0f;
    float avg_roughness_squared = 0.0f;
    int label;
#ifdef __KERNEL_METAL__
    if (kernel_data.integrator.use_surface_guiding && kernel_integrator_state.guiding_capacity > 0)
    {
      const float rand_guiding = hash_uint3_to_float(
          light_path_index, iteration, uint(bounce) ^ 0x67756964u);
      uint resampling_rng = lcg_init(
          hash_uint3(light_path_index, iteration, uint(bounce) ^ 0x72697330u));
      const float3 rand_resampling = lcg_step_float3(&resampling_rng);
      label = surface_shader_bsdf_gpu_guided_sample_closure(kg,
                                                            &sd,
                                                            sc,
                                                            rand_bsdf,
                                                            rand_guiding,
                                                            rand_resampling,
                                                            &eval,
                                                            &wo,
                                                            &pdf,
                                                            &mis_pdf,
                                                            &unguided_pdf,
                                                            &sampled_roughness,
                                                            &eta,
                                                            avg_roughness_squared,
                                                            true);
    }
    else
#endif
    {
      label = surface_shader_bsdf_sample_closure(kg,
                                                 &sd,
                                                 sc,
                                                 rand_bsdf,
                                                 &eval,
                                                 &wo,
                                                 &pdf,
                                                 &sampled_roughness,
                                                 &eta,
                                                 avg_roughness_squared);
      mis_pdf = pdf;
      unguided_pdf = pdf;
    }
    if (!(pdf > 0.0f) || bsdf_eval_is_zero(&eval)) {
      break;
    }

    /* In light-path order the diffuse receiver of a caustic lies after this sharp event, so the
     * closure filtering used by camera paths cannot know yet that this reflection or refraction
     * will become a caustic. Honor the scene controls explicitly, as photon tracing does. The
     * vertex at the current surface was cached above, preserving direct glossy visibility; only
     * continuation into the disabled caustic transport class is stopped. */
    if (!bdpt_caustic_event_enabled(label,
                                    kernel_data.integrator.caustics_reflective,
                                    kernel_data.integrator.caustics_refractive))
    {
      break;
    }

    if (!(label & (LABEL_TRANSPARENT | LABEL_SINGULAR))) {
      sharp_prefix = sharp_prefix && vcm_surface_is_sharp(kg, &sd);
      if (!sharp_prefix && !use_connections) {
        /* No later vertex is kept. */
        break;
      }
    }

    if (kernel_integrator_state.bdpt_coherent_history) {
      const int patch_index = coherent_patch_for_hit(kg, sd.object, sd.prim, sd.type);
      int patch_mode = 0;
      int incident_side = 0;
      if (patch_index >= 0) {
        const ccl_global KernelCoherentPatch *patch =
            &kernel_data_fetch(coherent_patches, patch_index);
        patch_mode = patch->mode;
        if (patch_mode == 2) {
          const float3 normal = patch->shape == 1 ? normalize(sd.P - patch->center) :
              normalize(cross(float3(patch->tangent_u), float3(patch->tangent_v)));
          incident_side = dot(ray.D, normal) < 0.0f ? 1 : -1;
        }
      }
      if (kernel_data.integrator.coherent_transport_mode == 1) {
        /* Native Ng is outward for validated Glass; shading setup flips it
         * toward the incoming ray and marks the hit backfacing from inside. */
        coherent_history = coherent_history_stream_after_interface(
            coherent_history, patch_index >= 0 ? patch_mode : 0, sd.object, label,
            (sd.runtime_flag & SR_BACKFACING) ? -1 : 1,
            uint(kernel_data.integrator.coherent_max_interface_events));
      }
      else {
        coherent_history = coherent_history_after_scatter(
            coherent_history, patch_index, patch_mode, label, incident_side);
      }
    }

    PolarizationSpectrumState next_polarization = light_polarization;
    if (polarization_enabled(kg) && (label & (LABEL_SINGULAR | LABEL_TRANSPARENT))) {
      next_polarization = polarization_surface_transport(kg, &sd, wo, light_polarization,
          false, sc, bsdf_eval_sum(&eval), (label & LABEL_SINGULAR) != 0);
    }
    float reverse_pdf = mis_pdf;
    Spectrum adjoint_eval = bsdf_eval_sum(&eval);
    if (!(label & LABEL_TRANSPARENT)) {
      if (label & LABEL_SINGULAR) {
        if (surface_shader_has_physical_grating(&sd) &&
            (sc->type == CLOSURE_BSDF_DIFFRACTION_SMOOTH_ID || bsdf_microfacet_has_delta(sc)))
        {
          /* Sampling already evaluated the complete forward atomic mixture.
           * Supported delta closures bypass continuous guiding proposals. */
          reverse_pdf = bdpt_reverse_pdf(kg, state, &sd, wo, true, nullptr, true);
          if (sd.runtime_flag & SR_CACHE_MISS) {
            kernel_integrator_state.queue_counter->cache_miss = true;
            return;
          }
        }
        adjoint_eval = bdpt_transpose_delta_eval(
            adjoint_eval, sc->N, sd.Ng, sd.wi, wo, eta, label & LABEL_TRANSMIT);
      }
      else {
        Spectrum reciprocal_eval;
        reverse_pdf = bdpt_reverse_pdf(kg, state, &sd, wo, true, &reciprocal_eval, false,
            polarization_enabled(kg) ? &light_polarization : nullptr,
            polarization_enabled(kg) ? &next_polarization : nullptr);
        if (sd.runtime_flag & SR_CACHE_MISS) {
          kernel_integrator_state.queue_counter->cache_miss = true;
          return;
        }
        /* Reuse the complete reciprocal shader evaluation. An eta-only correction is not
         * sufficient for direction-dependent layered shaders or distinct shading normals. */
        adjoint_eval = bdpt_transpose_surface_eval(reciprocal_eval, sd.Ng, sd.wi, wo);
      }
    }
    light_polarization = next_polarization;
    throughput *= adjoint_eval / pdf;
    unguided_factor *= safe_divide(pdf, unguided_pdf);
    if (!isfinite_safe(throughput) || is_zero(throughput)) {
      break;
    }

    if (label & LABEL_TRANSPARENT) {
      path_state_next(kg, state, label, sd.runtime_flag);
      if (INTEGRATOR_STATE(state, path, transparent_bounce) >=
          kernel_data.integrator.transparent_max_bounce)
      {
        break;
      }
      d_vcm = d_vcm_before_hit;
      d_vc = d_vc_before_hit;
      d_vm = d_vm_before_hit;
      /* Keep the emitter/previous-scattering origin so the next area conversion
       * uses the complete edge length across all null surfaces. */
      ray.D = normalize(wo);
      ray.tmin = intersection_t_offset(sd.ray_length);
      ray.tmax = FLT_MAX;
      ray.self.prim = sd.prim;
      ray.self.object = sd.object;
      ray.self.light_prim = PRIM_NONE;
      ray.self.light_object = OBJECT_NONE;
      bounce--;
      continue;
    }

    const float cos_out = max(fabsf(dot(sd.Ng, normalize(wo))), 1.0e-8f);
    if (bounce == 0 && !(label & LABEL_SINGULAR)) {
      camera_samples_subpath = true;
    }
    const float3 d_previous = make_float3(d_vcm.encoded(), d_vc.encoded(), d_vm.encoded());
    if (label & LABEL_SINGULAR) {
      const float3 next = BDPTMISWeight::Log::scatter_delta_light_vcm(
          d_previous, cos_out, mis_pdf, reverse_pdf);
      d_vcm = BDPTMISWeight::from_encoded(next.x);
      d_vc = BDPTMISWeight::from_encoded(next.y);
      d_vm = BDPTMISWeight::from_encoded(next.z);
    }
    else {
      /* The camera path samples this vertex and reaches the rest of the subpath with next event
       * estimation at the first vertex, and with a connection at any other. */
      const float selection_ratio = bounce != 0 ? (use_connections ? 1.0f : 0.0f) :
          bdpt_light_selection_ratio(kg,
                                     emitter_distribution,
                                     ray.P,
                                     sd.P,
                                     sd.time,
                                     sd.P,
                                     dot(sd.N, wo) >= 0.0f ? sd.N : -sd.N,
                                     sd.runtime_flag,
                                     sd.object);
      const float3 next = BDPTMISWeight::Log::scatter_light_vcm(
          d_previous, cos_out, bdpt_safe_pdf(mis_pdf), reverse_pdf, selection_ratio, merge_here);
      d_vcm = BDPTMISWeight::from_encoded(next.x);
      d_vc = BDPTMISWeight::from_encoded(next.y);
      d_vm = BDPTMISWeight::from_encoded(next.z);
    }

    path_state_next(kg, state, label, sd.runtime_flag);
    if (label & LABEL_TRANSMIT) {
      path_media_enter_exit<false>(kg, state, &sd);
    }

    ray.P = ray_offset(sd.P, dot(sd.Ng, wo) >= 0.0f ? sd.Ng : -sd.Ng);
    ray.tmin = 0.0f;
    ray.D = normalize(wo);
    ray.tmax = FLT_MAX;
    ray.self.prim = sd.prim;
    ray.self.object = sd.object;
    ray.self.light_prim = PRIM_NONE;
    ray.self.light_object = OBJECT_NONE;

    if (bounce >= 3) {
      /* RIS contribution weights depend on rejected candidates. Keep roulette a function
       * of the selected path, so bidirectional MIS remains a deterministic partition. */
      const float continuation = bdpt_light_path_continuation_probability(throughput,
                                                                          unguided_factor);
      if (lcg_step_float(&rng) >= continuation) {
        break;
      }
      throughput /= continuation;
      /* Camera MIS omits roulette probabilities. Use the same directional densities here;
       * survival compensation belongs in throughput, preserving a common MIS partition. */
    }
  }
}

/* On the GPU all subpaths of a pass are traced by threads that run in lockstep: one long
 * subpath holds up every short one next to it. When only the vertices of caustics are kept,
 * most subpaths end at their first surface. A first pass then only finds the ones that do not,
 * and a second traces those next to each other. */
ccl_device void integrator_bdpt_light_generate(KernelGlobals kg,
                                               IntegratorState state,
                                               const uint dispatch_index,
                                               const uint start_iteration,
                                               ccl_attr_maybe_unused const uint batch_samples,
                                               const uint stage = BDPT_LIGHT_STAGE_ALL)
{
  const uint storage_path_index = (stage == BDPT_LIGHT_STAGE_LISTED) ?
                                      kernel_integrator_state.bdpt_vertex_indices[dispatch_index] :
                                      dispatch_index;
  uint vcm_stored = 0u;
  uint vcm_kept = 0u;
  bdpt_light_generate_path(
      kg, state, storage_path_index, start_iteration, stage, &vcm_stored, &vcm_kept);
  if (vcm_stored != 0u) {
    const uint paths_per_cache = kernel_integrator_state.bdpt_light_path_count;
    vcm_finish_light_path(kg,
                          storage_path_index / paths_per_cache,
                          storage_path_index % paths_per_cache,
                          vcm_stored,
                          vcm_kept);
  }
}

/* Sensor connections are deliberately isolated from light generation. The manifold solver has a
 * large live working set; keeping it in a separate Metal kernel avoids inflating compile time and
 * register pressure for every emitted light path. The compact cache already contains one
 * reservoir-selected connectible vertex per path, so it is also an unbiased sensor reservoir. */
ccl_device void integrator_bdpt_cache_order(KernelGlobals kg,
                                            ccl_attr_maybe_unused const uint num_light_paths,
                                            const uint lane = 0,
                                            const uint width = 32,
                                            const uint cache = 0)
{
  const uint count = compact_indices(kernel_integrator_state.bdpt_vertex_indices +
                                         cache * kernel_integrator_state.bdpt_vertex_capacity,
                                     kernel_integrator_state.bdpt_light_path_count,
                                     lane,
                                     width);
  if (lane == 0) {
    kernel_integrator_state.bdpt_vertex_count[cache] = count;
  }
}

ccl_device void integrator_bdpt_sensor_connect(KernelGlobals kg,
                                               IntegratorState state,
                                               const uint dispatch_index,
                                               const uint start_iteration,
                                               const uint batch_samples,
                                               ccl_global float *render_buffer)
{
  if (!kernel_integrator_state.bdpt_vertex_count || !kernel_integrator_state.bdpt_vertices ||
      !bdpt_connections_enabled(kg))
  {
    return;
  }
  const uint paths_per_cache = kernel_integrator_state.bdpt_light_path_count;
  const uint cache = dispatch_index / paths_per_cache;
  const uint vertex_index = dispatch_index % paths_per_cache;
  const uint iteration = start_iteration + cache;
  const uint samples_per_cache = batch_samples / kernel_integrator_state.bdpt_cache_count;
  const uint vertex_count = min(kernel_integrator_state.bdpt_vertex_count[cache],
                                kernel_integrator_state.bdpt_vertex_capacity);
  if (vertex_index >= vertex_count) {
    return;
  }

  const uint storage_index =
      kernel_integrator_state
          .bdpt_vertex_indices[cache * kernel_integrator_state.bdpt_vertex_capacity +
                               vertex_index];
  KernelBDPTVertex light_vertex = kernel_integrator_state.bdpt_vertices[storage_index];
  if (light_vertex.sensor_complete) {
    return;
  }
  if (kernel_integrator_state.bdpt_coherent_history && light_vertex.object != OBJECT_NONE &&
      (kernel_data_fetch(object_flag, light_vertex.object) & SD_OBJECT_COHERENT_DETECTOR) &&
      coherent_history_owned_candidate(
          kg, kernel_integrator_state.bdpt_coherent_history[storage_index],
          light_vertex.emitter_object))
  {
    kernel_integrator_state.bdpt_vertices[storage_index].sensor_complete = 1;
    return;
  }
  kernel_integrator_state.bdpt_vertices[storage_index].sensor_complete = 1;
  uint rng = lcg_init(
      hash_uint3(vertex_index, iteration, uint(kernel_data.integrator.seed) ^ 0x73656e73u));
  photon_state_init(kg, state, rng, iteration);
  INTEGRATOR_STATE_WRITE(state, path, flag) = light_vertex.flag;
  const uint path_length = bdpt_vertex_path_length(&light_vertex);
  const uint selection_count = bdpt_vertex_selection_count(&light_vertex);
  if (selection_count == 0u || path_length < 2u) {
    return;
  }
  INTEGRATOR_STATE_WRITE(state, path, bounce) = path_length - 2u;
  INTEGRATOR_STATE_WRITE(state, path, transparent_bounce) = light_vertex.path_length >> 20u;

#ifdef __VOLUME__
  /* Dedicated sensor work does not inherit the generating light path's medium stack. Rebuild the
   * containing media at a cached medium vertex so the connection shadow ray receives the same
   * transmittance treatment as ordinary volume NEE. */
  if (light_vertex.type == PRIMITIVE_VOLUME && kernel_data.integrator.use_volumes) {
    packed_normal packed_incoming;
    packed_incoming.value = light_vertex.incoming;
    INTEGRATOR_STATE_WRITE(state, ray, P) = light_vertex.P;
    INTEGRATOR_STATE_WRITE(state, ray, D) = packed_incoming.decode();
    INTEGRATOR_STATE_WRITE(state, ray, tmin) = 0.0f;
    INTEGRATOR_STATE_WRITE(state, ray, tmax) = FLT_MAX;
    INTEGRATOR_STATE_WRITE(state, ray, time) = photon_unpack_time(light_vertex.time_wavelength);
    integrator_volume_stack_init(kg, state, PATH_RAY_VISIBILITY_CAMERA);
  }
#endif

  ShaderData light_sd;
  light_sd.runtime_flag = 0;
  if (!bdpt_setup_light_vertex(kg, state, &light_vertex, &light_sd)) {
    if (light_sd.runtime_flag & SR_CACHE_MISS) {
      kernel_integrator_state.bdpt_vertices[storage_index].sensor_complete = 0;
      kernel_integrator_state.queue_counter->cache_miss = true;
    }
    return;
  }
  const float2 rand_lens = make_float2(lcg_step_float(&rng), lcg_step_float(&rng));
  bdpt_connect_light_vertex_to_camera(kg,
                                      state,
                                      &light_vertex,
                                      &light_sd,
                                      kernel_integrator_state.bdpt_polarization ?
                                          polarization_unpack(kernel_integrator_state.bdpt_polarization[storage_index]) :
                                          polarization_unpolarized(),
                                      selection_count,
                                      iteration,
                                      samples_per_cache,
                                      render_buffer,
                                      rand_lens);
  if (light_sd.runtime_flag & SR_CACHE_MISS) {
    kernel_integrator_state.bdpt_vertices[storage_index].sensor_complete = 0;
    kernel_integrator_state.queue_counter->cache_miss = true;
  }
}

CCL_NAMESPACE_END
