/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Metal light-vertex-cache bidirectional path tracing.
 *
 * Light subpaths are generated in one GPU pass and one connectible vertex per path is selected
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
#ifdef __MNEE__
#  include "kernel/integrator/mnee.h"
#endif
#include "kernel/sample/lcg.h"
#include "kernel/util/compact_indices.h"
#include "util/atomic.h"

CCL_NAMESPACE_BEGIN

ccl_device_inline bool bdpt_camera_supported()
{
  const CameraType camera_type = CameraType(kernel_data.cam.type);
  if (kernel_data.cam.interocular_offset != 0.0f || camera_type == CAMERA_CUSTOM) {
    return false;
  }
  return camera_type == CAMERA_PERSPECTIVE ||
         ((camera_type == CAMERA_PANORAMA || camera_type == CAMERA_ORTHOGRAPHIC) &&
          kernel_data.cam.aperturesize == 0.0f && kernel_data.cam.num_motion_steps == 0);
}

ccl_device_inline bool bdpt_enabled_for_surface_path(ConstIntegratorState state)
{
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  return kernel_data.integrator.use_bidirectional_path_tracing && bdpt_camera_supported() &&
         !(path_flag & (PATH_RAY_SHADOW_CATCHER_HIT | PATH_RAY_SHADOW_CATCHER_PASS |
                        PATH_RAY_BDPT_UNSUPPORTED));
}

/* The first camera-volume scatter competes with NEE and a light-to-sensor
 * connection even though later camera connections use the unsupported-path
 * fallback. Preserve that partition when its continuation hits an emitter.
 * Additional real events are deliberately not covered by this predicate. */
ccl_device_inline bool bdpt_enabled_for_emission(ConstIntegratorState state)
{
  if (bdpt_enabled_for_surface_path(state)) {
    return true;
  }
  const uint32_t flag = INTEGRATOR_STATE(state, path, flag);
  return kernel_data.integrator.use_bidirectional_path_tracing && bdpt_camera_supported() &&
         !(flag & (PATH_RAY_SHADOW_CATCHER_HIT | PATH_RAY_SHADOW_CATCHER_PASS)) &&
         (flag & PATH_RAY_BDPT_UNSUPPORTED) &&
         INTEGRATOR_STATE(state, path, bounce) == 1 &&
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
      return Spectrum(photon_spectral_kernel(light_wavelength, camera_wavelength) /
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

ccl_device_inline float bdpt_infinite_position_pdf(const float3 P, const float3 direction)
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

ccl_device_inline float bdpt_point_emission_direction_pdf(const float3 light_P,
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

ccl_device_inline float bdpt_emission_mis_weight_infinite(IntegratorState state,
                                                          const float direct_pdf_w,
                                                          const float position_pdf,
                                                          const float selection_ratio)
{
  if (INTEGRATOR_STATE(state, path, bounce) == 0 ||
      (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_MIS_SKIP))
  {
    return 1.0f;
  }
  const float emission_pdf_w = direct_pdf_w * position_pdf;
  const BDPTMISWeight w_camera =
      BDPTMISWeight(direct_pdf_w) * selection_ratio *
          BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vcm)) +
      BDPTMISWeight(emission_pdf_w) *
          BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vc));
  return (BDPTMISWeight(1.0f) + w_camera).inverse();
}

ccl_device_inline float bdpt_emission_mis_weight_surface(KernelGlobals kg,
                                                         IntegratorState state,
                                                         const ccl_private ShaderData *sd)
{
  if (INTEGRATOR_STATE(state, path, bounce) == 0 ||
      (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_MIS_SKIP))
  {
    return 1.0f;
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
      BDPTMISWeight(emission_pdf_w) *
          BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vc));
  return (BDPTMISWeight(1.0f) + w_camera).inverse();
}

ccl_device_inline float bdpt_emission_mis_weight_lamp(KernelGlobals kg,
                                                      IntegratorState state,
                                                      const ccl_global KernelLight *klight,
                                                      const float3 ray_P,
                                                      const float3 ray_D,
                                                      const float distance,
                                                      const float nee_pdf_w)
{
  if (INTEGRATOR_STATE(state, path, bounce) == 0 ||
      (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_MIS_SKIP))
  {
    return 1.0f;
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
  const BDPTMISWeight d_vc = BDPTMISWeight::from_encoded(
                                 INTEGRATOR_STATE(state, path, bdpt_d_vc)) /
                             cos_at_light;
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
    ccl_private Spectrum *reciprocal_eval = nullptr)
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
  const float pdf = surface_shader_bsdf_eval(kg,
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
  return pdf;
}

/* Balance-heuristic weight for the next-event strategy in the presence of light tracing and
 * vertex connections. This is Georgiev's recursive form of Veach MIS (SmallVCM eqs. 44-45).
 * The current implementation evaluates the exact emission density for triangle and area lights;
 * other analytic lights retain the regular two-strategy balance weight until their singular
 * measures are handled explicitly. */
ccl_device_inline float bdpt_nee_mis_weight(KernelGlobals kg,
                                            IntegratorState state,
                                            ccl_private ShaderData *sd,
                                            const ccl_private LightSample *ls,
                                            const float bsdf_pdf)
{
  const float direct_pdf = bdpt_safe_pdf(ls->pdf);
  BDPTMISWeight w_light = BDPTMISWeight(bsdf_pdf) / direct_pdf;

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
    const float position_pdf = bdpt_infinite_position_pdf(sd->P, -ls->D);
    const float reverse_pdf = bdpt_reverse_pdf(kg, state, sd, ls->D);
    const float cos_camera = max(fabsf(dot(sd->Ng, ls->D)), 1.0e-8f);
    const float emission_selection_ratio = kernel_data.integrator.distribution_pdf_lights /
                                           bdpt_safe_pdf(ls->pdf_selection);
    w_camera = BDPTMISWeight(emission_selection_ratio) * position_pdf * cos_camera *
               (BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vcm)) +
                BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vc)) *
                    reverse_pdf);
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
                                        bdpt_point_emission_direction_pdf(klight->co, -ls->D));
      const float reverse_pdf = bdpt_reverse_pdf(kg, state, sd, ls->D);
      const float cos_camera = max(fabsf(dot(sd->Ng, ls->D)), 1.0e-8f);
      w_camera = BDPTMISWeight(emission_pdf_w) * cos_camera / direct_pdf *
                 (BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vcm)) +
                  BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vc)) *
                      reverse_pdf);
    }
  }

  if (emission_position_pdf > 0.0f) {
    const float reverse_pdf = bdpt_reverse_pdf(kg, state, sd, ls->D);
    const float cos_camera = max(fabsf(dot(sd->Ng, ls->D)), 1.0e-8f);
    const BDPTMISWeight emission_to_direct = BDPTMISWeight(emission_position_pdf) *
                                             emission_side_pdf * cos_camera / M_PI_F / direct_pdf;
    w_camera = emission_to_direct *
               (BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vcm)) +
                BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vc)) *
                    reverse_pdf);
  }

  return (BDPTMISWeight(1.0f) + w_light + w_camera).inverse();
}

/* Match camera_sample_perspective()'s time interpolation in camera space. */
ccl_device_inline float3 bdpt_perspective_image_point(const float3 raster, const float time)
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

ccl_device_inline void bdpt_recursive_mis_before_measure_conversion(
    KernelGlobals kg,
    IntegratorState state,
    const ccl_private ShaderData *sd,
    ccl_private BDPTMISWeight *d_vcm_out,
    ccl_private BDPTMISWeight *d_vc_out)
{
  BDPTMISWeight d_vcm = BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vcm));
  BDPTMISWeight d_vc = BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vc));

  if (INTEGRATOR_STATE(state, path, bounce) == 0 && d_vcm == 0.0f) {
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
      const float3 image_origin = bdpt_perspective_image_point(zero_float3(), sd->time);
      const float3 image_dx = bdpt_perspective_image_point(make_float3(1, 0, 0), sd->time) -
                              image_origin;
      const float3 image_dy = bdpt_perspective_image_point(make_float3(0, 1, 0), sd->time) -
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
ccl_device_inline float bdpt_volume_nee_mis_weight(
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
    w_camera = BDPTMISWeight(emission_selection_ratio) * bdpt_infinite_position_pdf(P, -ls->D) *
               (d_vcm + d_vc * reverse_pdf);
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
                                        bdpt_point_emission_direction_pdf(klight->co, -ls->D));
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

ccl_device_inline void bdpt_recursive_mis_after_hit(KernelGlobals kg,
                                                    IntegratorState state,
                                                    const ccl_private ShaderData *sd)
{
  BDPTMISWeight d_vcm;
  BDPTMISWeight d_vc;
  bdpt_recursive_mis_before_measure_conversion(kg, state, sd, &d_vcm, &d_vc);

  const float distance2 = sqr(max(sd->ray_length, 1.0e-10f));
  const float cos_fixed = max(fabsf(dot(sd->Ng, sd->wi)), 1.0e-8f);
  d_vcm = d_vcm * distance2 / cos_fixed;
  d_vc /= cos_fixed;

  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vcm) = d_vcm.encoded();
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vc) = d_vc.encoded();
}

ccl_device_inline void bdpt_recursive_mis_undo_transparent_hit(IntegratorState state,
                                                               const ccl_private ShaderData *sd)
{
  /* A null event keeps the original ray origin and does not introduce a vertex
   * in the MIS path. Restore the directional measure before tracing the rest
   * of that same edge. A primary ray must recompute its camera/clip conversion
   * at the eventual scattering vertex. */
  if (INTEGRATOR_STATE(state, path, bounce) == 0) {
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vcm) = -INFINITY;
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vc) = -INFINITY;
  }
  else {
    const float cosine = max(fabsf(dot(sd->Ng, sd->wi)), 1.0e-8f);
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vcm) =
        (BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vcm)) *
         (BDPTMISWeight(cosine) / sqr(max(sd->ray_length, 1.0e-10f))))
            .encoded();
    INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vc) = (BDPTMISWeight::from_encoded(INTEGRATOR_STATE(
                                                          state, path, bdpt_d_vc)) *
                                                      BDPTMISWeight(cosine))
                                                         .encoded();
  }
}

ccl_device_inline void bdpt_recursive_mis_after_scatter(IntegratorState state,
                                                        const int label,
                                                        const float cos_out,
                                                        const float forward_pdf,
                                                        const float reverse_pdf)
{
  BDPTMISWeight d_vcm = BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vcm));
  BDPTMISWeight d_vc = BDPTMISWeight::from_encoded(INTEGRATOR_STATE(state, path, bdpt_d_vc));

  if (label & LABEL_SINGULAR) {
    d_vcm = 0.0f;
    d_vc *= cos_out;
  }
  else {
    const float2 next = BDPTMISWeight::Log::scatter(make_float2(d_vcm.encoded(), d_vc.encoded()),
                                                    cos_out,
                                                    bdpt_safe_pdf(forward_pdf),
                                                    reverse_pdf);
    d_vcm = BDPTMISWeight::from_encoded(next.x);
    d_vc = BDPTMISWeight::from_encoded(next.y);
  }

  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vcm) = d_vcm.encoded();
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vc) = d_vc.encoded();
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
  stored_vertex->light_group = light_group;
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
                                                         const uint path_length,
                                                         const uint transparent_bounce,
                                                         const uint flag,
                                                         const uint emitter_shader_flags,
                                                         const float wavelength_rand,
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
    *stored_vertex = local_vertex;
  }
  else {
    stored_vertex->path_length = bdpt_pack_vertex_support(
        stored_vertex->path_length & 0xffu, *selection_count, stored_vertex->path_length >> 20u);
  }
}

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

    const float3 image_origin = bdpt_perspective_image_point(zero_float3(), vertex_time);
    const float3 image_dx = bdpt_perspective_image_point(make_float3(1.0f, 0.0f, 0.0f),
                                                         vertex_time) -
                            image_origin;
    const float3 image_dy = bdpt_perspective_image_point(make_float3(0.0f, 1.0f, 0.0f),
                                                         vertex_time) -
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

    const float3 image_P = bdpt_perspective_image_point(*raster, vertex_time);
    const float3 image_X = bdpt_perspective_image_point(
        make_float3(raster->x + 1.0f, raster->y, raster->z), vertex_time);
    const float3 image_Y = bdpt_perspective_image_point(
        make_float3(raster->x, raster->y + 1.0f, raster->z), vertex_time);
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
ccl_device_inline bool bdpt_perspective_ray_to_raster(const float3 sensor_P,
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
  const float3 image_origin = bdpt_perspective_image_point(zero_float3(), time);
  const float3 image_dx = bdpt_perspective_image_point(make_float3(1.0f, 0.0f, 0.0f), time) -
                          image_origin;
  const float3 image_dy = bdpt_perspective_image_point(make_float3(0.0f, 1.0f, 0.0f), time) -
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

  const float3 image_P = bdpt_perspective_image_point(*raster, time);
  const float3 image_X = bdpt_perspective_image_point(
      make_float3(raster->x + 1.0f, raster->y, raster->z), time);
  const float3 image_Y = bdpt_perspective_image_point(
      make_float3(raster->x, raster->y + 1.0f, raster->z), time);
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

/* Splat one reservoir-selected light vertex onto a built-in sensor. */
ccl_device_inline void bdpt_connect_light_vertex_to_camera(
    KernelGlobals kg,
    IntegratorState state,
    const ccl_private KernelBDPTVertex *light_vertex,
    ccl_private ShaderData *light_sd,
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
  Spectrum manifold_throughput = one_spectrum();
  bool manifold_connection = false;

#ifdef __MNEE__
  /* A cached diffuse light vertex viewed through one or more specular interfaces is the SDS
   * transport class that ordinary BDPT sensor splats cannot connect. Walk the same exact
   * specular manifold used by Cycles MNEE, but in reverse: camera -> interfaces -> cached light
   * vertex. This yields both the physically valid endpoint direction and its transfer Jacobian. */
  if ((kernel_data.kernel_features & KERNEL_FEATURE_MNEE) &&
      light_vertex->type != PRIMITIVE_VOLUME &&
      CameraType(kernel_data.cam.type) == CAMERA_PERSPECTIVE)
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
    sensor_target.Ng = light_sd->N;
    sensor_target.t = distance;
    sensor_target.D = -direction;
    sensor_target.pdf = 1.0f;
    sensor_target.pdf_selection = 1.0f;
    sensor_target.eval_fac = 1.0f;
    sensor_target.object = light_vertex->object;
    sensor_target.prim = light_vertex->prim;
    sensor_target.shader = light_sd->shader;
    sensor_target.group = light_vertex->light_group;
    sensor_target.type = LIGHT_TRIANGLE;
    sensor_target.emitter_id = EMITTER_NONE;

    ShaderDataCausticsStorage manifold_sd_storage;
    ccl_private ShaderData *manifold_sd = AS_SHADER_DATA(&manifold_sd_storage);
    RNGState rng_state;
    path_state_rng_load(state, &rng_state);
    Spectrum candidate_throughput = zero_spectrum();
    float3 camera_wo = zero_float3();
    float3 light_wo = zero_float3();
    float light_distance = 0.0f;
    int manifold_vertex_count = 0;
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
                                                                     photon_unpack_wavelength_rand(
                                                                         light_vertex
                                                                             ->time_wavelength));
    if (manifold_result == SHADER_EVAL_CACHE_MISS) {
      light_sd->runtime_flag |= SR_CACHE_MISS;
      return;
    }
    float3 manifold_raster;
    float sensor_jacobian;
    if (manifold_vertex_count > 0 && isfinite_safe(candidate_throughput) &&
        bdpt_perspective_ray_to_raster(sensor_P,
                                      camera_wo,
                                      camera_sd->time,
                                      &manifold_raster,
                                      &sensor_jacobian))
    {
      /* The manifold routine validates camera-to-interface segments. Check the final free segment
       * from the cached endpoint back to the last interface; the hit at its upper bound is the
       * intended manifold vertex, while anything earlier is a true blocker. */
      Ray verify_ray ccl_optional_struct_init;
      bool verify_skip_self = true;
      verify_ray.P = shadow_ray_offset(kg, light_sd, light_wo, &verify_skip_self);
      verify_ray.D = light_wo;
      verify_ray.tmin = 0.0f;
      verify_ray.tmax = light_distance;
      verify_ray.time = camera_sd->time;
      verify_ray.self.object = verify_skip_self ? light_sd->object : OBJECT_NONE;
      verify_ray.self.prim = verify_skip_self ? light_sd->prim : PRIM_NONE;
      Intersection verify_isect;
      const bool early_blocker = scene_intersect(
                                     kg,
                                     &verify_ray,
                                     PATH_RAY_VISIBILITY_TRANSMIT,
                                     &verify_isect) &&
                                 verify_isect.t < light_distance - MNEE_MIN_DISTANCE;
      if (!early_blocker) {
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
  const bool volume_vertex = light_vertex->type == PRIMITIVE_VOLUME;
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
  const BDPTMISWeight w_light = (BDPTMISWeight(camera_pdf_area) / light_path_sample_ratio) *
                                (BDPTMISWeight::from_encoded(light_vertex->d_vcm) *
                                     selection_ratio +
                                 BDPTMISWeight::from_encoded(light_vertex->d_vc) * reverse_pdf);
  const float mis_weight = (BDPTMISWeight(1.0f) + w_light).inverse();

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
  const Spectrum contribution = Spectrum(light_vertex->throughput) * spectral_weight *
                                manifold_throughput * sensor_eval *
                                (mis_weight * normalization * connection_jacobian);
  if (!isfinite_safe(contribution) || is_zero(contribution)) {
    return;
  }

  Ray shadow_ray ccl_optional_struct_init;
  bool skip_self = !volume_vertex;
  shadow_ray.P = volume_vertex ? light_vertex->P :
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
  if (!volume_vertex && kernel_data.integrator.use_volumes) {
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
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, lightgroup) = light_vertex->light_group + 1;
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

/* Generate one complete light subpath. The pass deliberately uses the same intersection, shader
 * evaluation, closure sampling, volume attenuation/boundary handling and Russian roulette
 * conventions as camera paths. */
ccl_device void integrator_bdpt_light_generate(KernelGlobals kg,
                                               IntegratorState state,
                                               const uint storage_path_index,
                                               const uint start_iteration,
                                               const uint batch_samples)
{
  const uint paths_per_cache = kernel_integrator_state.bdpt_light_path_count;
  const uint cache = storage_path_index / paths_per_cache;
  const uint light_path_index = storage_path_index % paths_per_cache;
  const uint iteration = start_iteration + cache;
  kernel_integrator_state.bdpt_vertex_indices[storage_path_index] = ~0u;
  if (!bdpt_camera_supported()) {
    return;
  }
  uint rng = lcg_init(
      hash_uint3(light_path_index, iteration, uint(kernel_data.integrator.seed) ^ 0x62647074u));
  photon_state_init(state, rng, iteration);
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

  Ray ray ccl_optional_struct_init;
  Spectrum throughput;
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
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vcm) = d_vcm.encoded();
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vc) = d_vc.encoded();

#ifdef __VOLUME__
  if (kernel_data.integrator.use_volumes) {
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
        bdpt_reservoir_store_light_vertex(kg,
                                          &volume_sd,
                                          &ray,
                                          throughput,
                                          emitter_object,
                                          emitter_distribution,
                                          light_group,
                                          d_vcm,
                                          d_vc,
                                          uint(bounce + 2),
                                          INTEGRATOR_STATE(state, path, transparent_bounce),
                                          INTEGRATOR_STATE(state, path, flag),
                                          emitter_shader_flags,
                                          light_wavelength_rand,
                                          &reservoir_rng,
                                          &reservoir_slot,
                                          &selection_count,
                                          storage_path_index);

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
      volume_stack_enter_exit<false>(kg, state, &sd);
      ray.tmin = intersection_t_offset(sd.ray_length);
      ray.tmax = FLT_MAX;
      ray.self.prim = sd.prim;
      ray.self.object = sd.object;
      bounce--;
      continue;
    }
#endif

    surface_shader_prepare_closures(kg, state, &sd, path_visibility);

    const BDPTMISWeight d_vcm_before_hit = d_vcm;
    const BDPTMISWeight d_vc_before_hit = d_vc;
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

    if ((sd.runtime_flag & SR_BSDF_HAS_EVAL) &&
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
                                        uint(bounce + 2),
                                        INTEGRATOR_STATE(state, path, transparent_bounce),
                                        INTEGRATOR_STATE(state, path, flag),
                                        emitter_shader_flags,
                                        light_wavelength_rand,
                                        &reservoir_rng,
                                        &reservoir_slot,
                                        &selection_count,
                                        storage_path_index);
    }

    if (float(bounce) >= emitter_max_bounces) {
      break;
    }

    float3 rand_bsdf = lcg_step_float3(&rng);
    const ccl_private ShaderClosure *sc = surface_shader_bsdf_bssrdf_pick(&sd, &rand_bsdf);
    if (CLOSURE_IS_RAY_PORTAL(sc->type)) {
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
    else {
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

    float reverse_pdf = mis_pdf;
    Spectrum adjoint_eval = bsdf_eval_sum(&eval);
    if (!(label & LABEL_TRANSPARENT)) {
      if (label & LABEL_SINGULAR) {
        adjoint_eval = bdpt_transpose_delta_eval(
            adjoint_eval, sc->N, sd.Ng, sd.wi, wo, eta, label & LABEL_TRANSMIT);
      }
      else {
        Spectrum reciprocal_eval;
        reverse_pdf = bdpt_reverse_pdf(kg, state, &sd, wo, true, &reciprocal_eval);
        if (sd.runtime_flag & SR_CACHE_MISS) {
          kernel_integrator_state.queue_counter->cache_miss = true;
          return;
        }
        /* Reuse the complete reciprocal shader evaluation. An eta-only correction is not
         * sufficient for direction-dependent layered shaders or distinct shading normals. */
        adjoint_eval = bdpt_transpose_surface_eval(reciprocal_eval, sd.Ng, sd.wi, wo);
      }
    }
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
    if (label & LABEL_SINGULAR) {
      d_vcm = 0.0f;
      d_vc *= cos_out;
    }
    else {
      const float selection_ratio = bounce == 0 ?
          bdpt_light_selection_ratio(kg,
                                     emitter_distribution,
                                     ray.P,
                                     sd.P,
                                     sd.time,
                                     sd.P,
                                     dot(sd.N, wo) >= 0.0f ? sd.N : -sd.N,
                                     sd.runtime_flag,
                                     sd.object) : 1.0f;
      const float2 next = BDPTMISWeight::Log::scatter(make_float2(d_vcm.encoded(), d_vc.encoded()),
                                                      cos_out,
                                                      bdpt_safe_pdf(mis_pdf),
                                                      reverse_pdf,
                                                      selection_ratio);
      d_vcm = BDPTMISWeight::from_encoded(next.x);
      d_vc = BDPTMISWeight::from_encoded(next.y);
    }

    path_state_next(kg, state, label, sd.runtime_flag);
#ifdef __VOLUME__
    if (label & LABEL_TRANSMIT) {
      volume_stack_enter_exit<false>(kg, state, &sd);
    }
#endif

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

/* Sensor connections are deliberately isolated from light generation. The manifold solver has a
 * large live working set; keeping it in a separate Metal kernel avoids inflating compile time and
 * register pressure for every emitted light path. The compact cache already contains one
 * reservoir-selected connectible vertex per path, so it is also an unbiased sensor reservoir. */
ccl_device void integrator_bdpt_cache_order(const uint num_light_paths,
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
  if (!kernel_integrator_state.bdpt_vertex_count || !kernel_integrator_state.bdpt_vertices) {
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
  kernel_integrator_state.bdpt_vertices[storage_index].sensor_complete = 1;
  uint rng = lcg_init(
      hash_uint3(vertex_index, iteration, uint(kernel_data.integrator.seed) ^ 0x73656e73u));
  photon_state_init(state, rng, iteration);
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
