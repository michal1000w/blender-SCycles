/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Metal light-vertex-cache bidirectional path tracing.
 *
 * Light subpaths are generated in one GPU pass and every connectible surface vertex is appended
 * to a compact global cache. Camera paths sample this cache at their surface vertices. Recursive
 * MIS terms follow Georgiev's balance-heuristic formulation, so connecting a pair only requires
 * local PDFs and the two cached partial weights. */

#include "kernel/bvh/bvh.h"
#include "kernel/camera/camera.h"
#include "kernel/geom/shader_data.h"
#include "kernel/integrator/path_state.h"
#include "kernel/integrator/photon_mapping.h"
#include "kernel/integrator/state_flow.h"
#include "kernel/integrator/state_util.h"
#include "kernel/integrator/surface_shader.h"
#ifdef __MNEE__
#  include "kernel/integrator/mnee.h"
#endif
#include "kernel/sample/lcg.h"
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
         INTEGRATOR_STATE(state, path, volume_bounce) == 0 &&
         !(path_flag & (PATH_RAY_SHADOW_CATCHER_HIT | PATH_RAY_SHADOW_CATCHER_PASS));
}

ccl_device_inline float bdpt_safe_pdf(const float pdf)
{
  return max(pdf, 1.0e-20f);
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
                                                          const float position_pdf)
{
  if (INTEGRATOR_STATE(state, path, bounce) == 0 ||
      (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_MIS_SKIP))
  {
    return 1.0f;
  }
  const float emission_pdf_w = direct_pdf_w * position_pdf;
  const float w_camera = direct_pdf_w * INTEGRATOR_STATE(state, path, bdpt_d_vcm) +
                         emission_pdf_w * INTEGRATOR_STATE(state, path, bdpt_d_vc);
  return 1.0f / (1.0f + w_camera);
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

  /* The flat emitter CDF is proportional to triangle area, making this the complete position
   * density including emitter selection. A two-sided emitter samples either hemisphere with
   * equal probability. */
  const float side_pdf = (front && back) ? 0.5f : 1.0f;
  const float direct_pdf_a = kernel_data.integrator.distribution_pdf_triangles;
  const float cos_at_light = max(fabsf(dot(sd->N, sd->wi)), 1.0e-8f);
  const float emission_pdf_w = direct_pdf_a * side_pdf * cos_at_light * M_1_PI_F;
  const float w_camera = direct_pdf_a * INTEGRATOR_STATE(state, path, bdpt_d_vcm) +
                         emission_pdf_w * INTEGRATOR_STATE(state, path, bdpt_d_vc);
  return 1.0f / (1.0f + w_camera);
}

ccl_device_inline float bdpt_emission_mis_weight_lamp(KernelGlobals kg,
                                                      IntegratorState state,
                                                      const ccl_global KernelLight *klight,
                                                      const float3 ray_P,
                                                      const float3 ray_D,
                                                      const float distance)
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
  const float d_vcm = INTEGRATOR_STATE(state, path, bdpt_d_vcm) * sqr(distance) / cos_at_light;
  const float d_vc = INTEGRATOR_STATE(state, path, bdpt_d_vc) / cos_at_light;
  return 1.0f / (1.0f + direct_pdf_a * d_vcm + emission_pdf_w * d_vc);
}

/* Evaluate the compact reverse density used by recursive MIS. Actual endpoint contributions do a
 * full reciprocal shader reconstruction; duplicating an arbitrary shader graph at every recursive
 * PDF query causes pathological Metal compile time and register pressure. Swapping the fixed
 * direction here is exact for reciprocal closures with direction-independent mixture weights and
 * remains a variance-only approximation for layered directional selection weights. */
ccl_device_inline float bdpt_reverse_pdf(KernelGlobals kg,
                                         IntegratorState state,
                                         ccl_private ShaderData *sd,
                                         const float3 sampled_wo)
{
  const float3 original_wi = sd->wi;
  sd->wi = normalize(sampled_wo);
  BsdfEval reverse_eval;
  float roughness_squared = 0.0f;
  const float reverse_pdf = surface_shader_bsdf_eval(
      kg, state, sd, original_wi, &reverse_eval, SHADER_USE_MIS, roughness_squared);
  sd->wi = original_wi;
  return reverse_pdf;
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
  float w_light = bsdf_pdf / direct_pdf;

  float emission_position_pdf = 0.0f;
  float emission_side_pdf = 1.0f;
  float w_camera = 0.0f;
  if (ls->type == LIGHT_TRIANGLE) {
    emission_position_pdf = kernel_data.integrator.distribution_pdf_triangles;
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
    const float cos_camera = max(fabsf(dot(sd->N, ls->D)), 1.0e-8f);
    w_camera = position_pdf * cos_camera *
               (INTEGRATOR_STATE(state, path, bdpt_d_vcm) +
                INTEGRATOR_STATE(state, path, bdpt_d_vc) * reverse_pdf);
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
      const float cos_camera = max(fabsf(dot(sd->N, ls->D)), 1.0e-8f);
      w_camera = emission_pdf_w * cos_camera / direct_pdf *
                 (INTEGRATOR_STATE(state, path, bdpt_d_vcm) +
                  INTEGRATOR_STATE(state, path, bdpt_d_vc) * reverse_pdf);
    }
  }

  if (emission_position_pdf > 0.0f) {
    const float reverse_pdf = bdpt_reverse_pdf(kg, state, sd, ls->D);
    const float cos_camera = max(fabsf(dot(sd->N, ls->D)), 1.0e-8f);
    const float emission_to_direct = emission_position_pdf * emission_side_pdf * cos_camera /
                                     (M_PI_F * direct_pdf);
    w_camera = emission_to_direct *
               (INTEGRATOR_STATE(state, path, bdpt_d_vcm) +
                INTEGRATOR_STATE(state, path, bdpt_d_vc) * reverse_pdf);
  }

  return 1.0f / (1.0f + w_light + w_camera);
}

ccl_device_inline void bdpt_recursive_mis_after_hit(IntegratorState state,
                                                    const ccl_private ShaderData *sd)
{
  float d_vcm = INTEGRATOR_STATE(state, path, bdpt_d_vcm);
  float d_vc = INTEGRATOR_STATE(state, path, bdpt_d_vc);

  if (INTEGRATOR_STATE(state, path, bounce) == 0 && d_vcm == 0.0f) {
    float inverse_camera_pdf_w = 0.0f;
    if (kernel_data.cam.type == CAMERA_PERSPECTIVE && kernel_data.cam.aperturesize == 0.0f &&
        kernel_data.cam.interocular_offset == 0.0f && kernel_data.cam.num_motion_steps == 0 &&
        !kernel_data.cam.have_perspective_motion)
    {
      const ProjectionTransform raster_to_camera = kernel_data.cam.rastertocamera;
      const float3 image_origin = transform_perspective(
          &raster_to_camera, make_float3(0.0f, 0.0f, 0.0f));
      const float3 image_x = transform_perspective(
          &raster_to_camera, make_float3(1.0f, 0.0f, 0.0f));
      const float3 image_y = transform_perspective(
          &raster_to_camera, make_float3(0.0f, 1.0f, 0.0f));
      const float image_pixel_area = len(cross(image_x - image_origin, image_y - image_origin));
      const Transform world_to_camera = kernel_data.cam.worldtocamera;
      const float3 camera_D = normalize(
          transform_direction(&world_to_camera, INTEGRATOR_STATE(state, ray, D)));
      const float scale = image_origin.z / camera_D.z;
      const float3 image_P = camera_D * scale;
      const float image_distance2 = len_squared(image_P);
      const float cos_at_camera = fabsf(camera_D.z);
      inverse_camera_pdf_w = cos_at_camera * image_pixel_area /
                             max(image_distance2, 1.0e-20f);
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
      /* The orthographic endpoint is delta in direction and sampled in position. Express its
       * projected-area density in the solid-angle recurrence used below. */
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
    }
    else {
      /* Compact differentials are a conservative fallback for non-pinhole camera models. */
      const float camera_footprint = max(INTEGRATOR_STATE(state, ray, dD), 1.0e-8f);
      inverse_camera_pdf_w = sqr(camera_footprint);
    }
    d_vcm = kernel_integrator_state.bdpt_light_path_sample_ratio * inverse_camera_pdf_w;
    d_vc = 0.0f;
  }

  const float distance2 = sqr(max(sd->ray_length, 1.0e-10f));
  const float cos_fixed = max(fabsf(dot(sd->N, sd->wi)), 1.0e-8f);
  d_vcm *= distance2 / cos_fixed;
  d_vc /= cos_fixed;

  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vcm) = d_vcm;
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vc) = d_vc;
}

ccl_device_inline void bdpt_recursive_mis_after_scatter(IntegratorState state,
                                                        const int label,
                                                        const float cos_out,
                                                        const float forward_pdf,
                                                        const float reverse_pdf)
{
  float d_vcm = INTEGRATOR_STATE(state, path, bdpt_d_vcm);
  float d_vc = INTEGRATOR_STATE(state, path, bdpt_d_vc);

  if (label & LABEL_SINGULAR) {
    d_vcm = 0.0f;
    d_vc *= cos_out;
  }
  else {
    d_vc = cos_out / bdpt_safe_pdf(forward_pdf) * (d_vc * reverse_pdf + d_vcm);
    d_vcm = 1.0f / bdpt_safe_pdf(forward_pdf);
  }

  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vcm) = d_vcm;
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vc) = d_vc;
}

ccl_device_inline void bdpt_fill_light_vertex(ccl_private KernelBDPTVertex *stored_vertex,
                                              const ccl_private ShaderData *sd,
                                              const ccl_private Ray *ray,
                                              const Spectrum throughput,
                                              const int emitter_object,
                                              const int light_group,
                                              const float d_vcm,
                                              const float d_vc,
                                              const uint path_length,
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
  stored_vertex->light_group = light_group;
  stored_vertex->d_vcm = d_vcm;
  stored_vertex->d_vc = d_vc;
  stored_vertex->path_length = path_length;
  stored_vertex->flag = flag;
  stored_vertex->emitter_shader_flags = emitter_shader_flags;
}

ccl_device_inline void bdpt_store_light_vertex(KernelGlobals kg,
                                               const ccl_private ShaderData *sd,
                                               const ccl_private Ray *ray,
                                               const Spectrum throughput,
                                               const int emitter_object,
                                               const int light_group,
                                               const float d_vcm,
                                               const float d_vc,
                                               const uint path_length,
                                               const uint flag,
                                               const uint emitter_shader_flags,
                                               const float wavelength_rand)
{
  const uint slot = atomic_fetch_and_add_uint32(kernel_integrator_state.bdpt_vertex_count, 1);
  if (slot >= kernel_integrator_state.bdpt_vertex_capacity) {
    return;
  }

  ccl_private KernelBDPTVertex local_vertex;
  bdpt_fill_light_vertex(&local_vertex,
                         sd,
                         ray,
                         throughput,
                         emitter_object,
                         light_group,
                         d_vcm,
                         d_vc,
                         path_length,
                         flag,
                         emitter_shader_flags,
                         wavelength_rand);
  *(&kernel_integrator_state.bdpt_vertices[slot]) = local_vertex;
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
      *connection_jacobian = 1.0f / image_pixel_area;
    }
    else if (camera_type == CAMERA_PANORAMA) {
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

  BsdfEval light_eval;
  float roughness_squared = 0.0f;
  const uint emitter_shader_flags = (light_vertex->path_length == 2u) ?
                                        (light_vertex->emitter_shader_flags | SHADER_USE_MIS) :
                                        SHADER_USE_MIS;
  const float light_pdf = surface_shader_bsdf_eval(kg,
                                                   state,
                                                   light_sd,
                                                   light_incoming,
                                                   &light_eval,
                                                   emitter_shader_flags,
                                                   roughness_squared);
  if (!(light_pdf > 0.0f) || bsdf_eval_is_zero(&light_eval)) {
    return;
  }
  /* In the reciprocal orientation this is exactly the camera-to-light forward density, which is
   * the reverse density required by the light-side recursive MIS term. */
  const float reverse_pdf = light_pdf;

  const float cos_at_surface = max(fabsf(dot(light_sd->N, direction)), 1.0e-8f);
  const float camera_pdf_area = connection_jacobian * cos_at_surface;
  const float cos_to_light = max(fabsf(dot(light_sd->N, light_incoming)), 1.0e-8f);

  const float light_path_count = float(kernel_integrator_state.bdpt_light_path_count);
  const float light_path_sample_ratio = max(
      kernel_integrator_state.bdpt_light_path_sample_ratio, 1.0e-20f);
  const float w_light = (camera_pdf_area / light_path_sample_ratio) *
                        (light_vertex->d_vcm + light_vertex->d_vc * reverse_pdf);
  const float mis_weight = 1.0f / (1.0f + w_light);

  /* BsdfEval is f*cos(outgoing). The reciprocal evaluation above points toward the previous light
   * vertex and therefore carries cos_to_light; convert it to the camera-outgoing measure required
   * by this sensor splat. One cached map is reused for a camera batch, so its splat must represent
   * every sample in that batch. */
  const Spectrum sensor_eval = bsdf_eval_sum(&light_eval) * (cos_at_surface / cos_to_light);
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
  bool skip_self = true;
  shadow_ray.P = shadow_ray_offset(kg, light_sd, direction, &skip_self);
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

  IntegratorShadowState shadow_state = integrator_shadow_path_init(
      kg, state, DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW, false);
#ifdef __VOLUME__
  integrator_state_copy_volume_stack_to_shadow(kg, shadow_state, state);
#endif
  integrator_state_write_shadow_ray(shadow_state, &shadow_ray);
  integrator_state_write_shadow_ray_self(shadow_state, &shadow_ray);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, render_pixel_index) =
      uint(kernel_integrator_state.bdpt_buffer_offset + pixel_x +
           pixel_y * kernel_integrator_state.bdpt_buffer_stride);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, sample) = iteration;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, rng_pixel) = hash_uint2(
      uint(pixel_x), uint(pixel_y));
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, rng_offset) = 0;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, transparent_bounce) = 0;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, volume_bounds_bounce) = 0;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, diffuse_bounce) = 1;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, glossy_bounce) = 0;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, transmission_bounce) = 0;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, bounce) = light_vertex->path_length - 1u;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, throughput) = contribution;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, lightgroup) = light_vertex->light_group + 1;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, visibility) = PATH_RAY_VISIBILITY_CAMERA;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, flag) = PATH_RAY_SURFACE_PASS;
  if (!(kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_TREE)) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, bsdf_eval_average) = average(
        bsdf_eval_sum(&light_eval));
  }
  if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, pass_diffuse_weight) = PackedSpectrum(
        bsdf_eval_pass_diffuse_weight(&light_eval));
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, pass_glossy_weight) = PackedSpectrum(
        bsdf_eval_pass_glossy_weight(&light_eval));
  }
}

/* Generate one complete light subpath. The pass deliberately uses the same intersection, shader
 * evaluation, closure sampling, volume attenuation/boundary handling and Russian roulette
 * conventions as camera paths. Volume scattering vertices are not yet connectible and terminate
 * this surface-only light-subpath strategy as a valid zero sample. */
ccl_device void integrator_bdpt_light_generate(KernelGlobals kg,
                                               IntegratorState state,
                                               const uint light_path_index,
                                               const uint iteration,
                                               const uint batch_samples)
{
  if (!bdpt_camera_supported()) {
    return;
  }
  uint rng = lcg_init(hash_uint3(
      light_path_index, iteration, uint(kernel_data.integrator.seed) ^ 0x62647074u));
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
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vcm) = 1.0f;
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vc) = 0.0f;

  Ray ray ccl_optional_struct_init;
  Spectrum throughput;
  int emitter_object = OBJECT_NONE;
  int light_group = LIGHTGROUP_NONE;
  float emission_pdf = 0.0f;
  float direct_pdf = 0.0f;
  float emission_cosine = 1.0f;
  bool is_delta_emitter = false;
  bool is_finite_emitter = true;
  uint emitter_shader_flags = 0u;
  float emitter_max_bounces = FLT_MAX;
  const float time = lcg_step_float(&rng);
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
                             &emitter_max_bounces))
  {
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

  float d_vcm = direct_pdf / bdpt_safe_pdf(emission_pdf);
  float d_vc = is_delta_emitter ? 0.0f :
                                  (is_finite_emitter ? emission_cosine : 1.0f) /
                                      bdpt_safe_pdf(emission_pdf);
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vcm) = d_vcm;
  INTEGRATOR_STATE_WRITE(state, path, bdpt_d_vc) = d_vc;

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

  const uint cache_bounce = min(
      uint(lcg_step_float(&rng) * float(kernel_data.integrator.bdpt_max_bounces)),
      uint(kernel_data.integrator.bdpt_max_bounces - 1));
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
        return;
      }
      if (volume_event == PHOTON_VOLUME_SCATTERED) {
        /* Surface connections get a valid zero sample for this light path. Do not let an
         * unattenuated ray incorrectly reach the surface after a sampled medium collision. */
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

    const float cos_fixed = max(fabsf(dot(sd.N, sd.wi)), 1.0e-8f);
    if (bounce == 0 && !is_finite_emitter) {
      /* Infinite emitters sample a launch disk perpendicular to the ray. Its position density is
       * already in projected-area measure, so the first surface conversion has no distance term
       * (SmallVCM eq. 49). All later edges connect ordinary finite surface vertices. */
      d_vcm /= cos_fixed;
    }
    else {
      d_vcm *= sqr(max(sd.ray_length, 1.0e-10f)) / cos_fixed;
    }
    d_vc /= cos_fixed;

    if ((sd.runtime_flag & SR_BSDF_HAS_EVAL) &&
        !(sd.object_flag & SD_OBJECT_SHADOW_CATCHER))
    {
      if (uint(bounce) == cache_bounce) {
        bdpt_store_light_vertex(kg,
                                &sd,
                                &ray,
                                throughput,
                                emitter_object,
                                light_group,
                                d_vcm,
                                d_vc,
                                uint(bounce + 2),
                                INTEGRATOR_STATE(state, path, flag),
                                emitter_shader_flags,
                                light_wavelength_rand);
      }

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
    float2 sampled_roughness = one_float2();
    float eta = 1.0f;
    float avg_roughness_squared = 0.0f;
    const int label = surface_shader_bsdf_sample_closure(
        kg, &sd, sc, rand_bsdf, &eval, &wo, &pdf, &sampled_roughness, &eta, avg_roughness_squared);
    if (!(pdf > 0.0f) || bsdf_eval_is_zero(&eval)) {
      break;
    }

    throughput *= bsdf_eval_sum(&eval) / pdf;
    if (label & LABEL_TRANSMIT) {
      /* Cycles evaluates transmissive closures in radiance transport mode. A light subpath is
       * transported in the adjoint (importance) direction, whose refractive BSDF differs by the
       * squared relative IOR. This cancels over entry/exit pairs, while remaining essential for
       * nested dielectrics and connections whose endpoints lie in different media. */
      throughput *= sqr(eta);
    }
    if (!isfinite_safe(throughput)) {
      break;
    }

    if (label & LABEL_TRANSPARENT) {
      path_state_next(kg, state, label, sd.runtime_flag);
      ray.P = sd.P;
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

    const float reverse_pdf = (label & LABEL_SINGULAR) ? pdf :
                                                           bdpt_reverse_pdf(kg, state, &sd, wo);
    const float cos_out = max(fabsf(dot(sd.N, normalize(wo))), 1.0e-8f);
    if (label & LABEL_SINGULAR) {
      d_vcm = 0.0f;
      d_vc *= cos_out;
    }
    else {
      d_vc = cos_out / bdpt_safe_pdf(pdf) * (d_vc * reverse_pdf + d_vcm);
      d_vcm = 1.0f / bdpt_safe_pdf(pdf);
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
      const float continuation = min(saturatef(reduce_max(throughput)), 0.95f);
      if (lcg_step_float(&rng) >= continuation) {
        break;
      }
      throughput /= continuation;
      d_vcm /= continuation;
      d_vc /= continuation;
    }
  }

}

/* Sensor connections are deliberately isolated from light generation. The manifold solver has a
 * large live working set; keeping it in a separate Metal kernel avoids inflating compile time and
 * register pressure for every emitted light path. The compact cache already contains one
 * uniformly selected potential bounce per path, so it is also an unbiased sensor reservoir. */
ccl_device void integrator_bdpt_sensor_connect(KernelGlobals kg,
                                               IntegratorState state,
                                               const uint vertex_index,
                                               const uint iteration,
                                               const uint batch_samples)
{
  if (!kernel_integrator_state.bdpt_vertex_count || !kernel_integrator_state.bdpt_vertices) {
    return;
  }
  const uint vertex_count = min(*kernel_integrator_state.bdpt_vertex_count,
                                kernel_integrator_state.bdpt_vertex_capacity);
  if (vertex_index >= vertex_count) {
    return;
  }

  KernelBDPTVertex light_vertex = kernel_integrator_state.bdpt_vertices[vertex_index];
  uint rng = lcg_init(
      hash_uint3(vertex_index, iteration, uint(kernel_data.integrator.seed) ^ 0x73656e73u));
  photon_state_init(state, rng, iteration);
  INTEGRATOR_STATE_WRITE(state, path, flag) = light_vertex.flag;
  INTEGRATOR_STATE_WRITE(state, path, bounce) = light_vertex.path_length - 1u;

  ShaderData light_sd;
  if (!bdpt_setup_light_vertex(kg, state, &light_vertex, &light_sd)) {
    return;
  }
  const float2 rand_lens = make_float2(lcg_step_float(&rng), lcg_step_float(&rng));
  bdpt_connect_light_vertex_to_camera(kg,
                                      state,
                                      &light_vertex,
                                      &light_sd,
                                      uint(kernel_data.integrator.bdpt_max_bounces),
                                      iteration,
                                      batch_samples,
                                      rand_lens);
}

CCL_NAMESPACE_END
