/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/path_state.h"
#include "kernel/integrator/surface_shader.h"

#include "kernel/film/data_passes.h"
#include "kernel/film/denoising_passes.h"
#include "kernel/film/light_passes.h"

#include "kernel/light/sample.h"

#include "kernel/geom/motion_triangle.h"
#include "kernel/geom/triangle.h"

#include "kernel/integrator/guiding.h"
#include "kernel/integrator/nested_dielectrics.h"
#include "kernel/integrator/shadow_linking.h"
#include "kernel/integrator/subsurface.h"
#include "kernel/integrator/volume_stack.h"
#if defined(__BDPT__) && !defined(__KERNEL_METAL__)
/* Light-cache transport. The Metal kernels include it once, ahead of all shading kernels. */
#  include "kernel/integrator/bidirectional.h"
#  include "kernel/integrator/vertex_merging.h"
#  include "kernel/integrator/guiding_gpu.h"
#endif

#include "kernel/types.h"
#include "util/math_intersect.h"

CCL_NAMESPACE_BEGIN

ccl_device_forceinline void integrate_surface_shader_setup(KernelGlobals kg,
                                                           ConstIntegratorState state,
                                                           ccl_private ShaderData *sd)
{
  Intersection isect ccl_optional_struct_init;
  integrator_state_read_isect(state, &isect);

  Ray ray ccl_optional_struct_init;
  integrator_state_read_ray(state, &ray);

  shader_setup_from_ray(kg, sd, &ray, &isect);

#ifdef __SPECTRAL__
  shader_setup_wavelength(kg, sd, state);
#endif
}

ccl_device_forceinline float3 integrate_surface_ray_offset(KernelGlobals kg,
                                                           const ccl_private ShaderData *sd,
                                                           const float3 ray_P,
                                                           const float3 ray_D)
{
  /* No ray offset needed for other primitive types. */
  if (!(sd->type & PRIMITIVE_TRIANGLE)) {
    return ray_P;
  }

  /* Self intersection tests already account for the case where a ray hits the
   * same primitive. However precision issues can still cause neighboring
   * triangles to be hit. Here we test if the ray-triangle intersection with
   * the same primitive would miss, implying that a neighboring triangle would
   * be hit instead.
   *
   * This relies on triangle intersection to be watertight, and the object inverse
   * object transform to match the one used by ray intersection exactly.
   *
   * Potential improvements:
   * - It appears this happens when either barycentric coordinates are small,
   *   or dot(sd->Ng, ray_D)  is small. Detect such cases and skip test?
   * - Instead of ray offset, can we tweak P to lie within the triangle?
   */

  /* TODO: Investigate if there are better ray offsetting algorithms for each BVH.
   * Cycles and Custom BVH triangle tests aren't numerically identical, meaning
   * this method isn't ideal for them. */

  float3 verts[3];
  if (sd->type == PRIMITIVE_TRIANGLE) {
    triangle_vertices(kg, sd->object, sd->prim, verts);
  }
  else {
    kernel_assert(sd->type == PRIMITIVE_MOTION_TRIANGLE);
    motion_triangle_vertices(kg, sd->object, sd->prim, sd->time, verts);
  }

  float3 local_ray_P = ray_P;
  float3 local_ray_D = ray_D;

  if (!(sd->object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    const Transform itfm = object_get_inverse_transform(kg, sd);
    local_ray_P = transform_point(&itfm, local_ray_P);
    local_ray_D = transform_direction(&itfm, local_ray_D);
  }

  if (ray_triangle_intersect_self(local_ray_P, local_ray_D, verts)) {
    return ray_P;
  }
  return ray_offset(ray_P, sd->Ng);
}

CCL_NAMESPACE_END
#include "kernel/light/coherent.h"
#include "kernel/light/coherent_history_kernel.h"
#include "kernel/light/coherent_specular.h"
CCL_NAMESPACE_BEGIN

ccl_device_forceinline bool integrate_surface_holdout(KernelGlobals kg,
                                                      ConstIntegratorState state,
                                                      ccl_private ShaderData *sd,
                                                      ccl_global float *ccl_restrict render_buffer)
{
  /* Write holdout transparency to render buffer and stop if fully holdout. */
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

  if (((sd->runtime_flag & SR_HOLDOUT) || (sd->object_flag & SD_OBJECT_HOLDOUT_MASK)) &&
      (path_flag & PATH_RAY_TRANSPARENT_BACKGROUND))
  {
    const Spectrum holdout_weight = surface_shader_apply_holdout(sd);
    const Spectrum throughput = INTEGRATOR_STATE(state, path, throughput);
    const float transparent = average(holdout_weight * throughput);
    film_write_holdout(kg, state, path_flag, transparent, render_buffer);
    if (isequal(holdout_weight, one_spectrum())) {
      return false;
    }
  }

  return true;
}

ccl_device_forceinline void integrate_surface_emission(KernelGlobals kg,
                                                       IntegratorState state,
                                                       const ccl_private ShaderData *sd,
                                                       ccl_global float *ccl_restrict
                                                           render_buffer)
{
  const PathRayVisibility path_visibility = INTEGRATOR_STATE(state, path, visibility);
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

#ifdef __LIGHT_LINKING__
  if (!(path_visibility & PATH_RAY_VISIBILITY_CAMERA) &&
      !light_link_object_match(kg, light_link_receiver_forward(kg, state), sd->object))
  {
    return;
  }
#endif

#ifdef __SHADOW_LINKING__
  /* Indirect emission of shadow-linked emissive surfaces is done via shadow rays to dedicated
   * light sources. */
  if (kernel_data.kernel_features & KERNEL_FEATURE_SHADOW_LINKING) {
    if (!(path_visibility & PATH_RAY_VISIBILITY_CAMERA) &&
        kernel_data_fetch(objects, sd->object).shadow_set_membership != LIGHT_LINK_MASK_ALL)
    {
      return;
    }
  }
#endif

  /* Evaluate emissive closure. */
  const Spectrum L = surface_shader_emission(sd) * polarization_emission_weight(kg, state);

  float mis_weight;
#ifdef __BDPT__
  mis_weight = bdpt_enabled_for_emission(kg, state) ?
                   bdpt_emission_mis_weight_surface(kg, state, sd) :
                   light_sample_mis_weight_forward_surface(
                       kg, state, path_visibility, path_flag, sd);
#else
  mis_weight = light_sample_mis_weight_forward_surface(kg, state, path_visibility, path_flag, sd);
#endif

#ifdef __BDPT__
  if ((sd->shader_flag & (SD_MIS_FRONT | SD_MIS_BACK)) &&
      bdpt_volume_sensor_owns_camera_path(kg, state))
  {
    mis_weight = 0.0f;
  }
#endif

  guiding_record_surface_emission(kg, state, L, mis_weight);
  film_write_surface_emission(
      kg, state, L, mis_weight, render_buffer, object_lightgroup(kg, sd->object), sd->P);
}

ccl_device int integrate_surface_ray_portal(KernelGlobals kg,
                                            IntegratorState state,
                                            ccl_private ShaderData *sd,
                                            const ccl_private ShaderClosure *sc)
{
  const ccl_private RayPortalClosure *pc = (const ccl_private RayPortalClosure *)sc;

  float sum_sample_weight = 0.0f;
  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];

    if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
      sum_sample_weight += sc->sample_weight;
    }
  }
  if (sum_sample_weight <= 0.0f) {
    return LABEL_NONE;
  }

  if (len_squared(sd->P - pc->P) > 1e-9f) {
    /* if the ray origin is changed, unset the current object,
     * so we can potentially hit the same polygon again */
    INTEGRATOR_STATE_WRITE(state, isect, object) = OBJECT_NONE;
    INTEGRATOR_STATE_WRITE(state, ray, P) = pc->P;
  }
  else {
    INTEGRATOR_STATE_WRITE(state, ray, P) = integrate_surface_ray_offset(kg, sd, pc->P, pc->D);
  }
  INTEGRATOR_STATE_WRITE(state, ray, D) = pc->D;
  INTEGRATOR_STATE_WRITE(state, ray, tmin) = 0.0f;
  INTEGRATOR_STATE_WRITE(state, ray, tmax) = FLT_MAX;
#ifdef __RAY_DIFFERENTIALS__
  INTEGRATOR_STATE_WRITE(state, ray, dP) = differential_make_compact(sd->dP);
#endif

  const float pick_pdf = pc->sample_weight / sum_sample_weight;
  INTEGRATOR_STATE_WRITE(state, path, throughput) *= pc->weight / pick_pdf;

  const int label = LABEL_TRANSMIT | LABEL_RAY_PORTAL;
  path_state_next(kg, state, label, sd->runtime_flag);

  return label;
}

/* Branch off a shadow path and initialize common part of it.
 * THe common is between the surface shading and configuration of a special shadow ray for the
 * shadow linking. */
ccl_device_inline IntegratorShadowState
integrate_direct_light_shadow_init_common(KernelGlobals kg,
                                          IntegratorState state,
                                          const ccl_private Ray *ccl_restrict ray,
                                          const Spectrum bsdf_spectrum,
                                          const int light_group,
                                          const int mnee_vertex_count,
                                          const bool constant_light_shader,
                                          const bool bdpt_connection = false)
{
  const DeviceKernel next_kernel = (constant_light_shader) ?
                                       DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW :
                                       DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE;

  /* Branch off shadow kernel. */
  IntegratorShadowState shadow_state;
#ifdef __MNEE__
  if (mnee_vertex_count > 0) {
    /* Reuse shadow path that was already allocated by intersect_mnee. */
    shadow_state = integrator_state_get_mnee_shadow_state(state);
    integrator_shadow_path_next(
        shadow_state, DEVICE_KERNEL_INTEGRATOR_SHADOW_PATH_MNEE_PENDING, next_kernel);
  }
  else
#endif
      if (bdpt_connection)
  {
    shadow_state = integrator_bdpt_shadow_path_init(kg, state, next_kernel);
  }
  else {
    shadow_state = integrator_shadow_path_init(kg, state, next_kernel, false);
  }

#ifdef __VOLUME__
  /* Copy volume stack and enter/exit volume. */
  integrator_state_copy_volume_stack_to_shadow(kg, shadow_state, state);
#endif
#ifdef __NESTED_DIELECTRICS__
  integrator_state_copy_medium_stack_to_shadow(kg, shadow_state, state);
#endif

  /* Write shadow ray and associated state to global memory. */
  integrator_state_write_shadow_ray(shadow_state, ray);
  integrator_state_write_shadow_ray_self(shadow_state, ray);

  /* Copy state from main path to shadow path. */
  const Spectrum unlit_throughput = INTEGRATOR_STATE(state, path, throughput);
  const Spectrum throughput = unlit_throughput * bsdf_spectrum;

  if (!(kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_TREE)) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, bsdf_eval_average) = average(bsdf_spectrum);
  }

  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, render_pixel_index) = INTEGRATOR_STATE(
      state, path, render_pixel_index);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, rng_offset) = INTEGRATOR_STATE(
      state, path, rng_offset);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, rng_pixel) = INTEGRATOR_STATE(
      state, path, rng_pixel);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, sample) = INTEGRATOR_STATE(
      state, path, sample);

  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, transparent_bounce) = INTEGRATOR_STATE(
      state, path, transparent_bounce);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, volume_bounds_bounce) = INTEGRATOR_STATE(
      state, path, volume_bounds_bounce);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, glossy_bounce) = INTEGRATOR_STATE(
      state, path, glossy_bounce);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, throughput) = throughput;

  if ((kernel_data.kernel_features & KERNEL_FEATURE_NODE_PORTAL)) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, portal_bounce) = INTEGRATOR_STATE(
        state, path, portal_bounce);
  }

#ifdef __MNEE__
  if (mnee_vertex_count > 0) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, transmission_bounce) =
        INTEGRATOR_STATE(state, path, transmission_bounce) + mnee_vertex_count - 1;
    INTEGRATOR_STATE_WRITE(shadow_state,
                           shadow_path,
                           diffuse_bounce) = INTEGRATOR_STATE(state, path, diffuse_bounce) + 1;
    INTEGRATOR_STATE_WRITE(shadow_state,
                           shadow_path,
                           bounce) = INTEGRATOR_STATE(state, path, bounce) + mnee_vertex_count;
  }
  else
#endif
  {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, transmission_bounce) = INTEGRATOR_STATE(
        state, path, transmission_bounce);
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, diffuse_bounce) = INTEGRATOR_STATE(
        state, path, diffuse_bounce);
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, bounce) = INTEGRATOR_STATE(
        state, path, bounce);
  }

  /* Write Light-group, +1 as light-group is int but we need to encode into a uint8_t. */
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, lightgroup) = light_group + 1;

#if defined(__PATH_GUIDING__)
  if ((kernel_data.kernel_features & KERNEL_FEATURE_PATH_GUIDING)) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, unlit_throughput) = unlit_throughput;
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, path_segment) = INTEGRATOR_STATE(
        state, guiding, path_segment);
    INTEGRATOR_STATE(shadow_state, shadow_path, guiding_light_linking_mis_weight) = 0.0f;
  }
#endif

  return shadow_state;
}

/* Path tracing: sample point on light and evaluate light shader, then
 * queue shadow ray to be traced. */
template<uint64_t node_feature_mask>
#if defined(__KERNEL_HIP__)
/* Inlining the function makes gfx1102 crash rendering principled_bsdf_bevel_emission_137420.blend
 * using SDK 7.2.1. */
ccl_device_noinline
#elif defined(__KERNEL_GPU__)
ccl_device_forceinline
#else
/* MSVC has very long compilation time (x20) if we force inline this function */
ccl_device
#endif
    ShaderEvalResult
    integrate_surface_direct_light(KernelGlobals kg,
                                   IntegratorState state,
                                   ccl_private ShaderData *sd,
                                   const ccl_private RNGState *rng_state)
{
  /* Test if there is a light or BSDF that needs direct light. */
  if (!(kernel_data.integrator.use_direct_light && (sd->runtime_flag & SR_BSDF_HAS_EVAL))) {
    return SHADER_EVAL_EMPTY;
  }

  LightSample ls ccl_optional_struct_init;
  int mnee_vertex_count = 0;  // NOLINT

#ifdef __MNEE__
  if ((kernel_data.kernel_features & KERNEL_FEATURE_MNEE) &&
      (INTEGRATOR_STATE(state, path, mnee) & PATH_MNEE_SAMPLED))
  {
    /* MNEE already sampled a light and caustics casters. */
    integrator_state_read_mnee(state, &ls, &mnee_vertex_count);
  }
  else
#endif
  {
    /* Sample position on a light. */
    const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
    const uint bounce = INTEGRATOR_STATE(state, path, bounce);
    const float3 rand_light = path_state_rng_3D(kg, rng_state, PRNG_LIGHT);

    if (!light_sample_from_position(kg,
                                    rand_light,
                                    sd->time,
                                    sd->P,
                                    sd->N,
                                    light_link_receiver_nee(kg, sd),
                                    sd->runtime_flag,
                                    bounce,
                                    path_flag,
                                    &ls))
    {
      return SHADER_EVAL_EMPTY;
    }
  }

  kernel_assert(ls.pdf != 0.0f);

  if (kernel_data.integrator.coherent_specular_enabled &&
      coherent_detector_eligible(sd) && ls.type == LIGHT_POINT &&
      coherent_history_owned_direct_source(kg, ls.prim))
  {
    /* The deterministic detector estimator owns this direct diagonal. */
    return SHADER_EVAL_EMPTY;
  }

  const bool is_transmission = dot(ls.D, sd->N) < 0.0f;

  if (ls.prim != PRIM_NONE && ls.prim == sd->prim && ls.object == sd->object) {
    /* Skip self intersection if light direction lies in the same hemisphere as the geometric
     * normal. */
    if (dot(ls.D, is_transmission ? -sd->Ng : sd->Ng) > 0.0f) {
      return SHADER_EVAL_EMPTY;
    }
  }

#ifdef __MNEE__
  /* On a caustic caster, a caustic light's contribution is delivered to receivers by
   * MNEE and does not need to be computed again here. */
  if (kernel_data.kernel_features & KERNEL_FEATURE_MNEE) {
    if (mnee_vertex_count == 0 && is_transmission &&
        (sd->object_flag & SD_OBJECT_CAUSTICS_CASTER) && ls.type != LIGHT_TRIANGLE &&
        kernel_data_fetch(lights, ls.prim).use_caustics)
    {
      return SHADER_EVAL_EMPTY;
    }
  }
#endif

  /* Evaluate constant part of light shader, rest will optionally be done in another kernel. */
  Spectrum light_shader_eval ccl_optional_struct_init;
  const bool is_constant_light_shader = light_sample_shader_eval_nee_constant(
      kg, ls.shader, ls.prim, ls.type != LIGHT_TRIANGLE, light_shader_eval);

  /* Evaluate BSDF. */
  BsdfEval bsdf_eval ccl_optional_struct_init;
  float avg_roughness_squared = 0.0f;
  const float bsdf_pdf = surface_shader_bsdf_eval(
      kg, state, sd, ls.D, &bsdf_eval, ls.shader, avg_roughness_squared);
  if (polarization_enabled(kg) && mnee_vertex_count == 0 && !bsdf_eval_is_zero(&bsdf_eval)) {
    const auto sensitivity = polarization_surface_transport(kg, sd, ls.D,
        polarization_path_read(state), true, nullptr, bsdf_eval_sum(&bsdf_eval), false, ls.shader);
    bsdf_eval_mul(&bsdf_eval, sensitivity.value[0]);
  }
  /* The legacy direct-group estimator has a different path partition. In
   * specular-connection mode only declared detector endpoints receive a
   * coherent field; all other NEE remains ordinary incoherent radiometry. */
  const bool coherent_source = (kernel_data.kernel_features & KERNEL_FEATURE_COHERENT_DIRECT) &&
                               !kernel_data.integrator.coherent_specular_enabled &&
                               mnee_vertex_count == 0 && ls.type == LIGHT_POINT &&
                               kernel_data_fetch(lights, ls.prim).coherence_group != 0 &&
                               kernel_data_fetch(lights, ls.prim).coherence_length > 0.0f;
  Spectrum coherent_scale = one_spectrum();
  if (coherent_source) {
    coherent_scale = coherent_direct_light_scale(kg, state, sd, &ls);
    if (sd->runtime_flag & SR_CACHE_MISS) {
      return SHADER_EVAL_CACHE_MISS;
    }
  }
#ifdef __KERNEL_METAL__
  const Spectrum guiding_scattering_throughput = INTEGRATOR_STATE(state, path, throughput) *
                                                 bsdf_eval_sum(&bsdf_eval);
#endif

  Ray ray ccl_optional_struct_init;

#ifdef __MNEE__
  if (mnee_vertex_count > 0) {
    light_shader_eval *= integrator_state_read_mnee_throughput(state);
    bsdf_eval_mul(&bsdf_eval, light_shader_eval);

    if (bsdf_eval_is_zero(&bsdf_eval)) {
      return SHADER_EVAL_EMPTY;
    }

    integrator_state_read_mnee_ray(state, &ls, &ray);
  }
  else
#endif /* __MNEE__ */
  {
    float mis_weight;
#ifdef __BDPT__
    mis_weight = bdpt_enabled_for_surface_path(kg, state) ?
                     bdpt_nee_mis_weight(kg, state, sd, &ls, bsdf_pdf) :
                     light_sample_mis_weight_nee(kg, ls.pdf, bsdf_pdf);
    if (sd->runtime_flag & SR_CACHE_MISS) {
      return SHADER_EVAL_CACHE_MISS;
    }
#else
    mis_weight = light_sample_mis_weight_nee(kg, ls.pdf, bsdf_pdf);
#endif
    /* Keep the ordinary radiometric MIS partition, including light subpaths.
     * Only NEE estimates the additional direct-source cross terms. Thus its
     * multiplier is w_NEE + (I_coherent / I_incoherent - 1), not w_NEE times
     * that ratio. The correction may be negative; clamping it would erase
     * destructive interference. For ordinary PT, point-source w_NEE is one. */
    Spectrum direct_weight = make_spectrum(mis_weight);
    if (coherent_source) {
      direct_weight = coherent_scale + make_spectrum(mis_weight - 1.0f);
    }
    bsdf_eval_mul(&bsdf_eval, light_shader_eval * ls.eval_fac / ls.pdf * direct_weight);

    /* Path termination for constant light shader. */
    if (is_constant_light_shader && !(kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_TREE)) {
      const float terminate = path_state_rng_light_termination(kg, rng_state);
      if (light_sample_terminate(kg, &bsdf_eval, terminate)) {
        return SHADER_EVAL_EMPTY;
      }
    }
    /* For non-constant light shader, probabilistic termination happens in
     * SHADE_LIGHT_NEE when the full contribution is known. */
    else if (bsdf_eval_is_zero(&bsdf_eval)) {
      return SHADER_EVAL_EMPTY;
    }

    /* Create shadow ray. */
    light_sample_to_surface_shadow_ray(kg, sd, &ls, &ray);

#ifdef __RAY_DIFFERENTIALS__
    /* Widen ray differences, with same logic as forward sampling to ensure
     * both MIS strategies converge to the same result. */
    ray.dD = bsdf_widen_dD(kg, INTEGRATOR_STATE(state, ray, dD), avg_roughness_squared);
#endif
  }

  if (ray.self.object != OBJECT_NONE) {
    ray.P = integrate_surface_ray_offset(kg, sd, ray.P, ray.D);
  }

#ifdef __BDPT__
  if (bdpt_volume_sensor_owns_camera_path(kg, state, 1) &&
      bdpt_volume_sensor_supports_light(kg, ls.type, ls.prim))
  {
    return SHADER_EVAL_EMPTY;
  }
#endif
  /* Branch off shadow kernel. */
  IntegratorShadowState shadow_state = integrate_direct_light_shadow_init_common(
      kg,
      state,
      &ray,
      bsdf_eval_sum(&bsdf_eval),
      ls.group,
      mnee_vertex_count,
      is_constant_light_shader);

#ifdef __KERNEL_METAL__
  guiding_gpu_shadow_endpoint(shadow_state, sd->P);
  if (mnee_vertex_count == 0) {
    guiding_gpu_record_direct(shadow_state,
                              state,
                              sd->P,
                              ls.D,
                              guiding_scattering_throughput,
                              false,
                              guiding_gpu_surface_orientation(sd),
                              false,
                              ls.t);
    if (coherent_source && bdpt_enabled_for_surface_path(kg, state) && guiding_gpu_training()) {
      /* The combined NEE baseline and cross-term estimator can be signed.
       * Do not train a positive radiance proposal with this signed observation.
       * Ordinary light-subpath observations continue to train the baseline. */
      INTEGRATOR_STATE_WRITE(shadow_state, shadow_gpu_guiding, history_head) = ~0u;
      INTEGRATOR_STATE_WRITE(shadow_state, shadow_gpu_guiding, direct_record_index) = ~0u;
    }
  }
#endif

  if (is_transmission) {
    path_media_enter_exit<true>(kg, shadow_state, sd);
  }

  uint32_t shadow_flag = INTEGRATOR_STATE(state, path, flag);

  if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES) {
    PackedSpectrum pass_diffuse_weight;
    PackedSpectrum pass_glossy_weight;

    if (shadow_flag & PATH_RAY_ANY_PASS) {
      /* Indirect bounce, use weights from earlier surface or volume bounce. */
      pass_diffuse_weight = INTEGRATOR_STATE(state, path, pass_diffuse_weight);
      pass_glossy_weight = INTEGRATOR_STATE(state, path, pass_glossy_weight);
    }
    else {
      /* Direct light, use BSDFs at this bounce. */
      shadow_flag |= PATH_RAY_SURFACE_PASS;
      pass_diffuse_weight = PackedSpectrum(bsdf_eval_pass_diffuse_weight(&bsdf_eval));
      pass_glossy_weight = PackedSpectrum(bsdf_eval_pass_glossy_weight(&bsdf_eval));
    }

    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, pass_diffuse_weight) = pass_diffuse_weight;
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, pass_glossy_weight) = pass_glossy_weight;
  }

  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, visibility) = INTEGRATOR_STATE(
      state, path, visibility);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, flag) = shadow_flag;

  return SHADER_EVAL_OK;
}

#ifdef __BDPT__
/* Connect the current camera vertex to one uniformly selected entry of the global light-vertex
 * cache. Every entry is a per-path reservoir sample over the connectible vertices that path
 * actually reached. Uniform cache sampling plus the stored reservoir support converts the
 * selected vertex into an unbiased estimate of the sum over light-subpath lengths. */
ccl_device_forceinline_transport bool integrate_surface_bidirectional(
    KernelGlobals kg,
    IntegratorState state,
    ccl_private ShaderData *sd,
    ccl_attr_maybe_unused const ccl_private RNGState *rng_state)
{
  if (!bdpt_enabled_for_surface_path(kg, state) || !(sd->runtime_flag & SR_BSDF_HAS_EVAL) ||
      !kernel_integrator_state.bdpt_vertices || !kernel_integrator_state.bdpt_vertex_count ||
      !bdpt_connections_enabled(kg))
  {
    return false;
  }

  const uint cache = kernel_integrator_state.bdpt_cache_count > 1 ?
                         uint(INTEGRATOR_STATE(state, path, sample)) -
                             kernel_integrator_state.bdpt_cache_start_sample :
                         0u;
  if (cache >= kernel_integrator_state.bdpt_cache_count) {
    atomic_fetch_and_or_uint32(&kernel_integrator_state.queue_counter->bdpt_error, 1u);
    return false;
  }
  const uint vertex_count = min(kernel_integrator_state.bdpt_vertex_count[cache],
                                kernel_integrator_state.bdpt_vertex_capacity);
  if (vertex_count == 0 || kernel_integrator_state.bdpt_light_path_count == 0) {
    return false;
  }

  const uint bounce = uint(INTEGRATOR_STATE(state, path, bounce));
  const float select = hash_uint3_to_float(INTEGRATOR_STATE(state, path, rng_pixel),
                                           uint(INTEGRATOR_STATE(state, path, sample)),
                                           bounce ^ 0x62647074u);
  const uint vertex_index = min(uint(select * float(vertex_count)), vertex_count - 1u);
  const uint storage_index = kernel_integrator_state.bdpt_vertex_indices
      [cache * kernel_integrator_state.bdpt_vertex_capacity + vertex_index];
  const ccl_global KernelBDPTVertex *light_vertex =
      &kernel_integrator_state.bdpt_vertices[storage_index];
  /* Ownership is attached to the light-side prefix ending at its marked detector.
   * The camera suffix may contain any number of unmarked surfaces. Filtering only
   * when the current camera vertex is a detector would count the same source
   * class again through such suffixes. */
  if (kernel_integrator_state.bdpt_coherent_history &&
      coherent_history_owned_candidate(
          kg, kernel_integrator_state.bdpt_coherent_history[storage_index],
          light_vertex->emitter_object))
  {
    return false;
  }
  /* Medium records have no triangle, UVs or surface closure. Their sensor strategy
   * is evaluated separately; never reconstruct one as a surface intersection. */
  if (light_vertex->type == PRIMITIVE_VOLUME) {
    return false;
  }
  const uint light_path_length = light_vertex->path_length & 0xffu;
  const uint light_selection_count = (light_vertex->path_length >> 8u) & 0xfffu;
  const uint transparent_bounce = INTEGRATOR_STATE(state, path, transparent_bounce) +
                                  (light_vertex->path_length >> 20u);

  if (light_selection_count == 0u ||
      (transparent_bounce > 0u &&
       transparent_bounce >= uint(kernel_data.integrator.transparent_max_bounce)) ||
      light_path_length + bounce + 1u > uint(kernel_data.integrator.max_bounce + 1))
  {
    return false;
  }

  const float3 delta = light_vertex->P - sd->P;
  const float distance2 = len_squared(delta);
  if (!(distance2 > 1.0e-12f)) {
    return false;
  }
  const float distance = sqrtf(distance2);
  const float3 direction = delta / distance;

  BsdfEval camera_eval;
  float camera_roughness_squared = 0.0f;
  const float camera_pdf = surface_shader_bsdf_eval(
      kg, state, sd, direction, &camera_eval, SHADER_USE_MIS, camera_roughness_squared);
  if (!(camera_pdf > 0.0f) || bsdf_eval_is_zero(&camera_eval)) {
    return false;
  }
  const float camera_reverse_pdf = bdpt_reverse_pdf(kg, state, sd, direction);
  if (sd->runtime_flag & SR_CACHE_MISS) {
    return false;
  }

  packed_normal packed_incoming;
  packed_incoming.value = light_vertex->incoming;
  const float3 light_incoming = packed_incoming.decode();

  Ray light_ray ccl_optional_struct_init;
  light_ray.P = light_vertex->P - light_incoming;
  light_ray.D = light_incoming;
  light_ray.tmin = 0.0f;
  light_ray.tmax = 1.0f;
  light_ray.time = photon_unpack_time(light_vertex->time_wavelength);
#  ifdef __RAY_DIFFERENTIALS__
  light_ray.dP = differential_zero_compact();
  light_ray.dD = differential_zero_compact();
#  endif

  Intersection light_isect;
  light_isect.t = 1.0f;
  light_isect.u = light_vertex->u;
  light_isect.v = light_vertex->v;
  light_isect.prim = light_vertex->prim;
  light_isect.object = light_vertex->object;
  light_isect.type = light_vertex->type;

  ShaderData light_sd;
  shader_setup_from_ray(kg, &light_sd, &light_ray, &light_isect);
#  ifdef __SPECTRAL__
  light_sd.rand_wavelength = photon_unpack_wavelength_rand(light_vertex->time_wavelength);
#  endif
  bdpt_vertex_restore_medium(light_vertex->light_group, &light_sd);
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE>(
      kg, state, &light_sd, nullptr, PATH_RAY_VISIBILITY_GLOSSY, light_vertex->flag);
  if (light_sd.runtime_flag & SR_CACHE_MISS) {
    sd->runtime_flag |= SR_CACHE_MISS;
    return false;
  }
  surface_shader_prepare_closures(kg, state, &light_sd, PATH_RAY_VISIBILITY_GLOSSY);

  BsdfEval light_eval;
  float light_roughness_squared = 0.0f;
  const uint emitter_shader_flags = (light_path_length == 2u) ?
                                        (light_vertex->emitter_shader_flags | SHADER_USE_MIS) :
                                        SHADER_USE_MIS;
  const float light_pdf = surface_shader_bsdf_eval(kg,
                                                   state,
                                                   &light_sd,
                                                   -direction,
                                                   &light_eval,
                                                   emitter_shader_flags,
                                                   light_roughness_squared,
                                                   true);
  if (!(light_pdf > 0.0f) || bsdf_eval_is_zero(&light_eval)) {
    return false;
  }

  /* Rebuild the light vertex in the reciprocal orientation. Cycles closures may bake
   * direction-dependent layering data during shader evaluation, so swapping ShaderData::wi only
   * is not sufficient for Principled and arbitrary node graphs. */
  light_ray.P = light_vertex->P - direction;
  light_ray.D = direction;
  shader_setup_from_ray(kg, &light_sd, &light_ray, &light_isect);
#  ifdef __SPECTRAL__
  light_sd.rand_wavelength = photon_unpack_wavelength_rand(light_vertex->time_wavelength);
#  endif
  bdpt_vertex_restore_medium(light_vertex->light_group, &light_sd);
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE>(
      kg, state, &light_sd, nullptr, PATH_RAY_VISIBILITY_GLOSSY, light_vertex->flag);
  if (light_sd.runtime_flag & SR_CACHE_MISS) {
    sd->runtime_flag |= SR_CACHE_MISS;
    return false;
  }
  surface_shader_prepare_closures(kg, state, &light_sd, PATH_RAY_VISIBILITY_GLOSSY);

  BsdfEval light_adjoint_eval;
  float light_adjoint_roughness_squared = 0.0f;
  const float light_reverse_pdf = surface_shader_bsdf_eval(kg,
                                                           state,
                                                           &light_sd,
                                                           -light_incoming,
                                                           &light_adjoint_eval,
                                                           emitter_shader_flags,
                                                           light_adjoint_roughness_squared);
  if (!(light_reverse_pdf > 0.0f) || bsdf_eval_is_zero(&light_adjoint_eval)) {
    return false;
  }

  const float cos_camera = max(fabsf(dot(sd->Ng, direction)), 1.0e-8f);
  const float cos_light = max(fabsf(dot(light_sd.Ng, -direction)), 1.0e-8f);
  const float camera_pdf_area = camera_pdf * cos_light / distance2;
  const float light_pdf_area = light_pdf * cos_camera / distance2;
  /* The cache contains one reservoir-selected vertex per light path that reached a connectible
   * event. K*n/N is the exact global-selection support, where n is this path's actual number of
   * candidates. Unlike a configured maximum-bounce factor, it does not amplify sparse splats from
   * short paths. */
  const float cache_scale = float(vertex_count) * float(light_selection_count) /
                            float(kernel_integrator_state.bdpt_light_path_count);

  const float selection_ratio = light_path_length == 2u ?
      bdpt_light_selection_ratio(kg,
                                 light_vertex->emitter_distribution,
                                 light_vertex->emitter_P,
                                 light_sd.P,
                                 light_sd.time,
                                 light_sd.P,
                                 light_sd.N,
                                 light_sd.runtime_flag,
                                 light_sd.object) : 1.0f;
  /* Either end of the connection can also be merged with the vertex next to it. */
  const BDPTMISWeight w_light =
      BDPTMISWeight(camera_pdf_area) *
      (BDPTMISWeight::from_encoded(light_vertex->d_vcm) * selection_ratio +
       BDPTMISWeight::from_encoded(light_vertex->d_vc) * light_reverse_pdf +
       vcm_mis_vm_factor(kg));
  const BDPTMISWeight w_camera = BDPTMISWeight(light_pdf_area) *
                                 bdpt_camera_vertex_alternatives(kg, state, camera_reverse_pdf);
  /* Reservoir subsampling estimates the sum over all connectible light-path vertices.
   * Its inverse inclusion probability belongs in the contribution below. The recursive MIS
   * partition still describes those complete strategies, as do NEE and sensor connections.
   * Applying cache_scale here as well would make their weights fail to sum to one. */
  const float mis_weight = (BDPTMISWeight(1.0f) + w_light + w_camera).inverse();

  const Spectrum light_connection_eval = bdpt_transpose_surface_eval(
      bsdf_eval_sum(&light_adjoint_eval), light_sd.Ng, -light_incoming, -direction);
  const Spectrum spectral_weight = bdpt_light_vertex_spectral_weight(
      kg,
      state,
      light_vertex->time_wavelength,
      (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SPECTRAL) != 0u);
  Spectrum connection = Spectrum(light_vertex->throughput) * spectral_weight *
                        bsdf_eval_sum(&camera_eval) * light_connection_eval *
                        (cache_scale * mis_weight / distance2);
  if (polarization_enabled(kg)) {
    const auto camera_sensitivity = polarization_surface_transport(kg, sd, direction,
        polarization_path_read(state), true, nullptr, bsdf_eval_sum(&camera_eval), false);
    const auto light_sensitivity = polarization_surface_transport(kg, &light_sd, -light_incoming,
        camera_sensitivity, true, nullptr, bsdf_eval_sum(&light_adjoint_eval), false,
        emitter_shader_flags);
    connection *= polarization_spectrum_contract(light_sensitivity,
        polarization_unpack(kernel_integrator_state.bdpt_polarization[storage_index]));
  }
  if (!isfinite_safe(connection) || is_zero(connection)) {
    return false;
  }
  Ray ray ccl_optional_struct_init;
  bool skip_self = true;
  ray.P = shadow_ray_offset(kg, sd, direction, &skip_self);
  const float3 light_shadow_P = ray_offset(
      light_sd.P, dot(light_sd.Ng, -direction) >= 0.0f ? light_sd.Ng : -light_sd.Ng);
  const float3 shadow_delta = light_shadow_P - ray.P;
  const float shadow_distance = len(shadow_delta);
  if (!(shadow_distance > 1.0e-8f)) {
    return false;
  }
  ray.D = shadow_delta / shadow_distance;
  ray.tmin = 0.0f;
  ray.tmax = shadow_distance;
  ray.time = sd->time;
  ray.self.object = skip_self ? sd->object : OBJECT_NONE;
  ray.self.prim = skip_self ? sd->prim : PRIM_NONE;
  ray.self.light_object = light_vertex->emitter_object;
  ray.self.light_prim = PRIM_NONE;
#  ifdef __RAY_DIFFERENTIALS__
  ray.dP = differential_zero_compact();
  ray.dD = differential_zero_compact();
#  endif

  IntegratorShadowState shadow_state = integrate_direct_light_shadow_init_common(
      kg, state, &ray, connection, bdpt_vertex_light_group(light_vertex->light_group), 0, true, true);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, transparent_bounce) = transparent_bounce;
#  ifdef __NESTED_DIELECTRICS__
  if ((sd->shader_flag & SD_HAS_NESTED_PRIORITY) && dot(sd->Ng, direction) < 0.0f) {
    /* The connection leaves through the surface of a nested dielectric: it is inside or outside
     * of that medium from here on, as a transmitted direct light ray is. */
    path_media_enter_exit<true>(kg, shadow_state, sd);
  }
#  endif
  guiding_gpu_record_direct(shadow_state,
                            state,
                            sd->P,
                            direction,
                            INTEGRATOR_STATE(state, path, throughput) *
                                bsdf_eval_sum(&camera_eval),
                            false,
                            guiding_gpu_surface_orientation(sd),
                            true,
                            len(light_vertex->P - sd->P));

  uint32_t shadow_flag = INTEGRATOR_STATE(state, path, flag);
  if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES) {
    PackedSpectrum pass_diffuse_weight;
    PackedSpectrum pass_glossy_weight;
    if (shadow_flag & PATH_RAY_ANY_PASS) {
      pass_diffuse_weight = INTEGRATOR_STATE(state, path, pass_diffuse_weight);
      pass_glossy_weight = INTEGRATOR_STATE(state, path, pass_glossy_weight);
    }
    else {
      shadow_flag |= PATH_RAY_SURFACE_PASS;
      pass_diffuse_weight = PackedSpectrum(bsdf_eval_pass_diffuse_weight(&camera_eval));
      pass_glossy_weight = PackedSpectrum(bsdf_eval_pass_glossy_weight(&camera_eval));
    }
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, pass_diffuse_weight) = pass_diffuse_weight;
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, pass_glossy_weight) = pass_glossy_weight;
  }
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, visibility) = INTEGRATOR_STATE(
      state, path, visibility);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, flag) = shadow_flag;
  return true;
}
#endif

/* Path tracing: bounce off or through surface with new direction. */
ccl_device_forceinline int integrate_surface_bsdf_bssrdf_bounce(
    KernelGlobals kg,
    IntegratorState state,
    ccl_private ShaderData *sd,
    const ccl_private RNGState *rng_state)
{
  /* Sample BSDF or BSSRDF. */
  if (!(sd->runtime_flag & (SR_BSDF | SR_BSSRDF))) {
    return LABEL_NONE;
  }

  float3 rand_bsdf = path_state_rng_3D(kg, rng_state, PRNG_SURFACE_BSDF);
  const ccl_private ShaderClosure *sc = surface_shader_bsdf_bssrdf_pick(sd, &rand_bsdf);

#ifdef __SUBSURFACE__
  /* BSSRDF closure, we schedule subsurface intersection kernel. */
  if (CLOSURE_IS_BSSRDF(sc->type)) {
#  if defined(__BDPT__) || defined(__PHOTON_MAPPING__)
    if (kernel_data.integrator.use_bidirectional_path_tracing) {
      INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_BDPT_UNSUPPORTED;
      INTEGRATOR_STATE_WRITE(state, path, flag) &= ~PATH_RAY_BDPT_VOLUME_SENSOR;
    }
    if (kernel_data.integrator.use_photon_mapping) {
      INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_PHOTON_MAPPING_UNSUPPORTED;
      INTEGRATOR_STATE_WRITE(state, path, flag) &= ~PATH_RAY_PHOTON_MAPPING_RECEIVER;
    }
#  endif
    if (polarization_enabled(kg)) {
      polarization_path_write(state, polarization_depolarized(polarization_path_read(state)));
    }
    return subsurface_bounce(kg, state, sd, sc);
  }
#endif
  if (CLOSURE_IS_RAY_PORTAL(sc->type)) {
#ifdef __PHOTON_MAPPING__
    if (kernel_data.integrator.use_photon_mapping) {
      INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_PHOTON_MAPPING_UNSUPPORTED;
      INTEGRATOR_STATE_WRITE(state, path, flag) &= ~PATH_RAY_PHOTON_MAPPING_RECEIVER;
    }
#endif
    if (polarization_enabled(kg)) {
      polarization_path_write(state, polarization_depolarized(polarization_path_read(state)));
    }
    return integrate_surface_ray_portal(kg, state, sd, sc);
  }

  /* BSDF closure, sample direction. */
  float bsdf_pdf = 0.0f;
  float unguided_bsdf_pdf = 0.0f;
  BsdfEval bsdf_eval ccl_optional_struct_init;
  float3 bsdf_wo ccl_optional_struct_init;
  int label;

  float2 bsdf_sampled_roughness = make_float2(1.0f, 1.0f);
  float bsdf_eta = 1.0f;
  float mis_pdf = 1.0f;

  float bsdf_avg_roughness_squared = 0.0f;

#ifdef __KERNEL_METAL__
  if (kernel_data.integrator.use_surface_guiding && kernel_integrator_state.guiding_capacity > 0) {
    const float rand_guiding = path_state_rng_1D(kg, rng_state, PRNG_SURFACE_BSDF_GUIDING);
    const float3 rand_resampling = path_state_rng_3D(kg, rng_state, PRNG_SURFACE_RIS_GUIDING_0);
    label = surface_shader_bsdf_gpu_guided_sample_closure(kg,
                                                          sd,
                                                          sc,
                                                          rand_bsdf,
                                                          rand_guiding,
                                                          rand_resampling,
                                                          &bsdf_eval,
                                                          &bsdf_wo,
                                                          &bsdf_pdf,
                                                          &mis_pdf,
                                                          &unguided_bsdf_pdf,
                                                          &bsdf_sampled_roughness,
                                                          &bsdf_eta,
                                                          bsdf_avg_roughness_squared);
    if (!(bsdf_pdf > 0.0f) || bsdf_eval_is_zero(&bsdf_eval)) {
      return LABEL_NONE;
    }
  }
  else
#endif
#if defined(__PATH_GUIDING__) && PATH_GUIDING_LEVEL >= 4
      if (kernel_data.integrator.use_surface_guiding &&
          (kernel_data.kernel_features & KERNEL_FEATURE_PATH_GUIDING))
  {
    label = surface_shader_bsdf_guided_sample_closure(kg,
                                                      state,
                                                      sd,
                                                      sc,
                                                      rand_bsdf,
                                                      &bsdf_eval,
                                                      &bsdf_wo,
                                                      &bsdf_pdf,
                                                      &mis_pdf,
                                                      &unguided_bsdf_pdf,
                                                      &bsdf_sampled_roughness,
                                                      &bsdf_eta,
                                                      rng_state,
                                                      bsdf_avg_roughness_squared);

    if (bsdf_pdf == 0.0f || bsdf_eval_is_zero(&bsdf_eval)) {
      return LABEL_NONE;
    }

    INTEGRATOR_STATE_WRITE(state, path, unguided_throughput) *= bsdf_pdf / unguided_bsdf_pdf;
  }
  else
#endif
  {
    label = surface_shader_bsdf_sample_closure(kg,
                                               sd,
                                               sc,
                                               rand_bsdf,
                                               &bsdf_eval,
                                               &bsdf_wo,
                                               &bsdf_pdf,
                                               &bsdf_sampled_roughness,
                                               &bsdf_eta,
                                               bsdf_avg_roughness_squared);

    if (bsdf_pdf == 0.0f || bsdf_eval_is_zero(&bsdf_eval)) {
      return LABEL_NONE;
    }
    mis_pdf = bsdf_pdf;
    unguided_bsdf_pdf = bsdf_pdf;
  }

#ifdef __BDPT__
  /* Resolve reciprocal texture reads before changing the ray, throughput or training record.
   * A miss replays this scattering stage with the same random sample after tiles are loaded. */
  float reverse_pdf = 0.0f;
  const bool update_bdpt_mis = bdpt_enabled_for_surface_path(kg, state) &&
                               !(label & LABEL_TRANSPARENT);
  if (update_bdpt_mis) {
    const bool grating_delta = (label & LABEL_SINGULAR) &&
        surface_shader_has_physical_grating(sd) &&
        (sc->type == CLOSURE_BSDF_DIFFRACTION_SMOOTH_ID || bsdf_microfacet_has_delta(sc));
    if (grating_delta) {
      /* Keep the forward mass supplied by the sampled closure mixture. */
      reverse_pdf = bdpt_reverse_pdf(kg, state, sd, bsdf_wo, false, nullptr, true);
    }
    else {
      reverse_pdf = (label & LABEL_SINGULAR) ? mis_pdf : bdpt_reverse_pdf(kg, state, sd, bsdf_wo);
    }
    if (sd->runtime_flag & SR_CACHE_MISS) {
      return LABEL_CACHE_MISS;
    }
  }
#endif
#ifdef __KERNEL_METAL__
  if (kernel_data.integrator.use_surface_guiding && kernel_integrator_state.guiding_capacity > 0) {
    INTEGRATOR_STATE_WRITE(state, path, unguided_throughput) *= bsdf_pdf / unguided_bsdf_pdf;
  }
#endif

  if (label & LABEL_TRANSPARENT) {
#ifdef __BDPT__
    if (bdpt_enabled_for_emission(kg, state)) {
      bdpt_recursive_mis_undo_transparent_hit(state, sd);
    }
#endif
    /* Only need to modify start distance for transparent. */
    INTEGRATOR_STATE_WRITE(state, ray, tmin) = intersection_t_offset(sd->ray_length);
  }
  else {
    /* Setup ray with changed origin and direction. */
    const float3 D = normalize(bsdf_wo);
    INTEGRATOR_STATE_WRITE(state, ray, P) = integrate_surface_ray_offset(kg, sd, sd->P, D);
    INTEGRATOR_STATE_WRITE(state, ray, D) = D;
    INTEGRATOR_STATE_WRITE(state, ray, tmin) = 0.0f;
    INTEGRATOR_STATE_WRITE(state, ray, tmax) = FLT_MAX;
#ifdef __RAY_DIFFERENTIALS__
    INTEGRATOR_STATE_WRITE(state, ray, dP) = differential_make_compact(sd->dP);

    /* Widen ray differences, with same logic as NEE sampling to ensure
     * both MIS strategies converge to the same result. */
    const float dD = bsdf_widen_dD(
        kg, INTEGRATOR_STATE(state, ray, dD), bsdf_avg_roughness_squared);
    INTEGRATOR_STATE_WRITE(state, ray, dD) = dD;
#endif
  }

  if (polarization_enabled(kg)) {
    polarization_path_write(state, polarization_surface_transport(kg, sd, normalize(bsdf_wo),
        polarization_path_read(state), true, sc, bsdf_eval_sum(&bsdf_eval),
        (label & LABEL_SINGULAR) != 0));
  }

  /* Update throughput. */
  const Spectrum bsdf_weight = bsdf_eval_sum(&bsdf_eval) / bsdf_pdf;
  INTEGRATOR_STATE_WRITE(state, path, throughput) *= bsdf_weight;

#ifdef __KERNEL_METAL__
  GuidingVirtualDistance source_distance;
  guiding_gpu_record_bounce(
      state,
      sd->P,
      normalize(bsdf_wo),
      bsdf_pdf,
      label,
      false,
      guiding_gpu_surface_orientation(sd),
      source_distance.scatter_scale(
          label & LABEL_SINGULAR,
          safe_sqrtf(min(bsdf_sampled_roughness.x, bsdf_sampled_roughness.y)),
          bsdf_eta,
          dot(sd->N, sd->wi),
          dot(sd->N, normalize(bsdf_wo))));
#endif

#ifdef __BDPT__
  if (update_bdpt_mis) {
    const float cos_out = max(fabsf(dot(sd->Ng, normalize(bsdf_wo))), 1.0e-8f);
    bdpt_recursive_mis_after_scatter(kg, state, label, cos_out, mis_pdf, reverse_pdf);
  }
#endif

  if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES) {
    if (INTEGRATOR_STATE(state, path, bounce) == 0) {
      INTEGRATOR_STATE_WRITE(state, path, pass_diffuse_weight) = bsdf_eval_pass_diffuse_weight(
          &bsdf_eval);
      INTEGRATOR_STATE_WRITE(state, path, pass_glossy_weight) = bsdf_eval_pass_glossy_weight(
          &bsdf_eval);
    }
  }

  /* Update path state */
  if (!(label & LABEL_TRANSPARENT)) {
    const float min_ray_pdf = INTEGRATOR_STATE(state, path, min_ray_pdf);
    INTEGRATOR_STATE_WRITE(state, path, mis_ray_pdf) = mis_pdf;
    INTEGRATOR_STATE_WRITE(state, path, mis_origin_n) = sd->N;
    INTEGRATOR_STATE_WRITE(state, path, min_ray_pdf) = fminf(unguided_bsdf_pdf, min_ray_pdf);

#ifdef __LIGHT_LINKING__
    if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_LINKING) {
      INTEGRATOR_STATE_WRITE(state, path, mis_ray_object) = sd->object;
    }
#endif
  }

  path_state_next(kg, state, label, sd->runtime_flag);

#ifdef __PHOTON_MAPPING__
  if (kernel_data.integrator.use_photon_mapping && !(label & LABEL_TRANSPARENT)) {
    if (surface_shader_photon_mapping_receiver(kg, sc, sd->wi)) {
      INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_PHOTON_MAPPING_RECEIVER;
      INTEGRATOR_STATE_WRITE(state, path, flag) &= ~PATH_RAY_PHOTON_MAPPING_UNSUPPORTED;
    }
    else {
      INTEGRATOR_STATE_WRITE(state, path, flag) &= ~PATH_RAY_PHOTON_MAPPING_RECEIVER;
    }
  }
#endif

  guiding_record_surface_bounce(kg,
                                state,
                                bsdf_weight,
                                bsdf_pdf,
                                sd->N,
                                normalize(bsdf_wo),
                                bsdf_sampled_roughness,
                                bsdf_eta);

  return label;
}

#ifdef __NESTED_DIELECTRICS__
/* Continue the path through a surface that lies inside a medium of higher priority. Like the
 * boundary of a volume it is no bounce of any kind: only the start of the ray moves, so that
 * multiple importance sampling and the bidirectional recurrence see one uninterrupted edge.
 * Shadow rays pass through the same surfaces. */
template<uint64_t node_feature_mask>
ccl_device_forceinline int integrate_surface_nested_false_intersection(
    KernelGlobals kg,
    IntegratorState state,
    ccl_private ShaderData *sd)
{
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

  /* A medium the path enters here becomes the outer side of later surfaces, which needs the
   * index of refraction its shader evaluates to at the wavelength of the path. */
  if (!(sd->runtime_flag & SR_BACKFACING) && !(sd->shader_flag & SD_HAS_ONLY_VOLUME)) {
    surface_shader_eval<node_feature_mask>(
        kg, state, sd, nullptr, INTEGRATOR_STATE(state, path, visibility), path_flag);
    if (sd->runtime_flag & SR_CACHE_MISS) {
      return LABEL_CACHE_MISS;
    }
  }

  /* Path termination was decided at the intersection, as for any other surface. */
  const float continuation_probability = (path_flag & PATH_RAY_TERMINATE_ON_NEXT_SURFACE) ?
                                             0.0f :
                                             float(INTEGRATOR_STATE(
                                                 state, path, continuation_probability));
  if (continuation_probability == 0.0f) {
    return LABEL_NONE;
  }
  if (continuation_probability != 1.0f) {
    INTEGRATOR_STATE_WRITE(state, path, throughput) /= continuation_probability;
  }

  /* Pass through without counting a bounce, only sanity check in case self intersection gets
   * us stuck. */
  const uint32_t volume_bounds_bounce = INTEGRATOR_STATE(state, path, volume_bounds_bounce) + 1;
  INTEGRATOR_STATE_WRITE(state, path, volume_bounds_bounce) = volume_bounds_bounce;
  if (volume_bounds_bounce > VOLUME_BOUNDS_MAX) {
    return LABEL_NONE;
  }
  INTEGRATOR_STATE_WRITE(state, path, rng_offset) += PRNG_BOUNCE_NUM;

  /* Only modify start distance. */
  INTEGRATOR_STATE_WRITE(state, ray, tmin) = intersection_t_offset(sd->ray_length);

  path_media_enter_exit<false>(kg, state, sd);

  return LABEL_TRANSMIT | LABEL_TRANSPARENT;
}
#endif

#ifdef __VOLUME__
ccl_device_forceinline int integrate_surface_volume_only_bounce(IntegratorState state,
                                                                ccl_private ShaderData *sd)
{
  if (!path_state_volume_next(state)) {
    return LABEL_NONE;
  }

  /* Only modify start distance. */
  INTEGRATOR_STATE_WRITE(state, ray, tmin) = intersection_t_offset(sd->ray_length);

  return LABEL_TRANSMIT | LABEL_TRANSPARENT;
}
#endif

ccl_device_forceinline bool integrate_surface_terminate(IntegratorState state,
                                                        const uint32_t path_flag)
{
  const float continuation_probability = (path_flag & PATH_RAY_TERMINATE_ON_NEXT_SURFACE) ?
                                             0.0f :
                                             float(INTEGRATOR_STATE(
                                                 state, path, continuation_probability));
  if (continuation_probability == 0.0f) {
    return true;
  }
  if (continuation_probability != 1.0f) {
    INTEGRATOR_STATE_WRITE(state, path, throughput) /= continuation_probability;
  }

  return false;
}

#if defined(__AO__)
ccl_device_forceinline void integrate_surface_ao(KernelGlobals kg,
                                                 IntegratorState state,
                                                 const ccl_private ShaderData *ccl_restrict sd,
                                                 const ccl_private RNGState *ccl_restrict
                                                     rng_state)
{
  const PathRayVisibility path_visibility = INTEGRATOR_STATE(state, path, visibility);
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

  if (!(kernel_data.kernel_features & KERNEL_FEATURE_AO_ADDITIVE) &&
      !(path_visibility & PATH_RAY_VISIBILITY_CAMERA))
  {
    return;
  }

  /* Skip AO for paths that were split off for shadow catchers to avoid double-counting. */
  if (path_flag & PATH_RAY_SHADOW_CATCHER_PASS) {
    return;
  }

  const float2 rand_bsdf = path_state_rng_2D(kg, rng_state, PRNG_SURFACE_BSDF);

  float3 ao_N;
  const Spectrum ao_weight = surface_shader_ao(
      sd, kernel_data.integrator.ao_additive_factor, &ao_N);

  float3 ao_D;
  float ao_pdf;
  sample_cos_hemisphere(ao_N, rand_bsdf, &ao_D, &ao_pdf);

  bool skip_self = true;

  Ray ray ccl_optional_struct_init;
  ray.P = shadow_ray_offset(kg, sd, ao_D, &skip_self);
  ray.D = ao_D;
  if (skip_self) {
    ray.P = integrate_surface_ray_offset(kg, sd, ray.P, ray.D);
  }
  ray.tmin = 0.0f;
  ray.tmax = kernel_data.integrator.ao_bounces_distance;
  ray.time = sd->time;
  ray.self.object = (skip_self) ? sd->object : OBJECT_NONE;
  ray.self.prim = (skip_self) ? sd->prim : PRIM_NONE;
  ray.self.light_object = OBJECT_NONE;
  ray.self.light_prim = PRIM_NONE;
  ray.dP = differential_zero_compact();
  ray.dD = differential_zero_compact();

  /* Branch off shadow kernel. */
  IntegratorShadowState shadow_state = integrator_shadow_path_init(
      kg, state, DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW, true);

#  ifdef __VOLUME__
  /* Copy volume stack and enter/exit volume. */
  integrator_state_copy_volume_stack_to_shadow(kg, shadow_state, state);
#  endif
#  ifdef __NESTED_DIELECTRICS__
  integrator_state_copy_medium_stack_to_shadow(kg, shadow_state, state);
#  endif

  /* Write shadow ray and associated state to global memory. */
  integrator_state_write_shadow_ray(shadow_state, &ray);
  integrator_state_write_shadow_ray_self(shadow_state, &ray);

  /* Copy state from main path to shadow path. */
  const uint16_t bounce = INTEGRATOR_STATE(state, path, bounce);
  const uint16_t transparent_bounce = INTEGRATOR_STATE(state, path, transparent_bounce);
  const uint32_t shadow_flag = INTEGRATOR_STATE(state, path, flag) | PATH_RAY_SHADOW_FOR_AO;
  const Spectrum throughput = INTEGRATOR_STATE(state, path, throughput) * surface_shader_alpha(sd);

  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, render_pixel_index) = INTEGRATOR_STATE(
      state, path, render_pixel_index);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, rng_offset) = INTEGRATOR_STATE(
      state, path, rng_offset);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, rng_pixel) = INTEGRATOR_STATE(
      state, path, rng_pixel);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, sample) = INTEGRATOR_STATE(
      state, path, sample);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, visibility) = INTEGRATOR_STATE(
      state, path, visibility);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, flag) = shadow_flag;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, bounce) = bounce;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, transparent_bounce) = transparent_bounce;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, volume_bounds_bounce) = INTEGRATOR_STATE(
      state, path, volume_bounds_bounce);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, throughput) = throughput;

  if (kernel_data.kernel_features & KERNEL_FEATURE_AO_ADDITIVE) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, unshadowed_throughput) = ao_weight;
  }
}
#endif /* defined(__AO__) */

#ifdef __KERNEL_METAL_VISIBLE_SHADING__
/* The large surface integration stages are compiled once as Metal visible functions (see
 * `kernel.metal`), so the regular and ray-tracing surface kernels share them and they compile
 * concurrently with the kernels. Table indices are part of the host and kernel contract. */
enum MetalSurfaceStage {
  METAL_SURFACE_STAGE_DIRECT_LIGHT = 0,
  METAL_SURFACE_STAGE_BIDIRECTIONAL = 1,
  METAL_SURFACE_STAGE_BSDF_BSSRDF_BOUNCE = 2,
  METAL_SURFACE_STAGE_PHOTON_GATHER = 3,
  METAL_SURFACE_STAGE_COHERENT_SPECULAR = 4,
  METAL_SURFACE_STAGE_EMISSION = 5,
  METAL_SURFACE_STAGE_DATA_PASSES = 6,
  METAL_SURFACE_STAGE_DENOISING_FEATURES = 7,
  METAL_SURFACE_STAGE_VERTEX_MERGING = 8,
};

ccl_device_inline int integrate_surface_stage(const MetalSurfaceStage stage,
                                              IntegratorState state,
                                              ccl_private ShaderData *sd,
                                              const ccl_private RNGState *rng_state,
                                              ccl_global float *render_buffer = nullptr,
                                              ccl_private float3 *result = nullptr,
                                              ccl_private float3 *secondary_result = nullptr)
{
  return metal_ancillaries->vft_surface[stage](&launch_params_metal,
                                               metal_ancillaries,
                                               state,
                                               sd,
                                               rng_state,
                                               render_buffer,
                                               result,
                                               secondary_result);
}
#endif

template<uint64_t node_feature_mask>
ccl_device_forceinline ShaderEvalResult
integrate_surface_direct_light_stage(KernelGlobals kg,
                                     IntegratorState state,
                                     ccl_private ShaderData *sd,
                                     const ccl_private RNGState *rng_state)
{
#ifdef __KERNEL_METAL_VISIBLE_SHADING__
  /* Direct light does not depend on the node feature mask, so the regular and ray-tracing
   * surface kernels share one function. */
  (void)kg;
  return ShaderEvalResult(
      integrate_surface_stage(METAL_SURFACE_STAGE_DIRECT_LIGHT, state, sd, rng_state));
#else
  return integrate_surface_direct_light<node_feature_mask>(kg, state, sd, rng_state);
#endif
}

template<uint64_t node_feature_mask>
ccl_device int integrate_surface(KernelGlobals kg,
                                 IntegratorState state,
                                 ccl_global float *ccl_restrict render_buffer)

{
  PROFILING_INIT_FOR_SHADER(kg, PROFILING_SHADE_SURFACE_SETUP);

  /* Setup shader data. */
  ShaderData sd;
  integrate_surface_shader_setup(kg, state, &sd);
  PROFILING_SHADER(sd.object, sd.shader);

  int continue_path_label = 0;

  const PathRayVisibility path_visibility = INTEGRATOR_STATE(state, path, visibility);
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

#if defined(__BDPT__) && defined(__KERNEL_GPU__)
  /* Primary-volume emitter MIS also mutates the recurrence at a surface hit.
   * Include that continuation in the retry stages so a later shader cache miss
   * cannot repeat its measure conversion or film writes. */
  const bool staged_bdpt = bdpt_enabled_for_emission(kg, state);
  const uint surface_stage = staged_bdpt ? INTEGRATOR_STATE(state, path, bdpt_surface_stage) : 0;
#else
  /* The CPU loads image tiles synchronously: shading never resumes from a later stage. */
  ccl_attr_maybe_unused constexpr bool staged_bdpt = false;
  constexpr uint surface_stage = 0;
#endif
  /* 0: initial shading, 1: vertex merging pending, 2: direct light pending, 3: connection
   * pending, 4: scatter pending.
   * Texture retries reconstruct closures but never repeat completed film or shadow writes. */

#ifdef __NESTED_DIELECTRICS__
  /* Nested dielectrics: find the medium around the surface, which its dielectric closures are
   * relative to. A surface inside a medium of higher priority does not exist. */
  if (nested_dielectric_surface_setup(kg, state, &sd)) {
    return integrate_surface_nested_false_intersection<node_feature_mask>(kg, state, &sd);
  }
#endif

  /* Skip most work for volume bounding surface. */
#ifdef __VOLUME__
  if (!(sd.shader_flag & SD_HAS_ONLY_VOLUME)) {
#endif
#ifdef __SUBSURFACE__
    /* Can skip shader evaluation for BSSRDF exit point without bump mapping. */
    if (!(path_flag & PATH_RAY_SUBSURFACE) || ((sd.shader_flag & SD_HAS_BSSRDF_BUMP)))
#endif
    {
      /* Evaluate shader. */
      PROFILING_EVENT(PROFILING_SHADE_SURFACE_EVAL);
      surface_shader_eval<node_feature_mask>(kg,
                                             state,
                                             &sd,
                                             surface_stage == 0 ? render_buffer : nullptr,
                                             path_visibility,
                                             path_flag);
    }

    if (sd.runtime_flag & SR_CACHE_MISS) {
      return LABEL_CACHE_MISS;
    }

    if (surface_stage == 0) {
#ifdef __SPECTRAL__
    if (sd.runtime_flag & (SR_BSDF_HAS_DISPERSION | SR_BSDF_HAS_SPECTRAL_TRANSMISSION)) {
      update_path_throughput_for_dispersion(kg, state, sd.rand_wavelength);
    }
#endif

    /* After shader evaluation, in case of texture cache miss. */
    guiding_record_surface_segment(kg, state, &sd);
#ifdef __KERNEL_METAL__
    if (sd.runtime_flag & SR_BSDF_HAS_EVAL) {
      guiding_gpu_record_importance(state,
                                    sd.P,
                                    sd.wi,
                                    fabsf(dot(sd.Ng, sd.wi)),
                                    false,
                                    guiding_gpu_surface_orientation(&sd));
    }
#endif
    }

#ifdef __SUBSURFACE__
    if (path_flag & PATH_RAY_SUBSURFACE) {
      /* When coming from inside subsurface scattering, setup a diffuse
       * closure to perform lighting at the exit point. */
      subsurface_shader_data_setup(kg, &sd);
      INTEGRATOR_STATE_WRITE(state, path, flag) &= ~PATH_RAY_SUBSURFACE;
    }
    else
#endif
    {
      /* Filter closures. */
      surface_shader_prepare_closures(kg, state, &sd, path_visibility);

      if (surface_stage == 0) {
#ifdef __BDPT__
        if (bdpt_enabled_for_emission(kg, state)) {
          bdpt_recursive_mis_after_hit(kg, state, &sd);
        }
#endif

      /* Evaluate holdout. */
      if (!integrate_surface_holdout(kg, state, &sd, render_buffer)) {
        return LABEL_NONE;
      }

#ifdef __PHOTON_MAPPING__
      if (kernel_data.integrator.use_photon_mapping) {
#  ifdef __KERNEL_METAL_VISIBLE_SHADING__
        Spectrum photon_L = zero_spectrum();
        integrate_surface_stage(
            METAL_SURFACE_STAGE_PHOTON_GATHER, state, &sd, nullptr, render_buffer, &photon_L);
#  else
        const Spectrum photon_L = photon_mapping_gather(kg, state, &sd, render_buffer);
#  endif
        photon_mapping_write(kg, state, photon_L, render_buffer);
      }
#endif

      /* Write emission. */
      if (sd.runtime_flag & SR_EMISSION) {
#  ifdef __KERNEL_METAL_VISIBLE_SHADING__
        integrate_surface_stage(
            METAL_SURFACE_STAGE_EMISSION, state, &sd, nullptr, render_buffer);
#  else
        integrate_surface_emission(kg, state, &sd, render_buffer);
#  endif
      }

      /* Perform path termination. Most paths have already been terminated in
       * the intersect_closest kernel, this is just for emission and for dividing
       * throughput by the probability at the right moment.
       *
       * Also ensure we don't do it twice for SSS at both the entry and exit point. */
      if (integrate_surface_terminate(state, path_flag)) {
        return LABEL_NONE;
      }

      /* Write render passes. */
#ifdef __KERNEL_METAL_VISIBLE_SHADING__
#  ifdef __PASSES__
      integrate_surface_stage(
          METAL_SURFACE_STAGE_DATA_PASSES, state, &sd, nullptr, render_buffer);
#  endif
#  ifdef __DENOISING_FEATURES__
      integrate_surface_stage(
          METAL_SURFACE_STAGE_DENOISING_FEATURES, state, &sd, nullptr, render_buffer);
#  endif
#else
#  ifdef __PASSES__
      PROFILING_EVENT(PROFILING_SHADE_SURFACE_PASSES);
      film_write_data_passes(kg, state, &sd, render_buffer);
#  endif

#  ifdef __DENOISING_FEATURES__
      film_write_denoising_features_surface(kg, state, &sd, render_buffer);
#  endif
#endif
      }
      else if (((sd.runtime_flag & SR_HOLDOUT) || (sd.object_flag & SD_OBJECT_HOLDOUT_MASK)) &&
               (path_flag & PATH_RAY_TRANSPARENT_BACKGROUND))
      {
        /* Reconstruct the holdout-adjusted closures without repeating the film contribution. */
        surface_shader_apply_holdout(&sd);
      }
    }

#ifdef __BDPT__
    if (staged_bdpt && surface_stage == 0) {
      INTEGRATOR_STATE_WRITE(state, path, bdpt_surface_stage) = 1;
    }
#endif

    /* Load random number state. */
    RNGState rng_state;
    path_state_rng_load(state, &rng_state);

#if defined(__PATH_GUIDING__) && PATH_GUIDING_LEVEL >= 4
    if (kernel_data.kernel_features & KERNEL_FEATURE_PATH_GUIDING) {
      surface_shader_prepare_guiding(kg, state, &sd, &rng_state);
#  ifdef __BDPT__
      /* OpenPGL initializes a vertex's guiding distribution stochastically, so a light subpath
       * cannot evaluate the density with which a camera path would have sampled it. The
       * bidirectional MIS recursion requires that density, so its vertices sample unguided.
       * The field is still trained, and vertices the recursion does not cover remain guided. */
      if (bdpt_enabled_for_surface_path(kg, state)) {
        INTEGRATOR_STATE_WRITE(state, guiding, use_surface_guiding) = false;
      }
#  endif
      guiding_write_debug_passes(kg, state, &sd, render_buffer);
    }
#endif
#ifdef __BDPT__
    if (surface_stage < 2) {
      /* Merge with the light subpath vertices around this vertex. */
      if (kernel_data.integrator.use_vertex_merging) {
#  ifdef __KERNEL_METAL_VISIBLE_SHADING__
        const bool merged = integrate_surface_stage(METAL_SURFACE_STAGE_VERTEX_MERGING,
                                                    state,
                                                    &sd,
                                                    nullptr,
                                                    render_buffer) == 0;
#  else
        const bool merged = vcm_merge(kg, state, &sd, render_buffer);
#  endif
        if (!merged) {
          return LABEL_CACHE_MISS;
        }
      }
      if (staged_bdpt) {
        INTEGRATOR_STATE_WRITE(state, path, bdpt_surface_stage) = 2;
      }
    }
#endif
    if (surface_stage < 3) {
      if (surface_stage <= 1 && kernel_data.integrator.coherent_specular_enabled &&
          (sd.object_flag & SD_OBJECT_COHERENT_DETECTOR))
      {
        /* Diagonals and signed pairs for declared coherent source paths are
         * owned here. Ordinary NEE/BDPT strategies for this class are
         * filtered; unrelated light paths retain their estimators. */
        Spectrum primary_direct;
        const bool light_passes = kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES;
#ifdef __KERNEL_METAL_VISIBLE_SHADING__
        Spectrum intensity = zero_spectrum();
        integrate_surface_stage(METAL_SURFACE_STAGE_COHERENT_SPECULAR,
                                state,
                                &sd,
                                nullptr,
                                nullptr,
                                &intensity,
                                light_passes ? &primary_direct : nullptr);
#else
        Spectrum intensity = coherent_specular_complete_intensity(
            kg, state, &sd, light_passes ? &primary_direct : nullptr);
#endif
        /* The declared Lambertian detector reradiates an unpolarized field.
         * Camera-side analyzers therefore contract its intensity with A_I. */
        const Spectrum detector_weight = polarization_emission_weight(kg, state);
        intensity *= detector_weight;
        if (light_passes) primary_direct *= detector_weight;
        ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);
        if (light_passes) {
          const Spectrum throughput = INTEGRATOR_STATE(state, path, throughput);
          film_write_coherent_surface_light_passes(
              kg, state, throughput * intensity, throughput * primary_direct, buffer);
        }
        if (!is_zero(intensity)) {
          film_write_combined_pass(kg,
                                   path_visibility,
                                   path_flag,
                                   INTEGRATOR_STATE(state, path, sample),
                                   INTEGRATOR_STATE(state, path, throughput) * intensity,
                                   buffer);
        }
      }
      /* Direct light. */
      PROFILING_EVENT(PROFILING_SHADE_SURFACE_DIRECT_LIGHT);
      const ShaderEvalResult result = integrate_surface_direct_light_stage<node_feature_mask>(
          kg, state, &sd, &rng_state);
      if (result == SHADER_EVAL_CACHE_MISS) {
        return LABEL_CACHE_MISS;
      }
#ifdef __BDPT__
      if (staged_bdpt) {
        INTEGRATOR_STATE_WRITE(state, path, bdpt_surface_stage) = 3;
      }
#endif
    }

    if (surface_stage < 4) {
#ifdef __BDPT__
#  ifdef __KERNEL_METAL_VISIBLE_SHADING__
      integrate_surface_stage(METAL_SURFACE_STAGE_BIDIRECTIONAL, state, &sd, &rng_state);
#  elif defined(__KERNEL_GPU__)
      integrate_surface_bidirectional(kg, state, &sd, &rng_state);
#  else
      if (kernel_data.integrator.use_bidirectional_path_tracing) {
        integrate_surface_bidirectional(kg, state, &sd, &rng_state);
      }
#  endif
      if (sd.runtime_flag & SR_CACHE_MISS) {
        return LABEL_CACHE_MISS;
      }
#endif

#if defined(__AO__)
      /* Ambient occlusion pass. */
      if (kernel_data.kernel_features & KERNEL_FEATURE_AO) {
        PROFILING_EVENT(PROFILING_SHADE_SURFACE_AO);
        integrate_surface_ao(kg, state, &sd, &rng_state);
      }
#endif
#ifdef __BDPT__
      if (staged_bdpt) {
        INTEGRATOR_STATE_WRITE(state, path, bdpt_surface_stage) = 4;
      }
#endif
    }

    PROFILING_EVENT(PROFILING_SHADE_SURFACE_INDIRECT_LIGHT);
#ifdef __KERNEL_METAL_VISIBLE_SHADING__
    continue_path_label = integrate_surface_stage(
        METAL_SURFACE_STAGE_BSDF_BSSRDF_BOUNCE, state, &sd, &rng_state);
#else
    continue_path_label = integrate_surface_bsdf_bssrdf_bounce(kg, state, &sd, &rng_state);
#endif
    if (continue_path_label == LABEL_CACHE_MISS) {
      return LABEL_CACHE_MISS;
    }

#ifdef __PHOTON_MAPPING__
    /* The synthetic diffuse bounce at a BSSRDF exit is not a local photon-map receiver. */
    if (kernel_data.integrator.use_photon_mapping && (path_flag & PATH_RAY_SUBSURFACE)) {
      INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_PHOTON_MAPPING_UNSUPPORTED;
      INTEGRATOR_STATE_WRITE(state, path, flag) &= ~PATH_RAY_PHOTON_MAPPING_RECEIVER;
    }
#endif
#ifdef __VOLUME__
  }
  else {
    if (integrate_surface_terminate(state, path_flag)) {
      return LABEL_NONE;
    }

#  ifdef __DENOISING_FEATURES__
    film_write_denoising_features_surface_volume(kg, state, &sd, render_buffer);
#  endif

    PROFILING_EVENT(PROFILING_SHADE_SURFACE_INDIRECT_LIGHT);
    continue_path_label = integrate_surface_volume_only_bounce(state, &sd);
  }
#endif

  if (continue_path_label & LABEL_TRANSMIT) {
    /* Enter/Exit volumes and nested dielectric media. */
    path_media_enter_exit<false>(kg, state, &sd);
  }

  return continue_path_label;
}

template<DeviceKernel current_kernel>
ccl_device_forceinline void integrator_shade_surface_next_kernel(IntegratorState state)
{
  if (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SUBSURFACE) {
    integrator_path_next(state, current_kernel, DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE);
  }
  else {
    kernel_assert(INTEGRATOR_STATE(state, ray, tmax) != 0.0f);
    integrator_path_next(state, current_kernel, DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST);
  }
}

template<uint64_t node_feature_mask = KERNEL_FEATURE_NODE_MASK_SURFACE &
                                      ~KERNEL_FEATURE_NODE_RAYTRACE,
         DeviceKernel current_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE>
ccl_device_forceinline void integrator_shade_surface(KernelGlobals kg,
                                                     IntegratorState state,
                                                     ccl_global float *ccl_restrict render_buffer)
{
  const int continue_path_label = integrate_surface<node_feature_mask>(kg, state, render_buffer);
  if (continue_path_label == LABEL_CACHE_MISS) {
    integrator_path_cache_miss_sorted(state, current_kernel);
    return;
  }

#ifdef __MNEE__
  /* Cleanup MNEE flag and shadow path if it was not reused for shadow trace. */
  if ((kernel_data.kernel_features & KERNEL_FEATURE_MNEE) &&
      (INTEGRATOR_STATE(state, path, mnee) & PATH_MNEE_SAMPLED))
  {
    INTEGRATOR_STATE_WRITE(state, path, mnee) &= ~PATH_MNEE_SAMPLED;

    const IntegratorShadowState shadow_state = integrator_state_get_mnee_shadow_state(state);
    if (INTEGRATOR_STATE(shadow_state, shadow_path, queued_kernel) ==
        DEVICE_KERNEL_INTEGRATOR_SHADOW_PATH_MNEE_PENDING)
    {
      integrator_shadow_path_terminate(shadow_state,
                                       DEVICE_KERNEL_INTEGRATOR_SHADOW_PATH_MNEE_PENDING);
    }
  }
#endif

  if (continue_path_label == LABEL_NONE) {
    integrator_path_terminate(kg, state, render_buffer, current_kernel);
    return;
  }

#ifdef __SHADOW_LINKING__
  /* No need to cast shadow linking rays at a transparent bounce: the lights will be accumulated
   * via the main path in this case. BSSRDF bounces continue with intersect_subsurface. */
  if ((continue_path_label & (LABEL_TRANSPARENT | LABEL_SUBSURFACE_SCATTER)) == 0) {
    if (shadow_linking_schedule_intersection_kernel<current_kernel>(kg, state)) {
      return;
    }
  }
#endif

  integrator_shade_surface_next_kernel<current_kernel>(state);
}

ccl_device_forceinline void integrator_shade_surface_raytrace(
    KernelGlobals kg, IntegratorState state, ccl_global float *ccl_restrict render_buffer)
{
  integrator_shade_surface<KERNEL_FEATURE_NODE_MASK_SURFACE,
                           DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE>(
      kg, state, render_buffer);
}

CCL_NAMESPACE_END
