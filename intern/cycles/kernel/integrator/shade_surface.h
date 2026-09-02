/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/path_state.h"
#include "kernel/integrator/restir.h"
#include "kernel/integrator/surface_shader.h"

#include "kernel/film/data_passes.h"
#include "kernel/film/denoising_passes.h"
#include "kernel/film/light_passes.h"

#include "kernel/light/sample.h"

#include "kernel/geom/motion_triangle.h"
#include "kernel/geom/triangle.h"

#include "kernel/integrator/guiding.h"
#include "kernel/integrator/shadow_linking.h"
#include "kernel/integrator/subsurface.h"
#include "kernel/integrator/volume_stack.h"

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
  const Spectrum L = surface_shader_emission(sd);

  float mis_weight;
#ifdef __KERNEL_METAL__
  const float forward_weight = light_sample_mis_weight_forward_surface(
      kg, state, path_visibility, path_flag, sd);
  mis_weight = ((path_flag & PATH_RAY_RESTIR_DIRECT) && !(path_flag & PATH_RAY_MIS_SKIP)) ?
                   /* Keep forward-only emitters that cannot be sampled by NEE. */
                   (restir_di_surface_nee_supported(sd, path_flag) ? 0.0f : 1.0f) :
                   bdpt_enabled_for_surface_path(state) ?
                   bdpt_emission_mis_weight_surface(kg, state, sd) :
                   forward_weight;
#else
  mis_weight = light_sample_mis_weight_forward_surface(kg, state, path_visibility, path_flag, sd);
#endif

  guiding_record_surface_emission(kg, state, L, mis_weight);
  film_write_surface_emission(
      kg, state, L, mis_weight, render_buffer, object_lightgroup(kg, sd->object));
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
                                          const bool constant_light_shader)
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
  {
    shadow_state = integrator_shadow_path_init(kg, state, next_kernel, false);
  }

#ifdef __VOLUME__
  /* Copy volume stack and enter/exit volume. */
  integrator_state_copy_volume_stack_to_shadow(kg, shadow_state, state);
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

#ifdef __KERNEL_METAL__
  if (kernel_data.kernel_features & KERNEL_FEATURE_RESTIR_PT) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_reconnection) = 0u;
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_jacobian) = 1.0f;
  }
#endif

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

#ifdef __KERNEL_METAL__
/* Hybrid shift from the replayed current prefix to the source reservoir's stored suffix. The
 * connection is accepted only in the reciprocal rough/large-footprint domain and is evaluated by
 * an ordinary Cycles shadow path, so transparency and volumes on the connecting edge remain exact. */
ccl_device_forceinline bool integrate_surface_restir_pt_reconnection(
    KernelGlobals kg, IntegratorState state, ccl_private ShaderData *sd)
{
  if (!kernel_data.integrator.use_restir_pt ||
      kernel_integrator_state.restir_pt_phase == 0u ||
      !(sd->runtime_flag & SR_BSDF_HAS_EVAL))
  {
    return false;
  }
  const uint pixel_index = INTEGRATOR_STATE(state, path, render_pixel_index);
  const ccl_global KernelReSTIRPTReservoir *source = restir_pt_replay_source(pixel_index);
  if (!source) {
    return false;
  }
  const uint path_length = source->path_data & 0xffu;
  const uint rc_length = (source->path_data >> 8u) & 0xffu;
  const uint technique = (source->path_data >> 16u) & 0xffu;
  const uint bounce = uint(INTEGRATOR_STATE(state, path, bounce));
  if (rc_length < 2u || rc_length > path_length || bounce + 2u != rc_length ||
      !(source->rc_wi_pdf > 0.0f) || !(source->inverse_partial_jacobian > 0.0f) ||
      surface_shader_average_roughness(sd) < kernel_data.integrator.restir_pt_min_roughness)
  {
    return false;
  }

  packed_normal packed_rc_N;
  packed_rc_N.value = source->rc_normal;
  const float3 rc_N = packed_rc_N.decode();
  const float3 rc_P = make_float3(source->rc_P);
  const float3 delta = rc_P - sd->P;
  const float distance2 = len_squared(delta);
  if (!(distance2 > 1.0e-12f)) {
    return false;
  }
  const float distance = sqrtf(distance2);
  const float3 D = delta / distance;
  const float cos_rc = fabsf(dot(rc_N, -D));
  const float cos_current = fabsf(dot(sd->Ng, D));
  if (!(cos_rc > 1.0e-7f) || !(cos_current > 1.0e-7f)) {
    return false;
  }

  BsdfEval bsdf_eval ccl_optional_struct_init;
  float roughness_squared = 0.0f;
  const float target_pdf = surface_shader_bsdf_eval(
      kg, state, sd, D, &bsdf_eval, SHADER_USE_MIS, roughness_squared);
  if (!(target_pdf > 0.0f) || bsdf_eval_is_zero(&bsdf_eval)) {
    return false;
  }
  const bool forced_nee = technique == RESTIR_PT_TECHNIQUE_NEE &&
                          rc_length == path_length && source->path_hash != 0u;
  float target_geometry = cos_rc / distance2;
  float target_density = target_pdf * target_geometry;
  float source_density = source->inverse_partial_jacobian * source->rc_wi_pdf;
  float jacobian = 0.0f;
  if (forced_nee) {
    const int emitter_object = int((source->path_hash >> 16u) & 0xffffu) - 1;
    const int emitter_prim = int(source->path_hash & 0xffffu) - 1;
    if (emitter_object < 0 || emitter_prim < 0) {
      return false;
    }
    ShaderData light_sd ccl_optional_struct_init;
    light_sd.P = rc_P;
    light_sd.Ng = rc_N;
    light_sd.wi = -D;
    light_sd.object = emitter_object;
    light_sd.prim = emitter_prim;
    light_sd.time = sd->time;
    float target_light_pdf = triangle_light_pdf(kg, &light_sd, distance);
#  ifdef __LIGHT_TREE__
    if (kernel_data.integrator.use_light_tree && target_light_pdf > 0.0f) {
      const uint lookup_offset = kernel_data_fetch(object_lookup_offset, emitter_object);
      const uint prim_offset = kernel_data_fetch(object_prim_offset, emitter_object);
      const uint triangle = kernel_data_fetch(
          triangle_to_tree, emitter_prim - int(prim_offset) + int(lookup_offset));
      target_light_pdf *= light_tree_pdf(kg,
                                         sd->P,
                                         sd->N,
                                         0.0f,
                                         INTEGRATOR_STATE(state, path, visibility),
                                         INTEGRATOR_STATE(state, path, flag),
                                         emitter_object,
                                         triangle,
                                         light_link_receiver_nee(kg, sd));
    }
#  endif
    if (!(target_light_pdf > 0.0f) || !isfinite_safe(target_light_pdf)) {
      return false;
    }
    source_density = source->inverse_partial_jacobian;
    target_density = target_light_pdf;
    jacobian = source_density / target_density;
  }
  const float dual_footprint = 1.0f / max(max(target_density, source_density), 1.0e-20f);
  const float required_footprint = kernel_data.integrator.restir_pt_footprint_threshold *
                                   max(INTEGRATOR_STATE(
                                           state, path, restir_pt_primary_footprint),
                                       1.0e-12f);
  if (dual_footprint < required_footprint) {
    return false;
  }
  if (!forced_nee) {
    jacobian = (source->inverse_partial_jacobian * target_geometry) /
               max(target_pdf * source->rc_wi_pdf, 1.0e-20f);
  }
  if (!(jacobian > 0.0f) || !isfinite_safe(jacobian)) {
    return false;
  }

  const Spectrum target_rc_throughput = INTEGRATOR_STATE(state, path, throughput) *
                                        (forced_nee ? bsdf_eval_sum(&bsdf_eval) :
                                                      bsdf_eval_sum(&bsdf_eval) / target_pdf);
  const Spectrum throughput_ratio = safe_divide_color(
      target_rc_throughput, Spectrum(source->rc_throughput));
  const Spectrum shifted_contribution = Spectrum(source->contribution) * throughput_ratio;
  if (is_zero(shifted_contribution) ||
      !isfinite_safe(reduce_max(fabs(shifted_contribution))))
  {
    return false;
  }

  Ray ray ccl_optional_struct_init;
  bool skip_self = true;
  ray.P = shadow_ray_offset(kg, sd, D, &skip_self);
  const float3 rc_shadow_P = ray_offset(rc_P, dot(rc_N, -D) >= 0.0f ? rc_N : -rc_N);
  const float3 shadow_delta = rc_shadow_P - ray.P;
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
  ray.self.light_object = OBJECT_NONE;
  ray.self.light_prim = PRIM_NONE;
#  ifdef __RAY_DIFFERENTIALS__
  ray.dP = differential_zero_compact();
  ray.dD = differential_zero_compact();
#  endif

  const int source_lightgroup = int(source->lightgroup & 0xffu) - 1;
  IntegratorShadowState shadow_state = integrate_direct_light_shadow_init_common(
      kg, state, &ray, one_spectrum(), source_lightgroup, 0, true);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, throughput) = shifted_contribution;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, bounce) = uint16_t(path_length - 2u);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, visibility) = PathRayVisibility(
      (source->lightgroup >> 8u) & 0xffu);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, flag) = source->path_flag;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_reconnection) = 1u;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_jacobian) = jacobian;
  return true;
}
#endif

/* Path tracing: sample point on light and evaluate light shader, then
 * queue shadow ray to be traced. */
template<uint64_t node_feature_mask>
#if defined(__KERNEL_GPU__)
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
#ifdef __KERNEL_METAL__
  float restir_weight = 0.0f;
  bool use_restir = false;
#endif

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
#ifdef __KERNEL_METAL__
    const uint32_t current_path_flag = INTEGRATOR_STATE(state, path, flag);
    const bool restir_surface_supported =
        !bdpt_enabled_for_surface_path(state) &&
        !(current_path_flag & PATH_RAY_SHADOW_CATCHER_PASS) &&
        surface_shader_average_roughness(sd) >=
            (kernel_data.integrator.use_restir_pt ?
                 kernel_data.integrator.restir_pt_min_roughness :
                 kernel_data.integrator.restir_min_roughness);
    /* ReSTIR PT owns indirect paths, while fresh ReSTIR DI candidate resampling is a lower-cost,
     * lower-variance estimator for primary many-light illumination. Do not force DI at later PT
     * vertices: doing so changes every indirect proposal and regresses heterogeneous emissive
     * production scenes. The primary DI path is disjoint from PT (which starts at path length 3),
     * so this forms the paper's unified direct/global path space without double counting. */
    const bool restir_pt_primary_di = kernel_data.integrator.use_restir_pt &&
                                      INTEGRATOR_STATE(state, path, bounce) == 0u;

    use_restir = (kernel_data.integrator.use_restir || restir_pt_primary_di) &&
                 (!kernel_data.integrator.use_restir_pt || restir_pt_primary_di) &&
                 restir_surface_supported;
    if (use_restir) {
      if (!restir_di_resample(kg, state, sd, rng_state, &ls, &restir_weight)) {
        /* This path could not build a trustworthy reservoir (for example it sampled a
         * non-constant emitter). Fall through to ordinary Cycles NEE locally; do not disable
         * ReSTIR for other surfaces or pixels. */
        use_restir = false;
      }
    }
    if (!use_restir)
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
  }

  kernel_assert(ls.pdf != 0.0f);

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

#ifdef __KERNEL_METAL__
  /* Cache the prefix-side factor before NEE's light-selection and emitter weights are applied.
   * If random replay finds no earlier hybrid vertex, a finite sampled-light endpoint can then be
   * forced as the reconnection vertex without resampling the light. */
  Spectrum restir_pt_nee_rc_throughput = zero_spectrum();
  float restir_pt_nee_source_pdf = 0.0f;
  if (kernel_data.integrator.use_restir_pt && mnee_vertex_count == 0 && bsdf_pdf > 0.0f &&
      /* The PSS Jacobian below is an area-measure mapping. Analytic delta and distant lights need
       * their own discrete/directional mapping and must not be treated as finite area endpoints. */
      ls.type == LIGHT_TRIANGLE && uint(INTEGRATOR_STATE(state, path, bounce)) > 0u &&
      isfinite_safe(ls.t) &&
      ls.t > 1.0e-6f && surface_shader_average_roughness(sd) >=
                             kernel_data.integrator.restir_pt_min_roughness)
  {
    restir_pt_nee_rc_throughput = INTEGRATOR_STATE(state, path, throughput) *
                                  bsdf_eval_sum(&bsdf_eval);
    restir_pt_nee_source_pdf = ls.pdf;
  }
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
#ifdef __KERNEL_METAL__
    mis_weight = bdpt_enabled_for_surface_path(state) ?
                              bdpt_nee_mis_weight(kg, state, sd, &ls, bsdf_pdf) :
                              light_sample_mis_weight_nee(kg, ls.pdf, bsdf_pdf);
#else
    mis_weight = light_sample_mis_weight_nee(kg, ls.pdf, bsdf_pdf);
#endif
#ifdef __KERNEL_METAL__
    const float sample_weight = use_restir ? restir_weight : 1.0f / ls.pdf;
    bsdf_eval_mul(&bsdf_eval, light_shader_eval * ls.eval_fac * sample_weight * mis_weight);
#else
    /* Preserve the established operation order and bit-exact output on unrelated devices. */
    bsdf_eval_mul(&bsdf_eval, light_shader_eval * ls.eval_fac / ls.pdf * mis_weight);
#endif

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

  /* Branch off shadow kernel. */
  IntegratorShadowState shadow_state = integrate_direct_light_shadow_init_common(
      kg,
      state,
      &ray,
      bsdf_eval_sum(&bsdf_eval),
      ls.group,
      mnee_vertex_count,
      is_constant_light_shader);

  if (is_transmission) {
#ifdef __VOLUME__
    volume_stack_enter_exit<true>(kg, shadow_state, sd);
#endif
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
  if (kernel_data.kernel_features & KERNEL_FEATURE_RESTIR_PT) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_path_hash) = INTEGRATOR_STATE(
        state, path, restir_pt_path_hash);
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_rc_P) = INTEGRATOR_STATE(
        state, path, restir_pt_rc_P);
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_rc_normal) = INTEGRATOR_STATE(
        state, path, restir_pt_rc_normal);
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_rc_throughput) = INTEGRATOR_STATE(
        state, path, restir_pt_rc_throughput);
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_rc_wi_pdf) = INTEGRATOR_STATE(
        state, path, restir_pt_rc_wi_pdf);
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_inverse_partial_jacobian) =
        INTEGRATOR_STATE(state, path, restir_pt_inverse_partial_jacobian);
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_rc_length) = INTEGRATOR_STATE(
        state, path, restir_pt_rc_length);
#ifdef __KERNEL_METAL__
    if (kernel_data.integrator.use_restir_pt &&
        kernel_integrator_state.restir_pt_phase == 0u &&
        INTEGRATOR_STATE(state, path, restir_pt_rc_length) == uint16_t(0xffffu) &&
        restir_pt_nee_source_pdf > 0.0f && ls.object >= 0 && ls.object < 0xffff &&
        ls.prim >= 0 && ls.prim < 0xffff &&
        !is_zero(restir_pt_nee_rc_throughput))
    {
      INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_rc_P) = ls.P;
      INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_rc_normal) =
          packed_normal(ls.Ng).value;
      INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_rc_throughput) =
          restir_pt_nee_rc_throughput;
      INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_rc_wi_pdf) =
          1.0f;
      INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_inverse_partial_jacobian) =
          restir_pt_nee_source_pdf;
      INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_rc_length) = uint16_t(
          min(uint(INTEGRATOR_STATE(state, path, bounce)) + 2u, 0xfffeu));
      INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, restir_pt_path_hash) =
          (uint(ls.object + 1) << 16u) | uint(ls.prim + 1);
    }
#endif
  }

  return SHADER_EVAL_OK;
}

#ifdef __KERNEL_METAL__
/* Connect the current camera vertex to one uniformly selected entry of the global light-vertex
 * cache. Every entry is a per-path reservoir sample over the connectible vertices that path
 * actually reached. Uniform cache sampling plus the stored reservoir support converts the
 * selected vertex into an unbiased estimate of the sum over light-subpath lengths. */
ccl_device_forceinline bool integrate_surface_bidirectional(KernelGlobals kg,
                                                            IntegratorState state,
                                                            ccl_private ShaderData *sd,
                                                            const ccl_private RNGState *rng_state)
{
  if (!bdpt_enabled_for_surface_path(state) || !(sd->runtime_flag & SR_BSDF_HAS_EVAL) ||
      !kernel_integrator_state.bdpt_vertices || !kernel_integrator_state.bdpt_vertex_count)
  {
    return false;
  }

  const uint vertex_count = min(*kernel_integrator_state.bdpt_vertex_count,
                                kernel_integrator_state.bdpt_vertex_capacity);
  if (vertex_count == 0 || kernel_integrator_state.bdpt_light_path_count == 0) {
    return false;
  }

  const uint bounce = uint(INTEGRATOR_STATE(state, path, bounce));
  const float select = hash_uint3_to_float(INTEGRATOR_STATE(state, path, rng_pixel),
                                           uint(INTEGRATOR_STATE(state, path, sample)),
                                           bounce ^ 0x62647074u);
  const uint vertex_index = min(uint(select * float(vertex_count)), vertex_count - 1u);
  const ccl_global KernelBDPTVertex *light_vertex =
      &kernel_integrator_state.bdpt_vertices[vertex_index];
  const uint light_path_length = light_vertex->path_length & 0xffu;
  const uint light_selection_count = (light_vertex->path_length >> 8u) & 0xffu;

  if (light_selection_count == 0u ||
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
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE>(
      kg, state, &light_sd, nullptr, PATH_RAY_VISIBILITY_GLOSSY, light_vertex->flag);
  if (light_sd.runtime_flag & SR_CACHE_MISS) {
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
                                                   light_roughness_squared);
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
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE>(
      kg, state, &light_sd, nullptr, PATH_RAY_VISIBILITY_GLOSSY, light_vertex->flag);
  if (light_sd.runtime_flag & SR_CACHE_MISS) {
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

  const float cos_camera = max(fabsf(dot(sd->N, direction)), 1.0e-8f);
  const float cos_light = max(fabsf(dot(light_sd.N, -direction)), 1.0e-8f);
  const float cos_light_previous = max(fabsf(dot(light_sd.N, -light_incoming)), 1.0e-8f);
  const float camera_pdf_area = camera_pdf * cos_light / distance2;
  const float light_pdf_area = light_pdf * cos_camera / distance2;
  /* The cache contains one reservoir-selected vertex per light path that reached a connectible
   * event. K*n/N is the exact global-selection support, where n is this path's actual number of
   * candidates. Unlike a configured maximum-bounce factor, it does not amplify sparse splats from
   * short paths. */
  const float cache_scale = float(vertex_count) * float(light_selection_count) /
                            float(kernel_integrator_state.bdpt_light_path_count);

  const float w_light = camera_pdf_area *
                        (light_vertex->d_vcm + light_vertex->d_vc * light_reverse_pdf);
  const float w_camera = light_pdf_area *
                         (INTEGRATOR_STATE(state, path, bdpt_d_vcm) +
                          INTEGRATOR_STATE(state, path, bdpt_d_vc) * camera_reverse_pdf);
  const float mis_weight = 1.0f / (1.0f + cache_scale * (w_light + w_camera));

  const Spectrum light_connection_eval = bsdf_eval_sum(&light_adjoint_eval) *
                                         (cos_light / cos_light_previous);
  const Spectrum spectral_weight = bdpt_light_vertex_spectral_weight(
      kg,
      state,
      light_vertex->time_wavelength,
      (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SPECTRAL) != 0u);
  Spectrum connection = Spectrum(light_vertex->throughput) * spectral_weight *
                        bsdf_eval_sum(&camera_eval) * light_connection_eval *
                        (cache_scale * mis_weight / distance2);
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
      kg, state, &ray, connection, light_vertex->light_group, 0, true);

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
#  ifdef __KERNEL_METAL__
    if (kernel_data.integrator.use_bidirectional_path_tracing) {
      INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_BDPT_UNSUPPORTED;
    }
    if (kernel_data.integrator.use_photon_mapping) {
      INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_PHOTON_MAPPING_UNSUPPORTED;
      INTEGRATOR_STATE_WRITE(state, path, flag) &= ~PATH_RAY_PHOTON_MAPPING_RECEIVER;
    }
#  endif
    return subsurface_bounce(kg, state, sd, sc);
  }
#endif
  if (CLOSURE_IS_RAY_PORTAL(sc->type)) {
#ifdef __KERNEL_METAL__
    if (kernel_data.integrator.use_photon_mapping) {
      INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_PHOTON_MAPPING_UNSUPPORTED;
      INTEGRATOR_STATE_WRITE(state, path, flag) &= ~PATH_RAY_PHOTON_MAPPING_RECEIVER;
    }
#endif
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

  if (label & LABEL_TRANSPARENT) {
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

  /* Update throughput. */
  const Spectrum bsdf_weight = bsdf_eval_sum(&bsdf_eval) / bsdf_pdf;
#ifdef __KERNEL_METAL__
  restir_pt_record_reconnection(state, sd, bsdf_pdf, bsdf_sampled_roughness, label);
#endif
  INTEGRATOR_STATE_WRITE(state, path, throughput) *= bsdf_weight;

#ifdef __KERNEL_METAL__
  if (bdpt_enabled_for_surface_path(state) && !(label & LABEL_TRANSPARENT)) {
    const float reverse_pdf = (label & LABEL_SINGULAR) ? bsdf_pdf :
                                                         bdpt_reverse_pdf(kg, state, sd, bsdf_wo);
    const float cos_out = max(fabsf(dot(sd->N, normalize(bsdf_wo))), 1.0e-8f);
    bdpt_recursive_mis_after_scatter(state, label, cos_out, bsdf_pdf, reverse_pdf);
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

#ifdef __KERNEL_METAL__
  if (kernel_data.integrator.use_photon_mapping && !(label & LABEL_TRANSPARENT)) {
    if (surface_shader_photon_mapping_receiver(sc, sd->wi)) {
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
                                             INTEGRATOR_STATE(
                                                 state, path, continuation_probability);
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
      surface_shader_eval<node_feature_mask>(
          kg, state, &sd, render_buffer, path_visibility, path_flag);
    }

    if (sd.runtime_flag & SR_CACHE_MISS) {
      return LABEL_CACHE_MISS;
    }

#ifdef __SPECTRAL__
    if (sd.runtime_flag & (SR_BSDF_HAS_DISPERSION | SR_BSDF_HAS_SPECTRAL_TRANSMISSION)) {
      update_path_throughput_for_dispersion(kg, state, sd.rand_wavelength);
    }
#endif

    /* After shader evaluation, in case of texture cache miss. */
    guiding_record_surface_segment(kg, state, &sd);

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

#ifdef __KERNEL_METAL__
      if (bdpt_enabled_for_surface_path(state)) {
        bdpt_recursive_mis_after_hit(state, &sd);
      }
#endif

      /* Evaluate holdout. */
      if (!integrate_surface_holdout(kg, state, &sd, render_buffer)) {
        return LABEL_NONE;
      }

#ifdef __KERNEL_METAL__
      if (kernel_data.integrator.use_photon_mapping) {
        const Spectrum photon_L = photon_mapping_gather(kg, state, &sd, render_buffer);
        photon_mapping_write(kg, state, photon_L, render_buffer);
      }
      restir_pt_record_primary(kg, state, &sd);
#endif

      /* Write emission. */
      if (sd.runtime_flag & SR_EMISSION) {
        integrate_surface_emission(kg, state, &sd, render_buffer);
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
#ifdef __PASSES__
      PROFILING_EVENT(PROFILING_SHADE_SURFACE_PASSES);
      film_write_data_passes(kg, state, &sd, render_buffer);
#endif

#ifdef __DENOISING_FEATURES__
      film_write_denoising_features_surface(kg, state, &sd, render_buffer);
#endif
    }

#ifdef __KERNEL_METAL__
    if (!restir_pt_begin_replay(state)) {
      return LABEL_NONE;
    }
    if (integrate_surface_restir_pt_reconnection(kg, state, &sd)) {
      return LABEL_NONE;
    }
#endif

    /* Load random number state. */
    RNGState rng_state;
    path_state_rng_load(state, &rng_state);

#if defined(__PATH_GUIDING__) && PATH_GUIDING_LEVEL >= 4
    if (kernel_data.kernel_features & KERNEL_FEATURE_PATH_GUIDING) {
      surface_shader_prepare_guiding(kg, state, &sd, &rng_state);
      guiding_write_debug_passes(kg, state, &sd, render_buffer);
    }
#endif
    /* Direct light. */
    PROFILING_EVENT(PROFILING_SHADE_SURFACE_DIRECT_LIGHT);
    const ShaderEvalResult result = integrate_surface_direct_light<node_feature_mask>(
        kg, state, &sd, &rng_state);
    if (result == SHADER_EVAL_CACHE_MISS) {
      return LABEL_CACHE_MISS;
    }

#ifdef __KERNEL_METAL__
    integrate_surface_bidirectional(kg, state, &sd, &rng_state);
#endif

#if defined(__AO__)
    /* Ambient occlusion pass. */
    if (kernel_data.kernel_features & KERNEL_FEATURE_AO) {
      PROFILING_EVENT(PROFILING_SHADE_SURFACE_AO);
      integrate_surface_ao(kg, state, &sd, &rng_state);
    }
#endif

    PROFILING_EVENT(PROFILING_SHADE_SURFACE_INDIRECT_LIGHT);
    continue_path_label = integrate_surface_bsdf_bssrdf_bounce(kg, state, &sd, &rng_state);
#ifdef __KERNEL_METAL__
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

  if (continue_path_label & LABEL_TRANSMIT) {
    /* Enter/Exit volume. */
    volume_stack_enter_exit<false>(kg, state, &sd);
  }
#endif

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
