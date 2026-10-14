/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/film/light_passes.h"

#include "kernel/integrator/guiding.h"
#include "kernel/integrator/path_state.h"
#include "kernel/integrator/shadow_catcher.h"
#include "kernel/integrator/volume_ior.h"

#include "kernel/light/light.h"

#include "kernel/bvh/bvh.h"

CCL_NAMESPACE_BEGIN

ccl_device_forceinline bool integrator_intersect_skip_lights(KernelGlobals kg,
                                                             IntegratorState state)
{
  /* When direct lighting is disabled for baking, we skip light sampling in
   * integrate_surface_direct_light for the first bounce. Therefore, in order
   * for MIS to be consistent, we also need to skip evaluating lights here. */
  return (kernel_data.integrator.filter_closures & FILTER_CLOSURE_DIRECT_LIGHT) &&
         (INTEGRATOR_STATE(state, path, bounce) == 1);
}

ccl_device_forceinline bool integrator_intersect_terminate(KernelGlobals kg,
                                                           IntegratorState state,
                                                           const int shader_flags)
{

  /* Optional AO bounce termination.
   * We continue evaluating emissive/transparent surfaces and volumes, similar
   * to direct lighting. Only if we know there are none can we terminate the
   * path immediately. */
  if (path_state_ao_bounce(kg, state)) {
    if (shader_flags & (SD_HAS_TRANSPARENT_SHADOW | SD_HAS_EMISSION)) {
      INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_TERMINATE_AFTER_TRANSPARENT;
    }
#ifdef __VOLUME__
    else if (!integrator_state_volume_stack_is_empty(kg, state)) {
      INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_TERMINATE_AFTER_VOLUME;
    }
#endif
    else {
      return true;
    }
  }

  /* Load random number state. */
  RNGState rng_state;
  path_state_rng_load(state, &rng_state);

  /* We perform path termination in this kernel to avoid launching shade_surface
   * and evaluating the shader when not needed. Only for emission and transparent
   * surfaces in front of emission do we need to evaluate the shader, since we
   * perform MIS as part of indirect rays. */
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  const float continuation_probability = path_state_continuation_probability(kg, state, path_flag);
  INTEGRATOR_STATE_WRITE(state, path, continuation_probability) = continuation_probability;

  guiding_record_continuation_probability(kg, state, continuation_probability);

  if (continuation_probability != 1.0f) {
    const float terminate = path_state_rng_1D(kg, &rng_state, PRNG_TERMINATE);

    if (continuation_probability == 0.0f || terminate >= continuation_probability) {
      if (shader_flags & SD_HAS_EMISSION) {
        /* Mark path to be terminated right after shader evaluation on the surface. */
        INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_TERMINATE_ON_NEXT_SURFACE;
      }
#ifdef __VOLUME__
      else if (!integrator_state_volume_stack_is_empty(kg, state)) {
        /* TODO: only do this for emissive volumes. */
        INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_TERMINATE_IN_NEXT_VOLUME;
      }
#endif
      else {
        return true;
      }
    }
  }

  return false;
}

#ifdef __SHADOW_CATCHER__
/* Split path if a shadow catcher was hit. */
ccl_device_forceinline void integrator_split_shadow_catcher(
    KernelGlobals kg,
    IntegratorState state,
    const ccl_private Intersection *ccl_restrict isect,
    ccl_global float *ccl_restrict render_buffer)
{
  /* Test if we hit a shadow catcher object, and potentially split the path to continue tracing two
   * paths from here. */
  const uint object_flags = intersection_get_object_flags(kg, isect);
  if (!kernel_shadow_catcher_is_path_split_bounce(kg, state, object_flags)) {
    return;
  }

  film_write_shadow_catcher_bounce_data(kg, state, render_buffer);

  /* Mark state as having done a shadow catcher split so that it stops contributing to
   * the shadow catcher matte pass, but keeps contributing to the combined pass. */
  INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_SHADOW_CATCHER_HIT;

  /* Copy current state to new state. */
  state = integrator_state_shadow_catcher_split(kg, state);

  /* Initialize new state.
   *
   * Note that the splitting leaves kernel and sorting counters as-is, so use INIT semantic for
   * the matte path. */

  /* Mark current state so that it will only track contribution of shadow catcher objects ignoring
   * non-catcher objects. */
  INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_SHADOW_CATCHER_PASS;

  /* Shadow catcher path does not use guiding.
   * Clear the path_segment to ensure we do not reference possibly stale data from the main path.
   */
#  ifdef __PATH_GUIDING__
  INTEGRATOR_STATE_WRITE(state, guiding, path_segment) = nullptr;
#  endif

  if (kernel_data.film.pass_background != PASS_UNUSED && !kernel_data.background.transparent) {
    /* If using background pass, schedule background shading kernel so that we have a background
     * to alpha-over on. The background kernel will then continue the path afterwards. */
    INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_SHADOW_CATCHER_BACKGROUND;
    integrator_path_init(state, DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND);
    return;
  }

#  ifdef __VOLUME__
  if (!integrator_state_volume_stack_is_empty(kg, state)) {
    /* Volume stack is not empty. Re-init the volume stack to exclude any non-shadow catcher
     * objects from it, and then continue shading volume and shadow catcher surface after. */
    integrator_path_init(state, DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK);
    return;
  }
#  endif

  /* Continue with shading shadow catcher surface. */
  const int shader = intersection_get_shader(kg, isect);
  const int flags = kernel_data_fetch(shaders, shader).flags;
  const bool use_caustics = kernel_data.integrator.use_caustics &&
                            (object_flags & SD_OBJECT_CAUSTICS_RECEIVER);
  const bool use_raytrace_kernel = (flags & SD_HAS_RAYTRACE);

  if (use_caustics) {
    integrator_path_init(state, DEVICE_KERNEL_INTEGRATOR_INTERSECT_MNEE);
  }
  else if (use_raytrace_kernel) {
    integrator_path_init_sorted(
        kg, state, DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE, shader);
  }
  else {
    integrator_path_init_sorted(kg, state, DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE, shader);
  }
}

/* Schedule next kernel to be executed after updating volume stack for shadow catcher. */
template<DeviceKernel current_kernel>
ccl_device_forceinline void integrator_intersect_next_kernel_after_shadow_catcher_volume(
    KernelGlobals kg, IntegratorState state)
{
  /* Continue with shading shadow catcher surface. Same as integrator_split_shadow_catcher, but
   * using NEXT instead of INIT. */
  Intersection isect ccl_optional_struct_init;
  integrator_state_read_isect(state, &isect);

  const int shader = intersection_get_shader(kg, &isect);
  const int shader_flags = kernel_data_fetch(shaders, shader).flags;
  const uint object_flags = intersection_get_object_flags(kg, &isect);
  const bool use_caustics = kernel_data.integrator.use_caustics &&
                            (object_flags & SD_OBJECT_CAUSTICS_RECEIVER);
  const bool use_raytrace_kernel = (shader_flags & SD_HAS_RAYTRACE);

  if (use_caustics) {
    integrator_path_next(state, current_kernel, DEVICE_KERNEL_INTEGRATOR_INTERSECT_MNEE);
  }
  else if (use_raytrace_kernel) {
    integrator_path_next_sorted(
        kg, state, current_kernel, DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE, shader);
  }
  else {
    integrator_path_next_sorted(
        kg, state, current_kernel, DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE, shader);
  }
}

/* Schedule next kernel to be executed after executing background shader for shadow catcher. */
template<DeviceKernel current_kernel>
ccl_device_forceinline void integrator_intersect_next_kernel_after_shadow_catcher_background(
    KernelGlobals kg, IntegratorState state)
{
#  ifdef __VOLUME__
  /* Same logic as integrator_split_shadow_catcher, but using NEXT instead of INIT. */
  if (!integrator_state_volume_stack_is_empty(kg, state)) {
    /* Volume stack is not empty. Re-init the volume stack to exclude any non-shadow catcher
     * objects from it, and then continue shading volume and shadow catcher surface after. */
    integrator_path_next(state, current_kernel, DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK);
    return;
  }
#  endif

  /* Continue with shading shadow catcher surface. */
  integrator_intersect_next_kernel_after_shadow_catcher_volume<current_kernel>(kg, state);
}
#endif

/* Schedule next kernel to be executed after intersect closest.
 *
 * Note that current_kernel is a template value since making this a variable
 * leads to poor performance with CUDA atomics. */
template<DeviceKernel current_kernel>
ccl_device_forceinline void integrator_intersect_next_kernel(
    KernelGlobals kg,
    IntegratorState state,
    const ccl_private Intersection *ccl_restrict isect,
    ccl_global float *ccl_restrict render_buffer,
    const bool hit,
    const bool skip_volume = false)
{
  /* Continue with volume kernel if we are inside a volume, regardless if we hit anything.
   * `skip_volume`: the volumes have nothing but an index of refraction, see volume_ior.h. */
#ifdef __VOLUME__
  if (!skip_volume && !integrator_state_volume_stack_is_empty(kg, state)) {
    const bool hit_surface = hit && !(isect->type & PRIMITIVE_LAMP);
    const int shader = (hit_surface) ? intersection_get_shader(kg, isect) : SHADER_NONE;
    const int flags = (hit_surface) ? kernel_data_fetch(shaders, shader).flags : 0;

    if (!integrator_intersect_terminate(kg, state, flags)) {
      if (kernel_data.integrator.volume_ray_marching) {
        integrator_path_next(
            state, current_kernel, DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME_RAY_MARCHING);
      }
      else {
        integrator_path_next(state, current_kernel, DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME);
      }
    }
    else {
      integrator_path_terminate(kg, state, render_buffer, current_kernel);
    }
    return;
  }
#endif

  if (hit) {
    /* Hit a surface, continue with light or surface kernel. */
    if (isect->type & PRIMITIVE_LAMP) {
      integrator_path_next(state, current_kernel, DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_FORWARD);
    }
    else {
      /* Hit a surface, continue with surface kernel unless terminated. */
      const int shader = intersection_get_shader(kg, isect);
      const int flags = kernel_data_fetch(shaders, shader).flags;

      if (!integrator_intersect_terminate(kg, state, flags)) {
        const uint object_flags = intersection_get_object_flags(kg, isect);
        const bool use_caustics = kernel_data.integrator.use_caustics &&
                                  (object_flags & SD_OBJECT_CAUSTICS_RECEIVER);
        const bool use_raytrace_kernel = (flags & SD_HAS_RAYTRACE);
        if (use_caustics) {
          integrator_path_next(state, current_kernel, DEVICE_KERNEL_INTEGRATOR_INTERSECT_MNEE);
        }
        else if (use_raytrace_kernel) {
          integrator_path_next_sorted(
              kg, state, current_kernel, DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE, shader);
        }
        else {
          integrator_path_next_sorted(
              kg, state, current_kernel, DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE, shader);
        }

#ifdef __SHADOW_CATCHER__
        /* Handle shadow catcher. */
        integrator_split_shadow_catcher(kg, state, isect, render_buffer);
#endif
      }
      else {
        integrator_path_terminate(kg, state, render_buffer, current_kernel);
      }
    }
  }
  else {
    /* Nothing hit, continue with background kernel. */
    if (integrator_intersect_skip_lights(kg, state)) {
      integrator_path_terminate(kg, state, render_buffer, current_kernel);
    }
    else {
      integrator_path_next(state, current_kernel, DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND);
    }
  }
}

/* Schedule next kernel to be executed after shade volume.
 *
 * The logic here matches integrator_intersect_next_kernel, except that
 * volume shading and termination testing have already been done. */
template<DeviceKernel current_kernel>
ccl_device_forceinline void integrator_intersect_next_kernel_after_volume(
    KernelGlobals kg,
    IntegratorState state,
    const ccl_private Intersection *ccl_restrict isect,
    ccl_global float *ccl_restrict render_buffer)
{
  if (isect->prim != PRIM_NONE) {
    /* Hit a surface, continue with light or surface kernel. */
    if (isect->type & PRIMITIVE_LAMP) {
      integrator_path_next(state, current_kernel, DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_FORWARD);
      return;
    }

    /* Hit a surface, continue with surface kernel unless terminated. */
    const int shader = intersection_get_shader(kg, isect);
    const int shader_flags = kernel_data_fetch(shaders, shader).flags;
    const uint object_flags = intersection_get_object_flags(kg, isect);
    const bool use_caustics = kernel_data.integrator.use_caustics &&
                              (object_flags & SD_OBJECT_CAUSTICS_RECEIVER);
    const bool use_raytrace_kernel = (shader_flags & SD_HAS_RAYTRACE);

    if (use_caustics) {
      integrator_path_next(state, current_kernel, DEVICE_KERNEL_INTEGRATOR_INTERSECT_MNEE);
    }
    else if (use_raytrace_kernel) {
      integrator_path_next_sorted(
          kg, state, current_kernel, DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE, shader);
    }
    else {
      integrator_path_next_sorted(
          kg, state, current_kernel, DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE, shader);
    }

#ifdef __SHADOW_CATCHER__
    /* Handle shadow catcher. */
    integrator_split_shadow_catcher(kg, state, isect, render_buffer);
#endif
    return;
  }
  /* Nothing hit, continue with background kernel. */
  if (integrator_intersect_skip_lights(kg, state)) {
    integrator_path_terminate(kg, state, render_buffer, current_kernel);
  }
  else {
    integrator_path_next(state, current_kernel, DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND);
  }
}

/* Setup mnee flag to signal last intersection with a caster. */
ccl_device_forceinline void integrator_intersect_closest_mnee_setup(KernelGlobals kg,
                                                                    IntegratorState state,
                                                                    const int last_isect_object)
{
  const PathRayVisibility path_visibility = INTEGRATOR_STATE(state, path, visibility);
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

#ifdef __MNEE__
  /* Path culling logic for MNEE (removes fireflies at the cost of bias) */
  if (kernel_data.integrator.use_caustics && !(path_flag & PATH_RAY_TRANSPARENT)) {
    /* The following firefly removal mechanism works by culling light connections when
     * a ray comes from a caustic caster directly after bouncing off a different caustic
     * receiver */
    bool from_caustic_caster = false;
    bool from_caustic_receiver = false;
    if (!(path_visibility & PATH_RAY_VISIBILITY_CAMERA) && last_isect_object != OBJECT_NONE) {
      const uint object_flags = kernel_data_fetch(object_flag, last_isect_object);
      from_caustic_receiver = (object_flags & SD_OBJECT_CAUSTICS_RECEIVER);
      from_caustic_caster = (object_flags & SD_OBJECT_CAUSTICS_CASTER);
    }

    const bool has_receiver_ancestor = INTEGRATOR_STATE(state, path, mnee) &
                                       PATH_MNEE_RECEIVER_ANCESTOR;
    INTEGRATOR_STATE_WRITE(state, path, mnee) &= ~PATH_MNEE_CULL_LIGHT_CONNECTION;
    const bool caster_transmission = from_caustic_caster &&
                                     (path_visibility & PATH_RAY_VISIBILITY_TRANSMIT);
    if (caster_transmission && has_receiver_ancestor) {
      INTEGRATOR_STATE_WRITE(state, path, mnee) |= PATH_MNEE_CULL_LIGHT_CONNECTION;
    }
    /* The manifold replacement only contains a reflective receiver followed by
     * refractions through casters. A reflection (including internal reflection
     * inside a cover) or scattering from another object ends that chain. */
    /* A singular receiver (including discrete diffraction orders) has no
     * continuous BSDF evaluation for the manifold light-connection strategy.
     * Keep its sampled paths: there is no MNEE contribution to replace them. */
    if (from_caustic_receiver && (path_flag & PATH_RAY_REFLECT) &&
        !(path_flag & PATH_RAY_SINGULAR))
    {
      INTEGRATOR_STATE_WRITE(state, path, mnee) |= PATH_MNEE_RECEIVER_ANCESTOR;
    }
    else if (!caster_transmission) {
      INTEGRATOR_STATE_WRITE(state, path, mnee) &= ~PATH_MNEE_RECEIVER_ANCESTOR;
    }
  }
#else
  (void)kg;
  (void)state;
  (void)last_isect_object;
#endif /* __MNEE__ */
}

#ifdef __VOLUME_IOR__
/* Intersection of a path inside of a volume that bends it, and the kernel that follows.
 *
 * A volume with nothing but an index of refraction is crossed here, up to the next surface. One
 * that also absorbs, scatters or emits is crossed one chord of the curve at a time, each followed
 * by the volume kernel: a path state with `volume_ior_segment` set is at such a chord.
 *
 * Not inlined on Metal: the kernel of scenes without such volumes would carry the loop of this
 * function through every intersection. */
ccl_device_outline_metal void integrator_intersect_closest_volume_ior(
    KernelGlobals kg,
    IntegratorState state,
    ccl_global float *ccl_restrict render_buffer,
    ccl_private Ray *ccl_restrict ray,
    ccl_private Intersection *ccl_restrict isect,
    const uint visibility,
    const int last_isect_prim,
    const int last_isect_object,
    const bool only_ior)
{
  const int segment = INTEGRATOR_STATE(state, path, volume_ior_segment);
  const bool pending = segment >= 2;
  VolumeIORState ior;
  ior.pending = false;
  ior.steps = (pending) ? segment - 2 : 0;
  ior.D = ray->D;
  ray->dD = volume_ior_path_footprint(state, ray->dD);

  const PathRayVisibility path_visibility = INTEGRATOR_STATE(state, path, visibility);
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

  VolumeIORLights lights;
  lights.enabled = kernel_data.integrator.use_light_mis &&
                   !integrator_intersect_skip_lights(kg, state);
  lights.last_prim = last_isect_prim;
  lights.last_object = last_isect_object;
  lights.last_type = INTEGRATOR_STATE(state, isect, type);
  lights.path_visibility = path_visibility;
  lights.path_flag = path_flag;

  uint16_t rng_offset = INTEGRATOR_STATE(state, path, rng_offset);
  if (pending) {
    /* Continue behind the chord the volume kernel has integrated, with the limit of an ambient
     * occlusion bounce if the caller has set one. */
    const float t_limit = (path_state_ao_bounce(kg, state)) ? ray->tmax : FLT_MAX;
    ray->tmax = INTEGRATOR_STATE(state, ray, tmax);
    volume_ior_resume(ray, INTEGRATOR_STATE(state, path, volume_ior_D), t_limit);
    /* Not at the surface or light the ray left any more. */
    lights.last_prim = PRIM_NONE;
    lights.last_object = OBJECT_NONE;
    lights.last_type = 0;
    /* New random numbers for the volume kernel of the next chord. */
    rng_offset += PRNG_BOUNCE_NUM;
  }

  RNGState rng_state;
  path_state_rng_load(state, &rng_state);
  const float rand = path_state_rng_1D(kg, &rng_state, PRNG_VOLUME_IOR_OFFSET);

  const VolumeIORResult result = volume_ior_intersect(
      kg, state, ray, (PathRayVisibility)visibility, isect, rand, &ior, &lights);

  const bool hit = (result == VOLUME_IOR_HIT);
  const bool is_segment = (result == VOLUME_IOR_SEGMENT);

  if (!pending) {
    integrator_intersect_closest_mnee_setup(kg, state, last_isect_object);
  }

  if (only_ior) {
    /* No media to integrate along the chord: the ray arrives right away. */
    volume_ior_arrive(ray, isect, hit, ior.D);
  }

  INTEGRATOR_STATE_WRITE(state, ray, P) = ray->P;
  INTEGRATOR_STATE_WRITE(state, ray, D) = ray->D;
  INTEGRATOR_STATE_WRITE(state, ray, tmin) = ray->tmin;
  INTEGRATOR_STATE_WRITE(state, ray, tmax) = ray->tmax;
  INTEGRATOR_STATE_WRITE(state, path, rng_offset) = rng_offset;
  INTEGRATOR_STATE_WRITE(
      state, path, volume_ior_segment) = (only_ior)   ? 0 :
                                         (is_segment) ? uint16_t(min(ior.steps, 65533) + 2) :
                                                        1;
  if (!only_ior) {
    INTEGRATOR_STATE_WRITE(state, path, volume_ior_D) = ior.D;
  }
  integrator_state_write_isect(state, isect);

  if (!pending && !is_segment) {
    /* Nothing for the volume kernel to integrate if the index of refraction is all there is. */
    integrator_intersect_next_kernel<DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST>(
        kg, state, isect, render_buffer, hit, only_ior);
    return;
  }

  /* Path termination is decided once per bounce, at its first chord. What the path hits in
   * the end is not known there: continue to it as if it were an emitter, the path ends after
   * its emission then. */
  if (!pending &&
      integrator_intersect_terminate(kg, state, SD_HAS_EMISSION | SD_HAS_TRANSPARENT_SHADOW))
  {
    integrator_path_terminate(
        kg, state, render_buffer, DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST);
    return;
  }

  if (kernel_data.integrator.volume_ray_marching) {
    integrator_path_next(state,
                         DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST,
                         DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME_RAY_MARCHING);
  }
  else {
    integrator_path_next(
        state, DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST, DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME);
  }
}
#endif /* __VOLUME_IOR__ */

ccl_device void integrator_intersect_closest(KernelGlobals kg,
                                             IntegratorState state,
                                             ccl_global float *ccl_restrict render_buffer)
{
  PROFILING_INIT(kg, PROFILING_INTERSECT_CLOSEST);

  /* Read ray from integrator state into local memory. */
  Ray ray ccl_optional_struct_init;
  integrator_state_read_ray(state, &ray);
  kernel_assert(ray.tmax != 0.0f);

  const uint visibility = path_state_ray_visibility(state);
  const int last_isect_prim = INTEGRATOR_STATE(state, isect, prim);
  const int last_isect_object = INTEGRATOR_STATE(state, isect, object);

  /* Trick to use short AO rays to approximate indirect light at the end of the path. */
  if (path_state_ao_bounce(kg, state)) {
    ray.tmax = kernel_data.integrator.ao_bounces_distance;

    if (last_isect_object != OBJECT_NONE) {
      const float object_ao_distance = kernel_data_fetch(objects, last_isect_object).ao_distance;
      if (object_ao_distance != 0.0f) {
        ray.tmax = object_ao_distance;
      }
    }
  }

  /* Scene Intersection. */
  Intersection isect ccl_optional_struct_init;
  isect.object = OBJECT_NONE;
  isect.prim = PRIM_NONE;
  ray.self.object = last_isect_object;
  ray.self.prim = last_isect_prim;
  ray.self.light_object = OBJECT_NONE;
  ray.self.light_prim = PRIM_NONE;

#ifdef __VOLUME_IOR__
  /* Inside of a volume with an index of refraction the ray is curved, see volume_ior.h. */
  bool only_ior;
  if (volume_ior_stack_active(kg, state, &only_ior) &&
      (INTEGRATOR_STATE(state, path, volume_ior_segment) >= 2 || volume_ior_path_bends(state)))
  {
    integrator_intersect_closest_volume_ior(kg,
                                            state,
                                            render_buffer,
                                            &ray,
                                            &isect,
                                            visibility,
                                            last_isect_prim,
                                            last_isect_object,
                                            only_ior);
    return;
  }
#endif

  bool hit = scene_intersect(kg, &ray, visibility, &isect);

  /* TODO: remove this and do it in the various intersection functions instead. */
  if (!hit) {
    isect.prim = PRIM_NONE;
  }

  integrator_intersect_closest_mnee_setup(kg, state, last_isect_object);

  const PathRayVisibility path_visibility = INTEGRATOR_STATE(state, path, visibility);
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

  /* Light intersection for MIS. */
  if (kernel_data.integrator.use_light_mis && !integrator_intersect_skip_lights(kg, state)) {
    /* NOTE: if we make lights visible to camera rays, we'll need to initialize
     * these in the path_state_init. */
    const int last_type = INTEGRATOR_STATE(state, isect, type);
    hit = lights_intersect(kg,
                           state,
                           &ray,
                           &isect,
                           last_isect_prim,
                           last_isect_object,
                           last_type,
                           path_visibility,
                           path_flag) ||
          hit;
  }

  /* Write intersection result into global integrator state memory. */
  integrator_state_write_isect(state, &isect);

  /* Setup up next kernel to be executed. */
  integrator_intersect_next_kernel<DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST>(
      kg, state, &isect, render_buffer, hit);
}

CCL_NAMESPACE_END
