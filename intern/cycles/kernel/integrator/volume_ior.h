/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Volume Index of Refraction
 *
 * A medium whose index of refraction `n` varies in space bends the light that crosses it: hot
 * air above a fire or a road, a gas escaping into another one, sugar dissolving in water. The
 * path of a ray follows the eikonal equation
 *
 *   d/ds (n dx/ds) = grad n
 *
 * which is integrated here with a leapfrog scheme on `v = n d`: straight segments of the step
 * length `h` of the volume, joined at steps where the direction changes by the gradient of the
 * index over the two half segments around it,
 *
 *   d' = normalize(n d + h grad n).
 *
 * The scheme is second order and, up to that order, takes a ray traced backwards along the same
 * polyline, which bidirectional methods rely on. The first step is offset by a random fraction
 * of `h`, so that what is left of the discretization is noise and not a pattern.
 *
 * The gradient is that of the refractivity `n - 1` the volume shaders of the stack evaluate with
 * SR_VOLUME_IOR_EVAL, from differences between three points around the step. For air the
 * refractivity is a few 1e-4: its gradient would be lost in the rounding of `n`.
 *
 * Rays are represented so that the rest of the integrator need not know about any of this. After
 * a direction change the ray origin is moved back along the new direction by the length of the
 * path so far: `ray.P + t * ray.D` still is the path at the arc length `t` from the last
 * scattering event for the current segment, `ray.D` is the direction the path arrives with, and
 * the distance to a hit is the length of the curved path. The origin is where the hit point sees
 * the scattering event, as in a mirage. Everything that reasons about the segment between two
 * path vertices (multiple importance sampling, ray differentials, bidirectional measures) takes
 * this straight ray as the edge, which is exact where the index is constant and consistent
 * between the strategies elsewhere.
 *
 * What is not bent: shadow rays and the connections of bidirectional methods. Light reaches a
 * shading point through hot air along a path whose direction differs from the straight one by
 * less than a milliradian, so direct light is right up to the focusing of light by the medium
 * (the shadowgraph), which sampled paths that reach an emitter and light subpaths do include. */

#include "kernel/globals.h"
#include "kernel/types.h"

#include "kernel/bvh/bvh.h"

#include "kernel/geom/shader_data.h"

#include "kernel/integrator/state.h"
#include "kernel/integrator/state_util.h"
#include "kernel/integrator/volume_shader.h"
#include "kernel/integrator/volume_stack.h"

#include "kernel/light/light.h"

CCL_NAMESPACE_BEGIN

enum VolumeIORResult {
  /* Nothing hit before the end of the ray. */
  VOLUME_IOR_MISS = 0,
  /* Surface or light hit. */
  VOLUME_IOR_HIT = 1,
  /* The ray ends where its direction changes, with nothing hit before. Returned for media that
   * also absorb, scatter or emit, for the caller to integrate them along the segment before it
   * continues with the same VolumeIORState. */
  VOLUME_IOR_SEGMENT = 2,
  /* A texture needed by a shader is not loaded yet. */
  VOLUME_IOR_CACHE_MISS = 3,
};

/* What a caller keeps of a curved ray between the segments of a bounce. */
struct VolumeIORState {
  /* Steps taken, up to the maximum after which the ray continues straight. */
  int steps;
  /* The ray starts where its direction changes to `D`, with nothing hit before: the caller
   * continues after VOLUME_IOR_SEGMENT. The steps up to `resume` are part of that change. */
  bool pending;
  float3 D;
  float resume;
};

#ifdef __VOLUME_IOR__

/* Longest straight run that is traced at once, in steps. */
#  define VOLUME_IOR_MAX_RUN 64
/* Change of direction that is applied at once. Smaller ones of consecutive steps are summed up
 * and applied at their center, as one thin lens: what this costs in accuracy is the square of
 * this angle times the length they were collected over, and it saves most rays through weakly
 * refracting air from being intersected and integrated in pieces. */
#  define VOLUME_IOR_BEND_SQUARED 1e-8f
/* Number of steps after which a smaller change is applied anyway, and the squared change that is
 * too small to matter at all. */
#  define VOLUME_IOR_MAX_DEFER 32
#  define VOLUME_IOR_MIN_BEND_SQUARED 1e-13f
/* Where the index does not vary at all, as in the air around a plume of hot gas, steps are this
 * many times longer at most. A step that then finds it to vary is taken again from where the
 * last one was, at the normal length: what is narrower than the longest step is not missed as
 * long as the air next to it is not exactly uniform. */
#  define VOLUME_IOR_MAX_STRIDE 4
#  define VOLUME_IOR_UNIFORM_SQUARED 1e-20f

/* Step length along a ray through the volumes the path is inside of: FLT_MAX if none of them has
 * an index of refraction. `only_ior` is set if there is nothing else to them, `detail_size` to
 * the size of the smallest features of the index, across which its gradient is measured. */
ccl_device_inline float volume_ior_stack_step_size(KernelGlobals kg,
                                                   ConstIntegratorState state,
                                                   ccl_private bool *only_ior,
                                                   ccl_private float *detail_size = nullptr)
{
  float step_size = FLT_MAX;
  float detail = FLT_MAX;
  *only_ior = true;

  for (int i = 0;; i++) {
    const VolumeStack entry = integrator_state_read_volume_stack(state, i);
    if (entry.shader == SHADER_NONE) {
      break;
    }
    const int shader_flag = kernel_data_fetch(shaders, (entry.shader & SHADER_MASK)).flags;
    if (!(shader_flag & SD_HAS_VOLUME)) {
      /* Medium of nested dielectrics, without a volume shader. */
      continue;
    }
    if (shader_flag & SD_HAS_VOLUME_IOR) {
      if (entry.object != OBJECT_NONE) {
        step_size = fminf(step_size, kernel_data_fetch(volume_ior_step_size, 2 * entry.object));
        detail = fminf(detail, kernel_data_fetch(volume_ior_step_size, 2 * entry.object + 1));
      }
    }
    if (!(shader_flag & SD_HAS_ONLY_VOLUME_IOR)) {
      *only_ior = false;
    }
  }

  if (detail_size) {
    *detail_size = detail;
  }
  return step_size;
}

/* Whether the path is inside of a volume that bends it. */
ccl_device_inline bool volume_ior_stack_active(KernelGlobals kg, ConstIntegratorState state)
{
  if (!(kernel_data.kernel_features & KERNEL_FEATURE_VOLUME_IOR)) {
    return false;
  }
  bool only_ior;
  return volume_ior_stack_step_size(kg, state, &only_ior) != FLT_MAX;
}

/* Refractivity `n - 1` of the volumes of the stack at a position. */
ccl_device_inline float volume_ior_eval(KernelGlobals kg,
                                        IntegratorState state,
                                        ccl_private ShaderData *ccl_restrict sd,
                                        const float3 P,
                                        const PathRayVisibility path_visibility,
                                        const uint32_t path_flag)
{
  sd->num_closure = 0;
  sd->num_closure_left = 0;
  sd->shader_flag = 0;
  sd->object_flag = 0;
  sd->runtime_flag = SR_IS_VOLUME_SHADER_EVAL | SR_VOLUME_IOR_EVAL |
                     (sd->runtime_flag & SR_CACHE_MISS);
  sd->volume_ior = 0.0f;

  for (int i = 0;; i++) {
    const VolumeStack entry = integrator_state_read_volume_stack(state, i);
    if (entry.shader == SHADER_NONE) {
      break;
    }
    const int shader_flag = kernel_data_fetch(shaders, (entry.shader & SHADER_MASK)).flags;
    if (!(shader_flag & SD_HAS_VOLUME_IOR)) {
      continue;
    }
    /* Motion blur of the volume moves the position. */
    sd->P = P;
    volume_shader_eval_entry<false, KERNEL_FEATURE_NODE_MASK_VOLUME>(
        kg, state, sd, entry, path_visibility, path_flag);
  }

  return sd->volume_ior;
}

/* Gradient of the index of refraction across a ray at the arc length `t`, times the step.
 *
 * Only the gradient across the ray bends it: along the ray it scales `n d`, which is normalized.
 * It is measured with three points around the ray, at the distance `offset` from it. Such a
 * triangle is exact for a linear field and, as it is turned by 60 degrees from one step to the
 * next, also for the curvature of the field over two steps. Returns the index at the step in
 * `n`. */
ccl_device_inline float3 volume_ior_step_kick(KernelGlobals kg,
                                              IntegratorState state,
                                              ccl_private ShaderData *ccl_restrict sd,
                                              const ccl_private Ray *ccl_restrict ray,
                                              const float t,
                                              const float step_size,
                                              const float offset,
                                              const int step,
                                              const PathRayVisibility path_visibility,
                                              const uint32_t path_flag,
                                              ccl_private float *n)
{
  const float3 P = ray->P + ray->D * t;

  float3 X;
  float3 Y;
  make_orthonormals(ray->D, &X, &Y);
  if (step & 1) {
    X = -X;
    Y = -Y;
  }
  /* Unit vectors at 0, 120 and 240 degrees. */
  const float3 e0 = X;
  const float3 e1 = -0.5f * X + (0.5f * M_SQRT3_F) * Y;
  const float3 e2 = -0.5f * X - (0.5f * M_SQRT3_F) * Y;

  const float f0 = volume_ior_eval(kg, state, sd, P + offset * e0, path_visibility, path_flag);
  const float f1 = volume_ior_eval(kg, state, sd, P + offset * e1, path_visibility, path_flag);
  const float f2 = volume_ior_eval(kg, state, sd, P + offset * e2, path_visibility, path_flag);

  /* A shader may give anything: keep the index positive. */
  *n = fmaxf(1.0f + (f0 + f1 + f2) * (1.0f / 3.0f), 0.01f);

  const float3 kick = (f0 * e0 + f1 * e1 + f2 * e2) * (step_size * 2.0f / (3.0f * offset));
  return (isfinite_safe(kick)) ? kick : zero_float3();
}

/* Lights that a camera path may hit on the way, see lights_intersect(). */
struct VolumeIORLights {
  bool enabled;
  int last_prim;
  int last_object;
  int last_type;
  PathRayVisibility path_visibility;
  uint32_t path_flag;
};

/* Closest intersection along a ray that the volumes of the stack bend, which must have an index
 * of refraction (see volume_ior_stack_active()).
 *
 * The ray is traced from `ray->tmin` to `ray->tmax`, along the length of its path. On return the
 * ray is the last straight segment of the path, from `tmin` on, as described at the top.
 *
 * `rand` offsets the steps. `ior` is zero at the start of a bounce and passed again unchanged
 * after VOLUME_IOR_SEGMENT, to continue behind the segment that was returned. */
ccl_device VolumeIORResult volume_ior_intersect(KernelGlobals kg,
                                                IntegratorState state,
                                                ccl_private Ray *ccl_restrict ray,
                                                const PathRayVisibility visibility,
                                                ccl_private Intersection *ccl_restrict isect,
                                                const float rand,
                                                ccl_private VolumeIORState *ior,
                                                const ccl_private VolumeIORLights *lights)
{
  bool only_ior;
  float detail_size;
  const float step_size = volume_ior_stack_step_size(kg, state, &only_ior, &detail_size);
  kernel_assert(step_size != FLT_MAX);
  /* Half a voxel, or less with steps that are shorter than one. */
  const float offset = 0.5f * fminf(detail_size, step_size);

  const int max_steps = kernel_data.integrator.volume_ior_max_steps;
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  const float t_limit = ray->tmax;

  ShaderDataTinyStorage sd_storage;
  ccl_private ShaderData *sd = AS_SHADER_DATA(&sd_storage);
  shader_setup_from_volume(sd, ray, OBJECT_NONE);
  /* For stochastic texture sampling. */
  sd->lcg_state = lcg_state_init(INTEGRATOR_STATE(state, path, rng_pixel),
                                 INTEGRATOR_STATE(state, path, rng_offset),
                                 INTEGRATOR_STATE(state, path, sample),
                                 0x696f7231);

  int last_prim = (lights) ? lights->last_prim : PRIM_NONE;
  int last_object = (lights) ? lights->last_object : OBJECT_NONE;
  int last_type = (lights) ? lights->last_type : 0;

  /* Start of the segment that is returned, and of the part of it that is still to be traced. */
  float segment_start = ray->tmin;
  float trace_start = ray->tmin;
  float next_step;

  if (ior->pending) {
    const float3 P = ray->P + ray->D * trace_start;
    ray->D = ior->D;
    ray->P = P - ray->D * trace_start;
    next_step = ior->resume;
    ior->pending = false;
  }
  else {
    next_step = trace_start + rand * step_size;
  }

  /* Direction changes that are too small to apply yet: their sum, and the sums of their sizes
   * and of the sizes times the positions, for the center they are applied at. */
  float3 carry = zero_float3();
  float carry_weight = 0.0f;
  float carry_center = 0.0f;
  int carry_steps = 0;
  /* Index of refraction at the last step. */
  float n = 1.0f;

  /* Number of steps traced at once: one where the ray bends, more and more where it does not. */
  int run = 1;
  /* Length of a step in steps, and the number of steps in a row without any gradient. */
  int stride = 1;
  int uniform_steps = 0;

  for (;;) {
    const bool bend = ior->steps < max_steps;
    const float trace_end = (bend) ? fminf(next_step + float(run - 1) * step_size, t_limit) :
                                     t_limit;

    ray->tmin = trace_start;
    ray->tmax = trace_end;
    isect->prim = PRIM_NONE;
    isect->object = OBJECT_NONE;
    bool hit = scene_intersect(kg, ray, visibility, isect);
    if (!hit) {
      isect->prim = PRIM_NONE;
      isect->t = trace_end;
    }
    if (lights && lights->enabled) {
      hit = lights_intersect(kg,
                             state,
                             ray,
                             isect,
                             last_prim,
                             last_object,
                             last_type,
                             lights->path_visibility,
                             lights->path_flag) ||
            hit;
    }
    const float t_end = (hit) ? isect->t : trace_end;

    /* Steps along what was traced, up to the first one at which the direction changes. */
    float3 D = ray->D;
    bool apply = false;
    while (bend && ior->steps < max_steps && next_step <= t_end && next_step < t_limit) {
      const float3 kick = volume_ior_step_kick(
          kg, state, sd, ray, next_step, step_size, offset, ior->steps, visibility, path_flag, &n);
      if (sd->runtime_flag & SR_CACHE_MISS) {
        return VOLUME_IOR_CACHE_MISS;
      }
      ior->steps++;

      if (len_squared(kick) < VOLUME_IOR_UNIFORM_SQUARED) {
        /* Longer steps through uniform air. */
        if (++uniform_steps >= 2) {
          stride = min(stride * 2, VOLUME_IOR_MAX_STRIDE);
        }
        carry_steps += (carry_weight > 0.0f);
        next_step += float(stride) * step_size;
        continue;
      }
      uniform_steps = 0;
      if (stride > 1) {
        /* The index varies again: go back to the first step that was left out. */
        next_step -= float(stride - 1) * step_size;
        stride = 1;
        continue;
      }

      const float weight = len(kick);
      carry += kick;
      carry_weight += weight;
      carry_center += weight * next_step;
      carry_steps += (carry_weight > 0.0f);
      next_step += step_size;

      D = safe_normalize_fallback(n * ray->D + carry, ray->D);
      const float bend_squared = len_squared(D - ray->D);
      if (bend_squared >= VOLUME_IOR_BEND_SQUARED ||
          (carry_steps >= VOLUME_IOR_MAX_DEFER && bend_squared >= VOLUME_IOR_MIN_BEND_SQUARED))
      {
        apply = true;
        break;
      }
    }

    if (!apply && bend && (hit || trace_end >= t_limit) && carry_weight > 0.0f) {
      /* What is left is applied before the ray leaves the volume, or the direction it leaves
       * with would be off by it. */
      D = safe_normalize_fallback(n * ray->D + carry, ray->D);
      apply = len_squared(D - ray->D) >= VOLUME_IOR_MIN_BEND_SQUARED;
    }

    if (apply) {
      /* Not before what was returned already, nor behind what was traced. */
      const float t_bend = clamp(carry_center / carry_weight, segment_start, t_end);
      carry = zero_float3();
      carry_weight = 0.0f;
      carry_center = 0.0f;
      carry_steps = 0;

      if (!only_ior) {
        /* The media up to here are integrated along a straight segment first. */
        ray->tmin = segment_start;
        ray->tmax = t_bend;
        isect->prim = PRIM_NONE;
        isect->object = OBJECT_NONE;
        isect->t = t_bend;
        ior->pending = true;
        ior->D = D;
        ior->resume = next_step;
        return VOLUME_IOR_SEGMENT;
      }

      const float3 P = ray->P + ray->D * t_bend;
      ray->D = D;
      ray->P = P - D * t_bend;
      /* Not at the surface or light the ray left any more. */
      ray->self.object = OBJECT_NONE;
      ray->self.prim = PRIM_NONE;
      last_prim = PRIM_NONE;
      last_object = OBJECT_NONE;
      last_type = 0;

      segment_start = t_bend;
      trace_start = t_bend;
      run = 1;
      continue;
    }

    if (hit || trace_end >= t_limit) {
      ray->tmin = segment_start;
      ray->tmax = t_limit;
      return (hit) ? VOLUME_IOR_HIT : VOLUME_IOR_MISS;
    }

    trace_start = trace_end;
    run = min(run * 2, VOLUME_IOR_MAX_RUN);
  }
}

#endif /* __VOLUME_IOR__ */

/* Closest intersection for the light subpaths of bidirectional methods and for photons, whose
 * loops keep `ior`, zero at first, between the segments of a bounce. */
ccl_device_inline VolumeIORResult volume_ior_light_path_intersect(
    KernelGlobals kg,
    IntegratorState state,
    ccl_private Ray *ccl_restrict ray,
    const PathRayVisibility visibility,
    ccl_private Intersection *ccl_restrict isect,
    ccl_private uint *rng,
    ccl_private VolumeIORState *ior)
{
#ifdef __VOLUME_IOR__
  if (volume_ior_stack_active(kg, state)) {
    const float rand = lcg_step_float(rng);
    const VolumeIORResult result = volume_ior_intersect(
        kg, state, ray, visibility, isect, rand, ior, nullptr);
    if (result != VOLUME_IOR_SEGMENT) {
      ior->pending = false;
      ior->steps = 0;
    }
    return result;
  }
  ior->pending = false;
  ior->steps = 0;
#else
  (void)state;
  (void)rng;
  (void)ior;
#endif
  return scene_intersect(kg, ray, visibility, isect) ? VOLUME_IOR_HIT : VOLUME_IOR_MISS;
}

CCL_NAMESPACE_END
