/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/bvh/bvh.h"

#include "kernel/geom/shader_data.h"

#include "kernel/integrator/intersect_closest.h"
#include "kernel/integrator/nested_dielectrics.h"
#include "kernel/integrator/volume_stack.h"

CCL_NAMESPACE_BEGIN

/* Surfaces which a probe ray of the volume stack records at most: two for every volume and
 * nested dielectric medium that the stacks have room for. */
#define VOLUME_STACK_PROBE_MAX_HITS (2 * (MAX_VOLUME_STACK_SIZE + MAX_MEDIUM_STACK_SIZE))

ccl_device_forceinline uint volume_stack_probe_hits(KernelGlobals kg)
{
  uint size = kernel_data.volume_stack_size;
#ifdef __NESTED_DIELECTRICS__
  if (nested_dielectrics_enabled(kg)) {
    size += kernel_data.medium_stack_size;
  }
#endif
  return 2 * size;
}

ccl_device void integrator_volume_stack_update_for_subsurface(KernelGlobals kg,
                                                              IntegratorState state,
                                                              const float3 from_P,
                                                              const float3 to_P)
{
#ifdef __VOLUME__
  PROFILING_INIT(kg, PROFILING_INTERSECT_VOLUME_STACK);

  ShaderDataTinyStorage stack_sd_storage;
  ccl_private ShaderData *stack_sd = AS_SHADER_DATA(&stack_sd_storage);

  kernel_assert(kernel_data.integrator.use_volumes ||
                kernel_data.integrator.use_nested_dielectrics);

  Ray volume_ray ccl_optional_struct_init;
  volume_ray.P = from_P;
  volume_ray.D = safe_normalize_len(to_P - from_P, &volume_ray.tmax);
  volume_ray.tmin = 0.0f;
  volume_ray.self.object = INTEGRATOR_STATE(state, isect, object);
  volume_ray.self.prim = INTEGRATOR_STATE(state, isect, prim);
  volume_ray.self.light_object = OBJECT_NONE;
  volume_ray.self.light_prim = PRIM_NONE;
  /* Store to avoid global fetches on every intersection step. */
  const uint max_hits = volume_stack_probe_hits(kg);

  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  const PathRayVisibility visibility = SHADOW_CATCHER_PATH_VISIBILITY(path_flag,
                                                                      PATH_RAY_VISIBILITY_ALL);

#  ifdef __VOLUME_RECORD_ALL__
  Intersection hits[VOLUME_STACK_PROBE_MAX_HITS + 1];
  const uint num_hits = scene_intersect_volume(kg, &volume_ray, hits, max_hits, visibility);
  if (num_hits > 0) {
    Intersection *isect = hits;

    qsort(hits, num_hits, sizeof(Intersection), intersections_compare);

    for (uint hit = 0; hit < num_hits; ++hit, ++isect) {
      /* Ignore self, SSS itself already enters and exits the object. */
      if (isect->object == volume_ray.self.object) {
        continue;
      }
      shader_setup_from_ray(kg, stack_sd, &volume_ray, isect);
      path_media_enter_exit<false>(kg, state, stack_sd);
    }
  }
#  else
  Intersection isect;
  int step = 0;
  while (step < max_hits && scene_intersect_volume(kg, &volume_ray, &isect, visibility))
  {
    /* Ignore self, SSS itself already enters and exits the object. */
    if (isect.object != volume_ray.self.object) {
      shader_setup_from_ray(kg, stack_sd, &volume_ray, &isect);
      path_media_enter_exit<false>(kg, state, stack_sd);
    }
    /* Move ray forward. */
    volume_ray.tmin = intersection_t_offset(isect.t);
    volume_ray.self.object = isect.object;
    volume_ray.self.prim = isect.prim;
    ++step;
  }
#  endif
#endif
}

#ifdef __VOLUME__
/* The origin of the path is inside the object of `stack_sd`: add it to the volume stack, or to
 * the nested dielectric media if its material has a nested priority. The index of refraction of
 * such a medium is the one of its shader, as its surface was not shaded. */
ccl_device_inline void integrator_volume_stack_init_add(KernelGlobals kg,
                                                        IntegratorState state,
                                                        const ccl_private ShaderData *stack_sd,
                                                        const int volume_stack_size,
                                                        ccl_private int *stack_index,
                                                        ccl_private int *medium_index)
{
#  ifdef __NESTED_DIELECTRICS__
  if ((stack_sd->shader_flag & SD_HAS_NESTED_PRIORITY) && nested_dielectrics_enabled(kg)) {
    for (int i = 0; i < *medium_index; ++i) {
      /* Don't add intersections twice. */
      if (INTEGRATOR_STATE_ARRAY(state, medium_stack, i, object) == stack_sd->object) {
        return;
      }
    }
    if (*medium_index < int(kernel_data.medium_stack_size) - 1) {
      const MediumStack new_entry = {stack_sd->object, stack_sd->shader, 0.0f};
      medium_stack_write<false>(state, *medium_index, new_entry);
      ++(*medium_index);
    }
    return;
  }
#  else
  (void)medium_index;
#  endif

  if (!(stack_sd->shader_flag & SD_HAS_VOLUME)) {
    return;
  }
  for (int i = 0; i < *stack_index; ++i) {
    /* Don't add intersections twice. */
    const VolumeStack entry = integrator_state_read_volume_stack(state, i);
    if (entry.object == stack_sd->object) {
      return;
    }
  }
  if (*stack_index < volume_stack_size - 1) {
    const VolumeStack new_entry = {stack_sd->object, stack_sd->shader};
    integrator_state_write_volume_stack(state, *stack_index, new_entry);
    ++(*stack_index);
  }
}
#endif

ccl_device void integrator_volume_stack_init(KernelGlobals kg,
                                             IntegratorState state,
                                             const PathRayVisibility visibility)
{
#ifdef __VOLUME__
  PROFILING_INIT(kg, PROFILING_INTERSECT_VOLUME_STACK);

  ShaderDataTinyStorage stack_sd_storage;
  ccl_private ShaderData *stack_sd = AS_SHADER_DATA(&stack_sd_storage);

  Ray volume_ray ccl_optional_struct_init;
  integrator_state_read_ray(state, &volume_ray);

  /* Trace ray in random direction. Any direction works, Z up is a guess to get the
   * fewest hits. */
  volume_ray.D = make_float3(0.0f, 0.0f, 1.0f);
  volume_ray.tmin = 0.0f;
  volume_ray.tmax = FLT_MAX;
  volume_ray.self.object = OBJECT_NONE;
  volume_ray.self.prim = PRIM_NONE;
  volume_ray.self.light_object = OBJECT_NONE;
  volume_ray.self.light_prim = PRIM_NONE;

  int stack_index = 0;
  int medium_index = 0;
  int enclosed_index = 0;

  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  const PathRayVisibility stack_visibility = SHADOW_CATCHER_PATH_VISIBILITY(path_flag, visibility);

  /* Initialize volume stack with background volume For shadow catcher the
   * background volume is always assumed to be CG. */
  if ((kernel_data.kernel_features & KERNEL_FEATURE_VOLUME) &&
      kernel_data.background.volume_shader != SHADER_NONE)
  {
    if (!(path_flag & PATH_RAY_SHADOW_CATCHER_PASS)) {
      INTEGRATOR_STATE_ARRAY_WRITE(
          state, volume_stack, stack_index, object) = kernel_data.background.object_index;
      INTEGRATOR_STATE_ARRAY_WRITE(
          state, volume_stack, stack_index, shader) = kernel_data.background.volume_shader;
      stack_index++;
    }
  }

  /* Store to avoid global fetches on every intersection step. */
  const uint volume_stack_size = kernel_data.volume_stack_size;
  const uint max_hits = volume_stack_probe_hits(kg);

#  ifdef __VOLUME_RECORD_ALL__
  Intersection hits[VOLUME_STACK_PROBE_MAX_HITS + 1];
  const uint num_hits = scene_intersect_volume(kg, &volume_ray, hits, max_hits, stack_visibility);
  if (num_hits > 0) {
    int enclosed_volumes[VOLUME_STACK_PROBE_MAX_HITS];
    Intersection *isect = hits;

    qsort(hits, num_hits, sizeof(Intersection), intersections_compare);

    for (uint hit = 0; hit < num_hits; ++hit, ++isect) {
      shader_setup_from_ray(kg, stack_sd, &volume_ray, isect);
      if (stack_sd->runtime_flag & SR_BACKFACING) {
        bool need_add = true;
        for (int i = 0; i < enclosed_index && need_add; ++i) {
          /* If ray exited the volume and never entered to that volume
           * it means that camera is inside such a volume.
           */
          if (enclosed_volumes[i] == stack_sd->object) {
            need_add = false;
          }
        }
        if (need_add) {
          integrator_volume_stack_init_add(
              kg, state, stack_sd, volume_stack_size, &stack_index, &medium_index);
        }
      }
      else if (enclosed_index < VOLUME_STACK_PROBE_MAX_HITS) {
        /* If ray from camera enters the volume, this volume shouldn't
         * be added to the stack on exit.
         */
        enclosed_volumes[enclosed_index++] = stack_sd->object;
      }
    }
  }
#  else
  /* CUDA does not support definition of a variable size arrays, so use the maximum possible. */
  int enclosed_volumes[VOLUME_STACK_PROBE_MAX_HITS];
  int step = 0;

  while (stack_index < volume_stack_size - 1 && enclosed_index < VOLUME_STACK_PROBE_MAX_HITS - 1 &&
         step < max_hits)
  {
    Intersection isect;
    if (!scene_intersect_volume(kg, &volume_ray, &isect, stack_visibility)) {
      break;
    }

    shader_setup_from_ray(kg, stack_sd, &volume_ray, &isect);
    if (stack_sd->runtime_flag & SR_BACKFACING) {
      /* If ray exited the volume and never entered to that volume
       * it means that camera is inside such a volume.
       */
      bool need_add = true;
      for (int i = 0; i < enclosed_index && need_add; ++i) {
        /* If ray exited the volume and never entered to that volume
         * it means that camera is inside such a volume.
         */
        if (enclosed_volumes[i] == stack_sd->object) {
          need_add = false;
        }
      }
      if (need_add) {
        integrator_volume_stack_init_add(
            kg, state, stack_sd, volume_stack_size, &stack_index, &medium_index);
      }
    }
    else {
      /* If ray from camera enters the volume, this volume shouldn't
       * be added to the stack on exit.
       */
      enclosed_volumes[enclosed_index++] = stack_sd->object;
    }

    /* Move ray forward. */
    volume_ray.tmin = intersection_t_offset(isect.t);
    volume_ray.self.object = isect.object;
    volume_ray.self.prim = isect.prim;
    ++step;
  }
#  endif

  /* Write terminator. */
  if (kernel_data.kernel_features & KERNEL_FEATURE_VOLUME) {
    const VolumeStack new_entry = {OBJECT_NONE, SHADER_NONE};
    integrator_state_write_volume_stack(state, stack_index, new_entry);
  }
#  ifdef __NESTED_DIELECTRICS__
  if (nested_dielectrics_enabled(kg)) {
    const MediumStack empty_entry = {OBJECT_NONE, SHADER_NONE, 0.0f};
    medium_stack_write<false>(state, medium_index, empty_entry);
    /* Only the medium of highest priority has its volume at the origin of the path. */
    medium_stack_sync_volume<false>(kg, state);
  }
#  endif
#endif
}

ccl_device void integrator_intersect_volume_stack(KernelGlobals kg, IntegratorState state)
{
#ifdef __VOLUME__
  integrator_volume_stack_init(kg, state, PATH_RAY_VISIBILITY_CAMERA);

#  ifdef __SHADOW_CATCHER__
  if (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SHADOW_CATCHER_PASS) {
    /* Volume stack re-init for shadow catcher, continue with shading of hit. */
    integrator_intersect_next_kernel_after_shadow_catcher_volume<
        DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK>(kg, state);
  }
  else
#  endif
  {
    /* Volume stack init for camera rays, continue with intersection of camera ray. */
    integrator_path_next(state,
                         DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK,
                         DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST);
  }
#endif
}

CCL_NAMESPACE_END
