/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/state.h"
#include "kernel/integrator/state_util.h"
#include "kernel/integrator/volume_stack.h"

CCL_NAMESPACE_BEGIN

/* Nested Dielectrics
 *
 * Overlapping refractive objects are resolved with the priority scheme of Schmidt and Budge,
 * "Simple Nested Dielectrics in Ray Traced Images" (2002). Materials with a nested priority are
 * media: a path keeps the list of media it is inside of, and where media overlap only the one
 * with the highest priority exists. A surface of a medium that lies inside a medium of higher
 * priority is not there (a false intersection), and paths and shadow rays pass through it. At
 * any other surface (a true intersection) the index of refraction on the outer side is the one
 * of the highest priority medium around it instead of 1.
 *
 * Media are ordered by priority, then by the nesting rank of their object and then by object
 * index. The rank orders media of the same priority, and with it media without any priority that
 * take part automatically, from their geometry: the object which encloses the smaller volume
 * has the higher rank, so an object inside another one always exists, as do ice in a drink and
 * a bubble in ice. Unlike the order in which a path entered the media, used by Waechter and
 * Raab, "Automatic Handling of Materials in Nested Volumes" (Ray Tracing Gems, 2019), this order
 * does not depend on the direction of the path, which light and camera subpaths of the
 * bidirectional integrators need to agree on.
 *
 * An object is one medium, whatever materials its surface has. Only where every material of
 * the object is a closed surface of its own, like a glass and its drink in one mesh, each of
 * them is a medium. They share the rank of the object, so among them the one which the path
 * entered last exists, which is right wherever one of them contains another.
 *
 * The volume of a medium follows the same rule: the volume stack only contains the volume of
 * the highest priority medium, besides the volumes of materials without a priority. */

#ifdef __NESTED_DIELECTRICS__

ccl_device_forceinline bool nested_dielectrics_enabled(KernelGlobals kg)
{
  return (kernel_data.kernel_features & KERNEL_FEATURE_NESTED_DIELECTRICS) != 0;
}

template<const bool shadow, typename IntegratorGenericState>
ccl_device_forceinline MediumStack medium_stack_read(const IntegratorGenericState state,
                                                     const int i)
{
  if constexpr (shadow) {
    const MediumStack entry = {INTEGRATOR_STATE_ARRAY(state, shadow_medium_stack, i, object),
                               INTEGRATOR_STATE_ARRAY(state, shadow_medium_stack, i, shader),
                               0.0f};
    return entry;
  }
  else {
    const MediumStack entry = {INTEGRATOR_STATE_ARRAY(state, medium_stack, i, object),
                               INTEGRATOR_STATE_ARRAY(state, medium_stack, i, shader),
                               INTEGRATOR_STATE_ARRAY(state, medium_stack, i, ior)};
    return entry;
  }

#  ifdef __KERNEL_GPU__
  /* Silence false positive warning with some GPU compilers. */
  MediumStack stack = {};
  return stack;
#  endif
}

template<const bool shadow, typename IntegratorGenericState>
ccl_device_forceinline void medium_stack_write(IntegratorGenericState state,
                                               const int i,
                                               const MediumStack entry)
{
  if constexpr (shadow) {
    INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_medium_stack, i, object) = entry.object;
    INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_medium_stack, i, shader) = entry.shader;
  }
  else {
    INTEGRATOR_STATE_ARRAY_WRITE(state, medium_stack, i, object) = entry.object;
    INTEGRATOR_STATE_ARRAY_WRITE(state, medium_stack, i, shader) = entry.shader;
    INTEGRATOR_STATE_ARRAY_WRITE(state, medium_stack, i, ior) = entry.ior;
  }
}

/* Index of refraction of a medium in the list: the one evaluated where the path entered it, or
 * the constant of its shader where the path never shaded its surface. */
ccl_device_forceinline float medium_ior(KernelGlobals kg, const ccl_private MediumStack &entry)
{
  return (entry.ior > 0.0f) ? entry.ior :
                              kernel_data_fetch(shaders, (entry.shader & SHADER_MASK)).nested_ior;
}

/* Order of a medium among the ones of the same priority: its priority and the nesting rank of
 * its object in one key. */
ccl_device_forceinline uint medium_order(KernelGlobals kg, const int shader, const int object)
{
  const uint priority = uint(kernel_data_fetch(shaders, (shader & SHADER_MASK)).nested_priority);
  const uint rank = kernel_data_fetch(object_flag, object) >> SD_OBJECT_NESTED_RANK_SHIFT;
  return (priority << (32 - SD_OBJECT_NESTED_RANK_SHIFT)) | rank;
}

/* Strict total order of the media of different objects. */
ccl_device_forceinline bool medium_overrides(const uint order_a,
                                             const int object_a,
                                             const uint order_b,
                                             const int object_b)
{
  return (order_a > order_b) || (order_a == order_b && object_a > object_b);
}

/* Every material of the object is a medium of its own. */
ccl_device_forceinline bool medium_per_shader(KernelGlobals kg, const int object)
{
  return (kernel_data_fetch(object_flag, object) & SD_OBJECT_NESTED_PER_SHADER) != 0;
}

ccl_device_forceinline bool medium_same(const int object_a,
                                        const int shader_a,
                                        const int object_b,
                                        const int shader_b,
                                        const bool per_shader)
{
  return object_a == object_b && (!per_shader || ((shader_a ^ shader_b) & SHADER_MASK) == 0);
}

/* Media without the plumbing of an integrator state, for manifold walks. */
struct MediumList {
  MediumStack entry[MAX_MEDIUM_STACK_SIZE];
  int size;
};

struct NestedDielectricHit {
  /* The surface lies inside a medium that overrides it, so it does not exist. */
  bool is_false;
  /* Index of refraction on the outer side of the surface. */
  float medium_ior;
};

/* Fold one medium of the list into the classification of a hit on `object`. */
struct NestedDielectricScan {
  /* Instances of the medium itself in the list, and the position of the last one. */
  bool per_shader;
  int self_count;
  int self_index;
  /* The medium of highest order among the others, and its position. */
  bool has_other;
  uint other_order;
  int other_object;
  int other_index;
  float other_ior;
};

ccl_device_forceinline NestedDielectricScan nested_dielectric_scan_begin(KernelGlobals kg,
                                                                         const int object)
{
  NestedDielectricScan scan;
  scan.per_shader = medium_per_shader(kg, object);
  scan.self_count = 0;
  scan.self_index = -1;
  scan.has_other = false;
  scan.other_order = 0;
  scan.other_object = OBJECT_NONE;
  scan.other_index = -1;
  scan.other_ior = 1.0f;
  return scan;
}

ccl_device_forceinline void nested_dielectric_scan_entry(KernelGlobals kg,
                                                         ccl_private NestedDielectricScan &scan,
                                                         const ccl_private MediumStack &entry,
                                                         const int index,
                                                         const int object,
                                                         const int shader)
{
  if (medium_same(entry.object, entry.shader, object, shader, scan.per_shader)) {
    scan.self_count++;
    scan.self_index = index;
    return;
  }
  const uint order = medium_order(kg, entry.shader, entry.object);
  /* Media of one object are in the order the path entered them. */
  if (!scan.has_other ||
      medium_overrides(order, entry.object, scan.other_order, scan.other_object) ||
      (order == scan.other_order && entry.object == scan.other_object))
  {
    scan.has_other = true;
    scan.other_order = order;
    scan.other_object = entry.object;
    scan.other_index = index;
    scan.other_ior = medium_ior(kg, entry);
  }
}

ccl_device_forceinline NestedDielectricHit nested_dielectric_scan_end(
    KernelGlobals kg,
    const ccl_private NestedDielectricScan &scan,
    const int object,
    const int shader,
    const bool backfacing)
{
  NestedDielectricHit hit;
  /* A medium of higher order around this surface replaces it. Another medium of the same
   * object does if the path entered it later, which it cannot have where it enters this one. */
  const uint order = medium_order(kg, shader, object);
  const bool overridden = scan.has_other &&
                          ((scan.other_object == object && scan.other_order == order) ?
                               (backfacing && scan.other_index > scan.self_index) :
                               medium_overrides(
                                   scan.other_order, scan.other_object, order, object));
  /* Surfaces inside the object itself, where a mesh intersects itself or instances of the list
   * got out of balance: only the outermost surface is a boundary of the medium. */
  const bool interior = backfacing ? (scan.self_count > 1) : (scan.self_count > 0);
  hit.is_false = overridden || interior;
  hit.medium_ior = scan.other_ior;
  return hit;
}

/* Classify the hit of a surface with a nested priority against the media of the path. */
template<const bool shadow, typename IntegratorGenericState>
ccl_device_inline NestedDielectricHit nested_dielectric_hit(KernelGlobals kg,
                                                            const IntegratorGenericState state,
                                                            const int object,
                                                            const int shader,
                                                            const bool backfacing)
{
  NestedDielectricScan scan = nested_dielectric_scan_begin(kg, object);
  for (int i = 0;; i++) {
    const MediumStack entry = medium_stack_read<shadow>(state, i);
    if (entry.shader == SHADER_NONE) {
      break;
    }
    nested_dielectric_scan_entry(kg, scan, entry, i, object, shader);
  }
  return nested_dielectric_scan_end(kg, scan, object, shader, backfacing);
}

ccl_device_inline NestedDielectricHit nested_dielectric_hit(KernelGlobals kg,
                                                            const ccl_private MediumList *list,
                                                            const int object,
                                                            const int shader,
                                                            const bool backfacing)
{
  NestedDielectricScan scan = nested_dielectric_scan_begin(kg, object);
  for (int i = 0; i < list->size; i++) {
    nested_dielectric_scan_entry(kg, scan, list->entry[i], i, object, shader);
  }
  return nested_dielectric_scan_end(kg, scan, object, shader, backfacing);
}

/* The medium that exists at the current position of the path, if any. */
template<const bool shadow, typename IntegratorGenericState>
ccl_device_inline bool medium_stack_top(KernelGlobals kg,
                                        const IntegratorGenericState state,
                                        ccl_private MediumStack *top)
{
  bool found = false;
  uint top_order = 0;
  for (int i = 0;; i++) {
    const MediumStack entry = medium_stack_read<shadow>(state, i);
    if (entry.shader == SHADER_NONE) {
      break;
    }
    const uint order = medium_order(kg, entry.shader, entry.object);
    if (!found || medium_overrides(order, entry.object, top_order, top->object) ||
        (order == top_order && entry.object == top->object))
    {
      found = true;
      top_order = order;
      *top = entry;
    }
  }
  return found;
}

/* The path crossed the surface of a medium: add it to, or remove it from the list. */
template<const bool shadow, typename IntegratorGenericState>
ccl_device_inline void medium_stack_enter_exit(KernelGlobals kg,
                                               IntegratorGenericState state,
                                               const int object,
                                               const int shader,
                                               const bool backfacing,
                                               const float ior)
{
  const MediumStack empty_entry = {OBJECT_NONE, SHADER_NONE, 0.0f};

  if (backfacing) {
    /* Remove the most recent instance, keeping the order of the others. A path which leaves a
     * medium it never entered has nothing to remove. */
    const bool per_shader = medium_per_shader(kg, object);
    int last = -1;
    int size = 0;
    for (;; size++) {
      const MediumStack entry = medium_stack_read<shadow>(state, size);
      if (entry.shader == SHADER_NONE) {
        break;
      }
      if (medium_same(entry.object, entry.shader, object, shader, per_shader)) {
        last = size;
      }
    }
    if (last < 0) {
      return;
    }
    for (int i = last; i < size - 1; i++) {
      medium_stack_write<shadow>(state, i, medium_stack_read<shadow>(state, i + 1));
    }
    medium_stack_write<shadow>(state, size - 1, empty_entry);
  }
  else {
    int size = 0;
    for (;; size++) {
      if (medium_stack_read<shadow>(state, size).shader == SHADER_NONE) {
        break;
      }
    }
    /* If we exceed the stack limit, ignore. */
    if (size >= int(kernel_data.medium_stack_size) - 1) {
      return;
    }
    const MediumStack new_entry = {object, shader, ior};
    medium_stack_write<shadow>(state, size, new_entry);
    medium_stack_write<shadow>(state, size + 1, empty_entry);
  }
}

ccl_device_inline void medium_list_enter_exit(KernelGlobals kg,
                                              ccl_private MediumList *list,
                                              const int object,
                                              const int shader,
                                              const bool backfacing,
                                              const float ior)
{
  if (backfacing) {
    const bool per_shader = medium_per_shader(kg, object);
    int last = -1;
    for (int i = 0; i < list->size; i++) {
      if (medium_same(list->entry[i].object, list->entry[i].shader, object, shader, per_shader)) {
        last = i;
      }
    }
    if (last < 0) {
      return;
    }
    for (int i = last; i < list->size - 1; i++) {
      list->entry[i] = list->entry[i + 1];
    }
    list->size--;
  }
  else if (list->size < int(kernel_data.medium_stack_size) - 1) {
    const MediumStack new_entry = {object, shader, ior};
    list->entry[list->size++] = new_entry;
  }
}

template<typename IntegratorGenericState>
ccl_device_inline void medium_list_from_state(const IntegratorGenericState state,
                                              ccl_private MediumList *list)
{
  list->size = 0;
  for (int i = 0; i < MAX_MEDIUM_STACK_SIZE - 1; i++) {
    const MediumStack entry = medium_stack_read<false>(state, i);
    if (entry.shader == SHADER_NONE) {
      break;
    }
    list->entry[list->size++] = entry;
  }
}

/* Make the volume stack contain the volume of the medium that exists at the position of the
 * path, and none of the media it overrides. */
template<const bool shadow, typename IntegratorGenericState>
ccl_device_inline void medium_stack_sync_volume(KernelGlobals kg, IntegratorGenericState state)
{
#  ifdef __VOLUME__
  if (!(kernel_data.kernel_features & KERNEL_FEATURE_VOLUME)) {
    return;
  }

  /* Remove the volumes of all media, keeping the order of the others: the world volume stays
   * the first entry. */
  int size = 0;
  for (int i = 0;; i++) {
    const VolumeStack entry = volume_stack_read<shadow>(state, i);
    if (entry.shader == SHADER_NONE) {
      break;
    }
    const int shader_flag = kernel_data_fetch(shaders, (entry.shader & SHADER_MASK)).flags;
    if (shader_flag & SD_HAS_NESTED_PRIORITY) {
      continue;
    }
    if (size != i) {
      volume_stack_write<shadow>(state, size, entry);
    }
    size++;
  }

  MediumStack top ccl_optional_struct_init;
  if (medium_stack_top<shadow>(kg, state, &top) &&
      (kernel_data_fetch(shaders, (top.shader & SHADER_MASK)).flags & SD_HAS_VOLUME) &&
      size < int(kernel_data.volume_stack_size) - 1)
  {
    const VolumeStack volume_entry = {top.object, top.shader};
    volume_stack_write<shadow>(state, size, volume_entry);
    size++;
  }

  const VolumeStack empty_entry = {OBJECT_NONE, SHADER_NONE};
  volume_stack_write<shadow>(state, size, empty_entry);
#  else
  (void)kg;
  (void)state;
#  endif
}

ccl_device_forceinline void integrator_state_medium_stack_clear(KernelGlobals kg,
                                                                IntegratorState state)
{
  if (nested_dielectrics_enabled(kg)) {
    INTEGRATOR_STATE_ARRAY_WRITE(state, medium_stack, 0, object) = OBJECT_NONE;
    INTEGRATOR_STATE_ARRAY_WRITE(state, medium_stack, 0, shader) = SHADER_NONE;
    INTEGRATOR_STATE_ARRAY_WRITE(state, medium_stack, 0, ior) = 0.0f;
  }
}

ccl_device_forceinline void integrator_state_copy_medium_stack_to_shadow(
    KernelGlobals kg, IntegratorShadowState shadow_state, ConstIntegratorState state)
{
  if (nested_dielectrics_enabled(kg)) {
    int index = 0;
    int shader;
    do {
      shader = INTEGRATOR_STATE_ARRAY(state, medium_stack, index, shader);

      INTEGRATOR_STATE_ARRAY_WRITE(shadow_state, shadow_medium_stack, index, object) =
          INTEGRATOR_STATE_ARRAY(state, medium_stack, index, object);
      INTEGRATOR_STATE_ARRAY_WRITE(shadow_state, shadow_medium_stack, index, shader) = shader;

      ++index;
    } while (shader != SHADER_NONE);
  }
}

/* Set the medium around a surface with a nested priority before evaluating its shader, and
 * return if the surface is a false intersection. */
ccl_device_inline bool nested_dielectric_surface_setup(KernelGlobals kg,
                                                       ConstIntegratorState state,
                                                       ccl_private ShaderData *sd)
{
  if (!(sd->shader_flag & SD_HAS_NESTED_PRIORITY) || !nested_dielectrics_enabled(kg)) {
    return false;
  }
  const NestedDielectricHit hit = nested_dielectric_hit<false>(
      kg, state, sd->object, sd->shader, (sd->runtime_flag & SR_BACKFACING) != 0);
  sd->medium_ior = hit.medium_ior;
  return hit.is_false;
}

#endif /* __NESTED_DIELECTRICS__ */

/* The path or shadow ray continues on the other side of a surface: update the media and the
 * volumes it is inside of. */
template<const bool shadow, typename IntegratorGenericState>
ccl_device_inline void path_media_enter_exit(KernelGlobals kg,
                                             IntegratorGenericState state,
                                             const ccl_private ShaderData *sd)
{
#ifdef __NESTED_DIELECTRICS__
  if ((sd->shader_flag & SD_HAS_NESTED_PRIORITY) && nested_dielectrics_enabled(kg)) {
    medium_stack_enter_exit<shadow>(kg,
                                    state,
                                    sd->object,
                                    sd->shader,
                                    (sd->runtime_flag & SR_BACKFACING) != 0,
                                    sd->interior_ior);
    medium_stack_sync_volume<shadow>(kg, state);
    return;
  }
#endif
#ifdef __VOLUME__
  volume_stack_enter_exit<shadow>(kg, state, sd);
#else
  (void)kg;
  (void)state;
  (void)sd;
#endif
}

CCL_NAMESPACE_END
