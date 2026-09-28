/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "kernel/globals.h"

CCL_NAMESPACE_BEGIN

/* Planar bounds seed the geometric solve; the exact triangle union decides
 * whether the solved endpoint is actually on the declared interface. Global
 * primitive IDs may be shared by instances, so object identity is essential. */
ccl_device_inline bool coherent_patch_contains_primitive(KernelGlobals kg,
                                                         const int patch_index,
                                                         const int object,
                                                         const int prim,
                                                         const int type = PRIMITIVE_TRIANGLE)
{
  if (patch_index < 0 || patch_index >= kernel_data.integrator.coherent_patch_count ||
      prim == PRIM_NONE || object == OBJECT_NONE) return false;
  const ccl_global KernelCoherentPatch *patch = &kernel_data_fetch(coherent_patches, patch_index);
  if (patch->primitive_type != (type & ~PRIMITIVE_MOTION) || patch->object != object || patch->primitive_offset < 0 || patch->primitive_count <= 0)
    return false;
  if (patch->shape == 2) {
    return prim >= patch->primitive_offset &&
           prim - patch->primitive_offset < patch->primitive_count;
  }
  int lower = 0, upper = patch->primitive_count;
  while (lower < upper) {
    const int middle = lower + (upper - lower) / 2;
    const int value = kernel_data_fetch(coherent_patch_primitives, patch->primitive_offset + middle);
    if (value < prim) lower = middle + 1;
    else upper = middle;
  }
  return lower < patch->primitive_count &&
         kernel_data_fetch(coherent_patch_primitives, patch->primitive_offset + lower) == prim;
}

ccl_device_inline bool coherent_patch_transition_valid(KernelGlobals kg,
                                                       const int previous_patch,
                                                       const int target_patch,
                                                       const int hit_object,
                                                       const int hit_prim,
                                                       const int hit_type = PRIMITIVE_TRIANGLE)
{
  /* No immediate repeat of the same planar face. A different face of the same
   * object is allowed only after exact target membership, never object skipping. */
  const bool repeated_sphere = target_patch >= 0 &&
      target_patch < kernel_data.integrator.coherent_patch_count &&
      kernel_data_fetch(coherent_patches, target_patch).shape == 1 &&
      kernel_data_fetch(coherent_patches, target_patch).mode == 2;
  return (target_patch != previous_patch || repeated_sphere) &&
         coherent_patch_contains_primitive(kg, target_patch, hit_object, hit_prim, hit_type);
}

ccl_device_inline int coherent_patch_for_hit(KernelGlobals kg, const int object, const int prim,
                                               const int type = PRIMITIVE_TRIANGLE)
{
  for (int index = 0; index < kernel_data.integrator.coherent_patch_count; index++) {
    if (coherent_patch_contains_primitive(kg, index, object, prim, type)) return index;
  }
  return -1;
}

CCL_NAMESPACE_END
