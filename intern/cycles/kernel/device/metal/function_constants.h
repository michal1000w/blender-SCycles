/* SPDX-FileCopyrightText: 2021-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

enum {
  Kernel_DummyConstant,
#define KERNEL_STRUCT_MEMBER(parent, type, name) KernelData_##parent##_##name,
#include "kernel/data_template.h"

  KernelData_kernel_features,

  /* Whether the rays of the kernel intersect pixel displaced surfaces, see
   * pixel_displacement_intersects(). */
  Kernel_PixelDisplacementRays,
  /* Pixel displacement functions specialized for one evaluator set, see
   * pixel_displacement_evaluator_set(). */
  Kernel_PixelDisplacementSpecialized,
  Kernel_PixelDisplacementEvaluatorSet,
  /* BVH traversal variant of the scene for the specialized functions: bit 0 for object motion,
   * bit 1 for curves. */
  Kernel_PixelDisplacementBVHFeatures,
};

#ifdef __KERNEL_METAL__
#  define KERNEL_STRUCT_MEMBER(parent, type, name) \
    constant type kernel_data_##parent##_##name \
        [[function_constant(KernelData_##parent##_##name)]];
#  include "kernel/data_template.h"

constant bool kernel_pixel_displacement_rays [[function_constant(Kernel_PixelDisplacementRays)]];
constant bool kernel_pixel_displacement_specialized
    [[function_constant(Kernel_PixelDisplacementSpecialized)]];
constant int kernel_pixel_displacement_evaluator_set
    [[function_constant(Kernel_PixelDisplacementEvaluatorSet)]];
constant int kernel_pixel_displacement_bvh_features
    [[function_constant(Kernel_PixelDisplacementBVHFeatures)]];

constant ulong kernel_data_kernel_features [[function_constant(KernelData_kernel_features)]];

#endif
