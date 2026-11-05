/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Constant Globals */

#pragma once

#include "kernel/types.h"

#include "kernel/integrator/state.h"
#include "kernel/util/profiler.h"

#include "util/color.h"
#include "util/types_image.h"

CCL_NAMESPACE_BEGIN

/* Not actually used, just a nullptr pointer that gets passed everywhere, which we
 * hope gets optimized out by the compiler. */
struct KernelGlobalsGPU {
  int unused[1];
};
using KernelGlobals = const ccl_global KernelGlobalsGPU *ccl_restrict;

struct KernelParamsCUDA {
  /* Global scene data and textures */
  KernelData data;
#define KERNEL_DATA_ARRAY(type, name) const type *name;
#define KERNEL_DATA_ARRAY_WRITABLE(type, name) type *name;
#include "kernel/data_arrays.h"

  /* Integrator state */
  IntegratorStateGPU integrator_state;
};

#ifdef __KERNEL_GPU__
__constant__ KernelParamsCUDA kernel_params;
#endif

#if defined(__KERNEL_GPU__) && defined(__KERNEL_GPU_TRANSPORT__)
/* Only rays of the intersection kernels intersect pixel displaced surfaces, as with the Metal
 * `pixel_displacement_rays` function constant. Those kernels pass the address of this object as
 * their KernelGlobals instead of nullptr, see ccl_gpu_intersect_kg. Both are constants in each
 * kernel, so after inlining the test costs nothing. */
__device__ KernelGlobalsGPU kernel_globals_intersect;
#  define ccl_gpu_intersect_kg (&kernel_globals_intersect)
#  define pixel_displacement_rays (kg != nullptr)
#endif

/* Abstraction macros */
#define kernel_data kernel_params.data
#define kernel_data_fetch(name, index) kernel_params.name[(index)]
#define kernel_data_write(name, index, value) kernel_params.name[(index)] = (value)
#define kernel_data_array(name) (kernel_params.name)
#define kernel_integrator_state kernel_params.integrator_state

CCL_NAMESPACE_END
