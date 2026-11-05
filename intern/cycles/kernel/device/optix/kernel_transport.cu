/* SPDX-FileCopyrightText: 2011-2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* OptiX kernels for scenes with light-cache transport (bidirectional path tracing, vertex merging
 * and photon mapping) or coherent light. These trace rays while shading: on OptiX that requires
 * ray generation programs, so the surface and volume shading kernels, the light-cache kernels and
 * MNEE run here instead of as CUDA kernels. The other kernels come from the CUDA module.
 *
 * The device loads this module instead of `kernel_optix` only for scenes that use these features,
 * see OptiXDevice::load_kernels(). Surface shading is in kernel_transport_surface.cu and
 * kernel_transport_surface_raytrace.cu: with all transport kernels in one module, compiling it
 * needs more than 20 GB of memory. */

#define __KERNEL_GPU_TRANSPORT__

/* Intersection, miss and hit programs. */
#include "kernel/device/optix/kernel.cu"

#include "kernel/integrator/bidirectional.h"
#include "kernel/integrator/intersect_mnee.h"
#include "kernel/integrator/photon_mapping.h"
/* Defines the background shader evaluation that photon_mapping.h declares. */
#include "kernel/integrator/shade_background.h"
#include "kernel/integrator/shade_surface.h"
#include "kernel/integrator/shade_volume.h"
#include "kernel/integrator/vertex_merging.h"

/* Volume shading. */

extern "C" __global__ void __raygen__kernel_optix_integrator_shade_volume()
{
  const int global_index = optixGetLaunchIndex().x;
  const int path_index = (kernel_params.path_index_array) ?
                             kernel_params.path_index_array[global_index] :
                             global_index;
  integrator_shade_volume(nullptr, path_index, kernel_params.render_buffer);
}

extern "C" __global__ void __raygen__kernel_optix_integrator_shade_volume_ray_marching()
{
  const int global_index = optixGetLaunchIndex().x;
  const int path_index = (kernel_params.path_index_array) ?
                             kernel_params.path_index_array[global_index] :
                             global_index;
  integrator_shade_volume_ray_marching(nullptr, path_index, kernel_params.render_buffer);
}

/* Manifold next event estimation, which BDPT sensor connections use too. */

extern "C" __global__ void __raygen__kernel_optix_integrator_intersect_mnee()
{
  const int global_index = optixGetLaunchIndex().x;
  const int path_index = (kernel_params.path_index_array) ?
                             kernel_params.path_index_array[global_index] :
                             global_index;
  integrator_intersect_mnee(nullptr, path_index);
}

/* Light-cache kernels. The launch width is the work size, the remaining arguments are launch
 * parameters, see OptiXDeviceQueue::enqueue(). */

extern "C" __global__ void __raygen__kernel_optix_integrator_photon_emit()
{
  const uint photon_index = optixGetLaunchIndex().x;
  integrator_photon_emit(
      nullptr, photon_index, photon_index, uint(kernel_params.light_cache_iteration));
}

extern "C" __global__ void __raygen__kernel_optix_integrator_bdpt_light_generate()
{
  const uint light_path_index = optixGetLaunchIndex().x;
  integrator_bdpt_light_generate(nullptr,
                                 light_path_index,
                                 light_path_index,
                                 uint(kernel_params.light_cache_iteration),
                                 uint(kernel_params.light_cache_batch_samples),
                                 uint(kernel_params.light_cache_stage));
}

extern "C" __global__ void __raygen__kernel_optix_integrator_bdpt_sensor_connect()
{
  const uint vertex_index = optixGetLaunchIndex().x;
  integrator_bdpt_sensor_connect(nullptr,
                                 vertex_index,
                                 vertex_index,
                                 uint(kernel_params.light_cache_iteration),
                                 uint(kernel_params.light_cache_batch_samples),
                                 kernel_params.render_buffer);
}
