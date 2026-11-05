/* SPDX-FileCopyrightText: 2011-2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Surface shading with ray-tracing nodes (AO and Bevel) of the transport kernels, and the direct
 * callables of those nodes, see kernel_transport.cu. Loaded only for scenes with such nodes. */

#define __KERNEL_GPU_TRANSPORT__

#include "kernel/device/optix/compat.h"
#include "kernel/device/optix/globals.h"

#include "kernel/device/gpu/image.h" /* Texture lookup uses normal CUDA intrinsics. */

#include "kernel/tables.h"

#include "kernel/integrator/state.h"
#include "kernel/integrator/state_flow.h"
#include "kernel/integrator/state_util.h"

/* The same definitions as the other transport module, kernel_transport.cu. */
#include "kernel/integrator/intersect_closest.h"
#include "kernel/integrator/intersect_dedicated_light.h"
#include "kernel/integrator/intersect_shadow.h"
#include "kernel/integrator/intersect_subsurface.h"
#include "kernel/integrator/intersect_volume_stack.h"

#include "kernel/integrator/bidirectional.h"
#include "kernel/integrator/intersect_mnee.h"
#include "kernel/integrator/photon_mapping.h"
/* Defines the background shader evaluation that photon_mapping.h declares. */
#include "kernel/integrator/shade_background.h"
#include "kernel/integrator/shade_surface.h"
#include "kernel/integrator/shade_volume.h"
#include "kernel/integrator/vertex_merging.h"

extern "C" __global__ void __raygen__kernel_optix_integrator_shade_surface_raytrace()
{
  const int global_index = optixGetLaunchIndex().x;
  const int path_index = (kernel_params.path_index_array) ?
                             kernel_params.path_index_array[global_index] :
                             global_index;
  integrator_shade_surface_raytrace(nullptr, path_index, kernel_params.render_buffer);
}
