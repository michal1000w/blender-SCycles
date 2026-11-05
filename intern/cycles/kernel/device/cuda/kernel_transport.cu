/* SPDX-FileCopyrightText: 2011-2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* CUDA kernels with light-cache transport (bidirectional path tracing, vertex merging and
 * photon mapping) and pixel-level displacement. The device loads this module instead of the
 * regular kernels only for scenes that use these features, see CUDADevice::load_kernels(). */

#define __KERNEL_GPU_TRANSPORT__

#include "kernel/device/cuda/kernel.cu"
