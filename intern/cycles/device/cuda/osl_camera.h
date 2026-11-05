/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_CUDA

#  include "util/string.h"
#  include "util/vector.h"

CCL_NAMESPACE_BEGIN

class CUDADevice;

/* Custom (OSL) cameras on CUDA, and on OptiX without OSL shading.
 *
 * These devices have no OSL back end for camera shaders that works with every kernel module, so
 * they run the translation of the compiled shader that Metal uses, see
 * `device/metal/osl_camera.h`. Its Metal source is compiled as CUDA with NVRTC, with a
 * compatibility layer for the Metal shading language (`osl_camera_compat.h`), and linked with
 * the camera ray kernel, which NVRTC compiles from the kernel sources that ship with Blender.
 * The kernel enables the camera code of Metal, `__KERNEL_METAL_OSL_CAMERA__` in
 * `camera_sample_custom()`, which calls the translated function and answers its image lookups.
 * The precompiled kernel modules are not changed: the device runs the camera ray kernel of the
 * built module instead of theirs.
 *
 * The kernel is compiled for the kernel module in use, regular or transport, so that it
 * initializes the same path state. Compiled modules are cached on disk.
 *
 * Returns the linked cubin in `cubin`, or false with a description in `error`. */
bool cuda_osl_camera_build(CUDADevice *device,
                           const string &camera_source,
                           bool transport,
                           vector<uint8_t> &cubin,
                           string &error);

CCL_NAMESPACE_END

#endif /* WITH_CUDA */
