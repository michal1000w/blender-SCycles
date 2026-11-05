/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_CUDA

#  include "device/cuda/kernel.h"
#  include "device/cuda/device_impl.h"

CCL_NAMESPACE_BEGIN

void CUDADeviceKernels::load(CUDADevice *device, CUmodule cuModule, const bool light_cache)
{
  for (int i = 0; i < (int)DEVICE_KERNEL_NUM; i++) {
    CUDADeviceKernel &kernel = kernels_[i];
    kernel = CUDADeviceKernel();

    /* Light-cache and GPU guiding kernels only exist in the transport module. */
    const DeviceKernel device_kernel = (DeviceKernel)i;
    const bool transport_only = device_kernel_is_light_cache(device_kernel) ||
                                device_kernel_is_gpu_guiding(device_kernel);
    if (transport_only ? !light_cache : !device_kernel_has_gpu_function(device_kernel)) {
      continue;
    }

    load_kernel(device, cuModule, device_kernel);
  }

  loaded = true;
}

bool CUDADeviceKernels::load_kernel(CUDADevice *device,
                                    CUmodule cuModule,
                                    const DeviceKernel device_kernel)
{
  CUDADeviceKernel &kernel = kernels_[(int)device_kernel];
  kernel = CUDADeviceKernel();

  const std::string function_name = std::string("kernel_gpu_") +
                                    device_kernel_as_string(device_kernel);
  cuda_device_assert(device,
                     cuModuleGetFunction(&kernel.function, cuModule, function_name.c_str()));

  if (kernel.function) {
    cuda_device_assert(device, cuFuncSetCacheConfig(kernel.function, CU_FUNC_CACHE_PREFER_L1));

    cuda_device_assert(
        device,
        cuOccupancyMaxPotentialBlockSize(
            &kernel.min_blocks, &kernel.num_threads_per_block, kernel.function, nullptr, 0, 0));
    return true;
  }

  LOG_ERROR << "Unable to load kernel " << function_name;
  return false;
}

const CUDADeviceKernel &CUDADeviceKernels::get(DeviceKernel kernel) const
{
  return kernels_[(int)kernel];
}

bool CUDADeviceKernels::available(DeviceKernel kernel) const
{
  return kernels_[(int)kernel].function != nullptr;
}

CCL_NAMESPACE_END

#endif /* WITH_CUDA */
