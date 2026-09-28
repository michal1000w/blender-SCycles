/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#ifdef WITH_METAL
#include "device/metal/device_impl.h"
#include "util/log.h"

#define DIFFRACTION_MPS_SOLVE
#define DIFFRACTION_POL_SPLIT
#define DIFFRACTION_INVERSE_REFINEMENT
#define DIFFRACTION_ANALYTIC_ADMITTANCE
#define DIFFRACTION_EXP_NORM 16.0f
#define DIFFRACTION_EXP_TERMS 64
#define DIFFRACTION_LARGE_ORDERS
#define DIFFRACTION_ADAPTIVE_QUEUE
#include "reference_pool.h"
#include "diffraction_shader_source.h"

CCL_NAMESPACE_BEGIN
bool MetalDevice::configure_diffraction_reference(DiffractionGratingCacheOptions &options,
                                                 string &error)
{
  @autoreleasepool {
    try {
      if (!mtlDevice) throw std::runtime_error("Metal device unavailable");
      const string device_name = info.description;
      auto pool = std::make_shared<DiffractionMetalReferencePool>(
          [NSString stringWithUTF8String:diffraction_metal_shader_source], mtlDevice,
          options.cancelled, [device_name] {
            LOG_INFO << "Initialized Metal diffraction pool slot on " << device_name;
          });
      options.reference_solver = [pool](const DiffractionGratingProfile &profile,
          double wavelength, double kx, double ky, int n, int retained,
          DiffractionGratingBlock &out, std::string &message) {
        return pool->solve(profile, wavelength, kx, ky, n, retained, out, message);
      };
      options.reference_backend_key = "metal-mps-wide64-conditioned-lossless-v3";
      error.clear();
      return true;
    }
    catch (const std::exception &exception) {
      error = exception.what();
      return false;
    }
  }
}
CCL_NAMESPACE_END
#endif
