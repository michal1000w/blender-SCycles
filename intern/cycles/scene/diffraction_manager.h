/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "scene/diffraction.h"
#include "scene/diffraction_albedo.h"
CCL_NAMESPACE_BEGIN
class DeviceScene;
class Device;

/* Scene update lock must be held by callers, as for other scene managers.
 * Cache handles are indices in the last successfully registered set. */
class DiffractionManager {
 public:
  /* Returns a stable index until set_caches replaces the set. Validation
   * failures return -1 and leave existing registrations unchanged. */
  /* Serialized scene-update operation, before parallel shader compilation.
   * Reuses an exact material/build request; callbacks are never retained.
   * Failure or cancellation does not publish a handle. A nonzero maximum_channels
   * rejects responses exceeding the caller's evaluator capacity, including hits. */
  int get_or_build(const DiffractionGratingProfile &profile,
                   const DiffractionGratingCacheOptions &options,
                   DiffractionGratingCacheStats &stats,
                   std::string &error,
                   int maximum_channels = 0);
  int add_cache(const DiffractionGratingCache &cache, std::string &error);
  bool set_caches(std::span<const DiffractionGratingCache> caches, std::string &error);
  /* Registers a separate one-sided GGX albedo table. Metal devices construct
   * natively; other devices use the CPU reference path. The exact request is
   * reused and a failed build never publishes a handle. */
  int get_or_build_albedo(const DiffractionAlbedoRequest &request,
                          Device *builder_device,
                          std::string &error,
                          const std::function<bool()> &cancelled = {});
  /* Two-sided dielectric tables use the Metal builder when available; other
   * devices use the CPU reference. A Metal error is never a CPU fallback. */
  int get_or_build_two_sided_albedo(const DiffractionTwoSidedAlbedoRequest &request,
                                    Device *builder_device,
                                    std::string &error,
                                    const std::function<bool()> &cancelled = {});
  size_t albedo_cache_count() const
  {
    return albedo_descriptors_.size();
  }
  size_t two_sided_albedo_cache_count() const
  {
    return two_sided_descriptors_.size();
  }
  size_t cache_count() const
  {
    return descriptors_.size() / 2;
  }
  bool need_update() const
  {
    return dirty_;
  }
  void device_update(DeviceScene *dscene);
  void device_free(DeviceScene *dscene);

 private:
  DiffractionGratingDeviceBuffers buffers_;
  /* Per cache: (node base,count,cell base,count),
   * (port base,active base,matrix base,mirror flag). All cell offsets are local
   * to their cache; kernels add these bases. Domain has two float4 per cache. */
  std::vector<int4> descriptors_;
  std::vector<float4> domains_;
  struct MaterialCache {
    DiffractionGratingProfile profile;
    DiffractionGratingCacheOptions options;
    DiffractionGratingCacheStats stats;
    int handle;
    size_t maximum_channels;
  };
  std::vector<MaterialCache> materials_;
  std::vector<DiffractionAlbedoRequest> albedo_requests_;
  std::vector<float> albedo_values_;
  std::vector<float> albedo_averages_;
  /* One per table: values base, averages base, mu count, phi count. */
  std::vector<int4> albedo_descriptors_;
  /* Wavelength min/max (nm), sample count, algorithm revision. */
  std::vector<float4> albedo_domains_;
  std::vector<DiffractionTwoSidedAlbedoRequest> two_sided_requests_;
  std::vector<float> two_sided_values_;
  std::vector<float> two_sided_integrals_;
  std::vector<float> two_sided_cross_;
  /* Values, integrals, cross bases and mu count; domain.w stores phi count. */
  std::vector<int4> two_sided_descriptors_;
  std::vector<float4> two_sided_domains_;
  bool dirty_ = false;
};
CCL_NAMESPACE_END
