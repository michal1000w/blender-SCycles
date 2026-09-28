/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction_manager.h"
#include "scene/diffraction_tensor.h"
#include "device/device.h"
#include "util/math.h"
#include <limits>
#include <tuple>
CCL_NAMESPACE_BEGIN

int DiffractionManager::get_or_build_albedo(const DiffractionAlbedoRequest &request,
                                             Device *builder_device,
                                             std::string &error,
                                             const std::function<bool()> &cancelled)
{
  if (cancelled && cancelled()) {
    error = "GGX diffraction albedo construction cancelled";
    return -1;
  }
  if (!diffraction_albedo_validate_request(request, error)) return -1;
  for (size_t i = 0; i < albedo_requests_.size(); ++i) {
    if (albedo_requests_[i] == request) {
      error.clear();
      return int(i);
    }
  }

  DiffractionAlbedoTable table;
  const bool built = builder_device && builder_device->info.type == DEVICE_METAL ?
                         builder_device->build_diffraction_albedo(request, table, error, cancelled) :
                         diffraction_albedo_build_cpu(request, table, error, cancelled);
  if (!built) return -1;
  if (cancelled && cancelled()) {
    error = "GGX diffraction albedo construction cancelled";
    return -1;
  }
  if (!(table.request == request)) {
    error = "GGX diffraction albedo builder returned a different request";
    return -1;
  }
  if (!diffraction_albedo_validate_table(table, error)) return -1;

  const size_t limit = std::numeric_limits<int>::max();
  if (albedo_cache_count() >= limit || albedo_values_.size() > limit - table.values.size() ||
      albedo_averages_.size() > limit - table.averages.size()) {
    error = "Combined GGX diffraction albedo buffers exceed int32 addressing";
    return -1;
  }
  /* Reserve every destination before changing any size. Allocation failure
   * must not leave a descriptor published without its domain/request. */
  try {
    albedo_values_.reserve(albedo_values_.size() + table.values.size());
    albedo_averages_.reserve(albedo_averages_.size() + table.averages.size());
    albedo_descriptors_.reserve(albedo_descriptors_.size() + 1);
    albedo_domains_.reserve(albedo_domains_.size() + 1);
    albedo_requests_.reserve(albedo_requests_.size() + 1);
  }
  catch (const std::bad_alloc &) {
    error = "Failed to allocate GGX diffraction albedo buffers";
    return -1;
  }
  const int handle = int(albedo_cache_count());
  const int4 descriptor = make_int4(int(albedo_values_.size()),
                                    int(albedo_averages_.size()),
                                    request.mu_count,
                                    request.phi_count);
  const float4 domain = make_float4(request.wavelength_min_nm,
                                     request.wavelength_max_nm,
                                     float(request.wavelength_count),
                                     float(request.algorithm_revision));
  albedo_values_.insert(albedo_values_.end(), table.values.begin(), table.values.end());
  albedo_averages_.insert(albedo_averages_.end(), table.averages.begin(), table.averages.end());
  albedo_descriptors_.push_back(descriptor);
  albedo_domains_.push_back(domain);
  albedo_requests_.push_back(request);
  dirty_ = true;
  error.clear();
  return handle;
}

int DiffractionManager::get_or_build_two_sided_albedo(
    const DiffractionTwoSidedAlbedoRequest &request,
    Device *builder_device,
    std::string &error,
    const std::function<bool()> &cancelled)
{
  DiffractionTwoSidedAlbedoRequest canonical = request;
  /* The film index has no effect at zero physical thickness. Share the
   * uncoated table regardless of the inactive Film IOR socket value. */
  if (canonical.film_thickness_nm == 0.0f) {
    canonical.film_ior = 1.0f;
  }
  if (cancelled && cancelled()) {
    error = "Two-sided GGX diffraction albedo construction cancelled";
    return -1;
  }
  for (size_t i = 0; i < two_sided_requests_.size(); ++i) {
    if (two_sided_requests_[i] == canonical) {
      error.clear();
      return int(i);
    }
  }
  DiffractionTwoSidedAlbedoTable table;
  const bool built = builder_device && builder_device->info.type == DEVICE_METAL ?
                         builder_device->build_diffraction_two_sided_albedo(
                             canonical, table, error, cancelled) :
                         diffraction_two_sided_albedo_build_cpu(
                             canonical, table, error, cancelled);
  if (!built) return -1;
  if (cancelled && cancelled()) {
    error = "Two-sided GGX diffraction albedo construction cancelled";
    return -1;
  }
  if (!(table.request == canonical) || !diffraction_two_sided_albedo_validate(table, error)) {
    if (error.empty()) error = "Two-sided GGX diffraction albedo request changed during construction";
    return -1;
  }
  const size_t limit = std::numeric_limits<int>::max();
  if (two_sided_albedo_cache_count() >= limit ||
      two_sided_values_.size() > limit - table.deficits.size() ||
      two_sided_integrals_.size() > limit - table.integrals.size() ||
      two_sided_cross_.size() > limit - table.cross_fractions.size())
  {
    error = "Two-sided GGX diffraction albedo buffers exceed int32 addressing";
    return -1;
  }
  try {
    two_sided_values_.reserve(two_sided_values_.size() + table.deficits.size());
    two_sided_integrals_.reserve(two_sided_integrals_.size() + table.integrals.size());
    two_sided_cross_.reserve(two_sided_cross_.size() + table.cross_fractions.size());
    two_sided_descriptors_.reserve(two_sided_descriptors_.size() + 1);
    two_sided_domains_.reserve(two_sided_domains_.size() + 1);
    two_sided_requests_.reserve(two_sided_requests_.size() + 1);
  }
  catch (const std::bad_alloc &) {
    error = "Failed to allocate two-sided GGX diffraction albedo buffers";
    return -1;
  }
  const int handle = int(two_sided_albedo_cache_count());
  const int4 descriptor = make_int4(int(two_sided_values_.size()),
                                    int(two_sided_integrals_.size()),
                                    int(two_sided_cross_.size()),
                                    canonical.mu_count |
                                        ((std::max(1, canonical.generalized_f0_count) - 1) << 8));
  const float4 domain = make_float4(380.0f, 780.0f,
                                     float(canonical.wavelength_count), float(canonical.phi_count));
  two_sided_values_.insert(two_sided_values_.end(), table.deficits.begin(), table.deficits.end());
  two_sided_integrals_.insert(
      two_sided_integrals_.end(), table.integrals.begin(), table.integrals.end());
  two_sided_cross_.insert(
      two_sided_cross_.end(), table.cross_fractions.begin(), table.cross_fractions.end());
  two_sided_descriptors_.push_back(descriptor);
  two_sided_domains_.push_back(domain);
  two_sided_requests_.push_back(canonical);
  dirty_ = true;
  error.clear();
  return handle;
}

/* Include numerical and resource settings in identity: a cache accepted for
 * one request must not silently satisfy a stricter request. Progress is a
 * per-call cancellation callback, not part of material identity. */
static auto build_key(const DiffractionGratingCacheOptions &o)
{
  return std::tie(o.reference_backend_key,
                  o.bounds.lower,
                  o.bounds.upper,
                  o.half_orders,
                  o.modal_power_tolerance,
                  o.modal_complex_tolerance,
                  o.maximum_half_orders,
                  o.retained_half_orders,
                  o.cutoff_margin,
                  o.tolerance,
                  o.complex_tolerance,
                  o.response_adaptive_splits,
                  o.curvature_adaptive_splits,
                  o.allow_chart_cells,
                  o.allow_quadratic_cells,
                  o.use_tensor_cells,
                  o.mirror_symmetry,
                  o.validation_workers,
                  o.maximum_depth,
                  o.maximum_nodes,
                  o.maximum_matrix_bytes,
                  o.reference_cache_matrix_bytes);
}

int DiffractionManager::get_or_build(const DiffractionGratingProfile &profile,
                                     const DiffractionGratingCacheOptions &options,
                                     DiffractionGratingCacheStats &stats,
                                     std::string &error,
                                     const int maximum_channels)
{
  stats = {};
  if (options.cancelled && options.cancelled()) {
    error = "Diffraction cache request cancelled";
    return -1;
  }
  if (bool(options.reference_solver) != !options.reference_backend_key.empty()) {
    error = "Custom diffraction solver requires a nonempty backend identity";
    return -1;
  }
  if (maximum_channels < 0) {
    error = "Invalid diffraction evaluator capacity";
    return -1;
  }
  const auto fits_evaluator = [&](const size_t channels) {
    if (maximum_channels && channels > size_t(maximum_channels)) {
      error = "Diffraction cache requires " + std::to_string(channels) +
              " polarization channels; evaluator supports " + std::to_string(maximum_channels);
      return false;
    }
    return true;
  };
  for (const MaterialCache &material : materials_) {
    if (material.profile == profile && build_key(material.options) == build_key(options)) {
      if (!fits_evaluator(material.maximum_channels))
        return -1;
      stats = material.stats;
      if (options.progress && !options.progress(stats)) {
        error = "Diffraction cache request cancelled";
        return -1;
      }
      error.clear();
      return material.handle;
    }
  }
  DiffractionGratingCache cache;
  const bool built = options.use_tensor_cells ?
                         diffraction_grating_build_tensor_cache(profile, options, cache, stats, error) :
                         diffraction_grating_build_cache(profile, options, cache, stats, error);
  if (!built)
    return -1;
  if (options.cancelled && options.cancelled()) {
    error = "Diffraction cache request cancelled";
    return -1;
  }
  size_t channels = 0;
  for (const auto &cell : cache.cells)
    channels = std::max(channels, 2 * cell.ports.size());
  if (!fits_evaluator(channels))
    return -1;
  const int handle = add_cache(cache, error);
  if (handle < 0)
    return -1;
  MaterialCache material{profile, options, stats, handle, channels};
  material.options.progress = {};
  material.options.cancelled = {};
  material.options.reference_solver = {};
  materials_.push_back(std::move(material));
  return handle;
}

int DiffractionManager::add_cache(const DiffractionGratingCache &cache, std::string &error)
{
  const size_t limit = std::numeric_limits<int>::max();
  if (cache_count() >= limit / 2) {
    error = "Too many diffraction caches";
    return -1;
  }
  DiffractionGratingDeviceBuffers packed;
  if (!diffraction_grating_device_buffers(cache, packed, error))
    return -1;
  for (int axis = 0; axis < 3; axis++) {
    const float lo = cache.bounds.lower[axis], hi = cache.bounds.upper[axis];
    if (!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi)) {
      error = "Invalid diffraction cache domain";
      return -1;
    }
  }
  auto fits = [&](const auto &target, const auto &source) {
    return source.size() <= limit && target.size() <= limit - source.size();
  };
  if (!fits(buffers_.nodes, packed.nodes) || !fits(buffers_.layout, packed.layout) ||
      !fits(buffers_.bounds, packed.bounds) || !fits(buffers_.ports, packed.ports) ||
      !fits(buffers_.active, packed.active) || !fits(buffers_.matrices, packed.matrices))
  {
    error = "Combined diffraction buffers exceed int32 addressing";
    return -1;
  }
  const int handle = cache_count();
  const int4 tree = make_int4(buffers_.nodes.size(),
                              packed.nodes.size(),
                              buffers_.layout.size() / 2,
                              packed.layout.size() / 2);
  const int4 data = make_int4(buffers_.ports.size(),
                              buffers_.active.size(),
                              buffers_.matrices.size(),
                              cache.mirror_symmetry);
  auto append = [](auto &target, const auto &source) {
    target.insert(target.end(), source.begin(), source.end());
  };
  append(buffers_.nodes, packed.nodes);
  append(buffers_.layout, packed.layout);
  append(buffers_.bounds, packed.bounds);
  append(buffers_.ports, packed.ports);
  append(buffers_.active, packed.active);
  append(buffers_.matrices, packed.matrices);
  descriptors_.push_back(tree);
  descriptors_.push_back(data);
  domains_.push_back(
      make_float4(cache.bounds.lower[0], cache.bounds.lower[1], cache.bounds.lower[2], 0));
  domains_.push_back(
      make_float4(cache.bounds.upper[0], cache.bounds.upper[1], cache.bounds.upper[2], 0));
  dirty_ = true;
  error.clear();
  return handle;
}

bool DiffractionManager::set_caches(std::span<const DiffractionGratingCache> caches,
                                    std::string &error)
{
  DiffractionManager replacement;
  for (const auto &cache : caches)
    if (replacement.add_cache(cache, error) < 0)
      return false;
  buffers_ = std::move(replacement.buffers_);
  descriptors_ = std::move(replacement.descriptors_);
  domains_ = std::move(replacement.domains_);
  materials_.clear();
  dirty_ = true;
  error.clear();
  return true;
}
CCL_NAMESPACE_END
