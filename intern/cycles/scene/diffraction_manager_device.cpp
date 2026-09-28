/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/devicescene.h"
#include "scene/diffraction_manager.h"
#include <algorithm>
CCL_NAMESPACE_BEGIN
void DiffractionManager::device_update(DeviceScene *dscene)
{
  if (!dirty_)
    return;
  auto upload = [](auto &destination, const auto &source) {
    if (destination.device->have_error())
      return false;
    if (source.empty()) {
      destination.free();
      return !destination.device->have_error();
    }
    destination.resize(source.size());
    if (!destination.data()) {
      destination.device->set_error("Failed to allocate diffraction cache host buffer");
      return false;
    }
    std::copy(source.begin(), source.end(), destination.data());
    destination.tag_modified();
    destination.copy_to_device();
    return !destination.device->have_error();
  };
  /* Scene aborts its update on a device error. Keep the full host set dirty so
   * recovery retries every array, including ones uploaded before the failure. */
  if (!upload(dscene->diffraction_nodes, buffers_.nodes) ||
      !upload(dscene->diffraction_layout, buffers_.layout) ||
      !upload(dscene->diffraction_bounds, buffers_.bounds) ||
      !upload(dscene->diffraction_ports, buffers_.ports) ||
      !upload(dscene->diffraction_active, buffers_.active) ||
      !upload(dscene->diffraction_matrices, buffers_.matrices) ||
      !upload(dscene->diffraction_descriptors, descriptors_) ||
      !upload(dscene->diffraction_domains, domains_) ||
      !upload(dscene->diffraction_albedo_values, albedo_values_) ||
      !upload(dscene->diffraction_albedo_averages, albedo_averages_) ||
      !upload(dscene->diffraction_albedo_descriptors, albedo_descriptors_) ||
      !upload(dscene->diffraction_albedo_domains, albedo_domains_) ||
      !upload(dscene->diffraction_two_sided_values, two_sided_values_) ||
      !upload(dscene->diffraction_two_sided_integrals, two_sided_integrals_) ||
      !upload(dscene->diffraction_two_sided_cross, two_sided_cross_) ||
      !upload(dscene->diffraction_two_sided_descriptors, two_sided_descriptors_) ||
      !upload(dscene->diffraction_two_sided_domains, two_sided_domains_))
    return;
  dscene->data.tables.num_diffraction_caches = int(cache_count());
  dscene->data.tables.num_diffraction_albedo_caches = int(albedo_cache_count());
  dscene->data.tables.num_diffraction_two_sided_caches = int(two_sided_albedo_cache_count());
  dirty_ = false;
}
void DiffractionManager::device_free(DeviceScene *dscene)
{
  dscene->diffraction_nodes.free();
  dscene->diffraction_layout.free();
  dscene->diffraction_bounds.free();
  dscene->diffraction_ports.free();
  dscene->diffraction_active.free();
  dscene->diffraction_matrices.free();
  dscene->diffraction_descriptors.free();
  dscene->diffraction_domains.free();
  dscene->diffraction_albedo_values.free();
  dscene->diffraction_albedo_averages.free();
  dscene->diffraction_albedo_descriptors.free();
  dscene->diffraction_albedo_domains.free();
  dscene->diffraction_two_sided_values.free();
  dscene->diffraction_two_sided_integrals.free();
  dscene->diffraction_two_sided_cross.free();
  dscene->diffraction_two_sided_descriptors.free();
  dscene->diffraction_two_sided_domains.free();
  dscene->data.tables.num_diffraction_caches = 0;
  dscene->data.tables.num_diffraction_albedo_caches = 0;
  dscene->data.tables.num_diffraction_two_sided_caches = 0;
  dirty_ = !descriptors_.empty() || !albedo_descriptors_.empty() ||
           !two_sided_descriptors_.empty();
}
CCL_NAMESPACE_END
