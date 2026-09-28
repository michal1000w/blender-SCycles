/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/util/diffraction_limits.h"
#include "kernel/globals.h"
#include "kernel/util/diffraction_scene_data.h"
CCL_NAMESPACE_BEGIN
ccl_device_inline DiffractionSceneData diffraction_scene_data(KernelGlobals kg)
{
  DiffractionSceneData data;
  data.nodes = kernel_data_array(diffraction_nodes);
  data.layout = kernel_data_array(diffraction_layout);
  data.bounds = kernel_data_array(diffraction_bounds);
  data.ports = kernel_data_array(diffraction_ports);
  data.active = kernel_data_array(diffraction_active);
  data.matrices = kernel_data_array(diffraction_matrices);
  data.descriptors = kernel_data_array(diffraction_descriptors);
  data.domains = kernel_data_array(diffraction_domains);
  data.cache_count = kernel_data.tables.num_diffraction_caches;
  return data;
}
ccl_device_inline bool diffraction_scene_cell_view(KernelGlobals kg,
                                                   const int handle,
                                                   const float3 query,
                                                   ccl_private DiffractionCacheCellView *view)
{
  const DiffractionSceneData data = diffraction_scene_data(kg);
  return diffraction_data_cell_view(&data, handle, query, view);
}
template<int MaxChannels = DIFFRACTION_MAX_CHANNELS, typename BoundaryPointer>
ccl_device_inline bool diffraction_scene_cell_jones(
    KernelGlobals kg,
    ccl_private const DiffractionCacheCellView *view,
    const BoundaryPointer boundary,
    const int incoming,
    ccl_private float2 *jones)
{
  const DiffractionSceneData data = diffraction_scene_data(kg);
  return diffraction_data_cell_jones<MaxChannels>(&data, view, boundary, incoming, jones);
}
template<int MaxChannels = DIFFRACTION_MAX_CHANNELS, typename BoundaryPointer>
ccl_device_inline bool diffraction_scene_cell_power(
    KernelGlobals kg,
    ccl_private const DiffractionCacheCellView *view,
    const BoundaryPointer boundary,
    const int incoming,
    ccl_private float *powers)
{
  const DiffractionSceneData data = diffraction_scene_data(kg);
  return diffraction_data_cell_power<MaxChannels>(&data, view, boundary, incoming, powers);
}
template<int MaxChannels = DIFFRACTION_MAX_CHANNELS>
ccl_device_inline bool diffraction_scene_sample(KernelGlobals kg,
                                                const int handle,
                                                const float3 incident,
                                                const bool incoming_substrate,
                                                const float upper_index,
                                                const float lower_index,
                                                const float wavelength,
                                                const float pitch,
                                                const float random,
                                                ccl_private DiffractionSceneSample *result)
{
  const DiffractionSceneData data = diffraction_scene_data(kg);
  return diffraction_data_sample<MaxChannels>(&data,
                                              handle,
                                              incident,
                                              incoming_substrate,
                                              upper_index,
                                              lower_index,
                                              wavelength,
                                              pitch,
                                              random,
                                              result);
}
template<int MaxChannels = DIFFRACTION_MAX_CHANNELS>
ccl_device_inline bool diffraction_scene_order_probability(KernelGlobals kg,
                                                           const int handle,
                                                           const float3 incident,
                                                           const bool incoming_substrate,
                                                           const float upper_index,
                                                           const float lower_index,
                                                           const float wavelength,
                                                           const float pitch,
                                                           const int relative_order,
                                                           const bool outgoing_substrate,
                                                           ccl_private float *power,
                                                           ccl_private float *probability)
{
  const DiffractionSceneData data = diffraction_scene_data(kg);
  return diffraction_data_order_probability<MaxChannels>(&data,
                                                         handle,
                                                         incident,
                                                         incoming_substrate,
                                                         upper_index,
                                                         lower_index,
                                                         wavelength,
                                                         pitch,
                                                         relative_order,
                                                         outgoing_substrate,
                                                         power,
                                                         probability);
}
template<int MaxChannels = DIFFRACTION_MAX_CHANNELS>
ccl_device_inline bool diffraction_scene_reverse_probability(
    KernelGlobals kg,
    const int handle,
    ccl_private const DiffractionSceneSample *forward,
    const bool incoming_substrate,
    const float upper_index,
    const float lower_index,
    const float wavelength,
    const float pitch,
    ccl_private float *power,
    ccl_private float *probability)
{
  const DiffractionSceneData data = diffraction_scene_data(kg);
  return diffraction_data_reverse_probability<MaxChannels>(&data,
                                                           handle,
                                                           forward,
                                                           incoming_substrate,
                                                           upper_index,
                                                           lower_index,
                                                           wavelength,
                                                           pitch,
                                                           power,
                                                           probability);
}
template<int MaxChannels = DIFFRACTION_MAX_CHANNELS>
ccl_device_inline bool diffraction_scene_direction_probability(
    KernelGlobals kg, const int handle, const float3 incident, const bool incoming_substrate,
    const float upper_index, const float lower_index, const float wavelength, const float pitch,
    const float3 outgoing, ccl_private float *power, ccl_private float *probability)
{
  const DiffractionSceneData data = diffraction_scene_data(kg);
  return diffraction_data_direction_probability<MaxChannels>(&data, handle, incident,
      incoming_substrate, upper_index, lower_index, wavelength, pitch, outgoing, power, probability);
}
CCL_NAMESPACE_END
