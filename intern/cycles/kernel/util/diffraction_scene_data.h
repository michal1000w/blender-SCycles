/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/util/diffraction_limits.h"
#include "kernel/util/diffraction_cache_view.h"
#include "kernel/util/diffraction_coordinates.h"
#include "kernel/util/diffraction_direction.h"
#include "kernel/util/diffraction_evaluate.h"
#include "kernel/util/diffraction_sample.h"
CCL_NAMESPACE_BEGIN
struct DiffractionSceneData {
  ccl_global const int4 *nodes;
  ccl_global const int4 *layout;
  ccl_global const float4 *bounds;
  ccl_global const int2 *ports;
  ccl_global const int *active;
  ccl_global const float2 *matrices;
  ccl_global const int4 *descriptors;
  ccl_global const float4 *domains;
  int cache_count;
};

/* Access the scene-owned cache through the normal kernel memory abstraction.
 * This is shared by CPU and GPU shading; a handle is valid only for the current
 * registered cache set. The manager publishes count only after upload succeeds. */
ccl_device_inline bool diffraction_data_cell_view(ccl_private const DiffractionSceneData *data,
                                                  const int handle,
                                                  const float3 query,
                                                  ccl_private DiffractionCacheCellView *view)
{
  return diffraction_cache_cell_view(data->descriptors,
                                     data->domains,
                                     data->nodes,
                                     data->layout,
                                     data->bounds,
                                     data->cache_count,
                                     handle,
                                     query,
                                     view);
}
/* The view and boundary basis must refer to the same folded query. Outputs
 * remain in that basis. Intensity-interpolated cells cannot supply coherent
 * Jones data and are explicitly rejected by the complex entry point. */
template<int MaxChannels = DIFFRACTION_MAX_CHANNELS, typename BoundaryPointer>
ccl_device_inline bool diffraction_data_cell_jones(
    ccl_private const DiffractionSceneData *data,
    ccl_private const DiffractionCacheCellView *view,
    const BoundaryPointer boundary,
    const int incoming,
    ccl_private float2 *jones)
{
  if (view->degree == 0 || view->ports <= 0 || view->ports > MaxChannels / 2)
    return false;
  if (view->degree == 3)
    return diffraction_cache_tensor_match<MaxChannels>(data->matrices + view->matrix_offset,
                                                       boundary,
                                                       2 * view->ports,
                                                       view->tensor_rank,
                                                       view->tensor_floats,
                                                       incoming,
                                                       view->coordinate,
                                                       jones);
  return diffraction_cache_chart_match<MaxChannels>(data->matrices + view->matrix_offset,
                                                    boundary,
                                                    2 * view->ports,
                                                    view->degree,
                                                    view->rotation,
                                                    incoming,
                                                    view->coordinate,
                                                    jones);
}

/* Unpolarized incoherent power column. No passivity clamp, transport-mode
 * index scaling or sampling normalization is applied here. */
template<int MaxChannels = DIFFRACTION_MAX_CHANNELS, typename BoundaryPointer>
ccl_device_inline bool diffraction_data_cell_power(
    ccl_private const DiffractionSceneData *data,
    ccl_private const DiffractionCacheCellView *view,
    const BoundaryPointer boundary,
    const int incoming,
    ccl_private float *powers)
{
  static_assert(MaxChannels > 0 && MaxChannels % 2 == 0);
  if (view->ports <= 0 || view->ports > MaxChannels / 2)
    return false;
  if (view->degree == 0)
    return diffraction_cache_intensity_match<MaxChannels>(data->matrices + view->matrix_offset,
                                                          boundary,
                                                          data->active + view->active_offset,
                                                          2 * view->active_ports,
                                                          2 * view->ports,
                                                          incoming,
                                                          view->coordinate,
                                                          powers);
  float2 jones[2 * MaxChannels];
  if (!diffraction_data_cell_jones<MaxChannels>(data, view, boundary, incoming, jones))
    return false;
  for (int port = 0; port < view->ports; port++) {
    float power = 0;
    for (int element = 0; element < 4; element++)
      power += 0.5f * len_squared(jones[4 * port + element]);
    powers[port] = power;
  }
  return true;
}
struct DiffractionSceneSample {
  float3 direction;
  float probability, throughput;
  /* Outgoing/incident index, matching Cycles closure eta; one for reflection. */
  float eta;
  int relative_order;
  bool transmission;
};

/* Incoherent, unpolarized scattering in the fixed cache frame: x crosses
 * grooves, y follows them, positive z points into the upper medium. The
 * incident vector carries propagation momentum (not the direction to the
 * previous vertex). Output throughput is flux-normalized; the BSDF must still
 * apply its transport-mode conversion and rough-surface Jacobian. Exterior
 * indices and pitch must match the registered material. */
template<int MaxChannels = DIFFRACTION_MAX_CHANNELS>
/* This substantial dispatch is shared by sampling and forward/reverse queries.
 * Let the compiler outline it instead of forcing a full matrix-solver expansion
 * at every call site. */
#ifdef __KERNEL_METAL_VISIBLE_SHADING__
ccl_device bool diffraction_data_power_column_impl(
#else
ccl_device bool diffraction_data_power_column(
#endif
    ccl_private const DiffractionSceneData *data,
    const int handle,
    const float3 incident,
    const bool incoming_substrate,
    const float upper_index,
    const float lower_index,
    const float wavelength,
    const float pitch,
    ccl_private DiffractionCacheCellView *output_view,
    ccl_private int *output_incoming_order,
    ccl_private float *powers)
{
  static_assert(MaxChannels > 0 && MaxChannels % 2 == 0);
  if (!(upper_index > 0) || !(lower_index > 0) || !isfinite_safe(upper_index) ||
      !isfinite_safe(lower_index))
    return false;
  const float ni = incoming_substrate ? lower_index : upper_index;
  DiffractionCacheCoordinates coordinates;
  DiffractionCacheCellView view;
  if (!diffraction_cache_coordinates(incident, ni, wavelength, pitch, false, &coordinates) ||
      !diffraction_data_cell_view(data, handle, coordinates.query, &view) || view.ports <= 0 ||
      view.ports > MaxChannels / 2)
    return false;
  const int incoming_order = view.reverse_orders ? -coordinates.incoming_order :
                                                   coordinates.incoming_order;
  const bool reverse_y = (view.cross_polarization_sign < 0) != view.reverse_orders;
  const float3 folded = make_float3(view.reverse_orders ? -incident.x : incident.x,
                                    reverse_y ? -incident.y : incident.y,
                                    incident.z);
  float2 boundary[2 * MaxChannels];
  float4 exterior[MaxChannels / 2];
  int incoming = -1;
  for (int port = 0; port < view.ports; port++) {
    const int2 channel = data->ports[view.port_offset + port];
    if (channel.x == incoming_order && bool(channel.y) == incoming_substrate)
      incoming = port;
    /* Guard subtraction before forming the relative order. */
    if ((incoming_order > 0 && channel.x < (-2147483647 - 1) + incoming_order) ||
        (incoming_order < 0 && channel.x > 2147483647 + incoming_order))
      return false;
    DiffractionGratingBoundary b;
    if (!diffraction_grating_boundary(folded,
                                      ni,
                                      channel.y ? lower_index : upper_index,
                                      wavelength,
                                      pitch,
                                      channel.x - incoming_order,
                                      &b))
      return false;
    if (view.degree == 3) {
      exterior[port] = make_float4(b.tangent_direction.x, b.tangent_direction.y,
                                  b.q2, channel.y ? lower_index : upper_index);
      continue;
    }
    bool reference = false;
    for (int a = 0; a < view.active_ports; a++)
      reference |= data->active[view.active_offset + a] == port;
    boundary[4 * port] = reference ? b.r_te : zero_float2();
    boundary[4 * port + 1] = reference ? b.r_tm : zero_float2();
    boundary[4 * port + 2] = reference ? b.transmission :
                                         (b.q2 > 0 ? make_float2(1, 1) : zero_float2());
    boundary[4 * port + 3] = b.tangent_direction;
  }
  if (incoming < 0)
    return false;
  if (view.degree == 3) {
    float2 jones[2 * MaxChannels];
    if (!diffraction_cache_tensor_admittance_match<MaxChannels>(
            data->matrices + view.matrix_offset, exterior, 2 * view.ports,
            view.tensor_rank, view.tensor_floats, incoming, view.coordinate, jones))
      return false;
    for (int port = 0; port < view.ports; port++) {
      float power = 0;
      for (int element = 0; element < 4; element++)
        power += 0.5f * len_squared(jones[4 * port + element]);
      powers[port] = power;
    }
  }
  else if (!diffraction_data_cell_power<MaxChannels>(data, &view, boundary, incoming, powers))
    return false;
  *output_view = view;
  *output_incoming_order = incoming_order;
  return true;
}
#ifdef __KERNEL_METAL_VISIBLE_SHADING__
/* The matrix solver is compiled once as a Metal visible function, see `kernel.metal`. */
template<int MaxChannels = DIFFRACTION_MAX_CHANNELS>
ccl_device_inline bool diffraction_data_power_column(
    ccl_private const DiffractionSceneData *data,
    const int handle,
    const float3 incident,
    const bool incoming_substrate,
    const float upper_index,
    const float lower_index,
    const float wavelength,
    const float pitch,
    ccl_private DiffractionCacheCellView *output_view,
    ccl_private int *output_incoming_order,
    ccl_private float *powers)
{
  if constexpr (MaxChannels == DIFFRACTION_MAX_CHANNELS) {
    return metal_ancillaries->vft_diffraction[0](&launch_params_metal,
                                                 metal_ancillaries,
                                                 data,
                                                 handle,
                                                 incident,
                                                 incoming_substrate,
                                                 upper_index,
                                                 lower_index,
                                                 wavelength,
                                                 pitch,
                                                 output_view,
                                                 output_incoming_order,
                                                 powers);
  }
  else {
    return diffraction_data_power_column_impl<MaxChannels>(data,
                                                           handle,
                                                           incident,
                                                           incoming_substrate,
                                                           upper_index,
                                                           lower_index,
                                                           wavelength,
                                                           pitch,
                                                           output_view,
                                                           output_incoming_order,
                                                           powers);
  }
}
#endif

template<int MaxChannels = DIFFRACTION_MAX_CHANNELS>
ccl_device_inline bool diffraction_data_sample(ccl_private const DiffractionSceneData *data,
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
  const float ni = incoming_substrate ? lower_index : upper_index;
  DiffractionCacheCellView view;
  int incoming_order;
  float powers[MaxChannels / 2];
  DiffractionOrderSample selected;
  if (!diffraction_data_power_column<MaxChannels>(data,
                                                  handle,
                                                  incident,
                                                  incoming_substrate,
                                                  upper_index,
                                                  lower_index,
                                                  wavelength,
                                                  pitch,
                                                  &view,
                                                  &incoming_order,
                                                  powers) ||
      !diffraction_sample_order(powers, view.ports, random, &selected))
    return false;
  const int2 outgoing = data->ports[view.port_offset + selected.port];
  int relative = outgoing.x - incoming_order; /* Checked in the loop above. */
  if (view.reverse_orders) {
    if (relative == (-2147483647 - 1))
      return false;
    relative = -relative;
  }
  float3 direction;
  if (!diffraction_grating_direction(incident,
                                     ni,
                                     outgoing.y ? lower_index : upper_index,
                                     wavelength,
                                     pitch,
                                     relative,
                                     bool(outgoing.y),
                                     &direction))
    return false;
  const float eta = bool(outgoing.y) != incoming_substrate ?
                        (outgoing.y ? lower_index : upper_index) / ni :
                        1.0f;
  if (!(eta > 0) || !isfinite_safe(eta))
    return false;
  result->eta = eta;
  result->direction = direction;
  result->relative_order = relative;
  result->transmission = bool(outgoing.y) != incoming_substrate;
  result->probability = selected.probability;
  result->throughput = selected.throughput;
  return true;
}
/* Discrete port probability, not a density per solid angle. Missing output
 * orders have zero mass; missing incident/cache data remain errors. */
template<int MaxChannels = DIFFRACTION_MAX_CHANNELS>
ccl_device_inline bool diffraction_data_order_probability(
    ccl_private const DiffractionSceneData *data,
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
  DiffractionCacheCellView view;
  int incoming_order;
  float powers[MaxChannels / 2];
  if (!diffraction_data_power_column<MaxChannels>(data,
                                                  handle,
                                                  incident,
                                                  incoming_substrate,
                                                  upper_index,
                                                  lower_index,
                                                  wavelength,
                                                  pitch,
                                                  &view,
                                                  &incoming_order,
                                                  powers))
    return false;
  float total = 0, selected = 0;
  for (int p = 0; p < view.ports; p++) {
    if (powers[p] < 0 || !isfinite_safe(powers[p]))
      return false;
    total += powers[p];
    const int2 channel = data->ports[view.port_offset + p];
    int relative = channel.x - incoming_order; /* Validated by power-column preparation. */
    if (view.reverse_orders) {
      if (relative == (-2147483647 - 1))
        continue;
      relative = -relative;
    }
    if (relative == relative_order && bool(channel.y) == outgoing_substrate)
      selected = powers[p];
  }
  if (!isfinite_safe(total))
    return false;
  *power = selected;
  *probability = total > 0 ? selected / total : 0;
  return true;
}
/* Recover a discrete mass for an already sampled direction. This is not a
 * continuous BSDF evaluator: the tolerance only accommodates round-trip float
 * frame conversion. Ambiguous matching orders are rejected, never merged. */
template<int MaxChannels = DIFFRACTION_MAX_CHANNELS>
ccl_device_inline bool diffraction_data_direction_probability(
    ccl_private const DiffractionSceneData *data,
    const int handle,
    const float3 incident,
    const bool incoming_substrate,
    const float upper_index,
    const float lower_index,
    const float wavelength,
    const float pitch,
    const float3 outgoing,
    ccl_private float *power,
    ccl_private float *probability)
{
  *power = *probability = 0;
  if (!isfinite_safe(outgoing) || outgoing.z == 0 ||
      fabsf(len_squared(outgoing) - 1) > 1e-4f)
    return false;
  DiffractionCacheCellView view;
  int incoming_order;
  float powers[MaxChannels / 2];
  if (!diffraction_data_power_column<MaxChannels>(data, handle, incident, incoming_substrate,
      upper_index, lower_index, wavelength, pitch, &view, &incoming_order, powers))
    return false;
  float total = 0, selected = 0;
  int matches = 0;
  for (int p = 0; p < view.ports; p++) {
    if (!(powers[p] >= 0) || !isfinite_safe(powers[p]))
      return false;
    total += powers[p];
    const int2 channel = data->ports[view.port_offset + p];
    /* Reflection and transmission are distinct discrete outcomes even when
     * both directions lie within the angular tolerance near grazing. Keep
     * their powers in the total normalization, but only match the right side. */
    if (bool(channel.y) != (outgoing.z < 0))
      continue;
    int relative = channel.x - incoming_order;
    if (view.reverse_orders) {
      if (relative == (-2147483647 - 1)) continue;
      relative = -relative;
    }
    float3 direction;
    if (diffraction_grating_direction(incident,
          incoming_substrate ? lower_index : upper_index,
          channel.y ? lower_index : upper_index, wavelength, pitch, relative,
          bool(channel.y), &direction) && len_squared(direction - outgoing) <= 16e-12f)
    {
      matches++;
      selected = powers[p];
    }
  }
  if (matches > 1 || !isfinite_safe(total))
    return false;
  *power = selected;
  *probability = total > 0 ? selected / total : 0;
  return true;
}

/* Reversing both propagation vectors preserves the signed reciprocal-lattice
 * kick: (-k_in)-(-k_out)=k_out-k_in. Re-evaluate the reverse incident column;
 * absorption means its normalization need not equal the forward one. */
template<int MaxChannels = DIFFRACTION_MAX_CHANNELS>
ccl_device_inline bool diffraction_data_reverse_probability(
    ccl_private const DiffractionSceneData *data,
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
  return diffraction_data_order_probability<MaxChannels>(data,
                                                         handle,
                                                         -forward->direction,
                                                         incoming_substrate !=
                                                             forward->transmission,
                                                         upper_index,
                                                         lower_index,
                                                         wavelength,
                                                         pitch,
                                                         forward->relative_order,
                                                         incoming_substrate,
                                                         power,
                                                         probability);
}
CCL_NAMESPACE_END
