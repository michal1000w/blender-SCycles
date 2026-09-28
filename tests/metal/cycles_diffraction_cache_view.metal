/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/util/diffraction_cache_view.h"
kernel void diffraction_cache_view(device const int4 *descriptors [[buffer(0)]],
                                   device const float4 *domains [[buffer(1)]],
                                   device const int4 *nodes [[buffer(2)]],
                                   device const int4 *layout [[buffer(3)]],
                                   device const float4 *bounds [[buffer(4)]],
                                   device const float4 *queries [[buffer(5)]],
                                   device float4 *output [[buffer(6)]],
                                   uint index [[thread_position_in_grid]])
{
  const float4 query = queries[index];
  DiffractionCacheCellView view;
  const bool valid = diffraction_cache_cell_view(
      descriptors, domains, nodes, layout, bounds, 2, int(query.w), query.xyz, &view);
  output[3 * index] =
      valid ? make_float4(view.matrix_offset, view.port_offset, view.active_offset, view.degree) :
              make_float4(-1, -1, -1, -1);
  output[3 * index + 1] = valid ? make_float4(view.coordinate.x,
                                              view.coordinate.y,
                                              view.coordinate.z,
                                              view.reverse_orders) :
                                  zero_float4();
  output[3 * index + 2] =
      valid ? make_float4(view.rotation.x, view.rotation.y, view.cross_polarization_sign, 1) :
              zero_float4();
}
