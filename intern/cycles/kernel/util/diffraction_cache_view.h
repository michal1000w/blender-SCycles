/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/util/diffraction_cache.h"
CCL_NAMESPACE_BEGIN

struct DiffractionCacheCellView {
  int matrix_offset, port_offset, active_offset;
  int ports, active_ports, degree;
  int tensor_rank, tensor_floats;
  float3 coordinate;
  float2 rotation;
  bool reverse_orders;
  float cross_polarization_sign;
};

/* Resolve the scene manager's cache-local offsets into global array offsets.
 * Arrays must come from validated DiffractionManager registration. Handle
 * range and absent/out-of-domain cells fail without nearest-cell fallback. */
ccl_device_inline bool diffraction_cache_cell_view(ccl_global const int4 *descriptors,
                                                   ccl_global const float4 *domains,
                                                   ccl_global const int4 *nodes,
                                                   ccl_global const int4 *layout,
                                                   ccl_global const float4 *bounds,
                                                   const int caches,
                                                   const int handle,
                                                   const float3 query,
                                                   ccl_private DiffractionCacheCellView *view)
{
  if (handle < 0 || handle >= caches)
    return false;
  const int4 tree = descriptors[2 * handle], data = descriptors[2 * handle + 1];
  const float4 lo = domains[2 * handle], hi = domains[2 * handle + 1];
  const bool mirror = data.w != 0;
  const int leaf = diffraction_cache_lookup(nodes + tree.x,
                                            tree.y,
                                            make_float3(lo.x, lo.y, lo.z),
                                            make_float3(hi.x, hi.y, hi.z),
                                            query,
                                            mirror);
  if (leaf < 0 || leaf >= tree.w)
    return false;
  const int cell = tree.z + leaf;
  const int4 offsets = layout[2 * cell], shape = layout[2 * cell + 1];
  const float4 a = bounds[2 * cell], b = bounds[2 * cell + 1];
  const DiffractionCacheSymmetry symmetry = diffraction_cache_symmetry(query);
  const float3 q = mirror ? symmetry.query : query;
  const float3 t = make_float3(
      (q.x - a.x) / (b.x - a.x), (q.y - a.y) / (b.y - a.y), (q.z - a.z) / (b.z - a.z));
  if (!isfinite_safe(t.x) || !isfinite_safe(t.y) || !isfinite_safe(t.z) || t.x < 0 || t.x > 1 ||
      t.y < 0 || t.y > 1 || t.z < 0 || t.z > 1)
    return false;
  view->matrix_offset = data.z + offsets.x;
  view->port_offset = data.x + offsets.y;
  view->active_offset = data.y + offsets.z;
  view->ports = shape.x;
  view->active_ports = shape.y;
  view->degree = offsets.w;
  view->tensor_rank = shape.z;
  view->tensor_floats = shape.w;
  view->coordinate = t;
  view->rotation = make_float2(a.w, b.w);
  view->reverse_orders = mirror && symmetry.reverse_orders;
  view->cross_polarization_sign = mirror ? symmetry.cross_polarization_sign : 1;
  return true;
}
CCL_NAMESPACE_END
