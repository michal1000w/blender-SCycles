/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "util/math.h"
CCL_NAMESPACE_BEGIN

/* Centered scalar lamellar profiles have x/y mirror symmetry. Fold both Bloch
 * and conical coordinates to the negative quadrant. An x reflection negates
 * port orders; either single reflection negates cross-polarized amplitudes.
 * These operations are their own inverse. */
struct DiffractionCacheSymmetry {
  float3 query;
  bool reverse_orders;
  float cross_polarization_sign;
};

ccl_device_inline DiffractionCacheSymmetry diffraction_cache_symmetry(const float3 query)
{
  DiffractionCacheSymmetry result;
  result.query = make_float3(-fabsf(query.x), -fabsf(query.y), query.z);
  result.reverse_orders = query.x > 0.0f;
  result.cross_polarization_sign = (result.reverse_orders != (query.y > 0.0f)) ? -1.0f : 1.0f;
  return result;
}

/* Return the prepared-cell index, or -1 for out-of-domain/invalid queries and
 * regions with no physical ports. Split-plane ownership belongs to the right
 * child. No nearest-cell extrapolation is performed. */
ccl_device_inline int diffraction_cache_lookup(ccl_global const int4 *nodes,
                                               const int node_count,
                                               const float3 lower,
                                               const float3 upper,
                                               const float3 original_query,
                                               const bool mirror_symmetry = false)
{
  const float3 query = mirror_symmetry ? diffraction_cache_symmetry(original_query).query :
                                         original_query;
  if (!isfinite_safe(query.x) || !isfinite_safe(query.y) || !isfinite_safe(query.z) ||
      query.x < lower.x || query.x > upper.x || query.y < lower.y || query.y > upper.y ||
      query.z < lower.z || query.z > upper.z)
    return -1;
  int index = 0;
  for (int step = 0; step < node_count; step++) {
    if (index < 0 || index >= node_count)
      return -1;
    const int4 node = nodes[index];
    if (node.x == -1)
      return node.y;
    if (node.x < 0 || node.x > 2)
      return -1;
    const float position = node.x == 0 ? query.x : node.x == 1 ? query.y : query.z;
    index = position < __int_as_float(node.w) ? node.y : node.z;
  }
  return -1;
}
CCL_NAMESPACE_END
