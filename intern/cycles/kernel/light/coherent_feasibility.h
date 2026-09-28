/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/light/coherent_geometry.h"

CCL_NAMESPACE_BEGIN

/* Conservative signed-distance interval of a finite planar rectangle (or a
 * point) against another interface plane. The center alone cannot establish
 * sidedness when a tilted rectangle crosses the plane. */
ccl_device_inline float2 coherent_feasibility_interval(const float3 center,
                                                        const float3 tangent_u,
                                                        const float3 tangent_v,
                                                        const float half_u,
                                                        const float half_v,
                                                        const float3 plane_center,
                                                        const float3 plane_normal)
{
  const float distance = dot(center - plane_center, plane_normal);
  const float radius = half_u * fabsf(dot(tangent_u, plane_normal)) +
                       half_v * fabsf(dot(tangent_v, plane_normal));
  return make_float2(distance - radius, distance + radius);
}

ccl_device_inline bool coherent_geometry_candidate_side_feasible(
    const float3 source,
    const float3 receiver,
    const ccl_private CoherentGeometryInterface
        patches[COHERENT_GEOMETRY_MAX_INTERFACES],
    const int count)
{
  for (int i = 0; i < count; i++) {
    const ccl_private CoherentGeometryInterface &patch = patches[i];
    const float3 normal = cross(patch.tangent_u, patch.tangent_v);
    if (!(len_squared(normal) > 0.0f)) {
      continue; /* The exact solver rejects invalid geometry. */
    }
    const float3 unit_normal = normalize(normal);
    const float3 before_center = i == 0 ? source : patches[i - 1].center;
    const float3 after_center = i + 1 == count ? receiver : patches[i + 1].center;
    const float2 before = i == 0 ?
                              coherent_feasibility_interval(source,
                                                            zero_float3(),
                                                            zero_float3(),
                                                            0.0f,
                                                            0.0f,
                                                            patch.center,
                                                            unit_normal) :
                              coherent_feasibility_interval(patches[i - 1].center,
                                                            patches[i - 1].tangent_u,
                                                            patches[i - 1].tangent_v,
                                                            patches[i - 1].half_u,
                                                            patches[i - 1].half_v,
                                                            patch.center,
                                                            unit_normal);
    const float2 after = i + 1 == count ?
                             coherent_feasibility_interval(receiver,
                                                           zero_float3(),
                                                           zero_float3(),
                                                           0.0f,
                                                           0.0f,
                                                           patch.center,
                                                           unit_normal) :
                             coherent_feasibility_interval(patches[i + 1].center,
                                                           patches[i + 1].tangent_u,
                                                           patches[i + 1].tangent_v,
                                                           patches[i + 1].half_u,
                                                           patches[i + 1].half_v,
                                                           patch.center,
                                                           unit_normal);
    /* This margin also covers cancellation in float world-space centers. It
     * may retain an impossible candidate, but must not prune a valid one. */
    const float before_extent = i == 0 ? 0.0f :
                                         patches[i - 1].half_u + patches[i - 1].half_v;
    const float after_extent = i + 1 == count ? 0.0f :
                                                 patches[i + 1].half_u + patches[i + 1].half_v;
    const float coordinate_scale = len(before_center) + len(patch.center) +
                                   len(after_center) + before_extent + after_extent;
    const float epsilon = max(1.0e-8f, 16.0f * FLT_EPSILON * coordinate_scale);
    bool before_positive = before.y >= -epsilon;
    bool before_negative = before.x <= epsilon;
    const bool after_positive = after.y >= -epsilon;
    const bool after_negative = after.x <= epsilon;
    if (patch.expected_incident_side > 0) {
      before_negative = false;
    }
    else if (patch.expected_incident_side < 0) {
      before_positive = false;
    }
    const bool feasible = patch.event == COHERENT_GEOMETRY_REFLECT ?
                              (before_positive && after_positive) ||
                                  (before_negative && after_negative) :
                              (before_positive && after_negative) ||
                                  (before_negative && after_positive);
    if (!feasible) {
      return false;
    }
  }
  return true;
}

CCL_NAMESPACE_END
