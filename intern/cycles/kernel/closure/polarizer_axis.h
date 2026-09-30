/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Object-space transmission axis of a linear polarizer film, unnormalized
 * (callers normalize after transforming it to world space).
 *
 * A physical polarizer's transmission axis lies in the plane of the film. The
 * reference direction for the angle is picked from the dominant component of
 * the object-space geometric normal:
 *
 *   Z-facing film: angle 0 along X, 90 degrees along Y (unchanged convention).
 *   X-facing film: angle 0 along Y, 90 degrees along Z.
 *   Y-facing film: angle 0 along X, 90 degrees along Z.
 *
 * so on upright side films angle 0 is horizontal and 90 degrees is vertical.
 * The normal's sign is ignored, so the front and back faces of a closed slab
 * share one axis and P*P = P holds. The axis is then projected into the true
 * tangent plane, which keeps it inside curved or tilted films. Using an axis
 * that is not in the film plane made the ray-transverse projection depend on
 * the ray direction, which produced radial pinwheel artifacts. */
ccl_device_inline float3 polarizer_object_axis(const float angle, const float3 object_normal)
{
  const float c = cosf(angle), s = sinf(angle);
  const float3 n = fabs(object_normal);
  float3 axis;
  if (n.z >= n.x && n.z >= n.y) {
    axis = make_float3(c, s, 0.0f);
  }
  else if (n.x >= n.y) {
    axis = make_float3(0.0f, c, s);
  }
  else {
    axis = make_float3(c, 0.0f, s);
  }
  /* The unprojected axis is perpendicular to the dominant axis, so it is at
   * least 35.3 degrees off the normal and keeps >= 57% of its length. The
   * check only guards against degenerate or NaN normals. */
  const float nn = len_squared(object_normal);
  if (nn > 1e-20f) {
    const float3 projected = axis - object_normal * (dot(axis, object_normal) / nn);
    if (len_squared(projected) > 1e-12f) {
      axis = projected;
    }
  }
  return axis;
}

CCL_NAMESPACE_END
