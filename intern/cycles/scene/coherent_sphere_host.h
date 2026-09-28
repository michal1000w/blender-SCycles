/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <array>
#include <cmath>

CCL_NAMESPACE_BEGIN

/* An unbaked native point under an arbitrary affine object transform is an
 * ellipsoid. Only exactly axis-permuted uniform transforms are represented
 * here; baked primitives already store their actual world sphere. */
inline bool coherent_sphere_transform_scale(
    const std::array<std::array<float, 3>, 3> &columns, float &scale)
{
  int used_axes = 0;
  scale = 1.0f;
  for (int column = 0; column < 3; ++column) {
    int nonzero = -1;
    float value = 0.0f;
    for (int row = 0; row < 3; ++row) {
      if (columns[column][row] == 0.0f) continue;
      if (nonzero != -1 || !std::isfinite(columns[column][row])) return false;
      nonzero = row;
      value = std::fabs(columns[column][row]);
    }
    if (nonzero == -1 || (used_axes & (1 << nonzero)) ||
        (column > 0 && value != scale))
    {
      return false;
    }
    used_axes |= 1 << nonzero;
    scale = value;
  }
  return true;
}

inline bool coherent_sphere_world_radius(const float local_radius,
                                         const float scale,
                                         const bool baked,
                                         float &world_radius)
{
  world_radius = baked ? local_radius : local_radius * scale;
  return std::isfinite(local_radius) && local_radius > 0.0f &&
         std::isfinite(world_radius) && world_radius > 0.0f &&
         (baked || double(world_radius) == double(local_radius) * double(scale));
}

inline bool coherent_sphere_source_outside(const std::array<double, 3> &source,
                                           const std::array<double, 3> &center,
                                           const float radius)
{
  double distance_squared = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double delta = source[axis] - center[axis];
    distance_squared += delta * delta;
  }
  return std::isfinite(distance_squared) &&
         distance_squared > double(radius) * double(radius);
}

CCL_NAMESPACE_END
