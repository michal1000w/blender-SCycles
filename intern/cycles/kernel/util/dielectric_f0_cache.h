/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* F0 is the native d-line value, not F0(n(lambda)). Concentrate eight nodes
 * around the coated tint kinks and keep eight over the remaining range. The
 * bounded cluster keeps all sixteen nodes strictly ordered even when the
 * d-line IOR exceeds three (where 4*F0 would otherwise exceed one). */
ccl_device_inline float dielectric_f0_cache_cluster(const float ior_d)
{
  const float ratio = (ior_d - 1.0f) / (ior_d + 1.0f);
  return min(ratio * ratio, 0.2f);
}

ccl_device_inline float dielectric_f0_cache_node(const float ior_d,
                                                const int count,
                                                const int node)
{
  if (count < 2) return 0.0f;
  if (count != 16) return float(node) / float(count - 1);
  const float cluster = dielectric_f0_cache_cluster(ior_d);
  if (!(cluster > 0.0f)) return float(node) / float(count - 1);
  const float factors[8] = {0.0f, 0.25f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f};
  if (node < 8) return factors[node] * cluster;
  const float lower = 4.0f * cluster;
  return lower + (1.0f - lower) * (float(node - 7) * 0.125f);
}

/* Coordinate is in node-index units, allowing the common lookup to choose
 * adjacent nodes without an additional per-material knot buffer. */
ccl_device_inline float dielectric_f0_cache_coordinate(const float ior_d,
                                                      const int count,
                                                      const float f0)
{
  const float value = saturatef(f0);
  if (count < 2) return 0.0f;
  if (count != 16 || !(dielectric_f0_cache_cluster(ior_d) > 0.0f))
    return value * float(count - 1);
  for (int node = 0; node < 15; ++node) {
    const float a = dielectric_f0_cache_node(ior_d, count, node);
    const float b = dielectric_f0_cache_node(ior_d, count, node + 1);
    if (value <= b) return float(node) + saturatef((value - a) / (b - a));
  }
  return 15.0f;
}

CCL_NAMESPACE_END
