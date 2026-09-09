/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/types_guiding.h"
#include "util/atomic.h"
#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Append-only within a drained group. Main states and their shadow branches share immutable
 * ancestors. The count deliberately exceeds capacity on overflow so the host rejects the render.
 */
struct GuidingHistory {
  ccl_global GuidingHistoryRecord *records;
  ccl_global uint *count;
  uint capacity;

  ccl_device_inline_method uint append(const uint parent,
                                       const uint field_index,
                                       const uint direction,
                                       const PackedSpectrum inverse_weight,
                                       const packed_float3 position = zero_float3(),
                                       const float distance_scale = 0.0f) const
  {
    const uint index = atomic_fetch_and_add_uint32(count, 1);
    if (index >= capacity) {
      return ~0u;
    }
    ccl_global GuidingHistoryRecord &record = records[index];
    record.field_index = field_index;
    record.direction = direction;
    record.inverse_weight = inverse_weight;
    record.parent = parent;
    record.position = position;
    record.radiance = 0.0f;
    record.source_weight = 0.0f;
    record.inverse_distance_weight = 0.0f;
    record.distance_weight = 0.0f;
    record.distance_scale = distance_scale;
    return index;
  }

  /* A completed direct/adjoint observation is not part of camera ancestry.
   * It shares the bounded allocation and lifetime, but has no parent or head. */
  ccl_device_inline_method void append_observation(const uint field_index,
                                                   const uint direction,
                                                   const packed_float3 position,
                                                   const float value,
                                                   const float distance = FLT_MAX) const
  {
    if (!(value > 0.0f) || !isfinite_safe(value)) {
      return;
    }
    const uint index = append(~0u, field_index, direction, zero_spectrum(), position);
    if (index != ~0u) {
      records[index].radiance = value;
      accumulate_source(index, value, distance);
    }
  }

  ccl_device_inline_method void accumulate_source(const uint index,
                                                  const float weight,
                                                  const float distance) const
  {
    if (weight > 0.0f && isfinite_safe(weight) && distance > 0.0f && distance < FLT_MAX &&
        isfinite_safe(distance) && isfinite_safe(weight / distance) &&
        isfinite_safe(weight * distance))
    {
      atomic_add_and_fetch_float(&records[index].source_weight, weight);
      atomic_add_and_fetch_float(&records[index].inverse_distance_weight, weight / distance);
      atomic_add_and_fetch_float(&records[index].distance_weight, weight * distance);
    }
  }

  ccl_device_inline_method void accumulate(const uint index, const float value) const
  {
    if (value > 0.0f && isfinite_safe(value)) {
      atomic_add_and_fetch_float(&records[index].radiance, value);
    }
  }
};

CCL_NAMESPACE_END
