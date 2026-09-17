/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/types_guiding.h"
#include "util/atomic.h"
#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Append-only within a drained group. Main states and their shadow branches share immutable
 * ancestors. Ancestry occupies [0, ancestry_capacity). A disjoint compact suffix holds drained
 * radiance vertices and BDPT extras for a deferred mixture fit. Histogram publication already
 * recorded every sample; compact overflow skips the GMM stream only. Ancestry overflow still
 * exceeds ancestry_capacity so the host can reject the render.
 */
struct GuidingHistory {
  ccl_global GuidingHistoryRecord *records;
  ccl_global uint *count;
  uint ancestry_capacity;
  uint observation_capacity;

  ccl_device_inline_method uint capacity() const
  {
    return ancestry_capacity + observation_capacity;
  }

  ccl_device_inline_method uint append(const uint parent,
                                       const uint field_index,
                                       const uint direction,
                                       const PackedSpectrum inverse_weight,
                                       const packed_float3 position = zero_float3(),
                                       const float distance_scale = 0.0f) const
  {
    const uint index = atomic_fetch_and_add_uint32(count, 1);
    if (index >= ancestry_capacity) {
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

  /* A completed direct/adjoint observation is not part of camera ancestry. */
  ccl_device_inline_method void append_observation(const uint field_index,
                                                   const uint direction,
                                                   const packed_float3 position,
                                                   const float value,
                                                   const float distance = FLT_MAX) const
  {
    if (!(value > 0.0f) || !isfinite_safe(value)) {
      return;
    }
    uint index;
    if (observation_capacity == 0) {
      index = append(~0u, field_index, direction, zero_spectrum(), position);
      if (index == ~0u) {
        return;
      }
    }
    else {
      index = atomic_fetch_and_add_uint32(count + 1, 1);
      if (index >= observation_capacity) {
        return;
      }
      index += ancestry_capacity;
      ccl_global GuidingHistoryRecord &record = records[index];
      record.field_index = field_index;
      record.direction = direction;
      record.inverse_weight = zero_spectrum();
      record.parent = ~0u;
      record.position = position;
      record.radiance = 0.0f;
      record.source_weight = 0.0f;
      record.inverse_distance_weight = 0.0f;
      record.distance_weight = 0.0f;
      record.distance_scale = 0.0f;
    }
    records[index].radiance = value;
    accumulate_source(index, value, distance);
  }

  /* Copy a completed ancestry vertex into the deferred GMM stream. Histogram publication already
   * recorded it; this keeps the exact direction for a later mixture fit. */
  ccl_device_inline_method void compact_completed(const uint ancestry_index) const
  {
    if (observation_capacity == 0 || ancestry_index >= ancestry_capacity) {
      return;
    }
    const ccl_global GuidingHistoryRecord &source = records[ancestry_index];
    if (source.field_index == ~0u || !(source.radiance > 0.0f) || !isfinite_safe(source.radiance)) {
      return;
    }
    const uint index = atomic_fetch_and_add_uint32(count + 1, 1);
    if (index >= observation_capacity) {
      return;
    }
    ccl_global GuidingHistoryRecord &record = records[ancestry_capacity + index];
    record.field_index = source.field_index;
    record.direction = source.direction;
    record.inverse_weight = zero_spectrum();
    record.parent = ~0u;
    record.position = source.position;
    record.radiance = source.radiance;
    record.source_weight = source.source_weight;
    record.inverse_distance_weight = source.inverse_distance_weight;
    record.distance_weight = source.distance_weight;
    record.distance_scale = 0.0f;
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
