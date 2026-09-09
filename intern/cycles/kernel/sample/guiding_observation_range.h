/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/sample/guiding_mixture_statistics.h"
#include "kernel/types_guiding.h"
#include "util/atomic.h"
#include "util/types_normal.h"

CCL_NAMESPACE_BEGIN

/* Partition drained histories by field without copying observation payloads.
 * Count, prefix, and scatter are separate ordered dispatches. Counters/error are
 * initialized to zero by the caller. Histories and ancestry remain immutable. */
struct GuidingObservationPartition {
  const ccl_global GuidingHistoryRecord *records;
  ccl_global uint *counts;
  ccl_global uint *offsets;
  ccl_global uint *cursors;
  ccl_global uint *indices;
  ccl_global uint *error;
  uint distributions;
  uint field_stride;
  uint capacity;

  ccl_device_inline_method uint field(const uint index) const
  {
    const ccl_global auto &record = records[index];
    if (record.field_index == ~0u || !(record.radiance > 0.0f) || !isfinite_safe(record.radiance))
    {
      return ~0u;
    }
    const uint distribution = record.field_index / field_stride;
    if (distribution >= distributions) {
      atomic_fetch_and_or_uint32(error, 1u);
      return ~0u;
    }
    return distribution;
  }

  ccl_device_inline_method void count(const uint index) const
  {
    const uint distribution = field(index);
    if (distribution != ~0u) {
      atomic_fetch_and_add_uint32(counts + distribution, 1);
    }
  }

  ccl_device_inline_method void prefix() const
  {
    uint total = 0;
    for (uint field = 0; field < distributions; ++field) {
      offsets[field] = total;
      cursors[field] = 0;
      if (counts[field] > capacity - total) {
        atomic_fetch_and_or_uint32(error, 2u);
        total = capacity;
      }
      else {
        total += counts[field];
      }
    }
  }

  ccl_device_inline_method void scatter(const uint index) const
  {
    const uint distribution = field(index);
    if (distribution == ~0u) {
      return;
    }
    const uint local = atomic_fetch_and_add_uint32(cursors + distribution, 1);
    if (local >= counts[distribution] || local >= capacity - offsets[distribution]) {
      atomic_fetch_and_or_uint32(error, 4u);
      return;
    }
    indices[offsets[distribution] + local] = index;
  }
};

/* Indexed view consumed directly by directional fitting and source collection.
 * Source distances are converted to the isotropic metric at read time. */
struct GuidingHistoryObservationRange {
  const ccl_global GuidingHistoryRecord *records;
  const ccl_global uint *indices;
  uint offset;
  float scene_scale;

  ccl_device_inline_method GuidingMixtureObservation observation(const uint index) const
  {
    const ccl_global auto &record = records[indices[offset + index]];
    packed_normal packed;
    packed.value = record.direction;
    const float3 direction = packed.decode();
    return {make_float4(direction.x, direction.y, direction.z, record.radiance),
            make_float4(record.position.x, record.position.y, record.position.z, 0),
            make_float4(record.source_weight,
                        record.inverse_distance_weight * scene_scale,
                        record.distance_weight / scene_scale,
                        0)};
  }

  ccl_device_inline_method float4 operator[](const uint index) const
  {
    return observation(index).direction_weight;
  }
};

CCL_NAMESPACE_END
