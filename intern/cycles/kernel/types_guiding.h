/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/types.h"
#include "util/types_spectrum.h"

CCL_NAMESPACE_BEGIN

/* Immutable until every main and shadow path in a training group has drained. */
struct GuidingHistoryRecord {
  uint field_index;
  uint direction;
  PackedSpectrum inverse_weight;
  uint parent;
  packed_float3 position;
  /* Atomically accumulated by main and shadow paths; consumed after the group drains. */
  float radiance;
  float source_weight;
  float inverse_distance_weight;
  float distance_weight;
  /* Immutable transfer through this vertex toward its parent. */
  float distance_scale;
};
static_assert(sizeof(GuidingHistoryRecord) == 56, "Guiding history host/device layout");

struct GuidingSpatialNode {
  /* Zero denotes a leaf; children otherwise occupy two consecutive node indices. */
  uint children;
  uint parent;
  uint axis;
  float split;
  uint depth;
  /* Visits also include zero-valued observations, independently of directional mass. */
  uint visits;
};

CCL_NAMESPACE_END
