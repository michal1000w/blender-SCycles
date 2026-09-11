/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/types.h"

CCL_NAMESPACE_BEGIN

/* Stable in-place removal of invalid indices. Exactly one full SIMD group calls
 * the Metal version. Every chunk is read before any lane overwrites its input;
 * compacted output never reaches the following input chunk. */
ccl_device_inline uint compact_indices(ccl_global uint *indices,
                                       const uint size,
                                       const uint lane = 0,
                                       const uint width = 32)
{
  uint count = 0;
#ifdef __KERNEL_METAL__
  for (uint begin = 0; begin < size; begin += width) {
    const uint index = begin + lane;
    const uint slot = index < size ? indices[index] : ~0u;
    const uint keep = slot != ~0u;
    const uint offset = metal::simd_prefix_exclusive_sum(keep);
    const uint kept = metal::simd_sum(keep);
    metal::simdgroup_barrier(metal::mem_flags::mem_device);
    if (keep) {
      indices[count + offset] = slot;
    }
    count += kept;
  }
#else
  (void)lane;
  (void)width;
  for (uint index = 0; index < size; ++index) {
    const uint slot = indices[index];
    if (slot != ~0u) {
      indices[count++] = slot;
    }
  }
#endif
  return count;
}

CCL_NAMESPACE_END
