/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/defines.h"

CCL_NAMESPACE_BEGIN

/* Packed intensity block in the ordinary lookup_table buffer. Four exact
 * integer-valued floats describe contiguous upper/lower order ranges, followed
 * by a row-major matrix of powers. Reflection and transmission share the same
 * port indexing. There are no complex fields in this ordinary-rendering data. */
enum {
  DIFFRACTION_TABLE_UPPER_FIRST = 0,
  DIFFRACTION_TABLE_UPPER_COUNT = 1,
  DIFFRACTION_TABLE_LOWER_FIRST = 2,
  DIFFRACTION_TABLE_LOWER_COUNT = 3,
  DIFFRACTION_TABLE_HEADER_SIZE = 4,
};

ccl_device_inline int diffraction_table_port(ccl_global const float *block,
                                             const int order,
                                             const bool substrate)
{
  const int start = int(
      block[substrate ? DIFFRACTION_TABLE_LOWER_FIRST : DIFFRACTION_TABLE_UPPER_FIRST]);
  const int count = int(
      block[substrate ? DIFFRACTION_TABLE_LOWER_COUNT : DIFFRACTION_TABLE_UPPER_COUNT]);
  if (order < start || order >= start + count) {
    return -1;
  }
  return order - start + (substrate ? int(block[DIFFRACTION_TABLE_UPPER_COUNT]) : 0);
}

/* A missing port is evanescent in this sample and contributes zero. The table
 * interpolation caller must also project onto the query's propagating ports. */
ccl_device_inline float diffraction_table_power(ccl_global const float *block,
                                                const int incoming_order,
                                                const bool incoming_substrate,
                                                const int outgoing_order,
                                                const bool outgoing_substrate)
{
  const int col = diffraction_table_port(block, incoming_order, incoming_substrate);
  const int row = diffraction_table_port(block, outgoing_order, outgoing_substrate);
  if (row < 0 || col < 0) {
    return 0.0f;
  }
  const int count = int(block[DIFFRACTION_TABLE_UPPER_COUNT]) +
                    int(block[DIFFRACTION_TABLE_LOWER_COUNT]);
  return block[DIFFRACTION_TABLE_HEADER_SIZE + row * count + col];
}

CCL_NAMESPACE_END
