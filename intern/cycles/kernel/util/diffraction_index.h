/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Host-validated, scene-owned lookup table. Preserve the supplied nonuniform
 * knots and interpolate n and k themselves, without RGB conversion or uniform
 * resampling. Extrapolation is deliberately rejected. */
ccl_device_inline bool diffraction_index_lookup(KernelGlobals kg,
                                                const int offset,
                                                const float wavelength,
                                                ccl_private float2 *index)
{
  if (offset < 0 || !isfinite_safe(wavelength)) {
    return false;
  }
  const int count = int(kernel_data_fetch(lookup_table, offset));
  if (count < 2 || count > (1 << 20)) {
    return false;
  }
  const int base = offset + 1;
  const float first = kernel_data_fetch(lookup_table, base);
  const float last = kernel_data_fetch(lookup_table, base + 3 * (count - 1));
  if (wavelength < first || wavelength > last) {
    return false;
  }
  int lo = 0, hi = count - 1;
  while (hi - lo > 1) {
    const int mid = lo + (hi - lo) / 2;
    if (wavelength < kernel_data_fetch(lookup_table, base + 3 * mid)) {
      hi = mid;
    }
    else {
      lo = mid;
    }
  }
  const int a = base + 3 * lo, b = base + 3 * hi;
  const float wa = kernel_data_fetch(lookup_table, a);
  const float wb = kernel_data_fetch(lookup_table, b);
  const float t = (wavelength - wa) / (wb - wa);
  const float2 ia = make_float2(kernel_data_fetch(lookup_table, a + 1),
                                kernel_data_fetch(lookup_table, a + 2));
  const float2 ib = make_float2(kernel_data_fetch(lookup_table, b + 1),
                                kernel_data_fetch(lookup_table, b + 2));
  *index = (1 - t) * ia + t * ib;
  return isfinite_safe(index->x) && isfinite_safe(index->y);
}

CCL_NAMESPACE_END
