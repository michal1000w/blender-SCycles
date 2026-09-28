/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/util/diffraction_boundary.h"
#include "kernel/util/diffraction_cache.h"
CCL_NAMESPACE_BEGIN

struct DiffractionCacheCoordinates {
  float3 query;
  int incoming_order;
  float cross_polarization_sign;
  bool reverse_orders;
};

/* Direction carries signed tangential wave-vector components in the grating
 * frame. The resulting Bloch coordinate is canonical in [-0.5, 0.5).
 * Preserve the integer order separately from the reduced coordinate. Input
 * and output port indices must both be reflected when reverse_orders is set.
 * Wavelength and pitch are vacuum lengths in the same units. */
ccl_device_inline bool diffraction_cache_coordinates(
    const float3 direction,
    const float incident_index,
    const float wavelength,
    const float pitch,
    const bool mirror_symmetry,
    ccl_private DiffractionCacheCoordinates *result)
{
  if (!(incident_index > 0) || !(wavelength > 0) || !(pitch > 0) ||
      !isfinite_safe(incident_index) || !isfinite_safe(wavelength) || !isfinite_safe(pitch) ||
      !isfinite_safe(direction.x) || !isfinite_safe(direction.y) || !isfinite_safe(direction.z))
    return false;
  const float2 x = make_float2(direction.x, 0), y = make_float2(direction.y, 0);
  const float2 z = make_float2(direction.z, 0);
  const float2 norm2 = diffraction_extended_add(
      diffraction_extended_add(diffraction_extended_mul(x, x), diffraction_extended_mul(y, y)),
      diffraction_extended_mul(z, z));
  if (!(norm2.x > 0) || !isfinite_safe(norm2.x))
    return false;
  const float2 scale = diffraction_extended_div(make_float2(incident_index, 0),
                                                diffraction_extended_sqrt(norm2));
  const float2 kx = diffraction_extended_mul(scale, x);
  const float2 ky = diffraction_extended_mul(scale, y);
  const float2 cycles = diffraction_extended_mul(
      kx, diffraction_extended_div(make_float2(pitch, 0), make_float2(wavelength, 0)));
  /* Leave room for endpoint canonicalization and mirror negation in int32. */
  if (!isfinite_safe(cycles.x) || fabsf(cycles.x) >= 2147483520.0f)
    return false;
  const float2 shifted = diffraction_extended_add(cycles, make_float2(0.5f, 0));
  const float integral = floorf(shifted.x);
  int order = int(integral) + int(floorf((shifted.x - integral) + shifted.y));
  const float order_high = float(order);
  const float order_low = float(order - int(order_high));
  const float2 remainder = diffraction_extended_add(cycles, make_float2(-order_high, -order_low));
  float bloch = remainder.x + remainder.y;
  /* Float conversion can round a value just below +0.5 onto the seam. Both
   * representations describe the same wave vector; use the canonical one. */
  if (bloch >= 0.5f) {
    bloch -= 1.0f;
    order++;
  }
  if (bloch < -0.5f) {
    bloch += 1.0f;
    order--;
  }
  float3 query = make_float3(bloch, ky.x + ky.y, wavelength);
  if (!isfinite_safe(query.y) || bloch < -0.5f || bloch >= 0.5f)
    return false;
  result->incoming_order = order;
  result->reverse_orders = false;
  result->cross_polarization_sign = 1;
  if (mirror_symmetry) {
    const DiffractionCacheSymmetry symmetry = diffraction_cache_symmetry(query);
    query = symmetry.query;
    result->reverse_orders = symmetry.reverse_orders;
    result->cross_polarization_sign = symmetry.cross_polarization_sign;
    if (symmetry.reverse_orders)
      result->incoming_order = -order;
  }
  result->query = query;
  return true;
}
CCL_NAMESPACE_END
