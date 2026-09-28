/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/closure/bsdf_diffraction_util.h"

CCL_NAMESPACE_BEGIN

/* Experimental intensity-only, smooth scalar interface. Not a Maxwell model.
 * Reflection uses the existing angle-dependent binary phase-screen powers;
 * transmission uses a thin phase plate with angle-independent optical delay.
 * R and T are reciprocal, direction-independent flux budgets (R + T <= 1).
 * Any unavailable diffracted power returns to specular reflection, not to a
 * direction-dependent renormalization of transmitted orders. This preserves
 * pairwise reciprocity and passivity, including at channel cutoffs. It can
 * overestimate reflection near cutoffs; it is not yet a production closure.
 */
struct FastDiffractionInterface {
  float wavelength, pitch, depth, duty;
  float incident_ior, transmitted_ior;
  float reflection_budget, transmission_budget;
  float transmission_phase;
};

ccl_device_inline bool fast_diffraction_interface_valid(const FastDiffractionInterface &p)
{
  return p.wavelength > 0 && isfinite_safe(p.wavelength) && p.pitch > 0 &&
         isfinite_safe(p.pitch) && p.depth >= 0 && isfinite_safe(p.depth) &&
         p.duty >= 0 && p.duty <= 1 && p.incident_ior > 0 &&
         isfinite_safe(p.incident_ior) && p.transmitted_ior > 0 &&
         isfinite_safe(p.transmitted_ior) && p.reflection_budget >= 0 &&
         p.transmission_budget >= 0 && p.reflection_budget + p.transmission_budget <= 1 &&
         isfinite_safe(p.transmission_phase) &&
         (p.incident_ior + max(p.incident_ior, p.transmitted_ior)) * p.pitch /
             p.wavelength < 4096;
}

ccl_device_inline int fast_diffraction_interface_order_bound(const FastDiffractionInterface &p,
                                                            const bool transmission)
{
  const float outgoing_ior = transmission ? p.transmitted_ior : p.incident_ior;
  return int(ceilf((p.incident_ior + outgoing_ior) * p.pitch / p.wavelength));
}

/* wi points away from the surface; its local z is positive. */
ccl_device_inline bool fast_diffraction_interface_direction(const FastDiffractionInterface &p,
                                                           const float3 wi,
                                                           const int order,
                                                           const bool transmission,
                                                           ccl_private float3 *wo)
{
  const float n = transmission ? p.transmitted_ior : p.incident_ior;
  const float x = (-p.incident_ior * wi.x + float(order) * p.wavelength / p.pitch) / n;
  const float y = -p.incident_ior * wi.y / n;
  const float z2 = 1 - x*x - y*y;
  if (!(wi.z > 0 && z2 > 0)) return false;
  *wo = make_float3(x, y, (transmission ? -1 : 1) * sqrtf(z2));
  return true;
}

ccl_device_inline float fast_diffraction_interface_off_diagonal(
    const FastDiffractionInterface &p, const float3 wi, const int order,
    const bool transmission, ccl_private float3 *wo)
{
  if ((!transmission && order == 0) ||
      !fast_diffraction_interface_direction(p, wi, order, transmission, wo)) return 0;
  const float phase = transmission ? p.transmission_phase :
      M_2PI_F * p.depth * p.incident_ior / p.wavelength * (wi.z + wo->z);
  return (transmission ? p.transmission_budget : p.reflection_budget) *
         diffraction_binary_power(order, phase, p.duty);
}

ccl_device_inline float fast_diffraction_interface_residual(const FastDiffractionInterface &p,
                                                           const float3 wi)
{
  float other = 0;
  for (int side = 0; side < 2; ++side) {
    const int bound = fast_diffraction_interface_order_bound(p, bool(side));
    for (int m = -bound; m <= bound; ++m) {
      float3 wo;
      other += fast_diffraction_interface_off_diagonal(p, wi, m, bool(side), &wo);
    }
  }
  /* Do not hide an invalid negative residual in this experiment. */
  return p.reflection_budget + p.transmission_budget - other;
}

/* Conditional on non-absorption: probability is power/(R+T), throughput R+T.
 * The eventual closure must also apply the appropriate radiance eta factor. */
ccl_device_inline bool fast_diffraction_interface_sample(
    const FastDiffractionInterface &p, const float3 wi, const float random,
    ccl_private int *order, ccl_private bool *transmission,
    ccl_private float3 *wo, ccl_private float *power, ccl_private float *probability)
{
  const float budget = p.reflection_budget + p.transmission_budget;
  if (!fast_diffraction_interface_valid(p) || !(wi.z > 0) ||
      !(random >= 0 && random < 1) || !(budget > 0)) return false;
  float u = random * budget, total = 0;
  for (int side = 0; side < 2; ++side) {
    const int bound = fast_diffraction_interface_order_bound(p, bool(side));
    for (int m = -bound; m <= bound; ++m) {
      const float value = fast_diffraction_interface_off_diagonal(p, wi, m, bool(side), wo);
      if (u < value) {
        *order = m; *transmission = bool(side); *power = value;
        *probability = value / budget;
        return true;
      }
      u -= value;
      total += value;
    }
  }
  *power = budget - total;
  if (!(*power > 0)) return false;
  *order = 0; *transmission = false; *probability = *power / budget;
  *wo = make_float3(-wi.x, -wi.y, wi.z);
  return true;
}

CCL_NAMESPACE_END
