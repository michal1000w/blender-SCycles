/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/util/diffraction_boundary.h"
CCL_NAMESPACE_BEGIN

/* Construct a propagating order in the local grating frame. The incident
 * vector carries the signed tangential propagation momentum; its normal sign
 * is immaterial. Reflected output has positive z, transmitted output negative
 * z. The order is relative to the incoming order, not an absolute cache port.
 * A false return includes evanescent/zero-normal-flux orders and invalid input;
 * it must not be interpreted as absorption when enumerating propagating ports.
 * All indices are real exterior indices, and lengths use the same units. */
ccl_device_inline bool diffraction_grating_direction(const float3 incident,
                                                     const float incident_index,
                                                     const float exterior_index,
                                                     const float wavelength,
                                                     const float pitch,
                                                     const int relative_order,
                                                     const bool transmission,
                                                     ccl_private float3 *outgoing)
{
  DiffractionGratingMomentum momentum;
  if (!diffraction_grating_momentum(incident,
                                    incident_index,
                                    exterior_index,
                                    wavelength,
                                    pitch,
                                    relative_order,
                                    &momentum) ||
      !(momentum.q2 > 0))
    return false;
  const float2 ox = diffraction_extended_div(momentum.x, make_float2(exterior_index, 0));
  const float2 oy = diffraction_extended_div(momentum.y, make_float2(exterior_index, 0));
  *outgoing = make_float3(ox.x + ox.y,
                          oy.x + oy.y,
                          (transmission ? -1.0f : 1.0f) * sqrtf(momentum.q2) / exterior_index);
  return isfinite_safe(outgoing->x) && isfinite_safe(outgoing->y) && isfinite_safe(outgoing->z);
}
/* Solid-angle Jacobian of one fixed propagating order, |d omega_o/d omega_i|.
 * Tangential momentum conservation gives d(o_t)/d(i_t) = n_i/n_o times I.
 * Since d omega = d(t_x)d(t_y)/|cos(theta)|, the area determinant is
 * (n_i/n_o)^2 |cos(theta_i)/cos(theta_o)|. The order's constant momentum
 * offset affects its outgoing cosine but not the transverse derivative.
 * Inputs are unit propagation directions in the same local grating frame.
 * This is a direction-map Jacobian, NOT an order probability or a BSDF PDF.
 * Grazing endpoints have no finite invertible map and are rejected. */
ccl_device_inline bool diffraction_grating_solid_angle_jacobian(
    const float3 incident,
    const float3 outgoing,
    const float incident_index,
    const float exterior_index,
    ccl_private float *jacobian)
{
  *jacobian = 0;
  const float ci = fabsf(incident.z), co = fabsf(outgoing.z);
  if (!(incident_index > 0) || !(exterior_index > 0) || !(ci > 0) || !(co > 0) ||
      !isfinite_safe(incident_index) || !isfinite_safe(exterior_index) ||
      !isfinite_safe(ci) || !isfinite_safe(co))
    return false;
  const float eta = incident_index / exterior_index;
  const float result = eta * eta * (ci / co);
  if (!(result > 0) || !isfinite_safe(result))
    return false;
  *jacobian = result;
  return true;
}
CCL_NAMESPACE_END
