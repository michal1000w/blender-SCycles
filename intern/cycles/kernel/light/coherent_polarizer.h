/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/light/coherent_field.h"

CCL_NAMESPACE_BEGIN

/* A physical object axis is independent of face orientation. Project it into
 * the ray's transverse plane. At the singular axis-parallel direction the
 * ideal transverse analyzer has no defined transmission direction. */
ccl_device_inline bool coherent_polarizer_axis(const float3 world_axis,
                                               const float3 transverse_s,
                                               const float3 transverse_p,
                                               ccl_private float2 *axis)
{
  const float2 projected = make_float2(dot(world_axis, transverse_s),
                                       dot(world_axis, transverse_p));
  const float norm = projected.x * projected.x + projected.y * projected.y;
  if (!(norm > 1.0e-12f) || !isfinite_safe(norm)) return false;
  *axis = projected * (1.0f / sqrtf(norm));
  return true;
}

/* Real rank-one Jones projector: absorption removes the orthogonal field.
 * This preserves complex phase and is passive, idempotent and reciprocal. */
ccl_device_inline CoherentJonesField coherent_polarizer_project(
    const CoherentJonesField field, const float2 unit_axis)
{
  const float2 amplitude = coherent_field_add(coherent_field_scale(field.s, unit_axis.x),
                                             coherent_field_scale(field.p, unit_axis.y));
  return {coherent_field_scale(amplitude, unit_axis.x),
          coherent_field_scale(amplitude, unit_axis.y)};
}

/* Dichroic analyzer coincident with a refracting interface. Both endpoint
 * projectors are necessary: P_out T P_in reverses as its transpose, unlike
 * applying only P_out T. IOR-matched panes reduce exactly to one projector. */
ccl_device_inline CoherentJonesField coherent_polarizer_transmit(
    const CoherentJonesField incoming,
    const float2 incident_axis,
    const float2 outgoing_axis,
    const CoherentScalarInterface factor_s,
    const CoherentScalarInterface factor_p)
{
  const CoherentJonesField selected = coherent_polarizer_project(incoming, incident_axis);
  const CoherentJonesField transmitted = coherent_field_interface_jones(
      selected, factor_s, factor_p, true, 1.0f);
  return coherent_polarizer_project(transmitted, outgoing_axis);
}

CCL_NAMESPACE_END
