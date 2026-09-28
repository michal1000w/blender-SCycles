/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/util/diffraction_table.h"
#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Cell-centered periodic Bloch and symmetric ky coordinates, with wavelength
 * samples at both endpoints. Each node offset addresses a packed power block.
 * Offsets are exact integer-valued floats; the host limits the table to 2^24
 * floats. The header stores refractive indices of the external lossless ports;
 * an absorbing substrate has lower index zero and no lower far-field ports. */
enum {
  DIFFRACTION_GRID_BLOCH_COUNT = 0,
  DIFFRACTION_GRID_TANGENT_COUNT = 1,
  DIFFRACTION_GRID_WAVELENGTH_COUNT = 2,
  DIFFRACTION_GRID_TANGENT_MAX = 3,
  DIFFRACTION_GRID_WAVELENGTH_MIN = 4,
  DIFFRACTION_GRID_WAVELENGTH_MAX = 5,
  DIFFRACTION_GRID_PITCH = 6,
  DIFFRACTION_GRID_UPPER_IOR = 7,
  DIFFRACTION_GRID_LOWER_IOR = 8,
  DIFFRACTION_GRID_CLOSED_PORT_REFLECTION = 9,
  DIFFRACTION_GRID_HEADER_SIZE = 10,
};

/* kx and ky are the incident propagation wavevector components divided by
 * vacuum k0. order_change is outgoing minus incoming grating momentum order.
 * Returns false for a query outside the supported spectral/propagating domain;
 * true with zero power can represent an evanescent outgoing order. */
ccl_device_inline bool diffraction_grid_power(ccl_global const float *grid,
                                              const float wavelength,
                                              const float kx,
                                              const float ky,
                                              const bool incoming_substrate,
                                              const int order_change,
                                              const bool outgoing_substrate,
                                              ccl_private float *power)
{
  *power = 0.0f;
  const float lower_lambda = grid[DIFFRACTION_GRID_WAVELENGTH_MIN];
  const float upper_lambda = grid[DIFFRACTION_GRID_WAVELENGTH_MAX];
  const float ni =
      grid[incoming_substrate ? DIFFRACTION_GRID_LOWER_IOR : DIFFRACTION_GRID_UPPER_IOR];
  if (!isfinite_safe(wavelength) || !isfinite_safe(kx) || !isfinite_safe(ky) ||
      wavelength < lower_lambda || wavelength > upper_lambda || !(ni > 0.0f) ||
      !(kx * kx + ky * ky < ni * ni))
  {
    return false;
  }
  const float step = wavelength / grid[DIFFRACTION_GRID_PITCH];
  const float no =
      grid[outgoing_substrate ? DIFFRACTION_GRID_LOWER_IOR : DIFFRACTION_GRID_UPPER_IOR];
  const float outgoing_x = kx + float(order_change) * step;
  if (!(no > 0.0f) || !(outgoing_x * outgoing_x + ky * ky < no * no)) {
    return true;
  }
  const int nk = int(grid[DIFFRACTION_GRID_BLOCH_COUNT]);
  const int nv = int(grid[DIFFRACTION_GRID_TANGENT_COUNT]);
  const int nl = int(grid[DIFFRACTION_GRID_WAVELENGTH_COUNT]);
  const float momentum = kx / step;
  const int incoming_order = int(floorf(momentum + 0.5f));
  const float bloch = momentum - incoming_order;
  const float x = (bloch + 0.5f) * nk - 0.5f;
  const float y = clamp(
      (ky / grid[DIFFRACTION_GRID_TANGENT_MAX] + 1.0f) * 0.5f * nv - 0.5f, 0.0f, float(nv - 1));
  const float z = (wavelength - lower_lambda) / (upper_lambda - lower_lambda) * (nl - 1);
  const int ix = int(floorf(x)), iy = min(int(y), nv - 2), iz = min(int(z), nl - 2);
  const float tx = x - ix, ty = y - iy, tz = z - iz;
  for (int a = 0; a < 2; a++) {
    const int unwrapped = ix + a;
    const int shift = unwrapped < 0 ? -1 : (unwrapped >= nk ? 1 : 0);
    const int u = unwrapped - shift * nk;
    for (int b = 0; b < 2; b++) {
      for (int c = 0; c < 2; c++) {
        const float weight = (a ? tx : 1.0f - tx) * (b ? ty : 1.0f - ty) * (c ? tz : 1.0f - tz);
        const int node = ((iz + c) * nv + iy + b) * nk + u;
        const int offset = int(grid[DIFFRACTION_GRID_HEADER_SIZE + node]);
        /* Unit diagonal padding preserves contraction and reciprocity in
         * the enlarged port space and uses the generic reflecting grazing
         * limit. This is an interpolation extension, not evanescent power. */
        if (grid[DIFFRACTION_GRID_CLOSED_PORT_REFLECTION] != 0.0f && order_change == 0 &&
            incoming_substrate == outgoing_substrate &&
            diffraction_table_port(grid + offset, incoming_order + shift, incoming_substrate) < 0)
        {
          *power += weight;
        }
        else {
          *power += weight * diffraction_table_power(grid + offset,
                                                     incoming_order + shift,
                                                     incoming_substrate,
                                                     incoming_order + order_change + shift,
                                                     outgoing_substrate);
        }
      }
    }
  }
  return true;
}

CCL_NAMESPACE_END
