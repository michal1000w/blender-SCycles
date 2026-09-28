/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/util/diffraction_boundary.h"

kernel void diffraction_boundary(device const float4 *inputs [[buffer(0)]],
                                 device float2 *outputs [[buffer(1)]],
                                 device uint *valid [[buffer(2)]],
                                 constant uint &cases [[buffer(3)]],
                                 uint index [[thread_position_in_grid]])
{
  const uint sample = index % cases;
  const float4 a = inputs[2 * sample], b = inputs[2 * sample + 1];
  DiffractionGratingBoundary result = {};
  valid[index] = diffraction_grating_boundary(a.xyz, a.w, b.x, b.y, b.z, int(b.w), &result);
  outputs[5 * index] = result.r_te;
  outputs[5 * index + 1] = result.r_tm;
  outputs[5 * index + 2] = result.transmission;
  outputs[5 * index + 3] = result.tangent_direction;
  outputs[5 * index + 4] = make_float2(result.q2, 0.0f);
}
