/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/util/diffraction_direction.h"
kernel void diffraction_direction(device const float4 *inputs [[buffer(0)]],
                                  device float2 *outputs [[buffer(1)]],
                                  device uint *valid [[buffer(2)]],
                                  constant uint &cases [[buffer(3)]],
                                  device const int *orders [[buffer(4)]],
                                  uint index [[thread_position_in_grid]])
{
  const uint sample = index % cases;
  const float4 a = inputs[2 * sample], b = inputs[2 * sample + 1];
  float3 result = zero_float3();
  const bool propagating = diffraction_grating_direction(
      a.xyz, a.w, b.x, b.y, b.z, orders[sample], sample % 2, &result);
  valid[index] = 1;
  outputs[5 * index] = make_float2(result.x, result.y);
  outputs[5 * index + 1] = make_float2(result.z, propagating);
  outputs[5 * index + 2] = outputs[5 * index + 3] = outputs[5 * index + 4] = zero_float2();
}
