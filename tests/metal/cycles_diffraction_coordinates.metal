/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/util/diffraction_coordinates.h"
kernel void diffraction_coordinates(device const float4 *inputs [[buffer(0)]],
                                    device float2 *outputs [[buffer(1)]],
                                    device uint *valid [[buffer(2)]],
                                    constant uint &cases [[buffer(3)]],
                                    uint index [[thread_position_in_grid]])
{
  const uint sample = index % cases;
  const float4 a = inputs[2 * sample], b = inputs[2 * sample + 1];
  DiffractionCacheCoordinates result = {};
  valid[index] = diffraction_cache_coordinates(a.xyz, a.w, b.y, b.z, sample % 2, &result);
  outputs[5 * index] = make_float2(result.query.x, result.query.y);
  outputs[5 * index + 1] = make_float2(result.query.z, result.incoming_order);
  outputs[5 * index + 2] = make_float2(result.cross_polarization_sign, result.reverse_orders);
  outputs[5 * index + 3] = outputs[5 * index + 4] = zero_float2();
}
