/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Must define GPU qualifiers before any Cycles utility header. */
#include "kernel/device/metal/compat.h"

#include "kernel/closure/bsdf_diffraction_util.h"
#include "kernel/util/diffraction_grid.h"

/* Tests the production header on an actual Metal device. This is a math test,
 * not a renderer performance benchmark. Input/output use explicit float4 slots
 * to avoid CPU/GPU float3 padding differences. */
kernel void diffraction_math(device const float4 *inputs [[buffer(0)]],
                             device float4 *outputs [[buffer(1)]],
                             uint index [[thread_position_in_grid]])
{
  const float4 a = inputs[2 * index];
  const float4 b = inputs[2 * index + 1];
  const int order = int(index % 17) - 8;
  float3 wo;
  const bool valid = diffraction_order_direction(a.xyz, a.w, order, b.x, index % 2, &wo);
  const float2 amplitude = diffraction_binary_amplitude(order, b.y, b.z, b.w);
  outputs[6 * index] = float4(wo, valid ? 1.0f : 0.0f);
  outputs[6 * index + 1] = float4(amplitude,
                                  diffraction_binary_power(order, b.y, b.z),
                                  diffraction_relief_phase(125.0f, 550.0f, 1.0f, b.x, a.z, wo.z));
  const float3 h = normalize(float3(2.0f * b.z - 1.0f, 2.0f * b.w - 1.0f, 0.1f + b.x));
  const float3 axis = float3(1.0f, 0.0f, 0.0f);
  const float delta = (int(index % 5) - 2) * a.w;
  const bool reflected = diffraction_facet_reflect(a.xyz, h, axis, delta, &wo);
  outputs[6 * index + 2] = float4(wo, reflected ? 1.0f : 0.0f);
  float4 geometry = float4(0.0f);
  if (reflected) {
    geometry.x = diffraction_reflection_jacobian(a.xyz, wo, h, axis, delta);
    geometry.y = diffraction_reflection_jacobian(wo, a.xyz, h, axis, delta);
    DiffractionReflectionRoot roots[2];
    const int count = diffraction_reflection_half_vectors(a.xyz, wo, axis, delta, roots);
    geometry.z = float(count);
    geometry.w = 2.0f;
    for (int r = 0; r < count; r++) {
      geometry.w = min(geometry.w, len(roots[r].h - h));
    }
  }
  outputs[6 * index + 3] = geometry;
  const bool transmitted = diffraction_facet_transmit(a.xyz, h, axis, b.x, delta, &wo);
  outputs[6 * index + 4] = float4(wo, transmitted ? 1.0f : 0.0f);
  geometry = float4(0.0f);
  if (transmitted) {
    geometry.x = diffraction_transmission_jacobian(a.xyz, wo, h, axis, b.x, delta);
    DiffractionTransmissionRoot roots[2];
    const int count = diffraction_transmission_half_vectors(a.xyz, wo, axis, b.x, delta, roots);
    geometry.z = float(count);
    geometry.w = 2.0f;
    for (int r = 0; r < count; r++) geometry.w = min(geometry.w, len(roots[r].h - h));
  }
  outputs[6 * index + 5] = geometry;

}

/* Packed table reads use the exact production reader and explicit port pairs. */
kernel void diffraction_table_test(device const float *block [[buffer(0)]],
                                    device const int4 *ports [[buffer(1)]],
                                    device float *outputs [[buffer(2)]],
                                    uint index [[thread_position_in_grid]])
{
  const int4 p = ports[index];
  outputs[index] = diffraction_table_power(block, p.x, p.y != 0, p.z, p.w != 0);
}

kernel void diffraction_grid_test(device const float *grid [[buffer(0)]],
                                   device const float4 *queries [[buffer(1)]],
                                   device const int4 *ports [[buffer(2)]],
                                   device float2 *outputs [[buffer(3)]],
                                   uint index [[thread_position_in_grid]])
{
  const float4 q = queries[index];
  const int4 p = ports[index];
  float power;
  const bool valid = diffraction_grid_power(grid, q.x, q.y, q.z, p.y != 0, p.x,
                                             p.w != 0, &power);
  outputs[index] = float2(power, valid ? 1.0f : 0.0f);
}
