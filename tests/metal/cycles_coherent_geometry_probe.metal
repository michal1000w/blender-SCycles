/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/light/coherent_geometry.h"

using namespace metal;

kernel void coherent_geometry_probe(device float4 *output [[buffer(0)]],
                                    uint index [[thread_position_in_grid]])
{
  if (index >= 103) return;
  CoherentGeometryInterface patches[COHERENT_GEOMETRY_MAX_INTERFACES]{};
  CoherentGeometryPath a{}, b{};
  if (index == 0) {
    const float3 u = float3(0.9323273301f, 0.0f, -0.3616154492f);
    const float3 v = float3(0.0f, 1.0f, 0.0f);
    const float3 n = normalize(cross(u, v));
    const float3 center = float3(1000.0f, -700.0f, 350.0f);
    const float3 source_a = center + n - 0.3f * u;
    const float3 source_b = source_a + 0.001f * u;
    const float3 receiver = center + 2.0f * n + 0.7f * u + 0.1f * v;
    patches[0].center = center;
    patches[0].tangent_u = u;
    patches[0].tangent_v = v;
    patches[0].half_u = patches[0].half_v = 10.0f;
    patches[0].ior_before = patches[0].ior_after = 1.0f;
    patches[0].event = COHERENT_GEOMETRY_REFLECT;
    const bool ok_a = coherent_geometry_connect(source_a, receiver, n, patches, 1, &a);
    const bool ok_b = coherent_geometry_connect(source_b, receiver, n, patches, 1, &b);
    output[0] = float4(ok_a ? 1.0f : 0.0f,
                       coherent_geometry_optical_difference(a, b),
                       a.spreading,
                       ok_b ? 1.0f : 0.0f);
    output[3] = float4(source_a, 0.0f);
    output[4] = float4(source_b, 0.0f);
    output[5] = float4(receiver, 0.0f);
    return;
  }
  if (index == 1) {
    patches[0].center = float3(0.0f);
    patches[0].tangent_u = float3(1.0f, 0.0f, 0.0f);
    patches[0].tangent_v = float3(0.0f, 1.0f, 0.0f);
    patches[0].half_u = patches[0].half_v = 10.0f;
    patches[0].ior_before = patches[0].ior_after = 1.0f;
    patches[0].event = COHERENT_GEOMETRY_REFLECT;
    const float3 source = float3(-0.35f, 0.17f, 1.0f);
    const float3 receiver = float3(0.85f, -0.12f, 2.0f);
    const bool ok = coherent_geometry_connect(source, receiver, float3(0, 0, 1), patches, 1, &a);
    output[1] = float4(ok ? 1.0f : 0.0f, a.optical_length, a.spreading, a.point[0].x);
    return;
  }
  if (index >= 84) {
    patches[0].center = float3(0.015000000596046448f, 0.0f, 0.0f);
    patches[0].tangent_u = float3(-4.656612517806025e-08f, 0.0f, 1.0f);
    patches[0].tangent_v = float3(0.0f, 1.0f, 0.0f);
    patches[0].half_u = patches[0].half_v = 0.020000001415610355f;
    patches[0].ior_before = 1.0f;
    patches[0].ior_after = patches[0].ior_opposite = 1.5f;
    patches[0].event = COHERENT_GEOMETRY_TRANSMIT;
    patches[0].expected_incident_side = 1;
    patches[1].center = float3(0.02250000089406967f, 0.019999999552965164f, 0.0f);
    patches[1].tangent_u = float3(1.0f, 0.0f, 0.0f);
    patches[1].tangent_v = float3(0.0f, 0.0f, 1.0f);
    patches[1].half_u = 0.00699999975040555f;
    patches[1].half_v = 0.014999999664723873f;
    patches[1].ior_before = patches[1].ior_after = patches[1].ior_opposite = 1.5f;
    patches[1].event = COHERENT_GEOMETRY_REFLECT;
    patches[2].center = float3(0.030000001192092896f, 0.0f, 0.0f);
    patches[2].tangent_u = float3(0.0f, 0.0f, -1.0f);
    patches[2].tangent_v = float3(0.0f, 1.0f, 0.0f);
    patches[2].half_u = patches[2].half_v = 0.020000001415610313f;
    patches[2].ior_before = 1.5f;
    patches[2].ior_after = patches[2].ior_opposite = 1.0f;
    patches[2].event = COHERENT_GEOMETRY_TRANSMIT;
    patches[2].expected_incident_side = -1;
    const bool unequal = index == 102;
    if (unequal) patches[1].center.y = 0.01f;
    const uint grid = index - 84;
    const float sy = unequal ? -25.0e-6f : (grid / 9 == 0 ? -25.0e-6f : 25.0e-6f);
    const float ry = unequal ? 0.0f : (int(grid / 3) % 3 - 1) * 18.0e-6f;
    const float rz = unequal ? 0.0f : (int(grid) % 3 - 1) * 18.0e-6f;
    const float3 source = float3(unequal ? 0.0149f : 0.0f, sy, 0.0f);
    const float3 receiver = float3(unequal ? 0.0305f : 0.05000000074505806f, ry, rz);
    const bool ok = coherent_geometry_connect(
        source, receiver, float3(-1.0f, 0.0f, 0.0f), patches, 3, &a);
    output[168 + grid] = float4(ok ? 1.0f : 0.0f,
                                a.optical_length_split.x + a.optical_length_split.y,
                                a.spreading,
                                a.point[1].x);
    return;
  }
  if (index >= 3) {
    const uint grid = index - 3;
    const float3 source_a = float3(-25.0e-6f, 0.0f, 1.0f);
    const float3 source_b = float3(25.0e-6f, 0.0f, 1.0f);
    const float3 receiver = float3(0.8f + float(grid % 9) * 0.05f,
                                   -0.2f + float(grid / 9) * 0.05f,
                                   2.0f);
    patches[0].center = float3(0.0f);
    patches[0].tangent_u = float3(1, 0, 0);
    patches[0].tangent_v = float3(0, 1, 0);
    patches[0].half_u = patches[0].half_v = 10.0f;
    patches[0].ior_before = patches[0].ior_after = 1.0f;
    patches[0].event = COHERENT_GEOMETRY_REFLECT;
    const bool ok_a = coherent_geometry_connect(source_a, receiver, float3(0, 0, 1),
                                                patches, 1, &a);
    const bool ok_b = coherent_geometry_connect(source_b, receiver, float3(0, 0, 1),
                                                patches, 1, &b);
    output[6 + 2 * grid] = float4(receiver, coherent_geometry_optical_difference(a, b));
    output[7 + 2 * grid] = float4(ok_a ? 1.0f : 0.0f,
                                  ok_b ? 1.0f : 0.0f,
                                  a.spreading,
                                  b.spreading);
    return;
  }
  patches[0].center = float3(0.0f);
  patches[1].center = float3(0.0f, 0.0f, -1.0f);
  for (int i = 0; i < 2; i++) {
    patches[i].tangent_u = float3(1.0f, 0.0f, 0.0f);
    patches[i].tangent_v = float3(0.0f, 1.0f, 0.0f);
    patches[i].half_u = patches[i].half_v = 10.0f;
    patches[i].event = COHERENT_GEOMETRY_TRANSMIT;
  }
  patches[0].ior_before = 1.0f;
  patches[0].ior_after = 1.5f;
  patches[1].ior_before = 1.5f;
  patches[1].ior_after = 1.0f;
  const bool ok = coherent_geometry_connect(
      float3(0, 0, 1), float3(0, 0, -2), float3(0, 0, 1), patches, 2, &a);
  output[2] = float4(ok ? 1.0f : 0.0f, a.optical_length, a.spreading, a.point[0].x);
}
