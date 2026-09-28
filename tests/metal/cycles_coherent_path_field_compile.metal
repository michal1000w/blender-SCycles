/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#include "kernel/device/metal/compat.h"
#include "kernel/light/coherent_path_field.h"

using namespace metal;

kernel void coherent_path_field_compile_probe(device float4 *output [[buffer(0)]])
{
  CoherentGeometryInterface patches[COHERENT_GEOMETRY_MAX_INTERFACES]{};
  bool mirrors[COHERENT_GEOMETRY_MAX_INTERFACES]{};
  CoherentGeometryPath path{};
  CoherentCompletedPathField field{};
  const float3 source = float3(0, 0, 0);
  const float3 detector = float3(0, 0, 1);
  const CoherentPathDetectorFrame frame{float3(1, 0, 0),
                                       float3(0, 1, 0),
                                       float3(0, 0, -1)};
  const bool solved = coherent_geometry_connect(source, detector, frame.normal, patches, 0, &path);
  const bool transported = coherent_path_field_transport(source,
                                                          detector,
                                                          frame,
                                                          &path,
                                                          patches,
                                                          mirrors,
                                                          float3(1),
                                                          float3(1),
                                                          0,
                                                          &field);
  const float3 cross = coherent_path_field_pair_cross(&field, &field, 0, 550e-9f, 1);
  output[0] = float4(solved && transported ? 1.0f : 0.0f,
                     field.physical_diagonal_rgb.x,
                     field.native_scalar_diagonal_rgb.x,
                     cross.x);
}
