/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#include "kernel/device/metal/compat.h"
#include "kernel/light/coherent_path_field.h"

using namespace metal;

kernel void coherent_path_field_probe(device float4 *output [[buffer(0)]])
{
  CoherentGeometryInterface patches[COHERENT_GEOMETRY_MAX_INTERFACES]{};
  bool mirrors[COHERENT_GEOMETRY_MAX_INTERFACES]{};
  CoherentGeometryPath path{};
  CoherentCompletedPathField field{};
  const float3 white = float3(1);
  const CoherentPathDetectorFrame lower{float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, -1)};
  const CoherentPathDetectorFrame upper{float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1)};

  /* Normal unit mirror: independent virtual source is three metres away. */
  patches[0].center = float3(0);
  patches[0].tangent_u = float3(1, 0, 0);
  patches[0].tangent_v = float3(0, 1, 0);
  patches[0].half_u = patches[0].half_v = 10;
  patches[0].ior_before = patches[0].ior_after = patches[0].ior_opposite = 1;
  patches[0].event = COHERENT_GEOMETRY_REFLECT;
  mirrors[0] = true;
  const float3 mirror_source = float3(0, 0, 1);
  const float3 mirror_detector = float3(0, 0, 2);
  bool solved = coherent_geometry_connect(mirror_source, mirror_detector, lower.normal,
                                         patches, 1, &path);
  bool transported = solved && coherent_path_field_transport(mirror_source, mirror_detector,
                                                              lower, &path, patches, mirrors,
                                                              white, white, 0, &field);
  output[0] = float4(transported ? 1 : 0, field.physical_diagonal_rgb.x,
                     field.native_scalar_diagonal_rgb.x, path.spreading);

  /* Normal air/glass/air slab, with the exact paraxial Jacobian 9/64. */
  mirrors[0] = false;
  patches[0].ior_after = patches[0].ior_opposite = 1.5f;
  patches[0].event = COHERENT_GEOMETRY_TRANSMIT;
  patches[1] = patches[0];
  patches[1].center = float3(0, 0, -1);
  patches[1].ior_before = 1.5f;
  patches[1].ior_after = patches[1].ior_opposite = 1;
  const float3 slab_detector = float3(0, 0, -2);
  solved = coherent_geometry_connect(mirror_source, slab_detector, upper.normal, patches, 2,
                                     &path);
  transported = solved && coherent_path_field_transport(mirror_source, slab_detector, upper,
                                                         &path, patches, mirrors, white, white,
                                                         0, &field);
  output[1] = float4(transported ? 1 : 0, field.physical_diagonal_rgb.x,
                     field.native_scalar_diagonal_rgb.x, path.spreading);

  /* Orthogonal incidence planes: physical two-bounce power is Rs*Rp. */
  const float inv_sqrt2 = 0.7071067811865475f;
  CoherentGeometryPath corner_path{};
  corner_path.count = 2;
  corner_path.point[0] = float3(0);
  corner_path.point[1] = float3(0, 1, 0);
  corner_path.spreading = 1;
  patches[0].event = COHERENT_GEOMETRY_REFLECT;
  patches[0].ior_before = patches[0].ior_after = 1;
  patches[0].ior_opposite = 1.5f;
  patches[0].tangent_u = float3(0, 0, 1);
  patches[0].tangent_v = float3(-inv_sqrt2, -inv_sqrt2, 0);
  patches[1].event = COHERENT_GEOMETRY_REFLECT;
  patches[1].ior_before = patches[1].ior_after = 1;
  patches[1].ior_opposite = 1.5f;
  patches[1].tangent_u = float3(1, 0, 0);
  patches[1].tangent_v = float3(0, -inv_sqrt2, -inv_sqrt2);
  transported = coherent_path_field_transport(float3(-1, 0, 0), float3(0, 1, 1), lower,
                                               &corner_path, patches, mirrors, white, white,
                                               0, &field);
  output[2] = float4(transported ? 1 : 0, field.physical_diagonal_rgb.x,
                     field.native_scalar_diagonal_rgb.x, corner_path.spreading);

  /* 60-degree glass-to-air TIR retains complex phase and unit power. */
  const float st = 0.8660254037844386f;
  const float ct = 0.5f;
  CoherentGeometryPath tir_path{};
  tir_path.count = 1;
  tir_path.point[0] = float3(0);
  tir_path.spreading = 1;
  patches[0].tangent_u = float3(1, 0, 0);
  patches[0].tangent_v = float3(0, 1, 0);
  patches[0].ior_before = patches[0].ior_after = 1.5f;
  patches[0].ior_opposite = 1;
  transported = coherent_path_field_transport(float3(-st, 0, ct), float3(st, 0, ct), lower,
                                               &tir_path, patches, mirrors, white, white, 0,
                                               &field);
  float imaginary = 0;
  for (int mode = 0; mode < COHERENT_PATH_WORLD_MODES; mode++) {
    imaginary += fabs(field.world_mode[mode].s.y) + fabs(field.world_mode[mode].p.y);
  }
  output[3] = float4(transported ? 1 : 0, field.physical_diagonal_rgb.x,
                     field.native_scalar_diagonal_rgb.x, imaginary);

  /* Direct pair: equal phase doubles each diagonal in its cross, pi negates
   * that cross, and zero coherence length erases it. */
  const float3 direct_source = float3(0);
  const float3 direct_detector = float3(0, 0, 1);
  solved = coherent_geometry_connect(direct_source, direct_detector, lower.normal, patches, 0,
                                     &path);
  transported = solved && coherent_path_field_transport(direct_source, direct_detector, lower,
                                                         &path, patches, mirrors, white, white,
                                                         0, &field);
  CoherentCompletedPathField phase_field = field;
  phase_field.source_phase_cycles = 0.5f;
  const float same = coherent_path_field_pair_cross(&field, &field, 0, 550e-9f, 1).x;
  const float opposite = coherent_path_field_pair_cross(&field, &phase_field, 0, 550e-9f, 1).x;
  const float incoherent = coherent_path_field_pair_cross(&field, &field, 0, 550e-9f, 0).x;
  output[4] = float4(transported ? 1 : 0, field.physical_diagonal_rgb.x, same, opposite);
  output[5] = float4(incoherent, 0, 0, 0);

  /* Actual saved slab face orientation. Symmetric detector z coordinates
   * must both solve and carry equal flux despite opposite patch u signs. */
  const CoherentPathDetectorFrame slab_frame{float3(0, 1, 0),
                                             float3(0, 0, 1),
                                             float3(-1, 0, 0)};
  patches[0].center = float3(0.30000001192092896f, 0, 0);
  patches[0].tangent_u = normalize(float3(-5.9604645e-8f, 0, 0.8f));
  patches[0].tangent_v = normalize(cross(normalize(float3(-1, 0, -4.37113883e-8f)),
                                           patches[0].tangent_u));
  patches[0].half_u = patches[0].half_v = 0.4f;
  patches[0].ior_before = 1;
  patches[0].ior_after = patches[0].ior_opposite = 1.5f;
  patches[0].event = COHERENT_GEOMETRY_TRANSMIT;
  patches[0].expected_incident_side = 1;
  patches[1].center = float3(0.6000000238418579f, 0, 0);
  patches[1].tangent_u = float3(0, 0, -1);
  patches[1].tangent_v = normalize(cross(normalize(float3(1, 0, -4.37113883e-8f)),
                                           patches[1].tangent_u));
  patches[1].half_u = patches[1].half_v = 0.4f;
  patches[1].ior_before = 1.5f;
  patches[1].ior_after = patches[1].ior_opposite = 1;
  patches[1].event = COHERENT_GEOMETRY_TRANSMIT;
  patches[1].expected_incident_side = -1;
  CoherentGeometryPath paired_paths[2][2]{};
  CoherentCompletedPathField paired_fields[2][2]{};
  bool paired_valid[2][2]{};
  for (int source_index = 0; source_index < 2; source_index++) {
    for (int side_index = 0; side_index < 2; side_index++) {
      const float source_y = source_index == 0 ? -5.0e-5f : 5.0e-5f;
      const float detector_z = side_index == 0 ? -0.05f : 0.05f;
      const float3 source = float3(0, source_y, 0);
      const float3 detector = float3(1, 0, detector_z);
      CoherentGeometryPath symmetric_path{};
      CoherentCompletedPathField symmetric_field{};
      const bool geometry_ok = coherent_geometry_connect(source,
                                                         detector,
                                                         slab_frame.normal,
                                                         patches,
                                                         2,
                                                         &symmetric_path);
      const bool field_ok = geometry_ok && coherent_path_field_transport(source,
                                                                         detector,
                                                                         slab_frame,
                                                                         &symmetric_path,
                                                                         patches,
                                                                         mirrors,
                                                                         float3(10),
                                                                         white,
                                                                         0,
                                                                         &symmetric_field);
      output[6 + 2 * source_index + side_index] =
          float4(field_ok ? 1 : 0,
                 symmetric_field.physical_diagonal_rgb.x,
                 symmetric_path.spreading,
                 symmetric_path.optical_length_split.x +
                     symmetric_path.optical_length_split.y);
      output[12 + 2 * source_index + side_index] =
          float4(symmetric_path.point[0].x,
                 symmetric_path.point[1].x,
                 patches[0].center.x,
                 patches[1].center.x);
      paired_valid[source_index][side_index] = field_ok;
      paired_paths[source_index][side_index] = symmetric_path;
      paired_fields[source_index][side_index] = symmetric_field;
    }
  }
  for (int side_index = 0; side_index < 2; side_index++) {
    const bool valid = paired_valid[0][side_index] && paired_valid[1][side_index];
    const float opd = valid ? coherent_geometry_optical_difference(paired_paths[0][side_index],
                                                                   paired_paths[1][side_index]) :
                              0.0f;
    const float3 same = valid ? coherent_path_field_pair_cross(&paired_fields[0][side_index],
                                                                &paired_fields[1][side_index],
                                                                opd, 550e-9f, 1) :
                                float3(0);
    CoherentCompletedPathField shifted = paired_fields[1][side_index];
    shifted.source_phase_cycles = 0.5f;
    const float3 opposite = valid ? coherent_path_field_pair_cross(&paired_fields[0][side_index],
                                                                    &shifted, opd, 550e-9f, 1) :
                                    float3(0);
    const float diagonal_sum = paired_fields[0][side_index].physical_diagonal_rgb.x +
                               paired_fields[1][side_index].physical_diagonal_rgb.x;
    output[10 + side_index] = float4(same.x, opposite.x, diagonal_sum, opd);
  }
}
