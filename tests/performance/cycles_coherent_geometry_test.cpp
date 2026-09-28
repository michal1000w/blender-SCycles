/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/light/coherent_geometry.h"

#include <cmath>
#include <cstdio>
#include <initializer_list>

using namespace ccl;

static int failures = 0;

static void check(const bool condition, const char *name)
{
  if (!condition) {
    std::fprintf(stderr, "FAIL %s\n", name);
    failures++;
  }
}

static CoherentGeometryInterface plane(const float z,
                                       const float n_before,
                                       const float n_after,
                                       const int event,
                                       const float half_size = 10.0f)
{
  CoherentGeometryInterface p{};
  p.center = make_float3(0.0f, 0.0f, z);
  p.tangent_u = make_float3(1.0f, 0.0f, 0.0f);
  p.tangent_v = make_float3(0.0f, 1.0f, 0.0f);
  p.half_u = p.half_v = half_size;
  p.ior_before = n_before;
  p.ior_after = n_after;
  p.event = event;
  return p;
}

static float finite_difference_spreading(
    const float3 source,
    const float3 receiver,
    const CoherentGeometryInterface patches[COHERENT_GEOMETRY_MAX_INTERFACES],
    const int count)
{
  constexpr float h = 0.002f;
  const float3 normal = make_float3(0.0f, 0.0f, 1.0f);
  CoherentGeometryPath px_plus{}, px_minus{}, py_plus{}, py_minus{};
  if (!coherent_geometry_connect(source, receiver + make_float3(h, 0, 0), normal, patches, count,
                                 &px_plus) ||
      !coherent_geometry_connect(source, receiver - make_float3(h, 0, 0), normal, patches, count,
                                 &px_minus) ||
      !coherent_geometry_connect(source, receiver + make_float3(0, h, 0), normal, patches, count,
                                 &py_plus) ||
      !coherent_geometry_connect(source, receiver - make_float3(0, h, 0), normal, patches, count,
                                 &py_minus))
  {
    return -1.0f;
  }
  const float3 dx = (px_plus.source_direction - px_minus.source_direction) / (2.0f * h);
  const float3 dy = (py_plus.source_direction - py_minus.source_direction) / (2.0f * h);
  return len(cross(dx, dy));
}

int main()
{
  CoherentGeometryInterface patches[COHERENT_GEOMETRY_MAX_INTERFACES]{};
  const float3 up = make_float3(0, 0, 1);
  CoherentGeometryPath path{};

  const float3 mirror_source = make_float3(-0.35f, 0.17f, 1.0f);
  const float3 mirror_receiver = make_float3(0.85f, -0.12f, 2.0f);
  patches[0] = plane(0, 1, 1, COHERENT_GEOMETRY_REFLECT);
  check(coherent_geometry_connect(mirror_source, mirror_receiver, up, patches, 1, &path),
        "mirror solve");
  const float3 virtual_source = make_float3(mirror_source.x, mirror_source.y, -mirror_source.z);
  const float3 virtual_segment = mirror_receiver - virtual_source;
  const float expected_mirror_length = len(virtual_segment);
  const float expected_mirror_spreading = fabsf(dot(up, virtual_segment)) /
                                          (expected_mirror_length * expected_mirror_length *
                                           expected_mirror_length);
  check(fabsf(path.optical_length - expected_mirror_length) < 1.0e-6f,
        "mirror image optical length");
  check(fabsf(path.spreading - expected_mirror_spreading) < 2.0e-6f,
        "mirror image analytic spreading");
  check(fabsf(path.spreading - finite_difference_spreading(
                                    mirror_source, mirror_receiver, patches, 1)) < 3.0e-5f,
        "mirror finite difference spreading");

  const float3 two_source = make_float3(-0.2f, 0.1f, 0.2f);
  const float3 two_receiver = make_float3(0.7f, -0.13f, 0.8f);
  patches[0] = plane(1, 1, 1, COHERENT_GEOMETRY_REFLECT);
  patches[1] = plane(0, 1, 1, COHERENT_GEOMETRY_REFLECT);
  check(coherent_geometry_connect(two_source, two_receiver, up, patches, 2, &path),
        "two distinct mirror objects solve");
  check(fabsf(path.spreading -
              finite_difference_spreading(two_source, two_receiver, patches, 2)) < 4.0e-5f,
        "two mirror Jacobian");

  const float3 slab_source = make_float3(0.0f, 0.0f, 1.0f);
  const float3 slab_receiver = make_float3(0.0f, 0.0f, -2.0f);
  patches[0] = plane(0, 1, 1.5f, COHERENT_GEOMETRY_TRANSMIT);
  patches[1] = plane(-1, 1.5f, 1, COHERENT_GEOMETRY_TRANSMIT);
  check(coherent_geometry_connect(slab_source, slab_receiver, up, patches, 2, &path),
        "two interface refractive slab solve");
  check(fabsf(path.optical_length - 3.5f) < 1.0e-6f, "slab optical length");
  check(fabsf(path.spreading - 0.140625f) < 2.0e-6f,
        "slab paraxial analytic spreading");
  check(fabsf(path.spreading -
              finite_difference_spreading(slab_source, slab_receiver, patches, 2)) < 3.0e-5f,
        "slab finite difference spreading");

  const float3 off_axis_receiver = make_float3(0.65f, 0.12f, -2.0f);
  check(coherent_geometry_connect(slab_source, off_axis_receiver, up, patches, 2, &path),
        "off axis slab solve");
  check(fabsf(path.spreading -
              finite_difference_spreading(slab_source, off_axis_receiver, patches, 2)) < 5.0e-5f,
        "off axis slab Jacobian");

  /* Optical paths differ by less than one visible wavelength while each path
   * exceeds two metres. Plain float subtraction loses a material phase. */
  const float3 thin_receiver = make_float3(0, 0, -1);
  CoherentGeometryPath direct_path{}, thin_path{};
  check(coherent_geometry_connect(slab_source, thin_receiver, up, patches, 0, &direct_path),
        "thin slab direct reference");
  const float thin_thickness = 1.0e-6f;
  patches[0] = plane(0, 1, 1.5f, COHERENT_GEOMETRY_TRANSMIT);
  patches[1] = plane(-thin_thickness, 1.5f, 1, COHERENT_GEOMETRY_TRANSMIT);
  check(coherent_geometry_connect(slab_source, thin_receiver, up, patches, 2, &thin_path),
        "thin refractive slab solve");
  const double opd_reference = 0.5 * double(thin_thickness);
  const double opd_measured = coherent_geometry_optical_difference(thin_path, direct_path);
  check(std::abs(opd_measured - opd_reference) < 1.0e-12,
        "compensated subwavelength refractive OPD");
  check(std::abs(double(thin_path.optical_length - direct_path.optical_length) - opd_reference) >
            1.0e-8,
        "fixture exposes naive float OPL cancellation");

  const float3 shallow_source = make_float3(-1, 0, 0.001f);
  const float3 shallow_receiver = make_float3(1, 0, 0.001f);
  const float3 side_receiver_normal = make_float3(1, 0, 0);
  check(coherent_geometry_connect(shallow_source, shallow_receiver, side_receiver_normal, patches, 0,
                                  &direct_path),
        "shallow direct reference");
  patches[0] = plane(0, 1, 1, COHERENT_GEOMETRY_REFLECT);
  check(coherent_geometry_connect(shallow_source, shallow_receiver, side_receiver_normal, patches, 1,
                                  &thin_path),
        "shallow mirror solve");
  const double mirror_opd_reference =
      std::sqrt(4.0 + 4.0 * double(shallow_source.z) * shallow_source.z) - 2.0;
  check(std::abs(double(coherent_geometry_optical_difference(thin_path, direct_path)) -
                 mirror_opd_reference) < 2.0e-11,
        "compensated subwavelength mirror OPD");

  /* A tilted patch translated far from the origin stresses affine hit-point
   * rounding. Compare two source paths using float inputs interpreted exactly
   * by an independent double virtual-source construction. */
  const float angle = 0.37f;
  const float3 tu = make_float3(std::cos(angle), 0, -std::sin(angle));
  const float3 tv = make_float3(0, 1, 0);
  const float3 normal = normalize(cross(tu, tv));
  const float3 origin = make_float3(1000, -700, 350);
  patches[0] = plane(0, 1, 1, COHERENT_GEOMETRY_REFLECT);
  patches[0].center = origin;
  patches[0].tangent_u = tu;
  patches[0].tangent_v = tv;
  const float3 tilted_a = origin + normal * 1.0f - tu * 0.3f;
  const float3 tilted_b = tilted_a + tu * 0.001f;
  const float3 tilted_receiver = origin + normal * 2.0f + tu * 0.7f + tv * 0.1f;
  CoherentGeometryPath tilted_path_a{}, tilted_path_b{};
  check(coherent_geometry_connect(tilted_a, tilted_receiver, normal, patches, 1,
                                  &tilted_path_a),
        "translated tilted mirror source A");
  check(coherent_geometry_connect(tilted_b, tilted_receiver, normal, patches, 1,
                                  &tilted_path_b),
        "translated tilted mirror source B");
  const double nx = double(tu.y) * tv.z - double(tu.z) * tv.y;
  const double ny = double(tu.z) * tv.x - double(tu.x) * tv.z;
  const double nz = double(tu.x) * tv.y - double(tu.y) * tv.x;
  const double inv_n = 1.0 / std::sqrt(nx * nx + ny * ny + nz * nz);
  const double n[3] = {nx * inv_n, ny * inv_n, nz * inv_n};
  const double c[3] = {origin.x, origin.y, origin.z};
  const double r[3] = {tilted_receiver.x, tilted_receiver.y, tilted_receiver.z};
  const float3 sources[2] = {tilted_a, tilted_b};
  double expected_range[2];
  for (int i = 0; i < 2; i++) {
    const double s[3] = {sources[i].x, sources[i].y, sources[i].z};
    const double height = (s[0] - c[0]) * n[0] + (s[1] - c[1]) * n[1] +
                          (s[2] - c[2]) * n[2];
    const double dx = r[0] - (s[0] - 2.0 * height * n[0]);
    const double dy = r[1] - (s[1] - 2.0 * height * n[1]);
    const double dz = r[2] - (s[2] - 2.0 * height * n[2]);
    expected_range[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
  }
  check(std::abs(double(coherent_geometry_optical_difference(tilted_path_a, tilted_path_b)) -
                 (expected_range[0] - expected_range[1])) < 2.0e-8,
        "translated tilted mirror pair OPD");

  for (const float scale : {0.01f, 1.0f, 100.0f}) {
    patches[0] = plane(0, 1, 1, COHERENT_GEOMETRY_REFLECT, 10.0f * scale);
    const float3 scaled_source = scale * mirror_source;
    const float3 scaled_receiver = scale * mirror_receiver;
    check(coherent_geometry_connect(scaled_source, scaled_receiver, up, patches, 1, &path),
          "scale mirror solve");
    const float expected = expected_mirror_spreading / (scale * scale);
    check(std::abs(path.spreading / expected - 1.0f) < 2.0e-4f,
          "scale mirror spreading");
  }

  patches[0] = plane(0, 1, 1, COHERENT_GEOMETRY_REFLECT, 0.01f);
  check(!coherent_geometry_connect(mirror_source, mirror_receiver, up, patches, 1, &path),
        "finite patch rejects external stationary point");
  patches[0] = plane(0, 1, 1, COHERENT_GEOMETRY_REFLECT);
  check(!coherent_geometry_connect(mirror_source, make_float3(0.85f, -0.12f, -2.0f), up,
                                    patches, 1, &path),
        "reflection rejects opposite side");
  patches[0] = plane(0, 1, 1.5f, COHERENT_GEOMETRY_TRANSMIT);
  check(!coherent_geometry_connect(mirror_source, mirror_receiver, up, patches, 1, &path),
        "transmission rejects same side");
  patches[0].expected_incident_side = -1;
  check(!coherent_geometry_connect(make_float3(0, 0, 1), make_float3(0, 0, -1),
                                    up, patches, 1, &path),
        "dielectric rejects incorrect incident medium side");
  patches[0].expected_incident_side = 1;
  check(coherent_geometry_connect(make_float3(0, 0, 1), make_float3(0, 0, -1),
                                   up, patches, 1, &path),
        "dielectric accepts declared incident side");
  patches[0] = plane(0, 1.5f, 1, COHERENT_GEOMETRY_TRANSMIT, 0.05f);
  check(!coherent_geometry_connect(make_float3(-0.8f, 0, 1),
                                    make_float3(0.8f, 0, -1), up, patches, 1, &path),
        "near critical finite aperture rejects path");
  patches[0] = plane(0, 1, 1, COHERENT_GEOMETRY_REFLECT);
  patches[1] = plane(0, 1, 1, COHERENT_GEOMETRY_REFLECT);
  check(!coherent_geometry_connect(make_float3(0, 0, 1), make_float3(0, 0, 1), up,
                                    patches, 2, &path),
        "repeated plane degeneracy rejected");
  check(!coherent_geometry_connect(make_float3(0, 0, 0), make_float3(0, 0, 0), up,
                                    patches, 0, &path),
        "zero distance rejected");

  std::printf("coherent_geometry failures=%d\n", failures);
  return failures ? 1 : 0;
}
