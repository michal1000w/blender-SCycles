/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#include "kernel/light/coherent_path_field.h"

#include <cmath>
#include <cstdio>

using namespace ccl;

static int failures = 0;

static void check(const bool condition, const char *label)
{
  if (!condition) {
    std::fprintf(stderr, "FAIL %s\n", label);
    failures++;
  }
}

static bool near(const double actual, const double expected, const double tolerance = 2e-5)
{
  return std::abs(actual - expected) <= tolerance;
}

static CoherentGeometryInterface plane(const float z,
                                       const float ni,
                                       const float nt,
                                       const int event)
{
  CoherentGeometryInterface patch{};
  patch.center = make_float3(0, 0, z);
  patch.tangent_u = make_float3(1, 0, 0);
  patch.tangent_v = make_float3(0, 1, 0);
  patch.half_u = patch.half_v = 10;
  patch.ior_before = ni;
  patch.ior_after = nt;
  patch.ior_opposite = nt;
  patch.event = event;
  return patch;
}

static CoherentPathDetectorFrame frame(const float3 normal)
{
  return {make_float3(1, 0, 0), make_float3(0, 1, 0), normal};
}

static double dielectric_t(const double ni, const double nt, const double ci, const bool p)
{
  const double ct = std::sqrt(1.0 - (ni / nt) * (ni / nt) * (1.0 - ci * ci));
  const double yi = p ? ni / ci : ni * ci;
  const double yt = p ? nt / ct : nt * ct;
  return 4.0 * yi * yt / ((yi + yt) * (yi + yt));
}

int main()
{
  constexpr double pi = 3.14159265358979323846;
  constexpr double base_factor = 1.0 / (4.0 * pi * pi);
  const float3 white = make_float3(1, 1, 1);
  const float3 source = make_float3(0, 0, 1);
  CoherentGeometryInterface patches[COHERENT_GEOMETRY_MAX_INTERFACES]{};
  bool mirror[COHERENT_GEOMETRY_MAX_INTERFACES]{};
  CoherentGeometryPath path{};
  CoherentCompletedPathField field{};

  /* Direct free space has total unit field power from the three equal
   * projected world dipoles and an analytic inverse-square receiver spread. */
  const float3 direct_detector = make_float3(0, 0, 2);
  check(coherent_geometry_connect(source, direct_detector, make_float3(0, 0, -1), patches, 0,
                                  &path),
        "direct geometry");
  check(coherent_path_field_transport(source,
                                      direct_detector,
                                      frame(make_float3(0, 0, -1)),
                                      &path,
                                      patches,
                                      mirror,
                                      make_float3(2, 4, 6),
                                      white,
                                      0,
                                      &field),
        "direct transport");
  check(near(field.physical_diagonal_rgb.x, 2.0 * base_factor, 1e-7),
        "direct absolute radiance");
  check(near(field.native_scalar_diagonal_rgb.z, 6.0 * base_factor, 1e-7),
        "direct native radiance");
  check(near(coherent_path_field_pair_cross(&field, &field, 0, 550e-9f, 1).x,
             2.0 * field.physical_diagonal_rgb.x, 1e-7),
        "identical coherent path cross");
  CoherentCompletedPathField phase_field = field;
  phase_field.source_phase_cycles = 0.5f;
  check(near(coherent_path_field_pair_cross(&field, &phase_field, 0, 550e-9f, 1).x,
             -2.0 * field.physical_diagonal_rgb.x, 1e-7),
        "pi source phase flips cross");
  check(coherent_path_field_pair_cross(&field, &field, 0, 550e-9f, 0).x == 0,
        "incoherent cross vanishes");

  /* Independent double Fresnel oracle: a normal slab transmits 0.96 through
   * each interface for both polarizations. Analytic connector spread is 9/64. */
  const float3 slab_detector = make_float3(0, 0, -2);
  const auto up_frame = frame(make_float3(0, 0, 1));
  patches[0] = plane(0, 1, 1.5f, COHERENT_GEOMETRY_TRANSMIT);
  patches[1] = plane(-1, 1.5f, 1, COHERENT_GEOMETRY_TRANSMIT);
  check(coherent_geometry_connect(source, slab_detector, up_frame.normal, patches, 2, &path),
        "normal slab geometry");
  check(coherent_path_field_transport(source, slab_detector, up_frame, &path, patches, mirror,
                                      white, white, 0, &field),
        "normal slab transport");
  const double normal_transmission = dielectric_t(1, 1.5, 1, false);
  const double normal_expected = base_factor * 0.140625 * normal_transmission * normal_transmission;
  check(near(field.physical_diagonal_rgb.x, normal_expected, 1e-7),
        "normal slab physical oracle");
  check(near(field.native_scalar_diagonal_rgb.x, normal_expected, 1e-7),
        "normal slab native oracle");

  /* Oblique slab: derive the interface cosines from independently measured
   * solved segment vectors, then use double s/p admittances. The physical
   * source modes remain correlated by polarization across both interfaces. */
  const float3 oblique_detector = make_float3(1.4f, 0, -2);
  check(coherent_geometry_connect(source, oblique_detector, up_frame.normal, patches, 2, &path),
        "oblique slab geometry");
  check(coherent_path_field_transport(source, oblique_detector, up_frame, &path, patches, mirror,
                                      white, white, 0, &field),
        "oblique slab transport");
  const double dx = double(path.point[0].x - source.x);
  const double ci = 1.0 / std::sqrt(1.0 + dx * dx);
  const double cs = std::sqrt(1.0 - (1.0 / 1.5) * (1.0 / 1.5) * (1.0 - ci * ci));
  const double ts = dielectric_t(1, 1.5, ci, false);
  const double tp = dielectric_t(1, 1.5, ci, true);
  const double ts_exit = dielectric_t(1.5, 1, cs, false);
  const double tp_exit = dielectric_t(1.5, 1, cs, true);
  const double physical = base_factor * path.spreading * 0.5 *
                          (ts * ts_exit + tp * tp_exit);
  const double native = base_factor * path.spreading *
                        (0.5 * (ts + tp)) * (0.5 * (ts_exit + tp_exit));
  check(near(field.physical_diagonal_rgb.x, physical, 2e-7),
        "oblique slab separate polarization power");
  check(near(field.native_scalar_diagonal_rgb.x, native, 2e-7),
        "oblique slab native Fresnel product");
  check(std::abs(physical - native) > 1e-7,
        "oblique slab distinguishes correlated from reset polarizations");

  /* Reversing the normal-incidence path exchanges n_i/n_t at each face and
   * preserves throughput and geometric spread. */
  const float3 reversed_source = make_float3(0, 0, -2);
  const float3 reversed_detector = source;
  CoherentGeometryInterface reversed[COHERENT_GEOMETRY_MAX_INTERFACES]{};
  reversed[0] = plane(-1, 1, 1.5f, COHERENT_GEOMETRY_TRANSMIT);
  reversed[1] = plane(0, 1.5f, 1, COHERENT_GEOMETRY_TRANSMIT);
  CoherentGeometryPath reversed_path{};
  CoherentCompletedPathField reversed_field{};
  check(coherent_geometry_connect(reversed_source, reversed_detector, make_float3(0, 0, -1),
                                  reversed, 2, &reversed_path),
        "reverse slab geometry");
  check(coherent_path_field_transport(reversed_source, reversed_detector,
                                      frame(make_float3(0, 0, -1)), &reversed_path, reversed,
                                      mirror, white, white, 0, &reversed_field),
        "reverse slab transport");
  check(near(reversed_field.physical_diagonal_rgb.x, normal_expected, 1e-7),
        "normal slab reciprocity");

  /* Two orthogonal incidence planes exchange the first bounce's s/p axes.
   * Therefore every mode receives one Rs and one Rp, whereas scalar path
   * tracing multiplies two separately averaged unpolarized reflectances. */
  const float inv_sqrt2 = 0.7071067811865475f;
  const float3 corner_source = make_float3(-1, 0, 0);
  const float3 corner_detector = make_float3(0, 1, 1);
  CoherentGeometryPath corner_path{};
  corner_path.count = 2;
  corner_path.point[0] = make_float3(0, 0, 0);
  corner_path.point[1] = make_float3(0, 1, 0);
  corner_path.spreading = 1;
  patches[0] = plane(0, 1, 1, COHERENT_GEOMETRY_REFLECT);
  patches[0].tangent_u = make_float3(0, 0, 1);
  patches[0].tangent_v = make_float3(-inv_sqrt2, -inv_sqrt2, 0);
  patches[0].ior_opposite = 1.5f;
  patches[1] = plane(0, 1, 1, COHERENT_GEOMETRY_REFLECT);
  patches[1].tangent_u = make_float3(1, 0, 0);
  patches[1].tangent_v = make_float3(0, -inv_sqrt2, -inv_sqrt2);
  patches[1].ior_opposite = 1.5f;
  check(coherent_path_field_transport(corner_source, corner_detector,
                                      frame(make_float3(0, 0, -1)), &corner_path, patches,
                                      mirror, white, white, 0, &field),
        "noncoplanar reflection transport");
  const double rs = 1.0 - dielectric_t(1, 1.5, inv_sqrt2, false);
  const double rp = 1.0 - dielectric_t(1, 1.5, inv_sqrt2, true);
  check(near(field.physical_diagonal_rgb.x, base_factor * rs * rp, 1e-7),
        "noncoplanar Jones frame exchange");
  check(near(field.native_scalar_diagonal_rgb.x,
             base_factor * 0.25 * (rs + rp) * (rs + rp), 1e-7),
        "noncoplanar native scalar product");

  /* Total internal reflection has unit power in each polarization but a
   * nonzero complex phase. A requested transmitted branch must be rejected. */
  constexpr float st = 0.8660254037844386f;
  constexpr float ct = 0.5f;
  const float3 tir_source = make_float3(-st, 0, ct);
  const float3 tir_detector = make_float3(st, 0, ct);
  patches[0] = plane(0, 1.5f, 1.5f, COHERENT_GEOMETRY_REFLECT);
  patches[0].ior_opposite = 1.0f;
  check(coherent_geometry_connect(tir_source, tir_detector, make_float3(0, 0, -1), patches, 1,
                                  &path),
        "TIR reflected geometry");
  check(coherent_path_field_transport(tir_source, tir_detector,
                                      frame(make_float3(0, 0, -1)), &path, patches, mirror,
                                      white, white, 0, &field),
        "TIR reflected transport");
  check(near(field.physical_diagonal_rgb.x, base_factor * path.spreading, 2e-7),
        "TIR unit physical power");
  check(near(field.native_scalar_diagonal_rgb.x, base_factor * path.spreading, 2e-7),
        "TIR unit native power");
  bool imaginary = false;
  for (int mode = 0; mode < COHERENT_PATH_WORLD_MODES; mode++) {
    imaginary |= std::abs(field.world_mode[mode].s.y) > 1e-5f ||
                 std::abs(field.world_mode[mode].p.y) > 1e-5f;
  }
  check(imaginary, "TIR retains complex phase");
  patches[0].event = COHERENT_GEOMETRY_TRANSMIT;
  patches[0].ior_after = 1.0f;
  check(!coherent_path_field_transport(tir_source, tir_detector,
                                       frame(make_float3(0, 0, -1)), &path, patches, mirror,
                                       white, white, 0, &field),
        "TIR transmitted branch rejected");

  std::printf("coherent_path_field tests: %s (%d failures)\n",
              failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}
