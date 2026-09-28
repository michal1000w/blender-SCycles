/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Saved mixed T/R/T fixture v4: exact float patch frames from its Blender mesh. */
#include "kernel/light/coherent_geometry.h"

#include <cmath>
#include <cstdio>
#include <initializer_list>

using namespace ccl;

static CoherentGeometryInterface entry()
{
  CoherentGeometryInterface p{};
  p.center = make_float3(0.015000000596046448f, 0.0f, 0.0f);
  p.tangent_u = make_float3(-4.656612517806025e-08f, 0.0f, 1.0f);
  p.tangent_v = make_float3(0.0f, 1.0f, 0.0f);
  p.half_u = p.half_v = 0.020000001415610355f;
  p.ior_before = 1.0f;
  p.ior_after = 1.5f;
  p.ior_opposite = 1.5f;
  p.event = COHERENT_GEOMETRY_TRANSMIT;
  p.expected_incident_side = 1;
  return p;
}

static CoherentGeometryInterface mirror()
{
  CoherentGeometryInterface p{};
  p.center = make_float3(0.02250000089406967f, 0.019999999552965164f, 0.0f);
  p.tangent_u = make_float3(1.0f, 0.0f, 0.0f);
  p.tangent_v = make_float3(0.0f, 0.0f, 1.0f);
  p.half_u = 0.00699999975040555f;
  p.half_v = 0.014999999664723873f;
  p.ior_before = p.ior_after = p.ior_opposite = 1.5f;
  p.event = COHERENT_GEOMETRY_REFLECT;
  return p;
}

static CoherentGeometryInterface exit_face()
{
  CoherentGeometryInterface p{};
  p.center = make_float3(0.030000001192092896f, 0.0f, 0.0f);
  p.tangent_u = make_float3(0.0f, 0.0f, -1.0f);
  p.tangent_v = make_float3(0.0f, 1.0f, 0.0f);
  p.half_u = p.half_v = 0.020000001415610313f;
  p.ior_before = 1.5f;
  p.ior_after = 1.0f;
  p.ior_opposite = 1.0f;
  p.event = COHERENT_GEOMETRY_TRANSMIT;
  p.expected_incident_side = -1;
  return p;
}

struct Reference {
  double optical_length;
  double spreading;
  double mirror_x;
};

/* Independent double Snell bisection to a receiver unfolded across the mirror. */
static Reference reference(const double sy,
                           const double ry,
                           const double rz,
                           const double source_x = 0.0,
                           const double receiver_x = double(0.05000000074505806f),
                           const double mirror_y = double(0.019999999552965164f))
{
  const double entry_x = double(entry().center.x);
  const double exit_x = double(exit_face().center.x);
  const double a = entry_x - source_x;
  const double t = exit_x - entry_x;
  const double b = receiver_x - exit_x;
  const double dy = 2.0 * mirror_y - ry - sy;
  const double rho = std::hypot(dy, rz);
  double lo = 0.0, hi = 1.0 - 1.0e-15;
  for (int iteration = 0; iteration < 64; iteration++) {
    const double u = 0.5 * (lo + hi);
    const double ca = std::sqrt(1.0 - u * u);
    const double sg = u / 1.5;
    const double cg = std::sqrt(1.0 - sg * sg);
    const double displacement = (a + b) * u / ca + t * sg / cg;
    if (displacement < rho) lo = u;
    else hi = u;
  }
  const double u = 0.5 * (lo + hi);
  const double ca = std::sqrt(1.0 - u * u);
  const double sg = u / 1.5;
  const double cg = std::sqrt(1.0 - sg * sg);
  const double entry_y = sy + a * u / ca * dy / rho;
  const double exit_unfolded_y = entry_y + t * sg / cg * dy / rho;
  const double dr_du = (a + b) / (ca * ca * ca) + t / (1.5 * cg * cg * cg);
  return {(a + b) / ca + 1.5 * t / cg,
          u / (rho * ca * dr_du),
          entry_x + t * (mirror_y - entry_y) / (exit_unfolded_y - entry_y)};
}

int main()
{
  const float3 detector_normal = make_float3(-1.0f, 0.0f, 0.0f);
  int solved_count = 0;
  double largest_opl_error = 0.0;
  double largest_spreading_relative_error = 0.0;
  double largest_mirror_x_error = 0.0;
  for (const float sy : {-25.0e-6f, 25.0e-6f}) {
    for (const float ry : {-18.0e-6f, 0.0f, 18.0e-6f}) {
      for (const float rz : {-18.0e-6f, 0.0f, 18.0e-6f}) {
        const float3 source = make_float3(0.0f, sy, 0.0f);
        const float3 receiver = make_float3(0.05000000074505806f, ry, rz);
        CoherentGeometryInterface patches[COHERENT_GEOMETRY_MAX_INTERFACES] = {
            entry(), mirror(), exit_face()};
        CoherentGeometryPath path{};
        const bool solved = coherent_geometry_connect(
            source, receiver, detector_normal, patches, 3, &path);
        if (!solved) continue;
        solved_count++;
        const Reference expected = reference(double(sy), double(ry), double(rz));
        const double measured_opl = double(path.optical_length_split.x) +
                                    double(path.optical_length_split.y);
        largest_opl_error = std::fmax(largest_opl_error,
                                      std::fabs(measured_opl - expected.optical_length));
        largest_spreading_relative_error = std::fmax(
            largest_spreading_relative_error,
            std::fabs(double(path.spreading) / expected.spreading - 1.0));
        largest_mirror_x_error = std::fmax(largest_mirror_x_error,
                                           std::fabs(double(path.point[1].x) - expected.mirror_x));
      }
    }
  }
  const bool pass = solved_count == 18 && largest_opl_error < 2.0e-8 &&
                    largest_spreading_relative_error < 1.0e-3 &&
                    largest_mirror_x_error < 2.0e-6;
  std::printf("mixed_saved_fixture solved=%d/18 max_opl_error=%.9g "
              "max_spreading_rel_error=%.9g max_mirror_x_error=%.9g pass=%d\n",
              solved_count, largest_opl_error, largest_spreading_relative_error,
              largest_mirror_x_error, int(pass));
  CoherentGeometryInterface unequal[COHERENT_GEOMETRY_MAX_INTERFACES] = {
      entry(), mirror(), exit_face()};
  unequal[1].center.y = 0.01f;
  CoherentGeometryPath unequal_path{};
  const float unequal_source_x = 0.0149f;
  const float unequal_receiver_x = 0.0305f;
  const bool unequal_solved = coherent_geometry_connect(
      make_float3(unequal_source_x, -25.0e-6f, 0.0f),
      make_float3(unequal_receiver_x, 0.0f, 0.0f),
      detector_normal, unequal, 3, &unequal_path);
  const Reference unequal_expected = reference(double(-25.0e-6f), 0.0, 0.0,
                                                double(unequal_source_x),
                                                double(unequal_receiver_x),
                                                double(unequal[1].center.y));
  const double unequal_opl_error = std::fabs(
      double(unequal_path.optical_length_split.x) +
      double(unequal_path.optical_length_split.y) - unequal_expected.optical_length);
  const double unequal_spread_error = std::fabs(
      double(unequal_path.spreading) / unequal_expected.spreading - 1.0);
  const double unequal_x_error = std::fabs(
      double(unequal_path.point[1].x) - unequal_expected.mirror_x);
  const bool unequal_pass = unequal_solved &&
                            unequal_expected.mirror_x > double(entry().center.x) &&
                            unequal_expected.mirror_x < double(exit_face().center.x) &&
                            unequal_opl_error < 2.0e-8 && unequal_spread_error < 1.0e-3 &&
                            unequal_x_error < 2.0e-6;
  std::printf("unequal_gap source_entry=%.9g solved=%d mirror_x=%.9g "
              "expected_x=%.9g opl_error=%.9g spread_rel_error=%.9g pass=%d\n",
              double(entry().center.x - unequal_source_x), int(unequal_solved),
              double(unequal_path.point[1].x), unequal_expected.mirror_x,
              unequal_opl_error, unequal_spread_error, int(unequal_pass));
  return pass && unequal_pass ? 0 : 1;
}
