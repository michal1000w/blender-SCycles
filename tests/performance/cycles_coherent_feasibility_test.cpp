/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#include "kernel/light/coherent_feasibility.h"

#include <cstdio>
#include <initializer_list>

using namespace ccl;

static int failures = 0;
static int checks = 0;

static void check(const bool condition, const char *name)
{
  checks++;
  if (!condition) {
    std::fprintf(stderr, "FAIL %s\n", name);
    failures++;
  }
}

static CoherentGeometryInterface patch(const float3 center,
                                       const float3 u,
                                       const float3 v,
                                       const float hu,
                                       const float hv,
                                       const int event,
                                       const int side)
{
  CoherentGeometryInterface result{};
  result.center = center;
  result.tangent_u = u;
  result.tangent_v = v;
  result.half_u = hu;
  result.half_v = hv;
  result.ior_before = 1.0f;
  result.ior_after = 1.0f;
  result.ior_opposite = 1.5f;
  result.event = event;
  result.expected_incident_side = side;
  return result;
}

int main()
{
  CoherentGeometryInterface sequence[COHERENT_GEOMETRY_MAX_INTERFACES]{};
  const float3 origin = make_float3(0.0f, 0.0f, 0.0f);
  const float3 above = make_float3(0.0f, 0.0f, 1.0f);
  const float3 below = make_float3(0.0f, 0.0f, -1.0f);
  sequence[0] = patch(origin,
                      make_float3(1, 0, 0),
                      make_float3(0, 1, 0),
                      2.0f,
                      2.0f,
                      COHERENT_GEOMETRY_REFLECT,
                      0);
  check(coherent_geometry_candidate_side_feasible(above, above, sequence, 1),
        "mirror same-side route");
  check(!coherent_geometry_candidate_side_feasible(above, below, sequence, 1),
        "mirror opposite-side route");
  sequence[0].event = COHERENT_GEOMETRY_TRANSMIT;
  sequence[0].expected_incident_side = 1;
  check(coherent_geometry_candidate_side_feasible(above, below, sequence, 1),
        "glass expected transmission side");
  check(!coherent_geometry_candidate_side_feasible(above, above, sequence, 1),
        "glass wrong transmission side");
  check(!coherent_geometry_candidate_side_feasible(below, above, sequence, 1),
        "glass wrong incident side");
  /* A finite tilted preceding rectangle spans z=0 even though its center is
   * below the next interface. Center-only rejection would be unsound. */
  sequence[0] = patch(make_float3(0, 0, -0.1f),
                      normalize(make_float3(1, 0, 1)),
                      make_float3(0, 1, 0),
                      1.0f,
                      1.0f,
                      COHERENT_GEOMETRY_REFLECT,
                      0);
  sequence[1] = patch(origin,
                      make_float3(1, 0, 0),
                      make_float3(0, 1, 0),
                      2.0f,
                      2.0f,
                      COHERENT_GEOMETRY_REFLECT,
                      0);
  check(coherent_geometry_candidate_side_feasible(below, above, sequence, 2),
        "finite rectangle spanning next plane retained");
  /* With a huge, nearly parallel rectangle, float angular error multiplied
   * by its extent can dominate the signed center distance. Keep this grazing
   * bound for the exact solver rather than incorrectly rejecting it. */
  sequence[0] = patch(make_float3(0, 0, -0.05f),
                      normalize(make_float3(1, 0, 1e-7f)),
                      make_float3(0, 1, 0),
                      100000.0f,
                      1.0f,
                      COHERENT_GEOMETRY_TRANSMIT,
                      -1);
  check(coherent_geometry_candidate_side_feasible(below, above, sequence, 2),
        "large near-parallel rectangle retained by extent-aware margin");

  /* Saved mixed T/R/T fixture: entry normal -X, internal mirror normal -Y,
   * exit normal +X. Both launch offsets and ROI corners have valid solved
   * routes, so a pre-solve test must keep them all. */
  const float3 source_y[2] = {make_float3(0, -25e-6f, 0),
                              make_float3(0, 25e-6f, 0)};
  sequence[0] = patch(make_float3(0.015000000596f, 0, 0),
                      make_float3(0, 0, 1),
                      make_float3(0, 1, 0),
                      0.02f,
                      0.02f,
                      COHERENT_GEOMETRY_TRANSMIT,
                      1);
  sequence[1] = patch(make_float3(0.022500000894f, 0.019999999553f, 0),
                      make_float3(1, 0, 0),
                      make_float3(0, 0, 1),
                      0.007f,
                      0.015f,
                      COHERENT_GEOMETRY_REFLECT,
                      0);
  sequence[2] = patch(make_float3(0.030000001192f, 0, 0),
                      make_float3(0, 1, 0),
                      make_float3(0, 0, 1),
                      0.02f,
                      0.02f,
                      COHERENT_GEOMETRY_TRANSMIT,
                      -1);
  sequence[0].ior_after = 1.5f;
  sequence[1].ior_before = 1.5f;
  sequence[1].ior_after = 1.5f;
  sequence[2].ior_before = 1.5f;
  for (const float3 source : source_y) {
    for (const float y : {-20e-6f, 0.0f, 20e-6f}) {
      for (const float z : {-20e-6f, 0.0f, 20e-6f}) {
        const float3 receiver = make_float3(0.050000000745f, y, z);
        const float3 detector_normal = make_float3(-1, 0, 0);
        check(coherent_geometry_candidate_side_feasible(source, receiver, sequence, 3),
              "mixed fixture valid T/R/T retained");
        CoherentGeometryPath mixed_path{};
        check(coherent_geometry_connect(
                  source, receiver, detector_normal, sequence, 3, &mixed_path),
              "mixed fixture T/R/T actually solved");
        const CoherentGeometryInterface direct[COHERENT_GEOMETRY_MAX_INTERFACES] = {
            sequence[0], sequence[2]};
        check(coherent_geometry_candidate_side_feasible(source, receiver, direct, 2),
              "mixed fixture valid direct T/T retained");
        CoherentGeometryPath direct_path{};
        check(coherent_geometry_connect(
                  source, receiver, detector_normal, direct, 2, &direct_path),
              "mixed fixture direct T/T actually solved");
      }
    }
  }
  std::printf("checks=%d failures=%d\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
