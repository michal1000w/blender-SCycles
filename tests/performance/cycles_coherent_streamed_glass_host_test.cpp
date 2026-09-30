/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

/* Standalone checks for streamed closed-convex Glass:
 * - host validation of closed, consistently outward, convex, connected meshes;
 * - strict exterior test used for coherent sources;
 * - the streamed history state machine that decides which native BDPT light
 *   prefixes the deterministic detector estimator owns;
 * - the streamed budget split between reflections and transmissions. */

#include "kernel/light/coherent_history.h"
#include "scene/coherent_convex_mesh.h"

#include <cstdio>
#include <vector>

using namespace ccl;

namespace {

using Tri = std::array<int, 3>;

std::vector<CoherentPlanarTriangle> mesh(const std::vector<CoherentPlanarPoint> &v,
                                         const std::vector<Tri> &faces)
{
  std::vector<CoherentPlanarTriangle> result;
  for (size_t i = 0; i < faces.size(); i++) {
    CoherentPlanarTriangle t{};
    for (int j = 0; j < 3; j++) {
      /* Exact float representation, as uploaded world vertices. */
      t.point[j] = {double(float(v[faces[i][j]][0])), double(float(v[faces[i][j]][1])),
                    double(float(v[faces[i][j]][2]))};
      t.vertex[j] = faces[i][j];
    }
    t.primitive = int(i);
    result.push_back(t);
  }
  return result;
}

const std::vector<CoherentPlanarPoint> cube_vertices(double s, double ox = 0.0)
{
  return {{-s + ox, -s, -s}, {s + ox, -s, -s}, {s + ox, s, -s}, {-s + ox, s, -s},
          {-s + ox, -s, s},  {s + ox, -s, s},  {s + ox, s, s},  {-s + ox, s, s}};
}

const std::vector<Tri> cube_faces = {{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7},
                                     {0, 1, 5}, {0, 5, 4}, {1, 2, 6}, {1, 6, 5},
                                     {2, 3, 7}, {2, 7, 6}, {3, 0, 4}, {3, 4, 7}};

}  // namespace

int main()
{
  int checks = 0, failures = 0;
  auto expect = [&](const bool condition, const char *what) {
    checks++;
    if (!condition) {
      failures++;
      std::printf("FAIL: %s\n", what);
    }
  };

  /* --- Host validation ---------------------------------------------------- */
  CoherentConvexHull hull;
  std::string error;
  const auto v = cube_vertices(0.005);
  expect(coherent_convex_mesh_validate(mesh(v, cube_faces), hull, error), "outward cube");
  expect(hull.planes.size() == 12, "hull planes");
  expect(coherent_convex_hull_strictly_outside(hull, {0.0, 0.0, 0.0051}), "just outside");
  expect(!coherent_convex_hull_strictly_outside(hull, {0.0, 0.0, 0.0}), "center inside");
  expect(!coherent_convex_hull_strictly_outside(hull, {0.001, 0.002, 0.0049}), "near face inside");
  expect(!coherent_convex_hull_strictly_outside(hull, {0.0, 0.0, double(0.005f)}), "on face");

  std::vector<Tri> reversed;
  for (const Tri &f : cube_faces) reversed.push_back({f[0], f[2], f[1]});
  expect(!coherent_convex_mesh_validate(mesh(v, reversed), hull, error) &&
             error.find("inward") != std::string::npos,
         "globally inward cube rejected with inward message");

  std::vector<Tri> open(cube_faces.begin(), cube_faces.end() - 1);
  expect(!coherent_convex_mesh_validate(mesh(v, open), hull, error), "open cube rejected");

  std::vector<Tri> one_flipped = cube_faces;
  one_flipped[5] = {one_flipped[5][0], one_flipped[5][2], one_flipped[5][1]};
  expect(!coherent_convex_mesh_validate(mesh(v, one_flipped), hull, error),
         "inconsistently oriented face rejected");

  auto concave = v;
  concave[6] = {0.0, 0.0, 0.0};
  expect(!coherent_convex_mesh_validate(mesh(concave, cube_faces), hull, error) &&
             error.find("convex") != std::string::npos,
         "concave vertex indentation rejected");

  /* Two separate cubes in one object: closed but not one connected volume. */
  auto two = v;
  const auto second = cube_vertices(0.005, 0.02);
  two.insert(two.end(), second.begin(), second.end());
  std::vector<Tri> two_faces = cube_faces;
  for (const Tri &f : cube_faces) two_faces.push_back({f[0] + 8, f[1] + 8, f[2] + 8});
  expect(!coherent_convex_mesh_validate(mesh(two, two_faces), hull, error),
         "disconnected volumes rejected");

  /* Rotated, non-axis-aligned cube, as world float vertices. */
  std::vector<CoherentPlanarPoint> rotated;
  const double c = std::cos(0.7), s = std::sin(0.7);
  for (const auto &p : v) {
    rotated.push_back({c * p[0] - s * p[1] + 0.1, s * p[0] + c * p[1] - 0.2,
                       0.3 * p[0] + p[2] * std::sqrt(1.0 - 0.09) * 1.0 + 0.05});
  }
  /* A shear keeps planarity and convexity. */
  expect(coherent_convex_mesh_validate(mesh(rotated, cube_faces), hull, error),
         "sheared rotated cube accepted");

  /* Tetrahedron: minimum closed convex mesh. */
  const std::vector<CoherentPlanarPoint> tet = {{0, 0, 0}, {0.01, 0, 0}, {0, 0.01, 0}, {0, 0, 0.01}};
  const std::vector<Tri> tet_faces = {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
  expect(coherent_convex_mesh_validate(mesh(tet, tet_faces), hull, error), "tetrahedron");
  expect(!coherent_convex_mesh_validate(mesh(tet, {tet_faces.begin(), tet_faces.end() - 1}),
                                        hull, error),
         "three faces rejected");

  /* Concave volumes: accepted without the convexity requirement. */
  expect(coherent_convex_mesh_validate(mesh(concave, cube_faces), hull, error, false),
         "concave accepted when convexity is not required");
  const auto cube = mesh(v, cube_faces);
  expect(std::abs(coherent_closed_mesh_winding(cube, {0.001, 0.002, 0.0}) - 1.0) < 1e-9, "winding inside 1");
  expect(std::abs(coherent_closed_mesh_winding(cube, {0.02, 0.0, 0.0})) < 1e-9, "winding outside 0");
  expect(coherent_closed_mesh_strictly_outside(cube, {0.0, 0.0, 0.0051}), "closed: just outside");
  expect(!coherent_closed_mesh_strictly_outside(cube, {0.0, 0.0, double(0.005f)}), "closed: on surface");
  expect(!coherent_closed_mesh_strictly_outside(cube, {0.0, 0.0, 0.0}), "closed: inside");
  /* Point inside the concave notch region is outside the concave volume. */
  const auto concave_mesh = mesh(concave, cube_faces);
  expect(coherent_closed_mesh_strictly_outside(concave_mesh, {0.0035, 0.0035, 0.0035}),
         "concave notch is outside");
  const auto far_cube = mesh(cube_vertices(0.005, 0.02), cube_faces);
  const auto touching_cube = mesh(cube_vertices(0.005, 0.009), cube_faces);
  expect(!coherent_triangle_sets_touch(cube, far_cube), "separated sets do not touch");
  expect(coherent_triangle_sets_touch(cube, touching_cube), "overlapping sets touch");

  /* --- Streamed history ---------------------------------------------------- */
  const int R = LABEL_SINGULAR | LABEL_REFLECT | LABEL_GLOSSY;
  const int T = LABEL_SINGULAR | LABEL_TRANSMIT;
  const int rough = LABEL_REFLECT | LABEL_GLOSSY;
  auto start = coherent_history_begin();
  auto owned = [](const CoherentPathHistory h, const uint max) {
    return coherent_history_valid(h) && !(h.metadata & COHERENT_HISTORY_STREAM_INSIDE) &&
           coherent_history_count(h) <= max;
  };

  /* Mirror behaviour is unchanged. */
  auto h = coherent_history_stream_after_interface(start, 1, 3, R, 1, 2u);
  expect(owned(h, 2) && coherent_history_count(h) == 1, "mirror R");
  expect(!coherent_history_valid(coherent_history_stream_after_interface(start, 1, 3, T, 1, 2u)),
         "mirror T invalid");
  expect(!coherent_history_valid(coherent_history_stream_after_interface(start, 0, 3, R, 1, 2u)),
         "unmarked invalid");
  expect(!coherent_history_valid(
             coherent_history_stream_after_interface(start, 1, 3, rough, 1, 2u)),
         "rough invalid");
  expect(!coherent_history_valid(coherent_history_stream_after_interface(h, 1, 3, R, 1, 1u)),
         "mirror over budget");
  for (int mirror_mode = 0; mirror_mode < 2; mirror_mode++) {
    /* Equivalence with the previous mirror-only streamed history. */
    const auto a = coherent_history_stream_after_scatter(start, mirror_mode == 1, R, 2u);
    const auto b = coherent_history_stream_after_interface(start, mirror_mode, 3, R, 1, 2u);
    expect(a.patches == b.patches && a.metadata == b.metadata, "mirror equivalence");
  }

  /* Exterior Glass reflection. */
  h = coherent_history_stream_after_interface(start, 2, 5, R, 1, 2u);
  expect(owned(h, 2) && coherent_history_count(h) == 1, "glass exterior R");
  expect(!coherent_history_valid(coherent_history_stream_after_interface(start, 2, 5, R, -1, 2u)),
         "internal R from outside-start invalid");
  h = coherent_history_stream_after_interface(h, 1, 3, R, 1, 2u);
  expect(owned(h, 2) && coherent_history_count(h) == 2, "glass R then mirror R");

  /* Entry and exit through one Glass object. */
  auto inside = coherent_history_stream_after_interface(start, 2, 5, T, 1, 2u);
  expect(coherent_history_valid(inside) && !owned(inside, 2), "inside not owned");
  auto tt = coherent_history_stream_after_interface(inside, 2, 5, T, -1, 2u);
  expect(owned(tt, 2) && coherent_history_count(tt) == 2, "TT owned");
  expect(!coherent_history_valid(coherent_history_stream_after_interface(inside, 2, 6, T, -1, 2u)),
         "exit through other object invalid");
  expect(!coherent_history_valid(coherent_history_stream_after_interface(inside, 2, 5, R, -1, 2u)),
         "internal reflection invalid");
  expect(!coherent_history_valid(coherent_history_stream_after_interface(inside, 1, 3, R, 1, 2u)),
         "mirror while inside invalid");
  expect(!coherent_history_valid(coherent_history_stream_after_interface(inside, 2, 5, T, 1, 2u)),
         "entry side while inside invalid");
  expect(!coherent_history_valid(coherent_history_stream_after_interface(start, 2, 5, T, 1, 1u)),
         "entry without exit budget invalid");
  expect(!coherent_history_valid(coherent_history_stream_after_interface(h, 2, 5, T, 1, 2u)),
         "entry after two events invalid");
  auto after_r = coherent_history_stream_after_interface(start, 1, 3, R, 1, 2u);
  expect(!coherent_history_valid(coherent_history_stream_after_interface(after_r, 2, 5, T, 1, 2u)),
         "R then entry invalid (three events)");
  expect(!coherent_history_valid(coherent_history_stream_after_interface(tt, 1, 3, R, 1, 2u)),
         "TT then R over budget");
  /* Invalid stays invalid. */
  auto dead = coherent_history_invalidate(start);
  expect(!coherent_history_valid(coherent_history_stream_after_interface(dead, 1, 3, R, 1, 2u)),
         "invalid persists");

  /* Longer chains: internal reflections while inside, within the budget. */
  auto in4 = coherent_history_stream_after_interface(start, 2, 5, T, 1, 4u);
  auto trt = coherent_history_stream_after_interface(in4, 2, 5, R, -1, 4u);
  expect(coherent_history_valid(trt) && !owned(trt, 4), "T R inside, not owned");
  trt = coherent_history_stream_after_interface(trt, 2, 5, T, -1, 4u);
  expect(owned(trt, 4) && coherent_history_count(trt) == 3, "TRT owned with four events");
  auto trrt = coherent_history_stream_after_interface(
      coherent_history_stream_after_interface(in4, 2, 5, R, -1, 4u), 2, 5, R, -1, 4u);
  trrt = coherent_history_stream_after_interface(trrt, 2, 5, T, -1, 4u);
  expect(owned(trrt, 4) && coherent_history_count(trrt) == 4, "TRRT owned");
  auto trrr = coherent_history_stream_after_interface(
      coherent_history_stream_after_interface(
          coherent_history_stream_after_interface(in4, 2, 5, R, -1, 4u), 2, 5, R, -1, 4u),
      2, 5, R, -1, 4u);
  expect(!coherent_history_valid(trrr), "no room left for the exit");
  auto ttr = coherent_history_stream_after_interface(
      coherent_history_stream_after_interface(in4, 2, 5, T, -1, 4u), 2, 5, R, 1, 4u);
  expect(owned(ttr, 4) && coherent_history_count(ttr) == 3, "exit then exterior reflection (concave)");
  auto tttt = coherent_history_stream_after_interface(
      coherent_history_stream_after_interface(
          coherent_history_stream_after_interface(in4, 2, 5, T, -1, 4u), 2, 5, T, 1, 4u),
      2, 5, T, -1, 4u);
  expect(owned(tttt, 4) && coherent_history_count(tttt) == 4, "re-entry of a concave volume");
  auto rrr = coherent_history_stream_after_interface(
      coherent_history_stream_after_interface(
          coherent_history_stream_after_interface(start, 1, 3, R, 1, 3u), 1, 4, R, 1, 3u),
      1, 3, R, 1, 3u);
  expect(owned(rrr, 3) && coherent_history_count(rrr) == 3, "three mirror reflections");
  expect(!coherent_history_valid(coherent_history_stream_after_interface(rrr, 1, 3, R, 1, 3u)),
         "fourth reflection over budget");

  /* --- Budget split -------------------------------------------------------- */
  KernelCoherentCandidate direct{};
  /* Kernel max_* bounces are stored plus one, as in Integrator::device_update. */
  const int max_bounce = 4, max_glossy = 3, max_transmission = 3;
  expect(coherent_history_candidate_within_budget(&direct, 0, 0, 0, max_bounce, max_glossy,
                                                  max_transmission, false, 3, 1024.0f, 2, 2),
         "TT within transmission 2");
  expect(!coherent_history_candidate_within_budget(&direct, 0, 0, 0, max_bounce, max_glossy, 2,
                                                   false, 3, 1024.0f, 2, 2),
         "TT rejected with transmission 1");
  expect(coherent_history_candidate_within_budget(&direct, 0, 0, 0, max_bounce, max_glossy, 1,
                                                  false, 3, 1024.0f, 2, 0),
         "RR ignores transmission budget");
  expect(!coherent_history_candidate_within_budget(&direct, 0, 0, 0, max_bounce, 2,
                                                   max_transmission, false, 3, 1024.0f, 2, 0),
         "RR rejected with glossy 1");
  expect(coherent_history_candidate_within_budget(&direct, 0, 0, 0, max_bounce, 2,
                                                  max_transmission, false, 3, 1024.0f, 2, 2),
         "TT ignores glossy budget");
  expect(!coherent_history_candidate_within_budget(&direct, 0, 0, 0, max_bounce, max_glossy,
                                                   max_transmission, false, 3, 1024.0f, 2, 3),
         "more transmissions than events rejected");
  expect(coherent_history_candidate_within_budget(&direct, 0, 0, 0, 6, 3, 3, false, 6, 1024.0f, 4, 2),
         "TRRT within max 5, glossy 2, transmission 2");
  expect(!coherent_history_candidate_within_budget(&direct, 0, 0, 0, 6, 2, 3, false, 6, 1024.0f, 4, 2),
         "TRRT rejected with glossy 1");

  std::printf("streamed glass host/history checks=%d failures=%d\n", checks, failures);
  return failures != 0;
}
