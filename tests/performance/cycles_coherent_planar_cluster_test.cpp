/* SPDX-License-Identifier: Apache-2.0 */
#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "scene/coherent_planar_cluster.h"
#include <iostream>
#include <stdexcept>
using namespace ccl;
int checks = 0;
void check(bool ok, const char *what)
{
  ++checks;
  if (!ok)
    throw std::runtime_error(what);
}
CoherentPlanarTriangle triangle(
    const std::vector<CoherentPlanarPoint> &p, int a, int b, int c, int id)
{
  return {{{p[a], p[b], p[c]}}, {{a, b, c}}, id};
}
std::vector<CoherentPlanarCluster> build(const std::vector<CoherentPlanarTriangle> &t)
{
  std::vector<CoherentPlanarCluster> c;
  std::string error;
  check(coherent_planar_cluster_build(t, c, error), error.c_str());
  return c;
}
bool contains_xy(const CoherentPlanarTriangle &t, double x, double y)
{
  for (int i = 0; i < 3; i++) {
    const auto &a = t.point[i], &b = t.point[(i + 1) % 3];
    if ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) < 0)
      return false;
  }
  return true;
}
int main()
{
  try {
    std::vector<CoherentPlanarPoint> p = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    auto a = triangle(p, 0, 1, 2, 37), b = triangle(p, 0, 2, 3, 35);
    auto c = build({a, b});
    check(c.size() == 1, "shared quad must merge");
    check(c[0].primitives == std::vector<int>({35, 37}), "global IDs sorted");
    // Disconnected near/exact coplanar components cannot silently merge.
    for (double height : {0.0, 1e-9, 1e-7, 1e-5}) {
      auto separate = b;
      separate.vertex = {4, 5, 6};
      separate.primitive = 39;
      for (auto &q : separate.point)
        q[2] = double(float(height));
      check(build({a, separate}).size() == 2, "disconnected nm sheets distinct");
    }
    auto reverse = b;
    std::swap(reverse.point[1], reverse.point[2]);
    std::swap(reverse.vertex[1], reverse.vertex[2]);
    check(build({a, reverse}).size() == 2, "opposite orientation distinct");
    auto folded = b;
    folded.point[2][2] = double(float(7e-9));
    check(build({a, folded}).size() == 2, "real shared-edge nanometre fold not float noise");
    folded.point[2][2] = .25;
    check(build({a, folded}).size() == 2, "folded mirror separate planes");
    // Intended rotated planar quad rounded to world float vertex representation.
    const double ca = std::cos(.43), sa = std::sin(.43), cb = std::cos(.71), sb = std::sin(.71);
    for (auto &q : p) {
      const double x = q[0], y = q[1];
      q = {double(float(1.3 + ca * x + sa * cb * y)),
           double(float(-.7 - sa * x + ca * cb * y)),
           double(float(2.1 + sb * y))};
    }
    auto rotated = std::vector<CoherentPlanarTriangle>{triangle(p, 0, 1, 2, 37),
                                                       triangle(p, 0, 2, 3, 35)};
    check(build(rotated).size() == 1, "rotated rounded quad merge");
    // An independently transformed instance shares global primitive IDs, not a plane.
    auto instance = rotated;
    for (auto &t : instance)
      for (auto &q : t.point)
        q = {double(float(q[0] + 4)), double(float(q[1] - 3)), double(float(q[2] + 2))};
    const auto ci = build(instance), cr = build(rotated);
    check(ci.size() == 1 && ci[0].primitives == cr[0].primitives,
          "instance topology/primitive identity");
    check(ci[0].origin != cr[0].origin, "instance world plane distinct");
    // Connected annulus: one enclosing rectangle, no fabricated triangles in its hole.
    p = {{-2, -2, 0},
         {2, -2, 0},
         {2, 2, 0},
         {-2, 2, 0},
         {-1, -1, 0},
         {1, -1, 0},
         {1, 1, 0},
         {-1, 1, 0}};
    std::vector<CoherentPlanarTriangle> ring;
    for (int i = 0; i < 4; i++) {
      const int j = (i + 1) % 4;
      ring.push_back(triangle(p, i, j, 4 + j, 100 + 2 * i));
      ring.push_back(triangle(p, i, 4 + j, 4 + i, 101 + 2 * i));
    }
    c = build(ring);
    check(c.size() == 1, "connected hole stays one plane");
    check(c[0].primitives.size() == 8, "hole exact triangle inventory");
    bool hit = false;
    for (const auto &t : ring)
      hit |= contains_xy(t, 0, 0);
    check(!hit, "center hole has no triangle despite enclosing rectangle");
    check(c[0].min_u <= 2 && c[0].max_u >= 2, "bounds enclose hole");
    auto bad = a;
    bad.point[2] = bad.point[1];
    std::string error;
    check(!coherent_planar_cluster_build({bad}, c, error), "degenerate rejects");
    bad = a;
    bad.point[1][0] = std::numeric_limits<double>::infinity();
    check(!coherent_planar_cluster_build({bad}, c, error), "nonfinite rejects");
    std::cout << "PASS " << checks << " checks\n";
  }
  catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << "\n";
    return 1;
  }
}
