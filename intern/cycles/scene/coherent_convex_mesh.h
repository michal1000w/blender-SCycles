/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "scene/coherent_planar_cluster.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

CCL_NAMESPACE_BEGIN

/* Host-only validation of a declared streamed Glass mesh: one closed,
 * edge-manifold, consistently and outward oriented, convex volume. The
 * triangles must already use the physical outward winding (the caller swaps
 * two vertices for negatively scaled objects, as native Cycles does for Ng).
 * The represented float world vertices are tested; no metre tolerance is
 * used, only the float-representation bound of the planar clustering. */
struct CoherentConvexHull {
  std::vector<coherent_planar_detail::Plane> planes;
};

namespace coherent_convex_detail {
/* Signed plane test with the same float-representation error bound used by
 * planar clustering. Returns +1 strictly outside, -1 strictly inside, 0 on the
 * plane within the bound. */
inline int plane_side(const coherent_planar_detail::Plane &plane, const CoherentPlanarPoint &point)
{
  using namespace coherent_planar_detail;
  const auto delta = sub(point, plane.origin);
  double error = 0.0, arithmetic_scale = 0.0;
  for (int j = 0; j < 3; j++) {
    const double position_error = 0.5 * (ulp(point[j]) + ulp(plane.origin[j]));
    error += std::abs(plane.cross_raw[j]) * position_error +
             std::abs(delta[j]) * plane.cross_error[j] + plane.cross_error[j] * position_error;
    arithmetic_scale += std::abs(plane.cross_raw[j] * delta[j]);
  }
  error += 64.0 * std::numeric_limits<double>::epsilon() * arithmetic_scale;
  const double value = dot(delta, plane.cross_raw);
  return value > error ? 1 : (value < -error ? -1 : 0);
}
}  // namespace coherent_convex_detail

inline bool coherent_convex_mesh_validate(const std::vector<CoherentPlanarTriangle> &triangles,
                                          CoherentConvexHull &hull,
                                          std::string &error,
                                          const bool require_convex = true)
{
  using namespace coherent_planar_detail;
  hull.planes.clear();
  if (triangles.size() < 4) {
    error = "Streamed Glass requires a closed convex triangle mesh (at least four faces)";
    return false;
  }
  std::vector<Plane> planes(triangles.size());
  struct Edge {
    int first = -1, second = -1;
    int a = 0, b = 0;
  };
  std::map<std::pair<int, int>, Edge> edges;
  for (size_t i = 0; i < triangles.size(); i++) {
    if (!plane(triangles[i], planes[i])) {
      error = "Streamed Glass has a degenerate or non-finite face";
      return false;
    }
    for (int j = 0; j < 3; j++) {
      const int a = triangles[i].vertex[j], b = triangles[i].vertex[(j + 1) % 3];
      if (a == b) {
        error = "Streamed Glass has a degenerate face";
        return false;
      }
      Edge &edge = edges[{std::min(a, b), std::max(a, b)}];
      if (edge.first < 0) {
        edge.first = int(i);
        edge.a = a;
        edge.b = b;
      }
      else if (edge.second >= 0 || edge.a != b || edge.b != a) {
        error =
            "Streamed Glass requires every edge to join exactly two consistently oriented faces";
        return false;
      }
      else {
        edge.second = int(i);
      }
    }
  }
  std::vector<std::vector<int>> adjacent(triangles.size());
  for (const auto &entry : edges) {
    if (entry.second.second < 0) {
      error = "Streamed Glass mesh is open; it must bound a closed volume";
      return false;
    }
    adjacent[entry.second.first].push_back(entry.second.second);
    adjacent[entry.second.second].push_back(entry.second.first);
  }
  std::vector<bool> visited(triangles.size(), false);
  std::vector<int> stack{0};
  visited[0] = true;
  for (size_t i = 0; i < stack.size(); i++) {
    for (const int next : adjacent[stack[i]]) {
      if (!visited[next]) {
        visited[next] = true;
        stack.push_back(next);
      }
    }
  }
  if (stack.size() != triangles.size()) {
    error = "Streamed Glass requires one connected closed volume per object";
    return false;
  }
  /* Signed volume relative to a vertex; positive for outward winding. */
  const CoherentPlanarPoint origin = triangles[0].point[0];
  double volume = 0.0, volume_scale = 0.0;
  for (const CoherentPlanarTriangle &t : triangles) {
    const auto a = sub(t.point[0], origin), b = sub(t.point[1], origin),
               c = sub(t.point[2], origin);
    const double v = dot(a, cross(b, c));
    volume += v;
    volume_scale += std::abs(v);
  }
  if (!(volume > 1.0e-9 * volume_scale)) {
    error = volume < 0.0 ? "Streamed Glass faces point inward; recalculate outside normals" :
                           "Streamed Glass has no enclosed volume";
    return false;
  }
  /* Convexity: every represented vertex lies on or behind every face plane. */
  for (size_t i = 0; i < triangles.size() && require_convex; i++) {
    for (const CoherentPlanarTriangle &t : triangles) {
      for (const CoherentPlanarPoint &p : t.point) {
        if (coherent_convex_detail::plane_side(planes[i], p) > 0) {
          error = "Streamed Glass requires a convex volume";
          return false;
        }
      }
    }
  }
  hull.planes = std::move(planes);
  return true;
}

namespace coherent_convex_detail {
inline double point_triangle_distance(const CoherentPlanarPoint &p, const CoherentPlanarTriangle &t)
{
  using namespace coherent_planar_detail;
  /* Ericson, closest point on triangle. */
  const auto &a = t.point[0], &b = t.point[1], &c = t.point[2];
  const auto ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  const double d1 = dot(ab, ap), d2 = dot(ac, ap);
  auto at = [&](const CoherentPlanarPoint &q) { return length(sub(p, q)); };
  if (d1 <= 0 && d2 <= 0) return at(a);
  const auto bp = sub(p, b);
  const double d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return at(b);
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const double v = d1 / (d1 - d3);
    return at({a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]});
  }
  const auto cp = sub(p, c);
  const double d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return at(c);
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const double w = d2 / (d2 - d6);
    return at({a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]});
  }
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return at({b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2])});
  }
  const double denom = 1.0 / (va + vb + vc);
  const double v = vb * denom, w = vc * denom;
  return at({a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w});
}

/* Segment [p,q] meets triangle (inclusive of boundaries). */
inline bool segment_hits_triangle(const CoherentPlanarPoint &p, const CoherentPlanarPoint &q,
                                  const CoherentPlanarTriangle &t)
{
  using namespace coherent_planar_detail;
  const auto e1 = sub(t.point[1], t.point[0]), e2 = sub(t.point[2], t.point[0]);
  const auto d = sub(q, p);
  const auto h = cross(d, e2);
  const double a = dot(e1, h);
  const double scale = length(e1) * length(e2) * length(d);
  if (std::abs(a) <= 1e-14 * scale) return false; /* Parallel; coplanar contact is caught elsewhere. */
  const double f = 1.0 / a;
  const auto s = sub(p, t.point[0]);
  const double u = f * dot(s, h);
  if (u < 0.0 || u > 1.0) return false;
  const auto qv = cross(s, e1);
  const double v = f * dot(d, qv);
  if (v < 0.0 || u + v > 1.0) return false;
  const double tt = f * dot(e2, qv);
  return tt >= 0.0 && tt <= 1.0;
}
}  // namespace coherent_convex_detail

/* Generalized winding number of a closed triangle mesh (Van Oosterom-Strackee). */
inline double coherent_closed_mesh_winding(const std::vector<CoherentPlanarTriangle> &triangles,
                                           const CoherentPlanarPoint &p)
{
  using namespace coherent_planar_detail;
  double total = 0.0;
  for (const auto &t : triangles) {
    const auto a = sub(t.point[0], p), b = sub(t.point[1], p), c = sub(t.point[2], p);
    const double la = length(a), lb = length(b), lc = length(c);
    const double numerator = dot(a, cross(b, c));
    const double denominator = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
    total += 2.0 * std::atan2(numerator, denominator);
  }
  return total / (4.0 * 3.14159265358979323846);
}

/* Strictly outside a closed outward mesh: winding ~0 and not on its surface. */
inline bool coherent_closed_mesh_strictly_outside(const std::vector<CoherentPlanarTriangle> &triangles,
                                                  const CoherentPlanarPoint &p)
{
  double scale = 0.0;
  for (const auto &t : triangles)
    for (const auto &q : t.point) scale = std::max({scale, std::abs(q[0]), std::abs(q[1]), std::abs(q[2])});
  scale = std::max({scale, std::abs(p[0]), std::abs(p[1]), std::abs(p[2])});
  for (const auto &t : triangles) {
    if (coherent_convex_detail::point_triangle_distance(p, t) <= 1e-9 * std::max(scale, 1e-30)) return false;
  }
  return std::abs(coherent_closed_mesh_winding(triangles, p)) < 0.5;
}

/* Any contact between two triangle sets: an edge of one meets a triangle of the other. */
inline bool coherent_triangle_sets_touch(const std::vector<CoherentPlanarTriangle> &a,
                                         const std::vector<CoherentPlanarTriangle> &b)
{
  auto edges_hit = [](const std::vector<CoherentPlanarTriangle> &x,
                      const std::vector<CoherentPlanarTriangle> &y) {
    for (const auto &s : x) {
      for (int j = 0; j < 3; j++) {
        for (const auto &t : y) {
          if (coherent_convex_detail::segment_hits_triangle(s.point[j], s.point[(j + 1) % 3], t)) return true;
        }
      }
    }
    return false;
  };
  return edges_hit(a, b) || edges_hit(b, a);
}

/* True when the point is strictly outside the validated convex volume. */
inline bool coherent_convex_hull_strictly_outside(const CoherentConvexHull &hull,
                                                  const CoherentPlanarPoint &point)
{
  for (const auto &plane : hull.planes) {
    if (coherent_convex_detail::plane_side(plane, point) > 0) {
      return true;
    }
  }
  return false;
}

CCL_NAMESPACE_END
