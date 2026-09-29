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
                                          std::string &error)
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
  for (size_t i = 0; i < triangles.size(); i++) {
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
