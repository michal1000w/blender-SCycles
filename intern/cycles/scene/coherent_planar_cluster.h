/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

CCL_NAMESPACE_BEGIN

/* Host-only geometry grouping. Coordinates are exact double representations of
 * the uploaded world-space float vertices; vertex IDs retain mesh topology. */
using CoherentPlanarPoint = std::array<double, 3>;
struct CoherentPlanarTriangle {
  std::array<CoherentPlanarPoint, 3> point;
  std::array<int, 3> vertex;
  int primitive;
};
struct CoherentPlanarCluster {
  CoherentPlanarPoint origin, normal, tangent_u, tangent_v;
  double min_u, max_u, min_v, max_v;
  std::vector<int> primitives;
};

namespace coherent_planar_detail {
inline CoherentPlanarPoint sub(const CoherentPlanarPoint &a, const CoherentPlanarPoint &b)
{
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
inline double dot(const CoherentPlanarPoint &a, const CoherentPlanarPoint &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline CoherentPlanarPoint cross(const CoherentPlanarPoint &a, const CoherentPlanarPoint &b)
{
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double length(const CoherentPlanarPoint &p)
{
  return std::sqrt(dot(p, p));
}
inline CoherentPlanarPoint scale(const CoherentPlanarPoint &p, const double f)
{
  return {p[0] * f, p[1] * f, p[2] * f};
}
inline double ulp(const double x)
{
  const float f = float(x);
  return std::max(double(std::nextafter(f, std::numeric_limits<float>::infinity())) - f,
                  double(f) - std::nextafter(f, -std::numeric_limits<float>::infinity()));
}
struct Plane {
  CoherentPlanarPoint origin, normal, u, v;
  CoherentPlanarPoint cross_raw, cross_error;
};
inline bool plane(const CoherentPlanarTriangle &triangle, Plane &p)
{
  for (const auto &point : triangle.point) {
    for (const double x : point)
      if (!std::isfinite(x) || !std::isfinite(float(x)))
        return false;
  }
  const auto a = sub(triangle.point[1], triangle.point[0]);
  const auto b = sub(triangle.point[2], triangle.point[0]);
  const auto c = cross(a, b);
  const double area = length(c), la = length(a);
  if (!(area > 0.0) || !(la > 0.0))
    return false;
  p.origin = triangle.point[0];
  p.normal = scale(c, 1.0 / area);
  p.u = scale(a, 1.0 / la);
  p.v = cross(p.normal, p.u);
  CoherentPlanarPoint da{}, db{};
  for (int j = 0; j < 3; j++) {
    da[j] = 0.5 * (ulp(triangle.point[1][j]) + ulp(triangle.point[0][j]));
    db[j] = 0.5 * (ulp(triangle.point[2][j]) + ulp(triangle.point[0][j]));
  }
  p.cross_raw = c;
  for (int i = 0; i < 3; i++) {
    const int j = (i + 1) % 3, k = (i + 2) % 3;
    p.cross_error[i] = std::abs(a[j]) * db[k] + std::abs(b[k]) * da[j] + da[j] * db[k] +
                       std::abs(a[k]) * db[j] + std::abs(b[j]) * da[k] + da[k] * db[j];
  }
  return true;
}

inline bool coplanar(const Plane &reference,
                     const Plane &candidate,
                     const CoherentPlanarTriangle &triangle)
{
  /* No fixed metre/angular tolerance and no neighbor-to-neighbor drift. The
   * bound describes float vertex representation, not a smoothing angle. */
  if (dot(reference.normal, candidate.normal) <= 0.0)
    return false;
  for (const auto &point : triangle.point) {
    const auto delta = sub(point, reference.origin);
    double error = 0.0, arithmetic_scale = 0.0;
    for (int j = 0; j < 3; j++) {
      const double position_error = 0.5 * (ulp(point[j]) + ulp(reference.origin[j]));
      error += std::abs(reference.cross_raw[j]) * position_error +
               std::abs(delta[j]) * reference.cross_error[j] +
               reference.cross_error[j] * position_error;
      arithmetic_scale += std::abs(reference.cross_raw[j] * delta[j]);
    }
    error += 64.0 * std::numeric_limits<double>::epsilon() * arithmetic_scale;
    if (std::abs(dot(delta, reference.cross_raw)) > error)
      return false;
  }
  return true;
}
}  // namespace coherent_planar_detail

inline bool coherent_planar_cluster_build(const std::vector<CoherentPlanarTriangle> &triangles,
                                          std::vector<CoherentPlanarCluster> &clusters,
                                          std::string &error)
{
  using namespace coherent_planar_detail;
  clusters.clear();
  error.clear();
  std::vector<Plane> planes(triangles.size());
  std::map<std::pair<int, int>, std::vector<size_t>> edges;
  for (size_t i = 0; i < triangles.size(); i++) {
    if (!plane(triangles[i], planes[i])) {
      error = "Coherent interface contains invalid or degenerate world-space triangles";
      return false;
    }
    for (int j = 0; j < 3; j++) {
      const int a = triangles[i].vertex[j], b = triangles[i].vertex[(j + 1) % 3];
      edges[{std::min(a, b), std::max(a, b)}].push_back(i);
    }
  }
  std::vector<std::vector<size_t>> neighbors(triangles.size());
  for (const auto &entry : edges) {
    if (entry.second.size() > 2) {
      error = "Coherent interface has a nonmanifold triangle edge";
      return false;
    }
    for (const size_t a : entry.second)
      for (const size_t b : entry.second) {
        if (a != b)
          neighbors[a].push_back(b);
      }
  }
  std::vector<bool> used(triangles.size(), false);
  for (size_t seed = 0; seed < triangles.size(); seed++) {
    if (used[seed])
      continue;
    const Plane &reference = planes[seed];
    CoherentPlanarCluster cluster{reference.origin,
                                  reference.normal,
                                  reference.u,
                                  reference.v,
                                  std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity(),
                                  {}};
    std::vector<size_t> queue{seed};
    used[seed] = true;
    for (size_t q = 0; q < queue.size(); q++) {
      const size_t current = queue[q];
      cluster.primitives.push_back(triangles[current].primitive);
      for (const auto &point : triangles[current].point) {
        const auto d = sub(point, reference.origin);
        const double x = dot(d, reference.u), y = dot(d, reference.v);
        cluster.min_u = std::min(cluster.min_u, x);
        cluster.max_u = std::max(cluster.max_u, x);
        cluster.min_v = std::min(cluster.min_v, y);
        cluster.max_v = std::max(cluster.max_v, y);
      }
      for (const size_t next : neighbors[current]) {
        if (!used[next] && coplanar(reference, planes[next], triangles[next])) {
          used[next] = true;
          queue.push_back(next);
        }
      }
    }
    if (!(cluster.max_u > cluster.min_u && cluster.max_v > cluster.min_v)) {
      error = "Coherent interface cluster has no finite planar area";
      return false;
    }
    std::sort(cluster.primitives.begin(), cluster.primitives.end());
    clusters.push_back(std::move(cluster));
  }
  return true;
}
CCL_NAMESPACE_END
