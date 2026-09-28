/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/light/coherent_geometry.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <initializer_list>
using namespace ccl;
using D = std::array<double, 3>;
D add(D a, D b)
{
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
D sub(D a, D b)
{
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
D mul(D a, double x)
{
  return {a[0] * x, a[1] * x, a[2] * x};
}
double dp(D a, D b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
D cp(D a, D b)
{
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
D norm(D a)
{
  return mul(a, 1 / std::sqrt(dp(a, a)));
}
D dbl(float3 a)
{
  return {a.x, a.y, a.z};
}
int checks = 0, failures = 0;
void check(bool ok, const char *name)
{
  checks++;
  if (!ok) {
    failures++;
    std::printf("FAIL %s\n", name);
  }
}
struct Reference {
  double length, spreading;
  std::array<D, 4> hit;
};
Reference reference(
    float3 source, float3 receiver, float3 detector, const CoherentGeometryInterface *p, int count)
{
  std::array<D, 5> image{};
  image[0] = dbl(source);
  for (int i = 0; i < count; i++) {
    D n = norm(cp(dbl(p[i].tangent_u), dbl(p[i].tangent_v)));
    image[i + 1] = sub(image[i], mul(n, 2 * dp(sub(image[i], dbl(p[i].center)), n)));
  }
  Reference r{};
  D start = dbl(receiver);
  for (int i = count - 1; i >= 0; i--) {
    D n = norm(cp(dbl(p[i].tangent_u), dbl(p[i].tangent_v)));
    D delta = sub(image[i + 1], start);
    double t = dp(sub(dbl(p[i].center), start), n) / dp(delta, n);
    r.hit[i] = add(start, mul(delta, t));
    start = r.hit[i];
  }
  const D delta = sub(dbl(receiver), image[count]);
  r.length = std::sqrt(dp(delta, delta));
  r.spreading = std::abs(dp(norm(dbl(detector)), delta)) / (r.length * r.length * r.length);
  return r;
}
int main()
{
  double maximum_opl = 0, maximum_opd = 0, maximum_spread = 0;
  for (int count : {2, 3, 4})
    for (float scene_scale : {.01f, 1.f, 100.f}) {
      const float3 source = make_float3(1.1f, -.7f, 2.3f) * scene_scale;
      float3 point = source, dir = normalize(make_float3(.9f, .3f, -.2f));
      CoherentGeometryInterface p[4]{};
      float3 ns[4] = {make_float3(1, .3f, .2f),
                      make_float3(.2f, 1, .4f),
                      make_float3(.4f, .1f, 1),
                      make_float3(1, -.7f, .3f)};
      for (int i = 0; i < count; i++) {
        point += (.8f + .2f * i) * scene_scale * dir;
        const float3 n = normalize(ns[i]);
        p[i].center = point;
        p[i].tangent_u = normalize(cross(n, make_float3(0, 0, 1)));
        p[i].tangent_v = normalize(cross(n, p[i].tangent_u));
        p[i].half_u = p[i].half_v = 10 * scene_scale;
        p[i].ior_before = p[i].ior_after = p[i].ior_opposite = 1;
        p[i].event = COHERENT_GEOMETRY_REFLECT;
        const float3 actual_n = normalize(cross(p[i].tangent_u, p[i].tangent_v));
        dir -= 2 * dot(dir, actual_n) * actual_n;
      }
      const float3 receiver = point + .9f * scene_scale * dir, detector = -normalize(dir);
      CoherentGeometryPath paths[2]{};
      Reference refs[2];
      for (int i = 0; i < 2; i++) {
        const float3 s = source + make_float3(0, float(i) * 5e-5f * scene_scale, 0);
        refs[i] = reference(s, receiver, detector, p, count);
        check(coherent_geometry_connect(s, receiver, detector, p, count, &paths[i]),
              "tilted translated multi-reflection solve");
        const double opl = double(paths[i].optical_length_split.x) +
                           paths[i].optical_length_split.y;
        const double error = std::abs(opl - refs[i].length);
        maximum_opl = std::max(maximum_opl, error / scene_scale);
        const double spread = std::abs(paths[i].spreading / refs[i].spreading - 1);
        maximum_spread = std::max(maximum_spread, spread);
        check(error < 2e-7 * scene_scale, "physical split OPL versus double isometry");
        check(spread < 3e-5, "analytic spreading versus double");
        for (int j = 0; j < count; j++) {
          const D hit = dbl(paths[i].point[j]);
          check(std::sqrt(dp(sub(hit, refs[i].hit[j]), sub(hit, refs[i].hit[j]))) <
                    3e-6 * scene_scale,
                "physical hit versus double plane intersection");
        }
      }
      const float3 n0 = normalize(cross(p[0].tangent_u, p[0].tangent_v));
      const int valid_side = dot(source - p[0].center, n0) > 0 ? 1 : -1;
      p[0].expected_incident_side = -valid_side;
      CoherentGeometryPath rejected{};
      check(!coherent_geometry_connect(source, receiver, detector, p, count, &rejected),
            "opposite incident side rejects");
      p[0].expected_incident_side = valid_side;
      check(coherent_geometry_connect(source, receiver, detector, p, count, &rejected),
            "declared incident side accepts");
      const float3 saved = p[0].center;
      p[0].center += .2f * scene_scale * p[0].tangent_u;
      p[0].half_u = .001f * scene_scale;
      check(!coherent_geometry_connect(source, receiver, detector, p, count, &rejected),
            "outside finite mirror patch rejects");
      p[0].center = saved;
      p[0].half_u = 10 * scene_scale;
      p[0].expected_incident_side = 0;
      const float2 delta = coherent_geometry_add_split(
          paths[0].optical_length_split,
          make_float2(-paths[1].optical_length_split.x, -paths[1].optical_length_split.y));
      const double opd = double(delta.x) + delta.y, expected = refs[0].length - refs[1].length;
      maximum_opd = std::max(maximum_opd, std::abs(opd - expected) / scene_scale);
      check(std::abs(opd - expected) < 3e-9 * scene_scale, "pair OPD versus double isometry");
    }
  std::printf(
      "checks=%d failures=%d max_scaled_OPL=%.12g max_scaled_OPD=%.12g "
      "max_spread_relative=%.12g\n",
      checks,
      failures,
      maximum_opl,
      maximum_opd,
      maximum_spread);
  return failures ? 1 : 0;
}
