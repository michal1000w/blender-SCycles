/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/geom/point_intersect.h"
#include <cstdio>
using namespace ccl;
int main()
{
  int checks = 0, failures = 0;
  auto check = [&](bool ok) { ++checks; failures += !ok; };
  const uint marker = SD_OBJECT_COHERENT_GLASS_POINT;
  for (bool enabled : {false, true}) for (uint flag : {0u, marker}) {
    check(point_coherent_glass_two_sided(enabled, flag, PRIMITIVE_POINT) ==
          (enabled && flag == marker));
    check(!point_coherent_glass_two_sided(enabled, flag, PRIMITIVE_MOTION_POINT));
    check(!point_coherent_glass_two_sided(enabled, flag, PRIMITIVE_TRIANGLE));
    check(!point_coherent_glass_two_sided(enabled, flag | SD_OBJECT_MOTION, PRIMITIVE_POINT));
    check(!point_coherent_glass_two_sided(enabled, flag | SD_OBJECT_HAS_VERTEX_MOTION,
                                        PRIMITIVE_POINT));
  }
  for (bool two : {false, true}) for (bool previous : {false, true})
    for (bool light : {false, true})
      check(point_coherent_skip_self(two, previous, light) ==
            (light || (previous && !two)));
  for (float scale : {.01f, 1.f, 100.f}) for (float speed : {.5f, 1.f, 3.f}) {
    const float3 center = make_float3(3.f, -2.f, 4.f) * scale;
    const float4 sphere = make_float4(center.x, center.y, center.z, scale);
    const float3 direction = make_float3(speed, 0, 0);
    float t = -1;
    const float tolerance = 2e-5f * scale / speed;
    for (bool two : {false, true}) {
      check(point_intersect_test(sphere, center - make_float3(2*scale,0,0), direction,
                                 0, 4*scale/speed, &t, two) &&
            fabsf(t-scale/speed) < tolerance);
      const bool inside = point_intersect_test(sphere, center, direction, 0,
                                               2*scale/speed, &t, two);
      check(inside == two && (!inside || fabsf(t-scale/speed) < tolerance));
      const bool clipped = point_intersect_test(sphere, center - make_float3(2*scale,0,0),
          direction, 1.5f*scale/speed, 4*scale/speed, &t, two);
      check(clipped == two && (!clipped || fabsf(t-3*scale/speed) < tolerance));
      check(!point_intersect_test(sphere, center, direction, 0, .5f*scale/speed, &t, two));
      check(!point_intersect_test(sphere, center + make_float3(0,2*scale,0), direction,
                                 0, 4*scale/speed, &t, two));
    }
    /* Offset just inside after entry; same primitive far exit must remain eligible. */
    const float3 inside_origin = center - make_float3(.999f*scale,0,0);
    check(point_intersect_test(sphere, inside_origin, direction, 0,
                              4*scale/speed, &t, true) &&
          fabsf(t-1.999f*scale/speed) < tolerance);
    /* Closest-hit tmax must retain a separate blocker before the far interface. */
    const float exit_t = t;
    const float3 blocker_center = inside_origin + make_float3(.5f*scale,0,0);
    const float4 blocker = make_float4(blocker_center.x, blocker_center.y,
                                      blocker_center.z, .1f*scale);
    check(point_intersect_test(blocker, inside_origin, direction, 0, exit_t, &t) &&
          t < exit_t && fabsf(t-.4f*scale/speed) < tolerance);
  }
  float t;
  check(point_intersect_test(make_float4(0,0,0,1), make_float3(-2,1,0),
                            make_float3(1,0,0), 0, 4, &t, true) && t == 2);
  check(point_intersect_test(make_float4(0,0,0,1), make_float3(-2,0,0),
                            make_float3(1,0,0), 0, 4, &t) && t == 1);
  check(!point_intersect_test(make_float4(0,0,0,1), zero_float3(),
                             make_float3(1,0,0), 0, 4, &t));
  std::printf("{\"checks\":%d,\"failures\":%d}\n", checks, failures);
  return failures != 0;
}
