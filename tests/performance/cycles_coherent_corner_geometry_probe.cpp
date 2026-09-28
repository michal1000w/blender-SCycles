/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/light/coherent_geometry.h"
#include <cmath>
#include <cstdio>
#include <initializer_list>

using namespace ccl;
int main()
{
  CoherentGeometryInterface floor{}, wall{};
  floor.center = make_float3(1, 0, 0);
  floor.tangent_u = make_float3(1, 0, 0);
  floor.tangent_v = make_float3(0, 1, 0);
  floor.half_u = 1;
  floor.half_v = .4f;
  wall.center = make_float3(0, 0, 1);
  wall.tangent_u = make_float3(0, 1, 0);
  wall.tangent_v = make_float3(0, 0, 1);
  wall.half_u = .4f;
  wall.half_v = 1;
  floor.ior_before = floor.ior_after = floor.ior_opposite = 1;
  wall.ior_before = wall.ior_after = wall.ior_opposite = 1;
  floor.event = wall.event = COHERENT_GEOMETRY_REFLECT;
  std::puts("source_y,receiver_z,sequence,physical,solved,spreading,expected,relative_error");
  for (const float sy : {-1e-5f, 1e-5f})
    for (const float z : {.65f,
                          .66f,
                          .661f,
                          .665f,
                          .666f,
                          .6665f,
                          .6666f,
                          .6667f,
                          .667f,
                          .67f,
                          .671f,
                          .675f,
                          .69f,
                          .8f})
    {
      const float3 source = make_float3(.3f, sy, .2f), receiver = make_float3(1, 0, z),
                   normal = make_float3(-1, 0, 0);
      for (int order = 0; order < 2; order++) {
        CoherentGeometryInterface patches[4]{};
        patches[0] = order ? wall : floor;
        patches[1] = order ? floor : wall;
        CoherentGeometryPath path{};
        const bool solved = coherent_geometry_connect(source, receiver, normal, patches, 2, &path);
        const double dx = 1 + double(source.x), dy = -double(source.y), dz = double(z) + source.z;
        const double radius = std::sqrt(dx * dx + dy * dy + dz * dz),
                     expected = dx / (radius * radius * radius);
        const bool physical = order ? (double(z) * source.x < source.z) :
                                      (double(z) * source.x > source.z);
        std::printf("%.9g,%.9g,%s,%d,%d,%.9g,%.12g,%.6g\n",
                    sy,
                    z,
                    order ? "wall-floor" : "floor-wall",
                    physical,
                    solved,
                    solved ? path.spreading : 0,
                    expected,
                    solved ? (path.spreading / expected - 1) : 0);
      }
    }
}
