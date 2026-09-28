/* SPDX-License-Identifier: Apache-2.0 */
#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "scene/coherent_sphere_host.h"
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace ccl;
int checks = 0;
void check(bool value, const char *label)
{
  ++checks;
  if (!value) throw std::runtime_error(label);
}
int main()
{
  try {
    const std::array<std::array<float, 3>, 3> identity = {{{1,0,0},{0,1,0},{0,0,1}}};
    float scale = 0, radius = 0;
    check(coherent_sphere_transform_scale(identity, scale) && scale == 1, "identity");
    auto matrix = identity;
    matrix = {{{0,-2,0},{0,0,2},{-2,0,0}}};
    check(coherent_sphere_transform_scale(matrix, scale) && scale == 2, "uniform signed permutation");
    matrix = identity; matrix[2][2] = 1.0000001192092896f;
    check(!coherent_sphere_transform_scale(matrix, scale), "one ULP ellipsoid rejected");
    matrix = identity; matrix[0][1] = 1e-10f;
    check(!coherent_sphere_transform_scale(matrix, scale), "tiny shear rejected");
    matrix = identity; matrix[1] = matrix[0];
    check(!coherent_sphere_transform_scale(matrix, scale), "singular duplicate axis rejected");
    matrix = identity; matrix[1] = {0,0,0};
    check(!coherent_sphere_transform_scale(matrix, scale), "zero scale rejected");
    matrix = identity; matrix[0][0] = std::numeric_limits<float>::infinity();
    check(!coherent_sphere_transform_scale(matrix, scale), "nonfinite transform rejected");
    check(coherent_sphere_world_radius(.3f, 2.f, false, radius) && radius == .6f, "exact scaled native radius");
    check(!coherent_sphere_world_radius(.3f, 1.1f, false, radius), "rounded instance radius rejected");
    check(coherent_sphere_world_radius(.3f, 1.1f, true, radius) && radius == .3f, "baked radius authoritative");
    for (float r : {0.f, -1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
      check(!coherent_sphere_world_radius(r, 1.f, true, radius), "invalid native radius rejected");
    check(coherent_sphere_source_outside({2,0,0}, {0,0,0}, 1), "exterior source");
    check(!coherent_sphere_source_outside({1,0,0}, {0,0,0}, 1), "surface source rejected");
    check(!coherent_sphere_source_outside({.5,0,0}, {0,0,0}, 1), "interior source rejected");
    check(coherent_sphere_source_outside({1e20,0,0}, {0,0,0}, 1e19f), "large finite no float squared overflow");
    check(!coherent_sphere_source_outside({1e-20,0,0}, {0,0,0}, 1e-19f), "tiny finite no float squared underflow");
    std::cout << checks << " checks PASS\n";
  }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
