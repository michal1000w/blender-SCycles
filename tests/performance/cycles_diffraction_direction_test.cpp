/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/util/diffraction_direction.h"
#include <iostream>
#include <random>
using namespace ccl;
int main()
{
  std::mt19937 rng(192845);
  std::uniform_real_distribution<float> uniform(-1, 1);
  int failures = 0, physical = 0, closed = 0;
  double maximum = 0;
  for (int i = 0; i < 20000; i++) {
    const float3 input = make_float3(uniform(rng), uniform(rng), uniform(rng));
    const float ni = i % 2 ? 1.0f : 1.5f, no = i % 3 ? 1.0f : 1.5f;
    const float wavelength = 380 + 400 * std::abs(uniform(rng)), pitch = i % 2 ? 740 : 1600;
    const int order = int(rng() % 9) - 4;
    const double norm = std::sqrt(double(input.x) * input.x + double(input.y) * input.y +
                                  double(input.z) * input.z);
    const double x = double(ni) * input.x / norm + double(order) * wavelength / pitch,
                 y = double(ni) * input.y / norm;
    const double q2 = no * no - x * x - y * y;
    for (bool transmission : {false, true}) {
      float3 out;
      const bool valid = diffraction_grating_direction(
          input, ni, no, wavelength, pitch, order, transmission, &out);
      if (valid != (q2 > 0)) {
        failures++;
        continue;
      }
      if (!valid) {
        closed++;
        continue;
      }
      physical++;
      const double error = std::max({std::abs(double(out.x) * no - x),
                                     std::abs(double(out.y) * no - y),
                                     std::abs(std::abs(double(out.z)) * no - std::sqrt(q2)),
                                     std::abs(double(len_squared(out)) - 1)});
      maximum = std::max(maximum, error);
      failures += error > 2e-6 || (out.z < 0) != transmission;
    }
  }
  float3 out;
  failures += diffraction_grating_direction(make_float3(1, 0, 0), 1, 1, 500, 1000, 0, false, &out);
  failures += diffraction_grating_direction(zero_float3(), 1, 1, 500, 1000, 0, false, &out);
  /* One order below exact tangential cancellation; direct int-to-float
   * conversion rounds the order upward and incorrectly produces x=0. */
  failures += !diffraction_grating_direction(
      make_float3(-1, 0, 0), 1, 1, 1, 33554432.0f, 33554431, false, &out);
  failures += out.x != -0x1p-25f;
  failures += !diffraction_grating_direction(
      make_float3(1, 0, 0), 1, 1, 1, 33554432.0f, -33554431, false, &out);
  failures += out.x != 0x1p-25f;
  std::cout << "{\"physical\":" << physical << ",\"closed\":" << closed
            << ",\"failures\":" << failures << ",\"maximum_error\":" << maximum << "}\n";
  return failures ? 1 : 0;
}
