/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/util/diffraction_coordinates.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
using namespace ccl;
int main()
{
  std::mt19937 rng(541279);
  std::uniform_real_distribution<float> random(-1, 1);
  int failures = 0, checked = 0;
  double maximum = 0;
  auto check = [&](float3 direction, float n, float wavelength, float pitch) {
    const double length = std::sqrt(double(direction.x) * direction.x +
                                    double(direction.y) * direction.y +
                                    double(direction.z) * direction.z);
    const double exact_x = n * double(direction.x) / length;
    const double exact_y = n * double(direction.y) / length;
    for (bool mirror : {false, true}) {
      DiffractionCacheCoordinates result;
      if (!diffraction_cache_coordinates(direction, n, wavelength, pitch, mirror, &result)) {
        failures++;
        continue;
      }
      const double sign = result.reverse_orders ? -1 : 1;
      const double reconstructed = sign * (double(result.query.x) + result.incoming_order) *
                                   wavelength / pitch;
      const double error = std::abs(reconstructed - exact_x);
      maximum = std::max(maximum, error);
      failures += error > 2e-6 ||
                  std::abs(double(result.query.y) - (mirror ? -std::abs(exact_y) : exact_y)) >
                      2e-6 ||
                  result.query.x < -0.5f || result.query.x >= 0.5f;
      checked++;
    }
  };
  for (int i = 0; i < 20000; i++)
    check(make_float3(random(rng), random(rng), random(rng)),
          1.5f,
          380 + 400 * std::abs(random(rng)),
          400 + 2800 * std::abs(random(rng)));
  for (float pitch : {250.0f,
                      std::nextafter(250.0f, 0.0f),
                      std::nextafter(250.0f, 1000.0f),
                      750.0f,
                      1250.0f,
                      1e10f})
    for (float x : {-1.0f, 1.0f})
      check(make_float3(x, 0, 0), 1, 500, pitch);
  DiffractionCacheCoordinates result;
  failures += diffraction_cache_coordinates(zero_float3(), 1, 500, 740, false, &result);
  failures += diffraction_cache_coordinates(make_float3(0, 0, 1), 1, 0, 740, false, &result);
  failures += diffraction_cache_coordinates(
      make_float3(std::numeric_limits<float>::infinity(), 0, 1), 1, 500, 740, false, &result);
  std::cout << "{\"checked\":" << checked << ",\"failures\":" << failures
            << ",\"maximum_momentum_error\":" << maximum << "}\n";
  return failures ? 1 : 0;
}
