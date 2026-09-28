/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "diffraction_packed_audit.h"
#include <iostream>
#include <limits>

int main()
{
  using namespace ccl;
  DiffractionGratingPowerBlock reference;
  reference.ports = {{0, false}, {1, false}};
  reference.matrix = {0.5, 0.25, 0.25, 0.5};
  auto stored = reference.ports;
  stored.push_back({2, false});
  float values[] = {0.5f, 0.25f, 0.125f};
  double error;
  int failures = 0;
  auto check = [&](bool condition, const char *name) {
    if (!condition) {
      std::cerr << name << '\n';
      failures++;
    }
  };
  check(diffraction_audit::power_column_error(stored, values, reference, 0, error) &&
            error == 0.125,
        "Extra output energy must count toward error");
  values[2] = 0;
  check(diffraction_audit::power_column_error(stored, values, reference, 0, error) && error == 0,
        "Closed zero-power output is valid");
  values[2] = std::numeric_limits<float>::quiet_NaN();
  check(!diffraction_audit::power_column_error(stored, values, reference, 0, error),
        "Nonfinite extra output must fail");
  values[2] = -0.125f;
  check(!diffraction_audit::power_column_error(stored, values, reference, 0, error),
        "Negative extra output must fail");
  stored = {{0, false}};
  check(!diffraction_audit::power_column_error(stored, values, reference, 0, error),
        "Missing physical output must fail");
  stored = {{0, false}, {0, false}, {1, false}};
  check(!diffraction_audit::power_column_error(stored, values, reference, 0, error),
        "Duplicate physical output must fail");
  stored = {{1, false}, {0, false}};
  values[0] = 0.25f;
  values[1] = 0.5f;
  check(diffraction_audit::power_column_error(stored, values, reference, 0, error) && error == 0,
        "Port order must not affect comparison");
  float2 matrix[32] = {}, boundaries[8] = {}, jones[8] = {};
  const float3 center = make_float3(0.5f, 0.5f, 0.5f);
  const float2 rotation = make_float2(1, 0);
  check(!diffraction_cache_chart_match<2>(matrix, boundaries, 4, 1, rotation, 0, center, jones),
        "Oversized chart must fail without truncation");
  check(!diffraction_cache_chart_match<2>(matrix, boundaries, 2, 1, rotation, 1, center, jones),
        "Invalid chart incoming port must fail");
  check(!diffraction_cache_chart_match<2>(matrix, boundaries, 2, 3, rotation, 0, center, jones),
        "Unsupported chart degree must fail");
  int active[] = {0};
  check(!diffraction_cache_intensity_match<2>(matrix, boundaries, active, 4, 2, 0, center, values),
        "Oversized feedback set must fail");
  check(
      !diffraction_cache_intensity_match<2>(matrix, boundaries, active, 0, 2, -1, center, values),
      "Negative incoming port must fail");
  std::cout << "12 packed power and dispatch checks; failures=" << failures << '\n';
  return failures != 0;
}
