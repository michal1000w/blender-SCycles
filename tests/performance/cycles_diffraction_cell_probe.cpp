/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
using namespace ccl;
int main(int argc, char **argv)
{
  if (argc != 11)
    return 2;
  const double pitch = std::atof(argv[1]);
  const std::complex<double> metal(.9, 6);
  const DiffractionGratingProfile p{pitch, 150, .41, 1, metal, 1, metal};
  DiffractionGratingCellBounds bounds;
  std::array<double, 3> q;
  for (int i = 0; i < 3; i++) {
    bounds.lower[i] = std::atof(argv[2 + i]);
    bounds.upper[i] = std::atof(argv[5 + i]);
    q[i] = std::atof(argv[8 + i]);
  }
  const int retained = int(std::ceil(pitch / 380.0 - .5));
  DiffractionGratingCell cell;
  DiffractionGratingChartCell chart;
  DiffractionGratingPackedCell packed;
  DiffractionGratingBlock reference, exact, actual;
  std::string error;
  const double x = q[0] * q[2] / pitch;
  if (!diffraction_grating_prepare_cell(p, bounds, 16, retained, .1, cell, error) ||
      !diffraction_grating_prepare_quadratic_chart(p, cell, 16, retained, chart, error) ||
      !diffraction_grating_pack_chart_cell(chart, packed, error) ||
      !diffraction_grating_chart_cell_match(p, chart, q, actual, error) ||
      !diffraction_grating_solve_reference(p, q[2], x, q[1], 16, retained, reference, error) ||
      !diffraction_grating_match_reference(p, q[2], x, q[1], reference, exact, error))
  {
    std::cerr << error << '\n';
    return 1;
  }
  DiffractionGratingPowerBlock a, b;
  diffraction_grating_power_block(actual, a);
  diffraction_grating_power_block(exact, b);
  if (a.ports.size() != b.ports.size() || actual.matrix.size() != exact.matrix.size())
    return 1;
  const size_t n = a.ports.size();
  double power_error = 0, complex_error = 0;
  for (size_t col = 0; col < n; col++) {
    if (a.ports[col].order != b.ports[col].order ||
        a.ports[col].substrate != b.ports[col].substrate)
      return 1;
    double column = 0;
    for (size_t row = 0; row < n; row++)
      column += std::abs(a.matrix[row * n + col] - b.matrix[row * n + col]);
    power_error = std::max(power_error, column);
  }
  for (size_t j = 0; j < actual.matrix.size(); j++)
    complex_error = std::hypot(complex_error, std::abs(actual.matrix[j] - exact.matrix[j]));
  /* Identity check against the exported float payload, not an accuracy metric. */
  uint64_t hash = 14695981039346656037ull;
  const auto *bytes = reinterpret_cast<const unsigned char *>(packed.matrices.data());
  for (size_t j = 0; j < packed.matrices.size() * sizeof(float2); j++) {
    hash ^= bytes[j];
    hash *= 1099511628211ull;
  }
  std::cout << std::setprecision(17) << "{\"cases\":[{\"pitch_nm\":" << pitch
            << ",\"degree\":" << packed.chart_degree
            << ",\"double_power_column_error\":" << power_error
            << ",\"double_complex_frobenius_error\":" << complex_error << ",\"matrix_fnv1a64\":\""
            << std::hex << hash << std::dec
            << "\",\"matrix_float2_count\":" << packed.matrices.size() << "}]}\n";
}
