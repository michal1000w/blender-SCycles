/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Physical-channel validation of held-out rational predictions on stdin. */
#include "scene/diffraction.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace ccl;
int main()
{
  const DiffractionGratingProfile p{740, 150, .41, 1, 1.5, 1, 1};
  int count;
  if (!(std::cin >> count) || count < 1 || count > 10000) return 2;
  double max_error = 0, max_conservation = 0, max_reference_conservation = 0;
  double minimum_gain = 1, maximum_gain = 1, reference_gain_error = 0;
  int empty_queries = 0;
  for (int q = 0; q < count; ++q) {
    double wavelength, bloch, ky;
    size_t size;
    if (!(std::cin >> wavelength >> bloch >> ky >> size)) return 2;
    DiffractionGratingBlock reference, exact, approximation, actual;
    std::string error;
    const double kx = bloch * wavelength / p.pitch;
    if (!diffraction_grating_solve_reference(p, wavelength, kx, ky, 16, 2, reference, error) ||
        !diffraction_grating_match_reference(p, wavelength, kx, ky, reference, exact, error)) {
      std::cerr << error << '\n'; return 1;
    }
    if (size != reference.matrix.size()) return 2;
    approximation = reference;
    for (auto &entry : approximation.matrix) {
      double re, im;
      if (!(std::cin >> re >> im) || !std::isfinite(re) || !std::isfinite(im)) return 2;
      entry = {re, im};
    }
    if (!diffraction_grating_match_reference(p, wavelength, kx, ky, approximation, actual, error)) {
      std::cerr << error << '\n'; return 1;
    }
    if (exact.ports.empty()) {
      if (!actual.ports.empty()) return 1;
      ++empty_queries;
      continue;
    }
    minimum_gain = std::min(minimum_gain, actual.minimum_power_gain);
    maximum_gain = std::max(maximum_gain, actual.maximum_power_gain);
    reference_gain_error = std::max(reference_gain_error,
        std::max(std::abs(exact.minimum_power_gain - 1), std::abs(exact.maximum_power_gain - 1)));
    DiffractionGratingPowerBlock a, b;
    diffraction_grating_power_block(actual, a);
    diffraction_grating_power_block(exact, b);
    const size_t n = b.ports.size();
    if (a.ports.size() != n) return 1;
    for (size_t col = 0; col < n; ++col) {
      if (a.ports[col].order != b.ports[col].order ||
          a.ports[col].substrate != b.ports[col].substrate) return 1;
      double error_sum = 0, sum = 0, reference_sum = 0;
      for (size_t row = 0; row < n; ++row) {
        error_sum += std::abs(a.matrix[row*n+col] - b.matrix[row*n+col]);
        sum += a.matrix[row*n+col]; reference_sum += b.matrix[row*n+col];
      }
      if (!std::isfinite(error_sum) || !std::isfinite(sum)) return 1;
      max_error = std::max(max_error, error_sum);
      max_conservation = std::max(max_conservation, std::abs(sum-1));
      max_reference_conservation = std::max(max_reference_conservation, std::abs(reference_sum-1));
    }
  }
  std::cout << std::setprecision(17) << "{\"held_out_queries\":" << count
            << ",\"empty_queries\":" << empty_queries
            << ",\"physical_queries\":" << count-empty_queries
            << ",\"minimum_coherent_power_gain\":" << minimum_gain
            << ",\"maximum_coherent_power_gain\":" << maximum_gain
            << ",\"direct_coherent_gain_error\":" << reference_gain_error
            << ",\"maximum_power_column_l1_error\":" << max_error
            << ",\"maximum_lossless_column_sum_error\":" << max_conservation
            << ",\"direct_lossless_column_sum_error\":" << max_reference_conservation
            << ",\"scope\":\"Supplied N16 query set; not modal convergence or a worst-case bound\"}\n";
}
