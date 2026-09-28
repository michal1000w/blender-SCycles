/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Direct modal-convergence audit, independent of cache interpolation. */
#include "scene/diffraction.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <numbers>
#include <random>

using namespace ccl;
using Clock = std::chrono::steady_clock;

static double column_error(const DiffractionGratingPowerBlock &a,
                           const DiffractionGratingPowerBlock &b)
{
  if (a.ports.size() != b.ports.size() || a.matrix.size() != b.matrix.size()) {
    return INFINITY;
  }
  const size_t n = a.ports.size();
  double maximum = 0.0;
  for (size_t col = 0; col < n; col++) {
    if (a.ports[col].order != b.ports[col].order ||
        a.ports[col].substrate != b.ports[col].substrate)
    {
      return INFINITY;
    }
    double sum = 0.0;
    for (size_t row = 0; row < n; row++) {
      sum += std::abs(a.matrix[row * n + col] - b.matrix[row * n + col]);
    }
    maximum = std::max(maximum, sum);
  }
  return maximum;
}

static void stats(const std::vector<double> &values)
{
  auto sorted = values;
  std::sort(sorted.begin(), sorted.end());
  double sum = 0.0;
  for (const double value : values) {
    sum += value;
  }
  std::cout << "{\"mean\":" << sum / values.size()
            << ",\"p95\":" << sorted[size_t(0.95 * (sorted.size() - 1))]
            << ",\"maximum\":" << sorted.back() << '}';
}

int main(int argc, char **argv)
{
  const bool targeted = argc == 8;
  if (argc > 5 && !targeted)
    return 2;
  const int queries = targeted ? 1 : argc > 1 ? std::atoi(argv[1]) : 32;
  std::array<double, 3> target{};
  if (targeted) {
    for (int j = 0; j < 3; j++)
      target[j] = std::strtod(argv[5 + j], nullptr);
    if (!std::isfinite(target[0]) || !std::isfinite(target[1]) || !std::isfinite(target[2]) ||
        target[0] < -0.5 || target[0] >= 0.5 || std::abs(target[1]) >= 1 || target[2] <= 0)
      return 2;
  }
  const int low_orders = argc > 2 ? std::atoi(argv[2]) : 64;
  const int high_orders = argc > 3 ? std::atoi(argv[3]) : 128;
  const unsigned seed = argc > 4 ? unsigned(std::strtoul(argv[4], nullptr, 10)) : 571291;
  if (queries < 1 || queries > 65536 || low_orders < 4 || high_orders <= low_orders ||
      high_orders > 512)
  {
    return 2;
  }
  std::cout << std::setprecision(12) << "{\"cases\":[";
  bool first = true;
  for (double pitch : {740.0, 1600.0}) {
    const std::complex<double> metal(0.9, 6.0);
    const DiffractionGratingProfile profile{pitch, 150.0, 0.41, 1.0, metal, 1.0, metal};
    const auto start = Clock::now();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<double> power_errors, complex_errors;
    double maximum_gain = 0.0, worst_power_error = -1;
    std::array<double, 3> worst_query{};
    for (int i = 0; i < queries; i++) {
      const double wavelength = targeted ? target[2] : 380.0 + 400.0 * uniform(rng);
      const double cosine = 0.0001 + 0.9999 * uniform(rng);
      const double phi = 2.0 * std::numbers::pi * uniform(rng);
      const double sine = std::sqrt(1.0 - cosine * cosine);
      const double x = sine * std::cos(phi), ky = targeted ? target[1] : sine * std::sin(phi);
      const double bloch = targeted ?
                               target[0] :
                               x * pitch / wavelength - std::floor(x * pitch / wavelength + 0.5);
      std::array<DiffractionGratingBlock, 2> blocks;
      std::array<DiffractionGratingPowerBlock, 2> powers;
      std::string error;
      for (int k = 0; k < 2; k++) {
        if (!diffraction_grating_solve_bloch(profile,
                                             wavelength,
                                             bloch * wavelength / pitch,
                                             ky,
                                             k ? high_orders : low_orders,
                                             blocks[k],
                                             error))
        {
          std::cerr << error << '\n';
          return 1;
        }
        diffraction_grating_power_block(blocks[k], powers[k]);
        maximum_gain = std::max(maximum_gain, blocks[k].maximum_power_gain);
      }
      const double power_error = column_error(powers[0], powers[1]);
      if (!std::isfinite(power_error) || blocks[0].matrix.size() != blocks[1].matrix.size()) {
        std::cerr << "Modal port mismatch\n";
        return 1;
      }
      double complex_error = 0.0;
      for (size_t j = 0; j < blocks[0].matrix.size(); j++) {
        complex_error = std::max(complex_error,
                                 std::abs(blocks[0].matrix[j] - blocks[1].matrix[j]));
      }
      if (power_error > worst_power_error) {
        worst_power_error = power_error;
        worst_query = {bloch, ky, wavelength};
      }
      power_errors.push_back(power_error);
      complex_errors.push_back(complex_error);
      if ((i + 1) % 8 == 0) {
        std::cerr << "pitch=" << pitch << " queries=" << i + 1 << '\n';
      }
    }
    std::cout << (first ? "" : ",") << "{\"pitch_nm\":" << pitch << ",\"queries\":" << queries
              << ",\"half_orders\":" << low_orders << ",\"reference_half_orders\":" << high_orders
              << ",\"seed\":" << seed
              << ",\"seconds\":" << std::chrono::duration<double>(Clock::now() - start).count()
              << ",\"maximum_power_gain\":" << maximum_gain << ",\"max_column_L1\":";
    stats(power_errors);
    std::cout << ",\"max_complex_amplitude_error\":";
    stats(complex_errors);
    std::cout << ",\"worst_power_query\":[" << worst_query[0] << "," << worst_query[1] << ","
              << worst_query[2] << "]}";
    first = false;
  }
  std::cout << "]}\n";
  return 0;
}
