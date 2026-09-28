/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Lazy adaptive-cell accuracy audit. Acceptance samples are fixed by each cell,
 * independent of held-out queries. This measures visited cells only: it is not
 * a complete cache build, a global error proof, or a rendering benchmark. */
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

struct CellNode {
  DiffractionGratingCell cell;
  bool accepted;
  double validation_error;
};

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
  const int queries = argc > 1 ? std::atoi(argv[1]) : 64;
  const double tolerance = argc > 2 ? std::atof(argv[2]) : 0.002;
  const int max_depth = argc > 3 ? std::atoi(argv[3]) : 6;
  const double margin = argc > 4 ? std::atof(argv[4]) : 0.25;
  const int half_orders = argc > 6 ? std::atoi(argv[6]) : 16;
  const int reference_orders = argc > 7 ? std::atoi(argv[7]) : 32;
  const unsigned seed = argc > 5 ? unsigned(std::strtoul(argv[5], nullptr, 10)) : 617923;
  if (queries < 1 || queries > 65536 || !(tolerance > 0.0) || !std::isfinite(tolerance) ||
      max_depth < 0 || max_depth > 12 || half_orders < 4 || reference_orders <= half_orders ||
      reference_orders > 128 || !(margin >= 0.0) || !std::isfinite(margin))
  {
    return 2;
  }
  std::cout << std::setprecision(12) << "{\"cases\":[";
  bool first = true;
  for (const double pitch : {740.0, 1600.0}) {
    const std::complex<double> metal(0.9, 6.0);
    const DiffractionGratingProfile profile{pitch, 150.0, 0.41, 1.0, metal, 1.0, metal};
    const int retained = int(std::ceil(pitch / 380.0 - 0.5));
    const auto start = Clock::now();
    std::map<std::array<double, 6>, CellNode> nodes;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<double> errors, truncation, feedback, depths;
    int unresolved = 0;
    std::string error;
    for (int probe = 0; probe < queries; probe++) {
      const double wavelength = 380.0 + 400.0 * uniform(rng);
      const double cosine = 0.0001 + 0.9999 * uniform(rng);
      const double phi = 2.0 * std::numbers::pi * uniform(rng);
      const double sine = std::sqrt(1.0 - cosine * cosine);
      const double kx = sine * std::cos(phi), ky = sine * std::sin(phi);
      const double bloch = kx * pitch / wavelength - std::floor(kx * pitch / wavelength + 0.5);
      const std::array<double, 3> query{bloch, ky, wavelength};
      DiffractionGratingCellBounds bounds;
      const std::array<double, 3> minimum{-0.5, -1.0, 380.0}, extent{1.0, 2.0, 400.0};
      for (int axis = 0; axis < 3; axis++) {
        const int index = std::clamp(
            int(4.0 * (query[axis] - minimum[axis]) / extent[axis]), 0, 3);
        bounds.lower[axis] = minimum[axis] + extent[axis] * index / 4.0;
        bounds.upper[axis] = minimum[axis] + extent[axis] * (index + 1) / 4.0;
      }
      CellNode *leaf = nullptr;
      int depth = 0;
      for (; depth <= max_depth; depth++) {
        const std::array<double, 6> key{bounds.lower[0],
                                        bounds.lower[1],
                                        bounds.lower[2],
                                        bounds.upper[0],
                                        bounds.upper[1],
                                        bounds.upper[2]};
        auto found = nodes.find(key);
        if (found == nodes.end()) {
          CellNode node{};
          if (!diffraction_grating_prepare_cell(
                  profile, bounds, half_orders, retained, margin, node.cell, error))
          {
            std::cerr << error << '\n';
            return 1;
          }
          for (int sample = 0; sample < 27; sample++) {
            std::array<double, 3> validation;
            int lattice = sample;
            for (int axis = 0; axis < 3; axis++) {
              const double fraction = 0.25 * (1 + lattice % 3);
              lattice /= 3;
              validation[axis] = bounds.lower[axis] +
                                 fraction * (bounds.upper[axis] - bounds.lower[axis]);
            }
            DiffractionGratingBlock direct;
            DiffractionGratingPowerBlock actual, exact;
            if (!diffraction_grating_cell_power(profile, node.cell, validation, actual, error) ||
                !diffraction_grating_solve_bloch(profile,
                                                 validation[2],
                                                 validation[0] * validation[2] / pitch,
                                                 validation[1],
                                                 half_orders,
                                                 direct,
                                                 error))
            {
              std::cerr << error << '\n';
              return 1;
            }
            diffraction_grating_power_block(direct, exact);
            node.validation_error = std::max(node.validation_error, column_error(actual, exact));
          }
          node.accepted = node.validation_error <= 0.5 * tolerance;
          if (!node.accepted && depth < max_depth) {
            node.cell = {};
          }
          found = nodes.emplace(key, std::move(node)).first;
        }
        if (found->second.accepted || depth == max_depth) {
          leaf = &found->second;
          break;
        }
        for (int axis = 0; axis < 3; axis++) {
          const double middle = 0.5 * (bounds.lower[axis] + bounds.upper[axis]);
          if (query[axis] < middle) {
            bounds.upper[axis] = middle;
          }
          else {
            bounds.lower[axis] = middle;
          }
        }
      }
      if (!leaf || leaf->cell.corners[0].scattering.matrix.empty()) {
        std::cerr << "Missing adaptive leaf\n";
        return 1;
      }
      unresolved += !leaf->accepted;
      DiffractionGratingPowerBlock actual, low, high;
      if (!diffraction_grating_cell_power(profile, leaf->cell, query, actual, error)) {
        std::cerr << error << '\n';
        return 1;
      }
      for (int n : {half_orders, reference_orders}) {
        DiffractionGratingBlock direct;
        if (!diffraction_grating_solve_bloch(
                profile, wavelength, bloch * wavelength / pitch, ky, n, direct, error))
        {
          std::cerr << error << '\n';
          return 1;
        }
        diffraction_grating_power_block(direct, n == half_orders ? low : high);
      }
      errors.push_back(column_error(actual, low));
      truncation.push_back(column_error(low, high));
      feedback.push_back(leaf->cell.feedback_channels);
      depths.push_back(depth);
      if ((probe + 1) % 32 == 0) {
        std::cerr << "pitch=" << pitch << " held_out=" << probe + 1 << " cells=" << nodes.size()
                  << '\n';
      }
    }
    size_t matrix_bytes = 0, accepted = 0;
    for (const auto &[key, node] : nodes) {
      accepted += node.accepted;
      for (const auto &corner : node.cell.corners) {
        matrix_bytes += corner.scattering.matrix.size() * sizeof(std::complex<double>);
      }
    }
    std::cout << (first ? "" : ",") << "{\"pitch_nm\":" << pitch << ",\"queries\":" << queries
              << ",\"half_orders\":" << half_orders
              << ",\"reference_half_orders\":" << reference_orders
              << ",\"tolerance\":" << tolerance << ",\"acceptance_tolerance\":" << 0.5 * tolerance
              << ",\"validation_samples_per_cell\":27,\"seed\":" << seed
              << ",\"cutoff_margin\":" << margin << ",\"visited_cells\":" << nodes.size()
              << ",\"accepted_cells\":" << accepted << ",\"unresolved_queries\":" << unresolved
              << ",\"visited_host_matrix_bytes\":" << matrix_bytes
              << ",\"seconds\":" << std::chrono::duration<double>(Clock::now() - start).count()
              << ",\"held_out_max_column_L1\":";
    stats(errors);
    std::cout << ",\"modal_truncation_max_column_L1\":";
    stats(truncation);
    std::cout << ",\"feedback_channels\":";
    stats(feedback);
    std::cout << ",\"refinement_depth\":";
    stats(depths);
    std::cout << ",\"sampled_accuracy_met\":"
              << (unresolved == 0 && *std::max_element(errors.begin(), errors.end()) <= tolerance ?
                      "true" :
                      "false")
              << '}';
    first = false;
  }
  std::cout << "]}\n";
  return 0;
}
