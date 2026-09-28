/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Compare intensity blending with a common passive Cayley chart of hybrid
 * operators. This is a representation audit, not a renderer benchmark. */
#include "scene/diffraction.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
using namespace ccl;
static double difference(const DiffractionGratingPowerBlock &a,
                         const DiffractionGratingPowerBlock &b)
{
  if (a.ports.size() != b.ports.size())
    return INFINITY;
  const size_t n = a.ports.size();
  double maximum = 0;
  for (size_t col = 0; col < n; col++) {
    if (a.ports[col].order != b.ports[col].order ||
        a.ports[col].substrate != b.ports[col].substrate)
      return INFINITY;
    double error = 0;
    for (size_t row = 0; row < n; row++)
      error += std::abs(a.matrix[row * n + col] - b.matrix[row * n + col]);
    maximum = std::max(maximum, error);
  }
  return maximum;
}
int main(int argc, char **argv)
{
  if (argc != 2)
    return 2;
  const double cutoff_margin = std::atof(argv[1]);
  if (!std::isfinite(cutoff_margin) || cutoff_margin < 0)
    return 2;
  std::cout << std::setprecision(12) << "{\"cases\":[";
  bool first = true;
  for (int test = 0; test < 4; test++) {
    const double pitch = test % 2 ? 1600 : 740;
    const std::complex<double> material = test == 2 ? std::complex<double>(1.5) :
                                                      std::complex<double>(0.9, 6);
    const DiffractionGratingProfile profile{pitch, 150, 0.41, 1, material, 1, material};
    const DiffractionGratingCellBounds bounds =
        test == 0 ?
            DiffractionGratingCellBounds{{-0.48046875, -0.96875, 389.375},
                                         {-0.4765625, -0.9609375, 390.9375}} :
        test == 1 ?
            DiffractionGratingCellBounds{{-0.5, -0.9921875, 380},
                                         {-0.49609375, -0.984375, 381.5625}} :
            DiffractionGratingCellBounds{{-0.03125, -0.0625, pitch / (test == 3 ? 4 : 1) - 5},
                                         {0.03125, 0.0625, pitch / (test == 3 ? 4 : 1) + 5}};
    DiffractionGratingCell cell;
    std::string error;
    if (!diffraction_grating_prepare_cell(profile, bounds, 16, 5, cutoff_margin, cell, error)) {
      std::cerr << error << '\n';
      return 1;
    }
    std::array<DiffractionGratingBlock, 8> operators;
    for (int i = 0; i < 8; i++)
      operators[i] = cell.corners[i].scattering;
    std::vector<DiffractionGratingReferenceChart> charts;
    double phase;
    if (!diffraction_grating_choose_chart(operators, phase, charts, error)) {
      std::cerr << error << '\n';
      return 1;
    }
    struct Candidate {
      DiffractionGratingChartCell cell;
      double phase, training_error = 0, held_out_error = 0;
      bool valid = true;
      bool training_valid = true;
    };
    std::vector<Candidate> candidates;
    candidates.push_back({{bounds, cell.corners[0].is_reference, charts}, phase});
    for (int index = 0; index < 12; index++) {
      const double trial_phase = 2 * std::numbers::pi * index / 12;
      Candidate candidate{{bounds, cell.corners[0].is_reference, {}}, trial_phase};
      for (const auto &op : operators) {
        DiffractionGratingReferenceChart chart;
        if (!diffraction_grating_reference_to_chart(op, trial_phase, chart, error)) {
          candidate.valid = false;
          break;
        }
        candidate.cell.corners.push_back(std::move(chart));
      }
      if (candidate.valid)
        candidates.push_back(std::move(candidate));
    }
    std::mt19937 rng(149371);
    std::uniform_real_distribution<double> uniform(0, 1);
    double power_max = 0, chart_max = 0, power_sum = 0, chart_sum = 0, maximum_gain = 0;
    double complex_max = 0, complex_frobenius_max = 0;
    constexpr int probes = 283;
    for (int sample = 0; sample < probes; sample++) {
      std::array<double, 3> q, t;
      int lattice = sample;
      for (int axis = 0; axis < 3; axis++) {
        t[axis] = sample < 27 ? 0.25 * (1 + lattice % 3) : uniform(rng);
        lattice /= 3;
        q[axis] = bounds.lower[axis] + t[axis] * (bounds.upper[axis] - bounds.lower[axis]);
      }
      auto blend = charts[0];
      std::fill(blend.matrix.begin(), blend.matrix.end(), std::complex<double>(0));
      for (int corner = 0; corner < 8; corner++) {
        const double weight = (corner & 1 ? t[0] : 1 - t[0]) * (corner & 2 ? t[1] : 1 - t[1]) *
                              (corner & 4 ? t[2] : 1 - t[2]);
        for (size_t j = 0; j < blend.matrix.size(); j++)
          blend.matrix[j] += weight * charts[corner].matrix[j];
      }
      DiffractionGratingHybrid interpolated;
      interpolated.is_reference = cell.corners[0].is_reference;
      DiffractionGratingBlock reference, direct, matched;
      DiffractionGratingPowerBlock powers, chart_powers, exact;
      const double x = q[0] * q[2] / pitch;
      if (!diffraction_grating_chart_to_reference(blend, interpolated.scattering, error) ||
          !diffraction_grating_match_hybrid(
              profile, q[2], x, q[1], interpolated, matched, error) ||
          !diffraction_grating_cell_power(profile, cell, q, powers, error) ||
          !diffraction_grating_solve_reference(profile, q[2], x, q[1], 16, 5, reference, error) ||
          !diffraction_grating_match_reference(profile, q[2], x, q[1], reference, direct, error))
      {
        std::cerr << error << '\n';
        return 1;
      }
      diffraction_grating_power_block(matched, chart_powers);
      diffraction_grating_power_block(direct, exact);
      for (auto &candidate : candidates) {
        if (!candidate.valid)
          continue;
        DiffractionGratingBlock candidate_matched;
        if (!diffraction_grating_chart_cell_match(
                profile, candidate.cell, q, candidate_matched, error))
        {
          candidate.valid = false;
          if (sample < 27)
            candidate.training_valid = false;
          continue;
        }
        DiffractionGratingPowerBlock candidate_power;
        diffraction_grating_power_block(candidate_matched, candidate_power);
        const double candidate_error = difference(candidate_power, exact);
        if (!std::isfinite(candidate_error)) {
          candidate.valid = false;
          if (sample < 27)
            candidate.training_valid = false;
          continue;
        }
        double &score = sample < 27 ? candidate.training_error : candidate.held_out_error;
        score = std::max(score, candidate_error);
      }
      const double pe = difference(powers, exact), ce = difference(chart_powers, exact);
      if (!std::isfinite(pe) || !std::isfinite(ce))
        return 1;
      /* Both operators use the same centered profile and Cartesian flux
       * basis. Do not align away a global phase: it is observable relative to
       * another coherent path. The Frobenius error bounds the output field
       * error for any unit-norm superposition of incident channels. */
      if (matched.matrix.size() != direct.matrix.size())
        return 1;
      double squared_error = 0;
      for (size_t i = 0; i < direct.matrix.size(); i++) {
        const double amplitude_error = std::abs(matched.matrix[i] - direct.matrix[i]);
        if (!std::isfinite(amplitude_error))
          return 1;
        complex_max = std::max(complex_max, amplitude_error);
        squared_error += amplitude_error * amplitude_error;
      }
      complex_frobenius_max = std::max(complex_frobenius_max, std::sqrt(squared_error));
      power_max = std::max(power_max, pe);
      chart_max = std::max(chart_max, ce);
      power_sum += pe;
      chart_sum += ce;
      maximum_gain = std::max(maximum_gain, matched.maximum_power_gain);
    }
    size_t best = 0;
    int valid_candidates = 0;
    for (size_t i = 0; i < candidates.size(); i++) {
      if (!candidates[i].training_valid)
        continue;
      valid_candidates++;
      if (!candidates[best].training_valid ||
          candidates[i].training_error < candidates[best].training_error)
        best = i;
    }
    if (!candidates[best].valid)
      return 1;
    std::cout << (first ? "" : ",") << "{\"case\":" << test << ",\"pitch_nm\":" << pitch
              << ",\"lossless\":" << (test == 2 ? "true" : "false") << ",\"probes\":" << probes
              << ",\"channels\":" << 2 * operators[0].ports.size()
              << ",\"cutoff_margin\":" << cutoff_margin
              << ",\"feedback_channels\":" << cell.feedback_channels << ",\"phase\":" << phase
              << ",\"power_mean_error\":" << power_sum / probes
              << ",\"power_max_error\":" << power_max
              << ",\"chart_mean_error\":" << chart_sum / probes
              << ",\"chart_max_error\":" << chart_max
              << ",\"chart_max_complex_coefficient_error\":" << complex_max
              << ",\"chart_max_complex_frobenius_error\":" << complex_frobenius_max
              << ",\"chart_candidates\":" << valid_candidates
              << ",\"selected_phase\":" << candidates[best].phase
              << ",\"selected_training_error\":" << candidates[best].training_error
              << ",\"selected_held_out_error\":" << candidates[best].held_out_error
              << ",\"baseline_training_error\":" << candidates[0].training_error
              << ",\"baseline_held_out_error\":" << candidates[0].held_out_error
              << ",\"maximum_polarization_gain\":" << maximum_gain << '}';
    first = false;
  }
  std::cout << "]}\n";
}
