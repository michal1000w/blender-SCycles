/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Audit tensor-quadratic Bernstein charts. No renderer/cache policy changes. */
#include "scene/diffraction.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
using namespace ccl;
using Complex = std::complex<double>;
int main()
{
  std::cout << std::setprecision(12) << "{\"cases\":[";
  for (int test = 0; test < 4; test++) {
    const double pitch = test % 2 ? 1600 : 740;
    const Complex material = test == 2 ? Complex(1.5) : Complex(0.9, 6);
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
    std::string error;
    DiffractionGratingCell cell;
    if (!diffraction_grating_prepare_cell(profile, bounds, 16, 5, 0.1, cell, error)) {
      std::cerr << error;
      return 1;
    }
    std::vector<DiffractionGratingBlock> samples;
    for (int sample = 0; sample < 27; sample++) {
      int lattice = sample;
      std::array<double, 3> q;
      for (int a = 0; a < 3; a++) {
        q[a] = bounds.lower[a] + 0.5 * (lattice % 3) * (bounds.upper[a] - bounds.lower[a]);
        lattice /= 3;
      }
      DiffractionGratingBlock reference;
      const double x = q[0] * q[2] / pitch;
      if (!diffraction_grating_solve_reference(profile, q[2], x, q[1], 16, 5, reference, error)) {
        std::cerr << error;
        return 1;
      }
      std::vector<unsigned char> keep(reference.ports.size(), 0);
      for (size_t i = 0; i < keep.size(); i++)
        for (size_t j = 0; j < cell.corners[0].scattering.ports.size(); j++)
          if (reference.ports[i].order == cell.corners[0].scattering.ports[j].order &&
              reference.ports[i].substrate == cell.corners[0].scattering.ports[j].substrate)
            keep[i] = cell.corners[0].is_reference[j];
      DiffractionGratingHybrid hybrid;
      if (!diffraction_grating_prepare_hybrid(
              profile, q[2], x, q[1], reference, keep, hybrid, error))
      {
        std::cerr << error;
        return 1;
      }
      if (hybrid.is_reference != cell.corners[0].is_reference)
        return 1;
      samples.push_back(std::move(hybrid.scattering));
    }
    double phase;
    std::vector<DiffractionGratingReferenceChart> controls;
    if (!diffraction_grating_choose_chart(samples, phase, controls, error)) {
      std::cerr << error;
      return 1;
    }
    const auto nodal_charts = controls;
    /* Interpolating quadratic values at 0, 1/2, 1 convert to Bernstein
     * controls B0=Y0, B1=2*Ymid-(Y0+Y1)/2, B2=Y1. Apply in each axis. */
    for (int axis = 0, stride = 1; axis < 3; axis++, stride *= 3)
      for (int i = 0; i < 27; i++)
        if ((i / stride) % 3 == 0)
          for (size_t j = 0; j < controls[i].matrix.size(); j++)
            controls[i + stride].matrix[j] = 2.0 * controls[i + stride].matrix[j] -
                                             0.5 * (controls[i].matrix[j] +
                                                    controls[i + 2 * stride].matrix[j]);
    double nodal_error = 0;
    for (int point = 0; point < 27; point++) {
      const double coordinate[3] = {0.5 * (point % 3), 0.5 * ((point / 3) % 3), 0.5 * (point / 9)};
      for (size_t j = 0; j < controls[0].matrix.size(); j++) {
        Complex value = 0;
        for (int index = 0; index < 27; index++) {
          double weight = 1;
          int lattice = index;
          for (int axis = 0; axis < 3; axis++) {
            const double t = coordinate[axis];
            const int digit = lattice % 3;
            lattice /= 3;
            weight *= digit == 0 ? (1 - t) * (1 - t) : digit == 1 ? 2 * t * (1 - t) : t * t;
          }
          value += weight * controls[index].matrix[j];
        }
        nodal_error = std::max(nodal_error,
                               std::abs(value - nodal_charts[point].matrix[j]) /
                                   std::max(1.0, std::abs(nodal_charts[point].matrix[j])));
      }
    }
    if (!std::isfinite(nodal_error) || nodal_error > 1e-10)
      return 1;
    double minimum_dissipation = INFINITY;
    const int channels = 2 * controls[0].ports.size();
    for (const auto &control : controls) {
      Eigen::MatrixXcd matrix(channels, channels);
      for (int r = 0; r < channels; r++)
        for (int c = 0; c < channels; c++)
          matrix(r, c) = control.matrix[r * channels + c];
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigen(0.5 * (matrix + matrix.adjoint()),
                                                            Eigen::EigenvaluesOnly);
      if (eigen.info() != Eigen::Success)
        return 1;
      minimum_dissipation = std::min(minimum_dissipation, eigen.eigenvalues().minCoeff());
    }
    std::mt19937 rng(149371);
    std::uniform_real_distribution<double> uniform(0, 1);
    double maximum_power_error = 0, maximum_complex_error = 0, maximum_gain = 0;
    double held_out_error = 0;
    for (int sample = 0; sample < 283; sample++) {
      int lattice = sample;
      std::array<double, 3> q;
      double weights[3][3];
      for (int a = 0; a < 3; a++) {
        const double t = sample < 27 ? 0.25 * (1 + lattice % 3) : uniform(rng);
        lattice /= 3;
        q[a] = bounds.lower[a] + t * (bounds.upper[a] - bounds.lower[a]);
        weights[a][0] = (1 - t) * (1 - t);
        weights[a][1] = 2 * t * (1 - t);
        weights[a][2] = t * t;
      }
      auto blend = controls[0];
      std::fill(blend.matrix.begin(), blend.matrix.end(), Complex(0));
      for (int i = 0; i < 27; i++) {
        const double weight = weights[0][i % 3] * weights[1][(i / 3) % 3] * weights[2][i / 9];
        for (size_t j = 0; j < blend.matrix.size(); j++)
          blend.matrix[j] += weight * controls[i].matrix[j];
      }
      DiffractionGratingHybrid interpolated;
      interpolated.is_reference = cell.corners[0].is_reference;
      DiffractionGratingBlock reference, exact, actual;
      const double x = q[0] * q[2] / pitch;
      if (!diffraction_grating_chart_to_reference(blend, interpolated.scattering, error) ||
          !diffraction_grating_match_hybrid(profile, q[2], x, q[1], interpolated, actual, error) ||
          !diffraction_grating_solve_reference(profile, q[2], x, q[1], 16, 5, reference, error) ||
          !diffraction_grating_match_reference(profile, q[2], x, q[1], reference, exact, error))
      {
        std::cerr << error;
        return 1;
      }
      DiffractionGratingPowerBlock a, b;
      diffraction_grating_power_block(actual, a);
      diffraction_grating_power_block(exact, b);
      if (a.ports.size() != b.ports.size() || actual.matrix.size() != exact.matrix.size())
        return 1;
      const size_t n = a.ports.size();
      double sample_error = 0;
      for (size_t col = 0; col < n; col++) {
        if (a.ports[col].order != b.ports[col].order ||
            a.ports[col].substrate != b.ports[col].substrate)
          return 1;
        double column_error = 0;
        for (size_t row = 0; row < n; row++)
          column_error += std::abs(a.matrix[row * n + col] - b.matrix[row * n + col]);
        sample_error = std::max(sample_error, column_error);
      }
      double complex_error = 0;
      for (size_t j = 0; j < actual.matrix.size(); j++)
        complex_error = std::hypot(complex_error, std::abs(actual.matrix[j] - exact.matrix[j]));
      if (!std::isfinite(sample_error) || !std::isfinite(complex_error))
        return 1;
      maximum_complex_error = std::max(maximum_complex_error, complex_error);
      maximum_power_error = std::max(maximum_power_error, sample_error);
      if (sample >= 27)
        held_out_error = std::max(held_out_error, sample_error);
      maximum_gain = std::max(maximum_gain, actual.maximum_power_gain);
    }
    std::cout << (test ? "," : "") << "{\"case\":" << test << ",\"channels\":" << channels
              << ",\"nodal_relative_error\":" << nodal_error << ",\"phase\":" << phase
              << ",\"minimum_control_dissipation\":" << minimum_dissipation
              << ",\"passive_controls_with_roundoff_tolerance\":"
              << (minimum_dissipation >= -1e-9 ? "true" : "false")
              << ",\"power_max_error\":" << maximum_power_error
              << ",\"held_out_power_max_error\":" << held_out_error
              << ",\"complex_frobenius_max_error\":" << maximum_complex_error
              << ",\"maximum_polarization_gain\":" << maximum_gain << '}';
  }
  std::cout << "]}\n";
}
