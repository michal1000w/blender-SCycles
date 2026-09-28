/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Cache approximation audit over uniform incident solid angle and wavelength.
 * This is not a renderer benchmark. Both references are finite modal solves. */
#include "kernel/util/diffraction_grid.h"
#include "scene/diffraction.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

using namespace ccl;
using Clock = std::chrono::steady_clock;

struct Probe {
  float wavelength, kx, ky;
  int incoming_order;
  DiffractionGratingPowerBlock low, high;
};

int main(int argc, char **argv)
{
  const int resolution = argc > 1 ? std::atoi(argv[1]) : 16;
  const int queries = argc > 2 ? std::atoi(argv[2]) : 256;
  const bool extension = argc > 3 && std::atoi(argv[3]) != 0;
  const int wavelengths = argc > 4 ? std::atoi(argv[4]) : 9;
  const double cutoff_margin = argc > 7 ? std::atof(argv[7]) : 0.0;
  const bool hybrid_cache = argc > 6 && std::atoi(argv[6]) != 0;
  const bool reference_cache = hybrid_cache || (argc > 5 && std::atoi(argv[5]) != 0);
  if (resolution < 2 || resolution > 128 || queries < 1 || queries > 65536 || wavelengths < 2 ||
      wavelengths > 128 || !std::isfinite(cutoff_margin) || cutoff_margin < 0.0 ||
      cutoff_margin > 4.0)
  {
    return 2;
  }
  std::cout << std::setprecision(12) << "{\"cases\":[";
  bool first = true;
  for (const double pitch : {740.0, 1600.0}) {
    const std::complex<double> metal(0.9, 6.0);
    const DiffractionGratingProfile profile{pitch, 150.0, 0.41, 1.0, metal, 1.0, metal};
    const DiffractionGratingGridConfig config{
        resolution, 2 * resolution, wavelengths, 380.0, 780.0, extension};
    std::vector<float> grid;
    std::string error;
    const auto start = Clock::now();
    const int retained = int(std::ceil(pitch / config.wavelength_min - 0.5));
    std::vector<DiffractionGratingBlock> reference_grid;
    if (reference_cache) {
      for (int l = 0; l < wavelengths; l++) {
        const double lambda = 380.0 + 400.0 * l / (wavelengths - 1);
        for (int v = 0; v < 2 * resolution; v++) {
          const double ky = -1.0 + 2.0 * v / (2 * resolution - 1);
          for (int k = 0; k < resolution; k++) {
            const double kx = (-0.5 + double(k) / (resolution - 1)) * lambda / pitch;
            DiffractionGratingBlock reference;
            if (!diffraction_grating_solve_reference(
                    profile, lambda, kx, ky, 16, retained, reference, error))
            {
              std::cerr << error << '\n';
              return 1;
            }
            reference_grid.push_back(std::move(reference));
          }
        }
      }
    }
    else if (!diffraction_grating_build_grid(profile, config, 16, grid, error)) {
      std::cerr << error << '\n';
      return 1;
    }
    const double build_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    std::mt19937 rng(617923);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<double> cache_errors, truncation_errors, feedback_counts;
    std::vector<Probe> probes;
    for (int i = 0; i < queries; i++) {
      Probe probe;
      probe.wavelength = float(380.0 + 400.0 * uniform(rng));
      const double cosine = 0.0001 + 0.9999 * uniform(rng);
      const double phi = 2.0 * std::numbers::pi_v<double> * uniform(rng);
      const double sine = std::sqrt(1.0 - cosine * cosine);
      probe.kx = float(sine * std::cos(phi));
      probe.ky = float(sine * std::sin(phi));
      probe.incoming_order = int(std::floor(probe.kx * pitch / probe.wavelength + 0.5));
      const double base = double(probe.kx) - probe.incoming_order * probe.wavelength / pitch;
      for (const int n : {16, 32}) {
        DiffractionGratingBlock block;
        if (!diffraction_grating_solve_bloch(
                profile, probe.wavelength, base, probe.ky, n, block, error))
        {
          std::cerr << error << '\n';
          return 1;
        }
        diffraction_grating_power_block(block, n == 16 ? probe.low : probe.high);
      }
      int col = -1;
      for (size_t j = 0; j < probe.low.ports.size(); j++) {
        if (probe.low.ports[j].order == probe.incoming_order) {
          col = int(j);
        }
      }
      if (col < 0 || probe.low.ports.size() != probe.high.ports.size()) {
        std::cerr << "Reference port mismatch\n";
        return 1;
      }
      DiffractionGratingPowerBlock reference_power;
      if (reference_cache) {
        const double bloch = base * pitch / probe.wavelength;
        const double x = (bloch + 0.5) * (resolution - 1);
        const double y = (probe.ky + 1.0) * 0.5 * (2 * resolution - 1);
        const double z = (probe.wavelength - 380.0) / 400.0 * (wavelengths - 1);
        const int ix = std::clamp(int(x), 0, resolution - 2);
        const int iy = std::clamp(int(y), 0, 2 * resolution - 2);
        const int iz = std::clamp(int(z), 0, wavelengths - 2);
        const double tx = x - ix, ty = y - iy, tz = z - iz;
        std::vector<unsigned char> keep(reference_grid.front().ports.size(), 0);
        if (hybrid_cache) {
          auto square_bounds = [](double lo, double hi) {
            return std::pair<double, double>{lo <= 0.0 && hi >= 0.0 ? 0.0 :
                                                                      std::min(lo * lo, hi * hi),
                                             std::max(lo * lo, hi * hi)};
          };
          const double bloch_lo = -0.5 + double(ix) / (resolution - 1);
          const double bloch_hi = -0.5 + double(ix + 1) / (resolution - 1);
          const double lambda_lo = 380.0 + 400.0 * iz / (wavelengths - 1);
          const double lambda_hi = 380.0 + 400.0 * (iz + 1) / (wavelengths - 1);
          const auto y2 = square_bounds(-1.0 + 2.0 * iy / (2 * resolution - 1),
                                        -1.0 + 2.0 * (iy + 1) / (2 * resolution - 1));
          int active = 0;
          for (size_t p = 0; p < keep.size(); p++) {
            const int order = reference_grid.front().ports[p].order;
            const std::array<double, 4> corners{(bloch_lo + order) * lambda_lo / pitch,
                                                (bloch_hi + order) * lambda_lo / pitch,
                                                (bloch_lo + order) * lambda_hi / pitch,
                                                (bloch_hi + order) * lambda_hi / pitch};
            const auto bounds = std::minmax_element(corners.begin(), corners.end());
            const auto x2 = square_bounds(*bounds.first, *bounds.second);
            const double q2_lo = 1.0 - x2.second - y2.second;
            const double q2_hi = 1.0 - x2.first - y2.first;
            keep[p] = q2_lo <= cutoff_margin + 1e-12 && q2_hi >= -cutoff_margin - 1e-12;
            active += 2 * keep[p];
          }
          feedback_counts.push_back(active);
          reference_power = probe.low;
          std::fill(reference_power.matrix.begin(), reference_power.matrix.end(), 0.0);
        }
        DiffractionGratingBlock interpolated = reference_grid.front();
        std::fill(
            interpolated.matrix.begin(), interpolated.matrix.end(), std::complex<double>(0.0));
        for (int a = 0; a < 2; a++) {
          for (int b = 0; b < 2; b++) {
            for (int c = 0; c < 2; c++) {
              const double weight = (a ? tx : 1.0 - tx) * (b ? ty : 1.0 - ty) *
                                    (c ? tz : 1.0 - tz);
              const size_t node = (size_t(iz + c) * (2 * resolution) + iy + b) * resolution + ix +
                                  a;
              if (hybrid_cache) {
                const double lambda = 380.0 + 400.0 * (iz + c) / (wavelengths - 1);
                const double corner_y = -1.0 + 2.0 * (iy + b) / (2 * resolution - 1);
                const double corner_x = (-0.5 + double(ix + a) / (resolution - 1)) * lambda /
                                        pitch;
                DiffractionGratingHybrid hybrid;
                DiffractionGratingBlock matched;
                if (!diffraction_grating_prepare_hybrid(profile,
                                                        lambda,
                                                        corner_x,
                                                        corner_y,
                                                        reference_grid[node],
                                                        keep,
                                                        hybrid,
                                                        error) ||
                    !diffraction_grating_match_hybrid(
                        profile, probe.wavelength, base, probe.ky, hybrid, matched, error))
                {
                  std::cerr << error << '\n';
                  return 1;
                }
                if (matched.maximum_power_gain > 1.0 + 1e-8) {
                  std::cerr << "Hybrid corner violates passivity\n";
                  return 1;
                }
                DiffractionGratingPowerBlock power;
                diffraction_grating_power_block(matched, power);
                if (power.ports.size() != reference_power.ports.size()) {
                  std::cerr << "Hybrid physical port mismatch\n";
                  return 1;
                }
                for (size_t j = 0; j < power.matrix.size(); j++) {
                  reference_power.matrix[j] += weight * power.matrix[j];
                }
              }
              else {
                for (size_t j = 0; j < interpolated.matrix.size(); j++) {
                  interpolated.matrix[j] += weight * reference_grid[node].matrix[j];
                }
              }
            }
          }
        }
        if (!hybrid_cache) {
          DiffractionGratingBlock physical;
          if (!diffraction_grating_match_reference(
                  profile, probe.wavelength, base, probe.ky, interpolated, physical, error))
          {
            std::cerr << error << '\n';
            return 1;
          }
          if (physical.maximum_power_gain > 1.0 + 1e-8) {
            std::cerr << "Interpolated reference operator violates passivity\n";
            return 1;
          }
          diffraction_grating_power_block(physical, reference_power);
        }
        if (reference_power.ports.size() != probe.low.ports.size()) {
          std::cerr << "Interpolated reference port mismatch\n";
          return 1;
        }
      }
      double cache_error = 0.0, truncation_error = 0.0;
      for (size_t row = 0; row < probe.low.ports.size(); row++) {
        float value;
        if (reference_cache) {
          value = float(reference_power.matrix[row * reference_power.ports.size() + col]);
        }
        else if (!diffraction_grid_power(grid.data(),
                                         probe.wavelength,
                                         probe.kx,
                                         probe.ky,
                                         false,
                                         probe.low.ports[row].order - probe.incoming_order,
                                         false,
                                         &value))
        {
          std::cerr << "Invalid cache probe\n";
          return 1;
        }
        const size_t offset = row * probe.low.ports.size() + col;
        cache_error += std::abs(value - probe.low.matrix[offset]);
        truncation_error += std::abs(probe.low.matrix[offset] - probe.high.matrix[offset]);
      }
      cache_errors.push_back(cache_error);
      truncation_errors.push_back(truncation_error);
      probes.push_back(std::move(probe));
    }
    auto stats = [](const std::vector<double> &values) {
      auto sorted = values;
      std::sort(sorted.begin(), sorted.end());
      double sum = 0.0;
      for (const double value : values) {
        sum += value;
      }
      std::cout << "{\"mean\":" << sum / values.size()
                << ",\"p95\":" << sorted[size_t(0.95 * (sorted.size() - 1))]
                << ",\"maximum\":" << sorted.back() << '}';
    };
    std::cout << (first ? "" : ",") << "{\"pitch_nm\":" << pitch
              << ",\"bloch_samples\":" << resolution << ",\"tangent_samples\":" << 2 * resolution
              << ",\"wavelength_samples\":" << wavelengths
              << ",\"build_seconds\":" << build_seconds
              << ",\"packed_bytes\":" << grid.size() * sizeof(float)
              << ",\"float_matrix_bytes_estimate\":"
              << (reference_cache ? reference_grid.size() * reference_grid.front().matrix.size() *
                                        2 * sizeof(float) :
                                    0)
              << ",\"host_matrix_bytes\":"
              << (reference_cache ? reference_grid.size() * reference_grid.front().matrix.size() *
                                        sizeof(std::complex<double>) :
                                    0)
              << ",\"reference_cache\":" << (reference_cache ? "true" : "false")
              << ",\"retained_half_orders\":" << retained
              << ",\"closed_port_reflection\":" << (extension ? "true" : "false")
              << ",\"cutoff_margin\":" << cutoff_margin << ",\"query_count\":" << queries
              << ",\"cache_vs_N16_L1\":";
    stats(cache_errors);
    std::cout << ",\"hybrid_cache\":" << (hybrid_cache ? "true" : "false");
    if (hybrid_cache) {
      std::cout << ",\"feedback_channels\":";
      stats(feedback_counts);
    }
    std::cout << ",\"N16_vs_N32_L1\":";
    stats(truncation_errors);
    std::cout << ",\"probes\":[";
    for (size_t i = 0; i < probes.size(); i++) {
      const auto &p = probes[i];
      std::cout << (i ? "," : "") << "{\"wavelength_nm\":" << p.wavelength << ",\"kx\":" << p.kx
                << ",\"ky\":" << p.ky << ",\"cache_L1\":" << cache_errors[i]
                << ",\"truncation_L1\":" << truncation_errors[i] << '}';
    }
    std::cout << "]}";
    first = false;
  }
  std::cout << "]}\n";
}
