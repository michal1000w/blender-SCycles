/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Standalone host response-generation benchmark. See the adjacent Python driver.
 * This measures precomputation, not rendering or GPU shading performance. */
#include "scene/diffraction.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace ccl;
using Clock = std::chrono::steady_clock;

int main(int argc, char **argv)
{
  const int repeats = argc > 1 ? std::atoi(argv[1]) : 3;
  if (repeats < 1 || repeats > 100) {
    return 2;
  }
  std::cout << std::setprecision(12) << "{\"cases\":[";
  bool first = true;
  for (const bool metal : {false, true}) {
    const std::complex<double> material = metal ? std::complex<double>(0.9, 6.0) : 1.5;
    for (const double pitch : {740.0, 1600.0}) {
      for (const int half_orders : {16, 32, 64}) {
        const DiffractionGratingProfile profile{pitch, 150.0, 0.41, 1.0, material, 1.0, material};
        const double wavelength = 550.0;
        const double base_kx = 0.19 * wavelength / pitch, ky = 0.23;
        DiffractionGratingBlock block;
        std::string error;
        if (!diffraction_grating_solve_bloch(
                profile, wavelength, base_kx, ky, half_orders, block, error))
        {
          std::cerr << error << '\n';
          return 1;
        }
        std::vector<double> angles, azimuths;
        for (const auto &port : block.ports) {
          if (!port.substrate) {
            const double x = base_kx + port.order * wavelength / pitch;
            angles.push_back(std::asin(std::hypot(x, ky)));
            azimuths.push_back(std::atan2(ky, x));
          }
        }
        std::vector<double> block_ms, separate_ms;
        DiffractionGratingResponse response;
        auto solve_block = [&]() {
          const auto start = Clock::now();
          if (!diffraction_grating_solve_bloch(
                  profile, wavelength, base_kx, ky, half_orders, block, error))
          {
            std::cerr << error << '\n';
            std::exit(1);
          }
          block_ms.push_back(
              std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        };
        auto solve_separate = [&]() {
          const auto start = Clock::now();
          for (size_t i = 0; i < angles.size(); i++) {
            if (!diffraction_grating_solve(
                    profile, wavelength, angles[i], azimuths[i], half_orders, response, error))
            {
              std::cerr << error << '\n';
              std::exit(1);
            }
          }
          separate_ms.push_back(
              std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        };
        /* Alternate order to reduce one systematic source of timing bias. */
        for (int i = 0; i < repeats; i++) {
          if (i % 2) {
            solve_separate();
            solve_block();
          }
          else {
            solve_block();
            solve_separate();
          }
        }
        DiffractionGratingPowerBlock power;
        diffraction_grating_power_block(block, power);
        auto numbers = [](const std::vector<double> &values) {
          std::cout << '[';
          for (size_t i = 0; i < values.size(); i++) {
            std::cout << (i ? "," : "") << values[i];
          }
          std::cout << ']';
        };
        std::cout << (first ? "" : ",") << "{\"metal\":" << (metal ? "true" : "false")
                  << ",\"pitch_nm\":" << pitch << ",\"half_orders\":" << half_orders
                  << ",\"top_incident_directions\":" << angles.size()
                  << ",\"all_ports\":" << block.ports.size()
                  << ",\"power_matrix_float_bytes\":" << power.matrix.size() * sizeof(float)
                  << ",\"jones_matrix_float_bytes\":" << block.matrix.size() * 2 * sizeof(float)
                  << ",\"maximum_power_gain\":" << block.maximum_power_gain << ",\"block_ms\":";
        numbers(block_ms);
        std::cout << ",\"separate_top_incidence_ms\":";
        numbers(separate_ms);
        std::cout << '}';
        first = false;
      }
    }
  }
  std::cout << "]}\n";
}
