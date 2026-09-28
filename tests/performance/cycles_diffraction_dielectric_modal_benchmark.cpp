/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace ccl;

/* Host preparation benchmark. Each case retains the full physical operator so
 * an eigensolver optimization can be checked independently of its timing. */
int main()
{
  const DiffractionGratingProfile profile{740, 150, double(float(.41)), 1, 1.5, 1, 1};
  std::cout << std::setprecision(17) << "{\"scope\":\"CPU modal preparation\",\"runs\":[";
  bool first = true, failed = false;
  for (int repeat = 0; repeat < 5; repeat++) {
    for (double ky : {0.0, -0.16666666666666663}) {
      for (int modes : {64, 128, 256}) {
        DiffractionGratingBlock reference, physical;
        std::string error;
        const auto started = std::chrono::steady_clock::now();
        const bool success = diffraction_grating_solve_reference(
                                 profile, 580, -0.2286036036036036, ky, modes, 2,
                                 reference, error) &&
                             diffraction_grating_match_reference(
                                 profile, 580, -0.2286036036036036, ky,
                                 reference, physical, error);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        if (!first) std::cout << ',';
        first = false;
        std::cout << "{\"repeat\":" << repeat << ",\"warmup\":"
                  << (repeat == 0 ? "true" : "false") << ",\"half_orders\":" << modes
                  << ",\"ky\":" << ky << ",\"seconds\":" << seconds;
        if (!success) {
          failed = true;
          std::cout << ",\"error\":" << std::quoted(error) << "}" << std::flush;
          continue;
        }
        std::cout << ",\"minimum_power_gain\":" << physical.minimum_power_gain
                  << ",\"maximum_power_gain\":" << physical.maximum_power_gain
                  << ",\"ports\":[";
        for (size_t i = 0; i < physical.ports.size(); i++) {
          if (i) std::cout << ',';
          std::cout << '[' << physical.ports[i].order << ','
                    << (physical.ports[i].substrate ? 1 : 0) << ']';
        }
        std::cout << "],\"matrix\":[";
        for (size_t i = 0; i < physical.matrix.size(); i++) {
          if (i) std::cout << ',';
          std::cout << '[' << physical.matrix[i].real() << ','
                    << physical.matrix[i].imag() << ']';
        }
        std::cout << "]}" << std::flush;
      }
    }
  }
  std::cout << "]}\n";
  return failed ? 1 : 0;
}
