/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include <iostream>
#include <iomanip>
using namespace ccl;
int main()
{
  std::cout << std::setprecision(17);
  for (const bool metal : {false, true})
    for (const double depth : {0., 150.})
      for (const double angle : {0., .7, 1.3})
        for (const double azimuth : {0., .6})
          for (const int modes : {16, 32}) {
            const std::complex<double> index = metal ? std::complex<double>(.7, 5.) : 1.5;
            const DiffractionGratingProfile profile{740, depth, .41, 1, index, 1,
                                                     metal ? index : 1.};
            DiffractionGratingResponse result;
            std::string error;
            if (!diffraction_grating_solve(profile, 551, angle, azimuth, modes, result, error)) {
              std::cerr << error;
              return 1;
            }
            for (const auto &order : result.orders) {
              for (const auto &value : order.reflection_jones)
                std::cout << value.real() << ' ' << value.imag() << '\n';
              for (const auto &value : order.transmission_jones)
                std::cout << value.real() << ' ' << value.imag() << '\n';
            }
          }
}
