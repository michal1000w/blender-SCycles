/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Fixed-topology reference-operator data for an offline interpolation experiment.
 * This is not a physical BSDF cache and does not establish modal convergence. */
#include "scene/diffraction.h"
#include <iomanip>
#include <string>
#include <iostream>
using namespace ccl;
int main(int argc, char **argv)
{
  const std::string axis = argc == 2 ? argv[1] : "wavelength";
  if (axis != "wavelength" && axis != "bloch" && axis != "ky") return 2;
  const DiffractionGratingProfile profile{740, 150, .41, 1, 1.5, 1, 1};
  const double bloch = -.0517578125, ky = -.946533203125;
  std::cout << std::setprecision(17)
            << "{\"scope\":\"N16 reference operators; not physical-power validation\","
            << "\"pitch_nm\":740,\"depth_nm\":150,\"duty\":0.41,\"ridge_ior\":1.5,"
            << "\"half_orders\":16,\"retained_half_orders\":2,\"bloch\":" << bloch
            << ",\"ky\":" << ky << ",\"axis\":" << std::quoted(axis) << ",\"samples\":[";
  for (int i = 0; i <= 160; ++i) {
    const double wavelength = axis == "wavelength" ? 380 + 20.0 * i / 160 : 384.8;
    const double b = axis == "bloch" ? -.1 + .1 * i / 160 : bloch;
    const double y = axis == "ky" ? -.98 + .08 * i / 160 : ky;
    const double coordinate = axis == "wavelength" ? wavelength : (axis == "bloch" ? b : y);
    DiffractionGratingBlock block;
    std::string error;
    if (!diffraction_grating_solve_reference(
            profile, wavelength, b * wavelength / profile.pitch, y, 16, 2, block, error)) {
      std::cerr << error << '\n';
      return 1;
    }
    std::cout << (i ? "," : "") << "{\"wavelength_nm\":" << wavelength
              << ",\"coordinate\":" << coordinate
              << ",\"fitting_sample\":" << (i % 2 ? "false" : "true")
              << ",\"matrix\":[";
    for (size_t j = 0; j < block.matrix.size(); ++j) {
      std::cout << (j ? "," : "") << '[' << block.matrix[j].real() << ','
                << block.matrix[j].imag() << ']';
    }
    std::cout << "]}";
  }
  std::cout << "]}\n";
}
