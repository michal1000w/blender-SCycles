/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include <cmath>
#include <iostream>
#include <limits>

int main()
{
  using namespace ccl;
  DiffractionGratingBlock block{};
  block.ports.push_back({0, false});
  block.matrix = {{0, 1.000001}, 0, 0, {0, -0.999999}};
  std::string error;
  if (!diffraction_grating_restore_lossless_reference(block, error) ||
      std::abs(block.matrix[0] - std::complex<double>(0, 1)) > 1e-12 ||
      std::abs(block.matrix[3] - std::complex<double>(0, -1)) > 1e-12)
    return 1;
  block.matrix[0] = 1.01;
  const auto rejected = block.matrix;
  if (diffraction_grating_restore_lossless_reference(block, error) ||
      error.empty() || block.matrix != rejected)
    return 2;
  block.matrix[0] = std::numeric_limits<double>::quiet_NaN();
  if (diffraction_grating_restore_lossless_reference(block, error) ||
      error.empty() || !std::isnan(block.matrix[0].real()))
    return 3;
  block.matrix.pop_back();
  if (diffraction_grating_restore_lossless_reference(block, error) || error.empty())
    return 4;
  std::cout << "Small defect corrected; large defect, nonfinite input and invalid dimensions rejected without publication\n";
}
