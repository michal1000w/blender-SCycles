/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Joint-coordinate reference data; this experiment is not a production cache. */
#include "scene/diffraction.h"
#include <charconv>
#include <iomanip>
#include <iostream>
#include <random>
using namespace ccl;
int main(int argc, char **argv)
{
  constexpr int side = 7, held_out = 128;
  const DiffractionGratingProfile profile{740, 150, .41, 1, 1.5, 1, 1};
  unsigned seed = 20260926;
  if (argc > 3 && argc != 8) return 2;
  const bool full_domain = argc == 3 && std::string(argv[2]) == "full";
  if (argc == 3 && !full_domain) return 2;
  if (argc >= 2) {
    const std::string text = argv[1];
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), seed);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) return 2;
  }
  std::mt19937 random(seed);
  double lower[3]={full_domain ? -.5 : -.1, full_domain ? -1 : -.98, 380};
  double upper[3]={full_domain ? .5 : 0, full_domain ? 1 : -.90, full_domain ? 780.0 : 400.0};
  if (argc == 8) {
    for (int i=0;i<6;++i) {
      const std::string text=argv[i+2];
      double &value=i<3?lower[i]:upper[i-3];
      const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
      if(parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size() || !std::isfinite(value)) return 2;
    }
    for(int i=0;i<3;++i) if(!(lower[i]<upper[i])) return 2;
    if(lower[0]<-.5 || upper[0]>.5 || lower[1]<-1 || upper[1]>1 || lower[2]<380 || upper[2]>780) return 2;
  }
  std::cout << std::setprecision(17)
            << "{\"scope\":\"Joint-coordinate N16 lossless reference experiment\","
            << "\"grid_side\":7,\"half_orders\":16,\"retained_half_orders\":2,"
            << "\"lower\":["<<lower[0]<<','<<lower[1]<<','<<lower[2]
            << "],\"upper\":["<<upper[0]<<','<<upper[1]<<','<<upper[2]<<"],\"seed\":"
            << seed << ",\"samples\":[";
  for (int i = 0; i < side * side * side + held_out; ++i) {
    const bool training = i < side * side * side;
    int index = i;
    double t[3];
    for (int a = 0; a < 3; ++a) {
      /* Explicit generator scaling is reproducible across C++ libraries. */
      t[a] = training ? double(index % side) / (side - 1) :
                        (double(random()) + .5) / 4294967296.0;
      index /= side;
    }
    const double b = lower[0]+(upper[0]-lower[0])*t[0];
    const double y = lower[1]+(upper[1]-lower[1])*t[1];
    const double w = lower[2]+(upper[2]-lower[2])*t[2];
    DiffractionGratingBlock block;
    std::string error;
    if (!diffraction_grating_solve_reference(profile, w, b * w / profile.pitch,
                                             y, 16, 2, block, error)) {
      std::cerr << error << '\n';
      return 1;
    }
    std::cout << (i ? "," : "") << "{\"query\":[" << b << ',' << y << ',' << w
              << "],\"fitting_sample\":" << (training ? "true" : "false")
              << ",\"matrix\":[";
    for (size_t j = 0; j < block.matrix.size(); ++j)
      std::cout << (j ? "," : "") << '[' << block.matrix[j].real() << ','
                << block.matrix[j].imag() << ']';
    std::cout << "]}";
    if (i % 64 == 0) std::cerr << "completed=" << i + 1 << '\n';
  }
  std::cout << "]}\n";
}
