/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction_index.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

namespace ccl {
struct TestIndexGlobals { const float *lookup_table; };
using KernelGlobals = const TestIndexGlobals *;
}
#define kernel_data_fetch(name, index) (kg->name[index])
#include "kernel/util/diffraction_index.h"
#undef kernel_data_fetch
using namespace ccl;

int main(int argc, char **argv)
{
  int failures = 0, checked = 0, rejected = 0;
  double maximum_error = 0;
  const std::vector<DiffractionIndexSample> samples = {
      {380, {1, .1}}, {517.3, {1.3, 2}}, {780, {2.6, 4}}};
  vector<float> packed;
  std::string error;
  failures += !diffraction_pack_index_table(samples, .7, false, true, packed, error);
  vector<float> buffer{99, 88, 77};
  buffer.insert(buffer.end(), packed.begin(), packed.end());
  TestIndexGlobals globals{buffer.data()};
  for (int i = 0; i <= 40000; ++i) {
    const float wavelength = 380 + float(i) * .01f;
    float2 actual;
    if (!diffraction_index_lookup(&globals, 3, wavelength, &actual)) {
      ++failures;
      continue;
    }
    const int hi = wavelength <= samples[1].wavelength ? 1 : 2;
    const auto &a = samples[hi - 1], &b = samples[hi];
    const double t = (double(wavelength) - a.wavelength) / (b.wavelength - a.wavelength);
    const auto expected = a.index * (1 - t) + b.index * t;
    const double e = std::max(std::abs(double(actual.x) + .7 - expected.real()),
                              std::abs(double(actual.y) - expected.imag()));
    maximum_error = std::max(maximum_error, e);
    failures += e > 3e-6;
    ++checked;
  }
  float2 value;
  failures += diffraction_index_lookup(&globals, 3, std::nextafter(380.f, 0.f), &value);
  failures += diffraction_index_lookup(&globals, 3, std::nextafter(780.f, 1000.f), &value);
  failures += diffraction_index_lookup(&globals, -1, 500, &value);
  failures += diffraction_index_lookup(&globals, 3, NAN, &value);
  const auto reject = [&](std::vector<DiffractionIndexSample> bad, bool lossless = false,
                          bool absorbing = false) {
    vector<float> result{42};
    const bool accepted = diffraction_pack_index_table(bad, 0, lossless, absorbing, result, error);
    failures += accepted || error.empty() || result.size() != 1 || result[0] != 42;
    ++rejected;
  };
  reject({});
  reject({samples[0]});
  auto bad = samples; bad[1].wavelength = 380; reject(bad);
  bad = samples; bad.insert(bad.begin() + 2, {517.300000001, {1, 1}}); reject(bad);
  bad = samples; bad[0].wavelength = 380.000000001; reject(bad);
  bad = samples; bad[2].wavelength = 779; reject(bad);
  bad = samples; bad[1].index = {-1, 1}; reject(bad);
  bad = samples; bad[1].index = {1, -.1}; reject(bad);
  bad = samples; bad[1].index = {NAN, 1}; reject(bad);
  bad = samples; bad[1].wavelength = INFINITY; reject(bad);
  reject(samples, true, false);
  bad = samples; bad[1].index = {1, 0}; reject(bad, false, true);
  const std::string valid_csv = "# Units are explicit\nwavelength_nm,n,k\n380,1,2\n780,2,3 # end\n";
  std::vector<DiffractionIndexSample> parsed;
  failures += !diffraction_parse_conductor_csv(valid_csv, parsed, error) || parsed.size() != 2;
  for (const std::string csv : {"wavelength_um,n,k\n.38,1,2\n.78,2,3",
                                "wavelength_nm,n,k\n380,,2\n780,2,3",
                                "wavelength_nm,n,k\n380,1,2,4\n780,2,3",
                                "wavelength_nm,n,k\n380,1,2x\n780,2,3",
                                "wavelength_nm,n,k\n380,1,0\n780,2,3"}) {
    const auto before = parsed;
    failures += diffraction_parse_conductor_csv(csv, parsed, error) || parsed != before;
    ++rejected;
  }
  if (argc != 2) return 2;
  std::ifstream file(argv[1]);
  const std::string contents((std::istreambuf_iterator<char>(file)), {});
  failures += !diffraction_parse_conductor_csv(contents, parsed, error) || parsed.size() != 206;
  std::cout << std::setprecision(12) << "{\"failures\":" << failures
            << ",\"interpolation_queries\":" << checked
            << ",\"invalid_tables_rejected\":" << rejected
            << ",\"maximum_index_error\":" << maximum_error
            << ",\"aluminum_samples\":" << parsed.size() << "}\n";
  return failures ? 1 : 0;
}
