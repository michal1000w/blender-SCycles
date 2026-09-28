/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#include "device/device.h"
#include "kernel/util/dielectric_dispersion.h"
#include "scene/diffraction_albedo.h"
#include "util/profiling.h"
#include "util/path.h"
#include "util/stats.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

using namespace ccl;

struct CacheComparison {
  double cpu_seconds = 0.0;
  double metal_cold_seconds = 0.0;
  double metal_warm_seconds = 0.0;
  double max_directional_error = 0.0;
  double max_integral_error = 0.0;
  double max_cross_error = 0.0;
  bool passed = false;
};

static double elapsed(const std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static CacheComparison compare(Device *device, const DiffractionTwoSidedAlbedoRequest &request)
{
  CacheComparison result;
  DiffractionTwoSidedAlbedoTable cpu, metal, repeated;
  std::string error;
  auto start = std::chrono::steady_clock::now();
  if (!diffraction_two_sided_albedo_build_cpu(request, cpu, error)) {
    std::cerr << "CPU cache build failed: " << error << '\n';
    return result;
  }
  result.cpu_seconds = elapsed(start);
  start = std::chrono::steady_clock::now();
  if (!device->build_diffraction_two_sided_albedo(request, metal, error)) {
    std::cerr << "Metal cache build failed: " << error << '\n';
    return result;
  }
  result.metal_cold_seconds = elapsed(start);
  start = std::chrono::steady_clock::now();
  if (!device->build_diffraction_two_sided_albedo(request, repeated, error)) {
    std::cerr << "Metal warm cache build failed: " << error << '\n';
    return result;
  }
  result.metal_warm_seconds = elapsed(start);
  if (!(cpu.request == metal.request) || !(metal.request == repeated.request) ||
      cpu.reflectance.size() != metal.reflectance.size() ||
      cpu.integrals.size() != metal.integrals.size() ||
      cpu.cross_fractions.size() != metal.cross_fractions.size()) {
    std::cerr << "Cache dimensions/request disagree\n";
    return result;
  }
  for (size_t i = 0; i < cpu.reflectance.size(); ++i) {
    result.max_directional_error = std::max({result.max_directional_error,
        std::abs(double(cpu.reflectance[i]) - metal.reflectance[i]),
        std::abs(double(cpu.transmittance[i]) - metal.transmittance[i]),
        std::abs(double(cpu.deficits[i]) - metal.deficits[i]),
        std::abs(double(metal.reflectance[i]) - repeated.reflectance[i]),
        std::abs(double(metal.transmittance[i]) - repeated.transmittance[i])});
  }
  for (size_t i = 0; i < cpu.integrals.size(); ++i) {
    result.max_integral_error = std::max(result.max_integral_error,
        std::abs(double(cpu.integrals[i]) - metal.integrals[i]));
  }
  for (size_t i = 0; i < cpu.cross_fractions.size(); ++i) {
    result.max_cross_error = std::max(result.max_cross_error,
        std::abs(double(cpu.cross_fractions[i]) - metal.cross_fractions[i]));
  }
  result.passed = result.max_directional_error <= 0.002 &&
                  result.max_integral_error <= 0.002 &&
                  result.max_cross_error <= 1e-6;
  return result;
}

int main(int argc, char **argv)
{
  if (argc != 3) {
    std::cerr << "Usage: cycles_diffraction_two_sided_gpu_cache_test OUTPUT.json SOURCE_ROOT\n"
                 "SOURCE_ROOT must contain source/kernel/device/metal/diffraction_albedo.metal\n";
    return 2;
  }
  path_init(argv[2]);
  const auto devices = Device::available_devices(DEVICE_MASK_METAL);
  if (devices.empty()) {
    std::cerr << "No Metal device available\n";
    return 2;
  }
  Stats stats;
  Profiler profiler;
  auto device = Device::create(devices.front(), stats, profiler, true);
  if (!device || device->have_error()) {
    std::cerr << "Metal device creation failed\n";
    return 2;
  }
  DiffractionTwoSidedAlbedoRequest uncoated;
  uncoated.wavelength_count = 16;
  DiffractionTwoSidedAlbedoRequest coated = uncoated;
  coated.film_ior = 1.5f;
  coated.film_thickness_nm = 500.0f;
  coated.wavelength_count = 40;
  const CacheComparison a = compare(device.get(), uncoated);
  const CacheComparison b = compare(device.get(), coated);
  DiffractionTwoSidedAlbedoRequest dispersive = uncoated;
  dispersive.generalized_f0_count = 2;
  dispersive.inv_abbe = 0.05f;
  const CacheComparison c = compare(device.get(), dispersive);
  DiffractionTwoSidedAlbedoRequest matched = dispersive;
  const float unit_effect = dielectric_ior_at_wavelength(
      matched.inside_ior, 1.0f, dielectric_wavelength_um(780.0f)) - matched.inside_ior;
  matched.inv_abbe = (1.0f - matched.inside_ior) / unit_effect;
  for (int i = 0; i < 128; ++i) {
    const float n = dielectric_ior_at_wavelength(matched.inside_ior, matched.inv_abbe,
                                                dielectric_wavelength_um(780.0f));
    if (n == 1.0f) break;
    matched.inv_abbe = std::nextafter(matched.inv_abbe, n > 1.0f ? INFINITY : -INFINITY);
  }
  if (dielectric_ior_at_wavelength(matched.inside_ior, matched.inv_abbe,
                                  dielectric_wavelength_um(780.0f)) != 1.0f) return 2;
  const CacheComparison d = compare(device.get(), matched);
  DiffractionTwoSidedAlbedoRequest subair = dispersive;
  subair.inv_abbe = matched.inv_abbe * 1.01f;
  const CacheComparison e = compare(device.get(), subair);
  DiffractionTwoSidedAlbedoRequest coated_generalized = dispersive;
  coated_generalized.generalized_f0_count = 16;
  coated_generalized.film_ior = 1.32f;
  coated_generalized.film_thickness_nm = 250.0f;
  const CacheComparison f = compare(device.get(), coated_generalized);
  DiffractionTwoSidedAlbedoRequest coated_matched = coated_generalized;
  coated_matched.inv_abbe = matched.inv_abbe;
  const CacheComparison g = compare(device.get(), coated_matched);
  std::ofstream out(argv[1]);
  if (!out) return 2;
  out.precision(9);
  out << "{\n  \"device\": \"" << devices.front().description << "\",\n"
      << "  \"scope\": \"two-sided CPU versus Metal cache, 512 facet samples\",\n";
  auto write = [&](const char *name, const CacheComparison &r,
                   const DiffractionTwoSidedAlbedoRequest &request, const bool last) {
    out << "  \"" << name << "\": {\"cpu_seconds\": " << r.cpu_seconds
        << ", \"metal_cold_seconds\": " << r.metal_cold_seconds
        << ", \"metal_warm_seconds\": " << r.metal_warm_seconds
        << ", \"max_directional_error\": " << r.max_directional_error
        << ", \"max_integral_error\": " << r.max_integral_error
        << ", \"max_cross_error\": " << r.max_cross_error
        << ", \"inside_ior\": " << request.inside_ior
        << ", \"inv_abbe\": " << request.inv_abbe
        << ", \"generalized_f0_count\": " << request.generalized_f0_count
        << ", \"wavelength_count\": " << request.wavelength_count
        << ", \"film_ior\": " << request.film_ior
        << ", \"film_thickness_nm\": " << request.film_thickness_nm
        << ", \"passed\": " << (r.passed ? "true" : "false") << "}"
        << (last ? "\n" : ",\n");
  };
  write("uncoated_16_lambda", a, uncoated, false);
  write("coated_40_lambda", b, coated, false);
  write("generalized_dispersion_2basis_16lambda", c, dispersive, false);
  write("generalized_matched_2basis_16lambda", d, matched, false);
  write("generalized_subair_2basis_16lambda", e, subair, false);
  write("coated_generalized_16basis_16lambda", f, coated_generalized, false);
  write("coated_matched_16basis_16lambda", g, coated_matched, true);
  out << "}\n";
  std::cout << "uncoated error " << a.max_directional_error << ", coated error "
            << b.max_directional_error << '\n';
  return a.passed && b.passed && c.passed && d.passed && e.passed && f.passed && g.passed ? 0 : 1;
}
