/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/util/diffraction_cache.h"
#include "kernel/util/diffraction_evaluate.h"
#include "scene/diffraction.h"
#include <algorithm>
#include <iomanip>
#include <random>
#include <sstream>

namespace diffraction_audit {
using namespace ccl;
using Complex = std::complex<double>;
/* Compare the union of stored and reference ports. A closed/extra stored
 * port has zero reference power; omitting it would hide spurious energy. */
inline bool power_column_error(const std::vector<DiffractionGratingPort> &stored,
                               const float *values,
                               const DiffractionGratingPowerBlock &reference,
                               int incoming,
                               double &error)
{
  error = 0;
  const size_t n = reference.ports.size();
  if (incoming < 0 || size_t(incoming) >= n || reference.matrix.size() != n * n)
    return false;
  std::vector<bool> seen(n, false);
  for (size_t out = 0; out < stored.size(); out++) {
    if (!std::isfinite(values[out]) || values[out] < 0)
      return false;
    double expected = 0;
    for (size_t j = 0; j < n; j++) {
      if (stored[out].order == reference.ports[j].order &&
          stored[out].substrate == reference.ports[j].substrate)
      {
        if (seen[j])
          return false;
        seen[j] = true;
        expected = reference.matrix[j * n + incoming];
        break;
      }
    }
    if (!std::isfinite(expected) || expected < 0)
      return false;
    error += std::abs(double(values[out]) - expected);
  }
  return std::all_of(seen.begin(), seen.end(), [](bool value) { return value; });
}
inline bool chart_match(const DiffractionGratingPackedCell &cell,
                        const float2 *boundary,
                        int incoming,
                        float3 t,
                        float2 *output)
{
  return diffraction_cache_chart_match<20>(cell.matrices.data(),
                                           boundary,
                                           int(cell.ports.size()) * 2,
                                           cell.chart_degree,
                                           cell.chart_rotation,
                                           incoming,
                                           t,
                                           output);
}
inline bool intensity_match(const DiffractionGratingPackedCell &cell,
                            const float2 *boundary,
                            int incoming,
                            float3 t,
                            float *output)
{
  return diffraction_cache_intensity_match<20>(cell.matrices.data(),
                                               boundary,
                                               cell.active_ports.data(),
                                               int(cell.active_ports.size()) * 2,
                                               int(cell.ports.size()) * 2,
                                               incoming,
                                               t,
                                               output);
}
inline std::string run(const DiffractionGratingProfile &p,
                       const DiffractionGratingCacheOptions &options,
                       const DiffractionGratingCache &cache,
                       int count,
                       unsigned seed,
                       bool sample_cache_bounds = false)
{
  std::mt19937 random(seed);
  std::uniform_real_distribution<double> uniform(0, 1);
  std::vector<double> errors;
  int failures = 0, empty = 0, over_tolerance = 0;
  std::array<double, 3> worst{};
  double maximum = 0;
  std::string first_error;
  auto fail = [&](const std::string &error) {
    failures++;
    if (first_error.empty())
      first_error = error;
  };
  for (int sample = 0; sample < count; sample++) {
    float3 original = make_float3(float(uniform(random) - 0.5),
                                  float(2 * uniform(random) - 1),
                                  float(380 + 400 * uniform(random)));
    if (sample_cache_bounds) {
      original = make_float3(
          float(cache.bounds.lower[0] + uniform(random) *
                    (cache.bounds.upper[0] - cache.bounds.lower[0])),
          float(cache.bounds.lower[1] + uniform(random) *
                    (cache.bounds.upper[1] - cache.bounds.lower[1])),
          float(cache.bounds.lower[2] + uniform(random) *
                    (cache.bounds.upper[2] - cache.bounds.lower[2])));
    }
    /* Half the probes target independent Rayleigh thresholds, with signed
     * offsets on either side. They are not builder acceptance coordinates. */
    if (!sample_cache_bounds && sample % 2) {
      const int order = int(random() % unsigned(2 * options.retained_half_orders + 1)) -
                        options.retained_half_orders;
      const double x = (double(original.x) + order) * original.z / p.pitch;
      if (std::abs(x) < 1) {
        const double delta = (sample % 4 == 1 ? -1 : 1) * std::pow(10.0, -3 - 4 * uniform(random));
        original.y = float(std::clamp(std::sqrt(1 - x * x) + delta, 0.0, 1.0) *
                           (random() % 2 ? 1 : -1));
      }
    }
    const float3 q = cache.mirror_symmetry ? diffraction_cache_symmetry(original).query : original;
    DiffractionGratingBlock reference, exact;
    DiffractionGratingPowerBlock power;
    std::string error;
    const double x = double(q.x) * q.z / p.pitch;
    if (!diffraction_grating_solve_reference(
            p, q.z, x, q.y, options.half_orders, options.retained_half_orders, reference, error) ||
        !diffraction_grating_match_reference(p, q.z, x, q.y, reference, exact, error))
    {
      fail(error);
      continue;
    }
    if (exact.ports.empty()) {
      empty++;
      continue;
    }
    diffraction_grating_power_block(exact, power);
    const int leaf = diffraction_cache_lookup(
        cache.nodes.data(),
        int(cache.nodes.size()),
        make_float3(cache.bounds.lower[0], cache.bounds.lower[1], cache.bounds.lower[2]),
        make_float3(cache.bounds.upper[0], cache.bounds.upper[1], cache.bounds.upper[2]),
        original,
        cache.mirror_symmetry);
    if (leaf < 0 || size_t(leaf) >= cache.cells.size()) {
      fail("Missing leaf for physical query");
      continue;
    }
    const auto &cell = cache.cells[leaf];
    if (cell.ports.size() > 10) {
      fail("Audit channel specialization exceeded");
      continue;
    }
    float3 t = make_float3(
        float((q.x - cell.bounds.lower[0]) / (cell.bounds.upper[0] - cell.bounds.lower[0])),
        float((q.y - cell.bounds.lower[1]) / (cell.bounds.upper[1] - cell.bounds.lower[1])),
        float((q.z - cell.bounds.lower[2]) / (cell.bounds.upper[2] - cell.bounds.lower[2])));
    float2 boundary[40];
    auto pair = [](Complex value) { return make_float2(value.real(), value.imag()); };
    for (size_t j = 0; j < cell.ports.size(); j++) {
      const double n = cell.ports[j].substrate ? p.substrate_ior.real() : p.incident_ior;
      const double kx = (double(q.x) + cell.ports[j].order) * q.z / p.pitch, ky = q.y;
      const double q2 = n * n - kx * kx - ky * ky;
      const Complex z = q2 > 0 ? Complex(std::sqrt(q2)) : Complex(0, std::sqrt(-q2));
      const bool active = std::find(cell.active_ports.begin(), cell.active_ports.end(), int(j)) !=
                          cell.active_ports.end();
      boundary[4 * j] = active ? pair((1.0 - z) / (1.0 + z)) : zero_float2();
      boundary[4 * j + 1] = active ? pair((z - n * n) / (z + n * n)) : zero_float2();
      boundary[4 * j + 2] = !active ?
                                make_float2(1, 1) :
                            q2 > 0 ?
                                make_float2(2 * std::sqrt(z.real()) / (1 + z.real()),
                                            2 * n * std::sqrt(z.real()) / (z.real() + n * n)) :
                                zero_float2();
      const double length = std::hypot(kx, ky);
      boundary[4 * j + 3] = length > 0 ? make_float2(kx / length, ky / length) : make_float2(1, 0);
    }
    auto find = [&](const DiffractionGratingPort &port) {
      for (size_t j = 0; j < cell.ports.size(); j++)
        if (cell.ports[j].order == port.order && cell.ports[j].substrate == port.substrate)
          return int(j);
      return -1;
    };
    bool valid = true;
    double query_error = 0;
    const int physical_ports = power.ports.size();
    for (int in = 0; in < physical_ports && valid; in++) {
      const int input = find(power.ports[in]);
      float values[10] = {};
      if (input < 0) {
        valid = false;
        break;
      }
      if (cell.operator_chart) {
        float2 jones[40];
        valid = chart_match(cell, boundary, input, t, jones);
        if (valid)
          for (size_t out = 0; out < cell.ports.size(); out++)
            for (int element = 0; element < 4; element++)
              values[out] += 0.5f * len_squared(jones[4 * out + element]);
      }
      else
        valid = intensity_match(cell, boundary, input, t, values);
      double column_error = 0;
      if (valid)
        valid = power_column_error(cell.ports, values, power, in, column_error);
      query_error = std::max(query_error, column_error);
    }
    if (!valid) {
      fail("Packed matcher or port mapping failed");
      continue;
    }
    errors.push_back(query_error);
    over_tolerance += query_error > options.tolerance;
    if (query_error > maximum) {
      maximum = query_error;
      worst = {q.x, q.y, q.z};
    }
  }
  std::sort(errors.begin(), errors.end());
  std::ostringstream out;
  out << std::setprecision(12) << "{\"requested_queries\":" << count
      << ",\"sampling\":" << std::quoted(sample_cache_bounds ? "uniform_cache_bounds" :
                                                                 "full_domain_and_cutoffs")
      << ",\"physical_queries\":" << errors.size() << ",\"empty_queries\":" << empty
      << ",\"failures\":" << failures << ",\"first_error\":" << std::quoted(first_error)
      << ",\"seed\":" << seed << ",\"over_construction_tolerance\":" << over_tolerance
      << ",\"maximum_power_column_error\":" << maximum
      << ",\"p95_error\":" << (errors.empty() ? 0 : errors[size_t(0.95 * (errors.size() - 1))])
      << ",\"held_out_threshold\":" << 2 * options.tolerance << ",\"passed\":"
      << (!errors.empty() && failures == 0 && maximum <= 2 * options.tolerance ? "true" : "false")
      << ",\"worst_query\":[" << worst[0] << ',' << worst[1] << ',' << worst[2] << "]}";
  return out.str();
}
}  // namespace diffraction_audit
