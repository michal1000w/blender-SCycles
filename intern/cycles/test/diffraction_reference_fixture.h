/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "scene/diffraction.h"
#include "util/math.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>

CCL_NAMESPACE_BEGIN

/* Shared host-generated fixtures for the CPU-float and Metal matching tests.
 * By default, exterior coefficients are generated in double precision to isolate
 * matching. construct_boundaries instead quantizes the query before solving and
 * supplies raw inputs for GPU coefficient construction. Neither mode tests
 * spatial/spectral interpolation. */
struct DiffractionReferenceFixture {
  int channels;
  std::vector<float2> charts;
  std::vector<float2> boundaries;
  std::vector<float4> boundary_inputs;
  std::vector<float2> rotations;
  std::vector<int> incoming;
  std::vector<int> active_ports;
  std::vector<float2> expected;
};

inline bool diffraction_reference_fixture(const double pitch,
                                          const std::complex<double> material,
                                          const int retained,
                                          const int cases,
                                          DiffractionReferenceFixture &fixture,
                                          std::string &error,
                                          const int feedback_ports = -1,
                                          const bool construct_boundaries = false)
{
  fixture = {};
  const bool lossless = material.imag() == 0.0;
  const int side_ports = 2 * retained + 1;
  const int ports = side_ports * (lossless ? 2 : 1), channels = 2 * ports;
  fixture.channels = channels;
  const DiffractionGratingProfile profile{pitch, 150.0, 0.41, 1.0, material, 1.0, material};
  std::mt19937 rng(519378);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  auto pair = [](const std::complex<double> value) {
    return make_float2(float(value.real()), float(value.imag()));
  };
  for (int i = 0; i < cases; i++) {
    double wavelength = lossless ? 500.0 + 200.0 * uniform(rng) : 380.0 + 400.0 * uniform(rng);
    const bool lower = lossless && i % 2;
    const double index = lower ? material.real() : 1.0;
    const double cosine = i % 8 == 1 ? 1e-5 : 0.01 + 0.99 * uniform(rng);
    const double phi = 2.0 * std::numbers::pi * uniform(rng);
    const double sine = std::sqrt(1.0 - cosine * cosine);
    double x = index * sine * std::cos(phi), ky = index * sine * std::sin(phi);
    if (i == 0) {
      wavelength = pitch > 1000.0 ? pitch / 4.0 : pitch;
      x = ky = 0.0;
    }
    float3 input_direction = make_float3(
        float(x / index), float(ky / index), float(i == 0 ? 1.0 : cosine));
    if (construct_boundaries) {
      wavelength = float(wavelength);
      const double norm = std::sqrt(double(input_direction.x) * input_direction.x +
                                    double(input_direction.y) * input_direction.y +
                                    double(input_direction.z) * input_direction.z);
      x = index * double(input_direction.x) / norm;
      ky = index * double(input_direction.y) / norm;
    }
    const int incoming_order = int(std::round(x * pitch / wavelength));
    const double kx = x - incoming_order * wavelength / pitch;
    DiffractionGratingBlock reference, physical;
    DiffractionGratingReferenceChart chart;
    if (!diffraction_grating_solve_reference(
            profile, wavelength, kx, ky, 16, retained, reference, error) ||
        !diffraction_grating_match_reference(
            profile, wavelength, kx, ky, reference, physical, error))
    {
      return false;
    }
    std::vector<unsigned char> keep(ports, 0), is_physical(ports, 0);
    if (feedback_ports >= 0) {
      if (feedback_ports > ports) {
        error = "Too many hybrid feedback ports";
        return false;
      }
      std::vector<std::pair<double, int>> cutoff_distance;
      for (int p = 0; p < ports; p++) {
        const auto &port = reference.ports[p];
        const double n = port.substrate ? material.real() : 1.0;
        const double px = kx + port.order * wavelength / pitch;
        cutoff_distance.emplace_back(std::abs(n * n - px * px - ky * ky), p);
      }
      std::sort(cutoff_distance.begin(), cutoff_distance.end());
      for (int p = 0; p < feedback_ports; p++) {
        keep[cutoff_distance[p].second] = 1;
      }
      for (int p = 0; p < ports; p++) {
        if (keep[p]) {
          fixture.active_ports.push_back(p);
        }
      }
      DiffractionGratingHybrid hybrid;
      if (!diffraction_grating_prepare_hybrid(
              profile, wavelength, kx, ky, reference, keep, hybrid, error))
      {
        return false;
      }
      /* Pad removed orders with zero rows/columns to give every GPU case the
       * same stride. These slots have T=0, R=0 and cannot feed back. */
      std::vector<int> mapping;
      for (const auto &hp : hybrid.scattering.ports) {
        int target = -1;
        for (int p = 0; p < ports; p++) {
          if (reference.ports[p].order == hp.order && reference.ports[p].substrate == hp.substrate)
          {
            target = p;
            is_physical[p] = !keep[p];
          }
        }
        if (target < 0) {
          error = "Unexpected hybrid fixture port";
          return false;
        }
        mapping.push_back(target);
      }
      std::vector<float2> padded(size_t(channels) * channels, zero_float2());
      const int hc = 2 * int(mapping.size());
      for (int row = 0; row < hc; row++) {
        for (int col = 0; col < hc; col++) {
          padded[size_t(2 * mapping[row / 2] + row % 2) * channels + 2 * mapping[col / 2] +
                 col % 2] = pair(hybrid.scattering.matrix[size_t(row) * hc + col]);
        }
      }
      fixture.charts.insert(fixture.charts.end(), padded.begin(), padded.end());
      fixture.rotations.push_back(make_float2(1.0f, 0.0f));
    }
    else {
      std::vector<DiffractionGratingReferenceChart> selected;
      double phase;
      if (!diffraction_grating_choose_chart(std::span(&reference, 1), phase, selected, error)) {
        return false;
      }
      chart = std::move(selected.front());
      for (const auto value : chart.matrix) {
        fixture.charts.push_back(pair(value));
      }
      fixture.rotations.push_back(pair(chart.rotation));
    }
    if (int(reference.ports.size()) != ports) {
      error = "Unexpected reference fixture port count";
      return false;
    }
    const int incoming = (lower ? side_ports : 0) + incoming_order + retained;
    fixture.incoming.push_back(incoming);
    int physical_input = -1;
    for (int p = 0; p < int(physical.ports.size()); p++) {
      if (physical.ports[p].order == incoming_order && physical.ports[p].substrate == lower) {
        physical_input = p;
      }
    }
    if (physical_input < 0) {
      error = "Fixture incoming wave is not propagating";
      return false;
    }
    for (int p = 0; p < ports; p++) {
      const auto &port = reference.ports[p];
      const double n = port.substrate ? material.real() : 1.0;
      const double px = kx + port.order * wavelength / pitch;
      const double transverse = std::hypot(px, ky);
      fixture.boundary_inputs.push_back(
          make_float4(input_direction.x, input_direction.y, input_direction.z, float(index)));
      fixture.boundary_inputs.push_back(make_float4(
          float(n), float(wavelength), float(pitch), float(port.order - incoming_order)));
      const double q2 = n * n - px * px - ky * ky;
      const std::complex<double> q = q2 >= 0.0 ? std::complex<double>(std::sqrt(q2), 0.0) :
                                                 std::complex<double>(0.0, std::sqrt(-q2));
      fixture.boundaries.push_back(pair((1.0 - q) / (1.0 + q)));
      fixture.boundaries.push_back(pair((q - n * n) / (q + n * n)));
      fixture.boundaries.push_back(
          q2 > 0.0 ? make_float2(float(2.0 * std::sqrt(q.real()) / (1.0 + q.real())),
                                 float(2.0 * n * std::sqrt(q.real()) / (q.real() + n * n))) :
                     zero_float2());
      fixture.boundaries.push_back(
          transverse > 0.0 ? make_float2(float(px / transverse), float(ky / transverse)) :
                             make_float2(1.0f, 0.0f));
      if (feedback_ports >= 0 && !keep[p]) {
        const size_t offset = fixture.boundaries.size() - 4;
        fixture.boundaries[offset] = fixture.boundaries[offset + 1] = zero_float2();
        fixture.boundaries[offset + 2] = is_physical[p] ? make_float2(1.0f, 1.0f) : zero_float2();
      }
      int physical_output = -1;
      for (int j = 0; j < int(physical.ports.size()); j++) {
        if (physical.ports[j].order == port.order && physical.ports[j].substrate == port.substrate)
        {
          physical_output = j;
        }
      }
      for (int row = 0; row < 2; row++) {
        for (int col = 0; col < 2; col++) {
          fixture.expected.push_back(physical_output < 0 ?
                                         zero_float2() :
                                         pair(physical.matrix[size_t(2 * physical_output + row) *
                                                                  (2 * physical.ports.size()) +
                                                              2 * physical_input + col]));
        }
      }
    }
  }
  return true;
}

CCL_NAMESPACE_END
