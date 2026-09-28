/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "util/math.h"
#include <complex>
#include <numbers>
#include <random>
#include <vector>

CCL_NAMESPACE_BEGIN

struct DiffractionBoundaryFixture {
  /* Two float4 per query: direction/index-in; index-out/wavelength/pitch/order. */
  std::vector<float4> inputs;
  /* Five float2: Rte, Rtm, Tte/Ttm, tangential direction, q2/zero. */
  std::vector<float2> expected;
};

inline DiffractionBoundaryFixture diffraction_boundary_fixture()
{
  DiffractionBoundaryFixture fixture;
  auto append = [&](float3 d, float ni, float no, float wavelength, float pitch, int order) {
    fixture.inputs.push_back(make_float4(d.x, d.y, d.z, ni));
    fixture.inputs.push_back(make_float4(no, wavelength, pitch, float(order)));
    const double norm2 = double(d.x) * d.x + double(d.y) * d.y + double(d.z) * d.z;
    const double scale = double(ni) / std::sqrt(norm2);
    const double x = scale * d.x, y = scale * d.y, z = scale * d.z;
    const double delta = double(order) * double(wavelength) / double(pitch);
    const double q2 = (double(no) - ni) * (double(no) + ni) + z * z - 2.0 * x * delta -
                      delta * delta;
    const std::complex<double> q = q2 > 0.0 ? std::complex<double>(std::sqrt(q2), 0.0) :
                                              std::complex<double>(0.0, std::sqrt(-q2));
    auto pair = [](std::complex<double> value) {
      return make_float2(float(value.real()), float(value.imag()));
    };
    fixture.expected.push_back(pair((1.0 - q) / (1.0 + q)));
    fixture.expected.push_back(pair((q - double(no) * no) / (q + double(no) * no)));
    fixture.expected.push_back(
        q2 > 0.0 ?
            make_float2(float(2.0 * std::sqrt(q.real()) / (1.0 + q.real())),
                        float(2.0 * no * std::sqrt(q.real()) / (q.real() + double(no) * no))) :
            zero_float2());
    const double transverse = std::hypot(x + delta, y);
    fixture.expected.push_back(
        transverse > 0.0 ? make_float2(float((x + delta) / transverse), float(y / transverse)) :
                           make_float2(1.0f, 0.0f));
    fixture.expected.push_back(make_float2(float(q2), 0.0f));
  };
  std::mt19937 rng(789521);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  for (int i = 0; i < 1024; i++) {
    const float z = i % 8 == 0 ? 1e-5f : 0.001f + 0.999f * uniform(rng);
    const float phi = 2.0f * std::numbers::pi_v<float> * uniform(rng);
    const float radius = std::sqrt(1.0f - z * z);
    const float ni = i % 3 == 0 ? 1.58f : 1.0f;
    const float no = i % 3 == 1 ? 1.5f : ni;
    append(make_float3(radius * std::cos(phi), radius * std::sin(phi), z),
           ni,
           no,
           380.0f + 400.0f * uniform(rng),
           i % 2 ? 1600.0f : 740.0f,
           i % 9 - 4);
  }
  for (int order : {-1, 1}) {
    append(make_float3(0, 0, 1), 1, 1, 740, 740, order);
    append(make_float3(0, 0, 2), 1, 1, 740, 740, order);
  }
  for (float sign : {-1.0f, 1.0f}) {
    for (float wavelength :
         {std::nextafter(740.0f, 0.0f), 740.0f, std::nextafter(740.0f, INFINITY)})
    {
      append(make_float3(sign, 0, 1e-5f), 1, 1, wavelength, 740, int(-2 * sign));
    }
  }
  append(make_float3(0, 1, 0), 1, 1, 550, 1600, 0);
  return fixture;
}

CCL_NAMESPACE_END
