/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "kernel/util/diffraction_boundary.h"
#include "kernel/util/diffraction_cache.h"
#include "kernel/util/diffraction_grid.h"
#include "kernel/util/diffraction_direction.h"
#include "kernel/util/diffraction_scene_data.h"
#include "kernel/util/diffraction_two_sided.h"
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include "kernel/util/diffraction_reference.h"
#include "scene/diffraction.h"
#include "scene/diffraction_manager.h"
#include "test/diffraction_boundary_fixture.h"
#include "test/diffraction_reference_fixture.h"

#include <cmath>
#include <iomanip>
#include <numbers>
#include <sstream>

#include <gtest/gtest.h>

CCL_NAMESPACE_BEGIN

namespace {
using Complex = std::complex<double>;

DiffractionGratingProfile profile(const Complex material)
{
  return {740.0, 150.0, 0.41, 1.0, material, Complex(1.0), material};
}

void expect_jones_power(const DiffractionGratingResponse &response)
{
  double ss = 0.0, pp = 0.0;
  Complex sp = 0.0;
  auto accumulate = [&](const std::array<Complex, 4> &J) {
    ss += std::norm(J[0]) + std::norm(J[2]);
    pp += std::norm(J[1]) + std::norm(J[3]);
    sp += std::conj(J[0]) * J[1] + std::conj(J[2]) * J[3];
  };
  for (const auto &order : response.orders) {
    accumulate(order.reflection_jones);
    if (response.has_transmission_jones) {
      accumulate(order.transmission_jones);
    }
    for (int c = 0; c < 2; c++) {
      EXPECT_NEAR(std::norm(order.reflection_jones[c]) + std::norm(order.reflection_jones[2 + c]),
                  order.reflection[c],
                  1e-10);
      if (response.has_transmission_jones) {
        EXPECT_NEAR(std::norm(order.transmission_jones[c]) +
                        std::norm(order.transmission_jones[2 + c]),
                    order.substrate_flux[c],
                    1e-10);
      }
    }
  }
  /* Maximum outgoing power over ALL unit Jones inputs, including elliptical
   * polarizations; checking only separate s and p inputs is insufficient. */
  const double maximum = 0.5 * (ss + pp + std::sqrt((ss - pp) * (ss - pp) + 4.0 * std::norm(sp)));
  EXPECT_LE(maximum, 1.0 + 1e-10);
  if (response.has_transmission_jones && std::abs(response.layer_absorption[0]) < 1e-10 &&
      std::abs(response.layer_absorption[1]) < 1e-10)
  {
    EXPECT_NEAR(ss, 1.0, 1e-10);
    EXPECT_NEAR(pp, 1.0, 1e-10);
    EXPECT_LT(std::abs(sp), 1e-10);
  }
}
}  // namespace

TEST(SceneDiffraction, TabulatedLayerOpticalConstants)
{
  auto p = profile(1.5);
  p.ridge_spectrum = {{500, Complex(1.3, 0.1)}, {600, Complex(1.7, 0.3)}};
  p.groove_spectrum = {{500, Complex(1)}, {600, Complex(1.2)}};
  for (double wavelength : {500.0, 550.0, 600.0}) {
    auto reference = p;
    reference.ridge_spectrum.clear();
    reference.groove_spectrum.clear();
    const double t = (wavelength - 500) / 100;
    reference.ridge_ior = (1 - t) * Complex(1.3, 0.1) + t * Complex(1.7, 0.3);
    reference.groove_ior = Complex(1 + 0.2 * t);
    DiffractionGratingBlock actual, expected;
    std::string error;
    ASSERT_TRUE(diffraction_grating_solve_bloch(p, wavelength, 0.1, 0.2, 8, actual, error))
        << error;
    ASSERT_TRUE(
        diffraction_grating_solve_bloch(reference, wavelength, 0.1, 0.2, 8, expected, error))
        << error;
    ASSERT_EQ(actual.matrix.size(), expected.matrix.size());
    for (size_t j = 0; j < actual.matrix.size(); j++)
      EXPECT_NEAR(std::abs(actual.matrix[j] - expected.matrix[j]), 0, 1e-10);
    EXPECT_LE(actual.maximum_power_gain, 1 + 1e-10);
  }
  DiffractionGratingBlock result;
  std::string error;
  EXPECT_FALSE(diffraction_grating_solve_bloch(p, 499, 0.1, 0.2, 8, result, error));
  EXPECT_FALSE(diffraction_grating_solve_bloch(p, 601, 0.1, 0.2, 8, result, error));
  p.ridge_spectrum[1].wavelength = 500;
  EXPECT_FALSE(diffraction_grating_solve_bloch(p, 500, 0.1, 0.2, 8, result, error));
  p.ridge_spectrum[1].wavelength = 600;
  p.ridge_spectrum[1].index = Complex(1.7, -0.3);
  EXPECT_FALSE(diffraction_grating_solve_bloch(p, 550, 0.1, 0.2, 8, result, error));
}

TEST(SceneDiffraction, LayerSpectrumCacheCoverage)
{
  auto p = profile(1.5);
  p.ridge_spectrum = {{500, Complex(1.3, 0.1)}, {600, Complex(1.7, 0.3)}};
  DiffractionGratingCacheOptions options;
  options.bounds = {{-0.03125, -0.0625, 540}, {0.03125, 0.0625, 560}};
  options.half_orders = 8;
  options.retained_half_orders = 3;
  options.tolerance = 0.1;
  DiffractionGratingCache cache;
  DiffractionGratingCacheStats stats;
  std::string error;
  ASSERT_TRUE(diffraction_grating_build_cache(p, options, cache, stats, error)) << error;
  EXPECT_FALSE(cache.cells.empty());
  options.bounds.upper[2] = 601;
  EXPECT_FALSE(diffraction_grating_build_cache(p, options, cache, stats, error));
  EXPECT_EQ(stats.reference_solves, 0);
  EXPECT_TRUE(cache.nodes.empty());
  options.bounds = {{-0.03125, 2, 540}, {0.03125, 3, 560}};
  p.ridge_spectrum[1].wavelength = 500;
  EXPECT_FALSE(diffraction_grating_build_cache(p, options, cache, stats, error));
  EXPECT_EQ(stats.reference_solves, 0);
  EXPECT_TRUE(cache.nodes.empty());
}

TEST(SceneDiffraction, CacheResolvesNarrowOpticalConstantFeature)
{
  auto p = profile(1.5);
  p.ridge_spectrum = {{500, Complex(1.5)},
                      {545, Complex(1.5)},
                      {545.125, Complex(1.8, 0.2)},
                      {545.25, Complex(1.5)},
                      {600, Complex(1.5)}};
  DiffractionGratingCacheOptions options;
  options.bounds = {{-0.001953125, -0.001953125, 540}, {0.001953125, 0.001953125, 560}};
  options.half_orders = 8;
  options.retained_half_orders = 3;
  options.tolerance = 0.1;
  DiffractionGratingCache cache;
  DiffractionGratingCacheStats stats;
  std::string error;
  ASSERT_TRUE(diffraction_grating_build_cache(p, options, cache, stats, error)) << error;
  EXPECT_GE(cache.cells.size(), 4);
  for (const auto &cell : cache.cells)
    for (double knot : {545.0, 545.125, 545.25})
      EXPECT_FALSE(cell.bounds.lower[2] < knot && cell.bounds.upper[2] > knot);
  p.ridge_spectrum[2].wavelength = 545.13;
  ASSERT_TRUE(diffraction_grating_build_cache(p, options, cache, stats, error)) << error;
  const float rounded = float(545.13);
  const float adjacent = std::nextafter(rounded, rounded < 545.13 ? INFINITY : -INFINITY);
  for (float boundary : {rounded, adjacent}) {
    bool found = false;
    for (const auto &cell : cache.cells)
      found |= cell.bounds.lower[2] == boundary || cell.bounds.upper[2] == boundary;
    EXPECT_TRUE(found);
  }
  options.maximum_depth = 1;
  EXPECT_FALSE(diffraction_grating_build_cache(p, options, cache, stats, error));
  EXPECT_TRUE(cache.nodes.empty());
}

TEST(SceneDiffraction, AbsorbingSubstrateDispersion)
{
  auto p = profile(Complex(0.9, 6));
  p.absorbing_substrate_spectrum = {{500, Complex(0.9, 6)}, {600, Complex(1.5, 4)}};
  for (double wavelength : {500.0, 550.0, 600.0}) {
    auto reference = p;
    reference.absorbing_substrate_spectrum.clear();
    const double t = (wavelength - 500) / 100;
    reference.substrate_ior = (1 - t) * Complex(0.9, 6) + t * Complex(1.5, 4);
    DiffractionGratingBlock a, b;
    std::string error;
    ASSERT_TRUE(diffraction_grating_solve_bloch(p, wavelength, 0.1, 0.2, 8, a, error)) << error;
    ASSERT_TRUE(diffraction_grating_solve_bloch(reference, wavelength, 0.1, 0.2, 8, b, error))
        << error;
    ASSERT_EQ(a.matrix.size(), b.matrix.size());
    for (size_t j = 0; j < a.matrix.size(); j++)
      EXPECT_NEAR(std::abs(a.matrix[j] - b.matrix[j]), 0, 1e-10);
    for (const auto &port : a.ports)
      EXPECT_FALSE(port.substrate);
    EXPECT_LE(a.maximum_power_gain, 1 + 1e-10);
  }
  DiffractionGratingBlock result;
  std::string error;
  EXPECT_FALSE(diffraction_grating_solve_bloch(p, 601, 0.1, 0.2, 8, result, error));
  p.absorbing_substrate_spectrum[1].index = Complex(1.5);
  EXPECT_FALSE(diffraction_grating_solve_bloch(p, 550, 0.1, 0.2, 8, result, error));
}

TEST(SceneDiffraction, FlatComplexFresnel)
{
  for (const Complex index : {Complex(1.5), Complex(0.9, 6.0)}) {
    for (double angle : {0.0, 0.4, 0.9}) {
      auto p = profile(index);
      p.depth = 0.0;
      DiffractionGratingResponse response;
      std::string error;
      ASSERT_TRUE(diffraction_grating_solve(p, 550.0, angle, 0.7, 12, response, error)) << error;
      const double ci = std::cos(angle);
      const Complex z = std::sqrt(index * index - std::pow(std::sin(angle), 2));
      const Complex rs = (ci - z) / (ci + z);
      const Complex rp = (index * index * ci - z) / (index * index * ci + z);
      const auto &zero = response.orders[12];
      EXPECT_NEAR(std::abs(zero.reflection_jones[0] - rs), 0.0, 1e-10);
      EXPECT_NEAR(std::abs(zero.reflection_jones[3] - rp), 0.0, 1e-10);
      EXPECT_LT(std::abs(zero.reflection_jones[1]), 1e-10);
      EXPECT_LT(std::abs(zero.reflection_jones[2]), 1e-10);
      for (int c = 0; c < 2; c++) {
        EXPECT_NEAR(response.layer_absorption[c], 0.0, 1e-10);
      }
      expect_jones_power(response);
    }
  }
}

TEST(SceneDiffraction, ZeroDepthIgnoresDegenerateInternalModes)
{
  auto p = profile(1.5);
  p.pitch = 1000.0;
  p.depth = 0.0;
  /* The absent layer would have q=0 for its +/-1 orders. Its material must
   * not prevent evaluation of the ordinary air/glass interface. */
  p.ridge_ior = p.groove_ior = Complex(0.55);
  DiffractionGratingResponse response;
  std::string error;
  ASSERT_TRUE(diffraction_grating_solve(p, 550.0, 0.0, 0.0, 8, response, error)) << error;
  for (const auto &order : response.orders) {
    for (int polarization = 0; polarization < 2; polarization++) {
      EXPECT_NEAR(order.reflection[polarization], order.order == 0 ? 0.04 : 0.0, 1e-12);
      EXPECT_NEAR(order.substrate_flux[polarization], order.order == 0 ? 0.96 : 0.0, 1e-12);
    }
  }
  expect_jones_power(response);
}

TEST(SceneDiffraction, DielectricConservationAndMetalPassivity)
{
  for (const Complex index : {Complex(1.5), Complex(0.9, 6.0)}) {
    for (const double angle : {0.0, 0.4, 0.9}) {
      DiffractionGratingResponse response;
      std::string error;
      ASSERT_TRUE(
          diffraction_grating_solve(profile(index), 550.0, angle, 0.7, 16, response, error))
          << error;
      EXPECT_LT(response.boundary_residual, 1e-10);
      for (int c = 0; c < 2; c++) {
        if (index.imag() == 0.0) {
          EXPECT_NEAR(response.layer_absorption[c], 0.0, 1e-10);
        }
        else {
          EXPECT_GE(response.layer_absorption[c], -1e-10);
          EXPECT_LE(response.layer_absorption[c], 1.0);
        }
      }
      expect_jones_power(response);
    }
  }
}

TEST(SceneDiffraction, IncidentMediumSimilarity)
{
  auto embedded = profile(Complex(0.9, 6.0));
  embedded.incident_ior = 1.58;
  embedded.groove_ior = 1.58;
  auto relative = embedded;
  relative.incident_ior = 1.0;
  relative.ridge_ior /= 1.58;
  relative.groove_ior /= 1.58;
  relative.substrate_ior /= 1.58;
  DiffractionGratingResponse a, b;
  std::string error;
  ASSERT_TRUE(diffraction_grating_solve(embedded, 550.0, 0.4, 0.7, 16, a, error)) << error;
  ASSERT_TRUE(diffraction_grating_solve(relative, 550.0 / 1.58, 0.4, 0.7, 16, b, error)) << error;
  for (size_t m = 0; m < a.orders.size(); m++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_LT(std::abs(a.orders[m].reflection_jones[j] - b.orders[m].reflection_jones[j]),
                1e-10);
    }
  }
}

TEST(SceneDiffraction, InvalidInputsAndExactCutoffsAreReported)
{
  auto p = profile(Complex(1.5));
  DiffractionGratingResponse r;
  std::string error;
  EXPECT_FALSE(diffraction_grating_solve(p, 0.0, 0.0, 0.0, 12, r, error));
  EXPECT_FALSE(error.empty());
  EXPECT_FALSE(diffraction_grating_solve(p, 740.0, 0.0, 0.0, 12, r, error));
  EXPECT_FALSE(error.empty());
  p.ridge_ior = Complex(1.5, -0.1);
  EXPECT_FALSE(diffraction_grating_solve(p, 550.0, 0.0, 0.0, 12, r, error));
}

TEST(SceneDiffraction, SpecularOrderComplexReciprocity)
{
  for (const Complex material : {Complex(1.5), Complex(0.9, 6.0)}) {
    DiffractionGratingResponse forward, reverse;
    std::string error;
    ASSERT_TRUE(diffraction_grating_solve(profile(material), 550.0, 0.4, 0.7, 16, forward, error))
        << error;
    ASSERT_TRUE(diffraction_grating_solve(
        profile(material), 550.0, 0.4, 0.7 + std::numbers::pi_v<double>, 16, reverse, error))
        << error;
    const auto &a = forward.orders[16].reflection_jones;
    const auto &b = reverse.orders[16].reflection_jones;
    /* Reversal flips the s basis and retains the p basis for these definitions.
     * Flux-normalized reciprocal scattering transposes the Jones matrix. */
    EXPECT_LT(std::abs(a[0] - b[0]), 1e-10);
    EXPECT_LT(std::abs(a[3] - b[3]), 1e-10);
    EXPECT_LT(std::abs(a[1] + b[2]), 1e-10);
    EXPECT_LT(std::abs(a[2] + b[1]), 1e-10);
  }
}

TEST(SceneDiffraction, IndependentNistConicalReference)
{
  /* Generated with NIST pySCATMECH 0.1.9, 16 Fourier orders on each side,
   * Single_Line_Grating, pitch 740 nm, height 150 nm, duty 0.41.
   * NIST's diffraction order convention is opposite to the solve's convention.
   * Compare equal truncations here; convergence is a separate validation. */
  struct Reference {
    double n, k, angle, azimuth;
    double powers[5][2];
  };
  const Reference cases[] = {{1.5,
                              0.0,
                              0.4,
                              0.7,
                              {{6.2844991046260919e-20, 6.4793370868914676e-20},
                               {0.016183308432771423, 0.020145908929171036},
                               {0.0025019825144088378, 0.0018949652866994834},
                               {3.1404976343950381e-18, 7.5768195667776913e-19},
                               {1.9426652281638134e-19, 6.6604333768889953e-20}}},
                             {1.5,
                              0.0,
                              0.9,
                              1.1,
                              {{9.095405161037016e-20, 1.1844467285923994e-19},
                               {0.032487059133080413, 0.018609960539445608},
                               {0.016936229173669888, 0.0027391555417126626},
                               {2.5167383148685057e-18, 1.6090001719651269e-18},
                               {1.6198858643696875e-19, 7.3121356612999773e-20}}},
                             {0.9,
                              6.0,
                              0.4,
                              0.7,
                              {{5.1198683656216562e-18, 5.3655149722170801e-18},
                               {0.65389920039239768, 0.66051077471506325},
                               {0.20830227979354765, 0.1893162962066075},
                               {1.292247852203561e-17, 1.6840319566087759e-17},
                               {2.7651462942932512e-18, 3.587871451679336e-18}}},
                             {0.9,
                              6.0,
                              0.9,
                              1.1,
                              {{3.6610219614852587e-17, 2.329866608589351e-18},
                               {0.26570761742944293, 0.33629341107841093},
                               {0.31050504526690448, 0.45338854163073605},
                               {7.3934128233948072e-17, 6.5151284961838323e-18},
                               {7.2028060356527628e-18, 1.955015795496109e-18}}}};
  for (const auto &c : cases) {
    DiffractionGratingResponse response;
    std::string error;
    ASSERT_TRUE(diffraction_grating_solve(
        profile(Complex(c.n, c.k)), 550.0, c.angle, c.azimuth, 16, response, error))
        << error;
    for (int m = -2; m <= 2; m++) {
      for (int p = 0; p < 2; p++) {
        EXPECT_NEAR(response.orders[16 + m].reflection[p], c.powers[m + 2][p], 1e-8);
      }
    }
  }
}

TEST(SceneDiffraction, CompleteBlockPassivityAndSingleIncidenceAgreement)
{
  for (const Complex material : {Complex(1.5), Complex(0.9, 6.0)}) {
    DiffractionGratingBlock block;
    DiffractionGratingResponse single;
    std::string error;
    ASSERT_TRUE(
        diffraction_grating_solve_block(profile(material), 550.0, 0.4, 0.7, 16, block, error))
        << error;
    ASSERT_TRUE(diffraction_grating_solve(profile(material), 550.0, 0.4, 0.7, 16, single, error))
        << error;
    EXPECT_LE(block.maximum_power_gain, 1.0 + 1e-10);
    EXPECT_GE(block.minimum_power_gain, -1e-10);
    if (material.imag() == 0.0) {
      EXPECT_NEAR(block.maximum_power_gain, 1.0, 1e-10);
      EXPECT_NEAR(block.minimum_power_gain, 1.0, 1e-10);
    }
    int input = -1;
    for (int p = 0; p < int(block.ports.size()); p++) {
      if (block.ports[p].order == 0 && !block.ports[p].substrate) {
        input = p;
      }
    }
    ASSERT_GE(input, 0);
    const int channels = 2 * int(block.ports.size());
    const double Ci[2][2] = {{-std::sin(0.7), std::cos(0.7)}, {std::cos(0.7), std::sin(0.7)}};
    for (int p = 0; p < int(block.ports.size()); p++) {
      const auto &port = block.ports[p];
      const double kx = std::sin(0.4) * std::cos(0.7) + port.order * 550.0 / 740.0;
      const double ky = std::sin(0.4) * std::sin(0.7);
      const double phi = std::atan2(ky, kx);
      const double sign = port.substrate ? 1.0 : -1.0;
      const double Co[2][2] = {{-std::sin(phi), sign * std::cos(phi)},
                               {std::cos(phi), sign * std::sin(phi)}};
      const auto &expected = port.substrate ? single.orders[16 + port.order].transmission_jones :
                                              single.orders[16 + port.order].reflection_jones;
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          Complex value = 0.0;
          for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
              value += Co[a][i] * block.matrix[size_t(2 * p + a) * channels + 2 * input + b] *
                       Ci[b][j];
            }
          }
          EXPECT_LT(std::abs(value - expected[2 * i + j]), 1e-10);
        }
      }
    }
  }
}

TEST(SceneDiffraction, CompleteBlockComplexReciprocity)
{
  for (const Complex material : {Complex(1.5), Complex(0.9, 6.0)}) {
    DiffractionGratingBlock a, b;
    std::string error;
    ASSERT_TRUE(diffraction_grating_solve_block(profile(material), 550.0, 0.4, 0.7, 16, a, error))
        << error;
    ASSERT_TRUE(diffraction_grating_solve_block(
        profile(material), 550.0, 0.4, 0.7 + std::numbers::pi_v<double>, 16, b, error))
        << error;
    ASSERT_EQ(a.ports.size(), b.ports.size());
    std::vector<int> reversed;
    for (const auto &port : a.ports) {
      int match = -1;
      for (int j = 0; j < int(b.ports.size()); j++) {
        if (b.ports[j].substrate == port.substrate && b.ports[j].order == -port.order) {
          match = j;
        }
      }
      ASSERT_GE(match, 0);
      reversed.push_back(match);
    }
    const int channels = 2 * int(a.ports.size());
    for (int row = 0; row < channels; row++) {
      for (int col = 0; col < channels; col++) {
        const int reverse_row = 2 * reversed[col / 2] + col % 2;
        const int reverse_col = 2 * reversed[row / 2] + row % 2;
        EXPECT_LT(std::abs(a.matrix[size_t(row) * channels + col] -
                           b.matrix[size_t(reverse_row) * channels + reverse_col]),
                  1e-10);
      }
    }
  }
}

TEST(SceneDiffraction, PowerInterpolationConservationReciprocityAndAccuracy)
{
  DiffractionGratingBlock low, high, center, rlow, rhigh, rcenter;
  std::string error;
  const auto p = profile(Complex(1.5));
  for (int reverse = 0; reverse < 2; reverse++) {
    const double phi = 0.7 + reverse * std::numbers::pi_v<double>;
    ASSERT_TRUE(
        diffraction_grating_solve_block(p, 550.0, 0.395, phi, 16, reverse ? rlow : low, error))
        << error;
    ASSERT_TRUE(
        diffraction_grating_solve_block(p, 550.0, 0.405, phi, 16, reverse ? rhigh : high, error))
        << error;
    ASSERT_TRUE(
        diffraction_grating_solve_block(p, 550.0, 0.4, phi, 16, reverse ? rcenter : center, error))
        << error;
  }
  DiffractionGratingPowerBlock a, b, ar, br, expected, interpolated, reversed;
  diffraction_grating_power_block(low, a);
  diffraction_grating_power_block(high, b);
  diffraction_grating_power_block(rlow, ar);
  diffraction_grating_power_block(rhigh, br);
  diffraction_grating_power_block(center, expected);
  const std::array<DiffractionGratingPowerSample, 2> samples = {{{&a, 0.5}, {&b, 0.5}}};
  const std::array<DiffractionGratingPowerSample, 2> reverse_samples = {{{&ar, 0.5}, {&br, 0.5}}};
  ASSERT_TRUE(diffraction_grating_interpolate_power(samples, center.ports, interpolated, error))
      << error;
  ASSERT_TRUE(
      diffraction_grating_interpolate_power(reverse_samples, rcenter.ports, reversed, error))
      << error;
  const size_t count = center.ports.size();
  for (size_t col = 0; col < count; col++) {
    double sum = 0.0;
    for (size_t row = 0; row < count; row++) {
      sum += interpolated.matrix[row * count + col];
      EXPECT_NEAR(
          interpolated.matrix[row * count + col], expected.matrix[row * count + col], 5e-4);
      int rr = -1, rc = -1;
      for (int j = 0; j < int(count); j++) {
        if (reversed.ports[j].substrate == center.ports[col].substrate &&
            reversed.ports[j].order == -center.ports[col].order)
        {
          rr = j;
        }
        if (reversed.ports[j].substrate == center.ports[row].substrate &&
            reversed.ports[j].order == -center.ports[row].order)
        {
          rc = j;
        }
      }
      ASSERT_GE(rr, 0);
      ASSERT_GE(rc, 0);
      EXPECT_NEAR(
          interpolated.matrix[row * count + col], reversed.matrix[size_t(rr) * count + rc], 1e-10);
    }
    EXPECT_NEAR(sum, 1.0, 1e-10);
  }
}

TEST(SceneDiffraction, PowerInterpolationAcrossOrderCutoffStaysPassive)
{
  DiffractionGratingBlock low, high, target;
  DiffractionGratingPowerBlock a, b, value;
  std::string error;
  const auto p = profile(Complex(0.9, 6.0));
  ASSERT_TRUE(diffraction_grating_solve_block(p, 550.0, 0.30, 0.7, 16, low, error)) << error;
  ASSERT_TRUE(diffraction_grating_solve_block(p, 550.0, 0.36, 0.7, 16, high, error)) << error;
  ASSERT_TRUE(diffraction_grating_solve_block(p, 550.0, 0.33, 0.7, 16, target, error)) << error;
  ASSERT_NE(low.ports.size(), high.ports.size());
  diffraction_grating_power_block(low, a);
  diffraction_grating_power_block(high, b);
  const std::array<DiffractionGratingPowerSample, 2> samples = {{{&a, 0.5}, {&b, 0.5}}};
  ASSERT_TRUE(diffraction_grating_interpolate_power(samples, target.ports, value, error)) << error;
  EXPECT_LE(value.maximum_column_sum, 1.0 + 1e-10);
  for (const double power : value.matrix) {
    EXPECT_GE(power, 0.0);
  }
  /* This proves the bound across a changing port set, not accuracy of such a
   * coarse cell. Response generation must refine cutoff neighborhoods. */
  const std::array<DiffractionGratingPowerSample, 2> negative = {{{&a, -0.5}, {&b, 1.5}}};
  EXPECT_FALSE(diffraction_grating_interpolate_power(negative, target.ports, value, error));
  const std::array<DiffractionGratingPort, 2> duplicate = {{target.ports[0], target.ports[0]}};
  EXPECT_FALSE(diffraction_grating_interpolate_power(samples, duplicate, value, error));
}

TEST(SceneDiffraction, DirectBlochCoordinatesAndSubstrateOnlyPorts)
{
  const auto p = profile(1.5);
  DiffractionGratingBlock angular, direct;
  std::string error;
  ASSERT_TRUE(diffraction_grating_solve_block(p, 550.0, 0.4, 0.7, 16, angular, error)) << error;
  ASSERT_TRUE(diffraction_grating_solve_bloch(
      p, 550.0, std::sin(0.4) * std::cos(0.7), std::sin(0.4) * std::sin(0.7), 16, direct, error))
      << error;
  ASSERT_EQ(angular.matrix.size(), direct.matrix.size());
  for (size_t i = 0; i < direct.matrix.size(); i++) {
    EXPECT_LT(std::abs(angular.matrix[i] - direct.matrix[i]), 1e-12);
  }
  /* ky above the upper-medium light cone: every upper order is evanescent,
   * but a lossless substrate still carries incident and outgoing power. */
  ASSERT_TRUE(diffraction_grating_solve_bloch(p, 550.0, 0.13, 1.2, 16, direct, error)) << error;
  ASSERT_FALSE(direct.ports.empty());
  for (const auto &port : direct.ports) {
    EXPECT_TRUE(port.substrate);
  }
  EXPECT_NEAR(direct.minimum_power_gain, 1.0, 1e-10);
  EXPECT_NEAR(direct.maximum_power_gain, 1.0, 1e-10);
  EXPECT_TRUE(diffraction_grating_solve_bloch(p, 550.0, 0.13, 2.0, 16, direct, error));
  EXPECT_TRUE(error.empty());
  EXPECT_TRUE(direct.matrix.empty());
}

TEST(SceneDiffraction, TruncationCannotOmitPropagatingPorts)
{
  auto p = profile(1.5);
  p.pitch = 1600.0;
  DiffractionGratingBlock block;
  std::string error;
  EXPECT_FALSE(diffraction_grating_solve_bloch(p, 550.0, 0.13, 0.2, 1, block, error));
  EXPECT_NE(error.find("omits propagating"), std::string::npos);
  EXPECT_TRUE(block.matrix.empty());
  ASSERT_TRUE(diffraction_grating_solve_bloch(p, 550.0, 0.13, 0.2, 16, block, error)) << error;
  EXPECT_NEAR(block.maximum_power_gain, 1.0, 1e-10);
  /* A displaced Bloch base must not evade the coverage check. */
  EXPECT_FALSE(diffraction_grating_solve_bloch(p, 550.0, 20.0, 0.2, 16, block, error));
  EXPECT_NE(error.find("omits propagating"), std::string::npos);
}

TEST(SceneDiffraction, PackedKernelPowerMatchesMaxwellBlock)
{
  for (const Complex material : {Complex(1.5), Complex(0.9, 6.0)}) {
    DiffractionGratingBlock block;
    std::string error;
    ASSERT_TRUE(
        diffraction_grating_solve_bloch(profile(material), 550.0, 0.13, 0.2, 16, block, error))
        << error;
    DiffractionGratingPowerBlock power;
    diffraction_grating_power_block(block, power);
    std::vector<float> packed;
    ASSERT_TRUE(diffraction_grating_pack_power(power, packed, error)) << error;
    for (size_t col = 0; col < power.ports.size(); col++) {
      double sum = 0.0;
      const auto &incoming = power.ports[col];
      for (size_t row = 0; row < power.ports.size(); row++) {
        const auto &outgoing = power.ports[row];
        const float value = diffraction_table_power(
            packed.data(), incoming.order, incoming.substrate, outgoing.order, outgoing.substrate);
        EXPECT_NEAR(value, power.matrix[row * power.ports.size() + col], 3e-8);
        sum += value;
      }
      EXPECT_LE(sum, 1.0 + 1e-7);
      EXPECT_EQ(
          diffraction_table_power(packed.data(), incoming.order, incoming.substrate, 100, false),
          0.0f);
    }
    power.matrix[0] = -0.01;
    EXPECT_FALSE(diffraction_grating_pack_power(power, packed, error));
    EXPECT_TRUE(packed.empty());
    power.matrix[0] = 2.0;
    EXPECT_FALSE(diffraction_grating_pack_power(power, packed, error));
  }
}

TEST(SceneDiffraction, SpectralGridConservationReciprocityAndFresnel)
{
  auto p = profile(1.5);
  p.depth = 0.0;
  const DiffractionGratingGridConfig config{8, 8, 3, 500.0, 600.0};
  std::vector<float> grid;
  std::string error;
  ASSERT_TRUE(diffraction_grating_build_grid(p, config, 8, grid, error)) << error;
  /* Cross both Bloch seams and zero with incident waves well away from grazing.
   * This is a flat-interface interpolation test, not relief-model validation. */
  for (const float wavelength : {500.0f, 537.0f, 600.0f}) {
    for (const float kx : {-0.43f, -0.38f, -0.001f, 0.001f, 0.38f, 0.43f}) {
      for (const float ky : {-0.2f, 0.2f}) {
        float r, t, reverse;
        ASSERT_TRUE(diffraction_grid_power(grid.data(), wavelength, kx, ky, false, 0, false, &r));
        ASSERT_TRUE(diffraction_grid_power(grid.data(), wavelength, kx, ky, false, 0, true, &t));
        EXPECT_NEAR(r + t, 1.0f, 2e-6f);
        ASSERT_TRUE(
            diffraction_grid_power(grid.data(), wavelength, -kx, -ky, true, 0, false, &reverse));
        EXPECT_NEAR(t, reverse, 2e-6f);
        const double ci = std::sqrt(1.0 - kx * kx - ky * ky);
        const double ct = std::sqrt(1.0 - (kx * kx + ky * ky) / 2.25);
        const double rs = (ci - 1.5 * ct) / (ci + 1.5 * ct);
        const double rp = (1.5 * ci - ct) / (1.5 * ci + ct);
        EXPECT_NEAR(r, 0.5 * (rs * rs + rp * rp), 0.003);
        float absent;
        ASSERT_TRUE(
            diffraction_grid_power(grid.data(), wavelength, kx, ky, false, 100, false, &absent));
        EXPECT_EQ(absent, 0.0f);
      }
    }
  }
  float invalid;
  EXPECT_FALSE(diffraction_grid_power(grid.data(), 450.0f, 0.0f, 0.0f, false, 0, false, &invalid));
  EXPECT_FALSE(diffraction_grid_power(grid.data(), 550.0f, 0.0f, 1.1f, false, 0, false, &invalid));
  EXPECT_FALSE(diffraction_grating_build_grid(p, config, 8, grid, error, []() { return true; }));
  EXPECT_TRUE(grid.empty());
  EXPECT_NE(error.find("cancelled"), std::string::npos);
}

TEST(SceneDiffraction, ReliefGridAllOrderReciprocityAndPassivity)
{
  const DiffractionGratingGridConfig config{8, 8, 3, 500.0, 600.0};
  for (const Complex material : {Complex(1.5), Complex(0.9, 6.0)}) {
    const auto p = profile(material);
    std::vector<float> grid;
    std::string error;
    ASSERT_TRUE(diffraction_grating_build_grid(p, config, 8, grid, error)) << error;
    for (const float wavelength : {500.0f, 537.0f, 600.0f}) {
      for (const float kx : {-0.85f, -0.38f, 0.0f, 0.38f, 0.85f}) {
        for (const float ky : {-0.2f, 0.2f}) {
          float sum = 0.0f;
          for (const bool lower : {false, true}) {
            const float index = lower ? (material.imag() == 0.0 ? 1.5f : 0.0f) : 1.0f;
            for (int order = -4; order <= 4; order++) {
              float forward;
              ASSERT_TRUE(diffraction_grid_power(
                  grid.data(), wavelength, kx, ky, false, order, lower, &forward));
              EXPECT_GE(forward, 0.0f);
              sum += forward;
              const float out_x = kx + order * wavelength / float(p.pitch);
              if (index > 0.0f && out_x * out_x + ky * ky < index * index) {
                float reverse;
                ASSERT_TRUE(diffraction_grid_power(
                    grid.data(), wavelength, -out_x, -ky, lower, order, false, &reverse));
                EXPECT_NEAR(forward, reverse, 2e-6f);
              }
            }
          }
          EXPECT_LE(sum, 1.0f + 2e-6f);
        }
      }
    }
  }
}

TEST(SceneDiffraction, MetalGrazingReflectionLimit)
{
  const auto p = profile(Complex(0.9, 6.0));
  for (const double wavelength : {450.0, 550.0, 650.0}) {
    for (const double azimuth : {0.3, 0.7, 1.2}) {
      DiffractionGratingResponse response;
      std::string error;
      ASSERT_TRUE(
          diffraction_grating_solve(p, wavelength, std::acos(1e-5), azimuth, 16, response, error))
          << error;
      const auto &zero = response.orders[16];
      EXPECT_NEAR(zero.reflection[0], 1.0, 0.001);
      EXPECT_NEAR(zero.reflection[1], 1.0, 0.001);
    }
  }
}

TEST(SceneDiffraction, GrazingExtensionPreservesReciprocityAndPassivity)
{
  const auto p = profile(Complex(0.9, 6.0));
  const DiffractionGratingGridConfig config{8, 8, 3, 500.0, 600.0, true};
  std::vector<float> grid;
  std::string error;
  ASSERT_TRUE(diffraction_grating_build_grid(p, config, 8, grid, error)) << error;
  for (const float cosine : {0.001f, 0.01f, 0.1f, 0.5f}) {
    const float kx = sqrtf(1.0f - cosine * cosine) * cosf(0.7f);
    const float ky = sqrtf(1.0f - cosine * cosine) * sinf(0.7f);
    float sum = 0.0f;
    for (int order = -3; order <= 3; order++) {
      float forward;
      ASSERT_TRUE(
          diffraction_grid_power(grid.data(), 550.0f, kx, ky, false, order, false, &forward));
      sum += forward;
      const float out_x = kx + order * 550.0f / 740.0f;
      if (out_x * out_x + ky * ky < 1.0f) {
        float reverse;
        ASSERT_TRUE(diffraction_grid_power(
            grid.data(), 550.0f, -out_x, -ky, false, order, false, &reverse));
        EXPECT_NEAR(forward, reverse, 2e-6f);
      }
    }
    EXPECT_GE(sum, 0.0f);
    EXPECT_LE(sum, 1.0f + 2e-6f);
  }
}

TEST(SceneDiffraction, ReferenceOrderBoundCoversCompleteDomain)
{
  DiffractionGratingCacheOptions options;
  options.bounds = {{-0.5, -1.5, 380}, {0.5, 1.5, 780}};
  for (const Complex material : {Complex(0.9, 6), Complex(1.5)}) {
    for (const double pitch : {740.0, 1600.0}) {
      auto p = profile(material);
      p.pitch = pitch;
      std::string error;
      const int bound = diffraction_grating_reference_order_bound(p, options, error);
      ASSERT_GE(bound, 0) << error;
      EXPECT_EQ(bound, material.imag() > 0 ? (pitch == 740 ? 2 : 4) : (pitch == 740 ? 3 : 6));
      const double n = material.imag() > 0 ? 1 : 1.5;
      for (const double wavelength : {380.0, 550.0, 780.0}) {
        for (const double bloch : {-0.5, -0.125, 0.0, 0.125, 0.5}) {
          for (int sign : {-1, 1}) {
            const double x = (sign * (bound + 1) + bloch) * wavelength / pitch;
            /* ky=0 maximizes q^2; every other conical direction is more closed. */
            EXPECT_LT(n * n - x * x, -options.cutoff_margin);
          }
        }
      }
    }
  }
  auto p = profile(Complex(0.9, 6));
  std::string error;
  p.pitch = 1.5 * 380 / std::sqrt(1 + options.cutoff_margin);
  EXPECT_EQ(diffraction_grating_reference_order_bound(p, options, error), 2);
  options.half_orders = 1;
  EXPECT_EQ(diffraction_grating_reference_order_bound(p, options, error), -1);
  EXPECT_FALSE(error.empty());
  options.half_orders = 16;
  options.bounds.lower[2] = 0;
  EXPECT_EQ(diffraction_grating_reference_order_bound(p, options, error), -1);
}

TEST(SceneDiffraction, ReducedReferenceWindowPreservesPhysicalOperator)
{
  DiffractionGratingCacheOptions options;
  options.bounds = {{-0.5, -1.5, 380}, {0.5, 1.5, 780}};
  for (const Complex material : {Complex(0.9, 6), Complex(1.5)}) {
    for (const double pitch : {740.0, 1600.0}) {
      auto p = profile(material);
      p.pitch = pitch;
      std::string error;
      const int bound = diffraction_grating_reference_order_bound(p, options, error);
      ASSERT_GE(bound, 0) << error;
      for (const double wavelength : {380.0, 550.0, 780.0}) {
        for (const double bloch : {-0.49, 0.0, 0.49}) {
          for (const double ky : {0.0, 0.99}) {
            const double kx = bloch * wavelength / pitch;
            DiffractionGratingBlock small, full, small_physical, full_physical;
            ASSERT_TRUE(diffraction_grating_solve_reference(p, wavelength, kx, ky, 16, bound, small, error)) << error;
            ASSERT_TRUE(diffraction_grating_solve_reference(p, wavelength, kx, ky, 16, 16, full, error)) << error;
            ASSERT_TRUE(diffraction_grating_match_reference(p, wavelength, kx, ky, small, small_physical, error)) << error;
            ASSERT_TRUE(diffraction_grating_match_reference(p, wavelength, kx, ky, full, full_physical, error)) << error;
            ASSERT_EQ(small_physical.ports.size(), full_physical.ports.size());
            for (size_t i = 0; i < small_physical.ports.size(); i++) {
              EXPECT_EQ(small_physical.ports[i].order, full_physical.ports[i].order);
              EXPECT_EQ(small_physical.ports[i].substrate, full_physical.ports[i].substrate);
            }
            ASSERT_EQ(small_physical.matrix.size(), full_physical.matrix.size());
            for (size_t i = 0; i < small_physical.matrix.size(); i++)
              EXPECT_LT(std::abs(small_physical.matrix[i] - full_physical.matrix[i]), 1e-9);
          }
        }
      }
    }
  }
}

TEST(SceneDiffraction, ReferencePortsMatchPhysicalMaxwellChannels)
{
  for (const Complex material : {Complex(1.5), Complex(0.9, 6.0)}) {
    for (const double depth : {0.0, 150.0}) {
      auto p = profile(material);
      p.depth = depth;
      for (const double wavelength : {450.0, 550.0, 650.0}) {
        DiffractionGratingBlock reference, matched, direct;
        std::string error;
        ASSERT_TRUE(
            diffraction_grating_solve_reference(p, wavelength, 0.13, 0.2, 16, 4, reference, error))
            << error;
        EXPECT_LE(reference.maximum_power_gain, 1.0 + 1e-10);
        if (material.imag() == 0.0) {
          EXPECT_NEAR(reference.minimum_power_gain, 1.0, 1e-10);
        }
        ASSERT_TRUE(diffraction_grating_match_reference(
            p, wavelength, 0.13, 0.2, reference, matched, error))
            << error;
        ASSERT_TRUE(diffraction_grating_solve_bloch(p, wavelength, 0.13, 0.2, 16, direct, error))
            << error;
        ASSERT_EQ(matched.matrix.size(), direct.matrix.size());
        ASSERT_EQ(matched.ports.size(), direct.ports.size());
        for (size_t i = 0; i < matched.ports.size(); i++) {
          EXPECT_EQ(matched.ports[i].order, direct.ports[i].order);
          EXPECT_EQ(matched.ports[i].substrate, direct.ports[i].substrate);
        }
        for (size_t i = 0; i < matched.matrix.size(); i++) {
          EXPECT_LT(std::abs(matched.matrix[i] - direct.matrix[i]), 1e-10);
        }
      }
    }
  }
}

TEST(SceneDiffraction, ReferencePortsComplexReciprocity)
{
  for (const Complex material : {Complex(1.5), Complex(0.9, 6.0)}) {
    const auto p = profile(material);
    DiffractionGratingBlock forward, reverse;
    std::string error;
    ASSERT_TRUE(diffraction_grating_solve_reference(p, 550.0, 0.13, 0.2, 16, 4, forward, error))
        << error;
    ASSERT_TRUE(diffraction_grating_solve_reference(p, 550.0, -0.13, -0.2, 16, 4, reverse, error))
        << error;
    const size_t channels = 2 * forward.ports.size();
    for (size_t row = 0; row < forward.ports.size(); row++) {
      for (size_t col = 0; col < forward.ports.size(); col++) {
        const size_t rr = (row / 9) * 9 + 8 - row % 9;
        const size_t rc = (col / 9) * 9 + 8 - col % 9;
        for (size_t a = 0; a < 2; a++) {
          for (size_t b = 0; b < 2; b++) {
            EXPECT_LT(std::abs(forward.matrix[(2 * row + a) * channels + 2 * col + b] -
                               reverse.matrix[(2 * rc + b) * channels + 2 * rr + a]),
                      1e-10);
          }
        }
      }
    }
  }
}

TEST(SceneDiffraction, ReferenceMatchingAtExactExternalGrazing)
{
  const auto p = profile(Complex(0.9, 6.0));
  DiffractionGratingBlock reference, physical;
  std::string error;
  ASSERT_TRUE(diffraction_grating_solve_reference(p, 740.0, 0.0, 0.0, 16, 3, reference, error))
      << error;
  ASSERT_TRUE(diffraction_grating_match_reference(p, 740.0, 0.0, 0.0, reference, physical, error))
      << error;
  ASSERT_EQ(physical.ports.size(), 1);
  EXPECT_EQ(physical.ports[0].order, 0);
  EXPECT_LE(physical.maximum_power_gain, 1.0 + 1e-10);
  for (const double delta : {-1e-8, 1e-8}) {
    DiffractionGratingResponse limit;
    ASSERT_TRUE(diffraction_grating_solve(p, 740.0 + delta, 0.0, 0.0, 16, limit, error)) << error;
    const auto &zero = limit.orders[16];
    EXPECT_NEAR(
        std::norm(physical.matrix[0]) + std::norm(physical.matrix[2]), zero.reflection[1], 1e-4);
    EXPECT_NEAR(
        std::norm(physical.matrix[1]) + std::norm(physical.matrix[3]), zero.reflection[0], 1e-4);
  }
}

TEST(SceneDiffraction, ReferenceInterpolationAcrossCutoff)
{
  for (const Complex material : {Complex(1.5), Complex(0.9, 6.0)}) {
    const auto p = profile(material);
    DiffractionGratingBlock low, high;
    std::string error;
    ASSERT_TRUE(diffraction_grating_solve_reference(p, 739.5, 0.0, 0.0, 16, 3, low, error))
        << error;
    ASSERT_TRUE(diffraction_grating_solve_reference(p, 740.5, 0.0, 0.0, 16, 3, high, error))
        << error;
    for (const double wavelength : {739.95, 740.0, 740.05}) {
      auto interpolated = low;
      const double t = wavelength - 739.5;
      for (size_t i = 0; i < low.matrix.size(); i++) {
        interpolated.matrix[i] = (1.0 - t) * low.matrix[i] + t * high.matrix[i];
      }
      DiffractionGratingBlock exact_reference, exact, approximate;
      ASSERT_TRUE(diffraction_grating_solve_reference(
          p, wavelength, 0.0, 0.0, 16, 3, exact_reference, error))
          << error;
      ASSERT_TRUE(diffraction_grating_match_reference(
          p, wavelength, 0.0, 0.0, exact_reference, exact, error))
          << error;
      ASSERT_TRUE(diffraction_grating_match_reference(
          p, wavelength, 0.0, 0.0, interpolated, approximate, error))
          << error;
      EXPECT_LE(approximate.maximum_power_gain, 1.0 + 1e-10);
      ASSERT_EQ(exact.matrix.size(), approximate.matrix.size());
      for (size_t i = 0; i < exact.matrix.size(); i++) {
        EXPECT_LT(std::abs(exact.matrix[i] - approximate.matrix[i]), 1e-4);
      }
    }
  }
}

TEST(SceneDiffraction, ReferenceMatchingRejectsInvalidOperators)
{
  const auto p = profile(Complex(0.9, 6.0));
  DiffractionGratingBlock reference, physical;
  std::string error;
  ASSERT_TRUE(diffraction_grating_solve_reference(p, 550.0, 0.0, 0.2, 16, 3, reference, error))
      << error;
  auto malformed = reference;
  malformed.ports[1] = malformed.ports[0];
  EXPECT_FALSE(
      diffraction_grating_match_reference(p, 550.0, 0.0, 0.2, malformed, physical, error));
  EXPECT_TRUE(physical.matrix.empty());
  malformed = reference;
  malformed.matrix[0] = Complex(INFINITY);
  EXPECT_FALSE(
      diffraction_grating_match_reference(p, 550.0, 0.0, 0.2, malformed, physical, error));
  EXPECT_FALSE(
      diffraction_grating_match_reference(p, 100.0, 0.0, 0.2, reference, physical, error));
  EXPECT_NE(error.find("cover all propagating"), std::string::npos);
}

TEST(SceneDiffraction, ReferenceChartsPreserveLosslessInterpolation)
{
  const auto p = profile(1.5);
  DiffractionGratingBlock low, high;
  DiffractionGratingReferenceChart a, b;
  std::string error;
  ASSERT_TRUE(diffraction_grating_solve_reference(p, 739.5, 0.0, 0.0, 16, 3, low, error)) << error;
  ASSERT_TRUE(diffraction_grating_solve_reference(p, 740.5, 0.0, 0.0, 16, 3, high, error))
      << error;
  ASSERT_TRUE(diffraction_grating_reference_to_chart(low, 0.5 * std::numbers::pi, a, error))
      << error;
  ASSERT_TRUE(diffraction_grating_reference_to_chart(high, 0.5 * std::numbers::pi, b, error))
      << error;
  EXPECT_NEAR(a.minimum_dissipation, 0.0, 1e-10);
  EXPECT_NEAR(b.minimum_dissipation, 0.0, 1e-10);
  DiffractionGratingBlock roundtrip;
  ASSERT_TRUE(diffraction_grating_chart_to_reference(a, roundtrip, error)) << error;
  for (size_t i = 0; i < low.matrix.size(); i++) {
    EXPECT_LT(std::abs(roundtrip.matrix[i] - low.matrix[i]), 1e-10);
  }
  for (const double wavelength : {739.95, 740.0, 740.05}) {
    auto interpolated = a;
    const double t = wavelength - 739.5;
    for (size_t i = 0; i < a.matrix.size(); i++) {
      interpolated.matrix[i] = (1.0 - t) * a.matrix[i] + t * b.matrix[i];
    }
    DiffractionGratingBlock reference, physical, exact_reference, exact;
    ASSERT_TRUE(diffraction_grating_chart_to_reference(interpolated, reference, error)) << error;
    EXPECT_NEAR(reference.minimum_power_gain, 1.0, 1e-10);
    EXPECT_NEAR(reference.maximum_power_gain, 1.0, 1e-10);
    ASSERT_TRUE(
        diffraction_grating_match_reference(p, wavelength, 0.0, 0.0, reference, physical, error))
        << error;
    EXPECT_NEAR(physical.minimum_power_gain, 1.0, 1e-10);
    EXPECT_NEAR(physical.maximum_power_gain, 1.0, 1e-10);
    ASSERT_TRUE(diffraction_grating_solve_reference(
        p, wavelength, 0.0, 0.0, 16, 3, exact_reference, error))
        << error;
    ASSERT_TRUE(diffraction_grating_match_reference(
        p, wavelength, 0.0, 0.0, exact_reference, exact, error))
        << error;
    for (size_t i = 0; i < exact.matrix.size(); i++) {
      EXPECT_LT(std::abs(exact.matrix[i] - physical.matrix[i]), 1e-4);
    }
  }
}

TEST(SceneDiffraction, ReferenceChartPoleRequiresAnotherChart)
{
  DiffractionGratingBlock transparent;
  transparent.ports = {{0, false}, {0, true}};
  transparent.matrix.assign(16, Complex(0.0));
  transparent.matrix[2] = transparent.matrix[7] = transparent.matrix[8] = transparent.matrix[13] =
      1.0;
  DiffractionGratingReferenceChart chart;
  std::string error;
  EXPECT_FALSE(diffraction_grating_reference_to_chart(transparent, 0.0, chart, error));
  EXPECT_TRUE(chart.matrix.empty());
  ASSERT_TRUE(
      diffraction_grating_reference_to_chart(transparent, 0.5 * std::numbers::pi, chart, error))
      << error;
  DiffractionGratingBlock restored;
  ASSERT_TRUE(diffraction_grating_chart_to_reference(chart, restored, error)) << error;
  for (size_t i = 0; i < 16; i++) {
    EXPECT_LT(std::abs(restored.matrix[i] - transparent.matrix[i]), 1e-12);
  }
}

TEST(SceneDiffraction, ReferenceChartReciprocityAndPassivity)
{
  for (const Complex material : {Complex(1.5), Complex(0.9, 6.0)}) {
    const auto p = profile(material);
    DiffractionGratingBlock forward, reverse;
    DiffractionGratingReferenceChart a, b;
    std::string error;
    ASSERT_TRUE(diffraction_grating_solve_reference(p, 550.0, 0.13, 0.2, 16, 3, forward, error))
        << error;
    ASSERT_TRUE(diffraction_grating_solve_reference(p, 550.0, -0.13, -0.2, 16, 3, reverse, error))
        << error;
    ASSERT_TRUE(diffraction_grating_reference_to_chart(forward, 1.2, a, error)) << error;
    ASSERT_TRUE(diffraction_grating_reference_to_chart(reverse, 1.2, b, error)) << error;
    EXPECT_GE(a.minimum_dissipation, -1e-10);
    EXPECT_GE(b.minimum_dissipation, -1e-10);
    const size_t ports = a.ports.size(), channels = 2 * ports;
    for (size_t row = 0; row < ports; row++) {
      for (size_t col = 0; col < ports; col++) {
        const size_t rr = (row / 7) * 7 + 6 - row % 7;
        const size_t rc = (col / 7) * 7 + 6 - col % 7;
        for (size_t i = 0; i < 2; i++) {
          for (size_t j = 0; j < 2; j++) {
            EXPECT_LT(std::abs(a.matrix[(2 * row + i) * channels + 2 * col + j] -
                               b.matrix[(2 * rc + j) * channels + 2 * rr + i]),
                      1e-9);
          }
        }
      }
    }
    /* An average of these two passive admittances remains passive, independent
     * of whether they are a useful approximation at a particular query. */
    for (size_t i = 0; i < a.matrix.size(); i++) {
      a.matrix[i] = 0.37 * a.matrix[i] + 0.63 * b.matrix[i];
    }
    DiffractionGratingBlock interpolated;
    ASSERT_TRUE(diffraction_grating_chart_to_reference(a, interpolated, error)) << error;
    EXPECT_LE(interpolated.maximum_power_gain, 1.0 + 1e-10);
    if (material.imag() == 0.0) {
      EXPECT_NEAR(interpolated.minimum_power_gain, 1.0, 1e-10);
    }
  }
}

namespace {
template<int N>
void check_float_reference_matching(const double pitch, const Complex material, const int retained)
{
  DiffractionReferenceFixture fixture;
  std::string error;
  constexpr int cases = 64;
  ASSERT_TRUE(diffraction_reference_fixture(pitch, material, retained, cases, fixture, error))
      << error;
  ASSERT_EQ(fixture.channels, N);
  float maximum_error = 0.0f;
  for (int i = 0; i < cases; i++) {
    float2 result[2 * N];
    ASSERT_TRUE(diffraction_reference_match<N>(fixture.charts.data() + size_t(i) * N * N,
                                               fixture.boundaries.data() + size_t(i) * 2 * N,
                                               fixture.rotations[i],
                                               fixture.incoming[i],
                                               result))
        << "case " << i;
    for (int j = 0; j < 2 * N; j++) {
      const float difference = len(result[j] - fixture.expected[size_t(i) * 2 * N + j]);
      maximum_error = std::max(maximum_error, difference);
      EXPECT_LT(difference, 3e-5f) << "case " << i << " coefficient " << j;
    }
    for (int col = 0; col < 2; col++) {
      float power = 0.0f;
      for (int row = 0; row < N; row++) {
        power += len_squared(result[2 * row + col]);
      }
      EXPECT_LE(power, 1.0f + 1e-4f);
      if (material.imag() == 0.0) {
        EXPECT_NEAR(power, 1.0f, 1e-4f);
      }
    }
  }
  std::ostringstream error_value;
  error_value << std::setprecision(12) << maximum_error;
  ::testing::Test::RecordProperty("matching_error_" + std::to_string(N), error_value.str());
}
}  // namespace

TEST(SceneDiffraction, FloatReferenceMatchingAgainstDoubleMaxwell)
{
  check_float_reference_matching<10>(740.0, Complex(0.9, 6.0), 2);
  check_float_reference_matching<18>(1600.0, Complex(0.9, 6.0), 4);
  check_float_reference_matching<20>(740.0, Complex(1.5), 2);
}

TEST(SceneDiffraction, ReferenceChartSelectionAvoidsPoleAngles)
{
  DiffractionGratingBlock reference;
  reference.ports = {{-1, false}, {0, false}, {1, false}, {-1, true}, {0, true}, {1, true}};
  reference.matrix.assign(144, Complex(0.0));
  const double pi = std::numbers::pi;
  const std::array<double, 6> poles{
      0.5 * pi, -0.5 * pi, 0.25 * pi, -0.25 * pi, 0.75 * pi, -0.75 * pi};
  for (int i = 0; i < 12; i++) {
    reference.matrix[12 * i + i] = -std::exp(Complex(0.0, -poles[i / 2]));
  }
  const std::array<DiffractionGratingBlock, 2> group{reference, reference};
  std::vector<DiffractionGratingReferenceChart> charts;
  double phase;
  std::string error;
  ASSERT_TRUE(diffraction_grating_choose_chart(group, phase, charts, error)) << error;
  ASSERT_EQ(charts.size(), 2);
  EXPECT_EQ(charts[0].rotation, charts[1].rotation);
  for (const auto &chart : charts) {
    EXPECT_LT(chart.frobenius_norm, 20.0);
    DiffractionGratingBlock restored;
    ASSERT_TRUE(diffraction_grating_chart_to_reference(chart, restored, error)) << error;
    for (size_t i = 0; i < reference.matrix.size(); i++) {
      EXPECT_LT(std::abs(reference.matrix[i] - restored.matrix[i]), 1e-10);
    }
  }
  EXPECT_FALSE(diffraction_grating_choose_chart({}, phase, charts, error));
  EXPECT_TRUE(charts.empty());
}

TEST(SceneDiffraction, HybridMatchingEqualsDirectMaxwell)
{
  for (const Complex material : {Complex(1.5), Complex(0.9, 6.0)}) {
    const auto p = profile(material);
    for (const double wavelength : {450.0, 740.0, 850.0}) {
      DiffractionGratingBlock reference, direct;
      std::string error;
      ASSERT_TRUE(
          diffraction_grating_solve_reference(p, wavelength, 0.0, 0.0, 16, 4, reference, error))
          << error;
      ASSERT_TRUE(
          diffraction_grating_match_reference(p, wavelength, 0.0, 0.0, reference, direct, error))
          << error;
      for (int pattern = 0; pattern < 4; pattern++) {
        std::vector<unsigned char> keep(reference.ports.size());
        for (size_t i = 0; i < keep.size(); i++) {
          keep[i] = pattern == 0 ? 0 :
                    pattern == 1 ? 1 :
                    pattern == 2 ? std::abs(reference.ports[i].order) == 1 :
                                   i % 2;
        }
        DiffractionGratingHybrid hybrid;
        DiffractionGratingBlock matched;
        ASSERT_TRUE(diffraction_grating_prepare_hybrid(
            p, wavelength, 0.0, 0.0, reference, keep, hybrid, error))
            << error;
        EXPECT_LE(hybrid.scattering.maximum_power_gain, 1.0 + 1e-10);
        if (material.imag() == 0.0) {
          EXPECT_NEAR(hybrid.scattering.minimum_power_gain, 1.0, 1e-10);
        }
        ASSERT_TRUE(
            diffraction_grating_match_hybrid(p, wavelength, 0.0, 0.0, hybrid, matched, error))
            << error;
        ASSERT_EQ(matched.matrix.size(), direct.matrix.size());
        for (size_t i = 0; i < matched.matrix.size(); i++) {
          EXPECT_LT(std::abs(matched.matrix[i] - direct.matrix[i]), 1e-10);
        }
      }
    }
  }
}

TEST(SceneDiffraction, HybridRejectsUnrepresentedCutoffs)
{
  const auto p = profile(Complex(0.9, 6.0));
  DiffractionGratingBlock reference, matched;
  DiffractionGratingHybrid hybrid;
  std::string error;
  ASSERT_TRUE(diffraction_grating_solve_reference(p, 739.0, 0.0, 0.0, 16, 3, reference, error))
      << error;
  std::vector<unsigned char> keep(reference.ports.size(), 0);
  ASSERT_TRUE(
      diffraction_grating_prepare_hybrid(p, 739.0, 0.0, 0.0, reference, keep, hybrid, error))
      << error;
  EXPECT_FALSE(diffraction_grating_match_hybrid(p, 741.0, 0.0, 0.0, hybrid, matched, error));
  EXPECT_NE(error.find("crossed"), std::string::npos);
  EXPECT_TRUE(matched.matrix.empty());
  ASSERT_TRUE(
      diffraction_grating_prepare_hybrid(p, 741.0, 0.0, 0.0, reference, keep, hybrid, error))
      << error;
  EXPECT_FALSE(diffraction_grating_match_hybrid(p, 739.0, 0.0, 0.0, hybrid, matched, error));
  EXPECT_NE(error.find("cover all propagating"), std::string::npos);
  for (size_t i = 0; i < keep.size(); i++) {
    keep[i] = std::abs(reference.ports[i].order) == 1;
  }
  ASSERT_TRUE(
      diffraction_grating_prepare_hybrid(p, 740.0, 0.0, 0.0, reference, keep, hybrid, error))
      << error;
  for (double wavelength : {739.0, 740.0, 741.0}) {
    ASSERT_TRUE(diffraction_grating_match_hybrid(p, wavelength, 0.0, 0.0, hybrid, matched, error))
        << error;
    EXPECT_LE(matched.maximum_power_gain, 1.0 + 1e-10);
  }
  hybrid.is_reference.pop_back();
  EXPECT_FALSE(diffraction_grating_match_hybrid(p, 740.0, 0.0, 0.0, hybrid, matched, error));
}

TEST(SceneDiffraction, HybridWithoutPhysicalChannels)
{
  const auto p = profile(Complex(0.9, 6.0));
  DiffractionGratingBlock reference, matched;
  DiffractionGratingHybrid hybrid;
  std::string error;
  ASSERT_TRUE(diffraction_grating_solve_reference(p, 550.0, 0.0, 2.0, 16, 3, reference, error))
      << error;
  std::vector<unsigned char> keep(reference.ports.size(), 0);
  ASSERT_TRUE(
      diffraction_grating_prepare_hybrid(p, 550.0, 0.0, 2.0, reference, keep, hybrid, error))
      << error;
  EXPECT_TRUE(hybrid.scattering.ports.empty());
  ASSERT_TRUE(diffraction_grating_match_hybrid(p, 550.0, 0.0, 2.0, hybrid, matched, error))
      << error;
  EXPECT_TRUE(matched.matrix.empty());
  EXPECT_FALSE(diffraction_grating_match_hybrid(p, 550.0, 0.0, 0.0, hybrid, matched, error));
  EXPECT_NE(error.find("cover all propagating"), std::string::npos);
}

namespace {
template<int C> void check_float_hybrid_matching()
{
  for (const Complex material : {Complex(1.5), Complex(0.9, 6.0)}) {
    DiffractionReferenceFixture fixture;
    std::string error;
    ASSERT_TRUE(diffraction_reference_fixture(740.0, material, 2, 64, fixture, error, C / 2))
        << error;
    const int n = fixture.channels;
    std::vector<float2> result(2 * n);
    for (int i = 0; i < 64; i++) {
      ASSERT_TRUE(
          diffraction_hybrid_match<C>(fixture.charts.data() + size_t(i) * n * n,
                                      fixture.boundaries.data() + size_t(i) * 2 * n,
                                      C ? fixture.active_ports.data() + i * (C / 2) : nullptr,
                                      n,
                                      fixture.incoming[i],
                                      result.data()))
          << i;
      for (int j = 0; j < 2 * n; j++) {
        EXPECT_LT(len(result[j] - fixture.expected[size_t(i) * 2 * n + j]), 3e-5f)
            << "feedback " << C << " case " << i << " coefficient " << j;
      }
    }
  }
}
}  // namespace

TEST(SceneDiffraction, FloatHybridMatchingAgainstDoubleMaxwell)
{
  check_float_hybrid_matching<0>();
  check_float_hybrid_matching<2>();
  check_float_hybrid_matching<4>();
}

TEST(SceneDiffraction, HybridCornerPowerInterpolationIsLosslessAndReciprocal)
{
  const auto p = profile(Complex(1.5));
  std::array<DiffractionGratingHybrid, 2> corners;
  std::string error;
  for (int i = 0; i < 2; i++) {
    const double wavelength = i ? 740.5 : 739.5;
    DiffractionGratingBlock reference;
    ASSERT_TRUE(
        diffraction_grating_solve_reference(p, wavelength, 0.0, 0.0, 16, 3, reference, error))
        << error;
    std::vector<unsigned char> keep(reference.ports.size());
    for (size_t j = 0; j < keep.size(); j++) {
      keep[j] = !reference.ports[j].substrate && std::abs(reference.ports[j].order) == 1;
    }
    ASSERT_TRUE(diffraction_grating_prepare_hybrid(
        p, wavelength, 0.0, 0.0, reference, keep, corners[i], error))
        << error;
  }
  for (double wavelength : {739.9, 740.0, 740.1}) {
    std::array<DiffractionGratingPowerBlock, 2> powers;
    for (int i = 0; i < 2; i++) {
      DiffractionGratingBlock matched;
      ASSERT_TRUE(
          diffraction_grating_match_hybrid(p, wavelength, 0.0, 0.0, corners[i], matched, error))
          << error;
      diffraction_grating_power_block(matched, powers[i]);
    }
    ASSERT_EQ(powers[0].matrix.size(), powers[1].matrix.size());
    const double t = wavelength - 739.5;
    auto mixed = powers[0];
    for (size_t i = 0; i < mixed.matrix.size(); i++) {
      mixed.matrix[i] = (1.0 - t) * powers[0].matrix[i] + t * powers[1].matrix[i];
    }
    const size_t n = mixed.ports.size();
    std::vector<size_t> reverse(n);
    for (size_t i = 0; i < n; i++) {
      const auto found = std::find_if(
          mixed.ports.begin(), mixed.ports.end(), [&](const DiffractionGratingPort &port) {
            return port.order == -mixed.ports[i].order &&
                   port.substrate == mixed.ports[i].substrate;
          });
      ASSERT_NE(found, mixed.ports.end());
      reverse[i] = size_t(found - mixed.ports.begin());
    }
    for (size_t col = 0; col < n; col++) {
      double sum = 0.0;
      for (size_t row = 0; row < n; row++) {
        sum += mixed.matrix[row * n + col];
        EXPECT_NEAR(
            mixed.matrix[row * n + col], mixed.matrix[reverse[col] * n + reverse[row]], 1e-10);
      }
      EXPECT_NEAR(sum, 1.0, 1e-10);
    }
  }
}

TEST(SceneDiffraction, PreparedCellMatchesCornersAndPreservesEnergy)
{
  const DiffractionGratingCellBounds bounds{{-0.03, -0.04, 735.0}, {0.03, 0.04, 745.0}};
  std::string error;
  for (const Complex material : {Complex(1.5), Complex(0.9, 6.0)}) {
    const auto p = profile(material);
    DiffractionGratingCell cell;
    ASSERT_TRUE(diffraction_grating_prepare_cell(p, bounds, 16, 3, 0.1, cell, error)) << error;
    EXPECT_GE(cell.feedback_channels, 4);
    for (int corner = 0; corner < 8; corner++) {
      const std::array<double, 3> query{corner & 1 ? bounds.upper[0] : bounds.lower[0],
                                        corner & 2 ? bounds.upper[1] : bounds.lower[1],
                                        corner & 4 ? bounds.upper[2] : bounds.lower[2]};
      DiffractionGratingPowerBlock power, exact;
      DiffractionGratingBlock direct;
      ASSERT_TRUE(diffraction_grating_cell_power(p, cell, query, power, error)) << error;
      ASSERT_TRUE(diffraction_grating_solve_bloch(
          p, query[2], query[0] * query[2] / p.pitch, query[1], 16, direct, error))
          << error;
      diffraction_grating_power_block(direct, exact);
      ASSERT_EQ(power.matrix.size(), exact.matrix.size());
      for (size_t i = 0; i < power.matrix.size(); i++) {
        EXPECT_NEAR(power.matrix[i], exact.matrix[i], 1e-10);
      }
    }
    for (const double wavelength : {739.9, 740.0, 740.1}) {
      DiffractionGratingPowerBlock power;
      ASSERT_TRUE(diffraction_grating_cell_power(p, cell, {0.0, 0.0, wavelength}, power, error))
          << error;
      EXPECT_LE(power.maximum_column_sum, 1.0 + 1e-10);
      if (material.imag() == 0.0) {
        const size_t n = power.ports.size();
        for (size_t col = 0; col < n; col++) {
          double sum = 0.0;
          for (size_t row = 0; row < n; row++) {
            sum += power.matrix[row * n + col];
          }
          EXPECT_NEAR(sum, 1.0, 1e-10);
        }
      }
    }
    DiffractionGratingPowerBlock power;
    EXPECT_FALSE(diffraction_grating_cell_power(p, cell, {0.1, 0.0, 740.0}, power, error));
    EXPECT_TRUE(power.matrix.empty());
  }
}

TEST(SceneDiffraction, CenteredLamellarMirrorSymmetryIncludesComplexPhase)
{
  for (const Complex material : {Complex(1.5), Complex(0.9, 6.0)}) {
    const auto p = profile(material);
    for (const bool artificial : {false, true}) {
      auto solve = [&](double x, double y, DiffractionGratingBlock &block, std::string &error) {
        return artificial ?
                   diffraction_grating_solve_reference(p, 550.0, x, y, 16, 3, block, error) :
                   diffraction_grating_solve_bloch(p, 550.0, x, y, 16, block, error);
      };
      std::string error;
      DiffractionGratingBlock original;
      ASSERT_TRUE(solve(0.17, 0.23, original, error)) << error;
      for (int mask = 1; mask < 4; mask++) {
        const bool x_flip = mask & 1, y_flip = mask & 2;
        DiffractionGratingBlock mirrored;
        ASSERT_TRUE(solve(x_flip ? -0.17 : 0.17, y_flip ? -0.23 : 0.23, mirrored, error)) << error;
        ASSERT_EQ(mirrored.ports.size(), original.ports.size());
        const int n = int(original.ports.size());
        std::vector<int> mapping(n);
        for (int i = 0; i < n; i++) {
          const auto &port = original.ports[i];
          const auto found = std::find_if(mirrored.ports.begin(),
                                          mirrored.ports.end(),
                                          [&](const DiffractionGratingPort &other) {
                                            return other.order ==
                                                       (x_flip ? -port.order : port.order) &&
                                                   other.substrate == port.substrate;
                                          });
          ASSERT_NE(found, mirrored.ports.end());
          mapping[i] = int(found - mirrored.ports.begin());
        }
        for (int row = 0; row < 2 * n; row++) {
          for (int col = 0; col < 2 * n; col++) {
            const int row_sign = row % 2 ? (y_flip ? -1 : 1) : (x_flip ? -1 : 1);
            const int col_sign = col % 2 ? (y_flip ? -1 : 1) : (x_flip ? -1 : 1);
            const Complex expected = double(row_sign * col_sign) *
                                     original.matrix[row * 2 * n + col];
            const Complex actual = mirrored.matrix[(2 * mapping[row / 2] + row % 2) * 2 * n +
                                                   2 * mapping[col / 2] + col % 2];
            EXPECT_LT(std::abs(actual - expected), 1e-10);
          }
        }
      }
    }
  }
}

TEST(SceneDiffraction, FloatExteriorBoundaryCoefficients)
{
  const auto fixture = diffraction_boundary_fixture();
  const int count = int(fixture.inputs.size() / 2);
  float maximum_error = 0.0f;
  for (int i = 0; i < count; i++) {
    const float4 a = fixture.inputs[2 * i], b = fixture.inputs[2 * i + 1];
    DiffractionGratingBoundary boundary;
    ASSERT_TRUE(diffraction_grating_boundary(
        make_float3(a.x, a.y, a.z), a.w, b.x, b.y, b.z, int(b.w), &boundary))
        << i;
    const std::array<float2, 4> values{
        boundary.r_te, boundary.r_tm, boundary.transmission, boundary.tangent_direction};
    for (int j = 0; j < 4; j++) {
      const float difference = len(values[j] - fixture.expected[5 * i + j]);
      maximum_error = std::max(maximum_error, difference);
      EXPECT_LT(difference, 3e-5f) << "case " << i << " coefficient " << j;
    }
    const float q2 = fixture.expected[5 * i + 4].x;
    EXPECT_NEAR(boundary.q2, q2, 3e-6f * (1.0f + fabsf(q2)));
    if (fabsf(q2) > 1e-12f || q2 == 0.0f) {
      EXPECT_EQ(boundary.q2 > 0.0f, q2 > 0.0f) << i;
    }
    EXPECT_NEAR(len_squared(boundary.r_te) + boundary.transmission.x * boundary.transmission.x,
                1.0f,
                3e-6f);
    EXPECT_NEAR(len_squared(boundary.r_tm) + boundary.transmission.y * boundary.transmission.y,
                1.0f,
                3e-6f);
  }
  std::ostringstream value;
  value << std::setprecision(12) << maximum_error;
  ::testing::Test::RecordProperty("maximum_boundary_error", value.str());
}

CCL_NAMESPACE_END

CCL_NAMESPACE_BEGIN
TEST(SceneDiffraction, PackedCellFloatEvaluationAndValidation)
{
  const DiffractionGratingCellBounds bounds{{-0.03, -0.04, 735.0}, {0.03, 0.04, 745.0}};
  for (const Complex material : {Complex(1.5), Complex(0.9, 6.0)}) {
    const auto p = profile(material);
    std::string error;
    DiffractionGratingCell cell;
    ASSERT_TRUE(diffraction_grating_prepare_cell(p, bounds, 16, 3, 0.1, cell, error)) << error;
    DiffractionGratingPackedCell packed;
    ASSERT_TRUE(diffraction_grating_pack_cell(cell, packed, error)) << error;
    const int ports = packed.ports.size(), channels = 2 * ports;
    ASSERT_LE(channels, 32);
    ASSERT_EQ(packed.matrices.size(), 8 * channels * channels);
    for (const float3 t : {make_float3(0, 0, 0),
                           make_float3(1, 1, 1),
                           make_float3(0.5f, 0.5f, 0.49f),
                           make_float3(0.5f, 0.5f, 0.5f),
                           make_float3(0.5f, 0.5f, 0.51f),
                           make_float3(0.17f, 0.73f, 0.37f)})
    {
      const std::array<double, 3> q{bounds.lower[0] + t.x * (bounds.upper[0] - bounds.lower[0]),
                                    bounds.lower[1] + t.y * (bounds.upper[1] - bounds.lower[1]),
                                    bounds.lower[2] + t.z * (bounds.upper[2] - bounds.lower[2])};
      DiffractionGratingPowerBlock expected;
      ASSERT_TRUE(diffraction_grating_cell_power(p, cell, q, expected, error)) << error;
      std::vector<float2> boundary;
      for (int port = 0; port < ports; port++) {
        const auto &entry = packed.ports[port];
        const double n = entry.substrate ? material.real() : p.incident_ior;
        const double x = (q[0] + entry.order) * q[2] / p.pitch, y = q[1];
        const double q2 = n * n - x * x - y * y;
        const Complex z = q2 > 0 ? Complex(std::sqrt(q2)) : Complex(0, std::sqrt(-q2));
        auto pair = [](Complex v) { return make_float2(float(v.real()), float(v.imag())); };
        const bool reference = cell.corners[0].is_reference[port];
        boundary.push_back(reference ? pair((1.0 - z) / (1.0 + z)) : zero_float2());
        boundary.push_back(reference ? pair((z - n * n) / (z + n * n)) : zero_float2());
        boundary.push_back(
            !reference ? make_float2(1, 1) :
            q2 > 0     ? make_float2(float(2 * std::sqrt(z.real()) / (1 + z.real())),
                                 float(2 * n * std::sqrt(z.real()) / (z.real() + n * n))) :
                         zero_float2());
        const double transverse = std::hypot(x, y);
        boundary.push_back(transverse > 0 ?
                               make_float2(float(x / transverse), float(y / transverse)) :
                               make_float2(1, 0));
      }
      for (int incoming = 0; incoming < int(expected.ports.size()); incoming++) {
        auto find_port = [&](const DiffractionGratingPort &target) {
          for (int j = 0; j < ports; j++)
            if (packed.ports[j].order == target.order &&
                packed.ports[j].substrate == target.substrate)
              return j;
          return -1;
        };
        const int input = find_port(expected.ports[incoming]);
        ASSERT_GE(input, 0);
        float powers[16];
        auto evaluate = [&]<int C>() {
          return diffraction_hybrid_cell_power<32, C>(packed.matrices.data(),
                                                      boundary.data(),
                                                      packed.active_ports.data(),
                                                      channels,
                                                      input,
                                                      t,
                                                      powers);
        };
        bool ok = false;
        switch (cell.feedback_channels) {
          case 0:
            ok = evaluate.operator()<0>();
            break;
          case 2:
            ok = evaluate.operator()<2>();
            break;
          case 4:
            ok = evaluate.operator()<4>();
            break;
          case 6:
            ok = evaluate.operator()<6>();
            break;
          case 8:
            ok = evaluate.operator()<8>();
            break;
          case 10:
            ok = evaluate.operator()<10>();
            break;
          case 12:
            ok = evaluate.operator()<12>();
            break;
        }
        ASSERT_TRUE(ok) << cell.feedback_channels;
        double sum = 0;
        for (int row = 0; row < int(expected.ports.size()); row++) {
          const int output = find_port(expected.ports[row]);
          ASSERT_GE(output, 0);
          EXPECT_NEAR(
              powers[output], expected.matrix[row * expected.ports.size() + incoming], 3e-5);
          sum += powers[output];
        }
        EXPECT_LE(sum, 1.0 + 3e-5);
        if (material.imag() == 0)
          EXPECT_NEAR(sum, 1.0, 3e-5);
      }
    }
    auto bad = cell;
    bad.corners[7].is_reference[0] ^= 1;
    EXPECT_FALSE(diffraction_grating_pack_cell(bad, packed, error));
    EXPECT_TRUE(packed.matrices.empty());
    bad = cell;
    bad.corners[3].scattering.matrix[0] = Complex(INFINITY);
    EXPECT_FALSE(diffraction_grating_pack_cell(bad, packed, error));
    EXPECT_TRUE(packed.matrices.empty());
    bad = cell;
    bad.feedback_channels += 2;
    EXPECT_FALSE(diffraction_grating_pack_cell(bad, packed, error));
  }
}
CCL_NAMESPACE_END

CCL_NAMESPACE_BEGIN
TEST(SceneDiffraction, CompleteCacheBuildAndLimits)
{
  DiffractionGratingCacheOptions options;
  options.bounds = {{-0.03125, -0.0625, 735}, {0.03125, 0.0625, 745}};
  options.tolerance = 0.1;
  DiffractionGratingCache cache;
  DiffractionGratingCacheStats stats;
  std::string error;
  const auto p = profile(Complex(0.9, 6));
  ASSERT_TRUE(diffraction_grating_build_cache(p, options, cache, stats, error)) << error;
  EXPECT_EQ(cache.nodes.size(), 1);
  EXPECT_EQ(cache.cells.size(), 1);
  EXPECT_EQ(stats.accepted_cells, 1);
  EXPECT_DOUBLE_EQ(stats.accepted_domain_fraction, 1.0);
  EXPECT_GT(stats.matrix_bytes, 0);
  EXPECT_LE(stats.maximum_accepted_error, options.tolerance);
  const float3 lo = make_float3(-0.03125f, -0.0625f, 735),
               hi = make_float3(0.03125f, 0.0625f, 745);
  for (const float3 q : {lo, hi, make_float3(0, 0, 740)})
    EXPECT_EQ(diffraction_cache_lookup(cache.nodes.data(), cache.nodes.size(), lo, hi, q), 0);
  EXPECT_EQ(diffraction_cache_lookup(
                cache.nodes.data(), cache.nodes.size(), lo, hi, make_float3(0, 0, 746)),
            -1);
  auto limited = options;
  limited.maximum_matrix_bytes = 0;
  EXPECT_FALSE(diffraction_grating_build_cache(p, limited, cache, stats, error));
  EXPECT_TRUE(cache.nodes.empty());
  EXPECT_TRUE(cache.cells.empty());
  limited = options;
  limited.tolerance = 1e-15;
  limited.maximum_nodes = 1;
  EXPECT_FALSE(diffraction_grating_build_cache(p, limited, cache, stats, error));
  EXPECT_TRUE(cache.nodes.empty());
  EXPECT_EQ(stats.visited_nodes, 1);
  limited = options;
  limited.progress = [](const DiffractionGratingCacheStats &) { return false; };
  EXPECT_FALSE(diffraction_grating_build_cache(p, limited, cache, stats, error));
  EXPECT_TRUE(cache.nodes.empty());
  EXPECT_TRUE(cache.cells.empty());
  EXPECT_EQ(stats.reference_solves, 0);
  EXPECT_DOUBLE_EQ(stats.accepted_domain_fraction, 0.0);
  limited = options;
  limited.bounds.lower[0] = -0.03;
  EXPECT_FALSE(diffraction_grating_build_cache(p, limited, cache, stats, error));
  EXPECT_EQ(stats.visited_nodes, 0);
}

TEST(SceneDiffraction, CacheLookupSplitOwnershipAndMalformedIndex)
{
  const float3 lo = make_float3(-0.5f, -1, 380), hi = make_float3(0.5f, 1, 780);
  int4 nodes[] = {
      make_int4(0, 1, 2, __float_as_int(0.0f)), make_int4(-1, 7, 0, 0), make_int4(-1, 9, 0, 0)};
  EXPECT_EQ(diffraction_cache_lookup(nodes, 3, lo, hi, make_float3(-0.1f, 0, 550)), 7);
  EXPECT_EQ(diffraction_cache_lookup(nodes, 3, lo, hi, make_float3(0, 0, 550)), 9);
  EXPECT_EQ(diffraction_cache_lookup(nodes, 3, lo, hi, make_float3(0.5f, 1, 780)), 9);
  EXPECT_EQ(diffraction_cache_lookup(nodes, 3, lo, hi, make_float3(NAN, 0, 550)), -1);
  nodes[0].z = 0;
  EXPECT_EQ(diffraction_cache_lookup(nodes, 3, lo, hi, make_float3(0, 0, 550)), -1);
  nodes[0].z = 3;
  EXPECT_EQ(diffraction_cache_lookup(nodes, 3, lo, hi, make_float3(0, 0, 550)), -1);
}
CCL_NAMESPACE_END

CCL_NAMESPACE_BEGIN
TEST(SceneDiffraction, CacheSolveReuseIsExactAndBounded)
{
  DiffractionGratingCacheOptions options;
  options.bounds = {{-0.03125, -0.0625, 735}, {0.03125, 0.0625, 745}};
  options.tolerance = 0.1;
  options.maximum_nodes = 256;
  options.allow_chart_cells = false;
  const auto p = profile(Complex(0.9, 6));
  DiffractionGratingCache probe, cached, uncached;
  DiffractionGratingCacheStats probe_stats, cached_stats, uncached_stats;
  std::string error;
  ASSERT_TRUE(diffraction_grating_build_cache(p, options, probe, probe_stats, error)) << error;
  ASSERT_GT(probe_stats.maximum_accepted_error, 1e-10);
  options.tolerance = 0.5 * probe_stats.maximum_accepted_error;
  options.reference_cache_matrix_bytes = 512 * 1024;
  ASSERT_TRUE(diffraction_grating_build_cache(p, options, cached, cached_stats, error)) << error;
  EXPECT_GT(cached.nodes.size(), 1);
  EXPECT_NEAR(cached_stats.accepted_domain_fraction, 1.0, 1e-15);
  EXPECT_GT(cached_stats.reference_cache_hits, 0);
  EXPECT_LE(cached_stats.peak_reference_matrix_bytes, options.reference_cache_matrix_bytes);
  options.reference_cache_matrix_bytes = 0;
  ASSERT_TRUE(diffraction_grating_build_cache(p, options, uncached, uncached_stats, error))
      << error;
  EXPECT_EQ(uncached_stats.reference_cache_hits, 0);
  EXPECT_EQ(uncached_stats.peak_reference_matrix_bytes, 0);
  EXPECT_LT(cached_stats.reference_solves, uncached_stats.reference_solves);
  ASSERT_EQ(cached.nodes.size(), uncached.nodes.size());
  ASSERT_EQ(cached.cells.size(), uncached.cells.size());
  for (size_t i = 0; i < cached.nodes.size(); i++) {
    EXPECT_EQ(cached.nodes[i].x, uncached.nodes[i].x);
    EXPECT_EQ(cached.nodes[i].y, uncached.nodes[i].y);
    EXPECT_EQ(cached.nodes[i].z, uncached.nodes[i].z);
    EXPECT_EQ(cached.nodes[i].w, uncached.nodes[i].w);
  }
  for (size_t i = 0; i < cached.cells.size(); i++) {
    const auto &a = cached.cells[i], &b = uncached.cells[i];
    EXPECT_EQ(a.bounds.lower, b.bounds.lower);
    EXPECT_EQ(a.bounds.upper, b.bounds.upper);
    EXPECT_EQ(a.active_ports, b.active_ports);
    ASSERT_EQ(a.matrices.size(), b.matrices.size());
    for (size_t j = 0; j < a.matrices.size(); j++) {
      EXPECT_EQ(a.matrices[j].x, b.matrices[j].x);
      EXPECT_EQ(a.matrices[j].y, b.matrices[j].y);
    }
  }
}
CCL_NAMESPACE_END

CCL_NAMESPACE_BEGIN
TEST(SceneDiffraction, HybridChartCellPreservesComplexCornersAndPassivity)
{
  const DiffractionGratingCellBounds bounds{{-0.03125, -0.0625, 735}, {0.03125, 0.0625, 745}};
  for (const Complex material : {Complex(1.5), Complex(0.9, 6)}) {
    const auto p = profile(material);
    std::string error;
    DiffractionGratingCell cell;
    ASSERT_TRUE(diffraction_grating_prepare_cell(p, bounds, 16, 5, 0.1, cell, error)) << error;
    DiffractionGratingChartCell chart;
    ASSERT_TRUE(diffraction_grating_prepare_chart_cell(cell, chart, error)) << error;
    for (int corner = 0; corner < 8; corner++) {
      const std::array<double, 3> q{corner & 1 ? bounds.upper[0] : bounds.lower[0],
                                    corner & 2 ? bounds.upper[1] : bounds.lower[1],
                                    corner & 4 ? bounds.upper[2] : bounds.lower[2]};
      DiffractionGratingBlock actual, expected;
      ASSERT_TRUE(diffraction_grating_chart_cell_match(p, chart, q, actual, error)) << error;
      ASSERT_TRUE(diffraction_grating_match_hybrid(
          p, q[2], q[0] * q[2] / p.pitch, q[1], cell.corners[corner], expected, error))
          << error;
      ASSERT_EQ(actual.matrix.size(), expected.matrix.size());
      for (size_t j = 0; j < actual.matrix.size(); j++)
        EXPECT_LT(std::abs(actual.matrix[j] - expected.matrix[j]), 1e-9);
    }
    for (int sample = 0; sample < 27; sample++) {
      std::array<double, 3> q;
      int lattice = sample;
      for (int a = 0; a < 3; a++) {
        q[a] = bounds.lower[a] + 0.25 * (1 + lattice % 3) * (bounds.upper[a] - bounds.lower[a]);
        lattice /= 3;
      }
      DiffractionGratingBlock actual;
      ASSERT_TRUE(diffraction_grating_chart_cell_match(p, chart, q, actual, error)) << error;
      EXPECT_LE(actual.maximum_power_gain, 1 + 1e-9);
      if (material.imag() == 0) {
        EXPECT_NEAR(actual.minimum_power_gain, 1, 1e-9);
        EXPECT_NEAR(actual.maximum_power_gain, 1, 1e-9);
      }
    }
    DiffractionGratingBlock actual;
    EXPECT_FALSE(diffraction_grating_chart_cell_match(p, chart, {0.5, 0, 740}, actual, error));
    EXPECT_TRUE(actual.matrix.empty());
    chart.corners[3].rotation = -chart.corners[3].rotation;
    EXPECT_FALSE(diffraction_grating_chart_cell_match(p, chart, {0, 0, 740}, actual, error));
    EXPECT_TRUE(actual.matrix.empty());
  }
}
CCL_NAMESPACE_END

CCL_NAMESPACE_BEGIN
TEST(SceneDiffraction, CacheSelectsValidatedChartBeforeRefining)
{
  DiffractionGratingCacheOptions options;
  options.bounds = {{-0.48046875, -0.96875, 389.375}, {-0.4765625, -0.9609375, 390.9375}};
  options.maximum_depth = 0;
  options.tolerance = 0.001;
  const auto p = profile(Complex(0.9, 6));
  DiffractionGratingCache cache;
  DiffractionGratingCacheStats stats;
  std::string error;
  options.allow_chart_cells = false;
  EXPECT_FALSE(diffraction_grating_build_cache(p, options, cache, stats, error));
  EXPECT_TRUE(cache.cells.empty());
  options.allow_chart_cells = true;
  ASSERT_TRUE(diffraction_grating_build_cache(p, options, cache, stats, error)) << error;
  ASSERT_EQ(cache.cells.size(), 1);
  EXPECT_EQ(cache.nodes.size(), 1);
  EXPECT_TRUE(cache.cells[0].operator_chart);
  EXPECT_NEAR(len_squared(cache.cells[0].chart_rotation), 1, 2e-7);
  EXPECT_EQ(stats.accepted_chart_cells, 1);
  EXPECT_EQ(stats.accepted_domain_fraction, 1.0);
  EXPECT_LE(stats.maximum_accepted_error, options.tolerance);
  DiffractionGratingCell original;
  DiffractionGratingChartCell chart;
  ASSERT_TRUE(diffraction_grating_prepare_cell(p, options.bounds, 16, 5, 0.1, original, error))
      << error;
  ASSERT_TRUE(diffraction_grating_prepare_chart_cell(original, chart, error)) << error;
  DiffractionGratingPackedCell packed;
  ASSERT_TRUE(diffraction_grating_pack_chart_cell(chart, packed, error)) << error;
  ASSERT_EQ(packed.matrices.size(), cache.cells[0].matrices.size());
  for (size_t i = 0; i < packed.matrices.size(); i++) {
    EXPECT_EQ(packed.matrices[i].x, cache.cells[0].matrices[i].x);
    EXPECT_EQ(packed.matrices[i].y, cache.cells[0].matrices[i].y);
  }
  chart.corners[7].rotation = -chart.corners[7].rotation;
  EXPECT_FALSE(diffraction_grating_pack_chart_cell(chart, packed, error));
  EXPECT_TRUE(packed.matrices.empty());
}
CCL_NAMESPACE_END

CCL_NAMESPACE_BEGIN
TEST(SceneDiffraction, MirrorCacheLookupAndComplexReconstruction)
{
  for (const Complex material : {Complex(0.9, 6), Complex(1.5)}) {
    const auto p = profile(material);
    DiffractionGratingCacheOptions options;
    options.bounds = {{-0.03125, -0.0625, 735}, {0.03125, 0.0625, 745}};
    options.tolerance = 0.1;
    options.mirror_symmetry = true;
    DiffractionGratingCache cache;
    DiffractionGratingCacheStats stats;
    std::string error;
    ASSERT_TRUE(diffraction_grating_build_cache(p, options, cache, stats, error)) << error;
    ASSERT_TRUE(cache.mirror_symmetry);
    EXPECT_EQ(cache.bounds.upper[0], 0);
    EXPECT_EQ(cache.bounds.upper[1], 0);
    ASSERT_EQ(cache.cells.size(), 1);
    EXPECT_EQ(stats.accepted_domain_fraction, 1);
    DiffractionGratingCell cell;
    DiffractionGratingChartCell chart;
    ASSERT_TRUE(diffraction_grating_prepare_cell(p, cache.bounds, 16, 5, 0.1, cell, error))
        << error;
    ASSERT_TRUE(diffraction_grating_prepare_chart_cell(cell, chart, error)) << error;
    for (float wavelength : {735.0f, 745.0f}) {
      for (float x_sign : {-1.0f, 1.0f}) {
        for (float y_sign : {-1.0f, 1.0f}) {
          const float3 query = make_float3(x_sign * 0.03125f, y_sign * 0.0625f, wavelength);
          const auto transform = diffraction_cache_symmetry(query);
          EXPECT_EQ(diffraction_cache_lookup(cache.nodes.data(),
                                             cache.nodes.size(),
                                             make_float3(-0.03125f, -0.0625f, 735),
                                             make_float3(0, 0, 745),
                                             query,
                                             true),
                    0);
          DiffractionGratingBlock folded, reference, exact;
          ASSERT_TRUE(diffraction_grating_chart_cell_match(
              p, chart, {transform.query.x, transform.query.y, transform.query.z}, folded, error))
              << error;
          const double x = double(query.x) * wavelength / p.pitch;
          ASSERT_TRUE(diffraction_grating_solve_reference(
              p, wavelength, x, query.y, 16, 5, reference, error))
              << error;
          ASSERT_TRUE(diffraction_grating_match_reference(
              p, wavelength, x, query.y, reference, exact, error))
              << error;
          ASSERT_EQ(folded.ports.size(), exact.ports.size());
          const int n = exact.ports.size();
          auto map = [&](int port) {
            const int order = transform.reverse_orders ? -exact.ports[port].order :
                                                         exact.ports[port].order;
            for (int j = 0; j < n; j++)
              if (folded.ports[j].order == order &&
                  folded.ports[j].substrate == exact.ports[port].substrate)
                return j;
            return -1;
          };
          for (int row = 0; row < 2 * n; row++) {
            for (int col = 0; col < 2 * n; col++) {
              const int r = map(row / 2), c = map(col / 2);
              ASSERT_GE(r, 0);
              ASSERT_GE(c, 0);
              const double sign = row % 2 == col % 2 ? 1 : transform.cross_polarization_sign;
              EXPECT_LT(
                  std::abs(exact.matrix[row * 2 * n + col] -
                           sign * folded.matrix[(2 * r + row % 2) * 2 * n + 2 * c + col % 2]),
                  1e-9);
            }
          }
        }
      }
    }
    /* Opposite directions share the same folded operator. Symmetry of that
     * operator then establishes reciprocal complex amplitudes, including at
     * interpolated interior queries rather than just the original corners. */
    for (const std::array<double, 3> q :
         {std::array<double, 3>{-0.01, -0.02, 739.9}, {-0.02, -0.04, 740.1}})
    {
      DiffractionGratingBlock matched;
      ASSERT_TRUE(diffraction_grating_chart_cell_match(p, chart, q, matched, error)) << error;
      const int n = 2 * matched.ports.size();
      for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
          EXPECT_LT(std::abs(matched.matrix[r * n + c] - matched.matrix[c * n + r]), 1e-9);
    }
    options.bounds.upper[0] = 0.0625;
    EXPECT_FALSE(diffraction_grating_build_cache(p, options, cache, stats, error));
    EXPECT_TRUE(cache.nodes.empty());
  }
}
CCL_NAMESPACE_END

CCL_NAMESPACE_BEGIN
TEST(SceneDiffraction, ParallelCacheValidationMatchesSerialExactly)
{
  const auto p = profile(Complex(0.9, 6));
  for (int mode = 0; mode < 4; mode++) {
    const bool use_chart = (mode & 1) != 0;
    DiffractionGratingCacheOptions options;
    options.bounds = use_chart ? DiffractionGratingCellBounds{{-0.48046875, -0.96875, 389.375},
                                                              {-0.4765625, -0.9609375, 390.9375}} :
                                 DiffractionGratingCellBounds{{-0.03125, -0.0625, 735},
                                                              {0.03125, 0.0625, 745}};
    options.allow_chart_cells = use_chart;
    options.curvature_adaptive_splits = (mode & 2) != 0;
    options.maximum_nodes = 256;
    options.tolerance = use_chart ? 0.001 : 0.1;
    DiffractionGratingCache serial, parallel;
    DiffractionGratingCacheStats a, b;
    std::string error;
    if (!use_chart) {
      ASSERT_TRUE(diffraction_grating_build_cache(p, options, serial, a, error)) << error;
      options.tolerance = a.maximum_accepted_error * 0.5;
    }
    ASSERT_TRUE(diffraction_grating_build_cache(p, options, serial, a, error)) << error;
    options.validation_workers = 4;
    ASSERT_TRUE(diffraction_grating_build_cache(p, options, parallel, b, error)) << error;
    ASSERT_EQ(serial.nodes.size(), parallel.nodes.size());
    ASSERT_EQ(serial.cells.size(), parallel.cells.size());
    EXPECT_EQ(a.maximum_accepted_error, b.maximum_accepted_error);
    EXPECT_EQ(a.accepted_domain_fraction, b.accepted_domain_fraction);
    EXPECT_EQ(a.accepted_chart_cells, b.accepted_chart_cells);
    for (size_t i = 0; i < serial.nodes.size(); i++) {
      EXPECT_EQ(serial.nodes[i].x, parallel.nodes[i].x);
      EXPECT_EQ(serial.nodes[i].y, parallel.nodes[i].y);
      EXPECT_EQ(serial.nodes[i].z, parallel.nodes[i].z);
      EXPECT_EQ(serial.nodes[i].w, parallel.nodes[i].w);
    }
    for (size_t i = 0; i < serial.cells.size(); i++) {
      const auto &x = serial.cells[i], &y = parallel.cells[i];
      EXPECT_EQ(x.bounds.lower, y.bounds.lower);
      EXPECT_EQ(x.bounds.upper, y.bounds.upper);
      EXPECT_EQ(x.operator_chart, y.operator_chart);
      EXPECT_EQ(x.chart_rotation.x, y.chart_rotation.x);
      EXPECT_EQ(x.chart_rotation.y, y.chart_rotation.y);
      EXPECT_EQ(x.active_ports, y.active_ports);
      ASSERT_EQ(x.ports.size(), y.ports.size());
      for (size_t j = 0; j < x.ports.size(); j++) {
        EXPECT_EQ(x.ports[j].order, y.ports[j].order);
        EXPECT_EQ(x.ports[j].substrate, y.ports[j].substrate);
      }
      ASSERT_EQ(x.matrices.size(), y.matrices.size());
      for (size_t j = 0; j < x.matrices.size(); j++) {
        EXPECT_EQ(x.matrices[j].x, y.matrices[j].x);
        EXPECT_EQ(x.matrices[j].y, y.matrices[j].y);
      }
    }
  }
}
CCL_NAMESPACE_END

CCL_NAMESPACE_BEGIN
TEST(SceneDiffraction, ComplexCacheRequiresPhaseAccuracy)
{
  const auto p = profile(Complex(0.9, 6));
  DiffractionGratingCacheOptions options;
  options.bounds = {{-0.03125, -0.0625, 735}, {0.03125, 0.0625, 745}};
  options.maximum_depth = 0;
  options.tolerance = 1;
  options.complex_tolerance = 1;
  DiffractionGratingCache cache;
  DiffractionGratingCacheStats stats;
  std::string error;
  ASSERT_TRUE(diffraction_grating_build_cache(p, options, cache, stats, error)) << error;
  EXPECT_TRUE(cache.complex_validated);
  ASSERT_FALSE(cache.cells.empty());
  for (const auto &cell : cache.cells)
    EXPECT_TRUE(cell.operator_chart);
  const double amplitude_error = stats.maximum_accepted_complex_error;
  ASSERT_GT(amplitude_error, 0);
  options.complex_tolerance = amplitude_error * 0.5;
  EXPECT_FALSE(diffraction_grating_build_cache(p, options, cache, stats, error));
  EXPECT_TRUE(cache.cells.empty());
  EXPECT_FALSE(cache.complex_validated);
  EXPECT_GT(stats.last_complex_error, options.complex_tolerance);
  EXPECT_LT(stats.last_validation_error, options.tolerance);
  options.maximum_depth = 16;
  options.maximum_nodes = 256;
  options.validation_workers = 4;
  ASSERT_TRUE(diffraction_grating_build_cache(p, options, cache, stats, error)) << error;
  EXPECT_TRUE(cache.complex_validated);
  EXPECT_GT(cache.cells.size(), 1);
  EXPECT_LE(stats.maximum_accepted_complex_error, options.complex_tolerance);
  for (const auto &cell : cache.cells)
    EXPECT_TRUE(cell.operator_chart);
  options.complex_tolerance = 1;
  options.allow_chart_cells = false;
  EXPECT_FALSE(diffraction_grating_build_cache(p, options, cache, stats, error));
}
CCL_NAMESPACE_END

CCL_NAMESPACE_BEGIN
TEST(SceneDiffraction, StoredComplexCacheLookupAndFloatMatching)
{
  const auto p = profile(Complex(0.9, 6));
  DiffractionGratingCacheOptions options;
  options.bounds = {{-0.03125, -0.0625, 735}, {0.03125, 0.0625, 745}};
  options.retained_half_orders = 1;
  options.cutoff_margin = 100;
  options.mirror_symmetry = true;
  options.complex_tolerance = 0.001;
  options.maximum_nodes = 1024;
  DiffractionGratingCache cache;
  DiffractionGratingCacheStats stats;
  std::string error;
  ASSERT_TRUE(diffraction_grating_build_cache(p, options, cache, stats, error)) << error;
  ASSERT_TRUE(cache.complex_validated);
  double maximum_error = 0;
  for (int sample = 0; sample < 37; sample++) {
    const float3 query = make_float3(
        float(-0.03125 + 0.0625 * (sample + 0.37) / 37),
        float(-0.0625 + 0.125 * std::fmod(sample * 0.618 + 0.17, 1.0)),
        float(735 + 10 * std::fmod(sample * 0.414 + 0.29, 1.0)));
    const auto symmetry = diffraction_cache_symmetry(query);
    const int leaf = diffraction_cache_lookup(cache.nodes.data(),
                                              int(cache.nodes.size()),
                                              make_float3(-0.03125f, -0.0625f, 735),
                                              make_float3(0, 0, 745),
                                              query,
                                              true);
    ASSERT_GE(leaf, 0);
    ASSERT_LT(leaf, cache.cells.size());
    const auto &cell = cache.cells[leaf];
    ASSERT_TRUE(cell.operator_chart);
    ASSERT_EQ(cell.ports.size(), 3);
    ASSERT_EQ(cell.active_ports.size(), 3);
    float3 t;
    t.x = float((symmetry.query.x - cell.bounds.lower[0]) /
                (cell.bounds.upper[0] - cell.bounds.lower[0]));
    t.y = float((symmetry.query.y - cell.bounds.lower[1]) /
                (cell.bounds.upper[1] - cell.bounds.lower[1]));
    t.z = float((symmetry.query.z - cell.bounds.lower[2]) /
                (cell.bounds.upper[2] - cell.bounds.lower[2]));
    float2 boundary[12];
    auto pair = [](Complex v) { return make_float2(float(v.real()), float(v.imag())); };
    for (int j = 0; j < 3; j++) {
      const double x = (double(symmetry.query.x) + cell.ports[j].order) * query.z / p.pitch;
      const double y = symmetry.query.y;
      const double q2 = 1 - x * x - y * y;
      const Complex z = q2 > 0 ? Complex(std::sqrt(q2)) : Complex(0, std::sqrt(-q2));
      boundary[4 * j] = pair((1.0 - z) / (1.0 + z));
      boundary[4 * j + 1] = pair((z - 1.0) / (z + 1.0));
      const float transmission = q2 > 0 ? float(2 * std::sqrt(z.real()) / (1 + z.real())) : 0;
      boundary[4 * j + 2] = make_float2(transmission, transmission);
      const double length = std::hypot(x, y);
      boundary[4 * j + 3] = length > 0 ? make_float2(float(x / length), float(y / length)) :
                                         make_float2(1, 0);
    }
    DiffractionGratingBlock reference, exact;
    const double kx = double(query.x) * query.z / p.pitch;
    ASSERT_TRUE(
        diffraction_grating_solve_reference(p, query.z, kx, query.y, 16, 1, reference, error))
        << error;
    ASSERT_TRUE(
        diffraction_grating_match_reference(p, query.z, kx, query.y, reference, exact, error))
        << error;
    const int n = 2 * int(exact.ports.size());
    auto stored_port = [&](int port) {
      const int order = exact.ports[port].order * (symmetry.reverse_orders ? -1 : 1);
      for (int j = 0; j < 3; j++)
        if (cell.ports[j].order == order)
          return j;
      return -1;
    };
    double squared_error = 0;
    for (int incoming = 0; incoming < n / 2; incoming++) {
      const int input = stored_port(incoming);
      ASSERT_GE(input, 0);
      float2 amplitudes[12];
      ASSERT_TRUE(diffraction_chart_cell_match<6>(
          cell.matrices.data(), boundary, cell.chart_rotation, input, t, amplitudes));
      for (int outgoing = 0; outgoing < n / 2; outgoing++) {
        const int output = stored_port(outgoing);
        ASSERT_GE(output, 0);
        for (int r = 0; r < 2; r++)
          for (int c = 0; c < 2; c++) {
            const float2 value = amplitudes[(2 * output + r) * 2 + c];
            const Complex actual = Complex(value.x, value.y) *
                                   double(r == c ? 1 : symmetry.cross_polarization_sign);
            squared_error += std::norm(actual -
                                       exact.matrix[(2 * outgoing + r) * n + 2 * incoming + c]);
          }
      }
    }
    maximum_error = std::max(maximum_error, std::sqrt(squared_error));
  }
  EXPECT_LT(maximum_error, 0.002);
  ::testing::Test::RecordProperty("maximum_stored_cache_complex_error",
                                  std::to_string(maximum_error));
}
CCL_NAMESPACE_END

CCL_NAMESPACE_BEGIN
TEST(SceneDiffraction, PassiveQuadraticChartAccuracyAndPacking)
{
  for (int test = 0; test < 3; test++) {
    auto p = profile(test == 1 ? Complex(1.5) : Complex(0.9, 6));
    if (test == 2)
      p.pitch = 1600;
    const double center = test == 2 ? 400 : 740;
    const DiffractionGratingCellBounds bounds{{-0.03125, -0.0625, center - 5},
                                              {0.03125, 0.0625, center + 5}};
    DiffractionGratingCell cell;
    DiffractionGratingChartCell chart;
    std::string error;
    ASSERT_TRUE(diffraction_grating_prepare_cell(p, bounds, 16, 5, 0.1, cell, error)) << error;
    ASSERT_TRUE(diffraction_grating_prepare_quadratic_chart(p, cell, 16, 5, chart, error))
        << error;
    ASSERT_EQ(chart.degree, 2);
    ASSERT_EQ(chart.corners.size(), 27);
    for (const auto &control : chart.corners)
      EXPECT_GE(control.minimum_dissipation, -1e-9);
    DiffractionGratingPackedCell packed;
    ASSERT_TRUE(diffraction_grating_pack_chart_cell(chart, packed, error)) << error;
    EXPECT_TRUE(packed.operator_chart);
    EXPECT_EQ(packed.chart_degree, 2);
    const size_t channels = 2 * packed.ports.size();
    EXPECT_EQ(packed.matrices.size(), 27 * channels * channels);
    for (int sample = 0; sample < 64; sample++) {
      int lattice = sample;
      std::array<double, 3> q;
      for (int a = 0; a < 3; a++) {
        const double t = sample < 27 ? 0.5 * (lattice % 3) :
                                       std::fmod((sample + 0.31) * (0.317 + 0.137 * a), 1.0);
        lattice /= 3;
        q[a] = bounds.lower[a] + t * (bounds.upper[a] - bounds.lower[a]);
      }
      DiffractionGratingBlock actual, reference, exact;
      ASSERT_TRUE(diffraction_grating_chart_cell_match(p, chart, q, actual, error)) << error;
      ASSERT_TRUE(diffraction_grating_solve_reference(
          p, q[2], q[0] * q[2] / p.pitch, q[1], 16, 5, reference, error))
          << error;
      ASSERT_TRUE(diffraction_grating_match_reference(
          p, q[2], q[0] * q[2] / p.pitch, q[1], reference, exact, error))
          << error;
      ASSERT_EQ(actual.matrix.size(), exact.matrix.size());
      double complex_error = 0;
      for (size_t j = 0; j < actual.matrix.size(); j++)
        complex_error = std::hypot(complex_error, std::abs(actual.matrix[j] - exact.matrix[j]));
      EXPECT_LT(complex_error, sample < 27 ? 1e-8 : 0.003) << test << " " << sample;
      EXPECT_LE(actual.maximum_power_gain, 1 + 1e-8);
      if (test == 1)
        EXPECT_NEAR(actual.minimum_power_gain, 1, 1e-8);
    }
    chart.corners.pop_back();
    EXPECT_FALSE(diffraction_grating_pack_chart_cell(chart, packed, error));
    EXPECT_TRUE(packed.matrices.empty());
    DiffractionGratingBlock invalid;
    EXPECT_FALSE(diffraction_grating_chart_cell_match(p, chart, {0, 0, center}, invalid, error));
  }
}
CCL_NAMESPACE_END

CCL_NAMESPACE_BEGIN
TEST(SceneDiffraction, CacheSelectsPassiveQuadraticBeforeSubdivision)
{
  const auto p = profile(Complex(1.5));
  DiffractionGratingCacheOptions options;
  options.bounds = {{-0.03125, -0.0625, 735}, {0.03125, 0.0625, 745}};
  options.maximum_depth = 0;
  options.tolerance = 0.001;
  DiffractionGratingCache cache;
  DiffractionGratingCacheStats stats;
  std::string error;
  EXPECT_FALSE(diffraction_grating_build_cache(p, options, cache, stats, error));
  options.allow_quadratic_cells = true;
  ASSERT_TRUE(diffraction_grating_build_cache(p, options, cache, stats, error)) << error;
  ASSERT_EQ(cache.cells.size(), 1);
  EXPECT_EQ(cache.cells[0].chart_degree, 2);
  EXPECT_EQ(stats.accepted_quadratic_cells, 1);
  EXPECT_LE(stats.maximum_accepted_error, options.tolerance);
  options.complex_tolerance = 0.001;
  ASSERT_TRUE(diffraction_grating_build_cache(p, options, cache, stats, error)) << error;
  EXPECT_TRUE(cache.complex_validated);
  EXPECT_EQ(stats.accepted_quadratic_cells, 1);
  EXPECT_LE(stats.maximum_accepted_complex_error, options.complex_tolerance);
}
CCL_NAMESPACE_END

CCL_NAMESPACE_BEGIN
TEST(SceneDiffraction, InplaceChartMatchingNearGrazingAndRealRotations)
{
  const Complex scattering(0.3, 0.2);
  for (double phase : {0.0,
                       std::numbers::pi,
                       std::numbers::pi - 0.001,
                       std::numbers::pi + 0.001,
                       0.5 * std::numbers::pi})
    for (double q : {1e-12, 1e-6, 0.01, 0.2}) {
      const Complex rotation = std::exp(Complex(0, phase));
      const Complex y = (1.0 - rotation * scattering) / (1.0 + rotation * scattering);
      const float2 pair = make_float2(float(y.real()), float(y.imag()));
      float2 chart[4] = {pair, zero_float2(), zero_float2(), pair};
      const float r = float((1 - q) / (1 + q)), t = float(2 * std::sqrt(q) / (1 + q));
      const float2 boundary[4] = {
          make_float2(r, 0), make_float2(-r, 0), make_float2(t, t), make_float2(1, 0)};
      float2 actual[4];
      ASSERT_TRUE(diffraction_reference_match_inplace<2>(
          chart,
          boundary,
          make_float2(float(rotation.real()), float(rotation.imag())),
          0,
          actual));
      for (int row = 0; row < 2; row++)
        for (int col = 0; col < 2; col++) {
          const double reflection = row ? r : -r;
          const Complex exact = row == col ?
                                    double(t) * t * scattering / (1.0 - reflection * scattering) -
                                        reflection :
                                    Complex(0);
          EXPECT_LT(std::abs(Complex(actual[2 * row + col].x, actual[2 * row + col].y) - exact),
                    2e-6)
              << phase << " " << q;
        }
    }
}
CCL_NAMESPACE_END

CCL_NAMESPACE_BEGIN
TEST(SceneDiffraction, QuadraticCacheChecksNearCellFaces)
{
  const auto p = profile(Complex(0.9, 6));
  DiffractionGratingCacheOptions options;
  options.bounds = {{-.125, -.71875, 630}, {0, -.6875, 680}};
  options.retained_half_orders = 2;
  options.allow_quadratic_cells = true;
  options.maximum_depth = 0;
  DiffractionGratingCache cache;
  DiffractionGratingCacheStats stats;
  std::string error;
  /* The former quarter-point-only rule accepted this region, whose held-out
   * physical power-column error reached 0.00249566 near two cell faces. */
  EXPECT_FALSE(diffraction_grating_build_cache(p, options, cache, stats, error));
  EXPECT_TRUE(cache.cells.empty());
  EXPECT_GT(stats.last_validation_error, options.tolerance);
  options.maximum_depth = 16;
  options.maximum_nodes = 256;
  ASSERT_TRUE(diffraction_grating_build_cache(p, options, cache, stats, error)) << error;
  EXPECT_GT(cache.cells.size(), 1);
  EXPECT_LE(stats.maximum_accepted_error, options.tolerance);
}
TEST(SceneDiffraction, DeviceBuffersFlattenAndRejectInvalidLayouts)
{
  DiffractionGratingCache cache;
  cache.bounds = {{-0.5, -1, 380}, {0.5, 1, 780}};
  cache.nodes = {
      make_int4(0, 1, 2, __float_as_int(0.0f)), make_int4(-1, 0, 0, 0), make_int4(-1, 1, 0, 0)};
  DiffractionGratingPackedCell cell;
  cell.bounds = cache.bounds;
  cell.ports = {{0, false}};
  cell.active_ports = {0};
  cell.matrices.resize(32, make_float2(0.25f, 0));
  cache.cells.push_back(cell);
  cell.operator_chart = true;
  cell.chart_degree = 2;
  cell.chart_rotation = make_float2(1, 0);
  cell.matrices.resize(108, make_float2(0.125f, 0));
  cache.cells.push_back(cell);
  DiffractionGratingDeviceBuffers buffers;
  std::string error;
  ASSERT_TRUE(diffraction_grating_device_buffers(cache, buffers, error)) << error;
  EXPECT_EQ(buffers.nodes.size(), 3);
  EXPECT_EQ(buffers.layout.size(), 4);
  EXPECT_EQ(buffers.layout[2].x, 32);
  EXPECT_EQ(buffers.layout[2].y, 1);
  EXPECT_EQ(buffers.layout[2].z, 1);
  EXPECT_EQ(buffers.layout[2].w, 2);
  EXPECT_EQ(buffers.matrices.size(), 140);
  EXPECT_EQ(buffers.matrices[32].x, 0.25f);
  const auto valid = cache;
  cache.cells[1].active_ports = {1};
  EXPECT_FALSE(diffraction_grating_device_buffers(cache, buffers, error));
  EXPECT_TRUE(buffers.nodes.empty());
  EXPECT_TRUE(buffers.matrices.empty());
  cache = valid;
  cache.nodes[0].y = 0;
  EXPECT_FALSE(diffraction_grating_device_buffers(cache, buffers, error));
  cache = valid;
  cache.cells[1].matrices.pop_back();
  EXPECT_FALSE(diffraction_grating_device_buffers(cache, buffers, error));
  cache = valid;
  cache.cells[1].bounds.upper[0] = cache.cells[1].bounds.lower[0];
  EXPECT_FALSE(diffraction_grating_device_buffers(cache, buffers, error));
  cache = valid;
  cache.nodes[1].y = 2;
  EXPECT_FALSE(diffraction_grating_device_buffers(cache, buffers, error));
}
TEST(SceneDiffraction, ManagerRegistrationIsTransactional)
{
  DiffractionManager manager;
  EXPECT_FALSE(manager.need_update());
  EXPECT_EQ(manager.cache_count(), 0);
  DiffractionGratingCache cache;
  cache.bounds = {{-0.5, -1, 380}, {0.5, 1, 780}};
  cache.nodes = {make_int4(-1, -1, 0, 0)};
  std::vector<DiffractionGratingCache> caches{cache, cache};
  std::string error;
  EXPECT_EQ(manager.add_cache(cache, error), 0);
  EXPECT_EQ(manager.add_cache(cache, error), 1);
  auto invalid = cache;
  invalid.nodes[0].y = 99;
  EXPECT_EQ(manager.add_cache(invalid, error), -1);
  EXPECT_EQ(manager.cache_count(), 2);
  EXPECT_EQ(manager.add_cache(cache, error), 2);
  ASSERT_TRUE(manager.set_caches(caches, error)) << error;
  EXPECT_TRUE(manager.need_update());
  EXPECT_EQ(manager.cache_count(), 2);
  caches[1].nodes[0].y = 99;
  EXPECT_FALSE(manager.set_caches(caches, error));
  EXPECT_FALSE(error.empty());
  EXPECT_EQ(manager.cache_count(), 2);
  EXPECT_TRUE(manager.need_update());
  ASSERT_TRUE(manager.set_caches({}, error)) << error;
  EXPECT_TRUE(error.empty());
  EXPECT_EQ(manager.cache_count(), 0);
  EXPECT_TRUE(manager.need_update());
}
TEST(SceneDiffraction, MaterialCacheReuseAndFailure)
{
  DiffractionManager manager;
  DiffractionGratingProfile profile{200, 0, 0.5, 1, 1.5, 1, 1.5};
  DiffractionGratingCacheOptions options;
  options.bounds = {{-0.001f, -0.001f, 549.9f}, {0.001f, 0.001f, 550.1f}};
  options.half_orders = 2;
  options.retained_half_orders = 1;
  options.tolerance = 1e-4;
  DiffractionGratingCacheStats stats;
  std::string error;
  const int first = manager.get_or_build(profile, options, stats, error);
  ASSERT_EQ(first, 0) << error;
  const auto initial_stats = stats;
  ASSERT_GT(stats.reference_solves, 0);
  int callbacks = 0;
  options.progress = [&](const auto &) { callbacks++; return true; };
  EXPECT_EQ(manager.get_or_build(profile, options, stats, error), first);
  EXPECT_EQ(callbacks, 1); /* A hit, not another builder traversal. */
  EXPECT_EQ(stats.reference_solves, initial_stats.reference_solves);
  EXPECT_EQ(manager.cache_count(), 1);
  options.progress = [](const auto &) { return false; };
  EXPECT_EQ(manager.get_or_build(profile, options, stats, error), -1);
  EXPECT_FALSE(error.empty());
  EXPECT_EQ(manager.cache_count(), 1);
  options.progress = {};
  auto different = profile;
  different.substrate_ior = 1.6;
  EXPECT_EQ(manager.get_or_build(different, options, stats, error), 1) << error;
  auto stricter = options;
  stricter.tolerance *= 0.5;
  EXPECT_EQ(manager.get_or_build(profile, stricter, stats, error), 2) << error;
  auto invalid = profile;
  invalid.pitch = -1;
  EXPECT_EQ(manager.get_or_build(invalid, options, stats, error), -1);
  EXPECT_EQ(manager.cache_count(), 3);
  /* Failed set replacement preserves material identities; successful clear
   * invalidates them, so the next request rebuilds at handle zero. */
  DiffractionGratingCache bad;
  EXPECT_FALSE(manager.set_caches(std::span(&bad, 1), error));
  EXPECT_EQ(manager.get_or_build(profile, options, stats, error), first);
  ASSERT_TRUE(manager.set_caches({}, error));
  EXPECT_EQ(manager.cache_count(), 0);
  EXPECT_EQ(manager.get_or_build(profile, options, stats, error), 0) << error;
  EXPECT_EQ(manager.cache_count(), 1);
}
TEST(SceneDiffraction, MaterialEvaluatorCapacity)
{
  DiffractionManager manager;
  DiffractionGratingProfile profile{200, 0, 0.5, 1, 1.5, 1, 1.5};
  DiffractionGratingCacheOptions options;
  options.bounds = {{-0.001f, -0.001f, 549.9f}, {0.001f, 0.001f, 550.1f}};
  options.half_orders = 2;
  options.retained_half_orders = 1;
  DiffractionGratingCacheStats stats;
  std::string error;
  /* Two ports, four polarization channels. Reject before registration. */
  EXPECT_EQ(manager.get_or_build(profile, options, stats, error, 2), -1);
  EXPECT_NE(error.find("requires 4"), std::string::npos) << error;
  EXPECT_EQ(manager.cache_count(), 0);
  EXPECT_FALSE(manager.need_update());
  ASSERT_EQ(manager.get_or_build(profile, options, stats, error, 4), 0) << error;
  EXPECT_EQ(manager.cache_count(), 1);
  /* A cache hit cannot bypass the requesting evaluator's smaller capacity. */
  EXPECT_EQ(manager.get_or_build(profile, options, stats, error, 2), -1);
  EXPECT_NE(error.find("supports 2"), std::string::npos) << error;
  EXPECT_EQ(manager.cache_count(), 1);
  EXPECT_EQ(manager.get_or_build(profile, options, stats, error, 4), 0);
  EXPECT_TRUE(error.empty());
  EXPECT_EQ(manager.get_or_build(profile, options, stats, error, -1), -1);
}
TEST(SceneDiffraction, OrderSolidAngleJacobian)
{
  int checked = 0, shifted = 0;
  for (const float ni : {1.0f, 1.5f})
    for (const float no : {0.7f, 1.0f, 1.5f})
      for (const float x : {-0.6f, -0.2f, 0.2f, 0.6f})
        for (const float y : {-0.25f, 0.25f})
          for (const int order : {-1, 0, 1})
            for (const bool transmission : {false, true}) {
              const float3 incident = make_float3(x, y, -std::sqrt(1 - x*x - y*y));
              float3 outgoing;
              if (!diffraction_grating_direction(incident, ni, no, 550, 1100, order,
                                                  transmission, &outgoing) ||
                  std::abs(outgoing.z) < 0.25f)
                continue;
              float jacobian, reverse;
              ASSERT_TRUE(diffraction_grating_solid_angle_jacobian(incident, outgoing, ni, no,
                                                                   &jacobian));
              ASSERT_TRUE(diffraction_grating_solid_angle_jacobian(-outgoing, -incident, no, ni,
                                                                   &reverse));
              EXPECT_NEAR(jacobian * reverse, 1, 3e-6);
              /* Perturb in orthonormal tangent directions on the incident
               * sphere. The cross product of output derivatives measures
               * output solid angle per unit incident solid angle. */
              const float3 a = normalize(cross(incident, make_float3(0, 0, 1)));
              const float3 b = cross(incident, a);
              constexpr float h = 0.0002f;
              float3 ap, am, bp, bm;
              ASSERT_TRUE(diffraction_grating_direction(normalize(incident + h*a), ni, no,
                  550, 1100, order, transmission, &ap));
              ASSERT_TRUE(diffraction_grating_direction(normalize(incident - h*a), ni, no,
                  550, 1100, order, transmission, &am));
              ASSERT_TRUE(diffraction_grating_direction(normalize(incident + h*b), ni, no,
                  550, 1100, order, transmission, &bp));
              ASSERT_TRUE(diffraction_grating_direction(normalize(incident - h*b), ni, no,
                  550, 1100, order, transmission, &bm));
              const float measured = len(cross((ap-am)/(2*h), (bp-bm)/(2*h)));
              EXPECT_NEAR(measured / jacobian, 1, 0.0015);
              checked++;
              shifted += order != 0;
            }
  EXPECT_GT(checked, 100);
  EXPECT_GT(shifted, 50);
  float jacobian = 99;
  EXPECT_FALSE(diffraction_grating_solid_angle_jacobian(
      make_float3(1, 0, 0), make_float3(0, 0, 1), 1, 1, &jacobian));
  EXPECT_EQ(jacobian, 0);
  EXPECT_FALSE(diffraction_grating_solid_angle_jacobian(
      make_float3(0, 0, -1), make_float3(1, 0, 0), 1, 1, &jacobian));
  EXPECT_FALSE(diffraction_grating_solid_angle_jacobian(
      make_float3(0, 0, -1), make_float3(0, 0, 1), 0, 1, &jacobian));
}
TEST(SceneDiffraction, GrazingDirectionProbabilityKeepsSidesDistinct)
{
  DiffractionGratingCache cache;
  cache.bounds = {{-0.5, -1, 500}, {0.5, 1, 600}};
  cache.nodes = {make_int4(-1, 0, 0, 0)};
  DiffractionGratingPackedCell cell;
  cell.bounds = cache.bounds;
  cell.ports = {{0, false}, {0, true}};
  cell.matrices.resize(8 * 16, zero_float2());
  for (int corner = 0; corner < 8; corner++)
    for (int c = 0; c < 4; c++) {
      cell.matrices[16*corner + 4*c + c] = make_float2(0.5f, 0);
      cell.matrices[16*corner + 4*((c+2)%4) + c] = make_float2(0.5f, 0);
    }
  cache.cells = {cell};
  DiffractionGratingDeviceBuffers buffers;
  std::string error;
  ASSERT_TRUE(diffraction_grating_device_buffers(cache, buffers, error)) << error;
  const int4 descriptors[] = {make_int4(0, 1, 0, 1), make_int4(0, 0, 0, 0)};
  const float4 domains[] = {make_float4(-0.5f, -1, 500, 0), make_float4(0.5f, 1, 600, 0)};
  DiffractionSceneData data{};
  data.cache_count = 1;
  data.descriptors = descriptors;
  data.domains = domains;
  data.nodes = buffers.nodes.data();
  data.layout = buffers.layout.data();
  data.bounds = buffers.bounds.data();
  data.ports = buffers.ports.data();
  data.active = buffers.active.data();
  data.matrices = buffers.matrices.data();
  for (bool below : {false, true})
    for (float z : {1e-5f, 1e-6f, 1e-7f})
      for (bool outgoing_below : {false, true}) {
        const float3 incident = make_float3(1, 0, below ? z : -z);
        float3 outgoing;
        ASSERT_TRUE(diffraction_grating_direction(incident, 1, 1, 550, 200, 0,
                                                  outgoing_below, &outgoing));
        float power, probability;
        ASSERT_TRUE(diffraction_data_direction_probability<4>(&data, 0, incident, below,
            1, 1, 550, 200, outgoing, &power, &probability));
        EXPECT_NEAR(power, 0.25f, 2e-6f);
        EXPECT_NEAR(probability, 0.5f, 2e-6f);
      }
}

TEST(SceneDiffraction, TwoSidedGlassAlbedoAndReciprocalReturn)
{
  DiffractionTwoSidedAlbedoRequest request;
  request.alpha_x = 0.35f;
  request.alpha_y = 0.6f;
  request.pitch_nm = 1200.0f;
  request.depth_nm = 125.0f;
  request.duty = 0.43f;
  request.inside_ior = 1.5f;
  request.wavelength_count = 3;
  request.mu_count = 4;
  request.phi_count = 8;
  request.facet_samples = 256;
  DiffractionTwoSidedAlbedoTable table;
  std::string error;
  ASSERT_TRUE(diffraction_two_sided_albedo_build_cpu(request, table, error)) << error;
  ASSERT_TRUE(diffraction_two_sided_albedo_validate(table, error)) << error;
  ASSERT_EQ(table.deficits.size(), table.reflectance.size());
  ASSERT_EQ(table.deficits.size(), table.transmittance.size());
  for (size_t i = 0; i < table.deficits.size(); i++) {
    EXPECT_GE(table.reflectance[i], 0.0f);
    EXPECT_GE(table.transmittance[i], 0.0f);
    EXPECT_NEAR(table.reflectance[i] + table.transmittance[i] + table.deficits[i],
                1.0f, 2.0e-6f);
  }
  /* Independent outgoing-direction quadrature of the native single-event
   * eval. This tests the tabulated escape estimator, beyond the algebraic
   * R+T+q construction identity, on both oriented IOR sides. */
  constexpr int mu_quadrature = 96, phi_quadrature = 192;
  const float mu = 4.0f / 9.0f;
  const float phi = M_2PI_F * 3.5f / 8.0f;
  const float3 incident = make_float3(std::sqrt(1.0f - mu * mu) * std::cos(phi),
                                      std::sqrt(1.0f - mu * mu) * std::sin(phi), mu);
  for (int side = 0; side < 2; side++) {
    DiffractionRoughDielectric param;
    ASSERT_TRUE(diffraction_dielectric_parameters(580.0f, request.pitch_nm,
                                                  request.depth_nm, request.duty,
                                                  side ? 1.5f : 1.0f,
                                                  side ? 1.0f : 1.5f,
                                                  request.alpha_x, request.alpha_y, &param));
    double integrated_reflection = 0.0, integrated_transmission = 0.0;
    for (int i = 0; i < mu_quadrature; i++) {
      const float outgoing_mu = (float(i) + 0.5f) / mu_quadrature;
      const float radius = std::sqrt(1.0f - outgoing_mu * outgoing_mu);
      for (int j = 0; j < phi_quadrature; j++) {
        const float outgoing_phi = M_2PI_F * (float(j) + 0.5f) / phi_quadrature;
        const float x = radius * std::cos(outgoing_phi);
        const float y = radius * std::sin(outgoing_phi);
        float pdf;
        integrated_reflection += diffraction_dielectric_eval<GGX>(
            &param, incident, make_float3(x, y, outgoing_mu), &pdf);
        integrated_transmission += diffraction_dielectric_eval<GGX>(
            &param, incident, make_float3(x, y, -outgoing_mu), &pdf);
      }
    }
    const double solid_angle = double(M_2PI_F) / (mu_quadrature * phi_quadrature);
    integrated_reflection *= solid_angle;
    integrated_transmission *= solid_angle;
    const size_t index = size_t((side * request.wavelength_count + 1) *
                                request.mu_count * request.phi_count +
                                2 * request.phi_count + 3);
    EXPECT_NEAR(table.reflectance[index], integrated_reflection, 0.02);
    EXPECT_NEAR(table.transmittance[index], integrated_transmission, 0.02);
    EXPECT_NEAR(table.deficits[index],
                1.0 - integrated_reflection - integrated_transmission, 0.02);
  }
  /* Both BSDF directions point away from the interface. On the opposite side
   * fixed tangent X remains, while the bitangent Y and normal Z reverse. */
  const float3 air_incident = normalize(make_float3(0.23f, 0.34f, 0.91f));
  const float3 glass_outgoing = normalize(make_float3(0.31f, -0.27f, -0.91f));
  const float3 glass_reverse = make_float3(glass_outgoing.x,
                                            -glass_outgoing.y,
                                            -glass_outgoing.z);
  const float q_air = diffraction_two_sided_albedo_lookup(table, 0, 580.0f, air_incident);
  const float q_glass = diffraction_two_sided_albedo_lookup(table, 1, 580.0f,
                                                             glass_reverse);
  EXPECT_GE(q_air, 0.0f);
  EXPECT_LE(q_air, 1.0f);
  EXPECT_GE(q_glass, 0.0f);
  EXPECT_LE(q_glass, 1.0f);
  DiffractionTwoSidedReturn lobe;
  ASSERT_TRUE(diffraction_two_sided_return(table.integrals[1], table.integrals[4],
                                           table.cross_fractions[1], &lobe));
  const float forward = diffraction_two_sided_eval(
      lobe, 0, 1, q_air, q_glass, 1.5f, glass_reverse.z);
  const float reverse = diffraction_two_sided_eval(
      lobe, 1, 0, q_glass, q_air, 1.0f, air_incident.z);
  EXPECT_NEAR(air_incident.z * forward,
              1.5f * 1.5f * glass_reverse.z * reverse,
              2.0e-6f);
  for (int side = 0; side < 2; side++) {
    const float sum = diffraction_two_sided_side_probability(lobe, side, 0) +
                      diffraction_two_sided_side_probability(lobe, side, 1);
    EXPECT_NEAR(sum, 1.0f, 2.0e-6f);
  }
  DiffractionManager manager;
  const int handle = manager.get_or_build_two_sided_albedo(request, nullptr, error);
  ASSERT_EQ(handle, 0) << error;
  EXPECT_EQ(manager.get_or_build_two_sided_albedo(request, nullptr, error), handle);
  EXPECT_EQ(manager.two_sided_albedo_cache_count(), 1);
  request.inside_ior = 0.8f;
  EXPECT_EQ(manager.get_or_build_two_sided_albedo(request, nullptr, error), -1);
  EXPECT_EQ(manager.two_sided_albedo_cache_count(), 1);
}

TEST(SceneDiffraction, TwoSidedGlassCoatedCacheAndSpectralCross)
{
  DiffractionTwoSidedAlbedoRequest request;
  request.alpha_x = 0.35f;
  request.alpha_y = 0.5f;
  request.pitch_nm = 1200.0f;
  request.depth_nm = 125.0f;
  request.duty = 0.43f;
  request.inside_ior = 1.5f;
  request.film_ior = 1.45f;
  request.film_thickness_nm = 250.0f;
  request.wavelength_count = 40;
  request.mu_count = 2;
  request.phi_count = 4;
  request.facet_samples = 64;
  std::string error;
  DiffractionTwoSidedAlbedoTable table;
  ASSERT_TRUE(diffraction_two_sided_albedo_build_cpu(request, table, error)) << error;
  ASSERT_TRUE(diffraction_two_sided_albedo_validate(table, error)) << error;

  /* Higher angular quadrature of physical coated Fresnel is independent of
   * the cache construction's 1,024-node cross-conductance quadrature. */
  for (const float wavelength_nm : {405.0f, 565.0f, 740.0f}) {
    constexpr int angular_samples = 2048;
    double direct = 0.0;
    for (int i = 0; i < angular_samples; i++) {
      const float mu = (float(i) + 0.5f) / angular_samples;
      direct += 2.0 * double(mu) *
                (1.0 - diffraction_thin_film_reflectance(
                           mu, 1.0f, request.inside_ior, request.film_ior,
                           request.film_thickness_nm / wavelength_nm));
    }
    direct /= angular_samples;
    const float position = (wavelength_nm - 380.0f) * (request.wavelength_count - 1) / 400.0f;
    const int a = int(position), b = a + 1;
    const float fraction = position - float(a);
    const float interpolated = (1.0f - fraction) * table.cross_fractions[a] +
                               fraction * table.cross_fractions[b];
    EXPECT_NEAR(interpolated, direct, 0.01) << wavelength_nm;
  }
  for (size_t i = 0; i < table.deficits.size(); i++) {
    EXPECT_GE(table.deficits[i], 0.0f);
    EXPECT_NEAR(table.reflectance[i] + table.transmittance[i] + table.deficits[i],
                1.0f, 2e-6f);
  }

  DiffractionManager manager;
  ASSERT_EQ(manager.get_or_build_two_sided_albedo(request, nullptr, error), 0) << error;
  auto changed = request;
  changed.film_thickness_nm = 260.0f;
  ASSERT_EQ(manager.get_or_build_two_sided_albedo(changed, nullptr, error), 1) << error;
  auto uncoated = request;
  uncoated.film_thickness_nm = 0.0f;
  uncoated.wavelength_count = 16;
  ASSERT_EQ(manager.get_or_build_two_sided_albedo(uncoated, nullptr, error), 2) << error;
  uncoated.film_ior = 1.8f;
  EXPECT_EQ(manager.get_or_build_two_sided_albedo(uncoated, nullptr, error), 2);
  EXPECT_EQ(manager.two_sided_albedo_cache_count(), 3);
  auto ordinary_thick_film = request;
  ordinary_thick_film.film_ior = 1.5f;
  ordinary_thick_film.film_thickness_nm = 2000.0f;
  ordinary_thick_film.wavelength_count = 136;
  ordinary_thick_film.facet_samples = 16;
  DiffractionTwoSidedAlbedoTable thick_table;
  ASSERT_TRUE(diffraction_two_sided_albedo_build_cpu(ordinary_thick_film, thick_table, error))
      << error;
  for (const float wavelength_nm : {390.0f, 555.0f, 725.0f}) {
    constexpr int angular_samples = 2048;
    double direct = 0.0;
    for (int i = 0; i < angular_samples; i++) {
      const float mu = (float(i) + 0.5f) / angular_samples;
      direct += 2.0 * double(mu) *
                (1.0 - diffraction_thin_film_reflectance(
                           mu, 1.0f, ordinary_thick_film.inside_ior,
                           ordinary_thick_film.film_ior,
                           ordinary_thick_film.film_thickness_nm / wavelength_nm));
    }
    direct /= angular_samples;
    const float position = (wavelength_nm - 380.0f) *
                           (ordinary_thick_film.wavelength_count - 1) / 400.0f;
    const int a = int(position), b = a + 1;
    const float fraction = position - float(a);
    const float interpolated = (1.0f - fraction) * thick_table.cross_fractions[a] +
                               fraction * thick_table.cross_fractions[b];
    EXPECT_NEAR(interpolated, direct, 0.01) << wavelength_nm;
  }
  changed.film_ior = 4.0f;
  changed.film_thickness_nm = 2000.0f; /* Requires more than 256 wavelength nodes. */
  EXPECT_EQ(manager.get_or_build_two_sided_albedo(changed, nullptr, error), -1);
  EXPECT_EQ(manager.two_sided_albedo_cache_count(), 3);
}
CCL_NAMESPACE_END
