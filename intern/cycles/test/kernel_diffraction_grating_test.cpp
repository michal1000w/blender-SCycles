/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Grating model of the Diffraction BSDF (kernel/closure/bsdf_diffraction_grating.h), checked
 * against the closure of the Al-Toto/blender `diffraction-bsdf` branch it was ported from
 * (diffraction_grating_branch_fixture.h) and against its own sampling. */

#include <gtest/gtest.h>

#include "kernel/globals.h"
#include "kernel/closure/bsdf.h"
#include "kernel/closure/bsdf_diffraction_grating.h"

#include "diffraction_grating_branch_fixture.h"

#include <cmath>
#include <memory>
#include <random>
#include <vector>

CCL_NAMESPACE_BEGIN

namespace {

struct GratingConfig {
  float roughness = 0.3f;
  float2 pitch = make_float2(1.6f, 1.6f);
  float2 depth = make_float2(110.0f, 110.0f);
  float2 duty = make_float2(0.45f, 0.45f);
  float2 rotation = make_float2(0.0f, 0.25f);
  bool rectangular = true;
  bool is_2d = false;
  float rand_wavelength = 0.5f;
  float3 N = make_float3(0.0f, 0.0f, 1.0f);
  float3 T = make_float3(1.0f, 0.0f, 0.0f);
};

/* Closure of this fork. */
struct PortedClosure {
  ShaderClosure storage = {};
  std::unique_ptr<ShaderData> sd = std::make_unique<ShaderData>();

  explicit PortedClosure(const GratingConfig &c)
  {
    sd->rand_wavelength = c.rand_wavelength;
    sd->runtime_flag = 0;
    DiffractionGratingBsdf *bsdf = reinterpret_cast<DiffractionGratingBsdf *>(&storage);
    bsdf->weight = one_spectrum();
    bsdf->N = c.N;
    bsdf_diffraction_grating_setup(sd.get(),
                                   bsdf,
                                   c.T,
                                   c.roughness,
                                   c.pitch,
                                   c.depth,
                                   c.duty,
                                   c.rotation,
                                   c.rectangular,
                                   c.is_2d);
  }

  const ShaderClosure *sc() const
  {
    return &storage;
  }
};

/* Closure of the branch, filled like its setup function does. */
struct BranchClosure {
  ShaderClosure storage = {};
  altoto::DiffractionExtraY extra = {};

  explicit BranchClosure(const GratingConfig &c)
  {
    altoto::DiffractionBsdf *bsdf = reinterpret_cast<altoto::DiffractionBsdf *>(&storage);
    bsdf->weight = one_spectrum();
    bsdf->N = c.N;
    bsdf->T = c.T;
    bsdf->roughness = sqr(saturatef(c.roughness));
    bsdf->pitch_um = fmaxf(c.pitch.x, 1e-4f);
    bsdf->groove_depth_nm = fmaxf(c.depth.x, 0.0f);
    bsdf->duty_cycle = saturatef(c.duty.x);
    bsdf->wavelength_um = sample_wavelength(c.rand_wavelength);
    bsdf->profile = c.rectangular ? altoto::DIFFRACTION_PROFILE_RECTANGULAR :
                                    altoto::DIFFRACTION_PROFILE_SINUSOIDAL;
    extra.pitch_um = fmaxf(c.pitch.y, 1e-4f);
    extra.groove_depth_nm = fmaxf(c.depth.y, 0.0f);
    extra.duty_cycle = saturatef(c.duty.y);
    extra.grating_type = c.is_2d ? altoto::DIFFRACTION_GRATING_2D : altoto::DIFFRACTION_GRATING_1D;
    extra.rotation_x = c.rotation.x;
    extra.rotation_y = c.rotation.y;
    bsdf->extra_y = &extra;
  }

  const ShaderClosure *sc() const
  {
    return &storage;
  }
};

float3 uniform_hemisphere(std::mt19937 &rng, float *pdf)
{
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  const float z = u(rng);
  const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
  const float phi = M_2PI_F * u(rng);
  *pdf = 1.0f / M_2PI_F;
  return make_float3(r * cosf(phi), r * sinf(phi), z);
}

float3 random_direction(std::mt19937 &rng, const float min_z = 0.02f)
{
  float pdf;
  float3 d;
  do {
    d = uniform_hemisphere(rng, &pdf);
  } while (d.z < min_z);
  return d;
}

std::vector<GratingConfig> test_configs()
{
  std::vector<GratingConfig> configs;
  for (const bool rectangular : {true, false}) {
    for (const bool is_2d : {false, true}) {
      for (const float roughness : {0.15f, 0.4f, 0.8f}) {
        GratingConfig c;
        c.rectangular = rectangular;
        c.is_2d = is_2d;
        c.roughness = roughness;
        c.pitch = is_2d ? make_float2(1.2f, 2.5f) : make_float2(1.6f, 1.6f);
        c.depth = rectangular ? make_float2(110.0f, 160.0f) : make_float2(180.0f, 90.0f);
        c.rotation = is_2d ? make_float2(0.07f, 0.29f) : make_float2(0.11f, 0.0f);
        c.rand_wavelength = 0.2f + 0.3f * float(configs.size() % 3);
        c.T = normalize(make_float3(1.0f, 0.3f, 0.2f));
        configs.push_back(c);
      }
    }
  }
  return configs;
}

/* Bessel function of the first kind, from its power series in double precision. */
double bessel_j(const int m, const double x)
{
  double term = 1.0;
  for (int k = 1; k <= m; k++) {
    term *= (x / 2.0) / k;
  }
  double sum = term;
  for (int k = 1; k < 200; k++) {
    term *= -(x * x / 4.0) / (double(k) * double(k + m));
    sum += term;
    if (std::abs(term) < 1e-17 * std::abs(sum)) {
      break;
    }
  }
  return sum;
}

}  // namespace

/* The value of every order is the branch's: same efficiencies, lattice and lobes. */
TEST(KernelDiffractionGrating, EvalValueMatchesBranch)
{
  std::mt19937 rng(1);
  int compared = 0;
  for (const GratingConfig &c : test_configs()) {
    const PortedClosure ported(c);
    const BranchClosure branch(c);
    for (int i = 0; i < 400; i++) {
      const float3 wi = random_direction(rng);
      const float3 wo = random_direction(rng);
      float pdf_ported, pdf_branch;
      const float value_ported = average(
          bsdf_diffraction_grating_eval(ported.sc(), wi, wo, &pdf_ported));
      const float value_branch = average(
          altoto::bsdf_diffraction_eval(branch.sc(), wi, wo, &pdf_branch));
      EXPECT_NEAR(value_ported, value_branch, 2e-4f * fabsf(value_branch) + 1e-6f)
          << "rectangular " << c.rectangular << " 2d " << c.is_2d << " roughness "
          << c.roughness;
      /* Same pdf: the branch only leaves an order out of the normalization when its half
       * vector is below the surface, which two upper hemisphere directions never produce. */
      EXPECT_NEAR(pdf_ported, pdf_branch, 2e-4f * fabsf(pdf_branch) + 1e-6f);
      compared++;
    }
  }
  EXPECT_EQ(compared, 12 * 400);
}

/* Efficiencies of the sinusoidal profile are J_m(A)^2, those of the rectangular profile sum to
 * one over all orders. */
TEST(KernelDiffractionGrating, Efficiencies)
{
  float eta[DIFFRACTION_GRATING_MAX_ORDERS];
  for (const float A : {0.2f, 0.7f, 1.5f, 3.0f, 6.0f, 12.0f}) {
    diffraction_grating_sinusoidal_efficiencies(A, DIFFRACTION_GRATING_MAX_ORDER, eta);
    for (int m = -DIFFRACTION_GRATING_MAX_ORDER; m <= DIFFRACTION_GRATING_MAX_ORDER; m++) {
      const double expected = std::pow(bessel_j(std::abs(m), A), 2.0);
      EXPECT_NEAR(eta[m + DIFFRACTION_GRATING_MAX_ORDER], expected, 2e-5) << "A " << A << " m "
                                                                         << m;
    }
  }
  /* Tiny arguments seed the recurrence with huge values: rescaling keeps them finite. */
  for (const float A : {0.15f, 0.16f, 0.2f}) {
    diffraction_grating_sinusoidal_efficiencies(A, DIFFRACTION_GRATING_MAX_ORDER, eta);
    for (int i = 0; i < DIFFRACTION_GRATING_MAX_ORDERS; i++) {
      EXPECT_TRUE(std::isfinite(eta[i]));
    }
  }
  for (const float phi : {0.3f, 1.0f, 2.5f, 3.14159f}) {
    for (const float g : {0.2f, 0.45f, 0.5f, 0.8f}) {
      diffraction_grating_rectangular_efficiencies(phi, g, DIFFRACTION_GRATING_MAX_ORDER, eta);
      double sum = 0.0;
      for (int i = 0; i < DIFFRACTION_GRATING_MAX_ORDERS; i++) {
        sum += eta[i];
      }
      /* Orders above 8 carry at most 4 g(1-g) * 2 / (pi^2 * 8). */
      EXPECT_LE(sum, 1.0 + 1e-5);
      EXPECT_GE(sum, 1.0 - 8.0 / (M_PI * M_PI * 8.0) - 1e-5);
      /* Compare with the branch's per-order formula. */
      for (int m = -DIFFRACTION_GRATING_MAX_ORDER; m <= DIFFRACTION_GRATING_MAX_ORDER; m++) {
        const float wavelength_nm = 500.0f;
        const float depth_nm = phi * wavelength_nm / (2.0f * M_2PI_F);
        EXPECT_NEAR(eta[m + DIFFRACTION_GRATING_MAX_ORDER],
                    altoto::diffraction_efficiency_rectangular(m, wavelength_nm, 1.0f, depth_nm, g),
                    1e-5f);
      }
    }
  }
}

/* A 2D grating whose second axis has no depth is the 1D grating of its first axis. */
TEST(KernelDiffractionGrating, FlatSecondAxisIs1D)
{
  std::mt19937 rng(2);
  for (const bool rectangular : {true, false}) {
    GratingConfig c1;
    c1.rectangular = rectangular;
    c1.roughness = 0.35f;
    c1.rotation = make_float2(0.13f, 0.0f);
    GratingConfig c2 = c1;
    c2.is_2d = true;
    c2.depth.y = 0.0f;
    c2.pitch.y = 3.0f;
    c2.rotation.y = c1.rotation.x + 0.25f;
    const PortedClosure g1(c1), g2(c2);
    for (int i = 0; i < 500; i++) {
      const float3 wi = random_direction(rng), wo = random_direction(rng);
      float pdf1, pdf2;
      const float v1 = average(bsdf_diffraction_grating_eval(g1.sc(), wi, wo, &pdf1));
      const float v2 = average(bsdf_diffraction_grating_eval(g2.sc(), wi, wo, &pdf2));
      EXPECT_NEAR(v1, v2, 1e-4f * v1 + 1e-6f);
      EXPECT_NEAR(pdf1, pdf2, 1e-4f * pdf1 + 1e-6f);
    }
  }
}

/* The pdf returned by eval is the density of the directions sample() produces: compare a
 * histogram of samples with the pdf integrated over the same bins, and check that sampled
 * eval/pdf estimates the same albedo as integrating eval. */
static void check_sampling(const GratingConfig &c,
                           const float3 wi,
                           const bool expect_match,
                           double *albedo_sampled,
                           double *albedo_integrated)
{
  const PortedClosure ported(c);
  const BranchClosure branch(c);
  const ShaderClosure *sc = ported.sc();

  constexpr int bins_theta = 8, bins_phi = 16;
  std::vector<double> histogram(bins_theta * bins_phi, 0.0);
  std::vector<double> pdf_integral(bins_theta * bins_phi, 0.0);
  auto bin_of = [&](const float3 w) {
    const int t = min(int(sqrtf(fmaxf(0.0f, 1.0f - w.z)) * bins_theta), bins_theta - 1);
    float phi = atan2f(w.y, w.x);
    if (phi < 0.0f) {
      phi += M_2PI_F;
    }
    const int p = min(int(phi / M_2PI_F * bins_phi), bins_phi - 1);
    return t * bins_phi + p;
  };

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  const int num_samples = 400000;
  double sampled = 0.0;
  int valid = 0;
  for (int i = 0; i < num_samples; i++) {
    Spectrum eval;
    float3 wo;
    float pdf;
    float2 roughness;
    float eta;
    const float3 rand = make_float3(u(rng), u(rng), u(rng));
    const int label = expect_match ?
                          bsdf_diffraction_grating_sample(
                              sc, c.N, wi, rand, &eval, &wo, &pdf, &roughness, &eta) :
                          altoto::bsdf_diffraction_sample(
                              branch.sc(), c.N, wi, rand, &eval, &wo, &pdf, &roughness, &eta);
    if (label == LABEL_NONE) {
      continue;
    }
    ASSERT_GT(pdf, 0.0f);
    sampled += average(eval) / pdf;
    histogram[bin_of(wo)] += 1.0;
    valid++;
  }
  *albedo_sampled = sampled / num_samples;

  /* Integrate eval and pdf with stratified uniform directions. */
  const int strata = 600;
  double integrated = 0.0;
  for (int i = 0; i < strata; i++) {
    for (int j = 0; j < strata; j++) {
      const float z = (i + u(rng)) / strata;
      const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
      const float phi = M_2PI_F * (j + u(rng)) / strata;
      const float3 wo = make_float3(r * cosf(phi), r * sinf(phi), z);
      float pdf;
      const float value = average(bsdf_diffraction_grating_eval(sc, wi, wo, &pdf));
      const double weight = M_2PI_F / (double(strata) * strata);
      integrated += value * weight;
      pdf_integral[bin_of(wo)] += pdf * weight;
    }
  }
  *albedo_integrated = integrated;

  if (!expect_match) {
    return;
  }
  double total_pdf = 0.0;
  for (int b = 0; b < bins_theta * bins_phi; b++) {
    total_pdf += pdf_integral[b];
    const double expected = pdf_integral[b] * num_samples;
    /* Poisson noise plus a little integration error of sharp lobes. */
    const double tolerance = 5.0 * sqrt(expected + 1.0) + 0.01 * expected + 5.0;
    EXPECT_NEAR(histogram[b], expected, tolerance) << "bin " << b;
  }
  EXPECT_NEAR(total_pdf, double(valid) / num_samples, 0.01);
  EXPECT_NEAR(*albedo_sampled, *albedo_integrated, 0.01 + 0.01 * *albedo_integrated);
}

TEST(KernelDiffractionGrating, SamplingMatchesPdf)
{
  std::mt19937 rng(3);
  for (GratingConfig c : test_configs()) {
    if (c.roughness < 0.3f) {
      continue; /* Too sharp for the stratified integration. */
    }
    for (const float3 wi : {normalize(make_float3(0.3f, -0.2f, 1.0f)),
                            normalize(make_float3(-0.8f, 0.5f, 0.35f))})
    {
      double sampled, integrated;
      check_sampling(c, wi, true, &sampled, &integrated);
      /* A passive surface. */
      EXPECT_LE(integrated, 1.0 + 1e-3);
    }
  }
}

/* Sampling of the branch and of the port estimate the same albedo, also at grazing incidence
 * where many orders are evanescent. */
TEST(KernelDiffractionGrating, SamplingMatchesBranch)
{
  GratingConfig c;
  c.roughness = 0.6f;
  c.pitch = make_float2(0.9f, 0.9f);
  c.depth = make_float2(150.0f, 150.0f);
  const float3 wi = normalize(make_float3(-0.9f, 0.1f, 0.25f));
  double sampled_port, integrated_port, sampled_branch, integrated_branch;
  check_sampling(c, wi, true, &sampled_port, &integrated_port);
  check_sampling(c, wi, false, &sampled_branch, &integrated_branch);
  EXPECT_NEAR(sampled_port, integrated_port, 0.01 + 0.01 * integrated_port);
  EXPECT_NEAR(sampled_branch, sampled_port, 0.01 + 0.01 * integrated_port);
}

/* Energy: no configuration reflects more than it receives, at any incidence. */
TEST(KernelDiffractionGrating, Passive)
{
  std::mt19937 rng(4);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for (GratingConfig c : test_configs()) {
    c.roughness = fmaxf(c.roughness, 0.4f);
    for (const float cos_i : {0.05f, 0.3f, 0.7f, 1.0f}) {
      const float3 wi = make_float3(sqrtf(1.0f - cos_i * cos_i), 0.0f, cos_i);
      const PortedClosure ported(c);
      double albedo = 0.0;
      const int strata = 300;
      for (int i = 0; i < strata; i++) {
        for (int j = 0; j < strata; j++) {
          const float z = (i + u(rng)) / strata;
          const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
          const float phi = M_2PI_F * (j + u(rng)) / strata;
          float pdf;
          albedo += average(bsdf_diffraction_grating_eval(
                        ported.sc(), wi, make_float3(r * cosf(phi), r * sinf(phi), z), &pdf)) *
                    M_2PI_F / (double(strata) * strata);
        }
      }
      EXPECT_LE(albedo, 1.02) << "cos_i " << cos_i;
      EXPECT_GE(albedo, 0.0);
    }
  }
}

/* Setup: the closure fits, uses the hero wavelength of the path and asks for spectral paths. */
TEST(KernelDiffractionGrating, Setup)
{
  GratingConfig c;
  c.rand_wavelength = 0.73f;
  PortedClosure ported(c);
  const DiffractionGratingBsdf *bsdf = reinterpret_cast<const DiffractionGratingBsdf *>(
      ported.sc());
  EXPECT_EQ(bsdf->type, CLOSURE_BSDF_DIFFRACTION_GRATING_ID);
  EXPECT_TRUE(ported.sd->runtime_flag & SR_BSDF_HAS_DISPERSION);
  const float wavelength = sample_wavelength(c.rand_wavelength);
  EXPECT_NEAR(len(bsdf->K1), wavelength / c.pitch.x, 1e-5f);
  EXPECT_EQ(diffraction_grating_order_x(bsdf), int(ceilf(c.pitch.x / wavelength)) + 1);
  EXPECT_FALSE(diffraction_grating_is_2d(bsdf));
  EXPECT_TRUE(diffraction_grating_is_rectangular(bsdf));
  EXPECT_NEAR(bsdf->alpha, sqr(c.roughness), 1e-6f);
  /* Bad inputs do not produce NaN. */
  GratingConfig bad;
  bad.pitch = make_float2(0.0f, -1.0f);
  bad.depth = make_float2(-5.0f, 1e6f);
  bad.duty = make_float2(-1.0f, 2.0f);
  bad.rotation = make_float2(0.25f, 0.25f); /* Parallel axes. */
  bad.is_2d = true;
  bad.T = make_float3(0.0f, 0.0f, 1.0f); /* Along the normal. */
  PortedClosure broken(bad);
  float pdf;
  const Spectrum value = bsdf_diffraction_grating_eval(
      broken.sc(), normalize(make_float3(0.2f, 0.1f, 1.0f)), normalize(make_float3(-0.3f, 0.0f, 1.0f)), &pdf);
  EXPECT_TRUE(isfinite_safe(value));
  EXPECT_TRUE(std::isfinite(pdf));
}

CCL_NAMESPACE_END
