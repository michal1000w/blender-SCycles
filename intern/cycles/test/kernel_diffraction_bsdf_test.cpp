/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <gtest/gtest.h>

#include "kernel/globals.h"
#include "kernel/closure/bsdf.h"
#include "kernel/closure/bsdf_diffraction.h"
#include "kernel/closure/bsdf_diffraction_thin_sheet.h"
#include "util/profiling.h"

#include "scene/shader.tables"

#include <random>
#include <vector>

CCL_NAMESPACE_BEGIN

static Spectrum thin_sheet_test_full_eval(KernelGlobals kg,
                                          ccl_private ShaderData *sd,
                                          const float3 wo)
{
  Spectrum result = zero_spectrum();
  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];
    float pdf;
    Spectrum value = zero_spectrum();
    if (sc->type == CLOSURE_BSDF_MICROFACET_GGX_ID) {
      value = bsdf_microfacet_ggx_eval(kg, sc, sd->wi, wo, &pdf);
    }
    else if (sc->type == CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID) {
      value = bsdf_thin_glass_transmission_eval(kg, sc, sd->wi, wo, &pdf);
    }
    else if (sc->type == CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_REFLECTION_ID ||
             sc->type == CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_TRANSMISSION_ID)
    {
      value = bsdf_diffraction_thin_sheet_eval(sc, sd->wi, wo, &pdf);
    }
    result += sc->weight * value;
  }
  return result;
}

TEST(KernelDiffractionBSDF, ThinSheetFullClosureConvergesToNativeThinGlass)
{
  std::vector<float> tables(table_ggx_E, table_ggx_E + 1024);
  tables.insert(tables.end(), table_ggx_Eavg, table_ggx_Eavg + 32);
  KernelGlobalsCPU global;
  global.lookup_table.data = tables.data();
  global.lookup_table.width = int(tables.size());
  global.data.tables.ggx_E = 0;
  global.data.tables.ggx_Eavg = 1024;
  Profiler profiler;
  ThreadKernelGlobalsCPU thread(global, nullptr, profiler, 0);
  const KernelGlobals kg = &thread;
  const float3 N = make_float3(0, 0, 1);
  const float3 tangent = make_float3(1, 0, 0);
  const float3 wi = normalize(make_float3(0.24f, -0.09f, 1.0f));
  const float3 reflected = normalize(make_float3(-0.19f, 0.12f, 1.0f));
  const float3 transmitted = normalize(make_float3(-0.17f, 0.05f, -1.0f));
  const FresnelCoeff tint = {make_spectrum(0.7f), make_spectrum(0.8f)};
  for (const float roughness : {0.2f, 0.6f, 1.0f}) {
    for (const float film_thickness : {0.0f, 180.0f}) {
      ShaderData native = {};
      native.wi = wi;
      native.N = native.Ng = N;
      native.num_closure_left = MAX_CLOSURE;
      native.rand_wavelength = 0.5f;
      native.shader_flag = SD_REQUIRES_WAVELENGTH;
      const FresnelThinFilm film = {film_thickness, 1.3f};
      bsdf_thin_glass_setup(kg, &native, true, true, tint, one_spectrum(), N,
                            sqr(roughness), 1.5f, film, PATH_RAY_VISIBILITY_CAMERA, 0);
      for (const float3 wo : {reflected, transmitted}) {
        const Spectrum expected = thin_sheet_test_full_eval(kg, &native, wo);
        for (const float depth : {0.0f, 0.01f, 0.001f}) {
          ShaderData grating = {};
          grating.wi = wi;
          grating.N = grating.Ng = N;
          grating.num_closure_left = MAX_CLOSURE;
          grating.rand_wavelength = 0.5f;
          grating.shader_flag = SD_REQUIRES_WAVELENGTH;
          bsdf_diffraction_thin_glass_setup(
              kg, &grating, true, true, tint, one_spectrum(), N, tangent,
              sqr(roughness), 1.5f, film, PATH_RAY_VISIBILITY_CAMERA, 0,
              1.0f, 450.0f, depth, 0.42f, 0.0f);
          const Spectrum actual = thin_sheet_test_full_eval(kg, &grating, wo);
          EXPECT_NEAR(average(actual), average(expected),
                      depth == 0.0f ? 1e-6f : 1e-3f * max(average(expected), 1e-4f));
        }
      }
    }
  }
}

TEST(KernelDiffractionBSDF, ThinSheetZeroReliefMatchesNativeCarriers)
{
  const float3 wi = normalize(make_float3(0.28f, -0.17f, 1.0f));
  const float3 h = normalize(make_float3(0.15f, 0.08f, 1.0f));
  for (const bool transmission : {false, true}) {
    const DiffractionThinSheetPort p = {0.18f, 0.42f, 0.0f, 0.41f, transmission};
    const float3 I = transmission ? diffraction_thin_sheet_flip_tangent(wi) : wi;
    const float3 O = 2.0f * dot(I, h) * h - I;
    const float3 wo = transmission ? -O : O;
    float pdf;
    const float value = diffraction_thin_sheet_eval(&p, wi, wo, &pdf);
    const float D = bsdf_aniso_D<GGX>(p.alpha, p.alpha, h);
    const float li = bsdf_aniso_lambda<GGX>(p.alpha, p.alpha, I);
    const float lo = bsdf_aniso_lambda<GGX>(p.alpha, p.alpha, O);
    EXPECT_NEAR(value, D / (4.0f * I.z * (1.0f + li + lo)), 2e-6f);
    EXPECT_NEAR(pdf, D / (4.0f * I.z * (1.0f + li)), 2e-6f);
    if (transmission) {
      const float3 straight = -wi;
      const DiffractionThinSheetPort smooth = {0, 0.42f, 0, 0.41f, true};
      float3 sampled;
      float mass, sample_pdf;
      ASSERT_TRUE(diffraction_thin_sheet_sample(
          &smooth, wi, make_float3(0.3f, 0.7f, 0.5f), &sampled, &mass, &sample_pdf));
      EXPECT_NEAR(len(sampled - straight), 0.0f, 1e-6f);
      EXPECT_FLOAT_EQ(mass, 1.0f);
      EXPECT_FLOAT_EQ(sample_pdf, 1.0f);
    }
  }
}

TEST(KernelDiffractionBSDF, ThinSheetPortsAreReciprocalAndSampleTheirOwnPdf)
{
  const float3 wi = normalize(make_float3(0.17f, -0.09f, 1.0f));
  const float3 h = normalize(make_float3(0.12f, 0.04f, 1.0f));
  for (const bool transmission : {false, true}) {
    const DiffractionThinSheetPort p = {0.2f, 0.34f, 2.3f, 0.43f, transmission};
    const float3 I = transmission ? diffraction_thin_sheet_flip_tangent(wi) : wi;
    for (int m = -2; m <= 2; m++) {
      float3 O;
      if (!diffraction_facet_reflect(I, h, make_float3(1, 0, 0),
                                     m * p.wavelength_over_pitch, &O) || O.z <= 0)
      {
        continue;
      }
      const float3 wo = transmission ? -O : O;
      float forward_pdf, reverse_pdf;
      const float forward = diffraction_thin_sheet_eval(&p, wi, wo, &forward_pdf);
      const float reverse = diffraction_thin_sheet_eval(
          &p, transmission ? -wo : wo, transmission ? -wi : wi, &reverse_pdf);
      EXPECT_NEAR(forward / fabsf(wo.z), reverse / wi.z, 2e-4f);
      EXPECT_GE(forward_pdf, 0.0f);
      EXPECT_GE(reverse_pdf, 0.0f);
    }
    for (float u : {0.07f, 0.3f, 0.7f, 0.97f}) {
      float3 wo;
      float value, pdf;
      if (!diffraction_thin_sheet_sample(&p, wi, make_float3(0.21f, 0.63f, u),
                                         &wo, &value, &pdf))
      {
        continue;
      }
      float evaluated_pdf;
      const float evaluated = diffraction_thin_sheet_eval(&p, wi, wo, &evaluated_pdf);
      EXPECT_NEAR(value, evaluated, 2e-5f);
      EXPECT_NEAR(pdf, evaluated_pdf, 2e-5f);
    }
  }
}

TEST(KernelDiffractionBSDF, ThinSheetSmallReliefConvergesToNativeAngularCarrier)
{
  const float3 wi = normalize(make_float3(0.24f, 0.11f, 1.0f));
  const float3 h = normalize(make_float3(0.14f, -0.08f, 1.0f));
  for (const float roughness : {0.2f, 0.6f, 1.0f}) {
    for (const bool transmission : {false, true}) {
      DiffractionThinSheetPort p = {sqr(roughness), 0.31f, 0.0f, 0.43f, transmission};
      if (transmission) {
        p.alpha = bsdf_thin_glass_transmission_roughness(p.alpha, 1.5f);
      }
      const float3 I = transmission ? diffraction_thin_sheet_flip_tangent(wi) : wi;
      const float3 O = 2.0f * dot(I, h) * h - I;
      const float3 wo = transmission ? -O : O;
      float flat_pdf;
      const float flat = diffraction_thin_sheet_eval(&p, wi, wo, &flat_pdf);
      ASSERT_GT(flat, 0.0f);
      for (const float relative_depth : {1e-2f, 1e-3f, 1e-4f}) {
        p.phase = M_2PI_F * relative_depth * (transmission ? 0.5f : 2.0f);
        float pdf;
        const float value = diffraction_thin_sheet_eval(&p, wi, wo, &pdf);
        const float tolerance = max(3e-5f, 8.0f * sqr(p.phase));
        EXPECT_NEAR(value / flat, 1.0f, tolerance);
        EXPECT_NEAR(pdf / flat_pdf, 1.0f, tolerance);
      }
    }
  }
}

TEST(KernelDiffractionBSDF, ThinSheetSmoothFacetPowerIsPassive)
{
  const float3 wi = normalize(make_float3(0.14f, 0.06f, 1.0f));
  for (const bool transmission : {false, true}) {
    const DiffractionThinSheetPort p = {0, 0.35f, 2.2f, 0.43f, transmission};
    const float3 I = transmission ? diffraction_thin_sheet_flip_tangent(wi) : wi;
    const float3 h = make_float3(0, 0, 1);
    float power = diffraction_thin_sheet_power(&p, 0, I, h);
    for (int m = -diffraction_thin_sheet_order_bound(&p);
         m <= diffraction_thin_sheet_order_bound(&p); m++)
    {
      float3 O;
      if (m != 0 && diffraction_facet_reflect(I, h, make_float3(1, 0, 0),
                                               m * p.wavelength_over_pitch, &O))
      {
        power += diffraction_thin_sheet_power(&p, m, I, h);
      }
    }
    EXPECT_NEAR(power, 1.0f, 2e-6f);
  }
}

TEST(KernelDiffractionBSDF, FacetOrderPowerIsPassiveAndReciprocal)
{
  std::mt19937 rng(826315);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  for (int trial = 0; trial < 10000; trial++) {
    const DiffractionReflection p = {
        0.2f, 0.3f, 0.15f + uniform(rng), 2.0f * uniform(rng), uniform(rng)};
    const float3 wi = normalize(
        make_float3(2 * uniform(rng) - 1, 2 * uniform(rng) - 1, 0.01f + uniform(rng)));
    const float3 h = make_float3(0, 0, 1), axis = make_float3(1, 0, 0);
    float total = 0.0f;
    const int limit = diffraction_reflection_max_order(&p);
    for (int m = -limit; m <= limit; m++) {
      float3 wo;
      if (m == 0 || !diffraction_facet_reflect(wi, h, axis, m * p.wavelength_over_pitch, &wo)) {
        continue;
      }
      const float power = diffraction_reflection_nonzero_power(&p, m, wi.z, wo.z);
      EXPECT_GE(power, 0.0f);
      EXPECT_FLOAT_EQ(power, diffraction_reflection_nonzero_power(&p, m, wo.z, wi.z));
      total += power;
    }
    EXPECT_LE(total, 1.0f + 1e-6f);
    EXPECT_NEAR(total + diffraction_reflection_zero_power(&p, wi, h), 1.0f, 2e-6f);
  }
}

TEST(KernelDiffractionBSDF, RoughReflectionReciprocity)
{
  std::mt19937 rng(342193);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  for (int trial = 0; trial < 20000; trial++) {
    const DiffractionReflection p = {0.05f + 0.7f * uniform(rng),
                                     0.05f + 0.7f * uniform(rng),
                                     0.15f + uniform(rng),
                                     1.5f * uniform(rng),
                                     uniform(rng)};
    const float3 wi = normalize(
        make_float3(2 * uniform(rng) - 1, 2 * uniform(rng) - 1, 0.05f + uniform(rng)));
    const float3 wo = normalize(
        make_float3(2 * uniform(rng) - 1, 2 * uniform(rng) - 1, 0.05f + uniform(rng)));
    float forward_pdf, reverse_pdf;
    const float forward = diffraction_reflection_eval(&p, wi, wo, &forward_pdf) / wo.z;
    const float reverse = diffraction_reflection_eval(&p, wo, wi, &reverse_pdf) / wi.z;
    EXPECT_NEAR(forward, reverse, 2e-5f * max(1.0f, max(forward, reverse)));
    EXPECT_GE(forward_pdf, 0.0f);
    EXPECT_GE(reverse_pdf, 0.0f);
    EXPECT_TRUE(isfinite_safe(forward));
  }
}

TEST(KernelDiffractionBSDF, FlatProfileReducesToGGX)
{
  const DiffractionReflection p = {0.2f, 0.4f, 0.35f, 0.0f, 0.5f};
  const float3 wi = normalize(make_float3(0.4f, 0.1f, 1.0f));
  const float3 wo = normalize(make_float3(-0.2f, 0.5f, 1.0f));
  const float3 h = normalize(wi + wo);
  const float D = bsdf_aniso_D<GGX>(p.alpha_x, p.alpha_y, h);
  const float li = bsdf_aniso_lambda<GGX>(p.alpha_x, p.alpha_y, wi);
  const float lo = bsdf_aniso_lambda<GGX>(p.alpha_x, p.alpha_y, wo);
  float pdf;
  const float value = diffraction_reflection_eval(&p, wi, wo, &pdf);
  EXPECT_NEAR(value, D / (4.0f * wi.z * (1.0f + li + lo)), 2e-6f);
  EXPECT_NEAR(pdf, D / (4.0f * wi.z * (1.0f + li)), 2e-6f);
}

TEST(KernelDiffractionBSDF, SmoothFacetSamplingConservesPower)
{
  const DiffractionReflection p = {0.0f, 0.0f, 0.34f, 0.28f, 0.5f};
  const float3 wi = normalize(make_float3(0.4f, 0.3f, 1.0f));
  for (int j = 0; j < 10000; j++) {
    float3 wo;
    float eval, pdf;
    ASSERT_TRUE(diffraction_reflection_sample(
        &p, wi, make_float3(0, 0, (j + 0.5f) / 10000), &wo, &eval, &pdf));
    EXPECT_GT(wo.z, 0.0f);
    EXPECT_FLOAT_EQ(eval, pdf);
    EXPECT_GT(pdf, 0.0f);
    EXPECT_LE(pdf, 1.0f);
    EXPECT_NEAR(len_squared(wo), 1.0f, 1e-6f);
  }
}

TEST(KernelDiffractionBSDF, SamplingMatchesIndependentSolidAngleIntegration)
{
  /* Integrate eval/pdf with uniform hemisphere samples, independently of the
   * normal/order sampler. Compare accepted probability, furnace energy, and
   * directional moments; null events remain in the sample denominator. */
  constexpr int count = 500000;
  for (const float cosine : {0.3f, 0.9f}) {
    const DiffractionReflection p = {0.3f, 0.5f, 0.37f, 0.24f, 0.42f};
    const float3 wi = make_float3(sqrtf(1.0f - cosine * cosine), 0.0f, cosine);
    std::mt19937 rng(625193);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    double sample_sum[4] = {}, sample_square[4] = {};
    double integral_sum[4] = {}, integral_square[4] = {};
    for (int j = 0; j < count; j++) {
      float3 wo;
      float value, pdf;
      const float3 rand = make_float3(uniform(rng), uniform(rng), uniform(rng));
      if (diffraction_reflection_sample(&p, wi, rand, &wo, &value, &pdf)) {
        const double values[4] = {1.0, value / pdf, wo.x, wo.z};
        EXPECT_LE(value, pdf * 1.000001f);
        for (int k = 0; k < 4; k++) {
          sample_sum[k] += values[k];
          sample_square[k] += values[k] * values[k];
        }
      }
      const float z = uniform(rng), phi = M_2PI_F * uniform(rng);
      const float r = sqrtf(1.0f - z * z);
      const float3 direction = make_float3(r * cosf(phi), r * sinf(phi), z);
      const float eval = diffraction_reflection_eval(&p, wi, direction, &pdf);
      const double values[4] = {(2.0 * M_PI) * pdf,
                                (2.0 * M_PI) * eval,
                                (2.0 * M_PI) * pdf * direction.x,
                                (2.0 * M_PI) * pdf * direction.z};
      for (int k = 0; k < 4; k++) {
        integral_sum[k] += values[k];
        integral_square[k] += values[k] * values[k];
      }
    }
    for (int k = 0; k < 4; k++) {
      const double sampled = sample_sum[k] / count;
      const double integrated = integral_sum[k] / count;
      const double variance = (sample_square[k] / count - sampled * sampled +
                               integral_square[k] / count - integrated * integrated) /
                              count;
      const double error = sqrt(max(0.0, variance));
      EXPECT_NEAR(sampled, integrated, 6.0 * error + 1e-4)
          << "cos_i=" << cosine << " moment=" << k;
      if (k < 2) {
        EXPECT_LE(integrated, 1.0 + 6.0 * error);
        EXPECT_GT(integrated, 0.0);
      }
    }
  }
}

CCL_NAMESPACE_END
