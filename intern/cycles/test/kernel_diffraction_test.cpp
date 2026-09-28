/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <gtest/gtest.h>

#include "kernel/closure/bsdf_diffraction_util.h"

#include <complex>
#include <random>

CCL_NAMESPACE_BEGIN

TEST(KernelDiffraction, FacetSamplingInversionAndReciprocalJacobians)
{
  std::mt19937 rng(852613);
  std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
  const float3 axis = normalize(make_float3(1.0f, 0.2f, 0.0f));
  int accepted = 0, double_roots = 0;
  for (int j = 0; j < 50000; j++) {
    const float3 h = normalize(
        make_float3(uniform(rng), uniform(rng), 0.1f + fabsf(uniform(rng))));
    const float3 wi = normalize(h + 0.9f * make_float3(uniform(rng), uniform(rng), uniform(rng)));
    const float delta = 1.8f * uniform(rng);
    float3 wo;
    if (!diffraction_facet_reflect(wi, h, axis, delta, &wo)) {
      continue;
    }
    accepted++;
    EXPECT_NEAR(len_squared(wo), 1.0f, 2e-6f);
    DiffractionReflectionRoot roots[2];
    const int count = diffraction_reflection_half_vectors(wi, wo, axis, delta, roots);
    ASSERT_GT(count, 0);
    double_roots += count == 2;
    float distance = 1.0f;
    for (int k = 0; k < count; k++) {
      const float3 root = roots[k].h;
      distance = min(distance, len(root - h));
      /* Verify the conservation equation directly. Reconstructing cos_o with
       * sqrt(1-|tangent|^2) is ill-conditioned at grazing; testing the vector
       * equation avoids hiding such conditioning behind a loose tolerance. */
      const float3 momentum = (dot(wi, root) + dot(wo, root)) * root + delta * roots[k].e;
      EXPECT_LT(len(wi + wo - momentum), 3e-6f);
      EXPECT_NEAR(dot(root, roots[k].e), 0.0f, 3e-7f);
      const float forward = diffraction_reflection_jacobian(wi, wo, root, axis, delta);
      const float reverse = diffraction_reflection_jacobian(wo, wi, root, axis, delta);
      const float a = dot(wi, root) * forward;
      const float b = dot(wo, root) * reverse;
      EXPECT_NEAR(a, b, 2e-6f * max(a, b));
    }
    /* The inverse azimuth is ill-conditioned when wi+wo approaches the axis:
     * perturbing v by one ulp rotates its transverse component by O(eps/|v_t|).
     * Keep a conditioning-aware normal bound alongside the strict 3e-6 momentum
     * residual above. An unconditional normal bound is not valid at this ring. */
    const float transverse = len(cross(axis, wi + wo));
    EXPECT_LT(distance, max(2e-5f, 8.0f * FLT_EPSILON / transverse));
  }
  EXPECT_GT(accepted, 10000);
  EXPECT_GT(double_roots, 100);
}

TEST(KernelDiffraction, FacetJacobianMatchesFiniteDifferences)
{
  std::mt19937 rng(317925);
  std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
  const float3 axis = make_float3(1.0f, 0.0f, 0.0f);
  constexpr float step = 0.0005f;
  int checked = 0;
  for (int j = 0; j < 10000; j++) {
    const float3 h = normalize(make_float3(uniform(rng), uniform(rng), 1.0f));
    const float3 wi = normalize(make_float3(uniform(rng), uniform(rng), 1.0f));
    const float delta = uniform(rng);
    float3 wo;
    if (!diffraction_facet_reflect(wi, h, axis, delta, &wo) || dot(wi, h) < 0.15f ||
        dot(wo, h) < 0.15f)
    {
      continue;
    }
    const float analytic = diffraction_reflection_jacobian(wi, wo, h, axis, delta);
    if (!(analytic > 0.01f && analytic < 20.0f)) {
      continue;
    }
    const float3 u = normalize(cross(h, axis));
    const float3 v = cross(h, u);
    float3 u0, u1, v0, v1;
    ASSERT_TRUE(diffraction_facet_reflect(wi, normalize(h - step * u), axis, delta, &u0));
    ASSERT_TRUE(diffraction_facet_reflect(wi, normalize(h + step * u), axis, delta, &u1));
    ASSERT_TRUE(diffraction_facet_reflect(wi, normalize(h - step * v), axis, delta, &v0));
    ASSERT_TRUE(diffraction_facet_reflect(wi, normalize(h + step * v), axis, delta, &v1));
    const float measured = 4.0f * step * step / len(cross(u1 - u0, v1 - v0));
    EXPECT_NEAR(measured, analytic, 0.01f * analytic);
    checked++;
  }
  EXPECT_GT(checked, 1000);
}

TEST(KernelDiffraction, ZeroOrderFacetMatchesMirror)
{
  const float3 h = normalize(make_float3(0.1f, 0.2f, 1.0f));
  const float3 wi = normalize(make_float3(0.5f, -0.2f, 1.0f));
  const float3 axis = make_float3(1.0f, 0.0f, 0.0f);
  float3 wo;
  DiffractionReflectionRoot roots[2];
  ASSERT_TRUE(diffraction_facet_reflect(wi, h, axis, 0.0f, &wo));
  EXPECT_LT(len(wo - (2.0f * dot(wi, h) * h - wi)), 3e-7f);
  ASSERT_EQ(diffraction_reflection_half_vectors(wi, wo, axis, 0.0f, roots), 1);
  EXPECT_LT(len(roots[0].h - h), 2e-7f);
  EXPECT_NEAR(
      diffraction_reflection_jacobian(wi, wo, h, axis, 0.0f), 1.0f / (4.0f * dot(wi, h)), 2e-7f);
}

TEST(KernelDiffraction, GratingEquationAndEvanescentOrders)
{
  const float3 wi = make_float3(0.3f, 0.4f, sqrtf(0.75f));
  for (float eta : {0.7f, 1.0f, 1.5f}) {
    for (bool transmission : {false, true}) {
      for (int m = -8; m <= 8; m++) {
        float3 wo;
        const bool valid = diffraction_order_direction(wi, 0.35f, m, eta, transmission, &wo);
        const double x = (double(m) * 0.35 - 0.3) / eta;
        const double y = -0.4 / eta;
        EXPECT_EQ(valid, x * x + y * y < 1.0);
        if (valid) {
          EXPECT_NEAR(len_squared(wo), 1.0f, 3e-7f);
          EXPECT_NEAR(wi.x + eta * wo.x, m * 0.35f, 2e-7f);
          EXPECT_NEAR(wi.y + eta * wo.y, 0.0f, 1e-7f);
          EXPECT_EQ(wo.z < 0.0f, transmission);
        }
        else {
          EXPECT_EQ(len_squared(wo), 0.0f);
        }
      }
    }
  }
}

TEST(KernelDiffraction, ReflectionReversesForEveryPropagatingOrder)
{
  const float3 wi = normalize(make_float3(0.5f, 0.2f, 1.0f));
  for (int m = -5; m <= 5; m++) {
    float3 wo, reverse;
    if (diffraction_order_direction(wi, 0.25f, m, 1.0f, false, &wo)) {
      ASSERT_TRUE(diffraction_order_direction(wo, 0.25f, m, 1.0f, false, &reverse));
      EXPECT_LT(len(reverse - wi), 2e-7f);
    }
  }
}

TEST(KernelDiffraction, BinaryCoefficientsMatchIndependentFourierIntegration)
{
  /* Integrate each constant segment separately: this avoids a sampled profile
   * accidentally changing the duty cycle. Midpoint integration is independent
   * of the analytic sine formula in the kernel. */
  constexpr int count = 4096;
  for (float duty : {0.1f, 0.3f, 0.5f, 0.85f}) {
    for (float phase : {0.0f, 0.7f, 3.14159265f, 5.0f}) {
      for (int m = -8; m <= 8; m++) {
        std::complex<double> amplitude(0.0, 0.0);
        for (int j = 0; j < count; j++) {
          const double u = (j + 0.5) / count;
          const double x0 = duty * u;
          const double x1 = duty + (1.0 - duty) * u;
          amplitude += double(duty) * std::polar(1.0, phase - 2.0 * M_PI * m * x0) +
                       (1.0 - duty) * std::polar(1.0, -2.0 * M_PI * m * x1);
        }
        EXPECT_NEAR(
            diffraction_binary_power(m, phase, duty), std::norm(amplitude / double(count)), 3e-7);
        const float2 coefficient = diffraction_binary_amplitude(m, phase, duty, 0.0f);
        EXPECT_NEAR(coefficient.x, amplitude.real() / count, 6e-7);
        EXPECT_NEAR(coefficient.y, amplitude.imag() / count, 6e-7);
      }
    }
  }
}

TEST(KernelDiffraction, ZeroOrderGrazingReflection)
{
  const float3 h = make_float3(0.0f, 0.0f, 1.0f);
  const float3 axis = make_float3(1.0f, 0.0f, 0.0f);
  for (float cosine : {1e-2f, 1e-3f, 1e-4f, 1e-5f}) {
    const float3 wi = normalize(make_float3(1.0f, 0.0f, cosine));
    float3 wo;
    ASSERT_TRUE(diffraction_facet_reflect(wi, h, axis, 0.0f, &wo));
    EXPECT_FLOAT_EQ(wo.z, wi.z);
    EXPECT_FLOAT_EQ(wo.x, -wi.x);
    EXPECT_NEAR(diffraction_reflection_jacobian(wi, wo, h, axis, 0.0f) * 4.0f * wi.z, 1.0f, 2e-7f);
  }
}

TEST(KernelDiffraction, TransmissionReversalChangesMediumAndFrame)
{
  const float3 wi = normalize(make_float3(0.4f, 0.3f, 1.0f));
  for (float eta : {0.65f, 1.0f, 1.5f, 2.4f}) {
    for (int m = -10; m <= 10; m++) {
      float3 wo, reverse;
      if (!diffraction_order_direction(wi, 0.2f, m, eta, true, &wo)) {
        continue;
      }
      /* Reverse interface normal while retaining the physical grating axis X.
       * The frame's Y axis flips to retain handedness. */
      const float3 reverse_wi = make_float3(wo.x, -wo.y, -wo.z);
      ASSERT_TRUE(
          diffraction_order_direction(reverse_wi, 0.2f / eta, m, 1.0f / eta, true, &reverse));
      const float3 world_reverse = make_float3(reverse.x, -reverse.y, -reverse.z);
      EXPECT_LT(len(world_reverse - wi), 5e-7f);
    }
  }
}

TEST(KernelDiffraction, TranslationPreservesPowerButChangesInterference)
{
  for (int m = -8; m <= 8; m++) {
    const float2 a = diffraction_binary_amplitude(m, 1.3f, 0.35f, 0.0f);
    const float2 b = diffraction_binary_amplitude(m, 1.3f, 0.35f, 0.23f);
    EXPECT_NEAR(dot(a, a), diffraction_binary_power(m, 1.3f, 0.35f), 2e-7f);
    EXPECT_NEAR(dot(a, a), dot(b, b), 2e-7f);
    const std::complex<double> expected = std::complex<double>(a.x, a.y) *
                                          std::polar(1.0, -2.0 * M_PI * m * 0.23);
    EXPECT_NEAR(b.x, expected.real(), 2e-7f);
    EXPECT_NEAR(b.y, expected.imag(), 2e-7f);
  }
  const float2 a = diffraction_binary_amplitude(1, M_PI_F, 0.5f, 0.0f);
  const float2 b = diffraction_binary_amplitude(1, M_PI_F, 0.5f, 0.5f);
  EXPECT_LT(dot(a + b, a + b), 1e-12f);
  EXPECT_GT(dot(a, a) + dot(b, b), 0.8f);
}

TEST(KernelDiffraction, BinaryParsevalAndSymmetry)
{
  for (float duty : {0.0f, 0.1f, 0.5f, 0.9f, 1.0f}) {
    for (float phase : {0.0f, 0.1f, 1.0f, 3.14159265f, 5.0f}) {
      double power = diffraction_binary_power(0, phase, duty);
      for (int m = 1; m <= 10000; m++) {
        const float p = diffraction_binary_power(m, phase, duty);
        EXPECT_GE(p, 0.0f);
        EXPECT_FLOAT_EQ(p, diffraction_binary_power(-m, phase, duty));
        power += 2.0 * p;
      }
      EXPECT_NEAR(power, 1.0, 5e-5);
    }
  }
  EXPECT_NEAR(diffraction_binary_power(0, M_PI_F, 0.5f), 0.0f, 1e-7f);
  EXPECT_NEAR(diffraction_binary_power(1, M_PI_F, 0.5f), 4.0 / (M_PI * M_PI), 1e-7);
  EXPECT_NEAR(diffraction_binary_power(2, M_PI_F, 0.5f), 0.0f, 1e-12f);
}

TEST(KernelDiffraction, ReliefPhaseAndFlatLimit)
{
  EXPECT_FLOAT_EQ(diffraction_relief_phase(0.0f, 550.0f, 1.0f, 1.0f, 1.0f, 1.0f), 0.0f);
  EXPECT_NEAR(diffraction_relief_phase(137.5f, 550.0f, 1.0f, 1.0f, 1.0f, 1.0f), M_PI_F, 1e-6f);
  EXPECT_FLOAT_EQ(diffraction_relief_phase(100.0f, 550.0f, 1.0f, 1.5f, 0.8f, -0.6f),
                  diffraction_relief_phase(100.0f, 550.0f, 1.5f, 1.0f, -0.6f, 0.8f));
}

CCL_NAMESPACE_END
