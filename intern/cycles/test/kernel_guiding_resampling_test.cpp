/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <gtest/gtest.h>

#include "kernel/sample/guiding_resampling.h"

#include <array>
#include <random>

CCL_NAMESPACE_BEGIN

namespace {
using Distribution = std::array<float, 4>;
std::mt19937 generator(716392);

float random_uniform()
{
  return (float(generator() >> 9) + 0.5f) * 0x1p-23f;
}

int sample_discrete(const Distribution &p)
{
  float value = random_uniform();
  for (int i = 0; i < 3; ++i) {
    if (value < p[i]) {
      return i;
    }
    value -= p[i];
  }
  return 3;
}

float mixture(const Distribution &p, const Distribution &q, const int x)
{
  return 0.5f * (p[x] + q[x]);
}

int resample(const Distribution &p,
             const Distribution &q,
             const Distribution &target,
             float &weight)
{
  const int candidate[2] = {sample_discrete(p), sample_discrete(q)};
  GuidingResamplingPair pair{{target[candidate[0]], target[candidate[1]]},
                             {mixture(p, q, candidate[0]), mixture(p, q, candidate[1])}};
  float effective_pdf;
  const int selected = pair.sample(random_uniform(), &effective_pdf);
  weight = selected >= 0 ? 1.0f / effective_pdf : 0.0f;
  return selected >= 0 ? candidate[selected] : 0;
}

void check_estimates(const double sum,
                     const double sum_squared,
                     const int count,
                     const double expected)
{
  const double mean = sum / count;
  const double variance = max(0.0, sum_squared / count - mean * mean);
  EXPECT_NEAR(mean, expected, 6.0 * sqrt(variance / count) + 1e-5);
}
}  // namespace

TEST(GuidingResampling, NullCandidatesAndFiniteLargeWeights)
{
  float pdf;
  GuidingResamplingPair empty{{0, 0}, {0.5f, 0.5f}};
  EXPECT_EQ(empty.sample(0.5f, &pdf), -1);
  EXPECT_EQ(pdf, 0.0f);
  GuidingResamplingPair one{{0, 3}, {0.5f, 0.5f}};
  EXPECT_EQ(one.sample(0.0f, &pdf), 1);
  EXPECT_FLOAT_EQ(pdf, 1.0f);
  GuidingResamplingPair large{{2e38f, 2e38f}, {1, 1}};
  EXPECT_EQ(large.sample(0.25f, &pdf), 0);
  EXPECT_NEAR(pdf, 1.0f, 1e-6f);
}

TEST(GuidingResampling, IntegratesWithPartialSupportAndDirectionalMIS)
{
  const Distribution p{0.0f, 0.2f, 0.3f, 0.5f};
  const Distribution q{0.6f, 0.3f, 0.1f, 0.0f};
  const Distribution target{2.0f, 0.0f, 8.0f, 0.3f};
  const Distribution integrand{0.3f, 0.0f, 12.0f, 0.8f};
  const Distribution mis_weight{0.2f, 0.6f, 0.8f, 0.3f};
  double expected = 0, sum = 0, sum_squared = 0;
  for (int i = 0; i < 4; ++i) {
    expected += double(integrand[i]) * mis_weight[i];
  }
  constexpr int count = 1000000;
  for (int i = 0; i < count; ++i) {
    float weight;
    const int x = resample(p, q, target, weight);
    const double estimate = double(integrand[x]) * mis_weight[x] * weight;
    sum += estimate;
    sum_squared += estimate * estimate;
  }
  check_estimates(sum, sum_squared, count, expected);
}

static void check_bidirectional_composition(const bool use_roulette)
{
  const Distribution forward_p{0.1f, 0.2f, 0.3f, 0.4f};
  const Distribution forward_q{0.5f, 0.1f, 0.2f, 0.2f};
  const Distribution reverse_p{0.4f, 0.3f, 0.2f, 0.1f};
  const Distribution reverse_q{0.2f, 0.2f, 0.1f, 0.5f};
  const Distribution target{0.1f, 2.0f, 0.5f, 3.0f};
  const auto conditional = [](const Distribution &p, const int first) {
    Distribution result;
    for (int i = 0; i < 4; ++i) {
      result[i] = p[(i + first) % 4];
    }
    return result;
  };
  const auto integrand = [](const int x, const int y) { return x == y ? 4.0 : 0.1 * (x + y + 1); };
  double expected = 0, sum = 0, sum_squared = 0;
  for (int x = 0; x < 4; ++x) {
    for (int y = 0; y < 4; ++y) {
      expected += integrand(x, y);
    }
  }
  constexpr int count = 1000000;
  for (int i = 0; i < count; ++i) {
    double estimate = 0;
    for (int reverse = 0; reverse < 2; ++reverse) {
      float w0, w1;
      const Distribution &p = reverse ? reverse_p : forward_p;
      const Distribution &q = reverse ? reverse_q : forward_q;
      const int a = resample(p, q, target, w0);
      const int b = resample(conditional(p, a), conditional(q, a), target, w1);
      const int x = reverse ? b : a, y = reverse ? a : b;
      const double pf = mixture(forward_p, forward_q, x) *
                        mixture(conditional(forward_p, x), conditional(forward_q, x), y);
      const double pr = mixture(reverse_p, reverse_q, y) *
                        mixture(conditional(reverse_p, y), conditional(reverse_q, y), x);
      const double mis = (reverse ? pr : pf) / (pf + pr);
      /* Each strategy may use a different roulette policy. A common partition of unity
       * can omit those survival probabilities, provided contributions compensate for them. */
      const double continuation = use_roulette ? (reverse ? 0.35 + 0.1 * a : 0.8 - 0.1 * b) : 1.0;
      if (!use_roulette || random_uniform() < continuation) {
        estimate += integrand(x, y) * mis * w0 * w1 / continuation;
      }
    }
    sum += estimate;
    sum_squared += estimate * estimate;
  }
  check_estimates(sum, sum_squared, count, expected);
}

TEST(GuidingResampling, BidirectionalCompositionWithDeterministicMIS)
{
  check_bidirectional_composition(false);
}

TEST(GuidingResampling, BidirectionalCompositionWithDifferentRoulettePolicies)
{
  check_bidirectional_composition(true);
}

CCL_NAMESPACE_END
