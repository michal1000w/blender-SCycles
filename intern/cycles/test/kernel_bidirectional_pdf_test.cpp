/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <gtest/gtest.h>

#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/closure/bsdf_microfacet_manifold.h"
#include "kernel/globals.h"
#include "kernel/integrator/bidirectional_transport.h"
#include "kernel/integrator/bidirectional_volume.h"
#include "kernel/light/area.h"
#include "kernel/light/point.h"
#include "kernel/sample/manifold.h"
#include "kernel/sample/sobol_burley.h"

#include <random>
#include <vector>

CCL_NAMESPACE_BEGIN

TEST(BidirectionalPDF, PrimaryVolumeEmissionSharesThreeStrategyPartition)
{
  std::mt19937 rng(70319);
  std::uniform_real_distribution<float> uniform(0.01f, 20.0f);
  for (int trial = 0; trial < 10000; ++trial) {
    const float phase = uniform(rng);
    const float nee = uniform(rng);
    const float camera_inverse_density = uniform(rng);
    const float emission = uniform(rng);
    const double light = double(camera_inverse_density) * emission;
    const double denominator = double(phase) * phase + double(nee) * nee + light * light;

    const float2 after_volume = BDPTMISWeight::Log::scatter(
        make_float2(BDPTMISWeight(camera_inverse_density).encoded(), -INFINITY),
        1.0f,
        phase,
        0.0f);
    const float phase_weight =
        (BDPTMISWeight(1.0f) +
         BDPTMISWeight(nee) * BDPTMISWeight::from_encoded(after_volume.x) +
         BDPTMISWeight(emission) * BDPTMISWeight::from_encoded(after_volume.y))
            .inverse();
    const double nee_weight = double(nee) * nee / denominator;
    const double sensor_weight = light * light / denominator;
    EXPECT_NEAR(phase_weight, double(phase) * phase / denominator, 2e-6);
    EXPECT_NEAR(phase_weight + nee_weight + sensor_weight, 1.0, 2e-6);
  }
  /* Ordinary two-strategy PT weighting is not a valid replacement for the
   * phase-hit term while the light-to-sensor strategy still contributes. */
  const double phase = 0.3, nee = 0.7, light = 0.2;
  const double all = phase * phase + nee * nee + light * light;
  EXPECT_GT(phase * phase / (phase * phase + nee * nee) +
                (nee * nee + light * light) / all,
            1.0);
}

template<int Exponent> static void check_mis_recurrence_expansion()
{
  std::mt19937 rng(52171);
  std::uniform_real_distribution<float> uniform(.25f, 1.75f);
  for (const int depth : {1, 2, 8, 32}) {
    for (int trial = 0; trial < 100; ++trial) {
      const float initial_cm = uniform(rng), initial_vc = uniform(rng);
      std::vector<float> forward(depth), reverse(depth), cosine(depth), selection(depth);
      float2 state = make_float2(BDPTMISRecurrence<Exponent>::factor(initial_cm),
                                 BDPTMISRecurrence<Exponent>::factor(initial_vc));
      for (int i = 0; i < depth; ++i) {
        forward[i] = uniform(rng);
        reverse[i] = trial % 7 == 0 && i == depth / 2 ? 0 : uniform(rng);
        cosine[i] = .5f * uniform(rng);
        selection[i] = trial % 11 == 0 ? 0 : uniform(rng);
        const float2 before = state;
        state = BDPTMISRecurrence<Exponent>::scatter(
            state, cosine[i], forward[i], reverse[i], selection[i]);
        if constexpr (Exponent == 1) {
          EXPECT_EQ(state.x, 1.0f / forward[i]);
          EXPECT_EQ(state.y,
                    cosine[i] / forward[i] * (before.y * reverse[i] + before.x * selection[i]));
        }
      }
      /* Enumerate complete products for each introduced strategy term, rather
       * than repeating the compact update. Each term propagates independently
       * through the remaining scattering events before its power is taken. */
      double expected = 0;
      for (int origin = -1; origin < depth; ++origin) {
        double ratio = origin < 0 ?
                           double(initial_vc) :
                           (origin == 0 ? double(initial_cm) : 1.0 / double(forward[origin - 1])) *
                               double(cosine[origin]) * selection[origin] / forward[origin];
        for (int i = origin + 1; i < depth; ++i) {
          ratio *= double(cosine[i]) * reverse[i] / forward[i];
        }
        expected += Exponent == 1 ? ratio : ratio * ratio;
      }
      ASSERT_TRUE(std::isfinite(state.y));
      EXPECT_NEAR(state.y, expected, 5e-6 * std::max(expected, 1e-20));
    }
  }
}

TEST(BidirectionalPDF, BalanceRecurrenceMatchesExplicitStrategyProducts)
{
  check_mis_recurrence_expansion<1>();
}

TEST(BidirectionalPDF, PowerRecurrenceMatchesExplicitStrategyProducts)
{
  check_mis_recurrence_expansion<2>();
  const auto result = BDPTMISRecurrence<2>::scatter(make_float2(4, 9), 1, 1, 1);
  EXPECT_EQ(result.y, 13); /* 2^2 + 3^2, never (2 + 3)^2. */
}

template<int Exponent> static void check_log_mis_range()
{
  using Log = BDPTMISLogRecurrence<Exponent>;
  EXPECT_EQ(Log::weight(-INFINITY, -INFINITY), 1);
  EXPECT_EQ(Log::product(-INFINITY, 1000), -INFINITY);
  EXPECT_TRUE(std::isfinite(Log::ratio(1e30f, 1e-30f)));
  EXPECT_EQ(Log::weight(Log::ratio(1e30f, 1e-30f), -INFINITY), 0);
  float2 state = make_float2(-INFINITY, 0);
  for (int i = 0; i < 32; ++i) {
    state = Log::scatter(state, 1, 1, i < 16 ? 1e20f : 1e-20f, 0);
    ASSERT_TRUE(std::isfinite(state.y));
  }
  /* An intermediate ordinary float product overflows, then the reciprocal
   * factors bring the final ratio back to one. Saturation cannot recover it. */
  EXPECT_NEAR(state.y, 0, 2e-4);
  EXPECT_NEAR(Log::weight(state.y, -INFINITY), .5, 5e-5);
}

TEST(BidirectionalPDF, LogMISRecoversAfterExtremeIntermediateRatios)
{
  check_log_mis_range<1>();
  check_log_mis_range<2>();
}

template<int Exponent> static void check_log_mis_partition()
{
  using Log = BDPTMISLogRecurrence<Exponent>;
  /* Direct complete products, stored as mantissa/exponent pairs. This reference
   * never uses logarithmic accumulation or the compact renderer recurrence. */
  constexpr int edges = 32;
  for (int trial = 0; trial < 100; ++trial) {
    std::mt19937 rng(542 + trial);
    std::uniform_real_distribution<float> uniform(.5f, 2.0f);
    double mantissas[edges + 1];
    int exponents[edges + 1];
    float logs[edges + 1];
    float forward[edges], reverse[edges];
    for (int i = 0; i < edges; ++i) {
      const float scale = trial % 3 == 2 || i < edges / 2 ? 1e20f : 1e-20f;
      forward[i] = uniform(rng) * scale;
      reverse[i] = uniform(rng) * (trial % 3 == 1 ? 1.0f / scale : scale);
    }
    int max_exponent = -100000;
    for (int strategy = 0; strategy <= edges; ++strategy) {
      const double count = strategy == edges ? 2097152 : 512;
      double mantissa = std::frexp(count, &exponents[strategy]);
      float log_density = std::log(float(count));
      for (int i = 0; i < edges; ++i) {
        const float density = i < strategy ? forward[i] : reverse[i];
        int exponent;
        mantissa *= std::frexp(double(density), &exponent);
        exponents[strategy] += exponent;
        mantissa = std::frexp(mantissa, &exponent);
        exponents[strategy] += exponent;
        log_density += std::log(density);
      }
      mantissas[strategy] = mantissa;
      logs[strategy] = Exponent * log_density;
      max_exponent = std::max(max_exponent, exponents[strategy]);
    }
    double probabilities[edges + 1], total = 0;
    for (int s = 0; s <= edges; ++s) {
      const double value = std::ldexp(mantissas[s], exponents[s] - max_exponent);
      probabilities[s] = s % 7 == 3 ? 0 : (Exponent == 1 ? value : value * value);
      total += probabilities[s];
    }
    double sum_weights = 0;
    for (int selected = 0; selected <= edges; ++selected) {
      if (selected % 7 == 3) {
        continue; /* Excluded deterministic-connection strategy. */
      }
      float alternatives = -INFINITY;
      for (int other = 0; other <= edges; ++other) {
        if (other != selected && other % 7 != 3) {
          alternatives = Log::sum(alternatives, logs[other] - logs[selected]);
        }
      }
      const float weight = Log::weight(alternatives, -INFINITY);
      EXPECT_NEAR(weight, probabilities[selected] / total, 2e-4);
      sum_weights += weight;
    }
    EXPECT_NEAR(sum_weights, 1, 2e-4);
  }
}

TEST(BidirectionalPDF, LogMISPartitionMatchesScaledCompleteProducts)
{
  check_log_mis_partition<1>();
  check_log_mis_partition<2>();
}

template<int Exponent> static void check_typed_mis_storage()
{
  using W = BDPTMISWeightT<Exponent>;
  EXPECT_EQ(W().encoded(), -INFINITY);
  EXPECT_TRUE(W::from_encoded(-INFINITY) == 0.0f);
  EXPECT_EQ(W(1.0f).encoded(), 0.0f);
  std::mt19937 rng(9631);
  std::uniform_real_distribution<float> uniform(.1f, 2.0f);
  for (int trial = 0; trial < 1000; ++trial) {
    const float direct = uniform(rng), emission = uniform(rng), cosine = uniform(rng);
    const float distance2 = uniform(rng), reverse = uniform(rng), selection = uniform(rng);
    W cm = W(direct) / emission;
    W vc = W(cosine) / emission;
    // The cache stores encoded terms. Geometry conversion and its null-surface
    // inverse must survive that boundary without interpreting a log as a PDF.
    const float stored_cm = (cm * distance2 / cosine).encoded();
    const float stored_vc = (vc / cosine).encoded();
    cm = W::from_encoded(stored_cm) * cosine / distance2;
    vc = W::from_encoded(stored_vc) * cosine;
    const W alternatives = W(selection) * (cm + vc * reverse);
    const double a = double(selection) * direct / emission;
    const double b = double(selection) * cosine / emission * reverse;
    const double expected = 1.0 / (1.0 + std::pow(a, Exponent) + std::pow(b, Exponent));
    EXPECT_NEAR((W(1.0f) + alternatives).inverse(), expected, 5e-6);
  }
  // Keep intermediates outside float product range recoverable across storage.
  W value(1.0f);
  for (int i = 0; i < 16; ++i) {
    value *= 1e20f;
  }
  value = W::from_encoded(value.encoded());
  for (int i = 0; i < 16; ++i) {
    value *= 1e-20f;
  }
  EXPECT_NEAR((W(1.0f) + value).inverse(), .5f, 5e-5);
}

TEST(BidirectionalPDF, TypedMISStorageGeometryAndEndpointWeights)
{
  check_typed_mis_storage<1>();
  check_typed_mis_storage<2>();
}

template<int Exponent> static void check_geometric_mis_partition(const bool with_media = false)
{
  using W = BDPTMISWeightT<Exponent>;
  std::mt19937 rng(92871);
  std::uniform_real_distribution<float> uniform(.4f, 1.6f);
  for (int edges = 3; edges <= 10; ++edges) {
    for (int trial = 0; trial < 32; ++trial) {
      // Folded paths with a finite area emitter and a fixed sensor endpoint.
      // Both directions lie above each surface's normal. Lambertian directional
      // PDFs are converted to area independently for the full-product oracle.
      std::vector<float3> p(edges + 1), normal(edges + 1);
      std::vector<float> prev(edges + 1), next(edges + 1), distance2(edges);
      std::vector<float> light_pdf(edges + 1), camera_pdf(edges + 1);
      std::vector<bool> delta(edges + 1, false);
      std::vector<float> rho(edges + 1, 1.0f), null_probability(edges, 1.0f);
      if (with_media) {
        for (float &q : null_probability) {
          q = .5f * uniform(rng);
        }
      }
      for (int i = 1; i < edges; ++i) {
        delta[i] = trial % 4 == 0 && i % 3 == 1;
      }
      for (int i = 0; i <= edges; ++i) {
        p[i] = make_float3(float(i), (i % 2 ? 1 : -1) * uniform(rng), .2f * uniform(rng));
      }
      for (int i = 0; i < edges; ++i) {
        distance2[i] = len_squared(p[i + 1] - p[i]);
      }
      next[0] = 1;
      light_pdf[0] = M_1_PI_F;
      for (int i = 1; i < edges; ++i) {
        const float3 to_prev = normalize(p[i - 1] - p[i]);
        const float3 to_next = normalize(p[i + 1] - p[i]);
        normal[i] = normalize(to_prev + to_next * (delta[i] ? 1.0f : uniform(rng)));
        prev[i] = dot(normal[i], to_prev);
        next[i] = dot(normal[i], to_next);
        ASSERT_GT(prev[i], 0);
        ASSERT_GT(next[i], 0);
        // Normalized cosine/uniform-hemisphere mixtures, with different
        // incident-direction-conditioned sampling weights in each direction.
        const float light_mix = trial % 3 ? .5f * uniform(rng) : 1.0f;
        const float camera_mix = trial % 3 ? .5f * uniform(rng) : 1.0f;
        light_pdf[i] = delta[i] ? 1.0f : (light_mix * next[i] + (1 - light_mix) * .5f) * M_1_PI_F;
        camera_pdf[i] = delta[i] ? 1.0f :
                                   (camera_mix * prev[i] + (1 - camera_mix) * .5f) * M_1_PI_F;
      }
      if (with_media) {
        for (int i = 2; i < edges; i += 3) {
          delta[i] = false;
          prev[i] = next[i] = 1.0f;  // Volume measure has no projected normal.
          light_pdf[i] = camera_pdf[i] = .25f * M_1_PI_F;
          rho[i] = uniform(rng);  // Direction-independent real-event intensity.
        }
      }
      const float emitter_area_pdf = .25f * uniform(rng);
      const float camera_omega_pdf = .5f * uniform(rng);
      const float selection_ratio = uniform(rng);
      const float sensor_samples = trial % 2 ? 4096.0f : 1.0f;
      const float emission_pdf = emitter_area_pdf * next[0] * M_1_PI_F;
      std::vector<double> light_area(edges), camera_area(edges);
      for (int i = 0; i < edges - 1; ++i) {
        light_area[i] = double(light_pdf[i]) * prev[i + 1] / distance2[i];
      }
      for (int i = 1; i < edges; ++i) {
        camera_area[i - 1] = double(camera_pdf[i]) * next[i - 1] / distance2[i - 1];
      }
      const double sensor_area_pdf = double(camera_omega_pdf) * next[edges - 1] /
                                     distance2[edges - 1];
      std::vector<double> probability(edges + 1);
      probability[0] = sensor_area_pdf * rho[edges - 1];
      for (int i = 0; i < edges - 1; ++i) {
        probability[0] *= camera_area[i] * rho[i];
      }
      for (int s = 1; s <= edges; ++s) {
        double product = emitter_area_pdf;
        for (int i = 0; i < s - 1; ++i) {
          product *= light_area[i] * rho[i + 1];
        }
        if (s < edges) {
          product *= sensor_area_pdf * rho[edges - 1];
          for (int i = s; i < edges - 1; ++i) {
            product *= camera_area[i] * rho[i];
          }
        }
        probability[s] = delta[s - 1] || delta[s] ? 0 :
                                                    product * (s == 1     ? selection_ratio :
                                                               s == edges ? sensor_samples :
                                                                            1.0f);
      }
      // Complete generalized path probabilities: ratio tracking deterministically
      // selects nulls on its connection edge. All other edges include the
      // random-walk null-event probabilities. Common Poisson factors cancel.
      for (int strategy = 0; strategy <= edges; ++strategy) {
        for (int edge = 0; edge < edges; ++edge) {
          if (strategy != edge + 1) {
            probability[strategy] *= null_probability[edge];
          }
        }
      }
      // Independently normalize full strategy densities, taking powers only
      // after constructing each probability and its strategy sample count.
      double total = 0;
      for (double &density : probability) {
        if constexpr (Exponent == 2) {
          density *= density;
        }
        total += density;
      }
      std::vector<W> light_cm(edges), light_vc(edges), camera_cm(edges), camera_vc(edges);
      W cm = W(emitter_area_pdf) / emission_pdf;
      W vc = W(next[0]) / emission_pdf;
      for (int i = 1; i < edges; ++i) {
        cm = (cm * distance2[i - 1] / prev[i])
                 .scaled_by_log_density(-std::log(null_probability[i - 1]));
        vc /= prev[i];
        light_cm[i] = W::from_encoded(cm.encoded());
        light_vc[i] = W::from_encoded(vc.encoded());
        if (delta[i]) {
          cm = W();
          vc *= next[i];
          continue;
        }
        const float2 scattered = W::Log::scatter(make_float2(cm.encoded(), vc.encoded()),
                                                 next[i],
                                                 light_pdf[i],
                                                 camera_pdf[i],
                                                 i == 1 ? selection_ratio : 1.0f);
        cm = W::from_encoded(scattered.x);
        vc = W::from_encoded(scattered.y);
      }
      cm = W(sensor_samples) / camera_omega_pdf;
      vc = W();
      for (int i = edges - 1; i >= 0; --i) {
        cm = (cm * distance2[i] / next[i]).scaled_by_log_density(-std::log(null_probability[i]));
        vc /= next[i];
        camera_cm[i] = W::from_encoded(cm.encoded());
        camera_vc[i] = W::from_encoded(vc.encoded());
        if (i && delta[i]) {
          cm = W();
          vc *= prev[i];
        }
        else if (i) {
          const float2 scattered = W::Log::scatter(
              make_float2(cm.encoded(), vc.encoded()), prev[i], camera_pdf[i], light_pdf[i]);
          cm = W::from_encoded(scattered.x);
          vc = W::from_encoded(scattered.y);
        }
      }
      double sum = 0;
      for (int s = 0; s <= edges; ++s) {
        if (probability[s] == 0) {
          continue;  // A deterministic connection cannot end at a delta vertex.
        }
        W alternatives;
        if (s == 0) {
          alternatives = W(emitter_area_pdf) * selection_ratio * camera_cm[0] +
                         W(emission_pdf) * camera_vc[0];
        }
        else if (s == 1) {
          const float direct_pdf = emitter_area_pdf * selection_ratio * distance2[0] / next[0];
          alternatives = W(camera_pdf[1]) / direct_pdf +
                         W(emission_pdf) * prev[1] / direct_pdf *
                             (camera_cm[1] + camera_vc[1] * light_pdf[1]);
        }
        else if (s == edges) {
          const int i = edges - 1;
          alternatives = W(float(sensor_area_pdf)) / sensor_samples *
                         (light_cm[i] + light_vc[i] * camera_pdf[i]);
        }
        else {
          alternatives = W(float(camera_area[s - 1])) *
                             (light_cm[s - 1] * (s == 2 ? selection_ratio : 1.0f) +
                              light_vc[s - 1] * camera_pdf[s - 1]) +
                         W(float(light_area[s - 1])) *
                             (camera_cm[s] + camera_vc[s] * light_pdf[s]);
        }
        const float weight = alternatives.connection_weight(
            s == 0 ? 0.0f : std::log(null_probability[s - 1]));
        EXPECT_NEAR(weight, probability[s] / total, 2e-5)
            << "edges=" << edges << " trial=" << trial << " strategy=" << s;
        sum += weight;
      }
      EXPECT_NEAR(sum, 1.0, 2e-5);
    }
  }
}

TEST(BidirectionalPDF, GeometricEndpointPartitionMatchesCompleteProbabilities)
{
  check_geometric_mis_partition<1>();
  check_geometric_mis_partition<2>();
}

TEST(BidirectionalPDF, NullTrackingConnectionsMatchCompletePathProbabilities)
{
  check_geometric_mis_partition<1>(true);
  check_geometric_mis_partition<2>(true);
  using W = BDPTMISWeightT<2>;
  const W large = W(1.0f).scaled_by_log_density(10000.0f);
  EXPECT_NEAR(large.connection_weight(-10000.0f), .5f, 1e-6f);
  EXPECT_EQ(W().connection_weight(-10000.0f), 1.0f);
}

TEST(BidirectionalPDF, WeightedNullEventsMatchSpectralTransport)
{
  const Spectrum extinction = make_float3(.3f, .8f, 1.4f);
  const Spectrum scattering = make_float3(.2f, .6f, .5f);
  for (const float majorant : {.6f, 1.0f, 1.4f, 3.0f, 7.0f}) {
    const auto event = BDPTVolumeEventDistribution::make(extinction, scattering, majorant);
    ASSERT_TRUE(event.valid);
    if (majorant >= 1.4f) {
      EXPECT_NEAR(event.real_probability * majorant, (.2 + .6 + .5) / 3, 1e-7);
    }
    else {
      EXPECT_LT(reduce_min(event.null_weight), 0);
      EXPECT_GT(event.real_probability, 0);
      EXPECT_GT(event.null_probability, 0);
    }
    for (int channel = 0; channel < 3; ++channel) {
      EXPECT_NEAR(event.real_probability * event.real_weight[channel] +
                      event.null_probability * event.null_weight[channel],
                  1.0 - double(extinction[channel] - scattering[channel]) / majorant,
                  2e-7);
    }
    for (const double distance : {.1, .7, 2.0}) {
      double poisson = std::exp(-majorant * distance);
      double moment[3] = {0, 0, 0};
      for (int count = 0; count < 256; ++count) {
        for (int channel = 0; channel < 3; ++channel) {
          moment[channel] += poisson * std::pow(double(event.null_probability), count) *
                             std::pow(double(event.null_weight[channel]), count);
        }
        poisson *= majorant * distance / (count + 1);
      }
      for (int channel = 0; channel < 3; ++channel) {
        const double expected = std::exp(-double(extinction[channel]) * distance);
        EXPECT_NEAR(moment[channel], expected, 5e-7);
        EXPECT_NEAR(moment[channel] * majorant * event.real_probability *
                        event.real_weight[channel],
                    expected * scattering[channel],
                    5e-7);
      }
    }
  }
}

TEST(BidirectionalPDF, WeightedNullEventSupportIsExplicit)
{
  const auto scattering = BDPTVolumeEventDistribution::make(one_spectrum(), one_spectrum(), 1);
  ASSERT_TRUE(scattering.valid);
  EXPECT_EQ(scattering.real_probability, 1);
  EXPECT_EQ(scattering.null_probability, 0);
  EXPECT_TRUE(scattering.sample_real(0.999f));
  const auto absorption = BDPTVolumeEventDistribution::make(one_spectrum(), zero_spectrum(), 1);
  ASSERT_TRUE(absorption.valid);
  EXPECT_EQ(absorption.real_probability, 0);
  EXPECT_FALSE(absorption.sample_real(0));
  EXPECT_TRUE(is_zero(absorption.null_weight));
  const auto underestimated = BDPTVolumeEventDistribution::make(
      one_spectrum(), one_spectrum(), .5f);
  ASSERT_TRUE(underestimated.valid);
  EXPECT_NEAR(underestimated.real_probability, 2.0 / 3.0, 1e-7);
  EXPECT_NEAR(underestimated.null_weight.x, -3, 1e-6);
  const auto signed_absorption = BDPTVolumeEventDistribution::make(
      one_spectrum(), zero_spectrum(), .5f);
  ASSERT_TRUE(signed_absorption.valid);
  EXPECT_EQ(signed_absorption.real_probability, 0);
  EXPECT_EQ(signed_absorption.null_weight.x, -1);
  const auto scattering_at_rate = BDPTVolumeEventDistribution::make(
      make_spectrum(2), one_spectrum(), 1);
  ASSERT_TRUE(scattering_at_rate.valid);
  EXPECT_EQ(scattering_at_rate.real_probability, .5f);
  EXPECT_EQ(scattering_at_rate.null_weight.x, -2);
  EXPECT_FALSE(BDPTVolumeEventDistribution::make(one_spectrum(), one_spectrum(), 0).valid);
  EXPECT_FALSE(BDPTVolumeEventDistribution::make(one_spectrum(), one_spectrum(), INFINITY).valid);
  EXPECT_FALSE(BDPTVolumeEventDistribution::make(zero_spectrum(), one_spectrum(), 1).valid);
}

TEST(BidirectionalPDF, SignedLightPathRoulettePreservesTransport)
{
  const Spectrum magnitudes = make_float3(.2f, .4f, .7f);
  const Spectrum unguided = make_float3(.9f, .8f, .7f);
  const float original = min(saturatef(reduce_max(magnitudes * unguided)), .95f);
  for (int signs = 0; signs < 8; ++signs) {
    Spectrum transport = magnitudes;
    for (int channel = 0; channel < 3; ++channel) {
      if (signs & (1 << channel)) {
        transport[channel] = -transport[channel];
      }
    }
    const float probability = bdpt_light_path_continuation_probability(transport, unguided);
    EXPECT_EQ(probability, original);
    ASSERT_GT(probability, 0);
    for (int channel = 0; channel < 3; ++channel) {
      // Integrate survival and termination outcomes independently; in particular,
      // an all-negative path must not disappear with probability one.
      EXPECT_NEAR(
          double(probability) * (transport[channel] / probability), transport[channel], 1e-7);
    }
  }
  EXPECT_EQ(bdpt_light_path_continuation_probability(zero_spectrum(), unguided), 0);
  EXPECT_EQ(bdpt_light_path_continuation_probability(make_spectrum(-2), unguided), .95f);
}

TEST(BidirectionalPDF, CausticSwitchesPreserveDiffuseAndNullEvents)
{
  for (const bool reflective : {false, true}) {
    for (const bool refractive : {false, true}) {
      EXPECT_TRUE(
          bdpt_caustic_event_enabled(LABEL_REFLECT | LABEL_DIFFUSE, reflective, refractive));
      EXPECT_TRUE(
          bdpt_caustic_event_enabled(LABEL_TRANSMIT | LABEL_DIFFUSE, reflective, refractive));
      EXPECT_TRUE(bdpt_caustic_event_enabled(
          LABEL_TRANSMIT | LABEL_SINGULAR | LABEL_TRANSPARENT, reflective, refractive));
      for (const int scattering : {LABEL_GLOSSY, LABEL_SINGULAR}) {
        EXPECT_EQ(bdpt_caustic_event_enabled(LABEL_REFLECT | scattering, reflective, refractive),
                  reflective);
        EXPECT_EQ(bdpt_caustic_event_enabled(LABEL_TRANSMIT | scattering, reflective, refractive),
                  refractive);
      }
    }
  }
}

TEST(BidirectionalPDF, RectangularNeeRequiresConditionalAreaDensity)
{
  /* Use Cycles' actual rectangle sampler, and independently enumerate the
   * camera-hit, NEE and sensor densities in the same path-area measure.
   * Emission is uniform in area; NEE is uniform in solid angle. Even with
   * one emitter, their position densities are not interchangeable. */
  const float3 receiver = make_float3(0, 0, 0);
  const float3 center = make_float3(0, 0, 1);
  const float3 axis_u = make_float3(1, 0, 0);
  const float3 axis_v = make_float3(0, 1, 0);
  KernelLight light{};
  light.type = LIGHT_AREA;
  light.co = center;
  light.area.axis_u = axis_u;
  light.area.axis_v = axis_v;
  light.area.dir = make_float3(0, 0, -1);
  light.area.len_u = light.area.len_v = 2.0f;
  light.area.invarea = 0.25f;
  const double solid_angle = 4.0 * std::atan(1.0 / std::sqrt(3.0));
  constexpr double emission_area = 0.25;
  constexpr double camera_area = 0.8;
  constexpr double light_samples = 4.0;
  double maximum_wrong_partition_error = 0.0;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      float3 endpoint = center;
      const float pdf_w = area_light_rect_sample(receiver,
                                                 &endpoint,
                                                 axis_u,
                                                 2.0f,
                                                 axis_v,
                                                 2.0f,
                                                 make_float2((x + 0.5f) / 8, (y + 0.5f) / 8),
                                                 true);
      EXPECT_NEAR(pdf_w, 1.0 / solid_angle, 1e-6);
      EXPECT_LE(fabsf(endpoint.x), 1.0f);
      EXPECT_LE(fabsf(endpoint.y), 1.0f);
      EXPECT_FLOAT_EQ(endpoint.z, 1.0f);
      const double distance2 = len_squared(endpoint - receiver);
      const double cosine = 1.0 / std::sqrt(distance2);
      const double nee_area = pdf_w * cosine / distance2;
      const LightEval evaluated = area_light_eval_from_intersection(
          &light, receiver, normalize(endpoint - receiver), std::sqrt(distance2));
      EXPECT_NEAR(evaluated.pdf, pdf_w, 1e-6);
      const double renderer_shape_ratio = evaluated.pdf * cosine * 4.0 / distance2;
      EXPECT_NEAR(renderer_shape_ratio, nee_area / emission_area, 1e-6);

      const double bsdf_area = (cosine / M_PI) * cosine / distance2;
      const double emitted_edge_area = (cosine / M_PI) * cosine / distance2;
      const double p_hit = camera_area * bsdf_area;
      const double p_nee = camera_area * nee_area;
      const double p_sensor = light_samples * emission_area * emitted_edge_area;
      const double total = p_hit + p_nee + p_sensor;

      /* These are the local recursive ratios; the independent path-density
       * enumeration above supplies their expected weights. */
      const double hit_weight = 1 / (1 + nee_area / bsdf_area + p_sensor / p_hit);
      const double nee_weight = 1 / (1 + bsdf_area / nee_area + p_sensor / p_nee);
      const double sensor_weight = 1 / (1 + p_hit / p_sensor +
                                        camera_area * emission_area / p_sensor *
                                            (nee_area / emission_area));
      EXPECT_NEAR(hit_weight, p_hit / total, 1e-12);
      EXPECT_NEAR(nee_weight, p_nee / total, 1e-12);
      EXPECT_NEAR(sensor_weight, p_sensor / total, 1e-12);
      EXPECT_NEAR(hit_weight + nee_weight + sensor_weight, 1.0, 1e-12);

      /* Capture the consequence of correcting only emitter selection: the
       * previous uniform-area assumption fails even when selection is one.
       * This diagnoses the density requirement, not renderer correctness. */
      const double wrong_hit = 1 / (1 + emission_area / bsdf_area + p_sensor / p_hit);
      const double wrong_sensor = 1 /
                                  (1 + p_hit / p_sensor + camera_area * emission_area / p_sensor);
      maximum_wrong_partition_error = std::max(
          maximum_wrong_partition_error, std::fabs(wrong_hit + nee_weight + wrong_sensor - 1));
    }
  }
  EXPECT_GT(maximum_wrong_partition_error, 0.05);
}

TEST(BidirectionalPDF, SphericalNeeAreaDensityMatchesSampleJacobian)
{
  KernelLight light{};
  light.type = LIGHT_POINT;
  light.co = zero_float3();
  light.spot.is_sphere = true;
  light.spot.radius = 1.0f;
  light.spot.eval_fac = 1.0f;
  const float3 receiver = make_float3(0, 0, 3);
  const float3 normal = make_float3(0, 0, -1);
  const double expected_pdf = 1.0 / (2 * M_PI * (1 - std::sqrt(8.0 / 9.0)));
  const auto sample = [&](const float u, const float v) {
    LightSample ls{};
    EXPECT_TRUE(point_light_sample(&light, make_float2(u, v), receiver, normal, 0, &ls));
    return ls;
  };
  constexpr float h = 0.0005f;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      const float u = (x + 0.5f) / 8, v = (y + 0.5f) / 8;
      const LightSample ls = sample(u, v);
      const LightEval eval = point_light_eval_from_intersection(
          &light, receiver, ls.D, ls.t, normal, 0);
      EXPECT_NEAR(ls.pdf, expected_pdf, expected_pdf * 1e-6);
      EXPECT_FLOAT_EQ(eval.pdf, ls.pdf);
      const float3 du = (sample(u + h, v).P - sample(u - h, v).P) / (2 * h);
      const float3 dv = (sample(u, v + h).P - sample(u, v - h).P) / (2 * h);
      const double numerical_area_pdf = 1.0 / len(cross(du, dv));
      const double evaluated_area_pdf = eval.pdf * fabsf(dot(ls.Ng, -ls.D)) / sqr(ls.t);
      EXPECT_NEAR(evaluated_area_pdf, numerical_area_pdf, numerical_area_pdf * 0.005);
    }
  }
}

TEST(BidirectionalPDF, ManifoldInterfaceSamplesHaveProductMeasure)
{
  /* Three independent uniform slope coordinates have sum variance 3/12.
   * Reusing one coordinate at all interfaces instead gives 9/12, even though
   * each interface's marginal distribution remains perfectly uniform. */
  double sum = 0.0, square_sum = 0.0, cross_sum = 0.0;
  constexpr uint samples = 65536;
  for (uint sample = 0; sample < samples; ++sample) {
    float values[3];
    for (uint vertex = 0; vertex < 3; ++vertex) {
      values[vertex] = sobol_burley_sample_2D(
                           sample, manifold_vertex_rng_dimension(vertex), 0x392ff712u, ~0u)
                           .x -
                       0.5f;
    }
    const double value = double(values[0]) + values[1] + values[2];
    sum += value;
    square_sum += value * value;
    cross_sum += double(values[0]) * values[1];
  }
  EXPECT_NEAR(sum / samples, 0.0, 0.003);
  EXPECT_NEAR(square_sum / samples, 0.25, 0.004);
  EXPECT_NEAR(cross_sum / samples, 0.0, 0.002);
}

TEST(BidirectionalPDF, ApertureCameraDensityMatchesDirectionalJacobian)
{
  const float3 dx = make_float3(.013f, 0, 0);
  const float3 dy = make_float3(0, .009f, 0);
  for (const float focus : {1.0f, 3.0f, 12.0f}) {
    for (const float aperture : {0.0f, .2f, .7f}) {
      for (const float shift : {-.3f, 0.0f, .8f}) {
        const float3 point = make_float3(shift, .4f, 1.0f) * focus;
        const float3 lens = make_float3(aperture, -.5f * aperture, 0);
        const auto direction = [&](const float x, const float y) {
          return normalize(point + focus * (x * dx + y * dy) - lens);
        };
        constexpr float h = .1f;
        const float3 du = (direction(h, 0) - direction(-h, 0)) / (2 * h);
        const float3 dv = (direction(0, h) - direction(0, -h)) / (2 * h);
        const float numerical_solid_angle = len(cross(du, dv));
        const float analytic_solid_angle = bdpt_camera_plane_inverse_pdf(
            point - lens, dx * focus, dy * focus);
        EXPECT_NEAR(analytic_solid_angle, numerical_solid_angle, numerical_solid_angle * 2e-4f);
      }
    }
  }
}

TEST(BidirectionalPDF, CameraClippingPreservesDirectStrategyPartition)
{
  /* Independently evaluate the three direct-path strategy densities in area measure:
   * camera BSDF hit, camera NEE, and light tracing to the sensor. Clipping changes
   * the intersection origin, not the camera or the physical path. */
  const float camera_distance = 8.0f;
  const float camera_pdf_w = 5000.0f;
  const float receiver_cosine = 0.7f;
  const float camera_pdf_a = camera_pdf_w * receiver_cosine / sqr(camera_distance);
  const float emitter_pdf = 2.0f;
  const float bsdf_hit_pdf = 0.03f;
  const float emitted_pdf_a = 0.02f;
  const float light_samples = 4096.0f;
  const float p_bsdf = camera_pdf_a * bsdf_hit_pdf;
  const float p_nee = camera_pdf_a * emitter_pdf;
  const float p_sensor = light_samples * emitter_pdf * emitted_pdf_a;
  const float sum = p_bsdf + p_nee + p_sensor;
  for (const float clipped_distance : {8.0f, 7.9f, 7.0f, 2.0f}) {
    const float d_vcm = light_samples *
                        bdpt_camera_clip_measure(
                            1.0f / camera_pdf_w, sqr(camera_distance), clipped_distance) *
                        sqr(clipped_distance) / receiver_cosine;
    const float nee_weight = 1.0f / (1.0f + bsdf_hit_pdf / emitter_pdf + emitted_pdf_a * d_vcm);
    EXPECT_NEAR(nee_weight, p_nee / sum, 1e-6f);
    EXPECT_NEAR(nee_weight + p_bsdf / sum + p_sensor / sum, 1.0f, 1e-6f);
  }
}

TEST(BidirectionalPDF, IndexMatchedTransmissionIsDiscrete)
{
  MicrofacetBsdf bsdf{};
  bsdf.N = make_float3(0, 0, 1);
  bsdf.ior = 1.0f + 1e-5f;
  bsdf.alpha_x = bsdf.alpha_y = 0.2f;
  bsdf.type = CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID;
  bsdf.fresnel_type = MicrofacetFresnel::DIELECTRIC;
  ASSERT_NE(bsdf_microfacet_eval_flag(&bsdf), 0);
  Spectrum eval;
  float3 wo;
  float pdf, eta;
  float2 roughness;
  const int label = bsdf_microfacet_sample<GGX>(nullptr,
                                                (const ShaderClosure *)&bsdf,
                                                bsdf.N,
                                                bsdf.N,
                                                make_float3(0.3f, 0.7f, 0.5f),
                                                &eval,
                                                &wo,
                                                &pdf,
                                                &roughness,
                                                &eta);
  EXPECT_NE(label & LABEL_TRANSMIT, 0);
  EXPECT_NE(label & LABEL_SINGULAR, 0);
  EXPECT_GT(pdf, 0.0f);
}

TEST(BidirectionalPDF, MicrofacetTransmissionReciprocity)
{
  std::mt19937 rng(73921);
  const auto random = [&]() { return (float(rng() >> 9) + 0.5f) * 0x1p-23f; };
  int checked = 0;
  for (const float ior : {1.45f, 1.0f / 1.45f}) {
    for (const float roughness : {0.18f, 0.55f}) {
      for (const float tilt : {0.0f, 0.35f}) {
        const float3 geometric_normal = make_float3(0, 0, 1);
        MicrofacetBsdf forward{};
        forward.N = normalize(make_float3(tilt, 0, 1));
        forward.ior = ior;
        forward.alpha_x = forward.alpha_y = sqr(roughness);
        forward.type = CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID;
        forward.fresnel_type = MicrofacetFresnel::DIELECTRIC;
        for (int sample = 0; sample < 1000; ++sample) {
          const float3 wi = normalize(make_float3(random() - 0.5f, random() - 0.5f, 1));
          Spectrum eval;
          float3 wo;
          float pdf, eta;
          float2 sampled_roughness;
          const int label = bsdf_microfacet_sample<GGX>(nullptr,
                                                        (const ShaderClosure *)&forward,
                                                        geometric_normal,
                                                        wi,
                                                        make_float3(random(), random(), random()),
                                                        &eval,
                                                        &wo,
                                                        &pdf,
                                                        &sampled_roughness,
                                                        &eta);
          if (!(label & LABEL_TRANSMIT) || !(pdf > 0.0f)) {
            continue;
          }
          MicrofacetBsdf reverse = forward;
          reverse.N = -forward.N;
          reverse.ior = 1.0f / ior;
          float reverse_pdf = 0.0f;
          const Spectrum reverse_eval = bsdf_microfacet_eval<GGX>(
              nullptr, (const ShaderClosure *)&reverse, wo, wi, &reverse_pdf);
          ASSERT_GT(reverse_pdf, 0.0f);
          const float camera_f = average(eval) / fabsf(dot(forward.N, wo));
          const float adjoint_f = average(reverse_eval) / fabsf(dot(reverse.N, wi));
          ASSERT_GT(camera_f, 0.0f);
          EXPECT_NEAR(adjoint_f / camera_f, 1.0f / sqr(eta), 2e-3f);
          const Spectrum transpose = bdpt_transpose_surface_eval(
              reverse_eval, geometric_normal, wi, wo);
          const Spectrum analytic = bdpt_transpose_delta_eval(
              eval, forward.N, geometric_normal, wi, wo, eta, true);
          ASSERT_GT(average(analytic), 0.0f);
          EXPECT_NEAR(average(transpose) / average(analytic), 1.0f, 2e-3f);
          ++checked;
        }
      }
    }
  }
  EXPECT_GT(checked, 6000);
}

TEST(BidirectionalPDF, GrazingAdjointMeasureIsFinite)
{
  const float3 normal = make_float3(0, 0, 1);
  const float3 grazing = make_float3(1, 0, 0);
  EXPECT_TRUE(is_zero(bdpt_transpose_surface_eval(one_spectrum(), normal, grazing, normal)));
  EXPECT_TRUE(is_zero(
      bdpt_transpose_delta_eval(one_spectrum(), normal, normal, grazing, normal, 1.45f, true)));
  EXPECT_TRUE(isfinite_safe(bdpt_transpose_surface_eval(one_spectrum(), normal, normal, grazing)));
}

TEST(BidirectionalPDF, ManifoldNormalsMatchGlassAndRefractionDistributions)
{
  constexpr int samples = 10000;
  constexpr float alpha = .3f;
  for (const ClosureType type : {CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID,
                                CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID,
                                CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID,
                                CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID}) {
    int inside = 0;
    for (int i = 0; i < samples; ++i) {
      const float2 h = mnee_sample_bsdf_dh(type, alpha, alpha, (i + .5f) / samples, .317f);
      ASSERT_TRUE(isfinite_safe(h.x) && isfinite_safe(h.y));
      inside += dot(h, h) <= alpha * alpha / (1 + alpha * alpha);
    }
    const bool beckmann = type == CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID ||
                          type == CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID;
    // Independently integrate the normal density up to tan(theta)=alpha.
    const double expected = beckmann ? 1 - std::exp(-1.0) : .5;
    EXPECT_NEAR(double(inside) / samples, expected, 2.0 / samples) << int(type);
  }
}

TEST(BidirectionalPDF, ManifoldTransmissionMatchesCameraBsdfInHalfVectorMeasure)
{
  for (const ClosureType type : {CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID,
                                CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID,
                                CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID,
                                CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID}) {
    const bool beckmann = type == CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID ||
                          type == CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID;
    for (const float energy_scale : {1.0f, 1.7f}) {
      for (const float cosine : {.3f, .7f, 1.0f}) {
        MicrofacetBsdf bsdf{};
        bsdf.type = type;
        bsdf.N = make_float3(0, 0, 1);
        bsdf.weight = make_float3(.3f, .5f, .8f);
        bsdf.alpha_x = bsdf.alpha_y = .35f;
        bsdf.ior = 1.45f;
        bsdf.energy_scale = energy_scale;
        bsdf.fresnel_type = MicrofacetFresnel::DIELECTRIC;
        ShaderClosure *closure = reinterpret_cast<ShaderClosure *>(&bsdf);
        const float3 wi = make_float3(safe_sqrtf(1 - cosine * cosine), 0, cosine);
        const float3 h = normalize(make_float3(.08f, -.03f, 1));
        const float3 wo = refract(-wi, h, 1 / bsdf.ior);
        ASSERT_LT(wo.z, 0);
        float pdf = 0;
        const Spectrum camera = beckmann ?
            bsdf_microfacet_beckmann_eval(nullptr, closure, wi, wo, &pdf) :
            bsdf_microfacet_ggx_eval(nullptr, closure, wi, wo, &pdf);
        ASSERT_GT(pdf, 0);
        const float D = beckmann ? bsdf_D<MicrofacetType::BECKMANN>(sqr(.35f), h.z) :
                                   bsdf_D<MicrofacetType::GGX>(sqr(.35f), h.z);
        const float half_vector_pdf = D * h.z;
        const float direction_jacobian = sqr(dot(wi, h) + bsdf.ior * dot(wo, h)) /
                                         (sqr(bsdf.ior) * fabsf(dot(wo, h)));
        const Spectrum expected = bsdf.weight * camera * direction_jacobian /
                                  (half_vector_pdf * h.z);
        const Spectrum actual = mnee_eval_bsdf_contribution(nullptr, closure, wi, wo);
        for (int c = 0; c < 3; ++c) {
          EXPECT_NEAR(actual[c], expected[c], 2e-5f * fabsf(expected[c]))
              << int(type) << " energy=" << energy_scale << " cosine=" << cosine;
        }
      }
    }
  }
}

CCL_NAMESPACE_END
