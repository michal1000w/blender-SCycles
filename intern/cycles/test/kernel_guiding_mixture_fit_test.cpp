/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <array>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "kernel/sample/guiding_mixture_conditional.h"
#include "kernel/sample/guiding_mixture_fit.h"
#include "kernel/sample/guiding_mixture_statistics.h"
#include "kernel/sample/guiding_observation_range.h"

CCL_NAMESPACE_BEGIN

TEST(GuidingMixtureFit, ConcentrationInvertsMeanResultant)
{
  for (const double k : {0.0, .0001, .01, .1, 1.0, 5.0, 32.0, 512.0, 4000.0, 16384.0}) {
    const double r = k == 0 ? 0 : 1.0 / std::tanh(k) - 1.0 / k;
    const float fitted = GuidingDirectionalMixtureFit<2>::concentration(float(r));
    EXPECT_NEAR(fitted, k, std::max(1e-5, k * .001));
  }
}

TEST(GuidingMixtureFit, ConditionalLobePriorMatchesDoubleMeanCosine)
{
  const float3 axis = normalize(make_float3(.2f, -.3f, 1));
  for (const float k : {0.0f, .001f, .1f, 1.0f, 10.0f, 100.0f, 4000.0f, 16384.0f}) {
    const GuidingSphericalGaussian lobe{axis, k};
    EXPECT_EQ(lobe.regularized(64, 0).concentration, k);
    EXPECT_EQ(lobe.regularized(0, .2f).concentration, 0);
    for (const float support : {.01f, 1.0f, 16.0f, 1024.0f, 1000000.0f}) {
      const auto result = lobe.regularized(support, .2f);
      const double input_mean = k == 0 ? 0 : 1 / std::tanh(double(k)) - 1 / double(k);
      const double expected = input_mean * double(support) / (double(support) + .2);
      const double actual = result.concentration == 0 ?
                                0 :
                                1 / std::tanh(double(result.concentration)) -
                                    1 / double(result.concentration);
      EXPECT_NEAR(actual, expected, 2e-6) << k << " support=" << support;
      EXPECT_EQ(len_squared(result.axis - axis), 0);
      EXPECT_GE(result.concentration, 0);
      EXPECT_LE(result.concentration, k);
    }
  }
}

TEST(GuidingMixtureFit, PriorSurvivesEveryConditionalQuery)
{
  GuidingGaussianMixture query;
  for (const int mode : {0, 1, 2}) {
    std::array<float, GuidingGaussianMixture::storage_size> storage{};
    storage[0] = 1;
    storage[1] = 1000;
    storage[4] = 1;
    storage[7] = 3;
    storage[8] = .01f;
    storage[11] = .01f;
    storage[13] = .01f;
    storage[14] = storage[15] = storage[16] = 1;
    storage[17] = mode;
    for (const float3 p : {zero_float3(), make_float3(.2f, -.4f, .8f), make_float3(-2, 1, -4)}) {
      storage[18] = 0;
      const auto before = query.component(storage.data(), 0, p);
      storage[18] = .2f;
      const auto after = query.component(storage.data(), 0, p);
      const double expected = (1 / std::tanh(double(before.concentration)) -
                               1 / double(before.concentration)) /
                              1.2;
      const double actual = 1 / std::tanh(double(after.concentration)) -
                            1 / double(after.concentration);
      EXPECT_NEAR(actual, expected, 2e-6) << mode;
      EXPECT_EQ(len_squared(before.axis - after.axis), 0);
    }
  }
}

TEST(GuidingMixtureFit, WeightPriorPreservesMassAndRevivesEmptyComponents)
{
  GuidingConditionalMixture publisher;
  std::array<float, GuidingGaussianMixture::storage_size> output{};
  float total = 0;
  for (int c = 0; c < GuidingGaussianMixture::components; ++c) {
    GuidingMixtureStatistics statistics;
    statistics.values[0] = c == 0 ? 16 : 0;
    statistics.values[22] = c == 0 ? 16 : 0;
    publisher.publish_component(output.data(),
                                c,
                                statistics.values,
                                16,
                                16,
                                {make_float3(0, 0, 1), c == 0 ? 1000.0f : 0.0f},
                                one_float3(),
                                16);
    const float mass = output[c * GuidingGaussianMixture::component_stride];
    EXPECT_GT(mass, 0);
    EXPECT_NEAR(mass,
                ((c == 0 ? 16.0 : 0.0) + .01) /
                    (16.0 + .01 * GuidingGaussianMixture::components),
                1e-7);
    total += mass;
  }
  EXPECT_NEAR(total, 1, 1e-6);
}

TEST(GuidingMixtureFit, GlobalSupportDoesNotCountSoftAssignmentsAsIndependentPaths)
{
  for (const float scale : {1.0f, 1e30f}) {
    GuidingMixtureStatistics components[2];
    double sum = 0, square = 0;
    for (int i = 0; i < 1000; ++i) {
      const float w = i == 17 ? 1.0f : .001f;
      const float responsibility = .125f * (1 + i % 7);
      const GuidingMixtureObservation observation{
          make_float4(0, 0, 1, w * scale), make_float4(0), make_float4(0)};
      components[0].record(observation, responsibility, scale, one_float3());
      components[1].record(observation, 1 - responsibility, scale, one_float3());
      sum += w;
      square += double(w) * w;
    }
    const double mass = double(components[0].values[0]) + components[1].values[0];
    const double squared =
        double(components[0].values[GuidingMixtureStatistics::global_squared_weight]) +
        components[1].values[GuidingMixtureStatistics::global_squared_weight];
    EXPECT_NEAR(mass * mass / squared, sum * sum / square, 1e-5);
    EXPECT_LT(mass * mass / squared, 4.0);
  }
}

TEST(GuidingMixtureFit, TwoPeaksWithinOneBinAndWeightScaleInvariance)
{
  const GuidingSphericalGaussian sources[2] = {{normalize(make_float3(.2f, .4f, 1)), 1500},
                                               {normalize(make_float3(.4f, .2f, 1)), 1500}};
  GuidingDirectionalTree<5> tree;
  ASSERT_EQ((tree.leaf_index(sources[0].axis) - tree.leaf_offset) / 64,
            (tree.leaf_index(sources[1].axis) - tree.leaf_offset) / 64);
  std::mt19937 generator(816);
  std::uniform_real_distribution<float> uniform(0, 1);
  std::vector<float4> samples;
  for (int i = 0; i < 8192; ++i) {
    float pdf;
    const float3 direction = sources[i % 2].sample(
        make_float2(uniform(generator), uniform(generator)), &pdf);
    samples.push_back(make_float4(direction.x, direction.y, direction.z, i % 2 ? .65f : .35f));
  }
  GuidingDirectionalMixtureFit<2> model;
  GuidingDirectionalMixtureFit<1> single;
  ASSERT_TRUE(model.initialize(samples.data(), samples.size()));
  ASSERT_TRUE(single.initialize(samples.data(), samples.size()));
  double previous = -1e30;
  for (int iteration = 0; iteration < 64; ++iteration) {
    ASSERT_TRUE(model.iterate(samples.data(), samples.size()));
    ASSERT_TRUE(single.iterate(samples.data(), samples.size()));
    double likelihood = 0;
    for (const auto &sample : samples) {
      likelihood += sample.w * std::log(model.pdf(make_float3(sample))) / samples.size();
    }
    EXPECT_GE(likelihood, previous - 2e-5) << "iteration " << iteration;
    previous = likelihood;
  }
  EXPECT_NEAR(model.weights[0] + model.weights[1], 1, 1e-6);
  for (int i = 0; i < 2; ++i) {
    const int closest = dot(model.lobes[0].axis, sources[i].axis) >
                                dot(model.lobes[1].axis, sources[i].axis) ?
                            0 :
                            1;
    EXPECT_LT(len(model.lobes[closest].axis - sources[i].axis), .003f);
    EXPECT_NEAR(model.weights[closest], i ? .65f : .35f, .01f);
    EXPECT_NEAR(model.lobes[closest].concentration, 1500, 100);
    /* These peaks are separated by many standard deviations, so known source
     * membership gives an independent high-precision moment reference. */
    double dx = 0, dy = 0, dz = 0, mass = 0;
    for (size_t j = i; j < samples.size(); j += 2) {
      const auto sample = samples[j];
      const double norm = std::sqrt(double(sample.x) * sample.x + double(sample.y) * sample.y +
                                    double(sample.z) * sample.z);
      dx += sample.w * sample.x / norm;
      dy += sample.w * sample.y / norm;
      dz += sample.w * sample.z / norm;
      mass += sample.w;
    }
    const double resultant = std::sqrt(dx * dx + dy * dy + dz * dz) / mass;
    const double expected_concentration = 1 / (1 - resultant);
    EXPECT_NEAR(model.lobes[closest].concentration, expected_concentration, 1.0);
  }
  double held_out_gain = 0;
  for (int i = 0; i < 10000; ++i) {
    const int source = uniform(generator) < .35f ? 0 : 1;
    float pdf;
    const float3 direction = sources[source].sample(
        make_float2(uniform(generator), uniform(generator)), &pdf);
    held_out_gain += std::log(model.pdf(direction)) - std::log(single.pdf(direction));
  }
  EXPECT_GT(held_out_gain / 10000, 1.0);
  std::array<float, GuidingGaussianMixture::storage_size> storage{};
  model.publish(storage.data());
  GuidingGaussianMixture query;
  double integral = 0;
  constexpr int resolution = 1024;
  for (int y = 0; y < resolution; ++y) {
    for (int x = 0; x < resolution; ++x) {
      const float3 direction = tree.square_to_direction(
          make_float2((x + .5f) / resolution, (y + .5f) / resolution));
      integral += query.pdf(storage.data(), direction) * (4 * M_PI) / (resolution * resolution);
    }
  }
  EXPECT_NEAR(integral, 1, 1e-4);
  const GuidingGaussianProduct uniform_profile = {
      {{make_float3(0, 0, 1), 0}, {make_float3(0, 0, 1), 0}}, {1, 0}};
  float3 sampled_mean = zero_float3();
  for (int i = 0; i < 30000; ++i) {
    float pdf;
    const float3 direction = query.sample_product(
        storage.data(),
        uniform_profile,
        make_float2(uniform(generator), uniform(generator)),
        &pdf);
    EXPECT_NEAR(pdf, query.pdf(storage.data(), direction), 2e-5f * max(pdf, 1.0f));
    sampled_mean += direction / 30000;
  }
  float3 expected_mean = zero_float3();
  for (int i = 0; i < model.active; ++i) {
    const double k = model.lobes[i].concentration;
    expected_mean += model.lobes[i].axis *
                     float(.95 * model.weights[i] * (1 / std::tanh(k) - 1 / k));
  }
  EXPECT_LT(len(sampled_mean - expected_mean), .008f);
  for (auto &sample : samples) {
    sample.w *= 1e30f;
  }
  GuidingDirectionalMixtureFit<2> bright;
  ASSERT_TRUE(bright.initialize(samples.data(), samples.size()));
  for (int iteration = 0; iteration < 64; ++iteration) {
    ASSERT_TRUE(bright.iterate(samples.data(), samples.size()));
  }
  for (int i = 0; i < 2; ++i) {
    EXPECT_LT(len(model.lobes[i].axis - bright.lobes[i].axis), 1e-5f);
    EXPECT_NEAR(model.weights[i], bright.weights[i], 1e-5f);
    EXPECT_NEAR(model.lobes[i].concentration, bright.lobes[i].concentration, 1.0f);
  }
}

TEST(GuidingMixtureFit, EmptyAndIsotropicObservations)
{
  GuidingDirectionalMixtureFit<1> model;
  ASSERT_FALSE(model.initialize(static_cast<const float4 *>(nullptr), 0));
  ASSERT_FALSE(model.iterate(static_cast<const float4 *>(nullptr), 0));
  const std::array<float4, 6> samples = {make_float4(1, 0, 0, 1),
                                         make_float4(-1, 0, 0, 1),
                                         make_float4(0, 1, 0, 1),
                                         make_float4(0, -1, 0, 1),
                                         make_float4(0, 0, 1, 1),
                                         make_float4(0, 0, -1, 1)};
  ASSERT_TRUE(model.initialize(samples.data(), samples.size()));
  ASSERT_TRUE(model.iterate(samples.data(), samples.size()));
  EXPECT_EQ(model.lobes[0].concentration, 0);
  EXPECT_NEAR(model.pdf(make_float3(0, 0, 1)), M_1_4PI_F, 1e-7f);
}

TEST(GuidingMixtureFit, CoincidentDirectionsAndEmptyPublication)
{
  GuidingDirectionalMixtureFit<16> model;
  std::array<float, GuidingGaussianMixture::storage_size> storage{};
  model.publish(storage.data());
  GuidingGaussianMixture query;
  EXPECT_NEAR(query.pdf(storage.data(), make_float3(0, 0, 1)), M_1_4PI_F, 1e-7f);
  std::array<float4, 64> observations;
  observations.fill(make_float4(0, 0, 1, 1e30f));
  ASSERT_TRUE(model.initialize(observations.data(), observations.size()));
  EXPECT_EQ(model.active, 1);
  ASSERT_TRUE(model.iterate(observations.data(), observations.size()));
  EXPECT_EQ(model.lobes[0].concentration, 16384);
  EXPECT_EQ(model.weights[0], 1);
  model.publish(storage.data());
  EXPECT_GT(query.pdf(storage.data(), make_float3(0, 0, 1)), 2000);
}

TEST(GuidingMixtureFit, SoftSourceStatisticsPreserveMassVarianceAndSupport)
{
  const float3 direction = make_float3(0, 0, 1);
  const GuidingMixtureObservation observation{
      make_float4(0, 0, 1, 4), make_float4(.2f, .3f, .4f, 0), make_float4(4, 2, 10, 0)};
  GuidingMixtureStatistics components[2];
  for (int i = 0; i < 64; ++i) {
    components[0].record(observation, .25f, 4, make_float3(1, .5f, .25f));
    components[1].record(observation, .75f, 4, make_float3(1, .5f, .25f));
  }
  EXPECT_NEAR(components[0].values[0] + components[1].values[0], 64, 1e-6);
  EXPECT_NEAR(components[0].values[GuidingMixtureStatistics::global_squared_weight] +
                  components[1].values[GuidingMixtureStatistics::global_squared_weight],
              64,
              1e-6);
  for (int i = 0; i < 2; ++i) {
    auto &statistics = components[i];
    const float *source = statistics.values + statistics.position_size;
    GuidingParallaxMoments model;
    const auto fit = model.fit(source, statistics.values[0], 64, direction);
    ASSERT_TRUE(fit.valid);
    EXPECT_LT(len(fit.target - make_float3(.2f, .15f, 2.1f)), 1e-5f);
    EXPECT_NEAR(fit.covariance[5], 1, 1e-5f);
    EXPECT_NEAR(source[0] * source[0] / source[1], 64, 1e-4f);
    EXPECT_NEAR(source[2] * source[2] / source[3], 64, 1e-4f);
    EXPECT_NEAR(statistics.values[0] * statistics.values[0] / statistics.values[22], 64, 1e-4f);
  }
  GuidingMixtureStatistics distant;
  auto unknown = observation;
  unknown.source = make_float4(0);
  for (int i = 0; i < 64; ++i) {
    distant.record(unknown, 1, 4, one_float3());
  }
  EXPECT_EQ(distant.values[0], 64);
  EXPECT_EQ(distant.values[distant.position_size], 0);
  GuidingParallaxMoments model;
  EXPECT_FALSE(model.fit(distant.values + distant.position_size, 64, 64, direction).valid);
}

TEST(GuidingMixtureFit, ConditionalPublicationAndPositionDependentPdf)
{
  const float3 extent = make_float3(1, .5f, .25f);
  const float3 target = make_float3(.2f, .4f, 1);
  GuidingMixtureStatistics statistics;
  for (int i = 0; i < 256; ++i) {
    const float3 p = make_float3((i % 8) * .01f, ((i / 8) % 8) * .01f, (i / 64) * .01f);
    const float3 delta = target - p * extent;
    const float distance = len(delta);
    const float3 direction = delta / distance;
    statistics.record({make_float4(direction.x, direction.y, direction.z, 1),
                       make_float4(p.x, p.y, p.z, 0),
                       make_float4(1, 1 / distance, distance, 0)},
                      1,
                      1,
                      extent);
  }
  std::array<float, GuidingGaussianMixture::storage_size> output{};
  const auto saved = statistics;
  GuidingConditionalMixture publisher;
  publisher.publish_component(output.data(),
                              0,
                              statistics.values,
                              statistics.values[0],
                              256,
                              {normalize(target), 10},
                              extent);
  for (int i = 0; i < statistics.storage_size; ++i) {
    EXPECT_EQ(statistics.values[i], saved.values[i]);
  }
  EXPECT_EQ(output[17], 2);
  GuidingGaussianMixture query;
  const float3 p = make_float3(.8f, .2f, -.5f);
  const auto fitted = query.component(output.data(), 0, p);
  EXPECT_LT(len(fitted.axis - normalize(target - p * extent)), 1e-5f);
  const GuidingGaussianProduct uniform = {{{make_float3(0, 0, 1), 0}, {make_float3(0, 0, 1), 0}},
                                          {1, 0}};
  std::mt19937 generator(913);
  std::uniform_real_distribution<float> random(0, 1);
  float3 mean = zero_float3();
  for (int i = 0; i < 20000; ++i) {
    float pdf;
    const float3 direction = query.sample_product(
        output.data(), uniform, make_float2(random(generator), random(generator)), &pdf, p);
    EXPECT_NEAR(pdf, query.pdf(output.data(), direction, p), 2e-5f * max(pdf, 1.0f));
    mean += direction / 20000;
  }
  EXPECT_LT(len(mean - fitted.axis * (.95f * (1 - 1 / fitted.concentration))), .008f);
  GuidingMixtureStatistics empty;
  publisher.publish_component(
      output.data(), 0, empty.values, 0, 0, {make_float3(0, 0, 1), 0}, extent);
  EXPECT_NEAR(query.pdf(output.data(), make_float3(0, 0, 1), p), M_1_4PI_F, 1e-7f);
}

TEST(GuidingMixtureFit, IndexedHistoryPartitionAndFitting)
{
  constexpr uint fields = 3, count = 192;
  std::array<GuidingHistoryRecord, count> records{};
  for (uint i = 0; i < count; ++i) {
    records[i].field_index = (i % fields) * 100 + 7;
    records[i].parent = i == 0 ? ~0u : i - 1;
    records[i].direction = packed_normal(make_float3(0, 0, 1)).value;
    records[i].radiance = 1;
    records[i].source_weight = 1;
    records[i].inverse_distance_weight = .25f;
    records[i].distance_weight = 4;
  }
  std::array<uint, fields> counts{}, offsets{}, cursors{};
  std::array<uint, count> indices{};
  uint error = 0;
  GuidingObservationPartition partition{records.data(),
                                        counts.data(),
                                        offsets.data(),
                                        cursors.data(),
                                        indices.data(),
                                        &error,
                                        fields,
                                        100,
                                        count};
  for (uint i = 0; i < count; ++i)
    partition.count(i);
  partition.prefix();
  for (uint i = 0; i < count; ++i)
    partition.scatter(i);
  ASSERT_EQ(error, 0u);
  for (uint f = 0; f < fields; ++f) {
    EXPECT_EQ(counts[f], 64u);
    GuidingHistoryObservationRange range{records.data(), indices.data(), offsets[f], 2};
    for (uint i = 0; i < counts[f]; ++i) {
      EXPECT_EQ(records[indices[offsets[f] + i]].field_index / 100, f);
      EXPECT_FLOAT_EQ(range.observation(i).source.y, .5f);
      EXPECT_FLOAT_EQ(range.observation(i).source.z, 2);
    }
    GuidingDirectionalMixtureFit<1> model;
    ASSERT_TRUE(model.initialize(range, counts[f]));
    ASSERT_TRUE(model.iterate(range, counts[f]));
    EXPECT_GT(model.pdf(make_float3(0, 0, 1)), 2000);
  }
  for (uint i = 0; i < count; ++i)
    EXPECT_EQ(records[i].parent, i == 0 ? ~0u : i - 1);
  partition.capacity = count - 1;
  partition.prefix();
  EXPECT_EQ(error, 2u);
  partition.capacity = count;
  counts.fill(0);
  error = 0;
  records[0].field_index = ~0u; /* Delta ancestry is not an observation. */
  records[1].radiance = 0;
  records[2].radiance = NAN;
  for (uint i = 0; i < count; ++i)
    partition.count(i);
  partition.prefix();
  for (uint i = 0; i < count; ++i)
    partition.scatter(i);
  EXPECT_EQ(error, 0u);
  for (uint f = 0; f < fields; ++f)
    EXPECT_EQ(counts[f], 63u);
  records[3].field_index = fields * 100;
  partition.count(3);
  EXPECT_EQ(error, 1u);
}

TEST(GuidingMixtureFit, DisjointBatchMergePreservesScaleAndEffectiveSupport)
{
  for (const float intensity : {1.0f, 1e30f}) {
    std::vector<GuidingMixtureObservation> observations;
    float maximum = 0;
    for (int i = 0; i < 321; ++i) {
      const float w = intensity * float(1 + i % 8);
      const float3 d = normalize(make_float3(.1f + .01f * (i % 5), -.2f, 1));
      observations.push_back({make_float4(d.x, d.y, d.z, w),
                              make_float4(.25f, .5f, .125f, 0),
                              make_float4(w, w * .5f, w * 2, 0)});
      maximum = std::max(maximum, w);
    }
    GuidingMixtureStatistics reference;
    double mass = 0, squared_mass = 0;
    for (uint i = 0; i < observations.size(); ++i) {
      const float responsibility = .25f + .125f * (i % 4);
      reference.record(observations[i], responsibility, maximum, make_float3(1, .5f, .25f));
      const double w = double(observations[i].direction_weight.w) / maximum * responsibility;
      mass += w;
      squared_mass += w * w;
    }
    for (const uint group_size : {1u, 17u, 257u}) {
      for (const bool reverse : {false, true}) {
        GuidingMixtureStatistics merged;
        float scale = 0;
        for (uint begin = 0; begin < observations.size(); begin += group_size) {
          const uint end = std::min(begin + group_size, uint(observations.size()));
          float local_scale = 0;
          for (uint j = begin; j < end; ++j) {
            const uint i = reverse ? observations.size() - 1 - j : j;
            local_scale = std::max(local_scale, observations[i].direction_weight.w);
          }
          GuidingMixtureStatistics batch;
          for (uint j = begin; j < end; ++j) {
            const uint i = reverse ? observations.size() - 1 - j : j;
            batch.record(
                observations[i], .25f + .125f * (i % 4), local_scale, make_float3(1, .5f, .25f));
          }
          ASSERT_TRUE(merged.merge(batch, local_scale, scale));
        }
        EXPECT_EQ(scale, maximum);
        for (int i = 0; i < merged.storage_size; ++i) {
          EXPECT_NEAR(merged.values[i],
                      reference.values[i],
                      2e-6f * std::max(1.0f, std::abs(reference.values[i])))
              << i;
        }
        EXPECT_NEAR(merged.values[0], mass, mass * 2e-6);
        EXPECT_NEAR(merged.values[22], squared_mass, squared_mass * 2e-6);
        EXPECT_NEAR(merged.values[merged.position_size + 1], squared_mass, squared_mass * 2e-6);
        EXPECT_NEAR(
            merged.values[merged.position_size + 3], .25 * squared_mass, squared_mass * 2e-6);
        const auto before = merged;
        const float before_scale = scale;
        GuidingMixtureStatistics invalid;
        invalid.values[invalid.storage_size - 1] = NAN;
        EXPECT_FALSE(merged.merge(invalid, 2 * scale, scale));
        EXPECT_EQ(scale, before_scale);
        for (int i = 0; i < merged.storage_size; ++i) {
          EXPECT_EQ(merged.values[i], before.values[i]);
          EXPECT_EQ(merged.errors[i], before.errors[i]);
        }
      }
    }
  }
}

TEST(GuidingMixtureFit, CenteredStreamingConcentrationMatchesDoubleReference)
{
  std::mt19937 rng(451);
  std::uniform_real_distribution<float> uniform(0, 1);
  const float3 axis = normalize(make_float3(-.65f, .2f, -.73f));
  for (const float k : {.5f, 10.0f, 1000.0f, 15000.0f}) {
    GuidingSphericalGaussian source{axis, k};
    std::vector<GuidingMixtureObservation> data;
    double mass = 0, square = 0;
    double delta[3] = {};
    for (int i = 0; i < 8192; ++i) {
      float pdf;
      const float3 d = source.sample(make_float2(uniform(rng), uniform(rng)), &pdf);
      const float w = float(1 + i % 8);
      data.push_back({make_float4(d.x, d.y, d.z, w), make_float4(0), make_float4(0)});
      const float3 normalized = normalize(d);
      mass += w;
      for (int j = 0; j < 3; ++j) {
        const double x = double(normalized[j]) - axis[j];
        delta[j] += w * x;
        square += w * x * x;
      }
    }
    double variance = square / mass;
    for (int j = 0; j < 3; ++j)
      variance -= (delta[j] / mass) * (delta[j] / mass);
    const double resultant = std::sqrt(std::max(0.0, 1.0 - variance));
    double lower = 0, upper = 16384;
    for (int step = 0; step < 80; ++step) {
      const double middle = (lower + upper) * .5;
      const double mean = 1 / std::tanh(middle) - 1 / middle;
      if (mean < resultant)
        lower = middle;
      else
        upper = middle;
    }
    const double expected = (lower + upper) * .5;
    for (const uint group : {1u, 17u, 1024u}) {
      GuidingMixtureStatistics merged;
      float scale = 0;
      for (uint begin = 0; begin < data.size(); begin += group) {
        const uint end = std::min(begin + group, uint(data.size()));
        float local_scale = 0;
        for (uint i = begin; i < end; ++i)
          local_scale = std::max(local_scale, data[i].direction_weight.w);
        GuidingMixtureStatistics batch;
        batch.direction_reference = axis;
        for (uint i = begin; i < end; ++i)
          batch.record(data[i], 1, local_scale, make_float3(1));
        ASSERT_TRUE(merged.merge(batch, local_scale, scale));
      }
      const auto fit = merged.directional_fit();
      EXPECT_NEAR(fit.concentration, expected, std::max(.001, expected * .0001))
          << k << " " << group;
      double length = 0;
      for (int j = 0; j < 3; ++j)
        length += std::pow(axis[j] + delta[j] / mass, 2);
      for (int j = 0; j < 3; ++j)
        EXPECT_NEAR(fit.axis[j], (axis[j] + delta[j] / mass) / std::sqrt(length), 2e-6);
      auto invalid = merged;
      invalid.direction_reference = make_float3(0, 0, 1);
      EXPECT_FALSE(merged.merge(invalid, scale, scale));
    }
  }
}

TEST(GuidingMixtureFit, DecayedPriorMatchesExplicitWeightedHistory)
{
  std::mt19937 rng(913);
  std::uniform_real_distribution<float> uniform(0, 1);
  std::vector<GuidingMixtureObservation> history;
  GuidingMixtureStatistics accumulated;
  accumulated.direction_reference = normalize(make_float3(.3f, -.2f, 1));
  float scale = 0;
  for (int epoch = 0; epoch < 12; ++epoch) {
    if (epoch > 0) {
      const auto next_reference = accumulated.directional_fit().axis;
      ASSERT_TRUE(accumulated.recenter(next_reference));
      scale *= .25f;
    }
    const GuidingSphericalGaussian source{normalize(make_float3(.3f + .002f * epoch, -.2f, 1)),
                                          1000};
    GuidingMixtureStatistics batch;
    batch.direction_reference = accumulated.direction_reference;
    for (int i = 0; i < 128; ++i) {
      float pdf;
      const float3 d = source.sample(make_float2(uniform(rng), uniform(rng)), &pdf);
      const float w = 1 + i % 2;
      const GuidingMixtureObservation observation{make_float4(d.x, d.y, d.z, w),
                                                  make_float4(.25f, .5f, .125f, 0),
                                                  make_float4(w, .5f * w, 2 * w, 0)};
      history.push_back(observation);
      batch.record(observation, 1, 2, make_float3(1));
    }
    ASSERT_TRUE(accumulated.merge(batch, 2, scale));
    GuidingMixtureStatistics reference;
    reference.direction_reference = accumulated.direction_reference;
    for (uint i = 0; i < history.size(); ++i) {
      auto observation = history[i];
      const float decay = std::pow(.25f, epoch - int(i / 128));
      observation.direction_weight.w *= decay;
      observation.source *= decay;
      reference.record(observation, 1, 2, make_float3(1));
    }
    for (int entry = 0; entry < accumulated.storage_size; ++entry) {
      EXPECT_NEAR(accumulated.values[entry],
                  reference.values[entry],
                  2e-6f * std::max(1.0f, std::abs(reference.values[entry])))
          << epoch << " " << entry;
    }
    EXPECT_NEAR(accumulated.directional_fit().concentration,
                reference.directional_fit().concentration,
                .1f);
  }
}

CCL_NAMESPACE_END
