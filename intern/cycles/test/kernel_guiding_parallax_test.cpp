/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <random>
#include <vector>

#include "kernel/sample/guiding_distribution.h"
#include "kernel/sample/guiding_field.h"
#include "kernel/sample/guiding_parallax.h"
#include "kernel/sample/guiding_position.h"
#include "kernel/sample/guiding_spherical_gaussian.h"

CCL_NAMESPACE_BEGIN

TEST(GuidingParallax, VirtualDistanceReflectionRefractionAndBranches)
{
  GuidingVirtualDistance distance;
  EXPECT_EQ(distance.scatter_scale(false, 0.5f, 1, .6f, .8f), 0);
  EXPECT_EQ(distance.scatter_scale(true, 0, 1, .6f, .8f), 1);
  EXPECT_EQ(distance.propagate(3, FLT_MAX, 0), 3);
  EXPECT_EQ(distance.propagate(3, FLT_MAX, 1), FLT_MAX);
  EXPECT_EQ(distance.propagate(3, 0, FLT_MAX), 3);
  EXPECT_EQ(distance.propagate(FLT_MAX, FLT_MAX, 1), FLT_MAX);
  EXPECT_EQ(distance.scatter_scale(true, 0, 1.5f, .8f, 0), FLT_MAX);
  EXPECT_EQ(distance.scatter_scale(true, 0, -1, .8f, .6f), FLT_MAX);
  const float forward = distance.scatter_scale(true, 0, 1.5f, .8f, .6f);
  const float reverse = distance.scatter_scale(true, 0, 1 / 1.5f, .6f, .8f);
  EXPECT_NEAR(forward * reverse, 1, 1e-6f);
  /* Two branches share the same immutable interface transforms, but not their endpoints. */
  const float branch_a = distance.propagate(2, distance.propagate(3, 5, reverse), forward);
  const float branch_b = distance.propagate(2, distance.propagate(3, 11, reverse), forward);
  EXPECT_NEAR(branch_a, 2 + 3 * (8.0 / 9.0) + 5, 1e-5);
  EXPECT_NEAR(branch_b - branch_a, 6, 1e-5);
  EXPECT_EQ(distance.propagate(2, distance.propagate(3, FLT_MAX, 0), forward),
            distance.propagate(2, 3, forward));
}

TEST(GuidingParallax, SharedSourceAndUnsupportedObservations)
{
  GuidingParallaxMoments model;
  std::array<float, GuidingParallaxMoments::storage_size> moments{};
  const float3 target = make_float3(1, 1, 2);
  for (int y = 0; y < 64; ++y) {
    for (int x = 0; x < 64; ++x) {
      const float3 p = make_float3((x + .5f) / 64 - .5f, (y + .5f) / 64 - .5f, 0);
      const float3 delta = target - p;
      model.record(moments.data(), 1, p, normalize(delta), len(delta));
    }
  }
  const GuidingParallaxFit fit = model.fit(moments.data(), 4096, 4096, make_float3(0, 0, 1));
  ASSERT_TRUE(fit.valid);
  EXPECT_LT(len(fit.target - target), 2e-5f);
  for (const float3 p : {make_float3(.4f, .2f, 0), make_float3(-.3f, -.1f, .5f)}) {
    EXPECT_LT(len(fit.direction(p) - normalize(target - p)), 2e-5f);
    EXPECT_GT(fit.concentration(p), 1000);
  }
  EXPECT_EQ(fit.concentration(fit.target), 0);
  EXPECT_NEAR(len(fit.direction(fit.target)), 1, 1e-6f);
  EXPECT_EQ(fit.concentration(make_float3(1e30f)), 0);
  EXPECT_NEAR(len(fit.direction(make_float3(1e30f))), 1, 1e-6f);
  EXPECT_FALSE(model.fit(moments.data(), 8192, 8192, make_float3(0, 0, 1)).valid);
  /* The same effective support must survive a change in batch weight. */
  auto decayed = moments;
  for (int i = 0; i < model.storage_size; ++i) {
    decayed[i] *= model.squared_weight_entry(i) ? .0625f : .25f;
  }
  const auto after_decay = model.fit(decayed.data(), 1024, 4096, make_float3(0, 0, 1));
  ASSERT_TRUE(after_decay.valid);
  EXPECT_LT(len(after_decay.target - fit.target), 1e-6f);
  moments.fill(0);
  model.record(moments.data(), 1, zero_float3(), make_float3(0, 0, 1), FLT_MAX);
  EXPECT_EQ(moments[0], 0);
  for (int i = 0; i < 128; ++i) {
    model.record(moments.data(), 1, zero_float3(), make_float3(0, 0, 1), i == 0 ? 1e-8f : 1);
  }
  EXPECT_FALSE(model.fit(moments.data(), 128, 128, make_float3(0, 0, 1)).valid);
  moments[0] = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(model.fit(moments.data(), 128, 128, make_float3(0, 0, 1)).valid);
}

TEST(GuidingParallax, AreaSourceQueryNormalizationAndSampling)
{
  GuidingParallaxMoments model;
  std::array<float, GuidingParallaxMoments::storage_size> moments{};
  for (int y = 0; y < 64; ++y) {
    for (int x = 0; x < 64; ++x) {
      const float3 p = make_float3((x % 8) / 8.0f - .5f, (y % 8) / 8.0f - .5f, 0);
      const float3 target = make_float3(
          1 + .7f * ((x + .5f) / 64 - .5f), 1 + .5f * ((y + .5f) / 64 - .5f), 2);
      model.record(moments.data(), 1, p, normalize(target - p), len(target - p));
    }
  }
  const auto fit = model.fit(moments.data(), 4096, 4096, make_float3(0, 0, 1));
  ASSERT_TRUE(fit.valid);
  const float3 near = make_float3(.9f, .7f, .5f), far = make_float3(-1, .3f, -1);
  EXPECT_GT(fit.concentration(far), 3 * fit.concentration(near));
  std::mt19937 rng(524);
  GuidingDirectionalTree<5> mapping;
  for (const float3 p : {near, far}) {
    const GuidingSphericalGaussian distribution{fit.direction(p), fit.concentration(p)};
    constexpr int resolution = 512, samples = 50000;
    double integral = 0;
    for (int y = 0; y < resolution; ++y) {
      for (int x = 0; x < resolution; ++x) {
        const float3 d = mapping.square_to_direction(
            make_float2((x + .5f) / resolution, (y + .5f) / resolution));
        integral += distribution.pdf(d) * (4 * M_PI / (resolution * resolution));
      }
    }
    EXPECT_NEAR(integral, 1, 2e-3);
    double sum = 0, squares = 0;
    for (int i = 0; i < samples; ++i) {
      const auto random = [&]() { return (float(rng() >> 9) + .5f) * 0x1p-23f; };
      float pdf;
      const float3 d = distribution.sample(make_float2(random(), random()), &pdf);
      ASSERT_GT(pdf, 0);
      EXPECT_NEAR(pdf, distribution.pdf(d), 2e-5f * pdf);
      const float cosine = dot(d, distribution.axis);
      sum += cosine;
      squares += cosine * cosine;
    }
    const double k = distribution.concentration;
    const double expected = 1 / std::tanh(k) - 1 / k;
    EXPECT_NEAR(sum / samples,
                expected,
                6 * std::sqrt((squares / samples - sqr(sum / samples)) / samples) + 1e-5);
  }
}

TEST(GuidingParallax, NearSourceNonlinearDirection)
{
  GuidingParallaxMoments parallax;
  GuidingPositionMoments linear;
  std::array<float, GuidingParallaxMoments::storage_size> source_moments{};
  std::array<float, GuidingPositionMoments::storage_size> linear_moments{};
  const float3 target = make_float3(.1f, -.1f, .25f);
  for (int y = 0; y < 64; ++y) {
    for (int x = 0; x < 64; ++x) {
      const float3 p = make_float3((x + .5f) / 64 - .5f, (y + .5f) / 64 - .5f, 0);
      const float3 d = normalize(target - p);
      parallax.record(source_moments.data(), 1, p, d, len(target - p));
      linear.record(linear_moments.data(), 1, p, d);
    }
  }
  const auto physical = parallax.fit(source_moments.data(), 4096, 4096, make_float3(0, 0, 1));
  const auto regression = linear.fit(linear_moments.data(), 4096);
  ASSERT_TRUE(physical.valid);
  ASSERT_TRUE(regression.valid);
  double physical_error = 0, linear_error = 0;
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      const float3 p = make_float3((x + .25f) / 16 - .5f, (y + .25f) / 16 - .5f, 0);
      const float3 expected = normalize(target - p);
      physical_error += len_squared(physical.direction(p) - expected);
      linear_error += len_squared(regression.direction(p) - expected);
    }
  }
  EXPECT_LT(physical_error / 256, 1e-8);
  EXPECT_GT(linear_error / 256, 1e-3);
}

TEST(GuidingParallax, MultipleEndpointsPreserveSourceVariance)
{
  GuidingParallaxMoments model;
  std::array<float, GuidingParallaxMoments::storage_size> moments{};
  for (int i = 0; i < 64; ++i) {
    /* One complete observation: weight 1 at distance 1, weight 3 at distance 3. */
    model.record_aggregate(moments.data(), 4, 2, 10, zero_float3(), make_float3(0, 0, 1));
  }
  const auto fit = model.fit(moments.data(), 256, 64, make_float3(0, 0, 1));
  ASSERT_TRUE(fit.valid);
  EXPECT_NEAR(fit.target.z, 2, 1e-6f);
  EXPECT_NEAR(fit.covariance[5], 1, 1e-6f);
  EXPECT_LT(fit.concentration(make_float3(1, 0, 0)), 100);
}

TEST(GuidingParallax, FieldPublicationUsesWorldMetricAndPreservesSupport)
{
  constexpr uint capacity = 3;
  std::array<GuidingSpatialNode, capacity> nodes{};
  std::vector<float> accumulation(capacity * GUIDING_FIELD_TYPES *
                                  GuidingField::accumulation_size);
  std::vector<float> sampling(capacity * GUIDING_FIELD_TYPES * GuidingField::sampling_size);
  uint counts[2] = {1, 1};
  GuidingField field{nodes.data(),
                     accumulation.data(),
                     sampling.data(),
                     counts,
                     capacity,
                     make_float3(-4, -2, -1),
                     make_float3(4, 2, 1)};
  const GuidingFieldType types[] = {GUIDING_FIELD_SURFACE_RADIANCE,
                                    GUIDING_FIELD_VOLUME_RADIANCE,
                                    GUIDING_FIELD_SURFACE_IMPORTANCE,
                                    GUIDING_FIELD_VOLUME_IMPORTANCE};
  for (int y = 0; y < 64; ++y) {
    for (int x = 0; x < 64; ++x) {
      const float3 p = make_float3((x + .5f) / 64 - .5f, (y + .5f) / 64 - .5f, 0);
      const float3 target = make_float3(1, 1, 2);
      const float3 d = normalize(target - p);
      for (const auto type : types) {
        field.record(
            field.record_index(0, type, d), 1, d, field.normalized_position(p), len(target - p));
      }
      field.record_visit(0);
    }
  }
  GuidingGaussianMixture mixture;
  int dominant = 0;
  for (const auto type : types) {
    field.publish(type, .05f);
    const float *storage = field.distribution(0, type) + field.tree_size;
    for (int i = 0; i < mixture.components; ++i) {
      if (storage[i * mixture.component_stride] > storage[dominant * mixture.component_stride]) {
        dominant = i;
      }
    }
    ASSERT_EQ(storage[dominant * mixture.component_stride + 17], 2);
    for (const float3 p : {make_float3(-.3f, -.2f, 0), make_float3(.3f, .2f, 0)}) {
      const auto component = mixture.component(storage, dominant, field.normalized_position(p));
      EXPECT_LT(len(component.axis - normalize(make_float3(1, 1, 2) - p)), 1e-4f);
      EXPECT_GT(component.concentration, 1000);
    }
  }
  const uint offset = field.parallax_offset + dominant * GuidingParallaxMoments::storage_size;
  const float w = accumulation[offset], w2 = accumulation[offset + 1];
  const float h = accumulation[offset + 2], h2 = accumulation[offset + 3];
  float total_observations = 0;
  for (int i = 0; i < mixture.components; ++i) {
    total_observations += accumulation[field.parallax_offset +
                                      i * GuidingParallaxMoments::storage_size];
  }
  /* Publication retains quarter-weight history without changing effective support. */
  EXPECT_NEAR(total_observations / .25f, 4096, 1e-3f);
  EXPECT_NEAR(w * w / w2, w / .25f, 1e-3f);
  field.begin_update();
  field.refine(0, 64);
  ASSERT_EQ(counts[0], 3u);
  for (uint child = 1; child < 3; ++child) {
    const float *data = accumulation.data() +
                        child * GUIDING_FIELD_TYPES * field.accumulation_size + offset;
    EXPECT_NEAR(data[0] * data[0] / data[1], w * w / w2, 1e-3f);
    EXPECT_NEAR(data[2] * data[2] / data[3], h * h / h2, 1e-3f);
  }
}

TEST(GuidingParallax, LargeFiniteRadianceKeepsFallbackNormalized)
{
  const float3 axis = normalize(make_float3(1, 2, 3));
  std::array<GuidingSpatialNode, 1> nodes{};
  std::vector<float> accumulation(GUIDING_FIELD_TYPES * GuidingField::accumulation_size);
  std::vector<float> sampling(GUIDING_FIELD_TYPES * GuidingField::sampling_size);
  uint counts[2] = {1, 1};
  GuidingField field{
      nodes.data(), accumulation.data(), sampling.data(), counts, 1, -one_float3(), one_float3()};
  for (int i = 0; i < 64; ++i) {
    field.record(field.record_index(0, GUIDING_FIELD_SURFACE_RADIANCE, axis), 1e20f / 64, axis);
  }
  field.publish(0, .05f);
  const float *storage = field.distribution(0, GUIDING_FIELD_SURFACE_RADIANCE) + field.tree_size;
  GuidingGaussianMixture mixture;
  for (int i = 0; i < mixture.components; ++i) {
    if (storage[i * mixture.component_stride] > 0) {
      const auto component = mixture.component(storage, i);
      EXPECT_NEAR(len(component.axis), 1, 1e-6f);
      if (storage[i * mixture.component_stride] > .5f) {
        EXPECT_LT(len(component.axis - axis), 1e-5f);
        EXPECT_GT(component.pdf(axis), 1.0f);
      }
    }
  }
}

CCL_NAMESPACE_END
