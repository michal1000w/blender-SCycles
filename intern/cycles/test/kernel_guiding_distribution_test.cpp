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
#include "kernel/sample/guiding_position.h"
#include "kernel/sample/guiding_spherical_gaussian.h"

CCL_NAMESPACE_BEGIN

using DirectionalTree = GuidingDirectionalTree<5>;
using TreeStorage = std::array<float, DirectionalTree::node_count>;

static float random_open(std::mt19937 &rng)
{
  return (float(rng() >> 9) + 0.5f) * 0x1p-23f;
}

TEST(GuidingPosition, PlanarParallaxAndDegenerateSupport)
{
  std::array<float, GuidingPositionMoments::storage_size> moments{};
  GuidingPositionMoments model;
  for (int y = 0; y < 64; ++y) {
    for (int x = 0; x < 64; ++x) {
      const float3 position = make_float3((x + 0.5f) / 64 - 0.5f, (y + 0.5f) / 64 - 0.5f, 0.0f);
      model.record(moments.data(), 1.0f, position, normalize(make_float3(0, 0, 2) - position));
    }
  }
  const GuidingPositionFit fit = model.fit(moments.data(), 4096);
  ASSERT_TRUE(fit.valid);
  EXPECT_LT(fit.residual_variance, 0.001f);
  for (const float3 position : {make_float3(0.3f, 0.2f, 0), make_float3(-0.4f, 0.1f, 0)}) {
    const float3 expected = normalize(make_float3(0, 0, 2) - position);
    EXPECT_LT(len(fit.direction(position) - expected), 0.003f);
    EXPECT_GT(len(normalize(fit.mean_direction) - expected), 0.1f);
  }
  moments.fill(0);
  for (int i = 0; i < 128; ++i) {
    model.record(moments.data(), 1.0f, make_float3(0.2f, 0.3f, 0.4f), make_float3(0, 0, 1));
  }
  const GuidingPositionFit point = model.fit(moments.data(), 128);
  ASSERT_TRUE(point.valid);
  EXPECT_NEAR(point.direction(make_float3(0.2f, 0.3f, 0.4f)).z, 1.0f, 1e-6f);
  const float3 distant = fit.direction(make_float3(1e30f, 1e30f, 1e30f));
  EXPECT_TRUE(isfinite_safe(distant));
  EXPECT_NEAR(len(distant), 1.0f, 1e-6f);
  moments.fill(0);
  for (int i = 0; i < 128; ++i) {
    model.record(moments.data(), 1.0f, zero_float3(), make_float3(0, 0, i % 2 ? 1 : -1));
  }
  const GuidingPositionFit isotropic = model.fit(moments.data(), 128);
  ASSERT_TRUE(isotropic.valid);
  EXPECT_NEAR(len(isotropic.direction(zero_float3())), 1.0f, 1e-6f);
}

TEST(GuidingPosition, RejectsUnsupportedAndDominatedFits)
{
  std::array<float, GuidingPositionMoments::storage_size> moments{};
  GuidingPositionMoments model;
  EXPECT_FALSE(model.fit(moments.data(), 1000).valid);
  for (int i = 0; i < 256; ++i) {
    model.record(moments.data(),
                 i == 0 ? 100000.0f : 1.0f,
                 make_float3(float(i) / 256, 0, 0),
                 make_float3(0, 0, 1));
  }
  EXPECT_FALSE(model.fit(moments.data(), 256).valid);
  moments[22] = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(model.fit(moments.data(), 256).valid);
}

TEST(GuidingPosition, SceneTranslationAndScale)
{
  GuidingPositionMoments model;
  GuidingPositionFit reference{};
  for (const float scale : {1.0f, 0.001f, 1000.0f}) {
    const float3 origin = scale == 1.0f ? zero_float3() : scale * make_float3(37, -19, 53);
    GuidingField field{};
    field.bounds_min = origin - make_float3(scale);
    field.bounds_max = origin + make_float3(scale);
    std::array<float, GuidingPositionMoments::storage_size> moments{};
    for (int y = 0; y < 64; ++y) {
      for (int x = 0; x < 64; ++x) {
        const float3 p = make_float3((x + .5f) / 64 - .5f, (y + .5f) / 64 - .5f, 0);
        const float3 world_position = origin + scale * p;
        const float3 world_light = origin + scale * make_float3(1, 1, 2);
        model.record(moments.data(),
                     1,
                     field.normalized_position(world_position),
                     normalize(world_light - world_position));
      }
    }
    const GuidingPositionFit fit = model.fit(moments.data(), 4096);
    ASSERT_TRUE(fit.valid);
    if (scale == 1.0f) {
      reference = fit;
    }
    for (const float3 p : {make_float3(-.3f, .2f, 0), make_float3(.4f, -.1f, 0)}) {
      const float3 expected = reference.direction((p + one_float3()) * .5f);
      const float3 actual = fit.direction(field.normalized_position(origin + scale * p));
      EXPECT_LT(len(actual - expected), 1e-4f);
    }
    EXPECT_NEAR(fit.residual_variance, reference.residual_variance, 1e-5f);
  }
}

TEST(GuidingPosition, PublishedMixtureNormalizationSamplingAndInheritance)
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
                     make_float3(-1, -1, -1),
                     make_float3(1, 1, 1)};
  std::mt19937 rng(867);
  for (int y = 0; y < 64; ++y) {
    for (int x = 0; x < 64; ++x) {
      const float3 p = make_float3((x + .5f) / 64 - .5f, (y + .5f) / 64 - .5f, 0);
      const float3 light = make_float3(
          1 + .3f * (random_open(rng) - .5f), 1 + .3f * (random_open(rng) - .5f), 2);
      const float3 d = normalize(light - p);
      field.record(field.record_index(0, GUIDING_FIELD_SURFACE_RADIANCE, d),
                   1,
                   d,
                   field.normalized_position(p));
      field.record_visit(0);
    }
  }
  field.publish(0, .05f);
  const float *storage = field.distribution(0, GUIDING_FIELD_SURFACE_RADIANCE) +
                         GuidingField::tree_size;
  int dominant = 0;
  for (int i = 1; i < GuidingGaussianMixture::components; ++i) {
    if (storage[i * GuidingGaussianMixture::component_stride] >
        storage[dominant * GuidingGaussianMixture::component_stride])
    {
      dominant = i;
    }
  }
  ASSERT_EQ(storage[dominant * GuidingGaussianMixture::component_stride + 17], 1.0f);
  GuidingGaussianMixture mixture;
  const float3 low = field.normalized_position(make_float3(-.3f, -.2f, 0));
  const float3 high = field.normalized_position(make_float3(.3f, .2f, 0));
  EXPECT_GT(len(mixture.component(storage, dominant, low).axis -
                mixture.component(storage, dominant, high).axis),
            .1f);
  const GuidingGaussianProduct profile{{{make_float3(0, 0, 1), 3}, {make_float3(0, 0, 1), 0}},
                                       {1, 0}};
  DirectionalTree mapping;
  for (const float3 position : {low, high}) {
    double mass = 0, product_mass = 0, expected_z = 0;
    constexpr int resolution = 512;
    const double cell_area = 4 * M_PI / (resolution * resolution);
    for (int y = 0; y < resolution; ++y) {
      for (int x = 0; x < resolution; ++x) {
        const float3 d = mapping.square_to_direction(
            make_float2((x + .5f) / resolution, (y + .5f) / resolution));
        mass += mixture.pdf(storage, d, position);
        const float pdf = mixture.pdf_product(storage, profile, d, position);
        product_mass += pdf;
        expected_z += pdf * d.z;
      }
    }
    EXPECT_NEAR(mass * cell_area, 1.0, 2e-3);
    EXPECT_NEAR(product_mass * cell_area, 1.0, 2e-3);
    constexpr int samples = 30000;
    double z = 0, z2 = 0;
    for (int i = 0; i < samples; ++i) {
      float pdf;
      const float3 d = mixture.sample_product(
          storage, profile, make_float2(random_open(rng), random_open(rng)), &pdf, position);
      ASSERT_GT(pdf, 0);
      EXPECT_NEAR(pdf, mixture.pdf_product(storage, profile, d, position), 2e-5f * pdf);
      z += d.z;
      z2 += d.z * d.z;
    }
    const double variance = z2 / samples - sqr(z / samples);
    EXPECT_NEAR(z / samples, expected_z * cell_area, 6 * std::sqrt(variance / samples) + 1e-4);
  }
  const uint offset = GuidingField::moments_offset +
                      dominant * GuidingPositionMoments::storage_size;
  const float effective_before = sqr(accumulation[offset] / sqrtf(accumulation[offset + 22]));
  /* The angular partition can put these unit-weight observations in several components. */
  float total_observations = 0;
  for (int i = 0; i < GuidingGaussianMixture::components; ++i) {
    const uint component_offset = GuidingField::moments_offset +
                                  i * GuidingPositionMoments::storage_size;
    total_observations += accumulation[component_offset];
  }
  /* Publication retains quarter-weight history without changing effective support. */
  EXPECT_NEAR(total_observations / .25f, 4096, 1e-3f);
  EXPECT_NEAR(effective_before, accumulation[offset] / .25f, 1e-3f);
  field.begin_update();
  field.refine(0, 64);
  ASSERT_EQ(counts[0], 3u);
  for (uint child = 1; child < 3; ++child) {
    const uint child_offset = child * GUIDING_FIELD_TYPES * GuidingField::accumulation_size +
                              offset;
    EXPECT_NEAR(sqr(accumulation[child_offset] / sqrtf(accumulation[child_offset + 22])),
                effective_before,
                1e-3f);
  }
}

TEST(GuidingSphericalGaussian, NormalizationAndProduct)
{
  const GuidingSphericalGaussian a{normalize(make_float3(1, 2, 3)), 5.0f};
  const GuidingSphericalGaussian b{normalize(make_float3(-2, 1, 3)), 12.0f};
  float product_integral;
  const GuidingSphericalGaussian product = a.product(b, &product_integral);
  double integral_a = 0, integral_b = 0, integral_product = 0;
  const DirectionalTree mapping;
  constexpr int resolution = 512;
  for (int y = 0; y < resolution; ++y) {
    for (int x = 0; x < resolution; ++x) {
      const float3 direction = mapping.square_to_direction(
          make_float2((x + 0.5f) / resolution, (y + 0.5f) / resolution));
      const float pa = a.pdf(direction), pb = b.pdf(direction);
      integral_a += pa;
      integral_b += pb;
      integral_product += double(pa) * pb;
      EXPECT_NEAR(
          pa * pb, product_integral * product.pdf(direction), 1e-5f * max(pa * pb, 1e-20f));
    }
  }
  const double cell_area = 4.0 * M_PI / (resolution * resolution);
  EXPECT_NEAR(integral_a * cell_area, 1.0, 1e-4);
  EXPECT_NEAR(integral_b * cell_area, 1.0, 1e-4);
  EXPECT_NEAR(integral_product * cell_area, product_integral, 1e-4);
}

TEST(GuidingSphericalGaussian, SamplingMoments)
{
  std::mt19937 rng(3829);
  const float3 axis = normalize(make_float3(1, -2, 3));
  constexpr int samples = 100000;
  for (const float concentration : {0.0f, 1e-5f, 0.01f, 1.0f, 32.0f, 16384.0f}) {
    const GuidingSphericalGaussian distribution{axis, concentration};
    double mean_cosine = 0;
    for (int i = 0; i < samples; ++i) {
      float pdf;
      const float3 direction = distribution.sample(make_float2(random_open(rng), random_open(rng)),
                                                   &pdf);
      ASSERT_TRUE(isfinite_safe(direction));
      ASSERT_GT(pdf, 0.0f);
      EXPECT_NEAR(len(direction), 1.0f, 2e-6f);
      mean_cosine += dot(axis, direction);
    }
    const double expected = concentration < 1e-4f ? double(concentration) / 3.0 :
                                                    1.0 / std::tanh(double(concentration)) -
                                                        1.0 / double(concentration);
    EXPECT_NEAR(
        mean_cosine / samples, expected, 6.0 * std::sqrt((1.0 - expected * expected) / samples));
  }
}

TEST(GuidingSphericalGaussian, FittedProductMixture)
{
  TreeStorage tree{};
  for (int i = 0; i < DirectionalTree::leaf_count; ++i) {
    tree[DirectionalTree::leaf_offset + i] = 1.0f + float(i % 17);
  }
  DirectionalTree directional;
  directional.build(tree.data(), 0.05f);
  std::array<float, GuidingGaussianMixture::storage_size> storage{};
  GuidingGaussianMixture mixture;
  mixture.build(storage.data(), tree.data());
  float mass = 0.0f;
  for (int i = 0; i < GuidingGaussianMixture::components; ++i) {
    mass += storage[GuidingGaussianMixture::component_stride * i];
    EXPECT_NEAR(len(mixture.component(storage.data(), i).axis), 1.0f, 1e-6f);
  }
  EXPECT_NEAR(mass, 1.0f, 1e-6f);
  const GuidingGaussianProduct profile{
      {{normalize(make_float3(1, 2, 3)), 5.0f}, {normalize(make_float3(-2, 1, -3)), 30.0f}},
      {0.3f, 0.7f}};
  constexpr int resolution = 256;
  double integral = 0.0, expected_z = 0.0;
  for (int y = 0; y < resolution; ++y) {
    for (int x = 0; x < resolution; ++x) {
      const float3 direction = directional.square_to_direction(
          make_float2((x + 0.5f) / resolution, (y + 0.5f) / resolution));
      const float pdf = mixture.pdf_product(storage.data(), profile, direction);
      integral += pdf;
      expected_z += pdf * direction.z;
    }
  }
  const double cell_area = 4.0 * M_PI / (resolution * resolution);
  EXPECT_NEAR(integral * cell_area, 1.0, 5e-4);
  std::mt19937 rng(17129);
  constexpr int samples = 100000;
  double z = 0.0, z2 = 0.0;
  for (int i = 0; i < samples; ++i) {
    float pdf;
    const float2 random = make_float2(random_open(rng), random_open(rng));
    const float3 direction = mixture.sample_product(storage.data(), profile, random, &pdf);
    if (i < 1024) {
      const float3 other = directional.square_to_direction(random);
      float fused_pdf, other_pdf, other_incident, incident;
      const float3 fused = mixture.sample_product(storage.data(),
                                                  profile,
                                                  random,
                                                  &fused_pdf,
                                                  zero_float3(),
                                                  &other,
                                                  &other_pdf,
                                                  &other_incident,
                                                  &incident);
      EXPECT_EQ(direction.x, fused.x);
      EXPECT_EQ(direction.y, fused.y);
      EXPECT_EQ(direction.z, fused.z);
      EXPECT_EQ(pdf, fused_pdf);
      EXPECT_NEAR(other_pdf, mixture.pdf_product(storage.data(), profile, other), 1e-6f);
      EXPECT_NEAR(other_incident, mixture.pdf(storage.data(), other), 1e-6f);
      EXPECT_NEAR(incident, mixture.pdf(storage.data(), direction), 1e-6f);
    }
    ASSERT_TRUE(isfinite_safe(direction));
    ASSERT_GT(pdf, 0.0f);
    EXPECT_NEAR(len(direction), 1.0f, 2e-6f);
    z += direction.z;
    z2 += direction.z * direction.z;
  }
  const double variance = z2 / samples - sqr(z / samples);
  EXPECT_NEAR(z / samples, expected_z * cell_area, 6.0 * std::sqrt(variance / samples));
}

TEST(GuidingSphericalGaussian, ProductDensityIndependentLobeSum)
{
  std::array<float, GuidingGaussianMixture::storage_size> storage{};
  GuidingGaussianMixture mixture;
  std::mt19937 rng(77129);
  for (const float concentration : {0.0f, 1.0f, 32.0f, 512.0f, 16384.0f}) {
    for (int i = 0; i < GuidingGaussianMixture::components; ++i) {
      float *entry = storage.data() + i * GuidingGaussianMixture::component_stride;
      const float3 axis = normalize(make_float3(float(i % 3) - 1, float(i % 5) - 2, 3));
      entry[0] = 1.0f / GuidingGaussianMixture::components;
      entry[1] = concentration;
      entry[2] = axis.x;
      entry[3] = axis.y;
      entry[4] = axis.z;
    }
    const GuidingGaussianProduct profile{{{normalize(make_float3(1, 2, 3)), concentration},
                                          {normalize(make_float3(-2, 1, 3)), 5.0f}},
                                         {.3f, .7f}};
    for (int sample = 0; sample < 2048; ++sample) {
      float pdf;
      const float3 direction = mixture.sample_product(
          storage.data(), profile, make_float2(random_open(rng), random_open(rng)), &pdf);
      float mass = 0, density = 0;
      for (int i = 0; i < GuidingGaussianMixture::components; ++i) {
        const auto illumination = mixture.component(storage.data(), i);
        for (int j = 0; j < 2; ++j) {
          float integral;
          const auto product = illumination.product(profile.lobes[j], &integral);
          const float weight = storage[i * GuidingGaussianMixture::component_stride] *
                               profile.weights[j] * integral;
          mass += weight;
          density += weight * product.pdf(direction);
        }
      }
      const float reference = mass > 0 ? .95f * density / mass + .05f * M_1_4PI_F : M_1_4PI_F;
      ASSERT_NEAR(pdf, reference, 2e-5f * max(reference, 1.0f))
          << "concentration=" << concentration << " sample=" << sample;
    }
  }
}

TEST(GuidingDistribution, EqualAreaMapping)
{
  const DirectionalTree distribution;
  for (int y = 0; y < 128; ++y) {
    for (int x = 0; x < 128; ++x) {
      const float2 uv = make_float2((x + 0.5f) / 128, (y + 0.5f) / 128);
      const float3 direction = distribution.square_to_direction(uv);
      const float2 restored = distribution.direction_to_square(direction);
      EXPECT_NEAR(len(direction), 1.0f, 2e-7f);
      EXPECT_NEAR(restored.x, uv.x, 2e-7f);
      EXPECT_NEAR(restored.y, uv.y, 2e-7f);
    }
  }
}

TEST(GuidingDistribution, EmptyAndInvalidTraining)
{
  DirectionalTree distribution;
  TreeStorage tree{};
  tree[DirectionalTree::leaf_offset] = std::numeric_limits<float>::infinity();
  tree[DirectionalTree::leaf_offset + 1] = std::numeric_limits<float>::quiet_NaN();
  tree[DirectionalTree::leaf_offset + 2] = -1.0f;
  distribution.build(tree.data(), 0.05f);
  for (const float value : tree) {
    EXPECT_EQ(value, 0.0f);
  }
  for (const float u : {0.0f, 0.3f, 1.0f}) {
    float pdf;
    const float3 direction = distribution.sample(tree.data(), make_float2(u, 0.4f), &pdf);
    EXPECT_NEAR(len(direction), 1.0f, 2e-7f);
    EXPECT_EQ(pdf, M_1_4PI_F);
    EXPECT_EQ(distribution.pdf(tree.data(), direction), pdf);
  }
}

TEST(GuidingDistribution, NormalizationAndExploration)
{
  DirectionalTree distribution;
  TreeStorage tree{};
  tree[DirectionalTree::leaf_offset + 147] = std::numeric_limits<float>::max();
  distribution.build(tree.data(), 0.05f);
  double integral = 0.0;
  for (int y = 0; y < DirectionalTree::resolution; ++y) {
    for (int x = 0; x < DirectionalTree::resolution; ++x) {
      const float3 direction = distribution.square_to_direction(make_float2(
          (x + 0.5f) / DirectionalTree::resolution, (y + 0.5f) / DirectionalTree::resolution));
      const float pdf = distribution.pdf(tree.data(), direction);
      EXPECT_GE(pdf, 0.05f * M_1_4PI_F * (1.0f - 1e-6f));
      integral += pdf * (4.0 * M_PI / DirectionalTree::leaf_count);
    }
  }
  EXPECT_NEAR(integral, 1.0, 2e-6);
}

TEST(GuidingDistribution, SamplePdfAndFrequencies)
{
  DirectionalTree distribution;
  TreeStorage tree{};
  for (int i = 0; i < DirectionalTree::leaf_count; ++i) {
    tree[DirectionalTree::leaf_offset + i] = 1.0f + (i % 11);
  }
  distribution.build(tree.data(), 0.05f);
  std::mt19937 rng(171);
  std::array<int, DirectionalTree::leaf_count> histogram{};
  constexpr int samples = 1000000;
  double integral_z2 = 0.0;
  for (int i = 0; i < samples; ++i) {
    float pdf;
    const float2 random = make_float2(random_open(rng), random_open(rng));
    const float3 direction = distribution.sample(tree.data(), random, &pdf);
    ASSERT_GT(pdf, 0.0f);
    const float evaluated = distribution.pdf(tree.data(), direction);
    ASSERT_NEAR(pdf, evaluated, 2e-6f);
    ++histogram[distribution.leaf_index(direction) - DirectionalTree::leaf_offset];
    integral_z2 += double(direction.z * direction.z) / pdf;
  }
  double chi_squared = 0.0;
  for (int i = 0; i < DirectionalTree::leaf_count; ++i) {
    const double expected = samples * double(tree[DirectionalTree::leaf_offset + i]) / tree[0];
    chi_squared += (histogram[i] - expected) * (histogram[i] - expected) / expected;
  }
  EXPECT_LT(chi_squared, 1250.0);
  EXPECT_NEAR(integral_z2 / samples, 4.0 * M_PI / 3.0, 0.02);
}

TEST(GuidingDistribution, CosineProductNormalizationAndSampling)
{
  DirectionalTree distribution;
  TreeStorage tree{};
  for (int i = 0; i < DirectionalTree::leaf_count; ++i) {
    tree[DirectionalTree::leaf_offset + i] = 1.0f + i % 11;
  }
  distribution.build(tree.data(), 0.05f);
  std::mt19937 rng(173);
  for (const float3 normal : {make_float3(1, 0, 0),
                              make_float3(0, 1, 0),
                              make_float3(0, 0, 1),
                              normalize(make_float3(1, 2, -3))})
  {
    for (const auto type : {GuidingDirectionalProduct::COSINE,
                            GuidingDirectionalProduct::TWO_SIDED_COSINE,
                            GuidingDirectionalProduct::PHASE})
    {
      const GuidingDirectionalProduct product{normal, 0.65f, type};
      std::array<double, DirectionalTree::leaf_count> probabilities{};
      double sum = 0;
      for (int y = 0; y < DirectionalTree::resolution; ++y) {
        for (int x = 0; x < DirectionalTree::resolution; ++x) {
          const float3 direction = distribution.square_to_direction(make_float2(
              (x + 0.5f) / DirectionalTree::resolution, (y + 0.5f) / DirectionalTree::resolution));
          const double probability = distribution.pdf_product(tree.data(), direction, product) *
                                     (4.0 * M_PI / DirectionalTree::leaf_count);
          ASSERT_GT(probability, 0.0);
          probabilities[distribution.leaf_index(direction) - DirectionalTree::leaf_offset] =
              probability;
          sum += probability;
        }
      }
      EXPECT_NEAR(sum, 1.0, 2e-6);
      constexpr int samples = 200000;
      std::array<int, DirectionalTree::leaf_count> histogram{};
      for (int i = 0; i < samples; ++i) {
        float pdf;
        const float3 direction = distribution.sample_product(
            tree.data(), make_float2(random_open(rng), random_open(rng)), product, &pdf);
        ASSERT_GT(pdf, 0.0f);
        ASSERT_NEAR(
            pdf, distribution.pdf_product(tree.data(), direction, product), 2e-5f * max(pdf, 1.0f))
            << "sample=" << i << " normal=" << normal.x << "," << normal.y << "," << normal.z
            << " product=" << type << " direction=" << direction.x << "," << direction.y << ","
            << direction.z;
        ++histogram[distribution.leaf_index(direction) - DirectionalTree::leaf_offset];
      }
      double chi_squared = 0;
      for (int i = 0; i < DirectionalTree::leaf_count; ++i) {
        const double expected = probabilities[i] * samples;
        chi_squared += (histogram[i] - expected) * (histogram[i] - expected) / expected;
      }
      EXPECT_LT(chi_squared, 1250.0);
    }
  }
}

TEST(GuidingDistribution, SparseWithoutExploration)
{
  DirectionalTree distribution;
  for (const int leaf : {0, 3, 144, 1023}) {
    TreeStorage tree{};
    tree[DirectionalTree::leaf_offset + leaf] = 1.0f;
    distribution.build(tree.data(), 0.0f);
    for (const float u : {0.0f, 0.25f, 0.99f, 1.0f}) {
      float pdf;
      const float3 direction = distribution.sample(tree.data(), make_float2(u, 0.5f), &pdf);
      EXPECT_TRUE(isfinite_safe(direction));
      EXPECT_GT(pdf, 0.0f);
      /* Zero/one inputs exercise the finite-precision edges of the CDF; avoid evaluating the
       * spherical seam itself, where two adjacent representations describe the same ray. */
      if (u > 0.0f && u < 0.99f) {
        EXPECT_EQ(distribution.leaf_index(direction), DirectionalTree::leaf_offset + leaf);
      }
    }
  }
}

TEST(GuidingDistribution, AdaptiveResolutionPreservesMassAndCountLeaves)
{
  DirectionalTree distribution;
  std::array<float, DirectionalTree::node_count> tree{};
  std::array<float, DirectionalTree::node_count> counts{};
  constexpr int first_leaf = DirectionalTree::leaf_offset;
  tree[first_leaf] = 1.0f;
  counts[first_leaf] = 1.0f;
  distribution.build_adaptive(tree.data(), counts.data(), 32.0f, 0.0f);
  /* One observation cannot establish a sharp angular distribution. */
  for (int i = first_leaf; i < DirectionalTree::node_count; ++i) {
    EXPECT_FLOAT_EQ(tree[i] / tree[0], 1.0f / DirectionalTree::leaf_count);
  }
  EXPECT_EQ(counts[first_leaf], 1.0f);

  tree.fill(0.0f);
  counts.fill(0.0f);
  constexpr int quarter = DirectionalTree::leaf_count / 4;
  tree[first_leaf] = 1.0f;
  tree[first_leaf + quarter] = 3.0f;
  counts[first_leaf] = 64.0f;
  counts[first_leaf + quarter] = 1.0f;
  distribution.build_adaptive(tree.data(), counts.data(), 32.0f, 0.0f);
  /* The well-observed lobe keeps full resolution. The sparse lobe spreads uniformly within
   * its quadrant, retaining the original 1:3 integrated mass ratio between the quadrants. */
  EXPECT_FLOAT_EQ(tree[1] / tree[0], 0.25f);
  EXPECT_FLOAT_EQ(tree[2] / tree[0], 0.75f);
  EXPECT_FLOAT_EQ(tree[first_leaf] / tree[0], 0.25f);
  for (int i = first_leaf + quarter; i < first_leaf + 2 * quarter; ++i) {
    EXPECT_FLOAT_EQ(tree[i] / tree[0], 0.75f / quarter);
  }
  EXPECT_EQ(counts[first_leaf], 64.0f);
  EXPECT_EQ(counts[first_leaf + quarter], 1.0f);
}

TEST(GuidingField, SurfaceOrientationConditioning)
{
  std::array<bool, GUIDING_FIELD_TYPES> seen{};
  for (int axis = 0; axis < 3; ++axis) {
    for (int sign = 0; sign < 2; ++sign) {
      float3 normal = make_float3(0.1f, 0.1f, 0.1f);
      normal[axis] = sign ? -1.0f : 1.0f;
      for (const bool importance : {false, true}) {
        const GuidingFieldType type = guiding_surface_field_type(normal, importance);
        EXPECT_GE(type, 0);
        EXPECT_LT(type, GUIDING_FIELD_TYPES);
        EXPECT_NE(type, GUIDING_FIELD_VOLUME_RADIANCE);
        EXPECT_NE(type, GUIDING_FIELD_VOLUME_IMPORTANCE);
        EXPECT_FALSE(seen[type]);
        seen[type] = true;
        EXPECT_EQ(type, guiding_surface_field_type(normal * 2.0f, importance));
      }
    }
  }
}

TEST(GuidingField, PublicationAndIndependentTargets)
{
  constexpr uint capacity = 7;
  std::array<GuidingSpatialNode, capacity> nodes{};
  std::vector<float> accumulation(capacity * GUIDING_FIELD_TYPES *
                                  GuidingField::accumulation_size);
  std::vector<float> sampling(capacity * GUIDING_FIELD_TYPES * GuidingField::sampling_size);
  std::array<uint, 2> counts = {1, 1};
  GuidingField field{nodes.data(),
                     accumulation.data(),
                     sampling.data(),
                     counts.data(),
                     capacity,
                     make_float3(-2, -1, -1),
                     make_float3(2, 1, 1)};
  const float3 north = normalize(make_float3(1, 1, 3));
  const float3 south = -north;
  for (int i = 0; i < 1024; ++i) {
    field.record(
        field.record_index(0, GUIDING_FIELD_SURFACE_RADIANCE, north), 10.0f / 1024, north);
    field.record(
        field.record_index(0, GUIDING_FIELD_SURFACE_IMPORTANCE, south), 20.0f / 1024, south);
  }
  EXPECT_EQ(field.distribution(0, GUIDING_FIELD_SURFACE_RADIANCE)[0], 0.0f);
  for (uint type = 0; type < GUIDING_FIELD_TYPES; ++type) {
    field.publish(type, 0.05f);
  }
  EXPECT_EQ(accumulation[field.record_index(0, GUIDING_FIELD_SURFACE_RADIANCE, north)], 2.5f);
  EXPECT_EQ(accumulation[field.record_index(0, GUIDING_FIELD_SURFACE_IMPORTANCE, south)], 5.0f);
  DirectionalTree directional;
  EXPECT_GT(directional.pdf(field.distribution(0, GUIDING_FIELD_SURFACE_RADIANCE), north),
            directional.pdf(field.distribution(0, GUIDING_FIELD_SURFACE_RADIANCE), south));
  EXPECT_LT(directional.pdf(field.distribution(0, GUIDING_FIELD_SURFACE_IMPORTANCE), north),
            directional.pdf(field.distribution(0, GUIDING_FIELD_SURFACE_IMPORTANCE), south));
  EXPECT_EQ(field.distribution(0, GUIDING_FIELD_VOLUME_RADIANCE)[0], 0.0f);
  for (int i = 0; i < 100; ++i) {
    field.record_visit(0);
  }
  field.begin_update();
  field.refine(0, 100);
  EXPECT_EQ(counts[0], 3u);
  EXPECT_EQ(field.find_leaf(make_float3(-1, 0, 0)), 1u);
  EXPECT_EQ(field.find_leaf(make_float3(1, 0, 0)), 2u);
  EXPECT_EQ(nodes[1].visits, 50u);
  /* A freshly created child must be queryable before it has collected a sample.
   * This includes the smooth mixture, not just the histogram used to seed it. */
  GuidingGaussianMixture mixture;
  for (uint child : {1u, 2u}) {
    for (auto type : {GUIDING_FIELD_SURFACE_RADIANCE, GUIDING_FIELD_SURFACE_IMPORTANCE}) {
      for (const float3 direction : {north, south, make_float3(0, 1, 0)}) {
        const float *parent = field.distribution(0, type);
        const float *inherited = field.distribution(child, type);
        EXPECT_EQ(directional.pdf(inherited, direction), directional.pdf(parent, direction));
        EXPECT_EQ(mixture.pdf(inherited + field.tree_size, direction),
                  mixture.pdf(parent + field.tree_size, direction));
      }
    }
  }
  for (uint i = 0; i < GUIDING_FIELD_TYPES * GuidingField::accumulation_size; ++i) {
    const uint offset = i % GuidingField::accumulation_size;
    const uint source_entry = (offset - GuidingField::parallax_offset) %
                              GuidingParallaxMoments::storage_size;
    const bool squared_weight = offset >= GuidingField::parallax_offset ?
                                    (source_entry == 1 || source_entry == 3) :
                                    offset >= GuidingField::moments_offset &&
                                        (offset - GuidingField::moments_offset) %
                                                GuidingPositionMoments::storage_size ==
                                            22;
    /* Two half-weight inherited copies conserve linear moments. Their sum of squared
     * weights is half the parent's; this preserves each copy's effective sample size. */
    EXPECT_EQ(accumulation[GUIDING_FIELD_TYPES * GuidingField::accumulation_size + i] +
                  accumulation[2 * GUIDING_FIELD_TYPES * GuidingField::accumulation_size + i],
              (squared_weight ? 0.5f : 1.0f) * accumulation[i]);
  }
  /* A subsequent asymmetric update must specialize only the observed child's distribution. */
  for (int i = 0; i < 64; ++i) {
    field.record(field.record_index(1, GUIDING_FIELD_SURFACE_RADIANCE, south), 100.0f / 64, south);
  }
  for (uint index = 0; index < counts[0] * GUIDING_FIELD_TYPES; ++index) {
    field.publish(index, 0.05f);
  }
  EXPECT_GT(directional.pdf(field.distribution(1, GUIDING_FIELD_SURFACE_RADIANCE), south),
            directional.pdf(field.distribution(1, GUIDING_FIELD_SURFACE_RADIANCE), north));
  EXPECT_LT(directional.pdf(field.distribution(2, GUIDING_FIELD_SURFACE_RADIANCE), south),
            directional.pdf(field.distribution(2, GUIDING_FIELD_SURFACE_RADIANCE), north));
}

TEST(GuidingField, PreservesSubBinAngularObservations)
{
  std::array<GuidingSpatialNode, 1> nodes{};
  std::vector<float> accumulation(GUIDING_FIELD_TYPES * GuidingField::accumulation_size);
  std::vector<float> sampling(GUIDING_FIELD_TYPES * GuidingField::sampling_size);
  std::array<uint, 2> counts = {1, 1};
  GuidingField field{nodes.data(),
                     accumulation.data(),
                     sampling.data(),
                     counts.data(),
                     1,
                     make_float3(-1),
                     make_float3(1)};
  const float3 direction = normalize(make_float3(1, 0.3f, 2));
  const uint index = field.record_index(0, GUIDING_FIELD_SURFACE_RADIANCE, direction);
  for (int i = 0; i < 128; ++i) {
    field.record(index, 1.0f, direction);
  }
  field.publish(0, 0.05f);
  GuidingGaussianMixture smooth;
  const float *storage = sampling.data() + GuidingField::tree_size;
  const int component = index / (GuidingField::bins / GuidingGaussianMixture::components);
  const GuidingSphericalGaussian fitted = smooth.component(storage, component);
  EXPECT_GT(dot(fitted.axis, direction), 1.0f - 1e-6f);
  EXPECT_GT(fitted.concentration, 10000.0f);
  const GuidingGaussianProduct uniform{{{make_float3(0, 0, 1), 0}, {make_float3(0, 0, 1), 0}},
                                       {1, 0}};
  for (const float3 query : {direction, -direction, normalize(make_float3(2, -1, 0.3f))}) {
    const float pdf = smooth.pdf(storage, query);
    EXPECT_GT(pdf, 0.0f);
    EXPECT_NEAR(pdf, smooth.pdf_product(storage, uniform, query), 1e-4f * max(pdf, 1.0f));
  }
}

TEST(GuidingField, RefinementRespectsMemoryBudgetAndAncestry)
{
  constexpr uint capacity = 17;
  std::array<GuidingSpatialNode, capacity> nodes{};
  std::vector<float> accumulation(capacity * GUIDING_FIELD_TYPES *
                                  GuidingField::accumulation_size);
  std::vector<float> sampling(capacity * GUIDING_FIELD_TYPES * GuidingField::sampling_size);
  std::array<uint, 2> counts = {1, 1};
  GuidingField field{nodes.data(),
                     accumulation.data(),
                     sampling.data(),
                     counts.data(),
                     capacity,
                     make_float3(-4, -2, -1),
                     make_float3(4, 2, 1)};
  nodes[0].visits = 100000;
  for (int iteration = 0; iteration < 12; ++iteration) {
    field.begin_update();
    for (uint index = 0; index < capacity; ++index) {
      field.refine(index, 10);
    }
    EXPECT_LE(counts[0], capacity);
    for (int x = -20; x < 20; ++x) {
      const uint leaf = field.find_leaf(make_float3(x * 0.2f, 0.1f, 0.3f));
      ASSERT_LT(leaf, counts[0]);
      EXPECT_EQ(nodes[leaf].children, 0u);
    }
  }
  EXPECT_EQ(counts[0], capacity);
  EXPECT_EQ(nodes[0].axis, 0u);
  EXPECT_EQ(nodes[0].split, 0.0f);
  EXPECT_EQ(nodes[1].axis, 0u);
  EXPECT_EQ(nodes[1].split, -2.0f);
  EXPECT_EQ(nodes[2].split, 2.0f);
  EXPECT_EQ(nodes[3].axis, 1u);
  EXPECT_EQ(nodes[3].split, 0.0f);
}

TEST(GuidingField, CooperativeRefinementPreservesEveryInheritedValue)
{
  constexpr uint capacity = 3;
  std::array<GuidingSpatialNode, capacity> nodes{};
  std::array<GuidingSpatialNode, capacity> cooperative_nodes{};
  std::vector<float> accumulation(capacity * GUIDING_FIELD_TYPES *
                                  GuidingField::accumulation_size);
  std::vector<float> sampling(capacity * GUIDING_FIELD_TYPES * GuidingField::sampling_size);
  for (size_t i = 0; i < accumulation.size(); ++i) {
    accumulation[i] = float(i % 137) * 0.125f;
  }
  for (size_t i = 0; i < sampling.size(); ++i) {
    sampling[i] = float(i % 53) * 0.0625f;
  }
  auto cooperative_accumulation = accumulation;
  auto cooperative_sampling = sampling;
  std::array<uint, 2> counts = {1, 1}, cooperative_counts = counts;
  nodes[0].visits = cooperative_nodes[0].visits = 1024;
  GuidingField scalar{nodes.data(),
                      accumulation.data(),
                      sampling.data(),
                      counts.data(),
                      capacity,
                      make_float3(-1),
                      make_float3(1)};
  GuidingField cooperative{cooperative_nodes.data(),
                           cooperative_accumulation.data(),
                           cooperative_sampling.data(),
                           cooperative_counts.data(),
                           capacity,
                           make_float3(-1),
                           make_float3(1)};
  scalar.refine(0, 64);
  const uint first = cooperative.refine_allocate(0, 64);
  ASSERT_EQ(first, 1u);
  for (uint lane = 0; lane < 32; ++lane) {
    cooperative.refine_copy(0, first, lane, 32);
  }
  EXPECT_EQ(accumulation, cooperative_accumulation);
  EXPECT_EQ(sampling, cooperative_sampling);
  EXPECT_EQ(counts, cooperative_counts);
  for (uint i = 0; i < capacity; ++i) {
    EXPECT_EQ(nodes[i].children, cooperative_nodes[i].children);
    EXPECT_EQ(nodes[i].parent, cooperative_nodes[i].parent);
    EXPECT_EQ(nodes[i].axis, cooperative_nodes[i].axis);
    EXPECT_EQ(nodes[i].split, cooperative_nodes[i].split);
    EXPECT_EQ(nodes[i].visits, cooperative_nodes[i].visits);
  }
}

TEST(GuidingDistribution, SpatialSplitUsesObservedSurfaceSpread)
{
  constexpr uint capacity = 3;
  std::vector<GuidingSpatialNode> nodes(capacity);
  std::vector<float> accumulation(capacity * GUIDING_FIELD_TYPES *
                                  GuidingField::accumulation_size);
  std::vector<float> sampling(capacity * GUIDING_FIELD_TYPES * GuidingField::sampling_size);
  std::array<uint, 2> counts = {1, 1};
  GuidingField field{nodes.data(),
                     accumulation.data(),
                     sampling.data(),
                     counts.data(),
                     capacity,
                     make_float3(-10, -1, -1),
                     make_float3(10, 1, 1)};
  const float3 direction = make_float3(0, 0, 1);
  for (int i = 0; i < 128; ++i) {
    const float3 position = make_float3(1, i < 64 ? -.75f : .25f, .5f);
    field.record_visit(0);
    field.record(field.record_index(0, GUIDING_FIELD_SURFACE_RADIANCE, direction),
                 1,
                 direction,
                 field.normalized_position(position));
  }
  field.begin_update();
  field.refine(0, 64);
  ASSERT_EQ(counts[0], 3u);
  EXPECT_EQ(nodes[0].axis, 1u);
  EXPECT_NEAR(nodes[0].split, -.25f, 1e-6);
  EXPECT_NE(field.find_leaf(make_float3(1, -.75f, .5f)),
            field.find_leaf(make_float3(1, .25f, .5f)));
}

TEST(GuidingDistribution, SpatialSplitDoesNotFitOneBrightObservation)
{
  std::vector<GuidingSpatialNode> nodes(1);
  std::vector<float> accumulation(GUIDING_FIELD_TYPES * GuidingField::accumulation_size);
  std::vector<float> sampling(GUIDING_FIELD_TYPES * GuidingField::sampling_size);
  std::array<uint, 2> counts = {1, 1};
  GuidingField field{nodes.data(),
                     accumulation.data(),
                     sampling.data(),
                     counts.data(),
                     1,
                     make_float3(-10, -1, -1),
                     make_float3(10, 1, 1)};
  const float3 direction = make_float3(0, 0, 1);
  for (int i = 0; i < 128; ++i) {
    const float3 position = make_float3(1, i < 64 ? -.75f : .25f, .5f);
    field.record(field.record_index(0, GUIDING_FIELD_SURFACE_RADIANCE, direction),
                 i == 0 ? 1e6f : 1,
                 direction,
                 field.normalized_position(position));
  }
  uint axis;
  float split;
  field.spatial_split(0, field.bounds_min, field.bounds_max, &axis, &split);
  EXPECT_EQ(axis, 0u);
  EXPECT_EQ(split, 0);
}

CCL_NAMESPACE_END
