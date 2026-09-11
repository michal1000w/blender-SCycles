/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "kernel/sample/guiding_distribution.h"
#include "kernel/sample/guiding_field.h"
#include "kernel/sample/guiding_mixture_conditional.h"
#include "kernel/sample/guiding_mixture_fit.h"
#include "kernel/sample/guiding_mixture_statistics.h"
#include "kernel/sample/guiding_observation_range.h"
#include "kernel/sample/guiding_parallax.h"
#include "kernel/sample/guiding_position.h"
#include "kernel/sample/guiding_spherical_gaussian.h"
#include "kernel/util/compact_indices.h"

using Tree = ccl::GuidingDirectionalTree<5>;

static void require(const bool condition, const char *message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template<int Capacity = 2>
static void test_mixture_fit(id<MTLDevice> device,
                             id<MTLLibrary> library,
                             id<MTLCommandQueue> queue,
                             const bool cooperative,
                             const uint count = 4096,
                             const uint cases = 2)
{
  using namespace ccl;
  std::array<GuidingSphericalGaussian, Capacity> sources;
  if constexpr (Capacity == 2) {
    sources = {{{normalize(make_float3(.2f, .4f, 1)), 1500},
                {normalize(make_float3(.4f, .2f, 1)), 1500}}};
  }
  else {
    for (int i = 0; i < Capacity; ++i) {
      const float z = 1 - 2 * (i + .5f) / Capacity;
      const float phi = i * 2.39996323f;
      const float radius = safe_sqrtf(1 - z * z);
      sources[i] = {normalize(make_float3(radius * cosf(phi), radius * sinf(phi), z)), 1500};
    }
  }
  std::mt19937 generator(816);
  std::uniform_real_distribution<float> uniform(0, 1);
  std::vector<float4> samples(count * cases);
  for (uint i = 0; i < count; ++i) {
    float pdf;
    const float3 direction = sources[i % Capacity].sample(
        make_float2(uniform(generator), uniform(generator)), &pdf);
    samples[i] = make_float4(
        direction.x, direction.y, direction.z, Capacity == 2 ? (i % 2 ? .65f : .35f) : 1.0f);
    for (uint test = 1; test < cases; ++test) {
      samples[test * count + i] = samples[i];
      if (test % 2) {
        samples[test * count + i].w *= 1e30f;
      }
    }
  }
  id<MTLBuffer> input = [device newBufferWithBytes:samples.data()
                                            length:samples.size() * sizeof(float4)
                                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> output = [device newBufferWithLength:2 * Capacity * cases * sizeof(float4)
                                             options:MTLResourceStorageModeShared];
  id<MTLBuffer> published = [device
      newBufferWithLength:cases * GuidingGaussianMixture::storage_size * sizeof(float)
                  options:MTLResourceStorageModeShared];
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline = [device
      newComputePipelineStateWithFunction:
          [library newFunctionWithName:Capacity == 16 ?
                                           (cooperative ? @"guiding_mixture_fit_cooperative_16" :
                                                          @"guiding_mixture_fit_16") :
                                           (cooperative ? @"guiding_mixture_fit_cooperative" :
                                                          @"guiding_mixture_fit")]
                                    error:&error];
  require(input && output && published && pipeline, "Mixture fit allocation/pipeline failed");
  constexpr int measured_runs = 5;
  std::array<double, measured_runs> durations;
  for (int run = -1; run < measured_runs; ++run) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder setBuffer:published offset:0 atIndex:3];
    if (cooperative) {
      [encoder dispatchThreadgroups:MTLSizeMake(cases, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
    }
    else {
      [encoder dispatchThreads:MTLSizeMake(cases, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(
                                    std::min(uint(pipeline.threadExecutionWidth), cases), 1, 1)];
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
      NSString *reason = command.error.localizedDescription ?: @"No Metal error supplied";
      std::fprintf(stderr,
                   "GUIDING_MIXTURE_COMMAND_ERROR capacity=%d cooperative=%d observations=%u "
                   "fields=%u run=%d status=%lu error_code=%ld description=%s\n",
                   Capacity,
                   int(cooperative),
                   count,
                   cases,
                   run,
                   static_cast<unsigned long>(command.status),
                   static_cast<long>(command.error.code),
                   reason.UTF8String);
    }
    require(command.status == MTLCommandBufferStatusCompleted, "Mixture fit GPU execution failed");
    if (run >= 0) {
      durations[run] = command.GPUEndTime - command.GPUStartTime;
    }
  }
  std::sort(durations.begin(), durations.end());
  const auto *result = static_cast<const float4 *>(output.contents);
  for (uint test = 0; test < cases; ++test) {
    GuidingDirectionalMixtureFit<Capacity> cpu;
    require(cpu.initialize(samples.data() + test * count, count),
            "Mixture CPU initialization failed");
    for (int iteration = 0; iteration < 64; ++iteration) {
      require(cpu.iterate(samples.data() + test * count, count), "Mixture CPU iteration failed");
    }
    float sum = 0;
    for (int component = 0; component < Capacity; ++component) {
      const auto fitted = result[2 * Capacity * test + 2 * component];
      const auto metadata = result[2 * Capacity * test + 2 * component + 1];
      require(metadata.y == 1 && metadata.z == Capacity, "Mixture Metal fit failed");
      require(len(make_float3(fitted) - cpu.lobes[component].axis) < 2e-5f,
              "Mixture Metal/CPU fitted direction mismatch");
      {
        int source = 0;
        for (int candidate = 1; candidate < Capacity; ++candidate) {
          if (dot(cpu.lobes[component].axis, sources[candidate].axis) >
              dot(cpu.lobes[component].axis, sources[source].axis))
          {
            source = candidate;
          }
        }
        double dx = 0, dy = 0, dz = 0, mass = 0;
        for (uint j = source; j < count; j += Capacity) {
          const auto sample = samples[test * count + j];
          const double norm = std::sqrt(double(sample.x) * sample.x + double(sample.y) * sample.y +
                                        double(sample.z) * sample.z);
          dx += sample.w * sample.x / norm;
          dy += sample.w * sample.y / norm;
          dz += sample.w * sample.z / norm;
          mass += sample.w;
        }
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz) / mass;
        if (std::abs(double(fitted.w) - 1 / (1 - r)) >= 1.0) {
          std::fprintf(stderr,
                       "FIT_REFERENCE count=%u case=%u component=%d gpu=%.9g reference=%.12g\n",
                       count,
                       test,
                       component,
                       fitted.w,
                       1 / (1 - r));
        }
        require(std::abs(double(fitted.w) - 1 / (1 - r)) < 1.0,
                "Mixture Metal concentration disagrees with independent double moments");
      }
      require(std::abs(fitted.w - cpu.lobes[component].concentration) < 3.0f,
              "Mixture Metal/CPU fitted concentration mismatch");
      require(std::abs(metadata.x - cpu.weights[component]) < 2e-5f,
              "Mixture Metal/CPU fitted weight mismatch");
      GuidingGaussianMixture query;
      const float *storage = static_cast<const float *>(published.contents) +
                             test * GuidingGaussianMixture::storage_size;
      const float host_pdf = query.pdf(storage, make_float3(fitted));
      require(std::abs(host_pdf - metadata.w) < 2e-4f * std::max(host_pdf, 1.0f),
              "Published fitted mixture Metal/CPU PDF mismatch");
      sum += metadata.x;
    }
    require(std::abs(sum - 1) < 1e-6f, "Mixture Metal weights not normalized");
  }
  std::printf(
      "GUIDING_MIXTURE_FIT_METAL observations=%u fields=%u components=%d cooperative=%d "
      "warm_gpu_median_seconds=%.9g min_seconds=%.9g max_seconds=%.9g measured_runs=5 "
      "iterations=64 peaks=passed "
      "bright_weights=passed host_agreement=passed\n",
      count * cases,
      cases,
      Capacity,
      cooperative,
      durations[measured_runs / 2],
      durations.front(),
      durations.back());
}

template<int Capacity = 2, bool BatchedHDR = false>
static void test_mixture_sources(id<MTLDevice> device,
                                 id<MTLLibrary> library,
                                 id<MTLCommandQueue> queue)
{
  using namespace ccl;
  constexpr uint count = 4096;
  std::array<float3, Capacity> targets;
  if constexpr (Capacity == 2) {
    targets = {make_float3(.2f, .4f, 1), make_float3(.4f, .2f, 1)};
  }
  else {
    for (int i = 0; i < Capacity; ++i) {
      const float z = 1 - 2 * (i + .5f) / Capacity;
      const float phi = i * 2.39996323f;
      const float r = safe_sqrtf(1 - z * z);
      targets[i] = normalize(make_float3(r * cosf(phi), r * sinf(phi), z));
    }
  }
  const float3 extent = make_float3(1, .5f, .25f);
  std::vector<float4> directions(count);
  std::vector<GuidingMixtureObservation> observations(count);
  std::mt19937 generator(376);
  std::uniform_real_distribution<float> uniform(-.03f, .03f);
  for (uint i = 0; i < count; ++i) {
    const float3 position = make_float3(
        uniform(generator), uniform(generator), uniform(generator));
    const float3 delta = targets[i % Capacity] - position * extent;
    const float distance = len(delta);
    const float3 direction = delta / distance;
    directions[i] = make_float4(
        direction.x, direction.y, direction.z, Capacity == 2 ? (i % 2 ? .65f : .35f) : 1.0f);
    if constexpr (BatchedHDR) {
      const float intensities[] = {1.0f, 8.0f, 2.0f, 32.0f};
      directions[i].w *= 1e28f * intensities[i / 1024];
    }
    const float w = directions[i].w;
    observations[i] = {directions[i],
                       make_float4(position.x, position.y, position.z, 0),
                       make_float4(w, w / distance, w * distance, 0)};
  }
  id<MTLBuffer> direction_buffer = [device newBufferWithBytes:directions.data()
                                                       length:directions.size() * sizeof(float4)
                                                      options:MTLResourceStorageModeShared];
  id<MTLBuffer> observation_buffer = [device
      newBufferWithBytes:observations.data()
                  length:observations.size() * sizeof(GuidingMixtureObservation)
                 options:MTLResourceStorageModeShared];
  id<MTLBuffer> output = [device
      newBufferWithLength:Capacity * GuidingMixtureStatistics::storage_size * sizeof(float)
                  options:MTLResourceStorageModeShared];
  id<MTLBuffer> fits = [device newBufferWithLength:2 * Capacity * sizeof(float4)
                                           options:MTLResourceStorageModeShared];
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline = [device
      newComputePipelineStateWithFunction:
          [library newFunctionWithName:Capacity == 2 ? @"guiding_mixture_source_statistics" :
                                                       @"guiding_mixture_source_statistics_16"]
                                    error:&error];
  require(direction_buffer && observation_buffer && output && fits && pipeline,
          "Mixture source allocation/pipeline failed");
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:direction_buffer offset:0 atIndex:0];
  [encoder setBuffer:observation_buffer offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBytes:&count length:sizeof(count) atIndex:3];
  [encoder setBuffer:fits offset:0 atIndex:4];
  [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  require(command.status == MTLCommandBufferStatusCompleted,
          "Mixture source GPU execution failed");
  GuidingDirectionalMixtureFit<Capacity> model;
  require(model.initialize(directions.data(), count), "Mixture source CPU init failed");
  for (int iteration = 0; iteration < 64; ++iteration) {
    require(model.iterate(directions.data(), count), "Mixture source CPU fit failed");
  }
  GuidingMixtureStatistics expected[Capacity];
  for (const auto &observation : observations) {
    const float3 direction = normalize(make_float3(observation.direction_weight));
    float density[Capacity], maximum = -FLT_MAX, sum = 0;
    for (int c = 0; c < Capacity; ++c) {
      density[c] = model.log_component(c, direction);
      maximum = std::max(maximum, density[c]);
    }
    for (int c = 0; c < Capacity; ++c) {
      density[c] = std::exp(density[c] - maximum);
      sum += density[c];
    }
    for (int c = 0; c < Capacity; ++c) {
      expected[c].record(observation, density[c] / sum, model.maximum_weight, extent);
    }
  }
  const float *actual = static_cast<const float *>(output.contents);
  for (int component = 0; component < Capacity; ++component) {
    const float *statistics = actual + component * GuidingMixtureStatistics::storage_size;
    for (int i = 0; i < GuidingMixtureStatistics::storage_size; ++i) {
      require(std::isfinite(statistics[i]) &&
                  std::abs(statistics[i] - expected[component].values[i]) <
                      2e-5f * std::max(std::abs(expected[component].values[i]), 1.0f),
              "Mixture source GPU/CPU statistic mismatch");
    }
    GuidingParallaxMoments source;
    const auto fitted = source.fit(statistics + GuidingMixtureStatistics::position_size,
                                   statistics[0],
                                   count,
                                   model.lobes[component].axis);
    require(fitted.valid, "Mixture source fit invalid");
    const auto *gpu_fits = static_cast<const float4 *>(fits.contents);
    require(gpu_fits[2 * component].w == 1 &&
                len(make_float3(gpu_fits[2 * component]) - fitted.target) < 1e-5f,
            "Mixture source Metal fit mismatch");
    GuidingPositionMoments position;
    const auto position_fit = position.fit(statistics, count);
    const float3 query = make_float3(.02f, -.01f, .01f);
    require(position_fit.valid && gpu_fits[2 * component + 1].w == 1 &&
                len(make_float3(gpu_fits[2 * component + 1]) - position_fit.direction(query)) <
                    1e-5f,
            "Mixture position Metal fit mismatch");
    require(len(position_fit.direction(query) - normalize(fitted.target - query * extent)) < .003f,
            "Mixture position query disagrees with known source geometry");

    float distance = FLT_MAX;
    for (const auto &target : targets) {
      distance = std::min(distance, len(fitted.target - target));
    }
    require(distance < 1e-4f, "Learned source target disagrees with known geometry");
    require(std::abs(fitted.covariance[0]) + std::abs(fitted.covariance[3]) +
                    std::abs(fitted.covariance[5]) <
                1e-5f,
            "Point source gained spurious extent");
  }
  /* Conditional iterations always read and write distinct models. */
  std::array<float, GuidingGaussianMixture::storage_size> initial{};
  model.publish(initial.data());
  id<MTLBuffer> models[2] = {[device newBufferWithBytes:initial.data()
                                                 length:sizeof(initial)
                                                options:MTLResourceStorageModeShared],
                             [device newBufferWithLength:sizeof(initial)
                                                 options:MTLResourceStorageModeShared]};
  id<MTLBuffer> scratch = [device
      newBufferWithLength:GuidingGaussianMixture::components *
                          GuidingMixtureStatistics::storage_size * sizeof(float)
                  options:MTLResourceStorageModeShared];
  id<MTLComputePipelineState> update = [device
      newComputePipelineStateWithFunction:
          [library newFunctionWithName:@"guiding_mixture_conditional_update"]
                                    error:&error];
  require(models[0] && models[1] && scratch && update,
          "Conditional mixture allocation/pipeline failed");
  for (int iteration = 0; iteration < 4; ++iteration) {
    id<MTLBuffer> input_model = models[iteration % 2];
    id<MTLBuffer> output_model = models[1 - iteration % 2];
    std::array<float, GuidingGaussianMixture::storage_size> before;
    std::memcpy(before.data(), input_model.contents, sizeof(before));
    id<MTLCommandBuffer> step = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [step computeCommandEncoder];
    [encoder setComputePipelineState:update];
    [encoder setBuffer:observation_buffer offset:0 atIndex:0];
    [encoder setBuffer:input_model offset:0 atIndex:1];
    [encoder setBuffer:output_model offset:0 atIndex:2];
    [encoder setBuffer:scratch offset:0 atIndex:3];
    [encoder setBytes:&count length:sizeof(count) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(update.threadExecutionWidth, 1, 1)];
    [encoder endEncoding];
    [step commit];
    [step waitUntilCompleted];
    require(step.status == MTLCommandBufferStatusCompleted,
            "Conditional mixture GPU update failed");
    require(std::memcmp(before.data(), input_model.contents, sizeof(before)) == 0,
            "Conditional update mutated its input distribution");
    std::array<GuidingMixtureStatistics, GuidingGaussianMixture::components> reference;
    GuidingConditionalMixture conditional;
    GuidingGaussianMixture query;
    for (int c = 0; c < query.components; ++c) {
      const int base = c * query.component_stride;
      reference[c].direction_reference = make_float3(
          before[base + 2], before[base + 3], before[base + 4]);
    }
    for (const auto &observation : observations) {
      const float3 direction = normalize(make_float3(observation.direction_weight));
      float densities[GuidingGaussianMixture::components];
      float maximum = -FLT_MAX;
      for (int c = 0; c < query.components; ++c) {
        densities[c] = conditional.log_component(
            before.data(), c, direction, make_float3(observation.position));
        maximum = std::max(maximum, densities[c]);
      }
      float sum = 0;
      for (int c = 0; c < query.components; ++c) {
        densities[c] = densities[c] > -FLT_MAX ? std::exp(densities[c] - maximum) : 0;
        sum += densities[c];
      }
      for (int c = 0; c < query.components; ++c) {
        reference[c].record(observation, densities[c] / sum, model.maximum_weight, extent);
      }
    }
    float total_weight = 0;
    float global_squared_weight = 0;
    for (const auto &statistics : reference) {
      total_weight += statistics.values[0];
      global_squared_weight += statistics.values[GuidingMixtureStatistics::global_squared_weight];
    }
    std::array<float, GuidingGaussianMixture::storage_size> expected_model{};
    for (int c = 0; c < query.components; ++c) {
      auto &statistics = reference[c];
      require(
          statistics.recenter(statistics.direction_reference + make_float3(.001f, -.002f, .003f)),
          "Conditional reference recentering failed");
      const float *merged = static_cast<const float *>(scratch.contents) +
                            c * GuidingMixtureStatistics::storage_size;
      for (int entry = 0; entry < GuidingMixtureStatistics::storage_size; ++entry) {
        if (!(std::isfinite(merged[entry]) &&
              std::abs(merged[entry] - statistics.values[entry]) <
                  2e-5f * std::max(std::abs(statistics.values[entry]), 1.0f)))
        {
          std::fprintf(
              stderr,
              "CENTERED_STAT component=%d entry=%d gpu=%.9g cpu=%.9g ref=(%.9g,%.9g,%.9g)\n",
              c,
              entry,
              merged[entry],
              statistics.values[entry],
              statistics.direction_reference.x,
              statistics.direction_reference.y,
              statistics.direction_reference.z);
        }
        require(std::isfinite(merged[entry]) &&
                    std::abs(merged[entry] - statistics.values[entry]) <
                        2e-5f * std::max(std::abs(statistics.values[entry]), 1.0f),
                "Grouped conditional statistics disagree with single-pass CPU collection");
      }
      conditional.publish_component(expected_model.data(),
                                    c,
                                    statistics.values,
                                    total_weight,
                                    count,
                                    statistics.directional_fit(),
                                    extent,
                                    total_weight * total_weight / global_squared_weight);
    }
    const float *actual_model = static_cast<const float *>(output_model.contents);
    for (int i = 0; i < query.storage_size; ++i) {
      require(std::isfinite(actual_model[i]) &&
                  std::abs(actual_model[i] - expected_model[i]) <
                      3e-4f * std::max(std::abs(expected_model[i]), 1.0f),
              "Conditional mixture GPU/CPU publication mismatch");
    }
    float mass = 0;
    for (int c = 0; c < query.components; ++c) {
      mass += actual_model[c * query.component_stride];
    }
    require(std::abs(mass - 1) < 1e-6f, "Conditional mixture lost normalized mass");
    for (int c = 0; c < Capacity; ++c) {
      require(actual_model[c * query.component_stride + 17] == 2,
              "Conditional mixture did not publish source models");
    }
  }
  std::printf(
      "GUIDING_MIXTURE_SOURCES_METAL observations=4096 components=%d learned_assignments=passed "
      "statistics=passed targets=passed conditional_iterations=4 immutable_input=passed "
      "group_size=257 batch_scale_changes=%d\n",
      Capacity,
      BatchedHDR);
}

static void test_observation_partition(id<MTLDevice> device,
                                       id<MTLLibrary> library,
                                       id<MTLCommandQueue> queue)
{
  using namespace ccl;
  constexpr uint count = 4096, fields = 16;
  std::vector<GuidingHistoryRecord> records(count);
  for (uint i = 0; i < count; ++i) {
    auto &r = records[i];
    r.field_index = (i % fields) * 100 + 7;
    r.parent = i == 0 ? ~0u : i - 1;
    r.direction = packed_normal(normalize(make_float3(1, int(i % fields) - 8, 3))).value;
    r.radiance = 1;
    r.source_weight = 1;
    r.inverse_distance_weight = .25f;
    r.distance_weight = 4;
  }
  id<MTLBuffer> input = [device newBufferWithBytes:records.data()
                                            length:records.size() * sizeof(GuidingHistoryRecord)
                                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> metadata = [device newBufferWithLength:49 * sizeof(uint)
                                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> indices = [device newBufferWithLength:(count + 16) * sizeof(uint)
                                              options:MTLResourceStorageModeShared];
  id<MTLBuffer> output = [device newBufferWithLength:2 * fields * sizeof(float4)
                                             options:MTLResourceStorageModeShared];
  NSError *error = nil;
  id<MTLComputePipelineState> partition = [device
      newComputePipelineStateWithFunction:[library
                                              newFunctionWithName:@"guiding_observation_partition"]
                                    error:&error];
  id<MTLComputePipelineState> fit = [device
      newComputePipelineStateWithFunction:
          [library newFunctionWithName:@"guiding_observation_indexed_fit"]
                                    error:&error];
  require(input && metadata && indices && output && partition && fit,
          "Observation partition allocation/pipeline failed");
  for (uint trial = 0; trial < 2; ++trial) {
    const uint capacity = count - trial;
    std::memset(metadata.contents, 0, metadata.length);
    std::memset(indices.contents, 0xab, indices.length);
    id<MTLCommandBuffer> command = [queue commandBuffer];
    for (uint phase = 0; phase < 3; ++phase) {
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:partition];
      [encoder setBuffer:input offset:0 atIndex:0];
      [encoder setBuffer:metadata offset:0 atIndex:1];
      [encoder setBuffer:indices offset:0 atIndex:2];
      [encoder setBytes:&count length:sizeof(count) atIndex:3];
      [encoder setBytes:&capacity length:sizeof(capacity) atIndex:4];
      [encoder setBytes:&phase length:sizeof(phase) atIndex:5];
      [encoder dispatchThreads:MTLSizeMake(phase == 1 ? 1 : count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(phase == 1 ? 1 : 64, 1, 1)];
      [encoder endEncoding];
    }
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
            "Observation partition GPU execution failed");
    const uint *meta = static_cast<const uint *>(metadata.contents);
    const uint *order = static_cast<const uint *>(indices.contents);
    require(trial ? (meta[48] & 2u) != 0 : meta[48] == 0,
            "Observation partition capacity guard failed");
    for (uint i = capacity; i < count + 16; ++i)
      require(order[i] == 0xababababu, "Observation partition wrote beyond capacity");
    require(std::memcmp(input.contents,
                        records.data(),
                        records.size() * sizeof(GuidingHistoryRecord)) == 0,
            "Observation partition changed histories");
    if (trial)
      continue;
    std::vector<bool> seen(count, false);
    for (uint f = 0; f < fields; ++f) {
      require(meta[f] == count / fields && meta[32 + f] == meta[f],
              "Observation partition count mismatch");
      for (uint i = 0; i < meta[f]; ++i) {
        const uint index = order[meta[16 + f] + i];
        require(index < count && !seen[index] && records[index].field_index / 100 == f,
                "Observation partition lost, duplicated, or misassigned a record");
        seen[index] = true;
      }
    }
    command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:fit];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:metadata offset:0 atIndex:1];
    [encoder setBuffer:indices offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(fields, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(fit.threadExecutionWidth, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted, "Indexed observation fit failed");
    const float4 *result = static_cast<const float4 *>(output.contents);
    for (uint f = 0; f < fields; ++f) {
      packed_normal direction;
      direction.value = records[f].direction;
      require(len(make_float3(result[2 * f]) - direction.decode()) < 1e-5f &&
                  result[2 * f].w == 16384,
              "Indexed observation fitter read incorrect directions");
      require(result[2 * f + 1].x == 1 && result[2 * f + 1].y == .5f && result[2 * f + 1].z == 2,
              "Indexed observation source metric conversion failed");
    }
  }
  std::printf(
      "GUIDING_OBSERVATION_PARTITION_METAL records=4096 fields=16 unique=passed immutable=passed "
      "overflow=passed indexed_fit=passed\n");
}

static void test_hdr(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue)
{
  using namespace ccl;
  GuidingGaussianMixture mixture;
  GuidingDirectionalTree<5> tree;
  std::array<float, Tree::node_count> weights{}, counts{};
  std::array<float, GuidingGaussianMixture::components * GuidingPositionMoments::storage_size>
      moments{};
  const float3 axis = normalize(make_float3(1, 2, 3));
  const int component = (tree.leaf_index(axis) - Tree::leaf_offset) /
                        (Tree::leaf_count / mixture.components);
  weights[Tree::leaf_offset + (tree.leaf_index(axis) - Tree::leaf_offset)] = 1e20f;
  tree.build(weights.data(), 0.0f);
  counts[5 + component] = 64;
  float *observation = moments.data() + component * GuidingPositionMoments::storage_size;
  observation[0] = 1e20f;
  for (int j = 0; j < 3; ++j) {
    observation[1 + j] = 1e20f * axis[j];
  }
  observation[22] = INFINITY; /* Squared weights overflow; fallback must remain normalized. */
  id<MTLBuffer> buffers[] = {[device newBufferWithBytes:weights.data()
                                                 length:sizeof(weights)
                                                options:MTLResourceStorageModeShared],
                             [device newBufferWithBytes:moments.data()
                                                 length:sizeof(moments)
                                                options:MTLResourceStorageModeShared],
                             [device newBufferWithBytes:counts.data()
                                                 length:sizeof(counts)
                                                options:MTLResourceStorageModeShared],
                             [device newBufferWithLength:mixture.storage_size * sizeof(float)
                                                 options:MTLResourceStorageModeShared],
                             [device newBufferWithLength:mixture.components * sizeof(float4)
                                                 options:MTLResourceStorageModeShared]};
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline = [device
      newComputePipelineStateWithFunction:[library newFunctionWithName:@"guiding_hdr_build"]
                                    error:&error];
  require(pipeline != nil, "HDR publication pipeline failed");
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  for (int i = 0; i < 5; ++i) {
    require(buffers[i] != nil, "HDR allocation failed");
    [encoder setBuffer:buffers[i] offset:0 atIndex:i];
  }
  [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  require(command.status == MTLCommandBufferStatusCompleted,
          "HDR publication GPU execution failed");
  const auto *result = static_cast<const float4 *>(buffers[4].contents);
  require(len(make_float3(result[component]) - axis) < 1e-5f,
          "HDR publication lost the normalized direction");
  require(std::isfinite(result[component].w) && result[component].w > 1,
          "HDR publication lost PDF mass");
  std::printf("GUIDING_HDR_METAL finite_radiance=1e20 normalization=passed\n");
}

static void test_parallax(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue)
{
  constexpr uint cases = 4, queries = 4;
  ccl::GuidingParallaxMoments model;
  std::array<float, cases * ccl::GuidingParallaxMoments::storage_size> zero{};
  const std::array<ccl::float4, queries> positions = {ccl::make_float4(.3f, .2f, 0, 0),
                                                      ccl::make_float4(-.4f, .1f, 0, 0),
                                                      ccl::make_float4(0, 0, -1, 0),
                                                      ccl::make_float4(1e30f, 1e30f, 1e30f, 0)};
  id<MTLBuffer> moments = [device newBufferWithBytes:zero.data()
                                              length:sizeof(zero)
                                             options:MTLResourceStorageModeShared];
  id<MTLBuffer> points = [device newBufferWithBytes:positions.data()
                                             length:sizeof(positions)
                                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> results = [device newBufferWithLength:cases * queries * 2 * sizeof(ccl::float4)
                                              options:MTLResourceStorageModeShared];
  require(moments && points && results, "Parallax allocation failed");
  NSError *error = nil;
  id<MTLComputePipelineState> recording = [device
      newComputePipelineStateWithFunction:[library newFunctionWithName:@"guiding_parallax_record"]
                                    error:&error];
  id<MTLComputePipelineState> fitting = [device
      newComputePipelineStateWithFunction:[library newFunctionWithName:@"guiding_parallax_fit"]
                                    error:&error];
  require(recording && fitting, "Parallax pipeline failed");
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:recording];
  [encoder setBuffer:moments offset:0 atIndex:0];
  [encoder dispatchThreads:MTLSizeMake(cases * 4096, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [encoder endEncoding];
  encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:fitting];
  [encoder setBuffer:moments offset:0 atIndex:0];
  [encoder setBuffer:points offset:0 atIndex:1];
  [encoder setBuffer:results offset:0 atIndex:2];
  [encoder dispatchThreads:MTLSizeMake(cases * queries, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  require(command.status == MTLCommandBufferStatusCompleted, "Parallax GPU execution failed");
  const auto *values = static_cast<const float *>(moments.contents);
  const auto *output = static_cast<const ccl::float4 *>(results.contents);
  for (uint test = 0; test < cases; ++test) {
    const auto expected = model.fit(values + test * model.storage_size,
                                    test == 3 ? 1e8f + 4095 : 4096,
                                    4096,
                                    ccl::make_float3(0, 0, 1));
    require(expected.valid == (test < 2), "Invalid GPU-recorded parallax fixture");
    if (test == 0) {
      require(ccl::len(expected.target - ccl::make_float3(1, 1, 2)) < 1e-4f,
              "Parallax source target does not match the emitter");
    }
    for (uint q = 0; q < queries; ++q) {
      const uint index = 2 * (test * queries + q);
      const auto p = ccl::make_float3(positions[q]);
      require((output[index + 1].x == 1) == expected.valid, "Parallax validity mismatch");
      require(ccl::len(ccl::make_float3(output[index]) - expected.direction(p)) < 2e-5f,
              "Parallax query direction mismatch");
      const float k = expected.concentration(p);
      require(std::isfinite(output[index].w) &&
                  std::abs(output[index].w - k) < 2e-3f * std::max(k, 1.0f),
              "Parallax query concentration mismatch");
      require(std::abs(output[index + 1].y - (2 + (40.0f / 9.0f) * (q + 1))) < 1e-5f,
              "Parallax virtual distance mismatch");
    }
  }
  std::printf(
      "GUIDING_PARALLAX_METAL records=16384 point=passed area=passed distant=passed "
      "dominated=passed refraction=passed queries=4\n");
}

static void test_position(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue)
{
  constexpr uint cases = 3, queries = 4;
  ccl::GuidingPositionMoments model;
  std::array<float, cases * ccl::GuidingPositionMoments::storage_size> values{};
  for (int y = 0; y < 64; ++y) {
    for (int x = 0; x < 64; ++x) {
      const ccl::float3 p = ccl::make_float3((x + .5f) / 64 - .5f, (y + .5f) / 64 - .5f, 0);
      const ccl::float3 d = ccl::normalize(ccl::make_float3(0, 0, 2) - p);
      model.record(values.data(), 1.0f, p, d);
      model.record(values.data() + model.storage_size,
                   1.0f,
                   ccl::zero_float3(),
                   ccl::make_float3(0, 0, x % 2 ? 1 : -1));
      model.record(values.data() + 2 * model.storage_size, x == 0 && y == 0 ? 1e8f : 1.0f, p, d);
    }
  }
  const std::array<ccl::float4, queries> positions = {ccl::make_float4(.3f, .2f, 0, 0),
                                                      ccl::make_float4(-.4f, .1f, 0, 0),
                                                      ccl::make_float4(0, 0, 0, 0),
                                                      ccl::make_float4(1e30f, 1e30f, 1e30f, 0)};
  id<MTLBuffer> moments = [device newBufferWithBytes:values.data()
                                              length:sizeof(values)
                                             options:MTLResourceStorageModeShared];
  id<MTLBuffer> points = [device newBufferWithBytes:positions.data()
                                             length:sizeof(positions)
                                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> results = [device newBufferWithLength:cases * queries * 2 * sizeof(ccl::float4)
                                              options:MTLResourceStorageModeShared];
  require(moments && points && results, "Position fit allocation failed");
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline = [device
      newComputePipelineStateWithFunction:[library newFunctionWithName:@"guiding_position_fit"]
                                    error:&error];
  require(pipeline != nil, "Position fitting pipeline failed");
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:moments offset:0 atIndex:0];
  [encoder setBuffer:points offset:0 atIndex:1];
  [encoder setBuffer:results offset:0 atIndex:2];
  [encoder dispatchThreads:MTLSizeMake(cases * queries, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  require(command.status == MTLCommandBufferStatusCompleted, "Position fitting execution failed");
  const auto *output = static_cast<const ccl::float4 *>(results.contents);
  for (uint test = 0; test < cases; ++test) {
    const auto expected = model.fit(values.data() + test * model.storage_size, 4096);
    require(expected.valid == (test < 2), "Invalid host position fixture");
    for (uint q = 0; q < queries; ++q) {
      const uint index = 2 * (test * queries + q);
      const ccl::float3 actual = ccl::make_float3(output[index]);
      require((output[index].w == 1.0f) == expected.valid,
              "Position validity host/Metal mismatch");
      require(ccl::isfinite_safe(actual) && std::abs(ccl::len(actual) - 1.0f) < 1e-5f,
              "Invalid conditioned direction");
      require(ccl::len(actual - expected.direction(ccl::make_float3(positions[q]))) < 1e-5f,
              "Conditioned direction host/Metal mismatch");
      require(std::abs(output[index + 1].x - expected.residual_variance) < 1e-5f,
              "Conditional residual variance host/Metal mismatch");
    }
  }
  std::printf(
      "GUIDING_POSITION_METAL planar=passed isotropic=passed dominated=passed queries=%u\n",
      queries);
}

static void test_history(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue)
{
  constexpr uint paths = 100000, full_capacity = paths * 10;
  const size_t record_bytes = full_capacity * sizeof(ccl::GuidingHistoryRecord);
  id<MTLBuffer> records = [device newBufferWithLength:record_bytes + 64
                                              options:MTLResourceStorageModeShared];
  id<MTLBuffer> count = [device newBufferWithLength:sizeof(uint)
                                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> heads = [device newBufferWithLength:paths * 2 * sizeof(uint)
                                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> results = [device newBufferWithLength:paths * 4 * sizeof(uint)
                                              options:MTLResourceStorageModeShared];
  require(records && count && heads && results, "History allocation failed");
  NSError *error = nil;
  id<MTLComputePipelineState> append = [device
      newComputePipelineStateWithFunction:[library newFunctionWithName:@"guiding_history_append"]
                                    error:&error];
  id<MTLComputePipelineState> read = [device
      newComputePipelineStateWithFunction:[library newFunctionWithName:@"guiding_history_read"]
                                    error:&error];
  id<MTLComputePipelineState> accumulate = [device
      newComputePipelineStateWithFunction:[library
                                              newFunctionWithName:@"guiding_history_accumulate"]
                                    error:&error];
  require(append && read && accumulate, "History pipeline compilation failed");
  for (uint group = 0; group < 4; ++group) {
    const uint capacity = group == 2 ? full_capacity - 37 : full_capacity;
    memset(records.contents, 0xab, record_bytes + 64);
    *static_cast<uint *>(count.contents) = 0;
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:append];
    [encoder setBuffer:records offset:0 atIndex:0];
    [encoder setBuffer:count offset:0 atIndex:1];
    [encoder setBuffer:heads offset:0 atIndex:2];
    [encoder setBytes:&capacity length:sizeof(capacity) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(paths, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted, "History append failed");
    require(*static_cast<uint *>(count.contents) == full_capacity,
            "History allocator lost records");
    const auto *bytes = static_cast<const unsigned char *>(records.contents);
    for (size_t i = size_t(capacity) * sizeof(ccl::GuidingHistoryRecord); i < record_bytes + 64;
         ++i)
    {
      require(bytes[i] == 0xab, "History overflow wrote outside capacity");
    }
    command = [queue commandBuffer];
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:accumulate];
    [encoder setBuffer:records offset:0 atIndex:0];
    [encoder setBuffer:heads offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(paths * 2, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted, "History accumulation failed");
    command = [queue commandBuffer];
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:read];
    [encoder setBuffer:records offset:0 atIndex:0];
    [encoder setBuffer:heads offset:0 atIndex:1];
    [encoder setBuffer:results offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(paths, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted, "History read failed");
    const auto *retained = static_cast<const ccl::GuidingHistoryRecord *>(records.contents);
    uint standalone_count = 0;
    for (uint i = 0; i < capacity; ++i) {
      const auto &record = retained[i];
      if (record.direction == 99 || record.direction == 100) {
        ++standalone_count;
        require(record.parent == ~0u, "Standalone observation entered camera ancestry");
        const bool finite = record.direction == 99;
        require(record.radiance == (finite ? 3.0f : 5.0f) &&
                    record.source_weight == (finite ? 3.0f : 0.0f) &&
                    record.inverse_distance_weight == (finite ? 1.5f : 0.0f) &&
                    record.distance_weight == (finite ? 6.0f : 0.0f),
                "Standalone observation lost radiance/source moments");
      }
    }
    if (capacity == full_capacity) {
      require(standalone_count == 2 * paths, "Standalone observations were dropped");
    }
    const auto *values = static_cast<const uint *>(results.contents);
    for (uint i = 0; i < paths; ++i) {
      require(values[4 * i + 2] == 0, "History changed ancestors or crossed paths");
      if (capacity == full_capacity) {
        require(values[4 * i] == 4 && values[4 * i + 1] == 8 && values[4 * i + 3] == 34,
                "Shadow history did not preserve its captured prefix");
      }
      else {
        require(values[4 * i] <= 4 && values[4 * i + 1] <= 8, "Invalid overflow history");
      }
    }
  }
  std::printf(
      "GUIDING_HISTORY_METAL paths=%u groups=4 overflow_guard=passed prefix_lifetime=passed "
      "aggregation=passed source_moments=passed standalone_observations=passed\n",
      paths);
}

static void test_field(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue)
{
  constexpr uint capacity = 31;
  const std::array<NSUInteger, 4> sizes = {
      capacity * sizeof(ccl::GuidingSpatialNode),
      capacity * ccl::GUIDING_FIELD_TYPES * ccl::GuidingField::accumulation_size * sizeof(float),
      capacity * ccl::GUIDING_FIELD_TYPES * ccl::GuidingField::sampling_size * sizeof(float),
      2 * sizeof(uint)};
  id<MTLBuffer> buffers[4];
  for (int i = 0; i < 4; ++i) {
    buffers[i] = [device newBufferWithLength:sizes[i] options:MTLResourceStorageModeShared];
    require(buffers[i] != nil, "Field allocation failed");
    memset(buffers[i].contents, 0, sizes[i]);
  }
  auto *counts = static_cast<uint *>(buffers[3].contents);
  counts[0] = 1;
  const std::array<const char *, 4> names = {"guiding_field_record",
                                             "guiding_field_begin",
                                             "guiding_field_refine",
                                             "guiding_field_publish"};
  id<MTLComputePipelineState> pipelines[4];
  for (int i = 0; i < 4; ++i) {
    NSError *error = nil;
    pipelines[i] = [device
        newComputePipelineStateWithFunction:
            [library newFunctionWithName:[NSString stringWithUTF8String:names[i]]]
                                      error:&error];
    require(pipelines[i] != nil, "Field pipeline compilation failed");
  }
  for (int iteration = 0; iteration < 12; ++iteration) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    for (int stage = 0; stage < 4; ++stage) {
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipelines[stage]];
      const uint work_size = stage == 0 ? 1000000 :
                             stage == 1 ? 1 :
                             stage == 2 ? capacity :
                                          capacity * ccl::GUIDING_FIELD_TYPES;
      if (stage == 1) {
        [encoder setBuffer:buffers[3] offset:0 atIndex:0];
      }
      else {
        for (int i = 0; i < 4; ++i) {
          [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        }
        [encoder setBytes:&capacity length:sizeof(capacity) atIndex:4];
      }
      [encoder dispatchThreads:MTLSizeMake(work_size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(std::min(work_size, 256u), 1, 1)];
      [encoder endEncoding];
    }
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted, "Field GPU execution failed");
    require(counts[0] <= capacity, "Field exceeded node capacity");
  }
  ccl::GuidingField field{static_cast<ccl::GuidingSpatialNode *>(buffers[0].contents),
                          static_cast<float *>(buffers[1].contents),
                          static_cast<float *>(buffers[2].contents),
                          counts,
                          capacity,
                          ccl::make_float3(-4, -2, -1),
                          ccl::make_float3(4, 2, 1)};
  require(counts[0] == capacity, "Spatial refinement did not fill the test budget");
  require(field.nodes[0].axis == 0 && std::abs(field.nodes[0].split) > 1e-4f &&
              std::abs(field.nodes[0].split) < .02f,
          "Spatial root did not follow the sampled positions");
  std::printf("GUIDING_SPATIAL_SPLIT_METAL root_axis=%u root_split=%.9g observed_mean=passed\n",
              field.nodes[0].axis,
              field.nodes[0].split);
  Tree directional;
  for (const float x : {-3.0f, 3.0f}) {
    const uint leaf = field.find_leaf(ccl::make_float3(x, 0, 0.1f));
    require(leaf < capacity && field.nodes[leaf].children == 0, "Invalid spatial lookup");
    const ccl::float3 expected = ccl::normalize(ccl::make_float3(1, 0.7f, x < 0 ? 3 : -3));
    const float *radiance = field.distribution(leaf, ccl::GUIDING_FIELD_SURFACE_RADIANCE);
    const float *importance = field.distribution(leaf, ccl::GUIDING_FIELD_SURFACE_IMPORTANCE);
    std::array<float, ccl::GuidingGaussianMixture::storage_size> host_smooth;
    ccl::GuidingGaussianMixture smooth;
    const float *observations = field.accumulation + leaf * ccl::GUIDING_FIELD_TYPES *
                                                         ccl::GuidingField::accumulation_size;
    smooth.build(host_smooth.data(),
                 radiance,
                 observations + ccl::GuidingField::moments_offset,
                 observations + ccl::GuidingField::bins);
    for (int i = 0; i < ccl::GuidingGaussianMixture::storage_size; ++i) {
      require(std::abs(host_smooth[i] - radiance[ccl::GuidingField::tree_size + i]) <=
                  2e-4f * std::max(1.0f, std::abs(host_smooth[i])),
              "Host/device smooth-field fit mismatch");
    }
    std::printf(
        "GUIDING_FIELD_QUERY x=%g leaf=%u visits=%u mass=%g expected_pdf=%g opposite_pdf=%g\n",
        x,
        leaf,
        field.nodes[leaf].visits,
        radiance[0],
        directional.pdf(radiance, expected),
        directional.pdf(radiance, -expected));
    require(directional.pdf(radiance, expected) > 20 * directional.pdf(radiance, -expected),
            "Radiance field failed to learn spatial variation");
    require(directional.pdf(importance, -expected) > 20 * directional.pdf(importance, expected),
            "Adjoint field failed to learn spatial variation");
    require(field.distribution(leaf, ccl::GUIDING_FIELD_VOLUME_RADIANCE)[0] == 0.0f,
            "Surface training leaked into volume distribution");
  }
  std::printf(
      "GUIDING_FIELD_METAL records=12000000 nodes=%u capacity=%u passed=1\n", counts[0], capacity);
}

static void test_bdpt_mis_recurrence(id<MTLDevice> device,
                                     id<MTLLibrary> library,
                                     id<MTLCommandQueue> queue)
{
  using namespace ccl;
  constexpr uint count = 1024, depth = 16;
  std::mt19937 rng(71832);
  std::uniform_real_distribution<float> random(.25f, 1.75f);
  std::vector<float4> events(count * depth);
  for (uint i = 0; i < events.size(); ++i) {
    events[i] = make_float4(
        .5f * random(rng), random(rng), i % 41 == 0 ? 0 : random(rng), random(rng));
  }
  id<MTLBuffer> input = [device newBufferWithBytes:events.data()
                                            length:events.size() * sizeof(float4)
                                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> output = [device newBufferWithLength:count * sizeof(float4)
                                             options:MTLResourceStorageModeShared];
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline = [device
      newComputePipelineStateWithFunction:[library newFunctionWithName:@"bdpt_mis_recurrence"]
                                    error:&error];
  require(input && output && pipeline, "BDPT MIS recurrence pipeline failed");
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:input offset:0 atIndex:0];
  [encoder setBuffer:output offset:0 atIndex:1];
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  require(command.status == MTLCommandBufferStatusCompleted,
          "BDPT MIS recurrence dispatch failed");
  const auto *values = static_cast<const float4 *>(output.contents);
  const auto check = [&](const bool logarithmic) {
    for (uint path = 0; path < count; ++path) {
      double sums[2] = {};
      for (int origin = -1; origin < int(depth); ++origin) {
        double ratio = .75;
        if (origin >= 0) {
          const auto e = events[path * depth + origin];
          ratio = (origin == 0 ? 1.25 : 1.0 / events[path * depth + origin - 1].y) * double(e.x) *
                  e.w / e.y;
        }
        for (uint i = origin + 1; i < depth; ++i) {
          const auto e = events[path * depth + i];
          ratio *= double(e.x) * e.z / e.y;
        }
        sums[0] += ratio;
        sums[1] += ratio * ratio;
      }
      for (int exponent = 0; exponent < 2; ++exponent) {
        const float stored = exponent == 0 ? values[path].y : values[path].w;
        const double actual = logarithmic ? std::exp(double(stored)) : stored;
        const double tolerance = logarithmic ? 1e-4 : 1e-5;
        require(std::isfinite(actual) && std::abs(actual - sums[exponent]) <
                                             tolerance * std::max(sums[exponent], 1e-20),
                "BDPT MIS Metal recurrence disagrees with independent strategy expansion");
      }
    }
  };
  check(false);
  for (uint path = 0; path < count; path += 2) {
    for (uint i = 0; i < depth; ++i) {
      events[path * depth + i] = make_float4(1, 1, i < depth / 2 ? 1e20f : 1e-20f, 0);
    }
  }
  std::memcpy(input.contents, events.data(), events.size() * sizeof(float4));
  pipeline = [device
      newComputePipelineStateWithFunction:[library newFunctionWithName:@"bdpt_mis_log_recurrence"]
                                    error:&error];
  require(pipeline != nil, "BDPT logarithmic MIS pipeline failed");
  command = [queue commandBuffer];
  encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:input offset:0 atIndex:0];
  [encoder setBuffer:output offset:0 atIndex:1];
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  require(command.status == MTLCommandBufferStatusCompleted,
          "BDPT logarithmic MIS dispatch failed");
  check(true);
  std::printf(
      "BDPT_MIS_RECURRENCE_METAL paths=%u depth=%u balance=passed power=passed "
      "log_extreme_recovery=passed\n",
      count,
      depth);
}

static void test_bdpt_mis_connection_weights(id<MTLDevice> device,
                                             id<MTLLibrary> library,
                                             id<MTLCommandQueue> queue)
{
  constexpr uint count = 1024;
  std::vector<ccl::float4> data(count);
  for (uint i = 0; i < count; ++i) {
    const float shift = i % 2 ? 10000.0f : 0.0f;
    data[i] = ccl::make_float4(shift + (int(i % 17) - 8) * .5f,
                               shift + (int(i % 13) - 6) * .5f,
                               -shift - float(i % 7) * .125f,
                               0);
  }
  id<MTLBuffer> input = [device newBufferWithBytes:data.data()
                                            length:count * sizeof(ccl::float4)
                                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> output = [device newBufferWithLength:count * sizeof(ccl::float2)
                                             options:MTLResourceStorageModeShared];
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline = [device
      newComputePipelineStateWithFunction:[library
                                              newFunctionWithName:@"bdpt_mis_connection_weights"]
                                    error:&error];
  require(input && output && pipeline, "BDPT null-connection MIS pipeline failed");
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:input offset:0 atIndex:0];
  [encoder setBuffer:output offset:0 atIndex:1];
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  require(command.status == MTLCommandBufferStatusCompleted,
          "BDPT null-connection MIS dispatch failed");
  const auto *values = static_cast<const ccl::float2 *>(output.contents);
  for (uint i = 0; i < count; ++i) {
    for (int exponent = 1; exponent <= 2; ++exponent) {
      // The large common factor cancels before this independent double-precision
      // direct normalization. Neither huge intermediate probability is exponentiated.
      const double a = double(data[i].x) + data[i].z;
      const double b = double(data[i].y) + data[i].z;
      const double expected = 1.0 / (1.0 + std::exp(exponent * a) + std::exp(exponent * b));
      const float actual = exponent == 1 ? values[i].x : values[i].y;
      require(std::isfinite(actual) && std::abs(actual - expected) < 5e-4,
              "BDPT null-connection MIS disagrees with normalized probabilities");
    }
  }
  std::printf(
      "BDPT_NULL_CONNECTION_MIS_METAL samples=%u balance=passed power=passed "
      "extreme_logs=passed\n",
      count);
}

static void test_bdpt_volume_homogeneous_events(id<MTLDevice> device,
                                                id<MTLLibrary> library,
                                                id<MTLCommandQueue> queue)
{
  constexpr uint count = 5 * 65536;
  id<MTLBuffer> output = [device newBufferWithLength:count * sizeof(ccl::float4)
                                             options:MTLResourceStorageModeShared];
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline = [device
      newComputePipelineStateWithFunction:
          [library newFunctionWithName:@"bdpt_volume_homogeneous_events"]
                                    error:&error];
  require(output && pipeline, "BDPT volume-event pipeline failed");
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:output offset:0 atIndex:0];
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  require(command.status == MTLCommandBufferStatusCompleted, "BDPT volume-event dispatch failed");
  const auto *values = static_cast<const ccl::float4 *>(output.contents);
  double sums[5][2][3] = {}, squares[5][2][3] = {};
  uint negative_weights[5] = {};
  for (uint i = 0; i < count; ++i) {
    require(values[i].w == 0 || values[i].w == 1, "Invalid or incomplete volume-event sample");
    const int type = int(values[i].w);
    for (int channel = 0; channel < 3; ++channel) {
      const float value = values[i][channel];
      require(std::isfinite(value) && (i % 5 >= 3 || value >= 0),
              "Invalid volume-event spectral weight");
      negative_weights[i % 5] += value < 0;
      sums[i % 5][type][channel] += value;
      squares[i % 5][type][channel] += double(value) * value;
    }
  }
  const double extinction[3] = {.3f, .8f, 1.4f}, scattering[3] = {.2f, .6f, .5f};
  for (int majorant = 0; majorant < 5; ++majorant) {
    if (majorant >= 3) {
      require(negative_weights[majorant] > 100, "Signed null transport was not exercised");
    }
    for (int type = 0; type < 2; ++type) {
      for (int channel = 0; channel < 3; ++channel) {
        const double transmittance = std::exp(-2 * extinction[channel]);
        const double expected = type ? scattering[channel] / extinction[channel] *
                                           (1 - transmittance) :
                                       transmittance;
        const double mean = sums[majorant][type][channel] / 65536;
        const double variance = std::max(squares[majorant][type][channel] / 65536 - mean * mean,
                                         0.0);
        const double tolerance = 6 * std::sqrt(variance / 65536) + 2e-4;
        std::printf(
            "BDPT_VOLUME_EVENT_MEAN majorant_case=%d event=%d channel=%d mean=%.9g expected=%.9g "
            "standard_error=%.9g\n",
            majorant,
            type,
            channel,
            mean,
            expected,
            std::sqrt(variance / 65536));
        require(std::abs(mean - expected) < tolerance,
                "BDPT volume-event mean disagrees with analytic spectral transport");
      }
    }
  }
  std::printf(
      "BDPT_VOLUME_EVENTS_METAL samples=%u majorants=5 survival=passed scattering=passed "
      "signed_weights=passed guard=passed\n",
      count);
}

static void test_bdpt_volume_piecewise_events(id<MTLDevice> device,
                                              id<MTLLibrary> library,
                                              id<MTLCommandQueue> queue)
{
  constexpr uint samples_per_mode = 131072;
  constexpr uint count = 4 * samples_per_mode;
  id<MTLBuffer> output = [device newBufferWithLength:count * sizeof(ccl::float4)
                                             options:MTLResourceStorageModeShared];
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline = [device
      newComputePipelineStateWithFunction:[library
                                              newFunctionWithName:@"bdpt_volume_piecewise_events"]
                                    error:&error];
  require(output && pipeline, "BDPT piecewise volume pipeline failed");
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:output offset:0 atIndex:0];
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  require(command.status == MTLCommandBufferStatusCompleted,
          "BDPT piecewise volume dispatch failed");

  double sums[4][3][3] = {}, squares[4][3][3] = {};
  uint collisions[4][3] = {}, negatives[4] = {};
  const auto *values = static_cast<const ccl::float4 *>(output.contents);
  for (uint i = 0; i < count; ++i) {
    require(std::isfinite(values[i].w) && values[i].w >= 0 && values[i].w < 768,
            "Invalid or incomplete piecewise volume sample");
    const uint mode = i % 4;
    const uint group = std::min(uint(values[i].w), 2u);
    ++collisions[mode][group];
    for (int channel = 0; channel < 3; ++channel) {
      const double value = values[i][channel];
      require(std::isfinite(value) && (mode >= 2 || value >= 0),
              "Invalid piecewise volume weight");
      negatives[mode] += value < 0;
      sums[mode][group][channel] += value;
      squares[mode][group][channel] += value * value;
    }
  }
  const double lengths[3] = {.7f, .8f, .5f};
  const double extinction[3][3] = {{1.1f, .9f, 1.3f}, {.4f, 1.4f, .8f}, {1.6f, .6f, 1.0f}};
  const double scattering[3][3] = {{.7f, .8f, .8f}, {.3f, 1.0f, .5f}, {1.0f, .4f, .8f}};
  for (int mode = 0; mode < 4; ++mode) {
    require(collisions[mode][2] > 1000, "Multiple real volume collisions were not exercised");
    require(mode < 2 || negatives[mode] > 100, "Piecewise signed transport was not exercised");
    for (int channel = 0; channel < 3; ++channel) {
      double tau_t = 0, tau_s = 0;
      for (int slab = 0; slab < 3; ++slab) {
        tau_t += lengths[slab] * extinction[slab][channel];
        tau_s += lengths[slab] * scattering[slab][channel];
      }
      // The ordered n-collision integral is exp(-tau_t) * tau_s^n / n!.
      // Sum n>=2 independently using exp(tau_s), including both slab orders.
      const double reference[3] = {std::exp(-tau_t),
                                   std::exp(-tau_t) * tau_s,
                                   std::exp(tau_s - tau_t) - std::exp(-tau_t) * (1 + tau_s)};
      for (int group = 0; group < 3; ++group) {
        const double mean = sums[mode][group][channel] / samples_per_mode;
        const double variance = std::max(
            squares[mode][group][channel] / samples_per_mode - mean * mean, 0.0);
        const double error = std::sqrt(variance / samples_per_mode);
        std::printf(
            "BDPT_VOLUME_PIECEWISE_MEAN mode=%d collisions=%d channel=%d mean=%.9g "
            "expected=%.9g standard_error=%.9g\n",
            mode,
            group,
            channel,
            mean,
            reference[group],
            error);
        require(std::abs(mean - reference[group]) < 6 * error + 2e-4,
                "Piecewise volume collision-order mean disagrees with analytic transport");
      }
    }
    std::printf(
        "BDPT_VOLUME_PIECEWISE_COUNTS mode=%d multiple_collisions=%u negative_weights=%u\n",
        mode,
        collisions[mode][2],
        negatives[mode]);
  }
  std::printf(
      "BDPT_VOLUME_PIECEWISE_METAL samples=%u modes=4 multiple_collisions=passed "
      "reversal=passed signed_weights=passed roulette=passed\n",
      count);
}

static void test_manifold_interface_samples(id<MTLDevice> device,
                                            id<MTLLibrary> library,
                                            id<MTLCommandQueue> queue)
{
  constexpr uint samples = 65536;
  id<MTLBuffer> output = [device newBufferWithLength:samples * sizeof(ccl::float4)
                                             options:MTLResourceStorageModeShared];
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline = [device
      newComputePipelineStateWithFunction:[library
                                              newFunctionWithName:@"manifold_interface_samples"]
                                    error:&error];
  require(output && pipeline, "Manifold random-sample pipeline failed");
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:output offset:0 atIndex:0];
  [encoder dispatchThreads:MTLSizeMake(samples, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  require(command.status == MTLCommandBufferStatusCompleted,
          "Manifold random-sample dispatch failed");
  const auto *values = static_cast<const ccl::float4 *>(output.contents);
  double sum = 0.0, squares = 0.0, cross = 0.0;
  for (uint i = 0; i < samples; ++i) {
    const double total = double(values[i].x) + values[i].y + values[i].z;
    sum += total;
    squares += total * total;
    cross += double(values[i].x) * values[i].y;
  }
  require(std::abs(sum / samples) < .003 && std::abs(squares / samples - .25) < .004 &&
              std::abs(cross / samples) < .002,
          "Correlated manifold-interface samples");
  std::printf("MANIFOLD_RANDOM_METAL samples=%u variance=%.9g cross=%.9g\n",
              samples,
              squares / samples,
              cross / samples);
}

static void test_manifold_bsdf_consistency(id<MTLDevice> device,
                                           id<MTLLibrary> library,
                                           id<MTLCommandQueue> queue)
{
  auto dispatch = [&](NSString *name, id<MTLBuffer> output, id<MTLBuffer> params, uint count) {
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:[library newFunctionWithName:name]
                                      error:&error];
    require(pipeline != nil, "Manifold BSDF pipeline failed");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:output offset:0 atIndex:0];
    if (params) {
      [encoder setBuffer:params offset:0 atIndex:1];
    }
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min(count, 64u), 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
      std::fprintf(stderr,
                   "Manifold BSDF Metal error: %s\n",
                   command.error.localizedDescription.UTF8String);
    }
    require(command.status == MTLCommandBufferStatusCompleted, "Manifold BSDF dispatch failed");
  };
  id<MTLBuffer> size = [device newBufferWithLength:sizeof(uint)
                                           options:MTLResourceStorageModeShared];
  require(size != nil, "Manifold BSDF size allocation failed");
  dispatch(@"manifold_bsdf_parameter_size", size, nil, 1);
  const uint parameter_size = *static_cast<const uint *>(size.contents);
  require(parameter_size > 0, "Manifold BSDF parameter size is zero");
  id<MTLBuffer> params = [device newBufferWithLength:parameter_size
                                             options:MTLResourceStorageModeShared];
  constexpr uint cases = 24, samples = 10000;
  id<MTLBuffer> output = [device newBufferWithLength:cases * samples * sizeof(ccl::float4)
                                             options:MTLResourceStorageModeShared];
  require(params && output, "Manifold BSDF allocation failed");
  std::memset(params.contents, 0, parameter_size);
  dispatch(@"manifold_bsdf_consistency", output, params, cases * samples);
  const auto *values = static_cast<const ccl::float4 *>(output.contents);
  uint inside[cases] = {};
  double maximum_error = 0;
  for (uint i = 0; i < cases * samples; ++i) {
    for (int c = 0; c < 3; ++c) {
      const double error = std::abs(double(values[i][c]) - 1);
      require(std::isfinite(error) && error < 2e-5,
              "Manifold transmission disagrees with camera BSDF on Metal");
      maximum_error = std::max(maximum_error, error);
    }
    require(std::isfinite(values[i].w), "Nonfinite manifold normal sample on Metal");
    inside[i % cases] += values[i].w <= .09f / 1.09f;
  }
  for (uint test = 0; test < cases; ++test) {
    const double expected = test / 6 >= 2 ? 1 - std::exp(-1.0) : .5;
    require(std::abs(double(inside[test]) / samples - expected) < 2.0 / samples,
            "Manifold normal distribution disagrees with analytic CDF on Metal");
  }
  std::printf("MANIFOLD_BSDF_METAL cases=%u samples_per_case=%u maximum_relative_error=%.9g\n",
              cases,
              samples,
              maximum_error);
}

static void test_compact_indices(id<MTLDevice> device,
                                 id<MTLLibrary> library,
                                 id<MTLCommandQueue> queue)
{
  NSError *error = nil;
  id<MTLFunction> function = [library newFunctionWithName:@"compact_indices_test"];
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                               error:&error];
  require(pipeline != nil && pipeline.threadExecutionWidth == 32,
          "Compaction needs a 32-lane SIMD group");
  std::mt19937 rng(93241);
  for (const uint size : {0u, 1u, 31u, 32u, 33u, 257u, 163840u}) {
    for (uint pattern = 0; pattern < 5; ++pattern) {
      std::vector<uint> expected(size + 32, 0xdeadbeefu);
      for (uint i = 0; i < size; ++i) {
        const bool keep = pattern == 0 || (pattern == 2 && i % 2 == 0) ||
                          (pattern == 3 && i >= size / 2) || (pattern == 4 && rng() % 7 != 0);
        expected[16 + i] = keep ? (i * 997u) ^ 0x21345678u : ~0u;
      }
      id<MTLBuffer> indices = [device newBufferWithBytes:expected.data()
                                                  length:expected.size() * sizeof(uint)
                                                 options:MTLResourceStorageModeShared];
      id<MTLBuffer> counts = [device newBufferWithLength:32 * sizeof(uint)
                                                 options:MTLResourceStorageModeShared];
      const uint expected_count = ccl::compact_indices(expected.data() + 16, size);
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:indices offset:0 atIndex:0];
      [encoder setBuffer:counts offset:0 atIndex:1];
      [encoder setBytes:&size length:sizeof(size) atIndex:2];
      [encoder dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      require(command.status == MTLCommandBufferStatusCompleted, "Compaction command failed");
      require(std::memcmp(indices.contents, expected.data(), expected.size() * sizeof(uint)) == 0,
              "SIMD compaction changed order, tail, or guard values");
      for (uint lane = 0; lane < 32; ++lane) {
        require(static_cast<uint *>(counts.contents)[lane] == expected_count,
                "SIMD compaction count mismatch");
      }
    }
  }
  printf(
      "BDPT_COMPACT_METAL sizes=7 patterns=5 stable_order=passed guards=passed "
      "all_lane_counts=passed\n");
}

int main(int argc, char **argv)
{
  @autoreleasepool {
    try {
      require(argc == 2, "Expected compiled Metal library path");
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      require(device != nil, "No Metal device available");
      NSError *error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                      error:&error];
      require(library != nil, error ? error.localizedDescription.UTF8String : "No library");
      id<MTLComputePipelineState> build = [device
          newComputePipelineStateWithFunction:[library newFunctionWithName:@"guiding_build"]
                                        error:&error];
      require(build != nil, "Cannot build guiding pipeline");
      id<MTLComputePipelineState> sample = [device
          newComputePipelineStateWithFunction:[library newFunctionWithName:@"guiding_sample"]
                                        error:&error];
      require(sample != nil, "Cannot build sampling pipeline");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      test_compact_indices(device, library, queue);
      constexpr int samples = 1000000;
      id<MTLBuffer> weights = [device newBufferWithLength:Tree::node_count * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
      id<MTLBuffer> counts = [device newBufferWithLength:Tree::node_count * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
      id<MTLBuffer> random = [device newBufferWithLength:samples * sizeof(ccl::float2)
                                                 options:MTLResourceStorageModeShared];
      id<MTLBuffer> results = [device newBufferWithLength:2 * samples * sizeof(ccl::float4)
                                                  options:MTLResourceStorageModeShared];
      require(weights && counts && random && results, "Metal allocation failed");
      auto *tree = static_cast<float *>(weights.contents);
      auto *random_values = static_cast<ccl::float2 *>(random.contents);
      auto *output = static_cast<ccl::float4 *>(results.contents);
      std::mt19937 rng(171);
      for (int i = 0; i < samples; ++i) {
        const float x = (float(rng() >> 9) + 0.5f) * 0x1p-23f;
        const float y = (float(rng() >> 9) + 0.5f) * 0x1p-23f;
        random_values[i] = ccl::make_float2(x, y);
      }
      for (uint test = 0; test < 12; ++test) {
        const ccl::GuidingDirectionalProduct product = {
            ccl::normalize(ccl::make_float3(1, 2, -3)),
            test == 6 ? -0.65f : 0.65f,
            test >= 6 ? ccl::GuidingDirectionalProduct::PHASE :
            test == 5 ? ccl::GuidingDirectionalProduct::TWO_SIDED_COSINE :
                        ccl::GuidingDirectionalProduct::COSINE};
        std::fill(tree, tree + Tree::node_count, 0.0f);
        auto *count_values = static_cast<float *>(counts.contents);
        std::fill(count_values, count_values + Tree::node_count, 0.0f);
        for (int i = 0; i < Tree::leaf_count; ++i) {
          tree[Tree::leaf_offset + i] = test == 9     ? 1.0f + (i % 11) :
                                        test % 3 == 0 ? 0.0f :
                                        test % 3 == 1 ? 1.0f + (i % 11) :
                                                        (i == 147 ? 1e30f : 0.0f);
          if (tree[Tree::leaf_offset + i] > 0.0f) {
            count_values[Tree::leaf_offset + i] = test == 10 ? (i % 17 == 0 ? 64 : 0) : 1;
          }
        }
        std::array<float, Tree::node_count> host_tree;
        std::copy(tree, tree + Tree::node_count, host_tree.begin());
        Tree distribution;
        std::array<float, Tree::node_count> host_counts;
        std::copy(count_values, count_values + Tree::node_count, host_counts.begin());
        if (test >= 9) {
          distribution.build_adaptive(host_tree.data(), host_counts.data(), 32.0f, 0.05f);
        }
        else {
          distribution.build(host_tree.data(), 0.05f);
        }
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:build];
        [encoder setBuffer:weights offset:0 atIndex:0];
        [encoder setBuffer:counts offset:0 atIndex:1];
        [encoder setBytes:&test length:sizeof(test) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:sample];
        [encoder setBuffer:weights offset:0 atIndex:0];
        [encoder setBuffer:random offset:0 atIndex:1];
        [encoder setBuffer:results offset:0 atIndex:2];
        [encoder setBytes:&test length:sizeof(test) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(samples, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "Metal command failed");
        for (int i = 0; i < Tree::node_count; ++i) {
          require(std::abs(tree[i] - host_tree[i]) <= 2e-5f * std::max(1.0f, host_tree[i]),
                  "Host/device tree build mismatch");
        }
        std::array<int, Tree::leaf_count> histogram{};
        double integral = 0.0;
        double integral_squared = 0.0;
        int boundary_mismatches = 0;
        for (int i = 0; i < samples; ++i) {
          const ccl::float3 direction = ccl::make_float3(output[2 * i]);
          const float pdf = output[2 * i].w;
          require(ccl::isfinite_safe(direction) && std::isfinite(pdf) && pdf > 0.0f,
                  "Nonfinite sample or invalid density");
          require(std::abs(ccl::len(direction) - 1.0f) < 2e-6f, "Nonunit direction");
          if (std::abs(pdf - output[2 * i + 1].x) > 2e-5f * std::max(pdf, 1.0f)) {
            ++boundary_mismatches;
          }
          ++histogram[distribution.leaf_index(direction) - Tree::leaf_offset];
          const double estimate = double(direction.z * direction.z) / pdf;
          integral += estimate;
          integral_squared += estimate * estimate;
        }
        double chi_squared = 0.0;
        for (int y = 0; y < Tree::resolution; ++y) {
          for (int x = 0; x < Tree::resolution; ++x) {
            const ccl::float3 direction = distribution.square_to_direction(
                ccl::make_float2((x + 0.5f) / Tree::resolution, (y + 0.5f) / Tree::resolution));
            const int i = distribution.leaf_index(direction) - Tree::leaf_offset;
            const float pdf = test >= 3 ? distribution.pdf_product(tree, direction, product) :
                                          distribution.pdf(tree, direction);
            const double expected = samples * double(pdf) * (4.0 * M_PI / Tree::leaf_count);
            chi_squared += (histogram[i] - expected) * (histogram[i] - expected) / expected;
          }
        }
        std::printf(
            "GUIDING_METAL device=%s case=%d samples=%d gpu_seconds=%.9g "
            "chi_squared=%.9g integral_z2=%.9g pdf_mismatches=%d\n",
            device.name.UTF8String,
            test,
            samples,
            command.GPUEndTime - command.GPUStartTime,
            chi_squared,
            integral / samples,
            boundary_mismatches);
        require(boundary_mismatches == 0, "Sample/evaluated density mismatch");
        require(chi_squared < 1250.0, "Sampling frequencies disagree with density");
        const double standard_error = std::sqrt(
            std::max(0.0, integral_squared / samples - std::pow(integral / samples, 2)) / samples);
        require(std::abs(integral / samples - 4.0 * M_PI / 3.0) < 6 * standard_error,
                "Importance-sampled integral disagrees with analytic result");
      }
      id<MTLComputePipelineState> gaussian = [device
          newComputePipelineStateWithFunction:[library
                                                  newFunctionWithName:@"guiding_gaussian_sample"]
                                        error:&error];
      require(gaussian != nil, "Cannot build spherical Gaussian pipeline");
      const float concentrations[] = {0.0f, 1e-5f, 0.01f, 1.0f, 32.0f, 16384.0f};
      id<MTLBuffer> smooth_storage = [device
          newBufferWithLength:ccl::GuidingGaussianMixture::storage_size * sizeof(float)
                      options:MTLResourceStorageModeShared];
      require(smooth_storage != nil, "Cannot allocate smooth mixture");
      std::array<float, Tree::node_count> smooth_tree{};
      for (int i = 0; i < Tree::leaf_count; ++i) {
        smooth_tree[Tree::leaf_offset + i] = 1.0f + float(i % 17);
      }
      Tree directional;
      directional.build(smooth_tree.data(), 0.05f);
      ccl::GuidingGaussianMixture mixture;
      auto *smooth_values = static_cast<float *>(smooth_storage.contents);
      mixture.build(smooth_values, smooth_tree.data());
      for (uint test = 0; test < 12; ++test) {
        if (test == 11) {
          mixture.build(smooth_values, smooth_tree.data());
          for (int i = 0; i < ccl::GuidingGaussianMixture::components; ++i) {
            float *entry = smooth_values + i * ccl::GuidingGaussianMixture::component_stride;
            for (int j = 0; j < 3; ++j) {
              entry[5 + j] = 2 * entry[2 + j];
              entry[14 + j] = 1;
            }
            for (int j = 8; j < 14; ++j) {
              entry[j] = j == 8 || j == 11 || j == 13 ? .03f : 0;
            }
            entry[17] = 2;
            entry[18] = .02f;
          }
        }
        if (test == 9) {
          for (int i = 0; i < ccl::GuidingGaussianMixture::components; ++i) {
            float *entry = smooth_values + i * ccl::GuidingGaussianMixture::component_stride;
            for (int j = 2; j <= 4; ++j) {
              entry[j] *= .9f;
            }
            entry[5] = entry[6] = entry[7] = .5f;
            entry[8] = .4f;
            entry[12] = -.3f;
            entry[16] = .2f;
            entry[17] = 1.0f;
          }
        }
        const ccl::float3 position = test == 9  ? ccl::make_float3(.2f, .3f, .4f) :
                                     test == 10 ? ccl::make_float3(.8f, .7f, .6f) :
                                                  ccl::zero_float3();
        const ccl::GuidingGaussianProduct profile{
            {{ccl::normalize(ccl::make_float3(1, 2, 3)), test == 8 ? 16384.0f : 5.0f},
             {ccl::normalize(ccl::make_float3(-2, 1, -3)), 30.0f}},
            {test == 8 ? 1.0f : 0.3f, test == 8 ? 0.0f : 0.7f}};
        ccl::GuidingSphericalGaussian distribution{ccl::normalize(ccl::make_float3(1, -2, 3)),
                                                   concentrations[std::min(test, 5u)]};
        if (test == 6) {
          const ccl::GuidingSphericalGaussian a{ccl::normalize(ccl::make_float3(1, 2, 3)), 5.0f};
          const ccl::GuidingSphericalGaussian b{ccl::normalize(ccl::make_float3(-2, 1, 3)), 12.0f};
          float integral;
          distribution = a.product(b, &integral);
        }
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:gaussian];
        [encoder setBuffer:random offset:0 atIndex:0];
        [encoder setBuffer:results offset:0 atIndex:1];
        [encoder setBytes:&test length:sizeof(test) atIndex:2];
        [encoder setBuffer:smooth_storage offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(samples, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
                "Gaussian Metal command failed");
        double mean_cosine = 0;
        for (int i = 0; i < samples; ++i) {
          const ccl::float3 direction = ccl::make_float3(output[2 * i]);
          const float pdf = output[2 * i].w;
          require(ccl::isfinite_safe(direction) && std::isfinite(pdf) && pdf > 0,
                  "Invalid Gaussian sample");
          require(std::abs(ccl::len(direction) - 1.0f) < 2e-6f, "Nonunit Gaussian direction");
          require(std::abs(pdf - output[2 * i + 1].x) <= 2e-5f * pdf,
                  "Gaussian sample/PDF mismatch");
          const float host_pdf = test >= 7 ? mixture.pdf_product(
                                                 smooth_values, profile, direction, position) :
                                             distribution.pdf(direction);
          require(std::abs(pdf - host_pdf) <= 0.01f * std::max(pdf, host_pdf),
                  "Gaussian host/device PDF mismatch");
          if (test >= 7) {
            const auto other = ccl::normalize(ccl::make_float3(1, -2, 1));
            const float expected[3] = {
                mixture.pdf_product(smooth_values, profile, other, position),
                mixture.pdf(smooth_values, other, position),
                mixture.pdf(smooth_values, direction, position)};
            for (int j = 0; j < 3; ++j) {
              const float actual = output[2 * i + 1][j + 1];
              require(std::isfinite(actual) &&
                          std::abs(actual - expected[j]) <= .01f * std::max(expected[j], 1e-7f),
                      "Fused guiding PDF disagrees with independent CPU query");
            }
          }
          mean_cosine += ccl::dot(distribution.axis, direction);
        }
        const double k = distribution.concentration;
        double expected = k < 1e-4 ? k / 3 : 1.0 / std::tanh(k) - 1.0 / k;
        if (test >= 7) {
          double mass = 0.0, moment = 0.0;
          for (int i = 0; i < ccl::GuidingGaussianMixture::components; ++i) {
            for (int j = 0; j < 2; ++j) {
              float integral;
              const auto product = mixture.component(smooth_values, i, position)
                                       .product(profile.lobes[j], &integral);
              const double weight =
                  smooth_values[ccl::GuidingGaussianMixture::component_stride * i] *
                  profile.weights[j] * integral;
              const double pk = product.concentration;
              const double mean = pk < 1e-4 ? pk / 3 : 1.0 / std::tanh(pk) - 1.0 / pk;
              mass += weight;
              moment += weight * mean * ccl::dot(distribution.axis, product.axis);
            }
          }
          expected = (1.0 - ccl::GuidingGaussianMixture::exploration) * moment / mass;
        }
        require(std::abs(mean_cosine / samples - expected) <
                    6 * std::sqrt((1 - expected * expected) / samples),
                "Gaussian sampling moment mismatch");
        std::printf("GUIDING_GAUSSIAN_METAL case=%u samples=%d mean_cosine=%.9g expected=%.9g\n",
                    test,
                    samples,
                    mean_cosine / samples,
                    expected);
      }
      id<MTLComputePipelineState> resampling = [device
          newComputePipelineStateWithFunction:[library newFunctionWithName:@"guiding_resample"]
                                        error:&error];
      require(resampling != nil, "Cannot build resampling pipeline");
      id<MTLCommandBuffer> resampling_command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> resampling_encoder = [resampling_command computeCommandEncoder];
      [resampling_encoder setComputePipelineState:resampling];
      [resampling_encoder setBuffer:random offset:0 atIndex:0];
      [resampling_encoder setBuffer:results offset:0 atIndex:1];
      [resampling_encoder dispatchThreads:MTLSizeMake(samples, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      [resampling_encoder endEncoding];
      [resampling_command commit];
      [resampling_command waitUntilCompleted];
      require(resampling_command.status == MTLCommandBufferStatusCompleted,
              "Metal resampling execution failed");
      const auto *estimates = static_cast<const float *>(results.contents);
      double sum = 0, sum_squared = 0;
      for (int i = 0; i < samples; ++i) {
        require(std::isfinite(estimates[i]), "Nonfinite resampling estimate");
        sum += estimates[i];
        sum_squared += double(estimates[i]) * estimates[i];
      }
      const double mean = sum / samples;
      const double variance = std::max(0.0, sum_squared / samples - mean * mean);
      require(std::abs(mean - 9.9) < 6.0 * std::sqrt(variance / samples) + 1e-5,
              "Biased Metal RIS and directional MIS composition");
      std::printf("GUIDING_RESAMPLING_METAL samples=%d mean=%.9g expected=9.9\n", samples, mean);
      test_field(device, library, queue);
      test_history(device, library, queue);
      test_position(device, library, queue);
      test_parallax(device, library, queue);
      test_hdr(device, library, queue);
      test_mixture_fit(device, library, queue, false, 4096, 32);
      test_mixture_fit(device, library, queue, true, 4096, 32);
      test_mixture_fit(device, library, queue, true, 4093);
      test_mixture_fit(device, library, queue, true, 17);
      test_mixture_fit<16>(device, library, queue, false, 4096, 8);
      test_mixture_fit<16>(device, library, queue, true, 4096, 8);
      test_mixture_fit<16>(device, library, queue, true, 4093);
      test_mixture_sources(device, library, queue);
      test_mixture_sources<16>(device, library, queue);
      test_mixture_sources<16, true>(device, library, queue);
      test_observation_partition(device, library, queue);
      test_bdpt_mis_recurrence(device, library, queue);
      test_bdpt_mis_connection_weights(device, library, queue);
      test_bdpt_volume_homogeneous_events(device, library, queue);
      test_bdpt_volume_piecewise_events(device, library, queue);
      test_manifold_interface_samples(device, library, queue);
      test_manifold_bsdf_consistency(device, library, queue);
      return 0;
    }
    catch (const std::exception &error) {
      std::fprintf(stderr, "GUIDING_METAL_FAILURE %s\n", error.what());
      return 1;
    }
  }
}
