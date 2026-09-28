/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "kernel/closure/bsdf_diffraction.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <array>
#include <chrono>

using namespace ccl;

/* Keep this estimator identical to the host prototype. All facet sampling,
 * order geometry/power, and masking remain calls into native Cycles headers. */
static float reference_albedo(const DiffractionReflection &p, const float3 wi, int samples)
{
  const float li = bsdf_aniso_lambda<GGX>(p.alpha_x, p.alpha_y, wi);
  double sum = 0;
  for (int s = 0; s < samples; ++s) {
    const float u = (s + .5f) / samples;
    const float v = (s + .5f) * .6180339887498949f;
    const float3 h = microfacet_ggx_sample_vndf(
        wi, p.alpha_x, p.alpha_y, make_float2(u, v - std::floor(v)));
    const float ci = dot(wi, h);
    float nonzero_mass = 0;
    const int limit = diffraction_reflection_max_order(&p);
    for (int order = -limit; order <= limit; ++order) {
      float3 wo;
      if (order == 0 || !diffraction_facet_reflect(
                            wi, h, make_float3(1, 0, 0), order * p.wavelength_over_pitch, &wo)) {
        continue;
      }
      const float mass = diffraction_reflection_nonzero_power(&p, order, ci, dot(wo, h));
      nonzero_mass += mass;
      if (wo.z > 0) {
        const float lo = bsdf_aniso_lambda<GGX>(p.alpha_x, p.alpha_y, wo);
        sum += mass * (1 + li) / (1 + li + lo);
      }
    }
    const float3 wo = 2 * ci * h - wi;
    if (wo.z > 0) {
      const float lo = bsdf_aniso_lambda<GGX>(p.alpha_x, p.alpha_y, wo);
      sum += std::max(0.0f, 1 - nonzero_mass) * (1 + li) / (1 + li + lo);
    }
  }
  return float(sum / samples);
}

/* Independent order-sampling quadrature, instead of the builder's exact order
 * sum. Its finite deterministic error is not a rigorous albedo bound. */
static float sampled_order_reference(const DiffractionReflection &p, const float3 wi)
{
  constexpr int samples = 16384;
  const float li = bsdf_aniso_lambda<GGX>(p.alpha_x, p.alpha_y, wi);
  double sum = 0;
  for (int s = 0; s < samples; ++s) {
    const float u = (s + .5f) / samples;
    const float v = (s + .5f) * .6180339887498949f;
    const float t = (s + .5f) * .7548776662466927f;
    const float3 h = microfacet_ggx_sample_vndf(
        wi, p.alpha_x, p.alpha_y, make_float2(u, v - std::floor(v)));
    float3 wo;
    float mass;
    if (diffraction_reflection_sample_order(&p, wi, h, t - std::floor(t), &wo, &mass) &&
        wo.z > 0) {
      const float lo = bsdf_aniso_lambda<GGX>(p.alpha_x, p.alpha_y, wo);
      sum += (1 + li) / (1 + li + lo);
    }
  }
  return float(sum / samples);
}

static float angular_lookup(const float *table, const unsigned mu_count,
                            const unsigned phi_count, const float3 w)
{
  const float mu = std::clamp(w.z, 0.0f, 1.0f);
  const float u = std::sqrt(mu) * (mu_count - 1);
  const unsigned i = std::min(unsigned(std::floor(u)), mu_count - 1);
  const unsigned k = std::min(i + 1, mu_count - 1);
  const float v = (std::atan2(w.y, w.x) / M_2PI_F + 1) * phi_count - .5f;
  const int j = (int(std::floor(v)) % int(phi_count) + int(phi_count)) % int(phi_count);
  const unsigned l = (j + 1) % phi_count;
  const float mi = float(i) / (mu_count - 1), mk = float(k) / (mu_count - 1);
  const float a = i == k ? 0 : std::clamp((mu - mi * mi) / (mk * mk - mi * mi), 0.0f, 1.0f);
  const float b = v - std::floor(v);
  return (1 - b) * ((1 - a) * table[i * phi_count + j] + a * table[k * phi_count + j]) +
         b * ((1 - a) * table[i * phi_count + l] + a * table[k * phi_count + l]);
}

int main(int argc, char **argv)
{
  if (argc < 2 || argc > 3 || (argc == 3 && strcmp(argv[2], "--compile-only"))) {
    fprintf(stderr, "usage: %s expanded.metal [--compile-only]\n", argv[0]);
    return 2;
  }
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return 2;
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
                                                   encoding:NSUTF8StringEncoding
                                                      error:&error];
    if (!source) return 2;
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
      fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 2;
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"diffraction_multiscatter_albedo"] error:&error];
    if (!pipeline) {
      fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 2;
    }
    id<MTLComputePipelineState> average_pipeline = [device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"diffraction_multiscatter_average"] error:&error];
    if (!average_pipeline) {
      fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 2;
    }
    if (argc == 3) {
      puts("Metal library and albedo pipeline compiled; no GPU dispatch");
      return 0;
    }

    constexpr unsigned mu_count = 8, phi_count = 16, count = mu_count * phi_count;
    constexpr unsigned samples = 512;
    std::vector<float4> directions;
    directions.reserve(count);
    for (unsigned m = 0; m < mu_count; ++m) {
      const float coordinate = float(m) / (mu_count - 1);
      const float mu = std::max(0.00001f, coordinate * coordinate);
      for (unsigned j = 0; j < phi_count; ++j) {
        const float phi = M_2PI_F * (j + .5f) / phi_count;
        const float r = std::sqrt(1 - mu * mu);
        directions.push_back(make_float4(r * std::cos(phi), r * std::sin(phi), mu, 0));
      }
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> size_buffer = [device newBufferWithLength:sizeof(unsigned)
                                                   options:MTLResourceStorageModeShared];
    id<MTLComputePipelineState> size_pipeline = [device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"diffraction_multiscatter_context_size"] error:&error];
    if (!size_pipeline) return 2;
    id<MTLCommandBuffer> size_command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> size_encoder = [size_command computeCommandEncoder];
    [size_encoder setComputePipelineState:size_pipeline];
    [size_encoder setBuffer:size_buffer offset:0 atIndex:0];
    [size_encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [size_encoder endEncoding];
    [size_command commit];
    [size_command waitUntilCompleted];
    if (size_command.status != MTLCommandBufferStatusCompleted) return 2;
    const unsigned context_size = *(const unsigned *)size_buffer.contents;
    if (context_size == 0 || context_size > 16 * 1024 * 1024) return 2;
    id<MTLBuffer> params = [device newBufferWithLength:context_size
                                               options:MTLResourceStorageModeShared];
    memset(params.contents, 0, context_size);
    id<MTLBuffer> input = [device newBufferWithBytes:directions.data()
                                             length:directions.size() * sizeof(float4)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> sample_buffer = [device newBufferWithBytes:&samples length:sizeof(samples)
                                                    options:MTLResourceStorageModeShared];
    int failures = 0;
    float max_error = 0;
    std::vector<std::array<float, 5>> profiles = {
        {.25f, .4f, .45f, 0.0f, .42f}, {.25f, .4f, .45f, .4f, .42f},
        {.05f, .05f, .45f, .4f, .42f}, {.8f, .8f, .45f, .4f, .42f},
        {.08f, .7f, .45f, .4f, .42f}, {.5f, .12f, .2f, .4f, .15f}};
    /* One physical profile over the visible domain, not independent arbitrary
     * wavelength/depth ratios. This validates dispatch layout, not spectral
     * interpolation accuracy or an adequate number of wavelength slices. */
    for (unsigned slice = 0; slice < 16; ++slice) {
      const float wavelength_nm = 380.0f + (400.0f * slice) / 15.0f;
      profiles.push_back({.25f, .4f, wavelength_nm / 1600.0f,
                          150.0f / wavelength_nm, .41f});
    }
    static_assert(sizeof(profiles[0]) == 5 * sizeof(float));
    id<MTLBuffer> output = [device newBufferWithLength:profiles.size() * count * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> parameter_buffer = [device newBufferWithBytes:profiles.data()
        length:profiles.size() * sizeof(profiles[0]) options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:params offset:0 atIndex:2];
    [encoder setBuffer:parameter_buffer offset:0 atIndex:3];
    [encoder setBuffer:sample_buffer offset:0 atIndex:4];
    [encoder setBytes:&count length:sizeof(count) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(count, profiles.size(), 1)
        threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
    [encoder endEncoding];
    id<MTLBuffer> averages = [device newBufferWithLength:profiles.size() * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    id<MTLComputeCommandEncoder> average_encoder = [command computeCommandEncoder];
    const unsigned shape[2] = {mu_count, phi_count};
    [average_encoder setComputePipelineState:average_pipeline];
    [average_encoder setBuffer:output offset:0 atIndex:0];
    [average_encoder setBuffer:averages offset:0 atIndex:1];
    [average_encoder setBytes:shape length:sizeof(shape) atIndex:2];
    [average_encoder dispatchThreadgroups:MTLSizeMake(profiles.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(average_pipeline.threadExecutionWidth, 1, 1)];
    [average_encoder endEncoding];
    const auto dispatch_start = std::chrono::steady_clock::now();
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 2;
    const double dispatch_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - dispatch_start).count();
    float max_average_error = 0;
    NSMutableArray *reports = [NSMutableArray array];
    for (size_t profile_index = 0; profile_index < profiles.size(); ++profile_index) {
      const auto &p = profiles[profile_index];
      const DiffractionReflection native_p{p[0], p[1], p[2], p[3], p[4]};
      const auto cpu_start = std::chrono::steady_clock::now();
      const float *actual = (const float *)output.contents + profile_index * count;
      float profile_error = 0;
      for (unsigned i = 0; i < count; ++i) {
        const float3 wi = make_float3(directions[i].x, directions[i].y, directions[i].z);
        const float expected = reference_albedo(native_p, wi, samples);
        const float error = std::abs(actual[i] - expected);
        max_error = std::max(max_error, error);
        profile_error = std::max(profile_error, error);
        failures += !std::isfinite(actual[i]) || error > 2e-3f;
      }
      double average_reference = 0;
      for (unsigned i = 0; i + 1 < mu_count; ++i) for (unsigned j = 0; j < phi_count; ++j) {
        const double x = double(i) / (mu_count - 1), y = double(i + 1) / (mu_count - 1);
        const double mu = x * x, h = y * y - mu;
        const double a = actual[i * phi_count + j];
        const double delta = actual[(i + 1) * phi_count + j] - a;
        average_reference += 2 * h * (mu * a + (mu * delta + h * a) / 2 + h * delta / 3) / phi_count;
      }
      const float average_actual = ((const float *)averages.contents)[profile_index];
      const float average_error = std::abs(average_actual - average_reference);
      max_average_error = std::max(max_average_error, average_error);
      failures += !std::isfinite(average_actual) || average_actual < 0 || average_actual > 1 ||
                  average_error > 1e-6f;
      const double cpu_seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - cpu_start).count();
      [reports addObject:@{@"alpha_x":@(p[0]), @"alpha_y":@(p[1]),
          @"wavelength_over_pitch":@(p[2]), @"depth_over_wavelength":@(p[3]),
          @"duty":@(p[4]), @"max_cpu_metal_abs_error":@(profile_error),
          @"projected_average":@(average_actual), @"average_absolute_error":@(average_error),
          @"cpu_reference_and_comparison_seconds":@(cpu_seconds)}];
    }
    /* Midpoints do not coincide with stored wavelengths. Check every angular
     * node against a separate native sampled-order reference. */
    float spectral_error = 0, spectral_underestimate = 0;
    unsigned spectral_count = 0;
    const float *tables = (const float *)output.contents;
    for (unsigned slice = 0; slice < 15; ++slice) {
      const float wavelength_nm = 380.0f + 400.0f * (slice + .5f) / 15.0f;
      const DiffractionReflection p{.25f, .4f, wavelength_nm / 1600.0f,
                                    150.0f / wavelength_nm, .41f};
      for (unsigned i = 0; i < count; ++i) {
        const float estimate = .5f * (tables[(6 + slice) * count + i] +
                                     tables[(7 + slice) * count + i]);
        const float3 wi = make_float3(directions[i].x, directions[i].y, directions[i].z);
        const float reference = sampled_order_reference(p, wi);
        spectral_error = std::max(spectral_error, std::abs(reference - estimate));
        spectral_underestimate = std::max(spectral_underestimate, reference - estimate);
        ++spectral_count;
      }
    }
    float combined_error = 0, combined_underestimate = 0;
    unsigned combined_count = 0;
    for (unsigned slice = 0; slice < 15; ++slice) {
      const float wavelength_nm = 380.0f + 400.0f * (slice + .5f) / 15.0f;
      const DiffractionReflection p{.25f, .4f, wavelength_nm / 1600.0f,
                                    150.0f / wavelength_nm, .41f};
      for (unsigned m = 0; m < 10; ++m) for (unsigned j = 0; j < 17; ++j) {
        const float mu = m == 0 ? .01f : (m == 9 ? .999999f : float(m) / 9);
        const float phi = M_2PI_F * (j + .37f) / 17;
        const float radius = std::sqrt(1 - mu * mu);
        const float3 wi = make_float3(radius * std::cos(phi), radius * std::sin(phi), mu);
        const float estimate = .5f * (
            angular_lookup(tables + (6 + slice) * count, mu_count, phi_count, wi) +
            angular_lookup(tables + (7 + slice) * count, mu_count, phi_count, wi));
        const float reference = sampled_order_reference(p, wi);
        if (!std::isfinite(reference) || !std::isfinite(estimate)) ++failures;
        combined_error = std::max(combined_error, std::abs(reference - estimate));
        combined_underestimate = std::max(combined_underestimate, reference - estimate);
        ++combined_count;
      }
    }
    constexpr float spectral_gate = .01f;
    NSDictionary *report = @{@"nodes":@(count), @"samples":@(samples),
        @"profiles":reports, @"max_cpu_metal_abs_error":@(max_error), @"failures":@(failures),
        @"device":device.name,
        @"batch_gpu_dispatch_and_wait_seconds":@(dispatch_seconds),
        @"batch_gpu_command_seconds":@(command.GPUEndTime-command.GPUStartTime),
        @"profile_count":@(profiles.size()), @"spectral_slices":@16,
        @"max_projected_average_error":@(max_average_error),
        @"projected_average_error_gate":@1e-6,
        @"combined_off_node_checks":@(combined_count),
        @"combined_max_absolute_error":@(combined_error),
        @"combined_max_albedo_underestimate":@(combined_underestimate),
        @"combined_off_node_passed":@(combined_error <= spectral_gate),
        @"spectral_midpoint_checks":@(spectral_count),
        @"spectral_reference_samples":@16384,
        @"spectral_max_absolute_error":@(spectral_error),
        @"spectral_max_albedo_underestimate":@(spectral_underestimate),
        @"spectral_absolute_error_gate":@(spectral_gate),
        @"spectral_midpoint_passed":@(spectral_error <= spectral_gate),
        @"scope":@"Same-estimator portability and batched build and projected-average reduction in one command buffer; excludes compilation and scene integration; angular and wavelength interpolation checked for one profile; not a global physical acceptance claim"};
    NSData *json = [NSJSONSerialization dataWithJSONObject:report
        options:NSJSONWritingPrettyPrinted error:&error];
    if (!json) return 2;
    fwrite(json.bytes, 1, json.length, stdout); puts("");
    return failures || spectral_error > spectral_gate || combined_error > spectral_gate ? 1 : 0;
  }
}
