/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "test/diffraction_reference_fixture.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

static bool run_case(id<MTLDevice> device,
                     id<MTLLibrary> library,
                     id<MTLCommandQueue> queue,
                     const double pitch,
                     const std::complex<double> material,
                     const int retained,
                     const unsigned evaluations,
                     NSMutableArray *reports,
                     const int feedback = -1)
{
  constexpr unsigned cases = 64;
  ccl::DiffractionReferenceFixture fixture;
  std::string message;
  const auto start = std::chrono::steady_clock::now();
  if (!ccl::diffraction_reference_fixture(pitch,
                                          material,
                                          retained,
                                          cases,
                                          fixture,
                                          message,
                                          feedback < 0 ? -1 : feedback / 2,
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
                                          true
#else
                                          false
#endif
                                          ))
  {
    fprintf(stderr, "Fixture generation: %s\n", message.c_str());
    return false;
  }
  const double fixture_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  const int channels = fixture.channels;
  NSError *error = nil;
  id<MTLFunction> function = [library
      newFunctionWithName:feedback < 0 ?
                              [NSString stringWithFormat:@"diffraction_reference_%d", channels] :
                              [NSString stringWithFormat:@"diffraction_hybrid_%d_%d",
                                                         channels,
                                                         feedback]];
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                               error:&error];
  id<MTLFunction> bench_function = [library
      newFunctionWithName:feedback < 0 ?
                              [NSString
                                  stringWithFormat:@"diffraction_reference_bench_%d", channels] :
                              [NSString stringWithFormat:@"diffraction_hybrid_bench_%d_%d",
                                                         channels,
                                                         feedback]];
  id<MTLComputePipelineState> bench_pipeline = [device
      newComputePipelineStateWithFunction:bench_function
                                    error:&error];
  if (!pipeline || !bench_pipeline) {
    fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
    return false;
  }
  id<MTLBuffer> charts = [device newBufferWithBytes:fixture.charts.data()
                                             length:fixture.charts.size() * sizeof(ccl::float2)
                                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> boundaries = [device
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
      newBufferWithBytes:fixture.boundary_inputs.data()
                  length:fixture.boundary_inputs.size() * sizeof(ccl::float4)
#else
      newBufferWithBytes:fixture.boundaries.data()
                  length:fixture.boundaries.size() * sizeof(ccl::float2)
#endif
                 options:MTLResourceStorageModeShared];
  id<MTLBuffer> rotations = [device
      newBufferWithBytes:fixture.rotations.data()
                  length:fixture.rotations.size() * sizeof(ccl::float2)
                 options:MTLResourceStorageModeShared];
  const int dummy_active = 0;
  id<MTLBuffer> active = [device
      newBufferWithBytes:fixture.active_ports.empty() ? &dummy_active : fixture.active_ports.data()
                  length:std::max(size_t(1), fixture.active_ports.size()) * sizeof(int)
                 options:MTLResourceStorageModeShared];
  id<MTLBuffer> incoming = [device newBufferWithBytes:fixture.incoming.data()
                                               length:fixture.incoming.size() * sizeof(int)
                                              options:MTLResourceStorageModeShared];
  id<MTLBuffer> outputs = [device newBufferWithLength:fixture.expected.size() * sizeof(ccl::float2)
                                              options:MTLResourceStorageModeShared];
  id<MTLBuffer> valid = [device newBufferWithLength:cases * sizeof(unsigned)
                                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> powers = [device newBufferWithLength:evaluations * sizeof(float)
                                             options:MTLResourceStorageModeShared];
  auto dispatch = [&](const bool benchmark, const unsigned count, double &milliseconds) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    id<MTLComputePipelineState> state = benchmark ? bench_pipeline : pipeline;
    [encoder setComputePipelineState:state];
    [encoder setBuffer:charts offset:0 atIndex:0];
    [encoder setBuffer:boundaries offset:0 atIndex:1];
    [encoder setBuffer:feedback < 0 ? rotations : active offset:0 atIndex:2];
    [encoder setBuffer:incoming offset:0 atIndex:3];
    [encoder setBuffer:benchmark ? powers : outputs offset:0 atIndex:4];
    if (benchmark) {
      [encoder setBytes:&cases length:sizeof(cases) atIndex:5];
    }
    else {
      [encoder setBuffer:valid offset:0 atIndex:5];
    }
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(state.threadExecutionWidth, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
      fprintf(stderr, "GPU failure: %s\n", command.error.localizedDescription.UTF8String);
      return false;
    }
    milliseconds = 1000.0 * (command.GPUEndTime - command.GPUStartTime);
    return true;
  };
  double elapsed;
  if (!dispatch(false, cases, elapsed)) {
    return false;
  }
  const ccl::float2 *actual = static_cast<const ccl::float2 *>(outputs.contents);
  const unsigned *flags = static_cast<const unsigned *>(valid.contents);
  unsigned failures = 0;
  float maximum_error = 0.0f, maximum_power_error = 0.0f;
  double maximum_polarization_gain = 0.0;
  for (unsigned i = 0; i < cases; i++) {
    if (!flags[i]) {
      fprintf(stderr, "Matching failed: channels=%d case=%u\n", channels, i);
      failures++;
    }
    for (int j = 0; j < 2 * channels; j++) {
      const size_t offset = size_t(i) * 2 * channels + j;
      const float difference = ccl::len(actual[offset] - fixture.expected[offset]);
      maximum_error = std::max(maximum_error, difference);
      if (!std::isfinite(difference) || difference > 3e-5f) {
        if (failures < 8) {
          fprintf(stderr,
                  "Matching mismatch: channels=%d case=%u coefficient=%d error=%.9g\n",
                  channels,
                  i,
                  j,
                  difference);
        }
        failures++;
      }
    }
    double ss = 0.0, pp = 0.0;
    std::complex<double> sp = 0.0;
    for (int row = 0; row < channels; row++) {
      const auto a = actual[size_t(i) * 2 * channels + 2 * row];
      const auto b = actual[size_t(i) * 2 * channels + 2 * row + 1];
      const std::complex<double> ca(a.x, a.y), cb(b.x, b.y);
      ss += std::norm(ca);
      pp += std::norm(cb);
      sp += std::conj(ca) * cb;
    }
    const double gain = 0.5 * (ss + pp + std::sqrt((ss - pp) * (ss - pp) + 4.0 * std::norm(sp)));
    maximum_polarization_gain = std::max(maximum_polarization_gain, gain);
    if (!std::isfinite(gain) || gain > 1.0 + 1e-4) {
      failures++;
    }
    for (int col = 0; col < 2; col++) {
      float power = 0.0f, expected_power = 0.0f;
      for (int row = 0; row < channels; row++) {
        const size_t offset = size_t(i) * 2 * channels + 2 * row + col;
        power += ccl::len_squared(actual[offset]);
        expected_power += ccl::len_squared(fixture.expected[offset]);
      }
      maximum_power_error = std::max(maximum_power_error, fabsf(power - expected_power));
      if (!std::isfinite(power) || power > 1.0f + 1e-4f || fabsf(power - expected_power) > 1e-4f) {
        failures++;
      }
    }
  }
  float maximum_checksum_error = 0.0f;
  NSMutableArray *times = [NSMutableArray array];
  NSMutableArray *warmup_times = [NSMutableArray array];
  if (!failures) {
    for (int warmup = 0; warmup < 2; warmup++) {
      if (!dispatch(true, evaluations, elapsed)) {
        return false;
      }
      [warmup_times addObject:@(elapsed)];
    }
    for (int repeat = 0; repeat < 3; repeat++) {
      if (!dispatch(true, evaluations, elapsed)) {
        return false;
      }
      [times addObject:@(elapsed)];
    }
    const float *sums = static_cast<const float *>(powers.contents);
    float expected_sums[cases] = {};
    for (unsigned i = 0; i < cases; i++) {
      for (int j = 0; j < 2 * channels; j++) {
        expected_sums[i] += ccl::len_squared(fixture.expected[size_t(i) * 2 * channels + j]);
      }
    }
    for (unsigned i = 0; i < evaluations; i++) {
      const float difference = fabsf(sums[i] - expected_sums[i % cases]);
      maximum_checksum_error = std::max(maximum_checksum_error, difference);
      if (!std::isfinite(sums[i]) || sums[i] < 0.0f || sums[i] > 2.0f + 2e-4f ||
          difference > 2e-4f)
      {
        failures++;
      }
    }
  }
  [reports addObject:@{
    @"channels" : @(channels),
    @"feedback_channels" : @(feedback),
    @"max_benchmark_checksum_error" : @(maximum_checksum_error),
    @"pitch_nm" : @(pitch),
    @"lossless" : @(material.imag() == 0.0),
    @"cases" : @(cases),
    @"failures" : @(failures),
    @"max_complex_error" : @(maximum_error),
    @"max_column_power_error" : @(maximum_power_error),
    @"fixture_generation_seconds" : @(fixture_seconds),
    @"evaluations" : @(evaluations),
    @"gpu_ms" : times,
    @"warmup_gpu_ms" : warmup_times,
    @"maximum_polarization_gain" : @(maximum_polarization_gain),
    @"thread_execution_width" : @(bench_pipeline.threadExecutionWidth),
    @"max_threads_per_group" : @(bench_pipeline.maxTotalThreadsPerThreadgroup)
  }];
  return failures == 0;
}

int main(int argc, const char **argv)
{
  if (argc < 2 || argc > 3) {
    return 2;
  }
  const unsigned evaluations = argc == 3 ? unsigned(std::atoi(argv[2])) : 262144;
  if (evaluations < 64 || evaluations > 4194304) {
    return 2;
  }
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "No Metal GPU; run this test outside the sandbox.\n");
      return 2;
    }
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
                                                 encoding:NSUTF8StringEncoding
                                                    error:&error];
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeFast;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
      fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 2;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSMutableArray *cases = [NSMutableArray array];
    bool success = run_case(device, library, queue, 740.0, {0.9, 6.0}, 2, evaluations, cases);
    success = run_case(device, library, queue, 1600.0, {0.9, 6.0}, 4, evaluations, cases) &&
              success;
    success = run_case(device, library, queue, 740.0, {1.5, 0.0}, 2, evaluations, cases) &&
              success;
    for (int feedback : {0, 2, 4}) {
      success = run_case(
                    device, library, queue, 740.0, {0.9, 6.0}, 2, evaluations, cases, feedback) &&
                success;
      success = run_case(
                    device, library, queue, 1600.0, {0.9, 6.0}, 4, evaluations, cases, feedback) &&
                success;
      success = run_case(
                    device, library, queue, 740.0, {1.5, 0.0}, 2, evaluations, cases, feedback) &&
                success;
    }
    NSDictionary *report = @{
      @"device" : device.name,
      @"cases" : cases,
      @"passed" : @(success),
      @"scope" :
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
          @"Hot-data GPU boundary construction and reference-port matching. Excludes cache "
          @"interpolation, ray traversal and renderer integration."
#else
          @"Hot-data reference-port matching only. Excludes cache interpolation, exterior "
          @"coefficient construction, ray traversal and renderer integration."
#endif
    };
    NSData *json = [NSJSONSerialization dataWithJSONObject:report options:0 error:&error];
    if (!json) {
      fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 2;
    }
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
    return success ? 0 : 1;
  }
}
