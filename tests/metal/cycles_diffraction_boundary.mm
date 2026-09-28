/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "test/diffraction_boundary_fixture.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

int main(int argc, const char **argv)
{
  if (argc != 3)
    return 2;
  const unsigned evaluations = unsigned(std::atoi(argv[2]));
  if (evaluations < 1035 || evaluations > 4194304)
    return 2;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device)
      return 2;
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
    if (error)
      fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:[library newFunctionWithName:@"diffraction_boundary"]
                                      error:&error];
    if (!pipeline)
      return 2;
    const auto fixture = ccl::diffraction_boundary_fixture();
    const unsigned cases = fixture.inputs.size() / 2;
    id<MTLBuffer> inputs = [device newBufferWithBytes:fixture.inputs.data()
                                               length:fixture.inputs.size() * sizeof(ccl::float4)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputs = [device newBufferWithLength:evaluations * 5 * sizeof(ccl::float2)
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> valid = [device newBufferWithLength:evaluations * sizeof(unsigned)
                                              options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSMutableArray *times = [NSMutableArray array];
    unsigned failures = 0;
    double max_error = 0, max_q2_error = 0, max_energy_error = 0;
    for (unsigned run = 0; run < 5; run++) {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:inputs offset:0 atIndex:0];
      [encoder setBuffer:outputs offset:0 atIndex:1];
      [encoder setBuffer:valid offset:0 atIndex:2];
      [encoder setBytes:&cases length:sizeof(cases) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(evaluations, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "%s\n", command.error.localizedDescription.UTF8String);
        return 2;
      }
      if (run >= 2)
        [times addObject:@(1000 * (command.GPUEndTime - command.GPUStartTime))];
      const auto *actual = static_cast<const ccl::float2 *>(outputs.contents);
      const auto *ok = static_cast<const unsigned *>(valid.contents);
      for (unsigned i = 0; i < evaluations; i++) {
        bool failed = !ok[i];
        const auto *expected = fixture.expected.data() + (i % cases) * 5;
        const auto *value = actual + i * 5;
        for (unsigned j = 0; j < 5; j++) {
          const double e = std::hypot(double(value[j].x) - expected[j].x,
                                      double(value[j].y) - expected[j].y);
          const double scaled = e / std::max(1.0, std::abs(double(expected[j].x)));
          if (j < 4)
            max_error = std::max(max_error, e);
          else
            max_q2_error = std::max(max_q2_error, scaled);
          failed |= !std::isfinite(e) || (j < 4 ? e > 3e-5 : scaled > 3e-6);
        }
        if (std::abs(expected[4].x) > 1e-12f || expected[4].x == 0.0f)
          failed |= (value[4].x > 0) != (expected[4].x > 0);
        for (unsigned p = 0; p < 2; p++) {
          const double t = p ? value[2].y : value[2].x;
          const double energy = double(value[p].x) * value[p].x + double(value[p].y) * value[p].y +
                                t * t;
          max_energy_error = std::max(max_energy_error, std::abs(energy - 1));
          failed |= !std::isfinite(energy) || std::abs(energy - 1) > 3e-6;
        }
        if (failed && failures++ < 12)
          fprintf(stderr,
                  "Boundary mismatch run %u query %u q2 %.9g expected %.9g\n",
                  run,
                  i % cases,
                  value[4].x,
                  expected[4].x);
      }
    }
    NSDictionary *report = @{
      @"device" : device.name,
      @"cases" : @(cases),
      @"evaluations" : @(evaluations),
      @"gpu_ms" : times,
      @"failures" : @(failures),
      @"passed" : @(failures == 0),
      @"max_coefficient_error" : @(max_error),
      @"max_scaled_q2_error" : @(max_q2_error),
      @"max_energy_error" : @(max_energy_error),
      @"math_mode" : @"fast",
      @"scope" :
          @"GPU boundary construction from float inputs against double reference; hot inputs, "
          @"includes output writes, excludes matching and renderer integration."
    };
    NSData *json = [NSJSONSerialization dataWithJSONObject:report options:0 error:&error];
    if (!json)
      return 2;
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
    return failures ? 1 : 0;
  }
}
