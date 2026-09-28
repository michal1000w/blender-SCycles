/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/util/diffraction_coordinates.h"
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
        newComputePipelineStateWithFunction:[library
                                                newFunctionWithName:@"diffraction_coordinates"]
                                      error:&error];
    if (!pipeline)
      return 2;
    auto fixture = ccl::diffraction_boundary_fixture();
    for (float pitch : {250.0f,
                        std::nextafter(250.0f, 0.0f),
                        std::nextafter(250.0f, 1000.0f),
                        750.0f,
                        1250.0f,
                        1e10f})
    {
      for (float sign : {-1.0f, 1.0f}) {
        fixture.inputs.push_back(ccl::make_float4(sign, 0, 0, 1));
        fixture.inputs.push_back(ccl::make_float4(1, 500, pitch, 0));
      }
    }
    fixture.expected.clear();
    for (size_t i = 0; i < fixture.inputs.size() / 2; i++) {
      const auto a = fixture.inputs[2 * i], b = fixture.inputs[2 * i + 1];
      ccl::DiffractionCacheCoordinates result;
      if (!ccl::diffraction_cache_coordinates(
              ccl::make_float3(a.x, a.y, a.z), a.w, b.y, b.z, i % 2, &result))
        return 2;
      fixture.expected.push_back(ccl::make_float2(result.query.x, result.query.y));
      fixture.expected.push_back(ccl::make_float2(result.query.z, result.incoming_order));
      fixture.expected.push_back(
          ccl::make_float2(result.cross_polarization_sign, result.reverse_orders));
      fixture.expected.push_back(ccl::zero_float2());
      fixture.expected.push_back(ccl::zero_float2());
    }
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
    double max_error = 0;
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
          max_error = std::max(max_error, e);
          failed |= !std::isfinite(e) || e > 3e-5;
        }
        if (failed && failures++ < 12)
          fprintf(stderr,
                  "Coordinate mismatch run %u query %u q2 %.9g expected %.9g\n",
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
      @"max_coordinate_error" : @(max_error),
      @"math_mode" : @"fast",
      @"scope" :
          @"GPU coordinate reduction against shared CPU mapping; independent CPU momentum checks "
          @"are separate."
    };
    NSData *json = [NSJSONSerialization dataWithJSONObject:report options:0 error:&error];
    if (!json)
      return 2;
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
    return failures ? 1 : 0;
  }
}
