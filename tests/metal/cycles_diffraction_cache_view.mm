/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "util/math.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <iostream>
using namespace ccl;
int main(int argc, const char **argv)
{
  if (argc != 2)
    return 2;
  @autoreleasepool {
    const std::array<int4, 4> descriptors = {make_int4(0, 1, 0, 1),
                                             make_int4(0, 0, 0, 0),
                                             make_int4(1, 1, 1, 1),
                                             make_int4(1, 1, 32, 1)};
    const std::array<float4, 4> domains = {make_float4(-0.5f, -1, 380, 0),
                                           make_float4(0.5f, 1, 780, 0),
                                           make_float4(-0.5f, -1, 380, 0),
                                           make_float4(0, 0, 780, 0)};
    const std::array<int4, 2> nodes = {make_int4(-1, 0, 0, 0), make_int4(-1, 0, 0, 0)};
    const std::array<int4, 4> layout = {make_int4(0, 0, 0, 0),
                                        make_int4(1, 1, 0, 0),
                                        make_int4(0, 0, 0, 2),
                                        make_int4(1, 1, 0, 0)};
    auto bounds = domains;
    bounds[0].w = bounds[2].w = 1;
    const std::array<float4, 7> queries = {make_float4(0, 0, 580, 0),
                                           make_float4(0.25f, 0.5f, 580, 1),
                                           make_float4(0.25f, -0.5f, 380, 1),
                                           make_float4(-0.25f, 0.5f, 780, 1),
                                           make_float4(0, 0, 580, -1),
                                           make_float4(0, 0, 580, 2),
                                           make_float4(0, 0, 800, 0)};
    const std::array<float4, 21> expected = {make_float4(0, 0, 0, 0),
                                             make_float4(0.5f, 0.5f, 0.5f, 0),
                                             make_float4(1, 0, 1, 1),
                                             make_float4(32, 1, 1, 2),
                                             make_float4(0.5f, 0.5f, 0.5f, 1),
                                             make_float4(1, 0, 1, 1),
                                             make_float4(32, 1, 1, 2),
                                             make_float4(0.5f, 0.5f, 0, 1),
                                             make_float4(1, 0, -1, 1),
                                             make_float4(32, 1, 1, 2),
                                             make_float4(0.5f, 0.5f, 1, 0),
                                             make_float4(1, 0, -1, 1),
                                             make_float4(-1, -1, -1, -1),
                                             zero_float4(),
                                             zero_float4(),
                                             make_float4(-1, -1, -1, -1),
                                             zero_float4(),
                                             zero_float4(),
                                             make_float4(-1, -1, -1, -1),
                                             zero_float4(),
                                             zero_float4()};
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
                                                 encoding:NSUTF8StringEncoding
                                                    error:&error];
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeFast;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
      std::cerr << error.localizedDescription.UTF8String;
      return 2;
    }
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:[library newFunctionWithName:@"diffraction_cache_view"]
                                      error:&error];
    if (!pipeline)
      return 2;
    auto upload = [&](const auto &data) {
      return [device newBufferWithBytes:data.data()
                                 length:sizeof(data)
                                options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> buffers[] = {upload(descriptors),
                               upload(domains),
                               upload(nodes),
                               upload(layout),
                               upload(bounds),
                               upload(queries),
                               [device newBufferWithLength:sizeof(expected)
                                                   options:MTLResourceStorageModeShared]};
    for (auto buffer : buffers)
      if (!buffer)
        return 2;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    for (int i = 0; i < 7; i++)
      [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    [encoder dispatchThreads:MTLSizeMake(queries.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
      return 2;
    const float *actual = static_cast<const float *>(buffers[6].contents);
    const float *reference = reinterpret_cast<const float *>(expected.data());
    int failures = 0;
    for (size_t i = 0; i < expected.size() * 4; i++)
      failures += actual[i] != reference[i];
    std::cout << "{\"cases\":7,\"compared_components\":84,\"failures\":" << failures << "}\n";
    return failures ? 1 : 0;
  }
}
