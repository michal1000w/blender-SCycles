/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/util/diffraction_scene_data.h"
#include "scene/diffraction.h"
#include "util/math.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
using namespace ccl;
#include <random>
int main(int argc, const char **argv)
{
  if (argc != 3 && argc != 4) return 2;
  const bool normal = argc == 4 && std::string(argv[3]) == "--normal";
  if (argc == 4 && !normal) return 2;
  std::ifstream input(argv[2], std::ios::binary);
  DiffractionGratingDeviceBuffers packed;
  auto read = [&](auto &v) {
    uint64_t count=0; input.read(reinterpret_cast<char *>(&count),sizeof(count));
    if(!input || count>10000000) return false;
    v.resize(count); input.read(reinterpret_cast<char *>(v.data()),count*sizeof(v[0]));
    return bool(input);
  };
  if(!read(packed.nodes)||!read(packed.layout)||!read(packed.bounds)||!read(packed.ports)||
     !read(packed.active)||!read(packed.matrices)||input.peek()!=EOF) return 2;
  @autoreleasepool {
    auto &nodes=packed.nodes; auto &layout=packed.layout; auto &bounds=packed.bounds;
    auto &ports=packed.ports; auto &active=packed.active; auto &matrices=packed.matrices;
    std::vector<int4> descriptors={make_int4(0,nodes.size(),0,layout.size()/2),make_int4(0,0,0,0)};
    std::vector<float4> domains={make_float4(-.5,-1,380,0),make_float4(.5,1,780,0)};
    std::vector<float4> queries,expected,reverse_checks;
    DiffractionSceneData data{nodes.data(),layout.data(),bounds.data(),ports.data(),active.data(),
      matrices.data(),descriptors.data(),domains.data(),1};
    std::mt19937 random(913771); std::uniform_real_distribution<float> unit(0,1);
    for(int q=0;q<4096;q++) {
      const float z=unit(random),phi=2*M_PI_F*unit(random),r=std::sqrt(1-z*z);
      const bool below=q%2;
      const float3 ray=normal?make_float3(0,0,below?1:-1):
                              make_float3(r*std::cos(phi),r*std::sin(phi),below?z:-z);
      const float wavelength=380+400*unit(random),u=unit(random);
      DiffractionSceneSample result;
      if(!diffraction_data_sample<20>(&data,0,ray,below,1,1,wavelength,740,u,&result))return 3;
      queries.push_back(make_float4(ray.x,ray.y,ray.z,740));
      queries.push_back(make_float4(0,below,u,wavelength));
      expected.push_back(make_float4(result.direction.x,result.direction.y,result.direction.z,result.eta));
      expected.push_back(make_float4(result.relative_order,result.transmission,result.probability,result.throughput));
      reverse_checks.push_back(zero_float4());
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
                                                 encoding:NSUTF8StringEncoding
                                                    error:&error];
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
      std::cerr << error.localizedDescription.UTF8String;
      return 2;
    }
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"diffraction_scene_sample_test"]
                                      error:&error];
    if (!pipeline) {
      std::cerr << "Metal sampler pipeline creation failed: "
                << error.localizedDescription.UTF8String << "\n";
      return 2;
    }
    auto upload = [&](const auto &data) {
      return [device newBufferWithBytes:data.data()
                                 length:data.size() * sizeof(data[0])
                                options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> buffers[] = {upload(descriptors),
                               upload(domains),
                               upload(nodes),
                               upload(layout),
                               upload(bounds),
                               upload(ports),
                               upload(active),
                               upload(matrices),
                               upload(queries),
                               [device newBufferWithLength:expected.size() * sizeof(float4)
                                                   options:MTLResourceStorageModeShared],
                               upload(reverse_checks)};
    for (auto buffer : buffers)
      if (!buffer)
        return 2;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    int failures = 0;
    double maximum_error = 0;
    double maximum_gpu_conservation_error = 0, maximum_cpu_conservation_error = 0;
    std::vector<double> gpu_ms;
    const size_t group_size = std::min(NSUInteger(64), pipeline.maxTotalThreadsPerThreadgroup);
    for (int run = 0; run < 7; run++) {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      for (int i = 0; i < 11; i++)
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
      [encoder dispatchThreads:MTLSizeMake(queries.size() / 2, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        std::cerr << command.error.localizedDescription.UTF8String;
        return 2;
      }
      const double milliseconds = 1000 * (command.GPUEndTime - command.GPUStartTime);
      if (!std::isfinite(milliseconds) || milliseconds <= 0)
        return 2;
      if (run >= 2)
        gpu_ms.push_back(milliseconds);
      const float *actual = static_cast<const float *>(buffers[9].contents);
      const float *reference = reinterpret_cast<const float *>(expected.data());
      /* This fixture is lossless with all propagating orders retained. Compare
       * throughput against unit flux independently of CPU/GPU agreement. */
      for (size_t q = 0; q < expected.size() / 2; q++) {
        const double gpu_error = std::abs(double(actual[8 * q + 7]) - 1);
        const double cpu_error = std::abs(double(reference[8 * q + 7]) - 1);
        maximum_gpu_conservation_error = std::max(maximum_gpu_conservation_error,
            std::isfinite(gpu_error) ? gpu_error : INFINITY);
        maximum_cpu_conservation_error = std::max(maximum_cpu_conservation_error,
            std::isfinite(cpu_error) ? cpu_error : INFINITY);
      }
      for (size_t i = 0; i < expected.size() * 4; i++) {
        const double difference = std::abs(actual[i] - reference[i]);
        if (std::isfinite(difference))
          maximum_error = std::max(maximum_error, difference);
        if (run == 0 && (!std::isfinite(actual[i]) || difference > 2e-6)) {
          const size_t query = i / 8;
          const float4 ray = queries[2 * query], parameters = queries[2 * query + 1];
          std::cerr << "query=" << query << " component=" << i % 8
                    << " cpu=" << reference[i] << " gpu=" << actual[i]
                    << " difference=" << difference << " ray=" << ray.x << "," << ray.y
                    << "," << ray.z << " wavelength=" << parameters.w << "\n";
        }
        failures += !std::isfinite(actual[i]) || difference > 2e-6;
      }
    }
    auto sorted = gpu_ms;
    std::sort(sorted.begin(), sorted.end());
    size_t cache_bytes = 0;
    for (int i = 0; i < 8; i++)
      cache_bytes += buffers[i].length;
    std::cout << "{\"cases_per_run\":" << queries.size() / 2
              << ",\"runs\":7,\"warmup_runs\":2,\"compared_components\":"
              << expected.size() * 4 * 7 << ",\"failures\":" << failures
              << ",\"maximum_component_error\":" << maximum_error
              << ",\"maximum_gpu_conservation_error\":" << maximum_gpu_conservation_error
              << ",\"maximum_cpu_conservation_error\":" << maximum_cpu_conservation_error
              << ",\"threadgroup_size\":" << group_size << ",\"cache_bytes\":" << cache_bytes
              << ",\"device\":\"" << device.name.UTF8String << "\",\"gpu_ms\":[";
    for (size_t i = 0; i < gpu_ms.size(); i++)
      std::cout << (i ? "," : "") << gpu_ms[i];
    std::cout << "],\"median_gpu_ms\":" << sorted[sorted.size() / 2]
              << ",\"measurement\":\"hot fixed-query sampler dispatch; excludes build, upload and "
                 "rendering\"}\n";
    return failures ? 1 : 0;
  }
}
