/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
std::vector<float> read(const char *path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f || f.tellg() <= 0 || f.tellg() % 4 != 0) return {};
  std::vector<float> v(size_t(f.tellg()) / 4);
  f.seekg(0); f.read(reinterpret_cast<char *>(v.data()), v.size()*4);
  if (!f) return {};
  return v;
}
int main(int argc, const char **argv) {
  if (argc != 6 && argc != 7) return 2;
  const bool cayley=argc==7;
  @autoreleasepool {
    auto model=read(argv[2]), queries=read(argv[3]), expected=read(argv[4]);
    if (model.size()!=(cayley?46482:46480) || queries.empty() || queries.size()%3 ||
        expected.size()!=queries.size()/3*(cayley?800:400)) return 2;
    size_t count=queries.size()/3;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if (!device) return 3;
    NSError *error=nil;
    NSString *source=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    MTLCompileOptions *options=[MTLCompileOptions new];
    options.mathMode=MTLMathModeSafe;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    if (!library) { std::cerr << error.localizedDescription.UTF8String; return 4; }
    id<MTLComputePipelineState> pipelines[3];
    for (int i=0;i<(cayley?3:2);++i) {
      pipelines[i]=[device newComputePipelineStateWithFunction:[library newFunctionWithName:
        i==0 ? @"tensor_coefficients" : (i==1 ? @"tensor_reconstruct" : @"tensor_cayley")] error:&error];
      if (!pipelines[i]) return 4;
    }
    auto upload=[&](const std::vector<float> &v) {
      return [device newBufferWithBytes:v.data() length:v.size()*4 options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> m=upload(model), q=upload(queries);
    id<MTLBuffer> c=[device newBufferWithLength:count*62*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> out=[device newBufferWithLength:count*400*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> scattering=[device newBufferWithLength:count*800*4 options:MTLResourceStorageModeShared];
    if (!m || !q || !c || !out || !scattering) return 5;
    id<MTLCommandQueue> queue=[device newCommandQueue];
    double seconds=0;
    for (int repeat=0;repeat<11;++repeat) {
      id<MTLCommandBuffer> command=[queue commandBuffer];
      for (int pass=0;pass<(cayley?3:2);++pass) {
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:pipelines[pass]];
        [encoder setBuffer:m offset:0 atIndex:0];
        [encoder setBuffer:pass==0?q:(pass==1?c:out) offset:0 atIndex:1];
        [encoder setBuffer:pass==0?c:(pass==1?out:scattering) offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(count*(pass==0?62:(pass==1?400:20)),1,1)
          threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [encoder endEncoding];
      }
      [command commit]; [command waitUntilCompleted];
      if (command.status!=MTLCommandBufferStatusCompleted) return 6;
      if (repeat) seconds+=command.GPUEndTime-command.GPUStartTime;
    }
    float *actual=static_cast<float *>((cayley?scattering:out).contents);
    double maximum=0;
    for (size_t i=0;i<expected.size();++i) {
      if (!std::isfinite(actual[i])) return 7;
      maximum=std::max(maximum,std::abs(double(actual[i])-expected[i]));
    }
    std::ofstream output(argv[5],std::ios::binary);
    output.write(reinterpret_cast<char *>(actual),expected.size()*4);
    if (!output) return 8;
    std::cout << "{\"device\":\"" << device.name.UTF8String << "\",\"queries\":" << count
              << ",\"mean_gpu_seconds\":" << seconds/10 << ",\"maximum_packed_error\":" << maximum
              << ",\"scope\":\"Experimental tensor stages; optional Cayley; excludes exterior matching and BSDF\"}\n";
  }
}
