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
int main(int argc,const char **argv) {
  if(argc!=5) return 2;
  @autoreleasepool {
    auto input=read(argv[2]),expected=read(argv[3]);
    if(input.empty() || input.size()%883 || expected.size()!=input.size()/883*80) return 2;
    const size_t count=input.size()/883;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device) return 3;
    NSError *error=nil;
    NSString *source=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    if(!library) {std::cerr<<error.localizedDescription.UTF8String;return 4;}
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:
      [library newFunctionWithName:@"tensor_match"] error:&error];
    if(!pipeline) return 4;
    id<MTLBuffer> in=[device newBufferWithBytes:input.data() length:input.size()*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> out=[device newBufferWithLength:expected.size()*4 options:MTLResourceStorageModeShared];
    if(!in || !out) return 5;
    id<MTLCommandQueue> queue=[device newCommandQueue];double seconds=0;
    for(int repeat=0;repeat<11;++repeat) {
      id<MTLCommandBuffer> command=[queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:in offset:0 atIndex:0];[encoder setBuffer:out offset:0 atIndex:1];
      [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
      [encoder endEncoding];[command commit];[command waitUntilCompleted];
      if(command.status!=MTLCommandBufferStatusCompleted) return 6;
      if(repeat) seconds+=command.GPUEndTime-command.GPUStartTime;
    }
    const float *actual=static_cast<const float *>(out.contents);double maximum=0;
    for(size_t i=0;i<expected.size();++i) {
      if(!std::isfinite(actual[i])) return 7;
      maximum=std::max(maximum,std::abs(double(actual[i])-expected[i]));
    }
    std::ofstream output(argv[4],std::ios::binary);
    output.write(reinterpret_cast<const char *>(actual),expected.size()*4);
    if(!output) return 8;
    std::cout<<"{\"incident_ports\":"<<count<<",\"maximum_component_error\":"<<maximum
      <<",\"mean_gpu_seconds\":"<<seconds/10<<",\"scope\":\"Production combined matcher only; CPU boundary construction; excludes tensor interpolation and rendering\"}\n";
  }
}
