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
  if(argc!=6) return 2;
  @autoreleasepool {
    auto input=read(argv[2]),queries=read(argv[3]),expected=read(argv[4]);
    if(input.size()<414 || (input.size()-414)%743 || queries.empty() || queries.size()%3 || expected.size()!=queries.size()/3*800) return 2;
    const size_t count=queries.size()/3;
    const int layout[2]={int(input.size()),int((input.size()-414)/743)};
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device) return 3;
    NSError *error=nil;
    NSString *source=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    if(!library) {std::cerr<<error.localizedDescription.UTF8String;return 4;}
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:
      [library newFunctionWithName:@"tensor_chart_test"] error:&error];
    if(!pipeline) return 4;
    id<MTLBuffer> in=[device newBufferWithBytes:input.data() length:input.size()*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> out=[device newBufferWithLength:expected.size()*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> q=[device newBufferWithBytes:queries.data() length:queries.size()*4 options:MTLResourceStorageModeShared];
    if(!in || !out || !q) return 5;
    id<MTLCommandQueue> queue=[device newCommandQueue];double seconds=0;
    for(int repeat=0;repeat<11;++repeat) {
      id<MTLCommandBuffer> command=[queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:in offset:0 atIndex:0];[encoder setBuffer:q offset:0 atIndex:1];
      [encoder setBuffer:out offset:0 atIndex:2];[encoder setBytes:layout length:sizeof(layout) atIndex:3];
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
    std::ofstream output(argv[5],std::ios::binary);
    output.write(reinterpret_cast<const char *>(actual),expected.size()*4);
    if(!output) return 8;
    std::cout<<"{\"queries\":"<<count<<",\"maximum_component_error\":"<<maximum
      <<",\"mean_gpu_seconds\":"<<seconds/10<<",\"scope\":\"Reusable tensor chart evaluator only; excludes matching and rendering\"}\n";
  }
}
