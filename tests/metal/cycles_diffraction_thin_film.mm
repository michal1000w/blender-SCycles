/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "kernel/closure/diffraction_thin_film.h"
#include <random>
#include <vector>
#include <cstdio>
using namespace ccl;
int main(int argc,char **argv) {
 if(argc!=2)return 2;
 @autoreleasepool {
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();if(!device)return 2;
  NSError *error=nil;
  NSString *source=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
  MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeFast;
  id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
  if(!library){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return 2;}
  id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"diffraction_thin_film"] error:&error];if(!pipeline)return 2;
  constexpr unsigned count=65536;
  std::vector<float4> input;std::vector<float> expected;std::mt19937 rng(618271);std::uniform_real_distribution<float> u(0,1);
  while(expected.size()<count) {
   float ni=1+u(rng),no=1+u(rng),nf=.8f+1.5f*u(rng),ci=.05f+.95f*u(rng),d=10*u(rng);
   const double ct2=1-pow(double(ni)/no,2)*(1-double(ci)*ci);if(ct2<=.001)continue;
   const float ct=sqrt(ct2);input.push_back(make_float4(ci,ct,ni,no));input.push_back(make_float4(nf,d,0,0));
   expected.push_back(diffraction_thin_film_pair_reflectance(ci,ct,ni,no,nf,d));
  }
  id<MTLBuffer> in=[device newBufferWithBytes:input.data() length:input.size()*sizeof(float4) options:MTLResourceStorageModeShared];
  id<MTLBuffer> out=[device newBufferWithLength:count*sizeof(float2) options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];[encoder setBuffer:in offset:0 atIndex:0];[encoder setBuffer:out offset:0 atIndex:1];
  [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
  [encoder endEncoding];[command commit];[command waitUntilCompleted];
  if(command.status!=MTLCommandBufferStatusCompleted)return 2;
  const float2 *actual=(const float2 *)out.contents;int failures=0;float max_error=0,max_reverse=0;
  for(unsigned i=0;i<count;++i) {
   float e=fabsf(actual[i].x-expected[i]),r=fabsf(actual[i].x-actual[i].y);
   max_error=fmaxf(max_error,e);max_reverse=fmaxf(max_reverse,r);
   if((!isfinite_safe(actual[i].x)||!isfinite_safe(actual[i].y))||e>2e-4f||r>2e-6f||actual[i].x<0||actual[i].x>1) {
    if(failures<8) { const auto a=input[2*i],b=input[2*i+1];fprintf(stderr,"case=%u expected=%g actual=%g reverse=%g ci=%g ct=%g ni=%g no=%g nf=%g d=%g\n",i,expected[i],actual[i].x,actual[i].y,a.x,a.y,a.z,a.w,b.x,b.y); }
    ++failures;
   }
  }
  NSDictionary *report=@{@"device":device.name,@"cases":@(count),@"max_cpu_error":@(max_error),@"max_reverse_error":@(max_reverse),@"failures":@(failures),@"passed":@(failures==0),@"scope":@"Single lossless film, not grating-film integration or a render benchmark"};
  NSData *json=[NSJSONSerialization dataWithJSONObject:report options:0 error:&error];fwrite(json.bytes,1,json.length,stdout);fputc('\n',stdout);
  return failures?1:0;
 }
}
