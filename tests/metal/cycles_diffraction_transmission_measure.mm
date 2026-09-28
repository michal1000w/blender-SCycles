/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <algorithm>
int main(int argc,char **argv) {
 if(argc!=2) return 2;
 @autoreleasepool {
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();
  if(!device) return 2;
  NSError *error=nil;
  NSString *source=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
  MTLCompileOptions *options=[MTLCompileOptions new]; options.mathMode=MTLMathModeFast;
  id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
  if(!library) { fprintf(stderr,"%s\n",error.localizedDescription.UTF8String); return 2; }
  id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"diffraction_transmission_measure"] error:&error];
  if(!pipeline) return 2;
  constexpr unsigned batch=1000000, batches=20, samples=batch*batches;
  id<MTLBuffer> buffer=[device newBufferWithLength:batch*4*sizeof(float) options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> queue=[device newCommandQueue];
  NSMutableArray *cases=[NSMutableArray array];
  int failures=0;
  for(float eta:{2.f/3,1.f,1.5f}) for(float delta:{-.8f,.3f,.9f}) {
   std::array<double,8> forward{},integral{},squares{};
   unsigned missing=0; double max_weight=0;
   float params[2]={eta,delta};
   for(unsigned run=0;run<batches;++run) {
    unsigned offset=run*batch;
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:params length:sizeof(params) atIndex:1];
    [encoder setBytes:&offset length:sizeof(offset) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(batch,1,1) threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted) return 2;
    const float *v=(const float *)buffer.contents;
    for(unsigned i=0;i<batch;++i,v+=4) {
     if(v[0]>=0) forward[int(v[0])]++;
     const int bin=int(v[1]); const double w=v[2];
     if(!std::isfinite(w)||w<0||bin<0||bin>=8) { ++failures; continue; }
     integral[bin]+=w; squares[bin]+=w*w; missing+=unsigned(v[3]); max_weight=std::max(max_weight,w);
    }
   }
   double ptotal=0,qtotal=0,max_error=0,max_se=0;
   for(int b=0;b<8;++b) {
    const double p=forward[b]/samples,q=integral[b]/samples;
    const double se=std::sqrt((squares[b]/samples-q*q+p*(1-p))/samples);
    const double e=std::abs(p-q);
    failures+=e>6*se+.0005; failures+=se>.005;
    ptotal+=p;qtotal+=q;max_error=std::max(max_error,e);max_se=std::max(max_se,se);
   }
   [cases addObject:@{@"eta":@(eta),@"delta":@(delta),@"forward_mass":@(ptotal),@"integrated_mass":@(qtotal),@"max_bin_error":@(max_error),@"max_bin_se":@(max_se),@"missing_inverse":@(missing),@"max_weight":@(max_weight)}];
   fprintf(stderr,"eta=%g delta=%g forward=%g integral=%g max_se=%g missing=%u\n",eta,delta,ptotal,qtotal,max_se,missing);
  }
  NSDictionary *report=@{@"device":device.name,@"samples_per_case":@(samples),@"cases":cases,@"failures":@(failures),@"passed":@(failures==0),@"scope":@"Geometry measure only; not BSDF energy, shader integration, or render benchmark",@"math_mode":@"fast"};
  NSData *json=[NSJSONSerialization dataWithJSONObject:report options:0 error:&error];
  fwrite(json.bytes,1,json.length,stdout); fputc('\n',stdout);
  return failures?1:0;
 }
}
