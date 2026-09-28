/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "kernel/closure/bsdf_diffraction_interface.h"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
using namespace ccl;

int main(int argc,const char **argv)
{
  if(argc!=2)return 2;
  @autoreleasepool {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device)return 2;
    NSError *error=nil;
    NSString *source=[NSString stringWithContentsOfFile:@(argv[1])
                         encoding:NSUTF8StringEncoding error:&error];
    MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeFast;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    if(!library){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return 2;}
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"fast_interface"] error:&error];
    if(!pipeline){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return 2;}
    constexpr unsigned count=16384;
    std::vector<float> inputs(count*16),expected(count*12);
    std::mt19937 rng(901713);std::uniform_real_distribution<float> unit(0,1);
    for(unsigned i=0;i<count;++i) {
      float *v=&inputs[i*16],*e=&expected[i*12];
      v[0]=380+400*unit(rng);v[1]=400+1600*unit(rng);v[2]=500*unit(rng);v[3]=unit(rng);
      v[4]=1+unit(rng);v[5]=1+unit(rng);v[6]=.01f+.8f*unit(rng);
      v[7]=i%3 ? (1-v[6])*unit(rng) : 0;v[8]=12*unit(rng);
      const float z=.02f+.98f*unit(rng),phi=M_2PI_F*unit(rng),s=sqrtf(1-z*z);
      v[9]=s*cosf(phi);v[10]=s*sinf(phi);v[11]=z;v[12]=unit(rng);
      if(i<12) {
        const float grazing_cosines[]={.3f,.1f,.01f,.001f,.0001f,.00001f};
        const float cosine=grazing_cosines[i%6];
        v[0]=550;v[1]=740;v[2]=0;v[3]=.5f;v[4]=v[5]=1;
        v[6]=i<6?1:0;v[7]=i<6?0:1;v[8]=0;
        v[9]=sqrtf(1-cosine*cosine);v[10]=0;v[11]=cosine;v[12]=.5f;
      }
      const FastDiffractionInterface p{v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]};
      const float3 wi=make_float3(v[9],v[10],v[11]);float3 wo;
      int order;bool transmission;float power,probability;
      if(!fast_diffraction_interface_sample(p,wi,v[12],&order,&transmission,&wo,&power,&probability))return 2;
      if(i<12 && (order!=0 || transmission!=(i>=6) || wo.x!=-wi.x ||
                  wo.z!=(i>=6?-wi.z:wi.z) || power!=1 || probability!=1))return 2;
      e[0]=wo.x;e[1]=wo.y;e[2]=wo.z;e[3]=power;e[4]=probability;
      e[5]=float(order);e[6]=float(transmission);e[7]=1;
      e[8]=fast_diffraction_interface_residual(p,wi);e[9]=power;e[10]=v[6]+v[7];
    }
    id<MTLBuffer> in=[device newBufferWithBytes:inputs.data() length:inputs.size()*sizeof(float)
                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> out=[device newBufferWithLength:expected.size()*sizeof(float)
                                options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];[encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBuffer:out offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(count,1,1)
          threadsPerThreadgroup:MTLSizeMake(std::min(NSUInteger(64),pipeline.maxTotalThreadsPerThreadgroup),1,1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted){
      fprintf(stderr,"%s\n",command.error.localizedDescription.UTF8String);return 2;
    }
    const float *actual=(const float *)out.contents;
    unsigned failures=0,branch_mismatches=0;double max_error=0,max_reciprocity=0;
    for(unsigned i=0;i<count;++i) {
      for(unsigned j=0;j<12;++j) {
        const float value=actual[i*12+j],reference=expected[i*12+j];
        if(!std::isfinite(value)){++failures;continue;}
        const double delta=std::abs(double(value)-reference);
        if(j>=5 && j<=7) {
          if(delta!=0){++failures;++branch_mismatches;}
        }
        else {
          max_error=std::max(max_error,delta);
          failures+=delta>(j<3?2e-4:2e-5);
        }
      }
      max_reciprocity=std::max(max_reciprocity,
          std::abs(double(actual[i*12+9])-actual[i*12+3]));
      failures+=actual[i*12+8]<-2e-6f;
    }
    printf("{\"device\":\"%s\",\"cases\":%u,\"failures\":%u,\"branch_mismatches\":%u,"
           "\"maximum_cpu_gpu_error\":%.12g,\"maximum_gpu_reciprocity_error\":%.12g,"
           "\"scope\":\"Metal fast-math scalar-interface experiment, not a render benchmark\"}\n",
           device.name.UTF8String,count,failures,branch_mismatches,max_error,max_reciprocity);
    return failures?1:0;
  }
}
