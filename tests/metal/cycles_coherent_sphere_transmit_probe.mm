/* SPDX-License-Identifier: Apache-2.0 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <simd/simd.h>
#include <cstdio>
#include <fstream>
#include <vector>
#include <array>
int main(int argc,const char**argv){
 if(argc!=2 && argc!=3)return 2;
 @autoreleasepool{
  id<MTLDevice>d=MTLCreateSystemDefaultDevice();NSError*e=nil;
  id<MTLLibrary>l=[d newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e];
  if(!l){fprintf(stderr,"%s\n",e.localizedDescription.UTF8String);return 2;}
  id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:[l newFunctionWithName:@"sphere_tt_probe"] error:&e];
  if(!p){fprintf(stderr,"%s\n",e.localizedDescription.UTF8String);return 2;}
  std::vector<std::array<simd_float4,4>> cases;
  if(argc==3){
    std::ifstream f(argv[2]);float x[14];
    while(f>>x[0]){for(int i=1;i<14;i++)if(!(f>>x[i]))return 2;
      cases.push_back({simd_float4{x[0],x[1],x[2],x[9]},simd_float4{x[3],x[4],x[5],x[10]},simd_float4{x[11],x[12],x[13],0},simd_float4{x[6],x[7],x[8],0}});}
  }else cases.push_back({simd_float4{-1,-1e-5f,.05f,.25f},simd_float4{.4f,.012f,-.018f,1.5f},simd_float4{-1,0,0,0},simd_float4{0,0,0,0}});
  printf("{\"device\":\"%s\",\"cases\":[",d.name.UTF8String);
  for(size_t ci=0;ci<cases.size();ci++){
  const auto &inputs=cases[ci];
  id<MTLBuffer>in=[d newBufferWithBytes:inputs.data() length:sizeof(inputs) options:MTLResourceStorageModeShared];
  id<MTLBuffer>out=[d newBufferWithLength:22*sizeof(simd_float4) options:MTLResourceStorageModeShared];
  id<MTLCommandQueue>q=[d newCommandQueue];id<MTLCommandBuffer>c=[q commandBuffer];
  id<MTLComputeCommandEncoder>enc=[c computeCommandEncoder];[enc setComputePipelineState:p];[enc setBuffer:in offset:0 atIndex:0];[enc setBuffer:out offset:0 atIndex:1];
  [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[enc endEncoding];[c commit];[c waitUntilCompleted];
  if(c.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"%s\n",c.error.localizedDescription.UTF8String);return 2;}
  const simd_float4*v=(const simd_float4*)out.contents;
  printf("%s{\"rows\":[",ci?",":"");
  for(int i=0;i<22;i++)printf("%s[%.17g,%.17g,%.17g,%.17g]",i?",":"",v[i].x,v[i].y,v[i].z,v[i].w);
  printf("]}");
  }
  printf("]}\n");
 }
}
