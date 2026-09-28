/* SPDX-License-Identifier: Apache-2.0 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <simd/simd.h>
#include "kernel/util/dielectric_dispersion.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace ccl;
static uint32_t bits(float f) {uint32_t b;std::memcpy(&b,&f,4);return b;}
__attribute__((noinline)) static simd_uint4 host_profile(simd_float4 p)
{
  const float um=dielectric_wavelength_um(p.x);
  return {bits(um),bits(dielectric_ior_at_wavelength(p.z,p.w,um)),
          bits(dielectric_ior_at_wavelength(p.z,p.w,p.y)),
          bits(dielectric_ior_at_wavelength(p.z,p.w,.78f))};
}
int main(int argc,const char **argv)
{
  if(argc!=2)return 2;
  @autoreleasepool {
    float matched=0x1.866fcp+0f;
    for(int j=0;j<64;j++) {
      const float n=dielectric_ior_at_wavelength(1.5f,matched,.78f);
      if(n==1)break;
      matched=std::nextafter(matched,n>1?2.0f:0.0f);
    }
    if(dielectric_ior_at_wavelength(1.5f,matched,.78f)!=1)return 3;
    std::vector<simd_float4> data={(simd_float4){780,.78f,1.5f,0x1.866fbep+0f},
                                  (simd_float4){780,.78f,1.5f,0x1.866fcp+0f},
                                  (simd_float4){780,.78f,1.5f,matched},
                                  (simd_float4){430,.43f,1.5f,.025f},
                                  (simd_float4){610,.61f,1.5f,.025f}};
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();if(!device)return 4;
    NSError *error=nil;
    id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"dielectric_dispersion_probe"] error:&error];
    if(!pipeline){std::fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return 5;}
    id<MTLBuffer> input=[device newBufferWithBytes:data.data() length:data.size()*sizeof(simd_float4)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> output=[device newBufferWithLength:data.size()*sizeof(simd_uint4)
                                               options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];[encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(data.size(),1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted)return 6;
    auto *actual=(const simd_uint4 *)output.contents;bool passed=true;
    for(size_t i=0;i<data.size();i++) {
      const simd_uint4 expected=host_profile(data[i]);
      const bool same=std::memcmp(&expected,&actual[i],sizeof(expected))==0;
      passed=passed&&same&&actual[i].y==actual[i].z;
      std::printf("case%zu nm=%a um=%a inv=%a cpu=[%08x,%08x,%08x,%08x] metal=[%08x,%08x,%08x,%08x] match=%d\n",
          i,data[i].x,data[i].y,data[i].w,expected.x,expected.y,expected.z,expected.w,
          actual[i].x,actual[i].y,actual[i].z,actual[i].w,same);
    }
    return !passed;
  }
}
