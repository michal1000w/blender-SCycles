/* SPDX-License-Identifier: Apache-2.0 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <simd/simd.h>
#include <cstdio>
#include <cmath>
int main(int argc,const char **argv){
 if(argc!=2)return 2;
 @autoreleasepool{
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();NSError*e=nil;
 id<MTLLibrary>l=[d newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e];
 id<MTLComputePipelineState>p=l?[d newComputePipelineStateWithFunction:[l newFunctionWithName:@"native_grating_polarizer_probe"] error:&e]:nil;
 if(!p){fprintf(stderr,"%s\n",e.localizedDescription.UTF8String);return 2;}
 simd_float4 input[24];for(int i=0;i<24;i++)input[i]={i%4*.37f,i%12<4?1.f:i%12<8?-1.f:0.f,i>=12?1.f:0.f,0};
 id<MTLBuffer>in=[d newBufferWithBytes:input length:sizeof(input) options:MTLResourceStorageModeShared];
 id<MTLBuffer>out=[d newBufferWithLength:264*sizeof(simd_float4) options:MTLResourceStorageModeShared];
 // The exercised grating map does not access scene resources. Reserve a zeroed
 // context binding; this is not an invented LUT or a replacement closure.
 id<MTLBuffer>params=[d newBufferWithLength:1024*1024 options:MTLResourceStorageModeShared];
 id<MTLCommandQueue>q=[d newCommandQueue];id<MTLCommandBuffer>c=[q commandBuffer];id<MTLComputeCommandEncoder>enc=[c computeCommandEncoder];
 [enc setComputePipelineState:p];[enc setBuffer:in offset:0 atIndex:0];[enc setBuffer:out offset:0 atIndex:1];[enc setBuffer:params offset:0 atIndex:2];
 [enc dispatchThreads:MTLSizeMake(24,1,1) threadsPerThreadgroup:MTLSizeMake(24,1,1)];[enc endEncoding];[c commit];[c waitUntilCompleted];
 if(c.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"%s\n",c.error.localizedDescription.UTF8String);return 2;}
 auto*v=(const simd_float4*)out.contents;int bad=0;
 printf("{\"device\":\"%s\",\"cases\":[",d.name.UTF8String);
 for(int i=0;i<24;i++){
  printf("%s{\"rows\":[",i?",":"");for(int j=0;j<11;j++){const auto a=v[11*i+j];printf("%s[%.9g,%.9g,%.9g,%.9g]",j?",":"",a.x,a.y,a.z,a.w);for(int k=0;k<4;k++)if(!std::isfinite(a[k]))bad++;}printf("]}");
  const float target=.5f;
  if(v[11*i].x!=96||v[11*i].z!=1||std::abs(v[11*i+2].x-target)>2e-6f||std::abs(v[11*i+3].x-target)>2e-6f)bad++;
  if(v[11*i+7].x!=1||v[11*i+7].z!=1||v[11*i+7].w!=1||std::abs(v[11*i+8].x-.5f)>2e-6f||std::abs(v[11*i+9].x-.5f)>2e-6f)bad++;
  const simd_float3 signed_target={-2*target,.3f*target,3*target};
  for(int k=0;k<3;k++)if(std::abs(v[11*i+5][k]-signed_target[k])>2e-6f||std::abs(v[11*i+6][k]-signed_target[k])>2e-6f)bad++;
 }
 printf("],\"failures\":%d}\n",bad);return bad?1:0;
 }
}
