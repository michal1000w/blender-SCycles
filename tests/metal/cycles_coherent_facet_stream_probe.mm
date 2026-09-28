/* SPDX-License-Identifier: Apache-2.0 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>
struct V4 {float x,y,z,w;};
int main(int argc,char **argv){@autoreleasepool {
 if(argc!=2)return 2;
 id<MTLDevice> device=MTLCreateSystemDefaultDevice();NSError *error=nil;
 id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
 if(!library){fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
 id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"coherent_facet_stream_probe"] error:&error];
 if(!pipeline){fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
 std::vector<V4> paths(86),cases;
 for(int i=0;i<83;i++)paths[i]={.01f*(1+i%3),1.0f+.0002f*i,1.27e-8f,.17f*i};
 for(float lc:{.005f,1e-7f,1e-9f})for(float z:{0.f,.371f,-1.9f,.001f})cases.push_back({z,lc,0,83});
 paths[83]={1e8f,0,0,0};paths[84]={1,0,0,0};paths[85]={1e8f,0,0,.5f};
 cases.push_back({0,1,83,3});
 id<MTLBuffer> input=[device newBufferWithBytes:paths.data() length:paths.size()*sizeof(V4) options:MTLResourceStorageModeShared];
 id<MTLBuffer> configs=[device newBufferWithBytes:cases.data() length:cases.size()*sizeof(V4) options:MTLResourceStorageModeShared];
 id<MTLBuffer> output=[device newBufferWithLength:cases.size()*sizeof(V4) options:MTLResourceStorageModeShared];
 id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
 [encoder setComputePipelineState:pipeline];[encoder setBuffer:input offset:0 atIndex:0];[encoder setBuffer:configs offset:0 atIndex:1];[encoder setBuffer:output offset:0 atIndex:2];
 [encoder dispatchThreads:MTLSizeMake(cases.size(),1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
 if(command.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"%s\n",command.error.description.UTF8String);return 2;}
 V4 *actual=(V4*)output.contents;int failures=0;double maximum=0;
 const float hi=550e-9f,low=float(550e-9-double(hi));const double wavelength=double(hi)+low;
 for(size_t i=0;i<cases.size();i++){
  auto c=cases[i];std::complex<double> total=0,direct=0;
  for(int route=0;route<int(c.w);route++){auto p=paths[int(c.z)+route];double length=double(p.y)+p.z;double angle=2*M_PI*std::remainder(length/wavelength+p.w+double(c.x)*length/(2*M_PI*double(c.y)),1.0);
   auto a=double(p.x)*std::exp(std::complex<double>(0,angle));total+=a;if(direct==std::complex<double>(0,0))direct=a;
  }
  double phase=2*M_PI*std::remainder((1+double(1.27e-8f))/wavelength+double(.13f)+double(c.x)*(1+double(1.27e-8f))/(2*M_PI*double(c.y)),1.0);
  double expected[4]={std::norm(total),std::real(direct*std::conj(total)),std::cos(phase),std::sin(phase)};
  float observed[4]={actual[i].x,actual[i].y,actual[i].z,actual[i].w};
  for(int k=0;k<4;k++){double e=std::abs(observed[k]-expected[k]);maximum=std::max(maximum,e);if(!std::isfinite(observed[k])||e>2e-4){failures++;fprintf(stderr,"case%zu component%d %.9g != %.9g\n",i,k,observed[k],expected[k]);}}
 }
 printf("{\"cases\":%zu,\"checks\":%zu,\"failures\":%d,\"max_absolute_error\":%.10g}\n",cases.size(),4*cases.size(),failures,maximum);
 return failures?1:0;
}}
