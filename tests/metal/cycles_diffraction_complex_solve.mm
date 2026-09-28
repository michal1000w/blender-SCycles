/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdint>
struct Pair { float x,y; };
using Complex=std::complex<double>;
static Complex complex_value(Pair p) { return {p.x,p.y}; }

int main(int argc,const char **argv)
{
  if (argc!=2) return 2;
  @autoreleasepool {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if (!device) return 2;
    NSError *error=nil;
    NSString *source=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    MTLCompileOptions *options=[MTLCompileOptions new]; options.mathMode=MTLMathModeSafe;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    if (!library) { fprintf(stderr,"%s\n",error.localizedDescription.UTF8String); return 2; }
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"complex_solve"] error:&error];
    if (!pipeline) return 2;
    id<MTLCommandQueue> queue=[device newCommandQueue];
    NSMutableArray *cases=[NSMutableArray array];
    unsigned failures=0;
    uint32_t state=917281;
    auto random=[&]() { state^=state<<13;state^=state>>17;state^=state<<5;
                       return float(int(state&65535)-32768)/32768.0f; };
    for (unsigned n:{3u,17u,66u,132u}) {
      unsigned rhs=n,stride=n+rhs,batch=5;
      std::vector<Pair> input(batch*n*stride),expected(batch*n*rhs);
      for (unsigned m=0;m<batch;++m) {
        float scale=m==1 ? 1e-8f : m==2 ? 1e8f : 1.0f;
        for (unsigned row=0;row<n;++row) {
          for (unsigned col=0;col<n;++col) {
            input[m*n*stride+row*stride+col]={scale*(random()/n+(row==col?2.0f:0.0f)),scale*random()/n};
            expected[m*n*rhs+row*rhs+col]={random(),random()};
          }
        }
        if (m==3) { // A permutation requires row pivoting even at the first step.
          for (unsigned row=0;row<n;++row)
            for (unsigned col=0;col<n;++col)
              input[m*n*stride+row*stride+col]={float(col==(row+1)%n),0};
        }
        if (m==4) { // Exactly rank deficient, must report failure.
          for (unsigned col=0;col<n;++col) input[m*n*stride+(n-1)*stride+col]={0,0};
        }
        for (unsigned row=0;row<n;++row)
          for (unsigned col=0;col<rhs;++col) {
            Complex value(0,0);
            for (unsigned k=0;k<n;++k)
              value+=complex_value(input[m*n*stride+row*stride+k])*complex_value(expected[m*n*rhs+k*rhs+col]);
            input[m*n*stride+row*stride+n+col]={float(value.real()),float(value.imag())};
          }
      }
      id<MTLBuffer> workspace=[device newBufferWithLength:input.size()*sizeof(Pair) options:MTLResourceStorageModeShared];
      id<MTLBuffer> status=[device newBufferWithLength:batch*sizeof(unsigned) options:MTLResourceStorageModeShared];
      NSMutableArray *times=[NSMutableArray array];
      double max_error=0,max_residual=0; unsigned singular_checks=0;
      for (unsigned run=0;run<5;++run) {
        memcpy(workspace.contents,input.data(),input.size()*sizeof(Pair));
        memset(status.contents,0,batch*sizeof(unsigned));
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline]; [encoder setBuffer:workspace offset:0 atIndex:0];
        [encoder setBuffer:status offset:0 atIndex:1]; [encoder setBytes:&n length:sizeof(n) atIndex:2];
        [encoder setBytes:&rhs length:sizeof(rhs) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(batch,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(128,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        if (command.status!=MTLCommandBufferStatusCompleted) return 2;
        if (run>=2) [times addObject:@(1000*(command.GPUEndTime-command.GPUStartTime))];
        auto actual=static_cast<const Pair *>(workspace.contents);
        auto codes=static_cast<const unsigned *>(status.contents);
        for (unsigned m=0;m<batch;++m) {
          if (m==4) { if (codes[m]!=2) ++failures; else ++singular_checks; continue; }
          if (codes[m]!=1) { ++failures;continue; }
          double residual2=0,b2=0;
          for (unsigned row=0;row<n;++row)
            for (unsigned col=0;col<rhs;++col) {
              Complex x=complex_value(actual[m*n*stride+row*stride+n+col]);
              double e=std::abs(x-complex_value(expected[m*n*rhs+row*rhs+col]));
              max_error=std::max(max_error,e);
              if (!std::isfinite(e)||e>2e-5) ++failures;
              Complex ax(0,0),b=complex_value(input[m*n*stride+row*stride+n+col]);
              for (unsigned k=0;k<n;++k)
                ax+=complex_value(input[m*n*stride+row*stride+k])*complex_value(actual[m*n*stride+k*stride+n+col]);
              residual2+=std::norm(ax-b); b2+=std::norm(b);
            }
          double residual=std::sqrt(residual2/b2); max_residual=std::max(max_residual,residual);
          if (!std::isfinite(residual)||residual>2e-6) ++failures;
        }
      }
      [cases addObject:@{@"size":@(n),@"rhs_columns":@(rhs),@"batch":@(batch),@"max_solution_error":@(max_error),
                        @"max_relative_residual":@(max_residual),@"singular_rejections":@(singular_checks),@"gpu_ms_after_two_warmups":times}];
    }
    NSDictionary *report=@{@"scope":@"Standalone pivoted complex solve; not full cache construction",@"device":device.name,
                           @"failures":@(failures),@"cases":cases};
    NSData *json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
    if (!json) return 2;
    fwrite(json.bytes,1,json.length,stdout); puts(""); return failures?1:0;
  }
}
