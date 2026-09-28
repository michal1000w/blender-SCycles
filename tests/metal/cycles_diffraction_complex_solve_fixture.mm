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
  if (argc!=3) return 2;
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
    FILE *fixture=fopen(argv[2],"rb");
    if (!fixture) return 2;
    unsigned records=0;
    if (fread(&records,sizeof(unsigned),1,fixture)!=1 || records>10000) return 2;
    for (unsigned record=0;record<records;++record) {
      unsigned n=0,rhs=0,batch=1;
      if (fread(&n,sizeof(unsigned),1,fixture)!=1 || fread(&rhs,sizeof(unsigned),1,fixture)!=1 ||
          n==0 || n>1024 || rhs==0 || rhs>2048) return 2;
      unsigned stride=n+rhs;
      std::vector<Pair> input(n*stride),expected(n*rhs);
      if (fread(input.data(),sizeof(Pair),input.size(),fixture)!=input.size() ||
          fread(expected.data(),sizeof(Pair),expected.size(),fixture)!=expected.size()) return 2;
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
      [cases addObject:@{@"record":@(record),@"size":@(n),@"rhs_columns":@(rhs),@"batch":@(batch),@"max_solution_error":@(max_error),
                        @"max_relative_residual":@(max_residual),@"singular_rejections":@(singular_checks),@"gpu_ms_after_two_warmups":times}];
    }
    if (fgetc(fixture)!=EOF) return 2;
    fclose(fixture);
    NSDictionary *report=@{@"scope":@"Captured propagation systems; standalone GPU solve, not full GPU propagation",@"device":device.name,
                           @"failures":@(failures),@"cases":cases};
    NSData *json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
    if (!json) return 2;
    fwrite(json.bytes,1,json.length,stdout); puts(""); return failures?1:0;
  }
}
