/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
struct Pair { float x,y; };
// Binary diagnostic protocol over stdin/stdout. Each request is operation,
// rows, inner, columns followed by two row-major complex-float matrices.
// Response is status then the output matrix. No CPU arithmetic fallback.
int main(int argc,const char **argv)
{
  if (argc!=2) return 2;
  @autoreleasepool {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice(); if (!device) return 2;
    fprintf(stderr,"Matrix diagnostic device: %s\n",device.name.UTF8String);
    NSError *error=nil;
    NSString *source=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    if (!library) { fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return 2; }
    id<MTLComputePipelineState> multiply=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rectangular_product"] error:&error];
    id<MTLComputePipelineState> solve=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"complex_solve"] error:&error];
    id<MTLComputePipelineState> residual=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rectangular_residual"] error:&error];
    if (!multiply||!solve||!residual) return 2;
    id<MTLCommandQueue> queue=[device newCommandQueue];
    unsigned header[4];
    while (true) {
      size_t got=fread(header,sizeof(unsigned),4,stdin);
      if (got==0 && feof(stdin)) break;
      if (got!=4) return 2;
      @autoreleasepool {
        unsigned operation=header[0],rows=header[1],inner=header[2],cols=header[3];
        if (operation>2||!rows||!inner||!cols||rows>2048||inner>2048||cols>4096 ||
            (operation==1 && rows!=inner)) return 2;
        std::vector<Pair> a(rows*inner),b(inner*cols),result(rows*cols);
        if (fread(a.data(),sizeof(Pair),a.size(),stdin)!=a.size() ||
            fread(b.data(),sizeof(Pair),b.size(),stdin)!=b.size()) return 2;
        std::vector<Pair> initial(rows*cols);
        if (operation==2 && fread(initial.data(),sizeof(Pair),initial.size(),stdin)!=initial.size()) return 2;
        id<MTLBuffer> ab=nil,bb=nil,output=nil,status=nil,cb=nil;
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        if (operation!=1) {
          ab=[device newBufferWithBytes:a.data() length:a.size()*sizeof(Pair) options:MTLResourceStorageModeShared];
          bb=[device newBufferWithBytes:b.data() length:b.size()*sizeof(Pair) options:MTLResourceStorageModeShared];
          output=[device newBufferWithLength:result.size()*sizeof(Pair) options:MTLResourceStorageModeShared];
          unsigned shape[4]={rows,inner,cols,0};
          [encoder setComputePipelineState:operation==2?residual:multiply];
          if (operation==2) {
            cb=[device newBufferWithBytes:initial.data() length:initial.size()*sizeof(Pair) options:MTLResourceStorageModeShared];
            [encoder setBuffer:cb offset:0 atIndex:4];
          }
          [encoder setBuffer:ab offset:0 atIndex:0];[encoder setBuffer:bb offset:0 atIndex:1];
          [encoder setBuffer:output offset:0 atIndex:2];[encoder setBytes:shape length:sizeof(shape) atIndex:3];
          [encoder dispatchThreads:MTLSizeMake(result.size(),1,1) threadsPerThreadgroup:MTLSizeMake(multiply.threadExecutionWidth,1,1)];
        }
        else {
          std::vector<Pair> workspace(rows*(rows+cols));
          for (unsigned row=0;row<rows;++row) {
            memcpy(workspace.data()+row*(rows+cols),a.data()+row*rows,rows*sizeof(Pair));
            memcpy(workspace.data()+row*(rows+cols)+rows,b.data()+row*cols,cols*sizeof(Pair));
          }
          output=[device newBufferWithBytes:workspace.data() length:workspace.size()*sizeof(Pair) options:MTLResourceStorageModeShared];
          status=[device newBufferWithLength:sizeof(unsigned) options:MTLResourceStorageModeShared];
          [encoder setComputePipelineState:solve];[encoder setBuffer:output offset:0 atIndex:0];
          [encoder setBuffer:status offset:0 atIndex:1];[encoder setBytes:&rows length:sizeof(rows) atIndex:2];
          [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
          [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(128,solve.maxTotalThreadsPerThreadgroup),1,1)];
        }
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        if (command.status!=MTLCommandBufferStatusCompleted) return 2;
        unsigned code=operation!=1?1:*static_cast<unsigned *>(status.contents);
        if (operation!=1) memcpy(result.data(),output.contents,result.size()*sizeof(Pair));
        else {
          const Pair *workspace=static_cast<const Pair *>(output.contents);
          for (unsigned row=0;row<rows;++row)
            memcpy(result.data()+row*cols,workspace+row*(rows+cols)+rows,cols*sizeof(Pair));
        }
        if (fwrite(&code,sizeof(unsigned),1,stdout)!=1 ||
            fwrite(result.data(),sizeof(Pair),result.size(),stdout)!=result.size()) return 2;
        fflush(stdout);
      }
    }
  }
  return 0;
}
