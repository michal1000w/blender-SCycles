/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>
#include <cstdint>
struct Pair { float x, y; };

int main(int argc, const char **argv)
{
  if (argc != 2) return 2;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return 2;
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe; // Required: error-free transforms forbid reassociation.
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) { fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 2; }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSMutableArray *cases = [NSMutableArray array];
    unsigned total_failures = 0;
    uint32_t state = 9321;
    auto random = [&]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                         return float(int(state & 65535) - 32768) / 32768.0f; };
    for (unsigned n : {3u, 17u, 66u, 132u}) {
      const unsigned batch = 4, count = batch*n*n;
      std::vector<Pair> a(count), b(count), reference(count);
      for (unsigned i = 0; i < count; ++i) {
        unsigned matrix = i/(n*n);
        float scale = matrix == 1 ? 1024.0f : matrix == 2 ? 0.001f : 1.0f;
        a[i] = {random()*scale, random()*scale};
        b[i] = {random()/scale, random()/scale};
      }
      // Alternating nearly cancelling terms in the fourth matrix.
      for (unsigned row=0; row<n; ++row)
        for (unsigned k=0; k+1<n; k+=2) a[3*n*n+row*n+k+1]=a[3*n*n+row*n+k];
      for (unsigned k=0; k+1<n; k+=2)
        for (unsigned col=0; col<n; ++col) {
          auto x=b[3*n*n+k*n+col];
          b[3*n*n+(k+1)*n+col]={-x.x+0.000001f,-x.y};
        }
      for (unsigned matrix=0; matrix<batch; ++matrix)
        for (unsigned row=0; row<n; ++row)
          for (unsigned col=0; col<n; ++col) {
            std::complex<double> sum(0,0);
            for (unsigned k=0; k<n; ++k) {
              auto x=a[matrix*n*n+row*n+k], y=b[matrix*n*n+k*n+col];
              sum += std::complex<double>(x.x,x.y)*std::complex<double>(y.x,y.y);
            }
            reference[matrix*n*n+row*n+col]={float(sum.real()),float(sum.imag())};
          }
      id<MTLBuffer> ab=[device newBufferWithBytes:a.data() length:count*sizeof(Pair) options:MTLResourceStorageModeShared];
      id<MTLBuffer> bb=[device newBufferWithBytes:b.data() length:count*sizeof(Pair) options:MTLResourceStorageModeShared];
      id<MTLBuffer> out=[device newBufferWithLength:count*sizeof(Pair) options:MTLResourceStorageModeShared];
      for (NSString *name in @[@"ordinary_product", @"compensated_product"]) {
        id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
        if (!pipeline) return 2;
        NSMutableArray *times=[NSMutableArray array];
        double maximum=0; unsigned failures=0;
        for (unsigned run=0; run<5; ++run) {
          id<MTLCommandBuffer> command=[queue commandBuffer];
          id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
          [encoder setComputePipelineState:pipeline];
          [encoder setBuffer:ab offset:0 atIndex:0]; [encoder setBuffer:bb offset:0 atIndex:1];
          [encoder setBuffer:out offset:0 atIndex:2]; [encoder setBytes:&n length:sizeof(n) atIndex:3];
          [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
          [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
          if (command.status != MTLCommandBufferStatusCompleted) return 2;
          if (run>=2) [times addObject:@(1000*(command.GPUEndTime-command.GPUStartTime))];
          const Pair *actual=static_cast<const Pair *>(out.contents);
          for (unsigned i=0; i<count; ++i) {
            double e=std::hypot(double(actual[i].x)-reference[i].x,double(actual[i].y)-reference[i].y);
            maximum=std::max(maximum,e);
            // Product-kernel gate, independent of the full Maxwell solver gate.
            double tolerance=2e-6+1e-6*std::hypot(reference[i].x,reference[i].y);
            if (!std::isfinite(e) || e>tolerance) ++failures;
          }
        }
        if ([name isEqualToString:@"compensated_product"]) total_failures+=failures;
        [cases addObject:@{@"kernel":name,@"size":@(n),@"batch":@(batch),@"failures_over_five_dispatches":@(failures),
                          @"max_complex_error":@(maximum),@"gpu_ms_after_two_warmups":times}];
      }
    }
    NSDictionary *report=@{@"scope":@"Standalone matrix products; not full cache construction or render timing",
                            @"math_mode":@"safe",@"device":device.name,@"compensated_failures":@(total_failures),@"cases":cases};
    NSData *json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
    if (!json) return 2;
    fwrite(json.bytes,1,json.length,stdout); puts("");
    return total_failures ? 1 : 0;
  }
}
