/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include <random>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
using namespace ccl;
#ifdef DIFFRACTION_TEST_COATED
constexpr bool test_coated=true;
#else
constexpr bool test_coated=false;
#endif
#ifdef DIFFRACTION_TEST_BECKMANN
constexpr MicrofacetType test_distribution=BECKMANN;
#else
constexpr MicrofacetType test_distribution=GGX;
#endif
int main(int argc,char **argv) {
 if(argc!=2)return 2;
 @autoreleasepool {
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();if(!device)return 2;
  NSError *error=nil;
  NSString *source=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
  const bool safe_math=std::getenv("CYCLES_DIFFRACTION_TEST_SAFE_MATH")!=nullptr;
  MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=safe_math?MTLMathModeSafe:MTLMathModeFast;
  id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
  if(!library){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return 2;}
  id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"diffraction_rough_dielectric"] error:&error];if(!pipeline)return 2;
  constexpr unsigned count=65536;
  std::vector<float4> input,expected;std::mt19937 rng(618271);std::uniform_real_distribution<float> u(0,1);
  for(unsigned i=0;i<count;++i) {
   const float z=.02f+.98f*u(rng),phi=2*M_PI_F*u(rng),r=sqrtf(1-z*z);
   const float4 a=make_float4(r*cosf(phi),r*sinf(phi),z,u(rng));
   // Cycle both interface orientations; entering-only tests miss the
   // critical-angle boundary and total internal reflection.
   const float ni=i%3==2?1.5f:1.0f,no=i%3==1?1.5f:1.0f;
   const float4 b=make_float4(i%8==0?0:.05f+.7f*u(rng),i%8==0?0:.05f+.7f*u(rng),ni,no);
   const float4 c=make_float4(.2f+.8f*u(rng),u(rng),.2f+.6f*u(rng),6*u(rng));
   const float4 d=make_float4(u(rng),u(rng),0,0);
   input.insert(input.end(),{a,b,c,d});
   const DiffractionRoughDielectric p{{c.x,c.y,c.z,b.z,b.w,c.w},b.x,b.y};
   float3 wo;float value,pdf;bool singular=false;
#ifdef DIFFRACTION_TEST_COATED
   const bool valid=diffraction_dielectric_sample_coated_continuous<test_distribution>(
       &p,make_float3(a.x,a.y,a.z),make_float3(a.w,d.x,d.y),1.35f,.47f,&wo,&value,&pdf);
#else
   const bool valid=diffraction_dielectric_sample<test_distribution>(&p,make_float3(a.x,a.y,a.z),make_float3(a.w,d.x,d.y),&wo,&value,&pdf,&singular);
#endif
   expected.push_back(make_float4(wo.x,wo.y,wo.z,value));
   expected.push_back(make_float4(pdf,float(valid),float(singular),0));
  }
  id<MTLBuffer> in=[device newBufferWithBytes:input.data() length:input.size()*sizeof(float4) options:MTLResourceStorageModeShared];
  id<MTLBuffer> out=[device newBufferWithLength:count*2*sizeof(float4) options:MTLResourceStorageModeShared];
  id<MTLBuffer> reevaluated=[device newBufferWithLength:count*sizeof(float4) options:MTLResourceStorageModeShared];
  id<MTLComputePipelineState> eval_pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"diffraction_rough_dielectric_reevaluate"] error:&error];
  if(!eval_pipeline)return 2;
  id<MTLBuffer> world_output=[device newBufferWithLength:count*sizeof(float4) options:MTLResourceStorageModeShared];
  id<MTLComputePipelineState> world_pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"diffraction_dielectric_world_frame"] error:&error];
  if(!world_pipeline)return 2;
  id<MTLCommandQueue> queue=[device newCommandQueue];
  id<MTLBuffer> size_buffer=[device newBufferWithLength:2*sizeof(unsigned) options:MTLResourceStorageModeShared];
  id<MTLComputePipelineState> size_pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"diffraction_context_size"] error:&error];
  if(!size_pipeline)return 2;
  id<MTLCommandBuffer> size_command=[queue commandBuffer];
  id<MTLComputeCommandEncoder> size_encoder=[size_command computeCommandEncoder];
  [size_encoder setComputePipelineState:size_pipeline];[size_encoder setBuffer:size_buffer offset:0 atIndex:0];
  [size_encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
  [size_encoder endEncoding];[size_command commit];[size_command waitUntilCompleted];
  if(size_command.status!=MTLCommandBufferStatusCompleted)return 2;
  const unsigned context_size=*(const unsigned *)size_buffer.contents;
  if(context_size==0||context_size>16*1024*1024)return 2;
  id<MTLBuffer> params=[device newBufferWithLength:context_size options:MTLResourceStorageModeShared];
  memset(params.contents,0,context_size);
  id<MTLBuffer> object_buffer=nil;
#ifdef DIFFRACTION_TEST_SURFACE_MIXTURE
  const unsigned object_size=((const unsigned *)size_buffer.contents)[1];
  if(object_size==0||object_size>1024*1024)return 2;
  object_buffer=[device newBufferWithLength:object_size options:MTLResourceStorageModeShared];
  memset(object_buffer.contents,0,object_size);
  id<MTLComputePipelineState> bind_pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"diffraction_bind_objects"] error:&error];
  if(!bind_pipeline)return 2;
  id<MTLCommandBuffer> binding=[queue commandBuffer];
  id<MTLComputeCommandEncoder> bind_encoder=[binding computeCommandEncoder];
  [bind_encoder setComputePipelineState:bind_pipeline];
  [bind_encoder setBuffer:params offset:0 atIndex:0];[bind_encoder setBuffer:object_buffer offset:0 atIndex:1];
  [bind_encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
  [bind_encoder endEncoding];[binding commit];[binding waitUntilCompleted];
  if(binding.status!=MTLCommandBufferStatusCompleted)return 2;
#endif
  id<MTLCommandBuffer> command=[queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];[encoder setBuffer:in offset:0 atIndex:0];[encoder setBuffer:out offset:0 atIndex:1];[encoder setBuffer:params offset:0 atIndex:2];
  [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
  [encoder endEncoding];
  encoder=[command computeCommandEncoder];
  [encoder setComputePipelineState:eval_pipeline];
  [encoder setBuffer:in offset:0 atIndex:0];[encoder setBuffer:out offset:0 atIndex:1];
  [encoder setBuffer:params offset:0 atIndex:2];[encoder setBuffer:reevaluated offset:0 atIndex:3];
  [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(eval_pipeline.threadExecutionWidth,1,1)];
  [encoder endEncoding];
  encoder=[command computeCommandEncoder];[encoder setComputePipelineState:world_pipeline];
  if(object_buffer)[encoder useResource:object_buffer usage:MTLResourceUsageRead];
  [encoder setBuffer:in offset:0 atIndex:0];[encoder setBuffer:world_output offset:0 atIndex:1];[encoder setBuffer:params offset:0 atIndex:2];
  [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(world_pipeline.threadExecutionWidth,1,1)];
  [encoder endEncoding];[command commit];[command waitUntilCompleted];
  if(command.status!=MTLCommandBufferStatusCompleted)return 2;
  NSMutableArray *gpu_times=[NSMutableArray new];
  double gpu_median_ms=0;
  if(std::getenv("CYCLES_DIFFRACTION_TEST_BENCHMARK")) {
    std::vector<double> timings;
    // Compilation, host reference generation and the initial warm dispatch
    // are excluded. Each timed command runs the same complete local sampler.
    for(int repeat=0;repeat<9;++repeat) {
      id<MTLCommandBuffer> timed=[queue commandBuffer];
      id<MTLComputeCommandEncoder> enc=[timed computeCommandEncoder];
      [enc setComputePipelineState:pipeline];
      [enc setBuffer:in offset:0 atIndex:0];[enc setBuffer:out offset:0 atIndex:1];
      [enc setBuffer:params offset:0 atIndex:2];
      [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
      [enc endEncoding];[timed commit];[timed waitUntilCompleted];
      if(timed.status!=MTLCommandBufferStatusCompleted)return 2;
      const double ms=1000*(timed.GPUEndTime-timed.GPUStartTime);
      if(!(ms>0)||!std::isfinite(ms))return 2;
      timings.push_back(ms);[gpu_times addObject:@(ms)];
    }
    std::sort(timings.begin(),timings.end());gpu_median_ms=timings[timings.size()/2];
  }
  const float4 *reeval=(const float4 *)reevaluated.contents;
  int reeval_failures=0,reciprocity_failures=0;double max_reeval_error=0,max_reciprocity_error=0;
  const float4 *actual=(const float4 *)out.contents;
  int failures=0,pointwise=0,branch_mismatches=0;double max_pdf_error=0,max_direction_error=0;
  double cpu_energy[8]={},gpu_energy[8]={};
  unsigned continuous_by_interface[3]={},transmission_by_interface[3]={};
  double max_same_direction_pdf_error=0;int same_direction_differences=0;
  NSMutableArray *diagnostics=[NSMutableArray new];
  auto vector_json=[](float4 v)->NSArray * {return @[@(v.x),@(v.y),@(v.z),@(v.w)];};
  auto bin=[](float4 v){return int(v.x>=0)+2*int(v.y>=0)+4*int(v.z>=0);};
  for(unsigned i=0;i<count;++i) {
   const float4 a=actual[2*i],b=actual[2*i+1],e=expected[2*i],f=expected[2*i+1];
   failures+=!isfinite_safe(a)||!isfinite_safe(b);
   branch_mismatches+=b.y!=f.y||b.z!=f.z;
   if(b.y&&!b.z) {
    ++continuous_by_interface[i%3];
    transmission_by_interface[i%3]+=a.z<0;
    const double err=std::abs(double(reeval[i].y)-b.x)/std::max(1.,double(b.x));
    max_reeval_error=std::max(max_reeval_error,err);
    reeval_failures+=!isfinite_safe(reeval[i])||err>2e-4;
    const double reciprocal_error=std::abs(double(reeval[i].z)-reeval[i].w)/std::max(1.,std::max(double(reeval[i].z),double(reeval[i].w)));
    max_reciprocity_error=std::max(max_reciprocity_error,reciprocal_error);
    reciprocity_failures+=reciprocal_error>3e-4;
    if(reciprocal_error>3e-4&&diagnostics.count<32)
      [diagnostics addObject:@{@"reciprocity_case":@(i),@"input_wi_random":vector_json(input[4*i]),@"input_roughness_indices":vector_json(input[4*i+1]),@"input_grating":vector_json(input[4*i+2]),@"input_random":vector_json(input[4*i+3]),@"sampled_direction_value":vector_json(a),@"forward":@(reeval[i].z),@"reverse":@(reeval[i].w),@"scaled_error":@(reciprocal_error)}];
   }
   if(b.y) {
    failures+=b.x<=0||a.w<0||a.w>b.x*1.00001f;
    gpu_energy[bin(a)]+=a.w/b.x;
   }
   if(f.y)cpu_energy[bin(e)]+=e.w/f.x;
   if(b.y&&f.y) {
    const double direction=len(make_float3(a.x-e.x,a.y-e.y,a.z-e.z));
    const double density=std::abs(double(b.x)-f.x)/std::max(1.,double(f.x));
    max_direction_error=std::max(max_direction_error,direction);max_pdf_error=std::max(max_pdf_error,density);
    pointwise+=direction>2e-4||density>2e-4;
    if(!b.z&&!f.z) {
     const float4 ia=input[4*i],ib=input[4*i+1],ic=input[4*i+2];
     const DiffractionRoughDielectric p{{ic.x,ic.y,ic.z,ib.z,ib.w,ic.w},ib.x,ib.y};
     float same_pdf;
     diffraction_dielectric_eval<test_distribution>(&p,make_float3(ia.x,ia.y,ia.z),make_float3(a.x,a.y,a.z),&same_pdf
#ifdef DIFFRACTION_TEST_COATED
         ,1.35f,.47f,p.facet.incident_ior==p.facet.transmitted_ior
#endif
         );
     const double same_error=std::abs(double(b.x)-same_pdf)/std::max(1.,double(same_pdf));
     max_same_direction_pdf_error=std::max(max_same_direction_pdf_error,same_error);
     same_direction_differences+=same_error>2e-4;
     if((density>.01||same_error>2e-4)&&diagnostics.count<32) {
      [diagnostics addObject:@{@"case":@(i),@"input_wi_random":vector_json(input[4*i]),@"input_roughness_indices":vector_json(input[4*i+1]),@"input_grating":vector_json(input[4*i+2]),@"sampled_direction_value":vector_json(a),@"cpu_sample_pdf":@(f.x),@"gpu_sample_pdf":@(b.x),@"cpu_at_gpu_direction_pdf":@(same_pdf),@"direction_error":@(direction),@"same_direction_scaled_error":@(same_error)}];
     }
    }
   }
  }
  const float4 *world=(const float4 *)world_output.contents;
  int world_failures=0,world_atoms=0,world_continuous=0;double world_max_error=0;
  for(unsigned i=0;i<count;++i) {
   const float4 w=world[i];
   const bool world_failed=!isfinite_safe(w)||w.w!=0||(w.y&&w.x>(w.z?2e-5:3e-4));
   world_failures+=world_failed;
   if(world_failed&&diagnostics.count<64)
     [diagnostics addObject:@{@"world_case":@(i),@"error":@(w.x),@"valid":@(w.y),@"singular":@(w.z),@"bad_event":@(w.w)}];
   if(w.y) {world_atoms+=w.z!=0;world_continuous+=w.z==0;world_max_error=std::max(world_max_error,double(w.x));}
  }
  failures+=world_failures;
  // Compare densities at exactly the stored GPU direction. Different rounded
  // sample directions can legitimately straddle a fold; identical directions
  // must agree within the existing density threshold across CPU and Metal.
  failures+=same_direction_differences;
  double max_bin_error=0;
  for(int i=0;i<8;++i)max_bin_error=std::max(max_bin_error,std::abs(cpu_energy[i]-gpu_energy[i])/count);
  failures+=reeval_failures+reciprocity_failures;
  failures+=max_bin_error>.001||branch_mismatches>count*.001;
  NSDictionary *report=@{@"coated_continuous_local_sampler":@(test_coated),@"world_closure_coated":@NO,@"coated_facet_cases":@(count),@"world_frame_failures":@(world_failures),@"world_frame_atoms":@(world_atoms),@"world_frame_continuous":@(world_continuous),@"world_frame_max_error":@(world_max_error),@"reciprocity_failures":@(reciprocity_failures),@"max_reciprocity_scaled_error":@(max_reciprocity_error),@"gpu_reeval_failures":@(reeval_failures),@"gpu_reeval_max_scaled_error":@(max_reeval_error),@"safe_math":@(safe_math),@"same_direction_max_pdf_scaled_error":@(max_same_direction_pdf_error),@"same_direction_differences":@(same_direction_differences),@"diagnostics":diagnostics,@"sampler_gpu_ms":gpu_times,@"sampler_gpu_median_ms":@(gpu_median_ms),@"local_distribution":@(test_distribution==BECKMANN?"Beckmann":"GGX"),@"world_distribution":@(test_distribution==BECKMANN?"Beckmann":"GGX"),@"device":device.name,@"cases":@(count),@"interface_coverage":@"equal index, entering glass, leaving glass",@"continuous_by_interface":@[@(continuous_by_interface[0]),@(continuous_by_interface[1]),@(continuous_by_interface[2])],@"continuous_transmission_by_interface":@[@(transmission_by_interface[0]),@(transmission_by_interface[1]),@(transmission_by_interface[2])],@"max_pdf_scaled_error":@(max_pdf_error),@"max_direction_error":@(max_direction_error),@"pointwise_differences":@(pointwise),@"branch_mismatches":@(branch_mismatches),@"max_energy_bin_error":@(max_bin_error),@"failures":@(failures),@"passed":@(failures==0),@"scope":@"Combined local rough dielectric sampler, same-direction CPU/Metal PDFs and energy bins; not renderer integration or differing-direction PDF acceptance"};
  NSData *json=[NSJSONSerialization dataWithJSONObject:report options:0 error:&error];fwrite(json.bytes,1,json.length,stdout);fputc('\n',stdout);
  return failures?1:0;
 }
}
