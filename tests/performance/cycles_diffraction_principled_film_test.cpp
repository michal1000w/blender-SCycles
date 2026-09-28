/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf.h"
#include <cstdio>
using namespace ccl;
#ifdef DIFFRACTION_TEST_BECKMANN
constexpr MicrofacetType test_distribution=BECKMANN;
#else
constexpr MicrofacetType test_distribution=GGX;
#endif
int main(){
 Profiler profiler;KernelGlobalsCPU base{};ThreadKernelGlobalsCPU kg(base,nullptr,profiler,0);KernelObject object{};kg.objects.data=&object;kg.objects.width=1;
 int total_fail=0,total_checked=0;
 for(float depth:{0.f,250.f}) for(float roughness:{0.f,.3f}) for(bool back:{false,true}) for(float ior:{1.f,1.5f}) {
 ShaderData sd{};sd.N=sd.Ng=make_float3(0,0,1);sd.wi=normalize(make_float3(.4f,.1f,1));sd.num_closure_left=4;sd.object=0;sd.rand_wavelength=.37f;
 if(back)sd.runtime_flag|=SR_BACKFACING;
 ShaderData limited{};limited.wi=sd.wi;limited.num_closure_left=ior==1.f?3:1;
 const int before=limited.num_closure_left;
 if(bsdf_diffraction_principled_transmission_setup(&limited,make_spectrum(.2f),make_spectrum(.8f),sd.N,make_float3(1,0,0),roughness,ior,1200,depth,.41f,one_spectrum(),0,1.7f,420) ||
    limited.num_closure!=0 || limited.num_closure_left!=before)return 3;
 if(!bsdf_diffraction_principled_transmission_setup(&sd,make_spectrum(.2f),make_spectrum(.8f),sd.N,make_float3(1,0,0),roughness,ior,1200,depth,.41f,one_spectrum(),0,1.7f,420))return 2;
 int fail=0;
 int checked=0;
 for(int i=0;i<1000;++i) for(int c=0;c<sd.num_closure;++c){
  Spectrum eval;float3 wo;float pdf,eta;float2 rough;
  int label=bsdf_sample(&kg,&sd,&sd.closure[c],make_float3((i+.5f)/1000.f,fmodf(i*.6180339887f+.3f,1.f),fmodf(i*.4142135623f+.2f,1.f)),&eval,&wo,&pdf,&rough,&eta);
  if(label==LABEL_NONE) continue;
  ++checked;
  float query_pdf;Spectrum queried=(label&LABEL_SINGULAR)?bsdf_eval_delta(&kg,&sd,&sd.closure[c],wo,&query_pdf):bsdf_eval(&kg,&sd,&sd.closure[c],wo,&query_pdf);

  fail+=pdf<=0||fabsf(pdf-query_pdf)>1e-5f*max(1.f,pdf)||fabsf(average(eval-queried))>1e-5f*max(1.f,average(eval));
 }
 printf("depth=%g roughness=%g back=%d ior=%g dispatcher_events=%d failures=%d\n",depth,roughness,back,ior,checked,fail);
 total_fail+=fail+(checked<500);total_checked+=checked;
 }
 /* Independent normal-incidence quarter-wave AR limit: n_f=sqrt(n_s),
  * d=lambda/(4*n_f). A flat lossless interface must transmit unit power. */
 ShaderData ar{};ar.N=ar.Ng=ar.wi=make_float3(0,0,1);ar.object=0;
 ar.rand_wavelength=.37f;ar.num_closure_left=4;
 const float nf=sqrtf(1.5f),wavelength=1000*sample_wavelength(ar.rand_wavelength);
 if(!bsdf_diffraction_principled_transmission_setup(&ar,one_spectrum(),one_spectrum(),
     ar.N,make_float3(1,0,0),0,1.5f,1200,0,.41f,one_spectrum(),0,nf,wavelength/(4*nf)))return 4;
 float reflected_pdf,transmitted_pdf;
 const Spectrum reflected=bsdf_eval_delta(&kg,&ar,&ar.closure[0],make_float3(0,0,1),&reflected_pdf);
 const Spectrum transmitted=bsdf_eval_delta(&kg,&ar,&ar.closure[0],make_float3(0,0,-1),&transmitted_pdf);
 total_fail+=reduce_max(fabs(reflected))*1e-6f>1e-6f ||
             reduce_max(fabs(transmitted*1e-6f-one_spectrum()))>1e-6f ||
             fabsf(transmitted_pdf*1e-6f-1)>1e-6f;
 printf("quarter_wave_reflectance=%g transmittance=%g\n",average(reflected)*1e-6f,average(transmitted)*1e-6f);
 printf("total_events=%d failures=%d\n",total_checked,total_fail);
 return total_fail!=0;
}
