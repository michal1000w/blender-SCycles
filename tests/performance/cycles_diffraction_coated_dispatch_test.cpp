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
 for(float phase:{0.f,1.4f}) for(float alpha:{0.f,.3f}) for(float ni:{1.f,1.5f}) for(float no:{1.f,1.5f}) {
 ShaderData sd{};sd.N=sd.Ng=make_float3(0,0,1);sd.wi=normalize(make_float3(.4f,.1f,1));sd.num_closure_left=4;sd.object=0;
 DiffractionRoughDielectric p{{.37f,.24f,.42f,ni,no,phase},alpha,alpha};
 ShaderData limited{};limited.wi=sd.wi;limited.num_closure_left=ni==no?3:1;
 const int before=limited.num_closure_left;
 if(bsdf_diffraction_dielectric_setup_coated<test_distribution>(&limited,make_spectrum(.2f),make_spectrum(.8f),sd.N,make_float3(1,0,0),&p,1.7f,.47f,64) ||
    limited.num_closure!=0 || limited.num_closure_left!=before)return 3;
 if(!bsdf_diffraction_dielectric_setup_coated<test_distribution>(&sd,make_spectrum(.2f),make_spectrum(.8f),sd.N,make_float3(1,0,0),&p,1.7f,.47f,64))return 2;
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
 printf("phase=%g alpha=%g ni=%g no=%g dispatcher_events=%d failures=%d\n",phase,alpha,ni,no,checked,fail);
 total_fail+=fail+(checked<500);total_checked+=checked;
 }
 printf("total_events=%d failures=%d\n",total_checked,total_fail);
 return total_fail!=0;
}
