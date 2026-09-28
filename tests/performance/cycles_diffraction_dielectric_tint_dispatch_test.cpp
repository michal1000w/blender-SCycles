/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#if defined(__aarch64__)
#  include <arm_neon.h>
#endif
#include "kernel/integrator/surface_shader.h"
#include <cstdio>
#include <random>
using namespace ccl;
template<MicrofacetType distribution>
static int check(KernelGlobals kg)
{
 int failures=0,events=0,rejected=0;float max_selection_error=0;
 std::mt19937 rng(483921);std::uniform_real_distribution<float> u(0,1);
 const Spectrum reflect=make_float3(.2f,.5f,.9f),transmit=make_float3(.8f,.3f,.1f);
 for(float alpha:{0.f,.3f}) for(float no:{1.f,1.5f}) {
  ShaderData base{},tinted{};
  base.N=base.Ng=make_float3(0,0,1);base.wi=normalize(make_float3(.4f,.1f,1));base.object=0;base.num_closure_left=3;
  tinted=base;
  const DiffractionRoughDielectric p{{.37f,.24f,.42f,1,no,1.4f},alpha,alpha};
  if(!bsdf_diffraction_dielectric_setup<distribution>(&base,one_spectrum(),base.N,make_float3(1,0,0),&p) ||
     !bsdf_diffraction_dielectric_setup_tinted<distribution>(&tinted,reflect,transmit,base.N,make_float3(1,0,0),&p)) return 1;
  // Blur must modify the auxiliary parameters, preserving their colors.
  for(int j=0;j<tinted.num_closure;++j) {
    if(tinted.closure[j].type==CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID)continue;
    auto *b=(DiffractionDielectricBsdf *)&tinted.closure[j];
    const Spectrum saved_reflection=b->extra->reflection,saved_transmission=b->extra->transmission;
    bsdf_blur(&tinted.closure[j],.7f);
    failures+=fabsf(bsdf_get_specular_roughness_squared(&tinted.closure[j])-.49f)>1e-6f;
    failures+=!is_zero(b->extra->reflection-saved_reflection)||!is_zero(b->extra->transmission-saved_transmission);
    b->extra->param.alpha_x=b->extra->param.alpha_y=alpha;
    failures+=!bsdf_diffraction_dielectric_has_transmission(&tinted.closure[j]);
    ShaderClosure filtered=tinted.closure[j];
    failures+=!bsdf_diffraction_dielectric_filter(&filtered,true,false);
    auto *f=(DiffractionDielectricBsdf *)&filtered;
    failures+=(f->disabled_lobes&LABEL_REFLECT)==0 || (f->disabled_lobes&LABEL_TRANSMIT)!=0;
    failures+=!bsdf_diffraction_dielectric_has_transmission(&filtered);
    failures+=bsdf_diffraction_dielectric_filter(&filtered,false,true) || filtered.sample_weight!=0;
    failures+=bsdf_diffraction_dielectric_has_transmission(&filtered);
  }
  ShaderClosure straight{};straight.type=CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID;straight.sample_weight=1;
  failures+=!bsdf_diffraction_dielectric_filter(&straight,true,false);
  failures+=bsdf_diffraction_dielectric_filter(&straight,false,true) || straight.sample_weight!=0;
  int picked[MAX_CLOSURE]={};
  for(int i=0;i<10000;++i) {
   float3 selection=make_float3(u(rng),u(rng),(i+.5f)/10000.f);
   const ShaderClosure *selected=surface_shader_bsdf_bssrdf_pick(&tinted,&selection);
   failures+=selected<&tinted.closure[0] || selected>=&tinted.closure[tinted.num_closure] ||
             selection.z<0 || selection.z>=1;
   ++picked[selected-&tinted.closure[0]];
   float3 wo;Spectrum value;float pdf,eta;float2 roughness;
   const int label=bsdf_sample(kg,&tinted,selected,selection,&value,&wo,&pdf,&roughness,&eta);
   if(label==LABEL_NONE) {++rejected;continue;}
   Spectrum reference=zero_spectrum(),actual=zero_spectrum();
   for(int side=0;side<2;++side) {
    ShaderData *sd=side?&tinted:&base;Spectrum sum=zero_spectrum();
    for(int j=0;j<sd->num_closure;++j) {
     float density;
     const Spectrum evaluated=(label&LABEL_SINGULAR)?
       bsdf_eval_delta(kg,sd,&sd->closure[j],wo,&density):
       bsdf_eval(kg,sd,&sd->closure[j],wo,&density);
     sum+=sd->closure[j].weight*evaluated;
    }
    if(side)actual=sum;else reference=sum;
   }
   BsdfEval combined;float average_roughness=0;
   float pdfs[MAX_CLOSURE];
   const float mixture_pdf=(label&LABEL_SINGULAR)?
       surface_shader_bsdf_eval_delta(kg,&tinted,wo,&combined):
       surface_shader_bsdf_eval_pdfs(kg,&tinted,wo,&combined,pdfs,0,average_roughness);
   BsdfEval sampled_sum;bsdf_eval_init(&sampled_sum,zero_spectrum());
   bsdf_eval_accum(&sampled_sum,selected,wo,value*selected->weight);
   float sampled_roughness=0;
   const float selected_weight=selected->sample_weight;
   const float sampled_mixture_pdf=_surface_shader_bsdf_eval_mis(
       kg,&tinted,wo,selected,&sampled_sum,pdf*selected_weight,selected_weight,0,
       pdf*selected_weight*bsdf_get_specular_roughness_squared(selected),sampled_roughness);
   failures+=fabsf(sampled_mixture_pdf-mixture_pdf)>2e-5f*max(1.f,mixture_pdf);
   failures+=reduce_max(fabs(bsdf_eval_sum(&sampled_sum)-actual))>2e-5f*max(1.f,reduce_max(fabs(actual)));
   float weighted_pdf=0,total_weight=0;
   for(int j=0;j<tinted.num_closure;++j) {
     float density;
     if(label&LABEL_SINGULAR)bsdf_eval_delta(kg,&tinted,&tinted.closure[j],wo,&density);
     else bsdf_eval(kg,&tinted,&tinted.closure[j],wo,&density);
     weighted_pdf+=density*tinted.closure[j].sample_weight;
     total_weight+=tinted.closure[j].sample_weight;
   }
   failures+=fabsf(mixture_pdf-weighted_pdf/total_weight)>2e-5f*max(1.f,mixture_pdf);
   failures+=reduce_max(fabs(bsdf_eval_sum(&combined)-actual))>2e-5f*max(1.f,reduce_max(fabs(actual)));
   reference*=((label&LABEL_TRANSMIT)?transmit:reflect);
   failures+=!isfinite_safe(actual)||reduce_max(fabs(actual-reference))>2e-5f*max(1.f,reduce_max(fabs(reference)));
   ++events;
  }
  float total=0;for(int j=0;j<tinted.num_closure;++j)total+=tinted.closure[j].sample_weight;
  for(int j=0;j<tinted.num_closure;++j) {
    const float error=fabsf(picked[j]-10000*tinted.closure[j].sample_weight/total);
    max_selection_error=max(max_selection_error,error);
    failures+=error>2; // Stratified uniform selection: at most two boundary samples.
  }
 }
 printf("distribution=%d weighted_events=%d rejected=%d max_selection_count_error=%g failures=%d\n",int(distribution),events,rejected,max_selection_error,failures);
 return failures+(events+rejected!=40000)+(events==0);
}
int main()
{
 Profiler profiler;KernelGlobalsCPU base{};ThreadKernelGlobalsCPU kg(base,nullptr,profiler,0);
 KernelObject object{};kg.objects.data=&object;kg.objects.width=1;
 return check<GGX>(&kg)+check<BECKMANN>(&kg)!=0;
}
