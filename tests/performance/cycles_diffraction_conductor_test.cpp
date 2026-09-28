/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf.h"
#include <cstdio>
using namespace ccl;
int main()
{
  Profiler profiler;KernelGlobalsCPU base{};base.data.film.is_rec709=1;
  ThreadKernelGlobalsCPU kg(base,nullptr,profiler,0);
  KernelObject object{};kg.objects.data=&object;kg.objects.width=1;
  int failures=0,events=0;float flat_error=0,reciprocity_error=0;
  for (bool beckmann:{false,true}) for (int model:{0,1,2,3})
  for (float alpha:{0.f,.25f}) for (float depth:{0.f,170.f}) for (float coverage:{.5f,1.f}) {
    ShaderData sd{};sd.N=sd.Ng=make_float3(0,0,1);
    sd.wi=normalize(make_float3(.4f,.2f,1));sd.object=0;sd.rand_wavelength=.37f;
    sd.num_closure_left=8;
    MicrofacetBsdf *source=(MicrofacetBsdf *)bsdf_alloc(&sd,sizeof(MicrofacetBsdf),one_spectrum());
    source->N=sd.N;source->T=make_float3(1,0,0);
    source->alpha_x=alpha;source->alpha_y=.5f*alpha;source->ior=1;
    if(beckmann)bsdf_microfacet_beckmann_setup(source);else bsdf_microfacet_ggx_setup(source);
    if(model==2) {
      auto *f=(FresnelGeneralizedSchlick *)closure_alloc_extra(&sd,sizeof(FresnelGeneralizedSchlick));
      *f={};f->f0=rgb_to_spectrum(make_float3(.02f,.03f,.04f));
      f->f90=one_spectrum();f->exponent=-1.5f;
      f->tint={one_spectrum(),zero_spectrum()};source->ior=1.5f;
      source->fresnel=f;source->fresnel_type=MicrofacetFresnel::GENERALIZED_SCHLICK;
    }
    else if(model==1) {
      auto *f=(FresnelF82Tint *)closure_alloc_extra(&sd,sizeof(FresnelF82Tint));
      *f={};f->f0=rgb_to_spectrum(make_float3(.4f,.6f,.8f));
      source->fresnel=f;source->fresnel_type=MicrofacetFresnel::F82_TINT;
    }
    else if(model==0) {
      auto *f=(FresnelConductor *)closure_alloc_extra(&sd,sizeof(FresnelConductor));
      *f={};f->ior={rgb_to_spectrum(make_float3(.3f,.8f,1.3f)),make_spectrum(3.f)};
      source->fresnel=f;source->fresnel_type=MicrofacetFresnel::CONDUCTOR;
    }
    const MicrofacetBsdf native_before=*source;
    const Spectrum fresnel_before=microfacet_fresnel(&kg,source,.6f,nullptr).reflectance;
    if(!bsdf_diffraction_conductor_split_setup(&sd,source,make_float3(1,0,0),1200,depth,.41f,coverage))return 2;
    if(sd.num_closure!=(depth==0?1:(coverage<1?2:1)))return 3;
    if(depth==0) {
      failures+=source->type!=native_before.type ||
          !isequal(source->weight,native_before.weight) ||
          source->sample_weight!=native_before.sample_weight ||
          source->fresnel!=native_before.fresnel ||
          source->fresnel_type!=native_before.fresnel_type ||
          source->energy_scale!=native_before.energy_scale ||
          source->diffraction_wavelength_nm!=0 ||
          !isequal(microfacet_fresnel(&kg,source,.6f,nullptr).reflectance,fresnel_before) ||
          (sd.runtime_flag&SR_BSDF_HAS_DISPERSION);
    }
    Spectrum sum=zero_spectrum();
    for(int c=0;c<sd.num_closure;++c)sum+=sd.closure[c].weight;
    if(reduce_max(fabs(sum-one_spectrum()))>1e-6f)return 4;
    const ShaderClosure *sc=&sd.closure[depth>0 && coverage<1?1:0];
    const auto *b=depth>0?(const DiffractionConductorBsdf *)sc:nullptr;
    for(int i=0;i<1000;++i) {
      Spectrum value;float3 wo;float pdf,eta;float2 rough;
      const int label=bsdf_sample(&kg,&sd,sc,make_float3((i+.5f)/1000,
          fmodf(i*.6180339887f+.3f,1.f),fmodf(i*.4142135623f+.2f,1.f)),
          &value,&wo,&pdf,&rough,&eta);
      if(label==LABEL_NONE)continue;
      ++events;float query_pdf;
      const Spectrum query=(label&LABEL_SINGULAR)?bsdf_eval_delta(&kg,&sd,sc,wo,&query_pdf):
                                                             bsdf_eval(&kg,&sd,sc,wo,&query_pdf);
      failures+=!(pdf>0) || fabsf(pdf-query_pdf)>3e-5f*max(1.f,pdf) ||
                reduce_max(fabs(value-query))>3e-5f*max(1.f,reduce_max(value));
      if(depth==0) {
        float reference_pdf;
        const Spectrum reference=(label&LABEL_SINGULAR)?
            bsdf_eval_delta(&kg,&sd,(const ShaderClosure *)&native_before,wo,&reference_pdf):
            bsdf_eval(&kg,&sd,(const ShaderClosure *)&native_before,wo,&reference_pdf);
        const float e=reduce_max(fabs(value-reference))/max(1.f,reduce_max(reference));
        flat_error=max(flat_error,e);
        failures+=e>3e-5f || fabsf(pdf-reference_pdf)>3e-5f*max(1.f,pdf);
      }
      if(!(label&LABEL_SINGULAR)) {
        float reverse_pdf;
        const Spectrum reverse=depth>0?
            bsdf_diffraction_conductor_eval(&kg,sc,wo,sd.wi,&reverse_pdf):
            (beckmann?bsdf_microfacet_eval<BECKMANN>(&kg,sc,wo,sd.wi,&reverse_pdf):
                      bsdf_microfacet_eval<GGX>(&kg,sc,wo,sd.wi,&reverse_pdf));
        const float error=reduce_max(fabs(value/wo.z-reverse/sd.wi.z))/max(1.f,reduce_max(value/wo.z));
        reciprocity_error=max(reciprocity_error,error);
        failures+=error>3e-4f;
        if(depth>0 && model==3 && !beckmann) {
          float old_pdf;
          const float old_value=diffraction_reflection_eval(&b->extra->param,sd.wi,wo,&old_pdf);
          failures+=fabsf(old_pdf-query_pdf)>3e-4f*max(1.f,old_pdf) ||
                    reduce_max(fabs(value-make_spectrum(old_value)))>3e-4f*max(1.f,old_value);
        }

      }
    }
  }
  /* Independent flat-interface grating equation, including a surrounding medium.
   * This checks outgoing momentum rather than repeating the setup conversion. */
  int displaced_orders=0;
  for (float medium:{1.f,1.5f}) for (bool beckmann:{false,true}) {
    ShaderData sd{};sd.N=sd.Ng=make_float3(0,0,1);
    sd.wi=normalize(make_float3(.1f,.2f,1));sd.object=0;sd.rand_wavelength=.37f;
    sd.num_closure_left=8;
    if(!bsdf_diffraction_glossy_setup(&kg,&sd,one_spectrum(),sd.N,make_float3(1,0,0),
                                    0,0,1200,170,.41f,medium,beckmann))return 5;
    for(int i=0;i<1000;++i) {
      Spectrum value;float3 wo;float pdf,eta;float2 rough;
      const int label=bsdf_sample(&kg,&sd,&sd.closure[0],
          make_float3((i+.5f)/1000,.3f,.7f),&value,&wo,&pdf,&rough,&eta);
      if(label==LABEL_NONE)continue;
      const float m=(sd.wi.x+wo.x)*medium*1200/(1000*sample_wavelength(sd.rand_wavelength));
      failures+=fabsf(m-roundf(m))>2e-4f || fabsf(sd.wi.y+wo.y)>2e-5f;
      displaced_orders+=fabsf(m)>.5f;
    }
  }
  failures+=displaced_orders==0;
  printf("events=%d failures=%d flat_max_scaled_error=%g reciprocity_max_scaled_error=%g\n",
         events,failures,flat_error,reciprocity_error);
  return failures!=0;
}
