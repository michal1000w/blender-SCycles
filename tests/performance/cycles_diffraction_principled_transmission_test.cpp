/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf.h"
#include <cstdio>
using namespace ccl;
static_assert(sizeof(DiffractionDielectricCoatingExtra)<=sizeof(ShaderClosure));
static_assert(sizeof(DiffractionDielectricGeneralizedExtra)<=sizeof(ShaderClosure));
int main()
{
  Profiler profiler;KernelGlobalsCPU base{};ThreadKernelGlobalsCPU kg(base,nullptr,profiler,0);
  KernelObject object{};kg.objects.data=&object;kg.objects.width=1;
  int failures=0,events=0,flat=0,reciprocal=0;float maximum_flat=0,maximum_reciprocity=0;
  for(bool back:{false,true}) for(bool colored:{false,true})
  for(float roughness:{0.f,.2f,.6f}) for(float depth:{0.f,250.f})
  for(float tint_scale:{0.f,.5f,1.f,1.5f,30.f}) {
    ShaderData sd{};sd.N=sd.Ng=make_float3(0,0,1);
    sd.wi=normalize(make_float3(.4f,.2f,1));sd.object=0;sd.rand_wavelength=.37f;sd.num_closure_left=8;
    if(back)sd.runtime_flag|=SR_BACKFACING;
    const Spectrum tint=colored?rgb_to_spectrum(make_float3(.3f,.7f,.9f)):one_spectrum();
    const Spectrum specular_tint=tint_scale*rgb_to_spectrum(make_float3(1,.7f,.3f));
    if(!bsdf_diffraction_principled_transmission_setup(&sd,one_spectrum(),tint,sd.N,make_float3(1,0,0),
                                    roughness,1.5f,1200,depth,.41f,specular_tint))return 2;
    MicrofacetBsdf native{};native.N=sd.N;native.alpha_x=native.alpha_y=sqr(roughness);
    native.ior=back?1/1.5f:1.5f;bsdf_microfacet_ggx_glass_setup(&native);
    FresnelGeneralizedSchlick fresnel=generalized_schlick_setup(1.5f,true,true,specular_tint,tint,{});
    fresnel.f0=saturate(fresnel.f0);
    native.fresnel=&fresnel;native.fresnel_type=MicrofacetFresnel::GENERALIZED_SCHLICK;
    const ShaderClosure *sc=&sd.closure[0];
    for(int i=0;i<2000;++i) {
      Spectrum value;float3 wo;float pdf,eta;float2 rough;
      const int label=bsdf_sample(&kg,&sd,sc,make_float3((i+.5f)/2000,
          fmodf(i*.6180339887f+.3f,1.f),fmodf(i*.4142135623f+.2f,1.f)),
          &value,&wo,&pdf,&rough,&eta);
      if(label==LABEL_NONE)continue;
      ++events;float qp;const Spectrum q=(label&LABEL_SINGULAR)?bsdf_eval_delta(&kg,&sd,sc,wo,&qp):bsdf_eval(&kg,&sd,sc,wo,&qp);
      failures+=!(pdf>0) || !isfinite_safe(value) || reduce_min(value)<0 ||
          fabsf(pdf-qp)>3e-5f*max(1.f,pdf) || reduce_max(fabs(value-q))>3e-5f*max(1.f,reduce_max(value));
      if(!(label&LABEL_SINGULAR)) {
        const auto *b=(const DiffractionDielectricBsdf *)sc;
        auto reverse=*(const DiffractionDielectricGeneralizedExtra *)b->extra;
        const bool trans=wo.z<0;
        float3 ri=wo,ro=sd.wi;
        float eta2=1;
        if(trans) {
          const auto original=reverse.base.param;
          diffraction_dielectric_parameters(1000*sample_wavelength(sd.rand_wavelength),1200,depth,.41f,
              original.facet.transmitted_ior,original.facet.incident_ior,
              original.alpha_x,original.alpha_y,&reverse.base.param);
          reverse.generalized_reference_f0=F0_from_ior(back?1.5f:1/1.5f);
          ri=make_float3(wo.x,-wo.y,-wo.z);ro=make_float3(sd.wi.x,-sd.wi.y,-sd.wi.z);
          eta2=sqr(original.facet.incident_ior/original.facet.transmitted_ior);
        }
        float rp;const Spectrum reversed=diffraction_dielectric_generalized_eval<GGX>(&reverse,ri,ro,&rp)*(trans?tint:one_spectrum());
        const Spectrum f=sc->weight*value/fabsf(wo.z)*eta2,g=reversed/fabsf(ro.z);
        const float error=reduce_max(fabs(f-g))/max(1.f,max(reduce_max(f),reduce_max(g)));
        maximum_reciprocity=max(maximum_reciprocity,error);failures+=error>3e-4f;++reciprocal;
      }
      if(depth==0) {
        float np;const Spectrum reference=(label&LABEL_SINGULAR)?bsdf_eval_delta(&kg,&sd,(ShaderClosure *)&native,wo,&np):
            bsdf_microfacet_eval<GGX>(&kg,(ShaderClosure *)&native,sd.wi,wo,&np);
        const float error=reduce_max(fabs(sc->weight*value-reference))/max(1.f,reduce_max(reference));
        maximum_flat=max(maximum_flat,error);failures+=error>3e-4f;++flat;
      }
    }
  }
  int facet_cases=0;float minimum_power=1,maximum_power=0;
  for(float ni:{1.f,1.5f}) for(float pitch:{400.f,1200.f,4000.f})
  for(float depth:{250.f,1000.f}) for(float duty:{.1f,.5f,.9f})
  for(float cosine:{.05f,.3f,1.f}) for(float tint_scale:{0.f,.5f,30.f}) {
    DiffractionDielectricGeneralizedExtra e{};
    if(!diffraction_dielectric_parameters(550,pitch,depth,duty,ni,ni==1?1.5f:1.f,.04f,.04f,&e.base.param))return 3;
    e.generalized_reference_f0=F0_from_ior(1.5f);
    e.generalized_f0=saturate(e.generalized_reference_f0*tint_scale*rgb_to_spectrum(make_float3(1,.7f,.3f)));
    const float3 wi=make_float3(sqrtf(1-cosine*cosine),0,cosine),h=make_float3(0,0,1);
    const auto *p=&e.base.param.facet;
    for(int side=0;side<2;++side) {
      const bool trans=side!=0;const float eta=p->incident_ior/p->transmitted_ior;
      const int bound=diffraction_dielectric_facet_active_order_bound(p,trans);
      for(int m=-bound;m<=bound;++m) {
        float3 wo;const float delta=m*p->wavelength_over_pitch;
        const bool valid=trans?diffraction_facet_transmit(wi,h,make_float3(1,0,0),eta,delta*eta,&wo):
                              diffraction_facet_reflect(wi,h,make_float3(1,0,0),delta,&wo);
        if(!valid)continue;
        const Spectrum power=diffraction_dielectric_generalized_power(&e,m,trans,wi,wo,h);
        minimum_power=min(minimum_power,reduce_min(power));maximum_power=max(maximum_power,reduce_max(power));
        failures+=!isfinite_safe(power) || reduce_min(power)<-1e-5f || reduce_max(power)>1.00001f;
      }
    }
    ++facet_cases;
  }
  for(int capacity:{1,2}) {
    ShaderData sd{};sd.N=sd.Ng=make_float3(0,0,1);sd.wi=sd.N;sd.num_closure_left=capacity;
    sd.rand_wavelength=.37f;
    const bool result=bsdf_diffraction_principled_transmission_setup(&sd,make_spectrum(.5f),
        make_spectrum(.25f),sd.N,make_float3(1,0,0),.3f,1.5f,1200,250,.41f,
        rgb_to_spectrum(make_float3(.2f,.6f,1)));
    failures+=result!=(capacity==2);
    if(capacity==1)failures+=sd.num_closure!=0 || sd.num_closure_left!=1;
    else failures+=sd.num_closure!=1 || sd.num_closure_left!=0;
  }
  printf("storage closure=%zu tint=%zu coating=%zu generalized=%zu capacity_cases=2\n",
         sizeof(ShaderClosure),sizeof(DiffractionDielectricTintExtra),
         sizeof(DiffractionDielectricCoatingExtra),sizeof(DiffractionDielectricGeneralizedExtra));
  printf("facet_cases=%d min_power=%g max_power=%g\n",facet_cases,minimum_power,maximum_power);
  printf("reciprocal=%d max_reciprocity=%g\n",reciprocal,maximum_reciprocity);
  printf("events=%d flat_comparisons=%d max_flat_error=%g failures=%d\n",events,flat,maximum_flat,failures);
  return failures!=0;
}
