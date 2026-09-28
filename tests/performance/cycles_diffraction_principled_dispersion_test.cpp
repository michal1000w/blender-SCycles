/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf.h"
#include <cstdio>
#include <array>
#include <cmath>
using namespace ccl;
static_assert(sizeof(DiffractionDielectricCoatingExtra)<=sizeof(ShaderClosure));
static_assert(sizeof(DiffractionDielectricGeneralizedExtra)<=sizeof(ShaderClosure));
/* Independent float64 evaluation at the supplied float directions. This uses
 * the closed-form isotropic GGX distribution and planar Fresnel pair, not the
 * grating inverse roots/Jacobian or the native float microfacet evaluator. */
static std::array<double,3> flat_double_reference(
    const DiffractionDielectricGeneralizedExtra &e,const float3 wi,const float3 wo)
{
  const auto &p=e.base.param;const bool trans=wo.z<0;
  const double ni=p.facet.incident_ior,no=p.facet.transmitted_ior;
  std::array<double,3> I={wi.x,wi.y,wi.z},O={wo.x,wo.y,wo.z},H;
  double length2=0;
  for(int c=0;c<3;++c) {H[c]=trans?ni*I[c]+no*O[c]:I[c]+O[c];length2+=H[c]*H[c];}
  const double inv=(H[2]<0?-1:1)/std::sqrt(length2);
  double ci=0,co=0;
  for(int c=0;c<3;++c) {H[c]*=inv;ci+=I[c]*H[c];co+=O[c]*H[c];}
  const double a2=double(p.alpha_x)*p.alpha_y;
  const double denom=H[0]*H[0]+H[1]*H[1]+a2*H[2]*H[2];
  const double D=a2/(3.14159265358979323846*denom*denom);
  auto lambda=[&](const std::array<double,3> &w) {
    return (std::sqrt(1+a2*(w[0]*w[0]+w[1]*w[1])/(w[2]*w[2]))-1)*.5;
  };
  double F=1;
  const double sint2=(ni/no)*(ni/no)*std::max(0.0,1-ci*ci);
  if(trans || sint2<1) {
    const double ct=trans?-co:std::sqrt(1-sint2);
    const double rs=(ni*ci-no*ct)/(ni*ci+no*ct),rp=(no*ci-ni*ct)/(no*ci+ni*ct);
    F=.5*(rs*rs+rp*rp);
  }
  const double t=std::min(1.0,std::max(0.0,(F-e.generalized_reference_f0)/(1-e.generalized_reference_f0)));
  const double common=D/I[2]*(trans?no*no*std::abs(ci*co)/length2:.25)/(1+lambda(I)+lambda(O));
  const Spectrum tint=trans?e.base.transmission:e.base.reflection;
  const double f0[3]={e.generalized_f0.x,e.generalized_f0.y,e.generalized_f0.z};
  const double color[3]={tint.x,tint.y,tint.z};std::array<double,3> result;
  for(int c=0;c<3;++c) {const double f=f0[c]+(1-f0[c])*t;result[c]=(trans?1-f:f)*color[c]*common;}
  return result;
}

int main()
{
  Profiler profiler;KernelGlobalsCPU base{};ThreadKernelGlobalsCPU kg(base,nullptr,profiler,0);
  KernelObject object{};kg.objects.data=&object;kg.objects.width=1;
  int failures=0,events=0,flat=0,reciprocal=0,native_disagreements=0,double_cases=0;float maximum_flat=0,maximum_reciprocity=0;double maximum_double_error=0;
  for(bool back:{false,true}) for(bool colored:{false,true})
  for(float roughness:{0.f,.2f,.6f}) for(float depth:{0.f,250.f})
  for(float wavelength_u:{.1f,.37f,.8f}) for(float inv_abbe:{.05f,.2f})
  for(bool white_specular:{false,true}) {
    ShaderData sd{};sd.N=sd.Ng=make_float3(0,0,1);
    sd.wi=normalize(make_float3(.4f,.2f,1));sd.object=0;sd.rand_wavelength=wavelength_u;sd.num_closure_left=8;sd.shader_flag|=SD_REQUIRES_WAVELENGTH;
    if(back)sd.runtime_flag|=SR_BACKFACING;
    const Spectrum tint=colored?rgb_to_spectrum(make_float3(.3f,.7f,.9f)):one_spectrum();
    const Spectrum specular_tint=white_specular?one_spectrum():.5f*rgb_to_spectrum(make_float3(1,.7f,.3f));
    if(!bsdf_diffraction_principled_transmission_setup(&sd,one_spectrum(),tint,sd.N,make_float3(1,0,0),
                                    roughness,1.5f,1200,depth,.41f,specular_tint,inv_abbe))return 2;
    MicrofacetBsdf native{};native.N=sd.N;native.alpha_x=native.alpha_y=sqr(roughness);
    native.ior=bsdf_glass_ior(&sd,back?1/1.5f:1.5f,inv_abbe);
    if(fabsf(native.ior-(back?1/1.5f:1.5f))<1e-6f)return 5;
    bsdf_microfacet_ggx_glass_setup(&native);
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
          reverse.generalized_reference_f0=F0_from_ior(reverse.base.param.facet.transmitted_ior/reverse.base.param.facet.incident_ior);
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
        if(error>3e-4f) {
          const auto *bb=(const DiffractionDielectricBsdf *)sc;
          const auto *ee=(const DiffractionDielectricGeneralizedExtra *)bb->extra;
          const auto *pp=&ee->base.param;
          printf("native_flat_disagreement wi %.17g %.17g %.17g wo %.17g %.17g %.17g ni %.17g no %.17g alpha %.17g ref_f0 %.17g f0 %.17g %.17g %.17g tint %.17g %.17g %.17g value %.17g %.17g %.17g native %.17g %.17g %.17g\n",
            double(sd.wi.x),double(sd.wi.y),double(sd.wi.z),double(wo.x),double(wo.y),double(wo.z),
            double(pp->facet.incident_ior),double(pp->facet.transmitted_ior),double(pp->alpha_x),
            double(ee->generalized_reference_f0),double(ee->generalized_f0.x),double(ee->generalized_f0.y),double(ee->generalized_f0.z),
            double(tint.x),double(tint.y),double(tint.z),double(value.x),double(value.y),double(value.z),
            double(reference.x),double(reference.y),double(reference.z));
        }
        maximum_flat=max(maximum_flat,error);native_disagreements+=error>3e-4f;++flat;
        if(label&LABEL_SINGULAR)failures+=error>3e-4f;
        else {
          const auto *b=(const DiffractionDielectricBsdf *)sc;
          const auto ref=flat_double_reference(*(const DiffractionDielectricGeneralizedExtra *)b->extra,sd.wi,wo);
          const Spectrum actual=sc->weight*value;
          const double delta=std::max({std::abs(actual.x-ref[0]),std::abs(actual.y-ref[1]),std::abs(actual.z-ref[2])});
          const double relative=delta/std::max({1.0,ref[0],ref[1],ref[2]});
          maximum_double_error=std::max(maximum_double_error,relative);failures+=relative>3e-4;++double_cases;
        }
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
  /* Find the nearest representable dispersion coefficient for the exact
   * index-matched wavelength. The analytic center makes n(lambda)=1. */
  ShaderData matched{};matched.N=matched.Ng=make_float3(0,0,1);
  matched.wi=normalize(make_float3(.4f,.2f,1));matched.object=0;
  matched.rand_wavelength=.9f;matched.shader_flag|=SD_REQUIRES_WAVELENGTH;
  matched.num_closure_left=8;
  const float wavelength=sample_wavelength(matched.rand_wavelength);
  const float fac=1/(1/sqr(.4861f)-1/sqr(.6563f));
  const float center=-1/(fac*(1/sqr(wavelength)-1/sqr(.5876f)));
  float lower=center,upper=center,inv_match=0;
  for(int i=0;i<64 && inv_match==0;++i) {
    if(bsdf_glass_ior(&matched,1.5f,lower)==1)inv_match=lower;
    else if(bsdf_glass_ior(&matched,1.5f,upper)==1)inv_match=upper;
    lower=nextafterf(lower,0);upper=nextafterf(upper,INFINITY);
  }
  if(inv_match==0)return 6;
  const Spectrum st=rgb_to_spectrum(make_float3(.2f,.5f,1));
  const Spectrum tt=rgb_to_spectrum(make_float3(.3f,.7f,.9f));
  if(!bsdf_diffraction_principled_transmission_setup(&matched,one_spectrum(),tt,
      matched.N,make_float3(1,0,0),.3f,1.5f,1200,250,.41f,st,inv_match))return 7;
  failures+=matched.num_closure!=2 || matched.closure[0].type!=CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID;
  failures+=reduce_max(fabs(matched.closure[0].weight-tt*(one_spectrum()-F0_from_ior(1.5f)*st)))>1e-6f;
  int matched_reflections=0;
  for(int i=0;i<2000;++i) {
    Spectrum v;float3 wo;float pdf,eta;float2 rough;
    const int label=bsdf_sample(&kg,&matched,&matched.closure[1],make_float3((i+.5f)/2000,
        fmodf(i*.6180339887f+.3f,1.f),fmodf(i*.4142135623f+.2f,1.f)),&v,&wo,&pdf,&rough,&eta);
    if(label==LABEL_NONE)continue;
    ++matched_reflections;float qp;const Spectrum q=bsdf_eval(&kg,&matched,&matched.closure[1],wo,&qp);
    failures+=!(label&LABEL_REFLECT) || !(pdf>0) || !isfinite_safe(v) ||
      fabsf(pdf-qp)>3e-5f*max(1.f,pdf) || reduce_max(fabs(v-q))>3e-5f*max(1.f,reduce_max(v));
  }
  failures+=matched_reflections==0;
  printf("double_reference_cases=%d maximum_double_error=%g native_float_disagreements=%d\n",double_cases,maximum_double_error,native_disagreements);
  printf("matched_index_inv_abbe=%g reflected_events=%d\n",inv_match,matched_reflections);
  printf("storage closure=%zu tint=%zu coating=%zu generalized=%zu capacity_cases=2\n",
         sizeof(ShaderClosure),sizeof(DiffractionDielectricTintExtra),
         sizeof(DiffractionDielectricCoatingExtra),sizeof(DiffractionDielectricGeneralizedExtra));
  printf("facet_cases=%d min_power=%g max_power=%g\n",facet_cases,minimum_power,maximum_power);
  printf("reciprocal=%d max_reciprocity=%g\n",reciprocal,maximum_reciprocity);
  printf("events=%d flat_comparisons=%d max_flat_error=%g failures=%d\n",events,flat,maximum_flat,failures);
  return failures!=0;
}
