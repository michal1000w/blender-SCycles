/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf.h"
#include <cstdio>
#include <array>
#include <cmath>
#include <complex>
using namespace ccl;
static_assert(sizeof(DiffractionDielectricCoatingExtra)<=sizeof(ShaderClosure));
static_assert(sizeof(DiffractionDielectricGeneralizedExtra)<=sizeof(ShaderClosure));
/* Independent complex interface-amplitude Airy series in float64, followed
 * by the declared passive Principled artistic film-tint rule. */
static std::array<double,3> film_power_reference(
    const DiffractionDielectricGeneralizedExtra &e,double ci,double co,bool trans)
{
  const double ni=e.base.param.facet.incident_ior,no=e.base.param.facet.transmitted_ior;
  const double nf=e.film_ior,d=e.film_thickness_over_wavelength;
  const double kt2=ni*ni*(1-ci*ci),ct2=1-kt2/(no*no);
  double F=1;
  if(trans || ct2>0) {
    const double ct=trans?-co:std::sqrt(ct2);
    using C=std::complex<double>;
    const C qi=ni*ci,qo=no*ct,qf=std::sqrt(C(nf*nf-kt2,0));
    const C phase=std::exp(C(0,4*3.14159265358979323846*d)*qf);
    F=0;
    for(int pol=0;pol<2;++pol) {
      const C yi=pol?qi/(ni*ni):qi,yf=pol?qf/(nf*nf):qf,yo=pol?qo/(no*no):qo;
      const C r1=(yi-yf)/(yi+yf),r2=(yf-yo)/(yf+yo);
      F+=.5*std::norm((r1+r2*phase)/(1.0+r1*r2*phase));
    }
  }
  const double scale=std::clamp((1-F)/(1-e.generalized_reference_f0),0.0,1.0);
  const double f0[3]={e.generalized_f0.x,e.generalized_f0.y,e.generalized_f0.z};
  std::array<double,3> result;
  for(int c=0;c<3;++c) {
    const double r=std::clamp(F*(1+scale*(f0[c]/e.generalized_reference_f0-1)),0.0,1.0);
    result[c]=trans?1-r:r;
  }
  return result;
}
/* Independent float64 evaluation at the supplied float directions. This uses
 * the closed-form isotropic GGX distribution and independent film amplitude series, not the
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
  const auto powers=film_power_reference(e,ci,co,trans);
  const double common=D/I[2]*(trans?no*no*std::abs(ci*co)/length2:.25)/(1+lambda(I)+lambda(O));
  const Spectrum tint=trans?e.base.transmission:e.base.reflection;
  const double f0[3]={e.generalized_f0.x,e.generalized_f0.y,e.generalized_f0.z};
  const double color[3]={tint.x,tint.y,tint.z};std::array<double,3> result;
  for(int c=0;c<3;++c) {result[c]=powers[c]*color[c]*common;}
  return result;
}

int main()
{
  Profiler profiler;KernelGlobalsCPU base{};ThreadKernelGlobalsCPU kg(base,nullptr,profiler,0);
  KernelObject object{};kg.objects.data=&object;kg.objects.width=1;
  int failures=0,events=0,flat=0,reciprocal=0,double_cases=0;float maximum_flat=0,maximum_reciprocity=0;double maximum_double_error=0;
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
                                    roughness,1.5f,1200,depth,.41f,specular_tint,inv_abbe,1.35f,420))return 2;
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
        ++flat;
        if(label&LABEL_SINGULAR) {
          const auto *b=(const DiffractionDielectricBsdf *)sc;
          const auto &e=*(const DiffractionDielectricGeneralizedExtra *)b->extra;
          const bool trans=wo.z<0;
          const auto powers=film_power_reference(e,sd.wi.z,wo.z,trans);
          const Spectrum actual=value*1e-6f;
          const Spectrum color=trans?tint:one_spectrum();
          const double error=std::max({std::abs(actual.x-powers[0]*color.x),std::abs(actual.y-powers[1]*color.y),std::abs(actual.z-powers[2]*color.z)});
          maximum_flat=max(maximum_flat,float(error));failures+=error>3e-4;
        }
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
    e.film_ior=1.7f;e.film_thickness_over_wavelength=420.f/550;
    e.generalized_f0=saturate(e.generalized_reference_f0*tint_scale*rgb_to_spectrum(make_float3(1,.7f,.3f)));
    const float3 wi=make_float3(sqrtf(1-cosine*cosine),0,cosine),h=make_float3(0,0,1);
    const auto *p=&e.base.param.facet;
    Spectrum total_power=zero_spectrum();
    for(int side=0;side<2;++side) {
      const bool trans=side!=0;const float eta=p->incident_ior/p->transmitted_ior;
      const int bound=diffraction_dielectric_facet_active_order_bound(p,trans);
      for(int m=-bound;m<=bound;++m) {
        float3 wo;const float delta=m*p->wavelength_over_pitch;
        const bool valid=trans?diffraction_facet_transmit(wi,h,make_float3(1,0,0),eta,delta*eta,&wo):
                              diffraction_facet_reflect(wi,h,make_float3(1,0,0),delta,&wo);
        if(!valid)continue;
        const Spectrum power=diffraction_dielectric_generalized_power(&e,m,trans,wi,wo,h);
        total_power+=power;
        minimum_power=min(minimum_power,reduce_min(power));maximum_power=max(maximum_power,reduce_max(power));
        failures+=!isfinite_safe(power) || reduce_min(power)<-1e-5f || reduce_max(power)>1.00001f;
      }
    }
    failures+=reduce_max(fabs(total_power-one_spectrum()))>2e-5f;
    ++facet_cases;
  }
  for(int capacity:{1,2}) {
    ShaderData sd{};sd.N=sd.Ng=make_float3(0,0,1);sd.wi=sd.N;sd.num_closure_left=capacity;
    sd.rand_wavelength=.37f;
    const bool result=bsdf_diffraction_principled_transmission_setup(&sd,make_spectrum(.5f),
        make_spectrum(.25f),sd.N,make_float3(1,0,0),.3f,1.5f,1200,250,.41f,
        rgb_to_spectrum(make_float3(.2f,.6f,1)),0,1.35f,420);
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
      matched.N,make_float3(1,0,0),.3f,1.5f,1200,250,.41f,st,inv_match,1.35f,420))return 7;
  failures+=matched.num_closure!=2 || matched.closure[0].type!=CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID;
  failures+=reduce_max(fabs(matched.closure[0].weight-tt))>1e-6f;
  int matched_reflections=0;
  for(int i=0;i<2000;++i) for(int c=0;c<matched.num_closure;++c) {
    Spectrum v;float3 wo;float pdf,eta;float2 rough;
    const int label=bsdf_sample(&kg,&matched,&matched.closure[c],make_float3((i+.5f)/2000,
        fmodf(i*.6180339887f+.3f,1.f),fmodf(i*.4142135623f+.2f,1.f)),&v,&wo,&pdf,&rough,&eta);
    if(label==LABEL_NONE)continue;
    ++matched_reflections;float qp;const Spectrum q=(label&LABEL_SINGULAR)?bsdf_eval_delta(&kg,&matched,&matched.closure[c],wo,&qp):bsdf_eval(&kg,&matched,&matched.closure[c],wo,&qp);
    failures+=!(label&(c==0?LABEL_TRANSMIT:LABEL_REFLECT)) || !(pdf>0) || !isfinite_safe(v) ||
      fabsf(pdf-qp)>3e-5f*max(1.f,pdf) || reduce_max(fabs(v-q))>3e-5f*max(1.f,reduce_max(v));
  }
  failures+=matched_reflections==0;
  printf("double_reference_cases=%d maximum_double_error=%g\n",double_cases,maximum_double_error);
  printf("matched_index_inv_abbe=%g dispatch_events=%d\n",inv_match,matched_reflections);
  printf("storage closure=%zu tint=%zu coating=%zu generalized=%zu capacity_cases=2\n",
         sizeof(ShaderClosure),sizeof(DiffractionDielectricTintExtra),
         sizeof(DiffractionDielectricCoatingExtra),sizeof(DiffractionDielectricGeneralizedExtra));
  printf("facet_cases=%d min_power=%g max_power=%g\n",facet_cases,minimum_power,maximum_power);
  printf("reciprocal=%d max_reciprocity=%g\n",reciprocal,maximum_reciprocity);
  printf("events=%d flat_comparisons=%d max_flat_error=%g failures=%d\n",events,flat,maximum_flat,failures);
  return failures!=0;
}
