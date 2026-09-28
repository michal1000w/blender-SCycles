/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf.h"
#include <complex>
#include <cstdio>
using namespace ccl;

/* Independent complex Fourier integral of a unit-amplitude binary phase mask. */
static double coefficient(int m,double phase,double duty)
{
  using C=std::complex<double>;
  const C c=(std::exp(C(0,phase))-1.0)*(1.0-std::exp(C(0,-2*M_PI*m*duty)))/C(0,2*M_PI*m);
  return std::norm(c);
}
static double fresnel(double ci,double ni,double no)
{
  const double ct2=1-(ni/no)*(ni/no)*(1-ci*ci);
  if(ct2<=0)return 1;
  const double ct=std::sqrt(ct2);
  const double s=(ni*ci-no*ct)/(ni*ci+no*ct),p=(no*ci-ni*ct)/(no*ci+ni*ct);
  return .5*(s*s+p*p);
}
int main()
{
  int failures=0,cases=0,events=0,reciprocal=0;double maximum_tail=0,maximum_pure_tail=0;float maximum_reciprocal=0;
  const int cap=DIFFRACTION_FAST_MAX_ORDER;
  const double bound=16/(M_PI*M_PI*cap),pure_bound=8/(M_PI*M_PI*cap);
  for(double wl:{380.,550.,780.}) for(double pitch:{100000.,1000000.})
  for(double ni:{1.,1.5}) for(double depth:{250.,2000.})
  for(double duty:{.1,.5,.9}) for(double ci:{.2,1.}) {
    const double no=ni==1?1.5:1,ix=std::sqrt(1-ci*ci);
    double tail=0,pure_tail=0;
    for(int side=0;side<2;++side) {
      const bool trans=side!=0;const double outgoing=trans?no:ni;
      const int full=int(std::ceil((ni+outgoing)*pitch/wl));
      for(int m=-full;m<=full;++m) {
        if(std::abs(m)<=cap)continue;
        const double ox=(m*wl/pitch-ni*ix)/outgoing;
        if(ox*ox>=1)continue;
        const double co=std::sqrt(1-ox*ox);
        const double phase=2*M_PI*depth/wl*(trans?ni-no:ni*(ci+co));
        const double power=coefficient(m,phase,duty);
        const double Fi=fresnel(ci,ni,no);
        const double Fo=trans?fresnel(co,no,ni):fresnel(co,ni,no);
        tail+=power*(trans?std::min(1-Fi,1-Fo):std::min(Fi,Fo));
        if(trans)pure_tail+=power;
      }
    }
    maximum_tail=std::max(maximum_tail,tail);maximum_pure_tail=std::max(maximum_pure_tail,pure_tail);
    failures+=tail>bound || pure_tail>pure_bound;
    DiffractionRoughDielectric p;
    failures+=!diffraction_dielectric_parameters(wl,pitch,depth,duty,ni,no,.04f,.04f,&p);
    failures+=diffraction_dielectric_facet_order_bound(&p.facet,false)>cap ||
              diffraction_dielectric_facet_order_bound(&p.facet,true)>cap;
    ++cases;
  }
  Profiler profiler;KernelGlobalsCPU base{};ThreadKernelGlobalsCPU kg(base,nullptr,profiler,0);
  KernelObject object{};kg.objects.data=&object;kg.objects.width=1;
  for(float pitch:{100000.f,1000000.f}) for(float roughness:{0.f,.2f})
  for(bool back:{false,true}) for(int material=0;material<4;++material) {
    ShaderData sd{};sd.N=sd.Ng=make_float3(0,0,1);sd.wi=normalize(make_float3(.3f,.1f,1));
    sd.rand_wavelength=.37f;sd.num_closure_left=8;sd.object=0;
    if(back)sd.runtime_flag|=SR_BACKFACING;
    auto setup=[&](ShaderData &sd) {
    bool ok=false;
    if(material==0)ok=bsdf_diffraction_glass_setup(&sd,one_spectrum(),one_spectrum(),sd.N,
        make_float3(1,0,0),roughness,1.5f,pitch,320,.41f,1,0,false);
    else if(material==1)ok=bsdf_diffraction_refraction_setup(&sd,one_spectrum(),sd.N,
        make_float3(1,0,0),roughness,1.5f,pitch,320,.41f,false);
    else if(material==2)ok=bsdf_diffraction_principled_transmission_setup(&sd,one_spectrum(),one_spectrum(),sd.N,
        make_float3(1,0,0),roughness,1.5f,pitch,320,.41f,make_float3(.3f,.7f,1),0,1.35f,420);
    else ok=bsdf_diffraction_glossy_setup(&kg,&sd,one_spectrum(),sd.N,make_float3(1,0,0),
        roughness*roughness,roughness*roughness,pitch,320,.41f,1.5f,false);
      return ok;
    };
    const bool ok=setup(sd);
    if(!ok || sd.num_closure==0)return 2;
    for(int i=0;i<64;++i) {
      Spectrum v;float3 wo;float pdf,eta;float2 rough;
      const ShaderClosure *sc=&sd.closure[0];
      const int label=bsdf_sample(&kg,&sd,sc,make_float3((i+.5f)/64,
          fmodf(i*.6180339887f+.3f,1.f),fmodf(i*.4142135623f+.2f,1.f)),&v,&wo,&pdf,&rough,&eta);
      if(label==LABEL_NONE)continue;
      float qp;const Spectrum q=(label&LABEL_SINGULAR)?bsdf_eval_delta(&kg,&sd,sc,wo,&qp):bsdf_eval(&kg,&sd,sc,wo,&qp);
      failures+=!(pdf>0) || !isfinite_safe(v) || reduce_min(v)<0 ||
          fabsf(pdf-qp)>3e-5f*max(1.f,pdf) || reduce_max(fabs(v-q))>3e-5f*max(1.f,reduce_max(v));
      if(!(label&LABEL_SINGULAR)) {
        const bool trans=label&LABEL_TRANSMIT;
        ShaderData reverse{};reverse.N=reverse.Ng=sd.N;reverse.object=0;
        reverse.num_closure_left=8;reverse.rand_wavelength=sd.rand_wavelength;
        reverse.wi=trans?make_float3(wo.x,-wo.y,-wo.z):wo;
        const float3 reversed_out=trans?make_float3(sd.wi.x,-sd.wi.y,-sd.wi.z):sd.wi;
        if(back!=trans)reverse.runtime_flag|=SR_BACKFACING;
        if(!setup(reverse))return 3;
        float reverse_pdf;const Spectrum rv=bsdf_eval(&kg,&reverse,&reverse.closure[0],reversed_out,&reverse_pdf);
        const Spectrum f=v/(fabsf(wo.z)*eta*eta),g=rv/fabsf(reversed_out.z);
        const float error=reduce_max(fabs(f-g))/max(1.f,max(reduce_max(f),reduce_max(g)));
        maximum_reciprocal=max(maximum_reciprocal,error);failures+=error>3e-4f;++reciprocal;
      }
      ++events;
    }
  }
  printf("reciprocal_cases=%d maximum_reciprocity_error=%g\n",reciprocal,maximum_reciprocal);
  printf("cap=%d facet_cases=%d maximum_omitted_power=%.12g bound=%.12g pure_omitted=%.12g pure_bound=%.12g events=%d failures=%d\n",
         cap,cases,maximum_tail,bound,maximum_pure_tail,pure_bound,events,failures);
  return failures!=0 || events<1000;
}
