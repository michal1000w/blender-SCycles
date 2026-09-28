/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction.h"
#include <cstdio>
#include <random>
using namespace ccl;
static double fresnel_reference(double ci, double ni, double no) {
 const double ratio=ni/no, ct2=1-ratio*ratio*(1-ci*ci);
 if(ct2<=0) return 1;
 const double ct=std::sqrt(ct2);
 const double rs=(ni*ci-no*ct)/(ni*ci+no*ct);
 const double rp=(no*ci-ni*ct)/(no*ci+ni*ct);
 return .5*(rs*rs+rp*rp);
}
static double power_reference(const DiffractionDielectricFacet &p,int m,bool trans,double ci,double co) {
 double budget;
 if(trans && m==0) {
  const double rs=(p.incident_ior*ci+p.transmitted_ior*co)/(p.incident_ior*ci-p.transmitted_ior*co);
  const double rp=(p.transmitted_ior*ci+p.incident_ior*co)/(p.transmitted_ior*ci-p.incident_ior*co);
  budget=1-.5*(rs*rs+rp*rp);
 }
 else {
  const double fi=fresnel_reference(ci,p.incident_ior,p.transmitted_ior);
  const double fo=trans?fresnel_reference(-co,p.transmitted_ior,p.incident_ior):
                        fresnel_reference(co,p.incident_ior,p.transmitted_ior);
  budget=trans?std::min(1-fi,1-fo):std::min(fi,fo);
 }
 const double phase=trans?p.transmission_phase:2*M_PI*p.height_over_wavelength*(ci+co);
 const double modulation=4*std::pow(std::sin(.5*phase),2);
 const double coefficient=m==0?1-modulation*p.duty*(1-p.duty):
   modulation*std::pow(std::sin(M_PI*m*p.duty)/(M_PI*m),2);
 return budget*coefficient;
}
int main() {
 std::mt19937 rng(823174);
 std::uniform_real_distribution<float> u(0,1);
 const float3 h=make_float3(0,0,1),axis=make_float3(1,0,0);
 int failures=0,reciprocal=0,flat=0;
 double max_reference=0;
 float max_reverse=0,min_residual=1,max_flat=0;
 for(int i=0;i<20000;++i) {
  const float z=.001f+.999f*u(rng),phi=2*M_PI_F*u(rng),r=sqrtf(1-z*z);
  const float3 wi=make_float3(r*cosf(phi),r*sinf(phi),z);
  DiffractionDielectricFacet p{.1f+u(rng),2*u(rng),u(rng),.5f+1.5f*u(rng),1.0f,10*u(rng)};
  const float eta=p.incident_ior/p.transmitted_ior;
  const float residual=diffraction_dielectric_facet_residual(&p,wi,h,axis);
  failures+=residual < -2e-6f || residual > 1 || !isfinite_safe(residual);
  min_residual=min(min_residual,residual);
  for(int side=0;side<2;++side) {
   const bool trans=side!=0;
   const int bound=diffraction_dielectric_facet_order_bound(&p,trans);
   for(int m=-bound;m<=bound;++m) {
    if(!trans && m==0) continue;
    float3 wo;
    bool valid=trans?diffraction_facet_transmit(wi,h,axis,eta,m*p.wavelength_over_pitch*eta,&wo):
                     diffraction_facet_reflect(wi,h,axis,m*p.wavelength_over_pitch,&wo);
    if(!valid) continue;
    const float power=diffraction_dielectric_facet_power(&p,m,trans,wi.z,wo.z);
    const double reference=power_reference(p,m,trans,wi.z,wo.z);
    max_reference=std::max(max_reference,std::abs(power-reference));
    failures+=std::abs(power-reference)>2e-5;
    DiffractionDielectricFacet reverse=p;
    if(trans) {reverse.incident_ior=p.transmitted_ior;reverse.transmitted_ior=p.incident_ior;reverse.wavelength_over_pitch*=eta;reverse.height_over_wavelength/=eta;}
    const float back=diffraction_dielectric_facet_power(&reverse,m,trans,trans?-wo.z:wo.z,trans?-wi.z:wi.z);
    max_reverse=max(max_reverse,fabsf(power-back));
    failures+=fabsf(power-back)>2e-6f || power<0 || power>1;
    ++reciprocal;
   }
  }
  p.height_over_wavelength=p.transmission_phase=0;
  const float result=diffraction_dielectric_facet_residual(&p,wi,h,axis);
  const float expected=fresnel_dielectric(wi.z,1/eta,nullptr);
  max_flat=max(max_flat,fabsf(result-expected));
  /* Very near critical angles reverse Fresnel is poorly conditioned in float. */
  failures+=fabsf(result-expected)>2e-4f;
  ++flat;
 }
 printf("max_double_reference_error=%.9g\n",max_reference);
 printf("reciprocal_pairs=%d max_reverse_error=%g min_residual=%g flat_cases=%d max_flat_error=%g failures=%d\n",reciprocal,max_reverse,min_residual,flat,max_flat,failures);
 return failures!=0;
}
