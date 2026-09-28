/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include <cstdio>
#include <random>
using namespace ccl;
#ifdef DIFFRACTION_TEST_BECKMANN
constexpr MicrofacetType test_distribution=BECKMANN;
#else
constexpr MicrofacetType test_distribution=GGX;
#endif
int main() {
 std::mt19937 rng(91842); std::uniform_real_distribution<float> u(0,1);
 auto direction=[&]() {float z=.02f+.98f*u(rng),phi=2*M_PI_F*u(rng),r=sqrtf(1-z*z);return make_float3(r*cosf(phi),r*sinf(phi),z);};
 int failures=0,reciprocal=0,sampled=0,flat=0; float max_reciprocal=0,max_flat=0;
 for(int i=0;i<20000;++i) {
  DiffractionRoughDielectric p{{.2f+.8f*u(rng),u(rng),u(rng),1,1.5f,6*u(rng)},.1f+.6f*u(rng),.1f+.6f*u(rng)};
  const float3 wi=direction(); float3 wo=direction(); if(i%2) wo.z=-wo.z;
  float pdf,reverse_pdf;
  const float forward=diffraction_dielectric_eval<test_distribution>(&p,wi,wo,&pdf)/fabsf(wo.z);
  DiffractionRoughDielectric reverse=p;
  float3 ri=wo,ro=wi;
  float factor=1;
  if(wo.z<0) {
   const float eta=p.facet.incident_ior/p.facet.transmitted_ior;
   reverse.facet.incident_ior=p.facet.transmitted_ior;reverse.facet.transmitted_ior=p.facet.incident_ior;
   reverse.facet.wavelength_over_pitch*=eta;reverse.facet.height_over_wavelength/=eta;
   ri=make_float3(wo.x,-wo.y,-wo.z);ro=make_float3(wi.x,-wi.y,-wi.z);factor=eta*eta;
  }
  const float back=diffraction_dielectric_eval<test_distribution>(&reverse,ri,ro,&reverse_pdf)/fabsf(ro.z);
  const float error=fabsf(factor*forward-back)/max(1.f,max(factor*forward,back));
  max_reciprocal=max(max_reciprocal,error);
  if(!isfinite_safe(error)||error>3e-4f) {if(failures<5)printf("reciprocity case=%d forward=%g reverse=%g error=%g\n",i,factor*forward,back,error);++failures;}
  ++reciprocal;
  float value;bool singular;
  if(diffraction_dielectric_sample<test_distribution>(&p,wi,make_float3(u(rng),u(rng),u(rng)),&wo,&value,&pdf,&singular)) {
   const bool bad=!isfinite_safe(value)||!isfinite_safe(pdf)||value<0||pdf<=0||value>pdf*1.00001f;
   if(bad&&failures<5)printf("sample case=%d value=%g pdf=%g lambda_o=%g\n",i,value,pdf,bsdf_aniso_lambda<test_distribution>(p.alpha_x,p.alpha_y,wo));
   failures+=bad;
   ++sampled;
  }
  /* No diffraction: compare with the conventional analytic GGX glass formula. */
  p.facet.height_over_wavelength=p.facet.transmission_phase=0;
  const float3 h=make_float3(0,0,1);
  const float eta=p.facet.incident_ior/p.facet.transmitted_ior;
  if(!diffraction_facet_transmit(wi,h,make_float3(1,0,0),eta,0,&wo)) continue;
  const float value_flat=diffraction_dielectric_eval<test_distribution>(&p,wi,wo,&pdf);
  const float F=fresnel_dielectric(wi.z,1/eta,nullptr);
  const float a=eta*wi.z+wo.z;
  const float common=bsdf_aniso_D<test_distribution>(p.alpha_x,p.alpha_y,h)*(-wo.z)/(a*a);
  const float expected=(1-F)*common/(1+bsdf_aniso_lambda<test_distribution>(p.alpha_x,p.alpha_y,wi)+bsdf_aniso_lambda<test_distribution>(p.alpha_x,p.alpha_y,wo));
  const float fe=fabsf(value_flat-expected)/max(1.f,expected);
  max_flat=max(max_flat,fe);failures+=fe>2e-4f;++flat;
 }
 /* Exactly matched indices and a flat profile are a unit straight-through
  * delta event even with nonzero roughness. */
 DiffractionRoughDielectric p{{.5f,0,.5f,1,1,0},.2f,.55f};
 for(int i=0;i<1000;++i) {
  const float3 wi=direction();float3 wo;float value,pdf;bool singular;
  failures+=!diffraction_dielectric_sample<test_distribution>(&p,wi,make_float3(u(rng),u(rng),u(rng)),&wo,&value,&pdf,&singular)||!singular||len(wi+wo)>1e-6f||fabsf(value-1)>1e-6f||fabsf(pdf-1)>1e-6f;
 }
 printf("reciprocal=%d max_scaled_reciprocal=%g sampled=%d flat=%d max_scaled_flat=%g failures=%d\n",reciprocal,max_reciprocal,sampled,flat,max_flat,failures);
 return failures!=0;
}
