/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf.h"
#include <cstdio>
using namespace ccl;
template<MicrofacetType type> int check(KernelGlobals kg)
{
  int failures=0;
  ShaderData sd{};sd.N=sd.Ng=make_float3(0,0,1);sd.wi=normalize(make_float3(.4f,.1f,1));
  sd.num_closure_left=2;sd.object=0;
  DiffractionRoughDielectric p{{.37f,.24f,.42f,1,1,1.4f},.2f,.35f};
  if(!bsdf_diffraction_coated_atom_setup<type>(
      &sd,make_spectrum(.7f),sd.N,make_float3(1,0,0),&p,1.7f,.47f,64))return 1;
  failures+=sd.num_closure!=1 || sd.num_closure_left!=0;
  ShaderClosure *sc=&sd.closure[0];
  failures+=!CLOSURE_IS_BSDF_SINGULAR(sc->type) || !bsdf_microfacet_has_delta(sc) ||
      !bsdf_diffraction_dielectric_has_transmission(sc);
  float first=0,last=0;
  for(float z:{.2f,.9f}) {
    sd.wi=make_float3(sqrtf(1-z*z),0,z);
    float expected;
    diffraction_dielectric_coated_straight_mass_quadrature<type>(&p,sd.wi,1.7f,.47f,64,&expected);
    Spectrum value;float3 wo;float pdf,eta;float2 roughness;
    const int label=bsdf_sample(kg,&sd,sc,make_float3(.3f,.4f,.5f),&value,&wo,&pdf,&roughness,&eta);
    float evaluated_pdf;const Spectrum evaluated=bsdf_eval_delta(kg,&sd,sc,wo,&evaluated_pdf);
    failures+=(label&(LABEL_SINGULAR|LABEL_TRANSMIT))!=(LABEL_SINGULAR|LABEL_TRANSMIT) ||
        (label&LABEL_REFLECT)!=0 || len(wo+sd.wi)!=0 || eta!=1 ||
        pdf!=1e6f || evaluated_pdf!=pdf || fabsf(average(value)-expected*1e6f)>2 ||
        len(value-evaluated)>2 || !is_zero(roughness);
    const Spectrum continuous=bsdf_eval(kg,&sd,sc,wo,&evaluated_pdf);
    failures+=!is_zero(continuous)||evaluated_pdf!=0;
    const Spectrum wrong=bsdf_eval_delta(kg,&sd,sc,sd.wi,&evaluated_pdf);
    failures+=!is_zero(wrong)||evaluated_pdf!=0;
    if(z==.2f)first=expected;else last=expected;
  }
  failures+=fabsf(first-last)<.001f;
  bsdf_blur(sc,.8f);p.alpha_x=p.alpha_y=.8f;
  float expected;
  diffraction_dielectric_coated_straight_mass_quadrature<type>(&p,sd.wi,1.7f,.47f,64,&expected);
  failures+=fabsf(bsdf_diffraction_coated_atom_mass(sc,sd.wi)-expected)>2e-6f;
  failures+=fabsf(last-expected)<.001f;
  failures+=!bsdf_diffraction_dielectric_filter(sc,true,false);
  failures+=bsdf_diffraction_dielectric_filter(sc,false,true)||sc->type!=CLOSURE_NONE_ID;
  ShaderData limited{};limited.wi=sd.wi;limited.num_closure_left=1;
  failures+=bsdf_diffraction_coated_atom_setup<type>(
      &limited,one_spectrum(),sd.N,make_float3(1,0,0),&p,1.7f,.47f,64);
  failures+=limited.num_closure!=0||limited.num_closure_left!=1;
  printf("distribution=%s first=%g last=%g blurred=%g failures=%d\n",
         type==GGX?"GGX":"Beckmann",first,last,expected,failures);
  return failures;
}
int main()
{
  Profiler profiler;KernelGlobalsCPU base{};ThreadKernelGlobalsCPU kg(base,nullptr,profiler,0);
  KernelObject object{};kg.objects.data=&object;kg.objects.width=1;
  return check<GGX>(&kg)+check<BECKMANN>(&kg)!=0;
}
