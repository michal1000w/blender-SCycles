/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include <cstdio>
#include <cstring>
#include <random>
using namespace ccl;
#ifdef DIFFRACTION_TEST_BECKMANN
constexpr MicrofacetType test_distribution=BECKMANN;
#else
constexpr MicrofacetType test_distribution=GGX;
#endif
int main() {
 std::mt19937 rng(19162);std::uniform_real_distribution<float> u(0,1);
 int failures=0,atoms=0,continuous=0;float max_atom_error=0,max_eval_error=0;
 for(float alpha:{0.f,.3f}) for(float no:{1.f,1.5f}) {
  DiffractionDielectricBsdf b{};
  b.type=CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID;
  b.N=normalize(make_float3(.2f,.3f,1.f));b.T=normalize(cross(make_float3(0,1,0),b.N));
  b.param={{.37f,.24f,.42f,1,no,1.4f},alpha,alpha};
  /* Exercise the same value-copy property needed when saving shader closures. */
  ShaderClosure storage{};std::memcpy(&storage,&b,sizeof(b));
  float3 X,Y;bsdf_diffraction_dielectric_frame(&b,&X,&Y);
  for(int i=0;i<10000;++i) {
   const float z=.05f+.95f*u(rng),phi=2*M_PI_F*u(rng),r=sqrtf(1-z*z);
   const float3 wi=(r*cosf(phi))*X+(r*sinf(phi))*Y+z*b.N;
   float3 wo;Spectrum eval;float pdf,eta;float2 roughness;
   const int label=bsdf_diffraction_dielectric_sample<test_distribution>(&storage,b.N,wi,make_float3(u(rng),u(rng),u(rng)),&eval,&wo,&pdf,&roughness,&eta);
   if(label==LABEL_NONE) continue;
   failures+=pdf<=0||!isfinite_safe(pdf)||!isfinite_safe(eval)||average(eval)<0;
   ShaderClosure disabled=storage;
   auto *masked=(DiffractionDielectricBsdf *)&disabled;
   masked->disabled_lobes=(label&LABEL_TRANSMIT)?LABEL_TRANSMIT:LABEL_REFLECT;
   float masked_pdf=1;
   const Spectrum masked_eval=bsdf_diffraction_dielectric_eval<test_distribution>(&disabled,wi,wo,&masked_pdf);
   failures+=masked_pdf!=0 || average(masked_eval)!=0 ||
       bsdf_diffraction_dielectric_delta_mass(&disabled,wi,wo)!=0;
   masked->disabled_lobes=(label&LABEL_TRANSMIT)?LABEL_REFLECT:LABEL_TRANSMIT;
   if(!(label&LABEL_SINGULAR)) {
     const Spectrum retained=bsdf_diffraction_dielectric_eval<test_distribution>(&disabled,wi,wo,&masked_pdf);
     failures+=masked_pdf!=pdf || average(retained)!=average(eval);
   }

   failures+=((label&LABEL_TRANSMIT)!=0)!=(dot(b.N,wo)<0);
   failures+=fabsf(eta-((label&LABEL_TRANSMIT)?no:1))>1e-6f;
   if(label&LABEL_SINGULAR) {
    const float mass=bsdf_diffraction_dielectric_delta_mass(&storage,wi,wo);
    const float error=fabsf(mass-pdf*1e-6f);
    max_atom_error=max(max_atom_error,error);failures+=error>2e-5f;++atoms;
   }
   else {
    float queried_pdf;const Spectrum queried=bsdf_diffraction_dielectric_eval<test_distribution>(&storage,wi,wo,&queried_pdf);
    const float error=fabsf(queried_pdf-pdf)/max(1.f,pdf);
    max_eval_error=max(max_eval_error,error);failures+=error>3e-4f||fabsf(average(queried-eval))/max(1.f,average(eval))>3e-4f;
    ++continuous;
   }
  }
 }
 /* Setup must reserve the complete mixed-measure pair, never truncate it. */
 ShaderData sd{};sd.num_closure_left=2;
 const DiffractionRoughDielectric split{{.37f,.24f,.42f,1,1,1.4f},.3f,.3f};
 const float atom=diffraction_dielectric_straight_mass(&split);
 failures+=!bsdf_diffraction_dielectric_setup(&sd,one_spectrum(),make_float3(0,0,1),make_float3(1,0,0),&split);
 failures+=sd.num_closure!=2 || sd.num_closure_left!=0;
 if(sd.num_closure==2) {
  failures+=sd.closure[0].type!=CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID ||
            sd.closure[1].type!=CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID;
  failures+=fabsf(average(sd.closure[0].weight)-atom)>1e-6f ||
            fabsf(average(sd.closure[1].weight)-(1-atom))>1e-6f;
  failures+=fabsf(sd.closure[0].sample_weight+sd.closure[1].sample_weight-1)>1e-6f;
 }
 ShaderData insufficient{};insufficient.num_closure_left=1;
 failures+=bsdf_diffraction_dielectric_setup(&insufficient,one_spectrum(),make_float3(0,0,1),make_float3(1,0,0),&split);
 failures+=insufficient.num_closure!=0 || insufficient.num_closure_left!=1;
 ShaderData reflection_only{};reflection_only.num_closure_left=1;
 failures+=!bsdf_diffraction_dielectric_setup(&reflection_only,one_spectrum(),make_float3(0,0,1),make_float3(1,0,0),&split,LABEL_TRANSMIT);
 failures+=reflection_only.num_closure!=1 || reflection_only.num_closure_left!=0;
 if(reflection_only.num_closure==1) {
   const auto *b=(const DiffractionDielectricBsdf *)&reflection_only.closure[0];
   failures+=b->disabled_lobes!=LABEL_TRANSMIT || fabsf(average(b->weight)-(1-atom))>1e-6f;
 }
 ShaderData neither{};neither.num_closure_left=2;
 failures+=bsdf_diffraction_dielectric_setup(&neither,one_spectrum(),make_float3(0,0,1),make_float3(1,0,0),&split,LABEL_TRANSMIT|LABEL_REFLECT);
 failures+=neither.num_closure!=0 || neither.num_closure_left!=2;
 printf("closure_bytes=%zu storage_bytes=%zu atoms=%d max_atom_error=%g continuous=%d max_eval_error=%g failures=%d\n",sizeof(DiffractionDielectricBsdf),sizeof(ShaderClosure),atoms,max_atom_error,continuous,max_eval_error,failures);
 return failures!=0;
}
