/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include <cstdio>
using namespace ccl;
int main()
{
 int failures=0;
 const DiffractionRoughDielectric p{{.37f,.24f,.42f,1,1,1.4f},.3f,.3f};
 const float atom=diffraction_dielectric_straight_mass(&p);
 const float3 N=make_float3(0,0,1),T=make_float3(1,0,0);
 ShaderData sd{};sd.num_closure_left=3;
 failures+=!bsdf_diffraction_dielectric_setup_tinted(&sd,make_spectrum(.2f),make_spectrum(.8f),N,T,&p);
 failures+=sd.num_closure!=2 || sd.num_closure_left!=0;
 if(sd.num_closure==2) {
  const auto *joint=(const DiffractionDielectricBsdf *)&sd.closure[1];
  failures+=!(joint->disabled_lobes & DIFFRACTION_DIELECTRIC_TINT_EXTRA);
  failures+=sd.closure[0].type!=CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID;
  failures+=fabsf(average(joint->weight)-(1-atom))>1e-6f;
  failures+=fabsf(average(sd.closure[0].weight)-.8f*atom)>1e-6f;
  failures+=fabsf(average(joint->extra->reflection)-.2f)>1e-6f || fabsf(average(joint->extra->transmission)-.8f)>1e-6f;
 }
 ShaderData limited{};limited.num_closure_left=2;
 failures+=bsdf_diffraction_dielectric_setup_tinted(&limited,make_spectrum(.2f),make_spectrum(.8f),N,T,&p);
 failures+=limited.num_closure!=0 || limited.num_closure_left!=2;
 ShaderData same{};same.num_closure_left=2;
 failures+=!bsdf_diffraction_dielectric_setup_tinted(&same,one_spectrum(),one_spectrum(),N,T,&p);
 failures+=same.num_closure!=2;
 ShaderData reflection{};reflection.num_closure_left=1;
 failures+=!bsdf_diffraction_dielectric_setup_tinted(&reflection,one_spectrum(),zero_spectrum(),N,T,&p);
 failures+=(reflection.runtime_flag&SR_BSDF_HAS_TRANSMISSION)!=0;
 /* Extras grow from the opposite end of ShaderData. Later closures and extras
  * must not overwrite an earlier joint closure's parameters or colors. */
 ShaderData mixed{};mixed.num_closure_left=8;
 failures+=!bsdf_diffraction_dielectric_setup_tinted(&mixed,make_spectrum(.2f),make_spectrum(.8f),N,T,&p);
 const auto *first=(const DiffractionDielectricBsdf *)&mixed.closure[1];
 const auto *first_extra=first->extra;
 DiffractionRoughDielectric p2=p;p2.alpha_x=.6f;
 failures+=!bsdf_diffraction_dielectric_setup_tinted(&mixed,make_spectrum(.7f),make_spectrum(.3f),N,T,&p2);
 const auto *second=(const DiffractionDielectricBsdf *)&mixed.closure[3];
 failures+=first_extra==second->extra;
 failures+=!bsdf_diffraction_dielectric_setup_tinted(&mixed,one_spectrum(),one_spectrum(),N,T,&p);
 failures+=mixed.num_closure!=6 || mixed.num_closure_left!=0;
 failures+=bsdf_diffraction_dielectric_setup_tinted(&mixed,one_spectrum(),one_spectrum(),N,T,&p);
 failures+=mixed.num_closure!=6 || mixed.num_closure_left!=0;
 failures+=first->extra!=first_extra;
 failures+=diffraction_dielectric_param(first)->alpha_x!=.3f;
 failures+=diffraction_dielectric_param(second)->alpha_x!=.6f;
 failures+=fabsf(average(first_extra->reflection)-.2f)>1e-6f;
 failures+=fabsf(average(first_extra->transmission)-.8f)>1e-6f;
 failures+=fabsf(average(second->extra->reflection)-.7f)>1e-6f;
 failures+=fabsf(average(second->extra->transmission)-.3f)>1e-6f;
 printf("tint_setup_failures=%d closure_bytes=%zu\n",failures,sizeof(DiffractionDielectricBsdf));
 return failures!=0;
}
