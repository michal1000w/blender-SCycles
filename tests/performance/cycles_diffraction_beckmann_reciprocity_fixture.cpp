/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include <cstdio>
using namespace ccl;
int main()
{
 DiffractionRoughDielectric p{{.85521405935287476f,.14083133637905121f,.66521966457366943f,1,1.5f,1.9346520900726318f},.6099353432655334f,.36753058433532715f};
 float3 wi=make_float3(-.25056535005569458f,.67943364381790161f,.68962806463241577f);
 float3 wo=make_float3(.7639737725257874f,-.45289772748947144f,-.4595952332019806f);
 for(int reverse=0;reverse<2;++reverse) {
  const float eta=p.facet.incident_ior/p.facet.transmitted_ior;
  float pdf;
  const float value=diffraction_dielectric_eval<BECKMANN>(&p,wi,wo,&pdf);
  printf("reverse=%d value=%g pdf=%g bsdf=%g\n",reverse,value,pdf,value/fabsf(wo.z));
  const int bound=diffraction_dielectric_facet_order_bound(&p.facet,true);
  for(int m=-bound;m<=bound;++m) {
   const float delta=m*p.facet.wavelength_over_pitch*eta;
   DiffractionTransmissionRoot roots[2];
   const int count=diffraction_transmission_half_vectors(wi,wo,make_float3(1,0,0),eta,delta,roots);
   for(int j=0;j<count;++j) {
    const float3 h=roots[j].h;if(h.z<=0)continue;
    const float power=diffraction_dielectric_facet_power(&p.facet,m,true,dot(wi,h),dot(wo,h));
    const float jac=diffraction_transmission_jacobian(wi,wo,h,make_float3(1,0,0),eta,delta);
    printf("m=%d root=%d h=(%.9g %.9g %.9g) power=%.9g jac=%.9g D=%.9g\n",m,j,h.x,h.y,h.z,power,jac,bsdf_aniso_D<BECKMANN>(p.alpha_x,p.alpha_y,h));
   }
  }
  const float3 oldwi=wi;wi=make_float3(wo.x,-wo.y,-wo.z);wo=make_float3(oldwi.x,-oldwi.y,-oldwi.z);
  p.facet.incident_ior=1.5f;p.facet.transmitted_ior=1;
  p.facet.wavelength_over_pitch*=eta;p.facet.height_over_wavelength/=eta;
 }
 return 0;
}
