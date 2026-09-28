/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include <cstdio>
using namespace ccl;
int main()
{
 int failures=0,cases=0,orders=0,evanescent=0;double max_momentum_error=0;
 for(float wavelength:{380.f,550.f,780.f})
 for(float pitch:{740.f,1600.f})
 for(float depth:{0.f,150.f,500.f}) {
   DiffractionRoughDielectric a,b;
   failures+=!diffraction_dielectric_parameters(wavelength,pitch,depth,.42f,1,1.5f,.2f,.4f,&a);
   failures+=!diffraction_dielectric_parameters(wavelength,pitch,depth,.42f,1.5f,1,.2f,.4f,&b);
   failures+=fabsf(b.facet.wavelength_over_pitch-a.facet.wavelength_over_pitch/1.5f)>1e-7f;
   failures+=fabsf(b.facet.height_over_wavelength-a.facet.height_over_wavelength*1.5f)>1e-6f;
   failures+=fabsf(a.facet.transmission_phase+b.facet.transmission_phase)>1e-6f;
   failures+=fabsf(a.facet.transmission_phase-M_2PI_F*depth/wavelength*(-.5f))>2e-6f;
   ++cases;
 }
 /* Independent transverse momentum equation, with directions pointing away
  * from the interface: n_i wi_parallel + n_o wo_parallel = m lambda/pitch.
  * Reflection uses n_o=n_i. No inverse-facet helper is used as an oracle. */
 for(float wavelength:{380.f,550.f,780.f}) for(float pitch:{740.f,1600.f})
 for(float ni:{1.f,1.5f}) for(float no:{1.f,1.5f})
 for(float x:{-.7f,0.f,.6f}) for(int side=0;side<2;++side) {
   DiffractionRoughDielectric p;
   if(!diffraction_dielectric_parameters(wavelength,pitch,150,.42f,ni,no,.2f,.4f,&p))return 2;
   const float y=.17f;
   const float3 wi=make_float3(x,y,sqrtf(1-x*x-y*y));
   const double outgoing_index=side?no:ni;
   for(int m=-5;m<=5;++m) {
     const double ox=(m*double(wavelength)/pitch-ni*double(x))/outgoing_index;
     const double oy=-ni*double(y)/outgoing_index;
     const bool expected=ox*ox+oy*oy<1;
     float3 wo;
     const bool valid=side?
       diffraction_facet_transmit(wi,make_float3(0,0,1),make_float3(1,0,0),ni/no,
                                   m*p.facet.wavelength_over_pitch*(ni/no),&wo):
       diffraction_facet_reflect(wi,make_float3(0,0,1),make_float3(1,0,0),
                                 m*p.facet.wavelength_over_pitch,&wo);
     failures+=valid!=expected;
     if(valid) {
       const double error=std::max(std::abs(double(wo.x)-ox),std::abs(double(wo.y)-oy));
       max_momentum_error=std::max(max_momentum_error,error);
       failures+=error>5e-7 || ((wo.z<0)!=(side!=0)) || fabsf(len_squared(wo)-1)>2e-6f;
       ++orders;
     }
     else ++evanescent;
   }
 }
 DiffractionRoughDielectric p{};
 failures+=diffraction_dielectric_parameters(0,740,150,.5f,1,1.5f,.2f,.2f,&p);
 failures+=diffraction_dielectric_parameters(550,0,150,.5f,1,1.5f,.2f,.2f,&p);
 failures+=diffraction_dielectric_parameters(550,740,-1,.5f,1,1.5f,.2f,.2f,&p);
 printf("physical_parameter_cases=%d propagating_orders=%d evanescent_orders=%d max_momentum_error=%.12g failures=%d\n",cases,orders,evanescent,max_momentum_error,failures);
 return failures!=0;
}
