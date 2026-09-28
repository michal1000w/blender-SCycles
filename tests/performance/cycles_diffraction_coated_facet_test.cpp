/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction.h"
#include <cstdio>
#include <random>
using namespace ccl;
int main()
{
  std::mt19937 rng(192473);std::uniform_real_distribution<float> random(0,1);
  const float3 h=make_float3(0,0,1),axis=make_float3(1,0,0);
  int failures=0,pairs=0;float max_reverse=0,max_flat=0,min_residual=1;
  for(int i=0;i<20000;++i) {
    const float ci=.01f+.99f*random(rng),phi=M_2PI_F*random(rng),s=sqrtf(1-ci*ci);
    const float3 wi=make_float3(s*cosf(phi),s*sinf(phi),ci);
    const float ni=i%3==2?1.5f:1,no=i%3==1?1.5f:1;
    DiffractionDielectricFacet p{.15f+random(rng),random(rng),random(rng),ni,no,6*random(rng)};
    const float nf=.8f+1.6f*random(rng),thickness=4*random(rng),eta=ni/no;
    const float residual=diffraction_dielectric_facet_residual(&p,wi,h,axis,nf,thickness);
    min_residual=min(min_residual,residual);
    failures+=!isfinite_safe(residual)||residual< -3e-5f||residual>1.00003f;
    for(int side=0;side<2;++side) {
      const bool trans=side!=0;
      const int bound=diffraction_dielectric_facet_order_bound(&p,trans);
      for(int m=-bound;m<=bound;++m) {
        if(!trans && m==0)continue;
        float3 wo;
        const bool valid=trans?
            diffraction_facet_transmit(wi,h,axis,eta,m*p.wavelength_over_pitch*eta,&wo):
            diffraction_facet_reflect(wi,h,axis,m*p.wavelength_over_pitch,&wo);
        if(!valid)continue;
        const float power=diffraction_dielectric_facet_power(&p,m,trans,ci,wo.z,nf,thickness);
        DiffractionDielectricFacet reverse=p;
        if(trans) {
          reverse.incident_ior=no;reverse.transmitted_ior=ni;
          reverse.wavelength_over_pitch*=eta;reverse.height_over_wavelength/=eta;
          reverse.transmission_phase=-p.transmission_phase;
        }
        const float back=diffraction_dielectric_facet_power(
            &reverse,m,trans,trans?-wo.z:wo.z,trans?-ci:ci,nf,thickness);
        max_reverse=max(max_reverse,fabsf(power-back));
        failures+=!isfinite_safe(power)||power<0||power>1||fabsf(power-back)>3e-5f;
        failures+=diffraction_dielectric_facet_power(&p,m,trans,ci,wo.z,nf,0)!=
                  diffraction_dielectric_facet_power(&p,m,trans,ci,wo.z);
        ++pairs;
      }
    }
    p.height_over_wavelength=p.transmission_phase=0;
    const float flat=diffraction_dielectric_facet_residual(&p,wi,h,axis,nf,thickness);
    const float expected=diffraction_thin_film_reflectance(ci,ni,no,nf,thickness);
    max_flat=max(max_flat,fabsf(flat-expected));
    failures+=fabsf(flat-expected)>2e-4f;
  }
  std::printf("cases=20000 pairs=%d min_residual=%.9g max_reverse=%.9g max_flat=%.9g failures=%d\n",
              pairs,min_residual,max_reverse,max_flat,failures);
  return failures!=0;
}
