/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Independent double Cauchy check and outgoing quadrature versus scalar-basis
 * VNDF escape estimates. Neither cache construction nor its normalization is
 * used as the outgoing-energy reference. */
#include "scene/diffraction_albedo.h"
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include "kernel/util/dielectric_dispersion.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
using namespace ccl;

static double independent_ior(double nd, double inv_abbe, double nm)
{
  const double ld=.5876, lc=.6563, lf=.4861, l=nm*.001;
  const double B=(nd-1)*inv_abbe/(1/(lf*lf)-1/(lc*lc));
  return nd-B/(ld*ld)+B/(l*l);
}

int main()
{
  DiffractionTwoSidedAlbedoRequest canonical;
  auto changed = canonical;
  changed.inv_abbe=.05f;
  if (changed==canonical) return 8;
  changed=canonical;changed.generalized_f0_count=2;
  if (changed==canonical) return 9;
  std::string invalid_error;
  changed.inv_abbe=-.01f;
  if(diffraction_two_sided_albedo_validate_request(changed,invalid_error))return 10;
  double max_n_error=0;
  for(float nd : {1.2f,1.5f,2.4f}) for(float abbe : {0.0f,.02f,.1f})
    for(int i=0;i<=100;i++) {
      const float nm=380+4*i;
      const float n=dielectric_ior_at_wavelength(nd,abbe,dielectric_wavelength_um(nm));
      max_n_error=std::max(max_n_error,std::abs(n-independent_ior(nd,abbe,nm)));
      if(abbe==0 && n!=nd)return 1;
    }
  if(max_n_error>8e-7)return 2;
  /* Tune the native float Cauchy endpoint to n(780nm)==1, where a null
   * transmission atom is separate from continuous generalized reflection. */
  /* Keep the old multiply-by-.001 exact-match tuple as a near-boundary
   * regression, rather than silently replacing all prior precision evidence. */
  const float legacy_n=dielectric_ior_at_wavelength(1.5f,0x1.866fbep+0f,dielectric_wavelength_um(780.0f));
  if(!std::isfinite(legacy_n) || std::abs(legacy_n-1.0f)>5e-7f)return 14;
  std::printf("legacy boundary n780 %.9g\n",legacy_n);
  DiffractionTwoSidedAlbedoRequest matched;
  matched.generalized_f0_count=2;matched.wavelength_count=2;
  matched.mu_count=2;matched.phi_count=2;matched.facet_samples=64;
  const float unit_effect=dielectric_ior_at_wavelength(matched.inside_ior,1.0f,dielectric_wavelength_um(780.0f))-matched.inside_ior;
  matched.inv_abbe=(1.0f-matched.inside_ior)/unit_effect;
  bool exact=false;
  for(int i=0;i<200;i++) {
    const float value=dielectric_ior_at_wavelength(matched.inside_ior,matched.inv_abbe,dielectric_wavelength_um(780.0f));
    if(value==1.0f){exact=true;break;}
    matched.inv_abbe=std::nextafter(matched.inv_abbe,value>1.0f?INFINITY:-INFINITY);
  }
  if(!exact)return 11;
  DiffractionTwoSidedAlbedoTable matched_table;
  std::string matched_error;
  if(!diffraction_two_sided_albedo_build_cpu(matched,matched_table,matched_error))return 12;
  for(int side=0;side<2;side++) for(int angle=0;angle<4;angle++) {
    const int index=(side*2+1)*4+angle;
    if(matched_table.reflectance[index]!=0 || matched_table.transmittance[index]!=1 ||
       matched_table.deficits[index]!=0)return 13;
  }
  std::puts("matched-index F0=0 straight transmission atom exact");
  DiffractionTwoSidedAlbedoRequest r;
  r.alpha_x=.35f;r.alpha_y=.5f;r.pitch_nm=1200;r.depth_nm=125;r.duty=.43f;
  r.inv_abbe=.05f;r.generalized_f0_count=2;
  r.wavelength_count=4;r.mu_count=4;r.phi_count=8;r.facet_samples=8192;
  DiffractionTwoSidedAlbedoTable table;
  std::string error;
  if(!diffraction_two_sided_albedo_build_cpu(r,table,error)) {
    std::fprintf(stderr,"cache failed: %s\n",error.c_str());return 3;
  }
  const float nm=380, n=dielectric_ior_at_wavelength(r.inside_ior,r.inv_abbe,dielectric_wavelength_um(nm));
  const float mu=1.0f/9,phi=M_2PI_F*3.5f/8,radial=std::sqrt(1-mu*mu);
  const float3 wi=make_float3(radial*std::cos(phi),radial*std::sin(phi),mu);
  double max_q_error=0,max_reciprocity_error=0;
  constexpr int out_mu_count=128,out_phi_count=256;
  for(int side=0;side<2;side++) for(float f0 : {0.0f,F0_from_ior(r.inside_ior),.3f,1.0f}) {
    DiffractionDielectricGeneralizedExtra e{};
    if(!diffraction_dielectric_parameters(nm,r.pitch_nm,r.depth_nm,r.duty,
          side?n:1,side?1:n,r.alpha_x,r.alpha_y,&e.base.param))return 4;
    e.generalized_f0=make_spectrum(f0);e.generalized_reference_f0=F0_from_ior(n);
    double energy=0;
    for(int i=0;i<out_mu_count;i++) {
      const float om=(i+.5f)/out_mu_count,orr=std::sqrt(1-om*om);
      for(int j=0;j<out_phi_count;j++) {
        const float op=M_2PI_F*(j+.5f)/out_phi_count;
        for(int event=0;event<2;event++) {
          const float3 wo=make_float3(orr*std::cos(op),orr*std::sin(op),event?-om:om);
          float pdf;
          const float value=average(diffraction_dielectric_generalized_eval<GGX>(&e,wi,wo,&pdf));
          energy+=value;
          if(i==39 && j==57) {
            auto reverse=e;
            float3 rwi=wo,rwo=wi;
            if(event) {
              std::swap(reverse.base.param.facet.incident_ior,reverse.base.param.facet.transmitted_ior);
              /* Fixed tangent X, opposing normal: local Y/Z reverse. */
              rwi=make_float3(wo.x,-wo.y,-wo.z);rwo=make_float3(wi.x,-wi.y,-wi.z);
              if(!diffraction_dielectric_parameters(nm,r.pitch_nm,r.depth_nm,r.duty,
                 side?1:n,side?n:1,r.alpha_x,r.alpha_y,&reverse.base.param))return 5;
            }
            const float back=average(diffraction_dielectric_generalized_eval<GGX>(&reverse,rwi,rwo,&pdf));
            const double ni=side?n:1,no=event?(side?1:n):ni;
            const double a=value/(std::abs(wo.z)*no*no),b=back/(wi.z*ni*ni);
            max_reciprocity_error=std::max(max_reciprocity_error,std::abs(a-b)/std::max(1e-8,std::max(a,b)));
          }
        }
      }
    }
    energy*=double(M_2PI_F)/(out_mu_count*out_phi_count);
    const double q0=diffraction_two_sided_albedo_lookup(table,side,nm,wi,0);
    const double q1=diffraction_two_sided_albedo_lookup(table,side,nm,wi,1);
    const double q=(1-f0)*q0+f0*q1;
    max_q_error=std::max(max_q_error,std::abs(1-energy-q));
    std::printf("side%d f0 %.6f energy %.6f cachedq %.6f sum %.6f\n",side,f0,energy,q,energy+q);
    if(energy<0 || energy>1.03 || q<0 || q>1)return 6;
  }
  std::printf("Cauchy max %.9g outgoingq max %.6f reciprocity relative max %.8g\n",max_n_error,max_q_error,max_reciprocity_error);
  return max_q_error<.035 && max_reciprocity_error<.002 ? 0:7;
}
