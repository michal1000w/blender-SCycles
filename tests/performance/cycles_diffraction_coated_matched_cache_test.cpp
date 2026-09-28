/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Validate the native physical coated branch at matched indices, including
 * its separate straight atom. The16bases must collapse to the identical
 * physical cache; selected energy rows use independent outgoing quadrature. */
#include "scene/diffraction_albedo.h"
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include "kernel/util/dielectric_dispersion.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
using namespace ccl;
int main()
{
  DiffractionTwoSidedAlbedoRequest r;
  r.alpha_x=.35f;r.alpha_y=.5f;r.pitch_nm=1200;r.depth_nm=125;r.duty=.43f;
  r.film_ior=2.4f;r.film_thickness_nm=300;r.wavelength_count=40;
  r.mu_count=3;r.phi_count=4;r.facet_samples=1024;
  const float um=dielectric_wavelength_um(780);
  const float shift=dielectric_ior_at_wavelength(r.inside_ior,1,um)-r.inside_ior;
  r.inv_abbe=(1-r.inside_ior)/shift;
  for(int i=0;i<200;i++) {
    const float n=dielectric_ior_at_wavelength(r.inside_ior,r.inv_abbe,um);
    if(n==1)break;
    r.inv_abbe=std::nextafter(r.inv_abbe,n>1?INFINITY:-INFINITY);
  }
  if(dielectric_ior_at_wavelength(r.inside_ior,r.inv_abbe,um)!=1)return 1;
  DiffractionTwoSidedAlbedoTable physical,generalized;
  std::string error;
  if(!diffraction_two_sided_albedo_build_cpu(r,physical,error))return 2;
  r.generalized_f0_count=16;
  if(!diffraction_two_sided_albedo_build_cpu(r,generalized,error)){fprintf(stderr,"%s\n",error.c_str());return 3;}
  const int stride=2*r.wavelength_count*r.mu_count*r.phi_count;
  for(int basis=0;basis<16;basis++)for(int side=0;side<2;side++)for(int angle=0;angle<12;angle++) {
    const int a=(side*r.wavelength_count+r.wavelength_count-1)*12+angle,b=basis*stride+a;
    if(physical.reflectance[a]!=generalized.reflectance[b] ||
       physical.transmittance[a]!=generalized.transmittance[b] ||
       physical.deficits[a]!=generalized.deficits[b])return 4;
  }
  double max_error=0;
  for(int side=0;side<2;side++)for(float mu:{.25f,1.0f}) {
    const float phi=M_PI_F*.25f,radial=std::sqrt(1-mu*mu);
    const float3 wi=make_float3(radial*std::cos(phi),radial*std::sin(phi),mu);
    DiffractionRoughDielectric p;
    if(!diffraction_dielectric_parameters(780,r.pitch_nm,r.depth_nm,r.duty,1,1,r.alpha_x,r.alpha_y,&p))return 5;
    float atom;
    if(!diffraction_dielectric_coated_straight_mass_quadrature<GGX>(&p,wi,r.film_ior,r.film_thickness_nm/780,64,&atom))return 6;
    constexpr int mus=128,phis=256;
    double energy=atom;
    for(int i=0;i<mus;i++)for(int j=0;j<phis;j++) {
      const float om=(i+.5f)/mus,orr=std::sqrt(1-om*om),op=M_2PI_F*(j+.5f)/phis;
      const float3 wo=make_float3(orr*std::cos(op),orr*std::sin(op),om);
      float pdf;
      energy+=diffraction_dielectric_eval<GGX>(&p,wi,wo,&pdf,r.film_ior,r.film_thickness_nm/780,true)*double(M_2PI_F)/(mus*phis);
    }
    const double q=diffraction_two_sided_albedo_lookup(generalized,side,780,wi,7);
    max_error=std::max(max_error,std::abs(energy+q-1));
    printf("side%d mu%.2f native_atom64 %.6f outgoing_continuous_plus_atom %.6f q %.6f sum %.6f\n",side,mu,atom,energy,q,energy+q);
  }
  printf("matched physicalcoated16basesexact; independentoutgoing/nativeatom maxerror %.6f\n",max_error);
  return max_error<.02?0:7;
}
