/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction_albedo.h"
#include "kernel/closure/diffraction_thin_sheet_model.h"
#include "kernel/util/dielectric_dispersion.h"
#include <cstdio>
#include <cmath>
using namespace ccl;
static int checks=0,failures=0;
static void check(bool ok,const char*name){checks++;if(!ok){failures++;printf("FAIL %s\n",name);}}
int main()
{
  for(int kind=0;kind<3;kind++){
    DiffractionAlbedoRequest r;
    r.thin_sheet=true;r.alpha_x=.36f;r.alpha_y=bsdf_thin_glass_transmission_roughness(r.alpha_x,r.inside_ior);
    r.pitch_nm=1150;r.depth_nm=320;r.duty=.42f;r.wavelength_count=3;r.mu_count=4;r.phi_count=8;r.facet_samples=128;
    if(kind==1)r.transmission_tint=.8f;
    if(kind==2){r.transmission_is_spectral=true;r.transmission_bt709=make_float3(.2f,.5f,.8f);r.film_ior=1.32f;r.film_thickness_nm=250;r.inv_abbe=.02f;}
    DiffractionAlbedoTable table;std::string error;
    bool built=diffraction_albedo_build_cpu(r,table,error);
    check(built,"actual CPU thin-sheet builder");if(!built){printf("builder error: %s\n",error.c_str());continue;}
    check(diffraction_albedo_validate_table(table,error),"actual cache validation");
    for(int l=0;l<r.wavelength_count;l++){
      float wavelength=380+400.f*l/(r.wavelength_count-1);
      float n=dielectric_ior_at_wavelength(r.inside_ior,r.inv_abbe,dielectric_wavelength_um(wavelength));
      DiffractionThinSheetModel model{};
      model.reflection={r.alpha_x,wavelength/r.pitch_nm,2*M_2PI_F*r.depth_nm/wavelength,r.duty,false};
      model.transmission={bsdf_thin_glass_transmission_roughness(r.alpha_x,n),wavelength/r.pitch_nm,M_2PI_F*(n-1)*r.depth_nm/wavelength,r.duty,true};
      model.ior=n;model.reflection_tint=r.reflection_tint;model.transmission_tint=diffraction_albedo_thin_sheet_transmission_tint(r,wavelength);
      model.film_ior=r.film_ior;model.film_over_wavelength=r.film_thickness_nm/wavelength;
      for(int i=0;i<r.mu_count;i++)for(int j=0;j<r.phi_count;j++){
        float x=float(i)/(r.mu_count-1),mu=fmaxf(x*x,1e-5f),radial=sqrtf(fmaxf(0,1-x*x*x*x));
        float phi=M_2PI_F*(j+.5f)/r.phi_count;
        float3 wi=make_float3(radial*cosf(phi),radial*sinf(phi),mu);
        float2 c=diffraction_thin_sheet_model_coefficients(&model,wi.z);
        const int group[4]={j,r.phi_count-1-j,
          (r.phi_count/2-1-j+r.phi_count)%r.phi_count,(j+r.phi_count/2)%r.phi_count};
        double sum=0;
        for(int k:group){
          const float angle=M_2PI_F*(k+.5f)/r.phi_count;
          float3 incident=make_float3(radial*cosf(angle),radial*sinf(angle),mu);
          sum+=c.x+c.y-diffraction_thin_sheet_model_escape(&model,incident,r.facet_samples);
        }
        float q=float(sum*.25);
        float cached=table.values[(l*r.mu_count+i)*r.phi_count+j];
        check(fabsf(cached-q)<2e-6f,"cached q equals actual scalar model at node");
        check(cached==table.values[(l*r.mu_count+i)*r.phi_count+group[1]] &&
              cached==table.values[(l*r.mu_count+i)*r.phi_count+group[2]],
              "finite table exact even azimuth symmetry");
        check(cached>=0 && cached<=c.x+c.y+2e-6f,"missing energy does not recycle absorption");
      }
      check(diffraction_albedo_average(table,wavelength)>=0,"finite projected q average");
      if(r.transmission_is_spectral)check(fabsf(model.transmission_tint-saturatef(bt709_to_spectral_transmission(r.transmission_bt709,dielectric_wavelength_um(wavelength))))<2e-7f,"native spectral transmission cache profile");
    }
    DiffractionAlbedoRequest changed=r;changed.transmission_bt709.x+=.01f;
    check(!(changed==r),"raw spectral tint participates in request identity");
  }
  DiffractionAlbedoRequest bad;bad.thin_sheet=true;bad.inside_ior=.9f;std::string error;
  check(!diffraction_albedo_validate_request(bad,error),"unsupported sub-air sheet explicitly rejected");
  bad.inside_ior=1.5f;bad.reflection_tint=1.1f;
  check(!diffraction_albedo_validate_request(bad,error),"active tint explicitly rejected");
  bad.reflection_tint=1;bad.phi_count=7;
  check(!diffraction_albedo_validate_request(bad,error),"odd azimuth count explicitly rejected");
  for(int phi:{2,6,16}) {
    DiffractionAlbedoTable t;t.request.thin_sheet=true;t.request.wavelength_count=2;
    t.request.mu_count=2;t.request.phi_count=phi;t.values.resize(4*phi);
    for(size_t i=0;i<t.values.size();i++)t.values[i]=float((i*7)%19)/20;
    double before=0;for(float x:t.values)before+=x;
    diffraction_albedo_symmetrize_thin_sheet(t);
    double after=0;for(float x:t.values)after+=x;
    check(fabs(before-after)<2e-6,"symmetrization preserves row-integral budget");
  }
  printf("checks=%d failures=%d\n",checks,failures);return failures?1:0;
}
