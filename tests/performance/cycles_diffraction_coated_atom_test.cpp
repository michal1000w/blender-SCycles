/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include <cstdio>
#include <random>
using namespace ccl;
template<MicrofacetType type> int check()
{
  std::mt19937 rng(71924);std::uniform_real_distribution<float> uniform(0,1);
  int failures=0;double max_error[4]={};
  constexpr int counts[]={16,64,256,4096};
  for(float alpha:{.1f,.35f,.8f})for(float z:{.2f,.8f})for(float film:{.47f,3.5f}) {
    DiffractionRoughDielectric p{{.37f,.24f,.42f,1,1,1.4f},alpha,alpha*.7f};
    const float r=sqrtf(1-z*z);
    const float3 wi=make_float3(.6f*r,.8f*r,z);
    const float coefficient=diffraction_dielectric_straight_mass(&p);
    double total=0,square=0;
    constexpr int samples=500000;
    for(int i=0;i<samples;++i) {
      const float2 u=make_float2(uniform(rng),uniform(rng));
      const float3 h=type==BECKMANN?
          microfacet_beckmann_sample_vndf(wi,p.alpha_x,p.alpha_y,u):
          microfacet_ggx_sample_vndf(wi,p.alpha_x,p.alpha_y,u);
      const float ci=dot(wi,h);
      const double value=coefficient*(1-diffraction_thin_film_pair_reflectance(ci,ci,1,1,1.7f,film));
      total+=value;square+=value*value;
    }
    const double mean=total/samples,se=sqrt((square/samples-mean*mean)/samples);
    printf("distribution=%s alpha=%g cosine=%g film=%g reference=%.9g se=%.9g",
           type==GGX?"GGX":"Beckmann",alpha,z,film,mean,se);
    for(int j=0;j<4;++j) {
      float mass,reverse;
      failures+=!diffraction_dielectric_coated_straight_mass_quadrature<type>(
          &p,wi,1.7f,film,counts[j],&mass);
      failures+=!diffraction_dielectric_coated_straight_mass_quadrature<type>(
          &p,make_float3(-wi.x,wi.y,wi.z),1.7f,film,counts[j],&reverse);
      failures+=mass!=reverse || mass<0 || mass>coefficient;
      const double error=std::abs(mass-mean);max_error[j]=std::max(max_error[j],error);
      /* Only the high-resolution reference candidate has an accuracy gate.
       * Small work bounds are measured, not silently accepted as defaults. */
      if(j==3)failures+=error>6*se+.002;
      printf(" n%d=%.9g",counts[j],mass);
    }
    float uncoated;
    failures+=!diffraction_dielectric_coated_straight_mass_quadrature<type>(&p,wi,1.7f,0,16,&uncoated);
    failures+=uncoated!=coefficient;
    printf("\n");fflush(stdout);
  }
  printf("max_absolute_errors=%g,%g,%g,%g failures=%d\n",max_error[0],max_error[1],max_error[2],max_error[3],failures);
  return failures;
}
int main(){return check<GGX>()+check<BECKMANN>()!=0;}
