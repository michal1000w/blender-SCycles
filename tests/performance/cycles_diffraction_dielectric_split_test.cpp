/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include <array>
#include <cstdio>
#include <random>
using namespace ccl;
int main() {
 constexpr int samples=500000;
 std::mt19937 rng(293182);std::uniform_real_distribution<float> u(0,1);
 int failures=0;
 auto bin=[](float3 v){return int(v.x>=0)+2*int(v.y>=0)+4*int(v.z>=0);};
 for(float cosine:{.3f,.9f}) for(float phase:{0.f,1.4f,M_PI_F}) {
  const DiffractionRoughDielectric p{{.37f,.24f,.5f,1,1,phase},.35f,.55f};
  const float3 wi=make_float3(sqrtf(1-cosine*cosine),0,cosine);
  const float atom=diffraction_dielectric_straight_mass(&p),budget=1-atom;
  std::array<double,8> mixed{},mixed2{},split{},split2{};
  int discrete_in_continuous=0;
  for(int i=0;i<samples;++i) {
   float3 wo;float value,pdf;bool singular;
   if(diffraction_dielectric_sample(&p,wi,make_float3(u(rng),u(rng),u(rng)),&wo,&value,&pdf,&singular)) {
    const double weight=value/pdf;mixed[bin(wo)]+=weight;mixed2[bin(wo)]+=weight*weight;
   }
   if(diffraction_dielectric_sample_continuous(&p,wi,make_float3(u(rng),u(rng),u(rng)),&wo,&value,&pdf)) {
    const double weight=budget*value/pdf;split[bin(wo)]+=weight;split2[bin(wo)]+=weight*weight;
    discrete_in_continuous+=len_squared(wi+wo)<1e-12f;
    float query_pdf;const float query=diffraction_dielectric_eval(&p,wi,wo,&query_pdf);
    failures+=fabsf(query_pdf/budget-pdf)>1e-5f*max(1.f,pdf);
    failures+=fabsf(query/budget-value)>1e-5f*max(1.f,value);
   }
  }
  double max_error=0,max_se=0;
  for(int b=0;b<8;++b) {
   const double a=mixed[b]/samples,c=split[b]/samples;
   const double se=sqrt(std::max(0.,(mixed2[b]/samples-a*a+split2[b]/samples-c*c)/samples));
   const double expected=c+(b==bin(-wi)?atom:0);
   max_error=std::max(max_error,std::abs(a-expected));max_se=std::max(max_se,se);
   failures+=std::abs(a-expected)>6*se+.0001 || se>.005;
  }
  failures+=discrete_in_continuous!=0;
  printf("cosine=%g phase=%g atom=%g max_bin_error=%g max_bin_se=%g atoms_in_continuous=%d\n",cosine,phase,atom,max_error,max_se,discrete_in_continuous);
 }
 printf("failures=%d\n",failures);return failures!=0;
}
