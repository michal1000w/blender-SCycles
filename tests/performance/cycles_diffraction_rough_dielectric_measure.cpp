/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
using namespace ccl;
#ifdef DIFFRACTION_TEST_BECKMANN
constexpr MicrofacetType test_distribution=BECKMANN;
#else
constexpr MicrofacetType test_distribution=GGX;
#endif
int main(int argc,char **argv) {
 const int samples=argc>1?std::atoi(argv[1]):1000000;
 if(samples<1000||samples>100000000)return 2;
 const bool matched_normal_only=argc==3 && std::strcmp(argv[2],"--matched-normal-only")==0;
 if(argc>3 || (argc==3 && !matched_normal_only))return 2;
 std::mt19937 rng(99273);std::uniform_real_distribution<float> u(0,1);
 int failures=0;
 for(float cosine:{.3f,.9f}) for(float ni:{2.f/3,1.f,1.5f}) {
  if(matched_normal_only && (cosine!=.9f || ni!=1))continue;
  const DiffractionRoughDielectric p{{.37f,.24f,.42f,ni,1,1.4f},.35f,.55f};
  const float3 wi=make_float3(sqrtf(1-cosine*cosine),0,cosine);
  float coated_atom=0;
#ifdef DIFFRACTION_TEST_COATED_MIXTURE
  if(!diffraction_dielectric_coated_straight_mass_quadrature<test_distribution>(
      &p,wi,1.35f,.47f,4096,&coated_atom))return 2;
#endif
  std::array<double,4> sum{},square{},integral{},integral_square{};
  for(int i=0;i<samples;++i) {
   float3 wo;float value,pdf;bool singular;
#ifdef DIFFRACTION_TEST_COATED_MIXTURE
   const bool valid=diffraction_dielectric_sample_coated_mixture<test_distribution>(
       &p,wi,make_float3(u(rng),u(rng),u(rng)),1.35f,.47f,coated_atom,&wo,&value,&pdf,&singular);
#elif defined(DIFFRACTION_TEST_COATED)
   const bool valid=diffraction_dielectric_sample_coated_continuous<test_distribution>(
       &p,wi,make_float3(u(rng),u(rng),u(rng)),1.35f,.47f,&wo,&value,&pdf);
#else
   const bool valid=diffraction_dielectric_sample<test_distribution>(&p,wi,make_float3(u(rng),u(rng),u(rng)),&wo,&value,&pdf,&singular);
#endif
   if(valid) {
    const double weight=value/pdf;
    const double v[4]={1,weight,wo.z<0?weight:0,wo.z>0?weight:0};
    for(int k=0;k<4;++k) {sum[k]+=v[k];square[k]+=v[k]*v[k];}
#ifdef DIFFRACTION_TEST_COATED_MIXTURE
    failures+=weight<0||weight>20.0001||!std::isfinite(weight);
#else
    failures+=weight<0||weight>1.00001||!std::isfinite(weight);
#endif
   }
   const float z=2*u(rng)-1,phi=2*M_PI_F*u(rng),r=sqrtf(1-z*z);
   wo=make_float3(r*cosf(phi),r*sinf(phi),z);
#ifdef DIFFRACTION_TEST_COATED_MIXTURE
   value=diffraction_dielectric_eval_coated_mixture<test_distribution>(
       &p,wi,wo,1.35f,.47f,coated_atom,&pdf);
#elif defined(DIFFRACTION_TEST_COATED)
   value=diffraction_dielectric_eval<test_distribution>(&p,wi,wo,&pdf,1.35f,.47f,ni==1);
#else
   value=diffraction_dielectric_eval<test_distribution>(&p,wi,wo,&pdf);
#endif
   const double v[4]={4*M_PI*pdf,4*M_PI*value,wo.z<0?4*M_PI*value:0,wo.z>0?4*M_PI*value:0};
   for(int k=0;k<4;++k) {integral[k]+=v[k];integral_square[k]+=v[k]*v[k];}
  }
#ifdef DIFFRACTION_TEST_COATED_MIXTURE
  const double atom=coated_atom;
#elif defined(DIFFRACTION_TEST_COATED)
  const double atom=0; /* Only the continuous coated proposal is under test. */
#else
  const double atom=ni==1?diffraction_binary_power(0,p.facet.transmission_phase,p.facet.duty):0;
#endif
  printf("cosine=%g ni=%g atom=%g",cosine,ni,atom);
  for(int k=0;k<4;++k) {
   const double a=sum[k]/samples,b=integral[k]/samples;
   const double se=sqrt(std::max(0.,(square[k]/samples-a*a+integral_square[k]/samples-b*b)/samples));
   double target=b+(k<3?atom:0);
#ifdef DIFFRACTION_TEST_COATED_MIXTURE
   if(k==0)target=b+diffraction_dielectric_coated_atom_probability(&p,coated_atom);
#endif
   failures+=std::abs(a-target)>6*se+.0005 || se>.005 || !std::isfinite(target);
   printf(" metric%d=%g/%g(se=%g)",k,a,target,se);
   if(k==1) failures+=a>1.00001;
  }
  printf("\n");fflush(stdout);
 }
 printf("failures=%d\n",failures);
 return failures!=0;
}
