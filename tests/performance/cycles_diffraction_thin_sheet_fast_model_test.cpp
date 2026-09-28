/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/globals.h"
#include "kernel/closure/bsdf.h"
#include "kernel/closure/diffraction_thin_sheet_model.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
using namespace ccl;
int main()
{
 printf("closure_size %zu extra_size %zu shader_slot %zu extra_slots %zu\n",sizeof(DiffractionThinSheetBsdf),sizeof(DiffractionThinSheetExtra),sizeof(ShaderClosure),(sizeof(DiffractionThinSheetExtra)+sizeof(ShaderClosure)-1)/sizeof(ShaderClosure));
 int checks=0,failures=0;double max_recip=0,max_row=0;
 for (float alpha : {.36f,.81f}) for (float tint : {1.f,.8f}) for (float film : {0.f,250.f}) {
  DiffractionThinSheetModel m{};
  m.reflection={alpha,550.f/1150.f,4*M_PI_F*320.f/550.f,.42f,false};
  float n=1.5f;m.transmission={alpha*sqrtf(3.4f*(n-1)*sqr(n-.5f)/(n*n*n)),550.f/1150.f,2*M_PI_F*(n-1)*320.f/550.f,.42f,true};
  m.ior=n;m.reflection_tint=1;m.transmission_tint=tint;m.film_ior=1.32f;m.film_over_wavelength=film/550;
  const float3 wi=normalize(make_float3(.4f,.2f,1.f));
  double dense=0;
  constexpr int M=96,P=192;
  for(int side : {1,-1}) for(int a=0;a<M;++a) for(int b=0;b<P;++b) {
   float mu=(a+.5f)/M,phi=2*M_PI_F*(b+.5f)/P,r=sqrtf(1-mu*mu);
   float3 wo=make_float3(r*cosf(phi),r*sinf(phi),side*mu);
   const bool t=side<0;
   double f=diffraction_thin_sheet_model_eval(&m,t,wi,wo);
   float3 ri=t ? make_float3(wo.x,-wo.y,-wo.z) : wo;
   float3 ro=t ? make_float3(wi.x,-wi.y,-wi.z) : wi;
   double rev=diffraction_thin_sheet_model_eval(&m,t,ri,ro);
   double error=std::abs(f/mu-rev/wi.z)/(1+std::abs(f/mu));max_recip=std::max(max_recip,error);
   checks++;if(!std::isfinite(f)||f<0||error>1e-4)failures++;
   dense+=f;
  }
  dense*=2*M_PI/(M*P);
  double estimate=diffraction_thin_sheet_model_escape(&m,wi,4096);
  double target=double(diffraction_thin_sheet_model_coefficients(&m,wi.z).x)+diffraction_thin_sheet_model_coefficients(&m,wi.z).y;
  double error=std::abs(dense-estimate);max_row=std::max(max_row,error);
  checks++;if(error>.02||estimate>target+2e-5||estimate<0)failures++;
  printf("alpha %.3f tint %.2f film %.0f target %.8f envelope_dense %.8f estimate %.8f q %.8f row_error %.8f\n",alpha,tint,film,target,dense,estimate,target-estimate,error);
 }
 for(float d : {0.f,.0001f,.001f,.01f}) {
  float b=diffraction_thin_sheet_model_relief_blend(4*M_PI_F*d/550,2*M_PI_F*.5f*d/550,.42f);
  checks++;if(b>1e-6f||b<0)failures++;
 }
 printf("checks %d failures %d reciprocity_relative %.9g row_abs %.9g\n",checks,failures,max_recip,max_row);return failures!=0;
}
