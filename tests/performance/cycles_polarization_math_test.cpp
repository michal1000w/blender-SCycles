/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/light/polarization_math.h"
#include <cstdio>
#include <cmath>
#include <initializer_list>
using namespace ccl;
static int checks=0,failures=0;
static void check(bool x,const char*s){checks++;if(!x){failures++;printf("FAIL %s\n",s);}}
int main(){
 PolarizationStokes unpolarized{{1,0,0,0}},camera{{1,0,0,0}};
 for(float angle:{0.f,.2f,.7f,1.5707963267948966f}) {
  auto first=polarization_mueller_from_jones(polarization_linear_filter(0));
  auto second=polarization_mueller_from_jones(polarization_linear_filter(angle));
  auto after=polarization_apply(second,polarization_apply(first,unpolarized));
  check(fabsf(after.value[0]-.5f*cosf(angle)*cosf(angle))<2e-7f,"ordered filters obey Malus");
  auto adjoint=polarization_apply(first,polarization_apply(second,camera,true),true);
  check(fabsf(polarization_contract(adjoint,unpolarized)-after.value[0])<2e-7f,"camera adjoint equals forward radiance");
  check(after.value[0]>=-1e-8f&&after.value[0]<=1,"passive polarizer");
 }
 // A 45-degree intermediate filter transmits light between crossed filters.
 auto x=polarization_mueller_from_jones(polarization_linear_filter(0));
 auto diagonal=polarization_mueller_from_jones(polarization_linear_filter(M_PI_4_F));
 auto y=polarization_mueller_from_jones(polarization_linear_filter(M_PI_2_F));
 auto three=polarization_apply(y,polarization_apply(diagonal,polarization_apply(x,unpolarized)));
 check(fabsf(three.value[0]-.125f)<1e-7f,"three ordered filters retain coherency");
 for(float angle:{0.f,.4f,atanf(1.5f),1.2f}) {
  auto s=coherent_field_dielectric(1,1.5f,cosf(angle),COHERENT_SCALAR_S);
  auto p=coherent_field_dielectric(1,1.5f,cosf(angle),COHERENT_SCALAR_P);
  PolarizationJones j{{{s.reflection,zero_float2()},{zero_float2(),p.reflection}}};
  auto m=polarization_mueller_from_jones(j);auto reflected=polarization_apply(m,unpolarized);
  float expected=.5f*(coherent_field_power(s.reflection)+coherent_field_power(p.reflection));
  check(fabsf(reflected.value[0]-expected)<2e-7f,"native unpolarized Fresnel mean");
  if(fabsf(angle-atanf(1.5f))<1e-6f){check(fabsf(p.reflection.x)<2e-7f,"Brewster p reflection vanishes");check(fabsf(reflected.value[1]-reflected.value[0])<2e-7f,"Brewster reflected glare is linear s");}
  auto filtered=polarization_apply(y,reflected);check(filtered.value[0]>=-1e-8f&&filtered.value[0]<=reflected.value[0]+1e-8f,"analyzer never amplifies glare");
 }
 auto mixed=polarization_mueller_from_jones(polarization_linear_filter(.3f));
 auto opposite=polarization_mueller_from_jones(polarization_linear_filter(1.1f));
 for(int i=0;i<4;i++)for(int j=0;j<4;j++)mixed.value[i][j]=.2f*mixed.value[i][j]+.8f*opposite.value[i][j];
 auto evaluated=polarization_apply(mixed,unpolarized);
 check(fabsf(evaluated.value[0]-.5f)<2e-7f,"closure mixture sums Mueller maps");
 auto dep=polarization_apply(polarization_depolarizer(.7f),evaluated);
 check(fabsf(dep.value[0]-.35f)<2e-7f&&dep.value[1]==0&&dep.value[2]==0&&dep.value[3]==0,"diffuse depolarizes while preserving scalar intensity");
 PolarizationJones retarder{{{make_float2(1,0),zero_float2()},{zero_float2(),make_float2(0,1)}}};
 auto circular=polarization_apply(polarization_mueller_from_jones(retarder),PolarizationStokes{{1,0,1,0}});
 check(fabsf(circular.value[3]-1)<1e-7f,"complex Jones retains circular Stokes");
 const float3 d=normalize(make_float3(.3f,.2f,1));
 auto axis=polarization_axis_filter(d,make_float3(1,0,0));
 auto axis_out=polarization_apply(polarization_mueller_from_jones(axis),unpolarized);
 check(fabsf(axis_out.value[0]-.5f)<3e-7f,"world axis transverse projection");
 printf("polarization_math checks%d failures%d\n",checks,failures);return failures?1:0;
}
