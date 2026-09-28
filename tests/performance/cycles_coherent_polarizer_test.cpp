/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/light/coherent_polarizer.h"
#include "kernel/light/coherent_path_field.h"
#include <cmath>
#include <cstdio>
using namespace ccl;
int main()
{
  int checks=0, failures=0;
  auto check=[&](float a,float b,float eps=3e-6f) { checks++; failures+=!std::isfinite(a)||fabsf(a-b)>eps; };
  auto axis=[](float a) {return make_float2(cosf(a),sinf(a));};
  const CoherentJonesField modes[2]={{make_float2(.707106781f,0),make_float2(0,0)},
                                   {make_float2(0,0),make_float2(.707106781f,0)}};
  for(int i=0;i<37;i++) {
    const float a=i*.137f;
    float single=0, parallel=0,crossed=0, relative=0,three=0;
    for(auto field:modes) {
      const auto first=coherent_polarizer_project(field,axis(a));
      single+=coherent_field_jones_power(first);
      parallel+=coherent_field_jones_power(coherent_polarizer_project(first,axis(a)));
      crossed+=coherent_field_jones_power(coherent_polarizer_project(first,axis(a+M_PI_2_F)));
      relative+=coherent_field_jones_power(coherent_polarizer_project(first,axis(a+.37f)));
      three+=coherent_field_jones_power(coherent_polarizer_project(
          coherent_polarizer_project(first,axis(a+M_PI_4_F)),axis(a+M_PI_2_F)));
    }
    check(single,.5f);check(parallel,.5f);check(crossed,0);check(relative,.5f*cosf(.37f)*cosf(.37f));check(three,.125f);
    // Transverse frame flip cannot rotate the physical object axis.
    float2 projected;
    const float3 world=make_float3(cosf(a),sinf(a),0);
    bool valid=coherent_polarizer_axis(world,make_float3(1,0,0),make_float3(0,-1,0),&projected);
    check(valid?1:0,1);check(projected.x,cosf(a));check(projected.y,-sinf(a));
  }
  const CoherentJonesField circular={make_float2(.707106781f,0),make_float2(0,.707106781f)};
  for(int i=0;i<7;i++) {
    const auto filtered=coherent_polarizer_project(circular,axis(.31f*i));
    check(coherent_field_jones_power(filtered),.5f);
    const auto twice=coherent_polarizer_project(filtered,axis(.31f*i));
    check(twice.s.x,filtered.s.x);check(twice.s.y,filtered.s.y);
    check(twice.p.x,filtered.p.x);check(twice.p.y,filtered.p.y);
  }
  // Interface Pout T Pin is reciprocal in flux-normalized endpoint bases.
  const auto fs=coherent_field_dielectric(1,1.5f,.7f,COHERENT_SCALAR_S);
  const auto fp=coherent_field_dielectric(1,1.5f,.7f,COHERENT_SCALAR_P);
  for(int i=0;i<13;i++) {
    const float2 ai=axis(.1f*i),ao=axis(.13f*i);
    const CoherentJonesField e0={make_float2(1,0),make_float2(0,0)};
    const CoherentJonesField e1={make_float2(0,0),make_float2(1,0)};
    auto f0=coherent_polarizer_transmit(e0,ai,ao,fs,fp),f1=coherent_polarizer_transmit(e1,ai,ao,fs,fp);
    auto r0=coherent_polarizer_transmit(e0,ao,ai,fs,fp),r1=coherent_polarizer_transmit(e1,ao,ai,fs,fp);
    check(f0.s.x,r0.s.x);check(f0.p.x,r1.s.x);check(f1.s.x,r0.p.x);check(f1.p.x,r1.p.x);
    check(coherent_field_jones_power(f0)+coherent_field_jones_power(f1)<=1.00001f?1:0,1);
  }
  // Actual Fresnel reflection followed by analyzer at Brewster's angle.
  const float brewster_cosine=1.0f/sqrtf(1.0f+1.5f*1.5f);
  const auto bs=coherent_field_dielectric(1,1.5f,brewster_cosine,COHERENT_SCALAR_S);
  const auto bp=coherent_field_dielectric(1,1.5f,brewster_cosine,COHERENT_SCALAR_P);
  for(int i=0;i<3;i++) {
    float intensity=0;
    for(auto field:modes) {
      field=coherent_field_interface_jones(field,bs,bp,false,-1);
      intensity+=coherent_field_jones_power(coherent_polarizer_project(field,axis(i*M_PI_4_F)));
    }
    const float rs=(1.5f*1.5f-1)/(1.5f*1.5f+1);
    check(intensity,.5f*rs*rs*(i==0?1:i==1?.5f:0));
  }
  // A real two-interface slab: the same physical axis must be idempotent.
  CoherentGeometryPath path{};
  path.count=2;path.point[0]=make_float3(0,0,0);path.point[1]=make_float3(0,0,1);
  path.spreading=1;path.optical_length_split=make_float2(3.5f,0);
  CoherentGeometryInterface patches[4]{};
  bool mirror[4]={false,false,false,false}, enabled[4]={true,true,false,false};
  float3 axes[4]{};
  for(int i=0;i<2;i++) {
    patches[i].tangent_u=make_float3(1,0,0);
    patches[i].tangent_v=make_float3(0,i==0?1:-1,0);
    patches[i].event=COHERENT_GEOMETRY_TRANSMIT;
    patches[i].ior_before=i==0?1:1.5f;patches[i].ior_after=i==0?1.5f:1;
  }
  const CoherentPathDetectorFrame detector={make_float3(1,0,0),make_float3(0,1,0),make_float3(0,0,-1)};
  for(int i=0;i<23;i++) {
    const float theta=.173f*i;
    axes[0]=axes[1]=make_float3(cosf(theta),sinf(theta),0);
    CoherentCompletedPathField field;
    check(coherent_path_field_transport(make_float3(0,0,-1),make_float3(0,0,2),detector,
          &path,patches,mirror,one_float3(),one_float3(),0,&field,axes,enabled)?1:0,1);
    const float base=.25f*M_1_PI_F*M_1_PI_F;
    check(field.physical_diagonal_rgb.x/base,.5f*.96f*.96f);
    // Crossed slabs physically absorb all rather than multiplying .5 twice.
    axes[1]=make_float3(-sinf(theta),cosf(theta),0);
    check(coherent_path_field_transport(make_float3(0,0,-1),make_float3(0,0,2),detector,
          &path,patches,mirror,one_float3(),one_float3(),0,&field,axes,enabled)?1:0,1);
    check(field.physical_diagonal_rgb.x/base,0);
  }
  // Matched substrate really connects through both sides of a closed slab.
  for(int flip=0;flip<2;flip++) {
    for(int i=0;i<2;i++) {
      patches[i].center=make_float3(0,0,float(i));
      patches[i].half_u=patches[i].half_v=1;
      patches[i].tangent_v=make_float3(0,((i+flip)&1)?-1:1,0);
      patches[i].ior_before=patches[i].ior_after=patches[i].ior_opposite=1;
      patches[i].expected_incident_side=0;
      axes[i]=make_float3(cosf(.37f),sinf(.37f),0);
    }
    CoherentGeometryPath matched;
    check(coherent_geometry_connect(make_float3(0,0,-1),make_float3(0,0,2),
          detector.normal,patches,2,&matched)?1:0,1);
    CoherentCompletedPathField field;
    check(coherent_path_field_transport(make_float3(0,0,-1),make_float3(0,0,2),detector,
          &matched,patches,mirror,one_float3(),one_float3(),0,&field,axes,enabled)?1:0,1);
    check(field.physical_diagonal_rgb.x/(matched.spreading*.25f*M_1_PI_F*M_1_PI_F),.5f);
  }
  float2 oblique_axis;
  const float3 direction=normalize(make_float3(1,0,1));
  const float3 oblique_s=make_float3(0,1,0), oblique_p=normalize(cross(oblique_s,direction));
  check(coherent_polarizer_axis(make_float3(1,0,0),oblique_s,oblique_p,&oblique_axis)?1:0,1);
  check(oblique_axis.x,0);check(fabsf(oblique_axis.y),1);
  printf("%d checks, %d failures\n",checks,failures);return failures?1:0;
}
