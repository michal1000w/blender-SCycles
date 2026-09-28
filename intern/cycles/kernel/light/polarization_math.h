/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/light/coherent_field.h"
CCL_NAMESPACE_BEGIN

/* Native incoherent polarization uses a coherency matrix, not a coherent
 * source phase. C=.5[[I+Q,U-iV],[U+iV,I-Q]]. A camera sensitivity propagates
 * with the transpose Mueller map; light radiance propagates forwards. */
struct PolarizationStokes { float value[4]; };
struct PolarizationJones { float2 value[2][2]; };
struct PolarizationMueller { float value[4][4]; };
ccl_device_inline float2 polarization_conjugate(const float2 a) {return make_float2(a.x,-a.y);}

ccl_device_inline PolarizationMueller polarization_mueller_from_jones(
    const ccl_private PolarizationJones &j)
{
  PolarizationMueller result{};
  for(int column=0;column<4;column++) {
    float2 c[2][2]={{zero_float2(),zero_float2()},{zero_float2(),zero_float2()}};
    if(column==0){c[0][0].x=.5f;c[1][1].x=.5f;}
    if(column==1){c[0][0].x=.5f;c[1][1].x=-.5f;}
    if(column==2){c[0][1].x=.5f;c[1][0].x=.5f;}
    if(column==3){c[0][1].y=-.5f;c[1][0].y=.5f;}
    float2 out[2][2]={{zero_float2(),zero_float2()},{zero_float2(),zero_float2()}};
    for(int a=0;a<2;a++)for(int b=0;b<2;b++)for(int p=0;p<2;p++)for(int q=0;q<2;q++)
      out[a][b]=coherent_field_add(out[a][b],coherent_field_mul(coherent_field_mul(j.value[a][p],c[p][q]),polarization_conjugate(j.value[b][q])));
    result.value[0][column]=out[0][0].x+out[1][1].x;
    result.value[1][column]=out[0][0].x-out[1][1].x;
    result.value[2][column]=2*out[0][1].x;
    result.value[3][column]=-2*out[0][1].y;
  }
  return result;
}
ccl_device_inline PolarizationMueller polarization_mueller_product(
    const ccl_private PolarizationMueller &a,const ccl_private PolarizationMueller &b)
{
  PolarizationMueller r{};
  for(int i=0;i<4;i++)for(int j=0;j<4;j++)for(int k=0;k<4;k++)r.value[i][j]+=a.value[i][k]*b.value[k][j];
  return r;
}
ccl_device_inline PolarizationStokes polarization_apply(
    const ccl_private PolarizationMueller &m,const ccl_private PolarizationStokes &s,
    const bool adjoint=false)
{
  PolarizationStokes result{};
  for(int i=0;i<4;i++)for(int j=0;j<4;j++)result.value[i]+=(adjoint?m.value[j][i]:m.value[i][j])*s.value[j];
  return result;
}
ccl_device_inline float polarization_contract(const ccl_private PolarizationStokes &a,
                                             const ccl_private PolarizationStokes &s)
{
  float result=0;for(int i=0;i<4;i++)result+=a.value[i]*s.value[i];return result;
}
ccl_device_inline PolarizationMueller polarization_depolarizer(const float weight)
{
  PolarizationMueller m{};m.value[0][0]=weight;return m;
}
ccl_device_inline PolarizationJones polarization_linear_filter(const float angle)
{
  const float c=cosf(angle),s=sinf(angle);
  return {{{make_float2(c*c,0),make_float2(c*s,0)},
           {make_float2(c*s,0),make_float2(s*s,0)}}};
}
ccl_device_inline PolarizationJones polarization_jones_product(
    const ccl_private PolarizationJones &a,const ccl_private PolarizationJones &b)
{
  PolarizationJones r{};
  for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++)
    r.value[i][j]=coherent_field_add(r.value[i][j],coherent_field_mul(a.value[i][k],b.value[k][j]));
  return r;
}

/* Express diagonal local incidence Jones transport in canonical transverse
 * frames. Input/output directions are physical light propagation directions.
 * The caller supplies the p gauge sign (reflection changes tangential sign). */
ccl_device_inline PolarizationJones polarization_incidence_jones(
    const float3 incoming,const float3 outgoing,const float3 normal,
    const float2 s_factor,const float2 p_factor)
{
  float3 iu,iv,ou,ov;make_orthonormals(incoming,&iu,&iv);make_orthonormals(outgoing,&ou,&ov);
  float3 s=cross(incoming,normal);
  if(len_squared(s)>1e-12f)s=normalize(s);else s=iu;
  const float3 ip=cross(s,incoming),op=cross(s,outgoing);
  const float3 ib[2]={iu,iv},ob[2]={ou,ov};PolarizationJones j{};
  for(int a=0;a<2;a++)for(int b=0;b<2;b++)
    j.value[a][b]=coherent_field_add(coherent_field_scale(s_factor,dot(ob[a],s)*dot(s,ib[b])),
                                    coherent_field_scale(p_factor,dot(ob[a],op)*dot(ip,ib[b])));
  return j;
}
ccl_device_inline PolarizationJones polarization_axis_filter(const float3 direction,
                                                             const float3 world_axis)
{
  float3 u,v;make_orthonormals(direction,&u,&v);
  const float3 projected=world_axis-direction*dot(world_axis,direction);
  if(!(len_squared(projected)>1e-12f))return {};
  const float3 axis=normalize(projected);const float a=dot(axis,u),b=dot(axis,v);
  return {{{make_float2(a*a,0),make_float2(a*b,0)},
           {make_float2(a*b,0),make_float2(b*b,0)}}};
}
CCL_NAMESPACE_END
