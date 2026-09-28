/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/light/coherent_polarizer.h"
using namespace metal;
kernel void coherent_polarizer_compile(device float4 *out [[buffer(0)]])
{
  float2 ai,ao;
  const float3 axis=float3(.5f,.5f,0);
  coherent_polarizer_axis(axis,float3(1,0,0),float3(0,1,0),&ai);
  coherent_polarizer_axis(axis,float3(1,0,0),float3(0,1,0),&ao);
  const auto fs=coherent_field_dielectric(1,1.5f,1,COHERENT_SCALAR_S);
  const auto fp=coherent_field_dielectric(1,1.5f,1,COHERENT_SCALAR_P);
  const auto field=coherent_polarizer_transmit({float2(1,0),float2(0,1)},ai,ao,fs,fp);
  out[0]=float4(field.s,field.p);
}
