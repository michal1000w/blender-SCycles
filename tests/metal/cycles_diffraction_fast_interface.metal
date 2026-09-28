/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/closure/bsdf_diffraction_interface.h"

kernel void fast_interface(device const float4 *input [[buffer(0)]],
                           device float4 *output [[buffer(1)]],
                           uint i [[thread_position_in_grid]])
{
  const float4 a=input[4*i], b=input[4*i+1], c=input[4*i+2], d=input[4*i+3];
  const FastDiffractionInterface p{a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w,c.x};
  const float3 wi=float3(c.y,c.z,c.w);
  float3 wo=float3(0);float power=0,probability=0;int order=0;bool transmission=false;
  const bool valid=fast_diffraction_interface_sample(
      p,wi,d.x,&order,&transmission,&wo,&power,&probability);
  const float residual=fast_diffraction_interface_residual(p,wi);
  float reverse_power=0,query_error=0;
  if(valid) {
    FastDiffractionInterface reverse=p;
    float3 reverse_wi=wo;
    if(transmission) {
      reverse.incident_ior=p.transmitted_ior;reverse.transmitted_ior=p.incident_ior;
      reverse_wi=float3(wo.x,-wo.y,-wo.z);
    }
    const float3 reverse_wo=transmission?float3(wi.x,-wi.y,-wi.z):wi;
    float reverse_probability;
    if(!fast_diffraction_interface_probability(reverse,reverse_wi,reverse_wo,
                                               &reverse_power,&reverse_probability))reverse_power=-1;
    float queried_power,queried_probability;
    if(!fast_diffraction_interface_probability(p,wi,wo,&queried_power,&queried_probability))query_error=1;
    else query_error=max(fabs(queried_power-power),fabs(queried_probability-probability));
  }
  output[3*i]=float4(wo,power);
  output[3*i+1]=float4(probability,float(order),float(transmission),float(valid));
  output[3*i+2]=float4(residual,reverse_power,p.reflection_budget+p.transmission_budget,query_error);
}
