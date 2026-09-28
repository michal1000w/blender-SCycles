/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/closure/bsdf_diffraction_util.h"
uint measure_hash(uint x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
float measure_uniform(thread uint &state) {
  state = measure_hash(state + 0x9e3779b9u);
  return float(state >> 8) * (1.0f / 16777216.0f);
}
float3 measure_sphere(thread uint &state) {
  const float z = 2 * measure_uniform(state) - 1;
  const float phi = 2 * M_PI_F * measure_uniform(state);
  const float r = sqrt(max(0.0f,1-z*z));
  return float3(r*cos(phi),r*sin(phi),z);
}
int measure_bin(float3 v) { return int(v.x>=0) + 2*int(v.y>=0) + 4*int(v.z>=0); }
kernel void diffraction_transmission_measure(device float4 *outputs [[buffer(0)]],
                                             constant float2 &params [[buffer(1)]],
                                             constant uint &offset [[buffer(2)]],
                                             uint i [[thread_position_in_grid]])
{
  uint state = measure_hash(i+offset+91231u);
  const float3 wi = normalize(float3(.6f,.2f,.8f)), axis = float3(1,0,0);
  float3 h = measure_sphere(state), out;
  if (dot(wi,h)<0) h=-h;
  float forward=-1, missing=0;
  DiffractionTransmissionRoot roots[2];
  if (diffraction_facet_transmit(wi,h,axis,params.x,params.y,&out)) {
    forward=float(measure_bin(out));
    missing=diffraction_transmission_half_vectors(wi,out,axis,params.x,params.y,roots)==0;
  }
  const float3 wo=measure_sphere(state);
  const int n=diffraction_transmission_half_vectors(wi,wo,axis,params.x,params.y,roots);
  float weight=0;
  for (int r=0;r<n;++r)
    weight+=2*diffraction_transmission_jacobian(wi,wo,roots[r].h,axis,params.x,params.y);
  outputs[i]=float4(forward,float(measure_bin(wo)),weight,missing);
}
