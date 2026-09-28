/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/device/metal/globals.h"
#include "kernel/tables.h"
#include "kernel/sample/guiding_field.h"
#include "kernel/device/metal/context_begin.h"
#include "kernel/integrator/surface_shader.h"
#include "kernel/device/metal/context_end.h"

kernel void native_grating_polarizer_probe(
    device const float4 *input [[buffer(0)]], device float4 *output [[buffer(1)]],
    constant KernelParamsMetal &params [[buffer(2)]], uint i [[thread_position_in_grid]])
{
  MetalKernelContext ctx(params);
  ShaderData sd{};
  sd.num_closure=1;sd.num_closure_left=7;
  sd.wi=float3(0,0,1);sd.N=sd.Ng=float3(0,0,input[i].y);
  thread MetalKernelContext::DiffractionStraightPolarizerBsdf *b=(thread MetalKernelContext::DiffractionStraightPolarizerBsdf *)&sd.closure[0];
  b->type=CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID;
  b->N=sd.N;b->weight=float3(1);b->sample_weight=1;
  const float3 physical_axis=float3(cos(input[i].x),sin(input[i].x),0);
  b->polarizer_axis=physical_axis;b->polarizer=1;
  MetalKernelContext::DiffractionPolarizerExtra<MetalKernelContext::DiffractionCoatedAtomExtra> coating{};
  if(input[i].z>0) {
    thread MetalKernelContext::DiffractionCoatedAtomBsdf *c=(thread MetalKernelContext::DiffractionCoatedAtomBsdf *)&sd.closure[0];
    c->type=CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID;c->extra=&coating.base;
    c->polarizer=1;coating.axis=physical_axis;
  }
  float3 read;
  const bool found=ctx.bsdf_diffraction_polarizer_axis(&sd.closure[0],&read);
  const auto map=ctx.polarization_closure_mueller(&sd,&sd.closure[0],-sd.wi,true);
  const auto first=ctx.polarization_spectrum_apply(map,ctx.polarization_unpolarized(),true);
  const auto second=ctx.polarization_spectrum_apply(map,first,true);
  auto weighted=ctx.polarization_unpolarized();weighted.value[0]=float3(-2,.3f,3);
  const auto signed_first=ctx.polarization_spectrum_apply(map,weighted,true);
  const auto signed_second=ctx.polarization_spectrum_apply(map,signed_first,true);
  output[11*i]=float4(sizeof(ShaderClosure),sizeof(*b),float(found),1.f);
  output[11*i+1]=float4(read,input[i].x);
  output[11*i+2]=float4(average(first.value[0]),average(first.value[1]),average(first.value[2]),average(first.value[3]));
  output[11*i+3]=float4(average(second.value[0]),average(second.value[1]),average(second.value[2]),average(second.value[3]));
  output[11*i+4]=float4(map.value[0][0],map.value[0][1],map.value[0][2],map.value[0][3]);
  output[11*i+5]=float4(signed_first.value[0],0);
  output[11*i+6]=float4(signed_second.value[0],0);
  ShaderData actual{};actual.num_closure_left=8;
  actual.N=actual.Ng=actual.wi=float3(0,0,1);actual.rand_wavelength=.371f;
  const bool made=ctx.bsdf_diffraction_glass_setup(&actual,float3(1),float3(1),actual.N,
      float3(1,0,0),.1f,1.f,1150,320,.42f,1,0,false);
  const bool attached=ctx.bsdf_diffraction_glass_set_polarizer(&actual,0,physical_axis);
  actual.closure[0].N=float3(0,0,input[i].y);
  const auto selected=&actual.closure[0];
  const auto wrapped=ctx.polarization_surface_transport(nullptr,&actual,-actual.wi,
      ctx.polarization_unpolarized(),true,selected,float3(1e6f),true);
  const auto wrapped_twice=ctx.polarization_surface_transport(nullptr,&actual,-actual.wi,
      wrapped,true,selected,float3(1e6f),true);
  output[11*i+7]=float4(actual.num_closure,actual.num_closure_left,float(made),float(attached));
  output[11*i+8]=float4(average(wrapped.value[0]),average(wrapped.value[1]),average(wrapped.value[2]),average(wrapped.value[3]));
  output[11*i+9]=float4(average(wrapped_twice.value[0]),average(wrapped_twice.value[1]),average(wrapped_twice.value[2]),average(wrapped_twice.value[3]));
  float3 actual_axis;const bool actual_found=ctx.bsdf_diffraction_polarizer_axis(selected,&actual_axis);
  output[11*i+10]=float4(actual_axis,float(actual_found));
}
