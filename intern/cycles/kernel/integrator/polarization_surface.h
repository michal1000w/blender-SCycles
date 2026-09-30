/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/integrator/polarization_state.h"
CCL_NAMESPACE_BEGIN

ccl_device_inline PolarizationMueller polarization_finish_closure_map(
    const ccl_private MicrofacetBsdf *bsdf,const float3 incoming,const float3 outgoing,
    const bool transmit,const ccl_private PolarizationMueller &map)
{
  if(!transmit||bsdf->fresnel_type!=MicrofacetFresnel::GENERALIZED_SCHLICK_POLARIZER)return map;
  const float3 axis=((const ccl_private FresnelGeneralizedSchlickPolarizer *)bsdf->fresnel)->axis;
  const auto pin=polarization_mueller_from_jones(polarization_axis_filter(incoming,axis));
  const auto pout=polarization_mueller_from_jones(polarization_axis_filter(outgoing,axis));
  return polarization_mueller_product(pout,polarization_mueller_product(map,pin));
}

/* Normalize the polarization map by the existing unpolarized closure value.
 * This leaves every native proposal/PDF unchanged. The dielectric Fresnel map
 * is physical; unsupported closures use the declared depolarizing Fast model. */
ccl_device_inline PolarizationMueller polarization_closure_mueller(
    const ccl_private ShaderData *sd,const ccl_private ShaderClosure *sc,
    const float3 wo,const bool adjoint)
{
  const float3 incoming=adjoint?-wo:-sd->wi,outgoing=adjoint?sd->wi:wo;
  if(sc->type==CLOSURE_BSDF_TRANSPARENT_ID) {
    PolarizationMueller identity{};for(int i=0;i<4;i++)identity.value[i][i]=1;return identity;
  }
  float3 grating_axis;
  if (bsdf_diffraction_polarizer_axis(sc,&grating_axis)) {
    /* The straight transmitted atom preserves its polarization. Rough grating
     * and multiple-return energy use the declared depolarizing Fast substrate. */
    PolarizationMueller substrate=polarization_depolarizer(1);
    if(sc->type==CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID ||
       sc->type==CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID) {
      for(int i=0;i<4;i++)substrate.value[i][i]=1;
    }
    /* Straight atoms are transmission measures by construction: their native
     * sampler always returns -wi and LABEL_TRANSMIT. Classify their event by
     * type, including opposing or degenerate stored normals, rather than
     * inferring it again from the shading normal. */
    if(sc->type==CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID ||
       sc->type==CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID || dot(sc->N,wo)<0) {
      const auto pin=polarization_mueller_from_jones(polarization_axis_filter(incoming,grating_axis));
      const auto pout=polarization_mueller_from_jones(polarization_axis_filter(outgoing,grating_axis));
      return polarization_mueller_product(pout,polarization_mueller_product(substrate,pin));
    }
    return substrate;
  }
  if(!CLOSURE_IS_BSDF_MICROFACET(sc->type))return polarization_depolarizer(1);
  const ccl_private MicrofacetBsdf *bsdf=(const ccl_private MicrofacetBsdf *)sc;
  const bool transmit=dot(sc->N,wo)<0;
  bool dielectric=bsdf->fresnel_type==MicrofacetFresnel::DIELECTRIC ||
                  bsdf->fresnel_type==MicrofacetFresnel::DIELECTRIC_TINT;
  if(bsdf->fresnel_type==MicrofacetFresnel::GENERALIZED_SCHLICK ||
     bsdf->fresnel_type==MicrofacetFresnel::GENERALIZED_SCHLICK_POLARIZER) {
    const ccl_private FresnelGeneralizedSchlick *f=(const ccl_private FresnelGeneralizedSchlick *)bsdf->fresnel;
    const Spectrum physical_f0=make_spectrum(F0_from_ior(bsdf->ior));
    dielectric=f->exponent<0 && f->thin_film.thickness<=THINFILM_THICKNESS_CUTOFF &&
        reduce_max(fabs(f->f0-physical_f0))<1e-6f &&
        reduce_max(fabs(f->f90-one_spectrum()))<1e-6f;
  }
  const bool constant_fresnel=bsdf->fresnel_type==MicrofacetFresnel::NONE;
  if(!dielectric&&!constant_fresnel)return polarization_finish_closure_map(bsdf,incoming,outgoing,
      transmit,polarization_depolarizer(1));
  float3 H=transmit?sd->wi+bsdf->ior*wo:sd->wi+wo;
  H=len_squared(H)>1e-12f?normalize(H):sc->N;
  if(dot(H,sc->N)<0)H=-H;
  float2 fs=make_float2(1,0),fp=make_float2(1,0);
  if(dielectric) {
    const float incident_ior=adjoint&&transmit?bsdf->ior:1;
    const float transmitted_ior=adjoint&&transmit?1:bsdf->ior;
    const float cosine=clamp(fabsf(dot(incoming,H)),0.f,1.f);
    const auto s=coherent_field_dielectric(incident_ior,transmitted_ior,cosine,COHERENT_SCALAR_S);
    const auto p=coherent_field_dielectric(incident_ior,transmitted_ior,cosine,COHERENT_SCALAR_P);
    if(!s.valid||!p.valid)return polarization_depolarizer(1);
    fs=transmit?make_float2(s.transmission,0):s.reflection;
    fp=transmit?make_float2(p.transmission,0):p.reflection;
  }
  const float baseline=.5f*(coherent_field_power(fs)+coherent_field_power(fp));
  if(!(baseline>0))return polarization_depolarizer(0);
  if(!transmit)fp=-fp;
  auto j=polarization_incidence_jones(incoming,outgoing,H,fs,fp);
  auto m=polarization_mueller_from_jones(j);
  const float single_fraction=1.0f/max(bsdf->energy_scale,1.0f);
  for(int i=0;i<4;i++)for(int k=0;k<4;k++)m.value[i][k]*=single_fraction/baseline;
  /* Native GGX energy compensation is a separate depolarizing Fast return,
   * never additional perfectly polarized single-facet Fresnel energy. */
  m.value[0][0]+=1-single_fraction;
  return polarization_finish_closure_map(bsdf,incoming,outgoing,transmit,m);
}

#ifdef __KERNEL_METAL_VISIBLE_SHADING__
/* Compiled once as a Metal visible function, see `kernel.metal`. */
ccl_device_inline PolarizationSpectrumState polarization_surface_transport(
    KernelGlobals /*kg*/,
    ccl_private ShaderData *sd,
    const float3 wo,
    const ccl_private PolarizationSpectrumState &incoming_state,
    const bool adjoint,
    const ccl_private ShaderClosure *sampled_closure,
    const Spectrum native_total,
    const bool sampled_delta,
    const uint light_shader_flags = 0,
    const bool reciprocal_forward = false)
{
  PolarizationSpectrumState result;
  metal_ancillaries->vft_polarization[0](&launch_params_metal,
                                         metal_ancillaries,
                                         &result,
                                         sd,
                                         wo,
                                         &incoming_state,
                                         adjoint,
                                         sampled_closure,
                                         native_total,
                                         sampled_delta,
                                         light_shader_flags,
                                         reciprocal_forward);
  return result;
}
#endif

#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device_noinline
#endif
PolarizationSpectrumState
#ifdef __KERNEL_METAL_VISIBLE_SHADING__
polarization_surface_transport_impl(
#else
polarization_surface_transport(
#endif
    KernelGlobals kg,ccl_private ShaderData *sd,const float3 wo,
    const ccl_private PolarizationSpectrumState &incoming_state,const bool adjoint,
    const ccl_private ShaderClosure *sampled_closure,const Spectrum native_total,
    const bool sampled_delta,const uint light_shader_flags=0,
    const bool reciprocal_forward=false)
{
  PolarizationSpectrumState out{{zero_spectrum(),zero_spectrum(),zero_spectrum(),zero_spectrum()}};
  Spectrum other_total=zero_spectrum();
  const bool atomic_mixture=sampled_delta ||
      (sampled_closure && sampled_closure->type==CLOSURE_BSDF_TRANSPARENT_ID);
  for(int i=0;i<sd->num_closure;i++) {
    const ccl_private ShaderClosure *sc=&sd->closure[i];
    if(sc==sampled_closure||!CLOSURE_IS_BSDF(sc->type)||_surface_shader_exclude(sc->type,light_shader_flags))continue;
    float pdf;
#ifdef __KERNEL_METAL_VISIBLE_SHADING__
    /* Called directly to keep the Metal visible function call depth bounded. */
    const Spectrum value=(atomic_mixture?bsdf_eval_delta_impl(kg,sd,sc,wo,&pdf):bsdf_eval(kg,sd,sc,wo,&pdf))*sc->weight;
#else
    const Spectrum value=(atomic_mixture?bsdf_eval_delta(kg,sd,sc,wo,&pdf):bsdf_eval(kg,sd,sc,wo,&pdf))*sc->weight;
#endif
    if(!(pdf>0)||is_zero(value))continue;
    other_total+=value;
    const auto transformed=polarization_spectrum_apply(polarization_closure_mueller(sd,sc,wo,adjoint),incoming_state,adjoint&&!reciprocal_forward);
    for(int k=0;k<4;k++)out.value[k]+=transformed.value[k]*value;
  }
  if(sampled_closure) {
    /* The sampler's selected-closure value can be an atomic mass unavailable
     * through ordinary eval. Recover it from the exact returned mixture sum. */
    const Spectrum selected=native_total-other_total;
    const auto transformed=polarization_spectrum_apply(polarization_closure_mueller(sd,sampled_closure,wo,adjoint),incoming_state,adjoint&&!reciprocal_forward);
    for(int k=0;k<4;k++)out.value[k]+=transformed.value[k]*selected;
  }
  for(int k=0;k<4;k++)out.value[k]=safe_divide(out.value[k],native_total);
  return out;
}
CCL_NAMESPACE_END
