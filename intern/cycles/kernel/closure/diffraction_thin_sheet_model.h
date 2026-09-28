/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/closure/bsdf_diffraction.h"
#include "kernel/closure/diffraction_thin_film.h"
CCL_NAMESPACE_BEGIN

struct DiffractionThinSheetModelPort {
  float alpha;
  float wavelength_over_pitch;
  float phase;
  float duty;
  bool transmission;
};

ccl_device_inline float3 diffraction_thin_sheet_model_port_flip_tangent(const float3 w)
{
  return make_float3(-w.x, -w.y, w.z);
}

ccl_device_inline int diffraction_thin_sheet_model_port_order_bound(ccl_private const DiffractionThinSheetModelPort *p)
{
  if (p->phase == 0.0f || !(p->wavelength_over_pitch > 0.0f)) {
    return 0;
  }
  return int(min(2.0f / p->wavelength_over_pitch, float(DIFFRACTION_FAST_MAX_ORDER)));
}

ccl_device_inline float diffraction_thin_sheet_model_port_power(ccl_private const DiffractionThinSheetModelPort *p,
                                                       const int order,
                                                       const float3 incident,
                                                       const float3 h)
{
  if (order != 0) {
    return diffraction_binary_power(order, p->phase, p->duty);
  }
  const float3 axis = make_float3(1.0f, 0.0f, 0.0f);
  const int bound = diffraction_thin_sheet_model_port_order_bound(p);
  float other = 0.0f;
  for (int m = -bound; m <= bound; m++) {
    float3 candidate;
    if (m != 0 && diffraction_facet_reflect(
                      incident, h, axis, m * p->wavelength_over_pitch, &candidate))
    {
      other += diffraction_binary_power(m, p->phase, p->duty);
    }
  }
  return max(0.0f, 1.0f - other);
}

/* Result and PDF are in outgoing solid angle; result is f*abs(cos_o). */
ccl_device_inline float diffraction_thin_sheet_model_port_eval(ccl_private const DiffractionThinSheetModelPort *p,
                                                     const float3 wi,
                                                     const float3 wo,
                                                     ccl_private float *pdf)
{
  *pdf = 0.0f;
  const float3 I = p->transmission ? diffraction_thin_sheet_model_port_flip_tangent(wi) : wi;
  const float3 O = p->transmission ? -wo : wo;
  if (!(I.z > 0.0f && O.z > 0.0f) || roughness_is_almost_specular(p->alpha, p->alpha)) {
    return 0.0f;
  }
  const float3 axis = make_float3(1.0f, 0.0f, 0.0f);
  float density = 0.0f;
  const int bound = diffraction_thin_sheet_model_port_order_bound(p);
  for (int m = -bound; m <= bound; m++) {
    const float delta = m * p->wavelength_over_pitch;
    DiffractionReflectionRoot roots[2];
    const int count = diffraction_reflection_half_vectors(I, O, axis, delta, roots);
    for (int r = 0; r < count; r++) {
      const float3 h = roots[r].h;
      if (!(h.z > 0.0f)) {
        continue;
      }
      const float power = diffraction_thin_sheet_model_port_power(p, m, I, h);
      const float J = diffraction_reflection_jacobian(I, O, h, axis, delta);
      density += power * bsdf_aniso_D<GGX>(p->alpha, p->alpha, h) * dot(I, h) * J / I.z;
    }
  }
  const float li = bsdf_aniso_lambda<GGX>(p->alpha, p->alpha, I);
  const float lo = bsdf_aniso_lambda<GGX>(p->alpha, p->alpha, O);
  *pdf = density / (1.0f + li);
  return density / (1.0f + li + lo);
}

ccl_device_inline bool diffraction_thin_sheet_model_port_sample(ccl_private const DiffractionThinSheetModelPort *p,
                                                       const float3 wi,
                                                       const float3 random,
                                                       ccl_private float3 *wo,
                                                       ccl_private float *value,
                                                       ccl_private float *pdf)
{
  *wo = zero_float3();
  *value = *pdf = 0.0f;
  const float3 I = p->transmission ? diffraction_thin_sheet_model_port_flip_tangent(wi) : wi;
  if (!(I.z > 0.0f)) {
    return false;
  }
  const bool singular = roughness_is_almost_specular(p->alpha, p->alpha);
  const float3 h = singular ? make_float3(0.0f, 0.0f, 1.0f) :
                              microfacet_ggx_sample_vndf(I, p->alpha, p->alpha, make_float2(random));
  const float3 axis = make_float3(1.0f, 0.0f, 0.0f);
  const int bound = diffraction_thin_sheet_model_port_order_bound(p);
  float u = random.z;
  float mass = 0.0f;
  float3 O = 2.0f * dot(I, h) * h - I;
  bool selected = false;
  for (int m = -bound; m <= bound; m++) {
    float3 candidate;
    if (m == 0 || !diffraction_facet_reflect(I, h, axis, m * p->wavelength_over_pitch, &candidate)) {
      continue;
    }
    const float power = diffraction_binary_power(m, p->phase, p->duty);
    if (u < power) {
      O = candidate;
      mass = power;
      selected = true;
      break;
    }
    u -= power;
  }
  if (!selected) {
    mass = diffraction_thin_sheet_model_port_power(p, 0, I, h);
  }
  if (!(O.z > 0.0f && mass > 0.0f)) {
    return false;
  }
  *wo = p->transmission ? -O : O;
  if (singular) {
    *value = *pdf = mass;
  }
  else {
    *value = diffraction_thin_sheet_model_port_eval(p, wi, *wo, pdf);
  }
  return *pdf > 0.0f;
}

/* Fast reciprocal microscopic-sheet component. All travel is in exterior air.
 * Coefficients are monochromatic scalar film/Beer sheet powers, without the
 * incoming-only native GGX compensation. The native-compatible carrier is
 * blended separately; reciprocity is a property of this component only. */
struct DiffractionThinSheetModel {
  DiffractionThinSheetModelPort reflection;
  DiffractionThinSheetModelPort transmission;
  float ior;
  float reflection_tint;
  float transmission_tint;
  float film_ior;
  float film_over_wavelength;
};

ccl_device_inline float2 diffraction_thin_sheet_model_coefficients(
    ccl_private const DiffractionThinSheetModel *p, const float mu)
{
  if (!(mu > 0.0f)) {
    return make_float2(1.0f, 0.0f);
  }
  const float ct2 = 1.0f - (1.0f - mu * mu) / (p->ior * p->ior);
  if (!(ct2 > 0.0f)) {
    return make_float2(1.0f, 0.0f);
  }
  const float ct = sqrtf(ct2);
  const float f0 = sqr((p->ior - 1.0f) / (p->ior + 1.0f));
  float front, back;
  if (p->film_over_wavelength > 0.0f) {
    front = diffraction_thin_film_pair_reflectance(
        mu, ct, 1.0f, p->ior, p->film_ior, p->film_over_wavelength);
    back = diffraction_thin_film_pair_reflectance(
        ct, mu, p->ior, 1.0f, p->film_ior, p->film_over_wavelength);
    /* Match the native generalized-film tint convention in scalar form. */
    front *= mix(1.0f, p->reflection_tint, saturatef((1.0f-front)/(1.0f-f0)));
    back *= mix(1.0f, p->reflection_tint, saturatef((1.0f-back)/(1.0f-f0)));
  }
  else {
    const float rs = (mu-p->ior*ct)/(mu+p->ior*ct);
    const float rp = (p->ior*mu-ct)/(p->ior*mu+ct);
    const float f = 0.5f*(rs*rs+rp*rp);
    front = back = mix(f0*p->reflection_tint, 1.0f,
                       saturatef((f-f0)/(1.0f-f0)));
  }
  const float color = powf(p->transmission_tint, 1.0f/ct);
  const float denominator = 1.0f-sqr(back*color);
  const float T = denominator > 0.0f ? color*(1.0f-front)*(1.0f-back)/denominator : 0.0f;
  return make_float2(saturatef(front+T*back*color), saturatef(T));
}

ccl_device_inline float diffraction_thin_sheet_model_coefficient(
    ccl_private const DiffractionThinSheetModel *p, const float mu, const bool transmission)
{
  const float2 c = diffraction_thin_sheet_model_coefficients(p, mu);
  return transmission ? c.y : c.x;
}

/* Reciprocity is imposed before summing facet preimages. Nonzero powers are
 * even in order; zero-order residuals depend on the incident direction and
 * therefore require the reverse residual too. This envelope never increases
 * the uncompensated microscopic first-event budget. */
ccl_device_inline float diffraction_thin_sheet_model_power(
    ccl_private const DiffractionThinSheetModel *model,
    ccl_private const DiffractionThinSheetModelPort *p,
    const int order, const float3 I, const float3 O, const float3 h)
{
  const float forward = diffraction_thin_sheet_model_coefficient(model, I.z, p->transmission) *
                        diffraction_thin_sheet_model_port_power(p, order, I, h);
  const float reverse = diffraction_thin_sheet_model_coefficient(model, O.z, p->transmission) *
                        diffraction_thin_sheet_model_port_power(p, order, O, h);
  return min(forward, reverse);
}

ccl_device_inline float diffraction_thin_sheet_model_eval(
    ccl_private const DiffractionThinSheetModel *model, const bool transmission,
    const float3 wi, const float3 wo)
{
  ccl_private const DiffractionThinSheetModelPort *p = transmission ?
      &model->transmission : &model->reflection;
  const float3 I = transmission ? diffraction_thin_sheet_model_port_flip_tangent(wi) : wi;
  const float3 O = transmission ? -wo : wo;
  if (!(I.z>0.0f && O.z>0.0f) || roughness_is_almost_specular(p->alpha,p->alpha)) {
    return 0.0f;
  }
  const float3 axis = make_float3(1.0f,0.0f,0.0f);
  float density = 0.0f;
  const int bound = diffraction_thin_sheet_model_port_order_bound(p);
  for (int m=-bound; m<=bound; ++m) {
    DiffractionReflectionRoot roots[2];
    const float delta = m*p->wavelength_over_pitch;
    const int count = diffraction_reflection_half_vectors(I,O,axis,delta,roots);
    for (int r=0;r<count;++r) {
      const float3 h=roots[r].h;
      if (!(h.z>0.0f)) continue;
      density += diffraction_thin_sheet_model_power(model,p,m,I,O,h)*
                 bsdf_aniso_D<GGX>(p->alpha,p->alpha,h)*dot(I,h)*
                 diffraction_reflection_jacobian(I,O,h,axis,delta)/I.z;
    }
  }
  return density/(1.0f+bsdf_aniso_lambda<GGX>(p->alpha,p->alpha,I)+
                       bsdf_aniso_lambda<GGX>(p->alpha,p->alpha,O));
}

/* Golden-ratio VNDF quadrature, integrating all propagating orders at each
 * sampled facet. No root inversion is needed during cache preparation. The
 * value is E_env; q=S-E_env. Singular ports use the exact flat-facet order sum. */
ccl_device_inline float diffraction_thin_sheet_model_escape_port_sample(
    ccl_private const DiffractionThinSheetModel *model, const float3 wi,
    const int port, const int sample, const int samples)
{
  ccl_private const DiffractionThinSheetModelPort *p = port ?
      &model->transmission : &model->reflection;
  const float3 I=port ? diffraction_thin_sheet_model_port_flip_tangent(wi) : wi;
  const bool singular=roughness_is_almost_specular(p->alpha,p->alpha);
  /* A singular port is evaluated exactly once by lane zero. */
  if (singular && sample != 0) return 0.0f;
  const float li=singular ? 0.0f : bsdf_aniso_lambda<GGX>(p->alpha,p->alpha,I);
  const float3 axis=make_float3(1.0f,0.0f,0.0f);
  const float u=(sample+0.5f)/samples, v=(sample+0.5f)*0.6180339887498949f;
  const float3 h=singular ? make_float3(0.0f,0.0f,1.0f) :
      microfacet_ggx_sample_vndf(I,p->alpha,p->alpha,make_float2(u,v-floorf(v)));
  float total=0.0f;
  const int bound=diffraction_thin_sheet_model_port_order_bound(p);
  for (int m=-bound;m<=bound;++m) {
    float3 O;
    if (!diffraction_facet_reflect(I,h,axis,m*p->wavelength_over_pitch,&O) || !(O.z>0.0f)) continue;
    const float lo=singular ? 0.0f : bsdf_aniso_lambda<GGX>(p->alpha,p->alpha,O);
    total += diffraction_thin_sheet_model_power(model,p,m,I,O,h)*(1.0f+li)/(1.0f+li+lo);
  }
  return total/(singular ? 1 : samples);
}

ccl_device_inline float diffraction_thin_sheet_model_escape(
    ccl_private const DiffractionThinSheetModel *model, const float3 wi, const int samples)
{
  float sum=0.0f;
  for (int port=0;port<2;++port) {
    ccl_private const DiffractionThinSheetModelPort *p=port ? &model->transmission : &model->reflection;
    const int count=roughness_is_almost_specular(p->alpha,p->alpha) ? 1 : samples;
    for (int s=0;s<count;++s) {
      sum += diffraction_thin_sheet_model_escape_port_sample(model,wi,port,s,samples);
    }
  }
  return sum;
}

ccl_device_inline float diffraction_thin_sheet_model_relief_blend(
    const float reflection_phase, const float transmission_phase, const float duty)
{
  /* Declared Fast compatibility transition: the reciprocal component is
   * fully active at 1/32 redistributed phase power. This is a material-only
   * weight, so it does not alter component reciprocity. */
  return saturatef(32.0f*max(1.0f-diffraction_binary_power(0,reflection_phase,duty),
                            1.0f-diffraction_binary_power(0,transmission_phase,duty)));
}

CCL_NAMESPACE_END
