/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "kernel/closure/bsdf_ashikhmin_shirley.h"
#include "kernel/closure/bsdf_diffraction.h"

CCL_NAMESPACE_BEGIN

/* Ashikhmin-Shirley samples the upper-hemisphere half-vector distribution
 * p_h = sqrt((n_x+1)(n_y+1))/(2*pi) * h.z^e, not a visible-normal NDF.
 * These helpers use the grating tangent as local X. */
ccl_device_inline float diffraction_ashikhmin_half_vector_pdf(
    const float alpha_x, const float alpha_y, const float3 h)
{
  if (!(h.z > 0.0f)) return 0.0f;
  const float nx = bsdf_ashikhmin_shirley_roughness_to_exponent(alpha_x);
  const float ny = bsdf_ashikhmin_shirley_roughness_to_exponent(alpha_y);
  float exponent = nx;
  if (nx != ny && h.z < 1.0f) {
    exponent = (nx * h.x * h.x + ny * h.y * h.y) /
               max(1e-12f, 1.0f - h.z * h.z);
  }
  return sqrtf((nx + 1.0f) * (ny + 1.0f)) * powf(h.z, exponent) *
         (0.5f * M_1_PI_F);
}

ccl_device_inline float3 diffraction_ashikhmin_sample_half_vector(
    const float alpha_x, const float alpha_y, float2 random)
{
  const float nx = bsdf_ashikhmin_shirley_roughness_to_exponent(alpha_x);
  const float ny = bsdf_ashikhmin_shirley_roughness_to_exponent(alpha_y);
  float phi, cos_theta;
  if (nx == ny) {
    phi = M_2PI_F * random.x;
    cos_theta = powf(random.y, 1.0f / (nx + 1.0f));
  }
  else if (random.x < .25f) {
    random.x *= 4.0f;
    bsdf_ashikhmin_shirley_sample_first_quadrant(nx, ny, random, &phi, &cos_theta);
  }
  else if (random.x < .5f) {
    random.x = 4.0f * (.5f - random.x);
    bsdf_ashikhmin_shirley_sample_first_quadrant(nx, ny, random, &phi, &cos_theta);
    phi = M_PI_F - phi;
  }
  else if (random.x < .75f) {
    random.x = 4.0f * (random.x - .5f);
    bsdf_ashikhmin_shirley_sample_first_quadrant(nx, ny, random, &phi, &cos_theta);
    phi = M_PI_F + phi;
  }
  else {
    random.x = 4.0f * (1.0f - random.x);
    bsdf_ashikhmin_shirley_sample_first_quadrant(nx, ny, random, &phi, &cos_theta);
    phi = M_2PI_F - phi;
  }
  return spherical_cos_to_direction(cos_theta, phi);
}

/* In the native zero-order model q = p_h/(4*c_i) and
 * f*cos_o = q * mu_o/max(mu_i,mu_o). For nonzero diffraction orders,
 * J_forward=c_o/det and J_reverse=c_i/det. Multiplying the native pump
 * by min(c_i,c_o)/c_o makes f reciprocal and keeps f*cos/q <= 1.
 * Rejected orders/facets remain null proposal events. */
ccl_device_inline float diffraction_ashikhmin_eval(
    ccl_private const DiffractionReflection *p, const float3 wi, const float3 wo,
    ccl_private float *pdf)
{
  *pdf = 0.0f;
  if (!(wi.z > 0.0f && wo.z > 0.0f) ||
      fmaxf(p->alpha_x, p->alpha_y) <= 1e-4f) return 0.0f;
  float value = 0.0f;
  const float max_mu = max(wi.z, wo.z);
  const float3 axis = make_float3(1.0f, 0.0f, 0.0f);
  const int bound = diffraction_reflection_max_order(p);
  for (int m = -bound; m <= bound; ++m) {
    const float delta = m * p->wavelength_over_pitch;
    DiffractionReflectionRoot roots[2];
    const int count = diffraction_reflection_half_vectors(wi, wo, axis, delta, roots);
    for (int r = 0; r < count; ++r) {
      const float3 h = roots[r].h;
      const float ci = dot(wi, h), co = dot(wo, h);
      if (!(h.z > 0.0f && ci > 0.0f && co > 0.0f)) continue;
      const float power = m == 0 ? diffraction_reflection_zero_power(p, wi, h) :
                                   diffraction_reflection_nonzero_power(p, m, ci, co);
      const float J = diffraction_reflection_jacobian(wi, wo, h, axis, delta);
      const float q = diffraction_ashikhmin_half_vector_pdf(p->alpha_x, p->alpha_y, h) *
                      power * J;
      *pdf += q;
      value += q * (wo.z / max_mu) * (min(ci, co) / co);
    }
  }
  return value;
}

ccl_device_inline bool diffraction_ashikhmin_sample_direction(
    ccl_private const DiffractionReflection *p, const float3 wi,
    const float3 random, ccl_private float3 *wo)
{
  *wo = zero_float3();
  if (!(wi.z > 0.0f)) return false;
  const float3 h = diffraction_ashikhmin_sample_half_vector(
      p->alpha_x, p->alpha_y, make_float2(random));
  if (!(dot(wi, h) > 0.0f)) return false;
  float mass;
  return diffraction_reflection_sample_order(p, wi, h, random.z, wo, &mass) &&
         wo->z > 0.0f;
}

ccl_device_inline Spectrum bsdf_diffraction_ashikhmin_eval(
    ccl_private const ShaderClosure *sc, const float3 wi, const float3 wo,
    ccl_private float *pdf)
{
  const ccl_private DiffractionBsdf *b = (const ccl_private DiffractionBsdf *)sc;
  float3 X, Y;
  make_orthonormals_safe_tangent(b->N, b->T, &X, &Y);
  const float3 I = make_float3(dot(wi, X), dot(wi, Y), dot(wi, b->N));
  const float3 O = make_float3(dot(wo, X), dot(wo, Y), dot(wo, b->N));
  return make_spectrum(diffraction_ashikhmin_eval(&b->param, I, O, pdf));
}

ccl_device_inline int bsdf_diffraction_ashikhmin_sample(
    ccl_private const ShaderClosure *sc, const float3 Ng, const float3 wi,
    const float3 random, ccl_private Spectrum *eval, ccl_private float3 *wo,
    ccl_private float *pdf, ccl_private float2 *roughness,
    ccl_private float *eta)
{
  const ccl_private DiffractionBsdf *b = (const ccl_private DiffractionBsdf *)sc;
  *eval = zero_spectrum();
  *pdf = 0.0f;
  *roughness = make_float2(b->param.alpha_x, b->param.alpha_y);
  *eta = 1.0f;
  float3 X, Y;
  make_orthonormals_safe_tangent(b->N, b->T, &X, &Y);
  const float3 I = make_float3(dot(wi, X), dot(wi, Y), dot(wi, b->N));
  const bool singular = max(b->param.alpha_x, b->param.alpha_y) <= 1e-4f;
  const float3 h = singular ? make_float3(0, 0, 1) :
      diffraction_ashikhmin_sample_half_vector(
          b->param.alpha_x, b->param.alpha_y, make_float2(random));
  float3 O;
  float mass;
  if (!(I.z > 0.0f && dot(I, h) > 0.0f) ||
      !diffraction_reflection_sample_order(&b->param, I, h, random.z, &O, &mass) ||
      !(O.z > 0.0f)) return LABEL_NONE;
  *wo = O.x * X + O.y * Y + O.z * b->N;
  if (!(dot(Ng, *wo) > 0.0f)) return LABEL_NONE;
  if (singular) {
    *pdf = mass * 1e6f;
    *eval = make_spectrum(*pdf);
    *roughness = zero_float2();
    return LABEL_REFLECT | LABEL_SINGULAR;
  }
  *eval = bsdf_diffraction_ashikhmin_eval(sc, wi, *wo, pdf);
  return *pdf > 0.0f ? LABEL_REFLECT | LABEL_GLOSSY : LABEL_NONE;
}

ccl_device_inline Spectrum bsdf_diffraction_ashikhmin_delta(
    ccl_private const ShaderClosure *sc, const float3 wi, const float3 wo,
    ccl_private float *pdf)
{
  *pdf = 0.0f;
  const ccl_private DiffractionBsdf *b = (const ccl_private DiffractionBsdf *)sc;
  const ccl_private DiffractionReflection *p = &b->param;
  if (max(p->alpha_x, p->alpha_y) > 1e-4f) return zero_spectrum();
  float3 X, Y;
  make_orthonormals_safe_tangent(b->N, b->T, &X, &Y);
  const float3 I = make_float3(dot(wi, X), dot(wi, Y), dot(wi, b->N));
  const float3 O = make_float3(dot(wo, X), dot(wo, Y), dot(wo, b->N));
  if (!(I.z > 0.0f && O.z > 0.0f)) return zero_spectrum();
  const float order = roundf((I.x + O.x) / p->wavelength_over_pitch);
  if (fabsf(order) > diffraction_reflection_max_order(p) ||
      sqr(I.x + O.x - order * p->wavelength_over_pitch) + sqr(I.y + O.y) > 16e-12f)
    return zero_spectrum();
  const float power = order == 0 ? diffraction_reflection_zero_power(
      p, I, make_float3(0, 0, 1)) :
      diffraction_reflection_nonzero_power(p, int(order), I.z, O.z);
  *pdf = power * 1e6f;
  return make_spectrum(*pdf);
}

/* Flat relief is an actual native Ashikhmin closure. In particular this
 * retains its original RGB/spectral weight and native sampling behavior. */
ccl_device_inline bool bsdf_diffraction_ashikhmin_glossy_setup(
    ccl_private ShaderData *sd, const Spectrum weight, const float3 N,
    const float3 T, const float alpha_x, const float alpha_y,
    const float pitch, const float depth, const float duty,
    const float medium_ior)
{
  if (!(pitch > 0.0f && pitch <= 1000000.0f && depth >= 0.0f && medium_ior > 0.0f) ||
      !isfinite_safe(depth) || !isfinite_safe(medium_ior)) return false;
  if (depth == 0.0f) {
    ccl_private MicrofacetBsdf *b = (ccl_private MicrofacetBsdf *)bsdf_alloc(
        sd, sizeof(MicrofacetBsdf), weight);
    if (!b) return false;
    b->N = N;
    b->T = T;
    b->ior = 1.0f;
    b->alpha_x = alpha_x;
    b->alpha_y = alpha_y;
    sd->runtime_flag |= bsdf_ashikhmin_shirley_setup(b);
    return true;
  }
  const float wavelength = 1000.0f * sample_wavelength(sd->rand_wavelength) / medium_ior;
  const DiffractionReflection p = {clamp(alpha_x, 1e-4f, 1.0f),
                                   clamp(alpha_y, 1e-4f, 1.0f),
                                   wavelength / pitch, depth / wavelength,
                                   saturatef(duty)};
  if (!(p.wavelength_over_pitch > 0.0f) || !isfinite_safe(p.wavelength_over_pitch))
    return false;
  ccl_private DiffractionBsdf *b = (ccl_private DiffractionBsdf *)bsdf_alloc(
      sd, sizeof(DiffractionBsdf), weight);
  if (!b) return false;
  b->N = N;
  b->T = T;
  b->param = p;
  b->type = CLOSURE_BSDF_DIFFRACTION_ASHIKHMIN_ID;
  sd->runtime_flag |= SR_BSDF | SR_BSDF_HAS_EVAL | SR_BSDF_HAS_DISPERSION;
  return true;
}

CCL_NAMESPACE_END
