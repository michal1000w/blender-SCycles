/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/closure/bsdf_diffraction_util.h"
#include "kernel/closure/diffraction_thin_film.h"
#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/util/diffraction_scene.h"

CCL_NAMESPACE_BEGIN

/* Scalar, flux-normalized binary phase-screen reflection on GGX facets.
 *
 * Nonzero orders retain the scalar Fourier powers evaluated at the actual
 * incident/outgoing angles. Power not carried by these orders returns to the
 * specular order. This completion is passive and reciprocal; it is NOT an RCWA
 * solution of a metal/dielectric relief boundary. Its physical accuracy needs
 * independent reference validation before exposing it as a material model.
 *
 * The local frame has N=(0,0,1), grating axis=(1,0,0). All lengths enter as
 * ratios to wavelength. No RGB wavelength approximation is used here.
 */
struct DiffractionReflection {
  float alpha_x, alpha_y;
  float wavelength_over_pitch;
  float height_over_wavelength;
  float duty;
};

/* Fast binary-profile truncation. Each nonzero Fourier power is bounded by
 * 4/(pi*m)^2. Omitting |m|>M in both reflection and transmission therefore
 * omits at most 16/(pi*pi*M) facet power for passive interface coefficients.
 * The joint model assigns its residual to order zero; pure refraction retains
 * omitted power as null events. This is an energy bound, not a pixel-error bound.
 * Ordinary CD/DVD profiles have fewer orders and are unaffected. */
enum { DIFFRACTION_FAST_MAX_ORDER = 256 };

ccl_device_inline int diffraction_reflection_max_order(ccl_private const DiffractionReflection *p)
{
  /* A flat profile has exactly one Fourier order. Avoid tracing zero-power
   * candidates, including their half-vector roots and Fresnel evaluations. */
  if (p->height_over_wavelength == 0.0f) return 0;
  /* Cap before integer conversion, including very small positive ratios. */
  return (p->wavelength_over_pitch > 0.0f && isfinite_safe(p->wavelength_over_pitch)) ?
             int(min(2.0f / p->wavelength_over_pitch, float(DIFFRACTION_FAST_MAX_ORDER))) :
             0;
}

ccl_device_inline float diffraction_reflection_nonzero_power(
    ccl_private const DiffractionReflection *p, const int m, const float ci, const float co)
{
  return diffraction_binary_power(m, M_2PI_F * p->height_over_wavelength * (ci + co), p->duty);
}

ccl_device_inline float diffraction_reflection_zero_power(
    ccl_private const DiffractionReflection *p, const float3 wi, const float3 h)
{
  const float3 axis = make_float3(1.0f, 0.0f, 0.0f);
  const float ci = dot(wi, h);
  float other = 0.0f;
  const int limit = diffraction_reflection_max_order(p);
  for (int m = -limit; m <= limit; m++) {
    float3 wo;
    if (m != 0 && diffraction_facet_reflect(wi, h, axis, m * p->wavelength_over_pitch, &wo)) {
      other += diffraction_reflection_nonzero_power(p, m, ci, dot(wo, h));
    }
  }
  /* Each nonzero power <= 4 sin(pi*m*duty)^2/(pi*m)^2, whose two-sided
   * infinite sum is 4*duty*(1-duty)<=1. The clamp only protects roundoff. */
  return max(0.0f, 1.0f - other);
}

/* Choose an order on one facet. The residual interval samples order zero,
 * avoiding a direction-dependent renormalization that breaks reciprocity. */
ccl_device_inline bool diffraction_reflection_sample_order(
    ccl_private const DiffractionReflection *p,
    const float3 wi,
    const float3 h,
    float u,
    ccl_private float3 *wo,
    ccl_private float *mass)
{
  const float3 axis = make_float3(1.0f, 0.0f, 0.0f);
  const float ci = dot(wi, h);
  float total = 0.0f;
  const int limit = diffraction_reflection_max_order(p);
  for (int m = -limit; m <= limit; m++) {
    float3 candidate;
    if (m == 0 ||
        !diffraction_facet_reflect(wi, h, axis, m * p->wavelength_over_pitch, &candidate))
    {
      continue;
    }
    const float power = diffraction_reflection_nonzero_power(p, m, ci, dot(candidate, h));
    if (u < power) {
      *wo = candidate;
      *mass = power;
      return true;
    }
    total += power;
    u -= power;
  }
  *mass = max(0.0f, 1.0f - total);
  *wo = 2.0f * ci * h - wi;
  return ci > 0.0f && *mass > 0.0f;
}

/* Returns f*cos(theta_o); pdf is measured in outgoing solid angle. */
ccl_device_inline float diffraction_reflection_eval(ccl_private const DiffractionReflection *p,
                                                    const float3 wi,
                                                    const float3 wo,
                                                    ccl_private float *pdf)
{
  *pdf = 0.0f;
  if (!(wi.z > 0.0f && wo.z > 0.0f) || roughness_is_almost_specular(p->alpha_x, p->alpha_y)) {
    return 0.0f;
  }
  const float lambda_i = bsdf_aniso_lambda<GGX>(p->alpha_x, p->alpha_y, wi);
  const float lambda_o = bsdf_aniso_lambda<GGX>(p->alpha_x, p->alpha_y, wo);
  const float3 axis = make_float3(1.0f, 0.0f, 0.0f);
  float density = 0.0f;
  const int limit = diffraction_reflection_max_order(p);
  for (int m = -limit; m <= limit; m++) {
    const float delta = m * p->wavelength_over_pitch;
    DiffractionReflectionRoot roots[2];
    const int count = diffraction_reflection_half_vectors(wi, wo, axis, delta, roots);
    for (int r = 0; r < count; r++) {
      const float3 h = roots[r].h;
      if (!(h.z > 0.0f)) {
        continue;
      }
      const float ci = dot(wi, h), co = dot(wo, h);
      const float power = m == 0 ? diffraction_reflection_zero_power(p, wi, h) :
                                   diffraction_reflection_nonzero_power(p, m, ci, co);
      const float J = diffraction_reflection_jacobian(wi, wo, h, axis, delta);
      density += power * bsdf_aniso_D<GGX>(p->alpha_x, p->alpha_y, h) * ci * J / wi.z;
    }
  }
  *pdf = density / (1.0f + lambda_i);
  return density / (1.0f + lambda_i + lambda_o);
}

ccl_device_inline bool diffraction_reflection_sample(ccl_private const DiffractionReflection *p,
                                                     const float3 wi,
                                                     const float3 rand,
                                                     ccl_private float3 *wo,
                                                     ccl_private float *eval,
                                                     ccl_private float *pdf)
{
  *eval = *pdf = 0.0f;
  *wo = zero_float3();
  if (!(wi.z > 0.0f)) {
    return false;
  }
  const bool singular = roughness_is_almost_specular(p->alpha_x, p->alpha_y);
  const float3 h = singular ?
                       make_float3(0.0f, 0.0f, 1.0f) :
                       microfacet_ggx_sample_vndf(wi, p->alpha_x, p->alpha_y, make_float2(rand));
  float mass;
  if (!diffraction_reflection_sample_order(p, wi, h, rand.z, wo, &mass) || !(wo->z > 0.0f)) {
    return false;
  }
  if (singular) {
    *eval = *pdf = mass;
  }
  else {
    *eval = diffraction_reflection_eval(p, wi, *wo, pdf);
  }
  return *pdf > 0.0f;
}

/* Direction density for one transmitted diffraction order on visible GGX
 * facets. eta=n_in/n_out; delta=m*lambda_vacuum/(n_out*pitch). This is a
 * geometry distribution, not an efficiency model or complete glass BSDF.
 * Rejected facets retain their probability as null events. The order weight
 * must be included by the caller when forming a mixture of diffraction orders.
 * Index-matched zero-order transmission is a delta event handled separately. */
ccl_device_inline float diffraction_transmission_ggx_order_pdf(
    const float alpha_x, const float alpha_y, const float3 wi, const float3 wo,
    const float eta, const float delta)
{
  if (!(wi.z > 0.0f && wo.z < 0.0f && alpha_x > 0.0f && alpha_y > 0.0f) ||
      roughness_is_almost_specular(alpha_x, alpha_y) || (eta == 1.0f && delta == 0.0f))
    return 0.0f;
  const float3 axis = make_float3(1.0f, 0.0f, 0.0f);
  DiffractionTransmissionRoot roots[2];
  const int count = diffraction_transmission_half_vectors(wi, wo, axis, eta, delta, roots);
  float density = 0.0f;
  for (int r = 0; r < count; ++r) {
    const float3 h = roots[r].h;
    if (h.z > 0.0f) {
      density += bsdf_aniso_D<GGX>(alpha_x, alpha_y, h) * dot(wi, h) *
                 diffraction_transmission_jacobian(wi, wo, h, axis, eta, delta);
    }
  }
  return density / (wi.z * (1.0f + bsdf_aniso_lambda<GGX>(alpha_x, alpha_y, wi)));
}

ccl_device_inline bool diffraction_transmission_ggx_sample_order(
    const float alpha_x, const float alpha_y, const float3 wi, const float2 rand,
    const float eta, const float delta, ccl_private float3 *wo, ccl_private float *pdf)
{
  *wo = zero_float3();
  *pdf = 0.0f;
  if (!(wi.z > 0.0f && alpha_x > 0.0f && alpha_y > 0.0f) ||
      roughness_is_almost_specular(alpha_x, alpha_y) || (eta == 1.0f && delta == 0.0f))
    return false;
  const float3 h = microfacet_ggx_sample_vndf(wi, alpha_x, alpha_y, rand);
  if (!diffraction_facet_transmit(wi, h, make_float3(1.0f, 0.0f, 0.0f), eta, delta, wo) ||
      !(wo->z < 0.0f))
    return false;
  *pdf = diffraction_transmission_ggx_order_pdf(alpha_x, alpha_y, wi, *wo, eta, delta);
  return *pdf > 0.0f;
}

/* Fast rough-dielectric facet coefficients. This reciprocal scalar
 * approximation uses conservative, pairwise Fresnel budgets; it is not a
 * Maxwell solution. In particular it does not model diffraction-assisted
 * transmission beyond the ordinary Fresnel critical angle. A flat profile
 * reduces to the ordinary lossless dielectric Fresnel interface.
 * wavelength_over_pitch is vacuum wavelength/(n_in*pitch), and
 * height_over_wavelength is height*n_in/vacuum wavelength. */
struct DiffractionDielectricFacet {
  float wavelength_over_pitch;
  float height_over_wavelength;
  float duty;
  float incident_ior, transmitted_ior;
  float transmission_phase;
};

ccl_device_inline float diffraction_dielectric_facet_power(
    ccl_private const DiffractionDielectricFacet *p, const int order,
    const bool transmission, const float ci, const float co,
    const float film_ior=1.0f, const float film_thickness_over_wavelength=0.0f,
    const float incident_reflectance=-1.0f, const bool pure_refraction=false)
{
  if (!(ci > 0.0f && (p->incident_ior / p->transmitted_ior) > 0.0f)) return 0.0f;
  /* Refraction BSDF is a transmission building block without Fresnel weighting.
   * Keep the Fourier probabilities subnormalized when orders do not propagate;
   * renormalizing that lost power would break forward/reverse symmetry. */
  if (pure_refraction) {
    return transmission && co<0 ?
        diffraction_binary_power(order,p->transmission_phase,p->duty) : 0.0f;
  }
  /* A lossless planar coating supplies the reciprocal interface budgets.
   * This remains the Fast scalar relief approximation, not a corrugated-film
   * Maxwell solution. Thickness is measured in vacuum wavelengths. */
  const bool coated=film_thickness_over_wavelength>0.0f;
  const float Fi = incident_reflectance >= 0.0f ? incident_reflectance : coated ? diffraction_thin_film_reflectance(
      ci,p->incident_ior,p->transmitted_ior,film_ior,film_thickness_over_wavelength) :
      fresnel_dielectric(ci, (p->transmitted_ior / p->incident_ior), nullptr);
  if (transmission) {
    if (!(co < 0.0f)) return 0.0f;
    /* Use both known cosines for the Snell pair. Reconstructing either
     * cosine through a second Snell evaluation amplifies rounding near the
     * critical angle and makes the forward/reverse powers disagree. */
    if (order == 0) {
      const float rs = ((p->incident_ior / p->transmitted_ior)*ci+co)/((p->incident_ior / p->transmitted_ior)*ci-co);
      const float rp = (ci+(p->incident_ior / p->transmitted_ior)*co)/(ci-(p->incident_ior / p->transmitted_ior)*co);
      const float pair_fresnel = coated ? diffraction_thin_film_pair_reflectance(
          ci,-co,p->incident_ior,p->transmitted_ior,film_ior,film_thickness_over_wavelength) :
          0.5f*(rs*rs+rp*rp);
      return (1.0f-pair_fresnel) *
             diffraction_binary_power(order, p->transmission_phase, p->duty);
    }
    const float Fo = coated ? diffraction_thin_film_reflectance(
        -co,p->transmitted_ior,p->incident_ior,film_ior,film_thickness_over_wavelength) :
        fresnel_dielectric(-co, (p->incident_ior / p->transmitted_ior), nullptr);
    /* min preserves source/receiver symmetry and bounds the sum of all
     * transmitted Fourier powers by 1-Fi for every incident direction. */
    return min(1.0f-Fi, 1.0f-Fo) *
           diffraction_binary_power(order, p->transmission_phase, p->duty);
  }
  if (!(co > 0.0f) || order == 0) return 0.0f; /* Residual mirror below. */
  const float Fo = coated ? diffraction_thin_film_reflectance(
      co,p->incident_ior,p->transmitted_ior,film_ior,film_thickness_over_wavelength) :
      fresnel_dielectric(co, (p->transmitted_ior / p->incident_ior), nullptr);
  return min(Fi,Fo) * diffraction_binary_power(
      order, M_2PI_F * p->height_over_wavelength * (ci+co), p->duty);
}

ccl_device_inline float diffraction_dielectric_facet_nonstraight_budget(
    ccl_private const DiffractionDielectricFacet *p, const float ci,
    const float film_ior=1.0f, const float film_thickness_over_wavelength=0.0f)
{
  if (p->incident_ior!=p->transmitted_ior) return 1.0f;
  const float straight=diffraction_binary_power(0,p->transmission_phase,p->duty);
  const float film_reflection=film_thickness_over_wavelength>0 ?
      diffraction_thin_film_pair_reflectance(ci,ci,p->incident_ior,p->transmitted_ior,
                                           film_ior,film_thickness_over_wavelength):0.0f;
  return (1.0f-straight)+film_reflection*straight;
}

ccl_device_inline int diffraction_dielectric_facet_order_bound(
    ccl_private const DiffractionDielectricFacet *p, const bool transmission)
{
  if (!((p->incident_ior / p->transmitted_ior) > 0.0f && p->wavelength_over_pitch > 0.0f)) return 0;
  const float bound = ceilf((transmission ? 1.0f + (p->transmitted_ior / p->incident_ior) : 2.0f) /
                            p->wavelength_over_pitch);
  /* Large valid pitches use the bounded Fast tail approximation instead of
   * losing their entire closure. Malformed geometry still fails setup. */
  return isfinite_safe(bound) && bound > 0.0f ?
             int(min(bound, float(DIFFRACTION_FAST_MAX_ORDER))) : 0;
}

/* Keep geometric range validation separate from the active Fourier support:
 * zero is a valid active bound but denotes invalid geometry in setup checks. */
ccl_device_inline int diffraction_dielectric_facet_active_order_bound(
    ccl_private const DiffractionDielectricFacet *p, const bool transmission)
{
  if (transmission ? p->transmission_phase == 0.0f : p->height_over_wavelength == 0.0f)
    return 0;
  return diffraction_dielectric_facet_order_bound(p, transmission);
}

ccl_device_inline float diffraction_dielectric_facet_residual(
    ccl_private const DiffractionDielectricFacet *p, const float3 wi,
    const float3 h, const float3 axis,
    const float film_ior=1.0f, const float film_thickness_over_wavelength=0.0f)
{
  /* Keep the dominant straight-through atom out of the accumulation.
   * The conditional closure divides this residual by the remaining budget;
   * adding a near-unit atom first would amplify rounding by 1 / budget. */
  const bool matched = p->incident_ior == p->transmitted_ior;
  const float ci = dot(wi,h);
  const float budget=diffraction_dielectric_facet_nonstraight_budget(
      p,ci,film_ior,film_thickness_over_wavelength);
  /* Every order shares the incident facet: evaluate its coating once. */
  const float Fi = film_thickness_over_wavelength > 0.0f ?
      diffraction_thin_film_reflectance(ci, p->incident_ior, p->transmitted_ior,
                                      film_ior, film_thickness_over_wavelength) :
      fresnel_dielectric(ci, p->transmitted_ior / p->incident_ior, nullptr);
  float other = 0.0f;
  for (int side=0; side<2; ++side) {
    const bool transmission = side != 0;
    const int bound = diffraction_dielectric_facet_active_order_bound(p,transmission);
    for (int m=-bound; m<=bound; ++m) {
      if (m==0 && (!transmission || matched)) continue;
      float3 wo;
      const float delta = m*p->wavelength_over_pitch;
      const bool valid = transmission ?
          diffraction_facet_transmit(wi,h,axis,(p->incident_ior / p->transmitted_ior),delta*(p->incident_ior / p->transmitted_ior),&wo) :
          diffraction_facet_reflect(wi,h,axis,delta,&wo);
      if (valid) other += diffraction_dielectric_facet_power(
          p,m,transmission,ci,dot(wo,h),film_ior,film_thickness_over_wavelength,Fi);
    }
  }
  /* Nonzero reflected powers sum to at most Fi, transmitted powers to
   * at most 1-Fi. Do not conceal an invalid negative residual in tests. */
  return budget-other;
}

struct DiffractionBsdf {
  SHADER_CLOSURE_BASE;
  DiffractionReflection param;
  float3 T;
};

static_assert(sizeof(ShaderClosure) >= sizeof(DiffractionBsdf), "DiffractionBsdf is too large!");

ccl_device_inline int bsdf_diffraction_label(ccl_private const DiffractionBsdf *bsdf)
{
  return LABEL_REFLECT |
         (roughness_is_almost_specular(bsdf->param.alpha_x, bsdf->param.alpha_y) ? LABEL_SINGULAR :
                                                                                   LABEL_GLOSSY);
}

ccl_device_inline Spectrum bsdf_diffraction_eval(ccl_private const ShaderClosure *sc,
                                                 const float3 wi,
                                                 const float3 wo,
                                                 ccl_private float *pdf)
{
  const ccl_private DiffractionBsdf *bsdf = (const ccl_private DiffractionBsdf *)sc;
  float3 X, Y;
  make_orthonormals_safe_tangent(bsdf->N, bsdf->T, &X, &Y);
  return make_spectrum(
      diffraction_reflection_eval(&bsdf->param,
                                  make_float3(dot(wi, X), dot(wi, Y), dot(wi, bsdf->N)),
                                  make_float3(dot(wo, X), dot(wo, Y), dot(wo, bsdf->N)),
                                  pdf));
}

ccl_device_inline int bsdf_diffraction_sample(ccl_private const ShaderClosure *sc,
                                              const float3 Ng,
                                              const float3 wi,
                                              const float3 rand,
                                              ccl_private Spectrum *eval,
                                              ccl_private float3 *wo,
                                              ccl_private float *pdf,
                                              ccl_private float2 *roughness,
                                              ccl_private float *eta)
{
  const ccl_private DiffractionBsdf *bsdf = (const ccl_private DiffractionBsdf *)sc;
  float3 X, Y;
  make_orthonormals_safe_tangent(bsdf->N, bsdf->T, &X, &Y);
  const float3 local_wi = make_float3(dot(wi, X), dot(wi, Y), dot(wi, bsdf->N));
  float3 local_wo;
  float value;
  *eval = zero_spectrum();
  *roughness = make_float2(bsdf->param.alpha_x, bsdf->param.alpha_y);
  *eta = 1.0f;
  if (!diffraction_reflection_sample(&bsdf->param, local_wi, rand, &local_wo, &value, pdf)) {
    return LABEL_NONE;
  }
  *wo = to_global(local_wo, X, Y, bsdf->N);
  if (!(dot(Ng, *wo) > 0.0f)) {
    *pdf = 0.0f;
    return LABEL_NONE;
  }
  *eval = make_spectrum(value);
  const int label = bsdf_diffraction_label(bsdf);
  if (label & LABEL_SINGULAR) {
    /* Match Cycles' discrete-event convention when this closure is mixed with
     * continuous lobes. The mathematical sampler above retains true masses. */
    *pdf *= 1e6f;
    *eval *= 1e6f;
  }
  return label;
}

ccl_device_inline void bsdf_diffraction_setup(KernelGlobals kg,
                                              ccl_private ShaderData *sd,
                                              const Spectrum weight,
                                              const float3 N,
                                              const float3 T,
                                              const float alpha_x,
                                              const float alpha_y,
                                              const float pitch_nm,
                                              const float depth_nm,
                                              const float duty,
                                            const float medium_ior)
{
#ifdef __SPECTRAL__
  if (!(pitch_nm > 0.0f && pitch_nm <= 1000000.0f && depth_nm >= 0.0f) || !isfinite_safe(depth_nm) || !(medium_ior > 0.0f) || !isfinite_safe(medium_ior))
  {
    return;
  }
  /* The existing bounded RGB-to-spectrum basis is also suitable for a passive
   * reflectance. Activate wavelength transport for achromatic gratings too. */
  const Spectrum spectral_weight = bsdf_spectral_transmission_color(kg, sd, weight);
  ccl_private DiffractionBsdf *bsdf = (ccl_private DiffractionBsdf *)bsdf_alloc(
      sd, sizeof(DiffractionBsdf), spectral_weight);
  if (!bsdf) {
    return;
  }
  bsdf->type = CLOSURE_BSDF_DIFFRACTION_ID;
  bsdf->N = N;
  bsdf->T = T;
  /* Node index is the incident medium at the grating, not the substrate.
   * Cycles does not infer an optical medium index from its volume stack. */
  const float wavelength_nm = 1000.0f * sample_wavelength(sd->rand_wavelength) / medium_ior;
  bsdf->param = {saturatef(alpha_x),
                 saturatef(alpha_y),
                 wavelength_nm / pitch_nm,
                 depth_nm / wavelength_nm,
                 saturatef(duty)};
  sd->runtime_flag |= SR_BSDF | SR_BSDF_HAS_DISPERSION;
  if (!roughness_is_almost_specular(bsdf->param.alpha_x, bsdf->param.alpha_y)) {
    sd->runtime_flag |= SR_BSDF_HAS_EVAL;
  }
#endif
}

CCL_NAMESPACE_END
