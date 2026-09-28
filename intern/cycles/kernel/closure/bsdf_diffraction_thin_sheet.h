/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/closure/bsdf_diffraction.h"
#include "kernel/closure/diffraction_thin_sheet_model.h"
#include "kernel/util/diffraction_albedo.h"

CCL_NAMESPACE_BEGIN

/* Fast scalar phase screen on a two-sided sheet. Each port retains the native
 * thin-glass spectral sheet coefficient; only its angular distribution is
 * changed. Transmission uses the native double-refraction approximation:
 * reflect the incident direction in the sheet plane, sample a GGX reflection,
 * then negate its output. This makes order zero exactly the native thin-glass
 * rough transmission carrier, including its special smooth straight ray.
 *
 * The relief phases are constant over a facet. This is a local phase-screen
 * approximation, not a solution of the corrugated two-interface boundary:
 * reflection 4*pi*n_exterior*depth/lambda, transmission
 * 2*pi*(n_sheet-n_exterior)*depth/lambda. At zero depth only order zero remains.
 * Missing/evanescent order power is returned to zero order. The legacy
 * incoming Fresnel/compensation carrier is native-compatible and approximate.
 * Cached mode adds a separate conservative microscopic reciprocal component
 * plus a finite-interpolant return, blended continuously by relief power. */
struct DiffractionThinSheetPort {
  float alpha;
  float wavelength_over_pitch;
  float phase;
  float duty;
  bool transmission;
};

ccl_device_inline float3 diffraction_thin_sheet_flip_tangent(const float3 w)
{
  return make_float3(-w.x, -w.y, w.z);
}

ccl_device_inline int diffraction_thin_sheet_order_bound(ccl_private const DiffractionThinSheetPort *p)
{
  if (p->phase == 0.0f || !(p->wavelength_over_pitch > 0.0f)) {
    return 0;
  }
  return int(min(2.0f / p->wavelength_over_pitch, float(DIFFRACTION_FAST_MAX_ORDER)));
}

ccl_device_inline float diffraction_thin_sheet_power(ccl_private const DiffractionThinSheetPort *p,
                                                       const int order,
                                                       const float3 incident,
                                                       const float3 h)
{
  if (order != 0) {
    return diffraction_binary_power(order, p->phase, p->duty);
  }
  const float3 axis = make_float3(1.0f, 0.0f, 0.0f);
  const int bound = diffraction_thin_sheet_order_bound(p);
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
ccl_device_inline float diffraction_thin_sheet_eval(ccl_private const DiffractionThinSheetPort *p,
                                                     const float3 wi,
                                                     const float3 wo,
                                                     ccl_private float *pdf)
{
  *pdf = 0.0f;
  const float3 I = p->transmission ? diffraction_thin_sheet_flip_tangent(wi) : wi;
  const float3 O = p->transmission ? -wo : wo;
  if (!(I.z > 0.0f && O.z > 0.0f) || roughness_is_almost_specular(p->alpha, p->alpha)) {
    return 0.0f;
  }
  const float3 axis = make_float3(1.0f, 0.0f, 0.0f);
  float density = 0.0f;
  const int bound = diffraction_thin_sheet_order_bound(p);
  for (int m = -bound; m <= bound; m++) {
    const float delta = m * p->wavelength_over_pitch;
    DiffractionReflectionRoot roots[2];
    const int count = diffraction_reflection_half_vectors(I, O, axis, delta, roots);
    for (int r = 0; r < count; r++) {
      const float3 h = roots[r].h;
      if (!(h.z > 0.0f)) {
        continue;
      }
      const float power = diffraction_thin_sheet_power(p, m, I, h);
      const float J = diffraction_reflection_jacobian(I, O, h, axis, delta);
      density += power * bsdf_aniso_D<GGX>(p->alpha, p->alpha, h) * dot(I, h) * J / I.z;
    }
  }
  const float li = bsdf_aniso_lambda<GGX>(p->alpha, p->alpha, I);
  const float lo = bsdf_aniso_lambda<GGX>(p->alpha, p->alpha, O);
  *pdf = density / (1.0f + li);
  return density / (1.0f + li + lo);
}

ccl_device_inline bool diffraction_thin_sheet_sample(ccl_private const DiffractionThinSheetPort *p,
                                                       const float3 wi,
                                                       const float3 random,
                                                       ccl_private float3 *wo,
                                                       ccl_private float *value,
                                                       ccl_private float *pdf)
{
  *wo = zero_float3();
  *value = *pdf = 0.0f;
  const float3 I = p->transmission ? diffraction_thin_sheet_flip_tangent(wi) : wi;
  if (!(I.z > 0.0f)) {
    return false;
  }
  const bool singular = roughness_is_almost_specular(p->alpha, p->alpha);
  const float3 h = singular ? make_float3(0.0f, 0.0f, 1.0f) :
                              microfacet_ggx_sample_vndf(I, p->alpha, p->alpha, make_float2(random));
  const float3 axis = make_float3(1.0f, 0.0f, 0.0f);
  const int bound = diffraction_thin_sheet_order_bound(p);
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
    mass = diffraction_thin_sheet_power(p, 0, I, h);
  }
  if (!(O.z > 0.0f && mass > 0.0f)) {
    return false;
  }
  *wo = p->transmission ? -O : O;
  if (singular) {
    *value = *pdf = mass;
  }
  else {
    *value = diffraction_thin_sheet_eval(p, wi, *wo, pdf);
  }
  return *pdf > 0.0f;
}

struct DiffractionThinSheetExtra {
  DiffractionThinSheetModel model;
  FresnelCoeff tint;
  Spectrum native_reflection;
  Spectrum native_transmission;
  int3 albedo_handles;
  float wavelength_nm;
  float blend;
};

struct DiffractionThinSheetBsdf {
  SHADER_CLOSURE_BASE;
  packed_float3 T;
  DiffractionThinSheetPort port;
  ccl_private const DiffractionThinSheetExtra *extra;
  bool return_only;
};
static_assert(sizeof(DiffractionThinSheetBsdf) <= sizeof(ShaderClosure));

/* Native thin glass applies planar GGX energy compensation to both of its
 * reflection-like carriers. Match that factor exactly in the flat limit.
 * Applying it to nonzero orders is an explicit approximation: planar GGX
 * albedo tables do not know the grating's order-dependent masking. */
ccl_device_inline Spectrum diffraction_thin_sheet_native_compensation(
    KernelGlobals kg, ccl_private ShaderData *sd, const float3 N, const float alpha,
    const Spectrum tint, const bool transmission)
{
  MicrofacetBsdf carrier;
  carrier.N = transmission ? -N : N;
  carrier.alpha_x = carrier.alpha_y = alpha;
  carrier.type = transmission ? CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID :
                                CLOSURE_BSDF_MICROFACET_GGX_ID;
  carrier.weight = one_spectrum();
  carrier.sample_weight = 1.0f;
  microfacet_ggx_preserve_energy(kg, &carrier,
                                 transmission ? reflect(sd->wi, N) : sd->wi,
                                 transmission ? one_spectrum() : tint);
  return carrier.weight * carrier.energy_scale;
}

ccl_device_inline bool bsdf_diffraction_thin_sheet_setup(ccl_private ShaderData *sd,
                                                          const Spectrum weight,
                                                          const float3 N,
                                                          const float3 tangent,
                                                          const float alpha,
                                                          const float pitch_nm,
                                                          const float depth_nm,
                                                          const float duty,
                                                          const float sheet_ior,
                                                          const bool transmission)
{
  if (is_zero(weight)) {
    return true;
  }
  const float wavelength_nm = 1000.0f * sample_wavelength(sd->rand_wavelength);
  if (!(wavelength_nm > 0.0f && pitch_nm > 0.0f && depth_nm >= 0.0f &&
        sheet_ior > 0.0f && duty >= 0.0f && duty <= 1.0f && alpha >= 0.0f && alpha <= 1.0f))
  {
    return false;
  }
  const float phase = M_2PI_F * depth_nm / wavelength_nm *
                      (transmission ? (sheet_ior - 1.0f) : 2.0f);
  if (!isfinite_safe(phase)) {
    return false;
  }
  ccl_private DiffractionThinSheetBsdf *bsdf =
      (ccl_private DiffractionThinSheetBsdf *)bsdf_alloc(sd, sizeof(DiffractionThinSheetBsdf), weight);
  if (bsdf == nullptr) {
    return false;
  }
  bsdf->N = safe_normalize_fallback(N, sd->N);
  float3 T = tangent - dot(tangent, bsdf->N) * bsdf->N;
  if (len_squared(T) < 1e-12f) {
    float3 unused;
    make_orthonormals(bsdf->N, &T, &unused);
  }
  bsdf->T = safe_normalize(T);
  bsdf->port = {alpha, wavelength_nm / pitch_nm, phase, duty, transmission};
  bsdf->extra = nullptr;
  bsdf->return_only = false;
  bsdf->type = transmission ? CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_TRANSMISSION_ID :
                              CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_REFLECTION_ID;
  sd->runtime_flag |= SR_BSDF | SR_BSDF_HAS_DISPERSION |
                      (transmission ? SR_BSDF_HAS_TRANSMISSION : 0) |
                      (roughness_is_almost_specular(alpha, alpha) ? 0 : SR_BSDF_HAS_EVAL);
  return true;
}

/* Preserve the native two-interface Fresnel and tint computation. The scalar
 * phase screen redistributes each port's already-absorbed spectral power. */
ccl_device_inline FresnelCoeff bsdf_diffraction_thin_glass_setup(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const bool reflective,
    const bool refractive,
    const FresnelCoeff tint,
    const Spectrum weight,
    const float3 N,
    const float3 T,
    const float roughness,
    const float ior,
    const FresnelThinFilm thinfilm,
    const PathRayVisibility ray_visibility,
    const uint32_t path_flag,
    const float diffraction,
    const float pitch_nm,
    const float depth_nm,
    const float duty,
    const float inv_abbe,
    const int3 albedo_handles = make_int3(-1,-1,-1))
{
  const float amount = saturatef(diffraction);
  if (!(amount > 0.0f && depth_nm > 0.0f && pitch_nm > 0.0f)) {
    return bsdf_thin_glass_setup(kg, sd, reflective, refractive, tint, weight, N,
                                 roughness, ior, thinfilm, ray_visibility, path_flag);
  }
  const bool back = (sd->runtime_flag & SR_BACKFACING) != 0;
  const float oriented_ior = bsdf_glass_ior(sd, back ? 1.0f / ior : ior, inv_abbe);
  const float sheet_ior = back ? 1.0f / oriented_ior : oriented_ior;
  const float wavelength_nm = 1000.0f * sample_wavelength(sd->rand_wavelength);
  if (!(isfinite_safe(sheet_ior) && sheet_ior > 0.0f && wavelength_nm > 0.0f &&
        duty >= 0.0f && duty <= 1.0f && isfinite_safe(depth_nm / wavelength_nm)))
  {
    return bsdf_thin_glass_setup(kg, sd, reflective, refractive, tint, weight, N,
                                 roughness, ior, thinfilm, ray_visibility, path_flag);
  }
  const bool cached = reflective && refractive && sheet_ior >= 1.0f &&
                      albedo_handles.x >= 0 && albedo_handles.y >= 0 && albedo_handles.z >= 0;
  const float reflection_phase = 2.0f * M_2PI_F * depth_nm / wavelength_nm;
  const float transmission_phase = M_2PI_F * (sheet_ior - 1.0f) * depth_nm / wavelength_nm;
  const float blend = diffraction_thin_sheet_model_relief_blend(reflection_phase,transmission_phase,duty);
  if (cached && !(blend>0.0f)) {
    return bsdf_thin_glass_setup(kg,sd,reflective,refractive,tint,weight,N,roughness,ior,
                                 thinfilm,ray_visibility,path_flag);
  }
  const float native_coverage=cached ? 1.0f-amount*blend : 1.0f-amount;
  const int extra_slots = (sizeof(DiffractionThinSheetExtra)+sizeof(ShaderClosure)-1)/sizeof(ShaderClosure);
  const int required_slots = (native_coverage > 0.0f ? 2 : 0)+(cached ? 4+extra_slots : 2);
  if (sd->num_closure_left < required_slots) {
    if (cached) {
      /* Never substitute a different native material when the reserved
       * completed-model allocation does not fit. No partial closures. */
      return bsdf_thin_glass_fresnel(kg,reflective,refractive,tint,thinfilm,sheet_ior,dot(N,sd->wi));
    }
    return bsdf_thin_glass_setup(kg, sd, reflective, refractive, tint, weight, N,
                                 roughness, ior, thinfilm, ray_visibility, path_flag);
  }
  const FresnelCoeff sheet = bsdf_thin_glass_fresnel(
      kg, reflective, refractive, tint, thinfilm, sheet_ior, dot(N, sd->wi));
  if (native_coverage > 0.0f) {
    /* The compatibility endpoint is the original native sheet, including its
     * unmodified d-line IOR. Dispersing this endpoint would create a nonzero
     * tiny-relief jump even when the completed component fades to zero. */
    bsdf_thin_glass_setup(kg, sd, reflective, refractive, tint, weight * native_coverage,
                         N, roughness, cached ? ior : sheet_ior, thinfilm, ray_visibility,path_flag);
  }
  const float alpha_reflection = saturatef(roughness);
  const float alpha_transmission = bsdf_thin_glass_transmission_roughness(
      alpha_reflection, sheet_ior);
  const Spectrum native_reflection_compensation = diffraction_thin_sheet_native_compensation(
      kg, sd, N, alpha_reflection, tint.reflectance, false);
  const Spectrum native_transmission_compensation = diffraction_thin_sheet_native_compensation(
      kg, sd, N, alpha_transmission, one_spectrum(), true);
  /* Planar multiple-bounce compensation belongs to the undiffracted carrier.
   * Fade its excess as binary phase power leaves order zero. This retains the
   * exact native flat limit without applying the whole planar correction to
   * widely redirected orders. It remains a declared approximate correction. */
  const Spectrum reflection_compensation = mix(
      one_spectrum(), native_reflection_compensation,
      diffraction_binary_power(0, reflection_phase, duty));
  const Spectrum transmission_compensation = mix(
      one_spectrum(), native_transmission_compensation,
      diffraction_binary_power(0, transmission_phase, duty));
  ccl_private DiffractionThinSheetExtra *extra = nullptr;
  if (cached) {
    extra = (ccl_private DiffractionThinSheetExtra *)closure_alloc_extra(sd,sizeof(DiffractionThinSheetExtra));
    if (extra == nullptr) return sheet; /* Whole allocation was checked above. */
    extra->model.reflection = {alpha_reflection,wavelength_nm/pitch_nm,reflection_phase,duty,false};
    extra->model.transmission = {alpha_transmission,wavelength_nm/pitch_nm,transmission_phase,duty,true};
    extra->model.ior=sheet_ior;
    extra->model.reflection_tint=extra->model.transmission_tint=1.0f;
    extra->model.film_ior=thinfilm.ior;
    extra->model.film_over_wavelength=thinfilm.thickness>THINFILM_THICKNESS_CUTOFF ? thinfilm.thickness/wavelength_nm : 0.0f;
    extra->tint={saturate(tint.reflectance),saturate(tint.transmittance)};
    extra->native_reflection=zero_spectrum();
    extra->native_transmission=zero_spectrum();
    extra->albedo_handles=albedo_handles;
    extra->wavelength_nm=wavelength_nm;
    extra->blend=blend;
  }
  for (int port=0;port<2;++port) {
    const Spectrum carrier=cached ? zero_spectrum() : (port ? sheet.transmittance*transmission_compensation : sheet.reflectance*reflection_compensation);
    const int lobes=cached ? 2 : 1;
    for (int lobe=0;lobe<lobes;++lobe) {
      const int before=sd->num_closure;
      const Spectrum w=weight*amount*(cached ? one_spectrum() : carrier);
      bsdf_diffraction_thin_sheet_setup(sd,w,N,T,port ? alpha_transmission : alpha_reflection,
                                        pitch_nm,depth_nm,duty,sheet_ior,port!=0);
      if (sd->num_closure>before && extra!=nullptr) {
        ccl_private DiffractionThinSheetBsdf *bsdf=(ccl_private DiffractionThinSheetBsdf *)&sd->closure[before];
        bsdf->extra=extra;
        bsdf->return_only=lobe!=0;
        Spectrum potential=zero_spectrum();
        float3 X,Y; make_orthonormals_safe_tangent(bsdf->N,bsdf->T,&X,&Y);
        const float3 I=make_float3(dot(sd->wi,X),dot(sd->wi,Y),dot(sd->wi,bsdf->N));
        for (int c=0;c<3;++c) {
          if (lobe) {
            potential[c]=0.5f*extra->blend*diffraction_albedo_lookup(kg,albedo_handles[c],wavelength_nm,make_float3(fabsf(I.x),fabsf(I.y),I.z)).x;
          }
          else {
            DiffractionThinSheetModel model=extra->model;
            model.reflection_tint=extra->tint.reflectance[c];
            model.transmission_tint=extra->tint.transmittance[c];
            const float2 coeff=diffraction_thin_sheet_model_coefficients(&model,I.z);
            const float q=diffraction_albedo_lookup(kg,albedo_handles[c],wavelength_nm,make_float3(fabsf(I.x),fabsf(I.y),I.z)).x;
            const float escaped=coeff.x+coeff.y>0.0f ? max(0.0f,1.0f-q/(coeff.x+coeff.y)) : 0.0f;
            potential[c]=mix(carrier[c],(port ? coeff.y : coeff.x)*escaped,extra->blend);
          }
        }
        bsdf->sample_weight *= max(0.0f,average(potential));
        sd->runtime_flag |= SR_BSDF_HAS_EVAL;
      }
    }
  }
  if (cached && native_coverage>0.0f) {
    const FresnelCoeff native=bsdf_thin_glass_fresnel(kg,reflective,refractive,tint,thinfilm,ior,dot(N,sd->wi));
    return {mix(native.reflectance,sheet.reflectance,amount*blend),
            mix(native.transmittance,sheet.transmittance,amount*blend)};
  }
  return sheet;
}

ccl_device_inline int bsdf_diffraction_thin_sheet_label(const ccl_private ShaderClosure *sc)
{
  const ccl_private DiffractionThinSheetBsdf *bsdf =
      (const ccl_private DiffractionThinSheetBsdf *)sc;
  return (bsdf->port.transmission ? LABEL_TRANSMIT : LABEL_REFLECT) |
         (!bsdf->return_only && roughness_is_almost_specular(bsdf->port.alpha, bsdf->port.alpha) ? LABEL_SINGULAR :
                                                                            LABEL_GLOSSY);
}

ccl_device_inline Spectrum bsdf_diffraction_thin_sheet_eval_legacy(const ccl_private ShaderClosure *sc,
                                                             const float3 wi,
                                                             const float3 wo,
                                                             ccl_private float *pdf)
{
  const ccl_private DiffractionThinSheetBsdf *bsdf =
      (const ccl_private DiffractionThinSheetBsdf *)sc;
  float3 X, Y;
  make_orthonormals_safe_tangent(bsdf->N, bsdf->T, &X, &Y);
  const float3 I = make_float3(dot(wi, X), dot(wi, Y), dot(wi, bsdf->N));
  const float3 O = make_float3(dot(wo, X), dot(wo, Y), dot(wo, bsdf->N));
  return make_spectrum(diffraction_thin_sheet_eval(&bsdf->port, I, O, pdf));
}

ccl_device_inline int bsdf_diffraction_thin_sheet_sample_legacy(const ccl_private ShaderClosure *sc,
                                                          const float3 Ng,
                                                          const float3 wi,
                                                          const float3 random,
                                                          ccl_private Spectrum *eval,
                                                          ccl_private float3 *wo,
                                                          ccl_private float *pdf,
                                                          ccl_private float2 *roughness,
                                                          ccl_private float *eta)
{
  const ccl_private DiffractionThinSheetBsdf *bsdf =
      (const ccl_private DiffractionThinSheetBsdf *)sc;
  float3 X, Y;
  make_orthonormals_safe_tangent(bsdf->N, bsdf->T, &X, &Y);
  const float3 I = make_float3(dot(wi, X), dot(wi, Y), dot(wi, bsdf->N));
  float3 O;
  float value;
  *eval = zero_spectrum();
  *roughness = make_float2(bsdf->port.alpha, bsdf->port.alpha);
  *eta = 1.0f;
  if (!diffraction_thin_sheet_sample(&bsdf->port, I, random, &O, &value, pdf)) {
    return LABEL_NONE;
  }
  *wo = to_global(O, X, Y, bsdf->N);
  if (!(bsdf->port.transmission ? dot(Ng, *wo) < 0.0f : dot(Ng, *wo) > 0.0f)) {
    *pdf = 0.0f;
    return LABEL_NONE;
  }
  *eval = make_spectrum(value);
  const int label = bsdf_diffraction_thin_sheet_label(sc);
  if (label & LABEL_SINGULAR) {
    *pdf *= 1e6f;
    *eval *= 1e6f;
    *roughness = zero_float2();
  }
  return label;
}

ccl_device_inline Spectrum bsdf_diffraction_thin_sheet_delta_legacy(const ccl_private ShaderClosure *sc,
                                                              const float3 wi,
                                                              const float3 wo,
                                                              ccl_private float *pdf)
{
  *pdf = 0.0f;
  const ccl_private DiffractionThinSheetBsdf *bsdf =
      (const ccl_private DiffractionThinSheetBsdf *)sc;
  if (!roughness_is_almost_specular(bsdf->port.alpha, bsdf->port.alpha)) {
    return zero_spectrum();
  }
  float3 X, Y;
  make_orthonormals_safe_tangent(bsdf->N, bsdf->T, &X, &Y);
  const float3 I0 = make_float3(dot(wi, X), dot(wi, Y), dot(wi, bsdf->N));
  const float3 O0 = make_float3(dot(wo, X), dot(wo, Y), dot(wo, bsdf->N));
  const float3 I = bsdf->port.transmission ? diffraction_thin_sheet_flip_tangent(I0) : I0;
  const float3 O = bsdf->port.transmission ? -O0 : O0;
  if (!(I.z > 0.0f && O.z > 0.0f)) {
    return zero_spectrum();
  }
  const float delta = I.x + O.x;
  const float order = bsdf->port.wavelength_over_pitch > 0.0f ?
                          roundf(delta / bsdf->port.wavelength_over_pitch) : 0.0f;
  if (!isfinite_safe(order) || fabsf(order) > diffraction_thin_sheet_order_bound(&bsdf->port)) {
    return zero_spectrum();
  }
  const int m = int(order);
  if (sqr(delta - m * bsdf->port.wavelength_over_pitch) + sqr(I.y + O.y) > 16e-12f)
  {
    return zero_spectrum();
  }
  const float mass = diffraction_thin_sheet_power(
      &bsdf->port, m, I, make_float3(0.0f, 0.0f, 1.0f));
  *pdf = mass * 1e6f;
  return make_spectrum(mass * 1e6f);
}

ccl_device_inline DiffractionThinSheetModel diffraction_thin_sheet_channel_model(
    ccl_private const DiffractionThinSheetExtra *extra, const int channel)
{
  DiffractionThinSheetModel p=extra->model;
  p.reflection_tint=extra->tint.reflectance[channel];
  p.transmission_tint=extra->tint.transmittance[channel];
  return p;
}

ccl_device_inline Spectrum diffraction_thin_sheet_cached_missing(
    KernelGlobals kg, ccl_private const DiffractionThinSheetExtra *extra,
    const float3 w, ccl_private Spectrum *average)
{
  Spectrum q=zero_spectrum();
  for (int c=0;c<3;++c) {
    const float2 data=diffraction_albedo_lookup(kg,extra->albedo_handles[c],extra->wavelength_nm,
                                               make_float3(fabsf(w.x),fabsf(w.y),fabsf(w.z)));
    q[c]=data.x; (*average)[c]=data.y;
  }
  return q;
}

ccl_device_inline Spectrum bsdf_diffraction_thin_sheet_eval(
    KernelGlobals kg,const ccl_private ShaderClosure *sc,const float3 wi,const float3 wo,
    ccl_private float *pdf)
{
  const ccl_private DiffractionThinSheetBsdf *b=(const ccl_private DiffractionThinSheetBsdf *)sc;
  if (b->extra==nullptr) return bsdf_diffraction_thin_sheet_eval_legacy(sc,wi,wo,pdf);
  float3 X,Y; make_orthonormals_safe_tangent(b->N,b->T,&X,&Y);
  const float3 I=make_float3(dot(wi,X),dot(wi,Y),dot(wi,b->N));
  const float3 O=make_float3(dot(wo,X),dot(wo,Y),dot(wo,b->N));
  const float mu=b->port.transmission ? -O.z : O.z;
  *pdf=0.0f;
  if (!(I.z>0.0f && mu>0.0f)) return zero_spectrum();
  ccl_private const DiffractionThinSheetExtra *e=b->extra;
  if (b->return_only) {
    *pdf=mu*M_1_PI_F;
    Spectrum avg,unused;
    const Spectrum qi=diffraction_thin_sheet_cached_missing(kg,e,I,&avg);
    const Spectrum qo=diffraction_thin_sheet_cached_missing(kg,e,O,&unused);
    Spectrum value=zero_spectrum();
    for (int c=0;c<3;++c) {
      value[c]=avg[c]>0.0f ? e->blend*qi[c]*qo[c]/(M_2PI_F*avg[c])*mu : 0.0f;
    }
    return value;
  }
  float proposal;
  const float raw=diffraction_thin_sheet_eval(&b->port,I,O,&proposal);
  *pdf=proposal;
  Spectrum value=(1.0f-e->blend)*raw*(b->port.transmission ? e->native_transmission : e->native_reflection);
  if (!roughness_is_almost_specular(b->port.alpha,b->port.alpha)) {
    /* Reuse facet roots/Jacobians across RGB coefficient channels. */
    const float3 FI=b->port.transmission ? diffraction_thin_sheet_flip_tangent(I) : I;
    const float3 FO=b->port.transmission ? -O : O;
    Spectrum Ci,Co;
    for (int c=0;c<3;++c) {
      const DiffractionThinSheetModel model=diffraction_thin_sheet_channel_model(e,c);
      Ci[c]=diffraction_thin_sheet_model_coefficient(&model,FI.z,b->port.transmission);
      Co[c]=diffraction_thin_sheet_model_coefficient(&model,FO.z,b->port.transmission);
    }
    Spectrum density=zero_spectrum();
    const float3 axis=make_float3(1,0,0);
    const int bound=diffraction_thin_sheet_order_bound(&b->port);
    for (int m=-bound;m<=bound;++m) {
      DiffractionReflectionRoot roots[2];
      const float delta=m*b->port.wavelength_over_pitch;
      const int count=diffraction_reflection_half_vectors(FI,FO,axis,delta,roots);
      for (int r=0;r<count;++r) {
        const float3 h=roots[r].h;
        if (!(h.z>0.0f)) continue;
        const float pi=diffraction_thin_sheet_power(&b->port,m,FI,h);
        const float po=diffraction_thin_sheet_power(&b->port,m,FO,h);
        const float geometry=bsdf_aniso_D<GGX>(b->port.alpha,b->port.alpha,h)*dot(FI,h)*
            diffraction_reflection_jacobian(FI,FO,h,axis,delta)/FI.z;
        density += min(Ci*pi,Co*po)*geometry;
      }
    }
    value += e->blend*density/(1.0f+bsdf_aniso_lambda<GGX>(b->port.alpha,b->port.alpha,FI)+
                                    bsdf_aniso_lambda<GGX>(b->port.alpha,b->port.alpha,FO));
  }
  return value;
}

ccl_device_inline Spectrum bsdf_diffraction_thin_sheet_delta(
    KernelGlobals kg,const ccl_private ShaderClosure *sc,const float3 wi,const float3 wo,
    ccl_private float *pdf)
{
  const ccl_private DiffractionThinSheetBsdf *b=(const ccl_private DiffractionThinSheetBsdf *)sc;
  if (b->return_only) { *pdf=0.0f; return zero_spectrum(); }
  const Spectrum raw=bsdf_diffraction_thin_sheet_delta_legacy(sc,wi,wo,pdf);
  if (b->extra==nullptr || *pdf==0.0f) return raw;
  float3 X,Y;make_orthonormals_safe_tangent(b->N,b->T,&X,&Y);
  const float3 I0=make_float3(dot(wi,X),dot(wi,Y),dot(wi,b->N));
  const float3 O0=make_float3(dot(wo,X),dot(wo,Y),dot(wo,b->N));
  const float3 I=b->port.transmission ? diffraction_thin_sheet_flip_tangent(I0) : I0;
  const float3 O=b->port.transmission ? -O0 : O0;
  const int order=int(roundf((I.x+O.x)/b->port.wavelength_over_pitch));
  ccl_private const DiffractionThinSheetExtra *e=b->extra;
  Spectrum result=(1.0f-e->blend)*raw*(b->port.transmission ? e->native_transmission : e->native_reflection);
  for (int c=0;c<3;++c) {
    const DiffractionThinSheetModel model=diffraction_thin_sheet_channel_model(e,c);
    ccl_private const DiffractionThinSheetModelPort *p=b->port.transmission ? &model.transmission : &model.reflection;
    result[c]+=e->blend*diffraction_thin_sheet_model_power(&model,p,order,I,O,make_float3(0,0,1))*1e6f;
  }
  (void)kg;
  return result;
}

ccl_device_inline int bsdf_diffraction_thin_sheet_sample(
    KernelGlobals kg,const ccl_private ShaderClosure *sc,const float3 Ng,const float3 wi,
    const float3 random,ccl_private Spectrum *eval,ccl_private float3 *wo,ccl_private float *pdf,
    ccl_private float2 *roughness,ccl_private float *eta)
{
  const ccl_private DiffractionThinSheetBsdf *b=(const ccl_private DiffractionThinSheetBsdf *)sc;
  if (b->extra==nullptr) return bsdf_diffraction_thin_sheet_sample_legacy(sc,Ng,wi,random,eval,wo,pdf,roughness,eta);
  float3 X,Y;make_orthonormals_safe_tangent(b->N,b->T,&X,&Y);
  const float3 I=make_float3(dot(wi,X),dot(wi,Y),dot(wi,b->N));
  float3 O;float dummy;
  const bool first=!b->return_only;
  *eta=1.0f;*eval=zero_spectrum();*pdf=0.0f;
  *roughness=make_float2(b->port.alpha,b->port.alpha);
  if (first) {
    if (!diffraction_thin_sheet_sample(&b->port,I,random,&O,&dummy,pdf)) return LABEL_NONE;
  }
  else {
    const float r=sqrtf(random.x),phi=M_2PI_F*random.y;
    O=make_float3(r*cosf(phi),r*sinf(phi),sqrtf(1.0f-r*r));
    if (b->port.transmission) O.z=-O.z;
    *roughness=one_float2();
  }
  *wo=to_global(O,X,Y,b->N);
  if (!(b->port.transmission ? dot(Ng,*wo)<0.0f : dot(Ng,*wo)>0.0f)) return LABEL_NONE;
  if (first && roughness_is_almost_specular(b->port.alpha,b->port.alpha)) {
    *eval=bsdf_diffraction_thin_sheet_delta(kg,sc,wi,*wo,pdf);
    *roughness=zero_float2();
    return (b->port.transmission ? LABEL_TRANSMIT : LABEL_REFLECT)|LABEL_SINGULAR;
  }
  *eval=bsdf_diffraction_thin_sheet_eval(kg,sc,wi,*wo,pdf);
  return (b->port.transmission ? LABEL_TRANSMIT : LABEL_REFLECT)|LABEL_GLOSSY;
}

/* Legacy signatures retained for standalone native-limit comparisons. */
ccl_device_inline Spectrum bsdf_diffraction_thin_sheet_eval(const ccl_private ShaderClosure *sc,
    const float3 wi,const float3 wo,ccl_private float *pdf)
{ return bsdf_diffraction_thin_sheet_eval_legacy(sc,wi,wo,pdf); }
ccl_device_inline Spectrum bsdf_diffraction_thin_sheet_delta(const ccl_private ShaderClosure *sc,
    const float3 wi,const float3 wo,ccl_private float *pdf)
{ return bsdf_diffraction_thin_sheet_delta_legacy(sc,wi,wo,pdf); }
ccl_device_inline int bsdf_diffraction_thin_sheet_sample(const ccl_private ShaderClosure *sc,
    const float3 Ng,const float3 wi,const float3 random,ccl_private Spectrum *eval,
    ccl_private float3 *wo,ccl_private float *pdf,ccl_private float2 *roughness,ccl_private float *eta)
{ return bsdf_diffraction_thin_sheet_sample_legacy(sc,Ng,wi,random,eval,wo,pdf,roughness,eta); }

CCL_NAMESPACE_END
