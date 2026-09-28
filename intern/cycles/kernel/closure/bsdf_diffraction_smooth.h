/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/closure/bsdf_diffraction_interface.h"
#include "kernel/util/diffraction_scene.h"
#include "kernel/util/diffraction_index.h"
CCL_NAMESPACE_BEGIN

/* Physical smooth lamellar grating. N faces the incident path as for other
 * closures; incoming_substrate identifies its side in the fixed cache frame. */
struct DiffractionSmoothBsdf {
  SHADER_CLOSURE_BASE;
  float3 T;
  float wavelength, pitch, upper_index, lower_index;
  int cache_handle, incoming_substrate;
  ccl_private FastDiffractionInterface *fast;
};
static_assert(sizeof(ShaderClosure) >= sizeof(DiffractionSmoothBsdf));

/* Shader compilation must advertise wavelength dependence and register the
 * cache before shading. This setup never substitutes an RGB wavelength. */
ccl_device bool bsdf_diffraction_smooth_setup(KernelGlobals kg,
                                              ccl_private ShaderData *sd,
                                              const float3 weight,
                                              const float3 normal,
                                              const float3 tangent,
                                              const int cache_handle,
                                              const float pitch,
                                              const float upper_index,
                                              const float lower_index,
                                              const bool incoming_substrate,
                                              const float depth = 0,
                                              const float duty = 0.5f,
                                              const float phase_contrast = 0,
                                              const float reflection_budget = 0,
                                              const float transmission_budget = 0,
                                              const float ridge_extinction = 0,
                                              const int ridge_table = -1,
                                              const int groove_table = -1,
                                              const int substrate_table = -1)
{
#ifdef __SPECTRAL__
  const bool fast_model = cache_handle == DIFFRACTION_FAST_CACHE_HANDLE;
  if (!(sd->shader_flag & SD_REQUIRES_WAVELENGTH) ||
      (!fast_model && (cache_handle < 0 || cache_handle >= kernel_data.tables.num_diffraction_caches)) || !(pitch > 0) ||
      !(upper_index > 0) || !(lower_index > 0) || !isfinite_safe(pitch) ||
      !isfinite_safe(upper_index) || !isfinite_safe(lower_index))
    return false;
  const float n2 = len_squared(normal);
  if (!(n2 > 0) || !isfinite_safe(n2))
    return false;
  const float3 N = normal / sqrtf(n2);
  const float3 T = tangent - dot(tangent, N) * N;
  const float t2 = len_squared(T);
  if (!(t2 > 0) || !isfinite_safe(t2))
    return false;
  const float wavelength = 1000.0f * sample_wavelength(sd->rand_wavelength);
  if (!isfinite_safe(wavelength))
    return false;
  FastDiffractionInterface fast_params{};
  if (fast_model) {
    float contrast = phase_contrast, extinction = ridge_extinction;
    float reflection = reflection_budget, transmission = transmission_budget;
    float2 index;
    if (ridge_table >= 0) {
      if (!diffraction_index_lookup(kg, ridge_table, wavelength, &index)) return false;
      contrast += index.x;
      extinction = index.y;
    }
    if (groove_table >= 0) {
      if (!diffraction_index_lookup(kg, groove_table, wavelength, &index)) return false;
      contrast -= index.x;
    }
    if (substrate_table >= 0) {
      if (!diffraction_index_lookup(kg, substrate_table, wavelength, &index)) return false;
      /* Normal-incidence Fresnel budget. Scale all indices together to avoid
       * overflow while retaining the same ratio for strongly absorbing media. */
      const float scale = max(upper_index, max(index.x, index.y));
      const float ni = upper_index / scale, n = index.x / scale, k = index.y / scale;
      reflection = (sqr(n - ni) + sqr(k)) / (sqr(n + ni) + sqr(k));
      transmission = 0;
    }
    const float attenuation = (1-duty) + duty * expf(-2*M_2PI_F*extinction*depth/wavelength);
    fast_params = {wavelength, pitch, depth, duty,
                   incoming_substrate ? lower_index : upper_index,
                   incoming_substrate ? upper_index : lower_index,
                   reflection, transmission*attenuation,
                   M_2PI_F*depth*contrast/wavelength};
    if (!fast_diffraction_interface_valid(fast_params)) return false;
  }
  else {
    const float4 lo = kernel_data_fetch(diffraction_domains, 2 * cache_handle);
    const float4 hi = kernel_data_fetch(diffraction_domains, 2 * cache_handle + 1);
    if (wavelength < lo.z || wavelength > hi.z) return false;
  }
  const Spectrum spectral_weight = bsdf_spectral_transmission_color(kg, sd, weight);
  ccl_private DiffractionSmoothBsdf *bsdf = (ccl_private DiffractionSmoothBsdf *)bsdf_alloc(
      sd, sizeof(DiffractionSmoothBsdf), spectral_weight);
  if (!bsdf)
    return false;
  bsdf->fast = nullptr;
  if (fast_model) {
    bsdf->fast = (ccl_private FastDiffractionInterface *)closure_alloc_extra(sd, sizeof(FastDiffractionInterface));
    if (!bsdf->fast) return false;
    *bsdf->fast = fast_params;
  }
  bsdf->type = CLOSURE_BSDF_DIFFRACTION_SMOOTH_ID;
  bsdf->N = N;
  bsdf->T = T / sqrtf(t2);
  bsdf->wavelength = wavelength;
  bsdf->pitch = pitch;
  bsdf->upper_index = upper_index;
  bsdf->lower_index = lower_index;
  bsdf->cache_handle = cache_handle;
  bsdf->incoming_substrate = incoming_substrate;
  sd->runtime_flag |= SR_BSDF | SR_BSDF_HAS_DISPERSION;
  return true;
#else
  return false;
#endif
}

/* Keep cache solver scratch out of each caller's BSDF dispatch frame on Metal.
 * Guiding calls the dispatch from multiple proposal branches. */
#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device
#endif
int bsdf_diffraction_smooth_sample(KernelGlobals kg,
                                              ccl_private const ShaderClosure *sc,
                                              const float3 Ng,
                                              const float3 wi,
                                              const float3 rand,
                                              ccl_private Spectrum *eval,
                                              ccl_private float3 *wo,
                                              ccl_private float *pdf,
                                              ccl_private float2 *roughness,
                                              ccl_private float *eta)
{
  ccl_private const DiffractionSmoothBsdf *bsdf = (ccl_private const DiffractionSmoothBsdf *)sc;
  *eval = zero_spectrum();
  *pdf = 0;
  *eta = 1;
  *roughness = zero_float2();
  if (bsdf->fast) {
    const float3 Y = cross(bsdf->N, bsdf->T);
    const float3 local_wi = make_float3(dot(wi,bsdf->T),dot(wi,Y),dot(wi,bsdf->N));
    float3 local_wo;float power,probability;int order;bool transmission;
    if (!fast_diffraction_interface_sample(*bsdf->fast,local_wi,rand.z,
                                           &order,&transmission,&local_wo,&power,&probability)) return LABEL_NONE;
    *wo = local_wo.x*bsdf->T + local_wo.y*Y + local_wo.z*bsdf->N;
    if ((dot(Ng,*wo)<0)!=transmission) return LABEL_NONE;
    *pdf = probability*1e6f;
    *eval = make_spectrum(power*1e6f);
    *eta = transmission ? bsdf->fast->transmitted_ior/bsdf->fast->incident_ior : 1;
    return LABEL_SINGULAR | (transmission ? LABEL_TRANSMIT : LABEL_REFLECT);
  }
  const float3 N = bsdf->incoming_substrate ? -bsdf->N : bsdf->N;
  const float3 tangent = bsdf->T - dot(bsdf->T, N) * N;
  const float length_squared = len_squared(tangent);
  if (!(length_squared > 0) || !isfinite_safe(length_squared))
    return LABEL_NONE;
  const float3 X = tangent / sqrtf(length_squared), Y = cross(N, X);
  const float3 incident = make_float3(-dot(wi, X), -dot(wi, Y), -dot(wi, N));
  DiffractionSceneSample sample;
  if (!diffraction_scene_sample(kg,
                                bsdf->cache_handle,
                                incident,
                                bool(bsdf->incoming_substrate),
                                bsdf->upper_index,
                                bsdf->lower_index,
                                bsdf->wavelength,
                                bsdf->pitch,
                                rand.z,
                                &sample))
    return LABEL_NONE;
  *wo = sample.direction.x * X + sample.direction.y * Y + sample.direction.z * N;
  if ((dot(Ng, *wo) < 0) != sample.transmission || (dot(bsdf->N, *wo) < 0) != sample.transmission)
    return LABEL_NONE;
  /* Cycles represents Dirac sampling using the same large MIS weight as its
   * smooth microfacets. Evaluation at an arbitrary direction remains zero. */
  *pdf = sample.probability * 1e6f;
  *eval = make_spectrum(sample.throughput * *pdf);
  *eta = sample.eta;
  return LABEL_SINGULAR | (sample.transmission ? LABEL_TRANSMIT : LABEL_REFLECT);
}
/* Probability mass for a known delta direction, in the closure's world frame.
 * Intended for path-strategy accounting; ordinary BSDF evaluation stays zero. */
#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device
#endif
bool bsdf_diffraction_smooth_delta_probability(
    KernelGlobals kg, ccl_private const ShaderClosure *sc, const float3 wi, const float3 wo,
    ccl_private float *power, ccl_private float *probability)
{
  ccl_private const DiffractionSmoothBsdf *bsdf =
      (ccl_private const DiffractionSmoothBsdf *)sc;
  if (bsdf->fast) {
    *power=0;*probability=0;
    const float3 Y=cross(bsdf->N,bsdf->T);
    const float3 local_wi=make_float3(dot(wi,bsdf->T),dot(wi,Y),dot(wi,bsdf->N));
    const float3 local_wo=make_float3(dot(wo,bsdf->T),dot(wo,Y),dot(wo,bsdf->N));
    return fast_diffraction_interface_probability(*bsdf->fast,local_wi,local_wo,power,probability);
  }
  const float3 N = bsdf->incoming_substrate ? -bsdf->N : bsdf->N;
  const float3 Y = cross(N, bsdf->T);
  const float3 incident = make_float3(-dot(wi, bsdf->T), -dot(wi, Y), -dot(wi, N));
  const float3 outgoing = make_float3(dot(wo, bsdf->T), dot(wo, Y), dot(wo, N));
  return diffraction_scene_direction_probability(kg, bsdf->cache_handle, incident,
      bool(bsdf->incoming_substrate), bsdf->upper_index, bsdf->lower_index,
      bsdf->wavelength, bsdf->pitch, outgoing, power, probability);
}
CCL_NAMESPACE_END
