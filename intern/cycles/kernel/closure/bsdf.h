/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

// clang-format off
#include "kernel/closure/bsdf_ashikhmin_velvet.h"
#include "kernel/closure/bsdf_diffuse.h"
#include "kernel/closure/bsdf_oren_nayar.h"
#include "kernel/closure/bsdf_phong_ramp.h"
#include "kernel/closure/bsdf_diffuse_ramp.h"
#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/closure/bsdf_diffraction.h"
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include "kernel/closure/bsdf_diffraction_conductor.h"
#include "kernel/closure/bsdf_diffraction_ashikhmin.h"
#include "kernel/closure/bsdf_diffraction_smooth.h"
#include "kernel/closure/bsdf_diffraction_thin_sheet.h"
#include "kernel/closure/bsdf_burley.h"
#include "kernel/closure/bsdf_sheen.h"
#include "kernel/closure/bsdf_transparent.h"
#include "kernel/closure/bsdf_ray_portal.h"
#include "kernel/closure/bsdf_ashikhmin_shirley.h"
#include "kernel/closure/bsdf_toon.h"
#include "kernel/closure/bsdf_hair.h"
#include "kernel/closure/bsdf_principled_hair_chiang.h"
#include "kernel/closure/bsdf_principled_hair_huang.h"
// clang-format on

CCL_NAMESPACE_BEGIN

/* Returns the square of the roughness of the closure if it has roughness,
 * 0 for singular closures and 1 otherwise. */
ccl_device_inline float bsdf_get_specular_roughness_squared(const ccl_private ShaderClosure *sc)
{
  if (CLOSURE_IS_BSDF_SINGULAR(sc->type)) {
    return 0.0f;
  }

  if ((sc->type == CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID || sc->type == CLOSURE_BSDF_DIFFRACTION_BECKMANN_ID)) {
    const ccl_private DiffractionDielectricBsdf *b=(const ccl_private DiffractionDielectricBsdf *)sc;
    return diffraction_dielectric_param(b)->alpha_x*diffraction_dielectric_param(b)->alpha_y;
  }

  if (bsdf_is_diffraction_conductor(sc->type)) {
    const ccl_private DiffractionConductorBsdf *b=(const ccl_private DiffractionConductorBsdf *)sc;
    return b->extra->param.alpha_x*b->extra->param.alpha_y;
  }
  if (sc->type == CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_REFLECTION_ID ||
      sc->type == CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_TRANSMISSION_ID) {
    const ccl_private DiffractionThinSheetBsdf *b =
        (const ccl_private DiffractionThinSheetBsdf *)sc;
    return b->return_only ? 1.0f : sqr(b->port.alpha);
  }
  if (sc->type == CLOSURE_BSDF_DIFFRACTION_ID ||
      sc->type == CLOSURE_BSDF_DIFFRACTION_ASHIKHMIN_ID) {
    const ccl_private DiffractionBsdf *bsdf = (const ccl_private DiffractionBsdf *)sc;
    return bsdf->param.alpha_x * bsdf->param.alpha_y;
  }

  if (CLOSURE_IS_BSDF_MICROFACET(sc->type)) {
    ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)sc;
    return bsdf->alpha_x * bsdf->alpha_y;
  }

  return 1.0f;
}

ccl_device_inline float bsdf_get_roughness_pass_squared(const ccl_private ShaderClosure *sc)
{
  if (sc->type == CLOSURE_BSDF_OREN_NAYAR_ID || sc->type == CLOSURE_BSDF_ROUGH_TRANSLUCENT_ID) {
    ccl_private OrenNayarBsdf *bsdf = (ccl_private OrenNayarBsdf *)sc;
    return sqr(sqr(bsdf->param.roughness));
  }

  /* For the Principled BSDF, we want the Roughness pass to return the value that
   * was set in the node. However, this value doesn't affect all closures (e.g.
   * diffuse), so skip those that don't really have a concept of roughness. */
  if (CLOSURE_IS_BSDF_DIFFUSE(sc->type)) {
    return -1.0f;
  }

  return bsdf_get_specular_roughness_squared(sc);
}

/* Widen the compact ray differential dD after a non-specular bounce so that
 * texture mip selection on subsequent hits reflects the BSDF lobe's angular
 * spread. This significantly save memory, and is needed to make image cache
 * memory usage scale with render tile size rather than overall resolution.
 *
 * This must be done consistently between next event estimation and forward
 * sampling for both to converge to the same result for MIS. This is not just
 * a theoretical concern, but can otherwise lead to seams.
 *
 * To achieve that, the sampled roughness is computed as a MIS weighted
 * average. This makes it so directions with high contribution from sharp
 * BSDFs have a lower roughness, as they will have a high MIS weight.
 *
 * The amount of widening is scaled by Filter Glossy, which is the setting
 * to control how much caustics are blurred out. For accurate caustics
 * image texture lookups and bump ray differentials must be small enough
 * to capture detail at secondary bounces. */
ccl_device_forceinline float bsdf_widen_dD(const KernelGlobals kg,
                                           const float prev_dD,
                                           const float avg_roughness_squared)
{
  if (!(avg_roughness_squared > 0.0f)) {
    return prev_dD;
  }

  const float scale = kernel_data.integrator.differential_widen_scale;
  return max(prev_dD, scale * sqrtf(avg_roughness_squared));
}

/* An additional term to smooth illumination on grazing angles when using bump mapping
 * based on "A Microfacet-Based Shadowing Function to Solve the Bump Terminator Problem"
 * by Alejandro Conty Estevez, Pascal Lecocq, and Clifford Stein. It preserves detail
 * close to the shadow terminator, and doesn't "wash out" intermediate bumps using a
 * Cook-Torrance GGX function for shading. */
ccl_device_inline float bump_shadowing_term(const ccl_private ShaderData *sd,
                                            const ccl_private ShaderClosure *sc,
                                            const float3 I,
                                            const bool is_eval)
{
  if (isequal(sc->N, sd->N)) {
    return 1.0f;
  }

  /* Smoothing doesn't apply to curve geometry. */
  if (sd->type & PRIMITIVE_CURVE) {
    return 1.0f;
  }

  /* In order to avoid artifacts at the shadow terminator when using smooth normals,
   * the BSDF evaluation functions allow for light leaking through the actual geometry
   * and only checks that the directions are in the correct hemisphere w.r.t. the
   * shading normal.
   * However, when using bump/normal mapping, this can lead to light leaking not just
   * "around" the shadow terminator, but to the rear side of supposedly opaque geometry.
   * In order to detect this case, we can ensure that the direction is also valid w.r.t.
   * the smoothed (but non-bump-mapped) normal `sd->N` (or `Ns` for short below).
   *
   * `dot(Ns, I) * dot(Ns, N)` tells us if I and N are on the same side of the smoothed geometry.
   * If incoming(I) and normal(N) are on the same side we reject refractions, `dot(N, I) < 0`.
   * If they are on different sides we reject reflections, `dot(N, I) > 0`. */
  const float cosNsI = dot(sd->N, I);
  const float cosNsN = dot(sd->N, sc->N);
  const float cosNI = dot(sc->N, I);
  const bool is_diffuse = CLOSURE_IS_BSDF_DIFFUSE(sc->type);
  if (cosNsI * cosNsN * cosNI < 0.0f && (is_eval || is_diffuse)) {
    return 0.0f;
  }

  /* The above test applies to all closures, but the softening only applies to diffuse ones. */
  if (!is_diffuse) {
    return 1.0f;
  }

  /* When bump map correction is not used do skip the smoothing. */
  if ((sd->shader_flag & SD_USE_BUMP_MAP_CORRECTION) == 0) {
    return 1.0f;
  }

  /* Get absolute incoming and shader normal deviation from smoothed normal, then clamp. */
  const float cos_i = fabsf(cosNsI);
  const float cos_d = fabsf(cosNsN);
  if (cos_d >= 1.0f || cos_i >= 1.0f) {
    return 1.0f;
  }
  if (cos_i < 1e-6f) {
    return 0.0f;
  }

  /* Get GGX shading values for final smoothing. */
  const float tan2_d = 1.0f / sqr(cos_d) - 1.0f;
  const float bump_alpha2 = saturatef(0.125f * tan2_d);

  /* Return smoothed value to avoid discontinuity at perpendicular angle. */
  return bsdf_G<MicrofacetType::GGX>(bump_alpha2, cos_i);
}

ccl_device_inline float shift_cos_in(float cos_in, const float frequency_multiplier)
{
  /* Shadow terminator workaround, taken from Appleseed.
   * SPDX-License-Identifier: MIT
   * Copyright (c) 2019 Francois Beaune, The appleseedhq Organization */
  cos_in = min(cos_in, 1.0f);

  const float angle = fast_acosf(cos_in);
  const float val = max(cosf(angle * frequency_multiplier), 0.0f) / cos_in;
  return val;
}

ccl_device_inline bool bsdf_is_transmission(const ccl_private ShaderClosure *sc, const float3 wo)
{
  return dot(sc->N, wo) < 0.0f;
}

#ifdef __KERNEL_METAL_VISIBLE_SHADING__
/* Closure dispatch is compiled once as a Metal visible function, see `kernel.metal`. */
ccl_device_inline int bsdf_sample(KernelGlobals /*kg*/,
                                  ccl_private ShaderData *sd,
                                  const ccl_private ShaderClosure *sc,
                                  const float3 rand,
                                  ccl_private Spectrum *eval,
                                  ccl_private float3 *wo,
                                  ccl_private float *pdf,
                                  ccl_private float2 *sampled_roughness,
                                  ccl_private float *eta)
{
  return metal_ancillaries->vft_bsdf_sample[0](&launch_params_metal,
                                               metal_ancillaries,
                                               sd,
                                               sc,
                                               rand,
                                               eval,
                                               wo,
                                               pdf,
                                               sampled_roughness,
                                               eta);
}

ccl_device_inline int bsdf_sample_impl(KernelGlobals kg,
#else
ccl_device_inline int bsdf_sample(KernelGlobals kg,
#endif
                                  ccl_private ShaderData *sd,
                                  const ccl_private ShaderClosure *sc,
                                  const float3 rand,
                                  ccl_private Spectrum *eval,
                                  ccl_private float3 *wo,
                                  ccl_private float *pdf,
                                  ccl_private float2 *sampled_roughness,
                                  ccl_private float *eta)
{
  /* For curves use the smooth normal, particularly for ribbons the geometric
   * normal gives too much darkening otherwise. */
  *eval = zero_spectrum();
  *pdf = 0.f;
  int label = LABEL_NONE;
  const float3 Ng = (sd->type & PRIMITIVE_CURVE) ? sc->N : sd->Ng;
  const float2 rand_xy = make_float2(rand);

  switch (sc->type) {
    case CLOSURE_BSDF_DIFFUSE_ID:
      label = bsdf_diffuse_sample(sc, Ng, sd->wi, rand_xy, eval, wo, pdf);
      *sampled_roughness = one_float2();
      *eta = 1.0f;
      break;
#if defined(__SVM__) || defined(__OSL__)
    case CLOSURE_BSDF_OREN_NAYAR_ID:
      label = bsdf_oren_nayar_sample(
          sc, Ng, sd->wi, rand_xy, eval, wo, pdf, sampled_roughness, eta);
      break;
    case CLOSURE_BSDF_ROUGH_TRANSLUCENT_ID:
      label = bsdf_rough_translucent_sample(
          sc, Ng, sd->wi, rand_xy, eval, wo, pdf, sampled_roughness, eta);
      break;
#  ifdef __OSL__
    case CLOSURE_BSDF_BURLEY_ID:
      label = bsdf_burley_sample(sc, Ng, sd->wi, rand_xy, eval, wo, pdf);
      *sampled_roughness = one_float2();
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_PHONG_RAMP_ID:
      label = bsdf_phong_ramp_sample(sc, Ng, sd->wi, rand_xy, eval, wo, pdf, sampled_roughness);
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_DIFFUSE_RAMP_ID:
      label = bsdf_diffuse_ramp_sample(sc, Ng, sd->wi, rand_xy, eval, wo, pdf);
      *sampled_roughness = one_float2();
      *eta = 1.0f;
      break;
#  endif
    case CLOSURE_BSDF_TRANSLUCENT_ID:
      label = bsdf_translucent_sample(sc, Ng, sd->wi, rand_xy, eval, wo, pdf);
      *sampled_roughness = one_float2();
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_TRANSPARENT_ID:
      label = bsdf_transparent_sample(sc, Ng, sd->wi, eval, wo, pdf);
      /* Polarized mixed shaders keep null and Glass atoms in one discrete
       * measure. The common scale cancels from their native throughput. */
      if ((kernel_data.kernel_features & KERNEL_FEATURE_POLARIZATION) && sd->num_closure > 1) {
        *eval *= 1e6f; *pdf *= 1e6f;
      }
      *sampled_roughness = zero_float2();
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_RAY_PORTAL_ID:
      /* ray portals are not handled by the BSDF code, we should never get here */
      kernel_assert(false);
      break;
    case CLOSURE_BSDF_MICROFACET_GGX_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID:
      label = bsdf_microfacet_ggx_sample(
          kg, sc, Ng, sd->wi, rand, eval, wo, pdf, sampled_roughness, eta);
      break;
    case CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID:
      label = bsdf_thin_glass_transmission_sample(
          kg, sc, Ng, sd->wi, rand, eval, wo, pdf, sampled_roughness, eta);
      break;
    case CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID:
      label = bsdf_microfacet_beckmann_sample(
          kg, sc, Ng, sd->wi, rand, eval, wo, pdf, sampled_roughness, eta);
      break;
    case CLOSURE_BSDF_DIFFRACTION_BECKMANN_ID:
      label=bsdf_diffraction_dielectric_sample<BECKMANN>(kg,sc,Ng,sd->wi,rand,eval,wo,pdf,sampled_roughness,eta);
      break;
    case CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID:
      label=bsdf_diffraction_dielectric_sample(kg,sc,Ng,sd->wi,rand,eval,wo,pdf,sampled_roughness,eta);
      break;
    case CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID:
    case CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID:
      *wo=-sd->wi; *pdf=1e6f; *eval=make_spectrum(1e6f);
      if(sc->type==CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID)
        *eval*=bsdf_diffraction_coated_atom_mass(sc,sd->wi);
      *eta=1; *sampled_roughness=zero_float2();
      label=LABEL_SINGULAR|LABEL_TRANSMIT;
      break;
    case CLOSURE_BSDF_DIFFRACTION_SMOOTH_ID:
      label = bsdf_diffraction_smooth_sample(
          kg, sc, Ng, sd->wi, rand, eval, wo, pdf, sampled_roughness, eta);
      break;
    case CLOSURE_BSDF_DIFFRACTION_CONDUCTOR_GGX_ID:
    case CLOSURE_BSDF_DIFFRACTION_CONDUCTOR_BECKMANN_ID:
      label=bsdf_diffraction_conductor_sample(kg,sc,Ng,sd->wi,rand,eval,wo,pdf,sampled_roughness,eta);
      break;
    case CLOSURE_BSDF_DIFFRACTION_ASHIKHMIN_ID:
      label=bsdf_diffraction_ashikhmin_sample(sc,Ng,sd->wi,rand,eval,wo,pdf,sampled_roughness,eta);
      break;
    case CLOSURE_BSDF_DIFFRACTION_ID:
      label = bsdf_diffraction_sample(sc, Ng, sd->wi, rand, eval, wo, pdf, sampled_roughness, eta);
      break;
    case CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_REFLECTION_ID:
    case CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_TRANSMISSION_ID:
      label = bsdf_diffraction_thin_sheet_sample(
          kg, sc, Ng, sd->wi, rand, eval, wo, pdf, sampled_roughness, eta);
      break;
    case CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID:
      label = bsdf_ashikhmin_shirley_sample(
          sc, Ng, sd->wi, rand_xy, eval, wo, pdf, sampled_roughness);
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_ASHIKHMIN_VELVET_ID:
      label = bsdf_ashikhmin_velvet_sample(sc, Ng, sd->wi, rand_xy, eval, wo, pdf);
      *sampled_roughness = one_float2();
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_DIFFUSE_TOON_ID:
      label = bsdf_diffuse_toon_sample(sc, Ng, sd->wi, rand_xy, eval, wo, pdf);
      *sampled_roughness = one_float2();
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_GLOSSY_TOON_ID:
      label = bsdf_glossy_toon_sample(sc, Ng, sd->wi, rand_xy, eval, wo, pdf);
      // double check if this is valid
      *sampled_roughness = one_float2();
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_HAIR_REFLECTION_ID:
      label = bsdf_hair_reflection_sample(
          sc, Ng, sd->wi, rand_xy, eval, wo, pdf, sampled_roughness);
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_HAIR_TRANSMISSION_ID:
      label = bsdf_hair_transmission_sample(
          sc, Ng, sd->wi, rand_xy, eval, wo, pdf, sampled_roughness);
      *eta = 1.0f;
      break;
#  ifdef __PRINCIPLED_HAIR__
    case CLOSURE_BSDF_HAIR_CHIANG_ID:
      label = bsdf_hair_chiang_sample(kg, sc, sd, rand, eval, wo, pdf, sampled_roughness);
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_HAIR_HUANG_ID:
      label = bsdf_hair_huang_sample(kg, sc, sd, rand, eval, wo, pdf, sampled_roughness);
      *eta = 1.0f;
      break;
#  endif
    case CLOSURE_BSDF_SHEEN_ID:
      label = bsdf_sheen_sample(sc, Ng, sd->wi, rand_xy, eval, wo, pdf);
      *sampled_roughness = one_float2();
      *eta = 1.0f;
      break;
#endif
    default:
      label = LABEL_NONE;
      break;
  }

  /* Test if BSDF sample should be treated as transparent for background. */
  if (label & LABEL_TRANSMIT) {
    const float threshold_squared = kernel_data.background.transparent_roughness_squared_threshold;

    if (threshold_squared >= 0.0f && !(label & LABEL_DIFFUSE)) {
      if (bsdf_get_specular_roughness_squared(sc) <= threshold_squared) {
        label |= LABEL_TRANSMIT_TRANSPARENT;
      }
    }
  }
  else if (label != LABEL_NONE) {
    /* Shadow terminator offset. */
    const float frequency_multiplier =
        kernel_data_fetch(objects, sd->object).shadow_terminator_shading_offset;
    if (frequency_multiplier > 1.0f) {
      const float cosNO = dot(*wo, sc->N);
      *eval *= shift_cos_in(cosNO, frequency_multiplier);
    }
    *eval *= bump_shadowing_term(sd, sc, *wo, false);
  }

#ifdef WITH_CYCLES_DEBUG
  kernel_assert(*pdf >= 0.0f);
  kernel_assert(eval->x >= 0.0f && eval->y >= 0.0f && eval->z >= 0.0f);
#endif

  return label;
}

ccl_device_inline void bsdf_roughness_eta(const ccl_private ShaderClosure *sc,
                                          const float3 wo,
                                          ccl_private float2 *roughness,
                                          ccl_private float *eta)
{
#ifdef __SVM__
  float alpha = 1.0f;
#endif
  switch (sc->type) {
    case CLOSURE_BSDF_DIFFUSE_ID:
      *roughness = one_float2();
      *eta = 1.0f;
      break;
#ifdef __SVM__
    case CLOSURE_BSDF_OREN_NAYAR_ID:
    case CLOSURE_BSDF_ROUGH_TRANSLUCENT_ID:
      *roughness = one_float2();
      *eta = 1.0f;
      break;
#  ifdef __OSL__
    case CLOSURE_BSDF_BURLEY_ID:
      *roughness = one_float2();
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_PHONG_RAMP_ID:
      alpha = phong_ramp_exponent_to_roughness(((const ccl_private PhongRampBsdf *)sc)->exponent);
      *roughness = make_float2(alpha, alpha);
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_DIFFUSE_RAMP_ID:
      *roughness = one_float2();
      *eta = 1.0f;
      break;
#  endif
    case CLOSURE_BSDF_TRANSLUCENT_ID:
      *roughness = one_float2();
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_TRANSPARENT_ID:
    case CLOSURE_BSDF_RAY_PORTAL_ID:
      *roughness = zero_float2();
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_MICROFACET_GGX_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID:
    case CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID: {
      const ccl_private MicrofacetBsdf *bsdf = (const ccl_private MicrofacetBsdf *)sc;
      *roughness = make_float2(bsdf->alpha_x, bsdf->alpha_y);
      *eta = (bsdf_is_transmission(sc, wo)) ? bsdf->ior : 1.0f;
      break;
    }
    case CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID:
    case CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID:
      *roughness=zero_float2(); *eta=1; break;
    case CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_REFLECTION_ID:
    case CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_TRANSMISSION_ID: {
      const ccl_private DiffractionThinSheetBsdf *b =
          (const ccl_private DiffractionThinSheetBsdf *)sc;
      *roughness = b->return_only ? one_float2() : make_float2(b->port.alpha, b->port.alpha);
      *eta = 1.0f;
      break;
    }
    case CLOSURE_BSDF_DIFFRACTION_BECKMANN_ID:
    case CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID: {
      const ccl_private DiffractionDielectricBsdf *b=(const ccl_private DiffractionDielectricBsdf *)sc;
      *roughness=make_float2(diffraction_dielectric_param(b)->alpha_x,diffraction_dielectric_param(b)->alpha_y);
      *eta=bsdf_is_transmission(sc,wo)?diffraction_dielectric_param(b)->facet.transmitted_ior/diffraction_dielectric_param(b)->facet.incident_ior:1;
      break;
    }
    case CLOSURE_BSDF_DIFFRACTION_SMOOTH_ID: {
      ccl_private const DiffractionSmoothBsdf *bsdf = (ccl_private const DiffractionSmoothBsdf *)sc;
      *roughness = zero_float2();
      *eta = bsdf_is_transmission(sc, wo) ?
                 (bsdf->incoming_substrate ? bsdf->upper_index / bsdf->lower_index :
                                             bsdf->lower_index / bsdf->upper_index) :
                 1;
      break;
    }
    case CLOSURE_BSDF_DIFFRACTION_CONDUCTOR_GGX_ID:
    case CLOSURE_BSDF_DIFFRACTION_CONDUCTOR_BECKMANN_ID: {
      const ccl_private DiffractionConductorBsdf *b=(const ccl_private DiffractionConductorBsdf *)sc;
      *roughness=make_float2(b->extra->param.alpha_x,b->extra->param.alpha_y);*eta=1;
      break;
    }
    case CLOSURE_BSDF_DIFFRACTION_ID: {
      const ccl_private DiffractionBsdf *bsdf = (const ccl_private DiffractionBsdf *)sc;
      *roughness = make_float2(bsdf->param.alpha_x, bsdf->param.alpha_y);
      *eta = 1.0f;
      break;
    }
    case CLOSURE_BSDF_DIFFRACTION_ASHIKHMIN_ID: {
      const ccl_private DiffractionBsdf *bsdf = (const ccl_private DiffractionBsdf *)sc;
      *roughness = make_float2(bsdf->param.alpha_x, bsdf->param.alpha_y);
      *eta = 1.0f;
      break;
    }
    case CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID: {
      const ccl_private MicrofacetBsdf *bsdf = (const ccl_private MicrofacetBsdf *)sc;
      *roughness = make_float2(bsdf->alpha_x, bsdf->alpha_y);
      *eta = 1.0f;
      break;
    }
    case CLOSURE_BSDF_ASHIKHMIN_VELVET_ID:
      *roughness = one_float2();
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_DIFFUSE_TOON_ID:
      *roughness = one_float2();
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_GLOSSY_TOON_ID:
      // double check if this is valid
      *roughness = one_float2();
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_HAIR_REFLECTION_ID:
      *roughness = make_float2(((ccl_private HairBsdf *)sc)->roughness1,
                               ((ccl_private HairBsdf *)sc)->roughness2);
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_HAIR_TRANSMISSION_ID:
      *roughness = make_float2(((ccl_private HairBsdf *)sc)->roughness1,
                               ((ccl_private HairBsdf *)sc)->roughness2);
      *eta = 1.0f;
      break;
#  ifdef __PRINCIPLED_HAIR__
    case CLOSURE_BSDF_HAIR_CHIANG_ID:
      alpha = ((ccl_private ChiangHairBSDF *)sc)->m0_roughness;
      *roughness = make_float2(alpha, alpha);
      *eta = 1.0f;
      break;
    case CLOSURE_BSDF_HAIR_HUANG_ID:
      alpha = ((ccl_private HuangHairBSDF *)sc)->roughness;
      *roughness = make_float2(alpha, alpha);
      *eta = 1.0f;
      break;
#  endif
    case CLOSURE_BSDF_SHEEN_ID:
      alpha = ((ccl_private SheenBsdf *)sc)->roughness;
      *roughness = make_float2(alpha, alpha);
      *eta = 1.0f;
      break;
#endif
    default:
      *roughness = one_float2();
      *eta = 1.0f;
      break;
  }
}

ccl_device_inline int bsdf_label(const KernelGlobals kg,
                                 const ccl_private ShaderClosure *sc,
                                 const float3 wo)
{
  /* For curves use the smooth normal, particularly for ribbons the geometric
   * normal gives too much darkening otherwise. */
  int label;
  switch (sc->type) {
    case CLOSURE_BSDF_DIFFUSE_ID:
    case CLOSURE_BSSRDF_BURLEY_ID:
    case CLOSURE_BSSRDF_RANDOM_WALK_ID:
    case CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID:
    case CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID:
      label = LABEL_REFLECT | LABEL_DIFFUSE;
      break;
#ifdef __SVM__
    case CLOSURE_BSDF_OREN_NAYAR_ID:
      label = LABEL_REFLECT | LABEL_DIFFUSE;
      break;
#  ifdef __OSL__
    case CLOSURE_BSDF_BURLEY_ID:
      label = LABEL_REFLECT | LABEL_DIFFUSE;
      break;
    case CLOSURE_BSDF_PHONG_RAMP_ID:
      label = LABEL_REFLECT | LABEL_GLOSSY;
      break;
    case CLOSURE_BSDF_DIFFUSE_RAMP_ID:
      label = LABEL_REFLECT | LABEL_DIFFUSE;
      break;
#  endif
    case CLOSURE_BSDF_TRANSLUCENT_ID:
    case CLOSURE_BSDF_ROUGH_TRANSLUCENT_ID:
      label = LABEL_TRANSMIT | LABEL_DIFFUSE;
      break;
    case CLOSURE_BSDF_TRANSPARENT_ID:
      label = LABEL_TRANSMIT | LABEL_TRANSPARENT;
      break;
    case CLOSURE_BSDF_RAY_PORTAL_ID:
      label = LABEL_TRANSMIT | LABEL_RAY_PORTAL;
      break;
    case CLOSURE_BSDF_MICROFACET_GGX_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID: {
      const ccl_private MicrofacetBsdf *bsdf = (const ccl_private MicrofacetBsdf *)sc;
      label = ((bsdf_is_transmission(sc, wo)) ? LABEL_TRANSMIT : LABEL_REFLECT) |
              ((bsdf_microfacet_eval_flag(bsdf)) ? LABEL_GLOSSY : LABEL_SINGULAR);
      break;
    }
    case CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID:
      label = LABEL_TRANSMIT | LABEL_GLOSSY;
      break;
    case CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID:
    case CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID:
      label=LABEL_SINGULAR|LABEL_TRANSMIT; break;
    case CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_REFLECTION_ID:
    case CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_TRANSMISSION_ID:
      label = bsdf_diffraction_thin_sheet_label(sc);
      break;
    case CLOSURE_BSDF_DIFFRACTION_BECKMANN_ID:
    case CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID: {
      const ccl_private DiffractionDielectricBsdf *b=(const ccl_private DiffractionDielectricBsdf *)sc;
      label=(bsdf_is_transmission(sc,wo)?LABEL_TRANSMIT:LABEL_REFLECT)|
          ((b->disabled_lobes & DIFFRACTION_DIELECTRIC_TWO_SIDED_MS) ||
           !roughness_is_almost_specular(diffraction_dielectric_param(b)->alpha_x,
                                         diffraction_dielectric_param(b)->alpha_y) ?
               LABEL_GLOSSY : LABEL_SINGULAR);
      break;
    }
    case CLOSURE_BSDF_DIFFRACTION_SMOOTH_ID:
      label = LABEL_SINGULAR | (bsdf_is_transmission(sc, wo) ? LABEL_TRANSMIT : LABEL_REFLECT);
      break;
    case CLOSURE_BSDF_DIFFRACTION_CONDUCTOR_GGX_ID:
    case CLOSURE_BSDF_DIFFRACTION_CONDUCTOR_BECKMANN_ID:
      label=bsdf_diffraction_conductor_label(sc);break;
    case CLOSURE_BSDF_DIFFRACTION_ASHIKHMIN_ID: {
      const ccl_private DiffractionBsdf *b=(const ccl_private DiffractionBsdf *)sc;
      label=LABEL_REFLECT | (max(b->param.alpha_x,b->param.alpha_y)<=1e-4f ? LABEL_SINGULAR : LABEL_GLOSSY);
      break;
    }
    case CLOSURE_BSDF_DIFFRACTION_ID:
      label = bsdf_diffraction_label((const ccl_private DiffractionBsdf *)sc);
      break;
    case CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID:
      label = LABEL_REFLECT | LABEL_GLOSSY;
      break;
    case CLOSURE_BSDF_ASHIKHMIN_VELVET_ID:
      label = LABEL_REFLECT | LABEL_DIFFUSE;
      break;
    case CLOSURE_BSDF_DIFFUSE_TOON_ID:
      label = LABEL_REFLECT | LABEL_DIFFUSE;
      break;
    case CLOSURE_BSDF_GLOSSY_TOON_ID:
      label = LABEL_REFLECT | LABEL_GLOSSY;
      break;
    case CLOSURE_BSDF_HAIR_REFLECTION_ID:
      label = LABEL_REFLECT | LABEL_GLOSSY;
      break;
    case CLOSURE_BSDF_HAIR_TRANSMISSION_ID:
      label = LABEL_TRANSMIT | LABEL_GLOSSY;
      break;
#  ifdef __PRINCIPLED_HAIR__
    case CLOSURE_BSDF_HAIR_CHIANG_ID:
      if (bsdf_is_transmission(sc, wo)) {
        label = LABEL_TRANSMIT | LABEL_GLOSSY;
      }
      else {
        label = LABEL_REFLECT | LABEL_GLOSSY;
      }
      break;
    case CLOSURE_BSDF_HAIR_HUANG_ID:
      label = LABEL_REFLECT | LABEL_GLOSSY;
      break;
#  endif
    case CLOSURE_BSDF_SHEEN_ID:
      label = LABEL_REFLECT | LABEL_DIFFUSE;
      break;
#endif
    default:
      label = LABEL_NONE;
      break;
  }

  /* Test if BSDF sample should be treated as transparent for background. */
  if (label & LABEL_TRANSMIT) {
    const float threshold_squared = kernel_data.background.transparent_roughness_squared_threshold;

    if (threshold_squared >= 0.0f) {
      if (bsdf_get_specular_roughness_squared(sc) <= threshold_squared) {
        label |= LABEL_TRANSMIT_TRANSPARENT;
      }
    }
  }
  return label;
}

ccl_device_inline bool bsdf_microfacet_has_delta(const ccl_private ShaderClosure *sc)
{
  if (bsdf_is_diffraction_conductor(sc->type))
    return (bsdf_diffraction_conductor_label(sc)&LABEL_SINGULAR)!=0;
  switch (sc->type) {
    case CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID:
    case CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID: return true;
    case CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_REFLECTION_ID:
    case CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_TRANSMISSION_ID:
      return (bsdf_diffraction_thin_sheet_label(sc) & LABEL_SINGULAR) != 0;
    case CLOSURE_BSDF_DIFFRACTION_BECKMANN_ID:
    case CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID: {
      const ccl_private DiffractionDielectricBsdf *b=(const ccl_private DiffractionDielectricBsdf *)sc;
      return !(b->disabled_lobes & DIFFRACTION_DIELECTRIC_TWO_SIDED_MS) &&
             roughness_is_almost_specular(diffraction_dielectric_param(b)->alpha_x,
                                          diffraction_dielectric_param(b)->alpha_y);
    }

    case CLOSURE_BSDF_MICROFACET_GGX_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID:
    case CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID: {
      ccl_private const MicrofacetBsdf *bsdf = (ccl_private const MicrofacetBsdf *)sc;
      return roughness_is_almost_specular(bsdf->alpha_x, bsdf->alpha_y);
    }
    default:
      return false;
  }
}

/* Exactly matched indices have a true straight transmission atom even at
 * finite roughness. Reflection is classified separately and remains rough. */
ccl_device_inline bool bsdf_microfacet_is_matched_transmission_atom(
    const ccl_private ShaderClosure *sc,const float3 wo)
{
  if (!(CLOSURE_IS_GLASS(sc->type) || CLOSURE_IS_REFRACTION(sc->type)) ||
      !CLOSURE_IS_BSDF_MICROFACET(sc->type) || dot(sc->N,wo)>=0) return false;
  return ((const ccl_private MicrofacetBsdf *)sc)->ior==1.0f;
}

/* Evaluate a known grating or smooth microfacet atom for sampled-event mixing.
 * This is a probability mass, scaled by Cycles' delta convention; it must not
 * be used as a solid-angle density for an arbitrary connection direction. */
#ifdef __KERNEL_METAL_VISIBLE_SHADING__
/* The grating atoms are compiled once as a Metal visible function, see `kernel.metal`. */
ccl_device_inline Spectrum bsdf_eval_delta(KernelGlobals /*kg*/,
                                           ccl_private ShaderData *sd,
                                           ccl_private const ShaderClosure *sc,
                                           const float3 wo,
                                           ccl_private float *pdf)
{
  return metal_ancillaries->vft_bsdf_eval_delta[0](
      &launch_params_metal, metal_ancillaries, sd, sc, wo, pdf);
}
#endif

#ifdef __KERNEL_METAL__
/* Keep the grating matrix workspace out of every specialized mixture caller. */
ccl_device __attribute__((noinline))
#else
ccl_device_inline
#endif
Spectrum
#ifdef __KERNEL_METAL_VISIBLE_SHADING__
bsdf_eval_delta_impl(KernelGlobals kg,
#else
bsdf_eval_delta(KernelGlobals kg,
#endif
                                                       ccl_private ShaderData *sd,
                                                       ccl_private const ShaderClosure *sc,
                                                       const float3 wo,
                                                       ccl_private float *pdf)
{
  *pdf = 0.0f;
  const bool transmission = dot(sc->N, wo) < 0.0f;
  const float3 Ng = (sd->type & PRIMITIVE_CURVE) ? sc->N : sd->Ng;
  if ((dot(Ng, wo) < 0.0f) != transmission) {
    return zero_spectrum();
  }
  Spectrum eval = zero_spectrum();
  if (bsdf_is_diffraction_conductor(sc->type)) {
    eval=bsdf_diffraction_conductor_delta(kg,sc,sd->wi,wo,pdf);
  }
  else if (sc->type == CLOSURE_BSDF_DIFFRACTION_ASHIKHMIN_ID) {
    eval=bsdf_diffraction_ashikhmin_delta(sc,sd->wi,wo,pdf);
  }
  else if (sc->type == CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_REFLECTION_ID ||
           sc->type == CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_TRANSMISSION_ID) {
    eval = bsdf_diffraction_thin_sheet_delta(kg, sc, sd->wi, wo, pdf);
  }
  else if (sc->type == CLOSURE_BSDF_DIFFRACTION_SMOOTH_ID) {
    float power, probability;
    if (!bsdf_diffraction_smooth_delta_probability(kg, sc, sd->wi, wo, &power, &probability)) {
      return zero_spectrum();
    }
    *pdf = probability * 1e6f;
    eval = make_spectrum(power * 1e6f);
  }
  else if (sc->type == CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID ||
           sc->type == CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID) {
    if (len_squared(wo+sd->wi)>16e-12f) return zero_spectrum();
    *pdf=1e6f; eval=make_spectrum(1e6f);
    if(sc->type==CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID)
      eval*=bsdf_diffraction_coated_atom_mass(sc,sd->wi);
  }
  else if ((sc->type == CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID || sc->type == CLOSURE_BSDF_DIFFRACTION_BECKMANN_ID)) {
    if (((const ccl_private DiffractionDielectricBsdf *)sc)->disabled_lobes &
        DIFFRACTION_DIELECTRIC_TWO_SIDED_MS)
    {
      return zero_spectrum();
    }
    float value_mass;
    *pdf=1e6f*bsdf_diffraction_dielectric_delta_mass(sc,sd->wi,wo,&value_mass);
    eval=make_spectrum(value_mass*1e6f)*diffraction_dielectric_tint(
        (const ccl_private DiffractionDielectricBsdf *)sc,bsdf_is_transmission(sc,wo));
    if (*pdf>0 && (((const ccl_private DiffractionDielectricBsdf *)sc)->disabled_lobes &
                  DIFFRACTION_DIELECTRIC_GENERALIZED))
      eval=1e6f*bsdf_diffraction_dielectric_generalized_delta_value(sc,sd->wi,wo);
  }
  else if (sc->type == CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID) {
    if (!bsdf_microfacet_has_delta(sc) || len_squared(wo + sd->wi) > 16e-12f) {
      return zero_spectrum();
    }
    *pdf = 1e6f;
    eval = make_spectrum(1e6f);
  }
  else {
    if (!bsdf_microfacet_has_delta(sc) &&
        !((kernel_data.kernel_features & KERNEL_FEATURE_POLARIZATION) &&
          bsdf_microfacet_is_matched_transmission_atom(sc,wo))) return zero_spectrum();
    ccl_private const MicrofacetBsdf *bsdf = (ccl_private const MicrofacetBsdf *)sc;
    const float cosine = dot(sc->N, sd->wi);
    if (!(cosine > 0.0f)) {
      return zero_spectrum();
    }
    float transmitted_cosine;
    const FresnelCoeff coeff = microfacet_fresnel(kg, bsdf, cosine, &transmitted_cosine);
    const float total = average(coeff.sum());
    const Spectrum branch = transmission ? coeff.transmittance : coeff.reflectance;
    if (!(total > 0.0f) || !(average(branch) > 0.0f)) {
      return zero_spectrum();
    }
    const float3 direction = transmission ?
        refract_angle(sd->wi, sc->N, transmitted_cosine, safe_divide(1.0f, bsdf->ior)) :
        2.0f * cosine * sc->N - sd->wi;
    if (len_squared(direction - wo) > 16e-12f) {
      return zero_spectrum();
    }
    *pdf = average(branch) / total * 1e6f;
    eval = branch * 1e6f;
  }
  if (!transmission) {
    const float frequency_multiplier =
        kernel_data_fetch(objects, sd->object).shadow_terminator_shading_offset;
    if (frequency_multiplier > 1.0f) {
      eval *= shift_cos_in(dot(wo, sc->N), frequency_multiplier);
    }
    eval *= bump_shadowing_term(sd, sc, wo, false);
  }
  return eval;
}

#ifdef __KERNEL_METAL_VISIBLE_SHADING__
/* Closure dispatch is compiled once as a Metal visible function, see `kernel.metal`. */
ccl_device_inline Spectrum bsdf_eval(KernelGlobals /*kg*/,
                                     ccl_private ShaderData *sd,
                                     const ccl_private ShaderClosure *sc,
                                     const float3 wo,
                                     ccl_private float *pdf)
{
  return metal_ancillaries->vft_bsdf_eval[0](
      &launch_params_metal, metal_ancillaries, sd, sc, wo, pdf);
}
#endif

#if defined(__KERNEL_METAL_VISIBLE_SHADING__)
ccl_device_inline
#elif defined(__KERNEL_METAL__) && defined(__KERNEL_METAL_TRANSPORT_FEATURES__) && \
    (__KERNEL_METAL_TRANSPORT_FEATURES__ & KERNEL_FEATURE_BDPT) && \
    (__KERNEL_METAL_TRANSPORT_FEATURES__ & KERNEL_FEATURE_PATH_GUIDING)
/* Joint transport queries this dispatch repeatedly for forward/reverse PDFs and
 * guide mixtures. Avoid duplicating the entire closure graph in every caller. */
ccl_device __attribute__((noinline))
#elif !defined(__KERNEL_CUDA__)
ccl_device
#else
ccl_device_inline
#endif
    Spectrum
#ifdef __KERNEL_METAL_VISIBLE_SHADING__
    bsdf_eval_impl(KernelGlobals kg,
#else
    bsdf_eval(KernelGlobals kg,
#endif
              ccl_private ShaderData *sd,
              const ccl_private ShaderClosure *sc,
              const float3 wo,
              ccl_private float *pdf)
{
  Spectrum eval = zero_spectrum();
  *pdf = 0.f;

  const float bump_shadowing = bump_shadowing_term(sd, sc, wo, true);
  if (bump_shadowing == 0.0f) {
    return zero_spectrum();
  }

  switch (sc->type) {
    case CLOSURE_BSDF_DIFFUSE_ID:
      eval = bsdf_diffuse_eval(sc, sd->wi, wo, pdf);
      break;
#if defined(__SVM__) || defined(__OSL__)
    case CLOSURE_BSDF_OREN_NAYAR_ID:
      eval = bsdf_oren_nayar_eval(sc, sd->wi, wo, pdf);
      break;
    case CLOSURE_BSDF_ROUGH_TRANSLUCENT_ID:
      eval = bsdf_rough_translucent_eval(sc, sd->wi, wo, pdf);
      break;
#  ifdef __OSL__
    case CLOSURE_BSDF_BURLEY_ID:
      eval = bsdf_burley_eval(sc, sd->wi, wo, pdf);
      break;
    case CLOSURE_BSDF_PHONG_RAMP_ID:
      eval = bsdf_phong_ramp_eval(sc, sd->wi, wo, pdf);
      break;
    case CLOSURE_BSDF_DIFFUSE_RAMP_ID:
      eval = bsdf_diffuse_ramp_eval(sc, sd->wi, wo, pdf);
      break;
#  endif
    case CLOSURE_BSDF_TRANSLUCENT_ID:
      eval = bsdf_translucent_eval(sc, sd->wi, wo, pdf);
      break;
    case CLOSURE_BSDF_TRANSPARENT_ID:
      eval = bsdf_transparent_eval(sc, sd->wi, wo, pdf);
      break;
    case CLOSURE_BSDF_RAY_PORTAL_ID:
      eval = bsdf_ray_portal_eval(sc, sd->wi, wo, pdf);
      break;
    case CLOSURE_BSDF_MICROFACET_GGX_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID:
      eval = bsdf_microfacet_ggx_eval(kg, sc, sd->wi, wo, pdf);
      break;
    case CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID:
      eval = bsdf_thin_glass_transmission_eval(kg, sc, sd->wi, wo, pdf);
      break;
    case CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID:
      eval = bsdf_microfacet_beckmann_eval(kg, sc, sd->wi, wo, pdf);
      break;
    case CLOSURE_BSDF_DIFFRACTION_BECKMANN_ID:
      eval=bsdf_diffraction_dielectric_eval<BECKMANN>(kg,sc,sd->wi,wo,pdf); break;
    case CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID:
      eval=bsdf_diffraction_dielectric_eval(kg,sc,sd->wi,wo,pdf); break;
    case CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID:
    case CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID:
    case CLOSURE_BSDF_DIFFRACTION_SMOOTH_ID:
      *pdf = 0;
      eval = zero_spectrum();
      break;
    case CLOSURE_BSDF_DIFFRACTION_CONDUCTOR_GGX_ID:
    case CLOSURE_BSDF_DIFFRACTION_CONDUCTOR_BECKMANN_ID:
      eval=bsdf_diffraction_conductor_eval(kg,sc,sd->wi,wo,pdf);break;
    case CLOSURE_BSDF_DIFFRACTION_ASHIKHMIN_ID:
      eval=bsdf_diffraction_ashikhmin_eval(sc,sd->wi,wo,pdf);break;
    case CLOSURE_BSDF_DIFFRACTION_ID:
      eval = bsdf_diffraction_eval(sc, sd->wi, wo, pdf);
      break;
    case CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_REFLECTION_ID:
    case CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_TRANSMISSION_ID:
      eval = bsdf_diffraction_thin_sheet_eval(kg, sc, sd->wi, wo, pdf);
      break;
    case CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID:
      eval = bsdf_ashikhmin_shirley_eval(sc, sd->wi, wo, pdf);
      break;
    case CLOSURE_BSDF_ASHIKHMIN_VELVET_ID:
      eval = bsdf_ashikhmin_velvet_eval(sc, sd->wi, wo, pdf);
      break;
    case CLOSURE_BSDF_DIFFUSE_TOON_ID:
      eval = bsdf_diffuse_toon_eval(sc, sd->wi, wo, pdf);
      break;
    case CLOSURE_BSDF_GLOSSY_TOON_ID:
      eval = bsdf_glossy_toon_eval(sc, sd->wi, wo, pdf);
      break;
#  ifdef __PRINCIPLED_HAIR__
    case CLOSURE_BSDF_HAIR_CHIANG_ID:
      eval = bsdf_hair_chiang_eval(kg, sd, sc, wo, pdf);
      break;
    case CLOSURE_BSDF_HAIR_HUANG_ID:
      eval = bsdf_hair_huang_eval(kg, sd, sc, wo, pdf);
      break;
#  endif
    case CLOSURE_BSDF_HAIR_REFLECTION_ID:
      eval = bsdf_hair_reflection_eval(sc, sd->wi, wo, pdf);
      break;
    case CLOSURE_BSDF_HAIR_TRANSMISSION_ID:
      eval = bsdf_hair_transmission_eval(sc, sd->wi, wo, pdf);
      break;
    case CLOSURE_BSDF_SHEEN_ID:
      eval = bsdf_sheen_eval(sc, sd->wi, wo, pdf);
      break;
#endif
    default:
      break;
  }

  eval *= bump_shadowing;

  /* Shadow terminator offset. */
  const float frequency_multiplier =
      kernel_data_fetch(objects, sd->object).shadow_terminator_shading_offset;
  if (frequency_multiplier > 1.0f) {
    const float cosNO = dot(wo, sc->N);
    if (cosNO >= 0.0f) {
      eval *= shift_cos_in(cosNO, frequency_multiplier);
    }
  }

#ifdef WITH_CYCLES_DEBUG
  kernel_assert(*pdf >= 0.0f);
  kernel_assert(eval.x >= 0.0f && eval.y >= 0.0f && eval.z >= 0.0f);
#endif
  return eval;
}

ccl_device void bsdf_blur(ccl_private ShaderClosure *sc, const float roughness)
{
  /* TODO: do we want to blur volume closures? */
#if defined(__SVM__) || defined(__OSL__)
  switch (sc->type) {
    case CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID: {
      ccl_private DiffractionCoatedAtomBsdf *bsdf=(ccl_private DiffractionCoatedAtomBsdf *)sc;
      bsdf->extra->param.alpha_x=max(bsdf->extra->param.alpha_x,roughness);
      bsdf->extra->param.alpha_y=max(bsdf->extra->param.alpha_y,roughness);
      break;
    }
    case CLOSURE_BSDF_MICROFACET_GGX_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID:
    case CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID:
      /* TODO: Recompute energy preservation after blur? */
      bsdf_microfacet_blur(sc, roughness);
      break;
    case CLOSURE_BSDF_DIFFRACTION_BECKMANN_ID:
    case CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID: {
      ccl_private DiffractionDielectricBsdf *b=(ccl_private DiffractionDielectricBsdf *)sc;
      if (b->disabled_lobes & (DIFFRACTION_DIELECTRIC_TWO_SIDED_MS |
                               DIFFRACTION_DIELECTRIC_CACHE_BASE))
      {
        break;
      }
      ccl_private DiffractionRoughDielectric *p=(b->disabled_lobes & DIFFRACTION_DIELECTRIC_TINT_EXTRA)?&b->extra->param:&b->param;
      p->alpha_x=max(p->alpha_x,roughness);
      p->alpha_y=max(p->alpha_y,roughness);
      break;
    }
    case CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_REFLECTION_ID:
    case CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_TRANSMISSION_ID: {
      ccl_private DiffractionThinSheetBsdf *b = (ccl_private DiffractionThinSheetBsdf *)sc;
      if (b->extra==nullptr) b->port.alpha = max(b->port.alpha, roughness);
      break;
    }
    case CLOSURE_BSDF_DIFFRACTION_CONDUCTOR_GGX_ID:
    case CLOSURE_BSDF_DIFFRACTION_CONDUCTOR_BECKMANN_ID: {
      ccl_private DiffractionConductorBsdf *b=(ccl_private DiffractionConductorBsdf *)sc;
      b->extra->param.alpha_x=max(b->extra->param.alpha_x,roughness);
      b->extra->param.alpha_y=max(b->extra->param.alpha_y,roughness);
      b->extra->carrier.alpha_x=b->extra->param.alpha_x;
      b->extra->carrier.alpha_y=b->extra->param.alpha_y;
      break;
    }
    case CLOSURE_BSDF_DIFFRACTION_ID: {
      ccl_private DiffractionBsdf *bsdf = (ccl_private DiffractionBsdf *)sc;
      bsdf->param.alpha_x = max(bsdf->param.alpha_x, roughness);
      bsdf->param.alpha_y = max(bsdf->param.alpha_y, roughness);
      break;
    }
    case CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID:
      bsdf_ashikhmin_shirley_blur(sc, roughness);
      break;
#  ifdef __PRINCIPLED_HAIR__
    case CLOSURE_BSDF_HAIR_CHIANG_ID:
      bsdf_hair_chiang_blur(sc, roughness);
      break;
    case CLOSURE_BSDF_HAIR_HUANG_ID:
      bsdf_hair_huang_blur(sc, roughness);
      break;
#  endif
    default:
      break;
  }
#endif
}

ccl_device_inline Spectrum bsdf_albedo(KernelGlobals kg,
                                       const ccl_private ShaderData *sd,
                                       const ccl_private ShaderClosure *sc,
                                       const bool reflection,
                                       const bool transmission)
{
  Spectrum albedo = one_spectrum();
  /* Some closures include additional components such as Fresnel terms that cause their albedo to
   * be below 1. The point of this function is to return a best-effort estimation of their albedo,
   * meaning the amount of reflected/refracted light that would be expected when illuminated by a
   * uniform white background.
   * This is used for the denoising albedo pass and diffuse/glossy/transmission color passes.
   * NOTE: This should always match the sample_weight of the closure - as in, if there's an albedo
   * adjustment in here, the sample_weight should also be reduced accordingly.
   * TODO(lukas): Consider calling this function to determine the sample_weight? Would be a bit of
   * extra overhead though. */
#if defined(__SVM__) || defined(__OSL__)
  if (sc->type == CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_REFLECTION_ID ||
      sc->type == CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_TRANSMISSION_ID) {
    const ccl_private DiffractionThinSheetBsdf *b=(const ccl_private DiffractionThinSheetBsdf *)sc;
    const bool enabled=b->port.transmission ? transmission : reflection;
    if (!enabled) return zero_spectrum();
    if (b->extra==nullptr) return one_spectrum();
    ccl_private const DiffractionThinSheetExtra *e=b->extra;
    float3 X,Y;make_orthonormals_safe_tangent(b->N,b->T,&X,&Y);
    const float3 I=make_float3(dot(sd->wi,X),dot(sd->wi,Y),dot(sd->wi,b->N));
    if (b->return_only) {
      Spectrum avg;
      return diffraction_thin_sheet_cached_missing(kg,e,I,&avg)*(0.5f*e->blend);
    }
    Spectrum coefficient=zero_spectrum();
    for (int c=0;c<3;++c) {
      const DiffractionThinSheetModel model=diffraction_thin_sheet_channel_model(e,c);
      const float2 coeff=diffraction_thin_sheet_model_coefficients(&model,I.z);
      const float q=diffraction_albedo_lookup(kg,e->albedo_handles[c],e->wavelength_nm,make_float3(fabsf(I.x),fabsf(I.y),I.z)).x;
      const float escaped=coeff.x+coeff.y>0.0f ? max(0.0f,1.0f-q/(coeff.x+coeff.y)) : 0.0f;
      coefficient[c]=(b->port.transmission ? coeff.y : coeff.x)*escaped;
    }
    return mix(b->port.transmission ? e->native_transmission : e->native_reflection,
               coefficient,e->blend);
  }
  if (sc->type == CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID ||
      sc->type == CLOSURE_BSDF_DIFFRACTION_BECKMANN_ID)
  {
    const ccl_private DiffractionDielectricBsdf *b =
        (const ccl_private DiffractionDielectricBsdf *)sc;
    if (b->disabled_lobes & DIFFRACTION_DIELECTRIC_TWO_SIDED_MS) {
      float3 X, Y;
      bsdf_diffraction_dielectric_frame(b, &X, &Y);
      const float3 I = make_float3(dot(sd->wi, X), dot(sd->wi, Y), dot(sd->wi, b->N));
      return diffraction_dielectric_ms_albedo(kg, b, I, reflection, transmission);
    }
    const bool reflect = reflection && !(b->disabled_lobes & LABEL_REFLECT);
    const bool transmit = transmission && !(b->disabled_lobes & LABEL_TRANSMIT);
    if (!reflect && !transmit) {
      return zero_spectrum();
    }
    /* Like native microfacet film albedo, use a smooth-interface estimate.
     * This preserves spectral lobe weights without an order loop, but does not
     * model rough masking or redistribution of evanescent grating orders. */
    const ccl_private DiffractionRoughDielectric *p = diffraction_dielectric_param(b);
    const float cosine = clamp(dot(sd->wi, b->N), 0.0f, 1.0f);
    const ccl_private DiffractionDielectricCoatingExtra *coating =
        diffraction_dielectric_coating(b);
    Spectrum F;
    if (b->disabled_lobes & DIFFRACTION_DIELECTRIC_GENERALIZED) {
      const ccl_private DiffractionDielectricGeneralizedExtra *e =
          (const ccl_private DiffractionDielectricGeneralizedExtra *)b->extra;
      F = diffraction_dielectric_generalized_fresnel(
          e, diffraction_dielectric_generalized_interface(e, cosine, false));
    }
    else if (b->disabled_lobes & DIFFRACTION_DIELECTRIC_PURE_REFRACTION) {
      F = zero_spectrum();
    }
    else {
      F = make_spectrum(coating ? diffraction_thin_film_reflectance(
          cosine, p->facet.incident_ior, p->facet.transmitted_ior,
          coating->film_ior, coating->film_thickness_over_wavelength) :
          fresnel_dielectric(cosine, p->facet.transmitted_ior / p->facet.incident_ior,
                             nullptr));
    }
    Spectrum T = one_spectrum() - F;
    float normalization = 1.0f;
    if (coating || (b->disabled_lobes & DIFFRACTION_DIELECTRIC_GENERALIZED)) {
      /* Matched-index transmission belongs to the separate straight atom. */
      if (p->facet.incident_ior == p->facet.transmitted_ior) {
        T = zero_spectrum();
      }
    }
    else {
      const float atom = diffraction_dielectric_straight_mass(p);
      T = max(T - make_spectrum(atom), zero_spectrum());
      normalization = 1.0f - atom;
    }
    albedo = reflect ? F * diffraction_dielectric_tint(b, false) : zero_spectrum();
    if (transmit) {
      albedo += T * diffraction_dielectric_tint(b, true);
    }
    albedo = normalization > 0.0f ? albedo / normalization : zero_spectrum();
  }
  else if (sc->type == CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID) {
    albedo = transmission ? one_spectrum() : zero_spectrum();
  }
  else if (sc->type == CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID) {
    albedo = transmission ? make_spectrum(bsdf_diffraction_coated_atom_mass(sc, sd->wi)) :
                            zero_spectrum();
  }
  else if (bsdf_is_diffraction_conductor(sc->type)) {
    const ccl_private DiffractionConductorBsdf *b=(const ccl_private DiffractionConductorBsdf *)sc;
    albedo=bsdf_microfacet_estimate_albedo(kg,sd->wi,&b->extra->carrier,reflection,transmission);
    if (reflection && b->extra->albedo_handle >= 0) {
      float3 X, Y;
      make_orthonormals_safe_tangent(b->N, b->T, &X, &Y);
      const float3 I = make_float3(dot(sd->wi, X), dot(sd->wi, Y), dot(sd->wi, b->N));
      if (I.z > 0.0f) {
        const float2 cached = diffraction_albedo_lookup(
            kg, b->extra->albedo_handle, b->extra->wavelength_nm, I);
        const float missing = 1.0f - cached.x;
        const float average_missing = 1.0f - cached.y;
        if (average_missing > 1.0e-7f) {
          const Spectrum color = b->extra->multiscatter_color;
          Spectrum multiple = safe_divide(
              color * cached.y, one_spectrum() - color * average_missing);
          if (b->extra->carrier.fresnel_type != MicrofacetFresnel::NONE) {
            multiple *= color;
          }
          albedo += multiple * missing;
        }
      }
    }
  }
  else if (CLOSURE_IS_BSDF_MICROFACET(sc->type)) {
    albedo = bsdf_microfacet_estimate_albedo(
        kg, sd->wi, (const ccl_private MicrofacetBsdf *)sc, reflection, transmission);
  }
#  ifdef __PRINCIPLED_HAIR__
  else if (sc->type == CLOSURE_BSDF_HAIR_CHIANG_ID) {
    /* TODO(lukas): Principled Hair could also be split into a glossy and a transmission component,
     * similar to Glass BSDFs. */
    albedo = bsdf_hair_chiang_albedo(sd, sc);
  }
  else if (sc->type == CLOSURE_BSDF_HAIR_HUANG_ID) {
    albedo = bsdf_hair_huang_albedo(sd, sc);
  }
#  endif
#endif
  return albedo;
}

ccl_device_inline Spectrum closure_albedo(KernelGlobals kg,
                                          const ccl_private ShaderData *sd,
                                          const ccl_private ShaderClosure *sc,
                                          const bool reflection,
                                          const bool transmission)
{
  return sc->weight * bsdf_albedo(kg, sd, sc, reflection, transmission);
}

/* Compute albedo used for layering this BSDF on top of another. The albedo is usually the
 * reflection albedo. */
ccl_device_inline_outline_metal Spectrum closure_layer_albedo(KernelGlobals kg,
                                                const ccl_private ShaderData *sd,
                                                const ccl_private ShaderClosure *sc)
{
  /* Use `reduce_max()` to keep compatibility, need to check OSL and OpenPBR spec if we should
   * remove tint for some closures. */
  return sc->weight * reduce_max(bsdf_albedo(kg, sd, sc, true, false));
}

CCL_NAMESPACE_END
