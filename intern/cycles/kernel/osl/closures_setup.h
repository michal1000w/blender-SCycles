/* SPDX-FileCopyrightText: 2009-2010 Sony Pictures Imageworks Inc., et al. All Rights Reserved.
 * SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Adapted code from Open Shading Language. */

#pragma once

#include "kernel/closure/alloc.h"
#include "kernel/closure/bsdf.h"
#include "kernel/closure/bssrdf.h"
#include "kernel/closure/emissive.h"
#include "kernel/closure/volume.h"
#include "kernel/light/coherent_detector.h"

#include "kernel/geom/object.h"

#include "kernel/osl/types.h"

CCL_NAMESPACE_BEGIN

#define OSL_CLOSURE_STRUCT_BEGIN(Upper, lower) \
  struct ccl_align(8) Upper##Closure { \
    const char *label;
#define OSL_CLOSURE_STRUCT_END(Upper, lower) \
  } \
  ;
#define OSL_CLOSURE_STRUCT_MEMBER(Upper, TYPE, type, name, key) type name;
#define OSL_CLOSURE_STRUCT_ARRAY_MEMBER(Upper, TYPE, type, name, key, size) type name[size];

#include "closures_template.h"

struct ccl_align(8) LayerClosure {
  const ccl_private OSLClosure *base;
  const ccl_private OSLClosure *top;
};

/* If we failed to allocate a layer-able closure, we need to zero out the albedo
 * so that lower layers aren't falsely blocked.
 * Therefore, to keep the code clean, set it to zero at the start and overwrite
 * later if it succeeded. */
ccl_device_forceinline void osl_zero_albedo(float3 *layer_albedo)
{
  if (layer_albedo != nullptr) {
    *layer_albedo = zero_float3();
  }
}

ccl_device_forceinline bool osl_closure_skip(KernelGlobals kg,
                                             const PathRayVisibility path_visibility,
                                             const int scattering)
{
  /* Caustic options */
  if ((scattering & LABEL_GLOSSY) && (path_visibility & PATH_RAY_VISIBILITY_DIFFUSE)) {
    const bool has_reflect = (scattering & LABEL_REFLECT);
    const bool has_transmit = (scattering & LABEL_TRANSMIT);
    const bool reflect_caustics_disabled = !kernel_data.integrator.caustics_reflective;
    const bool refract_caustics_disabled = !kernel_data.integrator.caustics_refractive;

    /* Reflective Caustics */
    if (reflect_caustics_disabled && has_reflect && !has_transmit) {
      return true;
    }
    /* Refractive Caustics */
    if (refract_caustics_disabled && has_transmit && !has_reflect) {
      return true;
    }
    /* Glass Caustics */
    if (reflect_caustics_disabled && refract_caustics_disabled) {
      return true;
    }
  }

  return false;
}

/* Diffuse */

ccl_device void osl_closure_diffuse_setup(KernelGlobals kg,
                                          ccl_private ShaderData *sd,
                                          const PathRayVisibility path_visibility,
                                          const uint32_t /*path_flag*/,
                                          const float3 weight,
                                          const ccl_private DiffuseClosure *closure,
                                          float3 * /*layer_albedo*/)
{
  if (osl_closure_skip(kg, path_visibility, LABEL_DIFFUSE)) {
    return;
  }

  const float3 N = safe_normalize_fallback(closure->N, sd->N);
  Spectrum detector_weight = rgb_to_spectrum(weight);
  if (kernel_data.integrator.coherent_specular_enabled &&
      (sd->object_flag & SD_OBJECT_COHERENT_DETECTOR))
  {
    detector_weight = coherent_detector_passive_color(detector_weight);
  }
  bsdf_diffuse_setup(sd, N, detector_weight);
}

/* Deprecated form, will be removed in OSL 2.0. */
ccl_device void osl_closure_oren_nayar_setup(KernelGlobals kg,
                                             ccl_private ShaderData *sd,
                                             const PathRayVisibility path_visibility,
                                             const uint32_t /*path_flag*/,
                                             const float3 weight,
                                             const ccl_private OrenNayarClosure *closure,
                                             float3 * /*layer_albedo*/)
{
  if (osl_closure_skip(kg, path_visibility, LABEL_DIFFUSE)) {
    return;
  }

  const float3 N = safe_normalize_fallback(closure->N, sd->N);
  const Spectrum color = rgb_to_spectrum(weight);
  bsdf_oren_nayar_setup(sd, N, color, closure->roughness, color);
}

ccl_device void osl_closure_oren_nayar_diffuse_bsdf_setup(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const PathRayVisibility path_visibility,
    const uint32_t /*path_flag*/,
    const float3 weight,
    const ccl_private OrenNayarDiffuseBSDFClosure *closure,
    float3 * /*layer_albedo*/)
{
  if (osl_closure_skip(kg, path_visibility, LABEL_DIFFUSE)) {
    return;
  }

  const float3 N = safe_normalize_fallback(closure->N, sd->N);
  const Spectrum closure_weight = rgb_to_spectrum(weight * closure->albedo);
  const Spectrum color = rgb_to_spectrum(closure->albedo);
  bsdf_oren_nayar_setup(sd, N, closure_weight, closure->roughness, color);
}

ccl_device void osl_closure_burley_diffuse_bsdf_setup(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const PathRayVisibility path_visibility,
    const uint32_t /*path_flag*/,
    const float3 weight,
    ccl_private const BurleyDiffuseBSDFClosure *closure,
    float3 * /*layer_albedo*/)
{
  if (osl_closure_skip(kg, path_visibility, LABEL_DIFFUSE)) {
    return;
  }

  ccl_private BurleyBsdf *bsdf = (ccl_private BurleyBsdf *)bsdf_alloc(
      sd, sizeof(BurleyBsdf), rgb_to_spectrum(weight * closure->albedo));
  if (!bsdf) {
    return;
  }

  bsdf->N = safe_normalize_fallback(closure->N, sd->N);

  sd->runtime_flag |= bsdf_burley_setup(bsdf, closure->roughness);
}

ccl_device void osl_closure_translucent_setup(KernelGlobals kg,
                                              ccl_private ShaderData *sd,
                                              const PathRayVisibility path_visibility,
                                              const uint32_t /*path_flag*/,
                                              const float3 weight,
                                              const ccl_private TranslucentClosure *closure,
                                              float3 * /*layer_albedo*/)
{
  if (osl_closure_skip(kg, path_visibility, LABEL_DIFFUSE)) {
    return;
  }

  const float3 N = safe_normalize_fallback(closure->N, sd->N);
  bsdf_translucent_setup(sd, N, rgb_to_spectrum(weight));
}

ccl_device void osl_closure_translucent_bsdf_setup(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const PathRayVisibility path_visibility,
    const uint32_t /*path_flag*/,
    const float3 weight,
    const ccl_private TranslucentBSDFClosure *closure,
    float3 * /*layer_albedo*/)
{
  if (osl_closure_skip(kg, path_visibility, LABEL_DIFFUSE)) {
    return;
  }

  const float3 N = safe_normalize_fallback(closure->N, sd->N);
  bsdf_translucent_setup(sd, N, rgb_to_spectrum(weight * closure->albedo));
}

ccl_device void osl_closure_reflection_setup(KernelGlobals kg,
                                             ccl_private ShaderData *sd,
                                             const PathRayVisibility path_visibility,
                                             const uint32_t /*path_flag*/,
                                             const float3 weight,
                                             const ccl_private ReflectionClosure *closure,
                                             float3 * /*layer_albedo*/)
{
  if (osl_closure_skip(kg, path_visibility, LABEL_SINGULAR)) {
    return;
  }

  ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
      sd, sizeof(MicrofacetBsdf), rgb_to_spectrum(weight));
  if (!bsdf) {
    return;
  }

  bsdf->N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  bsdf->alpha_x = bsdf->alpha_y = 0.0f;

  sd->runtime_flag |= bsdf_microfacet_ggx_setup(bsdf);
}

ccl_device void osl_closure_refraction_setup(KernelGlobals kg,
                                             ccl_private ShaderData *sd,
                                             const PathRayVisibility path_visibility,
                                             const uint32_t /*path_flag*/,
                                             const float3 weight,
                                             const ccl_private RefractionClosure *closure,
                                             float3 * /*layer_albedo*/)
{
  if (osl_closure_skip(kg, path_visibility, LABEL_SINGULAR)) {
    return;
  }

  ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
      sd, sizeof(MicrofacetBsdf), rgb_to_spectrum(weight));
  if (!bsdf) {
    return;
  }

  const bool backfacing = (sd->runtime_flag & SR_BACKFACING);

  bsdf->N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  bsdf->ior = backfacing ? 1.0f / closure->ior : closure->ior;
  bsdf->alpha_x = bsdf->alpha_y = 0.0f;

  sd->runtime_flag |= bsdf_microfacet_ggx_refraction_setup(bsdf);
}

ccl_device void osl_closure_transparent_setup(KernelGlobals /*kg*/,
                                              ccl_private ShaderData *sd,
                                              const PathRayVisibility /*path_visibility*/,
                                              const uint32_t path_flag,
                                              const float3 weight,
                                              const ccl_private TransparentClosure * /*closure*/,
                                              float3 * /*layer_albedo*/)
{
  bsdf_transparent_setup(sd, rgb_to_spectrum(weight), path_flag);
}

ccl_device void osl_closure_transparent_bsdf_setup(
    KernelGlobals /*kg*/,
    ccl_private ShaderData *sd,
    const PathRayVisibility /*path_visibility*/,
    const uint32_t path_flag,
    const float3 weight,
    const ccl_private TransparentBSDFClosure * /*closure*/,
    float3 * /*layer_albedo*/)
{
  bsdf_transparent_setup(sd, rgb_to_spectrum(weight), path_flag);
}

ccl_device void osl_closure_ray_portal_bsdf_setup(KernelGlobals /*kg*/,
                                                  ccl_private ShaderData *sd,
                                                  const PathRayVisibility /*path_visibility*/,
                                                  const uint32_t /*path_flag*/,
                                                  const float3 weight,
                                                  const ccl_private RayPortalBSDFClosure *closure,
                                                  float3 * /*layer_albedo*/)
{
  bsdf_ray_portal_setup(sd, rgb_to_spectrum(weight), closure->position, closure->direction);
}

ccl_device void osl_closure_coat_bsdf_setup(KernelGlobals kg,
                                            ccl_private ShaderData *sd,
                                            const PathRayVisibility path_visibility,
                                            const uint32_t /*path_flag*/,
                                            const float3 weight,
                                            const ccl_private CoatBSDFClosure *closure,
                                            float3 *layer_albedo)
{
  osl_zero_albedo(layer_albedo);

  if (osl_closure_skip(kg, path_visibility, LABEL_GLOSSY | LABEL_REFLECT)) {
    return;
  }

  Coat coat;
  coat.tint = rgb_to_spectrum(closure->color);
  coat.ior = closure->ior;
  coat.N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  coat.roughness = closure->roughness;
  coat.weight = rgb_to_spectrum(weight);
  const Spectrum albedo = bsdf_coat_setup(kg, sd, path_visibility, coat);

  if (layer_albedo) {
    *layer_albedo = spectrum_to_rgb(albedo);
  }
}

/* MaterialX closures */
ccl_device void osl_closure_dielectric_bsdf_setup(KernelGlobals kg,
                                                  ccl_private ShaderData *sd,
                                                  const PathRayVisibility path_visibility,
                                                  const uint32_t /*path_flag*/,
                                                  const float3 weight,
                                                  const ccl_private DielectricBSDFClosure *closure,
                                                  float3 *layer_albedo)
{
  osl_zero_albedo(layer_albedo);

  if (osl_closure_skip(kg, path_visibility, LABEL_GLOSSY | LABEL_REFLECT)) {
    return;
  }

  ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
      sd, sizeof(MicrofacetBsdf), rgb_to_spectrum(weight));
  if (!bsdf) {
    return;
  }

  ccl_private FresnelDielectricTint *fresnel = (ccl_private FresnelDielectricTint *)
      closure_alloc_extra(sd, sizeof(FresnelDielectricTint));

  if (!fresnel) {
    return;
  }

  const bool backfacing = (sd->runtime_flag & SR_BACKFACING);

  bsdf->N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  bsdf->alpha_x = closure->alpha_x;
  bsdf->alpha_y = closure->alpha_y;
  bsdf->ior = backfacing ? 1.0f / closure->ior : closure->ior;
  bsdf->T = closure->T;

  const bool beckmann = closure->distribution == make_string("beckmann", 14712237670914973463ull);
  const bool multiggx = closure->distribution == make_string("multi_ggx", 16842698693386468366ull);

  fresnel->thin_film = {closure->thinfilm_thickness, closure->thinfilm_ior};
  fresnel->tint = {rgb_to_spectrum(closure->reflection_tint),
                   bsdf_spectral_transmission_color(kg, sd, closure->transmission_tint)};
  bsdf_dielectric_tint_setup(kg, bsdf, sd, fresnel, beckmann, multiggx);

  if (layer_albedo != nullptr) {
    if (!(sd->runtime_flag & SR_BSDF_HAS_TRANSMISSION)) {
      *layer_albedo = closure_layer_albedo(kg, sd, (ccl_private ShaderClosure *)bsdf);
    }
    else {
      *layer_albedo = one_float3();
    }
  }
}

ccl_device void osl_closure_conductor_bsdf_setup(KernelGlobals kg,
                                                 ccl_private ShaderData *sd,
                                                 const PathRayVisibility path_visibility,
                                                 const uint32_t /*path_flag*/,
                                                 const float3 weight,
                                                 const ccl_private ConductorBSDFClosure *closure,
                                                 float3 * /*layer_albedo*/)
{
  if (osl_closure_skip(kg, path_visibility, LABEL_GLOSSY | LABEL_REFLECT)) {
    return;
  }

  ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
      sd, sizeof(MicrofacetBsdf), rgb_to_spectrum(weight));
  if (!bsdf) {
    return;
  }

  ccl_private FresnelConductor *fresnel = (ccl_private FresnelConductor *)closure_alloc_extra(
      sd, sizeof(FresnelConductor));
  if (!fresnel) {
    return;
  }

  bsdf->N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  bsdf->alpha_x = closure->alpha_x;
  bsdf->alpha_y = closure->alpha_y;
  bsdf->ior = 0.0f;
  bsdf->T = closure->T;

  bool preserve_energy = false;

  /* Beckmann */
  if (closure->distribution == make_string("beckmann", 14712237670914973463ull)) {
    sd->runtime_flag |= bsdf_microfacet_beckmann_setup(bsdf);
  }
  /* GGX (either single- or multi-scattering) */
  else {
    sd->runtime_flag |= bsdf_microfacet_ggx_setup(bsdf);
    preserve_energy = (closure->distribution == make_string("multi_ggx", 16842698693386468366ull));
  }

  fresnel->thin_film.thickness = closure->thinfilm_thickness;
  fresnel->thin_film.ior = closure->thinfilm_ior;

  fresnel->ior = {rgb_to_spectrum(closure->ior), rgb_to_spectrum(closure->extinction)};
  if (preserve_energy && closure->diffraction_weight > 0.0f) {
    bsdf_microfacet_setup_fresnel_conductor(kg, bsdf, sd->wi, fresnel, false);
    bsdf_diffraction_conductor_multiggx_split_setup(
        kg, sd, bsdf, closure->T, closure->diffraction_pitch, closure->diffraction_depth,
        closure->diffraction_duty, closure->diffraction_weight,
        closure->diffraction_albedo_handle_plus_one - 1);
  }
  else {
    bsdf_microfacet_setup_fresnel_conductor(kg, bsdf, sd->wi, fresnel, preserve_energy);
  }
  if (closure->diffraction_weight>0 && !preserve_energy) {
    bsdf_diffraction_conductor_split_setup(sd,bsdf,closure->T,closure->diffraction_pitch,
        closure->diffraction_depth,closure->diffraction_duty,closure->diffraction_weight);
  }
}

ccl_device void osl_closure_generalized_schlick_bsdf_setup(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const PathRayVisibility path_visibility,
    const uint32_t /*path_flag*/,
    const float3 weight,
    const ccl_private GeneralizedSchlickBSDFClosure *closure,
    float3 *layer_albedo)
{
  osl_zero_albedo(layer_albedo);

  const bool has_reflection = !is_zero(closure->reflection_tint);
  const bool has_transmission = !is_zero(closure->transmission_tint);

  int label = LABEL_GLOSSY | LABEL_REFLECT;
  if (has_transmission) {
    label |= LABEL_TRANSMIT;
  }

  if (osl_closure_skip(kg, path_visibility, label)) {
    return;
  }

  const int first_layer_closure = sd->num_closure;
  ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
      sd, sizeof(MicrofacetBsdf), rgb_to_spectrum(weight));
  if (!bsdf) {
    return;
  }

  ccl_private FresnelGeneralizedSchlick *fresnel = (ccl_private FresnelGeneralizedSchlick *)
      closure_alloc_extra(sd, closure->polarizer ? sizeof(FresnelGeneralizedSchlickPolarizer) : sizeof(FresnelGeneralizedSchlick));
  if (!fresnel) {
    return;
  }

  const bool backfacing = (sd->runtime_flag & SR_BACKFACING);

  bsdf->N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  bsdf->alpha_x = closure->alpha_x;
  bsdf->alpha_y = closure->alpha_y;
  bsdf->T = closure->T;

  if (closure->exponent < 0.0f) {
    /* Trick for principled BSDF: Since we use the real Fresnel equation and remap
     * to the F0...F90 range, this allows us to use the real IOR.
     * Computing it back from F0 might give a different result in case of specular
     * tinting. */
    const float ior = backfacing ? 1.0f / -closure->exponent : -closure->exponent;
    bsdf->ior = bsdf_glass_ior(sd, ior, closure->inv_abbe);
  }
  else {
    const float ior = ior_from_F0(average(closure->f0));
    bsdf->ior = backfacing ? 1.0f / ior : ior;
  }

  bool preserve_energy = false;

  /* Beckmann */
  if (closure->distribution == make_string("beckmann", 14712237670914973463ull)) {
    if (has_reflection && has_transmission) {
      sd->runtime_flag |= bsdf_microfacet_beckmann_glass_setup(bsdf);
    }
    else if (has_transmission) {
      sd->runtime_flag |= bsdf_microfacet_beckmann_refraction_setup(bsdf);
    }
    else {
      sd->runtime_flag |= bsdf_microfacet_beckmann_setup(bsdf);
    }
  }
  /* GGX (either single- or multi-scattering) */
  else {
    if (has_reflection && has_transmission) {
      sd->runtime_flag |= bsdf_microfacet_ggx_glass_setup(bsdf);
    }
    else if (has_transmission) {
      sd->runtime_flag |= bsdf_microfacet_ggx_refraction_setup(bsdf);
    }
    else {
      sd->runtime_flag |= bsdf_microfacet_ggx_setup(bsdf);
    }

    preserve_energy = (closure->distribution == make_string("multi_ggx", 16842698693386468366ull));
  }

  const bool reflective_caustics = (kernel_data.integrator.caustics_reflective ||
                                    (path_visibility & PATH_RAY_VISIBILITY_DIFFUSE) == 0);
  const bool refractive_caustics = (kernel_data.integrator.caustics_refractive ||
                                    (path_visibility & PATH_RAY_VISIBILITY_DIFFUSE) == 0);

  fresnel->tint.reflectance = reflective_caustics ? rgb_to_spectrum(closure->reflection_tint) :
                                                    zero_spectrum();
  fresnel->tint.transmittance = refractive_caustics ? bsdf_spectral_transmission_color(
                                                          kg, sd, closure->transmission_tint) :
                                                      zero_spectrum();
  fresnel->f0 = rgb_to_spectrum(closure->f0);
  fresnel->f90 = rgb_to_spectrum(closure->f90);
  fresnel->exponent = closure->exponent;
  fresnel->thin_film.thickness = closure->thinfilm_thickness;
  fresnel->thin_film.ior = closure->thinfilm_ior;
  const bool multiggx_diffraction = preserve_energy && closure->diffraction_weight > 0.0f &&
                                    !has_transmission;
  bsdf_microfacet_setup_fresnel_generalized_schlick(
      kg, bsdf, sd->wi, fresnel, preserve_energy && !multiggx_diffraction);
  if (closure->polarizer) {
    const float3 axis = object_polarizer_axis(kg, sd, closure->polarizer_angle);
    bsdf_microfacet_set_polarizer(bsdf, safe_normalize(axis));
  }

  if (multiggx_diffraction) {
    bsdf_diffraction_conductor_multiggx_split_setup(
        kg, sd, bsdf, closure->T, closure->diffraction_pitch, closure->diffraction_depth,
        closure->diffraction_duty, closure->diffraction_weight,
        closure->diffraction_albedo_handle_plus_one - 1);
  }
  else if (closure->diffraction_weight > 0.0f && !preserve_energy && !has_transmission) {
    bsdf_diffraction_conductor_split_setup(sd, bsdf, closure->T,
                                            closure->diffraction_pitch,
                                            closure->diffraction_depth,
                                            closure->diffraction_duty,
                                            closure->diffraction_weight);
  }

  if (layer_albedo != nullptr) {
    if (has_reflection && !has_transmission) {
      Spectrum total = zero_spectrum();
      for (int i = first_layer_closure; i < sd->num_closure; i++) {
        total += closure_layer_albedo(kg, sd, &sd->closure[i]);
      }
      *layer_albedo = total;
    }
    else {
      *layer_albedo = one_float3();
    }
  }
}

ccl_device void osl_closure_thin_glass_setup(KernelGlobals kg,
                                             ccl_private ShaderData *sd,
                                             const PathRayVisibility path_visibility,
                                             const uint32_t path_flag,
                                             const float3 weight,
                                             const ccl_private ThinGlassClosure *closure,
                                             float3 *layer_albedo)
{
  osl_zero_albedo(layer_albedo);

  const bool has_reflection = !is_zero(closure->reflection_tint);
  const bool has_transmission = !is_zero(closure->transmission_tint);

  int label = roughness_is_almost_specular(closure->roughness, closure->roughness) ?
                  LABEL_SINGULAR :
                  LABEL_GLOSSY;
  if (has_transmission) {
    label |= LABEL_TRANSMIT;
  }
  if (has_reflection) {
    label |= LABEL_REFLECT;
  }

  if (osl_closure_skip(kg, path_visibility, label)) {
    return;
  }

  const bool reflective_caustics = (kernel_data.integrator.caustics_reflective ||
                                    (path_visibility & PATH_RAY_VISIBILITY_DIFFUSE) == 0);
  const bool refractive_caustics = (kernel_data.integrator.caustics_refractive ||
                                    (path_visibility & PATH_RAY_VISIBILITY_DIFFUSE) == 0);

  const float3 valid_reflection_N = maybe_ensure_valid_specular_reflection(
      sd, safe_normalize_fallback(closure->N, sd->N));
  const FresnelThinFilm thinfilm = {closure->thinfilm_thickness, closure->thinfilm_ior};
  const FresnelCoeff tint = {rgb_to_spectrum(closure->reflection_tint),
                             bsdf_spectral_transmission_color(kg, sd, closure->transmission_tint)};
  /* ThinWall deliberately bypasses backface eta inversion in Principled OSL.
   * Its closure IOR is already the absolute sheet IOR on both faces. */
  const float absolute_ior = closure->ior;
  const bool use_diffraction = closure->diffraction_weight > 0.0f &&
                               closure->diffraction_depth > 0.0f;
  const FresnelCoeff fresnel = use_diffraction ? bsdf_diffraction_thin_glass_setup(kg,
                                                     sd,
                                                     reflective_caustics,
                                                     refractive_caustics,
                                                     tint,
                                                     rgb_to_spectrum(weight),
                                                     valid_reflection_N,
                                                     closure->T,
                                                     closure->roughness,
                                                     absolute_ior,
                                                     thinfilm,
                                                     path_visibility,
                                                     path_flag,
                                                     closure->diffraction_weight,
                                                     closure->diffraction_pitch,
                                                     closure->diffraction_depth,
                                                     closure->diffraction_duty,
                                                     closure->inv_abbe,
                                                     make_int3(closure->diffraction_thin_sheet_r_plus_one - 1,
                                                               closure->diffraction_thin_sheet_g_plus_one - 1,
                                                               closure->diffraction_thin_sheet_b_plus_one - 1)) : bsdf_thin_glass_setup(kg,
                                                     sd,
                                                     reflective_caustics,
                                                     refractive_caustics,
                                                     tint,
                                                     rgb_to_spectrum(weight),
                                                     valid_reflection_N,
                                                     closure->roughness,
                                                     closure->ior,
                                                     thinfilm,
                                                     path_visibility,
                                                     path_flag);

  if (layer_albedo != nullptr) {
    *layer_albedo = fresnel.transmittance * !!has_transmission +
                    fresnel.reflectance * !!has_reflection;
  }
}

ccl_device void osl_closure_thin_subsurface_setup(KernelGlobals kg,
                                                  ccl_private ShaderData *sd,
                                                  const PathRayVisibility path_visibility,
                                                  const uint32_t /*path_flag*/,
                                                  const float3 weight,
                                                  const ccl_private ThinSubsurfaceClosure *closure,
                                                  float3 *layer_albedo)
{
  osl_zero_albedo(layer_albedo);

  if (osl_closure_skip(kg, path_visibility, LABEL_DIFFUSE)) {
    return;
  }

  const float3 N = safe_normalize_fallback(closure->N, sd->N);
  bsdf_thin_subsurface_setup(
      sd, N, rgb_to_spectrum(weight), closure->anisotropy, closure->roughness, closure->color);
}

/* Standard microfacet closures */

/* The cache is prepared by the host, shared with the SVM physical closure.
 * This closure requires the shader's sampled-wavelength metadata. */
ccl_device void osl_closure_diffraction_smooth_setup(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const PathRayVisibility /*path_visibility*/,
    const uint32_t /*path_flag*/,
    const float3 weight,
    const ccl_private DiffractionSmoothClosure *closure,
    float3 *layer_albedo)
{
  osl_zero_albedo(layer_albedo);
  const int index = sd->num_closure;
  if (!bsdf_diffraction_smooth_setup(kg, sd, weight, closure->N, closure->T,
                                     closure->cache_handle, closure->pitch,
                                     closure->upper_index, closure->lower_index,
                                     bool(sd->runtime_flag & SR_BACKFACING)))
    return;
  if (layer_albedo) {
    /* Sum surviving flux before geometric-normal clipping. A randomly chosen
     * rejected order must not change attenuation of lower layers. */
    ccl_private const DiffractionSmoothBsdf *bsdf =
        (ccl_private const DiffractionSmoothBsdf *)&sd->closure[index];
    const float3 N = bsdf->incoming_substrate ? -bsdf->N : bsdf->N;
    const float3 Y = cross(N, bsdf->T);
    const float3 incident = make_float3(-dot(sd->wi, bsdf->T),
                                        -dot(sd->wi, Y), -dot(sd->wi, N));
    DiffractionSceneSample sample;
    if (diffraction_scene_sample(kg, bsdf->cache_handle, incident,
                                  bool(bsdf->incoming_substrate), bsdf->upper_index,
                                  bsdf->lower_index, bsdf->wavelength, bsdf->pitch,
                                  0.5f, &sample))
      *layer_albedo = weight * sample.throughput;
  }
}

ccl_device void osl_closure_diffraction_fast_setup(
    KernelGlobals kg, ccl_private ShaderData *sd,
    const PathRayVisibility /*path_visibility*/, const uint32_t /*path_flag*/,
    const float3 weight, const ccl_private DiffractionFastClosure *closure,
    float3 *layer_albedo)
{
  osl_zero_albedo(layer_albedo);
  const int index=sd->num_closure;
  if (!bsdf_diffraction_smooth_setup(kg,sd,weight,closure->N,closure->T,
      DIFFRACTION_FAST_CACHE_HANDLE,closure->pitch,closure->upper_index,closure->lower_index,
      bool(sd->runtime_flag & SR_BACKFACING),closure->depth,closure->duty,
      closure->phase_contrast,closure->reflection_budget,closure->transmission_budget,
      closure->ridge_extinction, closure->ridge_table, closure->groove_table,
      closure->substrate_table)) return;
  if (layer_albedo) {
    const ccl_private DiffractionSmoothBsdf *bsdf=
        (ccl_private const DiffractionSmoothBsdf *)&sd->closure[index];
    *layer_albedo=weight*(bsdf->fast->reflection_budget+bsdf->fast->transmission_budget);
  }
}

ccl_device void osl_closure_diffraction_refraction_setup(
    KernelGlobals kg, ccl_private ShaderData *sd,
    const PathRayVisibility path_visibility, const uint32_t /*path_flag*/,
    const float3 weight, const ccl_private DiffractionRefractionClosure *closure,
    float3 *layer_albedo)
{
  osl_zero_albedo(layer_albedo);
  if (osl_closure_skip(kg,path_visibility,LABEL_GLOSSY|LABEL_TRANSMIT)) return;
  const Spectrum w=rgb_to_spectrum(weight)*bsdf_spectral_transmission_color(
      kg,sd,max(float3(closure->color),zero_float3()));
  bsdf_diffraction_refraction_setup(sd,w,
      maybe_ensure_valid_specular_reflection(sd,safe_normalize_fallback(closure->N,sd->N)),
      closure->T,closure->roughness,max(closure->ior,1e-5f),
      closure->pitch,closure->depth,closure->duty,closure->beckmann!=0);
}

ccl_device void osl_closure_diffraction_glass_setup(
    KernelGlobals kg, ccl_private ShaderData *sd,
    const PathRayVisibility path_visibility, const uint32_t /*path_flag*/,
    const float3 weight, const ccl_private DiffractionGlassClosure *closure,
    float3 *layer_albedo)
{
  osl_zero_albedo(layer_albedo);
  const float3 color = closure->diffraction_two_sided_handle_plus_one > 0 ?
                           saturate(float3(closure->color)) :
                           max(float3(closure->color), zero_float3());
  const Spectrum w = rgb_to_spectrum(weight);
  const Spectrum reflection_color = closure->principled_transmission ? one_spectrum() :
                                                                       rgb_to_spectrum(color);
  const Spectrum reflection = osl_closure_skip(kg, path_visibility, LABEL_GLOSSY | LABEL_REFLECT) ?
                                  zero_spectrum() : w * reflection_color;
  const Spectrum transmission = osl_closure_skip(kg, path_visibility, LABEL_GLOSSY | LABEL_TRANSMIT) ?
                                  zero_spectrum() : w * bsdf_spectral_transmission_color(kg, sd, color);
  if (closure->principled_transmission) {
    if (closure->diffraction_two_sided_handle_plus_one > 0) {
      const Spectrum bounded_transmission = osl_closure_skip(
          kg, path_visibility, LABEL_GLOSSY | LABEL_TRANSMIT) ? zero_spectrum() :
          w * saturate(bsdf_spectral_transmission_color(kg, sd, color));
      bsdf_diffraction_principled_transmission_two_sided_setup(
          kg, sd, reflection, bounded_transmission,
          maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N)),
          closure->T, closure->roughness, max(closure->ior, 1e-5f),
          closure->pitch, closure->depth, closure->duty,
          rgb_to_spectrum(max(float3(closure->specular_tint), zero_float3())), closure->inv_abbe,
          max(closure->film_ior, 1e-5f), closure->film_thickness,
          closure->diffraction_two_sided_handle_plus_one - 1);
    }
    else {
      bsdf_diffraction_principled_transmission_setup(sd,reflection,transmission,
          maybe_ensure_valid_specular_reflection(sd,safe_normalize_fallback(closure->N,sd->N)),
          closure->T,closure->roughness,max(closure->ior,1e-5f),
          closure->pitch,closure->depth,closure->duty,
          rgb_to_spectrum(max(float3(closure->specular_tint),zero_float3())),closure->inv_abbe,
          max(closure->film_ior,1e-5f),closure->film_thickness);
    }
    return;
  }
  const int polarizer_first = sd->num_closure;
  const int polarizer_left = sd->num_closure_left;
  if (closure->diffraction_two_sided_handle_plus_one > 0) {
    /* The two-sided return applies one spectral tint per completed return.
     * Bound reconstruction before multiplying coverage and material weight. */
    const Spectrum bounded_reflection = osl_closure_skip(
                                            kg, path_visibility, LABEL_GLOSSY | LABEL_REFLECT) ?
                                            zero_spectrum() :
                                            w * min(max(reflection_color, zero_spectrum()),
                                                    one_spectrum());
    const Spectrum bounded_transmission = osl_closure_skip(
                                              kg, path_visibility, LABEL_GLOSSY | LABEL_TRANSMIT) ?
                                              zero_spectrum() :
                                              w * min(max(bsdf_spectral_transmission_color(
                                                              kg, sd, color),
                                                          zero_spectrum()),
                                                      one_spectrum());
    bsdf_diffraction_glass_two_sided_setup(
        kg, sd, bounded_reflection, bounded_transmission,
        maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N)),
        closure->T, closure->roughness, max(closure->ior, 1e-5f),
        closure->pitch, closure->depth, closure->duty,
        closure->diffraction_two_sided_handle_plus_one - 1,
        max(closure->film_ior, 1e-5f), closure->film_thickness);
  }
  else bsdf_diffraction_glass_setup(sd, reflection, transmission,
      maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N)),
      closure->T, closure->roughness, max(closure->ior, 1e-5f),
      closure->pitch, closure->depth, closure->duty,
      max(closure->film_ior, 1e-5f), closure->film_thickness, closure->beckmann != 0);
  if (closure->polarizer) {
    const float3 axis=object_polarizer_axis(kg,sd,closure->polarizer_angle);
    const bool attached=bsdf_diffraction_glass_set_polarizer(sd,polarizer_first,safe_normalize(axis));
    if (!attached) {sd->num_closure=polarizer_first;sd->num_closure_left=polarizer_left;}
    kernel_assert(attached);
  }
}

ccl_device void osl_closure_diffraction_setup(KernelGlobals kg,
                                             ccl_private ShaderData *sd,
                                             const PathRayVisibility path_visibility,
                                             const uint32_t /*path_flag*/,
                                             const float3 weight,
                                             const ccl_private DiffractionClosure *closure,
                                             float3 *layer_albedo)
{
  osl_zero_albedo(layer_albedo);
  if (osl_closure_skip(kg, path_visibility, LABEL_GLOSSY | LABEL_REFLECT)) {
    return;
  }
  const float3 N = maybe_ensure_valid_specular_reflection(
      sd, safe_normalize_fallback(closure->N, sd->N));
  if (closure->ashikhmin != 0) {
    bsdf_diffraction_ashikhmin_glossy_setup(
        sd, rgb_to_spectrum(weight), N, closure->T,
        closure->alpha_x, closure->alpha_y, closure->pitch, closure->depth,
        closure->duty, closure->medium_ior);
  }
  else {
    bsdf_diffraction_glossy_setup(kg, sd, rgb_to_spectrum(weight), N, closure->T,
                           closure->alpha_x, closure->alpha_y, closure->pitch, closure->depth,
                           closure->duty, closure->medium_ior, closure->beckmann != 0,
                           closure->albedo_handle - 1,
                           rgb_to_spectrum(saturate(one_float3() - closure->multiscatter_absorption)));
  }
  if (layer_albedo) {
    *layer_albedo = weight;
  }
}

ccl_device void osl_closure_microfacet_setup(KernelGlobals kg,
                                             ccl_private ShaderData *sd,
                                             const PathRayVisibility path_visibility,
                                             const uint32_t /*path_flag*/,
                                             const float3 weight,
                                             const ccl_private MicrofacetClosure *closure,
                                             float3 *layer_albedo)
{
  osl_zero_albedo(layer_albedo);

  const int label = (closure->refract) ? LABEL_TRANSMIT : LABEL_REFLECT;
  if (osl_closure_skip(kg, path_visibility, LABEL_GLOSSY | label)) {
    return;
  }

  const Spectrum spectral_weight = (closure->refract == 1) ?
                                       bsdf_spectral_transmission_color(kg, sd, weight) :
                                       rgb_to_spectrum(weight);
  ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
      sd, sizeof(MicrofacetBsdf), spectral_weight);
  if (!bsdf) {
    return;
  }

  const bool backfacing = (sd->runtime_flag & SR_BACKFACING);

  bsdf->N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  bsdf->alpha_x = closure->alpha_x;
  bsdf->alpha_y = closure->alpha_y;
  bsdf->ior = backfacing ? 1.0f / closure->ior : closure->ior;
  bsdf->T = closure->T;

  /* Beckmann */
  if (closure->distribution == make_string("beckmann", 14712237670914973463ull)) {
    if (closure->refract == 1) {
      sd->runtime_flag |= bsdf_microfacet_beckmann_refraction_setup(bsdf);
    }
    else if (closure->refract == 2) {
      sd->runtime_flag |= bsdf_microfacet_beckmann_glass_setup(bsdf);
    }
    else {
      sd->runtime_flag |= bsdf_microfacet_beckmann_setup(bsdf);
    }
  }
  /* Ashikhmin-Shirley */
  else if (closure->distribution == make_string("ashikhmin_shirley", 11318482998918370922ull)) {
    sd->runtime_flag |= bsdf_ashikhmin_shirley_setup(bsdf);
  }
  /* GGX (either single- or multi-scattering) */
  else {
    if (closure->refract == 1) {
      sd->runtime_flag |= bsdf_microfacet_ggx_refraction_setup(bsdf);
    }
    else if (closure->refract == 2) {
      sd->runtime_flag |= bsdf_microfacet_ggx_glass_setup(bsdf);
    }
    else {
      sd->runtime_flag |= bsdf_microfacet_ggx_setup(bsdf);
    }

    if (closure->distribution == make_string("multi_ggx", 16842698693386468366ull)) {
      /* Since there's no dedicated color input, the weight is the best we got. */
      bsdf_microfacet_setup_fresnel_constant(kg, bsdf, sd->wi, rgb_to_spectrum(weight));
    }
  }

  if (layer_albedo != nullptr) {
    if (closure->refract == 0) {
      *layer_albedo = closure_layer_albedo(kg, sd, (ccl_private ShaderClosure *)bsdf);
    }
    else {
      *layer_albedo = one_float3();
    }
  }
}

/* Special-purpose Microfacet closures */

ccl_device void osl_closure_microfacet_f82_tint_setup(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const PathRayVisibility path_visibility,
    const uint32_t /*path_flag*/,
    const float3 weight,
    const ccl_private MicrofacetF82TintClosure *closure,
    float3 * /*layer_albedo*/)
{
  if (osl_closure_skip(kg, path_visibility, LABEL_GLOSSY | LABEL_REFLECT)) {
    return;
  }

  ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
      sd, sizeof(MicrofacetBsdf), rgb_to_spectrum(weight));
  if (!bsdf) {
    return;
  }

  ccl_private FresnelF82Tint *fresnel = (ccl_private FresnelF82Tint *)closure_alloc_extra(
      sd, sizeof(FresnelF82Tint));
  if (!fresnel) {
    return;
  }

  bsdf->N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  bsdf->alpha_x = closure->alpha_x;
  bsdf->alpha_y = closure->alpha_y;
  bsdf->ior = 0.0f;
  bsdf->T = closure->T;

  bool preserve_energy = false;

  /* Beckmann */
  if (closure->distribution == make_string("beckmann", 14712237670914973463ull)) {
    sd->runtime_flag |= bsdf_microfacet_beckmann_setup(bsdf);
  }
  /* GGX (either single- or multi-scattering) */
  else {
    sd->runtime_flag |= bsdf_microfacet_ggx_setup(bsdf);
    preserve_energy = (closure->distribution == make_string("multi_ggx", 16842698693386468366ull));
  }

  fresnel->f0 = rgb_to_spectrum(closure->f0);
  fresnel->thin_film.thickness = closure->thinfilm_thickness;
  fresnel->thin_film.ior = closure->thinfilm_ior;

  if (preserve_energy && closure->diffraction_weight > 0.0f) {
    bsdf_microfacet_setup_fresnel_f82_tint(
        kg, bsdf, sd->wi, fresnel, rgb_to_spectrum(closure->f82), false);
    bsdf_diffraction_conductor_multiggx_split_setup(
        kg, sd, bsdf, closure->T, closure->diffraction_pitch, closure->diffraction_depth,
        closure->diffraction_duty, closure->diffraction_weight,
        closure->diffraction_albedo_handle_plus_one - 1);
  }
  else {
    bsdf_microfacet_setup_fresnel_f82_tint(
        kg, bsdf, sd->wi, fresnel, rgb_to_spectrum(closure->f82), preserve_energy);
  }
  if (closure->diffraction_weight>0 && !preserve_energy) {
    bsdf_diffraction_conductor_split_setup(sd,bsdf,closure->T,closure->diffraction_pitch,
        closure->diffraction_depth,closure->diffraction_duty,closure->diffraction_weight);
  }
}

ccl_device void osl_closure_microfacet_multi_ggx_glass_setup(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const PathRayVisibility path_visibility,
    const uint32_t /*path_flag*/,
    const float3 weight,
    const ccl_private MicrofacetMultiGGXGlassClosure *closure,
    float3 * /*layer_albedo*/)
{
  /* Technically, the MultiGGX closure may also transmit. However,
   * since this is set statically and only used for caustic flags, this
   * is probably as good as it gets. */
  if (osl_closure_skip(kg, path_visibility, LABEL_GLOSSY | LABEL_REFLECT)) {
    return;
  }

  ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
      sd, sizeof(MicrofacetBsdf), rgb_to_spectrum(weight));
  if (!bsdf) {
    return;
  }

  const bool backfacing = (sd->runtime_flag & SR_BACKFACING);

  bsdf->N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  bsdf->alpha_x = closure->alpha_x;
  bsdf->alpha_y = bsdf->alpha_x;
  bsdf->ior = backfacing ? 1.0f / closure->ior : closure->ior;

  bsdf->T = zero_float3();

  sd->runtime_flag |= bsdf_microfacet_ggx_glass_setup(bsdf);
  bsdf_microfacet_setup_fresnel_constant(kg, bsdf, sd->wi, rgb_to_spectrum(closure->color));
}

ccl_device void osl_closure_microfacet_multi_ggx_aniso_setup(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const PathRayVisibility path_visibility,
    const uint32_t /*path_flag*/,
    const float3 weight,
    const ccl_private MicrofacetMultiGGXClosure *closure,
    float3 *layer_albedo)
{
  osl_zero_albedo(layer_albedo);

  if (osl_closure_skip(kg, path_visibility, LABEL_GLOSSY | LABEL_REFLECT)) {
    return;
  }

  ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)bsdf_alloc(
      sd, sizeof(MicrofacetBsdf), rgb_to_spectrum(weight));
  if (!bsdf) {
    return;
  }

  bsdf->N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  bsdf->alpha_x = closure->alpha_x;
  bsdf->alpha_y = closure->alpha_y;
  bsdf->ior = 1.0f;

  bsdf->T = closure->T;

  sd->runtime_flag |= bsdf_microfacet_ggx_setup(bsdf);
  bsdf_microfacet_setup_fresnel_constant(kg, bsdf, sd->wi, rgb_to_spectrum(closure->color));

  if (layer_albedo != nullptr) {
    *layer_albedo = closure_layer_albedo(kg, sd, (ccl_private ShaderClosure *)bsdf);
  }
}

/* Ashikhmin Velvet */

ccl_device void osl_closure_ashikhmin_velvet_setup(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const PathRayVisibility path_visibility,
    const uint32_t /*path_flag*/,
    const float3 weight,
    const ccl_private AshikhminVelvetClosure *closure,
    float3 * /*layer_albedo*/)
{
  if (osl_closure_skip(kg, path_visibility, LABEL_DIFFUSE)) {
    return;
  }

  ccl_private VelvetBsdf *bsdf = (ccl_private VelvetBsdf *)bsdf_alloc(
      sd, sizeof(VelvetBsdf), rgb_to_spectrum(weight));
  if (!bsdf) {
    return;
  }

  bsdf->N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  bsdf->sigma = closure->sigma;

  sd->runtime_flag |= bsdf_ashikhmin_velvet_setup(bsdf);
}

/* Sheen */
ccl_device void osl_closure_sheen_bsdf_setup(KernelGlobals kg,
                                             ccl_private ShaderData *sd,
                                             const PathRayVisibility path_visibility,
                                             const uint32_t /*path_flag*/,
                                             const float3 weight,
                                             const ccl_private SheenBSDFClosure *closure,
                                             float3 *layer_albedo)
{
  osl_zero_albedo(layer_albedo);

  if (osl_closure_skip(kg, path_visibility, LABEL_DIFFUSE)) {
    return;
  }

  const float3 N = safe_normalize_fallback(closure->N, sd->N);
  const Spectrum color = rgb_to_spectrum(closure->albedo);
  const Spectrum albedo = bsdf_sheen_setup(
      kg, sd, rgb_to_spectrum(weight) * color, N, closure->roughness);

  if (layer_albedo) {
    *layer_albedo = spectrum_to_rgb(albedo);
  }
}

ccl_device void osl_closure_diffuse_toon_setup(KernelGlobals kg,
                                               ccl_private ShaderData *sd,
                                               const PathRayVisibility path_visibility,
                                               const uint32_t /*path_flag*/,
                                               const float3 weight,
                                               const ccl_private DiffuseToonClosure *closure,
                                               float3 * /*layer_albedo*/)
{
  if (osl_closure_skip(kg, path_visibility, LABEL_DIFFUSE)) {
    return;
  }

  ccl_private ToonBsdf *bsdf = (ccl_private ToonBsdf *)bsdf_alloc(
      sd, sizeof(ToonBsdf), rgb_to_spectrum(weight));
  if (!bsdf) {
    return;
  }

  bsdf->N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  bsdf->size = closure->size;
  bsdf->smooth = closure->smooth;

  sd->runtime_flag |= bsdf_diffuse_toon_setup(bsdf);
}

ccl_device void osl_closure_glossy_toon_setup(KernelGlobals kg,
                                              ccl_private ShaderData *sd,
                                              const PathRayVisibility path_visibility,
                                              const uint32_t /*path_flag*/,
                                              const float3 weight,
                                              const ccl_private GlossyToonClosure *closure,
                                              float3 * /*layer_albedo*/)
{
  if (osl_closure_skip(kg, path_visibility, LABEL_GLOSSY)) {
    return;
  }

  ccl_private ToonBsdf *bsdf = (ccl_private ToonBsdf *)bsdf_alloc(
      sd, sizeof(ToonBsdf), rgb_to_spectrum(weight));
  if (!bsdf) {
    return;
  }

  bsdf->N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  bsdf->size = closure->size;
  bsdf->smooth = closure->smooth;

  sd->runtime_flag |= bsdf_glossy_toon_setup(bsdf);
}

/* Variable cone emissive closure
 *
 * This primitive emits in a cone having a configurable penumbra area where the light decays to 0
 * reaching the outer_angle limit. It can also behave as a lambertian emitter if the provided
 * angles are PI/2, which is the default
 */
ccl_device void osl_closure_emission_setup(KernelGlobals kg,
                                           ccl_private ShaderData *sd,
                                           const PathRayVisibility path_visibility,
                                           const uint32_t path_flag,
                                           float3 weight,
                                           const ccl_private GenericEmissiveClosure * /*closure*/,
                                           float3 * /*layer_albedo*/)
{
  if (sd->runtime_flag & SR_IS_VOLUME_SHADER_EVAL) {
    if ((path_visibility & PATH_RAY_VISIBILITY_SHADOW) || (path_flag & PATH_RAY_EXTINCTION)) {
      /* Don't need emission for shadows and extinction. */
      return;
    }
    weight *= object_volume_density(kg, sd->object);
  }
  emission_setup(sd, rgb_to_spectrum(weight));
}

/* Generic background closure
 *
 * We only have a background closure for the shaders to return a color in background shaders. No
 * methods, only the weight is taking into account
 */
ccl_device void osl_closure_background_setup(
    KernelGlobals /*kg*/,
    ccl_private ShaderData *sd,
    const PathRayVisibility /*path_visibility*/,
    const uint32_t /*path_flag*/,
    const float3 weight,
    const ccl_private GenericBackgroundClosure * /*closure*/,
    float3 * /*layer_albedo*/)
{
  background_setup(sd, rgb_to_spectrum(weight));
}

/* Uniform EDF
 *
 * This is a duplicate of emission above except an emittance value can be passed to the weight.
 * This is for MaterialX closure compatibility found in `stdosl.h`.
 */
ccl_device void osl_closure_uniform_edf_setup(KernelGlobals kg,
                                              ccl_private ShaderData *sd,
                                              const PathRayVisibility path_visibility,
                                              const uint32_t path_flag,
                                              float3 weight,
                                              const ccl_private UniformEDFClosure *closure,
                                              float3 * /*layer_albedo*/)
{
  weight *= closure->emittance;
  if (sd->runtime_flag & SR_IS_VOLUME_SHADER_EVAL) {
    if ((path_visibility & PATH_RAY_VISIBILITY_SHADOW) || (path_flag & PATH_RAY_EXTINCTION)) {
      /* Don't need emission for shadows and extinction. */
      return;
    }
    weight *= object_volume_density(kg, sd->object);
  }
  emission_setup(sd, rgb_to_spectrum(weight));
}

/* Holdout closure
 *
 * This will be used by the shader to mark the amount of holdout for the current shading point. No
 * parameters, only the weight will be used
 */
ccl_device void osl_closure_holdout_setup(KernelGlobals /*kg*/,
                                          ccl_private ShaderData *sd,
                                          const PathRayVisibility /*path_visibility*/,
                                          const uint32_t /*path_flag*/,
                                          const float3 weight,
                                          const ccl_private HoldoutClosure * /*closure*/,
                                          float3 * /*layer_albedo*/)
{
  closure_alloc(sd, sizeof(ShaderClosure), CLOSURE_HOLDOUT_ID, rgb_to_spectrum(weight));
  sd->runtime_flag |= SR_HOLDOUT;
}

ccl_device void osl_closure_diffuse_ramp_setup(KernelGlobals /*kg*/,
                                               ccl_private ShaderData *sd,
                                               const PathRayVisibility /*path_visibility*/,
                                               const uint32_t /*path_flag*/,
                                               const float3 weight,
                                               const ccl_private DiffuseRampClosure *closure,
                                               float3 * /*layer_albedo*/)
{
  ccl_private DiffuseRampBsdf *bsdf = (ccl_private DiffuseRampBsdf *)bsdf_alloc(
      sd, sizeof(DiffuseRampBsdf), rgb_to_spectrum(weight));

  if (!bsdf) {
    return;
  }

  bsdf->N = safe_normalize_fallback(closure->N, sd->N);

  bsdf->colors = (float3 *)closure_alloc_extra(sd, sizeof(float3) * 8);
  if (!bsdf->colors) {
    return;
  }

  for (int i = 0; i < 8; i++) {
    bsdf->colors[i] = closure->colors[i];
  }

  sd->runtime_flag |= bsdf_diffuse_ramp_setup(bsdf);
}

ccl_device void osl_closure_phong_ramp_setup(KernelGlobals /*kg*/,
                                             ccl_private ShaderData *sd,
                                             const PathRayVisibility /*path_visibility*/,
                                             const uint32_t /*path_flag*/,
                                             const float3 weight,
                                             const ccl_private PhongRampClosure *closure,
                                             float3 * /*layer_albedo*/)
{
  ccl_private PhongRampBsdf *bsdf = (ccl_private PhongRampBsdf *)bsdf_alloc(
      sd, sizeof(PhongRampBsdf), rgb_to_spectrum(weight));
  if (!bsdf) {
    return;
  }

  bsdf->N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  bsdf->exponent = closure->exponent;

  bsdf->colors = (float3 *)closure_alloc_extra(sd, sizeof(float3) * 8);
  if (!bsdf->colors) {
    return;
  }

  for (int i = 0; i < 8; i++) {
    bsdf->colors[i] = closure->colors[i];
  }

  sd->runtime_flag |= bsdf_phong_ramp_setup(bsdf);
}

ccl_device void osl_closure_bssrdf_setup(KernelGlobals /*kg*/,
                                         ccl_private ShaderData *sd,
                                         const PathRayVisibility /*path_visibility*/,
                                         const uint32_t path_flag,
                                         const float3 weight,
                                         const ccl_private BSSRDFClosure *closure,
                                         float3 * /*layer_albedo*/)
{
  ClosureType type;
  if (closure->method == make_string("burley", 186330084368958868ull)) {
    type = CLOSURE_BSSRDF_BURLEY_ID;
  }
  else if (closure->method == make_string("random_walk", 11360609267673527222ull)) {
    type = CLOSURE_BSSRDF_RANDOM_WALK_ID;
  }
  else if (closure->method == make_string("random_walk_skin", 3096325052680726300ull)) {
    type = CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID;
  }
  else if (closure->method == make_string("random_walk_legacy", 3162086485308246001ull)) {
    type = CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID;
  }
  else {
    return;
  }

  ccl_private Bssrdf *bssrdf = bssrdf_alloc(sd, rgb_to_spectrum(weight));
  if (!bssrdf) {
    return;
  }

  bssrdf->radius = closure->radius;

  bssrdf->albedo = closure->albedo;
  bssrdf->N = maybe_ensure_valid_specular_reflection(sd,
                                                     safe_normalize_fallback(closure->N, sd->N));
  bssrdf->alpha = closure->roughness;
  bssrdf->ior = closure->ior;
  bssrdf->anisotropy = closure->anisotropy;

  sd->runtime_flag |= bssrdf_setup(sd, bssrdf, path_flag, type);
}

/* MaterialX-compatible subsurface_bssrdf */
ccl_device void osl_closure_subsurface_bssrdf_setup(
    KernelGlobals /*kg*/,
    ccl_private ShaderData *sd,
    const PathRayVisibility /*path_visibility*/,
    const uint32_t path_flag,
    const float3 weight,
    const ccl_private SubsurfaceBSSRDFClosure *closure,
    float3 * /*layer_albedo*/)
{
  ccl_private Bssrdf *bssrdf = bssrdf_alloc(sd, rgb_to_spectrum(closure->albedo * weight));
  if (!bssrdf) {
    return;
  }

#if OSL_LIBRARY_VERSION_CODE >= 11401
  bssrdf->radius = closure->radius;
#else
  bssrdf->radius = closure->transmission_depth * closure->transmission_color;
#endif

  bssrdf->albedo = closure->albedo;
  bssrdf->N = maybe_ensure_valid_specular_reflection(sd,
                                                     safe_normalize_fallback(closure->N, sd->N));
  bssrdf->alpha = 1.0f;
  bssrdf->ior = 1.4f;
  bssrdf->anisotropy = closure->anisotropy;

  sd->runtime_flag |= bssrdf_setup(sd, bssrdf, path_flag, CLOSURE_BSSRDF_RANDOM_WALK_ID);
}

/* Hair */

ccl_device void osl_closure_hair_reflection_setup(KernelGlobals kg,
                                                  ccl_private ShaderData *sd,
                                                  const PathRayVisibility path_visibility,
                                                  const uint32_t /*path_flag*/,
                                                  const float3 weight,
                                                  const ccl_private HairReflectionClosure *closure,
                                                  float3 * /*layer_albedo*/)
{
  if (osl_closure_skip(kg, path_visibility, LABEL_GLOSSY)) {
    return;
  }

  ccl_private HairBsdf *bsdf = (ccl_private HairBsdf *)bsdf_alloc(
      sd, sizeof(HairBsdf), rgb_to_spectrum(weight));
  if (!bsdf) {
    return;
  }

  bsdf->N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  bsdf->T = closure->T;
  bsdf->roughness1 = closure->roughness1;
  bsdf->roughness2 = closure->roughness2;
  bsdf->offset = closure->offset;

  sd->runtime_flag |= bsdf_hair_reflection_setup(bsdf);
}

ccl_device void osl_closure_hair_transmission_setup(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const PathRayVisibility path_visibility,
    const uint32_t /*path_flag*/,
    const float3 weight,
    const ccl_private HairTransmissionClosure *closure,
    float3 * /*layer_albedo*/)
{
  if (osl_closure_skip(kg, path_visibility, LABEL_GLOSSY)) {
    return;
  }

  ccl_private HairBsdf *bsdf = (ccl_private HairBsdf *)bsdf_alloc(
      sd, sizeof(HairBsdf), rgb_to_spectrum(weight));
  if (!bsdf) {
    return;
  }

  bsdf->N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  bsdf->T = closure->T;
  bsdf->roughness1 = closure->roughness1;
  bsdf->roughness2 = closure->roughness2;
  bsdf->offset = closure->offset;

  sd->runtime_flag |= bsdf_hair_transmission_setup(bsdf);
}

ccl_device void osl_closure_hair_chiang_setup(KernelGlobals kg,
                                              ccl_private ShaderData *sd,
                                              const PathRayVisibility path_visibility,
                                              const uint32_t /*path_flag*/,
                                              const float3 weight,
                                              const ccl_private ChiangHairClosure *closure,
                                              float3 * /*layer_albedo*/)
{
#ifdef __HAIR__
  if (osl_closure_skip(kg, path_visibility, LABEL_GLOSSY)) {
    return;
  }

  ccl_private ChiangHairBSDF *bsdf = (ccl_private ChiangHairBSDF *)bsdf_alloc(
      sd, sizeof(ChiangHairBSDF), rgb_to_spectrum(weight));
  if (!bsdf) {
    return;
  }

  bsdf->N = maybe_ensure_valid_specular_reflection(sd, safe_normalize_fallback(closure->N, sd->N));
  bsdf->sigma = closure->sigma;
  bsdf->v = closure->v;
  bsdf->s = closure->s;
  bsdf->alpha = closure->alpha;
  bsdf->eta = closure->eta;
  bsdf->m0_roughness = closure->m0_roughness;

  sd->runtime_flag |= bsdf_hair_chiang_setup(sd, bsdf);
#endif
}

ccl_device void osl_closure_hair_huang_setup(KernelGlobals kg,
                                             ccl_private ShaderData *sd,
                                             const PathRayVisibility path_visibility,
                                             const uint32_t path_flag,
                                             const float3 weight,
                                             const ccl_private HuangHairClosure *closure,
                                             float3 * /*layer_albedo*/)
{
#ifdef __HAIR__
  if (osl_closure_skip(kg, path_visibility, LABEL_GLOSSY)) {
    return;
  }

  if (closure->r_lobe <= 0.0f && closure->tt_lobe <= 0.0f && closure->trt_lobe <= 0.0f) {
    return;
  }

  ccl_private HuangHairBSDF *bsdf = (ccl_private HuangHairBSDF *)bsdf_alloc(
      sd, sizeof(HuangHairBSDF), rgb_to_spectrum(weight));
  if (!bsdf) {
    return;
  }

  ccl_private HuangHairExtra *extra = (ccl_private HuangHairExtra *)closure_alloc_extra(
      sd, sizeof(HuangHairExtra));
  if (!extra) {
    return;
  }

  bsdf->N = safe_normalize_fallback(closure->N, sd->N);
  bsdf->sigma = closure->sigma;
  bsdf->roughness = closure->roughness;
  bsdf->tilt = closure->tilt;
  bsdf->eta = closure->eta;
  bsdf->aspect_ratio = closure->aspect_ratio;

  bsdf->extra = extra;
  bsdf->extra->R = closure->r_lobe;
  bsdf->extra->TT = closure->tt_lobe;
  bsdf->extra->TRT = closure->trt_lobe;

  bsdf->extra->pixel_coverage = 1.0f;

  /* For camera ray, check if the hair covers more than one pixel, in which case a nearfield model
   * is needed to prevent ribbon-like appearance. */
  if ((path_visibility & PATH_RAY_VISIBILITY_CAMERA) && (sd->type & PRIMITIVE_CURVE)) {
    /* Interpolate radius between curve keys. */
    const KernelCurve kcurve = kernel_data_fetch(curves, sd->prim);
    const int k0 = kcurve.first_key + PRIMITIVE_UNPACK_SEGMENT(sd->type);
    const int k1 = k0 + 1;
    const int position_offset = kernel_data_fetch(objects, sd->object).position_offset;
    const float radius = mix(kernel_data_fetch(curve_keys, position_offset + k0).w,
                             kernel_data_fetch(curve_keys, position_offset + k1).w,
                             sd->u);

    bsdf->extra->pixel_coverage = 0.5f * sd->dP / radius;
  }

  sd->runtime_flag |= bsdf_hair_huang_setup(sd, bsdf, path_flag);
#endif
}

/* Volume */

ccl_device void osl_closure_absorption_setup(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const PathRayVisibility /*path_visibility*/,
    const uint32_t /*path_flag*/,
    float3 weight,
    const ccl_private VolumeAbsorptionClosure * /*closure*/,
    float3 * /*layer_albedo*/)
{
  volume_extinction_setup(sd, rgb_to_spectrum(weight * object_volume_density(kg, sd->object)));
}

ccl_device void osl_closure_henyey_greenstein_setup(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const PathRayVisibility /*path_visibility*/,
    const uint32_t /*path_flag*/,
    float3 weight,
    const ccl_private VolumeHenyeyGreensteinClosure *closure,
    float3 * /*layer_albedo*/)
{
  weight *= object_volume_density(kg, sd->object);
  volume_extinction_setup(sd, rgb_to_spectrum(weight));

  ccl_private HenyeyGreensteinVolume *volume = (ccl_private HenyeyGreensteinVolume *)bsdf_alloc(
      sd, sizeof(HenyeyGreensteinVolume), rgb_to_spectrum(weight));
  if (!volume) {
    return;
  }

  volume->g = closure->g;

  sd->runtime_flag |= volume_henyey_greenstein_setup(volume);
}

ccl_device void osl_closure_fournier_forand_setup(
    KernelGlobals /*kg*/,
    ccl_private ShaderData *sd,
    const PathRayVisibility /*path_visibility*/,
    const uint32_t /*path_flag*/,
    const float3 weight,
    const ccl_private VolumeFournierForandClosure *closure,
    float3 * /*layer_albedo*/)
{
  volume_extinction_setup(sd, rgb_to_spectrum(weight));

  ccl_private FournierForandVolume *volume = (ccl_private FournierForandVolume *)bsdf_alloc(
      sd, sizeof(FournierForandVolume), rgb_to_spectrum(weight));
  if (!volume) {
    return;
  }

  sd->runtime_flag |= volume_fournier_forand_setup(volume, closure->B, closure->IOR);
}

ccl_device void osl_closure_draine_setup(KernelGlobals /*kg*/,
                                         ccl_private ShaderData *sd,
                                         const PathRayVisibility /*path_visibility*/,
                                         const uint32_t /*path_flag*/,
                                         const float3 weight,
                                         const ccl_private VolumeDraineClosure *closure,
                                         float3 * /*layer_albedo*/)
{
  volume_extinction_setup(sd, rgb_to_spectrum(weight));

  ccl_private DraineVolume *volume = (ccl_private DraineVolume *)bsdf_alloc(
      sd, sizeof(DraineVolume), rgb_to_spectrum(weight));
  if (!volume) {
    return;
  }

  volume->g = closure->g;
  volume->alpha = closure->alpha;

  sd->runtime_flag |= volume_draine_setup(volume);
}

ccl_device void osl_closure_rayleigh_setup(KernelGlobals /*kg*/,
                                           ccl_private ShaderData *sd,
                                           const PathRayVisibility /*path_visibility*/,
                                           const uint32_t /*path_flag*/,
                                           const float3 weight,
                                           const ccl_private VolumeRayleighClosure * /*closure*/,
                                           float3 * /*layer_albedo*/)
{
  volume_extinction_setup(sd, rgb_to_spectrum(weight));

  ccl_private RayleighVolume *volume = (ccl_private RayleighVolume *)bsdf_alloc(
      sd, sizeof(RayleighVolume), rgb_to_spectrum(weight));
  if (!volume) {
    return;
  }

  sd->runtime_flag |= volume_rayleigh_setup(volume);
}

ccl_device void osl_closure_anisotropic_vdf_setup(KernelGlobals kg,
                                                  ccl_private ShaderData *sd,
                                                  const PathRayVisibility /*path_visibility*/,
                                                  const uint32_t /*path_flag*/,
                                                  float3 weight,
                                                  const ccl_private AnisotropicVDFClosure *closure,
                                                  float3 * /*layer_albedo*/)
{
  if (!(sd->runtime_flag & SR_IS_VOLUME_SHADER_EVAL)) {
    return;
  }

  weight *= object_volume_density(kg, sd->object) * closure->extinction;
  volume_extinction_setup(sd, rgb_to_spectrum(weight));

  ccl_private HenyeyGreensteinVolume *volume = (ccl_private HenyeyGreensteinVolume *)bsdf_alloc(
      sd, sizeof(HenyeyGreensteinVolume), rgb_to_spectrum(weight * closure->albedo));
  if (!volume) {
    return;
  }

  volume->g = closure->anisotropy;

  sd->runtime_flag |= volume_henyey_greenstein_setup(volume);
}

CCL_NAMESPACE_END
