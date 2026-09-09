/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Functions to evaluate shaders. */

#pragma once

#include "kernel/closure/bsdf.h"
#include "kernel/closure/emissive.h"

#include "kernel/film/light_passes.h"

#include "kernel/integrator/guiding.h"
#include "kernel/integrator/state_util.h"
#include "kernel/sample/guiding_resampling.h"

#ifdef __SVM__
#  include "kernel/svm/svm.h"
#endif
#ifdef __OSL__
#  include "kernel/osl/osl.h"
#endif

CCL_NAMESPACE_BEGIN

#ifdef __KERNEL_METAL__
ccl_device_inline bool surface_shader_is_hair_closure(const ClosureType type)
{
  return type == CLOSURE_BSDF_HAIR_REFLECTION_ID || type == CLOSURE_BSDF_HAIR_TRANSMISSION_ID ||
         type == CLOSURE_BSDF_HAIR_CHIANG_ID || type == CLOSURE_BSDF_HAIR_HUANG_ID;
}

/* Photon density estimation is well behaved for diffuse and sufficiently broad non-delta surface
 * lobes. Narrow glossy/transmission lobes remain ray traced. Roughness values returned by Cycles
 * are microfacet alpha, while the UI threshold is artist-facing perceptual roughness. */
ccl_device_inline bool surface_shader_photon_mapping_receiver(ccl_private const ShaderClosure *sc,
                                                              const float3 wi)
{
  if (!CLOSURE_IS_BSDF(sc->type) || CLOSURE_IS_BSDF_SINGULAR(sc->type) ||
      surface_shader_is_hair_closure(sc->type))
  {
    return false;
  }
  if (CLOSURE_IS_BSDF_DIFFUSE(sc->type)) {
    return true;
  }
  if (!(CLOSURE_IS_BSDF_GLOSSY(sc->type) || CLOSURE_IS_BSDF_TRANSMISSION(sc->type) ||
        CLOSURE_IS_GLASS(sc->type)))
  {
    return false;
  }

  float2 roughness;
  float eta;
  bsdf_roughness_eta(sc, wi, &roughness, &eta);
  const float alpha = max(roughness.x, roughness.y);
  const float threshold = sqr(kernel_data.integrator.photon_roughness_threshold);
  return alpha > 1.0e-8f && alpha >= threshold;
}
#endif

/* Guiding */

#if defined(__PATH_GUIDING__)

ccl_device float surface_shader_average_sample_weight_squared_roughness(
    const ccl_private ShaderData *sd)
{
  float avg_squared_roughness = 0.0f;
  float sum_sample_weight = 0.0f;
  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];

    if (!CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
      continue;
    }
    avg_squared_roughness += sc->sample_weight * bsdf_get_specular_roughness_squared(sc);
    sum_sample_weight += sc->sample_weight;
  }

  avg_squared_roughness = avg_squared_roughness > 0.0f ?
                              avg_squared_roughness / sum_sample_weight :
                              0.0f;
  return avg_squared_roughness;
}

ccl_device_inline void surface_shader_prepare_guiding(KernelGlobals kg,
                                                      IntegratorState state,
                                                      ccl_private ShaderData *sd,
                                                      const ccl_private RNGState *rng_state)
{
  /* Have any BSDF to guide? */
  if (!(kernel_data.integrator.use_surface_guiding && (sd->runtime_flag & SR_BSDF_HAS_EVAL))) {
    INTEGRATOR_STATE_WRITE(state, guiding, use_surface_guiding) = false;
    return;
  }

  const float surface_guiding_probability = kernel_data.integrator.surface_guiding_probability;
  const int guiding_directional_sampling_type =
      kernel_data.integrator.guiding_directional_sampling_type;
  const float guiding_roughness_threshold = kernel_data.integrator.guiding_roughness_threshold;
  float rand_bsdf_guiding = path_state_rng_1D(kg, rng_state, PRNG_SURFACE_BSDF_GUIDING);

  /* Compute proportion of diffuse BSDF and BSSRDFs. */
  float diffuse_sampling_fraction = 0.0f;
  float bssrdf_sampling_fraction = 0.0f;
  float bsdf_bssrdf_sampling_sum = 0.0f;

  bool fully_opaque = true;

  for (int i = 0; i < sd->num_closure; i++) {
    ShaderClosure *sc = &sd->closure[i];
    if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
      const float sweight = sc->sample_weight;
      kernel_assert(sweight >= 0.0f);

      bsdf_bssrdf_sampling_sum += sweight;
      if (CLOSURE_IS_BSDF_DIFFUSE(sc->type) && sc->type < CLOSURE_BSDF_TRANSLUCENT_ID) {
        diffuse_sampling_fraction += sweight;
      }
      if (CLOSURE_IS_BSSRDF(sc->type)) {
        bssrdf_sampling_fraction += sweight;
      }

      if (CLOSURE_IS_BSDF_TRANSPARENT(sc->type) || CLOSURE_IS_BSDF_TRANSMISSION(sc->type)) {
        fully_opaque = false;
      }
    }
  }

  if (bsdf_bssrdf_sampling_sum > 0.0f) {
    diffuse_sampling_fraction /= bsdf_bssrdf_sampling_sum;
    bssrdf_sampling_fraction /= bsdf_bssrdf_sampling_sum;
  }

  /* Initial guiding */
  /* The roughness because the function returns `alpha.x * alpha.y`.
   * In addition alpha is squared again. */
  float avg_roughness = surface_shader_average_sample_weight_squared_roughness(sd);
  avg_roughness = safe_sqrtf(avg_roughness);
  if (!fully_opaque || avg_roughness < guiding_roughness_threshold ||
      ((guiding_directional_sampling_type == GUIDING_DIRECTIONAL_SAMPLING_TYPE_PRODUCT_MIS) &&
       (diffuse_sampling_fraction <= 0.0f)) ||
      !guiding_bsdf_init(kg, sd->P, sd->N, rand_bsdf_guiding))
  {
    INTEGRATOR_STATE_WRITE(state, guiding, use_surface_guiding) = false;
    INTEGRATOR_STATE_WRITE(state, guiding, surface_guiding_sampling_prob) = 0.f;
    return;
  }

  INTEGRATOR_STATE_WRITE(state, guiding, use_surface_guiding) = true;
  if (kernel_data.integrator.guiding_directional_sampling_type ==
      GUIDING_DIRECTIONAL_SAMPLING_TYPE_PRODUCT_MIS)
  {
    INTEGRATOR_STATE_WRITE(
        state, guiding, surface_guiding_sampling_prob) = surface_guiding_probability *
                                                         diffuse_sampling_fraction;
  }
  else if (kernel_data.integrator.guiding_directional_sampling_type ==
           GUIDING_DIRECTIONAL_SAMPLING_TYPE_RIS)
  {
    INTEGRATOR_STATE_WRITE(
        state, guiding, surface_guiding_sampling_prob) = surface_guiding_probability;
  }
  else {  // GUIDING_DIRECTIONAL_SAMPLING_TYPE_ROUGHNESS
    INTEGRATOR_STATE_WRITE(
        state, guiding, surface_guiding_sampling_prob) = surface_guiding_probability *
                                                         avg_roughness;
  }
  INTEGRATOR_STATE_WRITE(state, guiding, bssrdf_sampling_prob) = bssrdf_sampling_fraction;
  INTEGRATOR_STATE_WRITE(state, guiding, sample_surface_guiding_rand) = rand_bsdf_guiding;

  kernel_assert(INTEGRATOR_STATE(state, guiding, surface_guiding_sampling_prob) > 0.0f &&
                INTEGRATOR_STATE(state, guiding, surface_guiding_sampling_prob) <= 1.0f);
}
#endif

ccl_device_inline void surface_shader_prepare_closures(KernelGlobals kg,
                                                       ConstIntegratorState state,
                                                       ccl_private ShaderData *sd,
                                                       const PathRayVisibility path_visibility)
{
  /* Filter out closures. */
  if (kernel_data.integrator.filter_closures) {
    const int filter_closures = kernel_data.integrator.filter_closures;
    if (filter_closures & FILTER_CLOSURE_EMISSION) {
      sd->closure_emission_background = zero_spectrum();
    }

    if (path_visibility & PATH_RAY_VISIBILITY_CAMERA) {
      if (filter_closures & FILTER_CLOSURE_DIRECT_LIGHT) {
        sd->runtime_flag &= ~SR_BSDF_HAS_EVAL;
      }
      bool has_bsdf_closure = false;
      for (int i = 0; i < sd->num_closure; i++) {
        ccl_private ShaderClosure *sc = &sd->closure[i];

        const bool filter_diffuse = (filter_closures & FILTER_CLOSURE_DIFFUSE);
        const bool filter_glossy = (filter_closures & FILTER_CLOSURE_GLOSSY);
        const bool filter_transmission = (filter_closures & FILTER_CLOSURE_TRANSMISSION);
        const bool filter_glass = filter_glossy && filter_transmission;
        if ((CLOSURE_IS_BSDF_DIFFUSE(sc->type) && filter_diffuse) ||
            (CLOSURE_IS_BSDF_GLOSSY(sc->type) && filter_glossy) ||
            (CLOSURE_IS_BSDF_TRANSMISSION(sc->type) && filter_transmission) ||
            (CLOSURE_IS_GLASS(sc->type) && filter_glass))
        {
          sc->type = CLOSURE_NONE_ID;
          sc->sample_weight = 0.0f;
        }
        else if ((CLOSURE_IS_BSDF_TRANSPARENT(sc->type) &&
                  (filter_closures & FILTER_CLOSURE_TRANSPARENT)))
        {
          sc->type = CLOSURE_HOLDOUT_ID;
          sc->sample_weight = 0.0f;
          sd->runtime_flag |= SR_HOLDOUT;
        }
        else if (CLOSURE_IS_BSDF(sc->type)) {
          has_bsdf_closure = true;
        }
      }
      if (!has_bsdf_closure) {
        sd->runtime_flag &= ~SR_BSDF;
      }
    }
  }

#ifdef __KERNEL_METAL__
  /* Partition caustic transport without overlap: photon mapping handles sufficiently sharp
   * glossy/transmission lobes reached after a supported broad receiver event, while ordinary path
   * tracing keeps rough caustics and unsupported receivers. */
  if (kernel_data.integrator.use_photon_mapping &&
      (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_PHOTON_MAPPING_RECEIVER) &&
      !(INTEGRATOR_STATE(state, path, flag) & PATH_RAY_PHOTON_MAPPING_UNSUPPORTED))
  {
    for (int i = 0; i < sd->num_closure; i++) {
      ccl_private ShaderClosure *sc = &sd->closure[i];
      if (!surface_shader_is_hair_closure(sc->type) && !CLOSURE_IS_BSDF_DIFFUSE(sc->type) &&
          (CLOSURE_IS_BSDF_GLOSSY(sc->type) || CLOSURE_IS_BSDF_TRANSMISSION(sc->type) ||
           CLOSURE_IS_GLASS(sc->type)))
      {
        if (!surface_shader_photon_mapping_receiver(sc, sd->wi)) {
          sc->type = CLOSURE_NONE_ID;
          sc->sample_weight = 0.0f;
        }
      }
    }
  }
#endif

  /* Filter glossy.
   *
   * Blurring of bsdf after bounces, for rays that have a small likelihood
   * of following this particular path (diffuse, rough glossy) */
  if (kernel_data.integrator.filter_glossy != FLT_MAX
#ifdef __MNEE__
      && !(INTEGRATOR_STATE(state, path, mnee) & PATH_MNEE_VALID)
#endif
  )
  {
    const float blur_pdf = kernel_data.integrator.filter_glossy *
                           INTEGRATOR_STATE(state, path, min_ray_pdf);

    if (blur_pdf < 1.0f) {
      const float blur_roughness = sqrtf(1.0f - blur_pdf) * 0.5f;

      for (int i = 0; i < sd->num_closure; i++) {
        ccl_private ShaderClosure *sc = &sd->closure[i];
        if (CLOSURE_IS_BSDF(sc->type)) {
          bsdf_blur(sc, blur_roughness);
        }
      }

      /* NOTE: this is a sufficient condition. If `blur_roughness < THRESH < original_roughness`
       * then the flag was already set. */
      if (!roughness_is_almost_specular(blur_roughness, blur_roughness)) {
        sd->runtime_flag |= SR_BSDF_HAS_EVAL;
      }
    }
  }
}

/* BSDF */
#ifdef WITH_CYCLES_DEBUG
ccl_device_inline void surface_shader_validate_bsdf_sample(const KernelGlobals kg,
                                                           const ccl_private ShaderClosure *sc,
                                                           const float3 wo,
                                                           const int org_label,
                                                           const float2 org_roughness,
                                                           const float org_eta)
{
  /* Validate the #bsdf_label and #bsdf_roughness_eta functions
   * by estimating the values after a BSDF sample. */
  kernel_assert(org_label == bsdf_label(kg, sc, wo));
  (void)kg;

  float2 comp_roughness;
  float comp_eta;
  bsdf_roughness_eta(sc, wo, &comp_roughness, &comp_eta);
  kernel_assert(org_eta == comp_eta);
  kernel_assert(org_roughness.x == comp_roughness.x);
  kernel_assert(org_roughness.y == comp_roughness.y);
}
#endif

ccl_device_forceinline bool _surface_shader_exclude(ClosureType type,
                                                    const uint light_shader_flags)
{
  if (!(light_shader_flags & SHADER_EXCLUDE_ANY)) {
    return false;
  }
  if (light_shader_flags & SHADER_EXCLUDE_DIFFUSE) {
    if (CLOSURE_IS_BSDF_DIFFUSE(type)) {
      return true;
    }
  }
  if (light_shader_flags & SHADER_EXCLUDE_GLOSSY) {
    if (CLOSURE_IS_BSDF_GLOSSY(type)) {
      return true;
    }
  }
  if (light_shader_flags & SHADER_EXCLUDE_TRANSMIT) {
    if (CLOSURE_IS_BSDF_TRANSMISSION(type)) {
      return true;
    }
  }
  /* Glass closures are both glossy and transmissive, so only exclude them if both are filtered. */
  const uint exclude_glass = SHADER_EXCLUDE_TRANSMIT | SHADER_EXCLUDE_GLOSSY;
  if ((light_shader_flags & exclude_glass) == exclude_glass) {
    if (CLOSURE_IS_GLASS(type)) {
      return true;
    }
  }
  return false;
}

ccl_device_inline float _surface_shader_bsdf_eval_mis(KernelGlobals kg,
                                                      ccl_private ShaderData *sd,
                                                      const float3 wo,
                                                      const ccl_private ShaderClosure *skip_sc,
                                                      ccl_private BsdfEval *result_eval,
                                                      float sum_pdf,
                                                      float sum_sample_weight,
                                                      const uint light_shader_flags,
                                                      float sum_pdf_roughness_squared,
                                                      ccl_private float &r_avg_roughness_squared)
{
  /* This is the veach one-sample model with balance heuristic,
   * some PDF factors drop out when using balance heuristic weighting. */
  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];

    if (sc == skip_sc) {
      continue;
    }

    if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
      if (CLOSURE_IS_BSDF(sc->type)) {
        float bsdf_pdf = 0.0f;
        const Spectrum eval = bsdf_eval(kg, sd, sc, wo, &bsdf_pdf);

        if (bsdf_pdf != 0.0f) {
          if (!_surface_shader_exclude(sc->type, light_shader_flags)) {
            bsdf_eval_accum(result_eval, sc, wo, eval * sc->weight);
          }

          const float weight = bsdf_pdf * sc->sample_weight;
          sum_pdf += weight;

          /* See #bsdf_widen_dD for the motivation for this weighted average roughness. */
          sum_pdf_roughness_squared += weight * bsdf_get_specular_roughness_squared(sc);
        }
      }

      sum_sample_weight += sc->sample_weight;
    }
  }

  r_avg_roughness_squared = (sum_pdf > 0.0f) ? sum_pdf_roughness_squared / sum_pdf : 0.0f;

  return (sum_sample_weight > 0.0f) ? sum_pdf / sum_sample_weight : 0.0f;
}

ccl_device_inline float surface_shader_bsdf_eval_pdfs(const KernelGlobals kg,
                                                      ccl_private ShaderData *sd,
                                                      const float3 wo,
                                                      ccl_private BsdfEval *result_eval,
                                                      ccl_private float *pdfs,
                                                      const uint light_shader_flags,
                                                      ccl_private float &r_avg_roughness_squared)
{
  /* This is the veach one-sample model with balance heuristic, some pdf
   * factors drop out when using balance heuristic weighting. */
  float sum_pdf = 0.0f;
  float sum_sample_weight = 0.0f;
  float sum_pdf_roughness_squared = 0.0f;
  bsdf_eval_init(result_eval, zero_spectrum());
  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];

    if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
      if (CLOSURE_IS_BSDF(sc->type)) {
        float bsdf_pdf = 0.0f;
        const Spectrum eval = bsdf_eval(kg, sd, sc, wo, &bsdf_pdf);
        kernel_assert(bsdf_pdf >= 0.0f);
        if (bsdf_pdf != 0.0f) {
          if (!_surface_shader_exclude(sc->type, light_shader_flags)) {
            bsdf_eval_accum(result_eval, sc, wo, eval * sc->weight);
          }

          const float weight = bsdf_pdf * sc->sample_weight;
          sum_pdf += weight;
          kernel_assert(weight >= 0.0f);
          pdfs[i] = weight;

          /* See #bsdf_widen_dD for the motivation for this weighted average roughness. */
          sum_pdf_roughness_squared += weight * bsdf_get_specular_roughness_squared(sc);
        }
        else {
          pdfs[i] = 0.0f;
        }
      }
      else {
        pdfs[i] = 0.0f;
      }

      sum_sample_weight += sc->sample_weight;
    }
    else {
      pdfs[i] = 0.0f;
    }
  }
  if (sum_pdf > 0.0f) {
    for (int i = 0; i < sd->num_closure; i++) {
      pdfs[i] /= sum_pdf;
    }
  }

  r_avg_roughness_squared = (sum_pdf > 0.0f) ? sum_pdf_roughness_squared / sum_pdf : 0.0f;

  return (sum_sample_weight > 0.0f) ? sum_pdf / sum_sample_weight : 0.0f;
}

#ifdef __KERNEL_METAL__
ccl_device_inline float3 surface_shader_gpu_guiding_normal(const ccl_private ShaderData *sd)
{
  /* Orient the product toward this query's incoming direction, including transmission. */
  return dot(sd->N, sd->wi) >= 0.0f ? sd->N : -sd->N;
}

ccl_device_inline bool surface_shader_gpu_guiding_continuous(const ccl_private ShaderClosure *sc)
{
  if (!CLOSURE_IS_BSDF(sc->type) || CLOSURE_IS_BSDF_SINGULAR(sc->type) ||
      CLOSURE_IS_BSDF_TRANSPARENT(sc->type) || CLOSURE_IS_RAY_PORTAL(sc->type))
  {
    return false;
  }
  if (CLOSURE_IS_BSDF_MICROFACET(sc->type)) {
    return bsdf_microfacet_eval_flag((const ccl_private MicrofacetBsdf *)sc) != 0;
  }
  return true;
}

/* A single query supplies both sampling and density evaluation. The field is immutable for
 * the entire render batch, so retaining this pointer avoids repeated spatial traversal. */
struct SurfaceGuidingProposal {
  const ccl_global float *weights;
  float3 position;
  float probability;
  float bsdf_fraction;
  GuidingDirectionalProduct product;
  bool use_smooth_product;
  GuidingGaussianProduct smooth_product;
  bool use_resampling;
};

ccl_device_inline GuidingGaussianProduct
surface_shader_gpu_guiding_glossy_product(const ccl_private ShaderData *sd)
{
  GuidingGaussianProduct product{};
  float3 axes[2] = {zero_float3(), zero_float3()};
  float concentrations[2] = {0.0f, 0.0f};
  for (int i = 0; i < sd->num_closure; ++i) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];
    if (!surface_shader_gpu_guiding_continuous(sc)) {
      continue;
    }
    const ccl_private MicrofacetBsdf *bsdf = (const ccl_private MicrofacetBsdf *)sc;
    const float alpha = max(bsdf->alpha_x, bsdf->alpha_y);
    const bool thin = bsdf->type == CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID;
    const bool refractive = CLOSURE_IS_REFRACTION(bsdf->type) || CLOSURE_IS_GLASS(bsdf->type);
    const float inverse_ior = refractive ? safe_divide(1.0f, bsdf->ior) : 1.0f;
    const float3 transmission = thin       ? -sd->wi :
                                refractive ? refract(-sd->wi, bsdf->N, inverse_ior) :
                                             zero_float3();
    float reflection = (CLOSURE_IS_REFRACTION(bsdf->type) || thin) ? 0.0f : 1.0f;
    if (CLOSURE_IS_GLASS(bsdf->type)) {
      reflection = fresnel_dielectric_cos(max(dot(bsdf->N, sd->wi), 0.0f), bsdf->ior);
    }
    const float weights[2] = {sc->sample_weight * reflection,
                              is_zero(transmission) ? 0.0f :
                                                      sc->sample_weight * (1.0f - reflection)};
    const float3 directions[2] = {reflect(-sd->wi, bsdf->N), transmission};
    const float widths[2] = {alpha, thin ? alpha : alpha * fabsf(1.0f - inverse_ior)};
    for (int lobe = 0; lobe < 2; ++lobe) {
      /* Match the central width of a GGX lobe with a smooth proposal. The ordinary BSDF
       * component retains its longer tails; actual scattering values are never approximated. */
      const float concentration = min(0.35f / max(sqr(widths[lobe]), 1e-8f), 16384.0f);
      product.weights[lobe] += weights[lobe];
      axes[lobe] += weights[lobe] * directions[lobe];
      concentrations[lobe] += weights[lobe] * concentration;
    }
  }
  for (int lobe = 0; lobe < 2; ++lobe) {
    const float length = len(axes[lobe]);
    product.lobes[lobe] = {length > 0.0f ? axes[lobe] / length : make_float3(0, 0, 1),
                           safe_divide(concentrations[lobe], product.weights[lobe])};
  }
  const float mass = product.weights[0] + product.weights[1];
  product.weights[0] = safe_divide(product.weights[0], mass);
  product.weights[1] = safe_divide(product.weights[1], mass);
  return product;
}

ccl_device_inline SurfaceGuidingProposal
surface_shader_gpu_guiding_query(const ccl_private ShaderData *sd, const bool light_path)
{
  SurfaceGuidingProposal proposal{};
  if (!kernel_data.integrator.use_surface_guiding ||
      kernel_integrator_state.guiding_capacity == 0 ||
      !(kernel_data.integrator.surface_guiding_probability > 0.0f) ||
      !(sd->runtime_flag & SR_BSDF_HAS_EVAL))
  {
    return proposal;
  }
  float total_weight = 0.0f;
  float bsdf_weight = 0.0f;
  float roughness = 0.0f;
  bool microfacet_only = true;
  for (int i = 0; i < sd->num_closure; ++i) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];
    if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
      if ((CLOSURE_IS_GLASS(sc->type) || CLOSURE_IS_REFRACTION(sc->type)) &&
          fabsf(((const ccl_private MicrofacetBsdf *)sc)->ior - 1.0f) < 1e-4f)
      {
        /* The microfacet sampler treats index-matched transmission as a Dirac event even
         * at nonzero roughness. Keep this whole surface on its original closure sampler;
         * its tiny continuous reflection branch and discrete transmission share one closure. */
        return proposal;
      }
      total_weight += sc->sample_weight;
      if (surface_shader_gpu_guiding_continuous(sc)) {
        bsdf_weight += sc->sample_weight;
        roughness += sc->sample_weight * bsdf_get_specular_roughness_squared(sc);
        microfacet_only &= CLOSURE_IS_BSDF_MICROFACET(sc->type);
      }
    }
  }
  if (!(bsdf_weight > 0.0f) || !(roughness > 0.0f) ||
      safe_sqrtf(roughness / bsdf_weight) < kernel_data.integrator.guiding_roughness_threshold)
  {
    return proposal;
  }
  const GuidingField field = guiding_gpu_field();
  const GuidingFieldType type = guiding_surface_field_type(guiding_gpu_surface_orientation(sd),
                                                           light_path);
  proposal.position = field.normalized_position(sd->P);
  proposal.weights = field.distribution(field.find_leaf(sd->P), type);
  if (!(proposal.weights[0] > 0.0f)) {
    return proposal;
  }
  proposal.bsdf_fraction = bsdf_weight / total_weight;
  proposal.probability = clamp(kernel_data.integrator.surface_guiding_probability, 0.0f, 1.0f);
  proposal.use_resampling = bsdf_weight == total_weight &&
                            kernel_data.integrator.guiding_directional_sampling_type ==
                                GUIDING_DIRECTIONAL_SAMPLING_TYPE_RIS;
  if (kernel_data.integrator.guiding_directional_sampling_type ==
      GUIDING_DIRECTIONAL_SAMPLING_TYPE_ROUGHNESS)
  {
    proposal.probability *= saturatef(safe_sqrtf(roughness / bsdf_weight));
  }
  proposal.product = {surface_shader_gpu_guiding_normal(sd),
                      0.0f,
                      sd->runtime_flag & SR_BSDF_HAS_TRANSMISSION ?
                          GuidingDirectionalProduct::TWO_SIDED_COSINE :
                          GuidingDirectionalProduct::COSINE};
  /* Resampling uses smooth incident-radiance targets and candidate products. The direct
   * mixture retains the adaptive histogram on broad surfaces and smooth narrow glossy lobes. */
  /* Alpha is the square of the user-facing roughness. */
  const bool glossy_product = microfacet_only && safe_sqrtf(roughness / bsdf_weight) < 0.25f;
  proposal.use_smooth_product = glossy_product || proposal.use_resampling;
  if (glossy_product) {
    proposal.smooth_product = surface_shader_gpu_guiding_glossy_product(sd);
  }
  else if (proposal.use_resampling) {
    /* A broad vMF lobe approximately matches the cosine hemisphere's first moment.
     * RIS evaluates the actual BSDF; this smooth product is only its candidate proposal. */
    const bool two_sided = sd->runtime_flag & SR_BSDF_HAS_TRANSMISSION;
    proposal.smooth_product = {{{proposal.product.axis, 3.0f}, {-proposal.product.axis, 3.0f}},
                               {two_sided ? 0.5f : 1.0f, two_sided ? 0.5f : 0.0f}};
  }
  return proposal;
}

ccl_device_inline float surface_shader_gpu_guiding_distribution_pdf(
    const ccl_private SurfaceGuidingProposal &proposal, const float3 direction)
{
  GuidingField::DirectionalTree directional;
  GuidingGaussianMixture smooth;
  return proposal.use_smooth_product ?
             smooth.pdf_product(proposal.weights + GuidingField::tree_size,
                                proposal.smooth_product,
                                direction,
                                proposal.position) :
             directional.pdf_product(proposal.weights, direction, proposal.product);
}

ccl_device_inline float3
surface_shader_gpu_guiding_distribution_sample(const ccl_private SurfaceGuidingProposal &proposal,
                                               const float2 random,
                                               ccl_private float *pdf)
{
  GuidingField::DirectionalTree directional;
  GuidingGaussianMixture smooth;
  return proposal.use_smooth_product ?
             smooth.sample_product(proposal.weights + GuidingField::tree_size,
                                   proposal.smooth_product,
                                   random,
                                   pdf,
                                   proposal.position) :
             directional.sample_product(proposal.weights, random, proposal.product, pdf);
}

ccl_device_inline float surface_shader_gpu_guiding_pdf_from_query(
    const ccl_private SurfaceGuidingProposal &proposal,
    const float3 direction,
    const float bsdf_pdf)
{
  /* Guided candidates are accepted only where the underlying continuous BSDF has support.
   * Do not assign MIS weight to an unreachable reverse strategy just because the learned
   * proposal has uniform exploration outside that support. */
  if (!(bsdf_pdf > 0.0f)) {
    return 0.0f;
  }
  if (proposal.probability == 0.0f) {
    return bsdf_pdf;
  }
  const float guide_pdf = surface_shader_gpu_guiding_distribution_pdf(proposal, direction);
  const float probability = proposal.use_resampling ? 0.5f : proposal.probability;
  return (1.0f - probability) * bsdf_pdf + probability * proposal.bsdf_fraction * guide_pdf;
}

ccl_device_inline float surface_shader_gpu_guiding_pdf(const ccl_private ShaderData *sd,
                                                       const float3 direction,
                                                       const float bsdf_pdf,
                                                       const bool light_path)
{
  const SurfaceGuidingProposal proposal = surface_shader_gpu_guiding_query(sd, light_path);
  return surface_shader_gpu_guiding_pdf_from_query(proposal, direction, bsdf_pdf);
}

#endif

#ifndef __KERNEL_CUDA__
ccl_device
#else
ccl_device_inline
#endif
    float
    surface_shader_bsdf_eval(KernelGlobals kg,
                             ccl_attr_maybe_unused IntegratorState state,
                             ccl_private ShaderData *sd,
                             const float3 wo,
                             ccl_private BsdfEval *bsdf_eval,
                             const uint light_shader_flags,
                             ccl_private float &r_avg_roughness_squared,
                             ccl_attr_maybe_unused const bool guiding_light_path = false)
{
  bsdf_eval_init(bsdf_eval, zero_spectrum());

  float pdf = _surface_shader_bsdf_eval_mis(kg,
                                            sd,
                                            wo,
                                            nullptr,
                                            bsdf_eval,
                                            0.0f,
                                            0.0f,
                                            light_shader_flags,
                                            0.0f,
                                            r_avg_roughness_squared);

  /* If the light does not use MIS, then it is only sampled via NEE, so the probability of hitting
   * the light using BSDF sampling is zero. */
  if (!(light_shader_flags & SHADER_USE_MIS)) {
    pdf = 0.0f;
  }

#if defined(__PATH_GUIDING__) && PATH_GUIDING_LEVEL >= 4
  if ((kernel_data.kernel_features & KERNEL_FEATURE_PATH_GUIDING)) {
    if (pdf > 0.0f && INTEGRATOR_STATE(state, guiding, use_surface_guiding)) {
      const float guiding_sampling_prob = INTEGRATOR_STATE(
          state, guiding, surface_guiding_sampling_prob);
      const float bssrdf_sampling_prob = INTEGRATOR_STATE(state, guiding, bssrdf_sampling_prob);
      const float guide_pdf = guiding_bsdf_pdf(kg, wo);

      if (kernel_data.integrator.guiding_directional_sampling_type ==
          GUIDING_DIRECTIONAL_SAMPLING_TYPE_RIS)
      {
        pdf = (0.5f * guide_pdf * (1.0f - bssrdf_sampling_prob)) + 0.5f * pdf;
      }
      else {
        pdf = (guiding_sampling_prob * guide_pdf * (1.0f - bssrdf_sampling_prob)) +
              (1.0f - guiding_sampling_prob) * pdf;
      }
    }
  }
#endif

#ifdef __KERNEL_METAL__
  if (light_shader_flags & SHADER_USE_MIS) {
    pdf = surface_shader_gpu_guiding_pdf(sd, wo, pdf, guiding_light_path);
  }
#endif

  return pdf;
}

/* Randomly sample a BSSRDF or BSDF proportional to ShaderClosure.sample_weight. */
const ccl_device_inline ccl_private ShaderClosure *surface_shader_bsdf_bssrdf_pick(
    const ccl_private ShaderData *ccl_restrict sd, ccl_private float3 *rand_bsdf)
{
  int sampled = 0;

  if (sd->num_closure > 1) {
    /* Pick a BSDF or based on sample weights. */
    float sum = 0.0f;

    for (int i = 0; i < sd->num_closure; i++) {
      const ccl_private ShaderClosure *sc = &sd->closure[i];

      if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
        sum += sc->sample_weight;
      }
    }

    const float r = (*rand_bsdf).z * sum;
    float partial_sum = 0.0f;

    for (int i = 0; i < sd->num_closure; i++) {
      const ccl_private ShaderClosure *sc = &sd->closure[i];

      if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
        const float next_sum = partial_sum + sc->sample_weight;

        if (r < next_sum) {
          sampled = i;

          /* Rescale to reuse for direction sample, to better preserve stratification. */
          (*rand_bsdf).z = (r - partial_sum) / sc->sample_weight;
          break;
        }

        partial_sum = next_sum;
      }
    }
  }

  return &sd->closure[sampled];
}

/* Return weight for picked BSSRDF. */
ccl_device_inline Spectrum
surface_shader_bssrdf_sample_weight(const ccl_private ShaderData *ccl_restrict sd,
                                    const ccl_private ShaderClosure *ccl_restrict bssrdf_sc)
{
  Spectrum weight = bssrdf_sc->weight;

  if (sd->num_closure > 1) {
    float sum = 0.0f;
    for (int i = 0; i < sd->num_closure; i++) {
      const ccl_private ShaderClosure *sc = &sd->closure[i];

      if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
        sum += sc->sample_weight;
      }
    }
    weight *= sum / bssrdf_sc->sample_weight;
  }

  return weight;
}

#if defined(__PATH_GUIDING__)
/* Sample direction for picked BSDF, and return evaluation and pdf for all
 * BSDFs combined using MIS. */

ccl_device int surface_shader_bsdf_guided_sample_closure_mis(
    KernelGlobals kg,
    IntegratorState state,
    ccl_private ShaderData *sd,
    const ccl_private ShaderClosure *sc,
    const float3 rand_bsdf,
    ccl_private BsdfEval *bsdf_eval,
    ccl_private float3 *wo,
    ccl_private float *bsdf_pdf,
    ccl_private float *unguided_bsdf_pdf,
    ccl_private float2 *sampled_roughness,
    ccl_private float *eta,
    ccl_private float &r_avg_roughness_squared)
{
  /* BSSRDF should already have been handled elsewhere. */
  kernel_assert(CLOSURE_IS_BSDF(sc->type));

  r_avg_roughness_squared = 0.0f;

  const bool use_surface_guiding = INTEGRATOR_STATE(state, guiding, use_surface_guiding);
  const float guiding_sampling_prob = INTEGRATOR_STATE(
      state, guiding, surface_guiding_sampling_prob);
  const float bssrdf_sampling_prob = INTEGRATOR_STATE(state, guiding, bssrdf_sampling_prob);

  /* Decide between sampling guiding distribution and BSDF. */
  bool sample_guiding = false;
  float rand_bsdf_guiding = INTEGRATOR_STATE(state, guiding, sample_surface_guiding_rand);

  if (use_surface_guiding && rand_bsdf_guiding < guiding_sampling_prob) {
    sample_guiding = true;
    rand_bsdf_guiding /= guiding_sampling_prob;
  }
  else {
    rand_bsdf_guiding -= guiding_sampling_prob;
    rand_bsdf_guiding /= (1.0f - guiding_sampling_prob);
  }

  /* Initialize to zero. */
  int label = LABEL_NONE;
  Spectrum eval = zero_spectrum();
  bsdf_eval_init(bsdf_eval, eval);

  *unguided_bsdf_pdf = 0.0f;
  float guide_pdf = 0.0f;

  if (sample_guiding) {
    /* Sample guiding distribution. */
    guide_pdf = guiding_bsdf_sample(kg, make_float2(rand_bsdf), wo);
    *bsdf_pdf = 0.0f;

    if (guide_pdf != 0.0f) {
      float unguided_bsdf_pdfs[MAX_CLOSURE];

      *unguided_bsdf_pdf = surface_shader_bsdf_eval_pdfs(
          kg, sd, *wo, bsdf_eval, unguided_bsdf_pdfs, 0, r_avg_roughness_squared);
      *bsdf_pdf = (guiding_sampling_prob * guide_pdf * (1.0f - bssrdf_sampling_prob)) +
                  ((1.0f - guiding_sampling_prob) * (*unguided_bsdf_pdf));
      float sum_pdfs = 0.0f;

      if (*unguided_bsdf_pdf > 0.0f) {
        int idx = -1;
        for (int i = 0; i < sd->num_closure; i++) {
          sum_pdfs += unguided_bsdf_pdfs[i];
          if (rand_bsdf_guiding <= sum_pdfs) {
            idx = i;
            break;
          }
        }

        kernel_assert(idx >= 0);
        /* Set the default idx to the last in the list.
         * in case of numerical problems and rand_bsdf_guiding is just >=1.0f and
         * the sum of all unguided_bsdf_pdfs is just < 1.0f. */
        idx = (rand_bsdf_guiding > sum_pdfs) ? sd->num_closure - 1 : idx;

        label = bsdf_label(kg, &sd->closure[idx], *wo);
      }
      else {
        *bsdf_pdf = 0.0f;
        *unguided_bsdf_pdf = 0.0f;
      }
    }

    kernel_assert(reduce_min(bsdf_eval_sum(bsdf_eval)) >= 0.0f);

    *sampled_roughness = make_float2(1.0f, 1.0f);
    *eta = 1.0f;
  }
  else {
    /* Sample BSDF. */
    *bsdf_pdf = 0.0f;
    label = bsdf_sample(
        kg, sd, sc, rand_bsdf, &eval, wo, unguided_bsdf_pdf, sampled_roughness, eta);
#  ifdef WITH_CYCLES_DEBUG
    /* Code path to validate the estimation of the label, sampled roughness and eta. This should be
     * activated from time to time when the BSDFs change to check if everything is still working
     * correctly. */
    if (*unguided_bsdf_pdf > 0.0f) {
      surface_shader_validate_bsdf_sample(kg, sc, *wo, label, *sampled_roughness, *eta);
    }
#  endif

    if (*unguided_bsdf_pdf != 0.0f) {
      bsdf_eval_init(bsdf_eval, sc, *wo, eval * sc->weight);

      kernel_assert(reduce_min(bsdf_eval_sum(bsdf_eval)) >= 0.0f);

      const float roughness_squared = bsdf_get_specular_roughness_squared(sc);
      if (sd->num_closure > 1) {
        const float sweight = sc->sample_weight;
        const float weight = (*unguided_bsdf_pdf) * sweight;
        *unguided_bsdf_pdf = _surface_shader_bsdf_eval_mis(kg,
                                                           sd,
                                                           *wo,
                                                           sc,
                                                           bsdf_eval,
                                                           weight,
                                                           sweight,
                                                           0,
                                                           weight * roughness_squared,
                                                           r_avg_roughness_squared);
        kernel_assert(reduce_min(bsdf_eval_sum(bsdf_eval)) >= 0.0f);
      }
      else {
        r_avg_roughness_squared = roughness_squared;
      }
      *bsdf_pdf = *unguided_bsdf_pdf;

      if (use_surface_guiding) {
        guide_pdf = guiding_bsdf_pdf(kg, *wo);
        *bsdf_pdf *= 1.0f - guiding_sampling_prob;
        *bsdf_pdf += guiding_sampling_prob * guide_pdf * (1.0f - bssrdf_sampling_prob);
      }
    }

    kernel_assert(reduce_min(bsdf_eval_sum(bsdf_eval)) >= 0.0f);
  }

  return label;
}

ccl_device int surface_shader_bsdf_guided_sample_closure_ris(
    KernelGlobals kg,
    IntegratorState state,
    ccl_private ShaderData *sd,
    const ccl_private ShaderClosure *sc,
    const float3 rand_bsdf,
    const ccl_private RNGState *rng_state,
    ccl_private BsdfEval *bsdf_eval,
    ccl_private float3 *wo,
    ccl_private float *bsdf_pdf,
    ccl_private float *mis_pdf,
    ccl_private float *unguided_bsdf_pdf,
    ccl_private float2 *sampled_roughness,
    ccl_private float *eta,
    ccl_private float &r_avg_roughness_squared)
{
  /* BSSRDF should already have been handled elsewhere. */
  kernel_assert(CLOSURE_IS_BSDF(sc->type));

  r_avg_roughness_squared = 0.0f;

  const bool use_surface_guiding = INTEGRATOR_STATE(state, guiding, use_surface_guiding);
  const float guiding_sampling_prob = INTEGRATOR_STATE(
      state, guiding, surface_guiding_sampling_prob);
  const float bssrdf_sampling_prob = INTEGRATOR_STATE(state, guiding, bssrdf_sampling_prob);

  /* Decide between sampling guiding distribution and BSDF. */
  const float rand_bsdf_guiding = INTEGRATOR_STATE(state, guiding, sample_surface_guiding_rand);

  /* Initialize to zero. */
  int label = LABEL_NONE;
  Spectrum eval = zero_spectrum();
  bsdf_eval_init(bsdf_eval, eval);

  *unguided_bsdf_pdf = 0.0f;
  float guide_pdf = 0.0f;

  if (use_surface_guiding && guiding_sampling_prob > 0.0f) {
    /* Performing guided sampling using RIS */

    // selected RIS candidate
    int ris_idx = 0;

    // directional roughness for each of the two RIS candidates
    float ris_avg_roughness_squared[2] = {0.0f, 0.0f};

    // meta data for the two RIS candidates
    GuidingRISSample ris_samples[2];
    ris_samples[0].rand = rand_bsdf;
    ris_samples[1].rand = path_state_rng_3D(kg, rng_state, PRNG_SURFACE_RIS_GUIDING_0);

    // ----------------------------------------------------
    // generate the first RIS candidate using a BSDF sample
    // ----------------------------------------------------
    ris_samples[0].label = bsdf_sample(kg,
                                       sd,
                                       sc,
                                       ris_samples[0].rand,
                                       &ris_samples[0].eval,
                                       &ris_samples[0].wo,
                                       &ris_samples[0].bsdf_pdf,
                                       &ris_samples[0].sampled_roughness,
                                       &ris_samples[0].eta);

    bsdf_eval_init(
        &ris_samples[0].bsdf_eval, sc, ris_samples[0].wo, ris_samples[0].eval * sc->weight);
    if (ris_samples[0].bsdf_pdf > 0.0f) {
      const float roughness_squared = bsdf_get_specular_roughness_squared(sc);
      if (sd->num_closure > 1) {
        const float sweight = sc->sample_weight;
        const float weight = ris_samples[0].bsdf_pdf * sweight;
        ris_samples[0].bsdf_pdf = _surface_shader_bsdf_eval_mis(kg,
                                                                sd,
                                                                ris_samples[0].wo,
                                                                sc,
                                                                &ris_samples[0].bsdf_eval,
                                                                weight,
                                                                sweight,
                                                                0,
                                                                weight * roughness_squared,
                                                                ris_avg_roughness_squared[0]);
        kernel_assert(reduce_min(bsdf_eval_sum(&ris_samples[0].bsdf_eval)) >= 0.0f);
      }
      else {
        ris_avg_roughness_squared[0] = roughness_squared;
      }
      ris_samples[0].avg_bsdf_eval = average(ris_samples[0].bsdf_eval.sum);
      ris_samples[0].guide_pdf = guiding_bsdf_pdf(kg, ris_samples[0].wo);
      ris_samples[0].guide_pdf *= (1.0f - bssrdf_sampling_prob);
      ris_samples[0].incoming_radiance_pdf = guiding_surface_incoming_radiance_pdf(
          kg, ris_samples[0].wo);
      ris_samples[0].bsdf_pdf = max(0.0f, ris_samples[0].bsdf_pdf);
    }

    // ------------------------------------------------------------------------------
    // generate the second RIS candidate using a sample from the guiding distribution
    // ------------------------------------------------------------------------------
    float unguided_bsdf_pdfs[MAX_CLOSURE];
    bsdf_eval_init(&ris_samples[1].bsdf_eval, eval);
    ris_samples[1].guide_pdf = guiding_bsdf_sample(
        kg, make_float2(ris_samples[1].rand), &ris_samples[1].wo);
    ris_samples[1].guide_pdf *= (1.0f - bssrdf_sampling_prob);
    ris_samples[1].incoming_radiance_pdf = guiding_surface_incoming_radiance_pdf(
        kg, ris_samples[1].wo);
    ris_samples[1].bsdf_pdf = surface_shader_bsdf_eval_pdfs(kg,
                                                            sd,
                                                            ris_samples[1].wo,
                                                            &ris_samples[1].bsdf_eval,
                                                            unguided_bsdf_pdfs,
                                                            0,
                                                            ris_avg_roughness_squared[1]);
    ris_samples[1].label = ris_samples[0].label;
    ris_samples[1].avg_bsdf_eval = average(ris_samples[1].bsdf_eval.sum);
    ris_samples[1].bsdf_pdf = max(0.0f, ris_samples[1].bsdf_pdf);

    // ------------------------------------------------------------------------------
    // calculate the RIS target functions for each RIS candidate
    // ------------------------------------------------------------------------------
    int num_ris_candidates = 0;
    float sum_ris_weights = 0.0f;
    if (calculate_ris_target(&ris_samples[0], guiding_sampling_prob)) {
      sum_ris_weights += ris_samples[0].ris_weight;
      num_ris_candidates++;
    }
    kernel_assert(ris_samples[0].ris_weight >= 0.0f);
    kernel_assert(sum_ris_weights >= 0.0f);

    if (calculate_ris_target(&ris_samples[1], guiding_sampling_prob)) {
      sum_ris_weights += ris_samples[1].ris_weight;
      num_ris_candidates++;
    }
    kernel_assert(ris_samples[1].ris_weight >= 0.0f);
    kernel_assert(sum_ris_weights >= 0.0f);

    // ------------------------------------------------------------------------------
    // Sample/Select a sample from the RIS candidates proportional to the target
    // ------------------------------------------------------------------------------
    if (num_ris_candidates == 0 || !(sum_ris_weights > 1e-10f)) {
      *bsdf_pdf = 0.0f;
      *mis_pdf = 0.0f;
      return label;
    }

    const float rand_ris_select = rand_bsdf_guiding * sum_ris_weights;

    float sum_ris = 0.0f;
    for (int i = 0; i < 2; i++) {
      sum_ris += ris_samples[i].ris_weight;
      if (rand_ris_select <= sum_ris) {
        ris_idx = i;
        break;
      }
    }

    kernel_assert(sum_ris >= 0.0f);
    kernel_assert(ris_idx < 2);

    // ------------------------------------------------------------------------------
    // Fill in the sample data for the selected RIS candidate
    // ------------------------------------------------------------------------------
    guide_pdf = ris_samples[ris_idx].ris_target * (2.0f / sum_ris_weights);
    *unguided_bsdf_pdf = ris_samples[ris_idx].bsdf_pdf;
    *mis_pdf = 0.5f * (ris_samples[ris_idx].bsdf_pdf + ris_samples[ris_idx].guide_pdf);
    *bsdf_pdf = guide_pdf;

    *wo = ris_samples[ris_idx].wo;
    label = ris_samples[ris_idx].label;

    *sampled_roughness = ris_samples[ris_idx].sampled_roughness;
    *eta = ris_samples[ris_idx].eta;
    *bsdf_eval = ris_samples[ris_idx].bsdf_eval;
    r_avg_roughness_squared = ris_avg_roughness_squared[ris_idx];

    kernel_assert(isfinite_safe(guide_pdf));
    kernel_assert(isfinite_safe(*bsdf_pdf));

    if (!(*bsdf_pdf > 1e-10f)) {
      *bsdf_pdf = 0.0f;
      *mis_pdf = 0.0f;
      return label;
    }

    kernel_assert(*bsdf_pdf > 0.0f);
    kernel_assert(*bsdf_pdf >= 1e-20f);
    kernel_assert(guide_pdf >= 0.0f);

    /// select label sampled_roughness and eta
    if (ris_idx == 1 && ris_samples[1].bsdf_pdf > 0.0f) {

      const float rnd = path_state_rng_1D(kg, rng_state, PRNG_SURFACE_RIS_GUIDING_1);

      float sum_pdfs = 0.0f;
      int idx = -1;
      for (int i = 0; i < sd->num_closure; i++) {
        sum_pdfs += unguided_bsdf_pdfs[i];
        if (rnd <= sum_pdfs) {
          idx = i;
          break;
        }
      }
      // kernel_assert(idx >= 0);
      /* Set the default idx to the last in the list.
       * in case of numerical problems and rand_bsdf_guiding is just >=1.0f and
       * the sum of all unguided_bsdf_pdfs is just < 1.0f. */
      idx = (rnd > sum_pdfs) ? sd->num_closure - 1 : idx;

      label = bsdf_label(kg, &sd->closure[idx], *wo);
      bsdf_roughness_eta(&sd->closure[idx], *wo, sampled_roughness, eta);
    }

    kernel_assert(isfinite_safe(*bsdf_pdf));
    kernel_assert(*bsdf_pdf >= 0.0f);
    kernel_assert(reduce_min(bsdf_eval_sum(bsdf_eval)) >= 0.0f);
  }
  else {
    /* Sample BSDF. */
    *bsdf_pdf = 0.0f;
    label = bsdf_sample(
        kg, sd, sc, rand_bsdf, &eval, wo, unguided_bsdf_pdf, sampled_roughness, eta);
#  ifdef WITH_CYCLES_DEBUG
    // Code path to validate the estimation of the label, sampled roughness and eta
    // This should be activated from time to time when the BSDFs change to check if everything
    // is still working correctly.
    if (*unguided_bsdf_pdf > 0.0f) {
      surface_shader_validate_bsdf_sample(kg, sc, *wo, label, *sampled_roughness, *eta);
    }
#  endif

    if (*unguided_bsdf_pdf != 0.0f) {
      bsdf_eval_init(bsdf_eval, sc, *wo, eval * sc->weight);

      kernel_assert(reduce_min(bsdf_eval_sum(bsdf_eval)) >= 0.0f);

      const float roughness_squared = bsdf_get_specular_roughness_squared(sc);
      if (sd->num_closure > 1) {
        const float sweight = sc->sample_weight;
        const float weight = (*unguided_bsdf_pdf) * sweight;
        *unguided_bsdf_pdf = _surface_shader_bsdf_eval_mis(kg,
                                                           sd,
                                                           *wo,
                                                           sc,
                                                           bsdf_eval,
                                                           weight,
                                                           sweight,
                                                           0,
                                                           weight * roughness_squared,
                                                           r_avg_roughness_squared);
        kernel_assert(reduce_min(bsdf_eval_sum(bsdf_eval)) >= 0.0f);
      }
      else {
        r_avg_roughness_squared = roughness_squared;
      }
      *bsdf_pdf = *unguided_bsdf_pdf;
      *mis_pdf = *bsdf_pdf;
    }

    kernel_assert(reduce_min(bsdf_eval_sum(bsdf_eval)) >= 0.0f);
  }

  return label;
}

ccl_device int surface_shader_bsdf_guided_sample_closure(
    KernelGlobals kg,
    IntegratorState state,
    ccl_private ShaderData *sd,
    const ccl_private ShaderClosure *sc,
    const float3 rand_bsdf,
    ccl_private BsdfEval *bsdf_eval,
    ccl_private float3 *wo,
    ccl_private float *bsdf_pdf,
    ccl_private float *mis_pdf,
    ccl_private float *unguided_bsdf_pdf,
    ccl_private float2 *sampled_roughness,
    ccl_private float *eta,
    const ccl_private RNGState *rng_state,
    ccl_private float &r_avg_roughness_squared)
{
  int label = LABEL_NONE;
  r_avg_roughness_squared = 0.0f;
  if (kernel_data.integrator.guiding_directional_sampling_type ==
          GUIDING_DIRECTIONAL_SAMPLING_TYPE_PRODUCT_MIS ||
      kernel_data.integrator.guiding_directional_sampling_type ==
          GUIDING_DIRECTIONAL_SAMPLING_TYPE_ROUGHNESS)
  {
    label = surface_shader_bsdf_guided_sample_closure_mis(kg,
                                                          state,
                                                          sd,
                                                          sc,
                                                          rand_bsdf,
                                                          bsdf_eval,
                                                          wo,
                                                          bsdf_pdf,
                                                          unguided_bsdf_pdf,
                                                          sampled_roughness,
                                                          eta,
                                                          r_avg_roughness_squared);
    *mis_pdf = (*unguided_bsdf_pdf > 0.0f) ? *bsdf_pdf : 0.0f;
  }
  else if (kernel_data.integrator.guiding_directional_sampling_type ==
           GUIDING_DIRECTIONAL_SAMPLING_TYPE_RIS)
  {
    label = surface_shader_bsdf_guided_sample_closure_ris(kg,
                                                          state,
                                                          sd,
                                                          sc,
                                                          rand_bsdf,
                                                          rng_state,
                                                          bsdf_eval,
                                                          wo,
                                                          bsdf_pdf,
                                                          mis_pdf,
                                                          unguided_bsdf_pdf,
                                                          sampled_roughness,
                                                          eta,
                                                          r_avg_roughness_squared);
  }
  if (!(*unguided_bsdf_pdf > 0.0f)) {
    *bsdf_pdf = 0.0f;
    *mis_pdf = 0.0f;
  }
  return label;
}

#endif

/* Sample direction for picked BSDF, and return evaluation and pdf for all
 * BSDFs combined using MIS. */
ccl_device int surface_shader_bsdf_sample_closure(KernelGlobals kg,
                                                  ccl_private ShaderData *sd,
                                                  const ccl_private ShaderClosure *sc,
                                                  const float3 rand_bsdf,
                                                  ccl_private BsdfEval *bsdf_eval,
                                                  ccl_private float3 *wo,
                                                  ccl_private float *pdf,
                                                  ccl_private float2 *sampled_roughness,
                                                  ccl_private float *eta,
                                                  ccl_private float &r_avg_roughness_squared)
{
  /* BSSRDF should already have been handled elsewhere. */
  kernel_assert(CLOSURE_IS_BSDF(sc->type));

  int label;
  Spectrum eval = zero_spectrum();

  *pdf = 0.0f;
  label = bsdf_sample(kg, sd, sc, rand_bsdf, &eval, wo, pdf, sampled_roughness, eta);

  if (*pdf != 0.0f) {
    bsdf_eval_init(bsdf_eval, sc, *wo, eval * sc->weight);

    const float sweight = sc->sample_weight;
    const float roughness_squared = bsdf_get_specular_roughness_squared(sc);

    if (sd->num_closure > 1) {
      const float weight = *pdf * sweight;
      *pdf = _surface_shader_bsdf_eval_mis(kg,
                                           sd,
                                           *wo,
                                           sc,
                                           bsdf_eval,
                                           weight,
                                           sweight,
                                           0,
                                           weight * roughness_squared,
                                           r_avg_roughness_squared);
    }
    else {
      r_avg_roughness_squared = roughness_squared;
    }
  }
  else {
    bsdf_eval_init(bsdf_eval, zero_spectrum());
    r_avg_roughness_squared = 0.0f;
  }

  return label;
}

#ifdef __KERNEL_METAL__
ccl_device int surface_shader_bsdf_gpu_resampled_closure(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const ccl_private ShaderClosure *sc,
    const ccl_private SurfaceGuidingProposal &proposal,
    const float3 rand_bsdf,
    const float3 rand_guide,
    const float rand_select,
    ccl_private BsdfEval *bsdf_eval,
    ccl_private float3 *wo,
    ccl_private float *pdf,
    ccl_private float *mis_pdf,
    ccl_private float *unguided_pdf,
    ccl_private float2 *sampled_roughness,
    ccl_private float *eta,
    ccl_private float &r_avg_roughness_squared)
{
  struct Candidate {
    BsdfEval eval;
    float3 direction;
    float2 roughness;
    float eta;
    float bsdf_pdf;
    float guide_pdf;
    float average_roughness;
    int label;
  } candidates[2] = {};

  candidates[0].label = surface_shader_bsdf_sample_closure(kg,
                                                           sd,
                                                           sc,
                                                           rand_bsdf,
                                                           &candidates[0].eval,
                                                           &candidates[0].direction,
                                                           &candidates[0].bsdf_pdf,
                                                           &candidates[0].roughness,
                                                           &candidates[0].eta,
                                                           candidates[0].average_roughness);
  if (candidates[0].bsdf_pdf > 0.0f) {
    candidates[0].guide_pdf = surface_shader_gpu_guiding_distribution_pdf(proposal,
                                                                          candidates[0].direction);
  }

  candidates[1].direction = surface_shader_gpu_guiding_distribution_sample(
      proposal, make_float2(rand_guide), &candidates[1].guide_pdf);
  float closure_pdfs[MAX_CLOSURE];
  candidates[1].bsdf_pdf = surface_shader_bsdf_eval_pdfs(kg,
                                                         sd,
                                                         candidates[1].direction,
                                                         &candidates[1].eval,
                                                         closure_pdfs,
                                                         0,
                                                         candidates[1].average_roughness);

  GuidingResamplingPair pair{};
  GuidingField::DirectionalTree directional;
  GuidingGaussianMixture smooth;
  const float uniform = (sd->runtime_flag & SR_BSDF_HAS_TRANSMISSION) ? M_1_PI_F * 0.25f :
                                                                        M_1_PI_F * 0.5f;
  for (int i = 0; i < 2; ++i) {
    ccl_private Candidate &candidate = candidates[i];
    pair.proposal[i] = 0.5f * (candidate.bsdf_pdf + candidate.guide_pdf);
    if (candidate.bsdf_pdf > 0.0f && !bsdf_eval_is_zero(&candidate.eval)) {
      const float incident_pdf = proposal.use_smooth_product ?
                                     smooth.pdf(proposal.weights + GuidingField::tree_size,
                                                candidate.direction,
                                                proposal.position) :
                                     directional.pdf(proposal.weights, candidate.direction);
      pair.target[i] = average(bsdf_eval_sum(&candidate.eval)) *
                       ((1.0f - proposal.probability) * uniform +
                        proposal.probability * incident_pdf);
    }
  }
  const int selected = pair.sample(rand_select, pdf);
  *mis_pdf = 0.0f;
  *unguided_pdf = 0.0f;
  if (selected < 0) {
    bsdf_eval_init(bsdf_eval, zero_spectrum());
    return LABEL_NONE;
  }
  ccl_private Candidate &candidate = candidates[selected];
  if (selected == 1) {
    int closure = -1;
    float cumulative = 0.0f;
    for (int i = 0; i < sd->num_closure; ++i) {
      if (closure_pdfs[i] > 0.0f) {
        closure = i;
        cumulative += closure_pdfs[i];
        if (rand_guide.z < cumulative) {
          break;
        }
      }
    }
    if (closure < 0) {
      *pdf = 0.0f;
      bsdf_eval_init(bsdf_eval, zero_spectrum());
      return LABEL_NONE;
    }
    bsdf_roughness_eta(
        &sd->closure[closure], candidate.direction, &candidate.roughness, &candidate.eta);
    candidate.label = bsdf_label(kg, &sd->closure[closure], candidate.direction);
  }
  *bsdf_eval = candidate.eval;
  *wo = candidate.direction;
  *unguided_pdf = candidate.bsdf_pdf;
  *mis_pdf = pair.proposal[selected];
  *sampled_roughness = candidate.roughness;
  *eta = candidate.eta;
  r_avg_roughness_squared = candidate.average_roughness;
  return candidate.label;
}

ccl_device int surface_shader_bsdf_gpu_guided_sample_closure(
    KernelGlobals kg,
    ccl_private ShaderData *sd,
    const ccl_private ShaderClosure *sc,
    const float3 rand_bsdf,
    float rand_guiding,
    const float3 rand_resampling,
    ccl_private BsdfEval *bsdf_eval,
    ccl_private float3 *wo,
    ccl_private float *pdf,
    ccl_private float *mis_pdf,
    ccl_private float *unguided_pdf,
    ccl_private float2 *sampled_roughness,
    ccl_private float *eta,
    ccl_private float &r_avg_roughness_squared,
    const bool light_path = false)
{
  *mis_pdf = 0.0f;
  /* Guiding replaces only the continuous part of closure selection. Discrete/transparent
   * events retain their original mass, which also preserves their bidirectional MIS ratios. */
  if (!surface_shader_gpu_guiding_continuous(sc)) {
    const int label = surface_shader_bsdf_sample_closure(kg,
                                                         sd,
                                                         sc,
                                                         rand_bsdf,
                                                         bsdf_eval,
                                                         wo,
                                                         pdf,
                                                         sampled_roughness,
                                                         eta,
                                                         r_avg_roughness_squared);
    *unguided_pdf = *pdf;
    *mis_pdf = *pdf;
    return label;
  }
  const SurfaceGuidingProposal proposal = surface_shader_gpu_guiding_query(sd, light_path);
  const float probability = proposal.probability;
  if (probability > 0.0f && proposal.use_resampling) {
    return surface_shader_bsdf_gpu_resampled_closure(kg,
                                                     sd,
                                                     sc,
                                                     proposal,
                                                     rand_bsdf,
                                                     rand_resampling,
                                                     rand_guiding,
                                                     bsdf_eval,
                                                     wo,
                                                     pdf,
                                                     mis_pdf,
                                                     unguided_pdf,
                                                     sampled_roughness,
                                                     eta,
                                                     r_avg_roughness_squared);
  }
  if (probability > 0.0f && rand_guiding < probability) {
    rand_guiding /= probability;
    float guide_pdf;
    *wo = surface_shader_gpu_guiding_distribution_sample(
        proposal, make_float2(rand_bsdf), &guide_pdf);
    float closure_pdfs[MAX_CLOSURE];
    *unguided_pdf = surface_shader_bsdf_eval_pdfs(
        kg, sd, *wo, bsdf_eval, closure_pdfs, 0, r_avg_roughness_squared);
    *pdf = (1.0f - probability) * *unguided_pdf + probability * proposal.bsdf_fraction * guide_pdf;
    if (!(*unguided_pdf > 0.0f) || bsdf_eval_is_zero(bsdf_eval)) {
      *pdf = 0.0f;
      return LABEL_NONE;
    }
    int selected = -1;
    float cumulative = 0.0f;
    for (int i = 0; i < sd->num_closure; ++i) {
      if (closure_pdfs[i] > 0.0f) {
        selected = i;
        cumulative += closure_pdfs[i];
        if (rand_guiding < cumulative) {
          break;
        }
      }
    }
    if (selected < 0) {
      *pdf = 0.0f;
      return LABEL_NONE;
    }
    const ccl_private ShaderClosure *selected_closure = &sd->closure[selected];
    bsdf_roughness_eta(selected_closure, *wo, sampled_roughness, eta);
    *mis_pdf = *pdf;
    return bsdf_label(kg, selected_closure, *wo);
  }

  const int label = surface_shader_bsdf_sample_closure(kg,
                                                       sd,
                                                       sc,
                                                       rand_bsdf,
                                                       bsdf_eval,
                                                       wo,
                                                       unguided_pdf,
                                                       sampled_roughness,
                                                       eta,
                                                       r_avg_roughness_squared);
  *pdf = *unguided_pdf;
  if (probability > 0.0f && *pdf > 0.0f) {
    /* Dirac events carry discrete probability mass and have no continuous guide component. */
    *pdf = (label & (LABEL_SINGULAR | LABEL_TRANSPARENT)) ?
               (1.0f - probability) * *pdf :
               surface_shader_gpu_guiding_pdf_from_query(proposal, *wo, *pdf);
  }
  *mis_pdf = *pdf;
  return label;
}
#endif

ccl_device float surface_shader_average_roughness(const ccl_private ShaderData *sd)
{
  float roughness = 0.0f;
  float sum_weight = 0.0f;

  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];

    if (CLOSURE_IS_BSDF(sc->type)) {
      /* `sqrt()` once to undo the squaring from multiplying roughness on the
       * two axes, and once for the squared roughness convention. */
      const float value = bsdf_get_roughness_pass_squared(sc);
      if (value >= 0.0f) {
        const float weight = fabsf(average(sc->weight));
        roughness += weight * sqrtf(sqrtf(value));
        sum_weight += weight;
      }
    }
  }

  return (sum_weight > 0.0f) ? roughness / sum_weight : 1.0f;
}

ccl_device Spectrum surface_shader_transparency(const ccl_private ShaderData *sd)
{
  if (sd->shader_flag & SD_HAS_ONLY_VOLUME) {
    return one_spectrum();
  }
  if (sd->runtime_flag & (SR_TRANSPARENT | SR_RAY_PORTAL)) {
    return sd->closure_transparent_extinction;
  }
  return zero_spectrum();
}

ccl_device Spectrum surface_shader_alpha(const ccl_private ShaderData *sd)
{
  Spectrum alpha = one_spectrum() - surface_shader_transparency(sd);

  alpha = saturate(alpha);

  return alpha;
}

ccl_device Spectrum surface_shader_diffuse(KernelGlobals kg, const ccl_private ShaderData *sd)
{
  Spectrum eval = zero_spectrum();

  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];

    if (CLOSURE_IS_BSDF_DIFFUSE(sc->type) || CLOSURE_IS_BSSRDF(sc->type)) {
      eval += closure_albedo(kg, sd, sc, true, true);
    }
  }

  return eval;
}

ccl_device Spectrum surface_shader_glossy(KernelGlobals kg, const ccl_private ShaderData *sd)
{
  Spectrum eval = zero_spectrum();

  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];

    if (CLOSURE_IS_BSDF_GLOSSY(sc->type) || CLOSURE_IS_GLASS(sc->type)) {
      eval += closure_albedo(kg, sd, sc, true, false);
    }
  }

  return eval;
}

ccl_device Spectrum surface_shader_transmission(KernelGlobals kg, const ccl_private ShaderData *sd)
{
  Spectrum eval = zero_spectrum();

  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];

    if (CLOSURE_IS_BSDF_TRANSMISSION(sc->type) || CLOSURE_IS_GLASS(sc->type)) {
      eval += closure_albedo(kg, sd, sc, false, true);
    }
  }

  return eval;
}

ccl_device float3 surface_shader_average_normal(const ccl_private ShaderData *sd)
{
  float3 N = zero_float3();

  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];
    if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
      N += sc->N * fabsf(average(sc->weight));
    }
  }

  return (is_zero(N)) ? sd->N : normalize(N);
}

ccl_device Spectrum surface_shader_ao(const ccl_private ShaderData *sd,
                                      const float ao_factor,
                                      ccl_private float3 *N_)
{
  Spectrum eval = zero_spectrum();
  float3 N = zero_float3();

  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];

    if (CLOSURE_IS_BSDF_DIFFUSE(sc->type)) {
      const ccl_private DiffuseBsdf *bsdf = (const ccl_private DiffuseBsdf *)sc;
      eval += sc->weight * ao_factor;
      N += bsdf->N * fabsf(average(sc->weight));
    }
  }

  *N_ = (is_zero(N)) ? sd->N : normalize(N);
  return eval;
}

#ifdef __SUBSURFACE__
ccl_device float3 surface_shader_bssrdf_normal(const ccl_private ShaderData *sd)
{
  float3 N = zero_float3();

  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];

    if (CLOSURE_IS_BSSRDF(sc->type)) {
      const ccl_private Bssrdf *bssrdf = (const ccl_private Bssrdf *)sc;
      const float avg_weight = fabsf(average(sc->weight));

      N += bssrdf->N * avg_weight;
    }
  }

  return (is_zero(N)) ? sd->N : normalize(N);
}
#endif /* __SUBSURFACE__ */

/* Constant emission optimization */

ccl_device bool surface_shader_constant_emission(KernelGlobals kg,
                                                 const int shader,
                                                 ccl_private Spectrum *eval)
{
  const int shader_index = shader & SHADER_MASK;
  const int shader_flag = kernel_data_fetch(shaders, shader_index).flags;

  if (shader_flag & SD_HAS_CONSTANT_EMISSION) {
    const float3 emission_rgb = make_float3(
        kernel_data_fetch(shaders, shader_index).constant_emission[0],
        kernel_data_fetch(shaders, shader_index).constant_emission[1],
        kernel_data_fetch(shaders, shader_index).constant_emission[2]);
    *eval = rgb_to_spectrum(emission_rgb);

    return true;
  }

  return false;
}

/* Background */

ccl_device Spectrum surface_shader_background(const ccl_private ShaderData *sd)
{
  if (sd->runtime_flag & SR_EMISSION) {
    return sd->closure_emission_background;
  }
  return zero_spectrum();
}

/* Emission */

ccl_device Spectrum surface_shader_emission(const ccl_private ShaderData *sd)
{
  if (sd->runtime_flag & SR_EMISSION) {
    return emissive_simple_eval(sd->Ng, sd->wi) * sd->closure_emission_background;
  }
  return zero_spectrum();
}

/* Holdout */

ccl_device Spectrum surface_shader_apply_holdout(ccl_private ShaderData *sd)
{
  Spectrum weight = zero_spectrum();

  /* For objects marked as holdout, preserve transparency and remove all other
   * closures, replacing them with a holdout weight. */
  if (sd->object_flag & SD_OBJECT_HOLDOUT_MASK) {
    if ((sd->runtime_flag & SR_TRANSPARENT) && !(sd->shader_flag & SD_HAS_ONLY_VOLUME)) {
      weight = one_spectrum() - sd->closure_transparent_extinction;

      for (int i = 0; i < sd->num_closure; i++) {
        ccl_private ShaderClosure *sc = &sd->closure[i];
        if (!CLOSURE_IS_BSDF_TRANSPARENT(sc->type)) {
          sc->type = NBUILTIN_CLOSURES;
        }
      }

      sd->runtime_flag &= (~SR_CLOSURE_FLAG | SR_TRANSPARENT | SR_BSDF);
    }
    else {
      weight = one_spectrum();
    }
  }
  else {
    for (int i = 0; i < sd->num_closure; i++) {
      const ccl_private ShaderClosure *sc = &sd->closure[i];
      if (CLOSURE_IS_HOLDOUT(sc->type)) {
        weight += sc->weight;
      }
    }
  }

  return weight;
}

/* Surface Evaluation */

template<uint64_t node_feature_mask, typename ConstIntegratorGenericState>
ccl_device void surface_shader_eval(KernelGlobals kg,
                                    ConstIntegratorGenericState state,
                                    ccl_private ShaderData *ccl_restrict sd,
                                    ccl_global float *ccl_restrict buffer,
                                    const PathRayVisibility path_visibility,
                                    const uint32_t path_flag,
                                    bool use_caustics_storage = false)
{
  /* Initialize additional RNG for BSDFs and image textures. */
  sd->lcg_state = integrator_state_lcg_init(state, 0xb4bc3953);

  /* If path is being terminated, we are tracing a shadow ray or evaluating
   * emission, then we don't need to store closures. The emission and shadow
   * shader data also do not have a closure array to save GPU memory. */
  int max_closures;
  if ((path_visibility & PATH_RAY_VISIBILITY_SHADOW) ||
      (path_flag & (PATH_RAY_TERMINATE | PATH_RAY_EMISSION)))
  {
    max_closures = 0;
  }
  else {
    max_closures = use_caustics_storage ? CAUSTICS_MAX_CLOSURE : kernel_data.max_closures;
  }

  sd->num_closure = 0;
  sd->num_closure_left = max_closures;
  sd->closure_transparent_extinction = zero_spectrum();

#ifdef __OSL__
  if (kernel_data.kernel_features & KERNEL_FEATURE_OSL_SHADING) {
    osl_eval_nodes<SHADER_TYPE_SURFACE>(kg, state, sd, path_visibility, path_flag);
  }
  else
#endif
  {
#ifdef __SVM__
    svm_eval_nodes<node_feature_mask, SHADER_TYPE_SURFACE>(
        kg, state, sd, buffer, path_visibility, path_flag);
#else
    if (sd->object == OBJECT_NONE) {
      sd->closure_emission_background = make_spectrum(0.8f);
      sd->runtime_flag |= SR_EMISSION;
    }
    else {
      bsdf_diffuse_setup(sd, sd->N, make_spectrum(0.8f));
    }
#endif
  }
}

CCL_NAMESPACE_END
