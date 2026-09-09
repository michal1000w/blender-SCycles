/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/closure/bsdf_microfacet.h"

CCL_NAMESPACE_BEGIN

/* Sample bsdf in half-vector measure. */
ccl_device_inline float2 mnee_sample_bsdf_dh(ClosureType type,
                                             const float alpha_x,
                                             const float alpha_y,
                                             const float sample_u,
                                             const float sample_v)
{
  float alpha2;
  float cos_phi;
  float sin_phi;

  if (alpha_x == alpha_y) {
    const float phi = sample_v * M_2PI_F;
    fast_sincosf(phi, &sin_phi, &cos_phi);
    alpha2 = alpha_x * alpha_x;
  }
  else {
    float phi = atanf(alpha_y / alpha_x * tanf(M_2PI_F * sample_v + M_PI_2_F));
    if (sample_v > .5f) {
      phi += M_PI_F;
    }
    fast_sincosf(phi, &sin_phi, &cos_phi);
    const float alpha_x2 = alpha_x * alpha_x;
    const float alpha_y2 = alpha_y * alpha_y;
    alpha2 = 1.f / (cos_phi * cos_phi / alpha_x2 + sin_phi * sin_phi / alpha_y2);
  }

  /* Map sampled angles to micro-normal direction h. */
  float tan2_theta = alpha2;
  if (type == CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID ||
      type == CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID)
  {
    tan2_theta *= -logf(1.0f - sample_u);
  }
  else { /* type == CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID assumed */
    tan2_theta *= sample_u / (1.0f - sample_u);
  }
  const float cos2_theta = 1.0f / (1.0f + tan2_theta);
  const float sin_theta = safe_sqrtf(1.0f - cos2_theta);
  return make_float2(cos_phi * sin_theta, sin_phi * sin_theta);
}

/* Evaluate product term inside eq.6 at solution interface vi
 * divided by corresponding sampled pdf:
 * fr(vi)_do / pdf_dh(vi) x |do/dh| x |n.wo / n.h|
 * We assume here that the pdf (in half-vector measure) is the same as
 * the one calculation when sampling the microfacet normals from the
 * specular chain above: this allows us to simplify the bsdf weight */
ccl_device_inline Spectrum mnee_eval_bsdf_contribution(KernelGlobals kg,
                                                       ccl_private ShaderClosure *closure,
                                                       const float3 wi,
                                                       const float3 wo)
{
  ccl_private MicrofacetBsdf *bsdf = (ccl_private MicrofacetBsdf *)closure;

  const float cosNI = dot(bsdf->N, wi);
  const float cosNO = dot(bsdf->N, wo);

  const float3 Ht = normalize(-(bsdf->ior * wo + wi));
  const float cosHI = dot(Ht, wi);

  const float alpha2 = bsdf->alpha_x * bsdf->alpha_y;
  const float cosThetaM = dot(bsdf->N, Ht);

  /* Now calculate G1(i, m) and G1(o, m). */
  float G;
  float energy_scale;
  if (bsdf->type == CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID ||
      bsdf->type == CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID)
  {
    G = bsdf_G<MicrofacetType::BECKMANN>(alpha2, cosNI, cosNO);
    energy_scale = 1.0f;
  }
  else { /* bsdf->type == CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID assumed */
    G = bsdf_G<MicrofacetType::GGX>(alpha2, cosNI, cosNO);
    energy_scale = bsdf->energy_scale;
  }

  const FresnelCoeff fresnel = microfacet_fresnel(kg, bsdf, cosHI, nullptr);

  /*
   * bsdf_do = (1 - F) * D_do * G * |h.wi| / (n.wi * n.wo)
   *  pdf_dh = D_dh * cosThetaM
   *    D_do = D_dh * |dh/do|
   *
   * contribution = bsdf_do * |do/dh| * |n.wo / n.h| / pdf_dh
   *              = (1 - F) * G * |h.wi / (n.wi * n.h^2)|
   */
  return bsdf->weight * fresnel.transmittance * energy_scale * G *
         fabsf(cosHI / (cosNI * sqr(cosThetaM)));
}

CCL_NAMESPACE_END
