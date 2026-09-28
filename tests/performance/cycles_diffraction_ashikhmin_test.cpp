/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction_ashikhmin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace ccl;

static float fract(const float x) { return x - std::floor(x); }

int main()
{
  int failures = 0, samples = 0;
  float max_flat = 0, max_reciprocity = 0, max_furnace = 0;
  for (const float depth : {0.0f, .4f})
    for (const float ratio : {.45f, .71f})
      for (const float ax : {.2f, .55f}) {
        DiffractionReflection p{ax, .7f * ax, ratio, depth, .41f};
        MicrofacetBsdf native{};
        native.N = make_float3(0, 0, 1);
        native.T = make_float3(1, 0, 0);
        native.alpha_x = p.alpha_x;
        native.alpha_y = p.alpha_y;
        bsdf_ashikhmin_shirley_setup(&native);
        for (const float mu : {.15f, .55f, .95f}) {
          const float3 wi = make_float3(std::sqrt(1 - mu * mu), 0, mu);
          double furnace = 0;
          for (int i = 0; i < 8192; ++i) {
            const float3 random = make_float3((i + .5f) / 8192,
                fract((i + .5f) * .6180339887498949f),
                fract((i + .5f) * .7548776662466927f));
            float3 wo;
            if (!diffraction_ashikhmin_sample_direction(&p, wi, random, &wo)) continue;
            float pdf;
            const float value = diffraction_ashikhmin_eval(&p, wi, wo, &pdf);
            if (!(pdf > 0) || !(value >= 0) || !std::isfinite(value)) {
              ++failures;
              continue;
            }
            furnace += value / pdf;
            ++samples;
            float reverse_pdf;
            const float reverse = diffraction_ashikhmin_eval(&p, wo, wi, &reverse_pdf);
            const float reciprocity = std::fabs(value / wo.z - reverse / wi.z);
            max_reciprocity = std::max(max_reciprocity, reciprocity);
            failures += reciprocity > 4e-4f * std::max(1.0f, value / wo.z);
            if (depth == 0) {
              float native_pdf;
              const float native_value = bsdf_ashikhmin_shirley_eval(
                  (const ShaderClosure *)&native, wi, wo, &native_pdf).x;
              const float difference = std::max(std::fabs(value - native_value),
                                                std::fabs(pdf - native_pdf));
              max_flat = std::max(max_flat, difference);
              failures += difference > 4e-4f * std::max(1.0f, native_pdf);
            }
          }
          const float albedo = furnace / 8192;
          max_furnace = std::max(max_furnace, albedo);
          failures += albedo > 1.005f;
        }
      }
  /* Cycles' discrete-event convention: the roughness-floor grating orders
   * have sample and delta-evaluation masses scaled by the same 1e6 factor. */
  DiffractionBsdf atom{};
  atom.N = make_float3(0, 0, 1);
  atom.T = make_float3(1, 0, 0);
  atom.param = {.0001f, .0001f, .45f, .4f, .41f};
  const float3 wi = normalize(make_float3(.2f, .1f, 1));
  for (int i = 0; i < 1024; ++i) {
    const float3 random = make_float3(.2f, .7f, (i + .5f) / 1024);
    Spectrum value;
    float3 wo;
    float pdf, eta;
    float2 roughness;
    const int label = bsdf_diffraction_ashikhmin_sample(
        (const ShaderClosure *)&atom, atom.N, wi, random, &value, &wo,
        &pdf, &roughness, &eta);
    if (label == LABEL_NONE) continue;
    float query_pdf;
    const Spectrum query = bsdf_diffraction_ashikhmin_delta(
        (const ShaderClosure *)&atom, wi, wo, &query_pdf);
    failures += label != (LABEL_REFLECT | LABEL_SINGULAR) ||
                std::fabs(pdf - query_pdf) > 1e-4f * std::max(1.0f, pdf) ||
                std::fabs(value.x - query.x) > 1e-4f * std::max(1.0f, value.x);
  }
  std::printf("samples=%d max_flat_difference=%g max_reciprocity=%g "
              "max_white_furnace=%g failures=%d\n",
              samples, max_flat, max_reciprocity, max_furnace, failures);
  return failures ? 1 : 0;
}
