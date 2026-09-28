/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf.h"
#include <cstdio>
using namespace ccl;

/* A zero lobe has exactly zero albedo regardless of the grating approximation.
 * Smooth normal-incidence controls use analytic Fresnel powers. Rough/relief
 * albedo remains an approximation. Original failing results remain in the audit. */
int main()
{
  ShaderData sd{};
  sd.N = sd.Ng = sd.wi = make_float3(0, 0, 1);
  DiffractionDielectricGeneralizedExtra extra{};
  DiffractionDielectricBsdf bsdf{};
  bsdf.N = sd.N;
  bsdf.T = make_float3(1, 0, 0);
  bsdf.type = CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID;
  bsdf.weight = one_spectrum();
  bsdf.extra = &extra.base;
  bsdf.disabled_lobes = DIFFRACTION_DIELECTRIC_TINT_EXTRA |
                        DIFFRACTION_DIELECTRIC_GENERALIZED;
  extra.generalized_f0 = make_spectrum(.02f);
  extra.generalized_reference_f0 = .04f;
  diffraction_dielectric_parameters(550, 1200, 250, .5f, 1, 1.5f, .2f, .2f,
                                    &extra.base.param);
  int failures = 0;
  for (int transmission = 0; transmission < 2; ++transmission) {
    extra.base.reflection = transmission ? one_spectrum() : zero_spectrum();
    extra.base.transmission = transmission ? zero_spectrum() : one_spectrum();
    bsdf.disabled_lobes = DIFFRACTION_DIELECTRIC_TINT_EXTRA |
                          DIFFRACTION_DIELECTRIC_GENERALIZED |
                          (transmission ? LABEL_TRANSMIT : LABEL_REFLECT);
    Spectrum result = closure_albedo(nullptr, &sd, (ShaderClosure *)&bsdf,
                                     !transmission, transmission);
    bool correct = is_zero(result);
    std::printf("%s disabled-lobe albedo: (%g, %g, %g), expected (0, 0, 0): %s\n",
                transmission ? "transmission" : "reflection", double(result.x),
                double(result.y), double(result.z), correct ? "PASS" : "FAIL");
    failures += !correct;
  }
  extra.base.reflection = rgb_to_spectrum(make_float3(.3f, .5f, .7f));
  extra.base.transmission = rgb_to_spectrum(make_float3(.7f, .4f, .2f));
  bsdf.disabled_lobes = DIFFRACTION_DIELECTRIC_TINT_EXTRA |
                        DIFFRACTION_DIELECTRIC_GENERALIZED;
  const Spectrum tinted = closure_albedo(nullptr, &sd, (ShaderClosure *)&bsdf, true, true);
  failures += reduce_max(fabs(tinted - (.02f * extra.base.reflection + .98f * extra.base.transmission))) > 1e-6f;
  const auto original_param = extra.base.param;
  diffraction_dielectric_parameters(550, 1200, 250, .5f, 1, 1, .2f, .2f,
                                    &extra.base.param);
  bsdf.disabled_lobes |= LABEL_TRANSMIT;
  failures += reduce_max(fabs(closure_albedo(nullptr, &sd, (ShaderClosure *)&bsdf,
                                             true, true) - (.02f * extra.base.reflection))) > 1e-6f;
  extra.base.param = original_param;
  bsdf.param = extra.base.param;
  int combinations = 0;
  for (int distribution = 0; distribution < 2; ++distribution) {
    bsdf.type = distribution ? CLOSURE_BSDF_DIFFRACTION_BECKMANN_ID :
                               CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID;
    for (int disabled = 0; disabled < 4; ++disabled) {
      bsdf.disabled_lobes = (disabled & 1 ? LABEL_REFLECT : 0) |
                            (disabled & 2 ? LABEL_TRANSMIT : 0);
      for (int requested = 0; requested < 4; ++requested) {
        const int active = requested & ~disabled;
        const float expected = (active & 1 ? .04f : 0.f) + (active & 2 ? .96f : 0.f);
        const Spectrum result = closure_albedo(nullptr, &sd, (ShaderClosure *)&bsdf,
                                                requested & 1, requested & 2);
        failures += reduce_max(fabs(result - make_spectrum(expected))) > 1e-6f;
        ++combinations;
      }
    }
  }
  bsdf.type = CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID;
  bsdf.disabled_lobes = DIFFRACTION_DIELECTRIC_PURE_REFRACTION | int(LABEL_REFLECT);
  failures += !isequal(closure_albedo(nullptr, &sd, (ShaderClosure *)&bsdf, false, true),
                       one_spectrum());
  DiffractionDielectricCoatingExtra coating{};
  coating.base.param = extra.base.param;
  coating.base.reflection = one_spectrum();
  coating.base.transmission = make_spectrum(.7f);
  coating.film_ior = sqrtf(1.5f);
  coating.film_thickness_over_wavelength = .25f / coating.film_ior;
  bsdf.extra = &coating.base;
  bsdf.disabled_lobes = DIFFRACTION_DIELECTRIC_TINT_EXTRA |
                        DIFFRACTION_DIELECTRIC_COATING_EXTRA;
  /* Quarter-wave index-matched AR coating at normal incidence: R=0, T=1. */
  failures += reduce_max(fabs(closure_albedo(nullptr, &sd, (ShaderClosure *)&bsdf,
                                             true, false))) > 1e-6f;
  failures += reduce_max(fabs(closure_albedo(nullptr, &sd, (ShaderClosure *)&bsdf,
                                             false, true) - make_spectrum(.7f))) > 1e-6f;
  bsdf.type = CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID;
  failures += !is_zero(closure_albedo(nullptr, &sd, (ShaderClosure *)&bsdf, true, false));
  failures += !isequal(closure_albedo(nullptr, &sd, (ShaderClosure *)&bsdf, false, true),
                       one_spectrum());
  std::printf("%d distribution/mask/query combinations; failures=%d\n", combinations, failures);
  return failures ? 1 : 0;
}
