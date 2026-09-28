/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

/* Actual Glass closure dispatch against a small independent constant-q cache.
 * q and Q are deliberately simple so the two hemisphere integrals, reverse
 * etendue relation and branch probabilities have analytic references. */
#include "kernel/closure/bsdf.h"
#include "scene/diffraction_albedo.h"
#include "util/profiling.h"

#include <cmath>
#include <cstdio>
#include <string>

using namespace ccl;

static int failures = 0;
static int checks = 0;
static void check(bool okay, const char *name)
{
  checks++;
  if (!okay) {
    std::fprintf(stderr, "FAIL %s\n", name);
    failures++;
  }
}

static float fraction(const float x)
{
  return x - floorf(x);
}

int main()
{
  const int return_slots = 1 + (sizeof(DiffractionDielectricTwoSidedExtra) +
      sizeof(ShaderClosure) - 1) / sizeof(ShaderClosure);
  const int plain_pair_slots = 1 + return_slots;
  const int tinted_pair_slots = 1 + return_slots + (sizeof(DiffractionDielectricTintExtra) +
      sizeof(ShaderClosure) - 1) / sizeof(ShaderClosure);
  const int coated_pair_slots = 1 + return_slots + (sizeof(DiffractionDielectricCoatingExtra) +
      sizeof(ShaderClosure) - 1) / sizeof(ShaderClosure);

  constexpr int wavelength_count = 2, mu_count = 2, phi_count = 4;
  float values[2 * wavelength_count * mu_count * phi_count];
  for (int side = 0; side < 2; side++) {
    for (int i = 0; i < wavelength_count * mu_count * phi_count; i++) {
      const float phi = M_2PI_F * (float(i % phi_count) + 0.5f) / float(phi_count);
      values[side * wavelength_count * mu_count * phi_count + i] =
          (side == 0 ? 0.2f : 0.3f) * (1.0f + 0.12f * cosf(phi) + 0.08f * sinf(phi));
    }
  }
  float integrals[2 * wavelength_count] = {M_PI_F * 0.2f,
                                            M_PI_F * 0.2f,
                                            M_PI_F * 2.25f * 0.3f,
                                            M_PI_F * 2.25f * 0.3f};
  float cross[2] = {0.5f, 0.5f};
  const int4 descriptor = make_int4(0, 0, 0, mu_count);
  const float4 domain = make_float4(380.0f, 780.0f, wavelength_count, phi_count);
  KernelGlobalsCPU base{};
  base.diffraction_two_sided_values.data = values;
  base.diffraction_two_sided_values.width = int(sizeof(values) / sizeof(values[0]));
  base.diffraction_two_sided_integrals.data = integrals;
  base.diffraction_two_sided_integrals.width = int(sizeof(integrals) / sizeof(integrals[0]));
  base.diffraction_two_sided_cross.data = cross;
  base.diffraction_two_sided_cross.width = 2;
  base.diffraction_two_sided_descriptors.data = &descriptor;
  base.diffraction_two_sided_descriptors.width = 1;
  base.diffraction_two_sided_domains.data = &domain;
  base.diffraction_two_sided_domains.width = 1;
  base.data.tables.num_diffraction_two_sided_caches = 1;
  KernelObject object{};
  base.objects.data = &object;
  base.objects.width = 1;
  Profiler profiler;
  ThreadKernelGlobalsCPU globals(base, nullptr, profiler, 0);
  KernelGlobals kg = &globals;

  /* A real film is a reciprocal, lossless two-port even though its spectrum
   * changes with angle. Query the same Snell pair from both incident sides. */
  for (const float cosine : {0.2f, 0.5f, 0.9f}) {
    const float reverse_cosine = sqrtf(1.0f - (1.0f / 1.5f) * (1.0f / 1.5f) *
                                              (1.0f - cosine * cosine));
    for (const float wavelength : {380.0f, 550.0f, 780.0f}) {
      const float thickness_over_wavelength = 250.0f / wavelength;
      const float forward = diffraction_thin_film_pair_reflectance(
          cosine, reverse_cosine, 1.0f, 1.5f, 1.32f, thickness_over_wavelength);
      const float reverse = diffraction_thin_film_pair_reflectance(
          reverse_cosine, cosine, 1.5f, 1.0f, 1.32f, thickness_over_wavelength);
      check(forward >= 0.0f && forward <= 1.0f,
            "coated facet reflection remains passive");
      check(fabsf(forward - reverse) < 2e-6f,
            "coated facet power reciprocal under medium reversal");
      const float bare = diffraction_thin_film_pair_reflectance(
          cosine, reverse_cosine, 1.0f, 1.5f, 1.32f, 0.0f);
      check(fabsf(bare - fresnel_dielectric(cosine, 1.5f, nullptr)) < 2e-6f,
            "zero-thickness film equals bare interface");
    }
  }

  ShaderData insufficient{};
  insufficient.N = insufficient.Ng = make_float3(0, 0, 1);
  insufficient.wi = normalize(make_float3(0.2f, 0.1f, 1.0f));
  insufficient.num_closure_left = plain_pair_slots - 1;
  insufficient.rand_wavelength = 0.5f;
  check(!bsdf_diffraction_glass_two_sided_setup(kg,
                                                &insufficient,
                                                one_spectrum(),
                                                one_spectrum(),
                                                insufficient.N,
                                                make_float3(1, 0, 0),
                                                sqrtf(0.3f),
                                                1.5f,
                                                1600.0f,
                                                150.0f,
                                                0.5f,
                                                0),
        "near-capacity setup fails atomically");
  check(insufficient.num_closure == 0, "no partial closure after capacity failure");

  /* Allocation plumbing only: the constant-q cache below is deliberately
   * uncoated, so this does not claim coated transport energy correctness. */
  ShaderData coated_capacity{};
  coated_capacity.N = coated_capacity.Ng = make_float3(0, 0, 1);
  coated_capacity.wi = normalize(make_float3(0.2f, 0.1f, 1.0f));
  coated_capacity.rand_wavelength = 0.5f;
  coated_capacity.num_closure_left = coated_pair_slots - 1;
  check(!bsdf_diffraction_glass_two_sided_setup(kg, &coated_capacity,
      one_spectrum(), one_spectrum(), coated_capacity.N, make_float3(1,0,0),
      sqrtf(0.3f), 1.5f, 1600, 150, 0.5f, 0, 1.32f, 250.0f),
      "coated pair rejects insufficient closure slots");
  check(coated_capacity.num_closure == 0,
        "coated pair does not partially allocate on capacity failure");
  coated_capacity.num_closure_left = coated_pair_slots;
  check(bsdf_diffraction_glass_two_sided_setup(kg, &coated_capacity,
      one_spectrum(), one_spectrum(), coated_capacity.N, make_float3(1,0,0),
      sqrtf(0.3f), 1.5f, 1600, 150, 0.5f, 0, 1.32f, 250.0f),
      "coated pair fits sizeof-derived closure slots");
  check(coated_capacity.num_closure == 2 && coated_capacity.num_closure_left == 0,
        "coated pair consumes exact sizeof-derived slots");
  check((((const DiffractionDielectricBsdf *)&coated_capacity.closure[0])->disabled_lobes &
         DIFFRACTION_DIELECTRIC_COATING_EXTRA) != 0,
        "coated first event retains film transport");

  float max_pdf_error = 0.0f, max_eval_error = 0.0f;
  int reflection_count = 0, transmission_count = 0;
  for (int side = 0; side < 2; side++) {
    ShaderData sd{};
    sd.N = sd.Ng = side == 0 ? make_float3(0, 0, 1) : make_float3(0, 0, -1);
    sd.wi = normalize(make_float3(0.2f, 0.1f, side == 0 ? 1.0f : -1.0f));
    sd.rand_wavelength = 0.5f;
    sd.num_closure_left = plain_pair_slots;
    if (side == 1) sd.runtime_flag |= SR_BACKFACING;
    check(bsdf_diffraction_glass_two_sided_setup(kg,
                                                 &sd,
                                                 one_spectrum(),
                                                 one_spectrum(),
                                                 sd.N,
                                                 make_float3(1, 0, 0),
                                                 sqrtf(0.3f),
                                                 1.5f,
                                                 1600.0f,
                                                 150.0f,
                                                 0.5f,
                                                 0),
          "two-sided Glass setup");
    check(sd.num_closure == 2, "separate first-event and return closures");
    check(sd.num_closure_left == 0, "exact closure reservation");
    const ShaderClosure *single = &sd.closure[0];
    const ShaderClosure *multiple = &sd.closure[1];
    const auto *first = (const DiffractionDielectricBsdf *)single;
    const auto *ms = (const DiffractionDielectricBsdf *)multiple;
    check((first->disabled_lobes & DIFFRACTION_DIELECTRIC_CACHE_BASE) != 0,
          "first event retains cache roughness");
    check((ms->disabled_lobes & DIFFRACTION_DIELECTRIC_TWO_SIDED_MS) != 0,
          "return closure marked");
    check(!bsdf_microfacet_has_delta(multiple), "return is continuous");
    const Spectrum total_albedo = bsdf_albedo(kg, &sd, multiple, true, true);
    const float3 local_wi = make_float3(sd.wi.x, side == 0 ? sd.wi.y : -sd.wi.y,
                                        fabsf(sd.wi.z));
    const float expected_q = diffraction_two_sided_cache_lookup(
        kg, 0, 1000.0f * sample_wavelength(sd.rand_wavelength), side, local_wi).x;
    check(fabsf(average(total_albedo) - expected_q) < 2e-6f,
          "return row energy equals directional deficit");
    const float expected_eta = side == 0 ? 1.5f : 1.0f / 1.5f;
    for (int i = 0; i < 4096; i++) {
      const float3 random = make_float3((float(i) + 0.5f) / 4096.0f,
                                        fraction((float(i) + 0.5f) * 0.61803398875f),
                                        fraction((float(i) + 0.5f) * 0.75487766625f));
      Spectrum sampled;
      float3 wo;
      float pdf, eta;
      float2 roughness;
      const int label = bsdf_sample(
          kg, &sd, multiple, random, &sampled, &wo, &pdf, &roughness, &eta);
      if (label == LABEL_NONE) {
        check(false, "cosine return sample accepted");
        continue;
      }
      float queried_pdf;
      const Spectrum queried = bsdf_eval(kg, &sd, multiple, wo, &queried_pdf);
      max_pdf_error = max(max_pdf_error, fabsf(pdf - queried_pdf));
      max_eval_error = max(max_eval_error, fabsf(average(sampled - queried)));
      check((label & LABEL_GLOSSY) != 0, "return sample labeled glossy");
      if (label & LABEL_TRANSMIT) {
        transmission_count++;
        check(fabsf(eta - expected_eta) < 1e-6f, "transmission eta matches side");
      }
      else {
        reflection_count++;
        check(eta == 1.0f, "reflection eta unity");
      }
    }
    ShaderClosure *mutable_ms = &sd.closure[1];
    check(bsdf_diffraction_dielectric_filter(mutable_ms, true, false),
          "reflection-only filter retains transmission");
    check(average(bsdf_albedo(kg, &sd, mutable_ms, true, false)) == 0.0f,
          "filtered reflection albedo zero");
    const float before_alpha = diffraction_dielectric_param(first)->alpha_x;
    bsdf_blur(&sd.closure[0], 0.8f);
    bsdf_blur(&sd.closure[1], 0.8f);
    check(diffraction_dielectric_param(first)->alpha_x == before_alpha,
          "cache base not roughened by Filter Glossy");
    check(diffraction_dielectric_param(ms)->alpha_x == before_alpha,
          "return not roughened by Filter Glossy");
  }
  check(max_pdf_error < 2e-6f && max_eval_error < 2e-6f,
        "sample and eval agree in both directions");
  check(reflection_count > 1000 && transmission_count > 1000,
        "both return sides sampled");

  /* Reverse an off-normal transmission in the fixed-X, flipped-Y/N frame.
   * Projected BSDF eval divided by n_out^2 cos_out is reciprocal. */
  ShaderData front{}, back{};
  front.N = front.Ng = make_float3(0, 0, 1);
  back.N = back.Ng = make_float3(0, 0, -1);
  front.wi = normalize(make_float3(0.3f, 0.2f, 1));
  const float3 transmitted = normalize(make_float3(-0.2f, 0.4f, -1));
  back.wi = transmitted;
  front.rand_wavelength = back.rand_wavelength = 0.5f;
  front.num_closure_left = back.num_closure_left = plain_pair_slots;
  back.runtime_flag |= SR_BACKFACING;
  check(bsdf_diffraction_glass_two_sided_setup(kg, &front, one_spectrum(), one_spectrum(),
      front.N, make_float3(1,0,0), sqrtf(0.3f), 1.5f, 1600, 150, 0.5f, 0),
      "front reverse-pair setup");
  check(bsdf_diffraction_glass_two_sided_setup(kg, &back, one_spectrum(), one_spectrum(),
      back.N, make_float3(1,0,0), sqrtf(0.3f), 1.5f, 1600, 150, 0.5f, 0),
      "back reverse-pair setup");
  float p_front, p_back;
  const float v_front = average(bsdf_eval(kg, &front, &front.closure[1], transmitted, &p_front));
  const float v_back = average(bsdf_eval(kg, &back, &back.closure[1], front.wi, &p_back));
  const float k_front = v_front / (2.25f * fabsf(dot(front.N, transmitted)));
  const float k_back = v_back / fabsf(dot(back.N, front.wi));
  check(fabsf(k_front - k_back) < 2e-6f, "off-normal front/back reciprocity");

  /* Replace the analytic constant-q cache with the production first-event
   * cache. This checks tinted full-closure power, including the sampled
   * single-event term, against the finite-energy bound on both sides. */
  DiffractionTwoSidedAlbedoRequest request;
  request.alpha_x = request.alpha_y = 0.3f;
  request.wavelength_count = 2;
  request.mu_count = 4;
  request.phi_count = 8;
  request.facet_samples = 512;
  DiffractionTwoSidedAlbedoTable table;
  std::string error;
  check(diffraction_two_sided_albedo_build_cpu(request, table, error),
        "production two-sided cache built");
  if (table.deficits.empty()) return 2;
  const int4 physical_descriptor = make_int4(0, 0, 0, request.mu_count);
  const float4 physical_domain = make_float4(380.0f,
                                             780.0f,
                                             float(request.wavelength_count),
                                             float(request.phi_count));
  globals.diffraction_two_sided_values.data = table.deficits.data();
  globals.diffraction_two_sided_values.width = int(table.deficits.size());
  globals.diffraction_two_sided_integrals.data = table.integrals.data();
  globals.diffraction_two_sided_integrals.width = int(table.integrals.size());
  globals.diffraction_two_sided_cross.data = table.cross_fractions.data();
  globals.diffraction_two_sided_cross.width = int(table.cross_fractions.size());
  globals.diffraction_two_sided_descriptors.data = &physical_descriptor;
  globals.diffraction_two_sided_domains.data = &physical_domain;
  const Spectrum reflection_tint = rgb_to_spectrum(make_float3(0.65f, 0.8f, 0.95f));
  const Spectrum transmission_tint = rgb_to_spectrum(make_float3(0.4f, 0.7f, 0.9f));
  float largest_tinted_furnace = 0.0f;
  for (int side = 0; side < 2; side++) {
    ShaderData sd{};
    sd.N = sd.Ng = side == 0 ? make_float3(0, 0, 1) : make_float3(0, 0, -1);
    sd.wi = normalize(make_float3(0.35f, 0.12f, side == 0 ? 1.0f : -1.0f));
    sd.rand_wavelength = 0.5f;
    sd.num_closure_left = tinted_pair_slots;
    if (side == 1) sd.runtime_flag |= SR_BACKFACING;
    check(bsdf_diffraction_glass_two_sided_setup(kg,
                                                 &sd,
                                                 reflection_tint,
                                                 transmission_tint,
                                                 sd.N,
                                                 make_float3(1, 0, 0),
                                                 sqrtf(0.3f),
                                                 request.inside_ior,
                                                 request.pitch_nm,
                                                 request.depth_nm,
                                                 request.duty,
                                                 0),
          "tinted first-event plus return setup");
    check(sd.num_closure == 2 && sd.num_closure_left == 0,
          "tinted pair allocation exact");
    double total = 0.0;
    for (int closure = 0; closure < sd.num_closure; closure++) {
      double subtotal = 0.0;
      const ShaderClosure *sc = &sd.closure[closure];
      constexpr int samples = 8192;
      for (int i = 0; i < samples; i++) {
        const float3 random = make_float3((float(i) + 0.5f) / float(samples),
                                          fraction((float(i) + 0.5f) * 0.61803398875f),
                                          fraction((float(i) + 0.5f) * 0.75487766625f));
        Spectrum sampled;
        float3 wo;
        float pdf, eta;
        float2 roughness;
        const int label = bsdf_sample(
            kg, &sd, sc, random, &sampled, &wo, &pdf, &roughness, &eta);
        if (label != LABEL_NONE && pdf > 0.0f) {
          subtotal += double(average(sc->weight * sampled) / pdf);
        }
      }
      total += subtotal / double(samples);
    }
    largest_tinted_furnace = max(largest_tinted_furnace, float(total));
    check(total >= 0.0 && total <= 1.02, "tinted complete Glass furnace <= unity");
  }
  ShaderData tinted_front{}, tinted_back{};
  tinted_front.N = tinted_front.Ng = make_float3(0, 0, 1);
  tinted_back.N = tinted_back.Ng = make_float3(0, 0, -1);
  tinted_front.wi = normalize(make_float3(0.3f, 0.2f, 1));
  tinted_back.wi = transmitted;
  tinted_front.rand_wavelength = tinted_back.rand_wavelength = 0.5f;
  tinted_front.num_closure_left = tinted_back.num_closure_left = tinted_pair_slots;
  tinted_back.runtime_flag |= SR_BACKFACING;
  check(bsdf_diffraction_glass_two_sided_setup(kg, &tinted_front,
      reflection_tint, transmission_tint, tinted_front.N, make_float3(1,0,0),
      sqrtf(0.3f), 1.5f, 1600, 150, 0.5f, 0), "tinted front reciprocal setup");
  check(bsdf_diffraction_glass_two_sided_setup(kg, &tinted_back,
      reflection_tint, transmission_tint, tinted_back.N, make_float3(1,0,0),
      sqrtf(0.3f), 1.5f, 1600, 150, 0.5f, 0), "tinted back reciprocal setup");
  float tinted_pdf_forward, tinted_pdf_reverse;
  const Spectrum tinted_forward = bsdf_eval(
      kg, &tinted_front, &tinted_front.closure[1], transmitted, &tinted_pdf_forward);
  const Spectrum tinted_reverse = bsdf_eval(
      kg, &tinted_back, &tinted_back.closure[1], tinted_front.wi, &tinted_pdf_reverse);
  const float3 reciprocal_forward = spectrum_to_rgb(tinted_forward) /
                                    (2.25f * fabsf(dot(tinted_front.N, transmitted)));
  const float3 reciprocal_reverse = spectrum_to_rgb(tinted_reverse) /
                                    fabsf(dot(tinted_back.N, tinted_front.wi));
  check(len(reciprocal_forward - reciprocal_reverse) < 3e-6f,
        "chromatic off-normal front/back reciprocity");
  float single_pdf_forward, single_pdf_reverse;
  const Spectrum single_forward = bsdf_eval(
      kg, &tinted_front, &tinted_front.closure[0], transmitted, &single_pdf_forward);
  const Spectrum single_reverse = bsdf_eval(
      kg, &tinted_back, &tinted_back.closure[0], tinted_front.wi, &single_pdf_reverse);
  const float3 complete_forward = spectrum_to_rgb(single_forward + tinted_forward) /
                                  (2.25f * fabsf(dot(tinted_front.N, transmitted)));
  const float3 complete_reverse = spectrum_to_rgb(single_reverse + tinted_reverse) /
                                  fabsf(dot(tinted_back.N, tinted_front.wi));
  check(len(complete_forward - complete_reverse) < 3e-6f,
        "chromatic complete Glass front/back reciprocity");

  /* Rebuild the production cache with a real film. The first-event closure
   * and q table must now describe the same coated interface. */
  DiffractionTwoSidedAlbedoRequest coated_request = request;
  coated_request.film_ior = 1.32f;
  coated_request.film_thickness_nm = 250.0f;
  const double coated_optical_nm = double(coated_request.film_ior) *
                                   double(coated_request.film_thickness_nm);
  const int coated_minimum_nodes = max(16, int(ceil(1.0 + 400.0 * 16.0 *
                                                coated_optical_nm / (380.0 * 380.0))));
  coated_request.wavelength_count = ((coated_minimum_nodes + 7) / 8) * 8;
  check(coated_request.wavelength_count == 16,
        "coated fixture uses production-derived spectral resolution");
  DiffractionTwoSidedAlbedoTable coated_table;
  error.clear();
  check(diffraction_two_sided_albedo_build_cpu(coated_request, coated_table, error),
        "production coated two-sided cache built");
  if (coated_table.deficits.empty()) {
    std::fprintf(stderr, "coated cache: %s\n", error.c_str());
    return 2;
  }
  const float4 coated_domain = make_float4(380.0f, 780.0f,
                                            float(coated_request.wavelength_count),
                                            float(coated_request.phi_count));
  globals.diffraction_two_sided_values.data = coated_table.deficits.data();
  globals.diffraction_two_sided_values.width = int(coated_table.deficits.size());
  globals.diffraction_two_sided_integrals.data = coated_table.integrals.data();
  globals.diffraction_two_sided_integrals.width = int(coated_table.integrals.size());
  globals.diffraction_two_sided_cross.data = coated_table.cross_fractions.data();
  globals.diffraction_two_sided_cross.width = int(coated_table.cross_fractions.size());
  globals.diffraction_two_sided_domains.data = &coated_domain;
  for (int side = 0; side < 2; side++) {
    ShaderData sd{};
    sd.N = sd.Ng = side == 0 ? make_float3(0, 0, 1) : make_float3(0, 0, -1);
    sd.wi = normalize(make_float3(0.35f, 0.12f, side == 0 ? 1.0f : -1.0f));
    sd.rand_wavelength = 0.5f;
    sd.num_closure_left = coated_pair_slots;
    if (side == 1) sd.runtime_flag |= SR_BACKFACING;
    check(bsdf_diffraction_glass_two_sided_setup(kg, &sd, one_spectrum(), one_spectrum(),
        sd.N, make_float3(1,0,0), sqrtf(coated_request.alpha_x),
        coated_request.inside_ior, coated_request.pitch_nm, coated_request.depth_nm,
        coated_request.duty, 0, coated_request.film_ior,
        coated_request.film_thickness_nm), "coated complete Glass setup");
    check(sd.num_closure == 2 && sd.num_closure_left == 0,
          "coated complete Glass has both lobes");
    double total = 0.0, return_energy = 0.0;
    for (int closure = 0; closure < sd.num_closure; closure++) {
      double subtotal = 0.0;
      const ShaderClosure *sc = &sd.closure[closure];
      constexpr int samples = 8192;
      for (int i = 0; i < samples; i++) {
        const float3 random = make_float3((float(i) + 0.5f) / float(samples),
            fraction((float(i) + 0.5f) * 0.61803398875f),
            fraction((float(i) + 0.5f) * 0.75487766625f));
        Spectrum sampled;
        float3 wo;
        float pdf, eta;
        float2 roughness;
        const int label = bsdf_sample(kg, &sd, sc, random,
                                      &sampled, &wo, &pdf, &roughness, &eta);
        if (label != LABEL_NONE && pdf > 0.0f) {
          subtotal += double(average(sc->weight * sampled) / pdf);
        }
      }
      const double contribution = subtotal / double(samples);
      total += contribution;
      if (closure == 1) return_energy = contribution;
    }
    check(return_energy > 0.02, "coated return contributes positive energy");
    check(fabs(total - 1.0) < 0.05,
          "coated complete Glass white-world energy on both sides");
  }
  std::printf("checks=%d failures=%d max_pdf_error=%g max_eval_error=%g ",
              checks, failures, max_pdf_error, max_eval_error);
  std::printf("reflection_samples=%d transmission_samples=%d reciprocity_error=%g "
              "largest_tinted_furnace=%g\n",
              reflection_count, transmission_count, fabsf(k_front-k_back),
              largest_tinted_furnace);
  return failures == 0 ? 0 : 1;
}
