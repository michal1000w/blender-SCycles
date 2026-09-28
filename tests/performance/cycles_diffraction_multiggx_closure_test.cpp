/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

/* Standalone native closure check. The albedo table is built with production
 * host code and exposed through actual KernelGlobalsCPU arrays. */
#include "kernel/closure/bsdf.h"
#include "scene/diffraction_albedo.h"
#include "util/profiling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace ccl;

static float fraction(const float value)
{
  return value - std::floor(value);
}

int main()
{
  std::setbuf(stdout, nullptr);
  int failures = 0;
  int compared_samples = 0;
  float largest_sample_pdf_error = 0.0f;
  float largest_sample_value_error = 0.0f;
  float largest_reciprocity_error = 0.0f;
  float largest_furnace_error = 0.0f;
  float largest_pdf_mass_error = 0.0f;
  static_assert(sizeof(MicrofacetBsdf) == 96, "Diffraction wavelength must fit existing padding");
  static_assert(sizeof(DiffractionConductorExtra) == 144, "No extra closure slots");
#ifdef __SPECTRAL__
  {
    KernelGlobalsCPU base{};
    base.data.film.is_rec709 = 1;
    base.data.film.xyz_to_r = make_float4(3.2406f, -1.5372f, -0.4986f, 0);
    base.data.film.xyz_to_g = make_float4(-0.9689f, 1.8758f, 0.0415f, 0);
    base.data.film.xyz_to_b = make_float4(0.0557f, -0.2040f, 1.0570f, 0);
    Profiler profiler;
    ThreadKernelGlobalsCPU globals(base, nullptr, profiler, 0);
    KernelGlobals kg = &globals;
    float old_min_y = 0, corrected_min_y = 0;
    for (int nm = 380; nm <= 780; nm++) {
      const float3 sensor = wavelength_to_rgb_d65(kg, dielectric_wavelength_um(float(nm)));
      const Spectrum color = make_float3(0.7f, 0.4f, 0.15f);
      const Spectrum reconstructed = microfacet_diffraction_spectral_reflectance(kg, color, float(nm));
      const float3 y = make_float3(.2126f, .7152f, .0722f);
      old_min_y = std::min(old_min_y, dot(sensor * color, y));
      const float corrected_y = dot(sensor * reconstructed, y);
      corrected_min_y = std::min(corrected_min_y, corrected_y);
      failures += corrected_y < -1e-6f || reduce_min(reconstructed) < 0 ||
                  reduce_max(reconstructed) > 1.000001f || reconstructed.x != reconstructed.y ||
                  reconstructed.y != reconstructed.z;
      failures += !isequal(microfacet_diffraction_spectral_reflectance(kg, color, 0), color);
    }
    for (const float depth : {0.0f, 320.0f}) {
      for (const float coverage : {0.0f, 0.5f, 1.0f}) {
        ShaderData sd{};
        sd.N = sd.Ng = make_float3(0, 0, 1);
        sd.wi = sd.N;
        sd.rand_wavelength = .5f;
        sd.num_closure_left = MAX_CLOSURE;
        auto *native = (MicrofacetBsdf *)bsdf_alloc(&sd, sizeof(MicrofacetBsdf), one_spectrum());
        native->N = sd.N; native->T = make_float3(1, 0, 0);
        native->alpha_x = native->alpha_y = .3f;
        bsdf_microfacet_ggx_setup(native);
        FresnelF82Tint fresnel{};
        fresnel.f0 = make_float3(.7f, .4f, .15f);
        native->fresnel_type = MicrofacetFresnel::F82_TINT;
        native->fresnel = &fresnel;
        failures += !bsdf_diffraction_conductor_split_setup(&sd, native, native->T,
                                                             1200, depth, .41f, coverage);
        if (depth == 0 || coverage == 0) {
          failures += native->type != CLOSURE_BSDF_MICROFACET_GGX_ID ||
                      native->diffraction_wavelength_nm != 0 || sd.num_closure != 1 ||
                      sd.runtime_flag & SR_BSDF_HAS_DISPERSION;
          const Spectrum unmarked = microfacet_fresnel(kg, native, 1, nullptr).reflectance;
          failures += !isequal(unmarked, fresnel.f0);
        }
        else {
          auto *grating = (DiffractionConductorBsdf *)(coverage == 1 ?
              (ShaderClosure *)native : &sd.closure[1]);
          failures += grating->extra->carrier.diffraction_wavelength_nm <= 0;
          if (coverage < 1) {
            const Spectrum marked = microfacet_fresnel(kg, native, .5f, nullptr).reflectance;
            failures += native->diffraction_wavelength_nm <= 0 || marked.x != marked.y ||
                        marked.y != marked.z;
          }
        }
      }
    }
    MicrofacetBsdf zero{};
    zero.N = make_float3(0, 0, 1); zero.alpha_x = zero.alpha_y = .3f;
    bsdf_microfacet_ggx_setup(&zero);
    ShaderData zero_sd{}; zero_sd.num_closure_left = MAX_CLOSURE;
    failures += bsdf_diffraction_conductor_setup(&zero_sd, &zero, make_float3(1,0,0),
                                                 1200, 0, .41f) ||
                zero.diffraction_wavelength_nm != 0 || zero_sd.num_closure != 0;
    failures += !(old_min_y < -0.001f);
    std::printf("spectral_rgb_old_min_y=%g scalar_reflectance_min_y=%g microfacet_size=%zu\n",
                double(old_min_y), double(corrected_min_y), sizeof(MicrofacetBsdf));
  }
#endif
  const float3 N = make_float3(0.0f, 0.0f, 1.0f);
  const float3 T = make_float3(1.0f, 0.0f, 0.0f);

  for (const float alpha : {0.3f, 0.6f, 0.9f}) {
    DiffractionAlbedoRequest request;
    request.alpha_x = alpha;
    request.alpha_y = alpha;
    request.pitch_nm = 1600.0f;
    request.depth_nm = 150.0f;
    request.duty = 0.41f;
    request.facet_samples = 512;
    DiffractionAlbedoTable table;
    std::string error;
    if (!diffraction_albedo_build_cpu(request, table, error)) {
      std::fprintf(stderr, "albedo build failed (alpha=%g): %s\n", double(alpha), error.c_str());
      return 2;
    }
    const int4 descriptor = make_int4(0, 0, request.mu_count, request.phi_count);
    const float4 domain = make_float4(request.wavelength_min_nm,
                                      request.wavelength_max_nm,
                                      float(request.wavelength_count),
                                      float(request.algorithm_revision));
    KernelGlobalsCPU base{};
    base.data.film.is_rec709 = 1;
    base.diffraction_albedo_values.data = table.values.data();
    base.diffraction_albedo_values.width = int(table.values.size());
    base.diffraction_albedo_averages.data = table.averages.data();
    base.diffraction_albedo_averages.width = int(table.averages.size());
    base.diffraction_albedo_descriptors.data = &descriptor;
    base.diffraction_albedo_descriptors.width = 1;
    base.diffraction_albedo_domains.data = &domain;
    base.diffraction_albedo_domains.width = 1;
    base.data.tables.num_diffraction_albedo_caches = 1;
    KernelObject object{};
    base.objects.data = &object;
    base.objects.width = 1;
    Profiler profiler;
    ThreadKernelGlobalsCPU globals(base, nullptr, profiler, 0);
    KernelGlobals kg = &globals;

    for (const float cosine_i : {0.35f, 0.75f}) {
      ShaderData sd{};
      sd.N = sd.Ng = N;
      sd.wi = make_float3(std::sqrt(1.0f - cosine_i * cosine_i), 0.0f, cosine_i);
      sd.rand_wavelength = 0.5f;
      sd.num_closure_left = MAX_CLOSURE;
      if (!bsdf_diffraction_glossy_setup(kg, &sd, one_spectrum(), N, T, alpha, alpha,
                                         request.pitch_nm, request.depth_nm, request.duty,
                                         request.medium_ior, false, 0) ||
          sd.num_closure != 1)
      {
        std::fprintf(stderr, "compensated glossy setup failed\n");
        return 3;
      }
      const ShaderClosure *closure = &sd.closure[0];
      const auto *conductor = (const DiffractionConductorBsdf *)closure;
      if (conductor->extra->albedo_handle != 0) return 4;
      const float missing = 1.0f - diffraction_albedo_lookup(
          kg, 0, conductor->extra->wavelength_nm, sd.wi).x;

      /* Sample/eval must use the same complete mixture density, including
       * native single-scatter proposals and the cosine extra branch. */
      int accepted_samples = 0;
      int extra_branch_samples = 0;
      for (int s = 0; s < 4096; ++s) {
        const float3 random = make_float3((s + 0.5f) / 4096.0f,
                                          fraction((s + 0.5f) * 0.6180339887498949f),
                                          fraction((s + 0.5f) * 0.7548776662466927f));
        extra_branch_samples += random.z < missing;
        Spectrum sampled_value;
        float3 wo;
        float sampled_pdf, eta;
        float2 roughness;
        const int label = bsdf_sample(kg, &sd, closure, random, &sampled_value,
                                      &wo, &sampled_pdf, &roughness, &eta);
        if (label == LABEL_NONE) continue;
        ++accepted_samples;
        float queried_pdf;
        const Spectrum queried_value = bsdf_eval(kg, &sd, closure, wo, &queried_pdf);
        const float pdf_error = std::abs(sampled_pdf - queried_pdf);
        const float value_error = reduce_max(fabs(sampled_value - queried_value));
        largest_sample_pdf_error = std::max(largest_sample_pdf_error, pdf_error);
        largest_sample_value_error = std::max(largest_sample_value_error, value_error);
        failures += !std::isfinite(sampled_pdf) || !(sampled_pdf > 0.0f) ||
                    !std::isfinite(queried_pdf) ||
                    pdf_error > 5e-5f * std::max(1.0f, sampled_pdf) ||
                    value_error > 5e-5f * std::max(1.0f, reduce_max(sampled_value));
        ++compared_samples;
      }

      /* Independent uniform-solid-angle quadrature of f*cos_o; no samples
       * from either BSDF branch are used to estimate furnace energy. */
      double energy = 0.0;
      double pdf_mass = 0.0;
      constexpr int mu_count = 64, phi_count = 128;
      for (int i = 0; i < mu_count; ++i) {
        const float mu = (i + 0.5f) / float(mu_count);
        const float radial = std::sqrt(1.0f - mu * mu);
        for (int j = 0; j < phi_count; ++j) {
          const float phi = M_2PI_F * (j + 0.5f) / float(phi_count);
          const float3 wo = make_float3(radial * std::cos(phi), radial * std::sin(phi), mu);
          float pdf, reverse_pdf;
          const Spectrum forward = bsdf_diffraction_conductor_eval(kg, closure, sd.wi, wo, &pdf);
          const Spectrum reverse = bsdf_diffraction_conductor_eval(kg, closure, wo, sd.wi,
                                                                   &reverse_pdf);
          const float forward_scalar = average(forward);
          const float reverse_scalar = average(reverse);
          const float reciprocity_error = std::abs(forward_scalar / wo.z -
                                                   reverse_scalar / sd.wi.z);
          largest_reciprocity_error = std::max(largest_reciprocity_error, reciprocity_error);
          failures += !std::isfinite(forward_scalar) || !std::isfinite(pdf) ||
                      forward_scalar < 0.0f || pdf < 0.0f ||
                      reciprocity_error > 5e-4f *
                                             std::max(1.0f, forward_scalar / wo.z);
          energy += forward_scalar;
          pdf_mass += pdf;
        }
      }
      energy *= double(M_2PI_F) / double(mu_count * phi_count);
      pdf_mass *= double(M_2PI_F) / double(mu_count * phi_count);
      const float pdf_mass_error = std::abs(float(pdf_mass) -
                                            float(accepted_samples) / 4096.0f);
      largest_pdf_mass_error = std::max(largest_pdf_mass_error, pdf_mass_error);
      const float furnace_error = std::abs(float(energy) - 1.0f);
      largest_furnace_error = std::max(largest_furnace_error, furnace_error);
      std::printf("alpha=%g mu_i=%g furnace=%g error=%g pdf_mass=%g sampled_mass=%g\n",
                  double(alpha), double(cosine_i), energy, double(furnace_error), pdf_mass,
                  double(accepted_samples) / 4096.0);
      failures += furnace_error > 0.03f;
      failures += !std::isfinite(pdf_mass) || pdf_mass < 0.0 || pdf_mass > 1.01 ||
                  pdf_mass_error > 0.03f;
      failures += extra_branch_samples == 0 || extra_branch_samples == 4096;

      if (alpha == 0.6f && cosine_i == 0.35f) {
        /* Isolate the added lobe by comparing the same closure with and
         * without its table handle. Color is supplied independently of the
         * outer closure weight, as required for partial diffraction mixes. */
        const float3 probe_wo = normalize(make_float3(-0.2f, 0.3f, 1.0f));
        float extras[3] = {};
        const float colors[3] = {0.0f, 0.5f, 1.0f};
        for (int c = 0; c < 3; ++c) {
          ShaderData probe{};
          probe.N = probe.Ng = N;
          probe.wi = sd.wi;
          probe.rand_wavelength = sd.rand_wavelength;
          probe.num_closure_left = MAX_CLOSURE;
          if (!bsdf_diffraction_glossy_setup(kg, &probe, one_spectrum(), N, T,
                                             alpha, alpha, request.pitch_nm,
                                             request.depth_nm, request.duty,
                                             request.medium_ior, false, 0,
                                             make_spectrum(colors[c])))
            return 5;
          auto *test_closure = (DiffractionConductorBsdf *)&probe.closure[0];
          float pdf;
          const Spectrum full = bsdf_diffraction_conductor_eval(
              kg, &probe.closure[0], probe.wi, probe_wo, &pdf);
          const Spectrum full_reverse = bsdf_diffraction_conductor_eval(
              kg, &probe.closure[0], probe_wo, probe.wi, &pdf);
          test_closure->extra->albedo_handle = -1;
          const Spectrum single = bsdf_diffraction_conductor_eval(
              kg, &probe.closure[0], probe.wi, probe_wo, &pdf);
          const Spectrum single_reverse = bsdf_diffraction_conductor_eval(
              kg, &probe.closure[0], probe_wo, probe.wi, &pdf);
          const float added = average(full - single);
          const float added_reverse = average(full_reverse - single_reverse);
          extras[c] = added;
          failures += !std::isfinite(added) || added < -1e-7f ||
                      std::abs(added / probe_wo.z - added_reverse / probe.wi.z) > 2e-5f;
        }
        failures += std::abs(extras[0]) > 1e-7f || !(extras[1] > 0.0f) ||
                    !(extras[2] > extras[1]);
        std::printf("extra_lobe color=(0,.5,1): (%g,%g,%g)\n",
                    double(extras[0]), double(extras[1]), double(extras[2]));
      }

      if (alpha == 0.6f) {
        /* Full-coverage Metallic does not need Blender's planar GGX LUTs.
         * Set the Fresnel payload directly: F82's usual setup estimates
         * sampling albedo from a separate LUT unavailable in this fixture.
         * That estimate does not enter this closure's eval or sample PDF. */
        for (int metallic_case = 0; metallic_case < 4; ++metallic_case) {
          const bool conductor_case = metallic_case == 0;
          const float f0 = metallic_case <= 1 ? 1.0f :
                           metallic_case == 2 ? 0.4f : 0.02f;
          ShaderData metal_sd{};
          metal_sd.N = metal_sd.Ng = N;
          metal_sd.wi = sd.wi;
          metal_sd.rand_wavelength = sd.rand_wavelength;
          metal_sd.num_closure_left = MAX_CLOSURE;
          auto *carrier = (MicrofacetBsdf *)bsdf_alloc(
              &metal_sd, sizeof(MicrofacetBsdf), one_spectrum());
          if (!carrier) return 6;
          carrier->N = N;
          carrier->T = T;
          carrier->alpha_x = carrier->alpha_y = alpha;
          carrier->ior = conductor_case ? 1.0f : 0.0f;
          bsdf_microfacet_ggx_setup(carrier);
          if (conductor_case) {
            auto *fresnel = (FresnelConductor *)closure_alloc_extra(
                &metal_sd, sizeof(FresnelConductor));
            if (!fresnel) return 7;
            fresnel->thin_film.thickness = 0.0f;
            fresnel->thin_film.ior = 1.33f;
            fresnel->ior = {zero_spectrum(), one_spectrum()};
            carrier->fresnel_type = MicrofacetFresnel::CONDUCTOR;
            carrier->fresnel = fresnel;
          }
          else {
            auto *fresnel = (FresnelF82Tint *)closure_alloc_extra(
                &metal_sd, sizeof(FresnelF82Tint));
            if (!fresnel) return 8;
            fresnel->thin_film.thickness = 0.0f;
            fresnel->thin_film.ior = 1.33f;
            fresnel->f0 = make_spectrum(f0);
            fresnel->b = fresnel_f82tint_B(fresnel->f0, make_spectrum(f0));
            carrier->fresnel_type = MicrofacetFresnel::F82_TINT;
            carrier->fresnel = fresnel;
          }
          if (!bsdf_diffraction_conductor_multiggx_split_setup(
                  kg, &metal_sd, carrier, T, request.pitch_nm,
                  request.depth_nm, request.duty, 1.0f, 0) ||
              metal_sd.num_closure != 1)
          {
            std::fprintf(stderr, "Metallic full-coverage setup failed (%d)\n", metallic_case);
            return 9;
          }
          const ShaderClosure *metal_closure = &metal_sd.closure[0];
          const auto *metal_grating = (const DiffractionConductorBsdf *)metal_closure;
          failures += metal_grating->extra->albedo_handle != 0;

          for (int s = 0; s < 1024; ++s) {
            const float3 random = make_float3((s + 0.5f) / 1024.0f,
                                              fraction((s + 0.5f) * 0.6180339887f),
                                              fraction((s + 0.5f) * 0.7548776662f));
            Spectrum sampled;
            float3 wo;
            float sample_pdf, eta;
            float2 roughness;
            if (bsdf_sample(kg, &metal_sd, metal_closure, random, &sampled,
                            &wo, &sample_pdf, &roughness, &eta) == LABEL_NONE)
              continue;
            float eval_pdf;
            const Spectrum queried = bsdf_eval(kg, &metal_sd, metal_closure, wo, &eval_pdf);
            failures += !std::isfinite(sample_pdf) || !(sample_pdf > 0.0f) ||
                        !std::isfinite(eval_pdf) ||
                        std::abs(sample_pdf - eval_pdf) > 5e-5f * std::max(1.0f, sample_pdf) ||
                        reduce_max(fabs(sampled - queried)) >
                            5e-5f * std::max(1.0f, reduce_max(sampled));
            ++compared_samples;
          }

          double metal_energy = 0.0;
          constexpr int metal_mu_count = 64, metal_phi_count = 128;
          for (int i = 0; i < metal_mu_count; ++i) {
            const float mu = (i + 0.5f) / float(metal_mu_count);
            const float radial = std::sqrt(1.0f - mu * mu);
            for (int j = 0; j < metal_phi_count; ++j) {
              const float phi = M_2PI_F * (j + 0.5f) / float(metal_phi_count);
              const float3 wo = make_float3(radial * std::cos(phi), radial * std::sin(phi), mu);
              float pdf, reverse_pdf;
              const Spectrum forward = bsdf_diffraction_conductor_eval(
                  kg, metal_closure, metal_sd.wi, wo, &pdf);
              const Spectrum reverse = bsdf_diffraction_conductor_eval(
                  kg, metal_closure, wo, metal_sd.wi, &reverse_pdf);
              const float forward_scalar = average(forward);
              const float reverse_scalar = average(reverse);
              const float reciprocity_error = std::abs(forward_scalar / wo.z -
                                                       reverse_scalar / metal_sd.wi.z);
              largest_reciprocity_error = std::max(largest_reciprocity_error, reciprocity_error);
              failures += !std::isfinite(forward_scalar) || !std::isfinite(pdf) ||
                          forward_scalar < 0.0f || pdf < 0.0f ||
                          reciprocity_error > 5e-4f *
                                                 std::max(1.0f, forward_scalar / wo.z);
              metal_energy += forward_scalar;
            }
          }
          metal_energy *= double(M_2PI_F) / double(metal_mu_count * metal_phi_count);
          const float metal_error = std::abs(float(metal_energy) - 1.0f);
          if (conductor_case || f0 == 1.0f) {
            largest_furnace_error = std::max(largest_furnace_error, metal_error);
            failures += metal_error > 0.03f;
          }
          else {
            failures += !std::isfinite(metal_energy) || metal_energy < 0.0 ||
                        metal_energy > 1.03;
          }
          std::printf("metallic=%s f0=%g mu_i=%g furnace=%g\n",
                      conductor_case ? "conductor" : "f82", double(f0),
                      double(cosine_i), metal_energy);
        }
      }
    }
  }
  failures += compared_samples < 1000;
  std::printf("compared_samples=%d max_pdf_error=%g max_value_error=%g "
              "max_reciprocity_error=%g max_furnace_error=%g "
              "max_pdf_mass_error=%g failures=%d\n",
              compared_samples, double(largest_sample_pdf_error),
              double(largest_sample_value_error), double(largest_reciprocity_error),
              double(largest_furnace_error), double(largest_pdf_mass_error), failures);
  return failures ? 1 : 0;
}
