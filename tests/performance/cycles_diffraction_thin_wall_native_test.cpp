/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

/* Standalone native Thin Glass comparison. This is compiled directly against
 * the Cycles kernel when the normal GTest target is not configured. */

#include "kernel/globals.h"
#include "kernel/closure/bsdf.h"
#include "scene/shader.tables"
#include "util/profiling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

CCL_NAMESPACE_BEGIN

static Spectrum full_eval(KernelGlobals kg, ShaderData *sd, const float3 wo)
{
  Spectrum sum = zero_spectrum();
  for (int i = 0; i < sd->num_closure; i++) {
    const ShaderClosure *sc = &sd->closure[i];
    float pdf;
    Spectrum value = zero_spectrum();
    if (sc->type == CLOSURE_BSDF_MICROFACET_GGX_ID) {
      value = bsdf_microfacet_ggx_eval(kg, sc, sd->wi, wo, &pdf);
    }
    else if (sc->type == CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID) {
      value = bsdf_thin_glass_transmission_eval(kg, sc, sd->wi, wo, &pdf);
    }
    else if (sc->type == CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_REFLECTION_ID ||
             sc->type == CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_TRANSMISSION_ID)
    {
      value = bsdf_diffraction_thin_sheet_eval(sc, sd->wi, wo, &pdf);
    }
    sum += sc->weight * value;
  }
  return sum;
}

static ShaderData make_shader_data(const float3 wi, const float3 N)
{
  ShaderData sd = {};
  sd.wi = wi;
  sd.N = sd.Ng = N;
  sd.num_closure_left = MAX_CLOSURE;
  sd.rand_wavelength = 0.5f;
  sd.shader_flag = SD_REQUIRES_WAVELENGTH;
  return sd;
}

int thin_wall_native_main()
{
  std::vector<float> tables(table_ggx_E, table_ggx_E + 1024);
  tables.insert(tables.end(), table_ggx_Eavg, table_ggx_Eavg + 32);
  KernelGlobalsCPU base;
  base.lookup_table.data = tables.data();
  base.lookup_table.width = int(tables.size());
  base.data.tables.ggx_E = 0;
  base.data.tables.ggx_Eavg = 1024;
  Profiler profiler;
  ThreadKernelGlobalsCPU thread(base, nullptr, profiler, 0);
  const KernelGlobals kg = &thread;

  const float3 N = make_float3(0, 0, 1);
  const float3 T = make_float3(1, 0, 0);
  const float3 wi = normalize(make_float3(0.24f, -0.09f, 1.0f));
  const float3 directions[] = {
      normalize(make_float3(-0.19f, 0.12f, 1.0f)),
      normalize(make_float3(-0.17f, 0.05f, -1.0f)),
  };
  const FresnelCoeff tint = {make_spectrum(0.7f), make_spectrum(0.8f)};
  float largest_relative_error = 0.0f;
  int comparisons = 0;
  for (const float roughness : {0.2f, 0.6f, 1.0f}) {
    for (const float film_thickness : {0.0f, 180.0f}) {
      const FresnelThinFilm film = {film_thickness, 1.3f};
      ShaderData native = make_shader_data(wi, N);
      bsdf_thin_glass_setup(kg, &native, true, true, tint, one_spectrum(), N,
                            sqr(roughness), 1.5f, film, PATH_RAY_VISIBILITY_CAMERA, 0);
      for (const float3 wo : directions) {
        const float expected = average(full_eval(kg, &native, wo));
        if (!(std::isfinite(expected) && expected > 0.0f)) {
          std::fprintf(stderr, "Invalid native value at roughness %.2f film %.1f\n",
                       double(roughness), double(film_thickness));
          return 2;
        }
        for (const float depth : {0.0f, 0.01f, 0.001f}) {
          ShaderData grating = make_shader_data(wi, N);
          bsdf_diffraction_thin_glass_setup(
              kg, &grating, true, true, tint, one_spectrum(), N, T,
              sqr(roughness), 1.5f, film, PATH_RAY_VISIBILITY_CAMERA, 0,
              1.0f, 450.0f, depth, 0.42f, 0.0f);
          const float actual = average(full_eval(kg, &grating, wo));
          const float relative_error = std::abs(actual / expected - 1.0f);
          largest_relative_error = std::max(largest_relative_error, relative_error);
          comparisons++;
          if (!std::isfinite(actual) || relative_error > (depth == 0.0f ? 1e-5f : 1e-3f)) {
            for (int j = 0; j < native.num_closure; j++) {
              const ShaderClosure *s = &native.closure[j];
              float debug_pdf;
              const Spectrum debug_value = s->type == CLOSURE_BSDF_MICROFACET_GGX_ID ?
                  bsdf_microfacet_ggx_eval(kg, s, wi, wo, &debug_pdf) :
                  bsdf_thin_glass_transmission_eval(kg, s, wi, wo, &debug_pdf);
              std::fprintf(stderr, "native closure %d type %d weight %.6g\n", j, int(s->type),
                           double(average(s->weight)));
              std::fprintf(stderr, " value %.8g pdf %.8g\n", double(average(debug_value)),
                           double(debug_pdf));
            }
            for (int j = 0; j < grating.num_closure; j++) {
              const ShaderClosure *s = &grating.closure[j];
              float debug_pdf;
              const Spectrum debug_value = bsdf_diffraction_thin_sheet_eval(s, wi, wo, &debug_pdf);
              std::fprintf(stderr, "candidate closure %d type %d weight %.6g\n", j, int(s->type),
                           double(average(s->weight)));
              std::fprintf(stderr, " value %.8g pdf %.8g\n", double(average(debug_value)),
                           double(debug_pdf));
            }
            std::fprintf(stderr,
                         "FAIL roughness %.2f film %.1f depth %.4f %s native %.8g candidate %.8g relative %.8g\n",
                         double(roughness), double(film_thickness), double(depth),
                         wo.z > 0 ? "reflect" : "transmit", double(expected), double(actual),
                         double(relative_error));
            return 1;
          }
        }
      }
    }
  }
  std::printf("PASS native full-closure comparisons=%d max_relative_error=%.8g\n",
              comparisons, double(largest_relative_error));
  float largest_furnace = 0.0f;
  for (const float roughness : {0.2f, 0.6f, 1.0f}) {
    for (const float incidence : {0.0f, 0.5f, 0.8f}) {
      ShaderData sheet = make_shader_data(
          normalize(make_float3(incidence, 0.0f, 1.0f)), N);
      ShaderData flat = make_shader_data(sheet.wi, N);
      bsdf_thin_glass_setup(kg, &flat, true, true,
                            {one_spectrum(), one_spectrum()}, one_spectrum(), N,
                            sqr(roughness), 1.5f, {0.0f, 1.0f}, PATH_RAY_VISIBILITY_CAMERA, 0);
      bsdf_diffraction_thin_glass_setup(
          kg, &sheet, true, true, {one_spectrum(), one_spectrum()}, one_spectrum(), N, T,
          sqr(roughness), 1.5f, {0.0f, 1.0f}, PATH_RAY_VISIBILITY_CAMERA, 0,
          1.0f, 1150.0f, 320.0f, 0.42f, 0.0f);
      double integral = 0.0;
      double native_integral = 0.0;
      constexpr int polar_samples = 32;
      constexpr int azimuth_samples = 64;
      for (int side = -1; side <= 1; side += 2) {
        for (int polar = 0; polar < polar_samples; polar++) {
          const float mu = (polar + 0.5f) / polar_samples;
          const float sine = sqrtf(1.0f - sqr(mu));
          for (int azimuth = 0; azimuth < azimuth_samples; azimuth++) {
            const float phi = M_2PI_F * (azimuth + 0.5f) / azimuth_samples;
            const float3 wo = make_float3(sine * cosf(phi), sine * sinf(phi), side * mu);
            integral += average(full_eval(kg, &sheet, wo));
            native_integral += average(full_eval(kg, &flat, wo));
          }
        }
      }
      const float energy = float(integral * M_2PI_F / (polar_samples * azimuth_samples));
      const float native_energy = float(native_integral * M_2PI_F /
                                        (polar_samples * azimuth_samples));
      largest_furnace = std::max(largest_furnace, energy);
      std::printf("furnace roughness=%.2f incidence=%.2f grating=%.6f native=%.6f\n",
                  double(roughness), double(incidence), double(energy), double(native_energy));
    }
  }
  std::printf("largest sampled furnace energy=%.6f (approximate quadrature)\n",
              double(largest_furnace));
  return 0;
}

CCL_NAMESPACE_END

int main()
{
  return ccl::thin_wall_native_main();
}
