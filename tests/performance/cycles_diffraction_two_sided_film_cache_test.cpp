/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

/* Focused coated cache reference: double Airy conductance across dense
 * wavelengths, and outgoing BSDF quadrature against the cache's independent
 * facet escape estimator on both oriented sides at grazing incidence. */
#include "scene/diffraction_albedo.h"
#include "kernel/closure/bsdf_diffraction_dielectric.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace ccl;

static double independent_air_film_reflectance(const double cosine,
                                               const double substrate_ior,
                                               const double film_ior,
                                               const double thickness_nm,
                                               const double wavelength_nm)
{
  const double qi = cosine;
  const double qf = std::sqrt(film_ior * film_ior - (1.0 - cosine * cosine));
  const double qo = std::sqrt(substrate_ior * substrate_ior - (1.0 - cosine * cosine));
  const std::complex<double> phase = std::polar(
      1.0, 4.0 * M_PI * thickness_nm * qf / wavelength_nm);
  double reflected = 0.0;
  for (int polarization = 0; polarization < 2; polarization++) {
    const double yi = polarization ? qi : qi;
    const double yf = polarization ? qf / (film_ior * film_ior) : qf;
    const double yo = polarization ? qo / (substrate_ior * substrate_ior) : qo;
    const double r01 = (yi - yf) / (yi + yf);
    const double r12 = (yf - yo) / (yf + yo);
    const std::complex<double> amplitude = (r01 + r12 * phase) /
                                            (1.0 + r01 * r12 * phase);
    reflected += 0.5 * std::norm(amplitude);
  }
  return reflected;
}

static double independent_air_cross(const double substrate_ior,
                                    const double film_ior,
                                    const double thickness_nm,
                                    const double wavelength_nm)
{
  constexpr int samples = 4096;
  double sum = 0.0;
  for (int i = 0; i < samples; i++) {
    const double mu = (i + 0.5) / samples;
    sum += 2.0 * mu * (1.0 - independent_air_film_reflectance(
                                  mu, substrate_ior, film_ior, thickness_nm, wavelength_nm));
  }
  return sum / samples;
}

static double cross_lookup(const DiffractionTwoSidedAlbedoTable &table, const double wavelength_nm)
{
  const auto &r = table.request;
  const double coordinate = (wavelength_nm - 380.0) * (r.wavelength_count - 1) / 400.0;
  const int lower = std::clamp(int(coordinate), 0, r.wavelength_count - 1);
  const int upper = std::min(lower + 1, r.wavelength_count - 1);
  return (1.0 - (coordinate - lower)) * table.cross_fractions[lower] +
         (coordinate - lower) * table.cross_fractions[upper];
}

int main()
{
  DiffractionTwoSidedAlbedoRequest r;
  r.alpha_x = 0.35f;
  r.alpha_y = 0.5f;
  r.pitch_nm = 1200.0f;
  r.depth_nm = 125.0f;
  r.duty = 0.43f;
  r.inside_ior = 1.5f;
  r.film_ior = 2.4f;
  r.film_thickness_nm = 300.0f;
  r.wavelength_count = 40;
  r.mu_count = 4;
  r.phi_count = 8;
  r.facet_samples = 1024;
  DiffractionTwoSidedAlbedoTable table;
  std::string error;
  if (!diffraction_two_sided_albedo_build_cpu(r, table, error)) {
    std::fprintf(stderr, "cache construction failed: %s\n", error.c_str());
    return 1;
  }
  double max_cross_error = 0.0;
  for (int index = 0; index <= 200; index++) {
    const double wavelength_nm = 380.0 + 2.0 * index;
    const double direct = independent_air_cross(
        r.inside_ior, r.film_ior, r.film_thickness_nm, wavelength_nm);
    max_cross_error = std::max(max_cross_error,
                               std::abs(direct - cross_lookup(table, wavelength_nm)));
  }

  constexpr int out_mu_count = 96, out_phi_count = 192;
  const float mu = 1.0f / 9.0f;
  const float phi = M_2PI_F * 3.5f / 8.0f;
  const float radial = std::sqrt(1.0f - mu * mu);
  const float3 wi = make_float3(radial * std::cos(phi), radial * std::sin(phi), mu);
  const float wavelength_nm = 555.0f; /* Between adjacent cache nodes. */
  double max_escape_error = 0.0;
  for (int side = 0; side < 2; side++) {
    DiffractionRoughDielectric p;
    if (!diffraction_dielectric_parameters(wavelength_nm, r.pitch_nm, r.depth_nm, r.duty,
                                           side ? r.inside_ior : 1.0f,
                                           side ? 1.0f : r.inside_ior,
                                           r.alpha_x, r.alpha_y, &p)) return 2;
    double reflected = 0.0, transmitted = 0.0;
    for (int i = 0; i < out_mu_count; i++) {
      const float out_mu = (i + 0.5f) / out_mu_count;
      const float out_radial = std::sqrt(1.0f - out_mu * out_mu);
      for (int j = 0; j < out_phi_count; j++) {
        const float out_phi = M_2PI_F * (j + 0.5f) / out_phi_count;
        const float x = out_radial * std::cos(out_phi);
        const float y = out_radial * std::sin(out_phi);
        float pdf;
        reflected += diffraction_dielectric_eval<GGX>(
            &p, wi, make_float3(x, y, out_mu), &pdf,
            r.film_ior, r.film_thickness_nm / wavelength_nm);
        transmitted += diffraction_dielectric_eval<GGX>(
            &p, wi, make_float3(x, y, -out_mu), &pdf,
            r.film_ior, r.film_thickness_nm / wavelength_nm);
      }
    }
    const double solid_angle = double(M_2PI_F) / (out_mu_count * out_phi_count);
    const double direct_q = 1.0 - (reflected + transmitted) * solid_angle;
    const double cached_q = diffraction_two_sided_albedo_lookup(table, side, wavelength_nm, wi);
    max_escape_error = std::max(max_escape_error, std::abs(direct_q - cached_q));
    std::printf("side %d grazing q cache %.6f direct %.6f\n", side, cached_q, direct_q);
  }
  std::printf("dense cross max absolute %.6f; grazing escape max absolute %.6f\n",
              max_cross_error, max_escape_error);
  if (!(max_cross_error <= 0.015 && max_escape_error <= 0.04)) return 3;
  return 0;
}
