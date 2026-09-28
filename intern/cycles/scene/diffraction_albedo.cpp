/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction_albedo.h"

#include "kernel/closure/bsdf_diffraction.h"
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include "kernel/closure/diffraction_thin_sheet_model.h"
#include "kernel/util/dielectric_dispersion.h"
#include "kernel/util/dielectric_f0_cache.h"

#include <algorithm>
#include <cmath>
#include <limits>

CCL_NAMESPACE_BEGIN

namespace {

constexpr float golden_conjugate = 0.6180339887498949f;
constexpr size_t maximum_table_entries = 8 * 1024 * 1024;
constexpr double maximum_estimator_work = 128.0e6;

size_t direction_count(const DiffractionAlbedoRequest &r)
{
  return size_t(r.mu_count) * size_t(r.phi_count);
}

float node_mu(const int i, const int count)
{
  const float x = float(i) / float(count - 1);
  return x * x;
}

float node_wavelength(const DiffractionAlbedoRequest &r, const int i)
{
  return r.wavelength_min_nm +
         (r.wavelength_max_nm - r.wavelength_min_nm) * float(i) / float(r.wavelength_count - 1);
}

float estimate(const DiffractionReflection &p,
               const float3 wi,
               const int samples,
               const std::function<bool()> &cancelled,
               bool &was_cancelled)
{
  const float lambda_i = bsdf_aniso_lambda<GGX>(p.alpha_x, p.alpha_y, wi);
  const int limit = diffraction_reflection_max_order(&p);
  const float3 axis = make_float3(1.0f, 0.0f, 0.0f);
  double sum = 0.0;
  for (int s = 0; s < samples; ++s) {
    if ((s & 63) == 0 && cancelled && cancelled()) {
      was_cancelled = true;
      return 0.0f;
    }
    const float u = (float(s) + 0.5f) / float(samples);
    const float v = (float(s) + 0.5f) * golden_conjugate;
    const float3 h = microfacet_ggx_sample_vndf(
        wi, p.alpha_x, p.alpha_y, make_float2(u, v - std::floor(v)));
    const float ci = dot(wi, h);
    float nonzero_mass = 0.0f;
    for (int order = -limit; order <= limit; ++order) {
      float3 wo;
      if (order == 0 || !diffraction_facet_reflect(
                            wi, h, axis, float(order) * p.wavelength_over_pitch, &wo)) {
        continue;
      }
      const float mass = diffraction_reflection_nonzero_power(&p, order, ci, dot(wo, h));
      nonzero_mass += mass;
      if (wo.z > 0.0f) {
        const float lambda_o = bsdf_aniso_lambda<GGX>(p.alpha_x, p.alpha_y, wo);
        sum += double(mass) * (1.0f + lambda_i) / (1.0f + lambda_i + lambda_o);
      }
    }
    const float3 wo = 2.0f * ci * h - wi;
    if (wo.z > 0.0f) {
      const float lambda_o = bsdf_aniso_lambda<GGX>(p.alpha_x, p.alpha_y, wo);
      sum += double(std::max(0.0f, 1.0f - nonzero_mass)) * (1.0f + lambda_i) /
             (1.0f + lambda_i + lambda_o);
    }
  }
  return float(sum / double(samples));
}

double projected_average(const float *values, const int mu_count, const int phi_count)
{
  double integral = 0.0;
  for (int i = 0; i < mu_count - 1; ++i) {
    const double mu = node_mu(i, mu_count);
    const double h = double(node_mu(i + 1, mu_count)) - mu;
    for (int j = 0; j < phi_count; ++j) {
      const double a = values[i * phi_count + j];
      const double delta = double(values[(i + 1) * phi_count + j]) - a;
      integral += 2.0 * h * (mu * a + 0.5 * (mu * delta + h * a) + h * delta / 3.0);
    }
  }
  return integral / double(phi_count);
}

float angular_lookup(const DiffractionAlbedoTable &t, const int slice, const float3 &w)
{
  const int mu_count = t.request.mu_count;
  const int phi_count = t.request.phi_count;
  const float mu = std::clamp(w.z, 0.0f, 1.0f);
  const float coordinate = std::sqrt(mu) * float(mu_count - 1);
  const int i = std::min(int(std::floor(coordinate)), mu_count - 1);
  const int k = std::min(i + 1, mu_count - 1);
  const float phi = (std::atan2(w.y, w.x) / M_2PI_F + 1.0f) * float(phi_count) - 0.5f;
  const int j = (int(std::floor(phi)) % phi_count + phi_count) % phi_count;
  const int l = (j + 1) % phi_count;
  const float mi = node_mu(i, mu_count), mk = node_mu(k, mu_count);
  const float a = (i == k) ? 0.0f : std::clamp((mu - mi) / (mk - mi), 0.0f, 1.0f);
  const float b = phi - std::floor(phi);
  const size_t base = size_t(slice) * direction_count(t.request);
  const auto at = [&](const int m, const int p) { return t.values[base + size_t(m * phi_count + p)]; };
  return (1.0f - b) * ((1.0f - a) * at(i, j) + a * at(k, j)) +
         b * ((1.0f - a) * at(i, l) + a * at(k, l));
}

float spectral_coordinate(const DiffractionAlbedoRequest &r, const float wavelength_nm)
{
  return (std::clamp(wavelength_nm, r.wavelength_min_nm, r.wavelength_max_nm) -
          r.wavelength_min_nm) *
         float(r.wavelength_count - 1) / (r.wavelength_max_nm - r.wavelength_min_nm);
}

}  // namespace

bool diffraction_albedo_validate_request(const DiffractionAlbedoRequest &r, std::string &error)
{
  const auto finite = [](const float v) { return std::isfinite(v); };
  const double minimum_medium_wavelength = double(r.wavelength_min_nm) / double(r.medium_ior);
  const double smallest_order_spacing = minimum_medium_wavelength / double(r.pitch_nm);
  const double maximum_order = std::isfinite(smallest_order_spacing) &&
                                       smallest_order_spacing > 0.0 ?
                                   std::min(256.0, std::floor(2.0 / smallest_order_spacing)) :
                                   256.0;
  const double estimated_work = double(r.wavelength_count) * double(r.mu_count) *
                                double(r.phi_count) * double(r.facet_samples) *
                                (r.depth_nm == 0.0f ? 1.0 : 2.0 * maximum_order + 1.0) *
                                (r.thin_sheet ? 2.0 : 1.0);
  if (!finite(r.alpha_x) || !finite(r.alpha_y) || r.alpha_x < (r.thin_sheet ? 0.0f : 1e-4f) ||
      r.alpha_y < (r.thin_sheet ? 0.0f : 1e-4f) || r.alpha_x > 1.0f || r.alpha_y > 1.0f ||
      !finite(r.pitch_nm) || r.pitch_nm <= 0.0f || r.pitch_nm > 1000000.0f ||
      !finite(r.depth_nm) || r.depth_nm < 0.0f || r.depth_nm > 1000000.0f ||
      !finite(r.duty) || r.duty < 0.0f || r.duty > 1.0f ||
      !finite(r.medium_ior) || r.medium_ior <= 0.0f || r.medium_ior > 100.0f ||
      !finite(r.wavelength_min_nm) || !finite(r.wavelength_max_nm) ||
      r.wavelength_min_nm != 380.0f || r.wavelength_max_nm != 780.0f ||
      r.wavelength_count < 2 || r.wavelength_count > 1024 || r.mu_count < 2 ||
      r.mu_count > 256 || r.phi_count < 2 || r.phi_count > 512 ||
      r.facet_samples < 1 || r.facet_samples > 65536 || r.algorithm_revision != 1 ||
      size_t(r.wavelength_count) * size_t(r.mu_count) * size_t(r.phi_count) >
          maximum_table_entries || estimated_work > maximum_estimator_work)
  {
    error = "Invalid GGX diffraction albedo request";
    return false;
  }
  if (r.thin_sheet) {
    if (!finite(r.inside_ior) || r.inside_ior < 1 || r.inside_ior > 10 ||
        !finite(r.inv_abbe) || r.inv_abbe < 0 || !finite(r.film_ior) ||
        r.film_ior <= 0 || r.film_ior > 10 || !finite(r.film_thickness_nm) ||
        r.film_thickness_nm < 0 || r.film_thickness_nm > 1000000 ||
        !finite(r.reflection_tint) || r.reflection_tint < 0 || r.reflection_tint > 1 ||
        !finite(r.transmission_tint) || r.transmission_tint < 0 || r.transmission_tint > 1 ||
        r.medium_ior != 1.0f || (r.phi_count % 2) != 0 ||
        (r.transmission_is_spectral &&
         (!finite(r.transmission_bt709.x) || !finite(r.transmission_bt709.y) ||
          !finite(r.transmission_bt709.z)))) {
      error = "Invalid passive thin-sheet diffraction cache request";
      return false;
    }
    for (float wavelength : {r.wavelength_min_nm, r.wavelength_max_nm}) {
      const float n = dielectric_ior_at_wavelength(
          r.inside_ior, r.inv_abbe, dielectric_wavelength_um(wavelength));
      if (!finite(n) || n < 1 || n > 10) {
        error = "Invalid thin-sheet dispersion endpoint";
        return false;
      }
    }
  }
  error.clear();
  return true;
}

bool diffraction_albedo_validate_table(const DiffractionAlbedoTable &t, std::string &error)
{
  if (!diffraction_albedo_validate_request(t.request, error)) return false;
  if (t.values.size() != size_t(t.request.wavelength_count) * direction_count(t.request) ||
      t.averages.size() != size_t(t.request.wavelength_count)) {
    error = "GGX diffraction albedo table has inconsistent dimensions";
    return false;
  }
  for (const float value : t.values) {
    if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
      error = "GGX diffraction albedo table contains invalid values";
      return false;
    }
  }
  for (int s = 0; s < t.request.wavelength_count; ++s) {
    const double average = projected_average(t.values.data() + size_t(s) * direction_count(t.request),
                                             t.request.mu_count, t.request.phi_count);
    if (!std::isfinite(t.averages[s]) || t.averages[s] < 0.0f || t.averages[s] > 1.0f ||
        std::abs(double(t.averages[s]) - average) > 1e-5) {
      error = "GGX diffraction albedo table has invalid projected average";
      return false;
    }
  }
  error.clear();
  return true;
}

void diffraction_albedo_symmetrize_thin_sheet(DiffractionAlbedoTable &table)
{
  const DiffractionAlbedoRequest &r = table.request;
  if (!r.thin_sheet || r.phi_count < 2 || (r.phi_count % 2) != 0) return;
  for (int wavelength = 0; wavelength < r.wavelength_count; wavelength++) {
    for (int mu = 0; mu < r.mu_count; mu++) {
      const size_t base = (size_t(wavelength) * r.mu_count + mu) * r.phi_count;
      for (int j = 0; j < r.phi_count; j++) {
        const int a = j, b = r.phi_count - 1 - j;
        const int c = (r.phi_count / 2 - 1 - j + r.phi_count) % r.phi_count;
        const int d = (j + r.phi_count / 2) % r.phi_count;
        if (j != std::min({a, b, c, d})) continue;
        const float mean = float((double(table.values[base+a]) + table.values[base+b] +
                                  table.values[base+c] + table.values[base+d]) * 0.25);
        table.values[base+a] = table.values[base+b] =
            table.values[base+c] = table.values[base+d] = mean;
      }
    }
  }
}

float diffraction_albedo_thin_sheet_transmission_tint(
    const DiffractionAlbedoRequest &r, const float wavelength_nm)
{
  if (!r.transmission_is_spectral) return r.transmission_tint;
  const float3 bt709 = max(r.transmission_bt709, zero_float3());
  const float largest = max(bt709.x, max(bt709.y, bt709.z));
  const float scale = max(largest, 1.0f);
  const float3 bounded = saturate(bt709 / scale);
  return saturatef(bt709_to_spectral_transmission(
      bounded, dielectric_wavelength_um(wavelength_nm)) * scale);
}

bool diffraction_albedo_build_cpu(const DiffractionAlbedoRequest &r,
                                  DiffractionAlbedoTable &table,
                                  std::string &error,
                                  const std::function<bool()> &cancelled)
{
  if (!diffraction_albedo_validate_request(r, error)) return false;
  DiffractionAlbedoTable result;
  result.request = r;
  result.values.resize(size_t(r.wavelength_count) * direction_count(r));
  result.averages.resize(r.wavelength_count);
  for (int s = 0; s < r.wavelength_count; ++s) {
    if (cancelled && cancelled()) {
      error = "GGX diffraction albedo construction cancelled";
      return false;
    }
    const float medium_wavelength = node_wavelength(r, s) / r.medium_ior;
    const DiffractionReflection p{r.alpha_x,
                                  r.alpha_y,
                                  medium_wavelength / r.pitch_nm,
                                  r.depth_nm / medium_wavelength,
                                  r.duty};
    if (!std::isfinite(p.wavelength_over_pitch) || !std::isfinite(p.height_over_wavelength)) {
      error = "GGX diffraction albedo wavelength ratio is not finite";
      return false;
    }
    DiffractionThinSheetModel sheet{};
    if (r.thin_sheet) {
      const float wavelength = node_wavelength(r, s);
      const float n = dielectric_ior_at_wavelength(r.inside_ior, r.inv_abbe,
                                                  dielectric_wavelength_um(wavelength));
      sheet.reflection = {r.alpha_x, wavelength / r.pitch_nm,
                           2.0f * M_2PI_F * r.depth_nm / wavelength, r.duty, false};
      sheet.transmission = {bsdf_thin_glass_transmission_roughness(r.alpha_x, n),
                             wavelength / r.pitch_nm,
                             M_2PI_F * (n - 1.0f) * r.depth_nm / wavelength, r.duty, true};
      sheet.ior = n;
      sheet.reflection_tint = r.reflection_tint;
      sheet.transmission_tint = diffraction_albedo_thin_sheet_transmission_tint(
          r, wavelength);
      sheet.film_ior = r.film_ior;
      sheet.film_over_wavelength = r.film_thickness_nm / wavelength;
    }
    for (int i = 0; i < r.mu_count; ++i) {
      const float mu = node_mu(i, r.mu_count);
      const float radial = std::sqrt(std::max(0.0f, 1.0f - mu * mu));
      for (int j = 0; j < r.phi_count; ++j) {
        const float phi = M_2PI_F * (float(j) + 0.5f) / float(r.phi_count);
        const float3 wi = make_float3(radial * std::cos(phi), radial * std::sin(phi),
                                      std::max(mu, 1e-5f));
        bool was_cancelled = false;
        float value;
        if (r.thin_sheet) {
          if (cancelled && cancelled()) {
            error = "Thin-sheet diffraction cache construction cancelled";
            return false;
          }
          const float2 coefficients = diffraction_thin_sheet_model_coefficients(&sheet, wi.z);
          value = coefficients.x + coefficients.y -
                  diffraction_thin_sheet_model_escape(&sheet, wi, r.facet_samples);
        }
        else value = estimate(p, wi, r.facet_samples, cancelled, was_cancelled);
        if (was_cancelled) {
          error = "GGX diffraction albedo construction cancelled";
          return false;
        }
        if (!std::isfinite(value) || value < -1e-5f || value > 1.0f + 1e-5f) {
          error = "GGX diffraction albedo estimator produced an invalid value";
          return false;
        }
        result.values[size_t(s) * direction_count(r) + size_t(i * r.phi_count + j)] =
            std::clamp(value, 0.0f, 1.0f);
      }
    }
    result.averages[s] = float(projected_average(result.values.data() + size_t(s) * direction_count(r),
                                                  r.mu_count, r.phi_count));
  }
  if (r.thin_sheet) {
    diffraction_albedo_symmetrize_thin_sheet(result);
    for (int s = 0; s < r.wavelength_count; s++) {
      result.averages[s] = float(projected_average(
          result.values.data() + size_t(s) * direction_count(r), r.mu_count, r.phi_count));
    }
  }
  if (!diffraction_albedo_validate_table(result, error)) return false;
  table = std::move(result);
  return true;
}

float diffraction_albedo_lookup(const DiffractionAlbedoTable &t,
                                const float wavelength_nm,
                                const float3 &direction)
{
  if (!std::isfinite(wavelength_nm) || !std::isfinite(direction.x) ||
      !std::isfinite(direction.y) || !std::isfinite(direction.z) ||
      t.values.empty())
    return std::numeric_limits<float>::quiet_NaN();
  const float coordinate = spectral_coordinate(t.request, wavelength_nm);
  const int a = std::min(int(std::floor(coordinate)), t.request.wavelength_count - 1);
  const int b = std::min(a + 1, t.request.wavelength_count - 1);
  const float t_spec = coordinate - float(a);
  return (1.0f - t_spec) * angular_lookup(t, a, direction) +
         t_spec * angular_lookup(t, b, direction);
}

float diffraction_albedo_average(const DiffractionAlbedoTable &t, const float wavelength_nm)
{
  if (!std::isfinite(wavelength_nm) || t.averages.empty())
    return std::numeric_limits<float>::quiet_NaN();
  const float coordinate = spectral_coordinate(t.request, wavelength_nm);
  const int a = std::min(int(std::floor(coordinate)), t.request.wavelength_count - 1);
  const int b = std::min(a + 1, t.request.wavelength_count - 1);
  return (1.0f - (coordinate - float(a))) * t.averages[a] +
         (coordinate - float(a)) * t.averages[b];
}

namespace {

float two_sided_wavelength(const DiffractionTwoSidedAlbedoRequest &r, const int index)
{
  return 380.0f + 400.0f * float(index) / float(r.wavelength_count - 1);
}

size_t two_sided_direction_count(const DiffractionTwoSidedAlbedoRequest &r)
{
  return size_t(r.mu_count) * size_t(r.phi_count);
}

int two_sided_basis_count(const DiffractionTwoSidedAlbedoRequest &r)
{
  return r.generalized_f0_count == 0 ? 1 : r.generalized_f0_count;
}

float two_sided_ior(const DiffractionTwoSidedAlbedoRequest &r, const float lambda_nm)
{
  return dielectric_ior_at_wavelength(r.inside_ior, r.inv_abbe, dielectric_wavelength_um(lambda_nm));
}

size_t two_sided_index(const DiffractionTwoSidedAlbedoRequest &r,
                       const int side,
                       const int wavelength,
                       const int mu,
                       const int phi,
                       const int basis = 0)
{
  return (size_t((basis * 2 + side) * r.wavelength_count + wavelength) * two_sided_direction_count(r)) +
         size_t(mu * r.phi_count + phi);
}

float2 estimate_dielectric_escape(const DiffractionRoughDielectric &p,
                                  const float3 wi,
                                  const int samples,
                                  const float film_ior,
                                  const float film_thickness_over_wavelength,
                                  const float generalized_f0,
                                  const std::function<bool()> &cancelled,
                                  bool &was_cancelled)
{
  const float lambda_i = diffraction_dielectric_lambda<GGX>(p.alpha_x, p.alpha_y, wi);
  double reflected = 0.0, transmitted = 0.0;
  DiffractionDielectricGeneralizedExtra generalized{};
  generalized.base.param = p;
  generalized.generalized_f0 = make_spectrum(generalized_f0);
  generalized.generalized_reference_f0 = F0_from_ior(p.facet.transmitted_ior / p.facet.incident_ior);
  generalized.film_ior = film_ior;
  generalized.film_thickness_over_wavelength = film_thickness_over_wavelength;
  const bool matched = p.facet.incident_ior == p.facet.transmitted_ior;
  /* Native Principled uses physical coated Glass near matched indices. A film
   * keeps a direction-dependent straight atom; it is not the bare 1-F0 atom. */
  const bool physical = generalized_f0 < 0.0f ||
      (film_thickness_over_wavelength > 0.0f && generalized.generalized_reference_f0 <= 1e-5f);
  if (matched && physical && film_thickness_over_wavelength == 0.0f)
    return make_float2(0.0f, 1.0f);
  const float continuous_budget = !physical && matched ? generalized_f0 : 1.0f;
  /* At equal indices the generalized first event includes a straight-through
   * atom outside its continuous sampler. It has no masking loss. */
  if (matched && !physical) {
    transmitted = double(1.0f - generalized_f0) * double(samples);
  }
  for (int sample = 0; sample < samples; sample++) {
    if ((sample & 63) == 0 && cancelled && cancelled()) {
      was_cancelled = true;
      return zero_float2();
    }
    const float u = (float(sample) + 0.5f) / float(samples);
    const float v0 = (float(sample) + 0.5f) * golden_conjugate;
    const float w0 = (float(sample) + 0.5f) * 0.7548776662466927f;
    const float3 random = make_float3(u, v0 - std::floor(v0), w0 - std::floor(w0));
    float3 wo;
    float mass;
    bool singular;
    /* Each basis is monochromatic scalar power. The generalized proposal
     * then equals its power, so count-only escape is unbiased. */
    const bool sampled = !physical ?
        diffraction_dielectric_generalized_sample_direction<GGX>(
            &generalized, wi, random, &wo, &mass, &singular) :
        diffraction_dielectric_sample_direction<GGX>(
            &p, wi, random, &wo, &mass, &singular, false,
            film_ior, film_thickness_over_wavelength);
    if (!sampled)
    {
      continue;
    }
    const float lambda_o = diffraction_dielectric_lambda<GGX>(p.alpha_x, p.alpha_y, wo);
    const double escaped = singular && matched && wo.z < 0.0f ? 1.0 :
        double(continuous_budget * (1.0f + lambda_i) / (1.0f + lambda_i + lambda_o));
    if (wo.z > 0.0f) reflected += escaped;
    else transmitted += escaped;
  }
  return make_float2(float(reflected / double(samples)),
                     float(transmitted / double(samples)));
}

float diffuse_fresnel_cross_fraction(const float inside_ior,
                                    const float film_ior,
                                    const float film_thickness_over_wavelength,
                                    const float generalized_f0 = -1.0f)
{
  /* The air-side flux transmission integral equals the reverse conductance
   * after the n^2 etendue change of variables. It sets a physical, bounded
   * side-mixing strength, not a fitted image parameter. */
  constexpr int samples = 1024;
  double sum = 0.0;
  DiffractionDielectricGeneralizedExtra generalized{};
  generalized.generalized_f0 = make_spectrum(generalized_f0);
  generalized.generalized_reference_f0 = F0_from_ior(inside_ior);
  generalized.film_thickness_over_wavelength = film_thickness_over_wavelength;
  for (int i = 0; i < samples; i++) {
    const float mu = (float(i) + 0.5f) / float(samples);
    const float reflected = film_thickness_over_wavelength > 0.0f ?
                                diffraction_thin_film_reflectance(
                                    mu, 1.0f, inside_ior, film_ior,
                                    film_thickness_over_wavelength) :
                                fresnel_dielectric(mu, inside_ior, nullptr);
    const float power = generalized_f0 >= 0.0f ?
        average(diffraction_dielectric_generalized_fresnel(&generalized, reflected)) : reflected;
    sum += 2.0 * double(mu) * double(1.0f - power);
  }
  return std::clamp(float(sum / (double(samples) * std::min(1.0, double(inside_ior) * inside_ior))), 0.0f, 1.0f);
}

}  // namespace

bool diffraction_two_sided_albedo_validate_request(const DiffractionTwoSidedAlbedoRequest &r,
                                                   std::string &error)
{
  const auto finite = [](const float x) { return std::isfinite(x); };
  const double work = 2.0 * two_sided_basis_count(r) * r.wavelength_count * r.mu_count * r.phi_count * r.facet_samples;
  /* At the blue edge, the Airy round-trip period is approximately
   * lambda^2/(2*n_f*d). Require at least eight spectral nodes per shortest
   * period. The node and work budgets bound unusually thick/high-index films. */
  const double optical_thickness_nm = double(r.film_ior) * double(r.film_thickness_nm);
  const double minimum_film_wavelength_count = r.film_thickness_nm > 0.0f ?
      std::ceil(1.0 + 400.0 * 16.0 * optical_thickness_nm / (380.0 * 380.0)) : 2.0;
  if (!finite(r.alpha_x) || !finite(r.alpha_y) || r.alpha_x < 1e-4f ||
      r.alpha_y < 1e-4f || r.alpha_x > 1.0f || r.alpha_y > 1.0f ||
      !finite(r.pitch_nm) || r.pitch_nm <= 0.0f || !finite(r.depth_nm) ||
      r.depth_nm < 0.0f || !finite(r.duty) || r.duty < 0.0f || r.duty > 1.0f ||
      !finite(r.inside_ior) || r.inside_ior <= 1.0f || r.inside_ior > 10.0f ||
      !finite(r.inv_abbe) || r.inv_abbe < 0.0f ||
      (r.generalized_f0_count != 0 && r.generalized_f0_count != 2 && r.generalized_f0_count != 16) ||
      (r.generalized_f0_count == 2 && r.film_thickness_nm > 0.0f) ||
      (r.generalized_f0_count == 16 && r.film_thickness_nm <= 0.0f) ||
      !finite(r.film_ior) || r.film_ior < 1.0f || r.film_ior > 10.0f ||
      !finite(r.film_thickness_nm) || r.film_thickness_nm < 0.0f ||
      !std::isfinite(minimum_film_wavelength_count) ||
      r.wavelength_count < 2 || r.wavelength_count > 256 || r.mu_count < 2 ||
      r.mu_count > 64 || r.phi_count < 2 || r.phi_count > 128 ||
      r.facet_samples < 16 || r.facet_samples > 65536 || work > 128.0e6 ||
      r.wavelength_count < minimum_film_wavelength_count || r.algorithm_revision != 1)
  {
    error = "Invalid two-sided diffraction albedo request";
    return false;
  }
  for (float lambda_nm : {380.0f, 780.0f}) {
    const float n = two_sided_ior(r, lambda_nm);
    if (!finite(n) || n <= 0.0f || n > 10.0f) {
      error = "Invalid dispersed two-sided diffraction IOR";
      return false;
    }
  }
  error.clear();
  return true;
}

bool diffraction_two_sided_albedo_build_cpu(
    const DiffractionTwoSidedAlbedoRequest &r,
    DiffractionTwoSidedAlbedoTable &table,
    std::string &error,
    const std::function<bool()> &cancelled)
{
  if (!diffraction_two_sided_albedo_validate_request(r, error)) return false;
  DiffractionTwoSidedAlbedoTable next;
  next.request = r;
  const size_t directional_size = 2 * size_t(two_sided_basis_count(r)) * size_t(r.wavelength_count) * two_sided_direction_count(r);
  next.reflectance.resize(directional_size);
  next.transmittance.resize(directional_size);
  for (int basis = 0; basis < two_sided_basis_count(r); basis++) {
    for (int wavelength = 0; wavelength < r.wavelength_count; wavelength++) {
      const float lambda_nm = two_sided_wavelength(r, wavelength);
      const float film_ratio = r.film_thickness_nm / lambda_nm;
      for (int side = 0; side < 2; side++) {
        if (cancelled && cancelled()) {
          error = "Two-sided diffraction albedo construction cancelled";
          return false;
        }
        const float incident_ior = side == 0 ? 1.0f : two_sided_ior(r, lambda_nm);
        const float transmitted_ior = side == 0 ? two_sided_ior(r, lambda_nm) : 1.0f;
        DiffractionRoughDielectric p;
        if (!diffraction_dielectric_parameters(lambda_nm,
                                               r.pitch_nm,
                                               r.depth_nm,
                                               r.duty,
                                               incident_ior,
                                               transmitted_ior,
                                               r.alpha_x,
                                               r.alpha_y,
                                               &p))
        {
          error = "Two-sided diffraction albedo parameters are invalid";
          return false;
        }
        for (int mu_index = 0; mu_index < r.mu_count; mu_index++) {
          const float mu = node_mu(mu_index, r.mu_count);
          const float radial = std::sqrt(std::max(0.0f, 1.0f - mu * mu));
          for (int phi_index = 0; phi_index < r.phi_count; phi_index++) {
            const float phi = M_2PI_F * (float(phi_index) + 0.5f) / float(r.phi_count);
            const float3 wi = make_float3(radial * std::cos(phi),
                                          radial * std::sin(phi),
                                          std::max(mu, 1e-5f));
            bool was_cancelled = false;
            const float2 escaped = estimate_dielectric_escape(
                p, wi, r.facet_samples, r.film_ior, film_ratio,
                r.generalized_f0_count == 0 ? -1.0f : dielectric_f0_cache_node(r.inside_ior, r.generalized_f0_count, basis),
                cancelled, was_cancelled);
            if (was_cancelled) {
              error = "Two-sided diffraction albedo construction cancelled";
              return false;
            }
            const size_t index = two_sided_index(r, side, wavelength, mu_index, phi_index, basis);
            next.reflectance[index] = escaped.x;
            next.transmittance[index] = escaped.y;
          }
        }
      }
    }
  }
  if (!diffraction_two_sided_albedo_complete_from_directional(next, error)) return false;
  table = std::move(next);
  return true;
}

bool diffraction_two_sided_albedo_complete_from_directional(
    DiffractionTwoSidedAlbedoTable &table, std::string &error)
{
  const auto &r = table.request;
  if (!diffraction_two_sided_albedo_validate_request(r, error)) return false;
  const size_t directional_size = 2 * size_t(two_sided_basis_count(r)) * size_t(r.wavelength_count) * two_sided_direction_count(r);
  if (table.reflectance.size() != directional_size ||
      table.transmittance.size() != directional_size)
  {
    error = "Two-sided diffraction directional albedo dimensions are inconsistent";
    return false;
  }
  table.deficits.resize(directional_size);
  table.integrals.resize(2 * size_t(two_sided_basis_count(r)) * size_t(r.wavelength_count));
  table.cross_fractions.resize(two_sided_basis_count(r) * r.wavelength_count);
  for (size_t index = 0; index < directional_size; index++) {
    const float reflected = table.reflectance[index];
    const float transmitted = table.transmittance[index];
    const float raw_total = reflected + transmitted;
    if (!std::isfinite(reflected) || !std::isfinite(transmitted) ||
        reflected < -1e-6f || transmitted < -1e-6f || raw_total > 1.0f + 2e-6f)
    {
      error = "Two-sided diffraction escape estimator is invalid";
      return false;
    }
    table.reflectance[index] = std::clamp(reflected, 0.0f, 1.0f);
    table.transmittance[index] = std::clamp(transmitted, 0.0f, 1.0f);
    table.deficits[index] = std::max(0.0f, 1.0f -
        (table.reflectance[index] + table.transmittance[index]));
  }
  for (int basis = 0; basis < two_sided_basis_count(r); basis++) {
    for (int wavelength = 0; wavelength < r.wavelength_count; wavelength++) {
      const float lambda_nm = two_sided_wavelength(r, wavelength);
      table.cross_fractions[basis * r.wavelength_count + wavelength] = diffuse_fresnel_cross_fraction(
          two_sided_ior(r, lambda_nm), r.film_ior, r.film_thickness_nm / lambda_nm,
          r.generalized_f0_count == 0 ? -1.0f : dielectric_f0_cache_node(r.inside_ior, r.generalized_f0_count, basis));
      for (int side = 0; side < 2; side++) {
        const float incident_ior = side == 0 ? 1.0f : two_sided_ior(r, lambda_nm);
        const float *slice = table.deficits.data() +
                             two_sided_index(r, side, wavelength, 0, 0, basis);
        table.integrals[(basis * 2 + side) * r.wavelength_count + wavelength] =
            float(M_PI_F * incident_ior * incident_ior *
                  projected_average(slice, r.mu_count, r.phi_count));
      }
    }
  }
  return diffraction_two_sided_albedo_validate(table, error);
}

bool diffraction_two_sided_albedo_validate(const DiffractionTwoSidedAlbedoTable &table,
                                           std::string &error)
{
  const auto &r = table.request;
  if (!diffraction_two_sided_albedo_validate_request(r, error)) return false;
  if (table.deficits.size() != 2 * size_t(two_sided_basis_count(r)) * size_t(r.wavelength_count) * two_sided_direction_count(r) ||
      table.reflectance.size() != table.deficits.size() ||
      table.transmittance.size() != table.deficits.size() ||
      table.integrals.size() != 2 * size_t(two_sided_basis_count(r)) * size_t(r.wavelength_count) ||
      table.cross_fractions.size() != size_t(two_sided_basis_count(r)) * size_t(r.wavelength_count))
  {
    error = "Two-sided diffraction albedo dimensions are inconsistent";
    return false;
  }
  for (size_t i = 0; i < table.deficits.size(); i++) {
    const float q = table.deficits[i];
    const float reflected = table.reflectance[i], transmitted = table.transmittance[i];
    if (!std::isfinite(q + reflected + transmitted) || q < 0.0f || q > 1.0f ||
        reflected < 0.0f || transmitted < 0.0f || reflected > 1.0f || transmitted > 1.0f ||
        std::abs(double(q) + reflected + transmitted - 1.0) > 2.0e-6) {
      error = "Two-sided diffraction deficit is invalid";
      return false;
    }
  }
  for (int basis = 0; basis < two_sided_basis_count(r); basis++) {
    for (int wavelength = 0; wavelength < r.wavelength_count; wavelength++) {
      const float lambda_nm = two_sided_wavelength(r, wavelength);
      const float n = two_sided_ior(r, lambda_nm);
      const float cross = table.cross_fractions[basis * r.wavelength_count + wavelength];
      if (!std::isfinite(cross) || cross < 0.0f || cross > 1.0f) {
        error = "Two-sided diffraction cross fraction is invalid";
        return false;
      }
      for (int side = 0; side < 2; side++) {
        const float *slice = table.deficits.data() + two_sided_index(r, side, wavelength, 0, 0, basis);
        const double expected = M_PI_F * (side == 0 ? 1.0 : double(n) * n) *
                                projected_average(slice, r.mu_count, r.phi_count);
        const float observed = table.integrals[(basis * 2 + side) * r.wavelength_count + wavelength];
        if (!std::isfinite(observed) || observed < 0.0f ||
            std::abs(double(observed) - expected) > 1e-5 * std::max(1.0, expected))
        {
          error = "Two-sided diffraction deficit integral is invalid";
          return false;
        }
      }
    }
  }
  error.clear();
  return true;
}

float diffraction_two_sided_albedo_lookup(const DiffractionTwoSidedAlbedoTable &table,
                                          const int side,
                                          const float wavelength_nm,
                                          const float3 &direction,
                                          const int basis)
{
  if (basis < 0 || basis >= two_sided_basis_count(table.request) ||
      side < 0 || side > 1 || !std::isfinite(wavelength_nm) ||
      !std::isfinite(direction.x + direction.y + direction.z) || table.deficits.empty())
  {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const auto &r = table.request;
  const float position = std::clamp((wavelength_nm - 380.0f) / 400.0f, 0.0f, 1.0f) *
                         float(r.wavelength_count - 1);
  const int a = std::min(int(position), r.wavelength_count - 1);
  const int b = std::min(a + 1, r.wavelength_count - 1);
  const float t = position - float(a);
  const float mu = std::clamp(direction.z, 0.0f, 1.0f);
  const float mu_position = std::sqrt(mu) * float(r.mu_count - 1);
  const int i = std::min(int(mu_position), r.mu_count - 1);
  const int k = std::min(i + 1, r.mu_count - 1);
  const float mi = node_mu(i, r.mu_count), mk = node_mu(k, r.mu_count);
  const float u = i == k ? 0.0f : std::clamp((mu - mi) / (mk - mi), 0.0f, 1.0f);
  const float azimuth = (std::atan2(direction.y, direction.x) / M_2PI_F + 1.0f) *
                            float(r.phi_count) - 0.5f;
  const int j = (int(std::floor(azimuth)) % r.phi_count + r.phi_count) % r.phi_count;
  const int l = (j + 1) % r.phi_count;
  const float v = azimuth - std::floor(azimuth);
  const auto sample = [&](const int wavelength, const int mu, const int phi) {
    return table.deficits[two_sided_index(r, side, wavelength, mu, phi, basis)];
  };
  const auto angular = [&](const int wavelength) {
    return (1.0f - v) * ((1.0f - u) * sample(wavelength, i, j) +
                         u * sample(wavelength, k, j)) +
           v * ((1.0f - u) * sample(wavelength, i, l) +
                u * sample(wavelength, k, l));
  };
  return (1.0f - t) * angular(a) + t * angular(b);
}

CCL_NAMESPACE_END
