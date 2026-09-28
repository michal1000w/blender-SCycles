/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

/* Host-only feasibility probe for a grating-specific missing-energy lobe.
 * No renderer closure uses this approximation yet. */
#include "kernel/closure/bsdf_diffraction.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstring>

using namespace ccl;

static float fract(const float x) { return x - std::floor(x); }

/* The native unit-reflector sampler proposes visible facets and diffraction
 * orders. Its f*cos/pdf ratio is G2/G1, including zero for below-surface
 * orders. This measures the actual grating-dependent single-bounce albedo. */
static float albedo(const DiffractionReflection &p, const float3 wi, const int samples,
                    const bool sum_orders = true)
{
  const float li = bsdf_aniso_lambda<GGX>(p.alpha_x, p.alpha_y, wi);
  double sum = 0;
  for (int s = 0; s < samples; ++s) {
    const float u = (s + .5f) / samples;
    const float v = fract((s + .5f) * .6180339887498949f);
    const float3 h = microfacet_ggx_sample_vndf(wi, p.alpha_x, p.alpha_y,
                                                 make_float2(u, v));
    if (!sum_orders) {
      float3 wo;
      float mass;
      const float t = fract((s + .5f) * .7548776662466927f);
      if (diffraction_reflection_sample_order(&p, wi, h, t, &wo, &mass) && wo.z > 0) {
        const float lo = bsdf_aniso_lambda<GGX>(p.alpha_x, p.alpha_y, wo);
        sum += (1 + li) / (1 + li + lo);
      }
      continue;
    }
    /* Integrate the discrete order choice exactly at each sampled facet.
     * This is the conditional expectation of the previous one-order estimator,
     * including its null events, without a third quadrature dimension. */
    const float ci = dot(wi, h);
    float nonzero_mass = 0;
    const int limit = diffraction_reflection_max_order(&p);
    for (int order = -limit; order <= limit; ++order) {
      float3 wo;
      if (order == 0 || !diffraction_facet_reflect(
              wi, h, make_float3(1, 0, 0), order * p.wavelength_over_pitch, &wo))
        continue;
      const float mass = diffraction_reflection_nonzero_power(&p, order, ci, dot(wo, h));
      nonzero_mass += mass;
      if (wo.z > 0) {
        const float lo = bsdf_aniso_lambda<GGX>(p.alpha_x, p.alpha_y, wo);
        sum += mass * (1 + li) / (1 + li + lo);
      }
    }
    const float3 wo = 2 * ci * h - wi;
    if (wo.z > 0) {
      const float lo = bsdf_aniso_lambda<GGX>(p.alpha_x, p.alpha_y, wo);
      sum += std::max(0.0f, 1 - nonzero_mass) * (1 + li) / (1 + li + lo);
    }
  }
  return sum / samples;
}

static float3 direction(const float u, const float v)
{
  const float mu = std::sqrt(u); /* Uniform in projected-solid-angle measure. */
  const float phi = M_2PI_F * v;
  const float radius = std::sqrt(std::max(0.0f, 1 - mu * mu));
  return make_float3(radius * std::cos(phi), radius * std::sin(phi), mu);
}

template<int MU, int PHI> struct Cache {
  float value[MU][PHI];
  float average = 0;
  static float node_mu(const int i)
  {
    const float coordinate = float(i) / (MU - 1);
    return coordinate * coordinate;
  }

  void build(const DiffractionReflection &p, const int samples)
  {
    average = 0;
    for (int i = 0; i < MU; ++i) for (int j = 0; j < PHI; ++j) {
      const float mu = std::max(0.00001f, node_mu(i));
      value[i][j] = albedo(p, direction(mu * mu, (j + .5f) / PHI), samples);
    }
    /* Exact projected integral of the piecewise-linear mu interpolation.
     * A plain node average would give the wrong rejection normalization. */
    double integral = 0;
    for (int i = 0; i < MU - 1; ++i) for (int j = 0; j < PHI; ++j) {
      const double mu = node_mu(i), h = node_mu(i + 1) - mu;
      const double a = value[i][j], delta = value[i + 1][j] - a;
      integral += 2 * h * (mu * a + (mu * delta + h * a) / 2 + h * delta / 3) / PHI;
    }
    average = float(integral);
  }

  float lookup(const float3 w) const
  {
    const float mu = std::clamp(w.z, 0.0f, 1.0f);
    const float u = std::sqrt(mu) * (MU - 1);
    const float v = (std::atan2(w.y, w.x) / M_2PI_F + 1) * PHI - .5f;
    const int i = std::clamp(int(std::floor(u)), 0, MU - 1);
    const int k = std::min(i + 1, MU - 1);
    const int j = (int(std::floor(v)) % PHI + PHI) % PHI;
    const int l = (j + 1) % PHI;
    const float a = k == i ? 0.0f :
        std::clamp((mu - node_mu(i)) / (node_mu(k) - node_mu(i)), 0.0f, 1.0f);
    const float b = fract(v);
    return ((1 - a) * value[i][j] + a * value[k][j]) * (1 - b) +
           ((1 - a) * value[i][l] + a * value[k][l]) * b;
  }
};

template<int MU, int PHI>
static float multiscatter_value(const Cache<MU, PHI> &cache, const float3 wi,
                                const float3 wo)
{
  const float missing = 1 - cache.average;
  if (missing <= 0) return 0;
  return (1 - cache.lookup(wi)) * (1 - cache.lookup(wo)) *
         wo.z / (M_PI_F * missing);
}

/* Accepted density after at most K cosine-proposal rejection draws. The
 * omitted mass is a legitimate null event, as in native order sampling. */
template<int MU, int PHI>
static float capped_proposal_pdf(const Cache<MU, PHI> &cache, const float3 wo,
                                 const int attempts)
{
  const float e = cache.average;
  const float multiplier = e < 1 ? (1 - std::pow(e, attempts)) / (1 - e) : attempts;
  return (1 - cache.lookup(wo)) * wo.z / M_PI_F * multiplier;
}

template<int MU, int PHI, int BUILD_SAMPLES>
static int probe(const float alpha_x = .25f, const float alpha_y = .4f,
                 const float wavelength_over_pitch = .45f, const float duty = .42f)
{
  constexpr int REF_SAMPLES = 65536;
  int failures = 0;
  for (const float depth : {0.0f, .4f}) {
    DiffractionReflection p{alpha_x, alpha_y, wavelength_over_pitch, depth, duty};
    Cache<MU, PHI> cache;
    const auto start = std::chrono::steady_clock::now();
    cache.build(p, BUILD_SAMPLES);
    const double milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    float max_error = 0, max_furnace = 0;
    float worst_mu = 0;
    /* Shifted off-node directions plus both boundary strips. Testing at the
     * 8x16 cache centers alone conceals interpolation and grazing clamp error. */
    for (int i = 0; i < 10; ++i) for (int j = 0; j < 17; ++j) {
      const float u = i == 0 ? 0.0001f : (i == 9 ? 0.9999f : (i - .73f) / 8);
      const float3 w = direction(u, (j + .31f) / 17);
      const float exact = albedo(p, w, REF_SAMPLES, false);
      const float estimate = cache.lookup(w);
      if (std::fabs(exact - estimate) > max_error) {
        max_error = std::fabs(exact - estimate);
        worst_mu = w.z;
      }
      /* A factorized lobe normalized by the cache's projected average has
       * integral 1-E_cache(w); the native SS part integrates to E_exact(w). */
      max_furnace = std::max(max_furnace, exact - estimate);
      failures += !std::isfinite(estimate) || estimate < 0 || estimate > 1;
    }
    /* The f*cos convention gives f_ms(i,o)=f_ms(o,i), and the capped
     * rejection proposal integrates to 1-Ebar^K including its null mass. */
    constexpr int ATTEMPTS = 4;
    double integrated_pdf = 0;
    for (int i = 0; i < 64; ++i) for (int j = 0; j < 128; ++j) {
      const float3 wi = direction((i + .5f) / 64, (j + .5f) / 128);
      const float3 wo = direction(((i * 13) % 64 + .5f) / 64,
                                  ((j * 43) % 128 + .5f) / 128);
      const float forward = multiscatter_value(cache, wi, wo) / wo.z;
      const float reverse = multiscatter_value(cache, wo, wi) / wi.z;
      failures += std::fabs(forward - reverse) > 2e-6f;
    }
    /* Resolve the grazing boundary using a uniform solid-angle quadrature,
     * independent of the cache's analytic projected-measure integral. */
    for (int i = 0; i < 512; ++i) for (int j = 0; j < 128; ++j) {
      const float mu = (i + .5f) / 512;
      const float3 w = direction(mu * mu, (j + .5f) / 128);
      integrated_pdf += capped_proposal_pdf(cache, w, ATTEMPTS) * M_2PI_F / (512 * 128);
    }
    failures += std::fabs(integrated_pdf - (1 - std::pow(cache.average, ATTEMPTS))) > 1e-4f;
    std::printf("alpha=%g,%g wavelength/pitch=%g duty=%g depth/lambda=%g "
                "cache=%dx%d/%d samples build_ms=%.1f "
                "max_albedo_error=%.4f max_furnace_excess=%.4f pdf_mass=%.4f "
                "worst_mu=%.4f failures=%d\n",
                alpha_x, alpha_y, wavelength_over_pitch, duty,
                depth, MU, PHI, BUILD_SAMPLES, milliseconds, max_error, max_furnace,
                integrated_pdf, worst_mu, failures);
  }
  return failures ? 1 : 0;
}

int main(int argc, const char **argv)
{
  if (argc == 2 && std::strcmp(argv[1], "--profiles") == 0) {
    int failures = 0;
    for (const auto &p : {DiffractionReflection{.05f, .05f, .45f, .4f, .42f},
                          DiffractionReflection{.8f, .8f, .45f, .4f, .42f},
                          DiffractionReflection{.08f, .7f, .45f, .4f, .42f},
                          DiffractionReflection{.5f, .12f, .2f, .4f, .15f},
                          DiffractionReflection{.25f, .4f, .85f, .4f, .8f}}) {
      failures |= probe<8, 16, 512>(p.alpha_x, p.alpha_y, p.wavelength_over_pitch, p.duty);
    }
    return failures;
  }
  if (argc != 1) return 2;
  /* This sweep measures approximation error separately from algebraic PDF and
   * reciprocity checks. A zero exit code does not certify furnace passivity:
   * the reported furnace excess must also be reviewed before integration. */
  return probe<4, 8, 128>() | probe<8, 16, 512>() | probe<16, 32, 1024>() |
         probe<32, 64, 2048>();
}
