/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Compare visible-GGX facet sampling with independent integration of the inverse
 * directional density. This checks geometry measure, not BSDF energy/efficiency.
 * Uses actual visible-GGX sampling and the production summed density.
 * Optional uniform-facet proposal shares the inverse geometry with the density,
 * so this complements (does not replace) independent uniform-direction tests.
 * Singular densities can have heavy tails: statistical agreement alone is not
 * proof of convergence, so report standard errors as well as discrepancies. */
#include "kernel/closure/bsdf_diffraction.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
using namespace ccl;
int main(int argc, char **argv)
{
  const int samples = argc > 1 ? std::atoi(argv[1]) : 2000000;
  if (samples < 10000 || samples > 100000000) return 2;
  const bool facet_proposal = argc > 2 && std::strcmp(argv[2], "facet") == 0;
  constexpr int bins = 8;
  std::mt19937 rng(192736);
  std::uniform_real_distribution<float> uniform(0, 1);
  auto sphere = [&]() {
    const float z = 2 * uniform(rng) - 1, phi = 2 * M_PI_F * uniform(rng);
    const float r = std::sqrt(std::max(0.f, 1-z*z));
    return make_float3(r*std::cos(phi), r*std::sin(phi), z);
  };
  auto bin = [](float3 v) { return (v.x >= 0) + 2*(v.y >= 0) + 4*(v.z >= 0); };
  int failures = 0;
  for (float cosine : {.3f, .9f}) {
  for (float eta : {2.f/3, 1.f, 1.5f}) {
    for (float delta : {-.8f, .3f, .9f}) {
      const float3 wi = make_float3(std::sqrt(1-cosine*cosine),0,cosine);
      constexpr float alpha_x=.2f, alpha_y=.55f;
      std::array<double,bins> forward{}, integral{}, squares{};
      int missing = 0;
      for (int i = 0; i < samples; ++i) {
        float3 out; float sampled_pdf;
        if (diffraction_transmission_ggx_sample_order(alpha_x,alpha_y,wi,
            make_float2(uniform(rng),uniform(rng)),eta,delta,&out,&sampled_pdf)) {
          forward[bin(out)]++;
          failures += !std::isfinite(sampled_pdf) || sampled_pdf <= 0;
        }
        float3 wo;
        double proposal_pdf = 1/(4*M_PI);
        if (facet_proposal) {
          float3 h = sphere();
          if (dot(wi,h)<0) h=-h;
          if (!diffraction_facet_transmit(wi,h,make_float3(1,0,0),eta,delta,&wo)) continue;
          DiffractionTransmissionRoot roots[2];
          const int count=diffraction_transmission_half_vectors(wi,wo,make_float3(1,0,0),eta,delta,roots);
          proposal_pdf=0;
          for (int r=0;r<count;++r)
            proposal_pdf+=diffraction_transmission_jacobian(wi,wo,roots[r].h,make_float3(1,0,0),eta,delta)/(2*M_PI);
          if (!(proposal_pdf>0)) { ++missing; continue; }
        }
        else wo=sphere();
        const double weight = diffraction_transmission_ggx_order_pdf(
            alpha_x,alpha_y,wi,wo,eta,delta)/proposal_pdf;
        if (!std::isfinite(weight)) { ++failures; continue; }
        integral[bin(wo)] += weight;
        squares[bin(wo)] += weight*weight;
      }
      double ptotal = 0, qtotal = 0, max_error = 0, max_se = 0;
      for (int b = 0; b < bins; ++b) {
        const double p = forward[b]/samples, q = integral[b]/samples;
        const double se = std::sqrt((squares[b]/samples-q*q+p*(1-p))/samples);
        const double error = std::abs(p-q);
        failures += error > 6*se + .0005;
        /* Reject uninformative estimates even if their error bars cover zero. */
        failures += se > .005;
        max_error = std::max(max_error,error); max_se = std::max(max_se,se);
        ptotal += p; qtotal += q;
      }
      printf("cosine=%.8g eta=%.8g delta=%.8g forward_mass=%.8g integrated_mass=%.8g max_bin_error=%.8g max_bin_se=%.8g missing_inverse=%d/%d\n",
             cosine,eta,delta,ptotal,qtotal,max_error,max_se,missing,samples);
      fflush(stdout);
    }
  }
  }
  printf("proposal=%s failures=%d\n",facet_proposal?"uniform_facet":"uniform_direction",failures);
  return failures != 0;
}
