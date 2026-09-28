/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction_util.h"
#include <iostream>
#include <random>
using namespace ccl;
struct D {
  double x,y,z;
  D operator+(D b) const { return {x+b.x,y+b.y,z+b.z}; }
  D operator-(D b) const { return {x-b.x,y-b.y,z-b.z}; }
  D operator*(double b) const { return {x*b,y*b,z*b}; }
};
static double dotd(D a,D b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
static D crossd(D a,D b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
static D unit(D a) { return a*(1/std::sqrt(dotd(a,a))); }
static D convert(float3 a) { return {a.x,a.y,a.z}; }
static D oracle(D wi,D h,D axis,double eta,double delta) {
  wi=unit(wi); h=unit(h); axis=unit(axis);
  D e=unit(axis-h*dotd(axis,h));
  D t=e*delta-(wi-h*dotd(wi,h))*eta;
  return t-h*std::sqrt(1-dotd(t,t));
}
static float tested_jacobian(const float3 wi,const float3 wo,const float3 h,
                             const float3 axis,const float eta,const float delta)
{
#ifdef DIFFRACTION_TEST_ABSOLUTE_INDICES
  // Common scaling by two leaves ray geometry unchanged while exercising
  // the outgoing-index-squared factor in the absolute-index Jacobian.
  return diffraction_transmission_jacobian_indices(wi,wo,h,axis,2*eta,2,2*delta);
#else
  return diffraction_transmission_jacobian(wi,wo,h,axis,eta,delta);
#endif
}

int main()
{
  std::mt19937 rng(918213);
  std::uniform_real_distribution<float> uniform(-1, 1);
  auto direction = [&]() {
    return normalize(make_float3(uniform(rng), uniform(rng), uniform(rng)));
  };
  int failures = 0, transmitted = 0, recovered = 0, two_roots = 0, jacobians = 0;
  int unresolved_grazing_roots = 0, reverse_checks = 0;
  float max_roundtrip = 0, max_jacobian = 0;
  for (int i = 0; i < 200000; ++i) {
    const float3 wi = direction(), axis = direction();
    float3 h = direction();
    if (dot(wi, h) < 0) h = -h;
    const float eta = i % 3 == 0 ? 1.0f : i % 3 == 1 ? 1.5f : 1.0f / 1.5f;
    const float delta = i % 5 == 0 ? 0.0f : 1.5f * uniform(rng);
    float3 wo;
    if (!diffraction_facet_transmit(wi, h, axis, eta, delta, &wo)) continue;
    ++transmitted;
    failures += fabsf(len_squared(wo) - 1) > 2e-5f || dot(wo, h) >= 0;
    if (dot(wi,h) > .01f && -dot(wo,h) > .01f) {
      float3 reverse;
      const bool valid = diffraction_facet_transmit(wo, -h, axis, 1/eta, delta/eta, &reverse);
      failures += !valid || len(reverse-wi) > .0001f;
      ++reverse_checks;
    }
    if (eta == 1 && delta == 0) {
      failures += len(wo + wi) != 0;
      continue;
    }
    DiffractionTransmissionRoot roots[4];
#ifdef DIFFRACTION_TEST_ABSOLUTE_INDICES
    const int count = diffraction_transmission_half_vectors_momentum(
        wi,wo,axis,diffraction_symmetric_momentum(2*eta,wi,2,wo),2*delta,roots);
#else
    const int count = diffraction_transmission_half_vectors(wi, wo, axis, eta, delta, roots);
#endif
    failures += count > 2;
    two_roots += count == 2;
    float closest = 2;
    for (int j = 0; j < count; ++j) {
      closest = std::min(closest, len(roots[j].h - h));
      float3 mapped;
      if (!diffraction_facet_transmit(wi, roots[j].h, axis, eta, delta, &mapped)) {
        /* Float subtraction cannot resolve ct^2 below a few ulps.
         * Record these cutoff cases separately, never as regular-point passes. */
        if (std::abs(dot(wo, roots[j].h)) < .0005f) ++unresolved_grazing_roots;
        else ++failures;
        continue;
      }
      const float error = len(mapped - wo);
      max_roundtrip = std::max(max_roundtrip, error);
      failures += error > .002f;
    }
    /* Near a fold inverse recovery loses precision; test regular points. */
    const float a = eta * dot(wi, h) + dot(wo, h);
    if (fabsf(a) > .01f && len(cross(axis, h)) > .01f &&
        len(cross(axis, normalize(eta*wi+wo))) > .001f) {
      ++recovered;
      if (closest > .002f) std::cerr << "inverse closest=" << closest << " a=" << a << " delta=" << delta << " axial=" << len(cross(axis, normalize(eta*wi+wo))) << " count=" << count << "\n";
      failures += closest > .002f;
    }
    const float J = tested_jacobian(wi, wo, h, axis, eta, delta);
    if (J < .02f || J > 20 || dot(wi, h) < .1f || -dot(wo, h) < .1f ||
        len(cross(axis, h)) < .2f) continue;
    const float3 u = normalize(cross(h, fabsf(h.x) < .8f ? make_float3(1,0,0) : make_float3(0,1,0)));
    const float3 v = cross(h, u);
    constexpr double step = 1e-5;
    const D hd=unit(convert(h)), ud=unit(convert(u)), vd=unit(convert(v));
    const D up=oracle(convert(wi),hd+ud*step,convert(axis),eta,delta);
    const D um=oracle(convert(wi),hd-ud*step,convert(axis),eta,delta);
    const D vp=oracle(convert(wi),hd+vd*step,convert(axis),eta,delta);
    const D vm=oracle(convert(wi),hd-vd*step,convert(axis),eta,delta);
    const D area=crossd(up-um,vp-vm);
    const double numeric = 4*step*step / std::sqrt(dotd(area,area));
    const float relative = std::abs(numeric/J-1);
    if (relative>max_jacobian && relative>.001f) {
      std::cout.precision(10);
      std::cout << "jacobian_worst_fixture i=" << i << " eta=" << eta
                << " delta=" << delta << " wi=" << wi.x << ',' << wi.y << ',' << wi.z
                << " wo=" << wo.x << ',' << wo.y << ',' << wo.z
                << " h=" << h.x << ',' << h.y << ',' << h.z
                << " axis=" << axis.x << ',' << axis.y << ',' << axis.z
                << " J=" << J << " numeric=" << numeric << " relative=" << relative << '\n';
    }
    max_jacobian = std::max(max_jacobian, relative);
    if (relative > .015f) std::cerr << "jac eta=" << eta << " delta=" << delta << " a=" << a << " J=" << J << " numeric=" << numeric << "\n";
    failures += relative > .015f;
    ++jacobians;
  }
  int near_matched_checks = 0;
  double near_matched_max_relative = 0;
  for (int exponent = 2; exponent <= 24; ++exponent) {
    for (float sign : {-1.0f, 1.0f}) {
      const float eta = 1.0f + sign * std::ldexp(1.0f, -exponent);
      if (eta == 1.0f) continue;
      for (float ci : {.2f, .8f, 1.0f}) {
        const float3 wi = normalize(make_float3(std::sqrt(1-ci*ci), 0, ci));
        const float3 h = make_float3(0,0,1), axis = make_float3(1,0,0);
        float3 wo;
        const D wd = unit(convert(wi));
        const double ct2 = 1-double(eta)*eta*(1-wd.z*wd.z);
        if (ct2 <= 0) continue;
        if (!diffraction_facet_transmit(wi,h,axis,eta,0,&wo)) { ++failures; continue; }
        const double ct = std::sqrt(ct2);
        const double a = ((double(eta)-1)*(double(eta)+1))/(double(eta)*wd.z+ct);
        const double reference = ct/(a*a);
        const double actual = tested_jacobian(wi,wo,h,axis,eta,0);
        const double relative = std::abs(actual/reference-1);
        near_matched_max_relative = std::max(near_matched_max_relative,relative);
        failures += !std::isfinite(relative) || relative > 2e-5;
        ++near_matched_checks;
      }
    }
  }
  std::cout << "near_matched_checks=" << near_matched_checks
            << " near_matched_max_relative=" << near_matched_max_relative << '\n';
  std::cout << "transmitted=" << transmitted << " recovered=" << recovered
            << " reverse_checks=" << reverse_checks << " two_roots=" << two_roots << " jacobians=" << jacobians
            << " max_roundtrip=" << max_roundtrip << " max_jacobian_relative="
            << max_jacobian << " unresolved_grazing_roots=" << unresolved_grazing_roots << " failures=" << failures << '\n';
  return failures != 0 || recovered < 10000 || two_roots < 100 || jacobians < 10000;
}
