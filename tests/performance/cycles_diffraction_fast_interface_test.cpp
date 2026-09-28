/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction_interface.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
using namespace ccl;

int main()
{
  std::mt19937 rng(714908);
  std::uniform_real_distribution<float> unit(0, 1);
  int failures = 0, pairs = 0, transmissions = 0;
  double worst_power = 0, worst_direction = 0, minimum_residual = 1;
  for (int i = 0; i < 10000; ++i) {
    const float ni = 1 + 1.5f*unit(rng), nt = 1 + 1.5f*unit(rng);
    const float reflection = sqr((ni-nt)/(ni+nt));
    const float transmission = i%3 ? 1-reflection : 0;
    FastDiffractionInterface p{380+400*unit(rng), 400+2600*unit(rng),
                              1000*unit(rng), unit(rng), ni, nt,
                              reflection, transmission, 12*unit(rng)};
    failures += !fast_diffraction_interface_valid(p);
    const float z = .01f+.99f*unit(rng), phi = M_2PI_F*unit(rng);
    const float s = sqrtf(1-z*z);
    const float3 wi = make_float3(s*cosf(phi),s*sinf(phi),z);
    const float residual = fast_diffraction_interface_residual(p,wi);
    minimum_residual = std::min(minimum_residual,double(residual));
    failures += residual < -2e-6f;
    /* Specular reflection is its own reciprocal port, including all residual
     * power redirected from closed transmitted channels. */
    const float3 reflected = make_float3(-wi.x,-wi.y,wi.z);
    const double diagonal_error = std::abs(double(residual)-
        fast_diffraction_interface_residual(p,reflected));
    worst_power = std::max(worst_power,diagonal_error);
    failures += diagonal_error > 2e-6;
    for (int side = 0; side < 2; ++side) {
      const int bound = fast_diffraction_interface_order_bound(p,bool(side));
      for (int m=-bound;m<=bound;++m) {
        float3 wo;
        const float power = fast_diffraction_interface_off_diagonal(p,wi,m,bool(side),&wo);
        if (!(power>0)) continue;
        FastDiffractionInterface reverse=p;
        float3 reverse_wi=wo, expected=wi;
        if (side) {
          std::swap(reverse.incident_ior,reverse.transmitted_ior);
          reverse_wi=make_float3(wo.x,-wo.y,-wo.z);
          expected=make_float3(wi.x,-wi.y,-wi.z);
          ++transmissions;
        }
        float3 reverse_wo=zero_float3();
        const float reverse_power=fast_diffraction_interface_off_diagonal(
            reverse,reverse_wi,m,bool(side),&reverse_wo);
        const double error=std::abs(double(power)-reverse_power);
        const double direction=len(reverse_wo-expected);
        float evaluated_power=0,evaluated_probability=0;
        failures += !fast_diffraction_interface_probability(reverse,reverse_wi,expected,
                                                             &evaluated_power,&evaluated_probability);
        failures += std::abs(double(evaluated_power)-power)>2e-5;
        worst_power=std::max(worst_power,error);
        worst_direction=std::max(worst_direction,direction);
        failures += error>2e-5 || direction>1e-4;
        ++pairs;
      }
    }
  }
  FastDiffractionInterface p{550,740,0,.41f,1,1.5f,.04f,.96f,0};
  const float3 wi=make_float3(.6f,0,.8f);
  float3 wo;
  failures += std::abs(fast_diffraction_interface_off_diagonal(p,wi,0,true,&wo)-.96f)>1e-6;
  failures += len(wo-make_float3(-.4f,0,-sqrtf(.84f)))>1e-6;
  failures += std::abs(fast_diffraction_interface_residual(p,wi)-.04f)>1e-6;
  float queried_power=1,queried_probability=1;
  const float3 off_atom=normalize(wo+make_float3(0,.01f,0));
  failures += !fast_diffraction_interface_probability(p,wi,off_atom,&queried_power,&queried_probability);
  failures += queried_power!=0 || queried_probability!=0;
  failures += fast_diffraction_interface_probability(p,2*wi,wo,&queried_power,&queried_probability);
  p.incident_ior=1.5f;p.transmitted_ior=1;
  failures += std::abs(fast_diffraction_interface_residual(p,make_float3(.8f,0,.6f))-1)>1e-6;
  /* Compare stratified inverse-CDF frequencies with separately evaluated
   * powers. This checks the sampler, including its final residual branch. */
  p={550,1600,150,.41f,1,1.5f,.04f,.96f,2.4f};
  std::map<std::pair<int,bool>,int> counts;
  constexpr int samples=100000;
  for(int i=0;i<samples;++i) {
    int order;bool transmission;float power,probability;
    if(!fast_diffraction_interface_sample(p,wi,(i+.5f)/samples,&order,&transmission,
                                          &wo,&power,&probability)) {++failures;continue;}
    ++counts[{order,transmission}];
    failures += !(probability>0) || std::abs(probability-power)>1e-6;
  }
  double worst_frequency=0;
  for (int side=0;side<2;++side) {
    const int bound=fast_diffraction_interface_order_bound(p,bool(side));
    for(int m=-bound;m<=bound;++m) {
      const double power=(!side && m==0) ? fast_diffraction_interface_residual(p,wi) :
          fast_diffraction_interface_off_diagonal(p,wi,m,bool(side),&wo);
      const double error=std::abs(double(counts[{m,bool(side)}])/samples-power);
      worst_frequency=std::max(worst_frequency,error);
      failures += error>3.0/samples;
    }
  }
  std::cout<<"{\"failures\":"<<failures<<",\"reciprocal_pairs\":"<<pairs
           <<",\"transmission_pairs\":"<<transmissions
           <<",\"maximum_power_error\":"<<worst_power
           <<",\"maximum_direction_error\":"<<worst_direction
           <<",\"minimum_residual\":"<<minimum_residual
           <<",\"maximum_frequency_error\":"<<worst_frequency<<"}\n";
  return failures ? 1 : 0;
}
