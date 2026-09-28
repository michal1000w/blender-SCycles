/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction_util.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
using namespace ccl;

int main()
{
  std::mt19937 rng(43077);
  std::uniform_real_distribution<float> component(-2,2);
  int failures=0, naive_sign_losses=0;
  double max_scaled_error=0;
  constexpr double epsilon=std::numeric_limits<float>::epsilon();
  for(int i=0;i<200000;++i) {
    const float3 v=make_float3(component(rng),component(rng),component(rng));
    const double norm=double(v.x)*v.x+double(v.y)*v.y+double(v.z)*v.z;
    float kick=float(std::sqrt(norm));
    const float target=i%2?INFINITY:0;
    for(int j=0;j<i%4;++j) kick=std::nextafter(kick,target);
    const double reference=norm-double(kick)*kick;
    const float result=diffraction_fold_discriminant(v,kick);
    const float naive=len_squared(v)-kick*kick;
    /* Four rounded product residuals and sums: allow float-relative output
     * error plus eight epsilon-squared units of the uncancelled magnitude. */
    const double bound=2*epsilon*std::abs(reference)+8*epsilon*epsilon*norm;
    const double error=std::abs(double(result)-reference);
    max_scaled_error=std::max(max_scaled_error,error/norm);
    failures+=!std::isfinite(result) || error>bound;
    if(std::abs(reference)>bound) {
      failures+=(result>0)!=(reference>0) || result==0;
      naive_sign_losses+=(naive>0)!=(reference>0) || naive==0;
    }
  }
  std::printf("cases=200000 failures=%d naive_sign_losses=%d max_scaled_error=%.9g\n",
              failures,naive_sign_losses,max_scaled_error);
  return failures!=0;
}
