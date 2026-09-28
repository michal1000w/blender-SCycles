/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include <cmath>
#include <cstdio>
using namespace ccl;
int main()
{
  int failures=0,negative_unbounded=0;
  double maximum_g1_error=0,maximum_lambda_scaled_error=0;
  /* Independent double-precision Beckmann integral. Sweep the full practical
   * slope range densely, including both sides of the rational-fit cutoff. */
  for(int i=0;i<=200000;++i) {
    const double slope=std::pow(10.0,-6.0+12.0*i/200000.0);
    const double a=1.0/slope;
    const double exact=-0.5*std::erfc(a)+std::exp(-a*a)/(2*a*std::sqrt(M_PI));
    const float3 w=normalize(make_float3(float(slope),0,1));
    const float lambda=diffraction_dielectric_lambda<BECKMANN>(1,1,w);
    negative_unbounded+=bsdf_aniso_lambda<BECKMANN>(1,1,w)<0;
    const double g=1.0/(1.0+lambda),reference=1.0/(1.0+exact);
    maximum_g1_error=std::max(maximum_g1_error,std::abs(g-reference));
    maximum_lambda_scaled_error=std::max(maximum_lambda_scaled_error,
                                        std::abs(lambda-exact)/std::max(1.0,exact));
    failures+=!std::isfinite(lambda)||lambda<0||g>1||std::abs(g-reference)>.005;
  }
  printf("cases=200001 negative_unbounded=%d max_g1_absolute_error=%.12g max_lambda_scaled_error=%.12g failures=%d\n",
         negative_unbounded,maximum_g1_error,maximum_lambda_scaled_error,failures);
  return failures!=0;
}
