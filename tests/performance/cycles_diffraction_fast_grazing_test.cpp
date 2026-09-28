/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction_interface.h"
#include <cstdio>
#include <initializer_list>
using namespace ccl;
int main()
{
  int failures=0;
  for (bool transmission : {false,true}) {
    const FastDiffractionInterface p{550,740,0,.5f,1,1,
                                      transmission?0.f:1.f,transmission?1.f:0.f,0};
    for(float z : {.3f,.1f,.01f,.001f,.0001f,.00001f}) {
      const float3 wi=make_float3(sqrtf(1-z*z),0,z);
      const float3 expected=make_float3(-wi.x,0,transmission?-wi.z:wi.z);
      float3 direction=zero_float3(),sampled=zero_float3();
      const bool valid=fast_diffraction_interface_direction(p,wi,0,transmission,&direction);
      int order=99;bool selected_transmission=false;float power=0,probability=0;
      const bool sampled_ok=fast_diffraction_interface_sample(p,wi,.5f,&order,
          &selected_transmission,&sampled,&power,&probability);
      const float error=len(direction-expected);
      float evaluated_power=0,evaluated_probability=0;
      const bool evaluated=fast_diffraction_interface_probability(p,wi,expected,
                                         &evaluated_power,&evaluated_probability);
      const bool pass=valid && sampled_ok && order==0 && selected_transmission==transmission &&
          len(sampled-expected)<=4e-6f && error<=4e-6f && power==1 && probability==1 &&
          evaluated && evaluated_power==1 && evaluated_probability==1;
      failures+=!pass;
      printf("transmission=%d cos_i=%.8g valid=%d selected_transmission=%d error=%.8g pass=%d\n",
             transmission,z,valid,selected_transmission,error,pass);
    }
  }
  printf("failures=%d\n",failures);
  return failures?1:0;
}
