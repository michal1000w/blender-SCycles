/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Reproduces the unresolved float-direction fold from Metal query 11456.
 * Diagnostic only: a negative discriminant is not an accepted result. */
#include "kernel/closure/bsdf_diffraction_util.h"
#include <cstdio>
using namespace ccl;
int main() {
 const float3 wi=make_float3(-.641338885f,-.458013684f,.615555048f);
 const float eta=.928189099f, delta=-.664625049f;
 const float3 h=normalize(make_float3(2*.2144555f-1,2*.210995913f-1,.1f+eta));
 const float3 axis=make_float3(1,0,0);
 float3 wo; diffraction_facet_transmit(wi,h,axis,eta,delta,&wo);
 const float3 v=eta*wi+wo;
 double vd[3]={double(eta)*wi.x+wo.x,double(eta)*wi.y+wo.y,double(eta)*wi.z+wo.z};
 const double exact=vd[0]*vd[0]+vd[1]*vd[1]+vd[2]*vd[2]-double(delta)*delta;
 const float a=eta*dot(wi,h)+dot(wo,h);
 printf("rounded_v_discriminant=%.17g exact_float_direction_discriminant=%.17g facet_a_squared=%.17g\n",double(len_squared(v)-delta*delta),exact,double(a)*a);
 printf("wi_norm_error=%.17g wo_norm_error=%.17g h_norm_error=%.17g\n",double(len_squared(wi))-1,double(len_squared(wo))-1,double(len_squared(h))-1);
}
