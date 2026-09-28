/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "util/math.h"
CCL_NAMESPACE_BEGIN
ccl_device_inline float diffraction_film_fma(float a,float b,float c)
{
#ifdef __KERNEL_METAL__
  return metal::fma(a,b,c);
#else
  return fmaf(a,b,c);
#endif
}

/* Spectral Airy response of one lossless dielectric film. ci and ct are the
 * positive external cosines of a Snell pair. Keeping both endpoints makes the
 * zero-order coefficient symmetric under reversal, including rounded angles.
 * d_over_lambda is physical film thickness / vacuum wavelength.
 * This is an interface helper, not a model of a corrugated film boundary. */
ccl_device_inline float diffraction_thin_film_pair_reflectance(
    const float ci, const float ct, const float ni, const float no,
    const float nf, const float d_over_lambda)
{
  if (!(ci>0 && ct>0 && ni>0 && no>0 && nf>0 && d_over_lambda>=0) ||
      !isfinite_safe(ci+ct+ni+no+nf+d_over_lambda)) return 1.0f;
  const float qi=ni*ci, qo=no*ct;
  /* Difference of squares avoids subtracting two large transverse momenta
   * at the film critical angle. FMA retains the cancellation residual. */
  const float qfi2=diffraction_film_fma(qi,qi,(nf-ni)*(nf+ni));
  const float qfo2=diffraction_film_fma(qo,qo,(nf-no)*(nf+no));
  const float qf2=0.5f*(min(qfi2,qfo2)+max(qfi2,qfo2));
  const float k=M_2PI_F*d_over_lambda;
  float c,s;
  if (qf2>0) {
    const float qf=sqrtf(qf2),phase=k*qf;
    c=cosf(phase);
    s=phase==0 ? k : sinf(phase)/qf;
  }
  else if (qf2<0) {
    const float decay=sqrtf(-qf2);
    /* Divide the characteristic matrix by cosh(k*decay), which cancels
     * from the reflectance. No exponential overflow for thick films. */
    c=1;
    const float optical_decay=k*decay;
    /* tanh(10) rounds to 1 in float. Avoid evaluating fast GPU tanh in the
     * large-argument range, where intermediate exponentials can overflow. */
    s=(optical_decay>=10.0f ? 1.0f : tanhf(optical_decay))/decay;
  }
  else { c=1; s=k; } /* Continuous film-critical-angle limit. */
  float sum=0;
  for (int polarization=0;polarization<2;++polarization) {
    /* TE admittance q; reciprocal TM admittance q/n^2. Either choice of
     * reciprocal field convention has the same reflected power. */
    const float yi=polarization?qi/(ni*ni):qi;
    const float yo=polarization?qo/(no*no):qo;
    const float scale=polarization?nf*nf:1;
    const float a=yi*yo*scale*s,b=qf2*s/scale;
    const float numerator=sqr((yi-yo)*c)+sqr(a-b);
    const float denominator=sqr((yi+yo)*c)+sqr(a+b);
    if (!(denominator>0) || !isfinite_safe(denominator)) return 1;
    sum+=numerator/denominator;
  }
  return saturatef(.5f*sum);
}

ccl_device_inline float diffraction_thin_film_reflectance(
    const float ci, const float ni, const float no, const float nf,
    const float d_over_lambda)
{
  if (!(ci>0 && ni>0 && no>0)) return 1;
  const float ratio=ni/no;
  const float ct2=1-ratio*ratio*max(0.0f,1-ci*ci);
  if (!(ct2>0)) return 1; /* Evanescent exterior transports no transmitted flux. */
  return diffraction_thin_film_pair_reflectance(ci,sqrtf(ct2),ni,no,nf,d_over_lambda);
}
CCL_NAMESPACE_END
