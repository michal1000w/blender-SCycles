/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/diffraction_thin_film.h"
#include <complex>
#include <random>
#include <cstdio>
using namespace ccl;
/* Independent interface-amplitude Airy series, rather than the production
 * characteristic-matrix form. Complex film q supports frustrated TIR. */
static double reference(double ci,double ni,double no,double nf,double d) {
 const double kt2=ni*ni*(1-ci*ci),q02=ni*ni-kt2,q22=no*no-kt2;
 if(q02<=0||q22<=0)return 1;
 using C=std::complex<double>;
 const C q0=sqrt(q02),q1=std::sqrt(C(nf*nf-kt2,0)),q2=sqrt(q22);
 const C phase=std::exp(C(0,4*M_PI*d)*q1);
 double total=0;
 for(int p=0;p<2;++p) {
  const C y0=p?q0/(ni*ni):q0,y1=p?q1/(nf*nf):q1,y2=p?q2/(no*no):q2;
  const C r01=(y0-y1)/(y0+y1),r12=(y1-y2)/(y1+y2);
  total+=std::norm((r01+r12*phase)/(1.0+r01*r12*phase));
 }
 return total*.5;
}
int main() {
 std::mt19937 rng(619812);std::uniform_real_distribution<float> u(0,1);
 int failures=0,propagating=0,evanescent_film=0;double maximum=0;float reversal=0;
 for(int i=0;i<100000;++i) {
  const float ni=1+u(rng),no=1+u(rng),nf=.8f+1.5f*u(rng),ci=.05f+.95f*u(rng),d=10*u(rng);
  const double ct2=1-std::pow(double(ni)/no,2)*(1-double(ci)*ci);
  if(ct2<=.001) continue; // Separately test critical limits below.
  const float ct=float(sqrt(ct2));
  const float value=diffraction_thin_film_pair_reflectance(ci,ct,ni,no,nf,d);
  const float back=diffraction_thin_film_pair_reflectance(ct,ci,no,ni,nf,d);
  const double error=std::abs(value-reference(ci,ni,no,nf,d));
  maximum=std::max(maximum,error);reversal=std::max(reversal,std::abs(value-back));
  failures+=!std::isfinite(value)||value<0||value>1||error>2e-4||std::abs(value-back)>2e-6;
  ++propagating;evanescent_film+=nf*nf<ni*ni*(1-ci*ci);
 }
 /* Quarter-wave antireflection film, zero thickness, and opaque evanescent
  * film are independent analytic limits. */
 const float nf=sqrtf(1.5f);
 failures+=diffraction_thin_film_reflectance(1,1,1.5f,nf,.25f/nf)>1e-6f;
 failures+=fabsf(diffraction_thin_film_reflectance(1,1,1.5f,2,0)-.04f)>1e-6f;
 failures+=fabsf(diffraction_thin_film_reflectance(.5f,1.5f,1.5f,1,100000)-1)>1e-6f;
 /* Exactly q_f=0: ni=1.25, ci=.6, nf=1. The external media match. */
 const float critical=diffraction_thin_film_pair_reflectance(.6f,.6f,1.25f,1.25f,1,.4f);
 failures+=!isfinite_safe(critical)||critical<0||critical>1;
 printf("pairs=%d evanescent_film=%d max_reference_error=%.9g max_reverse_error=%g film_critical=%g failures=%d\n",propagating,evanescent_film,maximum,reversal,critical,failures);
 return failures!=0;
}
