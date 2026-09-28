/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/light/polarization_math.h"
#include <complex>
#include <cmath>
#include <cstdio>
using namespace ccl;
using Z=std::complex<double>;
static Z z(float2 a){return {a.x,a.y};}
static PolarizationStokes stokes(Z a,Z b){return {{float(std::norm(a)+std::norm(b)),float(std::norm(a)-std::norm(b)),float(2*std::real(a*std::conj(b))),float(-2*std::imag(a*std::conj(b)))}};}
int main()
{
 int checks=0,bad=0;double maximum=0;
 auto check=[&](double a,double b,double eps=3e-6){checks++;maximum=std::max(maximum,std::abs(a-b));bad+=!std::isfinite(a)||std::abs(a-b)>eps;};
 for(int n=0;n<31;n++) {
   const float c=.15f+.022f*n;
   const auto fs=coherent_field_dielectric(1.5f,1,c,COHERENT_SCALAR_S);
   const auto fp=coherent_field_dielectric(1.5f,1,c,COHERENT_SCALAR_P);
   PolarizationJones j{};j.value[0][0]=fs.reflection;j.value[1][1]=fp.reflection;
   if(n&1)j=polarization_jones_product(polarization_linear_filter(.173f*n),j);
   const auto m=polarization_mueller_from_jones(j);
   const Z e0=std::cos(.17*n)*std::polar(1.0,.11*n),e1=std::sin(.17*n)*std::polar(1.0,-.23*n);
   const Z o0=z(j.value[0][0])*e0+z(j.value[0][1])*e1;
   const Z o1=z(j.value[1][0])*e0+z(j.value[1][1])*e1;
   const auto in=stokes(e0,e1),expected=stokes(o0,o1),observed=polarization_apply(m,in);
   for(int k=0;k<4;k++)check(observed.value[k],expected.value[k]);
   check(observed.value[0]<=1.00001f?1:0,1);
   // A detector is a covector: .5*Stokes(analyzer) corresponds to aa^dagger.
   const Z a0=std::cos(.29*n)*std::polar(1.0,-.37*n),a1=std::sin(.29*n)*std::polar(1.0,.07*n);
   auto analyzer=stokes(a0,a1);for(float &v:analyzer.value)v*=.5f;
   const auto backward=polarization_apply(m,analyzer,true);
   const double response=std::norm(std::conj(a0)*o0+std::conj(a1)*o1);
   check(polarization_contract(analyzer,observed),response);
   check(polarization_contract(backward,in),response);
   if(fs.total_internal_reflection&&!(n&1))check(observed.value[0],1);
 }
 // The sign of circular Stokes V under complex TIR is checked against direct fields.
 const auto rs=coherent_field_dielectric(1.5f,1,.5f,COHERENT_SCALAR_S).reflection;
 const auto rp=coherent_field_dielectric(1.5f,1,.5f,COHERENT_SCALAR_P).reflection;
 PolarizationJones tir{};tir.value[0][0]=rs;tir.value[1][1]=rp;
 const auto out=polarization_apply(polarization_mueller_from_jones(tir),{{1,0,1,0}});
 check(out.value[3],std::imag(z(rp)*std::conj(z(rs))));
 check(std::abs(out.value[3])>.1f?1:0,1);
 // Transparent identity preserves circular and linear components; diffuse destroys them.
 PolarizationMueller identity{};for(int i=0;i<4;i++)identity.value[i][i]=1;
 PolarizationStokes circular{{1,0,0,1}};
 const auto preserved=polarization_apply(identity,circular),diffuse=polarization_apply(polarization_depolarizer(.8f),circular);
 for(int k=0;k<4;k++)check(preserved.value[k],circular.value[k]);
 check(diffuse.value[0],.8f);for(int k=1;k<4;k++)check(diffuse.value[k],0);
 printf("native polarization independent checks=%d failures=%d max_error=%.9g\n",checks,bad,maximum);
 return bad?1:0;
}
