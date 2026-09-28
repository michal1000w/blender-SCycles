/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "diffraction_admittance_candidate.h"
#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <random>
using namespace ccl;
using C=std::complex<double>;
using M=Eigen::MatrixXcd;
template<int N> bool run(double &maximum) {
  std::mt19937 random(740+N);
  std::uniform_real_distribution<double> uniform(-1,1);
  for(int trial=0;trial<24;++trial) {
    M h(N,N),seed(N,N);
    for(int r=0;r<N;++r) for(int c=0;c<N;++c) {
      h(r,c)=C(uniform(random),uniform(random))*.15;
      seed(r,c)=C(uniform(random),uniform(random));
    }
    h=((h+h.adjoint())*.5).eval();
    M anchor=seed.householderQr().householderQ()*M::Identity(N,N);
    if(trial==0) anchor=M::Identity(N,N);
    const M y=C(0,1)*h;
    float2 chart[N*N],u[N*N],boundary[2*N],jones[2*N]; float4 exterior[N/2];
    M reflection=M::Zero(N,N),transmission=M::Zero(N,N);
    for(int r=0;r<N;++r) for(int c=0;c<N;++c) {
      chart[r*N+c]=make_float2(y(r,c).real(),y(r,c).imag());
      u[r*N+c]=make_float2(anchor(r,c).real(),anchor(r,c).imag());
    }
    for(int p=0;p<N/2;++p) {
      const double z=trial==1?.001:.1+.8*std::abs(uniform(random));
      const double index=1+.8*std::abs(uniform(random));
      const double q2=(trial==2 && p>0)?0:((trial==3 && p>0)?-z*z:z*z);
      exterior[p]=make_float4(.6,.8,q2,index);
      const C normal=std::sqrt(C(q2,0));
      const C rs=(C(1)-normal)/(C(1)+normal),rp=(normal-index*index)/(normal+index*index);
      const double ts=q2>0?2*std::sqrt(z)/(1+z):0,tp=q2>0?2*index*std::sqrt(z)/(z+index*index):0;
      for(int a=0;a<2;++a) for(int b=0;b<2;++b) {
        const double projection=(a?.8:.6)*(b?.8:.6);
        reflection(2*p+a,2*p+b)=(double(a==b)-projection)*rs+projection*rp;
        transmission(2*p+a,2*p+b)=(double(a==b)-projection)*ts+projection*tp;
      }
    }
    // Independent dense complex algebra with no production matching helper.
    M identity=M::Identity(N,N);
    M scattering=anchor*(identity-y)*(identity+y).inverse();
    M exact=transmission*scattering*(identity-reflection*scattering).inverse()*transmission-reflection;
    for(int p=0;p<N/2;++p) {
      if(exterior[p].z<=0)continue;
      if(!diffraction_admittance_candidate<N>(chart,u,exterior,p,jones)) return false;
      for(int r=0;r<N;++r) for(int c=0;c<2;++c) {
        const double error=std::abs(C(jones[2*r+c].x,jones[2*r+c].y)-(exterior[r/2].z>0?exact(r,2*p+c):C(0)));
        if(!std::isfinite(error)) return false;
        maximum=std::max(maximum,error);
      }
    }
    if(diffraction_admittance_candidate<N>(chart,u,exterior,-1,jones)) return false;
  }
  return maximum<2e-5;
}
int main() {
  double maximum=0;
  const bool passed=run<2>(maximum)&&run<4>(maximum)&&run<20>(maximum)&&run<36>(maximum);
  std::cout<<"{\"passed\":"<<(passed?"true":"false")<<",\"maximum_complex_error\":"<<maximum<<"}\n";
  return passed?0:1;
}
