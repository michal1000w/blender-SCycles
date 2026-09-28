/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Test compressed Hermitian charts through the production two-polarization matcher. */
#include "scene/diffraction.h"
#include "kernel/util/diffraction_reference.h"
#include <iostream>
#include <iomanip>
#include <fstream>
using namespace ccl;
int main(int argc, char **argv) {
  if(argc!=1 && argc!=3 && argc!=4) return 2;
  std::ofstream fixture, expected_output;
  std::ifstream gpu;
  if(argc==4) { gpu.open(argv[3],std::ios::binary); if(!gpu) return 2; }
  if(argc>=3) { fixture.open(argv[1],std::ios::binary); expected_output.open(argv[2],std::ios::binary);
    if(!fixture || !expected_output) return 2; }
  using C=std::complex<double>;
  int count; double re,im;
  if (!(std::cin>>count) || count<1 || count>10000) return 2;
  float2 anchor[400];
  for(auto &v:anchor) {
    if(!(std::cin>>re>>im) || !std::isfinite(re) || !std::isfinite(im)) return 2;
    v=make_float2(re,im);
  }
  const DiffractionGratingProfile profile{740,150,.41,1,1.5,1,1};
  double maximum=0, conservation=0;
  int columns=0;
  for(int query=0;query<count;++query) {
    double b,y,w;
    if (!(std::cin>>b>>y>>w)) return 2;
    float2 chart[400];
    for(auto &v:chart) {
      if (!(std::cin>>re>>im) || !std::isfinite(re) || !std::isfinite(im)) return 2;
      /* Existing Cayley chart is Y=iH. */
      v=make_float2(-im,re);
    }
    DiffractionGratingBlock reference, exact;
    DiffractionGratingPowerBlock power;
    std::string error;
    if (!diffraction_grating_solve_reference(profile,w,b*w/740,y,16,2,reference,error) ||
        !diffraction_grating_match_reference(profile,w,b*w/740,y,reference,exact,error)) {
      std::cerr<<error;return 3;
    }
    diffraction_grating_power_block(exact,power);
    if(reference.ports.size()!=10) return 4;
    float2 boundary[40];
    for(int p=0;p<10;++p) {
      const double x=(b+reference.ports[p].order)*w/740;
      const double q2=1-x*x-y*y;
      const C z=q2>0?C(std::sqrt(q2)):C(0,std::sqrt(-q2));
      const C te=(1.0-z)/(1.0+z),tm=(z-1.0)/(z+1.0);
      boundary[4*p]=make_float2(te.real(),te.imag());
      boundary[4*p+1]=make_float2(tm.real(),tm.imag());
      float t=q2>0?2*std::sqrt(z.real())/(1+z.real()):0;
      boundary[4*p+2]=make_float2(t,t);
      const double length=std::hypot(x,y);
      boundary[4*p+3]=length?make_float2(x/length,y/length):make_float2(1,0);
    }
    auto find=[&](const DiffractionGratingPort &p) {
      for(int i=0;i<10;++i) if(reference.ports[i].order==p.order &&
                                reference.ports[i].substrate==p.substrate) return i;
      return -1;
    };
    for(size_t in=0;in<power.ports.size();++in) {
      float2 work[400],jones[40];
      std::copy(chart,chart+400,work);
      if(!diffraction_reference_match_anchor<20>(work,anchor,boundary,find(power.ports[in]),jones)) return 5;
      if(argc==4) { gpu.read(reinterpret_cast<char *>(jones),sizeof(jones)); if(!gpu) return 9; }
      if(argc>=3) {
        fixture.write(reinterpret_cast<const char *>(chart),sizeof(chart));
        fixture.write(reinterpret_cast<const char *>(anchor),sizeof(anchor));
        fixture.write(reinterpret_cast<const char *>(boundary),sizeof(boundary));
        const float input=float(find(power.ports[in]));
        fixture.write(reinterpret_cast<const char *>(&input),sizeof(input));
        expected_output.write(reinterpret_cast<const char *>(jones),sizeof(jones));
        if(!fixture || !expected_output) return 8;
      }
      double error_sum=0,sum=0;
      for(size_t out=0;out<power.ports.size();++out) {
        int p=find(power.ports[out]);
        if(p<0) return 6;
        double actual=0;
        for(int k=0;k<4;++k) actual+=.5*len_squared(jones[4*p+k]);
        double expected=power.matrix[out*power.ports.size()+in];
        error_sum+=std::abs(actual-expected);sum+=actual;
      }
      if(!std::isfinite(error_sum) || !std::isfinite(sum)) return 7;
      maximum=std::max(maximum,error_sum);conservation=std::max(conservation,std::abs(sum-1));
      ++columns;
    }
  }
  std::cout<<std::setprecision(12)<<"{\"queries\":"<<count<<",\"physical_columns\":"<<columns
    <<",\"maximum_power_error\":"<<maximum<<",\"maximum_conservation_error\":"<<conservation
    <<",\"scope\":\"Matrix-anchor matcher or supplied GPU amplitudes, local N16 domain\"}\n";
}
