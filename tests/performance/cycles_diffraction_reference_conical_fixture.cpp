/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <cstring>
using namespace ccl;
int main(int argc,char **argv)
{
  bool float_inputs=false;
  std::vector<int> orders{16,32};
  if(argc>1) {
    orders.clear();
    for(int i=1;i<argc;++i) {
      if(std::strcmp(argv[i],"--float-inputs")==0){float_inputs=true;continue;}
      char *end=nullptr;long n=std::strtol(argv[i],&end,10);
      if(!*argv[i]||*end||n<6||n>128)return 2;
      orders.push_back(int(n));
    }
  }
  if(orders.empty())orders={16,32};
  auto input=[&](double x){return float_inputs?double(float(x)):x;};
  std::cout<<std::setprecision(17)<<"[";
  bool first=true;
  for(int n:orders) for(int material=0;material<3;++material)
    for(int sample=0;sample<4;++sample) {
      const std::complex<double> index=material==0?std::complex<double>(input(1.7),0):std::complex<double>(input(.9),6);
      const double incident=material==2?1.5:1;
      const std::complex<double> groove(incident,0),substrate=material==0?std::complex<double>(input(1.3),0):index;
      const double pitch=900+sample*130,depth=50+sample*350,wavelength=420+sample*90,duty=input(.23+sample*.16);
      const double base=input(-.31+sample*.19),ky=input(-.45+sample*.27);
      const DiffractionGratingProfile profile{pitch,depth,duty,incident,index,groove,substrate};
      DiffractionGratingBlock block;std::string error;
      if(!diffraction_grating_solve_reference(profile,wavelength,base,ky,n,6,block,error)) {
        std::cerr<<error<<'\n'; return 1;
      }
      if(!first)std::cout<<',';first=false;
      std::cout<<"{\"half_orders\":"<<n<<",\"ridge\":["<<index.real()<<','<<index.imag()
               <<"],\"groove\":["<<groove.real()<<','<<groove.imag()<<"],\"substrate\":["<<substrate.real()<<','<<substrate.imag()
               <<"],\"incident_ior\":"<<incident<<",\"depth_nm\":"<<depth<<",\"ky\":"<<ky<<",\"pitch_nm\":"<<pitch<<",\"wavelength_nm\":"<<wavelength<<",\"duty\":"<<duty<<",\"kx\":"<<base
               <<",\"retained_half_orders\":6,\"boundary_residual\":"<<block.boundary_residual<<",\"ports\":[";
      for(size_t i=0;i<block.ports.size();++i) {
        if(i)std::cout<<',';
        std::cout<<'['<<block.ports[i].order<<','<<int(block.ports[i].substrate)<<']';
      }
      std::cout<<"],\"matrix\":[";
      for(size_t i=0;i<block.matrix.size();++i) {
        if(i)std::cout<<',';
        std::cout<<'['<<block.matrix[i].real()<<','<<block.matrix[i].imag()<<']';
      }
      std::cout<<"]}";
    }
  std::cout<<"]\n";
}
