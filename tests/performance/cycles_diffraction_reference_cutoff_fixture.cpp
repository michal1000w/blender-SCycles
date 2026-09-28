/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include <iostream>
#include <iomanip>
using namespace ccl;
int main()
{
  std::cout<<std::setprecision(17)<<"[";
  bool first=true;
  for(int n:{16,32}) for(bool metal:{false,true})
    for(double cutoff:{0.42,0.34}) for(double offset:{-1e-6,0.0,1e-6}) {
      const std::complex<double> index=metal?std::complex<double>(.9,6):std::complex<double>(1.5,0);
      const DiffractionGratingProfile profile{1000,150,.41,1,index,1,index};
      DiffractionGratingBlock block;std::string error;
      const double base=cutoff+offset;
      if(!diffraction_grating_solve_reference(profile,580,base,0,n,3,block,error)) {
        std::cerr<<error<<'\n'; return 1;
      }
      if(!first)std::cout<<',';first=false;
      std::cout<<"{\"half_orders\":"<<n<<",\"ridge\":["<<index.real()<<','<<index.imag()
               <<"],\"incident_ior\":1,\"depth_nm\":150,\"ky\":0,\"pitch_nm\":1000,\"wavelength_nm\":580,\"duty\":0.41,\"kx\":"<<base
               <<",\"retained_half_orders\":3,\"boundary_residual\":"<<block.boundary_residual<<",\"ports\":[";
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
