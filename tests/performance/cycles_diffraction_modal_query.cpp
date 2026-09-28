/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include <iomanip>
#include <iostream>
using namespace ccl;
int main(int argc, char **argv)
{
  if (argc!=4) return 2;
  double coordinates[3];
  for (int i=0;i<3;i++) {
    try {
      size_t end=0;
      coordinates[i]=std::stod(argv[i+1],&end);
      if (end!=std::string(argv[i+1]).size() || !std::isfinite(coordinates[i])) return 2;
    }
    catch (...) {return 2;}
  }
  const DiffractionGratingProfile profile{740,150,double(float(.41)),1,1.5,1,1};
  std::cout<<std::setprecision(17)<<"{\"wavelength_nm\":"<<coordinates[0]
           <<",\"kx\":"<<coordinates[1]<<",\"ky\":"<<coordinates[2]<<",\"runs\":[";
  bool first=true, failed=false;
  for (int modes : {16,32,64,128,256,512}) {
    DiffractionGratingBlock reference, physical;
    std::string error;
    const bool success=diffraction_grating_solve_reference(profile,coordinates[0],coordinates[1],
        coordinates[2],modes,2,reference,error) &&
      diffraction_grating_match_reference(profile,coordinates[0],coordinates[1],coordinates[2],
                                            reference,physical,error);
    if (!first) std::cout<<',';
    first=false;
    std::cout<<"{\"half_orders\":"<<modes;
    if (!success) {
      failed=true;std::cout<<",\"error\":"<<std::quoted(error)<<"}"<<std::flush;continue;
    }
    std::cout<<",\"reference_boundary_residual\":"<<reference.boundary_residual
             <<",\"physical_boundary_residual\":"<<physical.boundary_residual
             <<",\"minimum_power_gain\":"<<physical.minimum_power_gain
             <<",\"maximum_power_gain\":"<<physical.maximum_power_gain<<",\"ports\":[";
    for (size_t i=0;i<physical.ports.size();i++) {
      if (i) std::cout<<',';
      std::cout<<'['<<physical.ports[i].order<<','<<(physical.ports[i].substrate?1:0)<<']';
    }
    std::cout<<"],\"matrix\":[";
    for (size_t i=0;i<physical.matrix.size();i++) {
      if (i) std::cout<<',';
      std::cout<<'['<<physical.matrix[i].real()<<','<<physical.matrix[i].imag()<<']';
    }
    std::cout<<"]}"<<std::flush;
  }
  std::cout<<"]}\n";
  return failed?1:0;
}
