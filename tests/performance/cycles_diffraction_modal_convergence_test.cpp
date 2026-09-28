/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction_convergence.h"
#include <iostream>
#include <limits>
using namespace ccl;
int main()
{
  DiffractionGratingProfile flat{740,0,.41,1,1.5,1,1.5};
  DiffractionModalOptions options;
  options.minimum_half_orders=4; options.maximum_half_orders=16;
  options.power_tolerance=1e-10; options.complex_tolerance=1e-9;
  DiffractionGratingBlock result;
  std::vector<DiffractionModalObservation> observations;
  std::string error;
  if (!diffraction_grating_converged_reference(flat,551,0,0,2,options,result,observations,error) ||
      result.matrix.empty() || observations.size()!=3) return 1;
  DiffractionGratingBlock physical;
  if (!diffraction_grating_match_reference(flat,551,0,0,result,physical,error)) return 2;
  DiffractionGratingPowerBlock power;
  diffraction_grating_power_block(physical,power);
  int incoming=-1;
  for (int p=0;p<int(power.ports.size());p++)
    if (power.ports[p].order==0 && !power.ports[p].substrate) incoming=p;
  if (incoming<0) return 3;
  for (int p=0;p<int(power.ports.size());p++) {
    const double expected=power.ports[p].order!=0?0:(power.ports[p].substrate?.96:.04);
    if (std::abs(power.matrix[p*power.ports.size()+incoming]-expected)>1e-12) return 4;
  }
  options.maximum_half_orders=8;
  if (diffraction_grating_converged_reference(flat,551,0,0,2,options,result,observations,error) ||
      !result.matrix.empty() || observations.size()!=2 || error.empty()) return 5;
  options.maximum_half_orders=16;
  options.progress=[](int n){return n<8;};
  if (diffraction_grating_converged_reference(flat,551,0,0,2,options,result,observations,error) ||
      !result.matrix.empty() || observations.size()!=1 || error.find("cancelled")==std::string::npos) return 6;
  options.progress={}; options.power_tolerance=std::numeric_limits<double>::quiet_NaN();
  if (diffraction_grating_converged_reference(flat,551,0,0,2,options,result,observations,error) ||
      !result.matrix.empty() || !observations.empty()) return 7;
  options={}; options.maximum_half_orders=64;
  DiffractionGratingProfile relief{740,150,.41,1,1.5,1,1};
  if (!diffraction_grating_converged_reference(relief,551,0,0,2,options,result,observations,error)) {
    std::cerr<<error; return 8;
  }
  const auto dielectric=observations.back();
  DiffractionGratingProfile metal{1600,150,.41,1,{.7,5},1,{.7,5}};
  if (diffraction_grating_converged_reference(metal,400,std::sin(1.3)-1,0,5,
                                              options,result,observations,error) ||
      !result.matrix.empty() || observations.size()!=3 || error.find("budget")==std::string::npos) return 9;
  std::cout<<"{\"passed\":true,\"dielectric_half_orders\":"<<dielectric.half_orders
           <<",\"dielectric_spanning_difference\":"<<dielectric.spanning_power_difference
           <<",\"metal_budget_adjacent_difference\":"<<observations.back().adjacent_power_difference
           <<",\"metal_budget_spanning_difference\":"<<observations.back().spanning_power_difference<<"}\n";
}
