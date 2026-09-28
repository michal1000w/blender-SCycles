/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction_convergence.h"
#include <iostream>
#include <iomanip>
using namespace ccl;
int main() {
  DiffractionGratingProfile profile{740,30,.41,1,{.9,6},1,{.9,6}};
  profile.ridge_spectrum={{590,{.8,5.8}},{600.5,{.91,6.01}},{610,{1,6.2}}};
  profile.absorbing_substrate_spectrum=profile.ridge_spectrum;
  profile.groove_spectrum={{590,1},{610,1.02}};
  DiffractionModalOptions options;options.minimum_half_orders=4;options.maximum_half_orders=256;options.power_tolerance=.001;
  DiffractionGratingBlock block;std::vector<DiffractionModalObservation> observations;std::string error;
  bool accepted=diffraction_grating_converged_reference(profile,600,-0.00079180743243243248,-0.0009765625,2,options,block,observations,error);
  std::cout<<std::setprecision(17)<<"accepted="<<accepted<<" error="<<error<<'\n';
  for(const auto &o:observations)std::cout<<o.half_orders<<" adjacent="<<o.adjacent_power_difference<<" spanning="<<o.spanning_power_difference<<'\n';
  return accepted?0:1;
}
