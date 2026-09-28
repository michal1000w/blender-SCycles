/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace ccl;
int main(int argc, char **argv)
{
  if (argc != 2) return 2;
  DiffractionGratingProfile profile{1600,150,double(float(.41)),1,{1,1},1,{1,1}};
  std::ifstream input(argv[1]); std::string line;
  if (!std::getline(input,line) || line.find("wavelength_nm,n,k") != 0) return 2;
  while (std::getline(input,line)) {
    std::replace(line.begin(),line.end(),',',' ');
    std::istringstream row(line); double wavelength,n,k;
    if (!(row>>wavelength>>n>>k)) return 2;
    profile.ridge_spectrum.push_back({wavelength,{n,k}});
  }
  profile.absorbing_substrate_spectrum=profile.ridge_spectrum;
  std::cout<<std::setprecision(17)<<"{\"runs\":[";
  bool first=true;
  for (int modes : {32,64,128})
    for (double wavelength : {399.99,399.9999,399.999999,400.,400.000001,400.0001,400.01}) {
      DiffractionGratingBlock reference, physical;
      DiffractionGratingPowerBlock power;
      std::string error;
      if (!diffraction_grating_solve_reference(profile,wavelength,0,0,modes,5,reference,error) ||
          !diffraction_grating_match_reference(profile,wavelength,0,0,reference,physical,error)) {
        std::cerr<<error; return 1;
      }
      diffraction_grating_power_block(physical,power);
      int incoming=-1;
      for (int p=0;p<int(power.ports.size());p++)
        if (power.ports[p].order==0 && !power.ports[p].substrate) incoming=p;
      if (incoming<0) return 2;
      if (!first) std::cout<<',';
      first=false;
      std::cout<<"{\"half_orders\":"<<modes<<",\"wavelength_nm\":"<<wavelength<<",\"orders\":[";
      for (int p=0;p<int(power.ports.size());p++) {
        if (power.ports[p].substrate) return 2;
        if (p) std::cout<<',';
        std::cout<<"{\"order\":"<<power.ports[p].order<<",\"R\":"
                 <<power.matrix[p*power.ports.size()+incoming]<<'}';
      }
      DiffractionGratingResponse direct;
      if (diffraction_grating_solve(profile,wavelength,0,0,modes,direct,error)) {
        double difference=0;
        for (const auto &order : direct.orders) {
          double expected=0;
          for (int p=0;p<int(power.ports.size());p++)
            if (power.ports[p].order==order.order)
              expected=power.matrix[p*power.ports.size()+incoming];
          difference+=std::abs(expected-.5*(order.reflection[0]+order.reflection[1]));
        }
        if (!std::isfinite(difference) || difference>1e-8) return 3;
        std::cout<<"],\"direct_reflection_l1\":"<<difference<<"}"<<std::flush;
      }
      else {
        if (wavelength!=400 || error!="An external grazing order requires a limiting solution")
          return 4;
        std::cout<<"],\"direct_error\":"<<std::quoted(error)<<"}"<<std::flush;
      }
    }
  std::cout<<"]}\n";
}
