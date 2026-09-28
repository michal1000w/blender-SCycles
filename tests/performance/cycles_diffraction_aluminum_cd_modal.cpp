/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
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
  std::cout<<std::setprecision(15)<<"{\"pitch_nm\":1600,\"depth_nm\":150,\"runs\":[";
  bool first=true, failed=false;
  for(double angle:{0.,.7,1.3})for(double azimuth:{0.,.6})
  for(int modes:{16,32,64,96})for(double wavelength:{400.,550.,700.}) {
    DiffractionGratingResponse result;std::string error;
    if(!diffraction_grating_solve(profile,wavelength,angle,azimuth,modes,result,error)) {
      if(!first)std::cout<<",";first=false;
      std::cout<<"{\"half_orders\":"<<modes<<",\"angle\":"<<angle<<",\"azimuth\":"<<azimuth
               <<",\"wavelength_nm\":"<<wavelength<<",\"error\":"<<std::quoted(error)<<"}"<<std::flush;
      failed=true; continue;
    }
    if(!first)std::cout<<",";first=false;
    std::cout<<"{\"half_orders\":"<<modes<<",\"angle\":"<<angle<<",\"azimuth\":"<<azimuth<<",\"wavelength_nm\":"<<wavelength<<",\"orders\":[";
    bool first_order=true;
    for(const auto &order:result.orders) {
      if(!first_order)std::cout<<",";first_order=false;
      std::cout<<"{\"order\":"<<order.order<<",\"R\":"<<.5*(order.reflection[0]+order.reflection[1])
               <<",\"T\":"<<.5*(order.substrate_flux[0]+order.substrate_flux[1])<<"}";
    }
    std::cout<<"],\"layer_absorption\":"<<.5*(result.layer_absorption[0]+result.layer_absorption[1])
             <<",\"boundary_residual\":"<<result.boundary_residual<<"}"<<std::flush;
  }
  std::cout<<"]}\n";
  return failed ? 1 : 0;
}
