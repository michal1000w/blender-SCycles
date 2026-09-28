/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include <iostream>
#include <iomanip>
using namespace ccl;
int main()
{
  const DiffractionGratingProfile profile{740,150,double(float(.41)),1,1.5,1,1};
  std::cout<<std::setprecision(15)<<"{\"pitch_nm\":740,\"depth_nm\":150,\"runs\":[";
  bool first=true;
  for(double angle:{0.5,1.1,1.55})for(double azimuth:{0.,.6,1.2})
  for(int modes:{16,32,64,96})for(double wavelength:{380.,430.,480.,530.,580.,630.,680.,730.,780.}) {
    DiffractionGratingResponse result;std::string error;
    if(!diffraction_grating_solve(profile,wavelength,angle,azimuth,modes,result,error)) {
      std::cerr<<error;return 1;
    }
    if(!first)std::cout<<",";first=false;
    std::cout<<"{\"half_orders\":"<<modes<<",\"angle\":"<<angle<<",\"azimuth\":"<<azimuth<<",\"wavelength_nm\":"<<wavelength<<",\"orders\":[";
    bool first_order=true;
    for(const auto &order:result.orders) {
      if(!first_order)std::cout<<",";first_order=false;
      std::cout<<"{\"order\":"<<order.order<<",\"R\":"<<.5*(order.reflection[0]+order.reflection[1])
               <<",\"T\":"<<.5*(order.substrate_flux[0]+order.substrate_flux[1])<<"}";
    }
    std::cout<<"]}"<<std::flush;
  }
  std::cout<<"]}\n";
}
