/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Direct spectral reflectance of the saved relief_metal furnace profile.
 * Success means the solves completed; convergence must be assessed separately. */
#include "scene/diffraction.h"
#include "kernel/tables.h"
#include <iomanip>
#include <iostream>
using namespace ccl;
int main()
{
  const std::complex<double> metal(double(float(.9)), 6);
  const DiffractionGratingProfile profile{740,150,double(float(.41)),1,metal,1,metal};
  const double conversion[3][3]={{3.2404542,-1.5371385,-.4985314},
    {-.9692660,1.8760108,.0415560},{.0556434,-.2040259,1.0572252}};
  std::cout << std::setprecision(17)
            << "{\"scope\":\"Direct normal-incidence reflected power; absorbing substrate flux is not transmitted radiance.\","
               "\"color_space\":\"linear BT.709\",\"pitch_nm\":740,\"depth_nm\":150,\"duty\":"
            << profile.duty << ",\"metal_n\":" << metal.real() << ",\"metal_k\":6,\"runs\":[";
  bool first=true;
  for (int modes : {16,64,128,256}) {
    for (int subdivision : {4,8}) {
      /* Resolve quadrature independently at N128; retain the same quadrature
       * across all modal resolutions for the modal comparisons. */
      if (subdivision==8 && modes!=128) continue;
      double rgb[3]={},neutral[3]={}; int evaluations=0;
      for (int interval=0;interval<80;interval++) {
        for (int sample=0;sample<subdivision;sample++) {
          const double width=5.0/subdivision;
          const double wavelength=380+5*interval+width*(sample+.5);
          const double f=(wavelength-(380+5*interval))/5;
          double xyz[3],weight[3]={};
          for(int c=0;c<3;c++)
            xyz[c]=((1-f)*cie_color_match[interval][c]+f*cie_color_match[interval+1][c])*
              ((1-f)*cie_d65_spd[interval]+f*cie_d65_spd[interval+1])*
              CIE_D65_NORMALIZATION*width/400;
          for(int c=0;c<3;c++)for(int j=0;j<3;j++)weight[c]+=conversion[c][j]*xyz[j];
          DiffractionGratingResponse response;std::string error;
          if(!diffraction_grating_solve(profile,wavelength,0,0,modes,response,error)) {
            std::cerr<<"N="<<modes<<" wavelength="<<wavelength<<" "<<error<<"\n";
            return 1;
          }
          double reflected=0;
          for(const auto &order:response.orders)
            reflected+=.5*(order.reflection[0]+order.reflection[1]);
          for(int c=0;c<3;c++) {rgb[c]+=reflected*weight[c];neutral[c]+=weight[c];}
          evaluations++;
        }
      }
      if(!first)std::cout<<",";first=false;
      std::cout<<"{\"half_orders\":"<<modes<<",\"subdivision\":"<<subdivision
               <<",\"evaluations\":"<<evaluations<<",\"rgb\":[";
      for(int c=0;c<3;c++)std::cout<<(c?",":"")<<rgb[c];
      std::cout<<"],\"neutral_rgb\":[";
      for(int c=0;c<3;c++)std::cout<<(c?",":"")<<neutral[c];
      std::cout<<"]}"<<std::flush;
      std::cerr<<"Completed N="<<modes<<" subdivision="<<subdivision<<"\n"<<std::flush;
    }
  }
  std::cout<<"]}\n";
}
