/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include "kernel/tables.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <iomanip>
using namespace ccl;
int main()
{
  const DiffractionGratingProfile profile{740,150,double(float(.41)),1,1.5,1,1};
  const double conversion[3][3]={{3.2404542,-1.5371385,-.4985314},
    {-.9692660,1.8760108,.0415560},{.0556434,-.2040259,1.0572252}};
  std::vector<double> edges;
  for(int w=380;w<=780;w+=5)edges.push_back(w);
  edges.push_back(.9*740);std::sort(edges.begin(),edges.end());
  std::cout<<std::setprecision(14)<<"{\"color_space\":\"linear BT.709\",\"runs\":[";
  bool first=true;
  for(int modes:{16,64})for(int subdivision:{1,2,4,8}) {
    double rgb[2][3][3]={},neutral[3]={};int evaluations=0;
    for(size_t interval=1;interval<edges.size();interval++)for(int sample=0;sample<subdivision;sample++) {
      const double width=(edges[interval]-edges[interval-1])/subdivision;
      const double wavelength=edges[interval-1]+width*(sample+.5);
      const double coordinate=(wavelength-380)/5;const int i=int(coordinate);const double f=coordinate-i;
      double xyz[3],weight[3]={};
      for(int c=0;c<3;c++)xyz[c]=((1-f)*cie_color_match[i][c]+f*cie_color_match[i+1][c])*
        ((1-f)*cie_d65_spd[i]+f*cie_d65_spd[i+1])*CIE_D65_NORMALIZATION*width/400;
      for(int c=0;c<3;c++)for(int j=0;j<3;j++)weight[c]+=conversion[c][j]*xyz[j];
      for(int c=0;c<3;c++)neutral[c]+=weight[c];
      DiffractionGratingResponse response;std::string error;
      if(!diffraction_grating_solve(profile,wavelength,0,0,modes,response,error)){std::cerr<<error;return 1;}
      evaluations++;
      for(const auto &order:response.orders) {
        const double x=std::abs(order.order*wavelength/740);
        const int region=x<.25?0:(x>.9?2:1);
        for(int c=0;c<3;c++) {
          rgb[0][region][c]+=.5*(order.reflection[0]+order.reflection[1])*weight[c];
          rgb[1][region][c]+=.5*(order.substrate_flux[0]+order.substrate_flux[1])*weight[c];
        }
      }
    }
    if(!first)std::cout<<",";first=false;
    std::cout<<"{\"half_orders\":"<<modes<<",\"subdivision\":"<<subdivision<<",\"evaluations\":"<<evaluations<<",\"neutral_rgb\":[";
    for(int c=0;c<3;c++)std::cout<<(c?",":"")<<neutral[c];
    std::cout<<"],\"regions\":[";
    for(int side=0;side<2;side++)for(int region=0;region<3;region++) {
      if(side||region)std::cout<<",";
      std::cout<<"{\"side\":\""<<(side?"transmission":"reflection")<<"\",\"region\":\""
        <<(region==0?"central":region==1?"middle":"outer")<<"\",\"rgb\":[";
      for(int c=0;c<3;c++)std::cout<<(c?",":"")<<rgb[side][region][c];
      std::cout<<"]}";
    }
    std::cout<<"]}"<<std::flush;
  }
  std::cout<<"]}\n";
}
