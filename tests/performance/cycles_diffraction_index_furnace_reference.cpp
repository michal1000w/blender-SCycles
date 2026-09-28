/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Independent double-precision Fresnel integration. No packed lookup or
 * diffraction sampler is used in this reference. */
#include "scene/diffraction.h"
#include "kernel/tables.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace ccl;
int main(int argc, char **argv)
{
  if (argc != 2) return 2;
  std::ifstream input(argv[1]);
  std::string line;
  if (!std::getline(input,line)) return 2;
  if (!line.empty() && line.back()=='\r') line.pop_back();
  if (line != "wavelength_nm,n,k") return 2;
  std::vector<DiffractionIndexSample> samples;
  while (std::getline(input,line)) {
    if (line.empty()) continue;
    std::replace(line.begin(),line.end(),',',' ');
    std::istringstream row(line);
    double wavelength,n,k;
    if (!(row>>wavelength>>n>>k)) return 2;
    samples.push_back({wavelength,{n,k}});
  }
  if (samples.size()<2 || samples.front().wavelength>380 || samples.back().wavelength<780) return 2;
  const double conversion[3][3]={{3.2404542,-1.5371385,-.4985314},
    {-.9692660,1.8760108,.0415560},{.0556434,-.2040259,1.0572252}};
  std::cout<<std::setprecision(17)<<"{\"scope\":\"Spectral normal-incidence Fresnel furnace reference, linear BT.709; no diffraction-efficiency claim.\",\"runs\":[";
  bool first=true;
  for (int subdivision:{16,32,64}) {
    double rgb[3]={};
    for (int interval=0;interval<80;++interval) for(int sample=0;sample<subdivision;++sample) {
      const double width=5.0/subdivision;
      const double wavelength=380+5*interval+width*(sample+.5);
      const auto high=std::upper_bound(samples.begin(),samples.end(),wavelength,
        [](double w,const DiffractionIndexSample &s){return w<s.wavelength;});
      if (high==samples.begin() || high==samples.end()) return 2;
      const auto &lo=*(high-1), &hi=*high;
      const double t=(wavelength-lo.wavelength)/(hi.wavelength-lo.wavelength);
      const auto index=lo.index*(1-t)+hi.index*t;
      const double R=std::norm((index-1.0)/(index+1.0));
      const double f=(wavelength-(380+5*interval))/5;
      double xyz[3];
      for(int c=0;c<3;++c) {
        xyz[c]=((1-f)*cie_color_match[interval][c]+f*cie_color_match[interval+1][c])*
          ((1-f)*cie_d65_spd[interval]+f*cie_d65_spd[interval+1])*CIE_D65_NORMALIZATION*width/400;
      }
      for(int c=0;c<3;++c)for(int j=0;j<3;++j)rgb[c]+=R*conversion[c][j]*xyz[j];
    }
    if(!first)std::cout<<",";first=false;
    std::cout<<"{\"subdivision\":"<<subdivision<<",\"rgb\":["<<rgb[0]<<","<<rgb[1]<<","<<rgb[2]<<"]}";
  }
  std::cout<<"]}\n";
}
