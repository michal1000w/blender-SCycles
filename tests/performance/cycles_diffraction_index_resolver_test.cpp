/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include <iostream>
using namespace ccl;
int main() {
  DiffractionGratingProfile p{740,30,.41,1,1.5,1,{.9,6}};
  p.ridge_spectrum={{500,{1.4,.1}},{600,{1.6,.3}}};
  p.groove_spectrum={{500,1},{600,1.2}};
  p.absorbing_substrate_spectrum={{500,{.8,5}},{600,{1,7}}};
  std::complex<double> r,g,s;std::string error;
  if(!diffraction_grating_sample_indices(p,550,r,g,s,error)||
     std::abs(r-std::complex<double>(1.5,.2))>1e-14||std::abs(g-1.1)>1e-14||
     std::abs(s-std::complex<double>(.9,6))>1e-14)return 1;
  for(double w:{500.,600.})if(!diffraction_grating_sample_indices(p,w,r,g,s,error))return 2;
  for(double w:{499.,601.,-1.}) {
    if(diffraction_grating_sample_indices(p,w,r,g,s,error)||r!=0.0||g!=0.0||s!=0.0)return 3;
  }
  p.absorbing_substrate_spectrum[1].index={1,0};
  if(diffraction_grating_sample_indices(p,550,r,g,s,error)||r!=0.0||g!=0.0||s!=0.0)return 4;
  p.absorbing_substrate_spectrum.clear();p.ridge_spectrum[1].wavelength=500;
  if(diffraction_grating_sample_indices(p,550,r,g,s,error))return 5;
  std::cout<<"Spectral interpolation, endpoints, no extrapolation, topology and malformed-table checks passed\n";
}
