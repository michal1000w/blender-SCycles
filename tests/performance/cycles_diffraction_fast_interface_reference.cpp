/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction_interface.h"
#include "scene/diffraction.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
using namespace ccl;

/* Efficiency comparison, not a pass/fail accuracy certification. Two fixed
 * Maxwell truncations expose reference drift instead of calling N128 exact. */
int main()
{
  std::cout << std::setprecision(12) << "{\"scope\":\"Scalar approximation versus fixed N64/N128 Maxwell references; no modal convergence claim\",\"queries\":[";
  bool first = true;
  for (const double lower : {1.0, 1.5}) {
    const DiffractionGratingProfile profile{740, 150, double(float(.41)), 1, 1.5, 1, lower};
    for (const double wavelength : {450., 550., 650.}) {
      for (const double degrees : {0., 30., 60.}) {
        const double angle = degrees * M_PI / 180, azimuth = degrees ? M_PI / 4 : 0;
        DiffractionGratingResponse coarse, fine;
        std::string error;
        if (!diffraction_grating_solve(profile,wavelength,angle,azimuth,64,coarse,error) ||
            !diffraction_grating_solve(profile,wavelength,angle,azimuth,128,fine,error)) {
          std::cerr << error << '\n'; return 1;
        }
        using Powers = std::map<std::pair<int,bool>,double>;
        const auto powers = [](const DiffractionGratingResponse &response) {
          Powers result;
          for (const auto &order : response.orders) {
            result[{order.order,false}]=.5*(order.reflection[0]+order.reflection[1]);
            result[{order.order,true}]=.5*(order.substrate_flux[0]+order.substrate_flux[1]);
          }
          return result;
        };
        Powers low=powers(coarse), high=powers(fine), fast;
        const float R=float(sqr((1-lower)/(1+lower)));
        const FastDiffractionInterface p{float(wavelength),740,150,float(.41),1,float(lower),
                                        R,1-R,float(2*M_PI*150*.5/wavelength)};
        if (!fast_diffraction_interface_valid(p)) return 2;
        const float3 wi=make_float3(-sin(angle)*cos(azimuth),-sin(angle)*sin(azimuth),cos(angle));
        for (int side=0;side<2;++side) {
          const int bound=fast_diffraction_interface_order_bound(p,bool(side));
          for (int m=-bound;m<=bound;++m) {
            float3 wo;
            fast[{m,bool(side)}]=(!side && !m) ? fast_diffraction_interface_residual(p,wi) :
                fast_diffraction_interface_off_diagonal(p,wi,m,bool(side),&wo);
          }
        }
        /* The reference contains all propagating fast orders for this domain.
         * Also insert fast keys explicitly so no candidate power is omitted. */
        for (const auto &[key,value] : fast) high.try_emplace(key,0);
        double l1=0,drift=0,maximum=0,ft[2]={},rt[2]={};
        if(!first)std::cout<<','; first=false;
        std::cout<<"{\"substrate_ior\":"<<lower<<",\"wavelength_nm\":"<<wavelength
                 <<",\"angle_degrees\":"<<degrees<<",\"azimuth_degrees\":"<<(degrees?45:0)
                 <<",\"orders\":[";
        bool first_order=true;
        for (const auto &[key,value] : high) {
          const double approximation=fast[key];
          const double delta=std::abs(approximation-value);
          l1+=delta;maximum=std::max(maximum,delta);drift+=std::abs(value-low[key]);
          ft[key.second]+=approximation;rt[key.second]+=value;
          if(value==0 && low[key]==0 && approximation==0)continue;
          if(!first_order)std::cout<<',';first_order=false;
          std::cout<<"{\"order\":"<<key.first<<",\"transmission\":"<<(key.second?"true":"false")
                   <<",\"fast\":"<<approximation<<",\"n64\":"<<low[key]<<",\"n128\":"<<value<<'}';
        }
        std::cout<<"],\"l1_error\":"<<l1<<",\"maximum_order_error\":"<<maximum
                 <<",\"reference_l1_drift\":"<<drift<<",\"fast_reflection\":"<<ft[0]
                 <<",\"fast_transmission\":"<<ft[1]<<",\"reference_reflection\":"<<rt[0]
                 <<",\"reference_transmission\":"<<rt[1]<<'}'<<std::flush;
      }
    }
  }
  std::cout<<"]}\n";
}
