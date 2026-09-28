/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction_convergence.h"
#include <iomanip>
#include <iostream>
#include <cstdlib>
using namespace ccl;
int main(int argc, char **argv)
{
  /* Optional bounded diagnostic budget; the production solver still rejects
   * unsupported resolutions. This does not relax either convergence test. */
  int maximum_half_orders = 512;
  if (argc == 2) {
    char *end = nullptr;
    const long value = std::strtol(argv[1], &end, 10);
    if (!end || *end || value < 64 || value > 512) return 2;
    maximum_half_orders = int(value);
  }
  else if (argc != 1) return 2;
  const std::complex<double> metal(double(float(.9)),6);
  const DiffractionGratingProfile profile{740,150,double(float(.41)),1,metal,1,metal};
  DiffractionGratingCacheOptions cache;
  cache.bounds={{-.5,-1,380},{.5,1,780}};
  std::string error;
  const int retained=diffraction_grating_reference_order_bound(profile,cache,error);
  if(retained<0){std::cerr<<error;return 2;}
  DiffractionModalOptions options;
  options.power_tolerance=.00025;
  options.maximum_half_orders=maximum_half_orders;
  std::cout<<std::setprecision(17)<<"{\"scope\":\"Selected normal-incidence metal queries, complete physical power columns; empirical convergence, not a full-domain bound.\",\"retained_half_orders\":"<<retained<<",\"power_tolerance\":"<<options.power_tolerance<<",\"maximum_half_orders\":"<<options.maximum_half_orders<<",\"queries\":[";
  bool first=true,all_passed=true;
  for(double wavelength:{425.,475.,525.,575.,625.,675.,725.}) {
    DiffractionGratingBlock reference;
    std::vector<DiffractionModalObservation> observations;
    const bool passed=diffraction_grating_converged_reference(profile,wavelength,0,0,retained,
                       options,reference,observations,error);
    if(!passed && !reference.matrix.empty())return 3;
    all_passed &= passed;
    if(!first)std::cout<<",";first=false;
    std::cout<<"{\"wavelength_nm\":"<<wavelength<<",\"converged\":"<<(passed?"true":"false")<<",\"observations\":[";
    for(size_t i=0;i<observations.size();i++) {
      const auto &o=observations[i];
      std::cout<<(i?",":"")<<"{\"half_orders\":"<<o.half_orders<<",\"comparisons\":"<<o.comparisons<<",\"adjacent_power_difference\":"<<o.adjacent_power_difference<<",\"spanning_power_difference\":"<<o.spanning_power_difference
               <<",\"adjacent_complex_difference\":"<<o.adjacent_complex_difference
               <<",\"spanning_complex_difference\":"<<o.spanning_complex_difference<<"}";
    }
    std::cout<<"]}"<<std::flush;
    std::cerr<<"wavelength="<<wavelength<<" converged="<<passed;
    if(!passed)std::cerr<<" error="<<error;
    std::cerr<<"\n"<<std::flush;
  }
  std::cout<<"],\"all_converged\":"<<(all_passed?"true":"false")<<"}\n";
  return all_passed?0:1;
}
