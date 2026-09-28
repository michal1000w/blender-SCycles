/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include "scene/diffraction_convergence.h"
#include "kernel/util/diffraction_scene_data.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
using namespace ccl;
int main(int argc, char **argv)
{
  if (argc < 2) return 2;
  bool mirror = false, normal = false, float_duty = false, modal_reference = false;
  int queries = 4096, reference_half_orders = 16;
  unsigned seed = 913771;
  for (int i = 2; i < argc; i++) {
    if (std::string(argv[i]) == "--mirror") mirror = true;
    else if (std::string(argv[i]) == "--normal") normal = true;
    else if (std::string(argv[i]) == "--float-duty") float_duty = true;
    else if (std::string(argv[i]) == "--modal-reference") modal_reference = true;
    else if (std::string(argv[i]) == "--queries" && i + 1 < argc) {
      try { queries = std::stoi(argv[++i]); } catch (...) { return 2; }
    }
    else if (std::string(argv[i]) == "--reference-half-orders" && i + 1 < argc) {
      try { reference_half_orders = std::stoi(argv[++i]); } catch (...) { return 2; }
    }
    else if (std::string(argv[i]) == "--seed" && i + 1 < argc) {
      try { seed = std::stoul(argv[++i]); } catch (...) { return 2; }
    }
    else return 2;
  }
  if (queries < 1 || queries > 65536 || reference_half_orders < 2 ||
      reference_half_orders > 768 || (modal_reference && reference_half_orders < 64) ||
      (normal && queries % 2)) return 2;
  std::ifstream in(argv[1], std::ios::binary);
  DiffractionGratingDeviceBuffers b;
  auto read = [&](auto &v) {
    uint64_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!in || n > 10000000) return false;
    v.resize(n);
    in.read(reinterpret_cast<char *>(v.data()), n * sizeof(v[0]));
    return bool(in);
  };
  if (!read(b.nodes) || !read(b.layout) || !read(b.bounds) || !read(b.ports) ||
      !read(b.active) || !read(b.matrices) || in.peek() != EOF) return 2;
  int4 descriptors[2] = {make_int4(0, b.nodes.size(), 0, b.layout.size()/2), make_int4(0,0,0,mirror)};
  float4 domains[2] = {make_float4(-.5,-1,380,0), make_float4(mirror?0:.5,mirror?0:1,780,0)};
  DiffractionSceneData data{b.nodes.data(),b.layout.data(),b.bounds.data(),b.ports.data(),
    b.active.data(),b.matrices.data(),descriptors,domains,1};
  const DiffractionGratingProfile profile{740,150,float_duty?double(float(.41)):.41,1,1.5,1,1};
  std::mt19937 random(seed);
  std::uniform_real_distribution<float> unit(0,1);
  double maximum_error=0, maximum_conservation_error=0;
  int maximum_reference_orders_used=0;
  size_t reference_solves=0;
  float worst_power_wavelength=0, worst_conservation_wavelength=0;
  float3 worst_power_ray=make_float3(0,0,0);
  int worst_power_query=-1;
  for (int q=0;q<queries;q++) {
    float z=unit(random), phi=2*M_PI_F*unit(random), r=std::sqrt(1-z*z);
    bool below=q%2;
    float3 ray=make_float3(r*std::cos(phi),r*std::sin(phi),below?z:-z);
    float wavelength=380+400*unit(random), powers[10];
    if (normal) {
      ray=make_float3(0,0,below?1:-1);
      /* Both incident sides at each wavelength midpoint. */
      wavelength=380+400*(float(q/2)+.5f)/(queries/2);
    }
    DiffractionCacheCellView view; int order;
    if (!diffraction_data_power_column<20>(&data,0,ray,below,1,1,wavelength,740,&view,&order,powers)) {
      std::cerr<<"lookup failed "<<q;return 3;
    }
    DiffractionCacheCoordinates coordinates;
    diffraction_cache_coordinates(ray,1,wavelength,740,false,&coordinates);
    DiffractionGratingBlock block; DiffractionGratingPowerBlock reference; std::string error;
    const double kx=double(coordinates.query.x)*wavelength/740;
    if (modal_reference) {
      DiffractionModalOptions options;
      options.maximum_half_orders=reference_half_orders;
      options.power_tolerance=.00025;
      DiffractionGratingCacheOptions cache_options;
      cache_options.bounds={{-.5,-1,380},{.5,1,780}};
      const int retained=diffraction_grating_reference_order_bound(profile,cache_options,error);
      DiffractionGratingBlock reference_operator;
      std::vector<DiffractionModalObservation> observations;
      if (retained<0 || !diffraction_grating_converged_reference(profile,wavelength,kx,
          coordinates.query.y,retained,options,reference_operator,observations,error) ||
          !diffraction_grating_match_reference(profile,wavelength,kx,coordinates.query.y,
                                               reference_operator,block,error)) {
        std::cerr<<std::setprecision(17)<<"Reference convergence failed at query "<<q
                 <<" wavelength_nm="<<wavelength<<" kx="<<kx
                 <<" ky="<<coordinates.query.y<<": "<<error;
        for (const auto &observation : observations)
          std::cerr<<"; N="<<observation.half_orders
                   <<" adjacent="<<observation.adjacent_power_difference
                   <<" spanning="<<observation.spanning_power_difference;
        return 4;
      }
      reference_solves+=observations.size();
      maximum_reference_orders_used=std::max(maximum_reference_orders_used,
                                             observations.back().half_orders);
    }
    else {
      if (!diffraction_grating_solve_bloch(profile,wavelength,kx,coordinates.query.y,
                                         reference_half_orders,block,error)) {
        std::cerr<<error;return 4;
      }
      reference_solves++;
      maximum_reference_orders_used=reference_half_orders;
    }
    diffraction_grating_power_block(block,reference);
    int incoming=-1;
    for (int p=0;p<int(reference.ports.size());p++)
      if (reference.ports[p].order==coordinates.incoming_order && reference.ports[p].substrate==below) incoming=p;
    if(incoming<0)return 5;
    double l1=0,sum=0;
    for(int p=0;p<view.ports;p++) {
      int2 port=b.ports[view.port_offset+p]; double expected=0;
      for(int j=0;j<int(reference.ports.size());j++)
        if(reference.ports[j].order==(view.reverse_orders?-port.x:port.x) && reference.ports[j].substrate==bool(port.y))
          expected=reference.matrix[j*reference.ports.size()+incoming];
      if(!std::isfinite(powers[p]) || powers[p]<0)return 6;
      l1+=std::abs(powers[p]-expected);sum+=powers[p];
    }
    if (l1>maximum_error) {
      maximum_error=l1; worst_power_wavelength=wavelength;
      worst_power_query=q; worst_power_ray=ray;
    }
    if (std::abs(sum-1)>maximum_conservation_error) {
      maximum_conservation_error=std::abs(sum-1); worst_conservation_wavelength=wavelength;
    }
  }
  std::cout<<std::setprecision(17)<<"{\"queries\":"<<queries<<",\"seed\":"<<seed
    <<",\"modal_reference\":"<<(modal_reference?"true":"false")
    <<",\"modal_power_tolerance\":"<<(modal_reference?.00025:0)
    <<",\"reference_solves\":"<<reference_solves
    <<",\"maximum_reference_orders_used\":"<<maximum_reference_orders_used
    <<",\"reference_half_orders\":"<<reference_half_orders<<",\"duty_cycle\":"<<profile.duty
    <<",\"maximum_power_column_l1_error\":"<<maximum_error
    <<",\"maximum_conservation_error\":"<<maximum_conservation_error
    <<",\"normal_incidence\":"<<(normal?"true":"false")
    <<",\"worst_power_wavelength_nm\":"<<worst_power_wavelength
    <<",\"worst_power_query\":"<<worst_power_query
    <<",\"worst_power_ray\":["<<worst_power_ray.x<<","<<worst_power_ray.y<<","<<worst_power_ray.z<<"]"
    <<",\"worst_conservation_wavelength_nm\":"<<worst_conservation_wavelength<<"}\n";
  return maximum_error<=.001 && maximum_conservation_error<=.0001 ? 0 : 1;
}
