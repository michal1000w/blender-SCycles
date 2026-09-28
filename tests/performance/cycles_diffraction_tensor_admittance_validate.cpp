/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include "kernel/util/diffraction_scene_data.h"
#include <fstream>
#include <iostream>
#include <random>
#include "diffraction_admittance_candidate.h"
using namespace ccl;
int main(int argc, char **argv)
{
  if (argc != 2 && argc != 3 && argc != 4) return 2;
  std::ofstream fixture,expected;
  std::ifstream gpu;
  if(argc==3)gpu.open(argv[2],std::ios::binary);
  if(argc==4){fixture.open(argv[2],std::ios::binary);expected.open(argv[3],std::ios::binary);}
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
  int4 descriptors[2] = {make_int4(0, b.nodes.size(), 0, b.layout.size()/2), make_int4(0,0,0,0)};
  float4 domains[2] = {make_float4(-.5,-1,380,0), make_float4(.5,1,780,0)};
  DiffractionSceneData data{b.nodes.data(),b.layout.data(),b.bounds.data(),b.ports.data(),
    b.active.data(),b.matrices.data(),descriptors,domains,1};
  const DiffractionGratingProfile profile{740,150,.41,1,1.5,1,1};
  std::mt19937 random(913771);
  std::uniform_real_distribution<float> unit(0,1);
  double maximum_error=0, maximum_conservation_error=0;
  for (int q=0;q<4096;q++) {
    float z=unit(random), phi=2*M_PI_F*unit(random), r=std::sqrt(1-z*z);
    bool below=q%2;
    float3 ray=make_float3(r*std::cos(phi),r*std::sin(phi),below?z:-z);
    float wavelength=380+400*unit(random), powers[10];
    DiffractionCacheCellView view; int order;
    if (!diffraction_data_power_column<20>(&data,0,ray,below,1,1,wavelength,740,&view,&order,powers)) {
      std::cerr<<"lookup failed "<<q;return 3;
    }
    if(view.ports!=10 || view.reverse_orders)return 7;
    float4 exterior[10];int incoming_port=-1;
    for(int p=0;p<10;p++) {
      const int2 port=b.ports[view.port_offset+p];
      if(port.x==order && bool(port.y)==below)incoming_port=p;
      DiffractionGratingBoundary boundary;
      if(!diffraction_grating_boundary(ray,1,1,wavelength,740,port.x-order,&boundary))return 8;
      exterior[p]=make_float4(boundary.tangent_direction.x,boundary.tangent_direction.y,boundary.q2,1);
    }
    const float2 *anchor=b.matrices.data()+view.matrix_offset;
    DiffractionTensorFloatView model{anchor+400,0};float2 chart[400],jones[40];
    if(!diffraction_tensor_chart<20>(model,view.tensor_floats,view.tensor_rank,view.coordinate,chart)||
       !diffraction_admittance_candidate<20>(chart,anchor,exterior,incoming_port,jones))return 9;
    if(argc==4) {
      fixture.write(reinterpret_cast<const char *>(chart),sizeof(chart));
      fixture.write(reinterpret_cast<const char *>(anchor),400*sizeof(float2));
      fixture.write(reinterpret_cast<const char *>(exterior),sizeof(exterior));
      float padding[40]={},input=float(incoming_port);
      fixture.write(reinterpret_cast<const char *>(padding),sizeof(padding));
      fixture.write(reinterpret_cast<const char *>(&input),sizeof(input));
      expected.write(reinterpret_cast<const char *>(jones),sizeof(jones));
      if(!fixture||!expected)return 10;
    }
    if(argc==3) {
      gpu.read(reinterpret_cast<char *>(jones),sizeof(jones));
      if(!gpu || (q==4095 && gpu.peek()!=EOF))return 11;
    }
    for(int p=0;p<10;p++) {
      powers[p]=0;
      for(int i=0;i<4;i++)powers[p]+=.5f*len_squared(jones[4*p+i]);
    }
    DiffractionCacheCoordinates coordinates;
    diffraction_cache_coordinates(ray,1,wavelength,740,false,&coordinates);
    DiffractionGratingBlock block; DiffractionGratingPowerBlock reference; std::string error;
    if (!diffraction_grating_solve_bloch(profile,wavelength,
        double(coordinates.query.x)*wavelength/740,coordinates.query.y,16,block,error)) {
      std::cerr<<error;return 4;
    }
    diffraction_grating_power_block(block,reference);
    int incoming=-1;
    for (int p=0;p<int(reference.ports.size());p++)
      if (reference.ports[p].order==order && reference.ports[p].substrate==below) incoming=p;
    if(incoming<0)return 5;
    double l1=0,sum=0;
    for(int p=0;p<view.ports;p++) {
      int2 port=b.ports[view.port_offset+p]; double expected=0;
      for(int j=0;j<int(reference.ports.size());j++)
        if(reference.ports[j].order==port.x && reference.ports[j].substrate==bool(port.y))
          expected=reference.matrix[j*reference.ports.size()+incoming];
      if(!std::isfinite(powers[p]) || powers[p]<0)return 6;
      l1+=std::abs(powers[p]-expected);sum+=powers[p];
    }
    maximum_error=std::max(maximum_error,l1);
    maximum_conservation_error=std::max(maximum_conservation_error,std::abs(sum-1));
  }
  std::cout<<"{\"queries\":4096,\"seed\":913771,\"maximum_power_column_l1_error\":"<<maximum_error
    <<",\"maximum_conservation_error\":"<<maximum_conservation_error<<"}\n";
  return maximum_error<=.001 && maximum_conservation_error<=.0001 ? 0 : 1;
}
