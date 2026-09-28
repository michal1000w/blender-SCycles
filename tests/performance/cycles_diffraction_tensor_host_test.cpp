/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction_tensor.h"
#include "kernel/util/diffraction_tensor.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <random>
using namespace ccl;
int main(int argc,char **argv) {
  if(argc!=2)return 2;
  DiffractionGratingProfile profile{740,150,.41,1,1.5,1,1};
  DiffractionGratingCacheOptions options;options.half_orders=16;options.retained_half_orders=2;
  DiffractionGratingPackedCell cell;std::string error;
  const auto start=std::chrono::steady_clock::now();
  if(!diffraction_grating_prepare_tensor_cell(profile,{{0,.5,580},{.5,1,780}},options,1e-7,cell,error)) {
    std::cerr<<error;return 3;
  }
  DiffractionGratingCache cache;cache.bounds=cell.bounds;cache.nodes={make_int4(-1,0,0,0)};cache.cells={cell};
  DiffractionGratingDeviceBuffers buffers;
  if(!diffraction_grating_device_buffers(cache,buffers,error)){std::cerr<<error;return 4;}
  std::ofstream out(argv[1],std::ios::binary);
  out.write(reinterpret_cast<const char *>(cell.matrices.data()),cell.matrices.size()*sizeof(float2));
  if(!out)return 5;
  std::mt19937 random(7542);
  double maximum = 0;
  for(int sample=0;sample<128;++sample) {
    std::array<double,3> q;
    for(int a=0;a<3;++a)
      q[a]=float(cell.bounds.lower[a]+(double(random())+.5)/4294967296.0*(cell.bounds.upper[a]-cell.bounds.lower[a]));
    DiffractionGratingBlock actual_reference,actual,exact_reference,exact;
    if(!diffraction_grating_tensor_reference(cell,q,actual_reference,error) ||
       !diffraction_grating_match_reference(profile,q[2],q[0]*q[2]/740,q[1],actual_reference,actual,error) ||
       !diffraction_grating_solve_reference(profile,q[2],q[0]*q[2]/740,q[1],16,2,exact_reference,error) ||
       !diffraction_grating_match_reference(profile,q[2],q[0]*q[2]/740,q[1],exact_reference,exact,error))return 7;
    DiffractionGratingPowerBlock a,b;
    diffraction_grating_power_block(actual,a);diffraction_grating_power_block(exact,b);
    if(a.ports.size()!=b.ports.size())return 8;
    for(size_t col=0;col<b.ports.size();++col) {
      double difference=0;
      for(size_t row=0;row<b.ports.size();++row)
        difference+=std::abs(a.matrix[row*b.ports.size()+col]-b.matrix[row*b.ports.size()+col]);
      maximum=std::max(maximum,difference);
    }
  }
  if(!std::isfinite(maximum) || maximum>.001)return 9;
  std::cout<<"{\"rank\":"<<cell.tensor_rank<<",\"model_floats\":"<<buffers.layout[1].w
    <<",\"matrix_pairs\":"<<cell.matrices.size()<<",\"maximum_power_error\":"<<maximum<<",\"seconds\":"
    <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
  options.progress=[](const DiffractionGratingCacheStats &){return false;};
  if(diffraction_grating_prepare_tensor_cell(profile,cache.bounds,options,1e-7,cell,error) ||
     !cell.matrices.empty() || error!="Tensor cell construction cancelled")return 6;
}
