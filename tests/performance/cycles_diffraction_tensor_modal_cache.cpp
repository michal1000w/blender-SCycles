/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction_tensor.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace ccl;
int main(int argc,char **argv) {
  if(argc<2)return 2;
  const DiffractionGratingProfile profile{740,150,double(float(.41)),1,1.5,1,1};
  DiffractionGratingCacheOptions options;
  options.bounds={{-.5,-1,380},{.5,1,780}};options.retained_half_orders=2;
  options.maximum_half_orders=512;
  for (int i=2;i<argc;i++) {
    if (std::string(argv[i])=="--mirror") options.mirror_symmetry=true;
    else if (std::string(argv[i])=="--maximum-half-orders" && i+1<argc) {
      try {
        size_t end=0;const std::string value=argv[++i];
        options.maximum_half_orders=std::stoi(value,&end);
        if (end!=value.size()) return 2;
      }
      catch (...) {return 2;}
    }
    else return 2;
  }
  if (options.maximum_half_orders<16 || options.maximum_half_orders>512) return 2;
  options.maximum_nodes=127;options.maximum_depth=18;
  options.modal_power_tolerance=.00025;
  options.progress=[](const DiffractionGratingCacheStats &s){
    static size_t previous=0, solves=0;
    if(s.visited_nodes!=previous || s.reference_solves>=solves+256) {
      previous=s.visited_nodes;solves=s.reference_solves;
      std::cerr<<"nodes="<<previous<<" solves="<<solves<<" coverage="<<s.accepted_domain_fraction
               <<" bytes="<<s.matrix_bytes<<'\n';
    }
    return true;
  };
  DiffractionGratingCache cache;DiffractionGratingCacheStats stats;std::string error;
  auto start=std::chrono::steady_clock::now();
  const bool complete=diffraction_grating_build_tensor_cache(profile,options,cache,stats,error);
  std::cout<<std::setprecision(17)<<"{\"complete\":"<<(complete?"true":"false")<<",\"nodes\":"<<stats.visited_nodes
    <<",\"modal_power_tolerance\":"<<options.modal_power_tolerance
    <<",\"maximum_half_orders\":"<<options.maximum_half_orders
    <<",\"duty_cycle\":"<<profile.duty<<",\"cells\":"<<cache.cells.size()<<",\"bytes\":"<<stats.matrix_bytes<<",\"coverage\":"<<stats.accepted_domain_fraction
    <<",\"maximum_construction_error\":"<<stats.maximum_accepted_error<<",\"reference_solves\":"<<stats.reference_solves
    <<",\"reference_cache_hits\":"<<stats.reference_cache_hits
    <<",\"peak_reference_matrix_bytes\":"<<stats.peak_reference_matrix_bytes
    <<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
  if(!complete){std::cerr<<error;return 1;}
  std::ofstream out(argv[1],std::ios::binary);
  // Export validated production buffers for subsequent device tests.
  DiffractionGratingDeviceBuffers buffers;
  if(!diffraction_grating_device_buffers(cache,buffers,error))return 3;
  auto write=[&](const auto &v){uint64_t count=v.size();out.write(reinterpret_cast<char *>(&count),sizeof(count));out.write(reinterpret_cast<const char *>(v.data()),v.size()*sizeof(v[0]));};
  write(buffers.nodes);write(buffers.layout);write(buffers.bounds);write(buffers.ports);write(buffers.active);write(buffers.matrices);
  if(!out)return 4;
}
