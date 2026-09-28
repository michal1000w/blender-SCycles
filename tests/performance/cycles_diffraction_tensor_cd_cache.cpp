/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction_tensor.h"
#include <chrono>
#include <fstream>
#include <iostream>
using namespace ccl;
int main(int argc,char **argv) {
  if(argc!=2 && argc!=3)return 2;
  const DiffractionGratingProfile profile{1600,150,.41,1,1.5,1,1};
  DiffractionGratingCacheOptions options;
  options.bounds={{-.5,-1,380},{.5,1,780}};options.retained_half_orders=4;
  options.mirror_symmetry=argc==3 && std::string(argv[2])=="--mirror";
  options.maximum_nodes=127;options.maximum_depth=18;
  options.progress=[](const DiffractionGratingCacheStats &s){
    static size_t previous=0;
    if(s.visited_nodes!=previous){previous=s.visited_nodes;std::cerr<<"nodes="<<previous<<" coverage="<<s.accepted_domain_fraction<<" bytes="<<s.matrix_bytes<<'\n';}
    return true;
  };
  DiffractionGratingCache cache;DiffractionGratingCacheStats stats;std::string error;
  auto start=std::chrono::steady_clock::now();
  const bool complete=diffraction_grating_build_tensor_cache(profile,options,cache,stats,error);
  std::cout<<"{\"complete\":"<<(complete?"true":"false")<<",\"nodes\":"<<stats.visited_nodes
    <<",\"cells\":"<<cache.cells.size()<<",\"bytes\":"<<stats.matrix_bytes<<",\"coverage\":"<<stats.accepted_domain_fraction
    <<",\"maximum_construction_error\":"<<stats.maximum_accepted_error<<",\"reference_solves\":"<<stats.reference_solves
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
