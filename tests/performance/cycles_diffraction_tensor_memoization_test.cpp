/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction_tensor.h"
#include <cstring>
#include <iostream>
using namespace ccl;
template<typename T> bool equal(const std::vector<T> &a,const std::vector<T> &b)
{
  return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
bool equal(const DiffractionGratingDeviceBuffers &a,const DiffractionGratingDeviceBuffers &b)
{
  return equal(a.nodes,b.nodes)&&equal(a.layout,b.layout)&&equal(a.bounds,b.bounds)&&
         equal(a.ports,b.ports)&&equal(a.active,b.active)&&equal(a.matrices,b.matrices);
}
int main(int argc, char **argv)
{
  if (argc>2 || (argc==2 && std::string(argv[1])!="--relief")) return 2;
  const bool relief=argc==2;
  const DiffractionGratingProfile profile{740,relief?150.:0.,.41,1,1.5,1,relief?1.:1.5};
  DiffractionGratingCacheOptions options;
  options.bounds={{-.015625,-.015625,600},{.015625,.015625,610}};
  options.half_orders=4;options.retained_half_orders=2;options.maximum_half_orders=16;
  options.modal_power_tolerance=1e-8;options.modal_complex_tolerance=1e-8;options.maximum_nodes=31;
  if (relief) {
    options.bounds={{-.5,-1,380},{.5,1,780}};
    options.mirror_symmetry=true;options.half_orders=16;options.maximum_nodes=127;
    options.modal_power_tolerance=options.modal_complex_tolerance=0;
  }
  DiffractionGratingDeviceBuffers baseline;
  size_t baseline_solves=0;
  for (size_t budget : {size_t(0),size_t(64*1024*1024),size_t(6400)}) {
    options.reference_cache_matrix_bytes=budget;
    DiffractionGratingCache cache;DiffractionGratingCacheStats stats;std::string error;
    DiffractionGratingDeviceBuffers buffers;
    if (!diffraction_grating_build_tensor_cache(profile,options,cache,stats,error) ||
        !diffraction_grating_device_buffers(cache,buffers,error)) {std::cerr<<error;return 1;}
    if (stats.peak_reference_matrix_bytes>budget) return 2;
    if (!budget) {baseline=buffers;baseline_solves=stats.reference_solves;}
    else if (!equal(baseline,buffers)) return 3;
    if (budget>6400 && (!stats.reference_cache_hits || stats.reference_solves>=baseline_solves)) return 4;
    std::cout<<"budget="<<budget<<" solves="<<stats.reference_solves<<" hits="<<stats.reference_cache_hits
             <<" peak_bytes="<<stats.peak_reference_matrix_bytes<<" cells="<<cache.cells.size()<<" byte_identical=1\n"<<std::flush;
  }
}
