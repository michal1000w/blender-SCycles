/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction_tensor.h"
#include "scene/diffraction_manager.h"
#include <iostream>
#include <memory>
using namespace ccl;
int main()
{
  const DiffractionGratingProfile profile{740,0,.41,1,1.5,1,1.5};
  DiffractionGratingCacheOptions options;
  options.bounds={{-.015625,-.015625,600},{.015625,.015625,610}};
  options.half_orders=4; options.retained_half_orders=2;
  options.maximum_half_orders=16;
  options.modal_power_tolerance=1e-8;
  options.modal_complex_tolerance=1e-8;
  options.maximum_nodes=31;
  for (bool tensor : {false,true}) {
    DiffractionGratingCache cache;
    DiffractionGratingCacheStats stats;
    std::string error;
    auto build=[&]() {return tensor ? diffraction_grating_build_tensor_cache(profile,options,cache,stats,error):
                                     diffraction_grating_build_cache(profile,options,cache,stats,error);};
    if (!build() || cache.cells.empty() || stats.reference_solves<3) {
      std::cerr<<"tensor="<<tensor<<" "<<error;return 1;
    }
    std::cout<<"tensor="<<tensor<<" cells="<<cache.cells.size()<<" solves="<<stats.reference_solves<<'\n';
    options.maximum_half_orders=8;
    if (build() || !cache.cells.empty() || error.find("budget")==std::string::npos) return 2;
    options.maximum_half_orders=16;
  }
  DiffractionManager manager;
  DiffractionGratingCacheStats stats;
  std::string error;
  options.use_tensor_cells=true;
  const int first=manager.get_or_build(profile,options,stats,error);
  if (first<0 || manager.get_or_build(profile,options,stats,error)!=first) return 3;
  options.modal_power_tolerance=1e-7;
  const int second=manager.get_or_build(profile,options,stats,error);
  if (second<0 || second==first || manager.cache_count()!=2) return 4;
  options.maximum_half_orders=8;
  if (manager.get_or_build(profile,options,stats,error)>=0 || manager.cache_count()!=2) return 5;
  options.maximum_half_orders=16;
  options.reference_solver=[](const auto &p,double w,double x,double y,int n,int retained,auto &out,auto &message) {
    return diffraction_grating_solve_reference(p,w,x,y,n,retained,out,message);
  };
  if(manager.get_or_build(profile,options,stats,error)>=0||manager.cache_count()!=2)return 6;
  options.reference_backend_key="test-cpu-delegate-v1";
  const int delegated=manager.get_or_build(profile,options,stats,error);
  if(delegated<0||delegated==second||manager.cache_count()!=3)return 7;
  options.reference_backend_key="test-failing-backend-v1";
  options.reference_solver=[](const auto &,double,double,double,int,int,auto &,auto &message) {
    message="intentional backend failure";return false;
  };
  if(manager.get_or_build(profile,options,stats,error)>=0||manager.cache_count()!=3)return 8;
  options.reference_solver={};
  if(manager.get_or_build(profile,options,stats,error)>=0||manager.cache_count()!=3)return 9;
  auto lifetime=std::make_shared<int>(0);
  std::weak_ptr<int> released=lifetime;
  options.reference_backend_key="test-callback-lifetime-v1";
  options.reference_solver=[lifetime](const auto &p,double w,double x,double y,int n,int retained,auto &out,auto &message) {
    return diffraction_grating_solve_reference(p,w,x,y,n,retained,out,message);
  };
  options.cancelled=[lifetime] { return false; };
  const int owned=manager.get_or_build(profile,options,stats,error);
  if(owned<0||manager.cache_count()!=4)return 10;
  lifetime.reset();options.reference_solver={};options.cancelled={};
  if(!released.expired())return 11;
  options.reference_solver=[](const auto &p,double w,double x,double y,int n,int retained,auto &out,auto &message) {
    return diffraction_grating_solve_reference(p,w,x,y,n,retained,out,message);
  };
  options.cancelled=[] { return true; };
  if(manager.get_or_build(profile,options,stats,error)>=0||manager.cache_count()!=4||
     error.find("cancelled")==std::string::npos)return 12;
  options.cancelled={};
  if(manager.get_or_build(profile,options,stats,error)!=owned)return 13;
  std::cout<<"manager identity and failed-publication checks passed\n";
}
