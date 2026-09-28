/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction_convergence.h"
#include <iostream>
using namespace ccl;
int main() {
  DiffractionGratingProfile profile{740,0,.41,1,1.5,1,1.5};
  DiffractionGratingCacheOptions options;
  options.half_orders=4;options.retained_half_orders=2;options.maximum_half_orders=16;
  DiffractionGratingBlock original,custom;size_t count=0,calls=0;std::string error;
  if(!diffraction_grating_cache_reference(profile,options,600,.01,.02,original,count,error))return 1;
  options.reference_solver=[&](const auto &p,double w,double x,double y,int n,int retained,auto &out,auto &message) {
    ++calls;return diffraction_grating_solve_reference(p,w,x,y,n,retained,out,message);
  };
  if(!diffraction_grating_cache_reference(profile,options,600,.01,.02,custom,count,error)||
     custom.matrix!=original.matrix||count!=1||calls!=1)return 2;
  options.modal_power_tolerance=1e-8;options.modal_complex_tolerance=1e-8;calls=0;
  if(!diffraction_grating_cache_reference(profile,options,600,.01,.02,custom,count,error)||count!=3||calls!=3)return 3;
  options.reference_solver=[&](const auto &,double,double,double,int,int,auto &out,auto &message) {
    out=original;message="backend failure";return false;
  };
  if(diffraction_grating_cache_reference(profile,options,600,.01,.02,custom,count,error)||!custom.matrix.empty())return 4;
  options.modal_power_tolerance=0;options.modal_complex_tolerance=0;
  if(diffraction_grating_cache_reference(profile,options,600,.01,.02,custom,count,error)||!custom.matrix.empty())return 5;
  options.reference_solver=[](const auto &,double,double,double,int,int,auto &out,auto &) {
    out={};return true;
  };
  if(diffraction_grating_cache_reference(profile,options,600,.01,.02,custom,count,error)||!custom.matrix.empty())return 6;
  DiffractionModalOptions modal;modal.minimum_half_orders=4;modal.maximum_half_orders=16;
  calls=0;modal.reference_solver=[&](const auto &,double,double,double,int,int,auto &,auto &) {++calls;return false;};
  modal.progress=[](int){return false;};std::vector<DiffractionModalObservation> observations;
  if(diffraction_grating_converged_reference(profile,600,.01,.02,2,modal,custom,observations,error)||calls||!custom.matrix.empty())return 7;
  calls=0;
  options.modal_power_tolerance=1e-8;
  options.cancelled=[&] { return calls>=1; };
  options.reference_solver=[&](const auto &p,double w,double x,double y,int n,int retained,auto &out,auto &message) {
    ++calls;return diffraction_grating_solve_reference(p,w,x,y,n,retained,out,message);
  };
  if(diffraction_grating_cache_reference(profile,options,600,.01,.02,custom,count,error)||
     calls!=1||count!=1||!custom.matrix.empty()||error.find("cancelled")==std::string::npos)return 8;
  options.modal_power_tolerance=0;
  if(diffraction_grating_cache_reference(profile,options,600,.01,.02,custom,count,error)||
     calls!=1||count||!custom.matrix.empty())return 9;
  std::cout<<"Backend delegation, convergence, rejection and mid-refinement cancellation checks passed\n";
}
