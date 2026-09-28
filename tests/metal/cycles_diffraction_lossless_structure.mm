/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Diagnose arithmetic error separately from a bounded polar correction.
 * This does not change the production response or cache acceptance policy. */
#include <Eigen/Dense>
#include "diffraction_cache_export.h"
#include <chrono>
#include <iostream>
#ifdef DIFFRACTION_TEST_METAL
#  include "device/metal/diffraction/reference_pool.h"
#  include "diffraction_shader_source.h"
#endif

int main(int argc, char **argv)
{
  using namespace ccl;
  const bool cutoffs = argc == 2 && std::string(argv[1]) == "--cutoffs";
  const bool profiles = argc == 2 && std::string(argv[1]) == "--profiles";
  if (argc != 1 && !cutoffs && !profiles) return 2;
#ifdef DIFFRACTION_TEST_METAL
  @autoreleasepool {
  auto pool = std::make_shared<DiffractionMetalReferencePool>(
      [NSString stringWithUTF8String:diffraction_metal_shader_source],
      MTLCreateSystemDefaultDevice());
#endif
  const std::vector<ccl::DiffractionGratingProfile> cases = {
      {740,150,0.41,1,1.5,1,1.5},
      {1600,100,0.5,1,1.45,1,1},
      {900,350,0.3,1.33,2.4,1.33,1.5},
      {500,500,0.1,1,2,1,1},
      {740,0,0.41,1,1.5,1,1.5},
      {740,150,0.9,1,1,1.5,1.5}};
  std::cout << std::setprecision(17);
  std::cout << "profile,half_orders,query,gpu_unitarity,cpu_unitarity,max_component_error,corrected_unitarity,correction_max,corrected_cpu_error\n";
  for (int profile_index=0;profile_index<(profiles ? int(cases.size()) : 1);++profile_index) {
  const auto &profile=cases[profile_index];
  for (int half_orders : {4,16,32,64}) {
  if (profiles && half_orders!=16 && half_orders!=64) continue;
  for (int sample=0;sample<(cutoffs ? 24 : 27);++sample) {
    int digits=sample;
    double q[3]; const double lo[3]={-.0625,-.125,590},hi[3]={.0625,.125,610};
    for(int a=0;a<3;++a) {q[a]=lo[a]+.5*(digits%3)*(hi[a]-lo[a]);digits/=3;}
    if (profiles) {
      q[0] = -0.4+0.4*(sample%3);
      q[1] = -0.5+0.5*((sample/3)%3);
      q[2] = 400+150*(sample/9);
    }
    if (cutoffs) {
      q[0]=0.03; q[2]=600;
      const double exterior = sample/12 == 0 ? 1.0 : 1.5;
      const double order_x = (q[0]+1)*q[2]/740;
      const double offset = (sample%2 ? 1 : -1)*std::pow(10.0, -3-2*((sample%6)/2));
      q[1]=(sample%12 < 6 ? 1 : -1)*
           (std::sqrt(exterior*exterior-order_x*order_x)+offset);
    }
    ccl::DiffractionGratingBlock gpu,cpu; std::string error;
    const int retained=profiles ? int(std::ceil(std::max(profile.incident_ior,
        profile.substrate_ior.real())*profile.pitch/400+0.5)) : 2;
    if(!pool->solve(profile,q[2],q[0]*q[2]/profile.pitch,q[1],half_orders,retained,gpu,error) ||
       !ccl::diffraction_grating_solve_reference(profile,q[2],q[0]*q[2]/profile.pitch,q[1],half_orders,retained,cpu,error)) {
      std::cerr<<error;return 1;
    }
    const int n=2*cpu.ports.size();
    if(gpu.matrix.size()!=cpu.matrix.size())return 2;
    Eigen::MatrixXcd g(n,n),c(n,n);double difference=0;
    for(int r=0;r<n;++r)for(int col=0;col<n;++col) {
      g(r,col)=gpu.matrix[r*n+col];c(r,col)=cpu.matrix[r*n+col];
      difference=std::max(difference,std::abs(g(r,col)-c(r,col)));
    }
    const auto identity=Eigen::MatrixXcd::Identity(n,n).eval();
    const double residual=(g.adjoint()*g-identity).norm();
    if(!std::isfinite(residual)||residual>1e-4) {
      std::cerr<<"Response outside experimental correction bound: profile="<<profile_index
               <<" N="<<half_orders<<" query="<<sample<<" residual="<<residual;return 3;
    }
    Eigen::MatrixXcd corrected=g;
    for(int iteration=0;iteration<2;++iteration)
      corrected=(0.5*corrected*(3.0*identity-corrected.adjoint()*corrected)).eval();
    const double corrected_residual=(corrected.adjoint()*corrected-identity).norm();
    const double correction=(corrected-g).cwiseAbs().maxCoeff();
    const double corrected_error=(corrected-c).cwiseAbs().maxCoeff();
    std::cout<<profile_index<<','<<half_orders<<','<<sample<<','<<residual
             <<','<<(c.adjoint()*c-identity).norm()<<','<<difference
             <<','<<corrected_residual<<','<<correction<<','<<corrected_error<<'\n';
    if(!corrected.allFinite()||corrected_residual>1e-12||correction>1e-4||
       corrected_error>3e-4)return 4;
  }
  }
  }

#ifdef DIFFRACTION_TEST_METAL
  }
#endif
}
