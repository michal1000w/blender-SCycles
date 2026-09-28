/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include "kernel/util/diffraction_scene_data.h"
#include <fstream>
#include <iostream>
#include <random>
#include <Eigen/Dense>
#include "diffraction_admittance_candidate.h"
#include <iomanip>
using namespace ccl;
int main(int argc, char **argv)
{
  if (argc != 2) return 2;
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
  using C=std::complex<double>; using M=Eigen::MatrixXcd;
  for (int q=0;q<=397;q++) {
    float z=unit(random), phi=2*M_PI_F*unit(random), r=std::sqrt(1-z*z);
    bool below=q%2;
    float3 ray=make_float3(r*std::cos(phi),r*std::sin(phi),below?z:-z);
    float wavelength=380+400*unit(random); const float u=unit(random); (void)u;
    if(q!=397)continue;
    DiffractionCacheCellView view; int order; float powers[10];
    if(!diffraction_data_power_column<20>(&data,0,ray,below,1,1,wavelength,740,&view,&order,powers))return 3;
    if(view.ports!=10 || view.reverse_orders)return 4;
    float2 boundary[40],chart[400],jones[40]; float4 exterior_data[10]; int incoming=-1;
    for(int p=0;p<10;p++) {
      int2 port=b.ports[view.port_offset+p];
      if(port.x==order && bool(port.y)==below)incoming=p;
      DiffractionGratingBoundary exterior;
      if(!diffraction_grating_boundary(ray,1,1,wavelength,740,port.x-order,&exterior))return 5;
      exterior_data[p]=make_float4(exterior.tangent_direction.x,exterior.tangent_direction.y,exterior.q2,1);
      boundary[4*p]=exterior.r_te;boundary[4*p+1]=exterior.r_tm;
      boundary[4*p+2]=exterior.transmission;boundary[4*p+3]=exterior.tangent_direction;
    }
    const float2 *anchor=b.matrices.data()+view.matrix_offset;
    DiffractionTensorFloatView model{anchor+400,0};
    if(!diffraction_tensor_chart<20>(model,view.tensor_floats,view.tensor_rank,view.coordinate,chart) ||
       !diffraction_reference_match_anchor<20>(chart,anchor,boundary,incoming,jones))return 6;
    float2 admittance_jones[40];
    if(!diffraction_admittance_candidate<20>(chart,anchor,exterior_data,incoming,admittance_jones))return 7;
    double admittance_power=0;
    for(auto value:admittance_jones)admittance_power+=.5*(double(value.x)*value.x+double(value.y)*value.y);
    M Y(20,20),U(20,20),R=M::Zero(20,20),T=M::Zero(20,20);
    for(int i=0;i<20;i++)for(int j=0;j<20;j++) {
      auto y=chart[i*20+j],v=anchor[i*20+j];Y(i,j)=C(y.x,y.y);U(i,j)=C(v.x,v.y);
      if(i/2==j/2) {
        auto reflection=diffraction_reference_r(boundary,i/2,i%2,j%2);
        R(i,j)=C(reflection.x,reflection.y);
        T(i,j)=diffraction_reference_t(boundary,i/2,i%2,j%2);
      }
    }
    M V=U*(M::Identity(20,20)-Y), A=M::Identity(20,20)+Y-R*V;
    M rhs=T.middleCols(2*incoming,2), x=A.partialPivLu().solve(rhs);
    M exact=T*V*x-R.middleCols(2*incoming,2);
    double float_power=0,double_power=0,maximum=0;
    for(int i=0;i<20;i++)for(int j=0;j<2;j++) {
      C value(jones[2*i+j].x,jones[2*i+j].y);
      float_power+=.5*std::norm(value);double_power+=.5*std::norm(exact(i,j));
      maximum=std::max(maximum,std::abs(value-exact(i,j)));
    }
    // Diagnostic only: isolate anchor storage error using a double-precision
    // polar factor. This does not modify the cache or normalize its powers.
    Eigen::JacobiSVD<M> anchor_svd(U,Eigen::ComputeFullU|Eigen::ComputeFullV);
    M polar=anchor_svd.matrixU()*anchor_svd.matrixV().adjoint();
    M polar_v=polar*(M::Identity(20,20)-Y);
    M polar_a=M::Identity(20,20)+Y-R*polar_v;
    M polar_x=polar_a.partialPivLu().solve(rhs);
    M polar_jones=T*polar_v*polar_x-R.middleCols(2*incoming,2);
    double polar_power=.5*polar_jones.squaredNorm();
    M precise_r=M::Zero(20,20),precise_t=M::Zero(20,20);
    const double norm=std::sqrt(double(ray.x)*ray.x+double(ray.y)*ray.y+double(ray.z)*ray.z);
    for(int p=0;p<10;p++) {
      const int relative=b.ports[view.port_offset+p].x-order;
      const double px=ray.x/norm+double(relative)*wavelength/740,py=ray.y/norm;
      const double transverse=std::hypot(px,py),q2=1-px*px-py*py;
      const double dx=transverse>0?px/transverse:1,dy=transverse>0?py/transverse:0;
      const C normal=std::sqrt(C(q2,0));
      const C rs=(C(1)-normal)/(C(1)+normal),rp=(normal-C(1))/(normal+C(1));
      const double t=q2>0?2*std::sqrt(normal.real())/(1+normal.real()):0;
      for(int i=0;i<2;i++)for(int j=0;j<2;j++) {
        double projection=(i?dy:dx)*(j?dy:dx);
        precise_r(2*p+i,2*p+j)=(double(i==j)-projection)*rs+projection*rp;
        precise_t(2*p+i,2*p+j)=double(i==j)*t;
      }
    }
    M precise_a=M::Identity(20,20)+Y-precise_r*polar_v;
    M precise_rhs=precise_t.middleCols(2*incoming,2);
    M precise_x=precise_a.partialPivLu().solve(precise_rhs);
    M precise_jones=precise_t*polar_v*precise_x-precise_r.middleCols(2*incoming,2);
    const double precise_power=.5*precise_jones.squaredNorm();
    Eigen::JacobiSVD<M> svd(A);
    std::cout<<std::setprecision(12)<<"{\"query\":397,\"float_power\":"<<float_power
      <<",\"double_power_same_float_inputs\":"<<double_power<<",\"maximum_jones_error\":"<<maximum
      <<",\"double_polar_anchor_power\":"<<polar_power
      <<",\"double_polar_precise_boundary_power\":"<<precise_power
      <<",\"float_admittance_power\":"<<admittance_power
      <<",\"matching_condition\":"<<svd.singularValues()(0)/svd.singularValues()(19)
      <<",\"anchor_unitarity_residual\":"<<(U.adjoint()*U-M::Identity(20,20)).norm()<<"}\n";
  }
}
