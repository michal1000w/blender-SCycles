/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "propagation.h"
#include <functional>
#ifdef DIFFRACTION_LARGE_ORDERS
inline constexpr unsigned diffraction_resident_max_orders=256;
#else
inline constexpr unsigned diffraction_resident_max_orders=128;
#endif
struct DiffractionResidentProfile {
  unsigned half_orders,retained;
  float pitch,wavelength,depth,duty,incident,ridge_n,ridge_k,groove_n,groove_k,substrate_n,substrate_k,kx,ky;
};
static_assert(sizeof(DiffractionResidentProfile)==60);
inline DiffractionResidentMatrix::Matrix diffraction_resident_response(
    DiffractionResidentMatrix &e,const DiffractionResidentProfile &profile,unsigned &steps,
    const std::function<void(const char *,DiffractionResidentMatrix::Matrix)> &observe={})
{
  using Matrix=DiffractionResidentMatrix::Matrix;
  const unsigned count=2*profile.half_orders+1,m=2*count;
  if(profile.retained>profile.half_orders||!profile.half_orders||profile.half_orders>diffraction_resident_max_orders||
     !(profile.pitch>0)||!(profile.wavelength>0)||!(profile.depth>=0)||
     !(profile.duty>=0&&profile.duty<=1)||!(profile.incident>0))
    throw std::runtime_error("Invalid resident material profile");
  for(float v:{profile.pitch,profile.wavelength,profile.depth,profile.duty,profile.incident,
                profile.ridge_n,profile.ridge_k,profile.groove_n,profile.groove_k,
                profile.substrate_n,profile.substrate_k,profile.kx,profile.ky})
    if(!std::isfinite(v))throw std::runtime_error("Nonfinite resident material parameter");
  auto valid_index=[](float n,float k){float norm=n*n+k*k;return n>=0&&k>=0&&norm>0&&std::isfinite(norm);};
  if(!valid_index(profile.ridge_n,profile.ridge_k)||!valid_index(profile.groove_n,profile.groove_k)||
     !valid_index(profile.substrate_n,profile.substrate_k))throw std::runtime_error("Unsupported resident optical index");
  const double external=profile.substrate_k==0?std::max(profile.incident,profile.substrate_n):profile.incident;
  const double radius2=external*external-double(profile.ky)*profile.ky;
  if(radius2>0) {
    const double radius=std::sqrt(radius2),step=double(profile.wavelength)/profile.pitch;
    if(profile.kx-(profile.retained+1)*step>-radius||profile.kx+(profile.retained+1)*step<radius)
      throw std::runtime_error("Resident reference window omits propagating channels");
  }
#ifdef DIFFRACTION_ADAPTIVE_QUEUE
  /* N16/N32 response benchmarks favor a wider queue. Keep the validated
   * resource bound for larger matrices rather than extrapolating that result. */
  e.pending_command_limit=profile.half_orders<=32?256:16;
#endif
  auto material=[&](unsigned kind){unsigned size=kind<3?count:m;return e.material(size,size,kind,&profile,sizeof(profile));};
  auto add=[&](Matrix a,Matrix b){return e.add(a,b);};
  auto neg=[&](Matrix a){return e.scale(a,-1);};
  auto sub=[&](Matrix a,Matrix b){return add(a,neg(b));};
  auto mul=[&](Matrix a,Matrix b){return e.multiply(a,b);};
  auto solve=[&](Matrix a,Matrix b){return e.refined_solve(a,b,2);};
  if(profile.depth==0) {
    // Solve the flat-interface continuity equation directly. The zero-thickness
    // relief has no constitutive response and needs no internal modal basis.
    steps=0;
    Matrix im=e.identity(m),upper=material(3),lower=material(4);
    Matrix shared=solve(add(upper,lower),e.join(e.scale(upper,2),e.scale(lower,2)));
    Matrix left=e.slice(shared,0,0,m,m),right=e.slice(shared,0,m,m,m);
    Matrix exterior=e.block(sub(left,im),right,left,sub(right,im));
    unsigned channels=2*(2*profile.retained+1)*(profile.substrate_k==0?2:1);
    return e.retained(exterior,channels,&profile,sizeof(profile));
  }
  Matrix identity=e.identity(count),epsilon=material(0),inverse_e=solve(epsilon,identity);
  Matrix normal_e=solve(material(1),identity),k=material(2);
  Matrix ki=mul(k,inverse_e),ik=mul(inverse_e,k);
  float ky=profile.ky;
  Matrix p=e.block(e.scale(ki,ky),sub(identity,mul(ki,k)),sub(e.scale(inverse_e,ky*ky),identity),e.scale(ik,-ky));
  Matrix q=e.block(e.scale(k,-ky),sub(mul(k,k),epsilon),sub(normal_e,e.scale(identity,ky*ky)),e.scale(k,ky));
  Matrix internal=material(5);
  Matrix transformed_p=mul(p,internal);
#ifdef DIFFRACTION_ANALYTIC_ADMITTANCE
  Matrix inverse_internal=material(6);
  Matrix transformed_q=mul(inverse_internal,q);
  Matrix upper=mul(inverse_internal,material(3)),lower=mul(inverse_internal,material(4));
#else
  Matrix transformed_q=solve(internal,q);
  Matrix upper=solve(internal,material(3)),lower=solve(internal,material(4));
#endif
  auto response=[&](Matrix tp,Matrix tq,Matrix u,Matrix l,unsigned &local_steps) {
    const unsigned size=tp.rows;
    if(observe){observe("transformed_p",tp);observe("transformed_q",tq);observe("upper",u);observe("lower",l);}
    std::function<void(unsigned,Matrix)> step_observer;
    if(observe)step_observer=[&](unsigned step,Matrix matrix) {
      std::string name="propagation_step_"+std::to_string(step);observe(name.c_str(),matrix);
    };
    Matrix s=diffraction_resident_propagation(e,tp,tq,
      float(2*3.14159265358979323846*profile.depth/profile.wavelength),local_steps,step_observer);
    if(observe)observe("propagation",s);
    Matrix im=e.identity(size),zero=e.scale(im,0);
    Matrix c=e.block(e.scale(add(im,u),.5f),zero,zero,e.scale(add(im,l),.5f));
    Matrix d=e.block(e.scale(sub(im,u),.5f),zero,zero,e.scale(sub(im,l),.5f));
    Matrix result=solve(sub(c,mul(s,d)),sub(mul(s,c),d));
    if(observe)observe("exterior",result);
    return result;
  };
  Matrix exterior;
#ifdef DIFFRACTION_POL_SPLIT
  if(profile.ky==0) {
    // At exactly ky=0 the transformed generators and exterior admittances
    // are polarization-block diagonal. Preserve original side/polarization
    // order when joining the two independently propagated responses.
    Matrix pol[2];unsigned pol_steps[2];
    for(unsigned i=0;i<2;++i) {
      auto diagonal=[&](Matrix a){return e.slice(a,i*count,i*count,count,count);};
      pol[i]=response(diagonal(transformed_p),diagonal(transformed_q),
                      diagonal(upper),diagonal(lower),pol_steps[i]);
    }
    steps=std::max(pol_steps[0],pol_steps[1]);
    Matrix zero=e.scale(identity,0);
    auto side_block=[&](unsigned row,unsigned col) {
      return e.block(e.slice(pol[0],row*count,col*count,count,count),zero,zero,
                     e.slice(pol[1],row*count,col*count,count,count));
    };
    exterior=e.block(side_block(0,0),side_block(0,1),side_block(1,0),side_block(1,1));
  }
  else
#endif
  {
    exterior=response(transformed_p,transformed_q,upper,lower,steps);
  }
  unsigned channels=2*(2*profile.retained+1)*(profile.substrate_k==0?2:1);
  return e.retained(exterior,channels,&profile,sizeof(profile));
}
