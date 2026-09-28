/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "matrix.h"
#include <functional>

inline DiffractionResidentMatrix::Matrix diffraction_resident_propagation(
    DiffractionResidentMatrix &engine,DiffractionResidentMatrix::Matrix p,
    DiffractionResidentMatrix::Matrix q,float thickness,unsigned &steps,
    const std::function<void(unsigned,DiffractionResidentMatrix::Matrix)> &observe={})
{
  using Matrix=DiffractionResidentMatrix::Matrix;
  auto add=[&](Matrix a,Matrix b){return engine.add(a,b);};
  auto mul=[&](Matrix a,Matrix b){return engine.multiply(a,b);};
  auto neg=[&](Matrix a){return engine.scale(a,-1);};
  auto sub=[&](Matrix a,Matrix b){return add(a,neg(b));};
  auto solve=[&](Matrix a,Matrix b){return engine.refined_solve(a,b,2);};
  const unsigned m=p.rows;
  if(p.cols!=m||q.rows!=m||q.cols!=m)throw std::runtime_error("Invalid propagation dimensions");
  Matrix identity=engine.identity(m);
  Matrix pq=add(p,q),qp=sub(q,p);
  Matrix generator=engine.scale(engine.block(pq,qp,neg(qp),neg(pq)),0,.5f*thickness);
  float norm=engine.norm1(generator);
#ifndef DIFFRACTION_EXP_NORM
#  define DIFFRACTION_EXP_NORM 8.0f
#endif
#ifndef DIFFRACTION_EXP_TERMS
#  define DIFFRACTION_EXP_TERMS 40
#endif
  steps=unsigned(std::max(0.0f,std::ceil(std::log2(std::max(norm/DIFFRACTION_EXP_NORM,1.0f)))));
  if(steps>60)throw std::runtime_error("Propagation scaling exceeds supported range");
  Matrix g=engine.scale(generator,std::ldexp(1.0f,-int(steps)));
  Matrix term=engine.identity(2*m),increment=engine.scale(term,0);
  for(unsigned k=1;k<=DIFFRACTION_EXP_TERMS;++k) {
    term=engine.scale(mul(term,g),1.0f/k);
    increment=add(increment,term);
  }
  Matrix aa=engine.slice(increment,0,0,m,m),b=engine.slice(increment,0,m,m,m);
  Matrix c=engine.slice(increment,m,0,m,m),dd=engine.slice(increment,m,m,m,m);
  Matrix x=solve(add(identity,dd),engine.join(c,neg(dd)));
  Matrix dc=engine.slice(x,0,0,m,m),db=engine.slice(x,0,m,m,m);
  Matrix a=neg(dc),new_c=sub(aa,mul(b,dc)),d=mul(b,add(identity,db));
  b=db;c=new_c;
  bool regular=false;
  auto capture=[&](unsigned step) {
    if(observe)observe(step,engine.block(a,regular?b:add(identity,b),regular?c:add(identity,c),d));
  };
  capture(0);
  for(unsigned step=0;step<steps;++step) {
    if(!regular && std::max(engine.norm1(b),engine.norm1(c))>.25f) {
      b=add(identity,b);c=add(identity,c);regular=true;
    }
    Matrix an,bn,cn,dn;
    if(regular) {
      Matrix solved=solve(sub(identity,mul(a,d)),engine.join(mul(a,c),b));
      Matrix xc=engine.slice(solved,0,0,m,m),xf=engine.slice(solved,0,m,m,m);
      an=add(a,mul(b,xc));bn=mul(b,xf);
      cn=mul(c,add(c,mul(d,xc)));dn=add(d,mul(mul(c,d),xf));
    }
    else {
      Matrix ad=mul(a,d),z=solve(sub(identity,ad),ad);
      Matrix xc=mul(mul(add(identity,z),a),add(identity,c));
      Matrix xf=mul(add(identity,z),add(identity,b));
      an=add(a,mul(add(identity,b),xc));
      bn=add(add(engine.scale(b,2),mul(b,b)),mul(mul(add(identity,b),z),add(identity,b)));
      cn=add(add(engine.scale(c,2),mul(c,c)),mul(mul(add(identity,c),d),xc));
      dn=add(d,mul(mul(add(identity,c),d),xf));
    }
    a=an;b=bn;c=cn;d=dn;
    capture(step+1);
  }
  if(!regular){b=add(identity,b);c=add(identity,c);}
  return engine.block(a,b,c,d);
}
