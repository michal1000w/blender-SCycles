/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf_diffraction_util.h"
#include <cmath>
#include <cstdio>
#include <initializer_list>
using namespace ccl;

struct D {
  double x,y,z;
  D operator+(D b) const {return {x+b.x,y+b.y,z+b.z};}
  D operator-(D b) const {return {x-b.x,y-b.y,z-b.z};}
  D operator*(double s) const {return {s*x,s*y,s*z};}
};
static double dotd(D a,D b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
static D crossd(D a,D b) {return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static double length(D a) {return std::sqrt(dotd(a,a));}
static D unit(D a) {return a*(1/length(a));}
static D convert(float3 a) {return {a.x,a.y,a.z};}

/* Independent angle construction: solve v.e(theta)=kick in the plane of
 * v and axis, then follow the branch nearest the supplied normal. This calls
 * neither the production root solver nor its Jacobian. */
static D inverse(D v,D axis,double kick,D near)
{
  const D B=unit(v-axis*dotd(axis,v));
  const double x=dotd(v,axis),y=dotd(v,B);
  const double sign=dotd(near,B)>0?1:-1;
  const double phase=std::atan2(y,x),tilt=std::asin(sign*kick/length(v));
  const double theta[2]={phase+tilt,phase+std::acos(-1.)-tilt};
  D best{};double distance=1e30;
  for(double t:theta) {
    const D h=axis*std::cos(t)+B*std::sin(t);
    if(sign*dotd(h,B)<=0)continue;
    const double error=length(h-near);
    if(error<distance){best=h;distance=error;}
  }
  return best;
}

int main()
{
  struct Fixture {float3 wi,wo,h,axis;float delta;};
  const Fixture fixtures[]={
    {make_float3(-.7823698521f,.1415566206f,.6065138578f),
     make_float3(.4448600113f,.6808281541f,.5818698406f),
     make_float3(-.5454487801f,.6093267798f,-.5755054355f),
     make_float3(.5815280676f,-.7142629623f,.3894271255f),-1.484076738f},
    {make_float3(.2137854546f,-.1204970255f,.9694206119f),
     make_float3(.9131450653f,.1388998181f,-.3832400143f),
     make_float3(-.2180676162f,.8903943896f,.3995551467f),
     make_float3(-.4846869111f,.8423107862f,.2357776016f),-1.270391941f}
  };
  int failures=0;
  for(int i=0;i<2;++i) {
    const auto &f=fixtures[i];
    const float3 vf=diffraction_symmetric_momentum(2,f.wi,2,f.wo);
    DiffractionTransmissionRoot roots[2];
    const int count=diffraction_transmission_half_vectors_momentum(
        f.wi,f.wo,f.axis,vf,2*f.delta,roots);
    int nearest=-1;float distance=1e30f;
    for(int r=0;r<count;++r)if(len(roots[r].h-f.h)<distance) {
      distance=len(roots[r].h-f.h);nearest=r;
    }
    if(nearest<0){++failures;continue;}
    const float J=diffraction_transmission_jacobian_indices(
        f.wi,f.wo,roots[nearest].h,f.axis,2,2,2*f.delta);
    const D v=convert(vf),axis=unit(convert(f.axis)),near=convert(roots[nearest].h);
    const D outgoing=convert(f.wo),W=unit(outgoing);
    const D U=unit(crossd(W,{1,0,0})),V=crossd(W,U);
    /* Keep the rounded center momentum fixed. Perturb on the outgoing sphere
     * without re-rounding each perturbed momentum to float. This measures the
     * derivative underlying evaluation, not the discontinuous float program. */
    double previous=0;
    for(double step:{2e-9,1e-9}) {
      auto root=[&](D tangent,double sign) {
        const D ray=(W*std::cos(step)+tangent*(sign*std::sin(step)))*length(outgoing);
        return inverse(v+(ray-outgoing)*2,axis,2*f.delta,near);
      };
      const D du=root(U,1)-root(U,-1),dv=root(V,1)-root(V,-1);
      const double numeric=length(crossd(du,dv))/(4*step*step);
      const double error=std::abs(numeric/J-1);
      std::printf("fixture=%d step=%.3g J=%.9g inverse_difference=%.9g relative=%.9g\n",
                  i,step,J,numeric,error);
      failures+=!std::isfinite(numeric)||error>.001;
      if(previous)failures+=std::abs(numeric/previous-1)>.001;
      previous=numeric;
    }
  }
  std::printf("failures=%d\n",failures);
  return failures!=0;
}
