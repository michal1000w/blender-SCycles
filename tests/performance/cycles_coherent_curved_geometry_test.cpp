/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/light/coherent_curved_geometry.h"
#include <cmath>
#include <initializer_list>
#include <cstdio>
using namespace ccl;
static int checks=0, failures=0;
static void check(bool value,const char *name){checks++;if(!value){failures++;printf("FAIL %s\n",name);}}
static double oracle(double a,double b,double theta,double radius)
{
  double lo=fmax(0.,theta-acos(radius/b)),hi=fmin(theta,acos(radius/a));
  for(int i=0;i<90;i++){
    double p=(lo+hi)/2;
    double ls=sqrt(a*a+radius*radius-2*a*radius*cos(p));
    double lr=sqrt(b*b+radius*radius-2*b*radius*cos(theta-p));
    double derivative=a*sin(p)/ls-b*sin(theta-p)/lr;
    if(derivative>0)hi=p;else lo=p;
  }
  return (lo+hi)/2;
}
int main()
{
  for(float scale:{.01f,1.f,100.f}) {
    const float radius=.4f*scale;
    for(float angle:{0.f,.2f,.9f,1.5f}) {
      const float3 source=make_float3(0,0,2*scale);
      const float3 receiver=make_float3(3*scale*sinf(angle),0,3*scale*cosf(angle));
      CoherentGeometryInterface f{};CoherentGeometryPath p{};
      const float3 normal=-normalize(receiver);
      bool ok=coherent_sphere_reflect(source,receiver,normal,zero_float3(),radius,&f,&p);
      check(ok,"visible exterior branch");if(!ok)continue;
      const double a=len(source),b=sqrt(double(receiver.x)*receiver.x+double(receiver.z)*receiver.z);
      const double theta=atan2(double(receiver.x),double(receiver.z));
      const double phi=oracle(a,b,theta,radius);
      const double px=radius*sin(phi),pz=radius*cos(phi);
      const double length=hypot(px,double(source.z)-pz)+hypot(double(receiver.x)-px,double(receiver.z)-pz);
      const double actual=double(p.optical_length_split.x)+p.optical_length_split.y;
      check(fabs(actual-length)<2e-11*scale,"stationary split OPL vs independent double");
      check(len(p.point[0]-make_float3(px,0,pz))<3e-7f*scale,"root vs independent angular oracle");
      float3 u,v;make_orthonormals(normal,&u,&v);
      const float h=1e-3f*scale;
      CoherentGeometryPath plus[2]{},minus[2]{};
      bool neighbors=true;
      for(int i=0;i<2;i++){
        float3 delta=(i==0?u:v)*h;
        neighbors &= coherent_sphere_reflect(source,receiver+delta,normal,zero_float3(),radius,&f,&plus[i]);
        neighbors &= coherent_sphere_reflect(source,receiver-delta,normal,zero_float3(),radius,&f,&minus[i]);
      }
      float3 du=(plus[0].source_direction-minus[0].source_direction)/(2*h);
      float3 dv=(plus[1].source_direction-minus[1].source_direction)/(2*h);
      float fd=fabsf(dot(cross(du,dv),p.source_direction));
      check(neighbors && fabsf(fd/p.spreading-1)<8e-4f,"curvature Jacobian vs independent ray finite difference");
      if(angle==0){double ds=a-radius,dr=b-radius,H=1/ds+1/dr+2/radius;
        double exact=1/(ds*dr*H);exact*=exact;
        check(fabs(p.spreading/exact-1)<2e-6,"axial analytic spreading");}
    }
  }
  for(float angle:{1e-7f,2*acosf(.2f)-1e-3f,2*acosf(.2f)-1e-4f}) {
    const float3 source=make_float3(0,0,2);
    const float3 receiver=make_float3(2*sinf(angle),0,2*cosf(angle));
    CoherentGeometryInterface f{};CoherentGeometryPath p{};
    check(coherent_sphere_reflect(source,receiver,-normalize(receiver),zero_float3(),.4f,&f,&p),"tiny-angle / near-horizon branch retained");
    double theta=atan2(double(receiver.x),double(receiver.z));
    double b=hypot(double(receiver.x),double(receiver.z));
    double phi=oracle(2,b,theta,.4f);
    double px=.4f*sin(phi),pz=.4f*cos(phi);
    double exact=hypot(px,2-pz)+hypot(double(receiver.x)-px,double(receiver.z)-pz);
    check(fabs(double(p.optical_length_split.x)+p.optical_length_split.y-exact)<3e-12,"near-horizon split optical length");
    check(p.spreading>0 && dot(p.source_direction,normalize(p.point[0]))<0,"positive near-horizon geometric measure");
  }
  CoherentGeometryInterface f{};CoherentGeometryPath p{};
  check(!coherent_sphere_reflect(make_float3(0,0,.2f),make_float3(0,0,2),make_float3(0,0,-1),zero_float3(),.4f,&f,&p),"inside source rejected");
  check(!coherent_sphere_reflect(make_float3(0,0,2),make_float3(0,0,.2f),make_float3(0,0,-1),zero_float3(),.4f,&f,&p),"inside receiver rejected");
  check(!coherent_sphere_reflect(make_float3(0,0,2),make_float3(0,0,-2),make_float3(0,0,1),zero_float3(),.4f,&f,&p),"disjoint visible caps rejected");
  printf("%d checks, %d failures\n",checks,failures);return failures?1:0;
}
