/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/light/coherent_curved_geometry.h"
#include <cstdio>
#include <initializer_list>
using namespace ccl;
float3 rotate(float3 p)
{
  const float c = cosf(.47f), s = sinf(.47f);
  return make_float3(c*p.x-s*p.z,p.y,s*p.x+c*p.z);
}
void vec(float3 p) { std::printf(",%.17g,%.17g,%.17g",double(p.x),double(p.y),double(p.z)); }
int main()
{
  std::puts("scale,cx,cy,cz,radius,sx,sy,sz,rx,ry,rz,nx,ny,nz,solved,spreading,opl,hx,hy,hz");
  for(float scale:{.01f,1.f,100.f}) for(bool translated:{false,true})
    for(float y:{-.08f,0.f,.08f}) for(float z:{-.33f,-.25f,-.17f})
      for(float sy:{-1e-5f,1e-5f}) {
        float3 center=translated?make_float3(3.7f,-2.1f,4.3f):zero_float3();
        float3 source=center+rotate(make_float3(-.65f,sy,.35f)*scale);
        float3 receiver=center+rotate(make_float3(-.65f,y,z)*scale);
        float3 normal=normalize(rotate(make_float3(1,0,0)));
        CoherentGeometryInterface frame{};CoherentGeometryPath path{};
        bool ok=coherent_sphere_reflect(source,receiver,normal,center,.25f*scale,&frame,&path);
        std::printf("%.17g",double(scale));vec(center);std::printf(",%.17g",double(.25f*scale));vec(source);vec(receiver);vec(normal);
        std::printf(",%d,%.17g,%.17g",ok,double(path.spreading),double(path.optical_length_split.x)+double(path.optical_length_split.y));vec(path.point[0]);std::puts("");
      }
}
