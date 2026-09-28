/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/light/coherent_unfold_geometry.h"
#include <cstdio>
#include <cmath>
#include <initializer_list>
using namespace ccl;
static int checks=0, failures=0;
static void check(bool x,const char *s){checks++;if(!x){printf("FAIL %s\n",s);failures++;}}
static bool solve(bool tt,float3 s,float3 r,CoherentGeometryInterface *f,CoherentGeometryPath *p,float *phase){
  f[0].center=make_float3(-1,0,0);f[0].tangent_u=make_float3(0,1,0);f[0].tangent_v=make_float3(0,0,1);f[0].half_u=f[0].half_v=10;
  int end=tt?3:2;f[end]=f[0];f[end].center=make_float3(tt?1:-.9f,0,0);
  return coherent_unfold_sphere_connect(s,r,make_float3(1,0,0),f,end+1,1,tt,0,zero_float3(),.3f,1.5f,p,phase)==COHERENT_SPHERE_TT_OK;
}
int main(){
  for(bool tt:{false,true}){
    float3 s=make_float3(-.5f,.1f,.04f),r=tt?make_float3(.4f,.015f,-.03f):make_float3(-.6f,-.2f,-.03f);
    CoherentGeometryInterface f[4]{};CoherentGeometryPath p{};float phase;
    check(solve(tt,s,r,f,&p,&phase),"mixed route exists");
    double opl=double(p.optical_length_split.x)+p.optical_length_split.y;
    printf("%s OPL %.15g spread %.12g phase %.9g\n",tt?"TT":"R",opl,p.spreading,phase);
    // Independent double oracle with every input rounded to the exact float fixture.
    check(fabs(opl-(tt?3.407249005664111:2.126205728782771))<2e-11,"independent unfolded OPL");
    check(fabs(p.spreading/(tt?.15986276461304494:.010796240554419337)-1)<2e-5,"independent spreading");
    check(fabs(phase-(tt?-.5:0))<1e-6,"inherited Morse phase");
    for(int i=0;i<p.count;i++){
      float3 before=i?p.point[i-1]:s;
      float3 after=i+1==p.count?r:p.point[i+1];
      float3 incoming=normalize(p.point[i]-before),outgoing=normalize(after-p.point[i]);
      float3 n=normalize(cross(f[i].tangent_u,f[i].tangent_v));
      float ni=tt&&i==2?1.5f:1, no=tt&&i==1?1.5f:1;
      float3 g=ni*incoming-no*outgoing;
      check(len(g-n*dot(g,n))<3e-6f,"physical stationary law");
    }
    float3 du,dv;make_orthonormals(make_float3(1,0,0),&du,&dv);
    float3 derivatives[2];
    for(int i=0;i<2;i++){
      CoherentGeometryPath a,b;float ph;CoherentGeometryInterface fa[4]{},fb[4]{};
      float3 delta=(i?dv:du)*1e-3f;
      check(solve(tt,s,r+delta,fa,&a,&ph)&&solve(tt,s,r-delta,fb,&b,&ph),"FD neighbors");
      derivatives[i]=(a.source_direction-b.source_direction)/.002f;
    }
    float spread=fabsf(dot(cross(derivatives[0],derivatives[1]),p.source_direction));
    check(fabs(spread/p.spreading-1)<.003,"physical receiver Jacobian");
  }
  printf("%d checks %d failures\n",checks,failures);return failures?1:0;
}
