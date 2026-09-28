/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/light/coherent_unfold_internal_geometry.h"
#include <cstdio>
#include <cmath>
#include <initializer_list>
using namespace ccl;
static int checks=0,failures=0;
static void check(bool x,const char*s){checks++;if(!x){printf("FAIL %s\n",s);failures++;}}
int main(){
 for(int m:{1,2})for(float theta:{.3f,.6f,.9f,1.2f,2.4f}) {
  const float3 source=make_float3(2,0,0),receiver=make_float3(2*cosf(theta),2*sinf(theta),0),normal=-normalize(receiver);
  CoherentSphereInternalInventory a;
  auto status=coherent_sphere_internal_inventory(source,receiver,normal,zero_float3(),1,1.5f,m,&a);
  check(status==COHERENT_SPHERE_TT_OK||status==COHERENT_SPHERE_TT_EMPTY,"finite isolated inventory");
  check(a.count<=3*(m+1),"proved finite root bound");
  for(int k=0;k<a.count;k++) {
   const auto &p=a.path[k];
   CoherentSphereInternalInventory selected;
   check(coherent_sphere_internal_inventory(source,receiver,normal,zero_float3(),1,1.5f,m,&selected,k)==COHERENT_SPHERE_TT_OK && selected.count==a.count,"selected branch inventory count");
   check(selected.path[k].optical_length_split.x==p.optical_length_split.x && selected.path[k].optical_length_split.y==p.optical_length_split.y && selected.path[k].spreading==p.spreading && selected.morse_index[k]==a.morse_index[k],"selected branch equals exhaustive inventory");
   for(int i=0;i<p.count;i++) {
    const float3 before=i?p.point[i-1]:source,after=i+1==p.count?receiver:p.point[i+1];
    const float3 incoming=normalize(p.point[i]-before),outgoing=normalize(after-p.point[i]);
    const float3 n=normalize(p.point[i]);const float ni=i?1.5f:1,no=i+1==p.count?1:1.5f;
    const float3 gradient=ni*incoming-no*outgoing;
    check(len(gradient-n*dot(gradient,n))<8e-6f,"Snell and internal reflection residual");
    check(fabsf(len(p.point[i])-1)<2e-7f,"sphere point");
   }
   float3 u,v;make_orthonormals(normal,&u,&v);float3 derivatives[2];
   for(int axis=0;axis<2;axis++) {
    const float3 delta=(axis?v:u)*1e-3f;CoherentSphereInternalInventory plus,minus;
    auto ps=coherent_sphere_internal_inventory(source,receiver+delta,normal,zero_float3(),1,1.5f,m,&plus);
    auto ms=coherent_sphere_internal_inventory(source,receiver-delta,normal,zero_float3(),1,1.5f,m,&minus);
    check(ps==COHERENT_SPHERE_TT_OK&&ms==COHERENT_SPHERE_TT_OK&&plus.count==a.count&&minus.count==a.count,"neighbor branch inventory");
    derivatives[axis]=(plus.path[k].source_direction-minus.path[k].source_direction)/.002f;
   }
   const float fd=fabsf(dot(cross(derivatives[0],derivatives[1]),p.source_direction));
   check(fabsf(fd/p.spreading-1)<.004f,"curved spreading vs launch FD");
   check(a.maslov_phase_cycles[k]==-.25f*a.morse_index[k],"Morse phase cycles");
   printf("m%d theta%.9g branch%d t%.9g opl%.15g spread%.12g morse%d rx%.9g ry%.9g\n",m,theta,k,a.angular_momentum[k],double(p.optical_length_split.x)+p.optical_length_split.y,p.spreading,a.morse_index[k],receiver.x,receiver.y);
  }
 }
 {
  const float3 source=make_float3(1.2f,0,0),receiver=make_float3(2*cosf(.9f),2*sinf(.9f),0);
  CoherentSphereInternalInventory virtual_paths;
  const float3 vs=make_float3(3.2f-source.x,0,0);
  check(coherent_sphere_internal_inventory(vs,receiver,-normalize(receiver),zero_float3(),1,1.5f,1,&virtual_paths)==COHERENT_SPHERE_TT_OK,"virtual internal inventory");
  for(int branch=0;branch<virtual_paths.count;branch++) {
   CoherentGeometryInterface frames[4]{};frames[0].center=make_float3(1.6f,0,0);
   frames[0].tangent_u=make_float3(0,1,0);frames[0].tangent_v=make_float3(0,0,1);frames[0].half_u=frames[0].half_v=10;
   CoherentGeometryPath p;float phase;
   check(coherent_unfold_internal_connect(source,receiver,-normalize(receiver),frames,4,1,1,branch,zero_float3(),1,1.5f,&p,&phase)==COHERENT_SPHERE_TT_OK,"planar mirror prefix plus internal reflection");
   check(p.count==4&&fabsf(p.point[0].x-1.6f)<2e-7f,"physical prefix point");
   double opl=double(p.optical_length_split.x)+p.optical_length_split.y;
   const auto vp=virtual_paths.path[branch];
   check(fabs(opl-(double(vp.optical_length_split.x)+vp.optical_length_split.y))<2e-11,"unfolded internal compensated OPL");
   check(p.spreading==vp.spreading&&phase==virtual_paths.maslov_phase_cycles[branch],"unfolded internal spreading and Morse");
  }
 }
 printf("%d checks %d failures\n",checks,failures);return failures?1:0;
}
