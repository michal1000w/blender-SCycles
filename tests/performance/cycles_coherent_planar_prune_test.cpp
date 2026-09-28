/* SPDX-License-Identifier: Apache-2.0 */
#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "scene/coherent_planar_prune.h"
#include <cstdio>
using namespace ccl;
using P=CoherentPlanarPoint;
static int checks=0,failures=0;
static void check(bool v,const char*s){checks++;if(!v){printf("FAIL %s\n",s);failures++;}}
static bool reference_hit(const P&S,const P&R,const CoherentPlanarTriangle&f){
 using namespace coherent_planar_detail;auto n=cross(sub(f.point[1],f.point[0]),sub(f.point[2],f.point[0]));double nn=dot(n,n),ds=dot(sub(S,f.point[0]),n),dr=dot(sub(R,f.point[0]),n);if(ds*dr<=0)return false;
 P image;for(int i=0;i<3;i++)image[i]=S[i]-2*ds/nn*n[i];auto d=sub(R,image);double t=dot(sub(f.point[0],image),n)/dot(d,n);if(!(t>0&&t<1))return false;P x;for(int i=0;i<3;i++)x[i]=image[i]+t*d[i];
 for(int i=0;i<3;i++)if(dot(cross(sub(f.point[(i+1)%3],f.point[i]),sub(x,f.point[i])),n)<-1e-14*nn)return false;return true;
}
int main(){
 CoherentPlanarTriangle f{{P{-1,-1,0},P{1,-1,0},P{0,1,0}},{0,1,2},0};
 CoherentPlanarTriangle d{{P{-.1,-.1,2},P{.1,-.1,2},P{0,.1,2}},{0,1,2},0};P source{0,0,2};
 check(coherent_planar_reflection_may_reach(source,f,d),"central real reflection retained");
 for(auto&p:d.point)p[0]+=10;
 check(!coherent_planar_reflection_may_reach(source,f,d),"separated reflection footprint pruned");
 for(auto&p:d.point)p[2]=-2;
 check(!coherent_planar_reflection_may_reach(source,f,d),"opposite halfspace pruned");
 d.point[0][2]=2;check(coherent_planar_reflection_may_reach(source,f,d),"horizon crossing conservatively retained");
 check(coherent_planar_reflection_may_reach(P{0,0,0},f,d),"source plane grazing retained");
 std::vector<P>vertices;vertices.push_back(P{0,0,1});
 const int sides=16,rings=7;
 for(int ring=1;ring<=rings;ring++)for(int side=0;side<sides;side++){
  double theta=3.141592653589793*ring/(rings+1),phi=2*3.141592653589793*side/sides;
  vertices.push_back(P{float(sin(theta)*cos(phi)),float(sin(theta)*sin(phi)),float(cos(theta))});
 }
 vertices.push_back(P{0,0,-1});std::vector<CoherentPlanarTriangle>facets;
 auto add=[&](int a,int b,int c){facets.push_back({{vertices[a],vertices[b],vertices[c]},{a,b,c},int(facets.size())});};
 for(int side=0;side<sides;side++){int next=(side+1)%sides;add(0,1+side,1+next);
  for(int ring=0;ring<rings-1;ring++){int a=1+ring*sides+side,b=1+ring*sides+next,c=a+sides,e=b+sides;add(a,c,b);add(b,c,e);}
  add(1+(rings-1)*sides+side,int(vertices.size()-1),1+(rings-1)*sides+next);
 }
 source=P{0,0,3};
 P x{};for(int k=0;k<3;k++)x[k]=(facets[0].point[0][k]+facets[0].point[1][k]+facets[0].point[2][k])/3;
 using namespace coherent_planar_detail;
 const auto n=cross(sub(facets[0].point[1],facets[0].point[0]),sub(facets[0].point[2],facets[0].point[0]));
 const auto incoming=sub(x,source);P outgoing,R;
 for(int k=0;k<3;k++)outgoing[k]=incoming[k]-2*dot(incoming,n)/dot(n,n)*n[k];
 const double receiver_t=(3-x[2])/outgoing[2];
 for(int k=0;k<3;k++)R[k]=x[k]+receiver_t*outgoing[k];
 d.point={P{R[0]-.05,R[1]-.05,3},P{R[0]+.05,R[1]-.05,3},P{R[0],R[1]+.05,3}};
 int retained=0,true_hits=0;
 for(const auto&facet:facets){bool may=coherent_planar_reflection_may_reach(source,facet,d);retained+=may;
  for(int i=0;i<=12;i++)for(int j=0;j<=12-i;j++){
   P r;for(int k=0;k<3;k++)r[k]=(d.point[0][k]*(12-i-j)+d.point[1][k]*i+d.point[2][k]*j)/12;
   if(reference_hit(source,r,facet)){true_hits++;check(may,"actual facet root never pruned");}
  }
 }
 check(facets.size()>64&&facets.size()<=256,"ordinary faceted sphere exceeds old patch budget");
 check(retained>0&&retained<64,"pruned exact facets fit existing candidate budget");
 check(true_hits>0,"independent sampled facet roots exist");
 printf("facets%zu retained%d sampled roots%d checks%d failures%d\n",facets.size(),retained,true_hits,checks,failures);return failures?1:0;
}
