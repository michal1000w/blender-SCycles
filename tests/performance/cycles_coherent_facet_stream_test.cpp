/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/light/coherent_facet_stream.h"
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
using namespace ccl;
static int checks=0, failures=0;
static void check(bool ok,const char *name){checks++;if(!ok){failures++;fprintf(stderr,"FAIL %s\n",name);}}
static CoherentCompletedPathField scalar(float amplitude,float length,float phase){
 CoherentCompletedPathField f{};f.radiance_amplitude_rgb=make_float3(amplitude,amplitude,amplitude);
 f.optical_length_split=make_float2(length,0);f.source_phase_cycles=phase;return f;
}
int main(){
 const float2 wavelength=make_float2(550e-9f,float(550e-9-double(550e-9f)));
 const float3 S=make_float3(0,0,1), D=make_float3(.4f,.1f,1.5f), N=make_float3(0,0,-1);
 float3 vertices[3]={make_float3(-2,-2,0),make_float3(4,-2,0),make_float3(-2,4,0)};
 CoherentGeometryInterface patch;CoherentGeometryPath path;
 check(coherent_facet_connect(S,D,N,vertices,make_uint3(0,1,2),&patch,&path),"triangle reflection");
 const double length=std::sqrt(double(D.x)*D.x+double(D.y)*D.y+6.25);
 check(std::abs(double(path.optical_length_split.x)+path.optical_length_split.y-length)<1e-11,"physical split OPL");
 check(std::abs(path.spreading-(2.5/length)/(length*length))<1e-7,"analytic spreading");
 vertices[0]=make_float3(3,3,0);vertices[1]=make_float3(4,3,0);vertices[2]=make_float3(3,4,0);
 check(!coherent_facet_connect(S,D,N,vertices,make_uint3(0,1,2),&patch,&path),"finite triangle miss");
 float3 q0[3]={make_float3(0,0,0),make_float3(1,0,0),make_float3(1,1,0)};
 float3 q1[3]={make_float3(0,0,0),make_float3(1,1,0),make_float3(0,1,0)};
 const float3 edge=make_float3(.5f,.5f,1);
 int owners=int(coherent_facet_connect(edge,edge,N,q0,make_uint3(0,1,2),&patch,&path))+
            int(coherent_facet_connect(edge,edge,N,q1,make_uint3(0,2,3),&patch,&path));
 check(owners==1,"exact shared diagonal has one facet owner");
 check(!coherent_facet_connect(S,make_float3(0,0,-1),N,q0,make_uint3(0,1,2),&patch,&path),"opposite reflection sides rejected");
 q0[1]=q0[0];
 check(!coherent_facet_connect(S,D,N,q0,make_uint3(0,1,2),&patch,&path),"degenerate facet rejected");
 // Independent coordinate-plane unfolding, including the collapsed-order corner.
 float3 floor[3]={make_float3(0,0,0),make_float3(4,0,0),make_float3(0,0,4)};
 float3 wall[3]={make_float3(0,0,0),make_float3(0,4,0),make_float3(0,0,4)};
 CoherentGeometryInterface pair[2];
 const float3 pair_source=make_float3(.3f,.2f,1);
 for(float y : {.66f,.666f,.6665f,.6666f,.6667f,.667f,.67f,.8f}) {
  const float3 receiver=make_float3(1,y,1.2f);
  const bool floor_first=double(y)*double(pair_source.x)>double(pair_source.y);
  const float3 *a=floor_first?floor:wall, *b=floor_first?wall:floor;
  bool ok=coherent_facet_pair_connect(pair_source,receiver,make_float3(-1,0,0),
      a,make_uint3(0,1,2),b,make_uint3(3,4,5),pair,&path);
  check(ok,"ordered finite pair near corner solved");
  if(ok) {
   const double sx=pair_source.x,sy=pair_source.y,sz=pair_source.z;
   const double dy=receiver.y,dz=receiver.z;
   const double tx=1/(1+sx),ty=dy/(dy+sy);
   // Intersections in unfolded space, then fold the earlier one back.
   const double x0=-(1-(1+sx)*ty), y1=dy-(dy+sy)*tx;
   const double z0=dz+(sz-dz)*ty,z1=dz+(sz-dz)*tx;
   float3 h0=floor_first?make_float3(float(x0),0,float(z0)):make_float3(0,float(-y1),float(z1));
   float3 h1=floor_first?make_float3(0,float(y1),float(z1)):make_float3(float(-x0),0,float(z0));
   check(len(path.point[0]-h0)<5e-7f && len(path.point[1]-h1)<5e-7f,"both hits independent unfolded reference");
   const double l=std::sqrt((1+sx)*(1+sx)+(dy+sy)*(dy+sy)+(dz-sz)*(dz-sz));
   const double opl_error=std::abs(double(path.optical_length_split.x)+path.optical_length_split.y-l);
   check(opl_error<5e-11,"two R physical split OPL");
   check(std::abs(path.spreading-(1+sx)/(l*l*l))<2e-6,"two R analytic Jacobian");
  }
  check(!coherent_facet_pair_connect(pair_source,receiver,make_float3(-1,0,0),
      b,make_uint3(3,4,5),a,make_uint3(0,1,2),pair,&path),"opposite ordered family rejected");
 }
 float3 far_wall[3]={make_float3(0,3,3),make_float3(0,4,3),make_float3(0,3,4)};
 check(!coherent_facet_pair_connect(pair_source,make_float3(1,.8f,1),N,
     floor,make_uint3(0,1,2),far_wall,make_uint3(3,4,5),pair,&path),"second finite triangle miss rejected");
 check(!coherent_facet_pair_connect(pair_source,make_float3(1,.8f,1),N,
     floor,make_uint3(0,1,2),floor,make_uint3(0,1,2),pair,&path),"repeated coplanar face not physical two R");
 check(coherent_facet_pair_connect(pair_source,make_float3(1,.8f,1.2f),make_float3(-1,0,0),
     floor,make_uint3(0,1,2),wall,make_uint3(3,4,5),pair,&path),"two R field path");
 CoherentPathDetectorFrame pair_frame;
 pair_frame.normal=make_float3(-1,0,0);
 make_orthonormals(pair_frame.normal,&pair_frame.u,&pair_frame.v);
 CoherentCompletedPathField pair_field;const bool mirrors[2]={true,true};
 check(coherent_path_field_transport(pair_source,make_float3(1,.8f,1.2f),pair_frame,&path,pair,
     mirrors,make_float3(1,1,1),make_float3(1,1,1),0,&pair_field),"two R Jones field transport");
 check(std::abs(pair_field.physical_diagonal_rgb.x-path.spreading/(4*M_PI*M_PI))<2e-8,
       "two R absolute diagonal and world-mode normalization");
 const int routes=83;
 CoherentCompletedPathField fields[routes];
 const float lc=.005f;
 double expected=0;
 for(int i=0;i<routes;i++) fields[i]=scalar(.01f*(1+i%3),.0002f*i,.17f*i);
 for(int i=0;i<routes;i++)for(int j=0;j<routes;j++){
  double opd=double(fields[i].optical_length_split.x)-fields[j].optical_length_split.x;
  double phase=2*M_PI*(opd/(double(wavelength.x)+wavelength.y)+double(fields[i].source_phase_cycles)-fields[j].source_phase_cycles);
  expected+=double(fields[i].radiance_amplitude_rgb.x)*fields[j].radiance_amplitude_rgb.x*std::exp(-.5*opd*opd/(double(lc)*lc))*std::cos(phase);
 }
 std::mt19937 rng(17421);std::normal_distribution<float> normal;
 double mean=0,m2=0;const int samples=32768;
 for(int sample=0;sample<samples;sample++){
  CoherentStreamField sum;coherent_stream_clear(&sum);float z=normal(rng);
  for(int i=0;i<routes;i++) coherent_stream_add(&sum,&fields[i],zero_float2(),wavelength,lc,z,i==0,true);
  float3 direct;float3 result=coherent_stream_finish(&sum,&direct);
  check(result.x>=0&&std::isfinite(result.x),"nonnegative finite sample");
  double delta=result.x-mean;mean+=delta/(sample+1);m2+=delta*(result.x-mean);
 }
 double stderr=std::sqrt(m2/(samples-1)/samples);
 check(std::abs(mean-expected)<8*stderr+1e-5,"83-route Gaussian explicit-pair expectation");
 double max_phase_error=0;
 // Tiny Lc and large OPD: independently evaluate represented input in double.
 for(float length : {1e-4f,1e-7f,1e-9f}) for(float z : {.371f,-1.9f,.001f}) {
  const float2 opd=make_float2(1.0f,1.27e-8f);
  auto actual=coherent_stream_phase(opd,wavelength,length,z,.13f);
  double phase=(double(opd.x)+opd.y)/(double(wavelength.x)+wavelength.y)+double(.13f)+
      double(z)*(double(opd.x)+opd.y)/(2*M_PI*double(length));
  double angle=2*M_PI*std::remainder(phase,1.0);
  max_phase_error=std::max(max_phase_error,std::max(std::abs(actual.x-std::cos(angle)),std::abs(actual.y-std::sin(angle))));
  check(std::abs(actual.x-std::cos(angle))<2e-4 && std::abs(actual.y-std::sin(angle))<2e-4,
        "large OPD tiny Lc compensated Gaussian phase");
 }
 double random_cross=0;
 for(int i=0;i<32768;i++) random_cross+=coherent_stream_phase(make_float2(1,0),wavelength,1e-9f,normal(rng),0).x;
 check(std::abs(random_cross/32768)<.03,"tiny Lc does not artificially recover coherence");
 // Infinite coherence: exact pair sum, signed half-cross passes, no random phase.
 CoherentStreamField sum;coherent_stream_clear(&sum);std::complex<double> total=0,direct_ref=0;
 for(int i=0;i<routes;i++){
  coherent_stream_add(&sum,&fields[i],zero_float2(),wavelength,1.0f,0,i==0,true);
  double cycles=double(fields[i].optical_length_split.x)/(double(wavelength.x)+wavelength.y)+fields[i].source_phase_cycles;
  auto a=double(fields[i].radiance_amplitude_rgb.x)*std::exp(std::complex<double>(0,2*M_PI*std::remainder(cycles,1.0)));
  total+=a;if(i==0)direct_ref+=a;
 }
 float3 direct;auto result=coherent_stream_finish(&sum,&direct);
 check(std::abs(result.x-std::norm(total))<2e-5,"infinite-Lc explicit complex sum");
 check(std::abs(direct.x-std::real(direct_ref*std::conj(total)))<2e-6,"signed direct half-cross");
 // Cancellation and high dynamic range: (1e8 + 1 - 1e8)^2 = 1.
 coherent_stream_clear(&sum);
 auto a=scalar(1e8f,0,0), b=scalar(1,0,0), c=scalar(1e8f,0,.5f);
 coherent_stream_add(&sum,&a,zero_float2(),wavelength,1,0,false,true);
 coherent_stream_add(&sum,&b,zero_float2(),wavelength,1,0,false,true);
 coherent_stream_add(&sum,&c,zero_float2(),wavelength,1,0,false,true);
 result=coherent_stream_finish(&sum,&direct);
 check(result.x==1.0f,"compensated high-dynamic-range cancellation");
 printf("{\"checks\":%d,\"failures\":%d,\"routes\":83,\"samples\":%d,\"pair_reference\":%.10g,\"sample_mean\":%.10g,\"standard_error\":%.10g,\"tiny_lc_mean_cosine\":%.10g,\"max_phase_component_error\":%.10g}\n",checks,failures,samples,expected,mean,stderr,random_cross/32768,max_phase_error);
 return failures?1:0;
}
