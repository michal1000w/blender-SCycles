/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/light/coherent_passes.h"
#include <cmath>
#include <complex>
#include <cstdio>
using namespace ccl;
static int checks=0, failures=0;
static void check(bool ok,const char *name){checks++;if(!ok){failures++;std::fprintf(stderr,"FAIL %s\n",name);}}
static void near(Spectrum a,Spectrum b,const char *name){check(len(a-b)<2e-6f,name);}
static Spectrum sum(const CoherentSurfaceLightPasses &p){return p.diffuse_direct+p.diffuse_indirect+p.glossy_direct+p.glossy_indirect+p.transmission_direct+p.transmission_indirect;}
int main(){
  // Independent complex-field oracle, including destructive interference that
  // makes the chosen direct pass negative while total intensity stays positive.
  for(auto amplitudes:{std::pair{std::complex<double>(1,0),std::complex<double>(-2,0)},
                       std::pair{std::complex<double>(1,1),std::complex<double>(.3,-.7)}}){
    const float diagonal_a=std::norm(amplitudes.first),diagonal_b=std::norm(amplitudes.second);
    const float pair=2*std::real(amplitudes.first*std::conj(amplitudes.second));
    for(bool da:{false,true})for(bool db:{false,true}){
      const Spectrum total=make_spectrum(diagonal_a+diagonal_b+pair);
      const Spectrum direct=make_spectrum((da?diagonal_a:0)+(db?diagonal_b:0))+coherent_pass_direct_pair_share(make_spectrum(pair),da,db);
      const double expected_direct=(da?std::norm(amplitudes.first):0)+(db?std::norm(amplitudes.second):0)+(.5*(int(da)+int(db)))*2*std::real(amplitudes.first*std::conj(amplitudes.second));
      near(direct,make_spectrum(expected_direct),"half-pair independent complex oracle");
      const auto primary=coherent_pass_surface_split(total,direct,0,zero_spectrum(),zero_spectrum());
      near(sum(primary),total,"primary stored-pass recomposition");
      near(primary.glossy_direct+primary.glossy_indirect+primary.transmission_direct+primary.transmission_indirect,zero_spectrum(),"primary pure diffuse attribution");
      if(da&&!db&&diagonal_a==1&&diagonal_b==4)check(primary.diffuse_direct.x<0,"signed direct preserved");
      for(auto weights:{std::pair{1.f,0.f},std::pair{0.f,1.f},std::pair{0.f,0.f},std::pair{.3f,.4f}}){
        const auto secondary=coherent_pass_surface_split(total,direct,1,make_spectrum(weights.first),make_spectrum(weights.second));
        near(sum(secondary),total,"secondary first-scatter weighted recomposition");
        near(secondary.diffuse_direct+secondary.glossy_direct+secondary.transmission_direct,zero_spectrum(),"secondary all indirect");
        near(secondary.diffuse_indirect,make_spectrum(weights.first)*total,"secondary diffuse weight independent oracle");
        near(secondary.glossy_indirect,make_spectrum(weights.second)*total,"secondary glossy weight independent oracle");
      }
    }
  }
  // Native compositing multiplies Color into albedo-divided lighting passes.
  const Spectrum color=make_float3(.25,.5,.75),radiance=make_float3(.1,.3,.9);
  const auto p=coherent_pass_surface_split(radiance,color*.2f,0,zero_spectrum(),zero_spectrum());
  near(color*(p.diffuse_direct/color+p.diffuse_indirect/color),radiance,"RGB native Color/direct/indirect recomposition");
  std::printf("checks %d failures %d\n",checks,failures);return failures?1:0;
}
