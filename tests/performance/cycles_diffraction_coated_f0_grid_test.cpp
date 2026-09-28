/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Test-only prototype: no cache ABI change. Scalar generalized powers are
 * integrated by enumerating all orders of independently sampled VNDF facets.
 * Grid interpolation is compared with direct parameters using identical
 * facets, separating interpolation bias from stochastic quadrature error.
 * Selected direct rows are separately checked by outgoing BSDF quadrature. */
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include "kernel/util/dielectric_f0_cache.h"
#include "scene/diffraction_albedo.h"
#include <cstdlib>
#include <string>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace ccl;
static float tested_inside_ior=1.5f;

static double generalized(double physical, double f0, double reference)
{
  if(reference<=1e-5)return physical;
  const double s=std::clamp((1-physical)/(1-reference),0.0,1.0);
  return std::clamp(physical*(1-s+s*f0/reference),0.0,1.0);
}

static float3 direction(float mu,float phi)
{
  const float r=std::sqrt(std::max(0.0f,1-mu*mu));
  return make_float3(r*std::cos(phi),r*std::sin(phi),std::max(mu,1e-5f));
}

static DiffractionDielectricGeneralizedExtra parameters(float nf,float d,float nm,int side,float f0)
{
  DiffractionDielectricGeneralizedExtra e{};
  diffraction_dielectric_parameters(nm,1200,125,.43f,side?tested_inside_ior:1.0f,
      side?1.0f:tested_inside_ior,.35f,.5f,&e.base.param);
  e.generalized_f0=make_spectrum(f0);e.generalized_reference_f0=F0_from_ior(tested_inside_ior);
  e.film_ior=nf;e.film_thickness_over_wavelength=d/nm;
  return e;
}

/* Deterministic h-space quadrature independent of the production discrete
 * order sampler. Every order's flux power is weighted by its masking escape. */
static std::vector<double> direct_q(float nf,float d,float nm,int side,float3 wi,
                                   const std::vector<double>&f0,int samples)
{
  auto e=parameters(nf,d,nm,side,0);
  auto&p=e.base.param;
  const float ni=p.facet.incident_ior,nt=p.facet.transmitted_ior,eta=ni/nt;
  const double li=diffraction_dielectric_lambda<GGX>(p.alpha_x,p.alpha_y,wi);
  std::vector<double> energy(f0.size(),0),other(f0.size());
  for(int sample=0;sample<samples;sample++) {
    const float u=(sample+.5f)/samples;
    const float v0=(sample+.5f)*.6180339887498949f;
    const float3 h=microfacet_ggx_sample_vndf(wi,p.alpha_x,p.alpha_y,make_float2(u,v0-std::floor(v0)));
    const float ci=dot(wi,h);
    std::fill(other.begin(),other.end(),0);
    const double Pi=diffraction_thin_film_reflectance(ci,ni,nt,nf,d/nm);
    for(int event=0;event<2;event++) {
      const bool trans=event!=0;
      const int bound=diffraction_dielectric_facet_active_order_bound(&p.facet,trans);
      for(int m=-bound;m<=bound;m++) {
        if(!trans && m==0)continue;
        float3 wo;
        const float delta=m*p.facet.wavelength_over_pitch;
        const bool valid=trans?diffraction_facet_transmit(wi,h,make_float3(1,0,0),eta,delta*eta,&wo):
            diffraction_facet_reflect(wi,h,make_float3(1,0,0),delta,&wo);
        if(!valid)continue;
        const float co=dot(wo,h);
        if(!(ci>0) || (trans?!(co<0):!(co>0)))continue;
        const double Po=diffraction_thin_film_reflectance(trans?-co:co,trans?nt:ni,trans?ni:nt,nf,d/nm);
        const double pair=trans && m==0?diffraction_thin_film_pair_reflectance(ci,-co,ni,nt,nf,d/nm):0;
        const double binary=diffraction_binary_power(m,trans?p.facet.transmission_phase:
            M_2PI_F*p.facet.height_over_wavelength*(ci+co),p.facet.duty);
        const bool escaped=trans?wo.z<0:wo.z>0;
        const double mask=escaped?(1+li)/(1+li+diffraction_dielectric_lambda<GGX>(p.alpha_x,p.alpha_y,wo)):0;
        for(size_t k=0;k<f0.size();k++) {
          const double Fi=generalized(Pi,f0[k],e.generalized_reference_f0);
          const double Fo=generalized(Po,f0[k],e.generalized_reference_f0);
          const double power=(trans?(m==0?1-generalized(pair,f0[k],e.generalized_reference_f0):
              std::min(1-Fi,1-Fo)):std::min(Fi,Fo))*binary;
          other[k]+=power;energy[k]+=power*mask;
        }
      }
    }
    const float3 wo=2*ci*h-wi;
    const double mask=wo.z>0?(1+li)/(1+li+diffraction_dielectric_lambda<GGX>(p.alpha_x,p.alpha_y,wo)):0;
    for(size_t k=0;k<f0.size();k++)energy[k]+=(1-other[k])*mask;
  }
  for(double&x:energy)x=1-x/samples;
  return energy;
}

static double cross(float nf,float d,float nm,double f0)
{
  constexpr int count=4096;double sum=0;
  for(int i=0;i<count;i++) {
    const float mu=(i+.5f)/count;
    const double P=diffraction_thin_film_reflectance(mu,1,tested_inside_ior,nf,d/nm);
    sum+=2*mu*(1-generalized(P,f0,F0_from_ior(tested_inside_ior)));
  }
  return sum/count;
}

static double projected(const std::vector<double>&v,int mu_count,int phi_count)
{
  double sum=0;
  for(int i=0;i<mu_count-1;i++) {
    const double mu=std::pow(double(i)/(mu_count-1),2);
    const double h=std::pow(double(i+1)/(mu_count-1),2)-mu;
    for(int j=0;j<phi_count;j++) {
      const double a=v[i*phi_count+j],delta=v[(i+1)*phi_count+j]-a;
      sum+=2*h*(mu*a+.5*(mu*delta+h*a)+h*delta/3);
    }
  }
  return sum/phi_count;
}

int main(int argc,char**argv)
{
  const bool high=argc>1;
  if(high)tested_inside_ior=std::strtof(argv[1],nullptr);
  for(float nd:{1.0f,1.00000011920928955f,1.5f,3.0f,3.1f,10.0f}) {
    float previous=-1;
    for(int node=0;node<16;node++) {
      const float value=dielectric_f0_cache_node(nd,16,node);
      const float coordinate=dielectric_f0_cache_coordinate(nd,16,value);
      if(!std::isfinite(value) || value<=previous || std::abs(coordinate-node)>2e-5f)return 2;
      previous=value;
    }
  }
  const int mus=high?3:8,phis=high?4:16,facets=512;
  double max_outgoing_error=0;
  for(int mode:{0,1,2}) for(int nodes:{8,16}) {
    if(high && (mode!=2 || nodes!=16))continue;
    const double reference=F0_from_ior(tested_inside_ior);
    if(mode==2 && nodes!=16)continue;
    std::vector<double> knots;
    for(int i=0;i<16;i++)knots.push_back(dielectric_f0_cache_node(tested_inside_ior,16,i));
    const auto coordinate=[&](double f){
      if(mode==2){const auto it=std::upper_bound(knots.begin(),knots.end(),f);const int i=std::clamp(int(it-knots.begin())-1,0,nodes-2);return (i+(f-knots[i])/(knots[i+1]-knots[i]))/(nodes-1);}
      return mode?f*(1+reference)/(f+reference):f;};
    const auto inverse=[&](double u){
      if(mode==2){const double x=u*(nodes-1);const int i=std::min(int(x),nodes-2);return knots[i]+(x-i)*(knots[i+1]-knots[i]);}
      return mode?u*reference/(1+reference-u):u;};
    const int tests=nodes-1+65;
    std::vector<double> f0;
    for(int i=0;i<nodes;i++)f0.push_back(inverse(double(i)/(nodes-1)));
    for(int i=0;i<nodes-1;i++)f0.push_back(inverse((i+.5)/(nodes-1)));
    for(int i=0;i<65;i++)f0.push_back(double(i)/64);
    for(auto film:{std::array<float,2>{1.32f,250},std::array<float,2>{2.4f,300}}) {
      const float nf=film[0],d=film[1];
      if(high && nf<2)continue;
      const int wavelength_nodes=nf<2?16:40;
      const float nm=550;
      const double lc=(nm-380)*(wavelength_nodes-1)/400.0;
      const int il=int(lc);const double lt=lc-il;
      const float l0=380+400.0f*il/(wavelength_nodes-1),l1=380+400.0f*(il+1)/(wavelength_nodes-1);
      double max_q=0,max_Q=0,max_cross=0,max_energy_error=0,max_f0_only_q=0,max_lambda_only_q=0,max_f0_only_cross=0,max_lambda_only_cross=0;
      for(int side=0;side<2;side++) {
        std::vector<std::vector<double>> actual(f0.size(),std::vector<double>(mus*phis));
        std::vector<std::vector<double>> grid(f0.size(),std::vector<double>(mus*phis));
        for(int i=0;i<mus;i++) for(int j=0;j<phis;j++) {
          const float mu=std::pow(float(i)/(mus-1),2),phi=M_2PI_F*(j+.5f)/phis;
          const auto wi=direction(mu,phi);
          const auto a=direct_q(nf,d,l0,side,wi,f0,facets),b=direct_q(nf,d,l1,side,wi,f0,facets);
          const auto reference=direct_q(nf,d,nm,side,wi,f0,facets);
          for(int k=0;k<tests;k++) {
            const double fc=coordinate(f0[nodes+k])*(nodes-1);
            const int fi=std::min(int(fc),nodes-1),fj=std::min(fi+1,nodes-1);const double ft=fc-fi;
            const double value=(1-lt)*((1-ft)*a[fi]+ft*a[fj])+lt*((1-ft)*b[fi]+ft*b[fj]);
            actual[nodes+k][i*phis+j]=reference[nodes+k];
            grid[nodes+k][i*phis+j]=value;
            max_q=std::max(max_q,std::abs(value-reference[nodes+k]));
            max_f0_only_q=std::max(max_f0_only_q,std::abs((1-ft)*reference[fi]+ft*reference[fj]-reference[nodes+k]));
            max_lambda_only_q=std::max(max_lambda_only_q,std::abs((1-lt)*a[nodes+k]+lt*b[nodes+k]-reference[nodes+k]));
            max_energy_error=std::max(max_energy_error,std::abs(value-reference[nodes+k]));
          }
        }
        for(int k=0;k<tests;k++) {
          const double ar=projected(actual[nodes+k],mus,phis),gr=projected(grid[nodes+k],mus,phis);
          max_Q=std::max(max_Q,std::abs(ar-gr));
        }
      }
      for(int k=0;k<tests;k++) {
        const double fc=coordinate(f0[nodes+k])*(nodes-1);
        const int fi=std::min(int(fc),nodes-1),fj=std::min(fi+1,nodes-1);const double ft=fc-fi;
        const double value=(1-lt)*((1-ft)*cross(nf,d,l0,f0[fi])+ft*cross(nf,d,l0,f0[fj]))+
                              lt*((1-ft)*cross(nf,d,l1,f0[fi])+ft*cross(nf,d,l1,f0[fj]));
        const double actual=cross(nf,d,nm,f0[nodes+k]);
        max_cross=std::max(max_cross,std::abs(value-actual));
        max_f0_only_cross=std::max(max_f0_only_cross,std::abs((1-ft)*cross(nf,d,nm,f0[fi])+ft*cross(nf,d,nm,f0[fj])-actual));
        max_lambda_only_cross=std::max(max_lambda_only_cross,std::abs((1-lt)*cross(nf,d,l0,f0[nodes+k])+lt*cross(nf,d,l1,f0[nodes+k])-actual));
      }
      /* Separate outgoing solid-angle quadrature verifies selected h-integrals,
       * rather than checking only internal cache normalization identities. */
      if(!high && nodes==16 && mode==0)for(int side=0;side<2;side++)for(float mu:{.2f,.65f}) {
        const auto wi=direction(mu,.7f);
        const std::vector<double> selected{.04,.1,.5};
        const auto q=direct_q(nf,d,nm,side,wi,selected,4096);
        for(size_t k=0;k<selected.size();k++) {
          auto e=parameters(nf,d,nm,side,float(selected[k]));double energy=0;
          constexpr int omus=96,ophis=192;
          for(int i=0;i<omus;i++)for(int j=0;j<ophis;j++) {
            const auto wo=direction((i+.5f)/omus,M_2PI_F*(j+.5f)/ophis);
            float pdf;
            energy+=average(diffraction_dielectric_generalized_eval<GGX>(&e,wi,wo,&pdf));
            energy+=average(diffraction_dielectric_generalized_eval<GGX>(&e,wi,make_float3(wo.x,wo.y,-wo.z),&pdf));
          }
          energy*=double(M_2PI_F)/(omus*ophis);
          max_outgoing_error=std::max(max_outgoing_error,std::abs(energy+q[k]-1));
        }
      }
      std::printf("bulkIOR %.6f mode%d nodes%d film%.2f/%.0f qerror %.6f Q_over_etendue_error %.6f crosserror %.6f energyerror %.6f full_default_trials %.0f f0onlyq %.6f lambdaonlyq %.6f f0onlycross %.6f lambdaonlycross %.6f\n",
          tested_inside_ior,mode,nodes,nf,d,max_q,max_Q,max_cross,max_energy_error,2.0*nodes*wavelength_nodes*8*16*512,max_f0_only_q,max_lambda_only_q,max_f0_only_cross,max_lambda_only_cross);
    }
  }
  if(!high)std::printf("independent_outgoing_vs_all_orders_h_integral max %.6f\n",max_outgoing_error);
  else std::puts("highIOR bounded interpolation check: outgoing reference not repeated");
  DiffractionTwoSidedAlbedoRequest request;
  request.generalized_f0_count=16;request.inside_ior=tested_inside_ior;
  request.film_ior=2.4f;request.film_thickness_nm=300;request.wavelength_count=40;
  request.mu_count=3;request.phi_count=4;request.facet_samples=128;
  DiffractionTwoSidedAlbedoTable table;std::string error;
  if(!diffraction_two_sided_albedo_build_cpu(request,table,error)) {
    std::fprintf(stderr,"actual CPU16cache failed: %s\n",error.c_str());return 3;
  }
  if(!diffraction_two_sided_albedo_validate(table,error))return 4;
  std::printf("actual CPU16coatedcache PASS %zu deficits\n",table.deficits.size());
  return max_outgoing_error<.025?0:1;
}
