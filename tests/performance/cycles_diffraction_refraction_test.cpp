/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf.h"
#include <complex>
#include <cstdio>
using namespace ccl;

int main()
{
  Profiler profiler;KernelGlobalsCPU base{};ThreadKernelGlobalsCPU kg(base,nullptr,profiler,0);
  KernelObject object{};kg.objects.data=&object;kg.objects.width=1;
  int failures=0,events=0,flat=0,reciprocal=0,atoms=0;
  float max_flat=0,max_reciprocal=0;
  for(bool beckmann:{false,true}) for(bool back:{false,true})
  for(float ior:{1.f,1.5f}) for(float roughness:{0.f,.2f,.6f}) for(float depth:{0.f,250.f}) {
    ShaderData sd{};sd.N=sd.Ng=make_float3(0,0,1);sd.wi=normalize(make_float3(.4f,.2f,1));
    sd.object=0;sd.rand_wavelength=.37f;sd.num_closure_left=8;
    if(back)sd.runtime_flag|=SR_BACKFACING;
    if(!bsdf_diffraction_refraction_setup(&sd,one_spectrum(),sd.N,make_float3(1,0,0),
                                        roughness,ior,1200,depth,.41f,beckmann))return 2;
    const ShaderClosure *sc=&sd.closure[0];
    MicrofacetBsdf native{};native.N=sd.N;native.alpha_x=native.alpha_y=sqr(roughness);
    native.ior=back?1/ior:ior;
    if(beckmann)bsdf_microfacet_beckmann_refraction_setup(&native);
    else bsdf_microfacet_ggx_refraction_setup(&native);
    for(int i=0;i<1000;++i) {
      Spectrum value;float3 wo;float pdf,eta;float2 rough;
      const int label=bsdf_sample(&kg,&sd,sc,make_float3((i+.5f)/1000,
          fmodf(i*.6180339887f+.3f,1.f),fmodf(i*.4142135623f+.2f,1.f)),
          &value,&wo,&pdf,&rough,&eta);
      if(label==LABEL_NONE)continue;
      ++events;failures+=!(label&LABEL_TRANSMIT) || wo.z>=0 || !(pdf>0) ||
                         !isfinite_safe(value) || reduce_min(value)<0;
      float query_pdf;
      const Spectrum query=(label&LABEL_SINGULAR)?bsdf_eval_delta(&kg,&sd,sc,wo,&query_pdf):
                                                             bsdf_eval(&kg,&sd,sc,wo,&query_pdf);
      failures+=fabsf(pdf-query_pdf)>3e-5f*max(1.f,pdf) ||
                reduce_max(fabs(value-query))>3e-5f*max(1.f,reduce_max(value));
      if(ior==1) {
        ++atoms;failures+=!(label&LABEL_SINGULAR) || len(sd.wi+wo)>1e-6f;
      }
      if(!(label&LABEL_SINGULAR)) {
        if(depth==0) {
          float native_pdf;
          const Spectrum reference=beckmann?
              bsdf_microfacet_eval<BECKMANN>(&kg,(const ShaderClosure *)&native,sd.wi,wo,&native_pdf):
              bsdf_microfacet_eval<GGX>(&kg,(const ShaderClosure *)&native,sd.wi,wo,&native_pdf);
          const float error=reduce_max(fabs(value-reference))/max(1.f,reduce_max(reference));
          max_flat=max(max_flat,error);failures+=error>3e-4f;++flat;
        }
        const auto *bsdf=(const DiffractionDielectricBsdf *)sc;
        auto reverse=*diffraction_dielectric_param(bsdf);
        const float ratio=reverse.facet.incident_ior/reverse.facet.transmitted_ior;
        diffraction_dielectric_parameters(1000*sample_wavelength(sd.rand_wavelength),1200,depth,.41f,
            reverse.facet.transmitted_ior,reverse.facet.incident_ior,
            reverse.alpha_x,reverse.alpha_y,&reverse);
        const float3 ri=make_float3(wo.x,-wo.y,-wo.z),ro=make_float3(sd.wi.x,-sd.wi.y,-sd.wi.z);
        float reverse_pdf;
        const float reversed=beckmann?
            diffraction_dielectric_eval<BECKMANN>(&reverse,ri,ro,&reverse_pdf,1,0,false,true):
            diffraction_dielectric_eval<GGX>(&reverse,ri,ro,&reverse_pdf,1,0,false,true);
        const float forward=average(value)/fabsf(wo.z)*sqr(ratio);
        const float backward=reversed/fabsf(ro.z);
        const float error=fabsf(forward-backward)/max(1.f,max(forward,backward));
        if(error>3e-4f) printf("reciprocity_failure beck=%d back=%d ior=%g rough=%g depth=%g i=%d "
            "wi=(%.9g,%.9g,%.9g) wo=(%.9g,%.9g,%.9g) f=%.9g reverse=%.9g error=%g\n",
            beckmann,back,ior,roughness,depth,i,sd.wi.x,sd.wi.y,sd.wi.z,wo.x,wo.y,wo.z,
            forward,backward,error);
        max_reciprocal=max(max_reciprocal,error);failures+=error>3e-4f;++reciprocal;
      }
    }
  }
  /* Independent double complex Fourier coefficients and tangential wavevectors.
   * The stratified facet sampler must realize their subunit propagating mass. */
  double max_mass_error=0;
  for(float ni:{1.f,1.5f}) for(float theta:{0.f,.6f,1.2f}) {
    DiffractionRoughDielectric p;
    const float no=ni==1?1.5f:1.f;
    if(!diffraction_dielectric_parameters(550,1200,250,.41f,ni,no,0,0,&p))return 3;
    const float3 wi=make_float3(sinf(theta),0,cosf(theta));
    const double phase=p.facet.transmission_phase,duty=p.facet.duty;
    const std::complex<double> step=std::exp(std::complex<double>(0,phase));
    double expected=0;
    for(int m=-20;m<=20;++m) {
      const double x=(m*550.0/1200-ni*wi.x)/no;
      if(x*x>=1)continue;
      const auto amplitude=m==0?std::complex<double>(1-duty)+duty*step:
          (step-std::complex<double>(1))*sin(M_PI*m*duty)/(M_PI*m);
      expected+=std::norm(amplitude);
    }
    int accepted=0;constexpr int trials=20000;
    for(int i=0;i<trials;++i) {
      float3 wo;float mass;int order;bool trans;
      if(diffraction_dielectric_facet_sample(&p.facet,wi,make_float3(0,0,1),
          (i+.5f)/trials,&wo,&mass,&order,&trans,false,1,0,true)) {
        ++accepted;failures+=!trans || wo.z>=0;
        failures+=fabsf(ni*wi.x+no*wo.x-order*550.f/1200)>3e-5f;
      }
    }
    const double error=fabs(double(accepted)/trials-expected);
    max_mass_error=std::max(max_mass_error,error);
    failures+=expected>1+1e-10 || error>1.1/trials;
  }
  /* Passivity in the kernel's throughput convention. Count null events in the
   * denominator: normalizing by accepted events would hide lost order power.
   * The stronger per-event bound checks the G2/G1 masking ratio as well. */
  int energy_cases=0,energy_events=0;
  double maximum_energy=0,minimum_energy=1;
  float maximum_throughput=0;
  for(bool beckmann:{false,true}) for(bool back:{false,true})
  for(float roughness:{0.f,.2f,.6f}) for(float depth:{0.f,250.f,1000.f})
  for(float cosine:{.05f,.5f,1.f}) {
    ShaderData sd{};sd.N=sd.Ng=make_float3(0,0,1);
    sd.wi=make_float3(sqrtf(1-cosine*cosine),0,cosine);
    sd.object=0;sd.rand_wavelength=.37f;sd.num_closure_left=8;
    if(back)sd.runtime_flag|=SR_BACKFACING;
    if(!bsdf_diffraction_refraction_setup(&sd,one_spectrum(),sd.N,make_float3(1,0,0),
                                        roughness,1.5f,1200,depth,.41f,beckmann))return 4;
    double sum=0;constexpr int count=2048;
    unsigned state=0x813527ad;
    auto uniform=[&]() {state=1664525u*state+1013904223u;return float(state>>8)*0x1p-24f;};
    for(int i=0;i<count;++i) {
      Spectrum value;float3 wo;float pdf,eta;float2 rough;
      const float x=uniform(),y=uniform(),z=uniform();
      const int label=bsdf_sample(&kg,&sd,&sd.closure[0],make_float3(x,y,z),
                                  &value,&wo,&pdf,&rough,&eta);
      if(label==LABEL_NONE)continue;
      ++energy_events;
      const float throughput=average(value)/pdf;
      maximum_throughput=max(maximum_throughput,throughput);
      failures+=!isfinite_safe(throughput) || throughput<0 || throughput>1.00001f;
      sum+=throughput;
    }
    const double energy=sum/count;
    maximum_energy=std::max(maximum_energy,energy);
    minimum_energy=std::min(minimum_energy,energy);
    failures+=energy>1.00001 || energy<0;
    ++energy_cases;
  }
  printf("passivity_cases=%d samples_per_case=2048 accepted=%d min_energy=%g "
         "max_energy=%g max_event_throughput=%g\n",energy_cases,energy_events,
         minimum_energy,maximum_energy,maximum_throughput);
  printf("events=%d flat=%d reciprocity=%d matched_atoms=%d max_flat=%g max_reciprocity=%g "
         "max_facet_mass_error=%g failures=%d\n",events,flat,reciprocal,atoms,max_flat,
         max_reciprocal,max_mass_error,failures);
  return failures!=0;
}
