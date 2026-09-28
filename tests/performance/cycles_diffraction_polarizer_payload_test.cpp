/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/closure/bsdf.h"
#include "kernel/integrator/surface_shader.h"
#include "util/profiling.h"
#include "scene/shader.tables"
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
using namespace ccl;
int main()
{
 int checks=0,bad=0;
 auto expect=[&](bool b,const char *name){checks++;if(!b){bad++;printf("FAIL %s\n",name);}};
 const float3 axis=normalize(make_float3(.3f,.7f,0));
 for(int kind=0;kind<4;kind++) {
   ShaderData sd{};sd.num_closure_left=8;sd.N=sd.Ng=sd.wi=make_float3(0,0,1);sd.rand_wavelength=.5f;
   bool ok=bsdf_diffraction_glass_setup(&sd,one_spectrum(),kind==2?make_spectrum(.7f):one_spectrum(),
       sd.N,make_float3(1,0,0),.6f,kind==3?1:1.5f,1150,320,.42f,1.32f,kind==1?250:0,false);
   expect(ok,"actual Glass grating setup");
   float3 read;
   for(int i=0;i<sd.num_closure;i++)expect(!bsdf_diffraction_polarizer_axis(&sd.closure[i],&read),"OFF metadata absent");
   const ShaderData before=sd;
   expect(bsdf_diffraction_glass_set_polarizer(&sd,0,axis),"attach exact graph8 capacity");
   expect(sd.num_closure==before.num_closure,"closure list preserved");
   for(int i=0;i<sd.num_closure;i++) {
     expect(bsdf_diffraction_polarizer_axis(&sd.closure[i],&read),"all grating closures filtered");
     expect(len(read-axis)<1e-7f,"physical axis exact");
     expect(sd.closure[i].weight==before.closure[i].weight,"native scalar weight unchanged");
     expect(sd.closure[i].sample_weight==before.closure[i].sample_weight,"native proposal unchanged");
   }
   // The full native sample still has exactly its original direction/PDF/value.
   if(kind==0)for(int sample=0;sample<16;sample++) {
     const float3 random=make_float3((sample+.5f)/16,.371f,.713f);
     Spectrum ea,eb;float3 oa,ob;float pa,pb,eta_a,eta_b;float2 ra,rb;
     int la=bsdf_diffraction_dielectric_sample(nullptr,&before.closure[0],before.Ng,before.wi,random,&ea,&oa,&pa,&ra,&eta_a);
     int lb=bsdf_diffraction_dielectric_sample(nullptr,&sd.closure[0],sd.Ng,sd.wi,random,&eb,&ob,&pb,&rb,&eta_b);
     expect(la==lb,"native sample label unchanged");expect(pa==pb,"native sample PDF unchanged");
     expect(ea==eb,"native sample value unchanged");expect(oa==ob,"native sample direction unchanged");
   }
 }
 // Attaching a needed inline extra with no room must mutate nothing.
 ShaderData exhausted{};exhausted.num_closure_left=1;exhausted.N=exhausted.Ng=exhausted.wi=make_float3(0,0,1);exhausted.rand_wavelength=.5f;
 expect(bsdf_diffraction_glass_setup(&exhausted,one_spectrum(),one_spectrum(),exhausted.N,
     make_float3(1,0,0),.6f,1.5f,1150,320,.42f,1,0,false),"one inline carrier before capacity failure");
 const ShaderData copy=exhausted;
 expect(!bsdf_diffraction_glass_set_polarizer(&exhausted,0,axis),"capacity failure reported");
 expect(std::memcmp(&copy,&exhausted,sizeof(copy))==0,"capacity failure transactional");
 // Actual native sampler: a delta branch cannot acquire continuous diffuse value.
 KernelGlobalsCPU base{};base.data.kernel_features=KERNEL_FEATURE_POLARIZATION;
 KernelObject fixture_object{};base.objects.data=&fixture_object;base.objects.width=1;
 std::vector<float> tables;
 auto add=[&](const float *v,int n){int offset=int(tables.size());tables.insert(tables.end(),v,v+n);return offset;};
 base.data.tables.ggx_E=add(table_ggx_E,1024);base.data.tables.ggx_Eavg=add(table_ggx_Eavg,32);
 base.data.tables.ggx_gen_schlick_ior_s=add(table_ggx_gen_schlick_ior_s,32768);
 base.data.tables.ggx_gen_schlick_s=add(table_ggx_gen_schlick_s,32768);
 base.data.tables.ggx_glass_E=add(table_ggx_glass_E,32768);
 base.data.tables.ggx_glass_Eavg=add(table_ggx_glass_Eavg,1024);
 base.data.tables.ggx_glass_inv_E=add(table_ggx_glass_inv_E,32768);
 base.data.tables.ggx_glass_inv_Eavg=add(table_ggx_glass_inv_Eavg,1024);
 base.lookup_table.data=tables.data();base.lookup_table.width=int(tables.size());
 Profiler profiler;ThreadKernelGlobalsCPU globals(base,nullptr,profiler,0);KernelGlobals kg=&globals;
 for(int kind=0;kind<5;kind++) {
   const bool transparent=kind==1;
   const bool rough_matched=kind>=2;
   const bool near_matched=kind>=3;
   const float fixture_ior=kind==3?1.f+1e-5f:kind==4?1.f-1e-5f:rough_matched?1.f:1.5f;
   ShaderData sd{};sd.num_closure_left=8;sd.N=sd.Ng=sd.wi=make_float3(0,0,1);
   sd.type=PRIMITIVE_TRIANGLE;
   ccl_private ShaderClosure *selected;
   if(transparent) {
     selected=bsdf_alloc(&sd,sizeof(ShaderClosure),make_spectrum(.5f));
     selected->type=CLOSURE_BSDF_TRANSPARENT_ID;selected->N=sd.N;
   }
   else {
     auto *b=(MicrofacetBsdf *)bsdf_alloc(&sd,sizeof(MicrofacetBsdf),make_spectrum(.5f));
     auto *f=(FresnelGeneralizedSchlickPolarizer *)closure_alloc_extra(&sd,sizeof(FresnelGeneralizedSchlickPolarizer));
     b->N=sd.N;b->T=zero_float3();b->ior=fixture_ior;b->alpha_x=b->alpha_y=rough_matched?.36f:0;
     sd.runtime_flag|=bsdf_microfacet_ggx_glass_setup(b);
     f->base.thin_film={0,1};f->base.f0=make_spectrum(F0_from_ior(fixture_ior));f->base.f90=one_spectrum();
     f->base.exponent=-fixture_ior;f->base.tint={one_spectrum(),one_spectrum()};
     bsdf_microfacet_setup_fresnel_generalized_schlick(kg,b,sd.wi,&f->base,false);
     bsdf_microfacet_set_polarizer(b,make_float3(1,0,0));selected=(ShaderClosure *)b;
   }
   ShaderData alone=sd;
   bsdf_diffuse_setup(&sd,sd.N,make_spectrum(.5f));
   BsdfEval value;float3 wo;float pdf,eta,avg;float2 rough;
   const int label=surface_shader_bsdf_sample_closure(kg,&sd,selected,make_float3(.3f,.4f,.8f),&value,&wo,&pdf,&rough,&eta,avg);
   expect(((label&(LABEL_SINGULAR|LABEL_TRANSPARENT))!=0)!=near_matched,"event measure matches exact versus near index");
   Spectrum raw;float3 raw_wo;float raw_pdf,raw_eta;float2 raw_rough;
   bsdf_sample(kg,&alone,&alone.closure[0],make_float3(.3f,.4f,.8f),&raw,&raw_wo,&raw_pdf,&raw_rough,&raw_eta);
   if(near_matched) {
     float diffuse_pdf;const Spectrum diffuse=bsdf_eval(kg,&sd,&sd.closure[1],wo,&diffuse_pdf)*sd.closure[1].weight;
     const Spectrum expected=raw*selected->weight+diffuse;
     expect(len(spectrum_to_rgb(bsdf_eval_sum(&value)-expected))<1e-6f,"near-index continuous mixture value");
     const auto response=polarization_surface_transport(kg,&sd,wo,polarization_unpolarized(),true,selected,bsdf_eval_sum(&value),false);
     const float target=average(safe_divide(.5f*raw*selected->weight+diffuse,expected));
     expect(isfinite(average(response.value[0])) && fabsf(average(response.value[0])-target)<2e-6f,"near-index continuous mixture polarization");
     continue;
   }
   const float mass_scale=transparent?1e6f:1.f;
   expect(len(spectrum_to_rgb(bsdf_eval_sum(&value)-raw*make_spectrum(.5f*mass_scale)))<.1f,"continuous lobe absent from atomic value");
   expect(fabsf(pdf/(raw_pdf*mass_scale)-.5f)<1e-6f,"atomic mixture PDF includes diffuse selection only");
   const auto sensitivity=polarization_surface_transport(kg,&sd,wo,polarization_unpolarized(),true,selected,bsdf_eval_sum(&value),true);
   expect(fabsf(average(sensitivity.value[0])-(transparent?1.f:.5f))<2e-6f,"actual delta polarized mixture response");
 }
 printf("polarizer payload checks=%d failures=%d extraBase=%zu coat=%zu MS=%zu closure=%zu\n",checks,bad,
        sizeof(DiffractionDielectricTintExtra),sizeof(DiffractionDielectricCoatingExtra),
        sizeof(DiffractionDielectricTwoSidedExtra),sizeof(ShaderClosure));
 return bad?1:0;
}
