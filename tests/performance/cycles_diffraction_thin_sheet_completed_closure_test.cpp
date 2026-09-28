/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#define main previous_thin_wall_fixture_main
#include "cycles_diffraction_thin_wall_native_test.cpp"
#undef main
#include "scene/diffraction_albedo.h"
#include "kernel/svm/closure.h"
#include <cstring>
using namespace ccl;
static Spectrum completed_eval(KernelGlobals kg, ShaderData *sd, float3 wo)
{
 Spectrum value=zero_spectrum();
 for(int i=0;i<sd->num_closure;++i) {
  auto *sc=&sd->closure[i];float pdf=0;Spectrum v=zero_spectrum();
  if(sc->type==CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_REFLECTION_ID||sc->type==CLOSURE_BSDF_DIFFRACTION_THIN_SHEET_TRANSMISSION_ID)
   v=bsdf_diffraction_thin_sheet_eval(kg,sc,sd->wi,wo,&pdf);
  else if(sc->type==CLOSURE_BSDF_MICROFACET_GGX_ID) v=bsdf_microfacet_ggx_eval(kg,sc,sd->wi,wo,&pdf);
  else if(sc->type==CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID) v=bsdf_thin_glass_transmission_eval(kg,sc,sd->wi,wo,&pdf);
  value+=sc->weight*v;
 }
 return value;
}
int main()
{
 std::vector<float> lut(table_ggx_E,table_ggx_E+1024);lut.insert(lut.end(),table_ggx_Eavg,table_ggx_Eavg+32);
 KernelGlobalsCPU base{};base.lookup_table.data=lut.data();base.lookup_table.width=lut.size();base.data.tables.ggx_E=0;base.data.tables.ggx_Eavg=1024;
 Profiler profiler;ThreadKernelGlobalsCPU thread(base,nullptr,profiler,0);KernelGlobals kg=&thread;
 int checks=0,failures=0;double max_energy=0,max_value=0,max_pdf=0;
 for(float alpha : {.36f,1.f,0.f}) {
  DiffractionAlbedoRequest request;request.thin_sheet=true;request.alpha_x=alpha;request.alpha_y=bsdf_thin_glass_transmission_roughness(alpha,1.5f);
  request.pitch_nm=1150;request.depth_nm=175;request.duty=.42f;request.mu_count=24;request.phi_count=16;request.wavelength_count=16;request.facet_samples=512;
  DiffractionAlbedoTable table;std::string error;
  if(!diffraction_albedo_build_cpu(request,table,error)){printf("cache_error %s\n",error.c_str());return 1;}
  int4 descriptor=make_int4(0,0,request.mu_count,request.phi_count);float4 domain=make_float4(380,780,request.wavelength_count,1);
  thread.diffraction_albedo_values.data=table.values.data();thread.diffraction_albedo_values.width=table.values.size();
  thread.diffraction_albedo_averages.data=table.averages.data();thread.diffraction_albedo_averages.width=table.averages.size();
  thread.diffraction_albedo_descriptors.data=&descriptor;thread.diffraction_albedo_descriptors.width=1;
  thread.diffraction_albedo_domains.data=&domain;thread.diffraction_albedo_domains.width=1;thread.data.tables.num_diffraction_albedo_caches=1;
  for(float mu : {.2f,.6f,.95f}) {
   float3 wi=normalize(make_float3(sqrtf(1-mu*mu)*cosf(.33f),sqrtf(1-mu*mu)*sinf(.33f),mu));
   ShaderData sd=make_shader_data(wi,make_float3(0,0,1));
   auto setup=[&](ShaderData *out,float amount,int capacity,float depth){out->num_closure_left=capacity;
    bsdf_diffraction_thin_glass_setup(kg,out,true,true,{one_spectrum(),one_spectrum()},one_spectrum(),out->N,make_float3(1,0,0),alpha,1.5f,{0,1.32f},PATH_RAY_VISIBILITY_CAMERA,0,amount,1150,depth,.42f,0,make_int3(0,0,0));};
   setup(&sd,1,6,175);checks++;if(sd.num_closure!=4||sd.num_closure_left!=0)failures++;
   for(float coverage : {.5f,1.f}) {
    int capacity=coverage<1?8:6;
    ShaderData exact=make_shader_data(wi,sd.N),small=make_shader_data(wi,sd.N);setup(&exact,coverage,capacity,175);setup(&small,coverage,capacity-1,175);
    checks++;if(exact.num_closure!=(coverage<1?6:4)||small.num_closure!=0||small.num_closure_left!=capacity-1)failures++;
   }
   for(int c=0;c<sd.num_closure;++c) {
    const auto *sc=&sd.closure[c];auto *b=(const DiffractionThinSheetBsdf *)sc;
    float old_alpha=b->port.alpha;bsdf_blur((ShaderClosure *)sc,.99f);checks++;if(b->port.alpha!=old_alpha)failures++;
    for(int sample=0;sample<256;++sample) {
     float3 random=make_float3((sample+.5f)/256,fmodf((sample+.5f)*.61803398875f,1.f),fmodf((sample+.5f)*.41421356237f,1.f));
     Spectrum sampled;float3 wo;float pdf,eta;float2 rough;
     int label=bsdf_diffraction_thin_sheet_sample(kg,sc,sd.N,wi,random,&sampled,&wo,&pdf,&rough,&eta);
     if(label==LABEL_NONE)continue;
     float evaluated_pdf;Spectrum evaluated=(label&LABEL_SINGULAR)?bsdf_diffraction_thin_sheet_delta(kg,sc,wi,wo,&evaluated_pdf):bsdf_diffraction_thin_sheet_eval(kg,sc,wi,wo,&evaluated_pdf);
     double pe=std::abs(double(pdf)-evaluated_pdf)/(1+pdf),ve=reduce_max(fabs(sampled-evaluated))/(1+reduce_max(fabs(sampled)));max_pdf=std::max(max_pdf,pe);max_value=std::max(max_value,ve);
     checks++;if(pe>1e-5||ve>1e-5||!(pdf>0)||!std::isfinite(reduce_max(sampled)))failures++;
    }
   }
   if(alpha>0) {
    constexpr int M=96,P=192;double energy=0;
    for(int side : {1,-1})for(int m=0;m<M;++m)for(int p=0;p<P;++p){float cos=(m+.5f)/M,phi=2*M_PI_F*(p+.5f)/P,r=sqrtf(1-cos*cos);energy+=average(completed_eval(kg,&sd,make_float3(r*cosf(phi),r*sinf(phi),side*cos)));}
    energy*=2*M_PI/(M*P);max_energy=std::max(max_energy,std::abs(energy-1));checks++;if(std::abs(energy-1)>.02)failures++;
    printf("alpha %.3f mu %.3f completed_energy %.9f\n",alpha,mu,energy);
   }
   else {
    double energy=0;
    for(int c=0;c<sd.num_closure;++c) {
     const auto *b=(const DiffractionThinSheetBsdf *)&sd.closure[c];
     if(b->return_only) {
      Spectrum avg;float3 local=wi;
      energy+=average(b->weight*diffraction_thin_sheet_cached_missing(kg,b->extra,local,&avg))*.5f*b->extra->blend;
      continue;
     }
     float3 I=b->port.transmission ? diffraction_thin_sheet_flip_tangent(wi) : wi;
     for(int m=-diffraction_thin_sheet_order_bound(&b->port);m<=diffraction_thin_sheet_order_bound(&b->port);++m) {
      float3 O;if(!diffraction_facet_reflect(I,make_float3(0,0,1),make_float3(1,0,0),m*b->port.wavelength_over_pitch,&O)||O.z<=0)continue;
      float3 out=b->port.transmission ? -O : O;float pdf;
      energy+=average(b->weight*bsdf_diffraction_thin_sheet_delta(kg,(const ShaderClosure *)b,wi,out,&pdf))*1e-6;
     }
    }
    checks++;max_energy=std::max(max_energy,std::abs(energy-1));if(std::abs(energy-1)>.02)failures++;
    printf("alpha_singular mu %.3f complete_atom_plus_return_energy %.9f\n",mu,energy);
   }
   if (mu==.6f && alpha==.36f) {
    // Allocation-only SVM graph coverage; film table reuse here is deliberate,
    // and is not a coated energy gate.
    const auto scalar=[](float v){return SVMInputFloat{__float_as_uint(v)};};
    const auto triple=[&](float x,float y,float z){return SVMInputFloat3{scalar(x),scalar(y),scalar(z)};};
    for(float coverage : {0.f,.5f,1.f}) for(float film : {0.f,250.f}) {
     SVMNodePrincipledBsdfData data{};
     data.distribution=CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID;
     data.ior=scalar(1.5f);data.roughness=scalar(sqrtf(alpha));data.sheen_weight=scalar(.2f);data.coat_weight=scalar(.2f);
     data.metallic=scalar(.3f);data.transmission_weight=scalar(.4f);data.subsurface_weight=scalar(.2f);data.base_color=triple(1,1,1);
     data.alpha=scalar(1);data.diffuse_roughness=scalar(.2f);data.normal_offset=data.coat_normal_offset=SVM_STACK_INVALID;data.tangent_offset=0;
     data.specular_tint=triple(1,1,1);data.specular_ior_level=scalar(.5f);data.anisotropic=scalar(0);data.anisotropic_rotation=scalar(0);
     data.transmission_dispersion_scale=scalar(0);data.transmission_dispersion_abbe_number=scalar(20);
     data.emission_color=triple(0,0,0);data.emission_strength=scalar(0);data.sheen_tint=triple(1,1,1);data.sheen_roughness=scalar(.5f);
     data.coat_tint=triple(1,1,1);data.coat_roughness=scalar(.2f);data.coat_ior=scalar(1.5f);
     data.subsurface_method=CLOSURE_BSSRDF_RANDOM_WALK_ID;data.subsurface_radius=triple(1,1,1);data.subsurface_scale=scalar(.1f);
     data.subsurface_ior=scalar(1.5f);data.subsurface_anisotropy=scalar(0);data.thin_film_thickness=scalar(film);data.thin_film_ior=scalar(1.32f);
     data.thin_wall={1,SVM_STACK_INVALID,{0,0,0}};data.diffraction_weight=scalar(coverage);data.diffraction_pitch=scalar(1150);
     data.diffraction_depth=scalar(175);data.diffraction_duty=scalar(.42f);data.diffraction_albedo_handle=data.diffraction_two_sided_handle=-1;
     data.diffraction_thin_sheet_handle_r=data.diffraction_thin_sheet_handle_g=data.diffraction_thin_sheet_handle_b=0;
     std::vector<uint> words(sizeof(data)/sizeof(uint));std::memcpy(words.data(),&data,sizeof(data));
     thread.svm_nodes.data=words.data();thread.svm_nodes.width=words.size();thread.data.integrator.caustics_reflective=true;thread.data.integrator.caustics_refractive=true;
     SVMNodeClosureBsdf node{CLOSURE_BSDF_PRINCIPLED_ID,SVM_STACK_INVALID,{0,0,0}};
     ShaderData roomy=make_shader_data(wi,sd.N),reserved=make_shader_data(wi,sd.N);
     for(ShaderData *out : {&roomy,&reserved}) {
      out->num_closure_left=out==&roomy?MAX_CLOSURE:24;float stack[SVM_STACK_SIZE]={1,0,0};
      svm_node_closure_bsdf<~uint64_t(0),SHADER_TYPE_SURFACE>(kg,out,stack,one_spectrum(),node,PATH_RAY_VISIBILITY_CAMERA,0,0);
     }
     int used=MAX_CLOSURE-roomy.num_closure_left;
     checks++;if(used>24||reserved.num_closure_left!=24-used||roomy.num_closure!=reserved.num_closure)failures++;
     for(int c=0;c<roomy.num_closure;++c){checks++;if(roomy.closure[c].type!=reserved.closure[c].type||reduce_max(fabs(roomy.closure[c].weight-reserved.closure[c].weight))>1e-7f)failures++;}
     printf("svm_graph24 coverage %.1f film %.0f used %d closures %d reserved %d\n",coverage,film,used,roomy.num_closure,reserved.num_closure);
    }
   }
   // Exact depth zero uses the native setup even with valid cache handles.
   ShaderData zero=make_shader_data(wi,sd.N),native=make_shader_data(wi,sd.N);setup(&zero,1,6,0);
   bsdf_thin_glass_setup(kg,&native,true,true,{one_spectrum(),one_spectrum()},one_spectrum(),sd.N,alpha,1.5f,{0,1.32f},PATH_RAY_VISIBILITY_CAMERA,0);
   checks++;if(zero.num_closure!=native.num_closure)failures++;
  }
 }
 // Tiny-relief continuity uses a real matching dispersed cache. The endpoint
 // is the original native d-line sheet, not a dispersed compatibility carrier.
 DiffractionAlbedoRequest tiny_request;tiny_request.thin_sheet=true;tiny_request.alpha_x=.36f;
 tiny_request.alpha_y=bsdf_thin_glass_transmission_roughness(.36f,1.5f);tiny_request.inside_ior=1.5f;tiny_request.inv_abbe=.05f;
 tiny_request.pitch_nm=1150;tiny_request.depth_nm=.1f;tiny_request.duty=.42f;tiny_request.wavelength_count=2;tiny_request.mu_count=8;tiny_request.phi_count=8;tiny_request.facet_samples=64;
 DiffractionAlbedoTable tiny_table;std::string tiny_error;
 if(!diffraction_albedo_build_cpu(tiny_request,tiny_table,tiny_error)){printf("tiny_cache_error %s\n",tiny_error.c_str());return 1;}
 int4 td=make_int4(0,0,8,8);float4 tw=make_float4(380,780,2,1);
 thread.diffraction_albedo_values.data=tiny_table.values.data();thread.diffraction_albedo_values.width=tiny_table.values.size();
 thread.diffraction_albedo_averages.data=tiny_table.averages.data();thread.diffraction_albedo_averages.width=tiny_table.averages.size();
 thread.diffraction_albedo_descriptors.data=&td;thread.diffraction_albedo_descriptors.width=1;
 thread.diffraction_albedo_domains.data=&tw;thread.diffraction_albedo_domains.width=1;
 double max_tiny_error=0;
 for(float wavelength_random : {.1f,.5f,.9f}) for(float depth : {.0001f,.001f,.01f,.1f}) {
  float3 wi=normalize(make_float3(.24f,-.09f,1));ShaderData candidate=make_shader_data(wi,make_float3(0,0,1)),native=make_shader_data(wi,make_float3(0,0,1));
  candidate.rand_wavelength=native.rand_wavelength=wavelength_random;
  bsdf_diffraction_thin_glass_setup(kg,&candidate,true,true,{one_spectrum(),one_spectrum()},one_spectrum(),candidate.N,make_float3(1,0,0),.36f,1.5f,{0,1.32f},PATH_RAY_VISIBILITY_CAMERA,0,1,1150,depth,.42f,.05f,make_int3(0,0,0));
  bsdf_thin_glass_setup(kg,&native,true,true,{one_spectrum(),one_spectrum()},one_spectrum(),native.N,.36f,1.5f,{0,1.32f},PATH_RAY_VISIBILITY_CAMERA,0);
  for(float side : {1.f,-1.f}) {
   float3 wo=normalize(make_float3(-.17f,.05f,side));Spectrum actual=completed_eval(kg,&candidate,wo),expected=completed_eval(kg,&native,wo);
   double error=reduce_max(fabs(actual-expected))/(1+reduce_max(fabs(expected)));max_tiny_error=std::max(max_tiny_error,error);checks++;if(error>1e-4||!std::isfinite(error))failures++;
  }
 }
 printf("tiny_depth_dispersion_native_limit_relative %.9g\n",max_tiny_error);
 printf("checks %d failures %d max_energy_error %.9g sample_eval_value_error %.9g pdf_error %.9g closure_size %zu extra_size %zu slot %zu\n",checks,failures,max_energy,max_value,max_pdf,sizeof(DiffractionThinSheetBsdf),sizeof(DiffractionThinSheetExtra),sizeof(ShaderClosure));return failures!=0;
}
