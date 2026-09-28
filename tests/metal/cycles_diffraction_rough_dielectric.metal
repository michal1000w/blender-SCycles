/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/device/metal/globals.h"
#include "kernel/tables.h"
#ifdef DIFFRACTION_TEST_SURFACE_MIXTURE
#include "kernel/sample/guiding_field.h"
#endif
#include "kernel/device/metal/context_begin.h"
#ifdef DIFFRACTION_TEST_SURFACE_MIXTURE
#include "kernel/integrator/surface_shader.h"
#else
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#endif
#include "kernel/device/metal/context_end.h"
#ifdef DIFFRACTION_TEST_BECKMANN
#define TEST_DISTRIBUTION MetalKernelContext::BECKMANN
#else
#define TEST_DISTRIBUTION MetalKernelContext::GGX
#endif
kernel void diffraction_context_size(device uint *result [[buffer(0)]])
{ result[0]=sizeof(KernelParamsMetal);result[1]=sizeof(KernelObject); }
#ifdef DIFFRACTION_TEST_SURFACE_MIXTURE
kernel void diffraction_bind_objects(device KernelParamsMetal &params [[buffer(0)]],
                                    device const KernelObject *objects [[buffer(1)]])
{ params.objects=objects; }
#endif
kernel void diffraction_rough_dielectric(device const float4 *input [[buffer(0)]],
                                         device float4 *output [[buffer(1)]],
                                         constant KernelParamsMetal &params [[buffer(2)]],
                                         uint i [[thread_position_in_grid]])
{
 MetalKernelContext context(params);
 const float4 a=input[4*i],b=input[4*i+1],c=input[4*i+2],d=input[4*i+3];
 MetalKernelContext::DiffractionRoughDielectric p{{c.x,c.y,c.z,b.z,b.w,c.w},b.x,b.y};
 float3 wo;float value,pdf;bool singular=false;
#ifdef DIFFRACTION_TEST_COATED
 const bool valid=context.diffraction_dielectric_sample_coated_continuous<TEST_DISTRIBUTION>(
     &p,a.xyz,float3(a.w,d.x,d.y),1.35f,.47f,&wo,&value,&pdf);
#else
 const bool valid=context.diffraction_dielectric_sample<TEST_DISTRIBUTION>(&p,a.xyz,float3(a.w,d.x,d.y),&wo,&value,&pdf,&singular);
#endif
 output[2*i]=float4(wo,value);
 output[2*i+1]=float4(pdf,float(valid),float(singular),0);
}

/* A separate dispatch forces sampled directions through device memory, as they
 * are between renderer kernels, instead of permitting compiler expression reuse. */
kernel void diffraction_rough_dielectric_reevaluate(
    device const float4 *input [[buffer(0)]],
    device const float4 *sampled [[buffer(1)]],
    constant KernelParamsMetal &params [[buffer(2)]],
    device float4 *output [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
 MetalKernelContext context(params);
 const float4 a=input[4*i],b=input[4*i+1],c=input[4*i+2];
 MetalKernelContext::DiffractionRoughDielectric p{{c.x,c.y,c.z,b.z,b.w,c.w},b.x,b.y};
 float pdf=0,value=0,forward=0,back=0;
 if(sampled[2*i+1].y && !sampled[2*i+1].z) {
   const float3 wi=a.xyz,wo=sampled[2*i].xyz;
   value=context.diffraction_dielectric_eval<TEST_DISTRIBUTION>(&p,wi,wo,&pdf
#ifdef DIFFRACTION_TEST_COATED
       ,1.35f,.47f,p.facet.incident_ior==p.facet.transmitted_ior
#endif
       );
   forward=value/abs(wo.z);
   float3 ri=wo,ro=wi;
   if(wo.z<0) {
     const float eta=p.facet.incident_ior/p.facet.transmitted_ior;
     p.facet.incident_ior=b.w;p.facet.transmitted_ior=b.z;
     p.facet.wavelength_over_pitch*=eta;p.facet.height_over_wavelength/=eta;
     ri=float3(wo.x,-wo.y,-wo.z);ro=float3(wi.x,-wi.y,-wi.z);
     forward*=eta*eta;
   }
   float reverse_pdf;
   back=context.diffraction_dielectric_eval<TEST_DISTRIBUTION>(&p,ri,ro,&reverse_pdf
#ifdef DIFFRACTION_TEST_COATED
       ,1.35f,.47f,p.facet.incident_ior==p.facet.transmitted_ior
#endif
       )/abs(ro.z);
 }
 output[i]=float4(value,pdf,forward,back);
}

kernel void diffraction_dielectric_world_frame(
    device const float4 *input [[buffer(0)]],
    device float4 *output [[buffer(1)]],
    constant KernelParamsMetal &params [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
 MetalKernelContext context(params);
 const float4 a=input[4*i],b=input[4*i+1],c=input[4*i+2],d=input[4*i+3];
 MetalKernelContext::DiffractionDielectricBsdf closure;
 closure.type=CLOSURE_BSDF_DIFFRACTION_DIELECTRIC_ID;
 closure.disabled_lobes=0;
 closure.N=normalize(float3(.2f,.3f,1));
 closure.T=normalize(cross(float3(0,1,0),closure.N));
 closure.param={{c.x,c.y,c.z,b.z,b.w,c.w},b.x,b.y};
 float3 X,Y;context.bsdf_diffraction_dielectric_frame(&closure,&X,&Y);
 const float3 wi=a.x*X+a.y*Y+a.z*closure.N;
 float3 wo;Spectrum value;float pdf,eta;float2 roughness;
 const int label=context.bsdf_diffraction_dielectric_sample<TEST_DISTRIBUTION>(
     (thread ShaderClosure *)&closure,closure.N,wi,float3(a.w,d.x,d.y),
     &value,&wo,&pdf,&roughness,&eta);
 float error=0;
 if(label!=LABEL_NONE) {
   if(label&LABEL_SINGULAR) {
     const float mass=context.bsdf_diffraction_dielectric_delta_mass(
         (thread ShaderClosure *)&closure,wi,wo);
     error=abs(mass-pdf*1e-6f);
   }
   else {
     float evaluated_pdf;
     context.bsdf_diffraction_dielectric_eval<TEST_DISTRIBUTION>((thread ShaderClosure *)&closure,wi,wo,&evaluated_pdf);
     error=abs(pdf-evaluated_pdf)/max(1.0f,pdf);
   }
 }
 bool mask_bad=false;
 if(label!=LABEL_NONE) {
   closure.disabled_lobes=(label&LABEL_TRANSMIT)?LABEL_TRANSMIT:LABEL_REFLECT;
   float masked_pdf;Spectrum masked_value;
   masked_value=context.bsdf_diffraction_dielectric_eval<TEST_DISTRIBUTION>(
       (thread ShaderClosure *)&closure,wi,wo,&masked_pdf);
   mask_bad=masked_pdf!=0 || metal::any(masked_value!=Spectrum(0)) ||
       context.bsdf_diffraction_dielectric_delta_mass((thread ShaderClosure *)&closure,wi,wo)!=0;
   float3 masked_wo;float masked_eta;float2 masked_roughness;
   const int masked_label=context.bsdf_diffraction_dielectric_sample<TEST_DISTRIBUTION>(
       (thread ShaderClosure *)&closure,closure.N,wi,float3(a.w,d.x,d.y),
       &masked_value,&masked_wo,&masked_pdf,&masked_roughness,&masked_eta);
   mask_bad=mask_bad || masked_label!=LABEL_NONE || masked_pdf!=0 || metal::any(masked_value!=Spectrum(0));
 }
 // A small fixed setup suite executes the real shader-data allocator on GPU.
 // Keep it separate from the timing kernel and its random sampling workload.
 if(i<16) {
   const uint setup_case=i%8;
   ShaderData sd={};
   sd.num_closure_left=setup_case==3?2:3;
   MetalKernelContext::DiffractionRoughDielectric setup_param{{.37f,.24f,.42f,1,1,1.4f},.3f,.3f};
   if(i>=8)setup_param.alpha_x=setup_param.alpha_y=0;
   Spectrum reflection=Spectrum(.2f),transmission=Spectrum(.8f);
   if(setup_case==1)reflection=transmission;
   if(setup_case==2)transmission=Spectrum(0);
   if(setup_case==4)reflection=Spectrum(0);
   if(setup_case==5)reflection=transmission=Spectrum(0);
   if(setup_case==6)setup_param.facet.transmission_phase=0;
   if(setup_case==7)setup_param.facet.transmitted_ior=1.5f;
   const bool allocated=context.bsdf_diffraction_dielectric_setup_tinted<TEST_DISTRIBUTION>(
       &sd,reflection,transmission,float3(0,0,1),float3(1,0,0),&setup_param);
   const int expected=setup_case==0?2:setup_case==1?2:setup_case==2?1:setup_case==3?0:setup_case==4?2:setup_case==5?0:setup_case==6?1:1;
   mask_bad=mask_bad || sd.num_closure!=expected || allocated!=(expected>0) ||
       sd.num_closure_left!=(setup_case==3?2:3)-expected-int(setup_case==0 || setup_case==7);
   if((setup_case==0 || setup_case==7) && allocated) {
     thread auto *joint=(thread MetalKernelContext::DiffractionDielectricBsdf *)&sd.closure[sd.num_closure-1];
     MetalKernelContext::DiffractionDielectricBsdf filtered=*joint;
     const bool kept=context.bsdf_diffraction_dielectric_filter((thread ShaderClosure *)&filtered,true,false);
     mask_bad=mask_bad || !kept || !(filtered.disabled_lobes&LABEL_REFLECT) ||
         (filtered.disabled_lobes&LABEL_TRANSMIT) ||
         !(filtered.disabled_lobes&MetalKernelContext::DIFFRACTION_DIELECTRIC_TINT_EXTRA);
     mask_bad=mask_bad || context.bsdf_diffraction_dielectric_filter((thread ShaderClosure *)&filtered,false,true) ||
         filtered.type!=CLOSURE_NONE_ID || filtered.sample_weight!=0;
     ShaderClosure straight={};straight.type=CLOSURE_BSDF_DIFFRACTION_STRAIGHT_ID;straight.sample_weight=1;
     mask_bad=mask_bad || !context.bsdf_diffraction_dielectric_filter(&straight,true,false) ||
         context.bsdf_diffraction_dielectric_filter(&straight,false,true) || straight.sample_weight!=0;
     MetalKernelContext::DiffractionDielectricBsdf neutral=*joint;
     neutral.param=joint->extra->param;neutral.disabled_lobes=0;
     const float3 incoming=normalize(float3(.4f,.1f,1));
     for(int draw=0;draw<32;++draw) {
       const float3 random=float3((draw+.5f)/32.0f,metal::fract(draw*.6180339f+.2f),metal::fract(draw*.4142135f+.3f));
       float3 jo,no;Spectrum jv,nv;float jp,np,je,ne;float2 jr,nr;
       const int jl=context.bsdf_diffraction_dielectric_sample<TEST_DISTRIBUTION>(
           (thread ShaderClosure *)joint,float3(0,0,1),incoming,random,&jv,&jo,&jp,&jr,&je);
       const int nl=context.bsdf_diffraction_dielectric_sample<TEST_DISTRIBUTION>(
           (thread ShaderClosure *)&neutral,float3(0,0,1),incoming,random,&nv,&no,&np,&nr,&ne);
       const float tint=(jl&LABEL_TRANSMIT)?.8f:.2f;
       mask_bad=mask_bad || jl!=nl || abs(jp-np)>2e-5f*max(1.0f,np) ||
           (jl!=LABEL_NONE && (metal::length(jo-no)>1e-6f ||
            metal::any(abs(jv-tint*nv)>2e-5f*max(Spectrum(1),abs(tint*nv)))));
     }
   }
   if(setup_case==0 && sd.num_closure==2) {
     const float atom=context.diffraction_dielectric_straight_mass(&setup_param);
     thread auto *joint=(thread MetalKernelContext::DiffractionDielectricBsdf *)&sd.closure[1];
     mask_bad=mask_bad || abs(sd.closure[0].weight.x-.8f*atom)>1e-6f ||
         abs(sd.closure[1].weight.x-(1-atom))>1e-6f ||
         joint->extra->reflection.x!=.2f || joint->extra->transmission.x!=.8f;
   }
 }
 if(i==16) {
   ShaderData mixed={};mixed.num_closure_left=8;
   MetalKernelContext::DiffractionRoughDielectric p{{.37f,.24f,.42f,1,1,1.4f},.3f,.3f};
   const float3 N=float3(0,0,1),T=float3(1,0,0);
   const bool first_ok=context.bsdf_diffraction_dielectric_setup_tinted<TEST_DISTRIBUTION>(
       &mixed,Spectrum(.2f),Spectrum(.8f),N,T,&p);
   p.alpha_x=.6f;
   const bool second_ok=context.bsdf_diffraction_dielectric_setup_tinted<TEST_DISTRIBUTION>(
       &mixed,Spectrum(.7f),Spectrum(.3f),N,T,&p);
   const bool inline_ok=context.bsdf_diffraction_dielectric_setup_tinted<TEST_DISTRIBUTION>(
       &mixed,Spectrum(1),Spectrum(1),N,T,&p);
   const bool overflow=context.bsdf_diffraction_dielectric_setup_tinted<TEST_DISTRIBUTION>(
       &mixed,Spectrum(1),Spectrum(1),N,T,&p);
   mask_bad=mask_bad || !first_ok || !second_ok || !inline_ok || overflow ||
       mixed.num_closure!=6 || mixed.num_closure_left!=0;
   if(first_ok && second_ok) {
     thread auto *first=(thread MetalKernelContext::DiffractionDielectricBsdf *)&mixed.closure[1];
     thread auto *second=(thread MetalKernelContext::DiffractionDielectricBsdf *)&mixed.closure[3];
     mask_bad=mask_bad || first->extra==second->extra ||
         context.diffraction_dielectric_param(first)->alpha_x!=.3f ||
         context.diffraction_dielectric_param(second)->alpha_x!=.6f ||
         first->extra->reflection.x!=.2f || first->extra->transmission.x!=.8f ||
         second->extra->reflection.x!=.7f || second->extra->transmission.x!=.3f;
   }
 }
 if(i<128) {
   MetalKernelContext::DiffractionRoughDielectric p{{c.x,c.y,c.z,1,1,c.w},b.x,b.y};
   float atom,reverse,uncoated;
   const bool ok=context.diffraction_dielectric_coated_straight_mass_quadrature<TEST_DISTRIBUTION>(
       &p,a.xyz,1.7f,4*c.y,64,&atom);
   const bool reverse_ok=context.diffraction_dielectric_coated_straight_mass_quadrature<TEST_DISTRIBUTION>(
       &p,float3(-a.x,a.y,a.z),1.7f,4*c.y,64,&reverse);
   const bool uncoated_ok=context.diffraction_dielectric_coated_straight_mass_quadrature<TEST_DISTRIBUTION>(
       &p,a.xyz,1.7f,0,64,&uncoated);
   mask_bad=mask_bad || !ok || !reverse_ok || !uncoated_ok || atom!=reverse || atom<0 ||
       atom>context.diffraction_dielectric_straight_mass(&p) ||
       uncoated!=context.diffraction_dielectric_straight_mass(&p);
   ShaderData atom_sd={};atom_sd.wi=a.xyz;atom_sd.num_closure_left=2;
   const bool atom_allocated=context.bsdf_diffraction_coated_atom_setup<TEST_DISTRIBUTION>(
       &atom_sd,Spectrum(.7f),float3(0,0,1),float3(1,0,0),&p,1.7f,4*c.y,64);
   mask_bad=mask_bad || !atom_allocated || atom_sd.num_closure!=1 || atom_sd.num_closure_left!=0;
   if(atom_allocated) {
     const float stored=context.bsdf_diffraction_coated_atom_mass(&atom_sd.closure[0],a.xyz);
     mask_bad=mask_bad || abs(stored-atom)>2e-6f ||
         !context.bsdf_diffraction_dielectric_has_transmission(&atom_sd.closure[0]);
     float next_mass;
     const float3 next_wi=normalize(float3(.8f,.1f,.3f));
     context.diffraction_dielectric_coated_straight_mass_quadrature<TEST_DISTRIBUTION>(
         &p,next_wi,1.7f,4*c.y,64,&next_mass);
     mask_bad=mask_bad || abs(context.bsdf_diffraction_coated_atom_mass(
         &atom_sd.closure[0],next_wi)-next_mass)>2e-6f;
     mask_bad=mask_bad || !context.bsdf_diffraction_dielectric_filter(&atom_sd.closure[0],true,false) ||
         context.bsdf_diffraction_dielectric_filter(&atom_sd.closure[0],false,true);
   }
   float3 mixed_wo;float mixed_value,mixed_pdf;bool mixed_singular;
   const bool mixed_valid=context.diffraction_dielectric_sample_coated_mixture<TEST_DISTRIBUTION>(
       &p,a.xyz,float3(a.w,d.x,d.y),1.7f,4*c.y,atom,
       &mixed_wo,&mixed_value,&mixed_pdf,&mixed_singular);
   if(mixed_valid) {
     mask_bad=mask_bad || !(mixed_pdf>0) || !isfinite(mixed_pdf) ||
         !isfinite(mixed_value) || mixed_value<0 || mixed_value>20.0001f*mixed_pdf;
     if(!mixed_singular) {
       float checked_pdf;
       const float checked_value=context.diffraction_dielectric_eval_coated_mixture<TEST_DISTRIBUTION>(
           &p,a.xyz,mixed_wo,1.7f,4*c.y,atom,&checked_pdf);
       mask_bad=mask_bad || abs(checked_pdf-mixed_pdf)>2e-4f*max(1.0f,mixed_pdf) ||
           abs(checked_value-mixed_value)>2e-4f*max(1.0f,mixed_value);
     }
     else if(!context.roughness_is_almost_specular(p.alpha_x,p.alpha_y)) {
       const float probability=context.diffraction_dielectric_coated_atom_probability(&p,atom);
       mask_bad=mask_bad || mixed_pdf!=probability || mixed_value!=atom ||
           metal::any(mixed_wo!=-a.xyz);
     }
   }
 }
 if(i>=128 && i<160) {
   ShaderData coated_sd={};coated_sd.wi=wi;coated_sd.num_closure_left=4;
   MetalKernelContext::DiffractionRoughDielectric p{{c.x,c.y,c.z,b.z,b.w,c.w},b.x,b.y};
   const bool allocated=context.bsdf_diffraction_dielectric_setup_coated<TEST_DISTRIBUTION>(
       &coated_sd,Spectrum(.2f),Spectrum(.8f),closure.N,closure.T,&p,1.7f,4*c.y,64);
   const bool matched=b.z==b.w;
   mask_bad=mask_bad || !allocated || coated_sd.num_closure!=(matched?2:1) ||
       coated_sd.num_closure_left!=(matched?0:2);
   if(allocated) {
     thread ShaderClosure *main=&coated_sd.closure[coated_sd.num_closure-1];
     Spectrum value;float3 outgoing;float density,eta;float2 rough;
     const int label=context.bsdf_diffraction_dielectric_sample<TEST_DISTRIBUTION>(
         main,closure.N,wi,float3(a.w,d.x,d.y),&value,&outgoing,&density,&rough,&eta);
     if(label!=LABEL_NONE) {
       float evaluated_pdf;Spectrum evaluated;
       if(label&LABEL_SINGULAR) {
         float value_mass;
         evaluated_pdf=1e6f*context.bsdf_diffraction_dielectric_delta_mass(main,wi,outgoing,&value_mass);
         evaluated=Spectrum(value_mass*1e6f*((label&LABEL_TRANSMIT)?.8f:.2f));
       }
       else evaluated=context.bsdf_diffraction_dielectric_eval<TEST_DISTRIBUTION>(main,wi,outgoing,&evaluated_pdf);
       mask_bad=mask_bad || !isfinite(density) || !(density>0) ||
           abs(evaluated_pdf-density)>3e-4f*max(1.0f,density) ||
           metal::any(abs(value-evaluated)>3e-4f*max(Spectrum(1),abs(value)));
     }
   }
 }
 // Coated facet coefficients are validated separately from the uncoated rough
 // closure. Its constant straight-atom split must not be used for a coating.
 {
   MetalKernelContext::DiffractionDielectricFacet p{c.x,c.y,c.z,b.z,b.w,c.w};
   const float nf=1+c.z,thickness=4*c.y;
   const float3 h=float3(0,0,1),axis=float3(1,0,0);
   const float residual=context.diffraction_dielectric_facet_residual(
       &p,a.xyz,h,axis,nf,thickness);
   mask_bad=mask_bad || !isfinite(residual) || residual< -3e-5f || residual>1.00003f;
   const bool trans=i%2!=0;
   const int order=int(i%7)-3;
   float3 outgoing;
   const float eta=p.incident_ior/p.transmitted_ior;
   const bool valid=trans?
       context.diffraction_facet_transmit(a.xyz,h,axis,eta,order*p.wavelength_over_pitch*eta,&outgoing):
       context.diffraction_facet_reflect(a.xyz,h,axis,order*p.wavelength_over_pitch,&outgoing);
   if(valid && (trans || order!=0)) {
     const float power=context.diffraction_dielectric_facet_power(
         &p,order,trans,a.z,outgoing.z,nf,thickness);
     if(trans) {
       p.incident_ior=b.w;p.transmitted_ior=b.z;
       p.wavelength_over_pitch*=eta;p.height_over_wavelength/=eta;
       p.transmission_phase=-p.transmission_phase;
     }
     const float reverse=context.diffraction_dielectric_facet_power(
         &p,order,trans,trans?-outgoing.z:outgoing.z,trans?-a.z:a.z,nf,thickness);
     mask_bad=mask_bad || !isfinite(power) || power<0 || power>1 ||
         abs(power-reverse)>3e-5f;
   }
 }
#ifdef DIFFRACTION_TEST_SURFACE_MIXTURE
 if(i>=160 && i<416) {
   ShaderData sd={};sd.N=sd.Ng=closure.N;sd.wi=wi;sd.object=0;sd.num_closure_left=5;
   MetalKernelContext::DiffractionRoughDielectric p{{c.x,c.y,c.z,b.z,b.w,i%2?c.w:0},b.x,b.y};
   const bool allocated=context.bsdf_diffraction_dielectric_setup_coated<TEST_DISTRIBUTION>(
       &sd,Spectrum(.2f,.5f,.9f),Spectrum(.8f,.3f,.1f),sd.N,closure.T,&p,1.7f,.47f,64);
   mask_bad=mask_bad || !allocated;
   if(allocated) {
     if(i%2)context.bsdf_diffuse_setup(&sd,sd.N,Spectrum(.13f));
     if(i%4==0)sd.wi=normalize(-.7f*X+.2f*Y+.3f*sd.N);
     for(int trial=0;trial<4;++trial) {
       float3 random=float3(a.w,d.x,metal::fract(d.y+trial*.6180339887f));
       thread const ShaderClosure *selected=context.surface_shader_bsdf_bssrdf_pick(&sd,&random);
       float3 outgoing;Spectrum sampled_value;float sampled_pdf,eta;float2 roughness;
       const int sampled_label=context.bsdf_sample(nullptr,&sd,selected,random,
           &sampled_value,&outgoing,&sampled_pdf,&roughness,&eta);
       if(sampled_label==LABEL_NONE)continue;
       const bool delta=(sampled_label&LABEL_SINGULAR)!=0;
       BsdfEval full,skipped;float pdfs[MAX_CLOSURE],average_roughness=0;
       const float mixture=delta?context.surface_shader_bsdf_eval_delta(nullptr,&sd,outgoing,&full):
           context.surface_shader_bsdf_eval_pdfs(nullptr,&sd,outgoing,&full,pdfs,0,average_roughness);
       Spectrum sum=Spectrum(0);float weighted=0,total=0;
       for(int j=0;j<sd.num_closure;++j) {
         float density;
         const Spectrum value=delta?context.bsdf_eval_delta(nullptr,&sd,&sd.closure[j],outgoing,&density):
             context.bsdf_eval(nullptr,&sd,&sd.closure[j],outgoing,&density);
         sum+=value*sd.closure[j].weight;weighted+=density*sd.closure[j].sample_weight;
         total+=sd.closure[j].sample_weight;
       }
       context.bsdf_eval_init(&skipped,Spectrum(0));
       context.bsdf_eval_accum(&skipped,selected,outgoing,sampled_value*selected->weight);
       float skip_roughness=0;
       const float skip_pdf=context._surface_shader_bsdf_eval_mis(nullptr,&sd,outgoing,selected,
           &skipped,sampled_pdf*selected->sample_weight,selected->sample_weight,0,
           sampled_pdf*selected->sample_weight*context.bsdf_get_specular_roughness_squared(selected),skip_roughness);
       mask_bad=mask_bad || !isfinite(mixture) || !(mixture>0) ||
           abs(mixture-weighted/total)>3e-4f*max(1.0f,mixture) ||
           abs(mixture-skip_pdf)>3e-4f*max(1.0f,mixture) ||
           metal::any(abs(context.bsdf_eval_sum(&full)-sum)>3e-4f*max(Spectrum(1),abs(sum))) ||
           metal::any(abs(context.bsdf_eval_sum(&skipped)-sum)>3e-4f*max(Spectrum(1),abs(sum)));
     }
   }
 }
#endif
 const bool bad=mask_bad || (label!=LABEL_NONE &&
     (!(pdf>0)||!isfinite(pdf)||!all(isfinite(value))||
      (((label&LABEL_TRANSMIT)!=0)!=(dot(closure.N,wo)<0))||
      abs(eta-((label&LABEL_TRANSMIT)?b.w/b.z:1.0f))>1e-6f));
 output[i]=float4(error,float(label!=LABEL_NONE),float((label&LABEL_SINGULAR)!=0),float(bad));
}
