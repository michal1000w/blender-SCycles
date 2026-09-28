/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Numerical experiment only; production closures are unchanged. */
#define main thin_wall_previous_fixture_entry
#include "cycles_diffraction_thin_wall_native_test.cpp"
#undef main
using namespace ccl;
static double proposal_pdf(ShaderData *sd,float3 wo){
 double total=0,value=0;
 for(int k=0;k<sd->num_closure;k++)total+=sd->closure[k].sample_weight;
 for(int k=0;k<sd->num_closure;k++){
   float pdf=0;bsdf_diffraction_thin_sheet_eval(&sd->closure[k],sd->wi,wo,&pdf);
   value+=sd->closure[k].sample_weight*pdf/total;
 }
 return value;
}
int main(){
 std::vector<float> tables(table_ggx_E,table_ggx_E+1024);tables.insert(tables.end(),table_ggx_Eavg,table_ggx_Eavg+32);
 KernelGlobalsCPU base{};base.lookup_table.data=tables.data();base.lookup_table.width=int(tables.size());base.data.tables.ggx_E=0;base.data.tables.ggx_Eavg=1024;
 Profiler profiler;ThreadKernelGlobalsCPU thread(base,nullptr,profiler,0);KernelGlobals kg=&thread;
 constexpr int M=16,P=32,S=2*M*P;std::vector<float3>w;std::vector<double>measure;
 for(int side:{1,-1})for(int m=0;m<M;m++)for(int p=0;p<P;p++){
  float mu=(m+.5f)/M,phi=M_2PI_F*(p+.5f)/P,t=sqrtf(1-mu*mu);w.push_back(make_float3(t*cosf(phi),t*sinf(phi),side*mu));measure.push_back(mu*double(M_2PI_F)/(M*P));
 }
 int failures=0,checks=0;
 for(int c=0;c<3;c++){
  float roughness=c==0?1.f:.6f,tint=c==2?.8f:1.f;
  auto setup=[&](float3 wi){float3 N=make_float3(0,0,wi.z>0?1.f:-1.f);ShaderData sd=make_shader_data(wi,N);if(wi.z<0)sd.runtime_flag|=SR_BACKFACING;
   bsdf_diffraction_thin_glass_setup(kg,&sd,true,true,{one_spectrum(),make_spectrum(tint)},one_spectrum(),N,make_float3(1,0,0),roughness*roughness,1.5f,{0,1},PATH_RAY_VISIBILITY_CAMERA,0,1,1150,320,.42f,0);return sd;};
  std::vector<ShaderData> sd(S);std::vector<double>K(S*S),proposal(S*S),target(S),E(S),q(S);double Q=0,min_q=1,max_overshoot=0,max_recip=0,max_budget_error=0,max_sampling_error=0,max_pdf=0;int worst=0;
  for(int i=0;i<S;i++){
   sd[i]=setup(w[i]);FresnelCoeff f=bsdf_thin_glass_fresnel(kg,true,true,{one_spectrum(),make_spectrum(tint)},{0,1},1.5f,std::abs(w[i].z));target[i]=average(f.reflectance+f.transmittance);
   for(int j=0;j<S;j++){K[i*S+j]=average(full_eval(kg,&sd[i],w[j]))/std::abs(w[j].z);proposal[i*S+j]=proposal_pdf(&sd[i],w[j]);}
  }
  for(int i=0;i<S;i++){
   for(int j=0;j<S;j++)E[i]+=std::min(K[i*S+j],K[j*S+i])*measure[j];
   q[i]=target[i]-E[i];min_q=std::min(min_q,q[i]);max_overshoot=std::max(max_overshoot,E[i]-target[i]);Q+=q[i]*measure[i];if(q[i]<q[worst])worst=i;
  }
  if(min_q<0||!(Q>0)){failures++;printf("FAIL negative_missing_energy case=%d min_q=%.9g overshoot=%.9g\n",c,min_q,max_overshoot);continue;}
  for(int i=0;i<S;i++){
   double row=0,sampled_expectation=0;double beta=E[i]/target[i];
   for(int j=0;j<S;j++){
    double envelope=std::min(K[i*S+j],K[j*S+i]);double ret=q[i]*q[j]/Q,completed=envelope+ret;
    double reverse=std::min(K[j*S+i],K[i*S+j])+q[j]*q[i]/Q;max_recip=std::max(max_recip,std::abs(completed-reverse));
    double accept=K[i*S+j]>0?envelope/K[i*S+j]:0;
    double pdf=beta*proposal[i*S+j]*accept+(1-beta)*q[j]*std::abs(w[j].z)/Q;
    if(completed>0&&!(pdf>0))failures++;
    double V=completed*std::abs(w[j].z);row+=completed*measure[j];
    if(pdf>0)sampled_expectation+=pdf*(V/pdf)*(measure[j]/std::abs(w[j].z));
    max_pdf=std::max(max_pdf,pdf);
   }
   max_budget_error=std::max(max_budget_error,std::abs(row-target[i]));max_sampling_error=std::max(max_sampling_error,std::abs(sampled_expectation-target[i]));
  }
  // Independently resolve the smallest-q row, rather than clamping coarse negative estimates.
  constexpr int DM=128,DP=256;double dense=0;
  for(int side:{1,-1})for(int m=0;m<DM;m++)for(int p=0;p<DP;p++){
   float mu=(m+.5f)/DM,phi=M_2PI_F*(p+.5f)/DP,t=sqrtf(1-mu*mu);float3 wo=make_float3(t*cosf(phi),t*sinf(phi),side*mu);ShaderData reverse=setup(wo);
   double a=average(full_eval(kg,&sd[worst],wo))/mu,b=average(full_eval(kg,&reverse,w[worst]))/std::abs(w[worst].z);
   dense+=std::min(a,b)*mu;
  }
  dense*=double(M_2PI_F)/(DM*DP);
  checks++;if(max_recip>1e-12||max_budget_error>1e-10||max_sampling_error>1e-10||dense>target[worst])failures++;
  printf("case=%d roughness=%.3g tint=%.3g nodes=%d min_q=%.9g Q=%.9g discrete_reciprocity_abs=%.3g budget_error=%.3g native_accept_plus_return_expectation_error=%.3g dense_min_q_row_mu=%.9g row_estimate=%.9g dense_envelope_row=%.9g target=%.9g max_pdf=%.9g\n",c,roughness,tint,S,min_q,Q,max_recip,max_budget_error,max_sampling_error,std::abs(w[worst].z),E[worst],dense,target[worst],max_pdf);
 }
 printf("candidate_checks=%d failures=%d production_changed=0\n",checks,failures);return failures!=0;
}
