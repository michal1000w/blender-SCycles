/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Numerical experiment only; production closures are unchanged. */
#define main thin_wall_previous_fixture_entry
#include "cycles_diffraction_thin_wall_native_test.cpp"
#undef main
using namespace ccl;
int main(){
  std::vector<float> tables(table_ggx_E,table_ggx_E+1024);
  tables.insert(tables.end(),table_ggx_Eavg,table_ggx_Eavg+32);
  KernelGlobalsCPU base{};base.lookup_table.data=tables.data();base.lookup_table.width=int(tables.size());
  base.data.tables.ggx_E=0;base.data.tables.ggx_Eavg=1024;
  Profiler profiler;ThreadKernelGlobalsCPU thread(base,nullptr,profiler,0);KernelGlobals kg=&thread;
  constexpr int M=8,P=16,S=2*M*P;
  std::vector<float3> w;std::vector<double> measure;
  for(int side:{1,-1})for(int m=0;m<M;m++)for(int p=0;p<P;p++){
    float mu=(m+.5f)/M,phi=M_2PI_F*(p+.5f)/P,s=sqrtf(1-mu*mu);
    w.push_back(make_float3(s*cosf(phi),s*sinf(phi),side*mu));measure.push_back(mu*double(M_2PI_F)/(M*P));
  }
  int checks=0,failures=0;
  for(int case_id=0;case_id<3;case_id++){
    float roughness=case_id==0?1.f:.6f;float tint_value=case_id==2?.8f:1.f;
    std::vector<ShaderData> shader(S);std::vector<double> target(S),energy(S),q(S),K(S*S);
    double Q=0,max_raw_energy=0,min_q=1,min_target=1,max_asym=0,max_relative=0,return_error=0,full_max_energy=0;
    for(int i=0;i<S;i++){
      float3 N=make_float3(0,0,w[i].z>0?1.f:-1.f);
      shader[i]=make_shader_data(w[i],N);if(w[i].z<0)shader[i].runtime_flag|=SR_BACKFACING;
      FresnelCoeff sheet=bsdf_diffraction_thin_glass_setup(kg,&shader[i],true,true,
          {one_spectrum(),make_spectrum(tint_value)},one_spectrum(),N,make_float3(1,0,0),
          roughness*roughness,1.5f,{0,1},PATH_RAY_VISIBILITY_CAMERA,0,1,1150,320,.42f,0);
      target[i]=average(sheet.reflectance+sheet.transmittance);min_target=std::min(min_target,target[i]);
      for(int j=0;j<S;j++){
        double v=average(full_eval(kg,&shader[i],w[j]));
        if(!std::isfinite(v)||v<0)failures++;
        K[i*S+j]=v/std::abs(w[j].z);energy[i]+=K[i*S+j]*measure[j];
      }
      max_raw_energy=std::max(max_raw_energy,energy[i]);
      q[i]=target[i]-energy[i];min_q=std::min(min_q,q[i]);
      Q+=std::max(0.,q[i])*measure[i];
    }
    int worst_row=0, pair_i=0, pair_j=0;
    for(int i=0;i<S;i++){
      if(energy[i]>energy[worst_row])worst_row=i;
      double row=0;
      for(int j=0;j<S;j++){
        double a=K[i*S+j],b=K[j*S+i];
        if(std::abs(a-b)>max_asym){max_asym=std::abs(a-b);pair_i=i;pair_j=j;}
        if(std::max(a,b)>1e-3)max_relative=std::max(max_relative,std::abs(a-b)/std::max(a,b));
        double ms=Q>0?std::max(0.,q[i])*std::max(0.,q[j])/Q:0;
        row+=(a+ms)*measure[j];
        // The added return is symmetric, so base antisymmetry is preserved exactly.
        if(std::abs(((a+ms)-(b+ms))-(a-b))>1e-11)failures++;
      }
      return_error=std::max(return_error,std::abs(row-std::max(target[i],energy[i])));
      full_max_energy=std::max(full_max_energy,row);
    }
    double dense_energy=0;
    constexpr int DM=128,DP=256;
    for(int side:{1,-1})for(int m=0;m<DM;m++)for(int p=0;p<DP;p++){
      float mu=(m+.5f)/DM,phi=M_2PI_F*(p+.5f)/DP,s=sqrtf(1-mu*mu);
      dense_energy+=average(full_eval(kg,&shader[worst_row],make_float3(s*cosf(phi),s*sinf(phi),side*mu)));
    }
    dense_energy*=double(M_2PI_F)/(DM*DP);
    printf("dense_worst_coarse_row case=%d mu=%.9g target=%.9g coarse_energy=%.9g dense_energy=%.9g resolution=%dx%d per_side; exact_pair mu_i=%.9g mu_j=%.9g Kij=%.9g Kji=%.9g\n",case_id,std::abs(w[worst_row].z),target[worst_row],energy[worst_row],dense_energy,DM,DP,w[pair_i].z,w[pair_j].z,K[pair_i*S+pair_j],K[pair_j*S+pair_i]);
    checks++;
    if(return_error>1e-10||max_asym<=1e-4)failures++;
    printf("case=%d roughness=%.3g transmission_tint=%.3g nodes=%d target_min=%.9g raw_energy_max=%.9g raw_missing_min=%.9g Q=%.9g returned_energy_max=%.9g discrete_budget_error=%.3g base_and_completed_reciprocity_abs=%.9g relative=%.9g\n",case_id,roughness,tint_value,S,min_target,max_raw_energy,min_q,Q,full_max_energy,return_error,max_asym,max_relative);
  }
  printf("experiment_checks=%d failures=%d conclusion=symmetric_additive_return_cannot_repair_existing_first_event_reciprocity\n",checks,failures);
  return failures!=0;
}
