/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/light/coherent_sphere_transmit_geometry.h"
kernel void sphere_tt_probe(device const float4 *input [[buffer(0)]], device float4 *output [[buffer(1)]])
{
  const float3 source=input[0].xyz,receiver=input[1].xyz;
  const float R=input[0].w,n=input[1].w;
  CoherentSphereTTInventory paths{};
  const auto status=coherent_sphere_transmit_inventory(source,receiver,input[2].xyz,input[3].xyz,R,n,&paths);
  output[0]=float4(float(status),float(paths.count),0,0);
  for(int j=0;j<3;j++){
    if(j<paths.count){
      const auto path=paths.path[j];
      output[1+j*3]=float4(paths.angular_momentum[j],path.spreading,path.optical_length_split);
      output[2+j*3]=float4(path.point[0],float(paths.morse_index[j]));
      output[3+j*3]=float4(path.point[1],paths.maslov_phase_cycles[j]);
    }else{output[1+j*3]=output[2+j*3]=output[3+j*3]=float4(0);}
    if(j<paths.count){
      CoherentSphereTTInventory selected{};
      coherent_sphere_transmit_inventory(source,receiver,input[2].xyz,input[3].xyz,R,n,&selected,j);
      const auto path=selected.path[j];
      output[13+j*3]=float4(selected.angular_momentum[j],path.spreading,path.optical_length_split);
      output[14+j*3]=float4(path.point[0],float(selected.morse_index[j]));
      output[15+j*3]=float4(path.point[1],selected.maslov_phase_cycles[j]);
    }else{output[13+j*3]=output[14+j*3]=output[15+j*3]=float4(0);}
  }
  const float a=len(source-input[3].xyz),b=len(receiver-input[3].xyz),A=R/a,B=R/b;
  const float3 e0=(source-input[3].xyz)/a,rd=(receiver-input[3].xyz)/b;
  const float cosine=clamp(dot(e0,rd),-1.0f,1.0f);
  const float theta=atan2f(len(rd-e0*cosine),cosine);
  float l=0,h=1;
  for(int k=0;k<30;k++){const float m=(l+h)*.5f;if(coherent_sphere_tt_normalized_derivative(m,A,B,n)>0)h=m;else l=m;}
  const float tc=(l+h)*.5f;
  output[10]=float4(A,B,theta,coherent_sphere_tt_normalized_derivative(0,A,B,n));
  output[11]=float4(tc,coherent_sphere_tt_equation(-tc,A,B,n,theta),coherent_sphere_tt_equation(tc,A,B,n,theta),0);
  output[12]=float4(coherent_sphere_tt_equation(-1,A,B,n,theta),coherent_sphere_tt_equation(1,A,B,n,theta),0,0);
}
