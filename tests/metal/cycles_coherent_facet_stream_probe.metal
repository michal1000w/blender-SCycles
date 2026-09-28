/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/light/coherent_facet_stream.h"
kernel void coherent_facet_stream_probe(device const float4 *paths [[buffer(0)]],
    device const float4 *cases [[buffer(1)]], device float4 *output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
 CoherentStreamField sum;coherent_stream_clear(&sum);
 const float4 config=cases[id];
 for(int i=0;i<int(config.w);i++) {
  CoherentCompletedPathField field{};
  const float4 input_path=paths[int(config.z)+i];
  field.radiance_amplitude_rgb=float3(input_path.x);
  field.optical_length_split=float2(input_path.y,input_path.z);
  field.source_phase_cycles=input_path.w;
  coherent_stream_add(&sum,&field,float2(0),float2(550e-9f,1.8441641985202095e-14f),config.y,config.x,i==0,true);
 }
 float3 direct;
 const float3 total=coherent_stream_finish(&sum,&direct);
 const float2 phase=coherent_stream_phase(float2(1.0f,1.27e-8f),float2(550e-9f,1.8441641985202095e-14f),config.y,config.x,.13f);
 output[id]=float4(total.x,direct.x,phase.x,phase.y);
}
