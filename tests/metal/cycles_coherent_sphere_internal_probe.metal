/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/light/coherent_sphere_internal_geometry.h"
kernel void sphere_internal_probe(device const float4 *input [[buffer(0)]],device float4 *output [[buffer(1)]])
{
  CoherentSphereInternalInventory paths{};
  const auto status=coherent_sphere_internal_inventory(input[0].xyz,input[1].xyz,input[2].xyz,
      input[3].xyz,input[0].w,input[1].w,int(input[3].w),&paths);
  output[0]=float4(float(status),float(paths.count),0,0);
  for(int j=0;j<9;j++) {
    if(j<paths.count) {
      const auto path=paths.path[j];
      output[1+j*6]=float4(paths.angular_momentum[j],path.spreading,path.optical_length_split);
      output[2+j*6]=float4(path.point[0],float(paths.morse_index[j]));
      output[3+j*6]=float4(path.point[1],paths.maslov_phase_cycles[j]);
      output[4+j*6]=float4(path.point[2],float(paths.winding[j]));
      output[5+j*6]=float4(path.count==4?path.point[3]:float3(0),float(path.count));
      CoherentSphereInternalInventory selected{};
      const auto selected_status=coherent_sphere_internal_inventory(input[0].xyz,input[1].xyz,input[2].xyz,
          input[3].xyz,input[0].w,input[1].w,int(input[3].w),&selected,j);
      output[6+j*6]=float4(selected.path[j].optical_length_split-path.optical_length_split,
          selected.path[j].spreading/path.spreading-1,
          selected_status==status && selected.count==paths.count &&
          selected.morse_index[j]==paths.morse_index[j] ? 0 : 1);
    } else for(int k=0;k<6;k++) output[1+j*6+k]=float4(0);
  }
}
