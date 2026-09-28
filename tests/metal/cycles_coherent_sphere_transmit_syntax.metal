/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/light/coherent_sphere_transmit_geometry.h"
kernel void sphere_tt_syntax(device float4 *output [[buffer(0)]])
{
  CoherentSphereTTInventory paths{};
  const auto status=coherent_sphere_transmit_inventory(float3(2,0,0),float3(-1.99f,.2f,0),float3(1,0,0),float3(0),1,1.5f,&paths);
  output[0]=float4(float(status),float(paths.count),paths.path[0].spreading,paths.maslov_phase_cycles[0]);
}
