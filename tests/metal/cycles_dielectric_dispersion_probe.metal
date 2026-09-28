/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/util/dielectric_dispersion.h"
using namespace metal;
kernel void dielectric_dispersion_probe(device const float4 *input [[buffer(0)]],
                                         device uint4 *output [[buffer(1)]],
                                         uint i [[thread_position_in_grid]])
{
  const float4 p=input[i]; // nm, sampled native micrometres, d-line index, invAbbe
  const float um=dielectric_wavelength_um(p.x);
  output[i]=uint4(as_type<uint>(um),
      as_type<uint>(dielectric_ior_at_wavelength(p.z,p.w,um)),
      as_type<uint>(dielectric_ior_at_wavelength(p.z,p.w,p.y)),
      as_type<uint>(dielectric_ior_at_wavelength(p.z,p.w,0.78f)));
}
