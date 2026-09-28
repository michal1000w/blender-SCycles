/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/closure/diffraction_thin_film.h"
kernel void diffraction_thin_film(device const float4 *input [[buffer(0)]],
                                 device float2 *output [[buffer(1)]],
                                 uint i [[thread_position_in_grid]])
{
 const float4 a=input[2*i],b=input[2*i+1];
 output[i]=float2(diffraction_thin_film_pair_reflectance(a.x,a.y,a.z,a.w,b.x,b.y),
                  diffraction_thin_film_pair_reflectance(a.y,a.x,a.w,a.z,b.x,b.y));
}
