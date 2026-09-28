/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/util/diffraction_tensor.h"
kernel void tensor_chart_test(device const float *model [[buffer(0)]],
                               device const float *queries [[buffer(1)]],
                               device float2 *output [[buffer(2)]],
                               constant int2 &layout [[buffer(3)]],
                               uint index [[thread_position_in_grid]])
{
  float2 chart[400];
  const bool valid=diffraction_tensor_chart<20>(model,layout.x,layout.y,
      make_float3(queries[index*3],queries[index*3+1],queries[index*3+2]),chart);
  for(int i=0;i<400;++i) output[index*400+i]=valid?chart[i]:make_float2(NAN,NAN);
}
