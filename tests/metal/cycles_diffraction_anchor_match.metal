/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/util/diffraction_reference.h"
kernel void tensor_anchor_match(device const float *inputs [[buffer(0)]],
                         device float2 *output [[buffer(1)]],
                         uint index [[thread_position_in_grid]])
{
  device const float *record=inputs+index*1681;
  float2 chart[400], anchor[400], boundary[40], jones[40];
  for(int i=0;i<400;++i) chart[i]=make_float2(record[2*i],record[2*i+1]);
  for(int i=0;i<400;++i) anchor[i]=make_float2(record[800+2*i],record[801+2*i]);
  for(int i=0;i<40;++i) boundary[i]=make_float2(record[1600+2*i],record[1601+2*i]);
  const bool valid=diffraction_reference_match_anchor<20>(chart,anchor,boundary,int(record[1680]),jones);
  for(int i=0;i<40;++i) output[index*40+i]=valid?jones[i]:make_float2(NAN,NAN);
}
