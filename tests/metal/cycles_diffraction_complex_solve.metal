/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include <metal_stdlib>
using namespace metal;

float2 complex_divide(float2 a, float2 b)
{
  // Scale by the larger denominator component rather than squaring both.
  if (abs(b.x) >= abs(b.y)) {
    float ratio=b.y/b.x, denominator=b.x+b.y*ratio;
    return float2((a.x+a.y*ratio)/denominator,(a.y-a.x*ratio)/denominator);
  }
  float ratio=b.x/b.y, denominator=b.y+b.x*ratio;
  return float2((a.x*ratio+a.y)/denominator,(a.y*ratio-a.x)/denominator);
}

float2 subtract_product(float2 a,float2 b,float2 c)
{
  return float2(fma(-b.x,c.x,fma(b.y,c.y,a.x)),
                fma(-b.x,c.y,fma(-b.y,c.x,a.y)));
}

// One threadgroup owns a matrix and all of its RHS columns. Workspace is
// row-major [A | B]; on success B contains X. Status: 1 success, 2 singular,
// 3 nonfinite. No regularization or identity fallback is applied on failure.
kernel void complex_solve(device float2 *workspace [[buffer(0)]],
                          device uint *status [[buffer(1)]],
                          constant uint &n [[buffer(2)]],
                          constant uint &rhs [[buffer(3)]],
                          uint matrix [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]],
                          uint width [[threads_per_threadgroup]])
{
  threadgroup uint pivot_row;
  threadgroup uint state;
  threadgroup float2 pivot;
  uint stride=n+rhs,base=matrix*n*stride;
  if (lane==0) state=1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k=0;k<n;++k) {
    if (lane==0) {
      float best=0;
      pivot_row=k;
      for (uint row=k;row<n;++row) {
        float2 value=workspace[base+row*stride+k];
        if (!all(isfinite(value))) { state=3; break; }
        float size=max(abs(value.x),abs(value.y));
        if (size>best) { best=size; pivot_row=row; }
      }
      if (state==1 && best==0) state=2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (state!=1) break; // Uniform threadgroup condition.
    if (pivot_row!=k) {
      for (uint col=lane;col<stride;col+=width) {
        float2 value=workspace[base+k*stride+col];
        workspace[base+k*stride+col]=workspace[base+pivot_row*stride+col];
        workspace[base+pivot_row*stride+col]=value;
      }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (lane==0) pivot=workspace[base+k*stride+k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint col=lane;col<stride;col+=width) {
      workspace[base+k*stride+col]=complex_divide(workspace[base+k*stride+col],pivot);
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint row=lane;row<n;row+=width) {
      if (row==k) continue;
      float2 factor=workspace[base+row*stride+k];
      for (uint col=k+1;col<stride;++col) {
        workspace[base+row*stride+col]=subtract_product(workspace[base+row*stride+col],
                                                       factor,workspace[base+k*stride+col]);
      }
      workspace[base+row*stride+k]=float2(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
  if (lane==0) {
    if (state==1) {
      for (uint row=0;row<n;++row)
        for (uint col=n;col<stride;++col)
          if (!all(isfinite(workspace[base+row*stride+col]))) state=3;
    }
    status[matrix]=state;
  }
}
