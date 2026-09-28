/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
// Real embedding of a complex system; all conversions remain GPU-resident.
kernel void matrix_real_embedding(device const float2 *a [[buffer(0)]],
                                  device float *out [[buffer(1)]],
                                  constant uint4 &shape [[buffer(3)]], uint i [[thread_position_in_grid]])
{
  uint n=shape.x,row=i/(2*n),col=i%(2*n);
  float2 z=a[(row%n)*n+col%n];
  out[i]=(row<n)==(col<n)?z.x:((row<n)?-z.y:z.y);
}
kernel void matrix_real_rhs(device const float2 *b [[buffer(0)]],
                            device float *out [[buffer(1)]],
                            constant uint4 &shape [[buffer(3)]], uint i [[thread_position_in_grid]])
{
  uint count=shape.x*shape.y;float2 z=b[i%count];out[i]=i<count?z.x:z.y;
}
kernel void matrix_complex_solution(device const float *x [[buffer(0)]],
                                    device float2 *out [[buffer(1)]],
                                    constant uint4 &shape [[buffer(3)]], uint i [[thread_position_in_grid]])
{
  out[i]=float2(x[i],x[i+shape.x*shape.y]);
}
kernel void matrix_finite(device const float2 *a [[buffer(0)]],
                           device atomic_uint *status [[buffer(1)]],
                           uint i [[thread_position_in_grid]])
{
  if(!all(isfinite(a[i])))atomic_store_explicit(status,0u,memory_order_relaxed);
}
