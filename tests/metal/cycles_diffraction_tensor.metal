/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Experimental rank-62, seven-node tensor model. Output is packed Hermitian
 * chart data, not a physical scattering matrix or a complete BSDF. */
#include <metal_stdlib>
using namespace metal;
void tensor_basis(float x, device const float *nodes, device const float *weights,
                  thread float *b)
{
  int exact = -1;
  for (int i = 0; i < 7; ++i) if (x == nodes[i]) exact = i;
  if (exact >= 0) {
    for (int i = 0; i < 7; ++i) b[i] = float(i == exact);
    return;
  }
  float sum = 0;
  float nearest = x - nodes[0];
  for (int i = 1; i < 7; ++i)
    if (abs(x - nodes[i]) < abs(nearest)) nearest = x - nodes[i];
  for (int i = 0; i < 7; ++i) {
    b[i] = weights[i] * (nearest / (x - nodes[i]));
    sum += b[i];
  }
  for (int i = 0; i < 7; ++i) b[i] /= sum;
}
kernel void tensor_coefficients(device const float *model [[buffer(0)]],
                                device const float *queries [[buffer(1)]],
                                device float *coefficients [[buffer(2)]],
                                uint index [[thread_position_in_grid]])
{
  uint query = index / 62, rank = index % 62;
  float bx[7], by[7], bz[7];
  tensor_basis(queries[3*query], model, model+7, bx);
  tensor_basis(queries[3*query+1], model, model+7, by);
  tensor_basis(queries[3*query+2], model, model+7, bz);
  float value = 0;
  for (int z = 0; z < 7; ++z) {
    float zy = 0;
    for (int y = 0; y < 7; ++y) {
      float yx = 0;
      for (int x = 0; x < 7; ++x)
        yx += bx[x] * model[14 + ((z*7+y)*7+x)*62 + rank];
      zy += by[y] * yx;
    }
    value += bz[z] * zy;
  }
  coefficients[index] = value;
}
kernel void tensor_reconstruct(device const float *model [[buffer(0)]],
                               device const float *coefficients [[buffer(1)]],
                               device float *output [[buffer(2)]],
                               uint index [[thread_position_in_grid]])
{
  uint query = index / 400, entry = index % 400;
  constexpr uint directions = 14 + 343*62, mean = directions + 62*400;
  float value = model[mean + entry];
  for (int r = 0; r < 62; ++r)
    value += coefficients[query*62+r] * model[directions+r*400+entry];
  output[index] = value;
}
float2 tensor_mul(float2 a, float2 b) {
  return float2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}
float2 tensor_div(float2 a, float2 b) {
  return tensor_mul(a,float2(b.x,-b.y))/dot(b,b);
}
float2 tensor_h(device const float *packed, uint row, uint col) {
  if (row == col) return float2(packed[row],0);
  uint a=min(row,col), b=max(row,col);
  uint k=a*(39-a)/2+b-a-1;
  return float2(packed[20+k], (row<col?1.0f:-1.0f)*packed[210+k]) * M_SQRT1_2_F;
}
/* One RHS per lane, partial-pivot complex elimination. This deliberately
 * exposes solver cost; production would avoid solving unused columns. */
kernel void tensor_cayley(device const float *model [[buffer(0)]],
                          device const float *packed [[buffer(1)]],
                          device float2 *output [[buffer(2)]],
                          uint index [[thread_position_in_grid]])
{
  uint query=index/20, col=index%20;
  float2 a[400], rhs[20];
  for (uint row=0;row<20;++row) {
    for (uint k=0;k<20;++k) {
      float2 h=tensor_h(packed+query*400,row,k);
      a[row*20+k]=float2(float(row==k)-h.y,h.x);
    }
    float2 h=tensor_h(packed+query*400,row,col);
    rhs[row]=float2(float(row==col)+h.y,-h.x);
  }
  for (uint k=0;k<20;++k) {
    uint pivot=k;
    for (uint row=k+1;row<20;++row)
      if (dot(a[row*20+k],a[row*20+k]) > dot(a[pivot*20+k],a[pivot*20+k])) pivot=row;
    if (!(dot(a[pivot*20+k],a[pivot*20+k])>0)) {
      for(uint row=0;row<20;++row) output[query*400+row*20+col]=float2(NAN);
      return;
    }
    if (pivot!=k) {
      for(uint j=k;j<20;++j) { float2 v=a[k*20+j]; a[k*20+j]=a[pivot*20+j]; a[pivot*20+j]=v; }
      float2 v=rhs[k];rhs[k]=rhs[pivot];rhs[pivot]=v;
    }
    for(uint row=k+1;row<20;++row) {
      float2 factor=tensor_div(a[row*20+k],a[k*20+k]);
      for(uint j=k+1;j<20;++j) a[row*20+j]-=tensor_mul(factor,a[k*20+j]);
      rhs[row]-=tensor_mul(factor,rhs[k]);
    }
  }
  for(int row=19;row>=0;--row) {
    for(uint j=row+1;j<20;++j) rhs[row]-=tensor_mul(a[row*20+j],rhs[j]);
    rhs[row]=tensor_div(rhs[row],a[row*20+row]);
  }
  float2 phase=float2(model[46480],model[46481]);
  for(uint row=0;row<20;++row) output[query*400+row*20+col]=tensor_mul(phase,rhs[row]);
}
