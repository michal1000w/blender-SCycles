/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/util/diffraction_reference.h"
CCL_NAMESPACE_BEGIN
/* Direct-admittance matching for a matrix-anchored Hermitian chart. Exterior entries are tangent x/y,
 * signed squared normal momentum and refractive index. Rows use TE/TM axes;
 * TM equations are multiplied by q to stay finite at the Rayleigh threshold. */
template<int N, typename Chart, typename Anchor, typename Exterior>
#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device_inline
#endif
bool diffraction_reference_match_admittance(const Chart chart,
                                                        const Anchor anchor,
                                                        const Exterior exterior,
                                                        const int incoming,
                                                        ccl_private float2 *jones)
{
  static_assert(N > 0 && N % 2 == 0);
  if(incoming<0 || incoming>=N/2 || !(exterior[incoming].z>0))return false;
  float2 v[N*N],a[N*N],rhs[2*N];
  for(int i=0;i<N;i++)for(int j=0;j<N;j++) {
    float2 value=anchor[i*N+j];
    for(int k=0;k<N;k++)value=value-diffraction_complex_mul(anchor[i*N+k],chart[k*N+j]);
    v[i*N+j]=value;
  }
  for(int row=0;row<N;row++) {
    const int p=row/2,mode=row%2;
    const float4 e=exterior[p];
    const float ex=mode?e.x:-e.y,ey=mode?e.y:e.x;
    const float q=sqrtf(fabsf(e.z));
    const float2 normal=e.z>=0?make_float2(q,0):make_float2(0,q);
    for(int col=0;col<N;col++) {
      const float2 b=ex*(make_float2(float(2*p==col),0)+chart[2*p*N+col])+
                     ey*(make_float2(float(2*p+1==col),0)+chart[(2*p+1)*N+col]);
      const float2 w=ex*v[2*p*N+col]+ey*v[(2*p+1)*N+col];
      a[row*N+col]=mode?diffraction_complex_mul(normal,b-w)+(e.w*e.w)*(b+w):
                            b-w+diffraction_complex_mul(normal,b+w);
    }
    const float source=p==incoming && e.z>0 ? 2*sqrtf(q)*(mode?e.w:1):0;
    rhs[2*row]=make_float2(source*ex,0);rhs[2*row+1]=make_float2(source*ey,0);
  }
  if(!diffraction_complex_solve<N>(a,rhs))return false;
  for(int p=0;p<N/2;p++) {
    const float4 e=exterior[p];
    for(int j=0;j<2;j++) {
      float2 te=zero_float2(),tm=zero_float2();
      if(e.z>0) {
        float2 plus[2]={zero_float2(),zero_float2()},minus[2]={zero_float2(),zero_float2()};
        for(int c=0;c<2;c++)for(int k=0;k<N;k++) {
          const float2 b=make_float2(float(2*p+c==k),0)+chart[(2*p+c)*N+k];
          plus[c]=plus[c]+diffraction_complex_mul(b+v[(2*p+c)*N+k],rhs[2*k+j]);
          minus[c]=minus[c]+diffraction_complex_mul(b-v[(2*p+c)*N+k],rhs[2*k+j]);
        }
        const float root=sqrtf(sqrtf(e.z));
        te=root*(-e.y*plus[0]+e.x*plus[1])-make_float2(p==incoming?(j?e.x:-e.y):0,0);
        tm=make_float2(p==incoming?(j?e.y:e.x):0,0)-(root/e.w)*(e.x*minus[0]+e.y*minus[1]);
      }
      jones[4*p+j]=-e.y*te+e.x*tm;
      jones[4*p+2+j]=e.x*te+e.y*tm;
    }
  }
  return true;
}
CCL_NAMESPACE_END
