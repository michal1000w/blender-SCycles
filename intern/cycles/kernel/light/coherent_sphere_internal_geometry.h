/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/light/coherent_sphere_transmit_geometry.h"
CCL_NAMESPACE_BEGIN

/* Debye orders: T-R-T and T-R-R-T, with all isolated signed-angular-momentum
 * roots over every possible winding. Not yet connected to production ownership. */
#define COHERENT_SPHERE_INTERNAL_MAX_BRANCHES 9
struct CoherentSphereInternalInventory {
  CoherentGeometryPath path[COHERENT_SPHERE_INTERNAL_MAX_BRANCHES];
  CoherentGeometryInterface frame[COHERENT_SPHERE_INTERNAL_MAX_BRANCHES][4];
  float angular_momentum[COHERENT_SPHERE_INTERNAL_MAX_BRANCHES];
  float maslov_phase_cycles[COHERENT_SPHERE_INTERNAL_MAX_BRANCHES];
  int morse_index[COHERENT_SPHERE_INTERNAL_MAX_BRANCHES], winding[COHERENT_SPHERE_INTERNAL_MAX_BRANCHES];
  int count;
  CoherentSphereTTStatus status;
};
ccl_device_inline float coherent_sphere_internal_equation(
    const float t, const float A, const float B, const float n,
    const int chords, const float target)
{
  return chords * M_PI_F + 2 * asinf(t) - asinf(A*t) - asinf(B*t) -
         2*chords*asinf(t/n) - target;
}
ccl_device_inline float coherent_sphere_internal_derivative(
    const float t, const float A, const float B, const float n, const int chords)
{
  const float v = max(0.0f, 1-t*t);
  return 2 - A*sqrtf(v/(1-A*A*t*t)) - B*sqrtf(v/(1-B*B*t*t)) -
         (2*chords/n)*sqrtf(v/(1-t*t/(n*n)));
}
#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device_noinline
#endif
CoherentSphereTTStatus coherent_sphere_internal_roots(
    const float A, const float B, const float n, const int chords, const float theta,
    ccl_private float roots[COHERENT_SPHERE_INTERNAL_MAX_BRANCHES],
    ccl_private int windings[COHERENT_SPHERE_INTERNAL_MAX_BRANCHES], ccl_private int *count)
{
  *count = 0;
  float bounds[4] = {-1, 1, 0, 0};
  int intervals = 1;
  if (coherent_sphere_internal_derivative(0,A,B,n,chords) < 0) {
    float lo=0, hi=1;
    for (int j=0;j<32;j++) {
      const float mid=(lo+hi)*.5f;
      if (coherent_sphere_internal_derivative(mid,A,B,n,chords)<0) lo=mid;
      else hi=mid;
    }
    const float critical=(lo+hi)*.5f;
    bounds[1]=-critical; bounds[2]=critical; bounds[3]=1; intervals=3;
  }
  /* For p chords and n>=1, 0<=Delta(t)<=2*p*pi. For 0<theta<pi
   * exactly the windings 0..p-1 are possible. Each has <=3 monotone roots. */
  for (int winding=0;winding<chords;winding++) {
    const float target=theta+2*winding*M_PI_F;
    for (int k=0;k<=intervals;k++) {
      const float f=coherent_sphere_internal_equation(bounds[k],A,B,n,chords,target);
      if (fabsf(f)<1e-6f) return k==0||k==intervals ?
          COHERENT_SPHERE_TT_GRAZING : COHERENT_SPHERE_TT_CAUSTIC;
    }
    for (int k=0;k<intervals;k++) {
      float lo=bounds[k],hi=bounds[k+1];
      float fl=coherent_sphere_internal_equation(lo,A,B,n,chords,target);
      const float fh=coherent_sphere_internal_equation(hi,A,B,n,chords,target);
      if ((fl<0)==(fh<0)) continue;
      for (int j=0;j<32;j++) {
        const float mid=(lo+hi)*.5f;
        const float fm=coherent_sphere_internal_equation(mid,A,B,n,chords,target);
        if (fm==0) {lo=hi=mid;break;}
        if ((fm<0)==(fl<0)) {lo=mid;fl=fm;} else hi=mid;
      }
      if (*count>=COHERENT_SPHERE_INTERNAL_MAX_BRANCHES) return COHERENT_SPHERE_TT_INVALID;
      roots[*count]=(lo+hi)*.5f;windings[*count]=winding;(*count)++;
    }
  }
  for (int i=1;i<*count;i++) {
    const float t=roots[i]; const int w=windings[i]; int j=i;
    while(j>0 && roots[j-1]>t) {roots[j]=roots[j-1];windings[j]=windings[j-1];j--;}
    roots[j]=t;windings[j]=w;
  }
  return *count ? COHERENT_SPHERE_TT_OK : COHERENT_SPHERE_TT_EMPTY;
}

ccl_device_inline bool coherent_sphere_internal_inertia(
    const ccl_private float H[COHERENT_GEOMETRY_DIM][COHERENT_GEOMETRY_DIM],
    const int dim, ccl_private int *negative)
{
  float a[COHERENT_GEOMETRY_DIM][COHERENT_GEOMETRY_DIM];
  for(int i=0;i<dim;i++) for(int j=0;j<dim;j++) a[i][j]=H[i][j];
  for(int iteration=0;iteration<160;iteration++) {
    int p=0,q=1;float biggest=0;
    for(int i=0;i<dim;i++) for(int j=i+1;j<dim;j++)
      if(fabsf(a[i][j])>biggest){biggest=fabsf(a[i][j]);p=i;q=j;}
    if(biggest==0) break;
    const float angle=.5f*atan2f(2*a[p][q],a[q][q]-a[p][p]);
    const float c=cosf(angle),s=sinf(angle),app=a[p][p],aqq=a[q][q],apq=a[p][q];
    for(int k=0;k<dim;k++) if(k!=p&&k!=q) {
      const float x=a[k][p],y=a[k][q];a[k][p]=a[p][k]=c*x-s*y;a[k][q]=a[q][k]=s*x+c*y;
    }
    a[p][p]=c*c*app-2*c*s*apq+s*s*aqq;
    a[q][q]=s*s*app+2*c*s*apq+c*c*aqq;a[p][q]=a[q][p]=0;
  }
  float largest=0,smallest=1e30f;*negative=0;
  for(int i=0;i<dim;i++){largest=max(largest,fabsf(a[i][i]));smallest=min(smallest,fabsf(a[i][i]));*negative+=a[i][i]<0;}
  return largest>0&&smallest>1e-6f*largest;
}

#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device_noinline
#endif
CoherentSphereTTStatus coherent_sphere_internal_inventory(
    const float3 source,const float3 receiver,float3 receiver_normal,const float3 center,
    const float radius,const float ior,const int reflections,
    ccl_private CoherentSphereInternalInventory *out, const int selected_branch = -1)
{
  out->count=0;out->status=COHERENT_SPHERE_TT_INVALID;
  const float3 s=source-center,r=receiver-center;const float a=len(s),b=len(r);
  if(!(radius>0&&ior>1&&a>radius&&b>radius&&len(receiver_normal)>0) ||
      !isfinite_safe(radius+ior+a+b+len(receiver_normal)) || reflections<1||reflections>2) return out->status;
  receiver_normal=normalize(receiver_normal);
  const float3 e0=s/a,rd=r/b;const float cosine=clamp(dot(e0,rd),-1.f,1.f);
  const float3 perpendicular=rd-e0*cosine;const float sine=len(perpendicular);
  if(!(sine>1e-7f)){out->status=COHERENT_SPHERE_TT_AXIAL_RING;return out->status;}
  const float3 e1=perpendicular/sine;const float theta=atan2f(sine,cosine);
  const int chords=reflections+1,count=reflections+2,dim=2*count;
  float roots[COHERENT_SPHERE_INTERNAL_MAX_BRANCHES];int winding[COHERENT_SPHERE_INTERNAL_MAX_BRANCHES],branches=0;
  out->status=coherent_sphere_internal_roots(radius/a,radius/b,ior,chords,theta,roots,winding,&branches);
  if(out->status!=COHERENT_SPHERE_TT_OK)return out->status;
  if(selected_branch>=branches){out->status=COHERENT_SPHERE_TT_EMPTY;return out->status;}
  /* In selected mode count is the isolated root count; only path[selected]
   * is initialized. The production caller invokes every reserved branch. */
  for(int branch=0;branch<branches;branch++) {
    if(selected_branch>=0 && branch!=selected_branch) continue;
    const float t=roots[branch],phi=asinf(t)-asinf(radius/a*t),advance=M_PI_F-2*asinf(t/ior);
    float2 positions[COHERENT_GEOMETRY_MAX_INTERFACES+2][3];float3 N[4],basis[8];
    for(int k=0;k<3;k++) {
      positions[0][k]=make_float2(k==0?source.x:k==1?source.y:source.z,0);
      positions[count+1][k]=make_float2(k==0?receiver.x:k==1?receiver.y:receiver.z,0);
    }
    for(int k=0;k<count;k++) {
      const float angle=phi+k*advance;
      coherent_sphere_tt_point(center,radius,e0*cosf(angle)+e1*sinf(angle),positions[k+1],&N[k]);
      make_orthonormals(N[k],&basis[2*k],&basis[2*k+1]);
    }
    float3 v[5];float L[5],weights[5];CoherentGeometryPath path{};
    path.count=count;path.optical_length_split=make_float2(0,0);
    for(int k=0;k<=count;k++) {
      const float2 length=coherent_geometry_distance_split_affine(positions[k],positions[k+1]);
      L[k]=length.x+length.y;weights[k]=(k==0||k==count)?1:ior;
      float3 delta;
      for(int axis=0;axis<3;axis++) {
        const float2 d=coherent_geometry_add_split(positions[k+1][axis],make_float2(-positions[k][axis].x,-positions[k][axis].y));
        if(axis==0)delta.x=d.x+d.y;if(axis==1)delta.y=d.x+d.y;if(axis==2)delta.z=d.x+d.y;
      }
      if(!(L[k]>0)){out->status=COHERENT_SPHERE_TT_INVALID;return out->status;}
      v[k]=delta/L[k];path.segment_length[k]=L[k];
      path.optical_length_split=coherent_geometry_add_split(path.optical_length_split,coherent_geometry_product_split(length,weights[k]));
    }
    if(!(dot(v[0],N[0])<0&&dot(v[1],N[0])<0&&dot(v[count-1],N[count-1])>0&&dot(v[count],N[count-1])>0)) {
      out->status=COHERENT_SPHERE_TT_INVALID;return out->status;
    }
    for(int k=1;k<count-1;k++) if(!(dot(v[k],N[k])>0&&dot(v[k+1],N[k])<0)) {
      out->status=COHERENT_SPHERE_TT_INVALID;return out->status;
    }
    float H[COHERENT_GEOMETRY_DIM][COHERENT_GEOMETRY_DIM]={{0}};
    for(int i=0;i<dim;i++) for(int j=0;j<dim;j++) {
      const int si=i/2,sj=j/2;const float3 u=basis[i],w=basis[j];
      if(si==sj) {
        H[i][j]=weights[si]*(dot(u,w)-dot(u,v[si])*dot(w,v[si]))/L[si]+
                weights[si+1]*(dot(u,w)-dot(u,v[si+1])*dot(w,v[si+1]))/L[si+1];
        if(i==j) H[i][j]-=dot(weights[si]*v[si]-weights[si+1]*v[si+1],N[si])/radius;
      }
      else if(abs(si-sj)==1) {
        const int segment=max(si,sj);
        H[i][j]=-weights[segment]*(dot(u,w)-dot(u,v[segment])*dot(w,v[segment]))/L[segment];
      }
    }
    int morse;
    if(!coherent_sphere_internal_inertia(H,dim,&morse)){out->status=COHERENT_SPHERE_TT_CAUSTIC;return out->status;}
    float3 ru,rv,su,sv;make_orthonormals(receiver_normal,&ru,&rv);make_orthonormals(v[0],&su,&sv);float J[4];
    for(int axis=0;axis<2;axis++) {
      const float3 rb=axis?rv:ru,rhs3=(rb-v[count]*dot(v[count],rb))/L[count];
      float rhs[COHERENT_GEOMETRY_DIM]={0},q[COHERENT_GEOMETRY_DIM]={0};
      rhs[dim-2]=dot(basis[dim-2],rhs3);rhs[dim-1]=dot(basis[dim-1],rhs3);
      if(!coherent_geometry_linear_solve(dim,H,rhs,q)){out->status=COHERENT_SPHERE_TT_CAUSTIC;return out->status;}
      const float3 motion=basis[0]*q[0]+basis[1]*q[1],d=(motion-v[0]*dot(v[0],motion))/L[0];
      J[axis]=dot(su,d);J[2+axis]=dot(sv,d);
    }
    path.spreading=fabsf(J[0]*J[3]-J[1]*J[2]);path.source_direction=v[0];
    if(!isfinite_safe(path.spreading)){out->status=COHERENT_SPHERE_TT_CAUSTIC;return out->status;}
    path.optical_length=path.optical_length_split.x+path.optical_length_split.y;
    for(int k=0;k<count;k++) {
      path.point[k]=make_float3(positions[k+1][0].x+positions[k+1][0].y,positions[k+1][1].x+positions[k+1][1].y,positions[k+1][2].x+positions[k+1][2].y);
      const bool entry=k==0,exit=k==count-1;
      out->frame[branch][k]={path.point[k],basis[2*k],basis[2*k+1],0,0,entry?1:ior,exit?1:ior,entry?ior:1,
                             entry||exit?COHERENT_GEOMETRY_TRANSMIT:COHERENT_GEOMETRY_REFLECT,entry?1:-1};
    }
    out->path[branch]=path;out->angular_momentum[branch]=t*radius;out->winding[branch]=winding[branch];
    out->morse_index[branch]=morse;out->maslov_phase_cycles[branch]=-.25f*morse;
  }
  out->count=branches;out->status=COHERENT_SPHERE_TT_OK;return out->status;
}
CCL_NAMESPACE_END
