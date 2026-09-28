/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/light/coherent_unfold_geometry.h"
#include "kernel/light/coherent_sphere_internal_geometry.h"
CCL_NAMESPACE_BEGIN
#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device_noinline
#endif
CoherentSphereTTStatus coherent_unfold_internal_connect(
    const float3 source,const float3 receiver,const float3 receiver_normal,
    ccl_private CoherentGeometryInterface *frames,const int count,const int sphere_first,
    const int reflections,const int branch,const float3 center,const float radius,const float ior,
    ccl_private CoherentGeometryPath *path,ccl_private float *phase)
{
  const int sphere_count=reflections+2;
  *phase=0;
  if(count>4||sphere_first<0||sphere_first+sphere_count>count||reflections<1||reflections>2)
    return COHERENT_SPHERE_TT_INVALID;
  float2 si[3]={make_float2(source.x,0),make_float2(source.y,0),make_float2(source.z,0)};
  float2 ri[3]={make_float2(receiver.x,0),make_float2(receiver.y,0),make_float2(receiver.z,0)};
  for(int i=0;i<sphere_first;i++) coherent_unfold_image(si,frames[i]);
  float3 image_normal=receiver_normal;
  for(int i=count-1;i>=sphere_first+sphere_count;i--) {
    coherent_unfold_image(ri,frames[i]);
    const float3 n=normalize(cross(frames[i].tangent_u,frames[i].tangent_v));
    image_normal-=(2*dot(image_normal,n)/dot(n,n))*n;
  }
  const float3 vs=make_float3(si[0].x,si[1].x,si[2].x),vr=make_float3(ri[0].x,ri[1].x,ri[2].x);
  if(!(len(vs-center)>radius&&len(vr-center)>radius)) return COHERENT_SPHERE_TT_EMPTY;
  CoherentSphereInternalInventory inventory;
  const auto status=coherent_sphere_internal_inventory(vs,vr,image_normal,center,radius,ior,reflections,&inventory,branch);
  if(status!=COHERENT_SPHERE_TT_OK)return status;
  if(branch<0||branch>=inventory.count)return COHERENT_SPHERE_TT_EMPTY;
  for(int i=0;i<sphere_count;i++)frames[sphere_first+i]=inventory.frame[branch][i];
  *phase=inventory.maslov_phase_cycles[branch];
  return coherent_unfold_finish(source,receiver,si,ri,vs,vr,frames,count,sphere_first,sphere_count,
                                 inventory.path[branch],path)?COHERENT_SPHERE_TT_OK:COHERENT_SPHERE_TT_EMPTY;
}
CCL_NAMESPACE_END
