/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/globals.h"
#include "kernel/types.h"

#include "kernel/geom/gsplat.h"
#include "kernel/geom/motion_point.h"
#include "kernel/geom/object.h"
#include "kernel/geom/point_intersect_policy.h"

CCL_NAMESPACE_BEGIN

/* Point primitive intersection functions. */

#ifdef __POINTCLOUD__

template<bool use_backface_culling = true>
ccl_device_forceinline bool point_intersect_test(const float4 point,
                                                 const float3 ray_P,
                                                 const float3 ray_D,
                                                 const float ray_tmin,
                                                 const float ray_tmax,
                                                 ccl_private float *t,
                                                 const bool two_sided = false)
{
  const float3 center = make_float3(point);
  const float radius = point.w;

  const float rd2 = 1.0f / dot(ray_D, ray_D);

  const float3 c0 = center - ray_P;
  const float projC0 = dot(c0, ray_D) * rd2;
  const float3 perp = c0 - projC0 * ray_D;
  const float l2 = dot(perp, perp);
  const float r2 = radius * radius;
  if (!(l2 <= r2)) {
    return false;
  }

  const float td = sqrt((r2 - l2) * rd2);
  const float t_front = projC0 - td;
  const bool valid_front = (ray_tmin <= t_front) & (t_front <= ray_tmax);

  if (valid_front) {
    *t = t_front;
    return true;
  }
  if (two_sided) {
    const float t_back = projC0 + td;
    if ((ray_tmin <= t_back) && (t_back <= ray_tmax)) {
      *t = t_back;
      return true;
    }
  }
  return false;
}

ccl_device_forceinline bool point_intersect(KernelGlobals kg,
                                            ccl_private Intersection *isect,
                                            const float3 ray_P,
                                            const float3 ray_D,
                                            const float ray_tmin,
                                            const float ray_tmax,
                                            const int object,
                                            const int prim,
                                            const float time,
                                            const int type)
{
  const int position_offset = kernel_data_fetch(objects, object).position_offset;
  const float4 point = (type & PRIMITIVE_MOTION) ?
                           motion_point(kg, object, prim, time) :
                           kernel_data_fetch(points, position_offset + prim);

  const bool two_sided = kernel_data.integrator.coherent_specular_enabled &&
                         point_coherent_glass_two_sided(
                             true, kernel_data_fetch(object_flag, object), type);
  if (!point_intersect_test(point, ray_P, ray_D, ray_tmin, ray_tmax, &isect->t, two_sided)) {
    return false;
  }

  isect->prim = prim;
  isect->object = object;
  isect->type = type;
  isect->u = 0.0f;
  isect->v = 0.0f;
  return true;
}

ccl_device_inline void point_shader_setup(KernelGlobals kg,
                                          ccl_private ShaderData *sd,
                                          const ccl_private Intersection *isect,
                                          const ccl_private Ray *ray)
{
  sd->shader = kernel_data_fetch(points_shader, isect->prim);
  sd->P = ray->P + ray->D * isect->t;

  /* Texture coordinates, zero for now. */
#  ifdef __UV__
  sd->u = isect->u;
  sd->v = isect->v;
#  endif

  /* Compute point center for normal. */
  const int position_offset = kernel_data_fetch(objects, sd->object).position_offset;
  float3 center = make_float3((isect->type & PRIMITIVE_MOTION) ?
                                  motion_point(kg, sd->object, sd->prim, sd->time) :
                                  kernel_data_fetch(points, position_offset + sd->prim));
  if (!(sd->object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    object_position_transform(kg, sd, &center);
  }

  /* Normal */
#  if defined(__GSPLATS__)
  if (isect->type & PRIMITIVE_POINT)
#  endif
  {
    sd->Ng = normalize(sd->P - center);
  }
#  if defined(__GSPLATS__)
  else {
    sd->Ng = gsplat_normal(kg, sd->object, sd->prim, sd->time, sd->type);
    kernel_assert((kernel_data_fetch(object_flag, sd->object) & SD_OBJECT_TRANSFORM_APPLIED) == 0);
    object_normal_transform(kg, sd, &sd->Ng);
  }
#  endif

  sd->N = sd->Ng;

#  ifdef __DPDU__
  /* dPdu/dPdv: arbitrary tangent frame so bump mapping works. */
  make_orthonormals(sd->Ng, &sd->dPdu, &sd->dPdv);
#  endif
}

#endif

CCL_NAMESPACE_END
