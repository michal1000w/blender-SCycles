/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/globals.h"

#include "kernel/geom/object.h"

#include "util/math_float2.h"
#include "util/math_float3.h"
#include "util/math_intersect.h"

CCL_NAMESPACE_BEGIN

#ifdef __KERNEL_METAL__

ccl_device_inline bool pixel_displacement_active(KernelGlobals kg, const int prim)
{
  if (!kernel_data.integrator.use_pixel_displacement ||
      kernel_data.integrator.pixel_displacement_scale == 0.0f ||
      kernel_data.integrator.pixel_displacement_max_distance <= 0.0f)
  {
    return false;
  }

  const int shader = kernel_data_fetch(tri_shader, prim);
  const int shader_index = shader & SHADER_MASK;
  return (kernel_data_fetch(shaders, shader_index).flags & SD_HAS_DISPLACEMENT) != 0;
}

ccl_device_inline float3 pixel_displacement_base_position(ccl_private const float3 verts[3],
                                                          const float u,
                                                          const float v)
{
  return verts[0] + u * (verts[1] - verts[0]) + v * (verts[2] - verts[0]);
}

ccl_device_inline float3 pixel_displacement_face_normal(ccl_private const float3 verts[3],
                                                        const uint object_flag)
{
  if (object_negative_scale_applied(object_flag)) {
    return normalize(cross(verts[2] - verts[0], verts[1] - verts[0]));
  }
  return normalize(cross(verts[1] - verts[0], verts[2] - verts[0]));
}

ccl_device_inline float2 pixel_displacement_bary_from_point(ccl_private const float3 verts[3],
                                                            const float3 P)
{
  const float3 e0 = verts[1] - verts[0];
  const float3 e1 = verts[2] - verts[0];
  const float3 vp = P - verts[0];

  const float d00 = dot(e0, e0);
  const float d01 = dot(e0, e1);
  const float d11 = dot(e1, e1);
  const float d20 = dot(vp, e0);
  const float d21 = dot(vp, e1);
  const float denom = d00 * d11 - d01 * d01;

  if (fabsf(denom) < 1.0e-20f) {
    return make_float2(0.0f, 0.0f);
  }

  const float inv_denom = 1.0f / denom;
  return make_float2((d11 * d20 - d01 * d21) * inv_denom,
                     (d00 * d21 - d01 * d20) * inv_denom);
}

ccl_device_inline bool pixel_displacement_clip_greater_equal_zero(const float f0,
                                                                  const float f1,
                                                                  ccl_private float *slo,
                                                                  ccl_private float *shi)
{
  if (f0 >= 0.0f && f1 >= 0.0f) {
    return true;
  }

  const float denom = f0 - f1;
  if (fabsf(denom) < 1.0e-20f) {
    return false;
  }

  const float s = clamp(f0 / denom, 0.0f, 1.0f);
  if (f0 < 0.0f && f1 >= 0.0f) {
    *slo = max(*slo, s);
  }
  else if (f0 >= 0.0f && f1 < 0.0f) {
    *shi = min(*shi, s);
  }
  else {
    return false;
  }

  return *shi >= *slo;
}

ccl_device_inline bool pixel_displacement_intersect_surface(
    KernelGlobals kg,
    const float3 P,
    const float3 dir,
    const float tmin,
    const float tmax,
    const float time,
    const int object,
    const int prim,
    const bool motion,
    ccl_private const float3 verts[3],
    ccl_private float *r_u,
    ccl_private float *r_v,
    ccl_private float *r_t)
{
  (void)time;
  (void)motion;

  if (!pixel_displacement_active(kg, prim)) {
    return false;
  }

  const uint object_flag = kernel_data_fetch(object_flag, object);
  const float3 Ng = pixel_displacement_face_normal(verts, object_flag);
  const float max_distance = kernel_data.integrator.pixel_displacement_max_distance;

  const float origin_plane = dot(P - verts[0], Ng);
  const float dir_plane = dot(dir, Ng);

  float t0 = tmin;
  float t1 = tmax;
  if (fabsf(dir_plane) > 1.0e-8f) {
    const float ta = (-max_distance - origin_plane) / dir_plane;
    const float tb = (max_distance - origin_plane) / dir_plane;
    t0 = max(t0, min(ta, tb));
    t1 = min(t1, max(ta, tb));
  }
  else if (origin_plane < -max_distance || origin_plane > max_distance) {
    return false;
  }

  if (t1 <= t0) {
    return false;
  }

  const float3 projected_origin = P - Ng * origin_plane;
  const float3 projected_dir = dir - Ng * dir_plane;
  const float2 bary0 = pixel_displacement_bary_from_point(verts, projected_origin + projected_dir * t0);
  const float2 bary1 = pixel_displacement_bary_from_point(verts, projected_origin + projected_dir * t1);

  const float3 e0 = verts[1] - verts[0];
  const float3 e1 = verts[2] - verts[0];
  const float3 e2 = verts[2] - verts[1];
  const float min_edge_len = sqrtf(max(min(min(dot(e0, e0), dot(e1, e1)), dot(e2, e2)), 1.0e-8f));
  const float bary_eps = clamp(max_distance / min_edge_len + 5.0e-4f, 5.0e-4f, 0.25f);

  float slo = 0.0f;
  float shi = 1.0f;
  if (!pixel_displacement_clip_greater_equal_zero(
          bary0.x + bary_eps, bary1.x + bary_eps, &slo, &shi) ||
      !pixel_displacement_clip_greater_equal_zero(
          bary0.y + bary_eps, bary1.y + bary_eps, &slo, &shi) ||
      !pixel_displacement_clip_greater_equal_zero(1.0f - bary0.x - bary0.y + bary_eps,
                                                 1.0f - bary1.x - bary1.y + bary_eps,
                                                 &slo,
                                                 &shi))
  {
    return false;
  }

  const float t = mix(t0, t1, slo);
  float2 bary = mix(bary0, bary1, slo);
  bary.x = clamp(bary.x, 0.0f, 1.0f);
  bary.y = clamp(bary.y, 0.0f, 1.0f - bary.x);

  *r_u = bary.x;
  *r_v = bary.y;
  *r_t = t;
  return true;
}

#else

ccl_device_inline bool pixel_displacement_active(KernelGlobals /*kg*/, const int /*prim*/)
{
  return false;
}

ccl_device_inline float3 pixel_displacement_face_normal(ccl_private const float3 verts[3],
                                                        const uint object_flag)
{
  if (object_negative_scale_applied(object_flag)) {
    return normalize(cross(verts[2] - verts[0], verts[1] - verts[0]));
  }
  return normalize(cross(verts[1] - verts[0], verts[2] - verts[0]));
}

ccl_device_inline bool pixel_displacement_intersect_surface(
    KernelGlobals /*kg*/,
    const float3 /*P*/,
    const float3 /*dir*/,
    const float /*tmin*/,
    const float /*tmax*/,
    const float /*time*/,
    const int /*object*/,
    const int /*prim*/,
    const bool /*motion*/,
    ccl_private const float3 /*verts*/[3],
    ccl_private float * /*r_u*/,
    ccl_private float * /*r_v*/,
    ccl_private float * /*r_t*/)
{
  return false;
}

ccl_device_inline void pixel_displacement_displaced_geometry(
    KernelGlobals /*kg*/,
    const int /*object*/,
    const int /*prim*/,
    const float /*u*/,
    const float /*v*/,
    const float /*time*/,
    const bool /*motion*/,
    ccl_private const float3 /*verts*/[3],
    ccl_private float3 *P_obj,
    ccl_private float3 *Ng_obj,
    ccl_private float3 *dPdu_obj,
    ccl_private float3 *dPdv_obj)
{
  *P_obj = zero_float3();
  *Ng_obj = zero_float3();
  *dPdu_obj = zero_float3();
  *dPdv_obj = zero_float3();
}

ccl_device_inline void pixel_displacement_shader_setup(KernelGlobals /*kg*/,
                                                       ccl_private ShaderData * /*sd*/,
                                                       const float /*time*/,
                                                       const bool /*motion*/,
                                                       ccl_private const float3 /*verts*/[3])
{
}

#endif

CCL_NAMESPACE_END
