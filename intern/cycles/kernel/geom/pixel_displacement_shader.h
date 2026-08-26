/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef KERNEL_GEOM_PIXEL_DISPLACEMENT_SHADER_H_
#define KERNEL_GEOM_PIXEL_DISPLACEMENT_SHADER_H_

CCL_NAMESPACE_BEGIN

#if defined(__KERNEL_METAL__) && (defined(__KERNEL_METAL_PIXEL_DISPLACEMENT__) || \
                                  defined(__KERNEL_METAL_PIXEL_DISPLACEMENT_SHADE__))

ccl_device_inline float3 pixel_displacement_smooth_normal(KernelGlobals kg,
                                                          const int object,
                                                          const int prim,
                                                          const uint object_flag,
                                                          const int shader,
                                                          const float u,
                                                          const float v,
                                                          const float time,
                                                          const bool motion,
                                                          const float3 Ng)
{
  if (!(shader & SHADER_SMOOTH_NORMAL)) {
    return Ng;
  }

  if (motion) {
    int numsteps;
    int step;
    float t;
    uint3 tri_vindex;
    motion_triangle_compute_info(kg, object, time, prim, &tri_vindex, &numsteps, &step, &t);

    float3 normals[3];
    motion_triangle_normals(kg, object, prim, tri_vindex, numsteps, step, t, normals);
    const float3 N = safe_normalize(
        triangle_interpolate(u, v, normals[0], normals[1], normals[2]));
    return is_zero(N) ? Ng : N;
  }

  return triangle_smooth_normal(kg, Ng, object, object_flag, prim, u, v);
}

ccl_device_inline void pixel_displacement_setup_shader_data(KernelGlobals kg,
                                                            ccl_private ShaderData *sd,
                                                            const int object,
                                                            const int prim,
                                                            const int shader,
                                                            const float u,
                                                            const float v,
                                                            const float time,
                                                            const bool motion,
                                                            const float3 P_obj,
                                                            const float3 Ng_obj,
                                                            const float3 dPdu_obj,
                                                            const float3 dPdv_obj)
{
  const int displacement_shader = shader | SHADER_SMOOTH_NORMAL;
  sd->P = P_obj;
  sd->Ng = Ng_obj;
  sd->N = pixel_displacement_smooth_normal(kg,
                                           object,
                                           prim,
                                           kernel_data_fetch(object_flag, object),
                                           displacement_shader,
                                           u,
                                           v,
                                           time,
                                           motion,
                                           Ng_obj);
  sd->wi = sd->N;
  sd->shader = displacement_shader;
  sd->type = PRIMITIVE_TRIANGLE;
  sd->object = object;
  sd->prim = prim;
  sd->u = u;
  sd->v = v;
  sd->time = time;
  sd->ray_length = 0.0f;
  sd->flag = kernel_data_fetch(shaders, (sd->shader & SHADER_MASK)).flags;
  sd->object_flag = kernel_data_fetch(object_flag, object);
  sd->ray_P = zero_float3();
  sd->ray_dP = 0.0f;

#  ifdef __OBJECT_MOTION__
  if (sd->object_flag & SD_OBJECT_MOTION) {
    sd->ob_tfm_motion = object_fetch_transform_motion(kg, object, time);
    sd->ob_itfm_motion = transform_inverse(sd->ob_tfm_motion);
  }
#  endif

#  ifdef __DPDU__
  sd->dPdu = dPdu_obj;
  sd->dPdv = dPdv_obj;
#  endif

  if (!(sd->object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    object_position_transform(kg, sd, &sd->P);
    object_normal_transform(kg, sd, &sd->Ng);
    object_normal_transform(kg, sd, &sd->N);
#  ifdef __DPDU__
    object_dir_transform(kg, sd, &sd->dPdu);
    object_dir_transform(kg, sd, &sd->dPdv);
#  endif
  }

  sd->wi = sd->N;

#  ifdef __RAY_DIFFERENTIALS__
#    ifdef __DPDU__
  sd->dP = 0.5f * (len(sd->dPdu) + len(sd->dPdv));
#    else
  sd->dP = 0.5f * (len(dPdu_obj) + len(dPdv_obj));
#    endif
  sd->dI = 0.0f;
  sd->du.dx = 1.0f;
  sd->du.dy = 0.0f;
  sd->dv.dx = 0.0f;
  sd->dv.dy = 1.0f;
#  endif
}

ccl_device_noinline float3 pixel_displacement_eval_object_direct(KernelGlobals kg,
                                                                 const int object,
                                                                 const int prim,
                                                                 const float u,
                                                                 const float v,
                                                                 const float time,
                                                                 const bool motion,
                                                                 ccl_private const float3 verts[3])
{
  const int shader = kernel_data_fetch(tri_shader, prim);
  const uint object_flag = kernel_data_fetch(object_flag, object);
  const float3 P_obj = pixel_displacement_base_position(verts, u, v);
  const float3 Ng_obj = pixel_displacement_face_normal(verts, object_flag);
  const float3 dPdu_obj = verts[1] - verts[0];
  const float3 dPdv_obj = verts[2] - verts[0];

  ShaderData sd;
  pixel_displacement_setup_shader_data(
      kg, &sd, object, prim, shader, u, v, time, motion, P_obj, Ng_obj, dPdu_obj, dPdv_obj);

  ConstIntegratorBakeState state;
  const float3 P = sd.P;
  displacement_shader_eval(kg, state, &sd);

  float3 D = ensure_finite(sd.P - P);
  if (!(object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    object_inverse_dir_transform(kg, &sd, &D);
  }
  D *= kernel_data.integrator.pixel_displacement_scale;

  const float max_distance = kernel_data.integrator.pixel_displacement_max_distance;
  const float distance = len(D);
  if (distance > max_distance && distance > 0.0f) {
    D *= max_distance / distance;
  }

  return ensure_finite(D);
}

ccl_device_inline int pixel_displacement_cache_sample_index(const int grid,
                                                            const int u,
                                                            const int v)
{
  return u * (grid + 1) - (u * (u - 1)) / 2 + v;
}

ccl_device_inline float3 pixel_displacement_cache_sample(const int offset,
                                                         const int grid,
                                                         const int u,
                                                         const int v)
{
  const float4 D = kernel_data_fetch(pixel_displacement_data,
                                     offset + pixel_displacement_cache_sample_index(grid, u, v));
  return make_float3(D.x, D.y, D.z);
}

ccl_device_inline bool pixel_displacement_cache_lookup(KernelGlobals kg,
                                                       const int prim,
                                                       const bool motion,
                                                       ccl_private int *r_grid,
                                                       ccl_private int *r_offset)
{
  if (motion) {
    return false;
  }

  const int grid = int(kernel_data_fetch(pixel_displacement_info, prim));
  if (grid <= 0) {
    return false;
  }

  const int offset = kernel_data_fetch(pixel_displacement_offset, prim);
  if (offset < 0) {
    return false;
  }

  *r_grid = grid;
  *r_offset = offset;
  return true;
}

ccl_device_inline bool pixel_displacement_eval_object_cached(KernelGlobals kg,
                                                             const int prim,
                                                             const float u,
                                                             const float v,
                                                             const bool motion,
                                                             ccl_private float3 *r_D)
{
  int grid, offset;
  if (!pixel_displacement_cache_lookup(kg, prim, motion, &grid, &offset)) {
    return false;
  }

  float cu = clamp(u, 0.0f, 1.0f);
  float cv = clamp(v, 0.0f, 1.0f);
  const float sum = cu + cv;
  if (sum > 1.0f) {
    cu /= sum;
    cv /= sum;
  }

  const float fu = cu * float(grid);
  const float fv = cv * float(grid);
  int iu = min(int(floorf(fu)), grid);
  int iv = min(int(floorf(fv)), grid - iu);

  if (iu + iv >= grid) {
    *r_D = pixel_displacement_cache_sample(offset, grid, iu, iv);
    return true;
  }

  const float du = fu - float(iu);
  const float dv = fv - float(iv);

  if (du + dv <= 1.0f || iu + iv + 2 > grid) {
    const float3 D00 = pixel_displacement_cache_sample(offset, grid, iu, iv);
    const float3 D10 = pixel_displacement_cache_sample(offset, grid, iu + 1, iv);
    const float3 D01 = pixel_displacement_cache_sample(offset, grid, iu, iv + 1);
    *r_D = D00 * (1.0f - du - dv) + D10 * du + D01 * dv;
    return true;
  }

  const float3 D10 = pixel_displacement_cache_sample(offset, grid, iu + 1, iv);
  const float3 D11 = pixel_displacement_cache_sample(offset, grid, iu + 1, iv + 1);
  const float3 D01 = pixel_displacement_cache_sample(offset, grid, iu, iv + 1);
  *r_D = D10 * (1.0f - dv) + D11 * (du + dv - 1.0f) + D01 * (1.0f - du);
  return true;
}

ccl_device_inline float3 pixel_displacement_eval_object(KernelGlobals kg,
                                                        const int object,
                                                        const int prim,
                                                        const float u,
                                                        const float v,
                                                        const float time,
                                                        const bool motion,
                                                        ccl_private const float3 verts[3])
{
  float3 D;
  if (pixel_displacement_eval_object_cached(kg, prim, u, v, motion, &D)) {
    return D;
  }

  return pixel_displacement_eval_object_direct(kg, object, prim, u, v, time, motion, verts);
}

ccl_device_inline float3 pixel_displacement_position(KernelGlobals kg,
                                                     const int object,
                                                     const int prim,
                                                     const float u,
                                                     const float v,
                                                     const float time,
                                                     const bool motion,
                                                     ccl_private const float3 verts[3])
{
  return pixel_displacement_base_position(verts, u, v) +
         pixel_displacement_eval_object(kg, object, prim, u, v, time, motion, verts);
}

ccl_device_inline float2 pixel_displacement_offset_bary(const float u,
                                                        const float v,
                                                        const float du,
                                                        const float dv)
{
  float2 result = make_float2(u + du, v + dv);
  if (result.x < 0.0f || result.y < 0.0f || result.x + result.y > 1.0f) {
    result = make_float2(u - du, v - dv);
  }
  result.x = clamp(result.x, 0.0f, 1.0f);
  result.y = clamp(result.y, 0.0f, 1.0f - result.x);
  return result;
}

ccl_device_inline bool pixel_displacement_bary_inside(const float2 bary, const float eps)
{
  return bary.x >= -eps && bary.y >= -eps && bary.x + bary.y <= 1.0f + eps;
}

ccl_device_inline bool pixel_displacement_sample_tangent(KernelGlobals kg,
                                                         const int object,
                                                         const int prim,
                                                         const float u,
                                                         const float v,
                                                         const float time,
                                                         const bool motion,
                                                         ccl_private const float3 verts[3],
                                                         const float3 P,
                                                         const float2 direction,
                                                         const float eps,
                                                         ccl_private float3 *dP)
{
  const float2 bary = make_float2(u, v);
  const float2 delta = direction * eps;
  const float2 bary_plus = bary + delta;
  const float2 bary_minus = bary - delta;
  const bool plus_inside = pixel_displacement_bary_inside(bary_plus, 0.0f);
  const bool minus_inside = pixel_displacement_bary_inside(bary_minus, 0.0f);

  if (plus_inside && minus_inside) {
    const float3 P_plus = pixel_displacement_position(
        kg, object, prim, bary_plus.x, bary_plus.y, time, motion, verts);
    const float3 P_minus = pixel_displacement_position(
        kg, object, prim, bary_minus.x, bary_minus.y, time, motion, verts);
    *dP = (P_plus - P_minus) * (0.5f / eps);
    return true;
  }
  if (plus_inside) {
    const float3 P_plus = pixel_displacement_position(
        kg, object, prim, bary_plus.x, bary_plus.y, time, motion, verts);
    *dP = (P_plus - P) / eps;
    return true;
  }
  if (minus_inside) {
    const float3 P_minus = pixel_displacement_position(
        kg, object, prim, bary_minus.x, bary_minus.y, time, motion, verts);
    *dP = (P - P_minus) / eps;
    return true;
  }

  return false;
}

ccl_device_inline bool pixel_displacement_cached_geometry(KernelGlobals kg,
                                                          const int prim,
                                                          const float u,
                                                          const float v,
                                                          const bool motion,
                                                          ccl_private const float3 verts[3],
                                                          ccl_private float3 *P,
                                                          ccl_private float3 *dPdu,
                                                          ccl_private float3 *dPdv)
{
  int grid, offset;
  if (!pixel_displacement_cache_lookup(kg, prim, motion, &grid, &offset)) {
    return false;
  }

  float cu = clamp(u, 0.0f, 1.0f);
  float cv = clamp(v, 0.0f, 1.0f);
  const float sum = cu + cv;
  if (sum > 1.0f) {
    cu /= sum;
    cv /= sum;
  }

  const float fu = cu * float(grid);
  const float fv = cv * float(grid);
  int iu = min(int(floorf(fu)), grid - 1);
  int iv = min(int(floorf(fv)), grid - 1);
  if (iu + iv >= grid) {
    if (iv > 0) {
      iv = grid - 1 - iu;
    }
    else {
      iu = grid - 1;
    }
  }

  const float du = fu - float(iu);
  const float dv = fv - float(iv);
  const float inv_grid = 1.0f / float(grid);
  const float2 b00 = make_float2(float(iu), float(iv)) * inv_grid;
  const float2 b10 = make_float2(float(iu + 1), float(iv)) * inv_grid;
  const float2 b01 = make_float2(float(iu), float(iv + 1)) * inv_grid;
  const float3 P00 = pixel_displacement_base_position(verts, b00.x, b00.y) +
                     pixel_displacement_cache_sample(offset, grid, iu, iv);
  const float3 P10 = pixel_displacement_base_position(verts, b10.x, b10.y) +
                     pixel_displacement_cache_sample(offset, grid, iu + 1, iv);
  const float3 P01 = pixel_displacement_base_position(verts, b01.x, b01.y) +
                     pixel_displacement_cache_sample(offset, grid, iu, iv + 1);

  if (du + dv <= 1.0f || iu + iv + 2 > grid) {
    *P = P00 * (1.0f - du - dv) + P10 * du + P01 * dv;
    *dPdu = (P10 - P00) * float(grid);
    *dPdv = (P01 - P00) * float(grid);
    return true;
  }

  const float2 b11 = make_float2(float(iu + 1), float(iv + 1)) * inv_grid;
  const float3 P11 = pixel_displacement_base_position(verts, b11.x, b11.y) +
                     pixel_displacement_cache_sample(offset, grid, iu + 1, iv + 1);
  *P = P10 * (1.0f - dv) + P11 * (du + dv - 1.0f) + P01 * (1.0f - du);
  *dPdu = (P11 - P01) * float(grid);
  *dPdv = (P11 - P10) * float(grid);
  return true;
}

ccl_device_noinline void pixel_displacement_displaced_geometry(KernelGlobals kg,
                                                               const int object,
                                                               const int prim,
                                                               const float u,
                                                               const float v,
                                                               const float time,
                                                               const bool motion,
                                                               ccl_private const float3 verts[3],
                                                               ccl_private float3 *P_obj,
                                                               ccl_private float3 *Ng_obj,
                                                               ccl_private float3 *dPdu_obj,
                                                               ccl_private float3 *dPdv_obj)
{
  const uint object_flag = kernel_data_fetch(object_flag, object);
  const float3 face_Ng = pixel_displacement_face_normal(verts, object_flag);
  if (pixel_displacement_cached_geometry(kg, prim, u, v, motion, verts, P_obj, dPdu_obj, dPdv_obj))
  {
    *Ng_obj = safe_normalize(cross(*dPdu_obj, *dPdv_obj));
    if (dot(*Ng_obj, face_Ng) < 0.0f) {
      *Ng_obj = -*Ng_obj;
    }
    if (is_zero(*Ng_obj)) {
      *Ng_obj = face_Ng;
    }
    return;
  }

  *P_obj = pixel_displacement_position(kg, object, prim, u, v, time, motion, verts);
  const float eps = 1.0e-3f;

  const float2 dirs[4] = {make_float2(1.0f, 0.0f),
                          make_float2(0.0f, 1.0f),
                          make_float2(1.0f, -1.0f),
                          make_float2(1.0f, 1.0f)};
  float3 tangents[4];
  bool valid[4];
  for (int i = 0; i < 4; i++) {
    valid[i] = pixel_displacement_sample_tangent(
        kg, object, prim, u, v, time, motion, verts, *P_obj, dirs[i], eps, &tangents[i]);
  }

  *dPdu_obj = valid[0] ? tangents[0] : (verts[1] - verts[0]);
  *dPdv_obj = valid[1] ? tangents[1] : (verts[2] - verts[0]);

  float3 normal_sum = zero_float3();
  for (int i = 0; i < 4; i++) {
    if (!valid[i]) {
      continue;
    }
    for (int j = i + 1; j < 4; j++) {
      if (!valid[j]) {
        continue;
      }

      const float det = dirs[i].x * dirs[j].y - dirs[i].y * dirs[j].x;
      if (fabsf(det) <= 1.0e-7f) {
        continue;
      }

      float3 n = cross(tangents[i], tangents[j]);
      if (det < 0.0f) {
        n = -n;
      }
      if (dot(n, face_Ng) < 0.0f) {
        n = -n;
      }
      if (!is_zero(n)) {
        normal_sum += safe_normalize(n);
      }
    }
  }

  *Ng_obj = safe_normalize(normal_sum);
  if (is_zero(*Ng_obj)) {
    *Ng_obj = safe_normalize(cross(*dPdu_obj, *dPdv_obj));
    if (dot(*Ng_obj, face_Ng) < 0.0f) {
      *Ng_obj = -*Ng_obj;
    }
  }
  if (is_zero(*Ng_obj)) {
    *Ng_obj = face_Ng;
  }
}

ccl_device_noinline void pixel_displacement_shader_setup(KernelGlobals kg,
                                                         ccl_private ShaderData *sd,
                                                         const float time,
                                                         const bool motion,
                                                         ccl_private const float3 verts[3])
{
  if (!pixel_displacement_active(kg, sd->prim)) {
    return;
  }

  float3 P_obj;
  float3 Ng_obj;
  float3 dPdu_obj;
  float3 dPdv_obj;
  pixel_displacement_displaced_geometry(kg,
                                        sd->object,
                                        sd->prim,
                                        sd->u,
                                        sd->v,
                                        time,
                                        motion,
                                        verts,
                                        &P_obj,
                                        &Ng_obj,
                                        &dPdu_obj,
                                        &dPdv_obj);

  sd->P = P_obj;
  if (!(sd->object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    object_position_transform(kg, sd, &sd->P);
  }

  sd->Ng = Ng_obj;
  sd->N = Ng_obj;

#  ifdef __DPDU__
  sd->dPdu = dPdu_obj;
  sd->dPdv = dPdv_obj;
#  endif
}

#  ifdef __KERNEL_METAL_PIXEL_DISPLACEMENT__

struct PixelDisplacementHeightSample {
  float plane_distance;
  float height;
  float residual;
  float2 bary;
};

ccl_device_inline bool pixel_displacement_clip_bounds_axis(const float origin,
                                                           const float direction,
                                                           const float lower,
                                                           const float upper,
                                                           ccl_private float *t0,
                                                           ccl_private float *t1)
{
  if (fabsf(direction) < 1.0e-8f) {
    return origin >= lower && origin <= upper;
  }

  float ta = (lower - origin) / direction;
  float tb = (upper - origin) / direction;
  if (ta > tb) {
    const float tmp = ta;
    ta = tb;
    tb = tmp;
  }

  *t0 = max(*t0, ta);
  *t1 = min(*t1, tb);
  return *t1 >= *t0;
}

ccl_device_inline bool pixel_displacement_clip_bounds(const float3 bounds_min,
                                                      const float3 bounds_max,
                                                      const float3 ray_P,
                                                      const float3 ray_D,
                                                      ccl_private float *t0,
                                                      ccl_private float *t1)
{
  return pixel_displacement_clip_bounds_axis(
             ray_P.x, ray_D.x, bounds_min.x, bounds_max.x, t0, t1) &&
         pixel_displacement_clip_bounds_axis(
             ray_P.y, ray_D.y, bounds_min.y, bounds_max.y, t0, t1) &&
         pixel_displacement_clip_bounds_axis(ray_P.z, ray_D.z, bounds_min.z, bounds_max.z, t0, t1);
}

ccl_device_noinline bool pixel_displacement_height_sample(
    KernelGlobals kg,
    ccl_private const float3 verts[3],
    const int object,
    const int prim,
    const float time,
    const bool motion,
    const float3 ray_P,
    const float3 ray_D,
    const float3 Ng,
    const float origin_plane,
    const float dir_plane,
    const float t,
    const float bary_eps,
    ccl_private PixelDisplacementHeightSample *r_sample)
{
  const float plane_distance = origin_plane + t * dir_plane;
  const float3 ray_point = ray_P + ray_D * t;
  const float3 projected = ray_point - plane_distance * Ng;
  float2 bary = pixel_displacement_bary_from_point(verts, projected);

  if (!pixel_displacement_bary_inside(bary, bary_eps)) {
    return false;
  }

  bary.x = clamp(bary.x, 0.0f, 1.0f);
  bary.y = clamp(bary.y, 0.0f, 1.0f - bary.x);

  const float3 D = pixel_displacement_eval_object(
      kg, object, prim, bary.x, bary.y, time, motion, verts);
  const float height = dot(D, Ng);

  r_sample->plane_distance = plane_distance;
  r_sample->height = height;
  r_sample->residual = plane_distance - height;
  r_sample->bary = bary;
  return true;
}

ccl_device_inline float2 pixel_displacement_clamp_bary(float2 bary)
{
  bary.x = max(bary.x, 0.0f);
  bary.y = max(bary.y, 0.0f);
  const float sum = bary.x + bary.y;
  if (sum > 1.0f) {
    bary /= sum;
  }
  return bary;
}

ccl_device_inline bool pixel_displacement_intersect_micro_triangle(const float3 ray_P,
                                                                   const float3 ray_D,
                                                                   const float ray_tmin,
                                                                   const float ray_tmax,
                                                                   const float3 P0,
                                                                   const float3 P1,
                                                                   const float3 P2,
                                                                   const float2 b0,
                                                                   const float2 b1,
                                                                   const float2 b2,
                                                                   ccl_private float *r_t,
                                                                   ccl_private float2 *r_bary)
{
  const float3 e1 = P1 - P0;
  const float3 e2 = P2 - P0;
  const float3 h = cross(ray_D, e2);
  const float a = dot(e1, h);
  if (fabsf(a) < 1.0e-7f) {
    return false;
  }

  const float inv_a = 1.0f / a;
  const float3 s = ray_P - P0;
  const float u = inv_a * dot(s, h);
  if (u < -5.0e-4f || u > 1.0f + 5.0e-4f) {
    return false;
  }

  const float3 q = cross(s, e1);
  const float v = inv_a * dot(ray_D, q);
  if (v < -5.0e-4f || u + v > 1.0f + 5.0e-4f) {
    return false;
  }

  const float t = inv_a * dot(e2, q);
  if (!(t > ray_tmin && t < ray_tmax)) {
    return false;
  }

  const float w = 1.0f - u - v;
  const float2 bary = b0 * w + b1 * u + b2 * v;
  if (!pixel_displacement_bary_inside(bary, 5.0e-4f)) {
    return false;
  }

  *r_t = t;
  *r_bary = pixel_displacement_clamp_bary(bary);
  return true;
}

ccl_device_inline bool pixel_displacement_intersect_cached_cell(KernelGlobals kg,
                                                                const int offset,
                                                                const int grid,
                                                                const int iu,
                                                                const int iv,
                                                                const bool upper,
                                                                ccl_private const float3 verts[3],
                                                                const float3 ray_P,
                                                                const float3 ray_D,
                                                                const float ray_tmin,
                                                                ccl_private float *r_best_t,
                                                                ccl_private float2 *r_best_bary)
{
  if (iu < 0 || iv < 0 || iu + iv >= grid) {
    return false;
  }

  const float inv_grid = 1.0f / float(grid);
  const float2 b10 = make_float2(float(iu + 1), float(iv)) * inv_grid;
  const float2 b01 = make_float2(float(iu), float(iv + 1)) * inv_grid;
  const float3 P10 = pixel_displacement_base_position(verts, b10.x, b10.y) +
                     pixel_displacement_cache_sample(offset, grid, iu + 1, iv);
  const float3 P01 = pixel_displacement_base_position(verts, b01.x, b01.y) +
                     pixel_displacement_cache_sample(offset, grid, iu, iv + 1);

  float t;
  float2 bary;
  bool hit;
  if (upper && iu + iv + 2 <= grid) {
    const float2 b11 = make_float2(float(iu + 1), float(iv + 1)) * inv_grid;
    const float3 P11 = pixel_displacement_base_position(verts, b11.x, b11.y) +
                       pixel_displacement_cache_sample(offset, grid, iu + 1, iv + 1);
    hit = pixel_displacement_intersect_micro_triangle(
        ray_P, ray_D, ray_tmin, *r_best_t, P10, P11, P01, b10, b11, b01, &t, &bary);
  }
  else {
    const float2 b00 = make_float2(float(iu), float(iv)) * inv_grid;
    const float3 P00 = pixel_displacement_base_position(verts, b00.x, b00.y) +
                       pixel_displacement_cache_sample(offset, grid, iu, iv);
    hit = pixel_displacement_intersect_micro_triangle(
        ray_P, ray_D, ray_tmin, *r_best_t, P00, P10, P01, b00, b10, b01, &t, &bary);
  }
  if (hit) {
    *r_best_t = t;
    *r_best_bary = bary;
  }
  return hit;
}

ccl_device_inline float pixel_displacement_next_grid_crossing(const float x,
                                                              const float dx,
                                                              const float s)
{
  if (fabsf(dx) < 1.0e-8f) {
    return 2.0f;
  }

  const float value = x + dx * s;
  const float boundary = (dx > 0.0f) ? floorf(value + 1.0e-5f) + 1.0f :
                                       ceilf(value - 1.0e-5f) - 1.0f;
  return s + (boundary - value) / dx;
}

/* Intersect the cached piecewise-linear micromesh without height marching. The projected ray is
 * linear in barycentric space, so grid-line crossings give the complete ordered set of cells for
 * normal displacement. Tangential vector displacement falls back to the general solver below if
 * this fast path does not find a hit. */
ccl_device_noinline bool pixel_displacement_intersect_cached_micro_mesh(KernelGlobals kg,
                                                                        const int prim,
                                                                        const bool motion,
                                                                        ccl_private const float3
                                                                            verts[3],
                                                                        const float3 ray_P,
                                                                        const float3 ray_D,
                                                                        const float ray_tmin,
                                                                        const float ray_tmax,
                                                                        const float2 bary_t0,
                                                                        const float2 bary_t1,
                                                                        ccl_private float *r_t,
                                                                        ccl_private float2 *r_bary)
{
  int grid, offset;
  if (!pixel_displacement_cache_lookup(kg, prim, motion, &grid, &offset)) {
    return false;
  }

  /* The caller may provide a conservatively padded barycentric segment. Clamp its parameter
   * range to the actual projected triangle before running the grid DDA. Clamping both endpoints
   * independently bends grazing rays in barycentric space and causes missed microtriangles. */
  float slo = 0.0f;
  float shi = 1.0f;
  constexpr float bary_eps = 1.0e-6f;
  if (!pixel_displacement_clip_greater_equal_zero(
          bary_t0.x + bary_eps, bary_t1.x + bary_eps, &slo, &shi) ||
      !pixel_displacement_clip_greater_equal_zero(
          bary_t0.y + bary_eps, bary_t1.y + bary_eps, &slo, &shi) ||
      !pixel_displacement_clip_greater_equal_zero(1.0f - bary_t0.x - bary_t0.y + bary_eps,
                                                  1.0f - bary_t1.x - bary_t1.y + bary_eps,
                                                  &slo,
                                                  &shi))
  {
    return false;
  }

  const float2 b0 = pixel_displacement_clamp_bary(mix(bary_t0, bary_t1, slo));
  const float2 b1 = pixel_displacement_clamp_bary(mix(bary_t0, bary_t1, shi));
  const float gu0 = b0.x * float(grid);
  const float gv0 = b0.y * float(grid);
  const float gdu = (b1.x - b0.x) * float(grid);
  const float gdv = (b1.y - b0.y) * float(grid);
  const float gsum0 = gu0 + gv0;
  const float gdsum = gdu + gdv;

  bool hit = false;
  float best_t = ray_tmax;
  float2 best_bary = zero_float2();
  float s = 0.0f;

  for (int iteration = 0; iteration < 388 && s < 1.0f + 1.0e-6f; iteration++) {
    float next_s = 1.0f;
    next_s = min(next_s, pixel_displacement_next_grid_crossing(gu0, gdu, s));
    next_s = min(next_s, pixel_displacement_next_grid_crossing(gv0, gdv, s));
    next_s = min(next_s, pixel_displacement_next_grid_crossing(gsum0, gdsum, s));
    next_s = clamp(next_s, s + 1.0e-6f, 1.0f);

    const float sample_s = 0.5f * (s + next_s);
    const float gu = gu0 + gdu * sample_s;
    const float gv = gv0 + gdv * sample_s;
    const int iu = clamp(int(floorf(gu)), 0, grid - 1);
    const int iv = clamp(int(floorf(gv)), 0, grid - 1);
    const bool upper = (gu - floorf(gu)) + (gv - floorf(gv)) > 1.0f;

    /* The midpoint lies strictly inside the current DDA interval, so it uniquely identifies the
     * crossed cell. Scalar cached displacement cannot move a microtriangle into a neighboring
     * projected cell; vector displacement is kept on the general BVH2 path. */
    hit |= pixel_displacement_intersect_cached_cell(
        kg, offset, grid, iu, iv, upper, verts, ray_P, ray_D, ray_tmin, &best_t, &best_bary);

    if (hit) {
      *r_t = best_t;
      *r_bary = best_bary;
      return true;
    }

    if (next_s >= 1.0f) {
      break;
    }
    s = next_s;
  }

  return false;
}

/* Lightweight MetalRT entry point. Unlike the general fallback solver this only links cache
 * lookup, barycentric clipping and micromesh traversal into the intersection function. */
ccl_device_noinline bool pixel_displacement_intersect_cached_surface(KernelGlobals kg,
                                                                     const int object,
                                                                     const int prim,
                                                                     const bool motion,
                                                                     ccl_private const float3
                                                                         verts[3],
                                                                     const float3 ray_P,
                                                                     const float3 ray_D,
                                                                     const float ray_tmin,
                                                                     const float ray_tmax,
                                                                     ccl_private float *r_u,
                                                                     ccl_private float *r_v,
                                                                     ccl_private float *r_t)
{
  int cache_grid, cache_offset;
  if (!pixel_displacement_cache_lookup(kg, prim, motion, &cache_grid, &cache_offset)) {
    return false;
  }
  (void)cache_grid;
  (void)cache_offset;

  const uint object_flag = kernel_data_fetch(object_flag, object);
  const float3 Ng = pixel_displacement_face_normal(verts, object_flag);
  const float max_distance = kernel_data.integrator.pixel_displacement_max_distance;
  const float origin_plane = dot(ray_P - verts[0], Ng);
  const float dir_plane = dot(ray_D, Ng);
  float t0 = ray_tmin;
  float t1 = ray_tmax;

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

  const float3 projected_origin = ray_P - Ng * origin_plane;
  const float3 projected_dir = ray_D - Ng * dir_plane;
  const float2 bary_t0 = pixel_displacement_bary_from_point(verts,
                                                            projected_origin + projected_dir * t0);
  const float2 bary_t1 = pixel_displacement_bary_from_point(verts,
                                                            projected_origin + projected_dir * t1);

  const float3 e0 = verts[1] - verts[0];
  const float3 e1 = verts[2] - verts[0];
  const float3 e2 = verts[2] - verts[1];
  const float min_edge_len = sqrtf(max(min(min(dot(e0, e0), dot(e1, e1)), dot(e2, e2)), 1.0e-8f));
  const float bary_eps = clamp(max_distance / min_edge_len + 5.0e-4f, 5.0e-4f, 0.25f);
  float slo = 0.0f;
  float shi = 1.0f;
  if (!pixel_displacement_clip_greater_equal_zero(
          bary_t0.x + bary_eps, bary_t1.x + bary_eps, &slo, &shi) ||
      !pixel_displacement_clip_greater_equal_zero(
          bary_t0.y + bary_eps, bary_t1.y + bary_eps, &slo, &shi) ||
      !pixel_displacement_clip_greater_equal_zero(1.0f - bary_t0.x - bary_t0.y + bary_eps,
                                                  1.0f - bary_t1.x - bary_t1.y + bary_eps,
                                                  &slo,
                                                  &shi))
  {
    return false;
  }

  float2 hit_bary;
  if (!pixel_displacement_intersect_cached_micro_mesh(kg,
                                                      prim,
                                                      motion,
                                                      verts,
                                                      ray_P,
                                                      ray_D,
                                                      ray_tmin,
                                                      ray_tmax,
                                                      mix(bary_t0, bary_t1, slo),
                                                      mix(bary_t0, bary_t1, shi),
                                                      r_t,
                                                      &hit_bary))
  {
    return false;
  }
  *r_u = hit_bary.x;
  *r_v = hit_bary.y;
  return true;
}

ccl_device_noinline bool pixel_displacement_intersect_micro_patch(KernelGlobals kg,
                                                                  const int object,
                                                                  const int prim,
                                                                  const float time,
                                                                  const bool motion,
                                                                  ccl_private const float3
                                                                      verts[3],
                                                                  const float3 ray_P,
                                                                  const float3 ray_D,
                                                                  const float ray_tmin,
                                                                  const float ray_tmax,
                                                                  const int steps,
                                                                  ccl_private float *r_t,
                                                                  ccl_private float2 *r_bary)
{
  bool hit = false;
  float best_t = ray_tmax;
  float2 best_bary = make_float2(0.0f, 0.0f);
  const int grid = clamp(int(ceilf(sqrtf(float(max(steps, 1)))) + 2), 6, 14);
  const float inv_grid = 1.0f / float(grid);

  for (int i = 0; i < 14; i++) {
    if (i >= grid) {
      break;
    }
    for (int j = 0; j < 14; j++) {
      if (i + j >= grid) {
        break;
      }

      const float2 b00 = make_float2(float(i), float(j)) * inv_grid;
      const float2 b10 = make_float2(float(i + 1), float(j)) * inv_grid;
      const float2 b01 = make_float2(float(i), float(j + 1)) * inv_grid;
      const float3 P00 = pixel_displacement_position(
          kg, object, prim, b00.x, b00.y, time, motion, verts);
      const float3 P10 = pixel_displacement_position(
          kg, object, prim, b10.x, b10.y, time, motion, verts);
      const float3 P01 = pixel_displacement_position(
          kg, object, prim, b01.x, b01.y, time, motion, verts);

      float t;
      float2 bary;
      if (pixel_displacement_intersect_micro_triangle(
              ray_P, ray_D, ray_tmin, best_t, P00, P10, P01, b00, b10, b01, &t, &bary))
      {
        hit = true;
        best_t = t;
        best_bary = bary;
      }

      if (i + j + 2 > grid) {
        continue;
      }

      const float2 b11 = make_float2(float(i + 1), float(j + 1)) * inv_grid;
      const float3 P11 = pixel_displacement_position(
          kg, object, prim, b11.x, b11.y, time, motion, verts);
      if (pixel_displacement_intersect_micro_triangle(
              ray_P, ray_D, ray_tmin, best_t, P10, P11, P01, b10, b11, b01, &t, &bary))
      {
        hit = true;
        best_t = t;
        best_bary = bary;
      }
    }
  }

  if (!hit) {
    return false;
  }

  *r_t = best_t;
  *r_bary = best_bary;
  return true;
}

ccl_device_noinline bool pixel_displacement_solve_local_ray(KernelGlobals kg,
                                                            const int object,
                                                            const int prim,
                                                            const float3 ray_P,
                                                            const float3 ray_D,
                                                            const float ray_tmin,
                                                            const float ray_tmax,
                                                            const float seed_t,
                                                            const float time,
                                                            const bool motion,
                                                            ccl_private const float3 verts[3],
                                                            ccl_private float *r_t,
                                                            ccl_private float *r_u,
                                                            ccl_private float *r_v)
{
  const uint object_flag = kernel_data_fetch(object_flag, object);
  const float3 Ng = pixel_displacement_face_normal(verts, object_flag);
  const float max_distance = kernel_data.integrator.pixel_displacement_max_distance;
  /* Shadow rays start with tmin == 0 and can otherwise re-hit a neighboring triangle edge. */
  const float start_epsilon = max(1.0e-5f, max_distance * 1.0e-4f / max(len(ray_D), 1.0e-6f));
  const float hit_tmin = (ray_tmin <= 1.0e-7f) ? start_epsilon : ray_tmin;

  const float origin_plane = dot(ray_P - verts[0], Ng);
  const float dir_plane = dot(ray_D, Ng);
  const float abs_dir_plane = fabsf(dir_plane);

  const float3 bounds_pad = make_float3(max_distance);
  const float3 bounds_min = min(min(verts[0], verts[1]), verts[2]) - bounds_pad;
  const float3 bounds_max = max(max(verts[0], verts[1]), verts[2]) + bounds_pad;
  float bounds_t0 = hit_tmin;
  float bounds_t1 = ray_tmax;
  if (!pixel_displacement_clip_bounds(
          bounds_min, bounds_max, ray_P, ray_D, &bounds_t0, &bounds_t1))
  {
    return false;
  }

  float t0 = hit_tmin;
  float t1 = ray_tmax;
  if (abs_dir_plane > 1.0e-8f) {
    const float ta = (-max_distance - origin_plane) / dir_plane;
    const float tb = (max_distance - origin_plane) / dir_plane;
    t0 = max(t0, min(ta, tb));
    t1 = min(t1, max(ta, tb));
  }
  else if (origin_plane < -max_distance || origin_plane > max_distance) {
    return false;
  }

  t0 = max(t0, bounds_t0);
  t1 = min(t1, bounds_t1);

  if (seed_t > hit_tmin && abs_dir_plane > 1.0e-6f) {
    const float local_pad = 2.0f * max_distance / abs_dir_plane;
    t0 = max(t0, max(hit_tmin, seed_t - local_pad));
    t1 = min(t1, seed_t + local_pad);
  }

  if (t1 <= t0) {
    return false;
  }

  const float3 projected_origin = ray_P - Ng * origin_plane;
  const float3 projected_dir = ray_D - Ng * dir_plane;
  const float2 bary_at_t0 = pixel_displacement_bary_from_point(
      verts, projected_origin + projected_dir * t0);
  const float2 bary_at_t1 = pixel_displacement_bary_from_point(
      verts, projected_origin + projected_dir * t1);

  const float3 e0 = verts[1] - verts[0];
  const float3 e1 = verts[2] - verts[0];
  const float3 e2 = verts[2] - verts[1];
  const float min_edge_len = sqrtf(max(min(min(dot(e0, e0), dot(e1, e1)), dot(e2, e2)), 1.0e-8f));
  const float bary_final_eps = 5.0e-4f;
  const float bary_eps = clamp(
      max_distance / min_edge_len + bary_final_eps, bary_final_eps, 0.25f);

  float slo = 0.0f;
  float shi = 1.0f;
  if (!pixel_displacement_clip_greater_equal_zero(
          bary_at_t0.x + bary_eps, bary_at_t1.x + bary_eps, &slo, &shi) ||
      !pixel_displacement_clip_greater_equal_zero(
          bary_at_t0.y + bary_eps, bary_at_t1.y + bary_eps, &slo, &shi) ||
      !pixel_displacement_clip_greater_equal_zero(1.0f - bary_at_t0.x - bary_at_t0.y + bary_eps,
                                                  1.0f - bary_at_t1.x - bary_at_t1.y + bary_eps,
                                                  &slo,
                                                  &shi))
  {
    return false;
  }

  const float slab_t0 = t0;
  const float slab_t1 = t1;
  t0 = mix(slab_t0, slab_t1, slo);
  t1 = mix(slab_t0, slab_t1, shi);

  if (t1 <= t0) {
    return false;
  }

  const float2 clipped_bary_t0 = mix(bary_at_t0, bary_at_t1, slo);
  const float2 clipped_bary_t1 = mix(bary_at_t0, bary_at_t1, shi);
  float cached_t;
  float2 cached_bary;
  if (pixel_displacement_intersect_cached_micro_mesh(kg,
                                                     prim,
                                                     motion,
                                                     verts,
                                                     ray_P,
                                                     ray_D,
                                                     hit_tmin,
                                                     ray_tmax,
                                                     clipped_bary_t0,
                                                     clipped_bary_t1,
                                                     &cached_t,
                                                     &cached_bary))
  {
    *r_t = cached_t;
    *r_u = cached_bary.x;
    *r_v = cached_bary.y;
    return true;
  }

  int cache_grid, cache_offset;
  const bool use_cache = pixel_displacement_cache_lookup(
      kg, prim, motion, &cache_grid, &cache_offset);
  (void)cache_offset;

  int steps = clamp(kernel_data.integrator.pixel_displacement_steps, 8, 128);
  if (use_cache) {
    steps = clamp(cache_grid * 2, 8, 32);
    if (abs_dir_plane < 0.05f) {
      steps = min(steps * 2, 48);
    }
    else if (abs_dir_plane < 0.2f) {
      steps = min((steps * 3) / 2, 40);
    }
  }
  else {
    if (abs_dir_plane < 0.05f) {
      steps = min(steps * 4, 128);
    }
    else if (abs_dir_plane < 0.2f) {
      steps = min(steps * 2, 128);
    }
  }
  const float dt = (t1 - t0) / float(steps);

  bool have_prev = false;
  float prev_t = t0;
  float prev_f = 0.0f;

  for (int i = 0; i <= 128; i++) {
    if (i > steps) {
      break;
    }

    const float t = (i == steps) ? t1 : (t0 + dt * float(i));
    PixelDisplacementHeightSample sample;
    if (!pixel_displacement_height_sample(kg,
                                          verts,
                                          object,
                                          prim,
                                          time,
                                          motion,
                                          ray_P,
                                          ray_D,
                                          Ng,
                                          origin_plane,
                                          dir_plane,
                                          t,
                                          bary_eps,
                                          &sample))
    {
      continue;
    }

    const float f = sample.residual;
    const float2 bary = sample.bary;
    const bool bracket = have_prev &&
                         ((prev_f <= 0.0f && f >= 0.0f) || (prev_f >= 0.0f && f <= 0.0f));
    if (fabsf(f) <= 2.0e-4f || bracket) {
      float lo = bracket ? prev_t : max(t - dt, t0);
      float hi = t;
      float flo = bracket ? prev_f : f;
      float fhi = f;
      float2 root_bary = bary;

      const int refine_steps = use_cache ? 5 : 8;
      for (int refine = 0; refine < 8; refine++) {
        if (refine >= refine_steps) {
          break;
        }

        const float denom = fhi - flo;
        const float secant_t = (fabsf(denom) > 1.0e-7f) ? hi - fhi * (hi - lo) / denom :
                                                          0.5f * (lo + hi);
        const float tm = clamp(secant_t, mix(lo, hi, 0.2f), mix(lo, hi, 0.8f));

        PixelDisplacementHeightSample sample_m;
        if (!pixel_displacement_height_sample(kg,
                                              verts,
                                              object,
                                              prim,
                                              time,
                                              motion,
                                              ray_P,
                                              ray_D,
                                              Ng,
                                              origin_plane,
                                              dir_plane,
                                              tm,
                                              bary_eps,
                                              &sample_m))
        {
          break;
        }

        const float fm = sample_m.residual;
        const float2 bary_m = sample_m.bary;
        root_bary = bary_m;
        if ((flo <= 0.0f && fm >= 0.0f) || (flo >= 0.0f && fm <= 0.0f)) {
          hi = tm;
          fhi = fm;
        }
        else {
          lo = tm;
          flo = fm;
        }
      }

      const float root_t = 0.5f * (lo + hi);
      const float plane_distance = origin_plane + root_t * dir_plane;
      const float3 root_point = ray_P + ray_D * root_t;
      const float3 projected = root_point - plane_distance * Ng;
      root_bary = pixel_displacement_bary_from_point(verts, projected);
      if (!pixel_displacement_bary_inside(root_bary, bary_final_eps)) {
        have_prev = true;
        prev_t = t;
        prev_f = f;
        continue;
      }
      root_bary.x = clamp(root_bary.x, 0.0f, 1.0f);
      root_bary.y = clamp(root_bary.y, 0.0f, 1.0f - root_bary.x);

      if (root_t > hit_tmin && root_t < ray_tmax) {
        float refined_t = root_t;
        float2 refined_bary = root_bary;
        const float residual_limit = max(7.5e-4f, max_distance * 0.05f);
        bool use_refined = false;

        const int newton_steps = use_cache ? 1 : 3;
        for (int newton = 0; newton < 3; newton++) {
          if (newton >= newton_steps) {
            break;
          }

          float3 surface_P;
          float3 surface_Ng;
          float3 surface_dPdu;
          float3 surface_dPdv;
          pixel_displacement_displaced_geometry(kg,
                                                object,
                                                prim,
                                                refined_bary.x,
                                                refined_bary.y,
                                                time,
                                                motion,
                                                verts,
                                                &surface_P,
                                                &surface_Ng,
                                                &surface_dPdu,
                                                &surface_dPdv);

          (void)surface_Ng;
          const float3 residual = ray_P + ray_D * refined_t - surface_P;
          if (dot(residual, residual) <= residual_limit * residual_limit) {
            use_refined = true;
            break;
          }

          const float3 c0 = ray_D;
          const float3 c1 = -surface_dPdu;
          const float3 c2 = -surface_dPdv;
          const float det = dot(c0, cross(c1, c2));
          if (fabsf(det) <= 1.0e-8f) {
            break;
          }

          const float3 rhs = -residual;
          const float delta_t = dot(rhs, cross(c1, c2)) / det;
          const float delta_u = dot(c0, cross(rhs, c2)) / det;
          const float delta_v = dot(c0, cross(c1, rhs)) / det;
          const float max_t_step = max((hi - lo) * 0.75f, 0.02f + max_distance);
          const float2 next_bary = refined_bary + clamp(make_float2(delta_u, delta_v),
                                                        make_float2(-0.25f, -0.25f),
                                                        make_float2(0.25f, 0.25f));
          if (!pixel_displacement_bary_inside(next_bary, bary_final_eps)) {
            break;
          }

          refined_t = clamp(
              refined_t + clamp(delta_t, -max_t_step, max_t_step), hit_tmin, ray_tmax);
          refined_bary = next_bary;
          refined_bary.x = clamp(refined_bary.x, 0.0f, 1.0f);
          refined_bary.y = clamp(refined_bary.y, 0.0f, 1.0f - refined_bary.x);
        }

        if (!use_refined) {
          float3 surface_P;
          float3 surface_Ng;
          float3 surface_dPdu;
          float3 surface_dPdv;
          pixel_displacement_displaced_geometry(kg,
                                                object,
                                                prim,
                                                refined_bary.x,
                                                refined_bary.y,
                                                time,
                                                motion,
                                                verts,
                                                &surface_P,
                                                &surface_Ng,
                                                &surface_dPdu,
                                                &surface_dPdv);

          (void)surface_Ng;
          (void)surface_dPdu;
          (void)surface_dPdv;
          const float3 residual = ray_P + ray_D * refined_t - surface_P;
          use_refined = dot(residual, residual) <= residual_limit * residual_limit;
        }

        if (use_refined && refined_t > hit_tmin && refined_t < ray_tmax) {
          *r_t = refined_t;
          *r_u = refined_bary.x;
          *r_v = refined_bary.y;
          return true;
        }
      }
    }

    have_prev = true;
    prev_t = t;
    prev_f = f;
  }

  if (abs_dir_plane < 0.35f) {
    float micro_t;
    float2 micro_bary;
    if (pixel_displacement_intersect_micro_patch(kg,
                                                 object,
                                                 prim,
                                                 time,
                                                 motion,
                                                 verts,
                                                 ray_P,
                                                 ray_D,
                                                 hit_tmin,
                                                 ray_tmax,
                                                 steps,
                                                 &micro_t,
                                                 &micro_bary))
    {
      *r_t = micro_t;
      *r_u = micro_bary.x;
      *r_v = micro_bary.y;
      return true;
    }
  }

  return false;
}

ccl_device_inline bool pixel_displacement_solve_ray(KernelGlobals kg,
                                                    ccl_private ShaderData *sd,
                                                    const ccl_private Ray *ray,
                                                    const float time,
                                                    const bool motion,
                                                    ccl_private const float3 verts[3],
                                                    ccl_private float *r_t,
                                                    ccl_private float *r_u,
                                                    ccl_private float *r_v)
{
  float3 ray_P = ray->P;
  float3 ray_D = ray->D;
  if (!(sd->object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    object_inverse_position_transform(kg, sd, &ray_P);
    object_inverse_dir_transform(kg, sd, &ray_D);
  }

  return pixel_displacement_solve_local_ray(kg,
                                            sd->object,
                                            sd->prim,
                                            ray_P,
                                            ray_D,
                                            ray->tmin,
                                            ray->tmax,
                                            sd->ray_length,
                                            time,
                                            motion,
                                            verts,
                                            r_t,
                                            r_u,
                                            r_v);
}

ccl_device_inline bool pixel_displacement_intersect_displaced_surface(KernelGlobals kg,
                                                                      const float3 P,
                                                                      const float3 dir,
                                                                      const float tmin,
                                                                      const float tmax,
                                                                      const float time,
                                                                      const int object,
                                                                      const int prim,
                                                                      const bool motion,
                                                                      ccl_private const float3
                                                                          verts[3],
                                                                      ccl_private float *r_u,
                                                                      ccl_private float *r_v,
                                                                      ccl_private float *r_t)
{
  if (!pixel_displacement_active(kg, prim)) {
    return false;
  }

  return pixel_displacement_solve_local_ray(
      kg, object, prim, P, dir, tmin, tmax, 0.0f, time, motion, verts, r_t, r_u, r_v);
}

ccl_device_noinline void pixel_displacement_shader_setup_from_ray(KernelGlobals kg,
                                                                  ccl_private ShaderData *sd,
                                                                  const ccl_private Ray *ray,
                                                                  const float time,
                                                                  const bool motion,
                                                                  ccl_private const float3
                                                                      verts[3])
{
  if (!pixel_displacement_active(kg, sd->prim)) {
    return;
  }

  float hit_t = sd->ray_length;
  float hit_u = sd->u;
  float hit_v = sd->v;
  if (pixel_displacement_solve_ray(kg, sd, ray, time, motion, verts, &hit_t, &hit_u, &hit_v)) {
    sd->ray_length = hit_t;
    sd->u = hit_u;
    sd->v = hit_v;
  }

  pixel_displacement_shader_setup(kg, sd, time, motion, verts);
}

#  endif /* __KERNEL_METAL_PIXEL_DISPLACEMENT__ */

#endif

CCL_NAMESPACE_END

#endif /* KERNEL_GEOM_PIXEL_DISPLACEMENT_SHADER_H_ */
