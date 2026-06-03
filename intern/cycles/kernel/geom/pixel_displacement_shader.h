/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef KERNEL_GEOM_PIXEL_DISPLACEMENT_SHADER_H_
#define KERNEL_GEOM_PIXEL_DISPLACEMENT_SHADER_H_

CCL_NAMESPACE_BEGIN

#if defined(__KERNEL_METAL__) && \
    (defined(__KERNEL_METAL_PIXEL_DISPLACEMENT__) || \
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
  sd->N = pixel_displacement_smooth_normal(
      kg,
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

ccl_device_noinline float3 pixel_displacement_eval_object(KernelGlobals kg,
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

ccl_device_noinline void pixel_displacement_displaced_geometry(
    KernelGlobals kg,
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
  const float eps = 2.0e-3f;

  const float2 bu = pixel_displacement_offset_bary(u, v, eps, 0.0f);
  const float2 bv = pixel_displacement_offset_bary(u, v, 0.0f, eps);

  *P_obj = pixel_displacement_position(kg, object, prim, u, v, time, motion, verts);

  const float3 Pu = pixel_displacement_position(kg, object, prim, bu.x, bu.y, time, motion, verts);
  const float3 Pv = pixel_displacement_position(kg, object, prim, bv.x, bv.y, time, motion, verts);

  const float du_sign = (bu.x >= u) ? 1.0f : -1.0f;
  const float dv_sign = (bv.y >= v) ? 1.0f : -1.0f;
  *dPdu_obj = (Pu - *P_obj) * (du_sign / eps);
  *dPdv_obj = (Pv - *P_obj) * (dv_sign / eps);

  const uint object_flag = kernel_data_fetch(object_flag, object);
  if (object_negative_scale_applied(object_flag)) {
    *Ng_obj = safe_normalize(cross(*dPdv_obj, *dPdu_obj));
  }
  else {
    *Ng_obj = safe_normalize(cross(*dPdu_obj, *dPdv_obj));
  }
  if (is_zero(*Ng_obj)) {
    *Ng_obj = pixel_displacement_face_normal(verts, object_flag);
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

ccl_device_inline bool pixel_displacement_bary_inside(const float2 bary, const float eps)
{
  return bary.x >= -eps && bary.y >= -eps && bary.x + bary.y <= 1.0f + eps;
}

ccl_device_noinline bool pixel_displacement_height_residual(KernelGlobals kg,
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
                                                            ccl_private float *r_f,
                                                            ccl_private float2 *r_bary)
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
  *r_f = plane_distance - dot(D, Ng);
  *r_bary = bary;
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

  const float origin_plane = dot(ray_P - verts[0], Ng);
  const float dir_plane = dot(ray_D, Ng);
  if (fabsf(dir_plane) < 1.0e-8f) {
    return false;
  }

  float t0 = (max_distance - origin_plane) / dir_plane;
  float t1 = (-max_distance - origin_plane) / dir_plane;
  if (t0 > t1) {
    const float tmp = t0;
    t0 = t1;
    t1 = tmp;
  }

  t0 = max(t0, ray_tmin);
  t1 = min(t1, ray_tmax);

  if (seed_t > ray_tmin) {
    const float local_pad = 2.0f * max_distance / max(fabsf(dir_plane), 1.0e-6f);
    t0 = max(t0, max(ray_tmin, seed_t - local_pad));
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
  const float min_edge_len = sqrtf(
      max(min(min(dot(e0, e0), dot(e1, e1)), dot(e2, e2)), 1.0e-8f));
  const float bary_final_eps = 5.0e-4f;
  const float bary_eps = clamp(max_distance / min_edge_len + bary_final_eps, bary_final_eps, 0.25f);

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

  int steps = clamp(kernel_data.integrator.pixel_displacement_steps, 8, 128);
  if (fabsf(dir_plane) < 0.2f) {
    steps = min(steps * 2, 128);
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
    float f;
    float2 bary;
    if (!pixel_displacement_height_residual(kg,
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
                                            &f,
                                            &bary))
    {
      continue;
    }

    const bool bracket = have_prev && ((prev_f <= 0.0f && f >= 0.0f) ||
                                       (prev_f >= 0.0f && f <= 0.0f));
    if (fabsf(f) <= 2.0e-4f || bracket) {
      float lo = bracket ? prev_t : max(t - dt, t0);
      float hi = t;
      float flo = bracket ? prev_f : f;
      float fhi = f;
      float2 root_bary = bary;

      for (int refine = 0; refine < 8; refine++) {
        const float denom = fhi - flo;
        const float secant_t = (fabsf(denom) > 1.0e-7f) ? hi - fhi * (hi - lo) / denom :
                                                          0.5f * (lo + hi);
        const float tm = clamp(secant_t, mix(lo, hi, 0.2f), mix(lo, hi, 0.8f));

        float fm;
        float2 bary_m;
        if (!pixel_displacement_height_residual(kg,
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
                                                &fm,
                                                &bary_m))
        {
          break;
        }

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
      if (!pixel_displacement_bary_inside(root_bary, bary_eps)) {
        root_bary = bary;
      }
      root_bary.x = clamp(root_bary.x, 0.0f, 1.0f);
      root_bary.y = clamp(root_bary.y, 0.0f, 1.0f - root_bary.x);

      if (root_t <= ray_tmin || root_t >= ray_tmax) {
        return false;
      }

      float refined_t = root_t;
      float2 refined_bary = root_bary;
      const float residual_limit = max(7.5e-4f, max_distance * 0.05f);
      bool use_refined = false;

      for (int newton = 0; newton < 3; newton++) {
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
        const float2 next_bary = refined_bary +
                                  clamp(make_float2(delta_u, delta_v),
                                        make_float2(-0.25f, -0.25f),
                                        make_float2(0.25f, 0.25f));
        if (!pixel_displacement_bary_inside(next_bary, bary_final_eps)) {
          break;
        }

        refined_t = clamp(
            refined_t + clamp(delta_t, -max_t_step, max_t_step), ray_tmin, ray_tmax);
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

      if (!use_refined || refined_t <= ray_tmin || refined_t >= ray_tmax) {
        return false;
      }

      *r_t = refined_t;
      *r_u = refined_bary.x;
      *r_v = refined_bary.y;
      return true;
    }

    have_prev = true;
    prev_t = t;
    prev_f = f;
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

ccl_device_inline bool pixel_displacement_intersect_displaced_surface(
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
  if (!pixel_displacement_active(kg, prim)) {
    return false;
  }

  return pixel_displacement_solve_local_ray(kg,
                                            object,
                                            prim,
                                            P,
                                            dir,
                                            tmin,
                                            tmax,
                                            0.0f,
                                            time,
                                            motion,
                                            verts,
                                            r_t,
                                            r_u,
                                            r_v);
}

ccl_device_noinline void pixel_displacement_shader_setup_from_ray(KernelGlobals kg,
                                                                  ccl_private ShaderData *sd,
                                                                  const ccl_private Ray *ray,
                                                                  const float time,
                                                                  const bool motion,
                                                                  ccl_private const float3 verts[3])
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
