/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/light/coherent_geometry.h"

CCL_NAMESPACE_BEGIN

/* Exterior convex sphere, one ideal reflection. Both endpoints must be outside.
 * The intersection of their visible caps contains one stationary branch.
 * Refraction, interior reflectors and focal caustics require a different branch
 * inventory and are deliberately not represented by this connector. */
ccl_device_inline bool coherent_sphere_reflect(const float3 source,
                                               const float3 receiver,
                                               const float3 receiver_normal,
                                               const float3 center,
                                               const float radius,
                                               ccl_private CoherentGeometryInterface *frame,
                                               ccl_private CoherentGeometryPath *path)
{
  const float3 s = source - center, r = receiver - center;
  const float a = len(s), b = len(r);
  if (!(radius > 0.0f && a > radius && b > radius)) return false;
  const float3 e0 = s / a;
  const float3 rdir = r / b;
  const float cosine = clamp(dot(e0, rdir), -1.0f, 1.0f);
  const float3 perpendicular = rdir - e0 * cosine;
  const float sine = len(perpendicular);
  const float theta = atan2f(sine, cosine);
  float3 e1, unused;
  if (sine > 1.0e-7f) e1 = perpendicular / sine;
  else make_orthonormals(e0, &e1, &unused);
  float lower = max(0.0f, theta - acosf(radius / b));
  float upper = min(theta, acosf(radius / a));
  if (lower > upper) return false;
  float phi = (lower + upper) * 0.5f;
  /* Bisection does not depend on a manifold seed. Angular error is tangential,
   * hence contributes only second order to the stationary optical length. */
  for (int iteration = 0; iteration < 30; iteration++) {
    phi = (lower + upper) * 0.5f;
    const float3 n = e0 * cosf(phi) + e1 * sinf(phi);
    const float3 t = -e0 * sinf(phi) + e1 * cosf(phi);
    const float3 x = radius * n;
    const float derivative = dot(normalize(x - s) + normalize(x - r), t);
    if (derivative > 0.0f) upper = phi;
    else lower = phi;
  }
  const float3 raw_n = e0 * cosf(phi) + e1 * sinf(phi);
  const float2 norm = coherent_geometry_distance_split(raw_n, zero_float3());
  float2 p[3], sp[3], rp[3];
  float3 relative_s, relative_r, normal;
  for (int axis = 0; axis < 3; axis++) {
    const float component = axis == 0 ? raw_n.x : axis == 1 ? raw_n.y : raw_n.z;
    const float hi = component / norm.x;
    const float lo = (coherent_geometry_fma(-hi, norm.x, component) - hi * norm.y) / norm.x;
    const float2 n = coherent_geometry_add_split(make_float2(hi, 0), make_float2(lo, 0));
    const float c = axis == 0 ? center.x : axis == 1 ? center.y : center.z;
    const float ss = axis == 0 ? source.x : axis == 1 ? source.y : source.z;
    const float rr = axis == 0 ? receiver.x : axis == 1 ? receiver.y : receiver.z;
    p[axis] = coherent_geometry_add_split(make_float2(c, 0),
                                          coherent_geometry_product_split(n, radius));
    sp[axis] = make_float2(ss, 0);
    rp[axis] = make_float2(rr, 0);
    const float2 ds = coherent_geometry_add_split(p[axis], make_float2(-ss, 0));
    const float2 dr = coherent_geometry_add_split(p[axis], make_float2(-rr, 0));
    if (axis == 0) { relative_s.x = ds.x + ds.y; relative_r.x = dr.x + dr.y; normal.x = n.x + n.y; }
    if (axis == 1) { relative_s.y = ds.x + ds.y; relative_r.y = dr.x + dr.y; normal.y = n.x + n.y; }
    if (axis == 2) { relative_s.z = ds.x + ds.y; relative_r.z = dr.x + dr.y; normal.z = n.x + n.y; }
  }
  const float2 ls = coherent_geometry_distance_split_affine(p, sp);
  const float2 lr = coherent_geometry_distance_split_affine(p, rp);
  const float ds = ls.x + ls.y, dr = lr.x + lr.y;
  if (!(ds > 1.0e-7f && dr > 1.0e-7f)) return false;
  const float3 vs = relative_s / ds, vr = relative_r / dr;
  /* Native points expose only their front surface; reject grazing/inside-side
   * roots rather than suppressing a route which the native BVH cannot see. */
  if (!(dot(vs, normal) < 0.0f && dot(vr, normal) < 0.0f)) return false;
  float3 u, v, ru, rv, su, sv;
  make_orthonormals(normal, &u, &v);
  make_orthonormals(receiver_normal, &ru, &rv);
  make_orthonormals(vs, &su, &sv);
  const float curvature = -dot(vs + vr, normal) / radius;
  const float us = dot(u, vs), vs_t = dot(v, vs), ur = dot(u, vr), vr_t = dot(v, vr);
  const float h00 = (1 - us * us) / ds + (1 - ur * ur) / dr + curvature;
  const float h11 = (1 - vs_t * vs_t) / ds + (1 - vr_t * vr_t) / dr + curvature;
  const float h01 = -us * vs_t / ds - ur * vr_t / dr;
  const float determinant = h00 * h11 - h01 * h01;
  if (!(h00 > 0 && h11 > 0 && determinant > 1.0e-12f * h00 * h11)) return false;
  float jacobian[4];
  for (int axis = 0; axis < 2; axis++) {
    const float3 basis = axis == 0 ? ru : rv;
    const float3 rhs = (basis - vr * dot(vr, basis)) / dr;
    const float q0 = (h11 * dot(u, rhs) - h01 * dot(v, rhs)) / determinant;
    const float q1 = (h00 * dot(v, rhs) - h01 * dot(u, rhs)) / determinant;
    const float3 motion = u * q0 + v * q1;
    const float3 direction_derivative = (motion - vs * dot(vs, motion)) / ds;
    jacobian[axis] = dot(su, direction_derivative);
    jacobian[2 + axis] = dot(sv, direction_derivative);
  }
  path->spreading = fabsf(jacobian[0] * jacobian[3] - jacobian[1] * jacobian[2]);
  if (!(path->spreading > 0)) return false;
  path->point[0] = make_float3(p[0].x + p[0].y, p[1].x + p[1].y, p[2].x + p[2].y);
  path->segment_length[0] = ds;
  path->segment_length[1] = dr;
  path->optical_length_split = coherent_geometry_add_split(ls, lr);
  path->optical_length = path->optical_length_split.x + path->optical_length_split.y;
  path->source_direction = vs;
  path->count = 1;
  frame->center = path->point[0];
  frame->tangent_u = u;
  frame->tangent_v = v;
  /* Exterior convex reflection has positive stationary Hessian: no relative
   * Maslov shift. Mirror material phase is still handled by the field model. */
  return true;
}

CCL_NAMESPACE_END
