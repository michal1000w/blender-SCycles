/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/light/coherent_geometry.h"
CCL_NAMESPACE_BEGIN

/* Stationary TT connector used by native sphere path integration.
 * Exterior endpoints, a homogeneous lossless sphere, no internal reflections.
 * Count includes every isolated TT root, sorted by signed angular momentum.
 * A caustic/nonisolated branch invalidates the whole inventory, never just its
 * divergent contribution. Phase convention exp(+ik OPL): Maslov cycles=-m/4. */
enum CoherentSphereTTStatus {
  COHERENT_SPHERE_TT_OK,
  COHERENT_SPHERE_TT_EMPTY,
  COHERENT_SPHERE_TT_INVALID,
  COHERENT_SPHERE_TT_CAUSTIC,
  COHERENT_SPHERE_TT_AXIAL_RING,
  COHERENT_SPHERE_TT_GRAZING
};
struct CoherentSphereTTInventory {
  CoherentGeometryPath path[3];
  CoherentGeometryInterface frame[3][2];
  float angular_momentum[3], maslov_phase_cycles[3];
  int morse_index[3], count;
  CoherentSphereTTStatus status;
};
ccl_device_inline float coherent_sphere_tt_equation(
    float t, float A, float B, float n, float theta)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  return M_PI_F + 2 * asinf(t) - asinf(A * t) - asinf(B * t) - 2 * asinf(t / n) - theta;
}
ccl_device_inline float coherent_sphere_tt_normalized_derivative(float t,
                                                                 float A,
                                                                 float B,
                                                                 float n)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  const float v = max(0.0f, 1 - t * t);
  return 2 - A * sqrtf(v / (1 - A * A * t * t)) - B * sqrtf(v / (1 - B * B * t * t)) -
         (n == 1 ? 2.0f : (2 / n) * sqrtf(v / (1 - t * t / (n * n))));
}
ccl_device_inline void coherent_sphere_tt_point(
    float3 c, float R, float3 raw, ccl_private float2 p[3], ccl_private float3 *N)
{
  const float2 norm = coherent_geometry_distance_split(raw, zero_float3());
  for (int k = 0; k < 3; k++) {
    const float v = k == 0 ? raw.x : k == 1 ? raw.y : raw.z;
    const float hi = v / norm.x;
    const float lo = (coherent_geometry_fma(-hi, norm.x, v) - hi * norm.y) / norm.x;
    const float2 q = coherent_geometry_add_split(make_float2(hi, 0), make_float2(lo, 0));
    p[k] = coherent_geometry_add_split(make_float2(k == 0 ? c.x :
                                                   k == 1 ? c.y :
                                                            c.z,
                                                   0),
                                       coherent_geometry_product_split(q, R));
    if (k == 0)
      N->x = q.x + q.y;
    if (k == 1)
      N->y = q.x + q.y;
    if (k == 2)
      N->z = q.x + q.y;
  }
}
/* Jacobi inertia: no positive-Hessian assumption for a refractive lens. */
ccl_device_inline bool coherent_sphere_tt_inertia(
    const ccl_private float H[COHERENT_GEOMETRY_DIM][COHERENT_GEOMETRY_DIM],
    ccl_private int *negative)
{
  float a[4][4];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      a[i][j] = H[i][j];
  for (int step = 0; step < 40; step++) {
    int p = 0, q = 1;
    float biggest = 0;
    for (int i = 0; i < 4; i++)
      for (int j = i + 1; j < 4; j++)
        if (fabsf(a[i][j]) > biggest) {
          biggest = fabsf(a[i][j]);
          p = i;
          q = j;
        }
    if (biggest == 0)
      break;
    const float angle = .5f * atan2f(2 * a[p][q], a[q][q] - a[p][p]);
    const float c = cosf(angle), s = sinf(angle), app = a[p][p], aqq = a[q][q], apq = a[p][q];
    for (int k = 0; k < 4; k++)
      if (k != p && k != q) {
        const float x = a[k][p], y = a[k][q];
        a[k][p] = a[p][k] = c * x - s * y;
        a[k][q] = a[q][k] = s * x + c * y;
      }
    a[p][p] = c * c * app - 2 * c * s * apq + s * s * aqq;
    a[q][q] = s * s * app + 2 * c * s * apq + c * c * aqq;
    a[p][q] = a[q][p] = 0;
  }
  float largest = 0, smallest = 1e30f;
  *negative = 0;
  for (int i = 0; i < 4; i++) {
    largest = max(largest, fabsf(a[i][i]));
    smallest = min(smallest, fabsf(a[i][i]));
    *negative += a[i][i] < 0;
  }
  return largest > 0 && smallest > 1e-6f * largest;
}
/* Keep bounded root isolation separate from the later large Hessian scratch
 * arrays. This boundary also permits testing every isolated root directly. */
#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device_noinline
#endif
CoherentSphereTTStatus coherent_sphere_tt_roots(float A, float B, float ior, float theta,
                                               ccl_private float roots[3],
                                               ccl_private int *root_count)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  float bounds[4] = {-1, 1, 0, 0};
  int intervals = 1;
  if (ior > 1 && coherent_sphere_tt_normalized_derivative(0, A, B, ior) < 0) {
    float l = 0, h = 1;
    for (int k = 0; k < 30; k++) {
      const float m = (l + h) * .5f;
      if (coherent_sphere_tt_normalized_derivative(m, A, B, ior) > 0)
        h = m;
      else
        l = m;
    }
    const float t = (l + h) * .5f;
    bounds[1] = -t;
    bounds[2] = t;
    bounds[3] = 1;
    intervals = 3;
  }
  int count = 0;
  for (int k = 0; k <= intervals; k++) {
    const float f = coherent_sphere_tt_equation(bounds[k], A, B, ior, theta);
    if (fabsf(f) < 8e-7f) {
      const CoherentSphereTTStatus status = (k == 0 || k == intervals) ? COHERENT_SPHERE_TT_GRAZING :
                                                 COHERENT_SPHERE_TT_CAUSTIC;
      return status;
    }
  }
  for (int k = 0; k < intervals; k++) {
    float l = bounds[k], h = bounds[k + 1], fl = coherent_sphere_tt_equation(l, A, B, ior, theta);
    const float fh = coherent_sphere_tt_equation(h, A, B, ior, theta);
    if ((fl < 0) == (fh < 0))
      continue;
    for (int j = 0; j < 32; j++) {
      const float m = (l + h) * .5f, fm = coherent_sphere_tt_equation(m, A, B, ior, theta);
      if (fm == 0) {
        l = h = m;
        break;
      }
      if ((fl < 0) != (fm < 0))
        h = m;
      else {
        l = m;
        fl = fm;
      }
    }
    roots[count++] = (l + h) * .5f;
  }
  *root_count = count;
  return count ? COHERENT_SPHERE_TT_OK : COHERENT_SPHERE_TT_EMPTY;
}
ccl_device_inline CoherentSphereTTStatus
coherent_sphere_transmit_inventory(float3 source,
                                   float3 receiver,
                                   float3 receiver_normal,
                                   float3 center,
                                   float radius,
                                   float ior,
                                   ccl_private CoherentSphereTTInventory *out,
                                   const int selected_branch = -1)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  out->count = 0;
  out->status = COHERENT_SPHERE_TT_INVALID;
  const float3 s = source - center, r = receiver - center;
  const float a = len(s), b = len(r);
  const float receiver_normal_length = len(receiver_normal);
  if (!(radius > 0 && ior >= 1 && a > radius && b > radius) ||
      !isfinite_safe(radius + ior + a + b) || !(receiver_normal_length > 0) ||
      !isfinite_safe(receiver_normal_length))
    return out->status;
  receiver_normal /= receiver_normal_length;
  const float3 e0 = s / a, rd = r / b;
  const float cosine = clamp(dot(e0, rd), -1.0f, 1.0f);
  const float3 perp = rd - e0 * cosine;
  const float sine = len(perp);
  float3 e1, unused;
  if (sine > 1e-7f)
    e1 = perp / sine;
  else
    make_orthonormals(e0, &e1, &unused);
  const float theta = atan2f(sine, cosine), A = radius / a, B = radius / b;
  float roots[3];
  int count = 0;
  const CoherentSphereTTStatus root_status = coherent_sphere_tt_roots(A, B, ior, theta, roots, &count);
  if (root_status != COHERENT_SPHERE_TT_OK && root_status != COHERENT_SPHERE_TT_EMPTY) {
    out->status = root_status;
    return out->status;
  }
  if (sine < 1e-6f && count > 1) {
    out->status = COHERENT_SPHERE_TT_AXIAL_RING;
    return out->status;
  }
  /* Selected evaluation still reports the full isolated-root count, but only
   * path/frame[selected_branch] is initialized. All branch-domain errors above
   * remain global. Each candidate evaluates its own curvature/Hessian below. */
  if (selected_branch >= count) {
    out->status = COHERENT_SPHERE_TT_EMPTY;
    return out->status;
  }
  for (int branch = 0; branch < count; branch++) {
    if (selected_branch >= 0 && branch != selected_branch) continue;
    const float t = roots[branch];
    if (!(fabsf(t) < 1)) {
      out->status = COHERENT_SPHERE_TT_GRAZING;
      return out->status;
    }
    const float phi = asinf(t) - asinf(A * t), psi = phi + M_PI_F - 2 * asinf(t / ior);
    float2 p[2][3], sp[3], rp[3];
    float3 N[2];
    coherent_sphere_tt_point(center, radius, e0 * cosf(phi) + e1 * sinf(phi), p[0], &N[0]);
    coherent_sphere_tt_point(center, radius, e0 * cosf(psi) + e1 * sinf(psi), p[1], &N[1]);
    for (int k = 0; k < 3; k++) {
      sp[k] = make_float2(k == 0 ? source.x : k == 1 ? source.y : source.z, 0);
      rp[k] = make_float2(k == 0 ? receiver.x : k == 1 ? receiver.y : receiver.z, 0);
    }
    const float2 lengths[3] = {coherent_geometry_distance_split_affine(p[0], sp),
                               coherent_geometry_distance_split_affine(p[1], p[0]),
                               coherent_geometry_distance_split_affine(rp, p[1])};
    float3 v[3];
    float L[3];
    for (int k = 0; k < 3; k++) {
      float3 d;
      for (int axis = 0; axis < 3; axis++) {
        const float2 x = coherent_geometry_add_split(k == 0 ? p[0][axis] :
                                                     k == 1 ? p[1][axis] :
                                                              rp[axis],
                                                     make_float2(-(k == 0 ? sp[axis].x :
                                                                   k == 1 ? p[0][axis].x :
                                                                            p[1][axis].x),
                                                                 -(k == 0 ? sp[axis].y :
                                                                   k == 1 ? p[0][axis].y :
                                                                            p[1][axis].y)));
        if (axis == 0)
          d.x = x.x + x.y;
        if (axis == 1)
          d.y = x.x + x.y;
        if (axis == 2)
          d.z = x.x + x.y;
      }
      L[k] = lengths[k].x + lengths[k].y;
      if (!(L[k] > 0) || !isfinite_safe(L[k]))
        return out->status;
      v[k] = d / L[k];
    }
    if (!(dot(v[0], N[0]) < 0 && dot(v[1], N[0]) < 0 && dot(v[1], N[1]) > 0 &&
          dot(v[2], N[1]) > 0))
    {
      out->status = COHERENT_SPHERE_TT_INVALID;
      return out->status;
    }
    float3 basis[4];
    make_orthonormals(N[0], &basis[0], &basis[1]);
    make_orthonormals(N[1], &basis[2], &basis[3]);
    const float3 g[2] = {v[0] - ior * v[1], ior * v[1] - v[2]};
    float H[COHERENT_GEOMETRY_DIM][COHERENT_GEOMETRY_DIM] = {{0}};
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) {
        const float3 u = basis[i], w = basis[j];
        const int side = i / 2;
        if (i / 2 == j / 2) {
          const int outer = side == 0 ? 0 : 2;
          H[i][j] = (dot(u, w) - dot(u, v[outer]) * dot(w, v[outer])) / L[outer] +
                    ior * (dot(u, w) - dot(u, v[1]) * dot(w, v[1])) / L[1];
          if (i == j)
            H[i][j] -= dot(g[side], N[side]) / radius;
        }
        else
          H[i][j] = -ior * (dot(u, w) - dot(u, v[1]) * dot(w, v[1])) / L[1];
      }
    int morse = 0;
    if (!coherent_sphere_tt_inertia(H, &morse)) {
      out->status = COHERENT_SPHERE_TT_CAUSTIC;
      return out->status;
    }
    float3 ru, rv, su, sv;
    make_orthonormals(receiver_normal, &ru, &rv);
    make_orthonormals(v[0], &su, &sv);
    float J[4];
    for (int axis = 0; axis < 2; axis++) {
      const float3 rb = axis == 0 ? ru : rv, rhs3 = (rb - v[2] * dot(v[2], rb)) / L[2];
      float rhs[COHERENT_GEOMETRY_DIM] = {0}, q[COHERENT_GEOMETRY_DIM] = {0};
      rhs[2] = dot(basis[2], rhs3);
      rhs[3] = dot(basis[3], rhs3);
      if (!coherent_geometry_linear_solve(4, H, rhs, q)) {
        out->status = COHERENT_SPHERE_TT_CAUSTIC;
        return out->status;
      }
      const float3 motion = basis[0] * q[0] + basis[1] * q[1],
                   d = (motion - v[0] * dot(v[0], motion)) / L[0];
      J[axis] = dot(su, d);
      J[2 + axis] = dot(sv, d);
    }
    CoherentGeometryPath path{};
    path.count = 2;
    path.spreading = fabsf(J[0] * J[3] - J[1] * J[2]);
    path.source_direction = v[0];
    if (!isfinite_safe(path.spreading)) {
      out->status = COHERENT_SPHERE_TT_CAUSTIC;
      return out->status;
    }
    for (int k = 0; k < 2; k++)
      path.point[k] = make_float3(
          p[k][0].x + p[k][0].y, p[k][1].x + p[k][1].y, p[k][2].x + p[k][2].y);
    for (int k = 0; k < 3; k++)
      path.segment_length[k] = L[k];
    path.optical_length_split = coherent_geometry_add_split(
        coherent_geometry_add_split(lengths[0], coherent_geometry_product_split(lengths[1], ior)),
        lengths[2]);
    path.optical_length = path.optical_length_split.x + path.optical_length_split.y;
    out->path[branch] = path;
    out->angular_momentum[branch] = t * radius;
    out->morse_index[branch] = morse;
    out->maslov_phase_cycles[branch] = -.25f * morse;
    for (int k = 0; k < 2; k++)
      out->frame[branch][k] = {path.point[k],
                               basis[2 * k],
                               basis[2 * k + 1],
                               0,
                               0,
                               k == 0 ? 1 : ior,
                               k == 0 ? ior : 1,
                               k == 0 ? ior : 1,
                               COHERENT_GEOMETRY_TRANSMIT,
                               k == 0 ? 1 : -1};
  }
  out->count = count;
  out->status = count ? COHERENT_SPHERE_TT_OK : COHERENT_SPHERE_TT_EMPTY;
  return out->status;
}
CCL_NAMESPACE_END
