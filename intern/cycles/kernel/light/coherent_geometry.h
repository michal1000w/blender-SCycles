/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* A bounded planar connector. The caller enumerates ordered patches and checks
 * scene visibility and material closures; this routine solves one sequence. */
#define COHERENT_GEOMETRY_MAX_INTERFACES 4
#define COHERENT_GEOMETRY_DIM (2 * COHERENT_GEOMETRY_MAX_INTERFACES)

enum CoherentGeometryEvent { COHERENT_GEOMETRY_REFLECT = 0, COHERENT_GEOMETRY_TRANSMIT = 1 };

struct CoherentGeometryInterface {
  float3 center;
  float3 tangent_u;
  float3 tangent_v;
  float half_u;
  float half_v;
  float ior_before;
  float ior_after;
  float ior_opposite; /* Reflection Fresnel side; geometry retains travel medium. */
  int event;
  int expected_incident_side; /* +1: +normal/exterior, -1: -normal/interior, 0: unchecked. */
};

struct CoherentGeometryPath {
  float3 point[COHERENT_GEOMETRY_MAX_INTERFACES];
  float segment_length[COHERENT_GEOMETRY_MAX_INTERFACES + 1];
  float optical_length; /* Diagnostic only; use optical_length_split for phase. */
  /* Unevaluated high + low sum; subtract paired paths through the helper below. */
  float2 optical_length_split;
  /* d(source solid angle)/d(receiver surface area). Includes receiver cosine. */
  float spreading;
  float3 source_direction;
  int count;
};


ccl_device_inline float coherent_geometry_fma(const float a, const float b, const float c)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
#ifdef __KERNEL_METAL__
  return metal::fma(a, b, c);
#else
  return fmaf(a, b, c);
#endif
}

/* Under optimized CPU fast math, forced inlining lets the caller reassociate
 * TwoSum despite the local FP pragma. One out-of-line arithmetic boundary is
 * enough; ordinary CPU and Metal keep the inline implementation. */
#if defined(__clang__) && defined(__FAST_MATH__) && !defined(__KERNEL_METAL__)
ccl_device_noinline ccl_never_inline
#else
ccl_device_inline
#endif
float2 coherent_geometry_add_split(const float2 a, const float2 b)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  const float sum = a.x + b.x;
  const float bb = sum - a.x;
  const float error = (a.x - (sum - bb)) + (b.x - bb) + a.y + b.y;
  const float hi = sum + error;
  return make_float2(hi, error - (hi - sum));
}

ccl_device_inline float2 coherent_geometry_difference_split(const float a, const float b)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  return coherent_geometry_add_split(make_float2(a, 0.0f), make_float2(-b, 0.0f));
}

ccl_device_inline float2 coherent_geometry_product_split(const float2 a, const float b)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  const float hi = a.x * b;
  const float lo = coherent_geometry_fma(a.x, b, -hi) + a.y * b;
  return coherent_geometry_add_split(make_float2(hi, 0.0f), make_float2(lo, 0.0f));
}

/* Float input coordinates are the exact inputs to this expansion. FMA retains
 * low bits of the squared range; the sqrt correction retains the remaining
 * sub-ulp distance. It cannot restore precision lost before coordinates arrive. */
ccl_device_inline float2 coherent_geometry_distance_split(const float3 a, const float3 b)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  const float2 delta[3] = {coherent_geometry_difference_split(a.x, b.x),
                           coherent_geometry_difference_split(a.y, b.y),
                           coherent_geometry_difference_split(a.z, b.z)};
  float2 squared = make_float2(0.0f, 0.0f);
  for (int axis = 0; axis < 3; axis++) {
    const float hi = delta[axis].x * delta[axis].x;
    const float lo = coherent_geometry_fma(delta[axis].x, delta[axis].x, -hi) +
                     2.0f * delta[axis].x * delta[axis].y;
    squared = coherent_geometry_add_split(squared, make_float2(hi, lo));
  }
  const float root = sqrtf(squared.x);
  if (!(root > 0.0f)) return make_float2(0.0f, 0.0f);
  const float correction =
      (coherent_geometry_fma(-root, root, squared.x) + squared.y) / (2.0f * root);
  return coherent_geometry_add_split(make_float2(root, 0.0f),
                                     make_float2(correction, 0.0f));
}

ccl_device_inline float2 coherent_geometry_affine_component(
    const ccl_private CoherentGeometryInterface &patch, const float u, const float v, const int axis)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  const float center = axis == 0 ? patch.center.x : axis == 1 ? patch.center.y : patch.center.z;
  const float tangent_u =
      axis == 0 ? patch.tangent_u.x : axis == 1 ? patch.tangent_u.y : patch.tangent_u.z;
  const float tangent_v =
      axis == 0 ? patch.tangent_v.x : axis == 1 ? patch.tangent_v.y : patch.tangent_v.z;
  return coherent_geometry_add_split(
      coherent_geometry_add_split(make_float2(center, 0.0f),
                                  coherent_geometry_product_split(make_float2(u, 0.0f), tangent_u)),
      coherent_geometry_product_split(make_float2(v, 0.0f), tangent_v));
}

ccl_device_inline float2 coherent_geometry_distance_split_affine(const ccl_private float2 a[3],
                                                                  const ccl_private float2 b[3])
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  float2 squared = make_float2(0.0f, 0.0f);
  for (int axis = 0; axis < 3; axis++) {
    const float2 difference = coherent_geometry_add_split(
        a[axis], make_float2(-b[axis].x, -b[axis].y));
    const float hi = difference.x * difference.x;
    const float lo = coherent_geometry_fma(difference.x, difference.x, -hi) +
                     2.0f * difference.x * difference.y;
    squared = coherent_geometry_add_split(squared, make_float2(hi, lo));
  }
  const float root = sqrtf(squared.x);
  if (!(root > 0.0f)) return make_float2(0.0f, 0.0f);
  return coherent_geometry_add_split(
      make_float2(root, 0.0f),
      make_float2((coherent_geometry_fma(-root, root, squared.x) + squared.y) / (2.0f * root),
                  0.0f));
}

ccl_device_inline float coherent_geometry_optical_difference(const ccl_private CoherentGeometryPath &a,
                                                              const ccl_private CoherentGeometryPath &b)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  const float2 neg_b = make_float2(-b.optical_length_split.x, -b.optical_length_split.y);
  const float2 difference = coherent_geometry_add_split(a.optical_length_split, neg_b);
  return difference.x + difference.y;
}

ccl_device_inline float2 coherent_geometry_optical_length_split(
    const float3 source,
    const float3 receiver,
    const ccl_private CoherentGeometryInterface patches[COHERENT_GEOMETRY_MAX_INTERFACES],
    const int count,
    const ccl_private float q[COHERENT_GEOMETRY_DIM])
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  float2 points[COHERENT_GEOMETRY_MAX_INTERFACES + 2][3];
  points[0][0] = make_float2(source.x, 0.0f);
  points[0][1] = make_float2(source.y, 0.0f);
  points[0][2] = make_float2(source.z, 0.0f);
  points[count + 1][0] = make_float2(receiver.x, 0.0f);
  points[count + 1][1] = make_float2(receiver.y, 0.0f);
  points[count + 1][2] = make_float2(receiver.z, 0.0f);
  for (int i = 0; i < count; i++) {
    for (int axis = 0; axis < 3; axis++) {
      points[i + 1][axis] = coherent_geometry_affine_component(
          patches[i], q[2 * i], q[2 * i + 1], axis);
    }
  }
  float2 length = make_float2(0.0f, 0.0f);
  for (int s = 0; s <= count; s++) {
    const float n = s == 0 ? patches[0].ior_before : patches[s - 1].ior_after;
    const float2 distance = coherent_geometry_distance_split_affine(points[s], points[s + 1]);
    length = coherent_geometry_add_split(length, coherent_geometry_product_split(distance, n));
  }
  return length;
}

/* Dense bounded solve is deliberate: eight unknowns maximum, fixed stack size,
 * and partial pivoting makes nearly grazing rejected cases unambiguous. */
ccl_device_inline bool coherent_geometry_linear_solve(const int dim,
                                                       const ccl_private float input[COHERENT_GEOMETRY_DIM]
                                                                        [COHERENT_GEOMETRY_DIM],
                                                       const ccl_private float rhs[COHERENT_GEOMETRY_DIM],
                                                       ccl_private float solution[COHERENT_GEOMETRY_DIM])
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  float a[COHERENT_GEOMETRY_DIM][COHERENT_GEOMETRY_DIM + 1];
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) a[i][j] = input[i][j];
    a[i][dim] = rhs[i];
  }
  for (int k = 0; k < dim; k++) {
    int pivot = k;
    for (int i = k + 1; i < dim; i++) {
      if (fabsf(a[i][k]) > fabsf(a[pivot][k])) pivot = i;
    }
    float row_scale = 0.0f;
    for (int j = k; j < dim; j++) row_scale = fmaxf(row_scale, fabsf(a[pivot][j]));
    if (!(fabsf(a[pivot][k]) > fmaxf(1.0e-12f, 1.0e-8f * row_scale))) return false;
    if (pivot != k) {
      for (int j = k; j <= dim; j++) {
        const float tmp = a[k][j];
        a[k][j] = a[pivot][j];
        a[pivot][j] = tmp;
      }
    }
    const float inv = 1.0f / a[k][k];
    for (int i = k + 1; i < dim; i++) {
      const float factor = a[i][k] * inv;
      for (int j = k + 1; j <= dim; j++) a[i][j] -= factor * a[k][j];
      a[i][k] = 0.0f;
    }
  }
  for (int i = dim - 1; i >= 0; i--) {
    float value = a[i][dim];
    for (int j = i + 1; j < dim; j++) value -= a[i][j] * solution[j];
    solution[i] = value / a[i][i];
    if (!isfinite_safe(solution[i])) return false;
  }
  return true;
}

ccl_device_inline float3 coherent_geometry_basis(const ccl_private CoherentGeometryInterface &patch,
                                                  const int axis)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  return axis == 0 ? patch.tangent_u : patch.tangent_v;
}

ccl_device_inline float3 coherent_geometry_point(const ccl_private CoherentGeometryInterface &patch,
                                                  const float u,
                                                  const float v)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  return patch.center + u * patch.tangent_u + v * patch.tangent_v;
}

ccl_device_inline float3 coherent_geometry_point_relative(const ccl_private CoherentGeometryInterface &patch,
                                                           const float u,
                                                           const float v,
                                                           const float3 origin)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  return (patch.center - origin) + u * patch.tangent_u + v * patch.tangent_v;
}

/* Fermat gradient and Hessian for positive piecewise refractive indices.
 * Stationary reflection and transmission differ in their geometric side test. */
ccl_device_inline bool coherent_geometry_system(
    const float3 source,
    const float3 receiver,
    const ccl_private CoherentGeometryInterface patches[COHERENT_GEOMETRY_MAX_INTERFACES],
    const int count,
    const ccl_private float q[COHERENT_GEOMETRY_DIM],
    ccl_private float *r_length,
    ccl_private float gradient[COHERENT_GEOMETRY_DIM],
    ccl_private float hessian[COHERENT_GEOMETRY_DIM][COHERENT_GEOMETRY_DIM])
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  float3 points[COHERENT_GEOMETRY_MAX_INTERFACES + 2];
  points[0] = zero_float3();
  points[count + 1] = receiver - source;
  for (int i = 0; i < count; i++)
    points[i + 1] = coherent_geometry_point_relative(patches[i], q[2 * i], q[2 * i + 1], source);
  const int dim = 2 * count;
  for (int i = 0; i < dim; i++) {
    gradient[i] = 0.0f;
    for (int j = 0; j < dim; j++) hessian[i][j] = 0.0f;
  }
  float length = 0.0f;
  for (int s = 0; s <= count; s++) {
    const float3 delta = points[s + 1] - points[s];
    const float distance = len(delta);
    if (!(distance > 1.0e-12f) || !isfinite_safe(distance)) return false;
    const float3 direction = delta / distance;
    const float n = s == 0 ? patches[0].ior_before : patches[s - 1].ior_after;
    if (!(n > 0.0f) || !isfinite_safe(n)) return false;
    length += n * distance;
    for (int endpoint = 0; endpoint < 2; endpoint++) {
      const int surface_index = s - 1 + endpoint;
      if (surface_index < 0 || surface_index >= count) continue;
      const float sign = endpoint == 0 ? -1.0f : 1.0f;
      for (int a = 0; a < 2; a++) {
        const int row = 2 * surface_index + a;
        const float3 basis_a = coherent_geometry_basis(patches[surface_index], a);
        gradient[row] += sign * n * dot(direction, basis_a);
        for (int other = 0; other < 2; other++) {
          const int adjacent = s - 1 + other;
          if (adjacent < 0 || adjacent >= count) continue;
          const float other_sign = other == 0 ? -1.0f : 1.0f;
          for (int b = 0; b < 2; b++) {
            const float3 basis_b = coherent_geometry_basis(patches[adjacent], b);
            const float projected = dot(basis_a, basis_b) -
                                    dot(basis_a, direction) * dot(direction, basis_b);
            hessian[row][2 * adjacent + b] += sign * other_sign * n / distance * projected;
          }
        }
      }
    }
  }
  *r_length = length;
  return isfinite_safe(length);
}

/* Planar reflection is an isometry: unfold the source through the ordered
 * planes, then intersect backwards. This avoids a singular range Hessian when
 * two valid reflections approach a shared corner. Images determine geometry
 * and spreading only; phase still uses compensated physical segment lengths. */
ccl_device_inline bool coherent_geometry_connect_reflections(
    const float3 source,
    const float3 receiver,
    const float3 receiver_normal,
    const ccl_private CoherentGeometryInterface patches[COHERENT_GEOMETRY_MAX_INTERFACES],
    const int count,
    ccl_private CoherentGeometryPath *path)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  float3 image[COHERENT_GEOMETRY_MAX_INTERFACES + 1];
  image[0] = zero_float3();
  for (int i = 0; i < count; i++) {
    const float3 normal = normalize(cross(patches[i].tangent_u, patches[i].tangent_v));
    const float3 center = patches[i].center - source;
    image[i + 1] = image[i] - 2.0f * dot(image[i] - center, normal) * normal;
  }
  float q[COHERENT_GEOMETRY_DIM];
  float3 points[COHERENT_GEOMETRY_MAX_INTERFACES + 2];
  points[0] = zero_float3();
  points[count + 1] = receiver - source;
  float3 start = points[count + 1];
  for (int i = count - 1; i >= 0; i--) {
    const ccl_private CoherentGeometryInterface &p = patches[i];
    const float3 normal = normalize(cross(p.tangent_u, p.tangent_v));
    const float3 center = p.center - source;
    const float3 direction = image[i + 1] - start;
    const float denominator = dot(direction, normal);
    if (!(fabsf(denominator) > 1.0e-12f)) return false;
    const float t = dot(center - start, normal) / denominator;
    if (!(t > 0.0f && t < 1.0f)) return false;
    const float3 hit_relative = (start - center) + t * direction;
    q[2 * i] = dot(hit_relative, p.tangent_u);
    q[2 * i + 1] = dot(hit_relative, p.tangent_v);
    if (fabsf(q[2 * i]) > p.half_u || fabsf(q[2 * i + 1]) > p.half_v) return false;
    points[i + 1] = coherent_geometry_point_relative(p, q[2 * i], q[2 * i + 1], source);
    path->point[i] = source + points[i + 1];
    start = points[i + 1];
  }
  for (int i = 0; i < count; i++) {
    const ccl_private CoherentGeometryInterface &p = patches[i];
    const float3 normal = cross(p.tangent_u, p.tangent_v);
    const float3 incoming = points[i] - points[i + 1];
    const float3 outgoing = points[i + 2] - points[i + 1];
    const float incoming_distance = len(incoming);
    const float outgoing_distance = len(outgoing);
    if (!(incoming_distance > 1.0e-12f && outgoing_distance > 1.0e-12f)) return false;
    const float side_before = dot(incoming / incoming_distance, normal);
    const float side_after = dot(outgoing / outgoing_distance, normal);
    if (fabsf(side_before) <= 1.0e-6f || fabsf(side_after) <= 1.0e-6f ||
        side_before * side_after <= 0.0f ||
        (p.expected_incident_side != 0 && side_before * float(p.expected_incident_side) <= 0.0f))
      return false;
  }
  for (int i = 0; i <= count; i++) path->segment_length[i] = len(points[i + 1] - points[i]);
  path->source_direction = normalize(points[1]);
  path->optical_length_split = coherent_geometry_optical_length_split(source, receiver, patches, count, q);
  path->optical_length = path->optical_length_split.x + path->optical_length_split.y;
  const float3 image_delta = points[count + 1] - image[count];
  const float image_distance = len(image_delta);
  if (!(image_distance > 1.0e-12f)) return false;
  path->spreading = fabsf(dot(normalize(receiver_normal), image_delta / image_distance)) /
                    (image_distance * image_distance);
  return path->spreading > 0.0f && isfinite_safe(path->spreading);
}

ccl_device_inline bool coherent_geometry_connect(
    const float3 source,
    const float3 receiver,
    const float3 receiver_normal,
    const ccl_private CoherentGeometryInterface patches[COHERENT_GEOMETRY_MAX_INTERFACES],
    const int count,
    ccl_private CoherentGeometryPath *path)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  if (count < 0 || count > COHERENT_GEOMETRY_MAX_INTERFACES) return false;
  if (!(len_squared(receiver_normal) > 0.0f) || !isfinite_safe(receiver_normal)) return false;
  path->count = count;
  if (count == 0) {
    const float3 segment = receiver - source;
    const float distance = len(segment);
    if (!(distance > 1.0e-12f)) return false;
    path->source_direction = segment / distance;
    path->segment_length[0] = distance;
    path->optical_length = distance;
    path->optical_length_split = coherent_geometry_distance_split(source, receiver);
    path->spreading = fabsf(dot(receiver_normal, path->source_direction)) /
                      (distance * distance);
    return path->spreading > 0.0f;
  }
  float q[COHERENT_GEOMETRY_DIM];
  for (int i = 0; i < count; i++) {
    const ccl_private CoherentGeometryInterface &p = patches[i];
    if (!(p.half_u > 0.0f && p.half_v > 0.0f && p.ior_before > 0.0f &&
          p.ior_after > 0.0f) ||
        fabsf(dot(p.tangent_u, p.tangent_v)) > 1.0e-4f ||
        fabsf(len(p.tangent_u) - 1.0f) > 1.0e-4f ||
        fabsf(len(p.tangent_v) - 1.0f) > 1.0e-4f ||
        (p.event != COHERENT_GEOMETRY_REFLECT && p.event != COHERENT_GEOMETRY_TRANSMIT))
      return false;
    if (p.expected_incident_side < -1 || p.expected_incident_side > 1) return false;
    if (i + 1 < count && fabsf(p.ior_after - patches[i + 1].ior_before) > 1.0e-5f)
      return false;
    if (p.event == COHERENT_GEOMETRY_REFLECT &&
        fabsf(p.ior_before - p.ior_after) > 1.0e-5f)
      return false;
    const float3 guess_from_center = (source - p.center) +
                                     (float(i + 1) / float(count + 1)) * (receiver - source);
    q[2 * i] = dot(guess_from_center, p.tangent_u);
    q[2 * i + 1] = dot(guess_from_center, p.tangent_v);
  }
  bool all_reflections = true;
  for (int i = 0; i < count; i++) {
    all_reflections &= patches[i].event == COHERENT_GEOMETRY_REFLECT &&
                       patches[i].ior_before == patches[0].ior_before &&
                       patches[i].ior_after == patches[0].ior_before;
  }
  if (all_reflections) {
    return coherent_geometry_connect_reflections(source, receiver, receiver_normal, patches, count, path);
  }
  const int dim = 2 * count;
  ccl_private float gradient[COHERENT_GEOMETRY_DIM];
  ccl_private float hessian[COHERENT_GEOMETRY_DIM][COHERENT_GEOMETRY_DIM];
  float length = 0.0f;
  bool converged = false;
  for (int iteration = 0; iteration < 32; iteration++) {
    if (!coherent_geometry_system(source, receiver, patches, count, q, &length, gradient, hessian))
      return false;
    float norm2 = 0.0f;
    for (int j = 0; j < dim; j++) norm2 += gradient[j] * gradient[j];
    if (norm2 < 1.0e-10f) {
      converged = true;
      break;
    }
    float step[COHERENT_GEOMETRY_DIM];
    if (!coherent_geometry_linear_solve(dim, hessian, gradient, step)) return false;
    /* A full Newton step can pass through a near-coincident pair of interface
     * points, where the range Hessian changes rapidly. Keep each point's
     * proposed motion below half the shortest current segment, then use the
     * accurate optical-length Armijo test below. This remains a stationary
     * path solve; no path length or field value is clamped. */
    float shortest_segment = FLT_MAX;
    float longest_motion = 0.0f;
    float3 previous = zero_float3();
    for (int i = 0; i < count; i++) {
      const float3 point = coherent_geometry_point_relative(
          patches[i], q[2 * i], q[2 * i + 1], source);
      shortest_segment = fminf(shortest_segment, len(point - previous));
      previous = point;
      const float3 motion = step[2 * i] * patches[i].tangent_u +
                            step[2 * i + 1] * patches[i].tangent_v;
      longest_motion = fmaxf(longest_motion, len(motion));
    }
    shortest_segment = fminf(shortest_segment, len((receiver - source) - previous));
    if (!(shortest_segment > 1.0e-12f) || !isfinite_safe(shortest_segment)) return false;
    if (longest_motion > 0.5f * shortest_segment) {
      const float factor = 0.5f * shortest_segment / longest_motion;
      for (int j = 0; j < dim; j++) step[j] *= factor;
    }
    float decrease = 0.0f;
    for (int j = 0; j < dim; j++) decrease += gradient[j] * step[j];
    if (!(decrease > 0.0f)) return false;
    const float2 length_split = coherent_geometry_optical_length_split(
        source, receiver, patches, count, q);
    bool accepted = false;
    for (int trial = 0; trial < 16; trial++) {
      const float scale = ldexpf(1.0f, -trial);
      float candidate[COHERENT_GEOMETRY_DIM];
      for (int j = 0; j < dim; j++) candidate[j] = q[j] - scale * step[j];
      float candidate_gradient[COHERENT_GEOMETRY_DIM];
      float candidate_hessian[COHERENT_GEOMETRY_DIM][COHERENT_GEOMETRY_DIM];
      float candidate_length;
      if (coherent_geometry_system(source,
                                   receiver,
                                   patches,
                                   count,
                                   candidate,
                                   &candidate_length,
                                   candidate_gradient,
                                   candidate_hessian))
      {
        const float2 candidate_split = coherent_geometry_optical_length_split(
            source, receiver, patches, count, candidate);
        const float2 difference = coherent_geometry_add_split(
            candidate_split, make_float2(-length_split.x, -length_split.y));
        if (difference.x + difference.y <= -1.0e-4f * scale * decrease) {
          for (int j = 0; j < dim; j++) q[j] = candidate[j];
          accepted = true;
          break;
        }
      }
    }
    if (!accepted) return false;
  }
  if (!converged ||
      !coherent_geometry_system(source, receiver, patches, count, q, &length, gradient, hessian))
    return false;

  float3 points[COHERENT_GEOMETRY_MAX_INTERFACES + 2];
  points[0] = zero_float3();
  points[count + 1] = receiver - source;
  for (int i = 0; i < count; i++) {
    const ccl_private CoherentGeometryInterface &p = patches[i];
    if (fabsf(q[2 * i]) > p.half_u || fabsf(q[2 * i + 1]) > p.half_v) return false;
    points[i + 1] = coherent_geometry_point_relative(p, q[2 * i], q[2 * i + 1], source);
    path->point[i] = source + points[i + 1];
  }
  for (int i = 0; i < count; i++) {
    const ccl_private CoherentGeometryInterface &p = patches[i];
    const float3 normal = cross(p.tangent_u, p.tangent_v);
    const float3 incoming = points[i] - points[i + 1];
    const float3 outgoing = points[i + 2] - points[i + 1];
    const float incoming_distance = len(incoming);
    const float outgoing_distance = len(outgoing);
    if (!(incoming_distance > 1.0e-12f && outgoing_distance > 1.0e-12f)) return false;
    const float side_before = dot(incoming / incoming_distance, normal);
    const float side_after = dot(outgoing / outgoing_distance, normal);
    if (fabsf(side_before) <= 1.0e-6f || fabsf(side_after) <= 1.0e-6f) return false;
    if (p.expected_incident_side != 0 &&
        side_before * float(p.expected_incident_side) <= 0.0f)
      return false;
    if (p.event == COHERENT_GEOMETRY_REFLECT ?
            side_before * side_after <= 0.0f :
            side_before * side_after >= 0.0f)
      return false;
  }
  for (int s = 0; s <= count; s++) path->segment_length[s] = len(points[s + 1] - points[s]);
  path->optical_length = length;
  path->optical_length_split = coherent_geometry_optical_length_split(
      source, receiver, patches, count, q);
  path->source_direction = normalize(points[1]);

  /* Implicitly differentiate the stationary constraints with respect to two
   * receiver tangent coordinates. H dq/dR = n_last B_last^T P(d_last)/d_last.
   * Then differentiate the first source ray; determinant converts receiver
   * area to source solid angle, including all ideal interface focusing. */
  float3 receiver_u, receiver_v;
  make_orthonormals(normalize(receiver_normal), &receiver_u, &receiver_v);
  const float3 last_direction = normalize(points[count + 1] - points[count]);
  const float last_distance = path->segment_length[count];
  const float last_ior = patches[count - 1].ior_after;
  const float3 source_direction = path->source_direction;
  const float source_distance = path->segment_length[0];
  float2 angular_derivative[2];
  for (int axis = 0; axis < 2; axis++) {
    const float3 receiver_axis = axis == 0 ? receiver_u : receiver_v;
    const float3 projected = receiver_axis - last_direction * dot(last_direction, receiver_axis);
    float rhs[COHERENT_GEOMETRY_DIM] = {0.0f};
    for (int a = 0; a < 2; a++) {
      rhs[2 * (count - 1) + a] = last_ior / last_distance *
                                 dot(coherent_geometry_basis(patches[count - 1], a), projected);
    }
    float derivative[COHERENT_GEOMETRY_DIM];
    if (!coherent_geometry_linear_solve(dim, hessian, rhs, derivative)) return false;
    const float3 moving_first = derivative[0] * patches[0].tangent_u +
                                derivative[1] * patches[0].tangent_v;
    const float3 d_direction =
        (moving_first - source_direction * dot(source_direction, moving_first)) / source_distance;
    float3 source_u, source_v;
    make_orthonormals(source_direction, &source_u, &source_v);
    angular_derivative[axis] = make_float2(dot(d_direction, source_u),
                                           dot(d_direction, source_v));
  }
  path->spreading = fabsf(angular_derivative[0].x * angular_derivative[1].y -
                            angular_derivative[0].y * angular_derivative[1].x);
  return isfinite_safe(path->spreading) && path->spreading > 0.0f;
}

CCL_NAMESPACE_END
