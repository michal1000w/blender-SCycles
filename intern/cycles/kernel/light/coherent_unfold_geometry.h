/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/light/coherent_curved_geometry.h"
#include "kernel/light/coherent_sphere_transmit_geometry.h"
CCL_NAMESPACE_BEGIN

/* Planar mirrors surrounding one convex sphere block are exact isometries.
 * Split virtual endpoints and first variations preserve stationary phase to
 * second order in float endpoint rounding. No tangent-plane sphere model. */
ccl_device_inline void coherent_unfold_image(ccl_private float2 image[3],
    const ccl_private CoherentGeometryInterface &p)
{
  const float3 n = normalize(cross(p.tangent_u, p.tangent_v));
  const float nv[3] = {n.x, n.y, n.z}, c[3] = {p.center.x, p.center.y, p.center.z};
  float2 distance = make_float2(0, 0), norm_squared = make_float2(0, 0);
  for (int a = 0; a < 3; a++) {
    distance = coherent_geometry_add_split(distance, coherent_geometry_product_split(
        coherent_geometry_add_split(image[a], make_float2(-c[a], 0)), nv[a]));
    norm_squared = coherent_geometry_add_split(norm_squared,
        coherent_geometry_product_split(make_float2(nv[a], 0), nv[a]));
  }
  const float quotient = distance.x / norm_squared.x;
  const float residual = coherent_geometry_fma(-quotient, norm_squared.x, distance.x) +
                         distance.y - quotient * norm_squared.y;
  distance = coherent_geometry_add_split(make_float2(quotient, 0),
                                        make_float2(residual / norm_squared.x, 0));
  for (int a = 0; a < 3; a++)
    image[a] = coherent_geometry_add_split(image[a],
        coherent_geometry_product_split(distance, -2 * nv[a]));
}

ccl_device_inline bool coherent_unfold_reconstruct(const float3 endpoint,
    const float3 sphere_point, const ccl_private CoherentGeometryInterface *frames,
    const int first, const int stop, const int step, ccl_private CoherentGeometryPath *path)
{
  float3 images[COHERENT_GEOMETRY_MAX_INTERFACES + 1];
  images[0] = endpoint;
  int count = 0;
  for (int i = first; i != stop; i += step) {
    const float3 n = normalize(cross(frames[i].tangent_u, frames[i].tangent_v));
    images[count + 1] = images[count] - (2 * dot(images[count] - frames[i].center, n) / dot(n, n)) * n;
    count++;
  }
  float3 start = sphere_point;
  for (int i = stop - step; i != first - step; i -= step) {
    const ccl_private CoherentGeometryInterface &p = frames[i];
    const float3 n = normalize(cross(p.tangent_u, p.tangent_v));
    const float3 direction = images[count] - start;
    const float denominator = dot(direction, n);
    if (!(fabsf(denominator) > 1e-12f)) return false;
    const float t = dot(p.center - start, n) / denominator;
    if (!(t > 0 && t < 1)) return false;
    const float3 hit = start + t * direction;
    if (fabsf(dot(hit - p.center, p.tangent_u)) > p.half_u ||
        fabsf(dot(hit - p.center, p.tangent_v)) > p.half_v) return false;
    path->point[i] = hit;
    start = hit;
    count--;
  }
  return true;
}

ccl_device_inline bool coherent_unfold_finish(const float3 source, const float3 receiver,
    const ccl_private float2 source_image[3], const ccl_private float2 receiver_image[3],
    const float3 virtual_source, const float3 virtual_receiver,
    const ccl_private CoherentGeometryInterface *frames, const int count,
    const int sphere_first, const int sphere_count,
    const ccl_private CoherentGeometryPath &sphere_path, ccl_private CoherentGeometryPath *path)
{
  *path = sphere_path;
  for (int i = 0; i < sphere_count; i++) path->point[sphere_first + i] = sphere_path.point[i];
  if (!coherent_unfold_reconstruct(source, sphere_path.point[0], frames, 0, sphere_first, 1, path) ||
      !coherent_unfold_reconstruct(receiver, sphere_path.point[sphere_count - 1], frames,
                                  count - 1, sphere_first + sphere_count - 1, -1, path)) return false;
  const float3 end_direction = normalize(virtual_receiver - sphere_path.point[sphere_count - 1]);
  const float sv[3] = {virtual_source.x, virtual_source.y, virtual_source.z};
  const float rv[3] = {virtual_receiver.x, virtual_receiver.y, virtual_receiver.z};
  const float sd[3] = {sphere_path.source_direction.x, sphere_path.source_direction.y,
                      sphere_path.source_direction.z};
  const float rd[3] = {end_direction.x, end_direction.y, end_direction.z};
  for (int a = 0; a < 3; a++) {
    path->optical_length_split = coherent_geometry_add_split(path->optical_length_split,
        coherent_geometry_product_split(coherent_geometry_add_split(source_image[a],
            make_float2(-sv[a], 0)), -sd[a]));
    path->optical_length_split = coherent_geometry_add_split(path->optical_length_split,
        coherent_geometry_product_split(coherent_geometry_add_split(receiver_image[a],
            make_float2(-rv[a], 0)), rd[a]));
  }
  for (int i = 0; i <= count; i++) {
    const float3 before = i == 0 ? source : path->point[i - 1];
    const float3 after = i == count ? receiver : path->point[i];
    path->segment_length[i] = len(after - before);
    if (!(path->segment_length[i] > 1e-7f)) return false;
  }
  for (int i = 0; i < count; i++) {
    if (i >= sphere_first && i < sphere_first + sphere_count) continue;
    const float3 n = normalize(cross(frames[i].tangent_u, frames[i].tangent_v));
    const float3 before = i == 0 ? source : path->point[i - 1];
    const float3 after = i + 1 == count ? receiver : path->point[i + 1];
    const float side_in = dot(normalize(before - path->point[i]), n);
    const float side_out = dot(normalize(after - path->point[i]), n);
    if (fabsf(side_in) <= 1e-6f || fabsf(side_out) <= 1e-6f || side_in * side_out <= 0 ||
        (frames[i].expected_incident_side != 0 &&
         side_in * frames[i].expected_incident_side <= 0)) return false;
  }
  path->source_direction = normalize(path->point[0] - source);
  path->count = count;
  path->optical_length = path->optical_length_split.x + path->optical_length_split.y;
  return true;
}

/* Status preserves the analytic sphere inventory failures; a clipped physical
 * route is EMPTY, never an inventory error. */
#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device_noinline
#endif
CoherentSphereTTStatus coherent_unfold_sphere_connect(
    const float3 source, const float3 receiver, const float3 receiver_normal,
    ccl_private CoherentGeometryInterface *frames, const int count,
    const int sphere_first, const bool transmit, const int branch,
    const float3 center, const float radius, const float ior,
    ccl_private CoherentGeometryPath *path, ccl_private float *phase)
{
  const int sphere_count = transmit ? 2 : 1;
  if (!(radius > 0) || !isfinite_safe(radius) || !isfinite_safe(source) ||
      !isfinite_safe(receiver) || !isfinite_safe(center) ||
      !(len(receiver_normal) > 0) || !isfinite_safe(receiver_normal) ||
      (transmit && (!(ior >= 1) || !isfinite_safe(ior)))) return COHERENT_SPHERE_TT_INVALID;
  if (count < sphere_count || count > COHERENT_GEOMETRY_MAX_INTERFACES ||
      sphere_first < 0 || sphere_first + sphere_count > count)
    return COHERENT_SPHERE_TT_INVALID;
  float2 si[3] = {make_float2(source.x, 0), make_float2(source.y, 0), make_float2(source.z, 0)};
  float2 ri[3] = {make_float2(receiver.x, 0), make_float2(receiver.y, 0), make_float2(receiver.z, 0)};
  for (int i = 0; i < sphere_first; i++) coherent_unfold_image(si, frames[i]);
  float3 image_normal = receiver_normal;
  for (int i = count - 1; i >= sphere_first + sphere_count; i--) {
    coherent_unfold_image(ri, frames[i]);
    const float3 n = normalize(cross(frames[i].tangent_u, frames[i].tangent_v));
    image_normal -= (2 * dot(image_normal, n) / dot(n, n)) * n;
  }
  const float3 virtual_source = make_float3(si[0].x, si[1].x, si[2].x);
  const float3 virtual_receiver = make_float3(ri[0].x, ri[1].x, ri[2].x);
  CoherentGeometryPath sphere_path;
  *phase = 0;
  /* Unfolding can place a virtual endpoint inside the sphere even though both
   * physical endpoints are exterior. Such a mirror topology cannot realize
   * this exterior sphere block; it is an empty candidate, not a bad scene. */
  if (!(len(virtual_source - center) > radius &&
        len(virtual_receiver - center) > radius)) return COHERENT_SPHERE_TT_EMPTY;
  if (transmit) {
    CoherentSphereTTInventory inventory;
    const CoherentSphereTTStatus status = coherent_sphere_transmit_inventory(
        virtual_source, virtual_receiver, image_normal, center, radius, ior, &inventory, branch);
    if (status != COHERENT_SPHERE_TT_OK) return status;
    if (branch < 0 || branch >= inventory.count) return COHERENT_SPHERE_TT_EMPTY;
    sphere_path = inventory.path[branch];
    frames[sphere_first] = inventory.frame[branch][0];
    frames[sphere_first + 1] = inventory.frame[branch][1];
    *phase = inventory.maslov_phase_cycles[branch];
  }
  else if (!coherent_sphere_reflect(virtual_source, virtual_receiver, image_normal,
                                   center, radius, &frames[sphere_first], &sphere_path))
    return COHERENT_SPHERE_TT_EMPTY;
  return coherent_unfold_finish(source, receiver, si, ri, virtual_source, virtual_receiver,
                               frames, count, sphere_first, sphere_count, sphere_path, path) ?
      COHERENT_SPHERE_TT_OK : COHERENT_SPHERE_TT_EMPTY;
}
CCL_NAMESPACE_END
