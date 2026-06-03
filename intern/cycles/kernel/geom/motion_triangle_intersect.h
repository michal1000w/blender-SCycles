/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Motion Triangle Primitive
 *
 * These are stored as regular triangles, plus extra positions and normals at
 * times other than the frame center. Computing the triangle vertex positions
 * or normals at a given ray time is a matter of interpolation of the two steps
 * between which the ray time lies.
 *
 * The extra positions are stored as additional motion steps in ATTR_STD_POSITION.
 * Normals in ATTR_STD_VERTEX_NORMAL and ATTR_STD_CORNER_NORMAL.
 */

#pragma once

#include "kernel/globals.h"
#include "kernel/types.h"

#include "kernel/geom/geom_intersect.h"
#include "kernel/geom/motion_triangle.h"
#include "kernel/geom/object.h"
#include "kernel/geom/pixel_displacement.h"

#include "util/math_intersect.h"

CCL_NAMESPACE_BEGIN

/* Ray intersection. We simply compute the vertex positions at the given ray
 * time and do a ray intersection with the resulting triangle.
 */

ccl_device_inline bool motion_triangle_intersect(KernelGlobals kg,
                                                 ccl_private Intersection *isect,
                                                 const float3 P,
                                                 const float3 dir,
                                                 const float tmin,
                                                 const float tmax,
                                                 const float time,
                                                 const uint visibility,
                                                 const int object,
                                                 const int prim,
                                                 const int prim_addr)
{
  /* Get vertex locations for intersection. */
  float3 verts[3];
  motion_triangle_vertices(kg, object, prim, time, verts);
#ifdef __KERNEL_METAL_PIXEL_DISPLACEMENT__
  if (pixel_displacement_active(kg, prim)) {
#ifdef __VISIBILITY_FLAG__
    if (!(kernel_data_fetch(prim_visibility, prim_addr) & visibility)) {
      return false;
    }
#endif

    float t;
    float u;
    float v;
    if (pixel_displacement_intersect_displaced_surface(
            kg, P, dir, tmin, tmax, time, object, prim, true, verts, &u, &v, &t))
    {
      isect->t = t;
      isect->u = u;
      isect->v = v;
      isect->prim = prim;
      isect->object = object;
      isect->type = PRIMITIVE_MOTION_TRIANGLE;
      return true;
    }
    return false;
  }
#endif

  /* Ray-triangle intersection, unoptimized. */
  float t;
  float u;
  float v;
  if (ray_triangle_intersect(P, dir, tmin, tmax, verts[0], verts[1], verts[2], &u, &v, &t)) {
#ifdef __VISIBILITY_FLAG__
    /* Visibility flag test. we do it here under the assumption
     * that most triangles are culled by node flags.
     */
    if (kernel_data_fetch(prim_visibility, prim_addr) & visibility)
#endif
    {
      isect->t = t;
      isect->u = u;
      isect->v = v;
      isect->prim = prim;
      isect->object = object;
      isect->type = PRIMITIVE_MOTION_TRIANGLE;
      return true;
    }
  }
  return false;
}

/* Special ray intersection routines for local intersections. In that case we
 * only want to intersect with primitives in the same object, and if case of
 * multiple hits we pick a single random primitive as the intersection point.
 * Returns whether traversal should be stopped.
 */
#ifdef __BVH_LOCAL__
ccl_device_inline bool motion_triangle_intersect_local(KernelGlobals kg,
                                                       ccl_private LocalIntersection *local_isect,
                                                       const float3 P,
                                                       const float3 dir,
                                                       const float time,
                                                       const int object,
                                                       const int prim,
                                                       const float tmin,
                                                       const float tmax,
                                                       ccl_private uint *lcg_state,
                                                       const int max_hits)
{
  /* Get vertex locations for intersection. */
  float3 verts[3];
  motion_triangle_vertices(kg, object, prim, time, verts);
  /* Ray-triangle intersection, unoptimized. */
  float t;
  float u;
  float v;
#ifdef __KERNEL_METAL_PIXEL_DISPLACEMENT__
  bool use_pixel_displacement = false;
#endif
#ifdef __KERNEL_METAL_PIXEL_DISPLACEMENT__
  if (pixel_displacement_active(kg, prim)) {
    if (!pixel_displacement_intersect_displaced_surface(
            kg, P, dir, tmin, tmax, time, object, prim, true, verts, &u, &v, &t))
    {
      return false;
    }
    use_pixel_displacement = true;
  }
  else {
#endif
    if (!ray_triangle_intersect(P, dir, tmin, tmax, verts[0], verts[1], verts[2], &u, &v, &t)) {
      return false;
    }
#ifdef __KERNEL_METAL_PIXEL_DISPLACEMENT__
  }
#endif

  /* If no actual hit information is requested, just return here. */
  if (max_hits == 0) {
    return true;
  }

  const int hit_index = local_intersect_get_record_index(local_isect, t, lcg_state, max_hits);
  if (hit_index == -1) {
    return false;
  }

  /* Record intersection. */
  ccl_private Intersection *isect = &local_isect->hits[hit_index];
  isect->t = t;
  isect->u = u;
  isect->v = v;
  isect->prim = prim;
  isect->object = object;
  isect->type = PRIMITIVE_MOTION_TRIANGLE;

  /* Record geometric normal. */
#ifdef __KERNEL_METAL_PIXEL_DISPLACEMENT__
  if (use_pixel_displacement) {
    const uint object_flag = kernel_data_fetch(object_flag, object);
    local_isect->Ng[hit_index] = pixel_displacement_face_normal(verts, object_flag);
  }
  else {
#endif
    local_isect->Ng[hit_index] = normalize(cross(verts[1] - verts[0], verts[2] - verts[0]));
#ifdef __KERNEL_METAL_PIXEL_DISPLACEMENT__
  }
#endif

  return false;
}
#endif /* __BVH_LOCAL__ */

CCL_NAMESPACE_END
