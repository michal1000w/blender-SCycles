/* SPDX-FileCopyrightText: 2021-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Metal kernel entry points. */

/* NOTE: Must come prior to other includes. */
#include "kernel/device/metal/compat.h"
#include "kernel/device/metal/globals.h"

/* NOTE: Must come prior to the kernel.h. */
#include "kernel/device/metal/function_constants.h"

/* NOTE: Must come prior to the rest of the includes. */
#include "kernel/device/gpu/kernel.h"

/* The rest of the includes. */
#include "kernel/bvh/intersect_filter.h"
#include "kernel/geom/geom_intersect.h"
#include "kernel/geom/motion_triangle.h"
#include "kernel/geom/triangle.h"
#include "util/math_intersect.h"

/* MetalRT intersection handlers. */

#ifdef __KERNEL_METALRT__

/* Intersection return types. */

/* For a bounding box intersection function. */
struct BoundingBoxIntersectionResult {
  bool accept [[accept_intersection]];
  bool continue_search [[continue_search]];
  float distance [[distance]];
};

/* For a primitive intersection function. */
struct PrimitiveIntersectionResult {
  bool accept [[accept_intersection]];
  bool continue_search [[continue_search]];
};

enum { METALRT_HIT_TRIANGLE, METALRT_HIT_CURVE, METALRT_HIT_BOUNDING_BOX };

/* Hit functions. */

[[intersection(triangle,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data)]] PrimitiveIntersectionResult
__intersection__local_tri_single_hit(
    ray_data MetalKernelContext::MetalRTIntersectionLocalPayload_single_hit &payload [[payload]],
    uint primitive_id [[primitive_id]])
{
  PrimitiveIntersectionResult result;
  result.continue_search = true;
  result.accept = (payload.self_prim != primitive_id);
  return result;
}

[[intersection(triangle,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data,
               METALRT_TAGS METALRT_LIMITS)]] PrimitiveIntersectionResult
__intersection__local_tri_single_hit_mblur(
    ray_data MetalKernelContext::MetalRTIntersectionLocalPayload_single_hit &payload [[payload]],
#  if defined(__METALRT_MOTION__)
    uint object [[instance_id]],
#  endif
    uint primitive_id [[primitive_id]])
{
  PrimitiveIntersectionResult result;
  result.continue_search = true;
#  if defined(__METALRT_MOTION__)
  result.accept = (payload.self_prim != primitive_id) && (payload.self_object == object);
#  else
  result.accept = (payload.self_prim != primitive_id);
#  endif
  return result;
}

template<typename TReturn, uint intersection_type>
TReturn metalrt_local_hit(constant KernelParamsMetal &launch_params_metal,
                          ray_data MetalKernelContext::MetalRTIntersectionLocalPayload &payload,
                          const uint prim,
                          const float2 barycentrics,
                          const float ray_tmax)
{
  TReturn result;

#  ifdef __BVH_LOCAL__
  if (payload.self_prim == prim) {
    /* Only intersect with matching object and skip self-intersection. */
    result.accept = false;
    result.continue_search = true;
    return result;
  }

  const int max_hits = payload.max_hits;
  if (max_hits == 0) {
    /* Special case for when no hit information is requested, just report that something was hit.
     */
    result.accept = true;
    result.continue_search = false;
    return result;
  }

  /* Make a copy of the lcg_state in the private address space, allowing to use utility function
   * to find the hit index to write the intersection to. This function is used from both HW-RT
   * code-path and non-HW-RT, making it hard to deal with the address spaces in the function
   * signature. Hopefully, compiler is smart enough to eliminate this temporary copy. */
  uint lcg_state = payload.lcg_state;

  MetalKernelContext context(launch_params_metal);
  const int hit_index = context.local_intersect_get_record_index(
      &payload, ray_tmax, payload.has_lcg_state ? &lcg_state : nullptr, max_hits);

  payload.lcg_state = lcg_state;

  if (hit_index == -1) {
    result.accept = false;
    result.continue_search = true;
    return result;
  }

  payload.hits[hit_index].prim = prim;
  payload.hits[hit_index].t = ray_tmax;
  payload.hits[hit_index].u = barycentrics.x;
  payload.hits[hit_index].v = barycentrics.y;

  /* Continue tracing (without this the trace call would return after the first hit). */
  result.accept = false;
  result.continue_search = true;
#  endif
  return result;
}

[[intersection(triangle,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data)]] PrimitiveIntersectionResult
__intersection__local_tri(constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
                          ray_data MetalKernelContext::MetalRTIntersectionLocalPayload &payload
                          [[payload]],
                          uint primitive_id [[primitive_id]],
                          float2 barycentrics [[barycentric_coord]],
                          float ray_tmax [[distance]])
{
  /* instance_id, aka the user_id has been removed. If we take this function we optimized the
   * SSS for starting traversal from a primitive acceleration structure instead of the root of the
   * global AS. this means we will always be intersecting the correct object no need for the
   * user-id to check */
  return metalrt_local_hit<PrimitiveIntersectionResult, METALRT_HIT_TRIANGLE>(
      launch_params_metal, payload, primitive_id, barycentrics, ray_tmax);
}

[[intersection(triangle,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data,
               METALRT_TAGS METALRT_LIMITS)]] PrimitiveIntersectionResult
__intersection__local_tri_mblur(
    constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
    ray_data MetalKernelContext::MetalRTIntersectionLocalPayload &payload [[payload]],
    uint primitive_id [[primitive_id]],
#  if defined(__METALRT_MOTION__)
    uint object [[instance_id]],
#  endif
    float2 barycentrics [[barycentric_coord]],
    float ray_tmax [[distance]])
{
#  if defined(__METALRT_MOTION__)
  if (payload.self_object != object) {
    PrimitiveIntersectionResult result;
    result.continue_search = true;
    result.accept = false;
    return result;
  }
#  endif

  return metalrt_local_hit<PrimitiveIntersectionResult, METALRT_HIT_TRIANGLE>(
      launch_params_metal, payload, primitive_id, barycentrics, ray_tmax);
}

inline bool metalrt_curve_skip_end_cap(const int type, const float u)
{
  return ((u == 0.0f || u == 1.0f) && (type & PRIMITIVE_CURVE) != PRIMITIVE_CURVE_THICK_LINEAR);
}

inline Intersection get_intersection(constant KernelParamsMetal &launch_params_metal,
                                     const float t,
                                     const float2 uv,
                                     uint object,
                                     uint prim)
{
  Intersection isect;
  isect.t = t;
  isect.u = uv.x;
  isect.v = uv.y;
  isect.prim = prim;
  isect.object = object;
  isect.type = kernel_data_fetch(objects, object).primitive_type;

#  ifdef __HAIR__
  if (isect.type & PRIMITIVE_CURVE) {
    const KernelCurveSegment segment = kernel_data_fetch(curve_segments, prim);
    isect.type = segment.type;
    isect.prim = segment.prim;
  }
#  endif

  if (isect.type & PRIMITIVE_POINT) {
    isect.u = 0.0f;
    isect.v = 0.0f;
  }

  return isect;
}

template<uint intersection_type>
bool metalrt_shadow_all_hit(constant KernelParamsMetal &launch_params_metal,
                            ray_data MetalKernelContext::BVHShadowAllPayload &payload,
                            uint object,
                            uint prim,
                            const float2 uv,
                            const float t,
                            const ccl_private Ray *ray = nullptr)
{
#  if defined(__TRANSPARENT_SHADOWS__)
  MetalKernelContext context(launch_params_metal);

  KernelGlobals kg = nullptr;

  const Intersection isect = get_intersection(launch_params_metal, t, uv, object, prim);

#    ifdef __HAIR__
  if constexpr (intersection_type == METALRT_HIT_CURVE) {
    /* Filter out curve end-caps. */
    if (metalrt_curve_skip_end_cap(isect.type, isect.u)) {
      return true;
    }

    if ((isect.type & PRIMITIVE_CURVE) == PRIMITIVE_CURVE_RIBBON) {
      if (!context.curve_ribbon_accept(
              nullptr, isect.u, isect.t, ray, object, isect.prim, isect.type))
      {
        return true;
      }
    }
  }
#    endif /* __HAIR__ */

  constexpr uint enabled_primitive_types = (intersection_type == METALRT_HIT_CURVE) ?
                                               PRIMITIVE_CURVE :
                                               (PRIMITIVE_ALL & ~PRIMITIVE_CURVE);
  return context
      .bvh_shadow_all_anyhit_filter<MetalKernelContext::ISECT_TEST_ALL, enabled_primitive_types>(
          kg, payload.state, payload, payload.base.ray_self, payload.base.ray_visibility, isect);

#  else  /* __TRANSPARENT_SHADOWS__ */
  payload.throughput = zero_float3();
  return false;
#  endif /* __TRANSPARENT_SHADOWS__ */
}

[[intersection(triangle,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data,
               METALRT_TAGS METALRT_LIMITS)]] PrimitiveIntersectionResult
__intersection__tri_shadow_all(constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
                               ray_data MetalKernelContext::BVHShadowAllPayload &payload
                               [[payload]],
                               const unsigned int object [[instance_id]],
                               const unsigned int primitive_id [[primitive_id]],
                               const uint primitive_id_offset [[user_instance_id]],
                               const float2 uv [[barycentric_coord]],
                               const float t [[distance]])
{
  uint prim = primitive_id + primitive_id_offset;

  PrimitiveIntersectionResult result;
#  ifdef __KERNEL_METAL_PIXEL_DISPLACEMENT__
  MetalKernelContext context(launch_params_metal);
  if (context.pixel_displacement_active(nullptr, prim)) {
    result.accept = false;
    result.continue_search = true;
    return result;
  }
#  endif
  result.continue_search = metalrt_shadow_all_hit<METALRT_HIT_TRIANGLE>(
      launch_params_metal, payload, object, prim, uv, t);
  result.accept = !result.continue_search;
  return result;
}

[[intersection(triangle,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data,
               METALRT_TAGS METALRT_LIMITS)]] PrimitiveIntersectionResult
__intersection__volume_tri(constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
                           ray_data MetalKernelContext::MetalRTIntersectionShadowPayload &payload
                           [[payload]],
                           const unsigned int object [[instance_id]],
                           const unsigned int primitive_id [[primitive_id]],
                           const uint primitive_id_offset [[user_instance_id]])
{
  PrimitiveIntersectionResult result;
  result.continue_search = true;

  KernelGlobals kg = nullptr;
  MetalKernelContext context(launch_params_metal);

  uint prim = primitive_id + primitive_id_offset;

  if (context.bvh_volume_anyhit_triangle_filter(
          kg, object, prim, payload.self, payload.visibility))
  {
    result.accept = false;
    return result;
  }

  result.accept = true;
  return result;
}

template<typename TReturnType, uint intersection_type>
inline TReturnType metalrt_visibility_test(
    constant KernelParamsMetal &launch_params_metal,
    ray_data MetalKernelContext::MetalRTIntersectionPayload &payload,
    const uint object,
    uint prim,
    const float u,
    const float t = 0.0f,
    const ccl_private Ray *ray = nullptr)
{
  TReturnType result;

  if ((kernel_data_fetch(objects, object).visibility & payload.visibility) == 0) {
    result.accept = false;
    result.continue_search = true;
    return result;
  }

#  ifdef __HAIR__
  if constexpr (intersection_type == METALRT_HIT_CURVE) {
    const KernelCurveSegment segment = kernel_data_fetch(curve_segments, prim);
    int type = segment.type;
    prim = segment.prim;

    /* Filter out curve end-caps. */
    if (metalrt_curve_skip_end_cap(type, u)) {
      result.accept = false;
      result.continue_search = true;
      return result;
    }

    if ((type & PRIMITIVE_CURVE) == PRIMITIVE_CURVE_RIBBON) {
      MetalKernelContext context(launch_params_metal);
      if (!context.curve_ribbon_accept(nullptr, u, t, ray, object, prim, type)) {
        result.accept = false;
        result.continue_search = true;
        return result;
      }
    }
  }
#  endif

  bool two_sided_point = false;
#  ifdef __POINTCLOUD__
  if constexpr (intersection_type == METALRT_HIT_BOUNDING_BOX) {
    MetalKernelContext context(launch_params_metal);
    two_sided_point = kernel_data.integrator.coherent_specular_enabled &&
                      context.point_coherent_glass_two_sided(
                          true, kernel_data_fetch(object_flag, object),
                          kernel_data_fetch(objects, object).primitive_type);
  }
#  endif
  if (payload.self_object == object && payload.self_prim == prim && !two_sided_point) {
    result.accept = false;
    result.continue_search = true;
    return result;
  }
  result.accept = true;
  result.continue_search = true;
  return result;
}

template<typename TReturnType, uint intersection_type>
inline TReturnType metalrt_visibility_test_shadow(
    constant KernelParamsMetal &launch_params_metal,
    ray_data MetalKernelContext::MetalRTIntersectionShadowPayload &payload,
    const uint object,
    uint prim,
    const float u,
    const float t = 0.0f,
    const ccl_private Ray *ray = nullptr)
{
  TReturnType result;

  if ((kernel_data_fetch(objects, object).visibility & payload.visibility) == 0) {
    result.accept = false;
    return result;
  }

#  ifdef __HAIR__
  if constexpr (intersection_type == METALRT_HIT_CURVE) {
    const KernelCurveSegment segment = kernel_data_fetch(curve_segments, prim);
    int type = segment.type;
    prim = segment.prim;

    /* Filter out curve end-caps. */
    if (metalrt_curve_skip_end_cap(type, u)) {
      result.accept = false;
      result.continue_search = true;
      return result;
    }

    if ((type & PRIMITIVE_CURVE) == PRIMITIVE_CURVE_RIBBON) {
      MetalKernelContext context(launch_params_metal);
      if (!context.curve_ribbon_accept(nullptr, u, t, ray, object, prim, type)) {
        result.accept = false;
        result.continue_search = true;
        return result;
      }
    }
  }
#  endif

  MetalKernelContext context(launch_params_metal);

  /* Shadow ray early termination. */
#  ifdef __SHADOW_LINKING__
  if (context.intersection_skip_shadow_link(nullptr, payload.self, object)) {
    result.accept = false;
    result.continue_search = true;
    return result;
  }
#  endif

  const int type = (intersection_type == METALRT_HIT_BOUNDING_BOX) ?
                       kernel_data_fetch(objects, object).primitive_type : 0;
  if (context.intersection_skip_self_shadow_coherent_point(
          nullptr, payload.self, object, prim, type)) {
    result.accept = false;
    result.continue_search = true;
    return result;
  }
  else {
    result.accept = true;
    result.continue_search = false;
    return result;
  }

  result.accept = true;
  result.continue_search = true;
  return result;
}

[[intersection(triangle,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data,
               METALRT_TAGS METALRT_LIMITS)]] PrimitiveIntersectionResult
__intersection__tri(constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
                    ray_data MetalKernelContext::MetalRTIntersectionPayload &payload [[payload]],
                    const unsigned int object [[instance_id]],
                    const uint primitive_id_offset [[user_instance_id]],
                    const unsigned int primitive_id [[primitive_id]])
{
  PrimitiveIntersectionResult result;
  result.continue_search = true;

#  ifdef __KERNEL_METAL_PIXEL_DISPLACEMENT__
  MetalKernelContext context(launch_params_metal);
  if (context.pixel_displacement_active(nullptr, primitive_id + primitive_id_offset)) {
    result.accept = false;
    return result;
  }
#  endif

  if ((kernel_data_fetch(objects, object).visibility & payload.visibility) == 0) {
    result.accept = false;
    return result;
  }

  result.accept = (payload.self_object != object ||
                   payload.self_prim != (primitive_id + primitive_id_offset));
  return result;
}

[[intersection(triangle,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data,
               METALRT_TAGS METALRT_LIMITS)]] PrimitiveIntersectionResult
__intersection__tri_shadow(constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
                           ray_data MetalKernelContext::MetalRTIntersectionShadowPayload &payload
                           [[payload]],
                           const unsigned int object [[instance_id]],
                           const uint primitive_id_offset [[user_instance_id]],
                           const unsigned int primitive_id [[primitive_id]])
{
  uint prim = primitive_id + primitive_id_offset;
#  ifdef __KERNEL_METAL_PIXEL_DISPLACEMENT__
  MetalKernelContext context(launch_params_metal);
  if (context.pixel_displacement_active(nullptr, prim)) {
    PrimitiveIntersectionResult result;
    result.accept = false;
    result.continue_search = true;
    return result;
  }
#  endif
  PrimitiveIntersectionResult result =
      metalrt_visibility_test_shadow<PrimitiveIntersectionResult, METALRT_HIT_TRIANGLE>(
          launch_params_metal, payload, object, prim, 0.0f);
  return result;
}

/* Intersect a displaced triangle represented by a conservative MetalRT bounding box. The
 * function reports the micromesh distance to MetalRT, so closest-hit ordering remains handled by
 * hardware traversal. Base-triangle intersection functions above reject the same primitive to
 * avoid exposing the undisplaced plane. */
ccl_device_inline bool metalrt_pixel_displacement_intersect(
    constant KernelParamsMetal &launch_params_metal,
    const uint object,
    const uint prim,
    const float3 ray_origin,
    const float3 ray_direction,
    const float ray_tmin,
    const float ray_tmax,
    const float time,
    thread float *r_u,
    thread float *r_v,
    thread float *r_t)
{
  KernelGlobals kg = nullptr;
  MetalKernelContext context(launch_params_metal);
  float3 verts[3];
  bool motion = false;
#  ifdef __OBJECT_MOTION__
  motion = (kernel_data_fetch(objects, object).primitive_type == PRIMITIVE_MOTION_TRIANGLE);
  if (motion) {
    context.motion_triangle_vertices(kg, object, prim, time, verts);
  }
  else
#  endif
  {
    context.triangle_vertices(kg, object, prim, verts);
  }

#  ifdef __KERNEL_METAL_PIXEL_DISPLACEMENT__
  if (context.pixel_displacement_active(kg, prim)) {
    return context.pixel_displacement_intersect_cached_surface(kg,
                                                               object,
                                                               prim,
                                                               motion,
                                                               verts,
                                                               ray_origin,
                                                               ray_direction,
                                                               ray_tmin,
                                                               ray_tmax,
                                                               r_u,
                                                               r_v,
                                                               r_t);
  }
#  endif

  return context.ray_triangle_intersect(
      ray_origin, ray_direction, ray_tmin, ray_tmax, verts[0], verts[1], verts[2], r_u, r_v, r_t);
}

#  ifdef __BVH_LOCAL__
[[intersection(bounding_box,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data)]] BoundingBoxIntersectionResult
__intersection__local_pixel_displacement_single_hit(
    constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
    ray_data MetalKernelContext::MetalRTIntersectionLocalPayload_single_hit &payload [[payload]],
    const uint primitive_id [[primitive_id]],
    const float3 ray_origin [[origin]],
    const float3 ray_direction [[direction]],
    const float ray_tmin [[min_distance]],
    const float ray_tmax [[max_distance]])
{
  BoundingBoxIntersectionResult result;
  result.accept = false;
  result.continue_search = true;
  result.distance = ray_tmax;

  if (payload.self_prim == primitive_id) {
    return result;
  }

  const uint prim = primitive_id + payload.primitive_id_offset;
  float u, v, t;
  if (metalrt_pixel_displacement_intersect(launch_params_metal,
                                           payload.object,
                                           prim,
                                           ray_origin,
                                           ray_direction,
                                           ray_tmin,
                                           ray_tmax,
                                           0.0f,
                                           &u,
                                           &v,
                                           &t))
  {
    result.accept = true;
    result.distance = t;
    if (t < payload.pixel_displacement_t) {
      payload.pixel_displacement_t = t;
      payload.pixel_displacement_u = u;
      payload.pixel_displacement_v = v;
    }
  }
  return result;
}

[[intersection(bounding_box,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data)]] BoundingBoxIntersectionResult
__intersection__local_pixel_displacement(
    constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
    ray_data MetalKernelContext::MetalRTIntersectionLocalPayload &payload [[payload]],
    const uint primitive_id [[primitive_id]],
    const float3 ray_origin [[origin]],
    const float3 ray_direction [[direction]],
    const float ray_tmin [[min_distance]],
    const float ray_tmax [[max_distance]])
{
  BoundingBoxIntersectionResult result;
  result.accept = false;
  result.continue_search = true;
  result.distance = ray_tmax;

  const uint prim = primitive_id + payload.primitive_id_offset;
  float u, v, t;
  if (metalrt_pixel_displacement_intersect(launch_params_metal,
                                           payload.object,
                                           prim,
                                           ray_origin,
                                           ray_direction,
                                           ray_tmin,
                                           ray_tmax,
                                           0.0f,
                                           &u,
                                           &v,
                                           &t))
  {
    result = metalrt_local_hit<BoundingBoxIntersectionResult, METALRT_HIT_BOUNDING_BOX>(
        launch_params_metal, payload, primitive_id, float2(u, v), t);
    if (result.accept) {
      result.distance = t;
    }
  }
  return result;
}
#  endif /* __BVH_LOCAL__ */

[[intersection(bounding_box,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data,
               METALRT_TAGS METALRT_LIMITS)]] BoundingBoxIntersectionResult
__intersection__pixel_displacement(constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
                                   ray_data MetalKernelContext::MetalRTIntersectionPayload &payload
                                   [[payload]],
                                   const uint object [[instance_id]],
                                   const uint primitive_id [[primitive_id]],
                                   const uint primitive_id_offset [[user_instance_id]],
                                   const float3 ray_origin [[origin]],
                                   const float3 ray_direction [[direction]],
#  if defined(__METALRT_MOTION__)
                                   const float time [[time]],
#  endif
                                   const float ray_tmin [[min_distance]],
                                   const float ray_tmax [[max_distance]])
{
  const uint prim = primitive_id + primitive_id_offset;
  BoundingBoxIntersectionResult result;
  result.accept = false;
  result.continue_search = true;
  result.distance = ray_tmax;

#  ifndef __METALRT_MOTION__
  const float time = 0.0f;
#  endif
  float u, v, t;
  if (metalrt_pixel_displacement_intersect(launch_params_metal,
                                           object,
                                           prim,
                                           ray_origin,
                                           ray_direction,
                                           ray_tmin,
                                           ray_tmax,
                                           time,
                                           &u,
                                           &v,
                                           &t))
  {
    result = metalrt_visibility_test<BoundingBoxIntersectionResult, METALRT_HIT_BOUNDING_BOX>(
        launch_params_metal, payload, object, prim, u);
    if (result.accept) {
      result.distance = t;
#  ifdef __KERNEL_METAL_PIXEL_DISPLACEMENT__
      MetalKernelContext context(launch_params_metal);
      if (context.pixel_displacement_active(nullptr, prim) && t < payload.pixel_displacement_t) {
        payload.pixel_displacement_t = t;
        payload.pixel_displacement_u = u;
        payload.pixel_displacement_v = v;
      }
#  endif
    }
  }
  return result;
}

[[intersection(bounding_box,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data,
               METALRT_TAGS METALRT_LIMITS)]] BoundingBoxIntersectionResult
__intersection__pixel_displacement_shadow(
    constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
    ray_data MetalKernelContext::MetalRTIntersectionShadowPayload &payload [[payload]],
    const uint object [[instance_id]],
    const uint primitive_id [[primitive_id]],
    const uint primitive_id_offset [[user_instance_id]],
    const float3 ray_origin [[origin]],
    const float3 ray_direction [[direction]],
#  if defined(__METALRT_MOTION__)
    const float time [[time]],
#  endif
    const float ray_tmin [[min_distance]],
    const float ray_tmax [[max_distance]])
{
  const uint prim = primitive_id + primitive_id_offset;
  BoundingBoxIntersectionResult result;
  result.accept = false;
  result.continue_search = true;
  result.distance = ray_tmax;

#  ifndef __METALRT_MOTION__
  const float time = 0.0f;
#  endif
  float u, v, t;
  if (metalrt_pixel_displacement_intersect(launch_params_metal,
                                           object,
                                           prim,
                                           ray_origin,
                                           ray_direction,
                                           ray_tmin,
                                           ray_tmax,
                                           time,
                                           &u,
                                           &v,
                                           &t))
  {
    result =
        metalrt_visibility_test_shadow<BoundingBoxIntersectionResult, METALRT_HIT_BOUNDING_BOX>(
            launch_params_metal, payload, object, prim, u);
    if (result.accept) {
      result.distance = t;
    }
  }
  return result;
}

[[intersection(bounding_box,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data,
               METALRT_TAGS METALRT_LIMITS)]] BoundingBoxIntersectionResult
__intersection__pixel_displacement_shadow_all(
    constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
    ray_data MetalKernelContext::BVHShadowAllPayload &payload [[payload]],
    const uint object [[instance_id]],
    const uint primitive_id [[primitive_id]],
    const uint primitive_id_offset [[user_instance_id]],
    const float3 ray_origin [[origin]],
    const float3 ray_direction [[direction]],
#  if defined(__METALRT_MOTION__)
    const float time [[time]],
#  endif
    const float ray_tmin [[min_distance]],
    const float ray_tmax [[max_distance]])
{
  const uint prim = primitive_id + primitive_id_offset;
  BoundingBoxIntersectionResult result;
  result.accept = false;
  result.continue_search = true;
  result.distance = ray_tmax;

#  ifndef __METALRT_MOTION__
  const float time = 0.0f;
#  endif
  float u, v, t;
  if (metalrt_pixel_displacement_intersect(launch_params_metal,
                                           object,
                                           prim,
                                           ray_origin,
                                           ray_direction,
                                           ray_tmin,
                                           ray_tmax,
                                           time,
                                           &u,
                                           &v,
                                           &t))
  {
    result.continue_search = metalrt_shadow_all_hit<METALRT_HIT_BOUNDING_BOX>(
        launch_params_metal, payload, object, prim, float2(u, v), t);
    result.accept = !result.continue_search;
    if (result.accept) {
      result.distance = t;
    }
  }
  return result;
}

/* Primitive intersection functions. */

[[intersection(curve,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data,
               METALRT_TAGS METALRT_LIMITS)]] PrimitiveIntersectionResult
__intersection__curve(constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
                      ray_data MetalKernelContext::MetalRTIntersectionPayload &payload [[payload]],
                      const uint object [[instance_id]],
                      const uint primitive_id [[primitive_id]],
                      const uint primitive_id_offset [[user_instance_id]],
                      float distance [[distance]],
                      const float3 ray_P [[origin]],
                      const float3 ray_D [[direction]],
                      float u [[curve_parameter]],
                      const float ray_tmin [[min_distance]],
                      const float ray_tmax [[max_distance]]
#  if defined(__METALRT_MOTION__)
                      ,
                      const float time [[time]]
#  endif
)
{
  uint prim = primitive_id + primitive_id_offset;

  Ray ray;
  ray.P = ray_P;
  ray.D = ray_D;
#  if defined(__METALRT_MOTION__)
  ray.time = time;
#  endif

  PrimitiveIntersectionResult result =
      metalrt_visibility_test<PrimitiveIntersectionResult, METALRT_HIT_CURVE>(
          launch_params_metal, payload, object, prim, u, distance, &ray);

  return result;
}

[[intersection(curve,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data,
               METALRT_TAGS METALRT_LIMITS)]] PrimitiveIntersectionResult
__intersection__curve_shadow(constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
                             ray_data MetalKernelContext::MetalRTIntersectionShadowPayload &payload
                             [[payload]],
                             const uint object [[instance_id]],
                             const uint primitive_id [[primitive_id]],
                             const uint primitive_id_offset [[user_instance_id]],
                             float distance [[distance]],
                             const float3 ray_P [[origin]],
                             const float3 ray_D [[direction]],
                             float u [[curve_parameter]],
                             const float ray_tmin [[min_distance]],
                             const float ray_tmax [[max_distance]]
#  if defined(__METALRT_MOTION__)
                             ,
                             const float time [[time]]
#  endif
)
{
  uint prim = primitive_id + primitive_id_offset;

  Ray ray;
  ray.P = ray_P;
  ray.D = ray_D;
#  if defined(__METALRT_MOTION__)
  ray.time = time;
#  endif

  PrimitiveIntersectionResult result =
      metalrt_visibility_test_shadow<PrimitiveIntersectionResult, METALRT_HIT_CURVE>(
          launch_params_metal, payload, object, prim, u, distance, &ray);

  return result;
}

[[intersection(curve,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data,
               METALRT_TAGS METALRT_LIMITS)]] PrimitiveIntersectionResult
__intersection__curve_shadow_all(constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
                                 ray_data MetalKernelContext::BVHShadowAllPayload &payload
                                 [[payload]],
                                 const uint object [[instance_id]],
                                 const uint primitive_id [[primitive_id]],
                                 const uint primitive_id_offset [[user_instance_id]],
                                 const float3 ray_P [[origin]],
                                 const float3 ray_D [[direction]],
#  if defined(__METALRT_MOTION__)
                                 const float time [[time]],
#  endif
                                 float u [[curve_parameter]],
                                 float t [[distance]])
{
  uint prim = primitive_id + primitive_id_offset;

  PrimitiveIntersectionResult result;

  Ray ray;
  ray.P = ray_P;
  ray.D = ray_D;
#  if defined(__METALRT_MOTION__)
  /* TODO(sergey): The time is not really needed.
   * Only ray direction and origin are needed in curve_ribbon_accept(), so there might be a room
   * for cleanup here. */
  ray.time = time;
#  endif

  result.continue_search = metalrt_shadow_all_hit<METALRT_HIT_CURVE>(
      launch_params_metal, payload, object, prim, float2(u, 0), t, &ray);
  result.accept = !result.continue_search;

  return result;
}

#  ifdef __POINTCLOUD__
ccl_device_inline void metalrt_intersection_point_shadow_all(
    constant KernelParamsMetal &launch_params_metal,
    ray_data MetalKernelContext::BVHShadowAllPayload &payload,
    const uint object,
    const uint prim,
    const uint type,
    const float3 ray_P,
    const float3 ray_D,
    float time,
    const float ray_tmin,
    const float ray_tmax,
    thread BoundingBoxIntersectionResult &result)
{
  Intersection isect;
  isect.t = ray_tmax;

  MetalKernelContext context(launch_params_metal);
  if (context.point_intersect(
          nullptr, &isect, ray_P, ray_D, ray_tmin, isect.t, object, prim, time, type))
  {
    result.continue_search = metalrt_shadow_all_hit<METALRT_HIT_BOUNDING_BOX>(
        launch_params_metal, payload, object, prim, float2(isect.u, isect.v), isect.t);
    result.accept = !result.continue_search;

    if (result.accept) {
      result.distance = isect.t;
    }
  }
}

[[intersection(bounding_box,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data,
               METALRT_TAGS METALRT_LIMITS)]] BoundingBoxIntersectionResult
__intersection__point(constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
                      ray_data MetalKernelContext::MetalRTIntersectionPayload &payload [[payload]],
                      const uint object [[instance_id]],
                      const uint primitive_id [[primitive_id]],
                      const uint primitive_id_offset [[user_instance_id]],
                      const float3 ray_origin [[origin]],
                      const float3 ray_direction [[direction]],
#    if defined(__METALRT_MOTION__)
                      const float time [[time]],
#    endif
                      const float ray_tmin [[min_distance]],
                      const float ray_tmax [[max_distance]])
{
  const uint prim = primitive_id + primitive_id_offset;
  const int type = kernel_data_fetch(objects, object).primitive_type;

  BoundingBoxIntersectionResult result;
  result.accept = false;
  result.continue_search = true;
  result.distance = ray_tmax;

  Intersection isect;
  isect.t = ray_tmax;

#    ifndef __METALRT_MOTION__
  const float time = 0.0f;
#    endif

  MetalKernelContext context(launch_params_metal);
  if (context.point_intersect(
          nullptr, &isect, ray_origin, ray_direction, ray_tmin, isect.t, object, prim, time, type))
  {
    result = metalrt_visibility_test<BoundingBoxIntersectionResult, METALRT_HIT_BOUNDING_BOX>(
        launch_params_metal, payload, object, prim, isect.u);
    if (result.accept) {
      result.distance = isect.t;
    }
  }
  return result;
}

#  endif /* __POINTCLOUD__ */

[[intersection(bounding_box,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data,
               METALRT_TAGS METALRT_LIMITS)]] BoundingBoxIntersectionResult
__intersection__point_shadow(constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
                             ray_data MetalKernelContext::MetalRTIntersectionShadowPayload &payload
                             [[payload]],
                             const uint object [[instance_id]],
                             const uint primitive_id [[primitive_id]],
                             const uint primitive_id_offset [[user_instance_id]],
                             const float3 ray_origin [[origin]],
                             const float3 ray_direction [[direction]],
#  if defined(__METALRT_MOTION__)
                             const float time [[time]],
#  endif
                             const float ray_tmin [[min_distance]],
                             const float ray_tmax [[max_distance]])
{
  const uint prim = primitive_id + primitive_id_offset;
  const int type = kernel_data_fetch(objects, object).primitive_type;

  BoundingBoxIntersectionResult result;
  result.accept = false;
  result.continue_search = true;
  result.distance = ray_tmax;

#  ifdef __POINTCLOUD__

  Intersection isect;
  isect.t = ray_tmax;

#    ifndef __METALRT_MOTION__
  const float time = 0.0f;
#    endif

  MetalKernelContext context(launch_params_metal);
  if (context.point_intersect(
          nullptr, &isect, ray_origin, ray_direction, ray_tmin, isect.t, object, prim, time, type))
  {
    result =
        metalrt_visibility_test_shadow<BoundingBoxIntersectionResult, METALRT_HIT_BOUNDING_BOX>(
            launch_params_metal, payload, object, prim, isect.u);
    if (result.accept) {
      result.distance = isect.t;
    }
  }

#  endif /* __POINTCLOUD__ */

  return result;
}

[[intersection(bounding_box,
               metal::raytracing::triangle_data,
               metal::raytracing::curve_data,
               METALRT_TAGS METALRT_LIMITS)]] BoundingBoxIntersectionResult
__intersection__point_shadow_all(constant KernelParamsMetal &launch_params_metal [[buffer(1)]],
                                 ray_data MetalKernelContext::BVHShadowAllPayload &payload
                                 [[payload]],
                                 const uint object [[instance_id]],
                                 const uint primitive_id [[primitive_id]],
                                 const uint primitive_id_offset [[user_instance_id]],
                                 const float3 ray_origin [[origin]],
                                 const float3 ray_direction [[direction]],
#  if defined(__METALRT_MOTION__)
                                 const float time [[time]],
#  endif
                                 const float ray_tmin [[min_distance]],
                                 const float ray_tmax [[max_distance]])
{
  const uint prim = primitive_id + primitive_id_offset;
  const int type = kernel_data_fetch(objects, object).primitive_type;

  BoundingBoxIntersectionResult result;
  result.accept = false;
  result.continue_search = true;
  result.distance = ray_tmax;

#  ifdef __POINTCLOUD__

  metalrt_intersection_point_shadow_all(launch_params_metal,
                                        payload,
                                        object,
                                        prim,
                                        type,
                                        ray_origin,
                                        ray_direction,
#    if defined(__METALRT_MOTION__)
                                        time,
#    else
                                        0.0f,
#    endif
                                        ray_tmin,
                                        ray_tmax,
                                        result);

#  endif /* __POINTCLOUD__ */

  return result;
}

#endif /* __KERNEL_METALRT__ */

#ifdef __KERNEL_METAL_VISIBLE_SHADING__
/* --------------------------------------------------------------------
 * Separately compiled shading functions.
 *
 * The host compiles these once to GPU binaries and links them into every pipeline. Kernels call
 * them through the visible function tables in MetalAncillaries, so the large shader interpreter
 * and closure dispatch are no longer optimized again inside each kernel. */

static_assert(sizeof(Spectrum) == sizeof(float3), "Metal visible shading functions use float3");

template<typename T> T metal_visible_state(int state);
template<> int metal_visible_state<int>(int state)
{
  return state;
}
template<> IntegratorBakeState metal_visible_state<IntegratorBakeState>(int /*state*/)
{
  return IntegratorBakeState();
}

#  define CCL_METAL_SVM_VISIBLE_FUNCTION(index, mask, shader_type, state_type) \
    [[visible]] void cycles_metal_svm_##index(constant void *launch_params, \
                                              constant void *ancillaries, \
                                              int state, \
                                              thread void *sd, \
                                              device float *render_buffer, \
                                              uint path_visibility, \
                                              uint path_flag) \
    { \
      MetalKernelContext context(*(constant KernelParamsMetal *)launch_params, \
                                 (constant MetalAncillaries *)ancillaries); \
      context.svm_eval_nodes_impl<mask, shader_type, MetalKernelContext::SVM_EVAL_CORE>( \
          nullptr, \
          metal_visible_state<state_type>(state), \
          (thread ShaderData *)sd, \
          render_buffer, \
          PathRayVisibility(path_visibility), \
          path_flag); \
    }
CCL_METAL_SVM_FUNCTIONS(CCL_METAL_SVM_VISIBLE_FUNCTION)
#  undef CCL_METAL_SVM_VISIBLE_FUNCTION

/* Shader nodes that do not depend on the interpreter instantiation, see SVM_SHARED_CASE. */
[[visible]] int cycles_metal_svm_node(constant void *launch_params,
                                      constant void *ancillaries,
                                      uint node_type,
                                      int offset,
                                      thread void *sd,
                                      thread float *stack,
                                      uint path_visibility,
                                      uint path_flag)
{
  MetalKernelContext context(*(constant KernelParamsMetal *)launch_params,
                             (constant MetalAncillaries *)ancillaries);
  return context.svm_eval_nodes_impl<KERNEL_FEATURE_NODE_MASK_SURFACE,
                                     SHADER_TYPE_SURFACE,
                                     MetalKernelContext::SVM_EVAL_SHARED_NODE>(
      nullptr,
      0,
      (thread ShaderData *)sd,
      nullptr,
      PathRayVisibility(path_visibility),
      path_flag,
      node_type,
      offset,
      stack);
}

/* Surface closure node shared by every interpreter instantiation that creates BSDFs. */
[[visible]] int cycles_metal_svm_closure(constant void *launch_params,
                                         constant void *ancillaries,
                                         thread void *sd,
                                         thread float *stack,
                                         float3 closure_weight,
                                         int offset,
                                         uint path_visibility,
                                         uint path_flag)
{
  MetalKernelContext context(*(constant KernelParamsMetal *)launch_params,
                             (constant MetalAncillaries *)ancillaries);
  using SVMNodeClosureBsdf = MetalKernelContext::SVMNodeClosureBsdf;
  const ccl_global SVMNodeClosureBsdf &bsdf_node =
      context.svm_node_get<SVMNodeClosureBsdf>(nullptr, &offset);
  return context.svm_node_closure_bsdf<KERNEL_FEATURE_NODE_MASK_SURFACE, SHADER_TYPE_SURFACE>(
      nullptr,
      (thread ShaderData *)sd,
      stack,
      closure_weight,
      bsdf_node,
      PathRayVisibility(path_visibility),
      path_flag,
      offset);
}

[[visible]] float3 cycles_metal_bsdf_eval(constant void *launch_params,
                                          constant void *ancillaries,
                                          thread void *sd,
                                          thread const void *sc,
                                          float3 wo,
                                          thread float *pdf)
{
  MetalKernelContext context(*(constant KernelParamsMetal *)launch_params,
                             (constant MetalAncillaries *)ancillaries);
  return context.bsdf_eval_impl(
      nullptr, (thread ShaderData *)sd, (thread const ShaderClosure *)sc, wo, pdf);
}

[[visible]] float3 cycles_metal_bsdf_eval_delta(constant void *launch_params,
                                                constant void *ancillaries,
                                                thread void *sd,
                                                thread const void *sc,
                                                float3 wo,
                                                thread float *pdf)
{
  MetalKernelContext context(*(constant KernelParamsMetal *)launch_params,
                             (constant MetalAncillaries *)ancillaries);
  return context.bsdf_eval_delta_impl(
      nullptr, (thread ShaderData *)sd, (thread const ShaderClosure *)sc, wo, pdf);
}

[[visible]] void cycles_metal_polarization_surface_transport(constant void *launch_params,
                                                             constant void *ancillaries,
                                                             thread void *result,
                                                             thread void *sd,
                                                             float3 wo,
                                                             thread const void *incoming_state,
                                                             bool adjoint,
                                                             thread const void *sampled_closure,
                                                             float3 native_total,
                                                             bool sampled_delta,
                                                             uint light_shader_flags,
                                                             bool reciprocal_forward)
{
  MetalKernelContext context(*(constant KernelParamsMetal *)launch_params,
                             (constant MetalAncillaries *)ancillaries);
  using PolarizationSpectrumState = MetalKernelContext::PolarizationSpectrumState;
  *(thread PolarizationSpectrumState *)result = context.polarization_surface_transport_impl(
      nullptr,
      (thread ShaderData *)sd,
      wo,
      *(thread const PolarizationSpectrumState *)incoming_state,
      adjoint,
      (thread const ShaderClosure *)sampled_closure,
      native_total,
      sampled_delta,
      light_shader_flags,
      reciprocal_forward);
}

[[visible]] bool cycles_metal_diffraction_power_column(constant void *launch_params,
                                                       constant void *ancillaries,
                                                       thread const void *data,
                                                       int handle,
                                                       float3 incident,
                                                       bool incoming_substrate,
                                                       float upper_index,
                                                       float lower_index,
                                                       float wavelength,
                                                       float pitch,
                                                       thread void *output_view,
                                                       thread int *output_incoming_order,
                                                       thread float *powers)
{
  MetalKernelContext context(*(constant KernelParamsMetal *)launch_params,
                             (constant MetalAncillaries *)ancillaries);
  using DiffractionSceneData = MetalKernelContext::DiffractionSceneData;
  using DiffractionCacheCellView = MetalKernelContext::DiffractionCacheCellView;
  return context.diffraction_data_power_column_impl<DIFFRACTION_MAX_CHANNELS>(
      (thread const DiffractionSceneData *)data,
      handle,
      incident,
      incoming_substrate,
      upper_index,
      lower_index,
      wavelength,
      pitch,
      (thread DiffractionCacheCellView *)output_view,
      output_incoming_order,
      powers);
}

#  ifdef __MNEE__
[[visible]] int cycles_metal_mnee_sample(constant void *launch_params,
                                         constant void *ancillaries,
                                         int state,
                                         thread void *sd,
                                         thread void *sd_mnee,
                                         thread const void *rng_state,
                                         thread void *ls,
                                         thread float3 *throughput,
                                         thread float3 *r_receiver_wo,
                                         thread int *r_vertex_count,
                                         thread float3 *r_light_wo,
                                         bool consider_all_refractive,
                                         thread float *r_light_distance,
                                         float wavelength_rand_override,
                                         bool volume_endpoint,
                                         thread float3 *r_vertices,
                                         thread void *r_polarization)
{
  MetalKernelContext context(*(constant KernelParamsMetal *)launch_params,
                             (constant MetalAncillaries *)ancillaries);
  using RNGState = MetalKernelContext::RNGState;
  using LightSample = MetalKernelContext::LightSample;
  using PolarizationMueller = MetalKernelContext::PolarizationMueller;
  return int(context.kernel_path_mnee_sample_impl(nullptr,
                                                  state,
                                                  (thread ShaderData *)sd,
                                                  (thread ShaderData *)sd_mnee,
                                                  (thread const RNGState *)rng_state,
                                                  (thread LightSample *)ls,
                                                  throughput,
                                                  r_receiver_wo,
                                                  *r_vertex_count,
                                                  r_light_wo,
                                                  consider_all_refractive,
                                                  r_light_distance,
                                                  wavelength_rand_override,
                                                  volume_endpoint,
                                                  r_vertices,
                                                  (thread PolarizationMueller *)r_polarization));
}
#  endif

#  ifdef __KERNEL_METAL_PIXEL_DISPLACEMENT_SHADE__
[[visible]] void cycles_metal_pixel_displacement_shader_setup(constant void *launch_params,
                                                              constant void *ancillaries,
                                                              thread void *sd,
                                                              float time,
                                                              bool motion,
                                                              thread const float3 *verts)
{
  MetalKernelContext context(*(constant KernelParamsMetal *)launch_params,
                             (constant MetalAncillaries *)ancillaries);
  context.pixel_displacement_shader_setup_impl(
      nullptr, (thread ShaderData *)sd, time, motion, verts);
}
#  endif

[[visible]] int cycles_metal_bsdf_sample(constant void *launch_params,
                                         constant void *ancillaries,
                                         thread void *sd,
                                         thread const void *sc,
                                         float3 rand,
                                         thread float3 *eval,
                                         thread float3 *wo,
                                         thread float *pdf,
                                         thread float2 *sampled_roughness,
                                         thread float *eta)
{
  MetalKernelContext context(*(constant KernelParamsMetal *)launch_params,
                             (constant MetalAncillaries *)ancillaries);
  return context.bsdf_sample_impl(nullptr,
                                  (thread ShaderData *)sd,
                                  (thread const ShaderClosure *)sc,
                                  rand,
                                  eval,
                                  wo,
                                  pdf,
                                  sampled_roughness,
                                  eta);
}
#  define CCL_METAL_SURFACE_STAGE_BEGIN(name) \
    [[visible]] int cycles_metal_surface_##name(constant void *launch_params, \
                                                constant void *ancillaries, \
                                                int state, \
                                                thread void *sd_ptr, \
                                                thread const void *rng_ptr, \
                                                device float *render_buffer, \
                                                thread float3 *result, \
                                                thread float3 *secondary_result) \
    { \
      MetalKernelContext context(*(constant KernelParamsMetal *)launch_params, \
                                 (constant MetalAncillaries *)ancillaries); \
      thread ShaderData *sd = (thread ShaderData *)sd_ptr; \
      const thread MetalKernelContext::RNGState *rng_state = \
          (const thread MetalKernelContext::RNGState *)rng_ptr; \
      (void)sd; \
      (void)rng_state; \
      (void)render_buffer; \
      (void)result; \
      (void)secondary_result;
#  define CCL_METAL_SURFACE_STAGE_END }

/* Table order must match MetalSurfaceStage. */
CCL_METAL_SURFACE_STAGE_BEGIN(0)
return int(context.integrate_surface_direct_light<KERNEL_FEATURE_NODE_MASK_SURFACE &
                                                  ~KERNEL_FEATURE_NODE_RAYTRACE>(
    nullptr, state, sd, rng_state));
CCL_METAL_SURFACE_STAGE_END

CCL_METAL_SURFACE_STAGE_BEGIN(1)
return int(context.integrate_surface_bidirectional(nullptr, state, sd, rng_state));
CCL_METAL_SURFACE_STAGE_END

CCL_METAL_SURFACE_STAGE_BEGIN(2)
return context.integrate_surface_bsdf_bssrdf_bounce(nullptr, state, sd, rng_state);
CCL_METAL_SURFACE_STAGE_END

CCL_METAL_SURFACE_STAGE_BEGIN(3)
*result = context.photon_mapping_gather(nullptr, state, sd, render_buffer);
return 0;
CCL_METAL_SURFACE_STAGE_END

CCL_METAL_SURFACE_STAGE_BEGIN(4)
*result = context.coherent_specular_complete_intensity(nullptr, state, sd, secondary_result);
return 0;
CCL_METAL_SURFACE_STAGE_END

CCL_METAL_SURFACE_STAGE_BEGIN(5)
context.integrate_surface_emission(nullptr, state, sd, render_buffer);
return 0;
CCL_METAL_SURFACE_STAGE_END

CCL_METAL_SURFACE_STAGE_BEGIN(6)
#  ifdef __PASSES__
context.film_write_data_passes(nullptr, state, sd, render_buffer);
#  endif
return 0;
CCL_METAL_SURFACE_STAGE_END

CCL_METAL_SURFACE_STAGE_BEGIN(7)
#  ifdef __DENOISING_FEATURES__
context.film_write_denoising_features_surface(nullptr, state, sd, render_buffer);
#  endif
return 0;
CCL_METAL_SURFACE_STAGE_END

#  undef CCL_METAL_SURFACE_STAGE_BEGIN
#  undef CCL_METAL_SURFACE_STAGE_END
#endif /* __KERNEL_METAL_VISIBLE_SHADING__ */
