/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Volume Index of Refraction
 *
 * A medium whose index of refraction `n` varies in space bends the light that crosses it: hot
 * air above a fire or a road, a gas escaping into another one, sugar dissolving in water. The
 * path of a ray follows the eikonal equation, here with the parameter `sigma` of `ds = n dsigma`
 * and `v = n dx/ds`, in which it is the motion of a particle in a potential:
 *
 *   dx/dsigma = v,   dv/dsigma = grad(n^2) / 2
 *
 * The field
 *
 * Evaluating volume shaders along every ray costs many times what the rest of the path does.
 * The scene update evaluates them once instead, on a grid for every object with such a volume
 * (see VolumeManager::update_ior_grids()), which stores `n^2 - 1` per voxel. The field is the
 * trilinear interpolation of the voxels and its gradient the derivative of that, which jumps
 * from one voxel to the next: it is looked up at a random position within half a voxel of where
 * it is needed. What a ray gets on average is the gradient of the field smoothed over a voxel,
 * which is continuous, second order for a smooth index, and still the gradient of an index: at a
 * sharp change of it the law of Snell holds, and the field ends within a voxel of where the
 * voxels do. Three coarser levels have, for blocks of voxels, the mean of the gradient, how far
 * the gradient is from that mean anywhere in the block, and how far from a linear function of
 * the position.
 *
 * The steps
 *
 * Leapfrog: half a step along `v`, the change of `v` by the gradient there, the other half.
 * The scheme is second order, takes a ray traced backwards along the same curve up to that
 * order, which bidirectional methods rely on, and is exact where the gradient is constant.
 * A step is a voxel long where the gradient varies and a whole block where it does not, or
 * where what it does within the block is below what the footprint of the pixel can show: the
 * uniform air around a plume of hot gas costs next to nothing, and neither do paths behind a
 * diffuse bounce. With the random positions of the lookups, what is left of the discretization
 * is noise and not a pattern.
 *
 * The intersection
 *
 * The curve is not intersected with the scene step by step. A straight ray is traced along the
 * chord the curve is expected to follow, and the steps run along it for as long as the curve
 * stays within a tolerance of that ray: a fraction of the footprint of the pixel. Where it
 * leaves, the next ray starts where the path has got to on the last one: the rays are one line
 * from the start of the path to its end, which nothing slips through. What that line hits is
 * looked for again along the curve, for the place and the direction of the hit to be those of
 * the curve. Through the weak fields of hot air most paths need one intersection per bounce, as
 * without any refraction, and still leave the volume in the direction the curve has there.
 *
 * Rays are represented so that the rest of the integrator need not know about any of this: the
 * ray of the last chord with its origin moved back by the length of the path so far, so that
 * the distance to a hit is the length of the path from the last scattering event. Everything
 * that reasons about the segment between two path vertices (multiple importance sampling, ray
 * differentials, bidirectional measures) takes this straight ray as the edge, which is exact
 * where the index is constant and consistent between the strategies elsewhere. After the hit
 * the direction of the ray becomes the one the curve arrives with, see volume_ior_arrive().
 *
 * Media that absorb, scatter or emit along a curved ray are integrated along its chords, by
 * the volume kernel, one chord at a time: chords that the curve stays within half a voxel of,
 * which is one from the surface the path enters the volume at to the one it leaves at for most
 * paths.
 *
 * What is not bent: shadow rays and the connections of bidirectional methods. Light reaches a
 * shading point through hot air along a path whose direction differs from the straight one by
 * less than a milliradian, so direct light is right up to the focusing of light by the medium
 * (the shadowgraph), which sampled paths that reach an emitter and light subpaths do include.
 * Radiance is not scaled by the square of the index either: where the index of the medium
 * around a surface is far from that around the camera, as inside of a lens, the surface is
 * shaded as if it were in air. */

#include "kernel/globals.h"
#include "kernel/types.h"

#include "kernel/bvh/bvh.h"

#include "kernel/geom/object.h"

#include "kernel/integrator/state.h"
#include "kernel/integrator/state_util.h"
#include "kernel/integrator/volume_stack.h"

#include "kernel/light/light.h"

#include "kernel/sample/lcg.h"

CCL_NAMESPACE_BEGIN

enum VolumeIORResult {
  /* Nothing hit before the end of the ray. */
  VOLUME_IOR_MISS = 0,
  /* Surface or light hit. */
  VOLUME_IOR_HIT = 1,
  /* The ray is a chord of the curve that ends inside of the volume, with nothing hit before.
   * Returned for media that also absorb, scatter or emit, for the caller to integrate them
   * along the chord before it continues with the same VolumeIORState. */
  VOLUME_IOR_SEGMENT = 2,
};

/* What a caller keeps of a curved ray between the chords of a bounce. */
struct VolumeIORState {
  /* Steps taken, up to the maximum after which the ray continues straight. */
  int steps;
  /* Set with VOLUME_IOR_SEGMENT: the curve continues at the end of the ray. */
  bool pending;
  /* Direction of the curve at the end of the ray that was returned. */
  float3 D;
};

/* The direction the curve arrives with at the end of the ray volume_ior_intersect() returned,
 * which is a chord of the curve: turns the ray around its end. To be called after the media
 * along the chord were integrated. A light keeps the ray it was intersected with: its sample is
 * computed from that again. */
ccl_device_inline void volume_ior_arrive(ccl_private Ray *ccl_restrict ray,
                                         const ccl_private Intersection *ccl_restrict isect,
                                         const bool hit,
                                         const float3 D)
{
  if (hit && (isect->type & PRIMITIVE_LAMP)) {
    return;
  }
  const float t = (hit) ? isect->t : ray->tmin;
  const float3 P = ray->P + ray->D * t;
  ray->D = D;
  ray->P = P - D * t;
}

/* The ray that continues the curve behind the chord volume_ior_intersect() returned with
 * VOLUME_IOR_SEGMENT, up to the length `t_limit` of the path. */
ccl_device_inline void volume_ior_resume(ccl_private Ray *ccl_restrict ray,
                                         const float3 D,
                                         const float t_limit)
{
  const float t = ray->tmax;
  const float3 P = ray->P + ray->D * t;
  ray->D = D;
  ray->P = P - D * t;
  ray->tmin = t;
  ray->tmax = t_limit;
  /* Not at the surface the ray left any more. */
  ray->self.object = OBJECT_NONE;
  ray->self.prim = PRIM_NONE;
}

#ifdef __VOLUME_IOR__

/* Rays in a row along which a path is bent without a scattering event in between, see
 * volume_ior_path_bends(). */
#  define VOLUME_IOR_MAX_PASSES 16
/* Volumes with an index of refraction that overlap at a point and are all taken into account.
 * Their gradients add up. */
#  define VOLUME_IOR_MAX_FIELDS 3
/* A block is crossed in one step with its mean gradient if what the rest of the gradient may
 * turn the path by is below an angle: this fraction of the angle of the footprint of the pixel,
 * and no less than the smallest angle, which is also that of paths without a footprint. With the
 * gradient looked up along the step the angle is larger by a factor, and anything goes that is
 * within a fraction of the mean: a quarter of the footprint for a block, which the many paths of
 * a pixel average out. The lookup is in the middle of the step if that is what the part of the
 * gradient which is not linear in the position turns the path by, and at a random position
 * along the step, which leaves noise and no bias, if it is all that is not the mean. */
#  define VOLUME_IOR_BLOCK_FOOTPRINT 0.01f
#  define VOLUME_IOR_BLOCK_ANGLE 2e-7f
#  define VOLUME_IOR_BLOCK_LOOKUP 25.0f
#  define VOLUME_IOR_BLOCK_RELATIVE 0.01f
/* Distance between the curve and the ray traced for it, as a fraction of the footprint of the
 * pixel at that distance, and of a voxel for paths without a footprint. With one it is no less
 * than the smallest fraction of a voxel, and no more than a few voxels: the rays are to stay on
 * the side of a surface that the curve is on. */
#  define VOLUME_IOR_TOLERANCE_FOOTPRINT 0.25f
#  define VOLUME_IOR_TOLERANCE_VOXEL 0.05f
#  define VOLUME_IOR_TOLERANCE_VOXEL_MIN 1e-3f
#  define VOLUME_IOR_TOLERANCE_VOXEL_MAX 2.0f
/* Distance between the curve and the ray that has hit something from which the hit is looked for
 * again along the curve, as a fraction of the tolerance. Small: whether a path that grazes a
 * surface hits it is all or nothing, as at the near end of a mirage. */
#  define VOLUME_IOR_TOLERANCE_HIT 0.1f
/* Longest arc a ray is traced for that starts off the curve, in tolerances. */
#  define VOLUME_IOR_CHAIN_LENGTH 1e4f
/* Distance between the curve and the chord its media are integrated along, in voxels. */
#  define VOLUME_IOR_TOLERANCE_MEDIA 0.5f
/* The footprint of a path that was scattered is no smaller than this fraction of the angle of
 * the lobe it was sampled from, see volume_ior_path_footprint(). */
#  define VOLUME_IOR_FOOTPRINT_LOBE 0.25f
/* Angle that the part of the gradient which varies within a block turns the path by in a step
 * through its voxels at most, where a quarter of a voxel is no more than that. */
#  define VOLUME_IOR_MAX_TURN 0.01f
#  define VOLUME_IOR_MIN_FINE_STEP 0.25f
/* Shortest step, as a fraction of a voxel: progress where the limits of a step meet. */
#  define VOLUME_IOR_MIN_STEP 0.02f

/* Grid of a volume on the stack, in world space at the time of the ray. */
struct VolumeIORField {
  /* World to voxel index space. A voxel has its value at its center. */
  Transform tfm;
  /* Bound of the scale from index to world space of a gradient. */
  float gradient_scale;
  /* Voxels per step along the ray, inverted, for every axis. */
  float3 inv_step;
  int3 res;
  int voxel_offset;
  int block_offset_1;
  int block_offset_2;
  int block_offset_3;
};

/* Grid of a volume stack entry, -1 if it has no index of refraction. */
ccl_device_inline int volume_ior_find_grid(KernelGlobals kg, const VolumeStack entry)
{
  if (entry.object == OBJECT_NONE) {
    return -1;
  }
  int grid = kernel_data_fetch(volume_ior_object_grid, entry.object);
  while (grid >= 0) {
    if (kernel_data_fetch(volume_ior_grids, grid).shader == (entry.shader & SHADER_MASK)) {
      return grid;
    }
    grid = kernel_data_fetch(volume_ior_grids, grid).next;
  }
  return -1;
}

/* Whether the path is inside of a volume that bends it. `only_ior` is set if such volumes are
 * all it is inside of: there is nothing else to them. */
ccl_device_inline bool volume_ior_stack_active(KernelGlobals kg,
                                               ConstIntegratorState state,
                                               ccl_private bool *only_ior)
{
  *only_ior = true;
  if (!(kernel_data.kernel_features & KERNEL_FEATURE_VOLUME_IOR)) {
    return false;
  }

  bool active = false;
  for (int i = 0;; i++) {
    const VolumeStack entry = integrator_state_read_volume_stack(state, i);
    if (entry.shader == SHADER_NONE) {
      break;
    }
    const int shader_flag = kernel_data_fetch(shaders, (entry.shader & SHADER_MASK)).flags;
    if (!(shader_flag & SD_HAS_VOLUME)) {
      /* Medium of nested dielectrics, without a volume shader. */
      continue;
    }
    if ((shader_flag & SD_HAS_VOLUME_IOR) && volume_ior_find_grid(kg, entry) >= 0) {
      active = true;
    }
    if (!(shader_flag & SD_HAS_ONLY_VOLUME_IOR)) {
      *only_ior = false;
    }
  }
  return active;
}

ccl_device_inline bool volume_ior_stack_active(KernelGlobals kg, ConstIntegratorState state)
{
  bool only_ior;
  return volume_ior_stack_active(kg, state, &only_ior);
}

/* Whether the path is bent on the ray it starts now. A curve that follows a surface made of
 * facets passes it over and over: the rim of a lens, a ray that grazes the side of a box of hot
 * air. Every pass is a ray of its own, and a turn of all the kernels on a device that runs the
 * paths in step. After a number of them without a scattering event in between the path is
 * straight, and leaves. */
ccl_device_inline bool volume_ior_path_bends(IntegratorState state)
{
  const int bounce = INTEGRATOR_STATE(state, path, bounce);
  if (INTEGRATOR_STATE(state, path, volume_ior_bounce) != bounce) {
    INTEGRATOR_STATE_WRITE(state, path, volume_ior_bounce) = bounce;
    INTEGRATOR_STATE_WRITE(state, path, volume_ior_passes) = 0;
    return true;
  }
  const int passes = INTEGRATOR_STATE(state, path, volume_ior_passes) + 1;
  INTEGRATOR_STATE_WRITE(state, path, volume_ior_passes) = min(passes, VOLUME_IOR_MAX_PASSES);
  return passes < VOLUME_IOR_MAX_PASSES;
}

ccl_device_inline void volume_ior_field_setup(KernelGlobals kg,
                                              const int grid_index,
                                              const int object,
                                              const float time,
                                              ccl_private VolumeIORField *field)
{
  const ccl_global KernelVolumeIORGrid *grid = &kernel_data_fetch(volume_ior_grids, grid_index);
  Transform tfm = grid->tfm;
  if (!(kernel_data_fetch(object_flag, object) & SD_OBJECT_TRANSFORM_APPLIED)) {
    Transform itfm;
    object_fetch_transform_motion_test(kg, object, time, &itfm);
    /* The grid transform after the inverse one of the object. */
    const float4 c_x = make_float4(itfm.x.x, itfm.y.x, itfm.z.x, 0.0f);
    const float4 c_y = make_float4(itfm.x.y, itfm.y.y, itfm.z.y, 0.0f);
    const float4 c_z = make_float4(itfm.x.z, itfm.y.z, itfm.z.z, 0.0f);
    const float4 c_w = make_float4(itfm.x.w, itfm.y.w, itfm.z.w, 1.0f);
    const Transform grid_tfm = tfm;
    tfm.x = make_float4(
        dot(grid_tfm.x, c_x), dot(grid_tfm.x, c_y), dot(grid_tfm.x, c_z), dot(grid_tfm.x, c_w));
    tfm.y = make_float4(
        dot(grid_tfm.y, c_x), dot(grid_tfm.y, c_y), dot(grid_tfm.y, c_z), dot(grid_tfm.y, c_w));
    tfm.z = make_float4(
        dot(grid_tfm.z, c_x), dot(grid_tfm.z, c_y), dot(grid_tfm.z, c_z), dot(grid_tfm.z, c_w));
  }
  field->tfm = tfm;
  field->gradient_scale = sqrtf(len_squared(make_float3(tfm.x)) + len_squared(make_float3(tfm.y)) +
                                len_squared(make_float3(tfm.z)));
  field->inv_step = make_float3(1.0f / grid->step_x, 1.0f / grid->step_y, 1.0f / grid->step_z);
  field->res = make_int3(grid->res_x, grid->res_y, grid->res_z);
  field->voxel_offset = grid->voxel_offset;
  field->block_offset_1 = grid->block_offset_1;
  field->block_offset_2 = grid->block_offset_2;
  field->block_offset_3 = grid->block_offset_3;
}

/* Gradient of a field in world space from the one in its index space. */
ccl_device_inline float3 volume_ior_field_gradient(const ccl_private VolumeIORField *field,
                                                   const float3 g)
{
  return g.x * make_float3(field->tfm.x) + g.y * make_float3(field->tfm.y) +
         g.z * make_float3(field->tfm.z);
}

/* `grad(n^2) / 2` in index space and `n^2 - 1` of a field at a position of its index space:
 * the trilinear interpolation of the voxels and its derivative. */
ccl_device_inline float4 volume_ior_field_lookup(KernelGlobals kg,
                                                 const ccl_private VolumeIORField *field,
                                                 const float3 p)
{
  const int3 res = field->res;
  const float3 q_max = make_float3(res - make_int3(1));
  const float3 q = p - make_float3(0.5f);
  const float3 q_clamped = clamp(q, zero_float3(), q_max);
  /* Rounded down: the conversion of a vector alone rounds to the nearest on some devices. */
  const int3 i = min(make_int3(floor(q_clamped)), res - make_int3(2));
  const float3 t = q_clamped - make_float3(i);

  const int stride_y = res.x;
  const int stride_z = res.x * res.y;
  const int index = field->voxel_offset + i.x + i.y * stride_y + i.z * stride_z;

  const float v000 = kernel_data_fetch(volume_ior_voxels, index);
  const float v100 = kernel_data_fetch(volume_ior_voxels, index + 1);
  const float v010 = kernel_data_fetch(volume_ior_voxels, index + stride_y);
  const float v110 = kernel_data_fetch(volume_ior_voxels, index + stride_y + 1);
  const float v001 = kernel_data_fetch(volume_ior_voxels, index + stride_z);
  const float v101 = kernel_data_fetch(volume_ior_voxels, index + stride_z + 1);
  const float v011 = kernel_data_fetch(volume_ior_voxels, index + stride_z + stride_y);
  const float v111 = kernel_data_fetch(volume_ior_voxels, index + stride_z + stride_y + 1);

  /* Along x, at the four edges. */
  const float x00 = mix(v000, v100, t.x);
  const float x10 = mix(v010, v110, t.x);
  const float x01 = mix(v001, v101, t.x);
  const float x11 = mix(v011, v111, t.x);
  const float y0 = mix(x00, x10, t.y);
  const float y1 = mix(x01, x11, t.y);

  /* The index is that of the border outside of the grid. */
  const float gx = (q.x < 0.0f || q.x > q_max.x) ? 0.0f :
                                                   mix(mix(v100 - v000, v110 - v010, t.y),
                                                       mix(v101 - v001, v111 - v011, t.y),
                                                       t.z);
  const float gy = (q.y < 0.0f || q.y > q_max.y) ? 0.0f : mix(x10 - x00, x11 - x01, t.z);
  const float gz = (q.z < 0.0f || q.z > q_max.z) ? 0.0f : y1 - y0;

  return make_float4(0.5f * gx, 0.5f * gy, 0.5f * gz, mix(y0, y1, t.z));
}

/* Distance along the ray `p + t * d` to where it leaves the box from `lo` to `hi`, for `p`
 * inside of it. */
ccl_device_inline float volume_ior_box_exit(const float3 p,
                                            const float3 d,
                                            const float3 lo,
                                            const float3 hi)
{
  const float tx = (d.x > 0.0f) ? (hi.x - p.x) / d.x : (d.x < 0.0f) ? (lo.x - p.x) / d.x : FLT_MAX;
  const float ty = (d.y > 0.0f) ? (hi.y - p.y) / d.y : (d.y < 0.0f) ? (lo.y - p.y) / d.y : FLT_MAX;
  const float tz = (d.z > 0.0f) ? (hi.z - p.z) / d.z : (d.z < 0.0f) ? (lo.z - p.z) / d.z : FLT_MAX;
  return fmaxf(fminf(tx, fminf(ty, tz)), 0.0f);
}

/* Distance along the ray `p + t * d` to where it enters the box from zero to `hi`, FLT_MAX if
 * it does not. */
ccl_device_inline float volume_ior_box_enter(const float3 p, const float3 d, const float3 hi)
{
  float t0 = 0.0f;
  float t1 = FLT_MAX;
#  define VOLUME_IOR_SLAB(axis) \
    if (d.axis != 0.0f) { \
      const float ta = -p.axis / d.axis; \
      const float tb = (hi.axis - p.axis) / d.axis; \
      t0 = fmaxf(t0, fminf(ta, tb)); \
      t1 = fminf(t1, fmaxf(ta, tb)); \
    } \
    else if (p.axis < 0.0f || p.axis > hi.axis) { \
      return FLT_MAX; \
    }
  VOLUME_IOR_SLAB(x)
  VOLUME_IOR_SLAB(y)
  VOLUME_IOR_SLAB(z)
#  undef VOLUME_IOR_SLAB
  return (t0 <= t1) ? t0 : FLT_MAX;
}

/* How a field is stepped through. */
enum VolumeIORStep {
  /* A voxel at a time, with the gradient at the middle of the step. */
  VOLUME_IOR_STEP_VOXEL = 0,
  /* A block at once, with the mean gradient of the block: it is the same all over it. */
  VOLUME_IOR_STEP_MEAN = 1,
  /* A block at once, with the gradient at the middle of the step: it changes about linearly
   * with the position in the block, which the middle is exact for. */
  VOLUME_IOR_STEP_LINEAR = 2,
  /* A block at once, with the gradient at a random position along the step: it varies little
   * over the block, and what is left of that is noise without bias. */
  VOLUME_IOR_STEP_BLOCK = 3,
};

/* The step through a field that a path is at, kept from one step to the next: a block that is
 * crossed in more than one step, as the curvature of the path or the ray that is traced along
 * it may ask for, is looked up once. */
struct VolumeIORFieldStep {
  int mode;
  /* Length of the path that is left of the block, or of the block within which no larger one
   * can be crossed at once. */
  float remaining;
  /* Mean gradient of the block in world space. */
  float3 gradient;
  /* Length of a step through the voxels of the block, in steps of a voxel. */
  float fine_scale;
  /* Lengths of the path that are left of the blocks of the two larger levels which cannot be
   * crossed at once: no need to look at those levels again before. */
  float retry_2;
  float retry_3;
};

/* A block of a coarse level of a field: whether the step from `p` along `d` (per unit of the
 * length of the path, both in index space) may cross it at once, and how. Sets the length of
 * that step and the mean gradient in the block. */
ccl_device_inline int volume_ior_field_block(KernelGlobals kg,
                                             const ccl_private VolumeIORField *field,
                                             const int block_offset,
                                             const int shift,
                                             const float3 p,
                                             const float3 d,
                                             const int3 voxel,
                                             const float inv_n2,
                                             const float angle,
                                             const float fine_step,
                                             ccl_private float *step,
                                             ccl_private float3 *gradient,
                                             ccl_private float *strength)
{
  const int3 res = field->res;
  const int size = 1 << shift;
  const int3 block = make_int3(voxel.x >> shift, voxel.y >> shift, voxel.z >> shift);
  const int3 block_res = make_int3(
      (res.x + size - 1) >> shift, (res.y + size - 1) >> shift, (res.z + size - 1) >> shift);
  const int index = block_offset + 2 * (block.x + (block.y + block.z * block_res.y) * block_res.x);

  const float3 lo = make_float3(
      float(block.x << shift), float(block.y << shift), float(block.z << shift));
  const float3 hi = min(lo + make_float3(float(size)), make_float3(res));
  const float exit = volume_ior_box_exit(p, d, lo, hi);
  *step = exit;
  if (exit < fine_step) {
    /* A corner of the block, or along one of its sides: a step through the voxels gets
     * further. */
    return VOLUME_IOR_STEP_VOXEL;
  }

  const float4 data = kernel_data_fetch(volume_ior_blocks, index);
  const float3 mean = make_float3(data);
  const float deviation = data.w;
  *gradient = mean;
  /* How much the gradient varies in the block. */
  *strength = deviation;
  /* The angle a gradient that is left out turns the path by, per unit of that gradient. */
  const float turn = field->gradient_scale * exit * inv_n2;
  if (deviation * turn <= angle) {
    return VOLUME_IOR_STEP_MEAN;
  }
  const float lookup_angle = VOLUME_IOR_BLOCK_LOOKUP * angle;
  const float relative = VOLUME_IOR_BLOCK_RELATIVE * len(mean);
  const float nonlinear = kernel_data_fetch(volume_ior_blocks, index + 1).x;
  if (nonlinear * turn <= lookup_angle || nonlinear <= relative) {
    return VOLUME_IOR_STEP_LINEAR;
  }
  if (deviation * turn <= lookup_angle || deviation <= relative) {
    return VOLUME_IOR_STEP_BLOCK;
  }
  return VOLUME_IOR_STEP_VOXEL;
}

/* The step a field allows from the position `P` along the direction `D`: its length, and in
 * `field_step` how the gradient of the step is found. */
ccl_device_inline float volume_ior_field_step(KernelGlobals kg,
                                              const ccl_private VolumeIORField *field,
                                              const float3 P,
                                              const float3 D,
                                              const float inv_n2,
                                              const float angle,
                                              ccl_private VolumeIORFieldStep *field_step)
{
  const float3 d = transform_direction(&field->tfm, D);
  const float fine_step = 1.0f / fmaxf(reduce_max(fabs(d) * field->inv_step), 1e-20f);
  if (field_step->remaining > 0.0f) {
    /* Still in the block of the last step. */
    return (field_step->mode == VOLUME_IOR_STEP_VOXEL) ? fine_step * field_step->fine_scale :
                                                         field_step->remaining;
  }

  const float3 p = transform_point(&field->tfm, P);
  const float3 res = make_float3(field->res);
  /* A thousandth of a voxel behind where a block or the grid ends. */
  const float behind = 1e-3f / fmaxf(reduce_max(fabs(d)), 1e-20f);

  field_step->mode = VOLUME_IOR_STEP_MEAN;
  field_step->gradient = zero_float3();
  field_step->remaining = 0.0f;

  if (p.x < 0.0f || p.y < 0.0f || p.z < 0.0f || p.x >= res.x || p.y >= res.y || p.z >= res.z) {
    /* Outside of the grid the index is that of its border: straight up to where it starts. */
    const float enter = volume_ior_box_enter(p, d, res);
    return (enter == FLT_MAX) ? FLT_MAX : enter + behind;
  }

  const int3 voxel = make_int3(floor(p));
  float step = 0.0f;
  float3 mean = zero_float3();
  float strength = 0.0f;
  int mode = VOLUME_IOR_STEP_VOXEL;
  if (field_step->retry_3 <= 0.0f) {
    mode = volume_ior_field_block(kg,
                                  field,
                                  field->block_offset_3,
                                  VOLUME_IOR_BLOCK_SHIFT_3,
                                  p,
                                  d,
                                  voxel,
                                  inv_n2,
                                  angle,
                                  fine_step,
                                  &step,
                                  &mean,
                                  &strength);
    if (mode == VOLUME_IOR_STEP_VOXEL) {
      field_step->retry_3 = step;
    }
  }
  if (mode == VOLUME_IOR_STEP_VOXEL && field_step->retry_2 <= 0.0f) {
    mode = volume_ior_field_block(kg,
                                  field,
                                  field->block_offset_2,
                                  VOLUME_IOR_BLOCK_SHIFT_2,
                                  p,
                                  d,
                                  voxel,
                                  inv_n2,
                                  angle,
                                  fine_step,
                                  &step,
                                  &mean,
                                  &strength);
    if (mode == VOLUME_IOR_STEP_VOXEL) {
      field_step->retry_2 = step;
    }
  }
  if (mode == VOLUME_IOR_STEP_VOXEL) {
    mode = volume_ior_field_block(kg,
                                  field,
                                  field->block_offset_1,
                                  VOLUME_IOR_BLOCK_SHIFT_1,
                                  p,
                                  d,
                                  voxel,
                                  inv_n2,
                                  angle,
                                  fine_step,
                                  &step,
                                  &mean,
                                  &strength);
  }
  field_step->mode = mode;
  if (mode != VOLUME_IOR_STEP_VOXEL) {
    field_step->gradient = volume_ior_field_gradient(field, mean);
    field_step->remaining = step + behind;
    return field_step->remaining;
  }
  /* No need to look at the blocks again before the path leaves the smallest one. Shorter steps
   * where what the gradient varies by within the block may turn the path by much in a step of
   * a voxel, as at a change of the index within a voxel: the scheme is second order in that
   * angle. By what the block has, not by what a lookup finds, which would be biased. */
  field_step->remaining = step;
  const float turn = strength * field->gradient_scale * inv_n2 * fine_step;
  field_step->fine_scale = (turn > VOLUME_IOR_MAX_TURN) ?
                               fmaxf(VOLUME_IOR_MAX_TURN / turn, VOLUME_IOR_MIN_FINE_STEP) :
                               1.0f;
  return fine_step * field_step->fine_scale;
}

/* What a ray of volume_ior_intersect() is traced for. */
#  define VOLUME_IOR_PHASE_PLAN 0
/* Along the chord the curve is expected to follow. */
#  define VOLUME_IOR_PHASE_RAY 1
/* From the end of the path on the last ray over to the curve. */
#  define VOLUME_IOR_PHASE_GAP 2
/* The same, before a chord of the media is returned. */
#  define VOLUME_IOR_PHASE_CHORD_GAP 3
/* For the hit of the ray again: along the last step of the curve, */
#  define VOLUME_IOR_PHASE_LAST_STEP 4
/* along the steps before it, */
#  define VOLUME_IOR_PHASE_REST 5
/* and from the ray over to the curve. */
#  define VOLUME_IOR_PHASE_SIDE 6

/* Lights that a camera path may hit on the way, see lights_intersect(). */
struct VolumeIORLights {
  bool enabled;
  int last_prim;
  int last_object;
  int last_type;
  PathRayVisibility path_visibility;
  uint32_t path_flag;
};

/* Closest intersection along a ray that the volumes of the stack bend, which must have an index
 * of refraction (see volume_ior_stack_active()).
 *
 * The ray is traced from `ray->tmin` to `ray->tmax`, along the length of its path. On return the
 * ray is the last chord of the curve, from `tmin` on, as described at the top, and `ior->D` the
 * direction the curve has at its end, for volume_ior_arrive().
 *
 * `rand` seeds the positions of the lookups. `ior` is zero at the start of a bounce. After
 * VOLUME_IOR_SEGMENT the curve continues with the ray of volume_ior_resume() and the same `ior`.
 */
ccl_device VolumeIORResult volume_ior_intersect(KernelGlobals kg,
                                                IntegratorState state,
                                                ccl_private Ray *ccl_restrict ray,
                                                const PathRayVisibility visibility,
                                                ccl_private Intersection *ccl_restrict isect,
                                                const float rand,
                                                ccl_private VolumeIORState *ior,
                                                const ccl_private VolumeIORLights *lights)
{
  /* The grids of the stack. */
  VolumeIORField fields[VOLUME_IOR_MAX_FIELDS];
  int num_fields = 0;
  bool only_ior = true;
  for (int i = 0;; i++) {
    const VolumeStack entry = integrator_state_read_volume_stack(state, i);
    if (entry.shader == SHADER_NONE) {
      break;
    }
    const int shader_flag = kernel_data_fetch(shaders, (entry.shader & SHADER_MASK)).flags;
    if (!(shader_flag & SD_HAS_VOLUME)) {
      continue;
    }
    if (!(shader_flag & SD_HAS_ONLY_VOLUME_IOR)) {
      only_ior = false;
    }
    if (!(shader_flag & SD_HAS_VOLUME_IOR) || num_fields == VOLUME_IOR_MAX_FIELDS) {
      continue;
    }
    const int grid = volume_ior_find_grid(kg, entry);
    if (grid >= 0) {
      volume_ior_field_setup(kg, grid, entry.object, ray->time, &fields[num_fields]);
      num_fields++;
    }
  }
  kernel_assert(num_fields > 0);

  const int max_steps = kernel_data.integrator.volume_ior_max_steps;
  const float t_limit = ray->tmax;

  int last_prim = (lights) ? lights->last_prim : PRIM_NONE;
  int last_object = (lights) ? lights->last_object : OBJECT_NONE;
  int last_type = (lights) ? lights->last_type : 0;

  /* Position on the curve and the length of the path up to it. */
  float3 X = ray->P + ray->D * ray->tmin;
  float s = ray->tmin;
  float3 D = ray->D;
  ior->pending = false;
  if (!(t_limit > s)) {
    ior->D = D;
    isect->prim = PRIM_NONE;
    isect->object = OBJECT_NONE;
    return VOLUME_IOR_MISS;
  }

  /* Random numbers for the positions of the lookups. */
  uint lcg_state = lcg_init(__float_as_uint(rand) + uint(ior->steps) * 0x9e3779b9u);

  /* `v = n dx/ds` and the gradient at the start, and the size of a voxel there for the
   * tolerances. The index is that of the field the steps see, smoothed over a voxel by the
   * random positions of their lookups: along the path `n^2 - v^2` stays what it starts with. */
  float n2 = 1.0f;
  float3 gradient = zero_float3();
  float voxel_size = FLT_MAX;
  float voxel_size_max = 0.0f;
  for (int i = 0; i < num_fields; i++) {
    const float3 p = transform_point(&fields[i].tfm, X) + lcg_step_float3(&lcg_state) -
                     make_float3(0.5f);
    const float4 value = volume_ior_field_lookup(kg, &fields[i], p);
    n2 += value.w;
    gradient += volume_ior_field_gradient(&fields[i], make_float3(value));
    /* World length of the shortest side of a voxel: the smallest singular value of the
     * transform is bounded from below by the inverse of the norm of its rows. */
    const float scale = fmaxf(
        len(make_float3(fields[i].tfm.x)),
        fmaxf(len(make_float3(fields[i].tfm.y)), len(make_float3(fields[i].tfm.z))));
    voxel_size = fminf(voxel_size, 1.0f / fmaxf(scale, 1e-20f));
    const float scale_min = fminf(
        len(make_float3(fields[i].tfm.x)),
        fminf(len(make_float3(fields[i].tfm.y)), len(make_float3(fields[i].tfm.z))));
    voxel_size_max = fmaxf(voxel_size_max, 1.0f / fmaxf(scale_min, 1e-20f));
  }
  /* A shader may give anything: keep the index positive. */
  n2 = fmaxf(n2, 1e-4f);
  float3 v = D * sqrtf(n2);

  /* Footprint of the pixel along the path, for the tolerance. */
  const float footprint_P = ray->dP;
  const float footprint_D = ray->dD;
  const bool has_footprint = (footprint_P > 0.0f || footprint_D > 0.0f);
  const float min_step = VOLUME_IOR_MIN_STEP * voxel_size;
  /* The angle of the footprint: a pixel of an orthographic camera has none, the one its size
   * has over a typical grid stands in for it. */
  const float block_angle = fmaxf(VOLUME_IOR_BLOCK_FOOTPRINT *
                                      (footprint_D + footprint_P / (64.0f * voxel_size_max)),
                                  VOLUME_IOR_BLOCK_ANGLE);

  /* Start of the chord of the media that is returned, and the point of the curve where a ray
   * ended that is the furthest from that chord. */
  const float3 chord_X = X;
  const float chord_s = s;
  float3 far_X = X;
  /* Where the curve was when the last ray was traced, and where the path has got to on that
   * ray, if another one follows. */
  float3 X_start = X;
  float s_start = s;
  float3 end_last = X;
  bool has_end = false;
  /* The path has left the fields for good. */
  bool straight = false;

  /* The steps through the fields, see volume_ior_field_step(). */
  VolumeIORFieldStep field_steps[VOLUME_IOR_MAX_FIELDS];
  for (int i = 0; i < num_fields; i++) {
    field_steps[i].mode = VOLUME_IOR_STEP_VOXEL;
    field_steps[i].remaining = 0.0f;
    field_steps[i].gradient = zero_float3();
    field_steps[i].fine_scale = 1.0f;
    field_steps[i].retry_2 = 0.0f;
    field_steps[i].retry_3 = 0.0f;
  }
  /* Squared curvature of the path at the last step. */
  float curvature2 = len_squared(gradient - D * dot(gradient, D)) / sqr(n2);

  /* What the path starts at, which the first ray is not to hit again. */
  int self_object = ray->self.object;
  int self_prim = ray->self.prim;

  /* Every turn of the loop below traces one ray, for what its phase says: all the rays are
   * traced by the same call, which is what devices that run many paths in step need to be fast.
   * First the ray is planned that the curve is expected to follow, and traced after one across
   * the gap to it, if there is one. The curve is stepped along it then. A hit of that ray that
   * the curve is not right at is looked for again along the curve, in up to three rays. */
  int phase = VOLUME_IOR_PHASE_PLAN;
  float tolerance = 0.0f;
  bool bend = true;
  /* The ray along the curve: its origin, direction and the length of the path at its ends. */
  float3 O = X;
  float3 aim = D;
  float t_start = s;
  float t_end = t_limit;
  bool hit = false;
  float t_stop = t_limit;
  /* The last step of the curve along it. */
  bool stepped = false;
  float3 X_last = X;
  /* Start of the ray across a gap, and the ends of the rays along the curve. */
  float3 gap_start = X;
  float3 exact_start = X;
  float3 exact_end = X;
  float exact_length = 0.0f;
  /* Chord of the media that is returned if nothing is hit in the gap behind it. */
  float3 chord_D = D;
  float chord_length = 0.0f;

  for (;;) {
    if (phase == VOLUME_IOR_PHASE_PLAN) {
      /* The ray along the chord the curve is expected to follow from here. The rays are one
       * line from the start of the path to its end, with the curve next to it: nothing is
       * between two of them for the path to slip through. The next ray starts where the path
       * has got to on the last one, and aims at where the curve will be if it keeps its
       * curvature: across the curve, to half the tolerance on its outer side. */
      tolerance = (has_footprint) ?
                      clamp(VOLUME_IOR_TOLERANCE_FOOTPRINT * (footprint_P + footprint_D * s),
                            VOLUME_IOR_TOLERANCE_VOXEL_MIN * voxel_size,
                            VOLUME_IOR_TOLERANCE_VOXEL_MAX * voxel_size_max) :
                      VOLUME_IOR_TOLERANCE_VOXEL * voxel_size;
      bend = ior->steps < max_steps && !straight;

      O = X;
      aim = D;
      float reach = FLT_MAX;
      bool chained = false;
      const bool on_surface = !has_end && self_prim != PRIM_NONE;
      if (bend && curvature2 > 1e-24f) {
        const float inv_n2 = 1.0f / len_squared(v);
        const float3 curvature = (gradient - D * dot(gradient, D)) * inv_n2;
        const float kappa = sqrtf(curvature2);
        const float3 outward = curvature * (-1.0f / kappa);
        /* How far the start of the ray is on the outer side of the curve. */
        const float3 offset = (has_end) ? end_last - X : zero_float3();
        const float outside = dot(offset, outward);
        float arc;
        float3 target;
        if (on_surface) {
          /* A ray that leaves a surface goes along the curve, to stay on the same side of that
           * surface: it does not hit the surface it starts at. */
          arc = fminf(sqrtf(2.0f * tolerance / kappa), 0.5f / kappa);
          target = X + D * arc;
        }
        else if (outside >= 0.0f) {
          /* The longest arc whose chord, from where the ray starts to where it aims at, has the
           * curve no further than four fifths of the tolerance on its inner side. */
          arc = sqrtf(8.0f * (0.8f * tolerance + 0.5f * (outside + 0.5f * tolerance)) / kappa);
          /* A ray that starts off the curve takes long to get to a curve that is about
           * straight: better one that starts on it, after one more across the gap. */
          chained = has_end && arc <= VOLUME_IOR_CHAIN_LENGTH * tolerance;
          if (!chained) {
            arc = sqrtf(8.0f * (0.8f * tolerance + 0.25f * tolerance) / kappa);
          }
          /* Not further around than a chord is a fair guess for. */
          arc = fminf(arc, 0.5f / kappa);
          target = X + D * arc + curvature * (0.5f * arc * arc) + outward * (0.5f * tolerance);
        }
        else {
          /* A ray that starts on the inner side goes along the curve, which crosses it and is
           * the tolerance away on the other side at the end of the arc. */
          arc = fminf(sqrtf(2.0f * (tolerance - outside) / kappa), 0.5f / kappa);
          chained = arc <= VOLUME_IOR_CHAIN_LENGTH * tolerance;
          target = X + offset - outward * outside + D * arc + curvature * (0.5f * arc * arc) +
                   outward * (0.5f * kappa * arc * arc + outside);
          if (!chained) {
            /* From the curve itself. */
            arc = fminf(sqrtf(8.0f * (0.8f * tolerance + 0.25f * tolerance) / kappa),
                        0.5f / kappa);
            target = X + D * arc + curvature * (0.5f * arc * arc) + outward * (0.5f * tolerance);
          }
        }
        if (chained) {
          O = end_last;
        }
        const float target_distance = len(target - O);
        if (target_distance > 0.0f) {
          aim = (target - O) / target_distance;
          reach = 1.1f * target_distance;
        }
        else {
          O = X;
          chained = false;
        }
      }

      /* The length of the path along the ray: that of the curve where it is next to it. */
      t_start = s - dot(X - O, aim);
      t_end = (reach < t_limit - t_start) ? t_start + reach : t_limit;
      X_start = X;
      s_start = s;

      /* From where the path has got to on the last ray over to the curve first, if the ray
       * starts there. */
      phase = (has_end && !chained) ? VOLUME_IOR_PHASE_GAP : VOLUME_IOR_PHASE_RAY;
      gap_start = end_last;
      has_end = false;
    }

    /* The ray of this turn. */
    Ray trace = *ray;
    /* None of the rays of the curve that starts at a surface hits that surface: they may start
     * there as well. */
    trace.self.object = self_object;
    trace.self.prim = self_prim;
    if (phase == VOLUME_IOR_PHASE_RAY) {
      trace.P = O - aim * t_start;
      trace.D = aim;
      trace.tmin = t_start;
      trace.tmax = t_end;
    }
    else {
      float3 from = gap_start;
      float3 to = X;
      if (phase == VOLUME_IOR_PHASE_LAST_STEP) {
        from = exact_start;
        to = exact_end;
      }
      else if (phase == VOLUME_IOR_PHASE_REST) {
        from = X_start;
        to = X_last;
      }
      const float3 across = to - from;
      const float across_length = len(across);
      trace.P = from;
      trace.D = (across_length > 0.0f) ? across / across_length : D;
      trace.tmin = 0.0f;
      trace.tmax = across_length;
    }
    Intersection trace_isect ccl_optional_struct_init;
    trace_isect.prim = PRIM_NONE;
    trace_isect.object = OBJECT_NONE;
    const bool trace_hit = (trace.tmax > trace.tmin) &&
                           scene_intersect(kg, &trace, visibility, &trace_isect);

    /* What the turn ends with: the end of the path at a hit or at the end of the ray along the
     * curve, or the next ray along it. */
    bool finished = false;
    bool advance = false;
    bool passed = false;
    float3 end = X;
    float3 D_end = D;

    if (phase == VOLUME_IOR_PHASE_RAY) {
      hit = trace_hit;
      *isect = trace_isect;
      if (!hit) {
        isect->prim = PRIM_NONE;
        isect->object = OBJECT_NONE;
        isect->t = t_end;
      }
      if (lights && lights->enabled) {
        hit = lights_intersect(kg,
                               state,
                               &trace,
                               isect,
                               last_prim,
                               last_object,
                               last_type,
                               lights->path_visibility,
                               lights->path_flag) ||
              hit;
      }
      t_stop = (hit) ? isect->t : t_end;

      /* Steps along the curve, for as long as it stays next to the ray. */
      bool reached = !bend;
      bool left = false;
      stepped = false;
      X_last = X;
      /* The direction and the index a step is planned with are those of the step before it:
       * they change little from one step to the next, and with their square root and division
       * every step would wait for the one before it to be done. What the steps do to the path
       * does not depend on them, only how long they are. */
      float inv_n = 1.0f / sqrtf(len_squared(v));
      D = v * inv_n;
      while (!reached && !left) {
        if (ior->steps >= max_steps) {
          break;
        }
        ior->steps++;

        const float inv_n2 = sqr(inv_n);
        const float inv_n_next = 1.0f / sqrtf(len_squared(v));
        const float3 D_next = v * inv_n_next;

        const float3 R = X - O;
        const float along = dot(R, aim);
        const float cos_aim = dot(D, aim);
        if (!(cos_aim > 0.25f)) {
          left = true;
          break;
        }
        const float3 R_across = R - aim * along;
        const float3 D_across = D - aim * cos_aim;
        const float a = len_squared(D_across);

        /* As far as the fields allow. */
        float step = FLT_MAX;
        for (int i = 0; i < num_fields; i++) {
          step = fminf(
              step,
              volume_ior_field_step(kg, &fields[i], X, D, inv_n2, block_angle, &field_steps[i]));
        }
        if (step >= FLT_MAX) {
          /* No field ahead: the path is straight from here on. The ray that was traced is the
           * last one if it has the direction of the path, the next one is otherwise. */
          straight = true;
          if (!(a > 1e-12f) && len_squared(R_across) <= sqr(VOLUME_IOR_TOLERANCE_HIT * tolerance))
          {
            reached = true;
          }
          break;
        }
        bool step_reaches = false;

        /* Up to where the ray was traced. */
        if (t_stop != FLT_MAX) {
          const float ahead = fmaxf(t_stop - t_start - along, 0.0f);
          if (step * cos_aim >= ahead) {
            step = ahead / cos_aim;
            step_reaches = true;
          }
        }

        /* Up to where the path, if it were straight from here, is the tolerance away from the
         * ray: a little further, for the test behind the step to find it there. */
        if (len_squared(R_across + D_across * step) > sqr(1.25f * tolerance) && a > 1e-20f) {
          const float b = dot(R_across, D_across);
          const float c = len_squared(R_across) - sqr(1.25f * tolerance);
          const float away = (-b + sqrtf(fmaxf(b * b - a * c, 0.0f))) / a;
          if (away < step) {
            step = away;
            step_reaches = false;
          }
        }

        /* A step that turns the path by more than the arc of a chord is too long for a ray to
         * follow in any case. */
        if (sqr(sqr(step)) * curvature2 > 4.0f * sqr(tolerance)) {
          step = sqrtf(2.0f * tolerance / sqrtf(curvature2));
          step_reaches = false;
        }
        if (step < min_step && !step_reaches) {
          step = min_step;
        }

        /* Leapfrog in `sigma`. The length of a step never depends on the gradient it finds, or
         * the sum over the steps would not be an unbiased estimate of the integral of the
         * gradient along the path. */
        const float dsigma = step * inv_n;
        const float3 X_mid = X + v * (0.5f * dsigma);
        gradient = zero_float3();
        for (int i = 0; i < num_fields; i++) {
          const int mode = field_steps[i].mode;
          field_steps[i].remaining -= step;
          field_steps[i].retry_2 -= step;
          field_steps[i].retry_3 -= step;
          if (mode == VOLUME_IOR_STEP_MEAN) {
            gradient += field_steps[i].gradient;
          }
          else {
            /* In the middle of the step, or anywhere along one across a block in which the
             * gradient varies in no simple way: an estimate of the integral of the gradient over
             * the step without bias, however the gradient changes along it. */
            const float3 X_lookup = (mode == VOLUME_IOR_STEP_BLOCK) ?
                                        X + v * (lcg_step_float(&lcg_state) * dsigma) :
                                        X_mid;
            const float3 p = transform_point(&fields[i].tfm, X_lookup) +
                             lcg_step_float3(&lcg_state) - make_float3(0.5f);
            gradient += volume_ior_field_gradient(
                &fields[i], make_float3(volume_ior_field_lookup(kg, &fields[i], p)));
          }
        }
        if (!isfinite_safe(gradient)) {
          gradient = zero_float3();
        }
        v += gradient * dsigma;
        X_last = X;
        stepped = true;
        X = X_mid + v * (0.5f * dsigma);
        s += step;

        curvature2 = len_squared(gradient - D_next * dot(gradient, D_next)) * sqr(sqr(inv_n_next));
        inv_n = inv_n_next;
        D = D_next;

        if (step_reaches) {
          reached = true;
        }
        else {
          /* The distance from the ray as that of a vector: the difference of the squared
           * lengths has no digits left for it far along the ray. A ray that starts off the
           * curve has it come closer first. */
          const float3 R_new = X - O;
          const float3 R_new_across = R_new - aim * dot(R_new, aim);
          left = len_squared(R_new_across) > sqr(tolerance) &&
                 dot(R_new_across, v - aim * dot(v, aim)) > 0.0f;
        }
      }
      /* The direction of the curve where it is now. */
      D = safe_normalize_fallback(v, D);

      end = O + aim * (t_stop - t_start);
      D_end = D;
      bool exact = false;
      if (reached && hit && stepped && !(isect->type & PRIMITIVE_LAMP)) {
        /* Where the curve is not right at the ray that has hit something, the hit is looked for
         * again along the curve: the tolerance is for finding what a path may hit, the place
         * it hits it at is that of the curve. Caustics are the derivative of that place. */
        const float3 R = X - end;
        exact = len_squared(R - aim * dot(R, aim)) > sqr(VOLUME_IOR_TOLERANCE_HIT * tolerance);
      }
      if (exact) {
        /* First along the last step and a little further, for surfaces at an angle. Not back
         * from it: a straight line is on the outer side of the curve there, where a surface
         * next to the curve is. */
        const float3 last = X - X_last;
        const float last_length = len(last);
        const float margin = 4.0f * tolerance;
        if (last_length > 1e-3f * tolerance) {
          exact_start = X_last;
          exact_end = X + last * (margin / last_length);
          exact_length = last_length;
        }
        else {
          /* The difference of the positions has few digits of the direction. */
          exact_start = X;
          exact_end = X + D * margin;
          exact_length = 0.0f;
        }
        phase = VOLUME_IOR_PHASE_LAST_STEP;
      }
      else if (reached && (hit || t_end >= t_limit)) {
        /* The ray that was traced is the last one. */
        finished = true;
      }
      else {
        advance = true;
      }
    }
    else if (trace_hit) {
      /* What a ray across a gap or along the curve finds is what the path hits. */
      *isect = trace_isect;
      hit = true;
      end = trace.P + trace.D * trace_isect.t;
      D_end = D;
      if (phase == VOLUME_IOR_PHASE_LAST_STEP || phase == VOLUME_IOR_PHASE_REST) {
        /* The direction the curve has there, with the curvature of the last step. */
        const float ahead = (phase == VOLUME_IOR_PHASE_LAST_STEP) ?
                                trace_isect.t - exact_length :
                                trace_isect.t - trace.tmax - len(X - X_last);
        const float inv_n2 = 1.0f / len_squared(v);
        D_end = safe_normalize_fallback(D + (gradient - D * dot(gradient, D)) * (inv_n2 * ahead),
                                        D);
      }
      finished = true;
    }
    else if (phase == VOLUME_IOR_PHASE_GAP) {
      phase = VOLUME_IOR_PHASE_RAY;
    }
    else if (phase == VOLUME_IOR_PHASE_CHORD_GAP) {
      /* The media along the chord are integrated before the path continues. */
      ray->D = chord_D;
      ray->P = chord_X - chord_D * chord_s;
      ray->tmin = chord_s;
      ray->tmax = chord_s + chord_length;
      isect->prim = PRIM_NONE;
      isect->object = OBJECT_NONE;
      isect->t = ray->tmax;
      ior->pending = true;
      ior->D = D;
      return VOLUME_IOR_SEGMENT;
    }
    else if (phase == VOLUME_IOR_PHASE_LAST_STEP) {
      /* Not there: the curve up to the last step, as its chord. A surface that the curve
       * passes at a small angle is hit well before the ray gets to it. */
      phase = VOLUME_IOR_PHASE_REST;
    }
    else if (phase == VOLUME_IOR_PHASE_REST) {
      /* Nor there: the curve is next to the surface, on one side or the other. Over to it from
       * the ray, a little before its hit: the surface is in between if the curve is behind
       * it. */
      gap_start = O + aim * fmaxf(t_stop - t_start - 0.05f * tolerance, 0.0f);
      phase = VOLUME_IOR_PHASE_SIDE;
    }
    else {
      /* The curve does not hit what the ray has: next to a surface that is curved the other
       * way, as inside of a lens, a ray leaves where the curve stays in. */
      passed = true;
      advance = true;
    }

    if (finished) {
      /* From the start of the chord to the end of the path. */
      const float3 chord_start = (only_ior) ? X_start : chord_X;
      const float chord_start_s = (only_ior) ? s_start : chord_s;
      if (hit) {
        const float3 chord = end - chord_start;
        const float length = len(chord);
        ray->D = (length > 0.0f) ? chord / length : aim;
        ray->P = chord_start - ray->D * chord_start_s;
        ray->tmin = chord_start_s;
        isect->t = chord_start_s + length;
      }
      else {
        /* The ray that was traced, to its end. */
        ray->D = aim;
        ray->P = (only_ior) ? O - aim * t_start : chord_X - aim * chord_s;
        ray->tmin = (only_ior) ? t_start : chord_s;
        isect->prim = PRIM_NONE;
        isect->object = OBJECT_NONE;
      }
      ray->tmax = t_limit;
      ior->D = D_end;
      return (hit) ? VOLUME_IOR_HIT : VOLUME_IOR_MISS;
    }

    if (advance) {
      /* What was traced has no hit before the curve got here: continue with the next ray,
       * which is the last one if the steps have run out. It starts where the path is on this
       * one, or on the curve if this one has hit what the curve does not: that is the side of
       * the surface the path is on. */
      end_last = (passed) ? X : O + aim * dot(X - O, aim);
      has_end = !passed;
      /* Not at the surface or light the path left any more. */
      self_object = OBJECT_NONE;
      self_prim = PRIM_NONE;
      last_prim = PRIM_NONE;
      last_object = OBJECT_NONE;
      last_type = 0;
      phase = VOLUME_IOR_PHASE_PLAN;

      if (!only_ior) {
        /* The media up to here are integrated along the chord first, if the curve is too far
         * from it: half a voxel, for the detail of the media is of that size. Judged by the
         * point between two rays that was the furthest from the chord so far. */
        const float3 chord = X - chord_X;
        chord_length = len(chord);
        if (chord_length > min_step) {
          chord_D = chord / chord_length;
          const float3 R_far = far_X - chord_X;
          const float3 R_last = X_start - chord_X;
          const float far_distance2 = len_squared(R_far - chord_D * dot(R_far, chord_D));
          const float last_distance2 = len_squared(R_last - chord_D * dot(R_last, chord_D));
          if (last_distance2 > far_distance2) {
            far_X = X_start;
          }
          if (fmaxf(far_distance2, last_distance2) >
              sqr(fmaxf(tolerance, VOLUME_IOR_TOLERANCE_MEDIA * voxel_size)))
          {
            /* The path continues from the curve after the chord: across the gap to it
             * first. */
            gap_start = end_last;
            phase = VOLUME_IOR_PHASE_CHORD_GAP;
          }
        }
      }
    }
  }
}

#endif /* __VOLUME_IOR__ */

/* Angle of the footprint of a camera path: that of its ray, which scattering widens only with
 * the filter of glossy reflections, and no less than a fraction of the widest lobe the path was
 * sampled from. The solid angle of a lobe is about the inverse of the density of its samples. */
ccl_device_inline float volume_ior_path_footprint(ConstIntegratorState state, const float dD)
{
#ifdef __VOLUME_IOR__
  const float min_ray_pdf = INTEGRATOR_STATE(state, path, min_ray_pdf);
  if (min_ray_pdf < FLT_MAX && min_ray_pdf > 0.0f) {
    return fmaxf(dD, VOLUME_IOR_FOOTPRINT_LOBE * sqrtf(M_1_PI_F / min_ray_pdf));
  }
#else
  (void)state;
#endif
  return dD;
}

/* Size of the footprint of a pixel at a position, zero for cameras that this is not known of. */
ccl_device_inline float volume_ior_camera_footprint(KernelGlobals kg, const float3 P)
{
  const float pixel = 0.5f * (len(make_float3(kernel_data.cam.dx)) +
                              len(make_float3(kernel_data.cam.dy)));
  if (kernel_data.cam.type == CAMERA_ORTHOGRAPHIC) {
    return pixel;
  }
  if (kernel_data.cam.type == CAMERA_PERSPECTIVE) {
    const float3 camera_P = make_float3(kernel_data.cam.cameratoworld.x.w,
                                        kernel_data.cam.cameratoworld.y.w,
                                        kernel_data.cam.cameratoworld.z.w);
    return pixel * len(P - camera_P);
  }
  return 0.0f;
}

/* Closest intersection for the light subpaths of bidirectional methods and for photons, whose
 * loops keep `ior`, zero at first, between the chords of a bounce. */
ccl_device_inline VolumeIORResult
volume_ior_light_path_intersect(KernelGlobals kg,
                                IntegratorState state,
                                ccl_private Ray *ccl_restrict ray,
                                const PathRayVisibility visibility,
                                ccl_private Intersection *ccl_restrict isect,
                                ccl_private uint *rng,
                                ccl_private VolumeIORState *ior,
                                ccl_private bool *arrive)
{
  *arrive = false;
#ifdef __VOLUME_IOR__
  bool only_ior;
  if (volume_ior_stack_active(kg, state, &only_ior) &&
      (ior->pending || volume_ior_path_bends(state)))
  {
    const float rand = lcg_step_float(rng);
    if (ior->pending) {
      volume_ior_resume(ray, ior->D, FLT_MAX);
    }
    /* The footprint of the pixel that sees the start of the ray stands in for the one the path
     * does not have: what the path lights is seen no sharper than that. */
    ray->dP = volume_ior_camera_footprint(kg, ray->P + ray->D * ray->tmin);
    ray->dD = differential_zero_compact();
    const VolumeIORResult result = volume_ior_intersect(
        kg, state, ray, visibility, isect, rand, ior, nullptr);
    if (result != VOLUME_IOR_SEGMENT) {
      ior->steps = 0;
      if (only_ior) {
        volume_ior_arrive(ray, isect, result == VOLUME_IOR_HIT, ior->D);
      }
      else {
        *arrive = true;
      }
    }
    return result;
  }
  ior->pending = false;
  ior->steps = 0;
#else
  (void)state;
  (void)rng;
  (void)ior;
#endif
  return scene_intersect(kg, ray, visibility, isect) ? VOLUME_IOR_HIT : VOLUME_IOR_MISS;
}

CCL_NAMESPACE_END
