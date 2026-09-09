/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/sample/guiding_distribution.h"
#include "kernel/sample/guiding_spherical_gaussian.h"
#include "kernel/types_guiding.h"
#include "util/atomic.h"

CCL_NAMESPACE_BEGIN

enum GuidingFieldType {
  /* Six dominant-normal sectors keep opposing sides of thin surfaces separate even when
   * they occupy the same spatial cell. Volume transport has no surface orientation. */
  GUIDING_FIELD_SURFACE_RADIANCE = 0,
  GUIDING_FIELD_VOLUME_RADIANCE = 6,
  GUIDING_FIELD_SURFACE_IMPORTANCE = 7,
  GUIDING_FIELD_VOLUME_IMPORTANCE = 13,
  GUIDING_FIELD_TYPES = 14,
};

ccl_device_inline GuidingFieldType guiding_surface_field_type(const float3 normal,
                                                              const bool importance)
{
  const float3 magnitude = fabs(normal);
  const int axis = magnitude.x >= magnitude.y && magnitude.x >= magnitude.z ? 0 :
                   magnitude.y >= magnitude.z                               ? 1 :
                                                                              2;
  const int sector = 2 * axis + int(normal[axis] < 0.0f);
  return GuidingFieldType(
      (importance ? GUIDING_FIELD_SURFACE_IMPORTANCE : GUIDING_FIELD_SURFACE_RADIANCE) + sector);
}

/* A bounded, adaptively refined binary spatial tree with separate radiance and adjoint fields.
 *
 * Accumulation arrays are writable while the immutable sampling arrays are in use. Topology
 * changes and publication require a drained queue. Child nodes inherit half the accumulated
 * parent observations; later observations specialize them independently. Refinement changes a
 * proposal, never a rendered contribution, so inherited data cannot bias the image estimator.
 * All storage is caller-owned and contains no host pointers or dynamic allocations. */
struct GuidingField {
  using DirectionalTree = GuidingDirectionalTree<5>;
  ccl_static_constexpr uint bins = DirectionalTree::leaf_count;
  ccl_static_constexpr uint tree_size = DirectionalTree::node_count;
  ccl_static_constexpr uint sampling_size = tree_size + GuidingGaussianMixture::storage_size;
  /* Radiance-bin sums, a tree of contribution counts, and joint position/direction moments per
   * Gaussian component. Only count leaves are updated while tracing; inner counts are
   * scratch space during publication. */
  ccl_static_constexpr uint moments_offset = bins + tree_size;
  ccl_static_constexpr uint parallax_offset = moments_offset +
                                              GuidingGaussianMixture::components *
                                                  GuidingPositionMoments::storage_size;
  ccl_static_constexpr uint accumulation_size = parallax_offset +
                                                GuidingGaussianMixture::components *
                                                    GuidingParallaxMoments::storage_size;
  ccl_static_constexpr uint max_depth = 20;

  ccl_global GuidingSpatialNode *nodes;
  ccl_global float *accumulation;
  ccl_global float *sampling;
  /* allocated node count, followed by the fixed count at the start of refinement. */
  ccl_global uint *counts;
  uint capacity;
  float3 bounds_min;
  float3 bounds_max;

  ccl_device_inline_method uint find_leaf(const float3 P) const
  {
    uint index = 0;
    for (uint depth = 0; depth < max_depth; ++depth) {
      const ccl_global GuidingSpatialNode *node = &nodes[index];
      if (node->children == 0) {
        break;
      }
      index = node->children + uint(P[node->axis] >= node->split);
    }
    return index;
  }

  ccl_device_inline_method uint record_index(const uint node,
                                             const GuidingFieldType type,
                                             const float3 direction) const
  {
    DirectionalTree distribution;
    return (node * GUIDING_FIELD_TYPES + uint(type)) * accumulation_size +
           distribution.leaf_index(direction) - DirectionalTree::leaf_offset;
  }

  ccl_device_inline_method const ccl_global float *distribution(const uint node,
                                                                const GuidingFieldType type) const
  {
    return sampling + (node * GUIDING_FIELD_TYPES + uint(type)) * sampling_size;
  }

  ccl_device_inline_method void record_visit(const uint node) const
  {
    atomic_fetch_and_add_uint32(&nodes[node].visits, 1);
  }

  ccl_device_inline_method float3 normalized_position(const float3 P) const
  {
    return (P - bounds_min) / max(bounds_max - bounds_min, make_float3(1e-8f));
  }

  ccl_device_inline_method float scene_scale() const
  {
    const float3 extent = max(bounds_max - bounds_min, make_float3(1e-8f));
    return max(extent.x, max(extent.y, extent.z));
  }

  ccl_device_inline_method float3 metric_extent() const
  {
    return max(bounds_max - bounds_min, make_float3(1e-8f)) / scene_scale();
  }

  ccl_device_inline_method bool squared_moment(const uint offset) const
  {
    if (offset >= parallax_offset) {
      GuidingParallaxMoments model;
      return model.squared_weight_entry((offset - parallax_offset) % model.storage_size);
    }
    return offset >= moments_offset &&
           (offset - moments_offset) % GuidingPositionMoments::storage_size == 22;
  }

  ccl_device_inline_method void record_source(const uint index,
                                              const float weight,
                                              const float inverse_distance_weight,
                                              const float distance_weight,
                                              const float3 direction,
                                              const float3 position) const
  {
    const uint distribution = index / accumulation_size;
    const uint component = (index % accumulation_size) /
                           (bins / GuidingGaussianMixture::components);
    GuidingParallaxMoments source;
    source.record_aggregate(accumulation + distribution * accumulation_size + parallax_offset +
                                component * source.storage_size,
                            weight,
                            inverse_distance_weight * scene_scale(),
                            distance_weight / scene_scale(),
                            position * metric_extent(),
                            direction);
  }

  ccl_device_inline_method void record(const uint index,
                                       const float value,
                                       const float3 direction,
                                       const float3 position = zero_float3(),
                                       const float distance = FLT_MAX) const
  {
    if (value > 0.0f && isfinite_safe(value)) {
      atomic_add_and_fetch_float(&accumulation[index], value);
      atomic_add_and_fetch_float(&accumulation[index + bins + DirectionalTree::leaf_offset], 1.0f);
      const uint distribution = index / accumulation_size;
      const uint bin = index % accumulation_size;
      const uint component = bin / (bins / GuidingGaussianMixture::components);
      ccl_global float *moments = accumulation + distribution * accumulation_size +
                                  moments_offset +
                                  GuidingPositionMoments::storage_size * component;
      GuidingPositionMoments model;
      model.record(moments, value, position, direction);
      if (distance > 0.0f && distance < FLT_MAX && isfinite_safe(distance)) {
        record_source(index, value, value / distance, value * distance, direction, position);
      }
    }
  }

  /* Called once after rendering has drained, before launching the refinement kernel. */
  ccl_device_inline_method void begin_update() const
  {
    counts[1] = counts[0];
  }

  /* Use the observed spatial spread instead of splitting empty dimensions of a
   * surface cell. These are proposal statistics; they never scale transport.
   * Keep the geometric midpoint fallback for sparse/degenerate observations. */
  ccl_device_inline_method void spatial_split(const uint index,
                                              const float3 lower,
                                              const float3 upper,
                                              ccl_private uint *axis,
                                              ccl_private float *split) const
  {
    const float3 extent = upper - lower;
    *axis = extent.x >= extent.y && extent.x >= extent.z ? 0u : extent.y >= extent.z ? 1u : 2u;
    *split = lower[*axis] + 0.5f * extent[*axis];
    const float3 normalized_lower = normalized_position(lower);
    const float3 normalized_upper = normalized_position(upper);
    float scale = 0;
    for (uint type = 0; type < GUIDING_FIELD_TYPES; ++type) {
      for (int component = 0; component < GuidingGaussianMixture::components; ++component) {
        const ccl_global float *m = accumulation +
                                    (index * GUIDING_FIELD_TYPES + type) * accumulation_size +
                                    moments_offset +
                                    component * GuidingPositionMoments::storage_size;
        if (isfinite_safe(m[0])) {
          scale = max(scale, m[0]);
        }
      }
    }
    if (!(scale > 0)) {
      return;
    }
    float weight = 0, squared_weight = 0;
    float3 first = zero_float3(), second = zero_float3();
    for (uint type = 0; type < GUIDING_FIELD_TYPES; ++type) {
      for (int component = 0; component < GuidingGaussianMixture::components; ++component) {
        const ccl_global float *m = accumulation +
                                    (index * GUIDING_FIELD_TYPES + type) * accumulation_size +
                                    moments_offset +
                                    component * GuidingPositionMoments::storage_size;
        if (!(m[0] > 0) || !isfinite_safe(m[0]) || !(m[22] > 0) || !isfinite_safe(m[22])) {
          continue;
        }
        const float3 mean = make_float3(m[4], m[5], m[6]) / m[0];
        const float3 square = make_float3(m[7], m[10], m[12]) / m[0];
        /* Children inherit parent moments. An inherited component centered
         * outside this cell is not evidence for where to split this cell. */
        if (!isfinite_safe(mean) || !isfinite_safe(square) ||
            reduce_min(mean - normalized_lower) < 0 || reduce_min(normalized_upper - mean) < 0)
        {
          continue;
        }
        const float w = m[0] / scale;
        weight += w;
        squared_weight += (m[22] / scale) / scale;
        first += w * mean;
        second += w * square;
      }
    }
    if (!(weight > 0) || !(squared_weight > 0) || weight / sqrtf(squared_weight) < 8.0f) {
      return;
    }
    const float3 mean = first / weight;
    const float3 normalized_extent = normalized_upper - normalized_lower;
    const float3 roundoff = 4.0f * FLT_EPSILON * (fabs(second / weight) + mean * mean);
    const float3 variance = min(max(second / weight - mean * mean - roundoff, zero_float3()),
                                0.25f * normalized_extent * normalized_extent);
    const float3 scene_extent = bounds_max - bounds_min;
    const float3 metric_variance = variance * scene_extent * scene_extent;
    if (!isfinite_safe(metric_variance) || !(reduce_max(metric_variance) > 0)) {
      return;
    }
    *axis = metric_variance.x >= metric_variance.y && metric_variance.x >= metric_variance.z ? 0u :
            metric_variance.y >= metric_variance.z                                           ? 1u :
                                                                                               2u;
    /* Avoid arbitrarily thin children when a bright cluster lies at a boundary.
     * Both children retain the full-support parent proposal before specializing. */
    *split = clamp(bounds_min[*axis] + mean[*axis] * scene_extent[*axis],
                   lower[*axis] + 0.1f * extent[*axis],
                   upper[*axis] - 0.1f * extent[*axis]);
  }

  /* At most one invocation per pre-existing node. Child initialization completes before the
   * separate publication kernel; neither new children nor queries execute in this launch. */
  ccl_device_inline_method void refine(const uint index, const uint split_threshold) const
  {
    if (index >= counts[1]) {
      return;
    }
    ccl_global GuidingSpatialNode *node = &nodes[index];
    if (node->children != 0 || node->visits < split_threshold || node->depth >= max_depth) {
      return;
    }
    /* Failed reservations are rolled back. Only this kernel observes the transient count;
     * publication runs after all reservations complete. Failed indices never access storage. */
    const uint first = atomic_fetch_and_add_uint32(&counts[0], 2);
    if (first + 2 > capacity) {
      atomic_fetch_and_add_uint32(&counts[0], -2);
      return;
    }

    /* Recover this node's bounds from the immutable ancestral splits. Other leaves can refine
     * concurrently but cannot change any ancestor of this node. */
    float3 lower = bounds_min;
    float3 upper = bounds_max;
    uint current = index;
    while (current != 0) {
      const uint parent = nodes[current].parent;
      const ccl_global GuidingSpatialNode *ancestor = &nodes[parent];
      const uint axis = ancestor->axis;
      if (current == ancestor->children) {
        upper[axis] = min(upper[axis], ancestor->split);
      }
      else {
        lower[axis] = max(lower[axis], ancestor->split);
      }
      current = parent;
    }
    uint axis;
    float split;
    spatial_split(index, lower, upper, &axis, &split);
    node->axis = axis;
    node->split = split;
    for (uint child = first; child < first + 2; ++child) {
      nodes[child].children = 0;
      nodes[child].parent = index;
      nodes[child].axis = 0;
      nodes[child].split = 0.0f;
      nodes[child].depth = node->depth + 1;
      nodes[child].visits = node->visits / 2;
      for (uint i = 0; i < GUIDING_FIELD_TYPES * sampling_size; ++i) {
        sampling[child * GUIDING_FIELD_TYPES * sampling_size + i] =
            sampling[index * GUIDING_FIELD_TYPES * sampling_size + i];
      }
      for (uint i = 0; i < GUIDING_FIELD_TYPES * accumulation_size; ++i) {
        const uint offset = i % accumulation_size;
        const bool squared_weight = squared_moment(offset);
        accumulation[child * GUIDING_FIELD_TYPES * accumulation_size + i] =
            (squared_weight ? 0.25f : 0.5f) *
            accumulation[index * GUIDING_FIELD_TYPES * accumulation_size + i];
      }
    }
    node->children = first;
  }

  /* One invocation per node/type, in a kernel after refinement. */
  ccl_device_inline_method void publish(const uint index, const float exploration) const
  {
    if (index >= counts[0] * GUIDING_FIELD_TYPES) {
      return;
    }
    ccl_global float *tree = sampling + index * sampling_size;
    ccl_global float *observations = accumulation + index * accumulation_size;
    ccl_global float *counts = observations + bins;
    for (uint bin = 0; bin < bins; ++bin) {
      tree[DirectionalTree::leaf_offset + bin] = observations[bin];
      /* Refined children initially share their parent's observations. Retain a short history
       * so samples gathered in the child's own spatial region can replace that inherited fit.
       * Unvisited directions retain support through the publication's exploration mixture. */
      observations[bin] *= 0.25f;
    }
    DirectionalTree distribution;
    distribution.build_adaptive(tree, counts, 32.0f, exploration);
    GuidingGaussianMixture smooth;
    smooth.build(tree + tree_size,
                 tree,
                 observations + moments_offset,
                 counts,
                 observations + parallax_offset,
                 metric_extent());
    for (uint bin = 0; bin < bins; ++bin) {
      counts[DirectionalTree::leaf_offset + bin] *= 0.25f;
    }
    for (uint i = moments_offset; i < accumulation_size; ++i) {
      const bool squared_weight = squared_moment(i);
      observations[i] *= squared_weight ? 0.0625f : 0.25f;
    }
  }
};

CCL_NAMESPACE_END
