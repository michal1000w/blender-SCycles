/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

struct GuidingDirectionalProduct {
  enum Type { COSINE, TWO_SIDED_COSINE, PHASE };
  float3 axis;
  float anisotropy;
  Type type;

  ccl_device_inline_method float evaluate(const float3 direction) const
  {
    const float cosine = dot(axis, direction);
    if (type == PHASE) {
      /* The coarse proposal uses the first angular moment of the phase mixture. Clamp its
       * sharpness to the coarse representation; the actual phase evaluation is unchanged. */
      const float g = clamp(anisotropy, -0.95f, 0.95f);
      const float denominator = max(1.0f + g * g - 2.0f * g * cosine, 1e-6f);
      return (1.0f - g * g) / (denominator * sqrtf(denominator));
    }
    return max(type == TWO_SIDED_COSINE ? fabsf(cosine) : cosine, 0.02f);
  }
};

/* A complete directional quadtree over equal-solid-angle (azimuth, cos(theta)) coordinates.
 * The tree is a flat, relocation-free array and is shared by host tests and device queries.
 * Children of node i are 4*i+[1,4]. Leaves are in Morton order.
 *
 * Build into separate storage between render batches. Sampling an array while its weights are
 * changing would invalidate both the distribution and the densities used by bidirectional MIS.
 * The normalized density has units sr^-1, including the 1/(4*pi) mapping Jacobian. */
template<int Levels> struct GuidingDirectionalTree {
  static_assert(Levels > 0 && Levels <= 8);
  ccl_static_constexpr int resolution = 1 << Levels;
  ccl_static_constexpr int leaf_count = 1 << (2 * Levels);
  ccl_static_constexpr int node_count = (4 * leaf_count - 1) / 3;
  ccl_static_constexpr int leaf_offset = node_count - leaf_count;

  ccl_device_inline float2 direction_to_square(const float3 direction)
  {
    float phi = atan2f(direction.y, direction.x);
    if (phi < 0.0f) {
      phi += M_2PI_F;
    }
    return make_float2(min(phi * M_1_2PI_F, 0x1.fffffep-1f),
                       min(clamp(0.5f * (direction.z + 1.0f), 0.0f, 1.0f), 0x1.fffffep-1f));
  }

  ccl_device_inline float3 square_to_direction(const float2 square)
  {
    const float phi = M_2PI_F * square.x;
    const float z = 2.0f * square.y - 1.0f;
    const float r = safe_sqrtf(1.0f - z * z);
    return make_float3(r * cosf(phi), r * sinf(phi), z);
  }

  ccl_device_inline int leaf_index(const float3 direction)
  {
    const float2 square = direction_to_square(direction);
    const int x = int(square.x * resolution);
    const int y = int(square.y * resolution);
    int index = 0;
    for (int level = Levels - 1; level >= 0; --level) {
      index = 4 * index + 1 + ((x >> level) & 1) + 2 * ((y >> level) & 1);
    }
    return index;
  }

  ccl_device_inline float pdf(const ccl_global float *tree, const float3 direction)
  {
    if (!(tree[0] > 0.0f)) {
      return M_1_4PI_F;
    }
    return (tree[leaf_index(direction)] / tree[0]) * (leaf_count * M_1_4PI_F);
  }

  ccl_device_inline float3 sample(const ccl_global float *tree,
                                  float2 random,
                                  ccl_private float *sample_pdf)
  {
    random = clamp(random, zero_float2(), make_float2(0x1.fffffep-1f));
    if (!(tree[0] > 0.0f)) {
      *sample_pdf = M_1_4PI_F;
      return square_to_direction(random);
    }

    return sample_subtree(tree, 0, 0, 0, 0, random, sample_pdf);
  }

  ccl_device_inline float3 sample_subtree(const ccl_global float *tree,
                                          int node,
                                          int x,
                                          int y,
                                          const int start_level,
                                          float2 random,
                                          ccl_private float *sample_pdf)
  {
    for (int level = start_level; level < Levels; ++level) {
      const int first = 4 * node + 1;
      /* Use the same child sum as the builder, avoiding CDF gaps from reassociation with the
       * parent. Rescale the residual uniform variate for the next level and then the leaf. */
      const float total = ((tree[first] + tree[first + 1]) + tree[first + 2]) + tree[first + 3];
      float target = min(random.x * total, __uint_as_float(__float_as_uint(total) - 1));
      int child = 0;
      for (; child < 3; ++child) {
        if (target < tree[first + child]) {
          break;
        }
        target -= tree[first + child];
      }
      node = first + child;
      random.x = min(target / tree[node], 0x1.fffffep-1f);
      x = 2 * x + (child & 1);
      y = 2 * y + (child >> 1);
    }
    *sample_pdf = (tree[node] / tree[0]) * (leaf_count * M_1_4PI_F);
    /* Trigonometric round trips can move an exactly rounded boundary into the adjacent bin.
     * Keep the generated coordinate a few float ulps inside its selected cell, including at
     * the spherical seam and poles, so sample and evaluation use the same discontinuous PDF. */
    constexpr float margin = 4.0f * FLT_EPSILON;
    const float2 lower = make_float2(float(x) / resolution, float(y) / resolution);
    const float2 uv = lower + random / resolution;
    return square_to_direction(
        clamp(uv, lower + make_float2(margin), lower + make_float2(1.0f / resolution - margin)));
  }

  /* A bounded-cost approximation of the radiance/cosine product. Reweight the sixteen
   * second-level cells using their center cosine, then retain the learned distribution
   * inside each cell. This is a normalized piecewise proposal, not an approximation to its
   * evaluated PDF. A positive floor keeps support across cells straddling the horizon.
   * The physical BSDF is still evaluated at the sampled direction by the integrator. */
  ccl_device_inline float product_factor(const int cell, const GuidingDirectionalProduct product)
  {
    const int x = ((cell >> 2) & 1) * 2 + (cell & 1);
    const int y = (cell >> 3) * 2 + ((cell >> 1) & 1);
    const float z = (float(y) + 0.5f) * 0.5f - 1.0f;
    const float xy = (y == 0 || y == 3) ? 0.4677071733f : 0.6846531969f;
    const float3 center = make_float3((x == 0 || x == 3) ? xy : -xy, x < 2 ? xy : -xy, z);
    return product.evaluate(center);
  }

  ccl_device_inline float product_sum(const ccl_global float *tree,
                                      const GuidingDirectionalProduct product)
  {
    static_assert(Levels >= 2);
    float total = 0.0f;
    for (int cell = 0; cell < 16; ++cell) {
      total += tree[5 + cell] * product_factor(cell, product);
    }
    return total;
  }

  ccl_device_inline float pdf_product(const ccl_global float *tree,
                                      const float3 direction,
                                      const GuidingDirectionalProduct product)
  {
    if (!(tree[0] > 0.0f)) {
      return M_1_4PI_F;
    }
    const float2 uv = direction_to_square(direction);
    const int x = int(uv.x * 4);
    const int y = int(uv.y * 4);
    const int cell = ((x >> 1) + 2 * (y >> 1)) * 4 + (x & 1) + 2 * (y & 1);
    return tree[leaf_index(direction)] * product_factor(cell, product) /
           product_sum(tree, product) * (leaf_count * M_1_4PI_F);
  }

  ccl_device_inline float3 sample_product(const ccl_global float *tree,
                                          float2 random,
                                          const GuidingDirectionalProduct product,
                                          ccl_private float *sample_pdf)
  {
    random = clamp(random, zero_float2(), make_float2(0x1.fffffep-1f));
    if (!(tree[0] > 0.0f)) {
      *sample_pdf = M_1_4PI_F;
      return square_to_direction(random);
    }
    float weights[16];
    float total = 0.0f;
    for (int cell = 0; cell < 16; ++cell) {
      weights[cell] = tree[5 + cell] * product_factor(cell, product);
      total += weights[cell];
    }
    float target = min(random.x * total, __uint_as_float(__float_as_uint(total) - 1));
    int cell = 0;
    for (; cell < 15; ++cell) {
      if (target < weights[cell]) {
        break;
      }
      target -= weights[cell];
    }
    random.x = min(target / weights[cell], 0x1.fffffep-1f);
    const int x = ((cell >> 2) & 1) * 2 + (cell & 1);
    const int y = (cell >> 3) * 2 + ((cell >> 1) & 1);
    const float3 direction = sample_subtree(tree, 5 + cell, x, y, 2, random, sample_pdf);
    *sample_pdf *= tree[0] * product_factor(cell, product) / total;
    return direction;
  }

  /* Leaf inputs are nonnegative radiance estimates integrated over each directional bin.
   * Normalize before summing to avoid overflow. The uniform component preserves exploration
   * even for bins that training has not visited. An entirely empty field remains uniform. */
  ccl_device_inline void build(ccl_global float *tree, const float exploration)
  {
    float maximum = 0.0f;
    for (int i = leaf_offset; i < node_count; ++i) {
      tree[i] = (isfinite_safe(tree[i]) && tree[i] > 0.0f) ? tree[i] : 0.0f;
      maximum = max(maximum, tree[i]);
    }
    if (!(maximum > 0.0f)) {
      for (int i = 0; i < leaf_offset; ++i) {
        tree[i] = 0.0f;
      }
      return;
    }
    float sum = 0.0f;
    for (int i = leaf_offset; i < node_count; ++i) {
      tree[i] /= maximum;
      sum += tree[i];
    }
    const float uniform_weight = clamp(exploration, 0.0f, 1.0f) * sum / leaf_count;
    for (int i = leaf_offset; i < node_count; ++i) {
      tree[i] = (1.0f - clamp(exploration, 0.0f, 1.0f)) * tree[i] + uniform_weight;
    }
    for (int i = leaf_offset - 1; i >= 0; --i) {
      const int first = 4 * i + 1;
      tree[i] = ((tree[first] + tree[first + 1]) + tree[first + 2]) + tree[first + 3];
    }
  }

  /* Sparse directional regions use a coarser constant density until enough training records
   * reach them. Collapsing a subtree preserves its integrated mass exactly. This regularizes
   * the proposal only; actual BSDF/phase evaluation and Monte Carlo weights are unchanged.
   * Count leaves are retained for the next training batch; inner counts are caller-owned
   * scratch space. Negative child masses mark collapsed descendants during the forward pass. */
  ccl_device_inline void build_adaptive(ccl_global float *tree,
                                        ccl_global float *counts,
                                        const float minimum_records,
                                        const float exploration)
  {
    build(tree, 0.0f);
    for (int i = leaf_offset - 1; i >= 0; --i) {
      const int first = 4 * i + 1;
      counts[i] = ((counts[first] + counts[first + 1]) + counts[first + 2]) + counts[first + 3];
    }
    for (int i = 0; i < leaf_offset; ++i) {
      const bool collapse = tree[i] < 0.0f || counts[i] < minimum_records;
      tree[i] = fabsf(tree[i]);
      if (collapse) {
        const int first = 4 * i + 1;
        for (int child = 0; child < 4; ++child) {
          tree[first + child] = -0.25f * tree[i];
        }
      }
    }
    for (int i = leaf_offset; i < node_count; ++i) {
      tree[i] = fabsf(tree[i]);
    }
    build(tree, exploration);
  }
};

CCL_NAMESPACE_END
