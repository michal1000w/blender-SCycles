/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include "util/defines.h"

CCL_NAMESPACE_BEGIN

/* Conservative acceleration metadata, not a resampled displacement image. Every native pixel
 * contributes to its leaf's interval. Coarser leaves only reduce rejection efficiency. */
class DisplacementImageBounds {
 public:
  int width = 0, height = 0;
  int block_shift = 4;
  int base_width = 0, base_height = 0;
  std::vector<uint32_t> nodes;
  std::vector<size_t> level_offsets;

  static uint32_t pack(const uint16_t lo, const uint16_t hi)
  {
    return uint32_t(lo) | (uint32_t(hi) << 16);
  }

  static uint16_t lower(const uint32_t node)
  {
    return uint16_t(node);
  }

  static uint16_t upper(const uint32_t node)
  {
    return uint16_t(node >> 16);
  }

  static uint32_t merge(const uint32_t a, const uint32_t b)
  {
    return pack(std::min(lower(a), lower(b)), std::max(upper(a), upper(b)));
  }

  template<typename T>
  bool build(const std::span<const T> pixels,
             const int image_width,
             const int image_height,
             const int max_base_dimension = 256)
  {
    static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t>);
    *this = {};
    if (image_width <= 0 || image_height <= 0 || max_base_dimension < 1 ||
        max_base_dimension > 256 || size_t(image_width) > pixels.size() / size_t(image_height))
    {
      return false;
    }
    width = image_width;
    height = image_height;
    size_t block = size_t(1) << block_shift;
    while ((size_t(width) + block - 1) / block > size_t(max_base_dimension) ||
           (size_t(height) + block - 1) / block > size_t(max_base_dimension))
    {
      block_shift++;
      block *= 2;
    }
    base_width = int((size_t(width) + block - 1) / block);
    base_height = int((size_t(height) + block - 1) / block);
    nodes.resize(size_t(base_width) * base_height, pack(UINT16_MAX, 0));
    for (int by = 0; by < base_height; by++) {
      for (int bx = 0; bx < base_width; bx++) {
        uint16_t lo = UINT16_MAX, hi = 0;
        const size_t xend = std::min((size_t(bx) + 1) * block, size_t(width));
        const size_t yend = std::min((size_t(by) + 1) * block, size_t(height));
        for (size_t y = size_t(by) * block; y < yend; y++) {
          for (size_t x = size_t(bx) * block; x < xend; x++) {
            const uint16_t value = uint16_t(pixels[y * size_t(width) + x]) *
                                   (sizeof(T) == 1 ? 257 : 1);
            lo = std::min(lo, value);
            hi = std::max(hi, value);
          }
        }
        nodes[size_t(by) * base_width + bx] = pack(lo, hi);
      }
    }
    level_offsets.push_back(0);
    int w = base_width, h = base_height;
    while (w > 1 || h > 1) {
      const int next_w = (w + 1) / 2, next_h = (h + 1) / 2;
      const size_t source = level_offsets.back(), target = nodes.size();
      nodes.resize(target + size_t(next_w) * next_h, pack(UINT16_MAX, 0));
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          uint32_t &parent = nodes[target + size_t(y / 2) * next_w + x / 2];
          parent = merge(parent, nodes[source + size_t(y) * w + x]);
        }
      }
      level_offsets.push_back(target);
      w = next_w;
      h = next_h;
    }
    return true;
  }

  /* Inclusive native pixel rectangle. The caller handles texture wrapping and filtering
   * footprints before querying. At most four nodes cover the entire rectangle. */
  uint32_t query(int x0, int y0, int x1, int y1) const
  {
    if (nodes.empty() || x0 < 0 || y0 < 0 || x1 >= width || y1 >= height || x0 > x1 || y0 > y1) {
      return pack(0, UINT16_MAX);
    }
    x0 >>= block_shift;
    y0 >>= block_shift;
    x1 >>= block_shift;
    y1 >>= block_shift;
    size_t level = 0;
    int w = base_width;
    while (x1 - x0 > 1 || y1 - y0 > 1) {
      x0 /= 2;
      y0 /= 2;
      x1 /= 2;
      y1 /= 2;
      w = (w + 1) / 2;
      level++;
    }
    uint32_t result = pack(UINT16_MAX, 0);
    for (int y = y0; y <= y1; y++) {
      for (int x = x0; x <= x1; x++) {
        result = merge(result, nodes[level_offsets[level] + size_t(y) * w + x]);
      }
    }
    return result;
  }
};

CCL_NAMESPACE_END
