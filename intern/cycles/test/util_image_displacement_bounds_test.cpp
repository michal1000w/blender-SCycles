/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <gtest/gtest.h>

#include "util/image_displacement_bounds.h"

CCL_NAMESPACE_BEGIN

TEST(image_displacement_bounds, EveryRectangleContainsNativeExtrema)
{
  for (const auto [w, h] : {std::pair{1, 1}, {17, 9}, {4097, 3}, {37, 65}}) {
    std::vector<uint16_t> pixels(size_t(w) * h);
    uint32_t state = 19;
    auto random = [&]() { return state = 1664525u * state + 1013904223u; };
    for (auto &pixel : pixels) {
      pixel = uint16_t(random() >> 16);
    }
    pixels.front() = 0;
    pixels.back() = UINT16_MAX;
    DisplacementImageBounds bounds;
    ASSERT_TRUE(bounds.build<uint16_t>(pixels, w, h, 8));
    EXPECT_LE(bounds.base_width, 8);
    EXPECT_LE(bounds.base_height, 8);
    const auto extrema = std::minmax_element(pixels.begin(), pixels.end());
    EXPECT_EQ(bounds.lower(bounds.nodes.back()), *extrema.first);
    EXPECT_EQ(bounds.upper(bounds.nodes.back()), *extrema.second);
    EXPECT_EQ(bounds.upper(bounds.query(w - 1, h - 1, w - 1, h - 1)), UINT16_MAX);
    for (int trial = 0; trial < 200; trial++) {
      int x0 = int(random() % w), x1 = int(random() % w);
      int y0 = int(random() % h), y1 = int(random() % h);
      if (x0 > x1) {
        std::swap(x0, x1);
      }
      if (y0 > y1) {
        std::swap(y0, y1);
      }
      const uint32_t range = bounds.query(x0, y0, x1, y1);
      for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
          EXPECT_LE(bounds.lower(range), pixels[size_t(y) * w + x]);
          EXPECT_GE(bounds.upper(range), pixels[size_t(y) * w + x]);
        }
      }
    }
  }
}

TEST(image_displacement_bounds, ByteNormalizationAndInvalidInput)
{
  const std::vector<uint8_t> pixels{0, 1, 100, 255};
  DisplacementImageBounds bounds;
  ASSERT_TRUE(bounds.build<uint8_t>(pixels, 2, 2));
  EXPECT_EQ(bounds.lower(bounds.nodes.back()), 0);
  EXPECT_EQ(bounds.upper(bounds.nodes.back()), UINT16_MAX);
  EXPECT_EQ(bounds.query(-1, 0, 1, 1), bounds.pack(0, UINT16_MAX));
  EXPECT_FALSE(bounds.build<uint8_t>(pixels, 3, 2));
  EXPECT_TRUE(bounds.nodes.empty());
  EXPECT_FALSE(bounds.build<uint8_t>(pixels, 0, 1));
  EXPECT_FALSE(bounds.build<uint8_t>(pixels, 2, 2, 0));
}

TEST(image_displacement_bounds, ContinuousBilinearFootprints)
{
  using Extension = DisplacementImageBounds::Extension;
  for (const auto [w, h] : {std::pair{1, 1}, {37, 65}, {2053, 9}, {9, 2063}}) {
    std::vector<uint16_t> pixels(size_t(w) * h);
    uint32_t state = 71;
    auto random = [&]() { return state = 1664525u * state + 1013904223u; };
    for (auto &pixel : pixels) {
      pixel = uint16_t(random() >> 16);
    }
    DisplacementImageBounds bounds;
    ASSERT_TRUE(bounds.build<uint16_t>(pixels, w, h));
    for (const Extension extension : {Extension::Repeat, Extension::Extend, Extension::Clip}) {
      auto texel = [&](int x, int y) -> double {
        if (extension == Extension::Repeat) {
          x = ((x % w) + w) % w;
          y = ((y % h) + h) % h;
        }
        else if (extension == Extension::Clip && (x < 0 || y < 0 || x >= w || y >= h)) {
          return 0.0;
        }
        x = std::clamp(x, 0, w - 1);
        y = std::clamp(y, 0, h - 1);
        return pixels[size_t(y) * w + x];
      };
      for (int trial = 0; trial < 1000; trial++) {
        const double u0 = double(random() % 50000) / 10000.0 - 2.0;
        const double v0 = double(random() % 50000) / 10000.0 - 2.0;
        const double du = double(random() % 100) / 1000.0;
        const double dv = double(random() % 100) / 1000.0;
        const uint32_t range = bounds.query_linear(u0, v0, u0 + du, v0 + dv, extension);
        for (int sample = 0; sample < 10; sample++) {
          const double x = (u0 + du * double(random() % 1001) / 1000.0) * w - 0.5;
          const double y = (v0 + dv * double(random() % 1001) / 1000.0) * h - 0.5;
          const int ix = int(std::floor(x)), iy = int(std::floor(y));
          const double fx = x - ix, fy = y - iy;
          const double value = (1 - fy) * ((1 - fx) * texel(ix, iy) + fx * texel(ix + 1, iy)) +
                               fy * ((1 - fx) * texel(ix, iy + 1) + fx * texel(ix + 1, iy + 1));
          EXPECT_LE(bounds.lower(range), value + 1.0e-9);
          EXPECT_GE(bounds.upper(range), value - 1.0e-9);
        }
      }
    }
    EXPECT_EQ(bounds.query_linear(-1.0e100, 0, 0, 1, Extension::Repeat),
              bounds.pack(0, UINT16_MAX));
  }
}

CCL_NAMESPACE_END
