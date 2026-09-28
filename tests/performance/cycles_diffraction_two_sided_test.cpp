/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#define ccl_device_inline inline
#define ccl_private
constexpr float M_1_PI_F = 0.31830988618379067154f;
static inline bool isfinite_safe(float x) { return std::isfinite(x); }
static inline float sqr(float x) { return x * x; }
static inline float max(float x, float y) { return x > y ? x : y; }

#include "kernel/util/diffraction_two_sided.h"

static int checks = 0;
static void check(bool condition, const char *name)
{
  checks++;
  if (!condition) {
    std::fprintf(stderr, "failed: %s\n", name);
    std::exit(1);
  }
}

int main()
{
  for (double q0 : {0.02, 0.2, 1.0, 3.0}) {
    for (double q1 : {0.01, 0.3, 2.0, 4.0}) {
      for (double cross : {0.0, 0.1, 0.5, 1.0}) {
        ccl::DiffractionTwoSidedReturn r;
        check(ccl::diffraction_two_sided_return(float(q0), float(q1), float(cross), &r),
              "valid return");
        for (int side = 0; side < 2; side++) {
          const double sum = r.c[side][0] * q0 + r.c[side][1] * q1;
          check(std::abs(sum - 1.0) < 3e-7, "both side energy rows");
          check(r.c[side][0] >= 0 && r.c[side][1] >= 0, "nonnegative coefficients");
          const double probability =
              ccl::diffraction_two_sided_side_probability(r, side, 0) +
              ccl::diffraction_two_sided_side_probability(r, side, 1);
          check(std::abs(probability - 1.0) < 3e-7, "sampled side probabilities");
        }
        check(r.c[0][1] == r.c[1][0], "reciprocal cross coefficient");
        const float ni[2] = {1.0f, 1.5f};
        const float qi[2] = {0.35f, 0.41f};
        const float co[2] = {0.78f, 0.62f};
        const double eval01 = ccl::diffraction_two_sided_eval(r, 0, 1, qi[0], qi[1], ni[1], co[1]);
        const double eval10 = ccl::diffraction_two_sided_eval(r, 1, 0, qi[1], qi[0], ni[0], co[0]);
        const double lhs = ni[0] * ni[0] * co[0] * eval01;
        const double rhs = ni[1] * ni[1] * co[1] * eval10;
        check(std::abs(lhs - rhs) < 5e-7 * std::fmax(1.0, std::fabs(lhs)),
              "radiance adjoint reciprocity");
        for (int side = 0; side < 2; side++) {
          const float pdf = ccl::diffraction_two_sided_pdf(r, 0, side, co[side]);
          check(std::isfinite(pdf) && pdf >= 0.0f, "sample pdf finite");
        }
      }
    }
  }
  ccl::DiffractionTwoSidedReturn one_sided;
  check(ccl::diffraction_two_sided_return(0.0f, 0.4f, 1.0f, &one_sided),
        "one-side zero deficit");
  check(one_sided.c[0][0] == 0 && one_sided.c[0][1] == 0 &&
            std::abs(one_sided.c[1][1] * 0.4f - 1.0f) < 1e-6f,
        "one-side zero limit");
  check(!ccl::diffraction_two_sided_return(0.0f, 0.0f, 0.5f, &one_sided),
        "specular limit has no return lobe");
  std::printf("two-sided reciprocal return: %d checks passed\n", checks);
}
