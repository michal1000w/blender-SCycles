/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Two stratified candidates, one from each proposal. The returned reciprocal contribution
 * weight is stochastic; it is not the marginal density of the selected direction. MIS must
 * use a separately evaluated, deterministic partition of unity. */
struct GuidingResamplingPair {
  float target[2];
  float proposal[2];

  ccl_device_inline_method int sample(const float random, ccl_private float *effective_pdf) const
  {
    const float w0 = target[0] > 0.0f && proposal[0] > 0.0f ? target[0] / proposal[0] : 0.0f;
    const float w1 = target[1] > 0.0f && proposal[1] > 0.0f ? target[1] / proposal[1] : 0.0f;
    const float scale = max(w0, w1);
    *effective_pdf = 0.0f;
    if (!(scale > 0.0f) || !isfinite_safe(scale)) {
      return -1;
    }
    /* Scaling avoids overflow when adding two large, individually finite weights. */
    const float a = w0 / scale;
    const float b = w1 / scale;
    const float sum = a + b;
    const int selected = random * sum < a ? 0 : 1;
    *effective_pdf = (target[selected] / scale) * (2.0f / sum);
    return *effective_pdf > 0.0f && isfinite_safe(*effective_pdf) ? selected : -1;
  }
};

CCL_NAMESPACE_END
