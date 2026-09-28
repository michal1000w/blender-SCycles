/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "util/math.h"
CCL_NAMESPACE_BEGIN

struct DiffractionOrderSample {
  int port;
  float probability;
  float throughput;
};

/* Sample an incoherent power column. The discrete probability is P_m/sum(P),
 * while the path throughput multiplier is sum(P), preserving absorption.
 * This does not apply the radiance/importance refractive-index conversion,
 * direction Jacobians or polarization evolution: those belong to the BSDF.
 * No power clamp or passivity correction is performed. Invalid powers and an
 * all-zero column fail; zero-power ports cannot be selected. */
template<typename PowerPointer>
ccl_device_inline bool diffraction_sample_order(const PowerPointer powers,
                                                const int ports,
                                                const float random,
                                                ccl_private DiffractionOrderSample *sample)
{
  if (ports <= 0 || !isfinite_safe(random) || random < 0 || random >= 1)
    return false;
  float total = 0;
  int last = -1;
  for (int i = 0; i < ports; i++) {
    const float power = powers[i];
    if (!isfinite_safe(power) || power < 0)
      return false;
    total += power;
    if (power > 0)
      last = i;
  }
  if (!(total > 0) || !isfinite_safe(total))
    return false;
  const float target = random * total;
  float cumulative = 0;
  for (int i = 0; i < ports; i++) {
    const float power = powers[i];
    cumulative += power;
    /* The last positive bin owns an endpoint rounded onto total. */
    if (power > 0 && (target < cumulative || i == last)) {
      sample->port = i;
      sample->probability = power / total;
      sample->throughput = total;
      return sample->probability > 0 && isfinite_safe(sample->probability);
    }
  }
  return false;
}
CCL_NAMESPACE_END
