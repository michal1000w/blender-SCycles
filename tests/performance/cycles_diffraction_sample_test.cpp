/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/util/diffraction_sample.h"
#include <iostream>
#include <limits>
using namespace ccl;
int main()
{
  const float powers[] = {0, 0.125f, 0.25f, 0, 0.125f};
  int counts[5] = {}, failures = 0;
  constexpr int samples = 65536;
  double estimates[5] = {};
  for (int i = 0; i < samples; i++) {
    DiffractionOrderSample sample;
    if (!diffraction_sample_order(powers, 5, (i + 0.5f) / samples, &sample)) {
      failures++;
      continue;
    }
    if (sample.port < 0 || sample.port >= 5) {
      failures++;
      continue;
    }
    counts[sample.port]++;
    estimates[sample.port] += sample.throughput / samples;
    failures += sample.throughput != 0.5f || sample.probability != powers[sample.port] / 0.5f;
  }
  for (int i = 0; i < 5; i++)
    failures += estimates[i] != powers[i];
  DiffractionOrderSample sample;
  const float zero[] = {0, 0}, negative[] = {0.5f, -0.1f};
  const float nan[] = {0.5f, std::numeric_limits<float>::quiet_NaN()};
  const float gain[] = {0.75f, 0.75f};
  failures += diffraction_sample_order(zero, 2, 0.5f, &sample);
  failures += diffraction_sample_order(negative, 2, 0.5f, &sample);
  failures += diffraction_sample_order(nan, 2, 0.5f, &sample);
  failures += diffraction_sample_order(powers, 5, 1.0f, &sample);
  failures += !diffraction_sample_order(powers, 5, 0.0f, &sample) || sample.port != 1;
  failures += !diffraction_sample_order(powers, 5, std::nextafter(1.0f, 0.0f), &sample) ||
              sample.port != 4;
  failures += !diffraction_sample_order(gain, 2, 0.5f, &sample) || sample.throughput != 1.5f;
  std::cout << "{\"samples\":" << samples << ",\"failures\":" << failures << ",\"counts\":["
            << counts[0] << ',' << counts[1] << ',' << counts[2] << ',' << counts[3] << ','
            << counts[4] << "]}\n";
  return failures ? 1 : 0;
}
