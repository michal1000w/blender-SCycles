/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/util/diffraction_scene_data.h"
#ifndef DIFFRACTION_SCENE_LOWER_INDEX
#  define DIFFRACTION_SCENE_LOWER_INDEX 1.5f
#  define DIFFRACTION_SCENE_CACHE_COUNT 68
#  define DIFFRACTION_SCENE_SYNTHETIC 1
#endif
kernel void diffraction_scene_sample_test(device const int4 *descriptors [[buffer(0)]],
                                          device const float4 *domains [[buffer(1)]],
                                          device const int4 *nodes [[buffer(2)]],
                                          device const int4 *layout [[buffer(3)]],
                                          device const float4 *bounds [[buffer(4)]],
                                          device const int2 *ports [[buffer(5)]],
                                          device const int *active [[buffer(6)]],
                                          device const float2 *matrices [[buffer(7)]],
                                          device const float4 *inputs [[buffer(8)]],
                                          device float4 *output [[buffer(9)]],
                                          device const float4 *reverse_checks [[buffer(10)]],
                                          uint i [[thread_position_in_grid]])
{
  DiffractionSceneData data;
  data.descriptors = descriptors;
  data.domains = domains;
  data.nodes = nodes;
  data.layout = layout;
  data.bounds = bounds;
  data.ports = ports;
  data.active = active;
  data.matrices = matrices;
  data.cache_count = DIFFRACTION_SCENE_CACHE_COUNT;
  const float4 ray = inputs[2 * i], parameters = inputs[2 * i + 1];
  DiffractionSceneSample result;
  bool valid = diffraction_data_sample<20>(&data,
                                           int(parameters.x),
                                           ray.xyz,
                                           bool(parameters.y),
                                           1,
                                           DIFFRACTION_SCENE_LOWER_INDEX,
                                           parameters.w,
                                           ray.w,
                                           parameters.z,
                                           &result);
#ifndef DIFFRACTION_SCENE_SAMPLE_ONLY
  if (valid) {
    float power, probability;
    valid = diffraction_data_order_probability<20>(&data,
                                                   int(parameters.x),
                                                   ray.xyz,
                                                   bool(parameters.y),
                                                   1,
                                                   DIFFRACTION_SCENE_LOWER_INDEX,
                                                   parameters.w,
                                                   ray.w,
                                                   result.relative_order,
                                                   bool(parameters.y) != result.transmission,
                                                   &power,
                                                   &probability) &&
            fabsf(probability - result.probability) < 2e-6f &&
            fabsf(power - result.probability * result.throughput) < 2e-6f;
    if (valid)
      valid = diffraction_data_order_probability<20>(&data,
                                                     int(parameters.x),
                                                     ray.xyz,
                                                     bool(parameters.y),
                                                     1,
                                                     DIFFRACTION_SCENE_LOWER_INDEX,
                                                     parameters.w,
                                                     ray.w,
                                                     2147483647,
                                                     false,
                                                     &power,
                                                     &probability) &&
              power == 0 && probability == 0;
  }
  if (valid) {
    float power, probability;
    valid = diffraction_data_direction_probability<20>(&data, int(parameters.x), ray.xyz,
              bool(parameters.y), 1, DIFFRACTION_SCENE_LOWER_INDEX, parameters.w, ray.w, result.direction,
              &power, &probability) &&
            fabsf(probability - result.probability) < 2e-6f &&
            fabsf(power - result.probability * result.throughput) < 2e-6f;
  }
  /* Synthetic handles 0/1 are opposite unfurled domains; 2/3 are mirrored
   * domains supporting both directions. Each has reciprocal power 0.5. */
  if (valid && DIFFRACTION_SCENE_SYNTHETIC && int(parameters.x) < 4) {
    const int reverse_handle = int(parameters.x) < 2 ? (int(parameters.x) ^ 1) : int(parameters.x);
    float reverse_power, reverse_probability;
    valid = diffraction_data_reverse_probability<20>(&data,
                                                     reverse_handle,
                                                     &result,
                                                     bool(parameters.y),
                                                     1,
                                                     DIFFRACTION_SCENE_LOWER_INDEX,
                                                     parameters.w,
                                                     ray.w,
                                                     &reverse_power,
                                                     &reverse_probability) &&
            fabsf(reverse_power - 0.5f) < 2e-6f && fabsf(reverse_probability - 1) < 2e-6f;
  }
  if (valid && reverse_checks[i].w != 0) {
    float power, probability;
    valid = diffraction_data_reverse_probability<20>(&data,
                                                     int(reverse_checks[i].x),
                                                     &result,
                                                     bool(parameters.y),
                                                     1,
                                                     DIFFRACTION_SCENE_LOWER_INDEX,
                                                     parameters.w,
                                                     ray.w,
                                                     &power,
                                                     &probability) &&
            fabsf(power - reverse_checks[i].y) < 2e-6f &&
            fabsf(probability - reverse_checks[i].z) < 2e-6f;
  }
#endif
  output[2 * i] = valid ?
                      make_float4(
                          result.direction.x, result.direction.y, result.direction.z, result.eta) :
                      zero_float4();
  output[2 * i + 1] = valid ? make_float4(result.relative_order,
                                          result.transmission,
                                          result.probability,
                                          result.throughput) :
                              zero_float4();
}
