/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/util/diffraction_boundary.h"
#include "kernel/util/diffraction_cache.h"
#include "kernel/util/diffraction_coordinates.h"
#include "kernel/util/diffraction_evaluate.h"
#include "kernel/util/diffraction_sample.h"

kernel void diffraction_full_cache(device const int4 *nodes [[buffer(0)]],
                                   device const int4 *layout [[buffer(1)]],
                                   device const float4 *bounds [[buffer(2)]],
                                   device const int2 *ports [[buffer(3)]],
                                   device const int *active [[buffer(4)]],
                                   device const float2 *matrices [[buffer(5)]],
                                   device const float4 *queries [[buffer(6)]],
                                   device const float2 *boundaries [[buffer(7)]],
                                   device float *output [[buffer(8)]],
                                   device int *leaves [[buffer(9)]],
                                   constant float4 *domain [[buffer(10)]],
                                   device float4 *samples [[buffer(11)]],
                                   uint index [[thread_position_in_grid]])
{
  leaves[index] = -1;
  samples[index] = make_float4(-1, 0, 0, 0);
  for (int j = 0; j < 10; j++)
    output[10 * index + j] = 0;
  const bool mirror = domain[1].w != 0;
#ifdef DIFFRACTION_RAY_LOOKUP
  const float4 raw = queries[index];
  DiffractionCacheCoordinates coordinates;
  if (!diffraction_cache_coordinates(
          raw.xyz, domain[2].x, raw.w, domain[2].z, mirror, &coordinates))
    return;
  const float3 q = coordinates.query;
  const float4 query = make_float4(q.x, q.y, q.z, coordinates.incoming_order);
  const float3 incident_ray = make_float3(
      coordinates.reverse_orders ? -raw.x : raw.x, mirror ? -fabsf(raw.y) : raw.y, raw.z);
#else
  const float4 query = queries[index];
  const float3 q = mirror ? diffraction_cache_symmetry(query.xyz).query : query.xyz;
#endif
  const int leaf = diffraction_cache_lookup(
      nodes, int(domain[0].w), domain[0].xyz, domain[1].xyz, q, false);
  if (leaf < 0)
    return;
  const int4 entry = layout[2 * leaf], shape = layout[2 * leaf + 1];
  if (shape.x <= 0 || shape.x > 10)
    return;
  const float4 lower = bounds[2 * leaf], upper = bounds[2 * leaf + 1];
  const float3 t = (q - lower.xyz) / (upper.xyz - lower.xyz);
  int incoming = -1;
  for (int j = 0; j < shape.x; j++)
    if (ports[entry.y + j].x == int(query.w) && ports[entry.y + j].y == 0)
      incoming = j;
  if (incoming < 0)
    return;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[40];
  device const float4 *inputs = reinterpret_cast<device const float4 *>(boundaries) + 20 * index;
  for (int j = 0; j < shape.x; j++) {
#  ifdef DIFFRACTION_RAY_LOOKUP
    const float4 ray = make_float4(incident_ray.x, incident_ray.y, incident_ray.z, domain[2].x);
    const float4 parameters = make_float4(ports[entry.y + j].y ? domain[2].y : domain[2].x,
                                          raw.w,
                                          domain[2].z,
                                          ports[entry.y + j].x - coordinates.incoming_order);
#  else
    const float4 ray = inputs[2 * j], parameters = inputs[2 * j + 1];
#  endif
    DiffractionGratingBoundary result;
    if (!diffraction_grating_boundary(
            ray.xyz, ray.w, parameters.x, parameters.y, parameters.z, int(parameters.w), &result))
      return;
    bool reference = false;
    for (int a = 0; a < shape.y; a++)
      reference |= active[entry.z + a] == j;
    local_boundary[4 * j] = reference ? result.r_te : zero_float2();
    local_boundary[4 * j + 1] = reference ? result.r_tm : zero_float2();
    local_boundary[4 * j + 2] = reference     ? result.transmission :
                                result.q2 > 0 ? make_float2(1, 1) :
                                                zero_float2();
    local_boundary[4 * j + 3] = result.tangent_direction;
  }
#else
  device const float2 *local_boundary = boundaries + 40 * index;
#endif
  float power[10] = {};
  bool valid;
  if (entry.w) {
    float2 jones[40];
#ifdef DIFFRACTION_INPLACE_CHART
    constexpr bool inplace = true;
#else
    constexpr bool inplace = false;
#endif
    valid = diffraction_cache_chart_match<20, inplace>(matrices + entry.x,
                                                       local_boundary,
                                                       2 * shape.x,
                                                       entry.w,
                                                       make_float2(lower.w, upper.w),
                                                       incoming,
                                                       t,
                                                       jones);
    if (valid)
      for (int p = 0; p < shape.x; p++)
        for (int c = 0; c < 4; c++)
          power[p] += 0.5f * dot(jones[4 * p + c], jones[4 * p + c]);
  }
  else
    valid = diffraction_cache_intensity_match<20>(matrices + entry.x,
                                                  local_boundary,
                                                  active + entry.z,
                                                  2 * shape.y,
                                                  2 * shape.x,
                                                  incoming,
                                                  t,
                                                  power);
  if (!valid)
    return;
  leaves[index] = leaf;
  for (int j = 0; j < shape.x; j++)
    output[10 * index + j] = power[j];
#ifdef DIFFRACTION_SAMPLE_ORDERS
  const float random = float((index * 1664525u + 1013904223u) & 0xffffffu) * 0x1p-24f;
  DiffractionOrderSample selected;
  const bool sampled = diffraction_sample_order(power, shape.x, random, &selected);
  samples[index] = sampled ?
                       make_float4(selected.port, selected.probability, selected.throughput, 1) :
                       make_float4(-1, 0, 0, 0);
#endif
}
