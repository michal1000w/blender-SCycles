/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/device/metal/compat.h"
#include "kernel/util/diffraction_boundary.h"

template<int C>
inline bool cell_boundary(device const float4 *inputs,
                          device const int *active,
                          int channels,
                          thread float2 *boundary)
{
  for (int port = 0; port < channels / 2; port++) {
    const float4 a = inputs[2 * port], b = inputs[2 * port + 1];
    DiffractionGratingBoundary value = {};
    if (!diffraction_grating_boundary(a.xyz, a.w, b.x, b.y, b.z, int(b.w), &value))
      return false;
    bool reference = false;
    for (int j = 0; j < C / 2; j++)
      reference |= active[j] == port;
    if (!reference) {
      value.r_te = value.r_tm = zero_float2();
      value.transmission = make_float2(1, 1);
    }
    boundary[4 * port] = value.r_te;
    boundary[4 * port + 1] = value.r_tm;
    boundary[4 * port + 2] = value.transmission;
    boundary[4 * port + 3] = value.tangent_direction;
  }
  return true;
}

template<int C>
inline bool evaluate_cell(device const float2 *matrices,
                          device const float4 *inputs,
                          device const int *active,
                          int channels,
                          float4 query,
                          thread float *powers)
{
  float2 boundary[64];
  if (!cell_boundary<C>(inputs, active, channels, boundary))
    return false;
  return diffraction_hybrid_cell_power<32, C>(
      matrices, boundary, active, channels, int(query.w), query.xyz, powers);
}

#define CELL_KERNEL(C) \
  kernel void diffraction_cell_##C(device const float2 *matrices [[buffer(0)]], \
                                   device const float4 *inputs [[buffer(1)]], \
                                   device const int *active [[buffer(2)]], \
                                   device const float4 *queries [[buffer(3)]], \
                                   device float *outputs [[buffer(4)]], \
                                   device uint *valid [[buffer(5)]], \
                                   constant int2 &shape [[buffer(6)]], \
                                   uint index [[thread_position_in_grid]]) \
  { \
    const uint sample = index % shape.y; \
    float powers[16] = {}; \
    valid[index] = evaluate_cell<C>( \
        matrices, inputs + sample * shape.x, active, shape.x, queries[sample], powers); \
    for (int p = 0; p < shape.x / 2; p++) \
      outputs[index * (shape.x / 2) + p] = powers[p]; \
  }
CELL_KERNEL(0)
CELL_KERNEL(2)
CELL_KERNEL(4)
CELL_KERNEL(6)
CELL_KERNEL(8)
CELL_KERNEL(10)
CELL_KERNEL(12)
#undef CELL_KERNEL

#include "kernel/util/diffraction_cache.h"
kernel void diffraction_lookup(device const int4 *nodes [[buffer(0)]],
                               device const float4 *queries [[buffer(1)]],
                               device int4 *outputs [[buffer(2)]],
                               constant uint &mirror [[buffer(3)]],
                               uint index [[thread_position_in_grid]])
{
  const auto symmetry = diffraction_cache_symmetry(queries[index].xyz);
  const int leaf = diffraction_cache_lookup(nodes,
                                            7,
                                            make_float3(-0.5f, -1.0f, 380.0f),
                                            mirror ? make_float3(0.0f, 0.0f, 780.0f) :
                                                     make_float3(0.5f, 1.0f, 780.0f),
                                            queries[index].xyz,
                                            mirror != 0);
  outputs[index] = make_int4(
      leaf, int(symmetry.reverse_orders), int(symmetry.cross_polarization_sign), 0);
}

#define CHART_CELL_KERNEL(N, C, D, O) \
  kernel void diffraction_chart_cell_##N##_##C##_##D##_##O( \
      device const float2 *matrices [[buffer(0)]], \
      device const float4 *inputs [[buffer(1)]], \
      device const int *active [[buffer(2)]], \
      device const float4 *queries [[buffer(3)]], \
      device float *outputs [[buffer(4)]], \
      device uint *valid [[buffer(5)]], \
      constant int2 &shape [[buffer(6)]], \
      constant float2 &rotation [[buffer(7)]], \
      device float2 *amplitudes [[buffer(8)]], \
      uint index [[thread_position_in_grid]]) \
  { \
    const uint sample = index % shape.y; \
    const float4 query = queries[sample]; \
    float2 boundary[2 * N], jones[2 * N] = {}; \
    valid[index] = cell_boundary<C>(inputs + sample * N, active, N, boundary) && \
                   diffraction_chart_cell_match<N, D, O>( \
                       matrices, boundary, rotation, int(query.w), query.xyz, jones); \
    if (index < uint(shape.y)) { \
      for (int j = 0; j < 2 * N; j++) \
        amplitudes[index * 2 * N + j] = jones[j]; \
    } \
    for (int p = 0; p < N / 2; p++) { \
      outputs[index * (N / 2) + p] = 0.5f * (dot(jones[4 * p], jones[4 * p]) + \
                                             dot(jones[4 * p + 1], jones[4 * p + 1]) + \
                                             dot(jones[4 * p + 2], jones[4 * p + 2]) + \
                                             dot(jones[4 * p + 3], jones[4 * p + 3])); \
    } \
  }
CHART_CELL_KERNEL(6, 4, 1, 0)
CHART_CELL_KERNEL(6, 4, 1, 1)
CHART_CELL_KERNEL(12, 4, 1, 0)
CHART_CELL_KERNEL(12, 4, 1, 1)
CHART_CELL_KERNEL(18, 4, 1, 0)
CHART_CELL_KERNEL(18, 4, 1, 1)
CHART_CELL_KERNEL(6, 4, 2, 0)
CHART_CELL_KERNEL(6, 4, 2, 1)
CHART_CELL_KERNEL(12, 4, 2, 0)
CHART_CELL_KERNEL(12, 4, 2, 1)
CHART_CELL_KERNEL(18, 4, 2, 0)
CHART_CELL_KERNEL(18, 4, 2, 1)
#undef CHART_CELL_KERNEL

/* End-to-end fixed-profile fixture: all three retained ports remain reference
 * ports. Bounds/rotations and matrices are uploaded from a completed cache. */
kernel void diffraction_stored_cache(device const int4 *nodes [[buffer(0)]],
                                     device const float2 *matrices [[buffer(1)]],
                                     device const float4 *bounds [[buffer(2)]],
                                     device const float4 *queries [[buffer(3)]],
                                     device float2 *outputs [[buffer(4)]],
                                     device uint *valid [[buffer(5)]],
                                     constant uint &node_count [[buffer(6)]],
                                     device const int2 *layout [[buffer(7)]],
                                     uint index [[thread_position_in_grid]])
{
  const float4 original = queries[index];
  const auto symmetry = diffraction_cache_symmetry(original.xyz);
  const float3 q = symmetry.query;
  const int leaf = diffraction_cache_lookup(nodes,
                                            node_count,
                                            make_float3(-0.03125f, -0.0625f, 735),
                                            make_float3(0, 0, 745),
                                            original.xyz,
                                            true);
  valid[index] = 0;
  for (int i = 0; i < 12; i++)
    outputs[12 * index + i] = zero_float2();
  if (leaf < 0)
    return;
  const float4 lower = bounds[2 * leaf], upper = bounds[2 * leaf + 1];
  const float3 t = (q - lower.xyz) / (upper.xyz - lower.xyz);
  const float x = q.x * q.z / 740.0f;
  const float3 ray = make_float3(x, q.y, sqrt(max(0.0f, 1.0f - x * x - q.y * q.y)));
  float2 boundary[12], jones[12];
  for (int p = 0; p < 3; p++) {
    DiffractionGratingBoundary b = {};
    if (!diffraction_grating_boundary(ray, 1.0f, 1.0f, q.z, 740.0f, p - 1, &b))
      return;
    boundary[4 * p] = b.r_te;
    boundary[4 * p + 1] = b.r_tm;
    boundary[4 * p + 2] = b.transmission;
    boundary[4 * p + 3] = b.tangent_direction;
  }
  const int incoming = symmetry.reverse_orders ? 2 - int(original.w) : int(original.w);
  const int2 entry = layout[leaf];
  const float2 rotation = make_float2(lower.w, upper.w);
  const bool matched = entry.y == 1 ?
                           diffraction_chart_cell_match<6, 1>(
                               matrices + entry.x, boundary, rotation, incoming, t, jones) :
                           entry.y == 2 &&
                               diffraction_chart_cell_match<6, 2>(
                                   matrices + entry.x, boundary, rotation, incoming, t, jones);
  if (!matched)
    return;
  for (int p = 0; p < 3; p++) {
    const int source = symmetry.reverse_orders ? 2 - p : p;
    for (int j = 0; j < 4; j++)
      outputs[12 * index + 4 * p + j] = jones[4 * source + j] *
                                        ((j == 1 || j == 2) ? symmetry.cross_polarization_sign :
                                                              1.0f);
  }
  valid[index] = 1;
}
