/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "kernel/device/metal/compat.h"

#include "kernel/util/diffraction_boundary.h"

#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
template<int N, int C>
inline bool construct_boundary(device const float4 *inputs,
                               device const int *active,
                               thread float2 *boundary)
{
  for (int port = 0; port < N / 2; port++) {
    const float4 a = inputs[2 * port], b = inputs[2 * port + 1];
    DiffractionGratingBoundary result = {};
    if (!diffraction_grating_boundary(a.xyz, a.w, b.x, b.y, b.z, int(b.w), &result)) {
      return false;
    }
    if constexpr (C >= 0) {
      bool reference = false;
      for (int j = 0; j < C / 2; j++) {
        reference |= active[j] == port;
      }
      if (!reference) {
        result.r_te = result.r_tm = zero_float2();
        result.transmission = result.q2 > 0 ? make_float2(1, 1) : zero_float2();
      }
    }
    boundary[4 * port] = result.r_te;
    boundary[4 * port + 1] = result.r_tm;
    boundary[4 * port + 2] = result.transmission;
    boundary[4 * port + 3] = result.tangent_direction;
  }
  return true;
}
#endif

kernel void diffraction_reference_10(device const float2 *charts [[buffer(0)]],
                                     device const float2 *boundaries [[buffer(1)]],
                                     device const float2 *rotations [[buffer(2)]],
                                     device const int *incoming [[buffer(3)]],
                                     device float2 *outputs [[buffer(4)]],
                                     device uint *valid [[buffer(5)]],
                                     uint index [[thread_position_in_grid]])
{
  const uint sample = index;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[20];
  const bool boundary_ok = construct_boundary<10, -1>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 10, nullptr, local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 20;
  const bool boundary_ok = true;
#endif
  float2 result[20];
  valid[index] = boundary_ok && diffraction_reference_match<10>(charts + index * 100,
                                                                local_boundary,
                                                                rotations[index],
                                                                incoming[index],
                                                                result);
  for (int j = 0; j < 20; j++) {
    outputs[index * 20 + j] = result[j];
  }
}

kernel void diffraction_reference_bench_10(device const float2 *charts [[buffer(0)]],
                                           device const float2 *boundaries [[buffer(1)]],
                                           device const float2 *rotations [[buffer(2)]],
                                           device const int *incoming [[buffer(3)]],
                                           device float *outputs [[buffer(4)]],
                                           constant uint &cases [[buffer(5)]],
                                           uint index [[thread_position_in_grid]])
{
  const uint sample = index % cases;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[20];
  const bool boundary_ok = construct_boundary<10, -1>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 10, nullptr, local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 20;
  const bool boundary_ok = true;
#endif
  float2 result[20];
  const bool valid = boundary_ok && diffraction_reference_match<10>(charts + sample * 100,
                                                                    local_boundary,
                                                                    rotations[sample],
                                                                    incoming[sample],
                                                                    result);
  float power = 0.0f;
  for (int j = 0; j < 20; j++) {
    power += dot(result[j], result[j]);
  }
  outputs[index] = valid ? power : -1.0f;
}

kernel void diffraction_reference_18(device const float2 *charts [[buffer(0)]],
                                     device const float2 *boundaries [[buffer(1)]],
                                     device const float2 *rotations [[buffer(2)]],
                                     device const int *incoming [[buffer(3)]],
                                     device float2 *outputs [[buffer(4)]],
                                     device uint *valid [[buffer(5)]],
                                     uint index [[thread_position_in_grid]])
{
  const uint sample = index;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[36];
  const bool boundary_ok = construct_boundary<18, -1>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 18, nullptr, local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 36;
  const bool boundary_ok = true;
#endif
  float2 result[36];
  valid[index] = boundary_ok && diffraction_reference_match<18>(charts + index * 324,
                                                                local_boundary,
                                                                rotations[index],
                                                                incoming[index],
                                                                result);
  for (int j = 0; j < 36; j++) {
    outputs[index * 36 + j] = result[j];
  }
}

kernel void diffraction_reference_bench_18(device const float2 *charts [[buffer(0)]],
                                           device const float2 *boundaries [[buffer(1)]],
                                           device const float2 *rotations [[buffer(2)]],
                                           device const int *incoming [[buffer(3)]],
                                           device float *outputs [[buffer(4)]],
                                           constant uint &cases [[buffer(5)]],
                                           uint index [[thread_position_in_grid]])
{
  const uint sample = index % cases;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[36];
  const bool boundary_ok = construct_boundary<18, -1>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 18, nullptr, local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 36;
  const bool boundary_ok = true;
#endif
  float2 result[36];
  const bool valid = boundary_ok && diffraction_reference_match<18>(charts + sample * 324,
                                                                    local_boundary,
                                                                    rotations[sample],
                                                                    incoming[sample],
                                                                    result);
  float power = 0.0f;
  for (int j = 0; j < 36; j++) {
    power += dot(result[j], result[j]);
  }
  outputs[index] = valid ? power : -1.0f;
}

kernel void diffraction_reference_20(device const float2 *charts [[buffer(0)]],
                                     device const float2 *boundaries [[buffer(1)]],
                                     device const float2 *rotations [[buffer(2)]],
                                     device const int *incoming [[buffer(3)]],
                                     device float2 *outputs [[buffer(4)]],
                                     device uint *valid [[buffer(5)]],
                                     uint index [[thread_position_in_grid]])
{
  const uint sample = index;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[40];
  const bool boundary_ok = construct_boundary<20, -1>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 20, nullptr, local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 40;
  const bool boundary_ok = true;
#endif
  float2 result[40];
  valid[index] = boundary_ok && diffraction_reference_match<20>(charts + index * 400,
                                                                local_boundary,
                                                                rotations[index],
                                                                incoming[index],
                                                                result);
  for (int j = 0; j < 40; j++) {
    outputs[index * 40 + j] = result[j];
  }
}

kernel void diffraction_reference_bench_20(device const float2 *charts [[buffer(0)]],
                                           device const float2 *boundaries [[buffer(1)]],
                                           device const float2 *rotations [[buffer(2)]],
                                           device const int *incoming [[buffer(3)]],
                                           device float *outputs [[buffer(4)]],
                                           constant uint &cases [[buffer(5)]],
                                           uint index [[thread_position_in_grid]])
{
  const uint sample = index % cases;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[40];
  const bool boundary_ok = construct_boundary<20, -1>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 20, nullptr, local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 40;
  const bool boundary_ok = true;
#endif
  float2 result[40];
  const bool valid = boundary_ok && diffraction_reference_match<20>(charts + sample * 400,
                                                                    local_boundary,
                                                                    rotations[sample],
                                                                    incoming[sample],
                                                                    result);
  float power = 0.0f;
  for (int j = 0; j < 40; j++) {
    power += dot(result[j], result[j]);
  }
  outputs[index] = valid ? power : -1.0f;
}

kernel void diffraction_hybrid_10_0(device const float2 *matrices [[buffer(0)]],
                                    device const float2 *boundaries [[buffer(1)]],
                                    device const int *active [[buffer(2)]],
                                    device const int *incoming [[buffer(3)]],
                                    device float2 *outputs [[buffer(4)]],
                                    device uint *valid [[buffer(5)]],
                                    uint index [[thread_position_in_grid]])
{
  const uint sample = index;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[20];
  const bool boundary_ok = construct_boundary<10, 0>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 10,
      active + sample * 0,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 20;
  const bool boundary_ok = true;
#endif
  float2 result[20] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<0>(matrices + sample * 100,
                                                             local_boundary,
                                                             active + sample * 0,
                                                             10,
                                                             incoming[sample],
                                                             result);
  valid[index] = ok;
  for (int j = 0; j < 20; j++) {
    outputs[index * 20 + j] = result[j];
  }
}

kernel void diffraction_hybrid_bench_10_0(device const float2 *matrices [[buffer(0)]],
                                          device const float2 *boundaries [[buffer(1)]],
                                          device const int *active [[buffer(2)]],
                                          device const int *incoming [[buffer(3)]],
                                          device float *outputs [[buffer(4)]],
                                          constant uint &cases [[buffer(5)]],
                                          uint index [[thread_position_in_grid]])
{
  const uint sample = index % cases;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[20];
  const bool boundary_ok = construct_boundary<10, 0>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 10,
      active + sample * 0,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 20;
  const bool boundary_ok = true;
#endif
  float2 result[20] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<0>(matrices + sample * 100,
                                                             local_boundary,
                                                             active + sample * 0,
                                                             10,
                                                             incoming[sample],
                                                             result);
  float power = 0.0f;
  for (int j = 0; j < 20; j++) {
    power += dot(result[j], result[j]);
  }
  outputs[index] = ok ? power : -1.0f;
}

kernel void diffraction_hybrid_10_2(device const float2 *matrices [[buffer(0)]],
                                    device const float2 *boundaries [[buffer(1)]],
                                    device const int *active [[buffer(2)]],
                                    device const int *incoming [[buffer(3)]],
                                    device float2 *outputs [[buffer(4)]],
                                    device uint *valid [[buffer(5)]],
                                    uint index [[thread_position_in_grid]])
{
  const uint sample = index;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[20];
  const bool boundary_ok = construct_boundary<10, 2>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 10,
      active + sample * 1,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 20;
  const bool boundary_ok = true;
#endif
  float2 result[20] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<2>(matrices + sample * 100,
                                                             local_boundary,
                                                             active + sample * 1,
                                                             10,
                                                             incoming[sample],
                                                             result);
  valid[index] = ok;
  for (int j = 0; j < 20; j++) {
    outputs[index * 20 + j] = result[j];
  }
}

kernel void diffraction_hybrid_bench_10_2(device const float2 *matrices [[buffer(0)]],
                                          device const float2 *boundaries [[buffer(1)]],
                                          device const int *active [[buffer(2)]],
                                          device const int *incoming [[buffer(3)]],
                                          device float *outputs [[buffer(4)]],
                                          constant uint &cases [[buffer(5)]],
                                          uint index [[thread_position_in_grid]])
{
  const uint sample = index % cases;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[20];
  const bool boundary_ok = construct_boundary<10, 2>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 10,
      active + sample * 1,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 20;
  const bool boundary_ok = true;
#endif
  float2 result[20] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<2>(matrices + sample * 100,
                                                             local_boundary,
                                                             active + sample * 1,
                                                             10,
                                                             incoming[sample],
                                                             result);
  float power = 0.0f;
  for (int j = 0; j < 20; j++) {
    power += dot(result[j], result[j]);
  }
  outputs[index] = ok ? power : -1.0f;
}

kernel void diffraction_hybrid_10_4(device const float2 *matrices [[buffer(0)]],
                                    device const float2 *boundaries [[buffer(1)]],
                                    device const int *active [[buffer(2)]],
                                    device const int *incoming [[buffer(3)]],
                                    device float2 *outputs [[buffer(4)]],
                                    device uint *valid [[buffer(5)]],
                                    uint index [[thread_position_in_grid]])
{
  const uint sample = index;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[20];
  const bool boundary_ok = construct_boundary<10, 4>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 10,
      active + sample * 2,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 20;
  const bool boundary_ok = true;
#endif
  float2 result[20] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<4>(matrices + sample * 100,
                                                             local_boundary,
                                                             active + sample * 2,
                                                             10,
                                                             incoming[sample],
                                                             result);
  valid[index] = ok;
  for (int j = 0; j < 20; j++) {
    outputs[index * 20 + j] = result[j];
  }
}

kernel void diffraction_hybrid_bench_10_4(device const float2 *matrices [[buffer(0)]],
                                          device const float2 *boundaries [[buffer(1)]],
                                          device const int *active [[buffer(2)]],
                                          device const int *incoming [[buffer(3)]],
                                          device float *outputs [[buffer(4)]],
                                          constant uint &cases [[buffer(5)]],
                                          uint index [[thread_position_in_grid]])
{
  const uint sample = index % cases;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[20];
  const bool boundary_ok = construct_boundary<10, 4>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 10,
      active + sample * 2,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 20;
  const bool boundary_ok = true;
#endif
  float2 result[20] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<4>(matrices + sample * 100,
                                                             local_boundary,
                                                             active + sample * 2,
                                                             10,
                                                             incoming[sample],
                                                             result);
  float power = 0.0f;
  for (int j = 0; j < 20; j++) {
    power += dot(result[j], result[j]);
  }
  outputs[index] = ok ? power : -1.0f;
}

kernel void diffraction_hybrid_18_0(device const float2 *matrices [[buffer(0)]],
                                    device const float2 *boundaries [[buffer(1)]],
                                    device const int *active [[buffer(2)]],
                                    device const int *incoming [[buffer(3)]],
                                    device float2 *outputs [[buffer(4)]],
                                    device uint *valid [[buffer(5)]],
                                    uint index [[thread_position_in_grid]])
{
  const uint sample = index;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[36];
  const bool boundary_ok = construct_boundary<18, 0>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 18,
      active + sample * 0,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 36;
  const bool boundary_ok = true;
#endif
  float2 result[36] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<0>(matrices + sample * 324,
                                                             local_boundary,
                                                             active + sample * 0,
                                                             18,
                                                             incoming[sample],
                                                             result);
  valid[index] = ok;
  for (int j = 0; j < 36; j++) {
    outputs[index * 36 + j] = result[j];
  }
}

kernel void diffraction_hybrid_bench_18_0(device const float2 *matrices [[buffer(0)]],
                                          device const float2 *boundaries [[buffer(1)]],
                                          device const int *active [[buffer(2)]],
                                          device const int *incoming [[buffer(3)]],
                                          device float *outputs [[buffer(4)]],
                                          constant uint &cases [[buffer(5)]],
                                          uint index [[thread_position_in_grid]])
{
  const uint sample = index % cases;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[36];
  const bool boundary_ok = construct_boundary<18, 0>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 18,
      active + sample * 0,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 36;
  const bool boundary_ok = true;
#endif
  float2 result[36] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<0>(matrices + sample * 324,
                                                             local_boundary,
                                                             active + sample * 0,
                                                             18,
                                                             incoming[sample],
                                                             result);
  float power = 0.0f;
  for (int j = 0; j < 36; j++) {
    power += dot(result[j], result[j]);
  }
  outputs[index] = ok ? power : -1.0f;
}

kernel void diffraction_hybrid_18_2(device const float2 *matrices [[buffer(0)]],
                                    device const float2 *boundaries [[buffer(1)]],
                                    device const int *active [[buffer(2)]],
                                    device const int *incoming [[buffer(3)]],
                                    device float2 *outputs [[buffer(4)]],
                                    device uint *valid [[buffer(5)]],
                                    uint index [[thread_position_in_grid]])
{
  const uint sample = index;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[36];
  const bool boundary_ok = construct_boundary<18, 2>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 18,
      active + sample * 1,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 36;
  const bool boundary_ok = true;
#endif
  float2 result[36] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<2>(matrices + sample * 324,
                                                             local_boundary,
                                                             active + sample * 1,
                                                             18,
                                                             incoming[sample],
                                                             result);
  valid[index] = ok;
  for (int j = 0; j < 36; j++) {
    outputs[index * 36 + j] = result[j];
  }
}

kernel void diffraction_hybrid_bench_18_2(device const float2 *matrices [[buffer(0)]],
                                          device const float2 *boundaries [[buffer(1)]],
                                          device const int *active [[buffer(2)]],
                                          device const int *incoming [[buffer(3)]],
                                          device float *outputs [[buffer(4)]],
                                          constant uint &cases [[buffer(5)]],
                                          uint index [[thread_position_in_grid]])
{
  const uint sample = index % cases;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[36];
  const bool boundary_ok = construct_boundary<18, 2>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 18,
      active + sample * 1,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 36;
  const bool boundary_ok = true;
#endif
  float2 result[36] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<2>(matrices + sample * 324,
                                                             local_boundary,
                                                             active + sample * 1,
                                                             18,
                                                             incoming[sample],
                                                             result);
  float power = 0.0f;
  for (int j = 0; j < 36; j++) {
    power += dot(result[j], result[j]);
  }
  outputs[index] = ok ? power : -1.0f;
}

kernel void diffraction_hybrid_18_4(device const float2 *matrices [[buffer(0)]],
                                    device const float2 *boundaries [[buffer(1)]],
                                    device const int *active [[buffer(2)]],
                                    device const int *incoming [[buffer(3)]],
                                    device float2 *outputs [[buffer(4)]],
                                    device uint *valid [[buffer(5)]],
                                    uint index [[thread_position_in_grid]])
{
  const uint sample = index;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[36];
  const bool boundary_ok = construct_boundary<18, 4>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 18,
      active + sample * 2,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 36;
  const bool boundary_ok = true;
#endif
  float2 result[36] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<4>(matrices + sample * 324,
                                                             local_boundary,
                                                             active + sample * 2,
                                                             18,
                                                             incoming[sample],
                                                             result);
  valid[index] = ok;
  for (int j = 0; j < 36; j++) {
    outputs[index * 36 + j] = result[j];
  }
}

kernel void diffraction_hybrid_bench_18_4(device const float2 *matrices [[buffer(0)]],
                                          device const float2 *boundaries [[buffer(1)]],
                                          device const int *active [[buffer(2)]],
                                          device const int *incoming [[buffer(3)]],
                                          device float *outputs [[buffer(4)]],
                                          constant uint &cases [[buffer(5)]],
                                          uint index [[thread_position_in_grid]])
{
  const uint sample = index % cases;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[36];
  const bool boundary_ok = construct_boundary<18, 4>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 18,
      active + sample * 2,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 36;
  const bool boundary_ok = true;
#endif
  float2 result[36] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<4>(matrices + sample * 324,
                                                             local_boundary,
                                                             active + sample * 2,
                                                             18,
                                                             incoming[sample],
                                                             result);
  float power = 0.0f;
  for (int j = 0; j < 36; j++) {
    power += dot(result[j], result[j]);
  }
  outputs[index] = ok ? power : -1.0f;
}

kernel void diffraction_hybrid_20_0(device const float2 *matrices [[buffer(0)]],
                                    device const float2 *boundaries [[buffer(1)]],
                                    device const int *active [[buffer(2)]],
                                    device const int *incoming [[buffer(3)]],
                                    device float2 *outputs [[buffer(4)]],
                                    device uint *valid [[buffer(5)]],
                                    uint index [[thread_position_in_grid]])
{
  const uint sample = index;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[40];
  const bool boundary_ok = construct_boundary<20, 0>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 20,
      active + sample * 0,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 40;
  const bool boundary_ok = true;
#endif
  float2 result[40] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<0>(matrices + sample * 400,
                                                             local_boundary,
                                                             active + sample * 0,
                                                             20,
                                                             incoming[sample],
                                                             result);
  valid[index] = ok;
  for (int j = 0; j < 40; j++) {
    outputs[index * 40 + j] = result[j];
  }
}

kernel void diffraction_hybrid_bench_20_0(device const float2 *matrices [[buffer(0)]],
                                          device const float2 *boundaries [[buffer(1)]],
                                          device const int *active [[buffer(2)]],
                                          device const int *incoming [[buffer(3)]],
                                          device float *outputs [[buffer(4)]],
                                          constant uint &cases [[buffer(5)]],
                                          uint index [[thread_position_in_grid]])
{
  const uint sample = index % cases;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[40];
  const bool boundary_ok = construct_boundary<20, 0>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 20,
      active + sample * 0,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 40;
  const bool boundary_ok = true;
#endif
  float2 result[40] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<0>(matrices + sample * 400,
                                                             local_boundary,
                                                             active + sample * 0,
                                                             20,
                                                             incoming[sample],
                                                             result);
  float power = 0.0f;
  for (int j = 0; j < 40; j++) {
    power += dot(result[j], result[j]);
  }
  outputs[index] = ok ? power : -1.0f;
}

kernel void diffraction_hybrid_20_2(device const float2 *matrices [[buffer(0)]],
                                    device const float2 *boundaries [[buffer(1)]],
                                    device const int *active [[buffer(2)]],
                                    device const int *incoming [[buffer(3)]],
                                    device float2 *outputs [[buffer(4)]],
                                    device uint *valid [[buffer(5)]],
                                    uint index [[thread_position_in_grid]])
{
  const uint sample = index;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[40];
  const bool boundary_ok = construct_boundary<20, 2>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 20,
      active + sample * 1,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 40;
  const bool boundary_ok = true;
#endif
  float2 result[40] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<2>(matrices + sample * 400,
                                                             local_boundary,
                                                             active + sample * 1,
                                                             20,
                                                             incoming[sample],
                                                             result);
  valid[index] = ok;
  for (int j = 0; j < 40; j++) {
    outputs[index * 40 + j] = result[j];
  }
}

kernel void diffraction_hybrid_bench_20_2(device const float2 *matrices [[buffer(0)]],
                                          device const float2 *boundaries [[buffer(1)]],
                                          device const int *active [[buffer(2)]],
                                          device const int *incoming [[buffer(3)]],
                                          device float *outputs [[buffer(4)]],
                                          constant uint &cases [[buffer(5)]],
                                          uint index [[thread_position_in_grid]])
{
  const uint sample = index % cases;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[40];
  const bool boundary_ok = construct_boundary<20, 2>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 20,
      active + sample * 1,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 40;
  const bool boundary_ok = true;
#endif
  float2 result[40] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<2>(matrices + sample * 400,
                                                             local_boundary,
                                                             active + sample * 1,
                                                             20,
                                                             incoming[sample],
                                                             result);
  float power = 0.0f;
  for (int j = 0; j < 40; j++) {
    power += dot(result[j], result[j]);
  }
  outputs[index] = ok ? power : -1.0f;
}

kernel void diffraction_hybrid_20_4(device const float2 *matrices [[buffer(0)]],
                                    device const float2 *boundaries [[buffer(1)]],
                                    device const int *active [[buffer(2)]],
                                    device const int *incoming [[buffer(3)]],
                                    device float2 *outputs [[buffer(4)]],
                                    device uint *valid [[buffer(5)]],
                                    uint index [[thread_position_in_grid]])
{
  const uint sample = index;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[40];
  const bool boundary_ok = construct_boundary<20, 4>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 20,
      active + sample * 2,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 40;
  const bool boundary_ok = true;
#endif
  float2 result[40] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<4>(matrices + sample * 400,
                                                             local_boundary,
                                                             active + sample * 2,
                                                             20,
                                                             incoming[sample],
                                                             result);
  valid[index] = ok;
  for (int j = 0; j < 40; j++) {
    outputs[index * 40 + j] = result[j];
  }
}

kernel void diffraction_hybrid_bench_20_4(device const float2 *matrices [[buffer(0)]],
                                          device const float2 *boundaries [[buffer(1)]],
                                          device const int *active [[buffer(2)]],
                                          device const int *incoming [[buffer(3)]],
                                          device float *outputs [[buffer(4)]],
                                          constant uint &cases [[buffer(5)]],
                                          uint index [[thread_position_in_grid]])
{
  const uint sample = index % cases;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
  float2 local_boundary[40];
  const bool boundary_ok = construct_boundary<20, 4>(
      reinterpret_cast<device const float4 *>(boundaries) + sample * 20,
      active + sample * 2,
      local_boundary);
#else
  device const float2 *local_boundary = boundaries + sample * 40;
  const bool boundary_ok = true;
#endif
  float2 result[40] = {};
  const bool ok = boundary_ok && diffraction_hybrid_match<4>(matrices + sample * 400,
                                                             local_boundary,
                                                             active + sample * 2,
                                                             20,
                                                             incoming[sample],
                                                             result);
  float power = 0.0f;
  for (int j = 0; j < 40; j++) {
    power += dot(result[j], result[j]);
  }
  outputs[index] = ok ? power : -1.0f;
}
