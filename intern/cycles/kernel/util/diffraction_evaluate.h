/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/util/diffraction_limits.h"
#include "kernel/util/diffraction_reference.h"
#include "kernel/util/diffraction_tensor.h"
#include "kernel/util/diffraction_admittance.h"
CCL_NAMESPACE_BEGIN

/* Only the selected channel count owns matrix scratch. Keep dispatch inline
 * so outlined calls do not accumulate one workspace per smaller channel count. */
template<int N, typename BoundaryPointer>
#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device_inline
#endif
bool diffraction_cache_tensor_match_sized(ccl_global const float2 *matrices,
                                      const BoundaryPointer boundary,
                                      const int rank,
                                      const int model_floats,
                                      const int incoming,
                                      const float3 coordinate,
                                      ccl_private float2 *jones)
{
  float2 chart[N * N];
  const DiffractionTensorFloatView model{matrices + N * N, 0};
  return diffraction_tensor_chart<N>(model, model_floats, rank, coordinate, chart) &&
         diffraction_reference_match_anchor<N>(chart, matrices, boundary, incoming, jones);
}

template<int MaxChannels = DIFFRACTION_MAX_CHANNELS, int N = 2, typename BoundaryPointer>
ccl_device_inline bool diffraction_cache_tensor_match(ccl_global const float2 *matrices,
                                      const BoundaryPointer boundary,
                                      const int channels,
                                      const int rank,
                                      const int model_floats,
                                      const int incoming,
                                      const float3 coordinate,
                                      ccl_private float2 *jones)
{
  static_assert(MaxChannels > 0 && MaxChannels % 2 == 0);
  if (channels <= 0 || channels > MaxChannels || channels % 2 || incoming < 0 ||
      incoming >= channels / 2)
    return false;
  if (channels == N)
    return diffraction_cache_tensor_match_sized<N>(
        matrices, boundary, rank, model_floats, incoming, coordinate, jones);
  if constexpr (N < MaxChannels)
    return diffraction_cache_tensor_match<MaxChannels, N + 2>(
        matrices, boundary, channels, rank, model_floats, incoming, coordinate, jones);
  return false;
}

/* Only the selected channel count owns matrix scratch. Keep dispatch inline
 * so outlined calls do not accumulate one workspace per smaller channel count. */
template<int N, typename BoundaryPointer>
#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device_inline
#endif
bool diffraction_cache_tensor_admittance_match_sized(ccl_global const float2 *matrices,
                                      const BoundaryPointer boundary,
                                      const int rank,
                                      const int model_floats,
                                      const int incoming,
                                      const float3 coordinate,
                                      ccl_private float2 *jones)
{
  float2 chart[N * N];
  const DiffractionTensorFloatView model{matrices + N * N, 0};
  return diffraction_tensor_chart<N>(model, model_floats, rank, coordinate, chart) &&
         diffraction_reference_match_admittance<N>(chart, matrices, boundary, incoming, jones);
}

template<int MaxChannels = DIFFRACTION_MAX_CHANNELS, int N = 2, typename BoundaryPointer>
ccl_device_inline bool diffraction_cache_tensor_admittance_match(ccl_global const float2 *matrices,
                                      const BoundaryPointer boundary,
                                      const int channels,
                                      const int rank,
                                      const int model_floats,
                                      const int incoming,
                                      const float3 coordinate,
                                      ccl_private float2 *jones)
{
  static_assert(MaxChannels > 0 && MaxChannels % 2 == 0);
  if (channels <= 0 || channels > MaxChannels || channels % 2 || incoming < 0 ||
      incoming >= channels / 2)
    return false;
  if (channels == N)
    return diffraction_cache_tensor_admittance_match_sized<N>(
        matrices, boundary, rank, model_floats, incoming, coordinate, jones);
  if constexpr (N < MaxChannels)
    return diffraction_cache_tensor_admittance_match<MaxChannels, N + 2>(
        matrices, boundary, channels, rank, model_floats, incoming, coordinate, jones);
  return false;
}

/* Dispatch a prepared cell to a fixed-size solver without maximum-size
 * scratch storage in each specialization. MaxChannels is an explicit caller
 * capacity, not a physical order cutoff: larger cells fail rather than drop
 * ports. The caller supplies complete matrices and output storage for all
 * channels. Chart output is complex Jones data; intensity output is restricted
 * to incoherent transport. InPlace retains the solver's conditioning fallback. */
template<int MaxChannels = DIFFRACTION_MAX_CHANNELS, bool InPlace = false, int N = 2, typename BoundaryPointer>
ccl_device_inline bool diffraction_cache_chart_match(ccl_global const float2 *matrices,
                                                     BoundaryPointer boundary,
                                                     int channels,
                                                     int degree,
                                                     float2 rotation,
                                                     int incoming,
                                                     float3 t,
                                                     ccl_private float2 *jones)
{
  static_assert(MaxChannels > 0 && MaxChannels % 2 == 0);
  if (channels <= 0 || channels > MaxChannels || channels % 2 || incoming < 0 ||
      incoming >= channels / 2)
    return false;
  if (channels == N)
    return degree == 1 ? diffraction_chart_cell_match<N, 1, InPlace>(
                             matrices, boundary, rotation, incoming, t, jones) :
                         degree == 2 && diffraction_chart_cell_match<N, 2, InPlace>(
                                            matrices, boundary, rotation, incoming, t, jones);
  if constexpr (N < MaxChannels)
    return diffraction_cache_chart_match<MaxChannels, InPlace, N + 2>(
        matrices, boundary, channels, degree, rotation, incoming, t, jones);
  return false;
}
template<int MaxChannels = DIFFRACTION_MAX_CHANNELS, int C = 0, typename BoundaryPointer>
ccl_device_inline bool diffraction_cache_intensity_match(ccl_global const float2 *matrices,
                                                         BoundaryPointer boundary,
                                                         ccl_global const int *active,
                                                         int count,
                                                         int channels,
                                                         int incoming,
                                                         float3 t,
                                                         ccl_private float *power)
{
  static_assert(MaxChannels > 0 && MaxChannels % 2 == 0);
  if (count < 0 || count > channels || count % 2 || channels <= 0 || channels > MaxChannels ||
      channels % 2 || incoming < 0 || incoming >= channels / 2)
    return false;
  if (count == C)
    return diffraction_hybrid_cell_power<MaxChannels, C>(
        matrices, boundary, active, channels, incoming, t, power);
  if constexpr (C < MaxChannels)
    return diffraction_cache_intensity_match<MaxChannels, C + 2>(
        matrices, boundary, active, count, channels, incoming, t, power);
  return false;
}

CCL_NAMESPACE_END
