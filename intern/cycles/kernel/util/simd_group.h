/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Operations across the lanes of one Metal SIMD group or CUDA warp. Every lane of the group must
 * call them with the same control flow. Both have 32 lanes on the GPUs that use them. On Metal
 * they are the Metal intrinsics. */

#if defined(__KERNEL_METAL__)

#  define ccl_simd_width() metal::simd_sum(1u)
#  define ccl_simd_sum(value) metal::simd_sum(value)
#  define ccl_simd_max(value) metal::simd_max(value)
#  define ccl_simd_min(value) metal::simd_min(value)
#  define ccl_simd_broadcast(value, lane) metal::simd_broadcast(value, lane)
#  define ccl_simd_shuffle_xor(value, mask) metal::simd_shuffle_xor(value, mask)

#elif defined(__KERNEL_CUDA__)

CCL_NAMESPACE_BEGIN

#  define CCL_SIMD_FULL_MASK 0xFFFFFFFFu

/* Kernels using these launch whole warps, the number of active lanes is constant. */
ccl_device_forceinline uint ccl_simd_width()
{
  return 32u;
}

template<typename T> ccl_device_forceinline T ccl_simd_shuffle_xor(const T value, const uint mask)
{
  return __shfl_xor_sync(CCL_SIMD_FULL_MASK, value, int(mask));
}

ccl_device_forceinline float4 ccl_simd_shuffle_xor(const float4 value, const uint mask)
{
  return make_float4(ccl_simd_shuffle_xor(value.x, mask),
                     ccl_simd_shuffle_xor(value.y, mask),
                     ccl_simd_shuffle_xor(value.z, mask),
                     ccl_simd_shuffle_xor(value.w, mask));
}

template<typename T> ccl_device_forceinline T ccl_simd_broadcast(const T value, const uint lane)
{
  return __shfl_sync(CCL_SIMD_FULL_MASK, value, int(lane));
}

/* Butterfly reductions leave the result in every lane, as the Metal intrinsics do. */
template<typename T> ccl_device_forceinline T ccl_simd_sum(T value)
{
  for (uint offset = 16; offset > 0; offset /= 2) {
    value = value + ccl_simd_shuffle_xor(value, offset);
  }
  return value;
}

template<typename T> ccl_device_forceinline T ccl_simd_max(T value)
{
  for (uint offset = 16; offset > 0; offset /= 2) {
    value = max(value, ccl_simd_shuffle_xor(value, offset));
  }
  return value;
}

template<typename T> ccl_device_forceinline T ccl_simd_min(T value)
{
  for (uint offset = 16; offset > 0; offset /= 2) {
    value = min(value, ccl_simd_shuffle_xor(value, offset));
  }
  return value;
}

CCL_NAMESPACE_END

#endif
