/* SPDX-FileCopyrightText: 2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/types_float3.h"

CCL_NAMESPACE_BEGIN

#define SPECTRUM_CHANNELS 3

using Spectrum = float3;
using PackedSpectrum = packed_float3;

#define make_spectrum(f) make_float3(f)
#define load_spectrum(f) load_float3(f)
#define store_spectrum(s, f) store_float3(f)

#define zero_spectrum zero_float3
#define one_spectrum one_float3

#define FOREACH_SPECTRUM_CHANNEL(counter) \
  for (int counter = 0; counter < SPECTRUM_CHANNELS; counter++)

#define GET_SPECTRUM_CHANNEL(v, i) (((ccl_private float *)(&(v)))[i])

/* Component `i` of a vector variable (float3, float4, int3, uint3...) with scalar type T, usable
 * as lvalue. Metal and CPU vector types support subscripts directly; the CUDA, HIP and oneAPI
 * vector types do not. With a constant index after unrolling it compiles to a register access. */
#if defined(__KERNEL_METAL__) || !defined(__KERNEL_GPU__)
#  define CCL_CHANNEL(T, v, i) (v)[i]
#else
#  define CCL_CHANNEL(T, v, i) (((T *)(&(v)))[i])
#endif

/* Read component `i` of a float3, also when the index is only known at runtime. On CUDA, HIP and
 * oneAPI a select keeps the vector in registers, where a subscript would spill it to memory. */
#if defined(__KERNEL_METAL__) || !defined(__KERNEL_GPU__)
#  define CCL_FLOAT3_GET(v, i) (v)[i]
#else
ccl_device_forceinline float ccl_float3_select(const float3 v, const int i)
{
  return (i == 0) ? v.x : ((i == 1) ? v.y : v.z);
}
#  define CCL_FLOAT3_GET(v, i) ccl_float3_select(v, int(i))
#endif

CCL_NAMESPACE_END
