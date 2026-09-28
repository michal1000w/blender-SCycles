/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "util/math.h"

#if defined(__clang__) && !defined(__KERNEL_METAL__)
#  pragma float_control(precise, on, push)
#endif

CCL_NAMESPACE_BEGIN

ccl_device_inline float dielectric_precise_divide(const float a, const float b)
{
#ifdef __KERNEL_METAL__
  return metal::precise::divide(a, b);
#else
  return a / b;
#endif
}

/* Preserve represented vacuum wavelength in the native micrometre domain.
 * Multiplication by the approximate float literal .001 can shift an exact
 * matched-index boundary by one ULP even when all later arithmetic is precise. */
ccl_device_inline float dielectric_wavelength_um(const float wavelength_nm)
{
  return dielectric_precise_divide(wavelength_nm, 1000.0f);
}


/* OpenPBR Surface v1.1.1 (55)-(56), with the exact native Cycles d-line
 * constants and operation order. Indices are absolute, not backface-flipped.
 * Wavelength is in micrometres; table builders convert their vacuum nm once. */
ccl_device_inline float dielectric_ior_at_wavelength(const float ior_d,
                                                    const float inv_abbe,
                                                    const float wavelength_um)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  if (inv_abbe == 0.0f) return ior_d;
  constexpr float lambda_d = 0.5876f;
  constexpr float lambda_C = 0.6563f;
  constexpr float lambda_F = 0.4861f;
  constexpr float fac = 1.0f / (1.0f / (lambda_F * lambda_F) - 1.0f / (lambda_C * lambda_C));
  constexpr float inv_lambda_d_sq = 1.0f / (lambda_d * lambda_d);
  const float B = (ior_d - 1.0f) * inv_abbe * fac;
  /* The CPU backend can contract after inlining into a fast-math caller
   * despite local IR contraction flags. Materialize this native product's
   * rounded value before subtraction; the matched-index atom depends on it. */
  volatile float correction = B * inv_lambda_d_sq;
  const float A = ior_d - correction;
  return A + dielectric_precise_divide(B, wavelength_um * wavelength_um);
}

CCL_NAMESPACE_END

#if defined(__clang__) && !defined(__KERNEL_METAL__)
#  pragma float_control(pop)
#endif
