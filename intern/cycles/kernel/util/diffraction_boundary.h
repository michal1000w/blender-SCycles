/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/util/diffraction_reference.h"

CCL_NAMESPACE_BEGIN

/* Error-free transforms require ordered operations even when the surrounding
 * Metal kernel uses fast math. This is a two-float expansion, not IEEE FP64. */
#if defined(__KERNEL_METAL__)
#  define DIFFRACTION_PRECISE_SCOPE \
    _Pragma("clang fp reassociate(off)") _Pragma("clang fp contract(off)")
#else
#  define DIFFRACTION_PRECISE_SCOPE
#  if defined(__clang__)
#    pragma float_control(precise, on, push)
#  endif
#endif

ccl_device_inline float2 diffraction_extended_sum(const float a, const float b)
{
  DIFFRACTION_PRECISE_SCOPE
  const float sum = a + b;
  const float second = sum - a;
  const float error = (a - (sum - second)) + (b - second);
  return make_float2(sum, error);
}

ccl_device_inline float2 diffraction_extended_add(const float2 a, const float2 b)
{
  DIFFRACTION_PRECISE_SCOPE
  const float2 sum = diffraction_extended_sum(a.x, b.x);
  return diffraction_extended_sum(sum.x, (sum.y + a.y) + b.y);
}

ccl_device_inline float2 diffraction_extended_mul(const float2 a, const float2 b)
{
  DIFFRACTION_PRECISE_SCOPE
  const float product = a.x * b.x;
#if defined(__KERNEL_METAL__)
  const float residual = metal::fma(a.x, b.x, -product);
#else
  const float residual = fmaf(a.x, b.x, -product);
#endif
  const float error = ((residual + a.x * b.y) + a.y * b.x) + a.y * b.y;
  return diffraction_extended_sum(product, error);
}

ccl_device_inline float2 diffraction_extended_div(const float2 a, const float2 b)
{
  DIFFRACTION_PRECISE_SCOPE
  const float quotient = a.x / b.x;
  const float2 remainder = diffraction_extended_add(
      a, -diffraction_extended_mul(b, make_float2(quotient, 0.0f)));
  return diffraction_extended_sum(quotient, (remainder.x + remainder.y) / b.x);
}

ccl_device_inline float2 diffraction_extended_sqrt(const float2 a)
{
  DIFFRACTION_PRECISE_SCOPE
  const float root = sqrtf(a.x);
  const float2 remainder = diffraction_extended_add(
      a, -diffraction_extended_mul(make_float2(root, 0.0f), make_float2(root, 0.0f)));
  return diffraction_extended_sum(root, (remainder.x + remainder.y) / (2.0f * root));
}

/* Exterior coefficients for one diffraction order relative to the incoming
 * order. Direction is expressed in the grating frame and normalized here.
 * wavelength/pitch are vacuum lengths in the same units; both exterior indices
 * are real. q2 is the signed squared longitudinal momentum divided by k0^2. */
struct DiffractionGratingBoundary {
  float2 r_te, r_tm;
  float2 transmission;
  float2 tangent_direction;
  float q2;
};

/* Compensated tangential momentum and signed longitudinal square, shared by
 * Fresnel matching and outgoing-direction construction. No Fresnel work here. */
struct DiffractionGratingMomentum {
  float2 x, y;
  float q2;
};

ccl_device_inline bool diffraction_grating_momentum(
    const float3 direction,
    const float incident_index,
    const float exterior_index,
    const float wavelength,
    const float pitch,
    const int order,
    ccl_private DiffractionGratingMomentum *momentum)
{
  DIFFRACTION_PRECISE_SCOPE
  if (!(incident_index > 0.0f) || !(exterior_index > 0.0f) || !(wavelength > 0.0f) ||
      !(pitch > 0.0f) || !isfinite_safe(incident_index) || !isfinite_safe(exterior_index) ||
      !isfinite_safe(wavelength) || !isfinite_safe(pitch) || !isfinite_safe(direction.x) ||
      !isfinite_safe(direction.y) || !isfinite_safe(direction.z))
  {
    return false;
  }
  const float2 dx = make_float2(direction.x, 0.0f), dy = make_float2(direction.y, 0.0f);
  const float2 dz = make_float2(direction.z, 0.0f);
  const float2 norm2 = diffraction_extended_add(
      diffraction_extended_add(diffraction_extended_mul(dx, dx), diffraction_extended_mul(dy, dy)),
      diffraction_extended_mul(dz, dz));
  if (!(norm2.x > 0.0f) || !isfinite_safe(norm2.x)) {
    return false;
  }
  const float2 n_in = make_float2(incident_index, 0.0f);
  const float2 n_out = make_float2(exterior_index, 0.0f);
  const float2 scale = diffraction_extended_div(n_in, diffraction_extended_sqrt(norm2));
  const float2 x = diffraction_extended_mul(scale, dx);
  const float2 y = diffraction_extended_mul(scale, dy);
  const float2 z = diffraction_extended_mul(scale, dz);
  /* Split before conversion: float(order) alone loses unit-order increments
   * beyond 2^24, and converting a rounded INT_MAX back to int is undefined. */
  const int order_high = (order / 4096) * 4096;
  const float2 order_extended = diffraction_extended_sum(float(order_high), float(order - order_high));
  const float2 delta = diffraction_extended_mul(
      order_extended,
      diffraction_extended_div(make_float2(wavelength, 0.0f), make_float2(pitch, 0.0f)));
  const float2 index_difference = diffraction_extended_mul(diffraction_extended_add(n_out, -n_in),
                                                           diffraction_extended_add(n_out, n_in));
  /* Preserve the known incident normal component. Reconstructing it as
   * n^2-kx^2-ky^2 loses grazing incoming rays before any order is added. */
  float2 q2 = diffraction_extended_add(index_difference, diffraction_extended_mul(z, z));
  q2 = diffraction_extended_add(q2, -2.0f * diffraction_extended_mul(x, delta));
  q2 = diffraction_extended_add(q2, -diffraction_extended_mul(delta, delta));
  momentum->q2 = q2.x + q2.y;
  if (!isfinite_safe(momentum->q2)) {
    return false;
  }
  momentum->x = diffraction_extended_add(x, delta);
  momentum->y = y;
  return isfinite_safe(momentum->x.x) && isfinite_safe(momentum->x.y) &&
         isfinite_safe(momentum->y.x) && isfinite_safe(momentum->y.y);
}

ccl_device_inline bool diffraction_grating_boundary(
    const float3 direction,
    const float incident_index,
    const float exterior_index,
    const float wavelength,
    const float pitch,
    const int order,
    ccl_private DiffractionGratingBoundary *boundary)
{
  DIFFRACTION_PRECISE_SCOPE
  DiffractionGratingMomentum momentum;
  if (!diffraction_grating_momentum(
          direction, incident_index, exterior_index, wavelength, pitch, order, &momentum))
    return false;
  boundary->q2 = momentum.q2;
  const float px = momentum.x.x + momentum.x.y, py = momentum.y.x + momentum.y.y;
  const float transverse = sqrtf(px * px + py * py);
  boundary->tangent_direction = transverse > 0.0f ? make_float2(px, py) / transverse :
                                                    make_float2(1.0f, 0.0f);
  const float q = sqrtf(fabsf(boundary->q2));
  const float index2 = exterior_index * exterior_index;
  if (boundary->q2 > 0.0f) {
    boundary->r_te = make_float2((1.0f - q) / (1.0f + q), 0.0f);
    boundary->r_tm = make_float2((q - index2) / (q + index2), 0.0f);
    const float root = sqrtf(q);
    boundary->transmission = make_float2(2.0f * root / (1.0f + q),
                                         2.0f * exterior_index * root / (q + index2));
  }
  else {
    boundary->r_te = diffraction_complex_div(make_float2(1.0f, -q), make_float2(1.0f, q));
    boundary->r_tm = diffraction_complex_div(make_float2(-index2, q), make_float2(index2, q));
    boundary->transmission = zero_float2();
  }
  return isfinite_safe(boundary->r_te.x) && isfinite_safe(boundary->r_te.y) &&
         isfinite_safe(boundary->r_tm.x) && isfinite_safe(boundary->r_tm.y) &&
         isfinite_safe(boundary->transmission.x) && isfinite_safe(boundary->transmission.y) &&
         isfinite_safe(boundary->tangent_direction.x) &&
         isfinite_safe(boundary->tangent_direction.y);
}

#if defined(__clang__) && !defined(__KERNEL_METAL__)
#  pragma float_control(pop)
#endif

#undef DIFFRACTION_PRECISE_SCOPE

CCL_NAMESPACE_END
