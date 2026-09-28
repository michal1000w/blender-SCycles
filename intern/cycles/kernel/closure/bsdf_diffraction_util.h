/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Project the grating axis onto an individual microfacet. The caller must use
 * the same axis convention in sampling and evaluation. A facet parallel to the
 * axis is degenerate and has zero measure in a continuous normal distribution. */
ccl_device_inline float3 diffraction_facet_axis(const float3 h, const float3 axis)
{
  /* Double cross keeps the result perpendicular even when axis and h almost
   * coincide; direct subtraction magnifies unit-length roundoff in that case. */
  return safe_normalize(cross(cross(h, axis), h));
}

/* Reflect from a grating carried by the facet h. The tangential momentum kick
 * is delta = order * wavelength / pitch, measured across its grooves. */
ccl_device_inline bool diffraction_facet_reflect(
    const float3 wi, const float3 h, const float3 axis, const float delta, ccl_private float3 *wo)
{
  *wo = zero_float3();
  const float ci = dot(wi, h);
  if (delta == 0.0f) {
    /* Avoid recovering cos(theta) from 1-sin(theta)^2 at grazing incidence.
     * Its cancellation amplifies CPU/Metal rounding in reverse densities. */
    if (!(ci > 0.0f)) {
      return false;
    }
    *wo = 2.0f * ci * h - wi;
    return true;
  }
  const float3 e = diffraction_facet_axis(h, axis);
  if (!(ci > 0.0f) || is_zero(e)) {
    return false;
  }
  const float3 tangent = delta * e - (wi - ci * h);
  const float co2 = 1.0f - len_squared(tangent);
  if (!(co2 > 0.0f)) {
    return false;
  }
  *wo = tangent + sqrtf(co2) * h;
  return true;
}

/* Invert the reflection above. Diffraction can give TWO facet normals for the
 * same direction pair and order, so evaluation must sum both preimages. For
 * delta=0 this reduces to the ordinary reflection half-vector.
 *
 * wi + wo = a h + delta e(h), a = cos_i + cos_o > 0.
 * h lies in the plane spanned by wi+wo and axis. Its two candidate tilts have
 * sin(theta)=delta/|wi+wo|. The projection-axis sign selects the valid roots. */
struct DiffractionReflectionRoot {
  float3 h;
  float3 e;
};

ccl_device_inline int diffraction_reflection_half_vectors(
    const float3 wi,
    const float3 wo,
    const float3 axis,
    const float delta,
    ccl_private DiffractionReflectionRoot *roots)
{
  const float3 v = wi + wo;
  const float length2 = len_squared(v);
  if (!(length2 > delta * delta)) {
    return 0;
  }
  const float3 V = v / sqrtf(length2);
  if (delta == 0.0f) {
    roots[0].h = V;
    roots[0].e = diffraction_facet_axis(V, axis);
    return 1;
  }
  /* Work in the axis/B frame. Retain the facet axis e explicitly: recomputing
   * it by subtracting almost parallel Cartesian vectors loses its direction
   * for roots that approach the grating axis. */
  const float t = dot(axis, V);
  const float3 B = safe_normalize(cross(cross(axis, V), axis));
  if (is_zero(B)) {
    /* Exact axial symmetry is a singular ring of normals, not an isolated
     * preimage. It has zero solid-angle measure and no finite directional PDF. */
    return 0;
  }
  const float s = delta / sqrtf(length2);
  const float c = safe_sqrtf(1.0f - s * s);
  const float transverse = dot(V, B);
  int count = 0;
  if (t * s + transverse * c > 0.0f) {
    const float x = c * t - s * transverse;
    const float y = c * transverse + s * t;
    const float3 candidate = normalize(x * axis + y * B);
    if (dot(candidate, wi) > 0.0f && dot(candidate, wo) > 0.0f) {
      roots[count++] = {candidate, normalize(y * axis - x * B)};
    }
  }
  if (t * s - transverse * c > 0.0f) {
    const float x = c * t + s * transverse;
    const float y = c * transverse - s * t;
    const float3 candidate = normalize(x * axis + y * B);
    if (dot(candidate, wi) > 0.0f && dot(candidate, wo) > 0.0f) {
      roots[count++] = {candidate, normalize(-y * axis + x * B)};
    }
  }
  return count;
}

/* Solid-angle Jacobian |d omega_h / d omega_o| for each nondegenerate root.
 * Projecting d(wi+wo) onto the facet gives eigenvalues a along e and
 * b=a-delta*(axis.h)/|project(axis)| across e. Projection from the outgoing
 * sphere contributes |wo.h|. The zero-order limit is 1/(4*wi.h).
 * Swapping wi and wo gives the reverse Jacobian: ci*J_forward=co*J_reverse. */
ccl_device_inline float diffraction_reflection_jacobian(
    const float3 wi, const float3 wo, const float3 h, const float3 axis, const float delta)
{
  const float ci = dot(wi, h), co = dot(wo, h);
  const float t = dot(axis, h);
  const float projection = len(cross(axis, h));
  if (!(ci > 0.0f && co > 0.0f && projection > 0.0f)) {
    return 0.0f;
  }
  const float a = ci + co;
  const float determinant = fabsf(a * (a - delta * t / projection));
  return determinant > 0.0f ? co / determinant : 0.0f;
}

/* Transmitted facet geometry. eta=n_in/n_out and delta=m*lambda_vacuum/
 * (n_out*pitch). Directions point away from the interface; h faces wi.
 * These helpers specify direction measure only, not diffraction efficiency. */
ccl_device_inline bool diffraction_facet_transmit(const float3 wi,
                                                 const float3 h,
                                                 const float3 axis,
                                                 const float eta,
                                                 const float delta,
                                                 ccl_private float3 *wo)
{
  *wo = zero_float3();
  const float ci = dot(wi, h);
  if (!(eta > 0.0f) || !isfinite_safe(eta) || !isfinite_safe(delta) || !(ci > 0.0f))
    return false;
  if (delta == 0.0f && eta == 1.0f) {
    *wo = -wi;
    return true;
  }
  const float3 e = delta == 0.0f ? zero_float3() : diffraction_facet_axis(h, axis);
  if (delta != 0.0f && is_zero(e))
    return false;
  const float3 tangent = delta * e - eta * (wi - ci * h);
  /* Match the ordinary dielectric Snell construction for order zero. Using
   * the rounded transverse vector norm needlessly perturbs its Fresnel angle. */
  const float inverse_eta = 1.0f / eta;
  const float ct2 = delta == 0.0f ?
      (inverse_eta*inverse_eta - (1.0f-ci*ci)) / (inverse_eta*inverse_eta) :
      1.0f - len_squared(tangent);
  if (!(ct2 > 0.0f))
    return false;
  *wo = tangent - sqrtf(ct2) * h;
  return true;
}

struct DiffractionTransmissionRoot {
  float3 h;
  float3 e;
};

/* Round both products before addition so swapping interface endpoints does
 * not choose a different product for fused contraction. */
ccl_device_inline float diffraction_symmetric_product_sum(
    const float a,const float x,const float b,const float y)
{
  volatile float ax=a*x;
  volatile float by=b*y;
  return ax+by;
}
ccl_device_inline float3 diffraction_symmetric_momentum(
    const float ni,const float3 wi,const float no,const float3 wo)
{
  return make_float3(diffraction_symmetric_product_sum(ni,wi.x,no,wo.x),
                     diffraction_symmetric_product_sum(ni,wi.y,no,wo.y),
                     diffraction_symmetric_product_sum(ni,wi.z,no,wo.z));
}

/* Resolve the sign and magnitude of |v|^2-kick^2 near a diffraction fold.
 * Product residuals and compensated additions recover the cancellation bits
 * without requiring double precision on Metal. Volatile intermediates prevent
 * fast-math reassociation from deleting the error terms. */
ccl_device_inline float diffraction_fold_discriminant(const float3 v, const float kick)
{
  const float length2=len_squared(v), kick2=kick*kick;
  const float approximate=length2-kick2;
  if (fabsf(approximate)>1e-3f*max(length2,kick2)) return approximate;
  const float terms[4]={v.x,v.y,v.z,kick};
  float sum=0,correction=0;
  for(int i=0;i<4;++i) {
    volatile float product=terms[i]*terms[i];
#ifdef __KERNEL_METAL__
    const float residual=metal::fma(terms[i],terms[i],-product);
#else
    const float residual=fmaf(terms[i],terms[i],-product);
#endif
    const float term=i==3?-product:product;
    volatile float next=sum+term;
    volatile float back=next-sum;
    volatile float left=next-back;
    volatile float error_left=sum-left;
    volatile float error_right=term-back;
    correction+=(error_left+error_right)+(i==3?-residual:residual);
    sum=next;
  }
  return sum+correction;
}

ccl_device_inline int diffraction_transmission_half_vectors_momentum(
    const float3 wi, const float3 wo, const float3 axis, const float3 v, const float delta,
    ccl_private DiffractionTransmissionRoot *roots)
{
  if (!isfinite_safe(v) || !isfinite_safe(delta)) return 0;
  const float length2 = len_squared(v);
  const float discriminant=diffraction_fold_discriminant(v,delta);
  if (!(discriminant > 0.0f))
    return 0;
  const float3 V = v / sqrtf(length2);
  if (delta == 0.0f) {
    const float3 h = dot(V, wi) > 0.0f ? V : -V;
    if (!(dot(h, wi) > 0.0f && dot(h, wo) < 0.0f))
      return 0;
    roots[0] = {h, diffraction_facet_axis(h, axis)};
    return 1;
  }
  const float3 B = safe_normalize(cross(cross(axis, V), axis));
  if (is_zero(B))
    return 0; /* Axial singular ring, not a finite directional density. */
  const float t = dot(axis, V), u = dot(B, V);
  /* Preserve the positive discriminant established above. Computing 1-s*s
   * can round to zero even for a propagating root, creating an artificial
   * zero-width fold and an enormous erroneous density. */
  const float s = delta / sqrtf(length2);
  const float c = sqrtf(discriminant / length2);
  int count = 0;
  /* Unlike reflection, a=eta*cos_i+cos_o can have either sign. Enumerate
   * both, then orient h toward wi. Opposite normals describe the same axis;
   * the hemisphere tests retain at most two distinct transmission roots. */
  for (int normal_sign = -1; normal_sign <= 1; normal_sign += 2) {
    for (int axis_sign = -1; axis_sign <= 1; axis_sign += 2) {
      const float x = normal_sign * c * t - axis_sign * s * u;
      const float y = normal_sign * c * u + axis_sign * s * t;
      if (!(axis_sign * y > 0.0f))
        continue;
      const float3 h = normalize(x * axis + y * B);
      if (dot(h, wi) > 0.0f && dot(h, wo) < 0.0f)
        roots[count++] = {h, normalize(axis_sign * (y * axis - x * B))};
    }
  }
  return count;
}

ccl_device_inline int diffraction_transmission_half_vectors(
    const float3 wi, const float3 wo, const float3 axis, const float eta, const float delta,
    ccl_private DiffractionTransmissionRoot *roots)
{
  if (!(eta>0) || !isfinite_safe(eta)) return 0;
  return diffraction_transmission_half_vectors_momentum(wi,wo,axis,eta*wi+wo,delta,roots);
}

/* |d omega_h / d omega_o|. The zero-order limit is the usual refractive
 * half-vector Jacobian |cos_o|/(eta*cos_i+cos_o)^2. Index-matched straight
 * transmission is a separate delta event and has no finite facet density. */
ccl_device_inline float diffraction_transmission_jacobian(
    const float3 wi, const float3 wo, const float3 h, const float3 axis,
    const float eta, const float delta)
{
  const float ci = dot(wi, h), co = dot(wo, h);
  if (!(eta > 0.0f && ci > 0.0f && co < 0.0f) || !isfinite_safe(eta) ||
      !isfinite_safe(delta))
    return 0.0f;
  /* Rationalize eta*cos_i-cos_t for nearly index-matched zero-order
   * refraction. Direct subtraction loses relative precision as eta -> 1. */
  const float a = delta == 0.0f ? ((eta - 1.0f) * (eta + 1.0f)) / (eta * ci - co) :
                                eta * ci + co;
  float b = a;
  if (delta != 0.0f) {
    const float projection = len(cross(axis, h));
    if (!(projection > 0.0f))
      return 0.0f;
    b -= delta * dot(axis, h) / projection;
  }
  const float determinant = fabsf(a * b);
  return determinant > 0.0f ? -co / determinant : 0.0f;
}

/* Absolute-index form: reversal swaps ni/no without reconstructing either
 * from a rounded ratio. The outgoing solid-angle derivative contributes no^2. */
ccl_device_inline float diffraction_transmission_jacobian_indices(
    const float3 wi,const float3 wo,const float3 h,const float3 axis,
    const float ni,const float no,const float kick)
{
  const float ci=dot(wi,h),co=dot(wo,h);
  if (!(ni>0 && no>0 && ci>0 && co<0)) return 0;
  const float3 momentum=diffraction_symmetric_momentum(ni,wi,no,wo);
  float a=kick==0 ? ((ni-no)*(ni+no))/(ni*ci-no*co):dot(momentum,h);
  if (kick!=0 && a*a<1e-3f*len_squared(momentum)) {
    /* v=a*h+kick*e and h.e=0 imply a^2=|v|^2-kick^2. Recover its
     * magnitude at folds instead of cancelling rounded components of h. */
    a=copysignf(sqrtf(max(0.0f,diffraction_fold_discriminant(momentum,kick))),a);
  }
  float b=a;
  if(kick!=0) {
    const float projection=len(cross(axis,h));
    if(!(projection>0))return 0;
    /* Project v=a*h+kick*e perpendicular to the groove axis:
     * |v_perp|=|a-kick*(axis.h)/|h_perp||*|h_perp|.
     * This avoids a second cancellation near the axial singular ring. */
    b=len(cross(axis,momentum))/projection;
  }
  const float determinant=fabsf(a*b);
  return determinant>0 ? no*no*(-co)/determinant:0;
}

/* Local grating frame: x is across the grooves, y is along them, z is the normal.
 * Both directions point away from the interface, as elsewhere in Cycles. Wavelength
 * and pitch must use the same length unit. eta is n_out / n_in and wavelength is
 * measured in the incident medium. This also handles conical incidence. */
ccl_device_inline bool diffraction_order_direction(const float3 wi,
                                                   const float wavelength_over_pitch,
                                                   const int order,
                                                   const float eta,
                                                   const bool transmission,
                                                   ccl_private float3 *wo)
{
  *wo = zero_float3();
  if (!(eta > 0.0f) || !(wavelength_over_pitch > 0.0f) || !(wi.z > 0.0f)) {
    return false;
  }
  const float x = (order * wavelength_over_pitch - wi.x) / eta;
  const float y = -wi.y / eta;
  const float z2 = 1.0f - x * x - y * y;
  /* A grazing or evanescent order transports no power away from the interface. */
  if (!(z2 > 0.0f)) {
    return false;
  }
  *wo = make_float3(x, y, (transmission ? -1.0f : 1.0f) * sqrtf(z2));
  return true;
}

/* Squared Fourier coefficient of a binary, lossless thin phase screen. The
 * fraction `duty` has phase `phase`, the remainder has phase zero. This is a
 * scalar Fourier coefficient, NOT an electromagnetic diffraction efficiency:
 * propagating-mode flux factors and interface scattering belong to the BSDF.
 * Summing over all integer orders gives one (Parseval's identity). */
ccl_device_inline float diffraction_binary_power(const int order,
                                                 const float phase,
                                                 const float duty)
{
  const float f = saturatef(duty);
  if (f == 0.0f || f == 1.0f) {
    return order == 0 ? 1.0f : 0.0f;
  }
  const float modulation = 4.0f * sqr(sinf(0.5f * phase));
  if (order == 0) {
    return max(0.0f, 1.0f - modulation * f * (1.0f - f));
  }
  const float m_pi = float(order) * M_PI_F;
  return modulation * sqr(sinf(m_pi * f) / m_pi);
}

/* Complex Fourier coefficient (real, imaginary), with the raised interval at
 * [origin, origin + duty] modulo one period. Unlike power, this retains the
 * relative phase required when coherently combining different surface patches.
 * The Fourier convention is integral t(x) exp(-i 2 pi m x) dx. */
ccl_device_inline float2 diffraction_binary_amplitude(const int order,
                                                      const float phase,
                                                      const float duty,
                                                      const float origin)
{
  const float f = saturatef(duty);
  /* exp(i phase) - 1, avoiding cancellation for a shallow grating. */
  const float2 delta = make_float2(-2.0f * sqr(sinf(0.5f * phase)), sinf(phase));
  if (order == 0) {
    return make_float2(1.0f, 0.0f) + f * delta;
  }
  if (f == 0.0f || f == 1.0f) {
    return zero_float2();
  }
  const float m_pi = float(order) * M_PI_F;
  const float envelope = sinf(m_pi * f) / m_pi;
  const float angle = -m_pi * (f + 2.0f * origin);
  const float c = cosf(angle), s = sinf(angle);
  return envelope * make_float2(delta.x * c - delta.y * s, delta.x * s + delta.y * c);
}

/* Phase difference between the two levels of a relief grating, using the
 * surface scattering vector. Reflection has wo.z > 0, transmission wo.z < 0.
 * This form is invariant under exchanging source and receiver with their IORs. */
ccl_device_inline float diffraction_relief_phase(const float height,
                                                 const float vacuum_wavelength,
                                                 const float n_i,
                                                 const float n_o,
                                                 const float cos_i,
                                                 const float cos_o)
{
  return M_2PI_F * (height / vacuum_wavelength) * (n_i * cos_i + n_o * cos_o);
}

CCL_NAMESPACE_END
