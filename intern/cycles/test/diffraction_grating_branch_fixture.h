/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: BSD-3-Clause */

/* Reference for kernel_diffraction_grating_test.cpp: the Diffraction BSDF closure of the
 * Al-Toto/blender `diffraction-bsdf` branch (commit ca7b0ef6a83, file
 * intern/cycles/kernel/closure/bsdf_diffraction.h), unchanged except that it lives in namespace
 * `altoto`, the setup and blur functions are left out and the enums are local. */

#pragma once

#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/sample/mapping.h"

CCL_NAMESPACE_BEGIN

namespace altoto {

enum DiffractionProfile {
  DIFFRACTION_PROFILE_SINUSOIDAL = 0,
  DIFFRACTION_PROFILE_RECTANGULAR = 1,
};

enum DiffractionGrating {
  DIFFRACTION_GRATING_1D = 0,
  DIFFRACTION_GRATING_2D = 1,
};

constexpr int DIFFRACTION_MAX_ORDER = 8;

/* Y-axis grating parameters, allocated separately with closure_alloc_extra()
 * (the same technique the Metallic BSDF uses for its FresnelConductor data) since
 * it is already at the fixed ShaderClosure budget with just the X axis. */
struct DiffractionExtraY {
  float pitch_um;       /* Grating pitch, Y axis, in micrometers. */
  float groove_depth_nm; /* Groove depth, Y axis, in nanometers. */
  float duty_cycle;      /* Pit/land width fraction, Y axis; rectangular profile only. */
  int grating_type;      /* DiffractionGrating: 1D ignores this struct's Y-axis fields. */

  /* Rotation of each axis's lattice vector within the tangent-plane (X,Y) frame,
   * in turns (0..1 = 0..360 degrees), independent of one another. */
  float rotation_x;
  float rotation_y;
};

struct DiffractionBsdf {
  SHADER_CLOSURE_BASE;

  float3 T; /* Groove tangent; the X grating vector is derived from this, like Glossy's anisotropy. */

  float roughness;
  float pitch_um;       /* Grating pitch, X axis, in micrometers */
  float groove_depth_nm; /* Groove depth, X axis, in nanometers; sets the per-order energy split. */
  float duty_cycle;      /* Pit/land width fraction, X axis; only used by the rectangular profile. */
  float wavelength_um;   /* Hero wavelength for this path, baked in at setup time. */

  int profile; /* DiffractionProfile. */

  ccl_private DiffractionExtraY *extra_y; /* Y-axis grating parameters, see above. */
};

static_assert(sizeof(ShaderClosure) >= sizeof(DiffractionBsdf), "DiffractionBsdf is too large!");

/* How far above the highest order actually needed to seed Miller's algorithm. */
ccl_device_inline int diffraction_bessel_miller_start(const int max_order, const float A)
{
  const int margin = max(10, (int)(1.6f * A) + 8);
  return min(max_order + margin, 64);
}

/* Compute the per-order diffraction efficiency for all orders in a single pass using
 * Miller's algorithm, then normalizes the result. */
ccl_device_inline void diffraction_fill_sinusoidal_efficiencies(const float A,
                                                                 const int max_order,
                                                                 ccl_private float *eta_out)
{
  /* Skip the recurrence entirely rather than push it into its own small-argument instability */
  if (A < 0.15f) {
    for (int m = -max_order; m <= max_order; m++) {
      eta_out[m + max_order] = (m == 0) ? 1.0f : 0.0f;
    }
    return;
  }

  const int n_start = diffraction_bessel_miller_start(max_order, A);
  float raw[DIFFRACTION_MAX_ORDER + 1];
  for (int i = 0; i <= max_order; i++) {
    raw[i] = 0.0f;
  }

  float J_next = 0.0f;
  float J_curr = 1.0f;
  float sum = 0.0f;
  for (int n = n_start; n >= 1; n--) {
    const float J_prev = (2.0f * (float)n / A) * J_curr - J_next;
    if (n - 1 <= max_order) {
      raw[n - 1] = J_prev;
    }
    if (((n - 1) & 1) == 0 && (n - 1) > 0) {
      sum += 2.0f * J_prev;
    }
    J_next = J_curr;
    J_curr = J_prev;
  }
  sum += raw[0];
  const float inv_sum = (fabsf(sum) > 1e-8f) ? 1.0f / sum : 0.0f;

  for (int m = -max_order; m <= max_order; m++) {
    const float J = raw[abs(m)] * inv_sum;
    eta_out[m + max_order] = sqr(J);
  }
}

ccl_device_inline float diffraction_efficiency_rectangular(const int m,
                                                            const float wavelength_nm,
                                                            const float cos_theta_i,
                                                            const float groove_depth_nm,
                                                            const float duty_cycle)
{
  /* Round-trip phase delay of the pit floor relative to the land. */
  const float phi = (M_2PI_F / wavelength_nm) * 2.0f * groove_depth_nm * cos_theta_i;
  const float g = duty_cycle;

  if (m == 0) {
    const float re = (1.0f - g) + g * cosf(phi);
    const float im = g * sinf(phi);
    return sqr(re) + sqr(im);
  }

  const float mg = (float)m * g;
  const float sinc_mg = (fabsf(mg) < 1e-6f) ? 1.0f : sinf(M_PI_F * mg) / (M_PI_F * mg);
  const float amp = 2.0f * sinf(0.5f * phi) * g * sinc_mg;
  return sqr(amp);
}

/* Precompute the efficiency of every diffraction order,
 * instead of recalculating each one. */
ccl_device_inline void diffraction_fill_efficiencies(const float wavelength_nm,
                                                      const float cos_theta_i,
                                                      const float groove_depth_nm,
                                                      const float duty_cycle,
                                                      const int profile,
                                                      const int max_order,
                                                      ccl_private float *eta_out)
{
  if (profile == DIFFRACTION_PROFILE_RECTANGULAR) {
    for (int m = -max_order; m <= max_order; m++) {
      eta_out[m + max_order] = diffraction_efficiency_rectangular(
          m, wavelength_nm, cos_theta_i, groove_depth_nm, duty_cycle);
    }
    return;
  }

  const float A = (M_2PI_F / wavelength_nm) * groove_depth_nm * cos_theta_i;
  diffraction_fill_sinusoidal_efficiencies(A, max_order, eta_out);
}

/* Highest order that can still leave the surface for this pitch/wavelength pair;
 * anything beyond it is evanescent and skipped entirely. */
ccl_device_inline int diffraction_max_propagating_order(const float pitch_nm,
                                                         const float wavelength_nm)
{
  const int bound = (int)ceilf(pitch_nm / wavelength_nm) + 1;
  return min(bound, DIFFRACTION_MAX_ORDER);
}

/* Unit direction (within the tangent-plane X/Y frame) of a lattice axis rotated
 * by `rotation_turns` from the reference tangent (X). */
ccl_device_inline float2 diffraction_axis_dir(const float rotation_turns)
{
  const float angle = rotation_turns * M_2PI_F;
  return make_float2(cosf(angle), sinf(angle));
}

/* Reciprocal lattice vector for a single periodic direction. */
ccl_device_inline float2 diffraction_reciprocal_1d(const float pitch_nm,
                                                    const float rotation_turns)
{
  return diffraction_axis_dir(rotation_turns) / pitch_nm;
}

/* Reciprocal basis for a 2D oblique lattice. */
ccl_device_inline void diffraction_reciprocal_2d(const float pitch_x_nm,
                                                  const float pitch_y_nm,
                                                  const float rotation_x_turns,
                                                  const float rotation_y_turns,
                                                  ccl_private float2 *b1,
                                                  ccl_private float2 *b2)
{
  const float2 a1 = pitch_x_nm * diffraction_axis_dir(rotation_x_turns);
  const float2 a2 = pitch_y_nm * diffraction_axis_dir(rotation_y_turns);
  const float det = a1.x * a2.y - a2.x * a1.y;
  const float inv_det = (fabsf(det) > 1e-6f) ? 1.0f / det : 0.0f;
  *b1 = make_float2(a2.y, -a2.x) * inv_det;
  *b2 = make_float2(-a1.y, a1.x) * inv_det;
}

ccl_device Spectrum bsdf_diffraction_eval(const ccl_private ShaderClosure *sc,
                                          const float3 wi,
                                          const float3 wo,
                                          ccl_private float *pdf)
{
  const ccl_private DiffractionBsdf *bsdf = (const ccl_private DiffractionBsdf *)sc;
  const ccl_private DiffractionExtraY *extra_y = bsdf->extra_y;
  const float3 N = bsdf->N;
  const float cos_NI = dot(N, wi);
  const float cos_NO = dot(N, wo);
  if (cos_NI <= 0.0f || cos_NO <= 0.0f) {
    *pdf = 0.0f;
    return zero_spectrum();
  }

  float3 X, Y;
  if (is_zero(bsdf->T)) {
    /* Fall back to an arbitrary basis */
    make_orthonormals(N, &X, &Y);
  }
  else {
    make_orthonormals_tangent(N, bsdf->T, &X, &Y);
  }

  const bool is_2d = (extra_y->grating_type == DIFFRACTION_GRATING_2D);

  const float pitch_x_nm = bsdf->pitch_um * 1000.0f;
  const float pitch_y_nm = extra_y->pitch_um * 1000.0f;
  const float wavelength_nm = bsdf->wavelength_um * 1000.0f;
  const float alpha2 = sqr(fmaxf(bsdf->roughness, 1e-4f));
  const int max_order_x = diffraction_max_propagating_order(pitch_x_nm, wavelength_nm);
  const int max_order_y = is_2d ? diffraction_max_propagating_order(pitch_y_nm, wavelength_nm) :
                                  0;

  /* Reciprocal lattice vectors. */
  float2 b1, b2;
  if (is_2d) {
    diffraction_reciprocal_2d(
        pitch_x_nm, pitch_y_nm, extra_y->rotation_x, extra_y->rotation_y, &b1, &b2);
  }
  else {
    b1 = diffraction_reciprocal_1d(pitch_x_nm, extra_y->rotation_x);
    b2 = zero_float2();
  }

  float eta_x[2 * DIFFRACTION_MAX_ORDER + 1];
  diffraction_fill_efficiencies(
      wavelength_nm, cos_NI, bsdf->groove_depth_nm, bsdf->duty_cycle, bsdf->profile, max_order_x, eta_x);
  float eta_y[2 * DIFFRACTION_MAX_ORDER + 1];
  if (is_2d) {
    diffraction_fill_efficiencies(wavelength_nm,
                                  cos_NI,
                                  extra_y->groove_depth_nm,
                                  extra_y->duty_cycle,
                                  bsdf->profile,
                                  max_order_y,
                                  eta_y);
  }

  Spectrum sum = zero_spectrum();
  float pdf_sum = 0.0f;
  float weight_sum = 0.0f;

  const float Ix = dot(X, wi);
  const float Iy = dot(Y, wi);

  for (int m = -max_order_x; m <= max_order_x; m++) {
    const float eta_m = eta_x[m + max_order_x];
    if (eta_m < 1e-6f) {
      continue;
    }
    /* Order m's own contribution to the shift, shared by every n at this m */
    const float shift_x_m = (float)m * wavelength_nm * b1.x;
    const float shift_y_m = (float)m * wavelength_nm * b1.y;

    for (int n = -max_order_y; n <= max_order_y; n++) {
      const float eta_n = is_2d ? eta_y[n + max_order_y] : 1.0f;
      const float eta_mn = eta_m * eta_n;
      if (eta_mn < 1e-6f) {
        continue;
      }

      /* Virtual incident direction carrying both axes' grating momentum
       * keeping a valid unit vector so the standard energy-conserving
       * half-vector formula still applies. */
      const float Ix_v = Ix + shift_x_m + (is_2d ? (float)n * wavelength_nm * b2.x : 0.0f);
      const float Iy_v = Iy + shift_y_m + (is_2d ? (float)n * wavelength_nm * b2.y : 0.0f);
      if (sqr(Ix_v) + sqr(Iy_v) > 1.0f) {
        continue; /* Evanescent for this order pair at this incidence direction. */
      }
      const float cos_NI_v = safe_sqrtf(1.0f - sqr(Ix_v) - sqr(Iy_v));
      const float3 wi_v = Ix_v * X + Iy_v * Y + cos_NI_v * N;

      const float3 H_un = wi_v + wo;
      const float H_len = len(H_un);
      if (H_len < 1e-6f) {
        continue;
      }
      const float3 H = H_un / H_len;
      const float cos_NH = dot(N, H);
      if (cos_NH <= 0.0f) {
        continue;
      }

      const float D = bsdf_D<MicrofacetType::GGX>(alpha2, cos_NH);
      const float G = bsdf_G<MicrofacetType::GGX>(alpha2, cos_NI_v, cos_NO);
      const float lobe = eta_mn * D * G / (4.0f * cos_NI_v);
      sum += make_spectrum(lobe);

      const float cos_HI = dot(H, wi_v);
      if (cos_HI > 0.0f) {
        const float G1 = bsdf_G<MicrofacetType::GGX>(alpha2, cos_NI_v);
        pdf_sum += eta_mn * D * G1 / (4.0f * cos_NI_v);
      }
      weight_sum += eta_mn;
    }
  }

  *pdf = (weight_sum > 0.0f) ? pdf_sum / weight_sum : 0.0f;
  return sum;
}

ccl_device int bsdf_diffraction_sample(const ccl_private ShaderClosure *sc,
                                       const float3 Ng,
                                       const float3 wi,
                                       const float3 rand,
                                       ccl_private Spectrum *eval,
                                       ccl_private float3 *wo,
                                       ccl_private float *pdf,
                                       ccl_private float2 *sampled_roughness,
                                       ccl_private float *eta)
{
  *eta = 1.0f;

  const ccl_private DiffractionBsdf *bsdf = (const ccl_private DiffractionBsdf *)sc;
  const ccl_private DiffractionExtraY *extra_y = bsdf->extra_y;
  *sampled_roughness = make_float2(bsdf->roughness, bsdf->roughness);

  const float3 N = bsdf->N;
  const float cos_NI = dot(N, wi);
  if (cos_NI <= 0.0f) {
    return LABEL_NONE;
  }

  float3 X, Y;
  if (is_zero(bsdf->T)) {
    /* No groove tangent available (e.g. mesh without UVs): fall back to an arbitrary
     * but consistent basis. */
    make_orthonormals(N, &X, &Y);
  }
  else {
    make_orthonormals_tangent(N, bsdf->T, &X, &Y);
  }
  const float Ix = dot(X, wi);
  const float Iy = dot(Y, wi);

  const bool is_2d = (extra_y->grating_type == DIFFRACTION_GRATING_2D);

  const float pitch_x_nm = bsdf->pitch_um * 1000.0f;
  const float pitch_y_nm = extra_y->pitch_um * 1000.0f;
  const float wavelength_nm = bsdf->wavelength_um * 1000.0f;
  const int max_order_x = diffraction_max_propagating_order(pitch_x_nm, wavelength_nm);
  const int max_order_y = is_2d ? diffraction_max_propagating_order(pitch_y_nm, wavelength_nm) :
                                  0;
  const int order_count_x = 2 * max_order_x + 1;
  const int order_count_y = 2 * max_order_y + 1;

  float2 b1, b2;
  if (is_2d) {
    diffraction_reciprocal_2d(
        pitch_x_nm, pitch_y_nm, extra_y->rotation_x, extra_y->rotation_y, &b1, &b2);
  }
  else {
    b1 = diffraction_reciprocal_1d(pitch_x_nm, extra_y->rotation_x);
    b2 = zero_float2();
  }

  float eta_x[2 * DIFFRACTION_MAX_ORDER + 1];
  diffraction_fill_efficiencies(
      wavelength_nm, cos_NI, bsdf->groove_depth_nm, bsdf->duty_cycle, bsdf->profile, max_order_x, eta_x);
  float eta_y[2 * DIFFRACTION_MAX_ORDER + 1];
  if (is_2d) {
    diffraction_fill_efficiencies(wavelength_nm,
                                  cos_NI,
                                  extra_y->groove_depth_nm,
                                  extra_y->duty_cycle,
                                  bsdf->profile,
                                  max_order_y,
                                  eta_y);
  }

  /* Pick which order pair (m,n) to sample, weighted by its efficiency at the incidence angle. */
  float weights[(2 * DIFFRACTION_MAX_ORDER + 1) * (2 * DIFFRACTION_MAX_ORDER + 1)];
  float weight_sum = 0.0f;
  for (int i = 0; i < order_count_x; i++) {
    const int m = i - max_order_x;
    const float shift_x_m = (float)m * wavelength_nm * b1.x;
    const float shift_y_m = (float)m * wavelength_nm * b1.y;
    for (int j = 0; j < order_count_y; j++) {
      const int n = j - max_order_y;
      const int idx = i * order_count_y + j;
      const float Ix_shifted_i = Ix + shift_x_m + (is_2d ? (float)n * wavelength_nm * b2.x : 0.0f);
      const float Iy_shifted_j = Iy + shift_y_m + (is_2d ? (float)n * wavelength_nm * b2.y : 0.0f);
      if (sqr(Ix_shifted_i) + sqr(Iy_shifted_j) > 1.0f) {
        weights[idx] = 0.0f;
        continue;
      }
      const float eta_mn = eta_x[i] * (is_2d ? eta_y[j] : 1.0f);
      weights[idx] = eta_mn;
      weight_sum += eta_mn;
    }
  }
  if (weight_sum <= 0.0f) {
    return LABEL_NONE;
  }

  float u = rand.z * weight_sum;
  int m = -max_order_x;
  int n = -max_order_y;
  const int order_count = order_count_x * order_count_y;
  for (int idx = 0; idx < order_count; idx++) {
    if (u < weights[idx] || idx == order_count - 1) {
      m = (idx / order_count_y) - max_order_x;
      n = (idx % order_count_y) - max_order_y;
      break;
    }
    u -= weights[idx];
  }

  /* Sample a microfacet normal around the shifted virtual incident direction */
  const float shift_x = (float)m * wavelength_nm * b1.x + (is_2d ? (float)n * wavelength_nm * b2.x : 0.0f);
  const float shift_y = (float)m * wavelength_nm * b1.y + (is_2d ? (float)n * wavelength_nm * b2.y : 0.0f);
  const float Ix_shifted = Ix + shift_x;
  const float Iy_shifted = Iy + shift_y;
  const float Iz_shifted = safe_sqrtf(fmaxf(0.0f, 1.0f - sqr(Ix_shifted) - sqr(Iy_shifted)));
  const float3 local_I_shifted = make_float3(Ix_shifted, Iy_shifted, Iz_shifted);

  const float alpha = fmaxf(bsdf->roughness, 1e-4f);
  const float3 local_H = microfacet_ggx_sample_vndf(local_I_shifted, alpha, alpha, make_float2(rand));
  const float3 H = to_global(local_H, X, Y, N);

  /* Reflect the shifted virtual incident direction. */
  const float3 wi_v = local_I_shifted.x * X + local_I_shifted.y * Y + local_I_shifted.z * N;
  const float cos_HI = dot(H, wi_v);
  if (cos_HI <= 0.0f) {
    return LABEL_NONE;
  }
  *wo = 2.0f * cos_HI * H - wi_v;

  const float cos_NO = dot(N, *wo);
  const float cos_NgO = dot(Ng, *wo);
  if (cos_NO <= 0.0f || cos_NgO <= 0.0f) {
    return LABEL_NONE;
  }

  *eval = bsdf_diffraction_eval(sc, wi, *wo, pdf);
  if (*pdf <= 0.0f) {
    return LABEL_NONE;
  }

  return LABEL_REFLECT | LABEL_GLOSSY;
}

}  // namespace altoto

CCL_NAMESPACE_END
