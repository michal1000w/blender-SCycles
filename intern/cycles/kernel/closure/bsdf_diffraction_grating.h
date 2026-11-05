/* SPDX-FileCopyrightText: 2011-2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Diffraction BSDF, Grating model: reflection off a periodic 1D or 2D (oblique lattice) surface
 * grating with a sinusoidal or rectangular groove profile.
 *
 * Port of the Diffraction BSDF of the Al-Toto/blender `diffraction-bsdf` branch, which implements
 * "Rendering Iridescent Colors of Optical Disks" (Sun et al., EGWR 2000): for the hero wavelength
 * of the path, every propagating order (m, n) gets the scalar thin-phase efficiency of the groove
 * profile, and is rendered as a GGX lobe around the specular reflection of a virtual incident
 * direction that carries the grating momentum m * b1 + n * b2 in the tangent plane.
 *
 * Differences to the branch (same values and pdf, see doc/cycles_diffraction_grating.md):
 * - The lattice is resolved at setup: the closure stores the world space momentum of one order
 *   along each axis, so evaluation needs no tangent frame and the closure fits into one
 *   ShaderClosure (the branch needed an extra closure slot).
 * - The order pick walks the orders twice instead of keeping a 17 x 17 weight array on the stack.
 * - Miller's recurrence for the Bessel functions rescales instead of overflowing. */

#pragma once

#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/closure/bsdf_util.h"
#include "kernel/sample/mapping.h"
#include "kernel/svm/types.h"
#include "kernel/util/colorspace.h"

CCL_NAMESPACE_BEGIN

/* Highest diffraction order evaluated per axis. */
#define DIFFRACTION_GRATING_MAX_ORDER 8
#define DIFFRACTION_GRATING_MAX_ORDERS (2 * DIFFRACTION_GRATING_MAX_ORDER + 1)

/* Efficiencies below this are skipped, as in the branch. */
#define DIFFRACTION_GRATING_EFFICIENCY_CUTOFF 1e-6f

/* Flags in DiffractionGratingBsdf::packed. */
#define DIFFRACTION_GRATING_PACKED_RECTANGULAR (1u << 0)
#define DIFFRACTION_GRATING_PACKED_2D (1u << 1)
#define DIFFRACTION_GRATING_PACKED_ORDER_X_SHIFT 8u
#define DIFFRACTION_GRATING_PACKED_ORDER_Y_SHIFT 16u

struct DiffractionGratingBsdf {
  SHADER_CLOSURE_BASE;

  /* Tangential momentum of order (1, 0), as a shift of the unit incident direction in world
   * space: wavelength times the first reciprocal lattice vector. */
  float3 K1;

  /* Momentum of order (0, 1) in the frame (K1 / |K1|, N x K1 / |K1|). Zero for 1D gratings. */
  float k2_along, k2_across;

  /* GGX alpha of every order. */
  float alpha;

  /* Efficiency parameters per axis: 2 pi depth / wavelength, and duty cycle (rectangular). */
  float phase_x, phase_y;
  float duty_x, duty_y;

  /* Profile, dimensionality and highest propagating order per axis. */
  uint packed;
};

static_assert(sizeof(ShaderClosure) >= sizeof(DiffractionGratingBsdf),
              "DiffractionGratingBsdf is too large!");

/* Keep the order arrays out of the stack frame of the BSDF dispatch on Metal, which every closure
 * of every scene goes through: inlined they made renders without any grating 4% slower. */
#ifdef __KERNEL_METAL__
#  define ccl_device_diffraction_grating ccl_device __attribute__((noinline))
#elif defined(_MSC_VER) && !defined(__clang__) && !defined(__KERNEL_GPU__)
/* Same with MSVC on the CPU, see ccl_device_inline_outline_rare. */
#  define ccl_device_diffraction_grating ccl_device ccl_never_inline
#elif defined(__KERNEL_CUDA__)
/* Same on CUDA and OptiX, where inlining also multiplies kernel compile time. */
#  define ccl_device_diffraction_grating ccl_device_noinline
#else
#  define ccl_device_diffraction_grating ccl_device
#endif

/* -------------------------------------------------------------------------------------------- */
/* Efficiencies */

/* How far above the highest order actually needed to seed Miller's algorithm. */
ccl_device_inline int diffraction_grating_miller_start(const int max_order, const float A)
{
  const int margin = max(10, (int)(1.6f * A) + 8);
  return min(max_order + margin, 64);
}

/* Efficiency of orders -max_order..max_order of a sinusoidal phase grating with phase amplitude A:
 * J_m(A)^2, computed for all orders in one pass with Miller's downward recurrence, normalized by
 * J_0 + 2 sum J_2k = 1. */
ccl_device_inline void diffraction_grating_sinusoidal_efficiencies(const float A,
                                                                   const int max_order,
                                                                   ccl_private float *eta_out)
{
  /* The recurrence is unstable for tiny arguments, where all power stays specular anyway. */
  if (A < 0.15f) {
    for (int m = -max_order; m <= max_order; m++) {
      eta_out[m + max_order] = (m == 0) ? 1.0f : 0.0f;
    }
    return;
  }

  const int n_start = diffraction_grating_miller_start(max_order, A);
  float raw[DIFFRACTION_GRATING_MAX_ORDER + 1];
  for (int i = 0; i <= max_order; i++) {
    raw[i] = 0.0f;
  }

  float J_next = 0.0f;
  float J_curr = 1.0f;
  float sum = 0.0f;
  const float two_over_A = 2.0f / A;
  for (int n = n_start; n >= 1; n--) {
    float J_prev = ((float)n * two_over_A) * J_curr - J_next;
    J_next = J_curr;
    J_curr = J_prev;
    /* The unnormalized values grow quickly: rescale everything computed so far. This changes
     * nothing after normalization. */
    if (fabsf(J_prev) > 1e20f) {
      const float scale = 1e-20f;
      J_prev *= scale;
      J_next *= scale;
      J_curr *= scale;
      sum *= scale;
      for (int i = 0; i <= max_order; i++) {
        raw[i] *= scale;
      }
    }
    if (n - 1 <= max_order) {
      raw[n - 1] = J_prev;
    }
    if (((n - 1) & 1) == 0 && (n - 1) > 0) {
      sum += 2.0f * J_prev;
    }
  }
  sum += raw[0];
  const float inv_sum = (fabsf(sum) > 1e-30f) ? 1.0f / sum : 0.0f;

  for (int m = -max_order; m <= max_order; m++) {
    eta_out[m + max_order] = sqr(raw[abs(m)] * inv_sum);
  }
}

/* Efficiency of orders -max_order..max_order of a rectangular (pit and land) phase grating with
 * round trip phase phi between pit and land, and pit fraction g. Sums to one over all orders. */
ccl_device_inline void diffraction_grating_rectangular_efficiencies(const float phi,
                                                                    const float g,
                                                                    const int max_order,
                                                                    ccl_private float *eta_out)
{
  const float re = (1.0f - g) + g * cosf(phi);
  const float im = g * sinf(phi);
  eta_out[max_order] = sqr(re) + sqr(im);

  const float amplitude = 2.0f * sinf(0.5f * phi) * g;
  for (int m = 1; m <= max_order; m++) {
    const float mg = (float)m * g;
    const float sinc_mg = (fabsf(mg) < 1e-6f) ? 1.0f : sinf(M_PI_F * mg) / (M_PI_F * mg);
    const float eta = sqr(amplitude * sinc_mg);
    eta_out[max_order + m] = eta;
    eta_out[max_order - m] = eta;
  }
}

/* Efficiencies of all orders of one axis at incidence cosine cos_NI. phase = 2 pi depth / lambda,
 * the phase difference of the reflected wave is twice that times the incidence cosine. */
ccl_device_inline void diffraction_grating_efficiencies(const bool rectangular,
                                                        const float phase,
                                                        const float duty,
                                                        const float cos_NI,
                                                        const int max_order,
                                                        ccl_private float *eta_out)
{
  if (rectangular) {
    diffraction_grating_rectangular_efficiencies(2.0f * phase * cos_NI, duty, max_order, eta_out);
  }
  else {
    diffraction_grating_sinusoidal_efficiencies(phase * cos_NI, max_order, eta_out);
  }
}

/* -------------------------------------------------------------------------------------------- */
/* Lattice */

/* Highest order that can still leave the surface for this pitch/wavelength pair. */
ccl_device_inline int diffraction_grating_max_order(const float pitch_um, const float wavelength_um)
{
  const int bound = (int)ceilf(pitch_um / wavelength_um) + 1;
  return clamp(bound, 0, DIFFRACTION_GRATING_MAX_ORDER);
}

/* Unit direction (in the tangent frame X, Y) of a lattice axis rotated by `turns` from X. */
ccl_device_inline float2 diffraction_grating_axis(const float turns)
{
  const float angle = turns * M_2PI_F;
  return make_float2(cosf(angle), sinf(angle));
}

/* Frame of the closure for decoding the second lattice momentum. */
ccl_device_inline void diffraction_grating_frame(const ccl_private DiffractionGratingBsdf *bsdf,
                                                 ccl_private float3 *K2)
{
  const float k1_len = len(bsdf->K1);
  if (k1_len > 0.0f && (bsdf->k2_along != 0.0f || bsdf->k2_across != 0.0f)) {
    const float3 along = bsdf->K1 / k1_len;
    const float3 across = cross(bsdf->N, along);
    *K2 = bsdf->k2_along * along + bsdf->k2_across * across;
  }
  else {
    *K2 = zero_float3();
  }
}

/* -------------------------------------------------------------------------------------------- */
/* Closure */

/* Inputs are as on the node: roughness 0..1 (alpha = roughness^2), pitch in micrometers, groove
 * depth in nanometers, duty cycle 0..1, rotation in turns. Y values are only used in 2D mode.
 * T is the reference tangent (X axis of the lattice), zero to use an arbitrary one. */
ccl_device int bsdf_diffraction_grating_setup(ccl_private ShaderData *sd,
                                              ccl_private DiffractionGratingBsdf *bsdf,
                                              const float3 T,
                                              const float roughness,
                                              const float2 pitch_um,
                                              const float2 groove_depth_nm,
                                              const float2 duty_cycle,
                                              const float2 rotation,
                                              const bool rectangular,
                                              const bool is_2d)
{
  bsdf->type = CLOSURE_BSDF_DIFFRACTION_GRATING_ID;
  bsdf->alpha = sqr(saturatef(roughness));

  const float pitch_x = fmaxf(pitch_um.x, 1e-4f);
  const float pitch_y = fmaxf(pitch_um.y, 1e-4f);

#ifdef __SPECTRAL__
  const float wavelength_um = sample_wavelength(sd->rand_wavelength);
#else
  const float wavelength_um = 0.55f;
#endif

  const float wavelength_nm = 1000.0f * wavelength_um;
  bsdf->phase_x = M_2PI_F * fmaxf(groove_depth_nm.x, 0.0f) / wavelength_nm;
  bsdf->phase_y = M_2PI_F * fmaxf(groove_depth_nm.y, 0.0f) / wavelength_nm;
  bsdf->duty_x = saturatef(duty_cycle.x);
  bsdf->duty_y = saturatef(duty_cycle.y);

  /* Reciprocal lattice in the tangent frame, scaled by the wavelength. */
  float2 b1, b2;
  if (is_2d) {
    const float2 a1 = pitch_x * diffraction_grating_axis(rotation.x);
    const float2 a2 = pitch_y * diffraction_grating_axis(rotation.y);
    const float det = a1.x * a2.y - a2.x * a1.y;
    /* Parallel axes have no reciprocal lattice: all orders collapse onto the specular one.
     * The threshold is the branch's 1e-6 nm^2. */
    const float inv_det = (fabsf(det) > 1e-12f) ? wavelength_um / det : 0.0f;
    b1 = make_float2(a2.y, -a2.x) * inv_det;
    b2 = make_float2(-a1.y, a1.x) * inv_det;
  }
  else {
    b1 = diffraction_grating_axis(rotation.x) * (wavelength_um / pitch_x);
    b2 = zero_float2();
  }

  float3 X, Y;
  if (is_zero(T)) {
    make_orthonormals(bsdf->N, &X, &Y);
  }
  else {
    make_orthonormals_safe_tangent(bsdf->N, T, &X, &Y);
  }
  bsdf->K1 = b1.x * X + b1.y * Y;

  /* Second momentum relative to the first one, so that only K1 needs a world space vector. */
  const float b1_len = len(b1);
  if (b1_len > 0.0f) {
    const float2 along = b1 / b1_len;
    bsdf->k2_along = dot(b2, along);
    bsdf->k2_across = along.x * b2.y - along.y * b2.x;
  }
  else {
    bsdf->k2_along = 0.0f;
    bsdf->k2_across = 0.0f;
  }

  const uint max_order_x = (uint)diffraction_grating_max_order(pitch_x, wavelength_um);
  const uint max_order_y = is_2d ? (uint)diffraction_grating_max_order(pitch_y, wavelength_um) :
                                   0u;
  bsdf->packed = (rectangular ? DIFFRACTION_GRATING_PACKED_RECTANGULAR : 0u) |
                 (is_2d ? DIFFRACTION_GRATING_PACKED_2D : 0u) |
                 (max_order_x << DIFFRACTION_GRATING_PACKED_ORDER_X_SHIFT) |
                 (max_order_y << DIFFRACTION_GRATING_PACKED_ORDER_Y_SHIFT);

  sd->runtime_flag |= SR_BSDF_HAS_DISPERSION;
  return SR_BSDF | SR_BSDF_HAS_EVAL;
}

ccl_device_inline bool diffraction_grating_is_rectangular(
    const ccl_private DiffractionGratingBsdf *bsdf)
{
  return (bsdf->packed & DIFFRACTION_GRATING_PACKED_RECTANGULAR) != 0;
}

ccl_device_inline bool diffraction_grating_is_2d(const ccl_private DiffractionGratingBsdf *bsdf)
{
  return (bsdf->packed & DIFFRACTION_GRATING_PACKED_2D) != 0;
}

ccl_device_inline int diffraction_grating_order_x(const ccl_private DiffractionGratingBsdf *bsdf)
{
  return (int)((bsdf->packed >> DIFFRACTION_GRATING_PACKED_ORDER_X_SHIFT) & 0xffu);
}

ccl_device_inline int diffraction_grating_order_y(const ccl_private DiffractionGratingBsdf *bsdf)
{
  return (int)((bsdf->packed >> DIFFRACTION_GRATING_PACKED_ORDER_Y_SHIFT) & 0xffu);
}

ccl_device void bsdf_diffraction_grating_blur(ccl_private ShaderClosure *sc, const float roughness)
{
  ccl_private DiffractionGratingBsdf *bsdf = (ccl_private DiffractionGratingBsdf *)sc;
  bsdf->alpha = fmaxf(roughness, bsdf->alpha);
}

/* Shared per-evaluation state: incident direction, its tangential part, order momenta and the
 * efficiencies of both axes at the incidence angle. */
struct DiffractionGratingOrders {
  float3 wi_t;
  float3 K2;
  int max_order_x, max_order_y;
  bool is_2d;
  float eta_x[DIFFRACTION_GRATING_MAX_ORDERS];
  float eta_y[DIFFRACTION_GRATING_MAX_ORDERS];
};

ccl_device_inline void diffraction_grating_orders_init(
    const ccl_private DiffractionGratingBsdf *bsdf,
    const float3 wi,
    const float cos_NI,
    ccl_private DiffractionGratingOrders *orders)
{
  const bool rectangular = diffraction_grating_is_rectangular(bsdf);
  orders->is_2d = diffraction_grating_is_2d(bsdf);
  orders->max_order_x = diffraction_grating_order_x(bsdf);
  orders->max_order_y = orders->is_2d ? diffraction_grating_order_y(bsdf) : 0;
  orders->wi_t = wi - cos_NI * bsdf->N;
  diffraction_grating_frame(bsdf, &orders->K2);

  diffraction_grating_efficiencies(
      rectangular, bsdf->phase_x, bsdf->duty_x, cos_NI, orders->max_order_x, orders->eta_x);
  if (orders->is_2d) {
    diffraction_grating_efficiencies(
        rectangular, bsdf->phase_y, bsdf->duty_y, cos_NI, orders->max_order_y, orders->eta_y);
  }
  else {
    orders->eta_y[0] = 1.0f;
  }
}

/* Weight of order (m, n) for picking it and for normalizing the pdf: its efficiency, or zero if
 * it is evanescent or negligible. Also returns the tangential part of the virtual incident
 * direction. */
ccl_device_inline float diffraction_grating_order_weight(
    const ccl_private DiffractionGratingBsdf *bsdf,
    const ccl_private DiffractionGratingOrders *orders,
    const int m,
    const int n,
    ccl_private float3 *wi_v_t)
{
  const float eta = orders->eta_x[m + orders->max_order_x] *
                    orders->eta_y[n + orders->max_order_y];
  if (!(eta >= DIFFRACTION_GRATING_EFFICIENCY_CUTOFF)) {
    return 0.0f;
  }
  *wi_v_t = orders->wi_t + (float)m * bsdf->K1 + (float)n * orders->K2;
  if (len_squared(*wi_v_t) > 1.0f) {
    return 0.0f;
  }
  return eta;
}

/* Weight of an order that passes the per-axis cutoff, before the propagation check. */
ccl_device_inline bool diffraction_grating_order_skipped(
    const ccl_private DiffractionGratingOrders *orders, const int m)
{
  return !(orders->eta_x[m + orders->max_order_x] >= DIFFRACTION_GRATING_EFFICIENCY_CUTOFF);
}

/* Value (f * cos(theta_o)) and pdf of all orders, for orders set up for the incident direction. */
ccl_device float diffraction_grating_eval_orders(const ccl_private DiffractionGratingBsdf *bsdf,
                                                 const ccl_private DiffractionGratingOrders *orders,
                                                 const float3 wo,
                                                 ccl_private float *pdf)
{
  const float3 N = bsdf->N;
  const float cos_NO = dot(N, wo);
  *pdf = 0.0f;
  if (cos_NO <= 0.0f) {
    return 0.0f;
  }

  const float alpha2 = sqr(fmaxf(bsdf->alpha, 1e-4f));
  /* Masking of the outgoing direction is the same for every order. */
  const float lambda_o = bsdf_lambda<MicrofacetType::GGX>(alpha2, cos_NO);
  float value = 0.0f;
  float pdf_sum = 0.0f;
  float weight_sum = 0.0f;

  for (int m = -orders->max_order_x; m <= orders->max_order_x; m++) {
    if (diffraction_grating_order_skipped(orders, m)) {
      continue;
    }
    for (int n = -orders->max_order_y; n <= orders->max_order_y; n++) {
      float3 wi_v_t;
      const float eta = diffraction_grating_order_weight(bsdf, orders, m, n, &wi_v_t);
      if (eta == 0.0f) {
        continue;
      }
      /* Every propagating order normalizes the pdf, as when picking an order in sampling,
       * also one whose lobe cannot reach `wo`. */
      weight_sum += eta;

      const float cos_NI_v = safe_sqrtf(1.0f - len_squared(wi_v_t));
      const float3 wi_v = wi_v_t + cos_NI_v * N;
      const float3 H_un = wi_v + wo;
      const float H_len = len(H_un);
      if (H_len < 1e-6f || cos_NI_v <= 0.0f) {
        continue;
      }
      const float3 H = H_un / H_len;
      const float cos_NH = dot(N, H);
      if (cos_NH <= 0.0f) {
        continue;
      }

      const float D = bsdf_D<MicrofacetType::GGX>(alpha2, cos_NH);
      const float lambda_i = bsdf_lambda<MicrofacetType::GGX>(alpha2, cos_NI_v);
      const float common = eta * D / (4.0f * cos_NI_v);
      value += common / (1.0f + lambda_i + lambda_o);
      if (dot(H, wi_v) > 0.0f) {
        pdf_sum += common / (1.0f + lambda_i);
      }
    }
  }

  *pdf = (weight_sum > 0.0f) ? pdf_sum / weight_sum : 0.0f;
  return value;
}

ccl_device_diffraction_grating Spectrum bsdf_diffraction_grating_eval(const ccl_private ShaderClosure *sc,
                                                  const float3 wi,
                                                  const float3 wo,
                                                  ccl_private float *pdf)
{
  const ccl_private DiffractionGratingBsdf *bsdf = (const ccl_private DiffractionGratingBsdf *)sc;
  const float cos_NI = dot(bsdf->N, wi);
  *pdf = 0.0f;
  if (cos_NI <= 0.0f || dot(bsdf->N, wo) <= 0.0f) {
    return zero_spectrum();
  }

  DiffractionGratingOrders orders;
  diffraction_grating_orders_init(bsdf, wi, cos_NI, &orders);
  return make_spectrum(diffraction_grating_eval_orders(bsdf, &orders, wo, pdf));
}

ccl_device_diffraction_grating int bsdf_diffraction_grating_sample(const ccl_private ShaderClosure *sc,
                                               const float3 Ng,
                                               const float3 wi,
                                               const float3 rand,
                                               ccl_private Spectrum *eval,
                                               ccl_private float3 *wo,
                                               ccl_private float *pdf,
                                               ccl_private float2 *sampled_roughness,
                                               ccl_private float *eta)
{
  const ccl_private DiffractionGratingBsdf *bsdf = (const ccl_private DiffractionGratingBsdf *)sc;
  *eta = 1.0f;
  *sampled_roughness = make_float2(bsdf->alpha, bsdf->alpha);
  *pdf = 0.0f;

  const float3 N = bsdf->N;
  const float cos_NI = dot(N, wi);
  if (cos_NI <= 0.0f) {
    return LABEL_NONE;
  }

  DiffractionGratingOrders orders;
  diffraction_grating_orders_init(bsdf, wi, cos_NI, &orders);

  /* Pick an order (m, n) proportional to its efficiency at this incidence angle: sum the
   * weights, then walk the orders again to find the picked one. */
  float weight_sum = 0.0f;
  for (int m = -orders.max_order_x; m <= orders.max_order_x; m++) {
    if (diffraction_grating_order_skipped(&orders, m)) {
      continue;
    }
    for (int n = -orders.max_order_y; n <= orders.max_order_y; n++) {
      float3 wi_v_t;
      weight_sum += diffraction_grating_order_weight(bsdf, &orders, m, n, &wi_v_t);
    }
  }
  if (!(weight_sum > 0.0f)) {
    return LABEL_NONE;
  }

  float u = rand.z * weight_sum;
  float3 picked_wi_v_t = zero_float3();
  bool picked = false;
  for (int m = -orders.max_order_x; m <= orders.max_order_x && !picked; m++) {
    if (diffraction_grating_order_skipped(&orders, m)) {
      continue;
    }
    for (int n = -orders.max_order_y; n <= orders.max_order_y; n++) {
      float3 wi_v_t;
      const float w = diffraction_grating_order_weight(bsdf, &orders, m, n, &wi_v_t);
      if (w == 0.0f) {
        continue;
      }
      /* The last valid order takes what floating point round off leaves of u. */
      picked_wi_v_t = wi_v_t;
      if (u < w) {
        picked = true;
        break;
      }
      u -= w;
    }
  }

  /* Sample a microfacet normal around the virtual incident direction and reflect it. */
  const float cos_NI_v = safe_sqrtf(1.0f - len_squared(picked_wi_v_t));
  const float3 wi_v = picked_wi_v_t + cos_NI_v * N;
  float3 X, Y;
  make_orthonormals(N, &X, &Y);
  const float3 local_I = make_float3(dot(wi_v, X), dot(wi_v, Y), cos_NI_v);
  const float alpha = fmaxf(bsdf->alpha, 1e-4f);
  const float3 local_H = microfacet_ggx_sample_vndf(local_I, alpha, alpha, make_float2(rand));
  const float3 H = to_global(local_H, X, Y, N);

  const float cos_HI = dot(H, wi_v);
  if (cos_HI <= 0.0f) {
    return LABEL_NONE;
  }
  *wo = 2.0f * cos_HI * H - wi_v;

  if (dot(N, *wo) <= 0.0f || dot(Ng, *wo) <= 0.0f) {
    return LABEL_NONE;
  }

  /* All orders, reusing their efficiencies. */
  *eval = make_spectrum(diffraction_grating_eval_orders(bsdf, &orders, *wo, pdf));
  if (!(*pdf > 0.0f)) {
    return LABEL_NONE;
  }

  return LABEL_REFLECT | LABEL_GLOSSY;
}

CCL_NAMESPACE_END
