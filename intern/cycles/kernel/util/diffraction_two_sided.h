/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

CCL_NAMESPACE_BEGIN

/* An explicitly approximate, neutral and lossless two-hemisphere return lobe.
 * The single-event directional escape deficit q_s is in [0,1]. The integrated
 * deficit Q_s uses the etendue measure n_s^2 |cos(theta)| dOmega, so the
 * reciprocal kernel K_st = f_st/n_t^2 is symmetric. The coefficient matrix
 * enforces sum_t C_st Q_t = 1 on each side. */
struct DiffractionTwoSidedReturn {
  float q_integral[2];
  float c[2][2];
};

ccl_device_inline bool diffraction_two_sided_return(const float q0,
                                                     const float q1,
                                                     const float cross_fraction,
                                                     ccl_private DiffractionTwoSidedReturn *r)
{
  if (!(q0 >= 0.0f && q1 >= 0.0f && q0 + q1 > 0.0f) ||
      !isfinite_safe(q0 + q1 + cross_fraction) ||
      cross_fraction < 0.0f || cross_fraction > 1.0f)
  {
    return false;
  }
  /* The bounded cross conductance is shared by both directions. This gives
   * nonnegative entries and preserves both row sums exactly. A physically
   * tabulated return operator supplies cross_fraction, never RGB tint. */
  const float k = (q0 > 0.0f && q1 > 0.0f) ? cross_fraction / max(q0, q1) : 0.0f;
  r->q_integral[0] = q0;
  r->q_integral[1] = q1;
  r->c[0][1] = k;
  r->c[1][0] = k;
  r->c[0][0] = q0 > 0.0f ? max(0.0f, (1.0f - k * q1) / q0) : 0.0f;
  r->c[1][1] = q1 > 0.0f ? max(0.0f, (1.0f - k * q0) / q1) : 0.0f;
  return isfinite_safe(r->c[0][0] + r->c[0][1] + r->c[1][1]);
}

/* Cycles eval convention is f * |cos_o|, with pdf per outgoing solid angle.
 * n_out^2 follows from the symmetric etendue kernel. */
ccl_device_inline float diffraction_two_sided_eval(const ccl_private DiffractionTwoSidedReturn &r,
                                                     const int incoming_side,
                                                     const int outgoing_side,
                                                     const float q_in,
                                                     const float q_out,
                                                     const float n_out,
                                                     const float cos_out)
{
  return sqr(n_out) * q_in * r.c[incoming_side][outgoing_side] * q_out * cos_out;
}

ccl_device_inline float diffraction_two_sided_side_probability(
    const ccl_private DiffractionTwoSidedReturn &r, const int incoming_side, const int outgoing_side)
{
  return r.c[incoming_side][outgoing_side] * r.q_integral[outgoing_side];
}

/* Cosine-hemisphere proposal, conditional on the selected outgoing side.
 * It is deliberately simple; eval/pdf remains unbiased and the proposal can
 * be improved without changing the reciprocal lobe. */
ccl_device_inline float diffraction_two_sided_pdf(const ccl_private DiffractionTwoSidedReturn &r,
                                                    const int incoming_side,
                                                    const int outgoing_side,
                                                    const float cos_out)
{
  return diffraction_two_sided_side_probability(r, incoming_side, outgoing_side) *
         cos_out * M_1_PI_F;
}

CCL_NAMESPACE_END
