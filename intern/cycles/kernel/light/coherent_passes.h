/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "util/math.h"
#include "util/types_spectrum.h"

CCL_NAMESPACE_BEGIN

/* Coherent pairs have no unique classical-path decomposition. This explicit
 * convention assigns half of each signed pair to each contributing route.
 * At a primary Lambertian detector a route with no interface is direct;
 * every other route is indirect. Never clamp the resulting signed passes. */
ccl_device_inline Spectrum coherent_pass_direct_pair_share(const Spectrum pair,
                                                           const bool direct_a,
                                                           const bool direct_b)
{
  return pair * (.5f * (int(direct_a) + int(direct_b)));
}

struct CoherentSurfaceLightPasses {
  Spectrum diffuse_direct, diffuse_indirect;
  Spectrum glossy_direct, glossy_indirect;
  Spectrum transmission_direct, transmission_indirect;
};

/* A camera suffix that already scattered makes every new detector contribution
 * indirect. Native first-scatter pass weights select its camera-side lobe.
 * Values are stored radiance, before the native accessor divides by Color. */
ccl_device_inline CoherentSurfaceLightPasses coherent_pass_surface_split(
    const Spectrum total,
    const Spectrum primary_direct,
    const int camera_bounce,
    const Spectrum first_diffuse_weight,
    const Spectrum first_glossy_weight)
{
  CoherentSurfaceLightPasses out{};
  if (camera_bounce == 0) {
    out.diffuse_direct = primary_direct;
    out.diffuse_indirect = total - primary_direct;
  }
  else {
    out.diffuse_indirect = first_diffuse_weight * total;
    out.glossy_indirect = first_glossy_weight * total;
    out.transmission_indirect = (one_spectrum() - first_diffuse_weight - first_glossy_weight) * total;
  }
  return out;
}

CCL_NAMESPACE_END
