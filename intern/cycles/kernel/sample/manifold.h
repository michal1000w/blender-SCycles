/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/types.h"

CCL_NAMESPACE_BEGIN

/* Address each rough interface's slope sample without mutating the path RNG.
 * Stable indexing also preserves replay after a texture-cache miss. */
ccl_device_inline uint manifold_vertex_rng_dimension(const uint interface_index)
{
  return PRNG_SURFACE_BSDF + interface_index * PRNG_BOUNCE_NUM;
}

CCL_NAMESPACE_END
