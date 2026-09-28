/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/types.h"
CCL_NAMESPACE_BEGIN

/* Only declared static Glass points in the coherent connector model gain exit hits.
 * Native points, motion points and disabled scenes retain their intersection policy. */
ccl_device_inline bool point_coherent_glass_two_sided(const bool enabled,
                                                     const uint object_flags,
                                                     const int type)
{
  return enabled && (object_flags & SD_OBJECT_COHERENT_GLASS_POINT) &&
         ((type & (PRIMITIVE_ALL | PRIMITIVE_MOTION)) == PRIMITIVE_POINT) &&
         !(object_flags & (SD_OBJECT_MOTION | SD_OBJECT_HAS_VERTEX_MOTION));
}

/* Retain the sampled-light endpoint exclusion even when a previous sphere may
 * be intersected again at its far interface. Ray offsets still exclude the origin. */
ccl_device_inline bool point_coherent_skip_self(const bool two_sided,
                                               const bool previous_self,
                                               const bool light_self)
{
  return light_self || (previous_self && !two_sided);
}
CCL_NAMESPACE_END
