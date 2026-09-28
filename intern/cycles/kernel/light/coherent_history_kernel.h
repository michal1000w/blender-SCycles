/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/light/coherent_history.h"
#include "kernel/light/coherent_patch_membership.h"
#include "kernel/globals.h"

CCL_NAMESPACE_BEGIN

ccl_device_inline bool coherent_history_owned_candidate(KernelGlobals kg,
                                                        const CoherentPathHistory history,
                                                        const int emitter_object)
{
  if (!coherent_history_valid(history)) {
    return false;
  }
  const bool stream_facets = kernel_data.integrator.coherent_transport_mode == 1;
  if (stream_facets && coherent_history_count(history) >
      uint(kernel_data.integrator.coherent_max_interface_events)) return false;
  for (int index = 0; index < kernel_data.integrator.coherent_candidate_count; index++) {
    const ccl_global KernelCoherentCandidate *candidate =
        &kernel_data_fetch(coherent_candidates, index);
    const int candidate_emitter_object = kernel_data_fetch(lights, candidate->light).object_id;
    if ((stream_facets && candidate_emitter_object == emitter_object) ||
        coherent_history_matches_source(
            history, candidate, emitter_object, candidate_emitter_object))
    {
      return true;
    }
  }
  return false;
}

/* Direct NEE is replaced only for a source that has an actual zero-interface
 * candidate. Other point lights, including zero-coherence-length members of
 * a group, retain the ordinary direct-light estimator. */
ccl_device_inline bool coherent_history_owned_direct_source(KernelGlobals kg,
                                                            const int light_index)
{
  const ccl_global KernelLight *light = &kernel_data_fetch(lights, light_index);
  if (light->coherence_group <= 0 || light->coherence_length <= 0.0f) {
    return false;
  }
  for (int index = 0; index < kernel_data.integrator.coherent_candidate_count; index++) {
    const ccl_global KernelCoherentCandidate *candidate =
        &kernel_data_fetch(coherent_candidates, index);
    if (coherent_history_owns_direct_source(candidate, light_index, true)) {
      return true;
    }
  }
  return false;
}

CCL_NAMESPACE_END
