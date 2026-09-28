/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/types.h"

CCL_NAMESPACE_BEGIN

/* Per-light-vertex optional sidecar. The ordinary KernelBDPTVertex stays
 * unchanged when coherent specular connections are disabled. Emitter identity
 * is already carried by KernelBDPTVertex::emitter_object. Patch IDs are six
 * bits because the scene candidate budget limits participating patches to 64.
 * Events are two bits each; incident sides use another two bits each so the
 * same patch/event from opposite media cannot be confused. */
struct CoherentPathHistory {
  uint patches;
  uint metadata;
};
static_assert(sizeof(CoherentPathHistory) == 8, "Coherent history sidecar must stay compact");

#define COHERENT_HISTORY_VALID (1u << 11u)
#define COHERENT_HISTORY_COUNT_SHIFT 8u
#define COHERENT_HISTORY_SIDE_SHIFT 12u
#define COHERENT_HISTORY_MAX_EVENTS 4u

ccl_device_inline CoherentPathHistory coherent_history_begin()
{
  return {0u, COHERENT_HISTORY_VALID};
}

ccl_device_inline CoherentPathHistory coherent_history_invalidate(const CoherentPathHistory history)
{
  return {history.patches, history.metadata & ~COHERENT_HISTORY_VALID};
}

ccl_device_inline bool coherent_history_valid(const CoherentPathHistory history)
{
  return (history.metadata & COHERENT_HISTORY_VALID) != 0u;
}

ccl_device_inline uint coherent_history_count(const CoherentPathHistory history)
{
  return (history.metadata >> COHERENT_HISTORY_COUNT_SHIFT) & 7u;
}

ccl_device_inline CoherentPathHistory coherent_history_append(const CoherentPathHistory history,
                                                              const uint patch,
                                                              const uint event,
                                                              const int incident_side = 0)
{
  const uint count = coherent_history_count(history);
  if (!coherent_history_valid(history) || count >= COHERENT_HISTORY_MAX_EVENTS ||
      patch >= 64u || event >= 4u || incident_side < -1 || incident_side > 1)
  {
    return coherent_history_invalidate(history);
  }
  CoherentPathHistory result = history;
  result.patches |= patch << (6u * count);
  result.metadata |= event << (2u * count);
  result.metadata |= uint(incident_side + 1) << (COHERENT_HISTORY_SIDE_SHIFT + 2u * count);
  result.metadata = (result.metadata & ~(7u << COHERENT_HISTORY_COUNT_SHIFT)) |
                    ((count + 1u) << COHERENT_HISTORY_COUNT_SHIFT);
  return result;
}

/* Called only after the actual BSDF sample. An unmarked or rough/diffuse event
 * permanently leaves the deterministic source class. The caller maps the hit
 * object to a unique coherent patch index; -1 means no participating patch. */
ccl_device_inline CoherentPathHistory coherent_history_after_scatter(
    const CoherentPathHistory history,
    const int patch_index,
    const int patch_mode,
    const int label,
    const int incident_side)
{
  if (patch_index < 0 || !(label & LABEL_SINGULAR)) {
    return coherent_history_invalidate(history);
  }
  const bool reflect = (label & LABEL_REFLECT) != 0;
  const bool transmit = (label & LABEL_TRANSMIT) != 0;
  if (reflect == transmit || (patch_mode == 1 && !reflect) ||
      (patch_mode != 1 && patch_mode != 2))
  {
    return coherent_history_invalidate(history);
  }
  return coherent_history_append(history, uint(patch_index), transmit ? 1u : 0u,
                                 patch_mode == 1 ? 0 : incident_side);
}

/* The streaming inventory includes every triangle of each declared mirror.
 * Eligibility is proved from the actual typed hit at scatter time; no capped
 * facet signature is needed. Other histories remain native. */
ccl_device_inline CoherentPathHistory coherent_history_stream_after_scatter(
    const CoherentPathHistory history, const bool eligible_mirror, const int label,
    const uint max_interfaces = 1u)
{
  if (max_interfaces < 1u || max_interfaces > 2u ||
      coherent_history_count(history) >= max_interfaces)
    return coherent_history_invalidate(history);
  return coherent_history_after_scatter(history, eligible_mirror ? 0 : -1, 1, label, 0);
}

ccl_device_inline bool coherent_history_matches(
    const CoherentPathHistory history,
    const ccl_global KernelCoherentCandidate *candidate)
{
  const uint count = coherent_history_count(history);
  if (!coherent_history_valid(history) || count != uint(candidate->count)) {
    return false;
  }
  for (uint index = 0u; index < count; index++) {
    if (((history.patches >> (6u * index)) & 63u) != uint(candidate->patch[index]) ||
        ((history.metadata >> (2u * index)) & 3u) != uint(candidate->event[index]) ||
        (candidate->expected_incident_side[index] != 0 &&
         ((history.metadata >> (COHERENT_HISTORY_SIDE_SHIFT + 2u * index)) & 3u) !=
             uint(candidate->expected_incident_side[index] + 1)))
    {
      return false;
    }
  }
  return true;
}

ccl_device_inline bool coherent_history_matches_source(
    const CoherentPathHistory history,
    const ccl_global KernelCoherentCandidate *candidate,
    const int emitter_object,
    const int candidate_emitter_object)
{
  return emitter_object == candidate_emitter_object &&
         coherent_history_matches(history, candidate);
}

ccl_device_inline bool coherent_history_owns_detector_prefix(
    const bool marked_detector,
    const CoherentPathHistory history,
    const ccl_global KernelCoherentCandidate *candidate,
    const int emitter_object,
    const int candidate_emitter_object)
{
  return marked_detector && coherent_history_matches_source(
                                history, candidate, emitter_object, candidate_emitter_object);
}

ccl_device_inline bool coherent_history_owns_direct_source(
    const ccl_global KernelCoherentCandidate *candidate,
    const int light_index,
    const bool positive_coherence_length)
{
  return positive_coherence_length && candidate->count == 0 && candidate->light == light_index;
}

/* Test a candidate against the eye suffix's remaining bounce budget. The
 * detector's local direct contribution is allowed at the current vertex;
 * every preceding ideal interface consumes one continuation on the reverse
 * path before that light can be reached. A continuation that reaches its
 * configured maximum terminates before the next surface. */
ccl_device_inline bool coherent_history_candidate_within_budget(
    const ccl_global KernelCoherentCandidate *candidate,
    const int camera_bounce,
    const int camera_glossy_bounce,
    const int camera_transmission_bounce,
    const int max_bounce,
    const int max_glossy_bounce,
    const int max_transmission_bounce,
    const bool bdpt_enabled,
    const int bdpt_max_bounces,
    const float source_max_bounces,
    const int interfaces_override = -1)
{
  if (candidate->count < 0 || candidate->count > int(COHERENT_HISTORY_MAX_EVENTS) ||
      camera_bounce < 0 || camera_glossy_bounce < 0 || camera_transmission_bounce < 0)
  {
    return false;
  }
  int reflections = 0;
  int transmissions = 0;
  for (int index = 0; index < candidate->count; index++) {
    if (candidate->event[index] == 0) {
      reflections++;
    }
    else if (candidate->event[index] == 1) {
      transmissions++;
    }
    else {
      return false;
    }
  }
  const int interfaces = interfaces_override >= 0 ? interfaces_override : candidate->count;
  if (interfaces_override >= 0) {
    if (interfaces_override > 2) return false;
    reflections = interfaces_override;
    transmissions = 0;
  }
  if (camera_bounce + interfaces > source_max_bounces ||
      (bdpt_enabled && interfaces >= bdpt_max_bounces))
  {
    return false;
  }
  if (interfaces > 0 && camera_bounce + interfaces >= max_bounce) {
    return false;
  }
  if (reflections > 0 && camera_glossy_bounce + reflections >= max_glossy_bounce) {
    return false;
  }
  if (transmissions > 0 &&
      camera_transmission_bounce + transmissions >= max_transmission_bounce)
  {
    return false;
  }
  return true;
}

CCL_NAMESPACE_END
