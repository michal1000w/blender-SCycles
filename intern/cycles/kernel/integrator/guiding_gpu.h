/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/globals.h"
#include "kernel/integrator/state.h"
#include "kernel/sample/guiding_field.h"
#include "kernel/sample/guiding_history.h"
#include "kernel/sample/guiding_mixture_conditional.h"
#include "kernel/sample/guiding_mixture_fit_metal.h"
#include "kernel/sample/guiding_observation_range.h"
#include "kernel/util/colorspace.h"
#include "util/hash.h"
#include "util/types_normal.h"

CCL_NAMESPACE_BEGIN

#ifdef __KERNEL_METAL__

ccl_device_inline float3 guiding_gpu_surface_orientation(const ccl_private ShaderData *sd)
{
  /* Condition on the primitive's geometric orientation, not the incoming path direction.
   * Camera and light paths can approach a transmissive primitive from opposite sides; their
   * radiance/adjoint records must still address the same surface sector. Shader setup flips Ng
   * for backfacing hits, and keeps that parity in SR_BACKFACING. */
  return (sd->runtime_flag & SR_BACKFACING) ? -sd->Ng : sd->Ng;
}

ccl_device_inline GuidingField guiding_gpu_field()
{
  return {kernel_integrator_state.guiding_nodes,
          kernel_integrator_state.guiding_accumulation,
          kernel_integrator_state.guiding_sampling,
          kernel_integrator_state.guiding_counts,
          kernel_integrator_state.guiding_capacity,
          make_float3(kernel_data.integrator.guiding_bounds_min),
          make_float3(kernel_data.integrator.guiding_bounds_max)};
}

ccl_device void guiding_gpu_begin_update()
{
  guiding_gpu_field().begin_update();
}

ccl_device void guiding_gpu_refine(const uint index)
{
  if (metal::simd_sum(1u) != 32u) {
    const uint fields = kernel_integrator_state.guiding_capacity * GUIDING_FIELD_TYPES;
    atomic_fetch_and_or_uint32(kernel_integrator_state.guiding_partition + 3 * fields, 8u);
    return;
  }
  const GuidingField field = guiding_gpu_field();
  const uint node = index / 32;
  const uint lane = index % 32;
  uint first = lane == 0 ? field.refine_allocate(node, 1024) : 0;
  first = metal::simd_broadcast(first, 0);
  if (first != 0) {
    field.refine_copy(node, first, lane, 32);
  }
}

ccl_device void guiding_gpu_publish(const uint index)
{
  const GuidingField field = guiding_gpu_field();
  if (index >= field.counts[0] * GUIDING_FIELD_TYPES) {
    return;
  }
  using Stats = GuidingMixtureStatistics;
  ccl_global float *working = kernel_integrator_state.guiding_fit + index * Stats::working_size;
  ccl_global float *model = field.sampling + index * field.sampling_size + field.tree_size;
  float3 references[GuidingGaussianMixture::components];
  float total = 0;
  for (int c = 0; c < GuidingGaussianMixture::components; ++c) {
    const int base = c * GuidingGaussianMixture::component_stride;
    references[c] = make_float3(model[base + 2], model[base + 3], model[base + 4]);
    total += working[c * Stats::working_component_size];
  }
  /* Keep the histogram as initial support and for spatial adaptation. A trained
   * interval replaces its angular-region mixture with soft-assignment fits. */
  field.publish(index, 0.05f);
  if (!(total > 0.0f)) {
    return;
  }
  GuidingConditionalMixture conditional;
  for (int c = 0; c < GuidingGaussianMixture::components; ++c) {
    ccl_global float *component = working + c * Stats::working_component_size;
    Stats statistics;
    statistics.direction_reference = references[c];
    for (int i = 0; i < Stats::storage_size; ++i) {
      statistics.values[i] = component[i];
    }
    conditional.publish_component(model, c, component, total,
                                  kernel_integrator_state.guiding_fit_counts[index],
                                  statistics.directional_fit(), field.metric_extent());
  }
}

ccl_device_inline GuidingObservationPartition guiding_gpu_partition()
{
  const uint fields = kernel_integrator_state.guiding_capacity * GUIDING_FIELD_TYPES;
  ccl_global uint *storage = kernel_integrator_state.guiding_partition;
  return {kernel_integrator_state.guiding_history, storage, storage + fields,
          storage + 2 * fields, kernel_integrator_state.guiding_indices,
          storage + 3 * fields, fields, GuidingField::accumulation_size,
          kernel_integrator_state.guiding_history_capacity};
}

ccl_device void guiding_gpu_partition_count(const uint index)
{
  guiding_gpu_partition().count(index);
}

ccl_device void guiding_gpu_partition_prefix(const uint index)
{
  if (index == 0) {
    const auto partition = guiding_gpu_partition();
    partition.prefix();
    const GuidingObservationTasks tasks{kernel_integrator_state.guiding_fit_tasks,
                                        partition.distributions,
                                        kernel_integrator_state.guiding_fit_task_capacity};
    if (!tasks.build(partition.counts)) {
      atomic_fetch_and_or_uint32(partition.error, 128u);
    }
  }
}

ccl_device void guiding_gpu_partition_scatter(const uint index)
{
  guiding_gpu_partition().scatter(index);
}

ccl_device void guiding_gpu_fit(const uint index)
{
  const auto partition = guiding_gpu_partition();
  /* Apple GPU SIMD groups contain 32 lanes. Reject an incompatible execution
   * width explicitly rather than mixing independent fields in a reduction. */
  if (metal::simd_sum(1u) != 32u) {
    atomic_fetch_and_or_uint32(partition.error, 8u);
    return;
  }
  const uint group = index / 32;
  const uint lane = index % 32;
  const bool partial = group >= partition.distributions;
  const uint task = group - partition.distributions;
  const ccl_global uint *tasks = kernel_integrator_state.guiding_fit_tasks;
  if (partial &&
      task >=
          tasks[partition.distributions + 2 * kernel_integrator_state.guiding_fit_task_capacity])
  {
    return;
  }
  const uint distribution = partial ? tasks[partition.distributions + 2 * task] : group;
  const uint start = partial ? tasks[partition.distributions + 2 * task + 1] : 0;
  const uint full_count = partition.counts[distribution];
  if (!partial && full_count > GuidingObservationTasks::chunk_size) {
    return;
  }
  const uint count = min(full_count - start, GuidingObservationTasks::chunk_size);
  if (count == 0) {
    return;
  }
  if (partition.offsets[distribution] > partition.capacity ||
      full_count > partition.capacity - partition.offsets[distribution])
  {
    atomic_fetch_and_or_uint32(partition.error, 64u);
    return;
  }
  const GuidingField field = guiding_gpu_field();
  const GuidingHistoryObservationRange range{partition.records,
                                             partition.indices,
                                             partition.offsets[distribution] + start,
                                             field.scene_scale()};
  float maximum_weight = 0;
  for (uint i = lane; i < count; i += 32) {
    maximum_weight = max(maximum_weight, range[i].w);
  }
  maximum_weight = metal::simd_max(maximum_weight);
  const ccl_global float *model = field.sampling + distribution * field.sampling_size +
                                  field.tree_size;
  const float model_mass = metal::simd_sum(lane < GuidingGaussianMixture::components ?
      model[lane * GuidingGaussianMixture::component_stride] : 0.0f);
  if (!(model_mass > 0.0f)) {
    if (partial && lane < GuidingGaussianMixture::components) {
      ccl_global float *storage = kernel_integrator_state.guiding_fit_partials +
                                  task * GuidingMixtureStatistics::working_size +
                                  lane * GuidingMixtureStatistics::working_component_size;
      for (int i = 0; i < GuidingMixtureStatistics::working_component_size; ++i) {
        storage[i] = 0;
      }
    }
    return;
  }
  using Stats = GuidingMixtureStatistics;
  Stats batch;
  guiding_mixture_collect_cooperative(
      model, range, count, lane, maximum_weight, field.metric_extent(), batch);
  if (lane < GuidingGaussianMixture::components) {
    if (partial) {
      ccl_global float *storage = kernel_integrator_state.guiding_fit_partials +
                                  task * Stats::working_size +
                                  lane * Stats::working_component_size;
      for (int i = 0; i < Stats::storage_size; ++i) {
        storage[i] = batch.values[i];
        storage[Stats::storage_size + i] = batch.errors[i];
      }
      storage[2 * Stats::storage_size] = maximum_weight;
      return;
    }
    ccl_global float *storage = kernel_integrator_state.guiding_fit +
        distribution * Stats::working_size + lane * Stats::working_component_size;
    Stats accumulated;
    accumulated.direction_reference = batch.direction_reference;
    for (int i = 0; i < Stats::storage_size; ++i) {
      accumulated.values[i] = storage[i];
      accumulated.errors[i] = storage[Stats::storage_size + i];
    }
    float scale = storage[2 * Stats::storage_size];
    if (!accumulated.merge(batch, maximum_weight, scale)) {
      atomic_fetch_and_or_uint32(partition.error, 16u);
      return;
    }
    for (int i = 0; i < Stats::storage_size; ++i) {
      storage[i] = accumulated.values[i];
      storage[Stats::storage_size + i] = accumulated.errors[i];
    }
    storage[2 * Stats::storage_size] = scale;
  }
  if (!partial && lane == 0) {
    ccl_global uint *observations = kernel_integrator_state.guiding_fit_counts + distribution;
    /* Publication only uses this count for its 64-observation eligibility gate.
     * Effective support is computed from the full weighted moments. */
    *observations = min(64u, min(64u, *observations) + min(64u, count));
  }
}

ccl_device void guiding_gpu_fit_reduce(const uint index)
{
  const uint distribution = index / 32;
  const uint lane = index % 32;
  const auto partition = guiding_gpu_partition();
  const uint count = partition.counts[distribution];
  if (count <= GuidingObservationTasks::chunk_size || lane >= GuidingGaussianMixture::components) {
    return;
  }
  using Stats = GuidingMixtureStatistics;
  const GuidingField field = guiding_gpu_field();
  const ccl_global float *model = field.sampling + distribution * field.sampling_size +
                                  field.tree_size;
  const uint base = lane * GuidingGaussianMixture::component_stride;
  Stats accumulated;
  accumulated.direction_reference = make_float3(model[base + 2], model[base + 3], model[base + 4]);
  ccl_global float *storage = kernel_integrator_state.guiding_fit +
                              distribution * Stats::working_size +
                              lane * Stats::working_component_size;
  for (int i = 0; i < Stats::storage_size; ++i) {
    accumulated.values[i] = storage[i];
    accumulated.errors[i] = storage[Stats::storage_size + i];
  }
  float scale = storage[2 * Stats::storage_size];
  const uint first = kernel_integrator_state.guiding_fit_tasks[distribution];
  const uint chunks = 1 + (count - 1) / GuidingObservationTasks::chunk_size;
  const uint total = kernel_integrator_state
                         .guiding_fit_tasks[partition.distributions +
                                            2 * kernel_integrator_state.guiding_fit_task_capacity];
  if (first > total || chunks > total - first) {
    atomic_fetch_and_or_uint32(partition.error, 128u);
    return;
  }
  bool have_batch = false;
  for (uint chunk = first; chunk < first + chunks; ++chunk) {
    const ccl_global float *input = kernel_integrator_state.guiding_fit_partials +
                                    chunk * Stats::working_size +
                                    lane * Stats::working_component_size;
    const float batch_scale = input[2 * Stats::storage_size];
    if (!(batch_scale > 0)) {
      continue;
    }
    have_batch = true;
    Stats batch;
    batch.direction_reference = accumulated.direction_reference;
    for (int i = 0; i < Stats::storage_size; ++i) {
      batch.values[i] = input[i];
      batch.errors[i] = input[Stats::storage_size + i];
    }
    if (!accumulated.merge(batch, batch_scale, scale)) {
      atomic_fetch_and_or_uint32(partition.error, 16u);
      return;
    }
  }
  for (int i = 0; i < Stats::storage_size; ++i) {
    storage[i] = accumulated.values[i];
    storage[Stats::storage_size + i] = accumulated.errors[i];
  }
  storage[2 * Stats::storage_size] = scale;
  if (lane == 0 && have_batch) {
    kernel_integrator_state.guiding_fit_counts[distribution] = 64;
  }
}

ccl_device_inline bool guiding_gpu_training()
{
  return kernel_data.integrator.use_guiding && kernel_integrator_state.guiding_training;
}

ccl_device_inline void guiding_gpu_record_bounce(IntegratorState state,
                                                 const float3 P,
                                                 const float3 direction,
                                                 const float pdf,
                                                 const int label,
                                                 const bool volume,
                                                 const float3 normal = zero_float3(),
                                                 const float distance_scale = 0.0f)
{
  if (!guiding_gpu_training() || (label & LABEL_TRANSPARENT) || !(pdf > 0.0f)) {
    return;
  }
  const bool delta = label & LABEL_SINGULAR;
  if (delta && INTEGRATOR_STATE(state, gpu_guiding, history_head) == ~0u) {
    return;
  }
  const GuidingField field = guiding_gpu_field();
  const uint node = field.find_leaf(P);
  const GuidingFieldType type = volume ? GUIDING_FIELD_VOLUME_RADIANCE :
                                         guiding_surface_field_type(normal, false);
  const GuidingHistory history{kernel_integrator_state.guiding_history,
                               kernel_integrator_state.guiding_history_count,
                               kernel_integrator_state.guiding_history_capacity};
  /* Remove post-scatter throughput and the actual contribution PDF, as in CPU path segments. */
  const uint index = history.append(
      INTEGRATOR_STATE(state, gpu_guiding, history_head),
      delta ? ~0u : field.record_index(node, type, direction),
      packed_normal(direction).value,
      delta ? zero_spectrum() :
              safe_divide(one_spectrum(), INTEGRATOR_STATE(state, path, throughput) * pdf),
      field.normalized_position(P),
      distance_scale);
  if (index != ~0u) {
    INTEGRATOR_STATE_WRITE(state, gpu_guiding, history_head) = index;
  }
  if (!delta) {
    field.record_visit(node);
  }
}

ccl_device_inline void guiding_gpu_record_importance(ConstIntegratorState state,
                                                     const float3 P,
                                                     const float3 direction,
                                                     const float projected_area,
                                                     const bool volume,
                                                     const float3 normal = zero_float3())
{
  if (!guiding_gpu_training() || !kernel_data.integrator.use_bidirectional_path_tracing ||
      !(projected_area > 0.0f))
  {
    return;
  }
  const GuidingField field = guiding_gpu_field();
  const uint node = field.find_leaf(P);
  const GuidingFieldType type = volume ? GUIDING_FIELD_VOLUME_IMPORTANCE :
                                         guiding_surface_field_type(normal, true);
  /* Camera arrivals estimate adjoint incident flux. Remove the projected-area measure on
   * surfaces; volume arrivals are already measured per unit volume. This only trains a proposal.
   */
  const float importance = average(spectrum_to_rgb(INTEGRATOR_STATE(state, path, throughput))) /
                           projected_area;
  float source_distance = 0.0f, source_scale = 1.0f;
  float3 previous_position = P;
  uint ancestor = INTEGRATOR_STATE(state, gpu_guiding, history_head);
  bool finite_source = false;
  const float3 extent = max(field.bounds_max - field.bounds_min, make_float3(1e-8f));
  while (ancestor != ~0u) {
    const ccl_global auto &record = kernel_integrator_state.guiding_history[ancestor];
    const float3 position = field.bounds_min + make_float3(record.position) * extent;
    source_distance += source_scale * len(position - previous_position);
    if (!isfinite_safe(source_distance)) {
      break;
    }
    if (record.distance_scale == 0.0f) {
      finite_source = true;
      break;
    }
    if (!(record.distance_scale > 0.0f) || record.distance_scale == FLT_MAX) {
      break;
    }
    /* Adjoint traversal uses the reciprocal refraction transfer. */
    source_scale /= record.distance_scale;
    previous_position = position;
    ancestor = record.parent;
  }
  const GuidingHistory history{kernel_integrator_state.guiding_history,
                               kernel_integrator_state.guiding_history_count,
                               kernel_integrator_state.guiding_history_capacity};
  history.append_observation(field.record_index(node, type, direction),
                              packed_normal(direction).value,
                              field.normalized_position(P),
                              importance,
                              finite_source ? source_distance : FLT_MAX);
  /* Adjoint arrivals greatly outnumber complete radiance observations. Driving subdivision
   * with those arrivals would create spatial cells whose radiance estimates are undersampled.
   * Both fields share refinement driven by the camera-vertex observations. */
}

ccl_device_inline void guiding_gpu_record_history(uint index,
                                                  const Spectrum contribution,
                                                  float3 endpoint = make_float3(FLT_MAX))
{
  const GuidingField field = guiding_gpu_field();
  const float3 extent = max(field.bounds_max - field.bounds_min, make_float3(1e-8f));
  GuidingVirtualDistance transport;
  float downstream_distance = 0.0f, downstream_scale = 0.0f;
  const GuidingHistory history{kernel_integrator_state.guiding_history,
                               kernel_integrator_state.guiding_history_count,
                               kernel_integrator_state.guiding_history_capacity};
  while (index != ~0u) {
    const ccl_global GuidingHistoryRecord &record = history.records[index];
    const float3 position = field.bounds_min + make_float3(record.position) * extent;
    const float segment = len(position - endpoint);
    const float distance = transport.propagate(segment, downstream_distance, downstream_scale);
    if (record.field_index != ~0u) {
      const float weight = average(
          spectrum_to_rgb(contribution * Spectrum(record.inverse_weight)));
      history.accumulate(index, weight);
      history.accumulate_source(index, weight, distance);
    }
    endpoint = position;
    downstream_distance = distance;
    downstream_scale = record.distance_scale;
    index = record.parent;
  }
}

/* One completed path observation per vertex, rather than a count for every film fragment.
 * This runs after all main/shadow queues drain and before any record slot is reused. */
ccl_device void guiding_gpu_flush_history(const uint index)
{
  const ccl_global GuidingHistoryRecord &record = kernel_integrator_state.guiding_history[index];
  if (record.field_index == ~0u) {
    return;
  }
  packed_normal direction;
  direction.value = record.direction;
  guiding_gpu_field().record(
      record.field_index, record.radiance, direction.decode(), make_float3(record.position));
  guiding_gpu_field().record_source(record.field_index,
                                    record.source_weight,
                                    record.inverse_distance_weight,
                                    record.distance_weight,
                                    direction.decode(),
                                    make_float3(record.position));
}

ccl_device_inline void guiding_gpu_record_radiance(ConstIntegratorState state,
                                                   const Spectrum contribution,
                                                   const float3 endpoint = make_float3(FLT_MAX))
{
  if (guiding_gpu_training()) {
    guiding_gpu_record_history(
        INTEGRATOR_STATE(state, gpu_guiding, history_head), contribution, endpoint);
  }
}

ccl_device_inline void guiding_gpu_record_shadow_radiance(ConstIntegratorShadowState state,
                                                          const Spectrum contribution)
{
  if (!guiding_gpu_training()) {
    return;
  }
  const GuidingField field = guiding_gpu_field();
  guiding_gpu_record_history(INTEGRATOR_STATE(state, shadow_gpu_guiding, history_head),
                             contribution,
                             INTEGRATOR_STATE(state, shadow_gpu_guiding, history_endpoint));
  const uint direct_index = INTEGRATOR_STATE(state, shadow_gpu_guiding, direct_record_index);
  if (direct_index != ~0u) {
    const Spectrum weight = INTEGRATOR_STATE(state, shadow_gpu_guiding, direct_inverse_weight);
    packed_normal direction;
    direction.value = INTEGRATOR_STATE(state, shadow_gpu_guiding, direct_record_direction);
    const GuidingHistory history{kernel_integrator_state.guiding_history,
                                 kernel_integrator_state.guiding_history_count,
                                 kernel_integrator_state.guiding_history_capacity};
    history.append_observation(direct_index,
                                direction.value,
                                INTEGRATOR_STATE(state, shadow_gpu_guiding, direct_record_position),
                                average(spectrum_to_rgb(contribution * weight)),
                                INTEGRATOR_STATE(state, shadow_gpu_guiding, direct_record_distance));
  }
}

ccl_device_inline void guiding_gpu_shadow_endpoint(IntegratorShadowState state, const float3 P)
{
  if (guiding_gpu_training()) {
    INTEGRATOR_STATE_WRITE(state, shadow_gpu_guiding, history_endpoint) = P;
  }
}

ccl_device_inline void guiding_gpu_record_direct(IntegratorShadowState shadow_state,
                                                 ConstIntegratorState state,
                                                 const float3 P,
                                                 const float3 direction,
                                                 const Spectrum scattering_throughput,
                                                 const bool volume,
                                                 const float3 normal = zero_float3(),
                                                 const bool indirect_connection = false,
                                                 const float distance = FLT_MAX)
{
  guiding_gpu_shadow_endpoint(shadow_state, P);
  if (!guiding_gpu_training() ||
      (!indirect_connection && !kernel_data.integrator.use_guiding_direct_light))
  {
    return;
  }
  /* With MIS-aware training, an ordinary NEE estimate belongs to the outgoing radiance
   * propagated to earlier vertices. Adding it at this vertex instead trains the guide toward
   * the direct-light strategy, even where NEE already accounts for almost all its energy.
   * Emitter hits along the continuation supply the MIS-weighted direct training samples,
   * matching OpenPGL's path-segment preparation. Keep the shadow history endpoint above:
   * this contribution must still train the preceding path vertices. Indirect BDPT connections
   * remain observations of incident indirect radiance at this vertex. */
  if (!indirect_connection && kernel_data.integrator.use_guiding_mis_weights) {
    return;
  }
  const GuidingField field = guiding_gpu_field();
  const uint node = field.find_leaf(P);
  const GuidingFieldType type = volume ? GUIDING_FIELD_VOLUME_RADIANCE :
                                         guiding_surface_field_type(normal, false);
  INTEGRATOR_STATE_WRITE(shadow_state,
                         shadow_gpu_guiding,
                         direct_record_index) = field.record_index(node, type, direction);
  INTEGRATOR_STATE_WRITE(
      shadow_state, shadow_gpu_guiding, direct_record_direction) = packed_normal(direction).value;
  /* Shadow throughput already includes light PDF, MIS, light termination and shader evaluation.
   * Removing only camera throughput and scattering retains the radiance estimate and its PDF. */
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_gpu_guiding, direct_inverse_weight) = safe_divide(
      one_spectrum(), scattering_throughput);
  INTEGRATOR_STATE_WRITE(
      shadow_state, shadow_gpu_guiding, direct_record_position) = field.normalized_position(P);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_gpu_guiding, direct_record_distance) = distance;
  field.record_visit(node);
}

#endif

CCL_NAMESPACE_END
