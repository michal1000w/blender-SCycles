/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/film/light_passes.h"
#include "kernel/integrator/restir_pt.h"

CCL_NAMESPACE_BEGIN

#ifdef __KERNEL_METAL__

ccl_device_inline void restir_pt_begin_reuse(const uint pixel_index)
{
  if (pixel_index >= kernel_integrator_state.restir_pt_reservoir_capacity ||
      !kernel_integrator_state.restir_pt_initial ||
      !kernel_integrator_state.restir_pt_current)
  {
    return;
  }
  const ccl_global KernelReSTIRPTReservoir *initial =
      (kernel_integrator_state.restir_pt_phase == 1u) ?
          &kernel_integrator_state.restir_pt_initial[pixel_index] :
          &kernel_integrator_state.restir_pt_source[pixel_index];
  ccl_global KernelReSTIRPTReservoir *current =
      &kernel_integrator_state.restir_pt_current[pixel_index];
  *current = *initial;
  if (kernel_integrator_state.restir_pt_phase == 1u) {
    current->M = (initial->target > 0.0f) ? 1u : 0u;
  }
  else if (initial->target > 0.0f && initial->M > 0u) {
    /* Pairwise MIS streams the canonical domain once. Confidence participates in its balance
     * weights but is not a multiplicity in the pairwise normalization. */
    current->weight_sum = initial->target * initial->weight_sum;
    current->vector_sum = initial->vector_sum;
  }
  current->lock = 0u;
  current->replay_accounted = 0u;
}

ccl_device_inline void restir_pt_normalize(const uint pixel_index)
{
  if (pixel_index >= kernel_integrator_state.restir_pt_reservoir_capacity ||
      !kernel_integrator_state.restir_pt_current)
  {
    return;
  }
  ccl_global KernelReSTIRPTReservoir *reservoir =
      &kernel_integrator_state.restir_pt_current[pixel_index];
  if (reservoir->target > 0.0f && reservoir->weight_sum > 0.0f && reservoir->M > 0u) {
    if (kernel_integrator_state.restir_pt_phase > 1u) {
      reservoir->weight_sum /= reservoir->target;
    }
    else {
      reservoir->weight_sum /= float(reservoir->M) * reservoir->target;
      reservoir->vector_sum = reservoir->vector_sum / float(reservoir->M);
    }
  }
  reservoir->lock = 0u;
  reservoir->replay_accounted = 0u;
}

ccl_device_inline void restir_pt_compute_duplication(const uint pixel_index)
{
  if (pixel_index >= kernel_integrator_state.restir_pt_reservoir_capacity ||
      !kernel_integrator_state.restir_pt_current ||
      !kernel_integrator_state.restir_pt_duplication)
  {
    return;
  }
  const ccl_global KernelReSTIRPTReservoir *center =
      &kernel_integrator_state.restir_pt_current[pixel_index];
  if (!(center->target > 0.0f)) {
    kernel_integrator_state.restir_pt_duplication[pixel_index] = 0.0f;
    return;
  }

  const int delta = int(pixel_index) - kernel_integrator_state.restir_buffer_offset;
  const int x = delta % kernel_integrator_state.restir_buffer_stride;
  const int y = delta / kernel_integrator_state.restir_buffer_stride;
  uint duplicates = 0u;
  uint comparisons = 0u;
  for (int oy = -8; oy <= 8; ++oy) {
    const int ny = y + oy;
    if (ny < kernel_integrator_state.restir_buffer_full_y ||
        ny >= kernel_integrator_state.restir_buffer_full_y +
                  kernel_integrator_state.restir_buffer_height)
    {
      continue;
    }
    for (int ox = -8; ox <= 8; ++ox) {
      if (ox == 0 && oy == 0) {
        continue;
      }
      const int nx = x + ox;
      if (nx < kernel_integrator_state.restir_buffer_full_x ||
          nx >= kernel_integrator_state.restir_buffer_full_x +
                    kernel_integrator_state.restir_buffer_width)
      {
        continue;
      }
      const uint neighbor_index = uint(kernel_integrator_state.restir_buffer_offset + nx +
                                       ny * kernel_integrator_state.restir_buffer_stride);
      if (neighbor_index >= kernel_integrator_state.restir_pt_reservoir_capacity) {
        continue;
      }
      const ccl_global KernelReSTIRPTReservoir *neighbor =
          &kernel_integrator_state.restir_pt_current[neighbor_index];
      comparisons++;
      duplicates += uint(neighbor->target > 0.0f &&
                         neighbor->rng_pixel == center->rng_pixel &&
                         neighbor->sample == center->sample);
    }
  }
  kernel_integrator_state.restir_pt_duplication[pixel_index] =
      comparisons ? float(duplicates) / float(comparisons) : 0.0f;
}

ccl_device_inline void restir_pt_end_reuse(const uint pixel_index)
{
  if (pixel_index >= kernel_integrator_state.restir_pt_reservoir_capacity ||
      !kernel_integrator_state.restir_pt_current)
  {
    return;
  }
  const ccl_global KernelReSTIRPTReservoir *source = restir_pt_replay_source(pixel_index);
  if (!source) {
    return;
  }
  ccl_global KernelReSTIRPTReservoir *output =
      &kernel_integrator_state.restir_pt_current[pixel_index];
  if (kernel_integrator_state.restir_pt_phase > 1u) {
    const ccl_global KernelReSTIRPTReservoir *stage =
        &kernel_integrator_state.restir_pt_initial[pixel_index];
    const uint source_index = restir_pt_source_index(pixel_index);
    if (!stage->replay_accounted || source_index == UINT_MAX ||
        restir_pt_source_index(source_index) != pixel_index)
    {
      return;
    }
    const ccl_global KernelReSTIRPTReservoir *inverse_stage =
        &kernel_integrator_state.restir_pt_initial[source_index];
    if (!inverse_stage->replay_accounted) {
      return;
    }

    const float q_neighbor_neighbor = source->target;
    const float q_neighbor_canonical = stage->target * stage->weight_sum;
    const float q_canonical_neighbor = inverse_stage->target * inverse_stage->weight_sum;
    const float q_canonical_canonical = output->target;
    const float M_neighbor = float(max(stage->M, 1u));
    const float M_canonical = float(max(output->M, 1u));
    const float neighbor_denom = M_neighbor * q_neighbor_neighbor +
                                 M_canonical * q_neighbor_canonical;
    const float canonical_denom = M_neighbor * q_canonical_neighbor +
                                  M_canonical * q_canonical_canonical;
    if (!(neighbor_denom > 0.0f) || !(canonical_denom > 0.0f)) {
      return;
    }
    const float neighbor_mis = M_neighbor * q_neighbor_neighbor / neighbor_denom;
    const float canonical_mis = M_canonical * q_canonical_canonical / canonical_denom;
    const float neighbor_ratio = min(
        q_neighbor_canonical / max(q_neighbor_neighbor, 1.0e-20f), 1.0f);
    const float canonical_ratio = min(
        q_canonical_neighbor / max(q_canonical_canonical, 1.0e-20f), 1.0f);
    const float reciprocal_ratio = min(neighbor_ratio, canonical_ratio);
    /* Pairwise MIS reduces the effective confidence rapidly when either reciprocal mapping has
     * little overlap. This confidence must also be used by the final normalization; retaining the
     * full source M after downweighting is an energy-losing estimator. */
    if (reciprocal_ratio < 0.5f || !isfinite_safe(reciprocal_ratio)) {
      return;
    }
    const float confidence_factor = powf(reciprocal_ratio, 8.0f);
    const uint effective_M = max(uint(M_neighbor * confidence_factor + 0.5f), 1u);

    output->weight_sum *= canonical_mis;
    output->vector_sum = output->vector_sum * canonical_mis;
    const float canonical_weight = source->weight_sum * stage->weight_sum *
                                   neighbor_mis;
    const float weight = stage->target * canonical_weight;
    const float new_weight_sum = output->weight_sum + weight;
    output->vector_sum = output->vector_sum + Spectrum(stage->contribution) * canonical_weight;
    const float select = hash_uint4_to_float(pixel_index,
                                             uint(output->sample),
                                             kernel_integrator_state.restir_pt_phase ^ 0x706du,
                                             output->M + stage->M);
    const bool selected = !(output->target > 0.0f) || select * new_weight_sum < weight;
    output->weight_sum = new_weight_sum;
    output->M = min(output->M + effective_M, 65535u);
    output->replay_accounted = 1u;
    if (selected) {
      output->contribution = stage->contribution;
      output->target = stage->target;
      output->rng_pixel = stage->rng_pixel;
      output->sample = stage->sample;
      output->path_data = stage->path_data;
      output->pass_diffuse_weight = stage->pass_diffuse_weight;
      output->pass_glossy_weight = stage->pass_glossy_weight;
      output->lightgroup = stage->lightgroup;
      output->path_flag = stage->path_flag;
      output->path_hash = stage->path_hash;
      output->inverse_partial_jacobian = stage->inverse_partial_jacobian;
      output->rc_wi_pdf = stage->rc_wi_pdf;
      output->rc_P = stage->rc_P;
      output->rc_normal = stage->rc_normal;
      output->rc_throughput = stage->rc_throughput;
    }
    return;
  }
  /* Failed, mismatched, or occluded shifts do not contribute confidence. The canonical fresh
   * reservoir copied by BEGIN_REUSE remains untouched in that case. */
}

/* Write the selected initial path-tree estimate after all camera and shadow queues have drained. */
ccl_device_inline void restir_pt_finalize_initial(KernelGlobals kg,
                                                  const uint pixel_index,
                                                  ccl_global float *render_buffer)
{
  if (pixel_index >= kernel_integrator_state.restir_pt_reservoir_capacity) {
    return;
  }
  ccl_global KernelReSTIRPTReservoir *reservoir =
      (kernel_integrator_state.restir_pt_phase == 0u) ?
          &kernel_integrator_state.restir_pt_initial[pixel_index] :
          &kernel_integrator_state.restir_pt_current[pixel_index];
  if (!(reservoir->target > 0.0f) || !(reservoir->weight_sum > 0.0f)) {
    return;
  }

  const float output_weight = (kernel_integrator_state.restir_pt_phase == 0u) ?
                                  reservoir->weight_sum / reservoir->target :
                              (kernel_integrator_state.restir_pt_phase > 1u) ?
                                  reservoir->weight_sum / reservoir->target :
                                  reservoir->weight_sum /
                                      (float(max(reservoir->M, 1u)) * reservoir->target);
  const Spectrum contribution = (kernel_integrator_state.restir_pt_phase == 0u) ?
                                    reservoir->vector_sum :
                                (kernel_integrator_state.restir_pt_phase > 1u) ?
                                    Spectrum(reservoir->vector_sum) :
                                    reservoir->vector_sum / float(max(reservoir->M, 1u));
  if (!isfinite_safe(reduce_max(fabs(contribution)))) {
    return;
  }
  /* Persist the finalized UCW for next-frame/sample GRIS input. */
  reservoir->weight_sum = output_weight;
  reservoir->vector_sum = contribution;
  reservoir->lock = 0u;
  ccl_global float *buffer = render_buffer +
                             uint64_t(pixel_index) * kernel_data.film.pass_stride;
  const PathRayVisibility visibility = PathRayVisibility((reservoir->lightgroup >> 8u) & 0xffu);
  const uint32_t path_flag = reservoir->path_flag;
  film_write_combined_pass(
      kg, visibility, path_flag, int(reservoir->sample), contribution, buffer);
  /* Component and light-group passes are written from every exact initial path candidate in
   * light_passes.h. A resampled vector_sum mixes multiple techniques, path lengths, closure
   * weights, and light groups, so assigning it using the selected sample's metadata is incorrect.
   * Auxiliary replay phases deliberately write no component passes. */
}

#endif /* __KERNEL_METAL__ */

CCL_NAMESPACE_END
