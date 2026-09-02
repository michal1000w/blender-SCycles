/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/state.h"
#include "kernel/sample/lcg.h"

#include "util/atomic.h"

CCL_NAMESPACE_BEGIN

enum ReSTIRPTTechnique : uint {
  RESTIR_PT_TECHNIQUE_EMISSION = 0u,
  RESTIR_PT_TECHNIQUE_NEE = 1u,
  RESTIR_PT_TECHNIQUE_VOLUME_EMISSION = 2u,
  RESTIR_PT_TECHNIQUE_BACKGROUND = 3u,
};

#ifdef __KERNEL_METAL__

ccl_device_inline bool restir_pt_path_supported(const uint32_t path_flag)
{
  if (!kernel_data.integrator.use_restir_pt ||
      !kernel_integrator_state.restir_pt_initial ||
      /* Photon mapping already owns the indirect estimator. ReSTIR DI remains active through the
       * use_restir_pt setting, but streaming those same indirect paths into ReSTIR PT would count
       * their energy twice. */
      kernel_data.integrator.use_photon_mapping ||
      (path_flag & (PATH_RAY_SHADOW_CATCHER_PASS | PATH_RAY_SHADOW_CATCHER_HIT)))
  {
    return false;
  }

  /* BDPT and ReSTIR PT both estimate complete camera/light strategies. Keep them disjoint:
   * ordinary BDPT vertices retain recursive MIS, while a path that locally leaves BDPT support
   * can enter the complete ReSTIR PT estimator from that point onward. */
  if (kernel_data.integrator.use_bidirectional_path_tracing &&
      !(path_flag & PATH_RAY_BDPT_UNSUPPORTED))
  {
    return false;
  }
  return true;
}

ccl_device_inline uint restir_pt_pack_path_data(const uint path_length,
                                                const uint rc_length,
                                                const uint technique,
                                                const uint age)
{
  return min(path_length, 255u) | (min(rc_length, 255u) << 8u) |
         ((technique & 0xffu) << 16u) | ((min(age, 255u) & 0xffu) << 24u);
}

ccl_device_inline void restir_pt_record_primary(KernelGlobals kg,
                                                IntegratorState state,
                                                const ccl_private ShaderData *sd)
{
  if (!kernel_data.integrator.use_restir_pt) {
    return;
  }
  const uint vertex_key = hash_uint4(uint(sd->object),
                                     uint(sd->prim),
                                     uint(sd->shader),
                                     uint(INTEGRATOR_STATE(state, path, bounce)));
  INTEGRATOR_STATE_WRITE(state, path, restir_pt_path_hash) = hash_uint2(
      INTEGRATOR_STATE(state, path, restir_pt_path_hash), vertex_key);

  if (INTEGRATOR_STATE(state, path, bounce) == 0u) {
    const float camera_spread = max(INTEGRATOR_STATE(state, ray, dD), 1.0e-8f);
    INTEGRATOR_STATE_WRITE(state, path, restir_pt_primary_footprint) =
        sqr(camera_spread * max(sd->ray_length, 1.0e-4f));
  }

  if (INTEGRATOR_STATE(state, path, bounce) != 0 ||
      kernel_integrator_state.restir_pt_phase != 0u ||
      !kernel_integrator_state.restir_pt_current_surfaces)
  {
    return;
  }
  const uint pixel_index = INTEGRATOR_STATE(state, path, render_pixel_index);
  if (pixel_index >= kernel_integrator_state.restir_pt_reservoir_capacity) {
    return;
  }
  ccl_global KernelReSTIRPTSurface *surface =
      &kernel_integrator_state.restir_pt_current_surfaces[pixel_index];
  surface->P = sd->P;
  surface->normal = packed_normal(sd->Ng).value;
  surface->object = sd->object;
  surface->prim = sd->prim;
  surface->shader = uint(sd->shader);
  surface->valid = 1u;
  const float4 motion = (kernel_data.kernel_features & KERNEL_FEATURE_OBJECT_MOTION) ?
                            primitive_motion_vector(kg, sd) :
                            zero_float4();
  surface->motion_pre = make_float2(motion.x, motion.y);
  surface->motion_post = make_float2(motion.z, motion.w);
}

ccl_device_inline void restir_pt_record_reconnection(
    IntegratorState state,
    const ccl_private ShaderData *sd,
    const float bsdf_pdf,
    const float2 sampled_roughness,
    const int label)
{
  if (!kernel_data.integrator.use_restir_pt ||
      kernel_integrator_state.restir_pt_phase != 0u ||
      (label & (LABEL_TRANSPARENT | LABEL_SINGULAR)) || !(label & LABEL_DIFFUSE) ||
      !(label & LABEL_REFLECT))
  {
    return;
  }

  const uint bounce = INTEGRATOR_STATE(state, path, bounce);
  const float roughness = min(sampled_roughness.x, sampled_roughness.y);
  if (bounce > 0u &&
      INTEGRATOR_STATE(state, path, restir_pt_rc_length) == uint16_t(0xffffu))
  {
    const float previous_roughness = INTEGRATOR_STATE(
        state, path, restir_pt_previous_roughness);
    const float distance2 = sqr(max(sd->ray_length, 1.0e-6f));
    const float3 toward_previous = normalize(sd->wi);
    const float previous_pdf = max(INTEGRATOR_STATE(state, path, mis_ray_pdf), 0.0f);
    const float3 previous_N = INTEGRATOR_STATE(state, path, mis_origin_n);
    const float forward_density = previous_pdf *
                                  fabsf(dot(sd->Ng, toward_previous)) / distance2;
    const float inverse_density = bsdf_pdf *
                                  fabsf(dot(previous_N, -toward_previous)) / distance2;
    const float dual_footprint = 1.0f /
                                 max(max(forward_density, inverse_density), 1.0e-20f);
    const float required_footprint = kernel_data.integrator.restir_pt_footprint_threshold *
                                     max(INTEGRATOR_STATE(
                                             state, path, restir_pt_primary_footprint),
                                         1.0e-12f);
    /* For a diffuse reflection reconnection vertex the outgoing sampling density is independent
     * of the changed incident direction, so the p_k(omega_k) ratio in the PSS Jacobian is exactly
     * one. Glossy/transmissive vertices need explicit black-box PDF reconstruction and therefore
     * continue by random replay/fallback instead of using an incomplete Jacobian. */
    if (previous_roughness >= kernel_data.integrator.restir_pt_min_roughness &&
        roughness >= kernel_data.integrator.restir_pt_min_roughness &&
        dual_footprint >= required_footprint)
    {
      INTEGRATOR_STATE_WRITE(state, path, restir_pt_rc_P) = sd->P;
      INTEGRATOR_STATE_WRITE(state, path, restir_pt_rc_normal) = packed_normal(sd->Ng).value;
      INTEGRATOR_STATE_WRITE(state, path, restir_pt_rc_throughput) = INTEGRATOR_STATE(
          state, path, throughput);
      /* Cache the two source-edge factors separately. The hybrid PSS Jacobian crosses source and
       * target geometry terms; caching only their product cannot reconstruct that ratio. The
       * field names are retained to avoid expanding the persistent reservoir record. */
      const float source_geometry = fabsf(dot(sd->Ng, toward_previous)) / distance2;
      INTEGRATOR_STATE_WRITE(state, path, restir_pt_rc_wi_pdf) = source_geometry;
      INTEGRATOR_STATE_WRITE(state, path, restir_pt_inverse_partial_jacobian) =
          previous_pdf;
      INTEGRATOR_STATE_WRITE(state, path, restir_pt_rc_length) = uint16_t(
          min(bounce + 1u, 0xfffeu));
    }
  }
  INTEGRATOR_STATE_WRITE(state, path, restir_pt_previous_roughness) = roughness;
}

ccl_device_inline bool restir_pt_stream_reconnection(
    const uint pixel_index, const Spectrum contribution, const float jacobian)
{
  const ccl_global KernelReSTIRPTReservoir *source = restir_pt_replay_source(pixel_index);
  if (!source) {
    return true;
  }

  const uint source_M = restir_pt_source_confidence(pixel_index, source);

  if (kernel_integrator_state.restir_pt_phase > 1u) {
    /* Spatial pairwise MIS is deliberately staged. The reciprocal lane writes the inverse shift
     * into its own staging slot; END_REUSE reads both only after every camera and shadow queue has
     * drained, avoiding cross-lane races. */
    ccl_global KernelReSTIRPTReservoir *stage =
        &kernel_integrator_state.restir_pt_initial[pixel_index];
    if (!restir_pt_try_lock(&stage->lock)) {
      return true;
    }
    const float target = reduce_max(fabs(contribution));
    if (target > 0.0f && jacobian > 0.0f && isfinite_safe(target) && isfinite_safe(jacobian)) {
      stage->contribution = contribution;
      stage->target = target;
      stage->weight_sum = jacobian;
      stage->M = source_M;
      stage->rng_pixel = source->rng_pixel;
      stage->sample = source->sample;
      stage->path_data = source->path_data;
      stage->pass_diffuse_weight = source->pass_diffuse_weight;
      stage->pass_glossy_weight = source->pass_glossy_weight;
      stage->lightgroup = source->lightgroup;
      stage->path_flag = source->path_flag;
      stage->path_hash = source->path_hash;
      stage->inverse_partial_jacobian = source->inverse_partial_jacobian;
      stage->rc_wi_pdf = source->rc_wi_pdf;
      stage->rc_P = source->rc_P;
      stage->rc_normal = source->rc_normal;
      stage->rc_throughput = source->rc_throughput;
      stage->replay_accounted = 1u;
    }
    restir_pt_unlock(&stage->lock);
    return true;
  }

  ccl_global KernelReSTIRPTReservoir *output =
      &kernel_integrator_state.restir_pt_current[pixel_index];
  if (!restir_pt_try_lock(&output->lock)) {
    return true;
  }
  if (output->replay_accounted != 0u) {
    restir_pt_unlock(&output->lock);
    return true;
  }

  const float target = reduce_max(fabs(contribution));
  const float canonical_weight = source->weight_sum * float(source_M);
  const float weight = target * canonical_weight;
  if (!(target > 0.0f) || !(weight > 0.0f) || !isfinite_safe(target) ||
      !isfinite_safe(weight))
  {
    restir_pt_unlock(&output->lock);
    return true;
  }

  output->replay_accounted = 1u;
  output->M = min(output->M + source_M, 65535u);

  const float new_weight_sum = output->weight_sum + weight;
  output->vector_sum = output->vector_sum + contribution * canonical_weight;
  const float select = hash_uint4_to_float(pixel_index,
                                           uint(output->sample),
                                           kernel_integrator_state.restir_pt_phase ^ 0x7263u,
                                           output->M + source_M);
  const bool selected = !(output->target > 0.0f) || select * new_weight_sum < weight;
  output->weight_sum = new_weight_sum;
  if (selected) {
    output->contribution = contribution;
    output->target = target;
    output->rng_pixel = source->rng_pixel;
    output->sample = source->sample;
    output->path_data = (source->path_data & 0x00ffffffu) |
                        (min(((source->path_data >> 24u) & 0xffu) + 1u, 255u) << 24u);
    output->pass_diffuse_weight = source->pass_diffuse_weight;
    output->pass_glossy_weight = source->pass_glossy_weight;
    output->lightgroup = source->lightgroup;
    output->path_flag = source->path_flag;
    output->path_hash = source->path_hash;
    output->inverse_partial_jacobian = source->inverse_partial_jacobian;
    output->rc_wi_pdf = source->rc_wi_pdf;
    output->rc_P = source->rc_P;
    output->rc_normal = source->rc_normal;
    output->rc_throughput = source->rc_throughput;
  }
  restir_pt_unlock(&output->lock);
  return true;
}

ccl_device_inline bool restir_pt_try_lock(ccl_global uint *lock)
{
  /* Never spin in a GPU SIMD group: the lane owning the lock may be unable to run while sibling
   * lanes spin. A contended initial contribution falls back to the ordinary exact film write. */
  return atomic_exchange_uint32(lock, 1u) == 0u;
}

ccl_device_inline void restir_pt_unlock(ccl_global uint *lock)
{
  atomic_exchange_uint32(lock, 0u);
}

ccl_device_inline uint restir_pt_source_index(const uint pixel_index)
{
  if (kernel_integrator_state.restir_pt_phase == 0u) {
    return pixel_index;
  }

  const int delta = int(pixel_index) - kernel_integrator_state.restir_buffer_offset;
  const int x = delta % kernel_integrator_state.restir_buffer_stride;
  const int y = delta / kernel_integrator_state.restir_buffer_stride;
  if (kernel_integrator_state.restir_pt_phase == 1u) {
    if (!kernel_integrator_state.restir_pt_current_surfaces) {
      return pixel_index;
    }
    const ccl_global KernelReSTIRPTSurface *surface =
        &kernel_integrator_state.restir_pt_current_surfaces[pixel_index];
    const int nx = x + int(roundf(surface->motion_pre.x));
    const int ny = y + int(roundf(surface->motion_pre.y));
    if (nx < kernel_integrator_state.restir_buffer_full_x ||
        nx >= kernel_integrator_state.restir_buffer_full_x +
                  kernel_integrator_state.restir_buffer_width ||
        ny < kernel_integrator_state.restir_buffer_full_y ||
        ny >= kernel_integrator_state.restir_buffer_full_y +
                  kernel_integrator_state.restir_buffer_height)
    {
      return UINT_MAX;
    }
    return uint(kernel_integrator_state.restir_buffer_offset + nx +
                ny * kernel_integrator_state.restir_buffer_stride);
  }

  const uint iteration = kernel_integrator_state.restir_pt_spatial_iteration;
  const int radius = max(kernel_data.integrator.restir_pt_spatial_radius, 1);
  /* Guarantee one useful local reciprocal pair even for small viewport tiles. Additional passes
   * retain the wider hashed distribution requested by the user radius. */
  const int step = (iteration == 0u) ? 1 :
                                         1 + int(hash_uint2(iteration, 0x51a7u) % uint(radius));
  int nx = x;
  int ny = y;
  if ((iteration % 3u) != 1u) {
    const int cell = x / step;
    nx += (cell & 1) ? -step : step;
  }
  if ((iteration % 3u) != 0u) {
    const int cell = y / step;
    ny += (cell & 1) ? -step : step;
  }
  if (nx < kernel_integrator_state.restir_buffer_full_x ||
      nx >= kernel_integrator_state.restir_buffer_full_x +
                kernel_integrator_state.restir_buffer_width ||
      ny < kernel_integrator_state.restir_buffer_full_y ||
      ny >= kernel_integrator_state.restir_buffer_full_y +
                kernel_integrator_state.restir_buffer_height)
  {
    return UINT_MAX;
  }
  return uint(kernel_integrator_state.restir_buffer_offset + nx +
              ny * kernel_integrator_state.restir_buffer_stride);
}

ccl_device_inline bool restir_pt_primary_compatible(const uint pixel_index,
                                                    const uint source_index)
{
  if (!kernel_integrator_state.restir_pt_current_surfaces ||
      !kernel_integrator_state.restir_pt_source_surfaces ||
      source_index >= kernel_integrator_state.restir_pt_reservoir_capacity)
  {
    return false;
  }
  const ccl_global KernelReSTIRPTSurface *current =
      &kernel_integrator_state.restir_pt_current_surfaces[pixel_index];
  const ccl_global KernelReSTIRPTSurface *source =
      &kernel_integrator_state.restir_pt_source_surfaces[source_index];
  if (!current->valid || !source->valid || current->shader != source->shader) {
    return false;
  }
  packed_normal current_N;
  packed_normal source_N;
  current_N.value = current->normal;
  source_N.value = source->normal;
  if (dot(current_N.decode(), source_N.decode()) < 0.5f) {
    return false;
  }
  const float distance = len(make_float3(current->P) - make_float3(source->P));
  const float scale = max(len(make_float3(current->P)), 1.0f);
  /* Temporal reprojection should be conservative. Spatial shifts can start farther apart because
   * their reciprocal hybrid mapping is subsequently checked by the dual-footprint criterion. */
  const float position_fraction = (kernel_integrator_state.restir_pt_phase > 1u) ? 0.5f : 0.1f;
  return distance <= max(position_fraction * scale, 1.0e-4f);
}

ccl_device_inline uint restir_pt_temporal_post_source_index(const uint pixel_index)
{
  if (!kernel_integrator_state.restir_pt_current_surfaces) {
    return UINT_MAX;
  }
  const int delta = int(pixel_index) - kernel_integrator_state.restir_buffer_offset;
  const int x = delta % kernel_integrator_state.restir_buffer_stride;
  const int y = delta / kernel_integrator_state.restir_buffer_stride;
  const ccl_global KernelReSTIRPTSurface *surface =
      &kernel_integrator_state.restir_pt_current_surfaces[pixel_index];
  const int nx = x + int(roundf(surface->motion_post.x));
  const int ny = y + int(roundf(surface->motion_post.y));
  if (nx < kernel_integrator_state.restir_buffer_full_x ||
      nx >= kernel_integrator_state.restir_buffer_full_x +
                kernel_integrator_state.restir_buffer_width ||
      ny < kernel_integrator_state.restir_buffer_full_y ||
      ny >= kernel_integrator_state.restir_buffer_full_y +
                kernel_integrator_state.restir_buffer_height)
  {
    return UINT_MAX;
  }
  return uint(kernel_integrator_state.restir_buffer_offset + nx +
              ny * kernel_integrator_state.restir_buffer_stride);
}

ccl_device_inline uint restir_pt_resolved_source_index(const uint pixel_index)
{
  uint source_index = restir_pt_source_index(pixel_index);
  if (kernel_integrator_state.restir_pt_phase == 1u &&
      (source_index == UINT_MAX || !restir_pt_primary_compatible(pixel_index, source_index)))
  {
    const uint alternate = restir_pt_temporal_post_source_index(pixel_index);
    if (alternate != UINT_MAX && restir_pt_primary_compatible(pixel_index, alternate)) {
      source_index = alternate;
    }
  }
  return source_index;
}

ccl_device_inline const ccl_global KernelReSTIRPTReservoir *restir_pt_replay_source(
    const uint pixel_index)
{
  if (kernel_integrator_state.restir_pt_phase == 0u ||
      !kernel_integrator_state.restir_pt_source)
  {
    return nullptr;
  }
  const uint source_index = restir_pt_resolved_source_index(pixel_index);
  if (source_index == UINT_MAX || !restir_pt_primary_compatible(pixel_index, source_index)) {
    return nullptr;
  }
  const ccl_global KernelReSTIRPTReservoir *source =
      &kernel_integrator_state.restir_pt_source[source_index];
  /* A spatial full-replay suffix changes domains without the inverse shift needed by pairwise
   * MIS and can amplify weights exponentially. Only admit spatial candidates carrying a valid
   * reconnection PDF/Jacobian; all other pixels keep their canonical fresh reservoir locally. */
  if (kernel_integrator_state.restir_pt_phase > 1u &&
      (!(source->rc_wi_pdf > 0.0f) || !(source->inverse_partial_jacobian > 0.0f)))
  {
    return nullptr;
  }
  return (source->target > 0.0f && source->M > 0u) ? source : nullptr;
}

ccl_device_inline uint restir_pt_source_confidence(
    const uint pixel_index, const ccl_global KernelReSTIRPTReservoir *source)
{
  uint cap = (kernel_integrator_state.restir_pt_phase == 1u) ?
                 uint(kernel_data.integrator.restir_pt_temporal_history) :
                 65535u;
  if (kernel_integrator_state.restir_pt_phase == 1u &&
      kernel_data.integrator.restir_pt_decorrelate &&
      kernel_integrator_state.restir_pt_duplication)
  {
    const uint source_index = restir_pt_resolved_source_index(pixel_index);
    if (source_index != UINT_MAX) {
      const float duplication = saturatef(
          kernel_integrator_state.restir_pt_duplication[source_index]);
      const float age_score = saturatef(float((source->path_data >> 24u) & 0xffu) / 20.0f);
      const float adaptive_cap = mix(float(max(cap, 1u)),
                                     1.0f,
                                     powf(max(duplication, age_score), 0.1f));
      cap = max(uint(adaptive_cap + 0.5f), 1u);
    }
  }
  return min(source->M, max(cap, 1u));
}

ccl_device_inline bool restir_pt_begin_replay(IntegratorState state)
{
  if (kernel_integrator_state.restir_pt_phase == 0u ||
      INTEGRATOR_STATE(state, path, bounce) != 0)
  {
    return true;
  }
  const uint pixel_index = INTEGRATOR_STATE(state, path, render_pixel_index);
  const ccl_global KernelReSTIRPTReservoir *source = restir_pt_replay_source(pixel_index);
  if (!source) {
    return false;
  }
  /* Camera/lens/time samples remain in the current domain. All dimensions beginning at the
   * primary surface are replayed from the selected source path. */
  INTEGRATOR_STATE_WRITE(state, path, rng_pixel) = source->rng_pixel;
  INTEGRATOR_STATE_WRITE(state, path, sample) = source->sample;
  return true;
}

ccl_device_inline bool restir_pt_stream_replay(const uint pixel_index,
                                               const uint path_length,
                                               const uint technique,
                                               const Spectrum contribution,
                                               const Spectrum pass_diffuse_weight,
                                               const Spectrum pass_glossy_weight,
                                               const uint lightgroup,
                                               const uint path_visibility,
                                               const uint32_t path_flag,
                                               const uint path_hash)
{
  const ccl_global KernelReSTIRPTReservoir *source = restir_pt_replay_source(pixel_index);
  if (!source) {
    return true;
  }
  const uint source_path_length = source->path_data & 0xffu;
  const uint source_technique = (source->path_data >> 16u) & 0xffu;
  if (path_length != source_path_length || technique != source_technique) {
    return true;
  }

  const uint source_M = restir_pt_source_confidence(pixel_index, source);
  ccl_global KernelReSTIRPTReservoir *output =
      &kernel_integrator_state.restir_pt_current[pixel_index];
  if (!restir_pt_try_lock(&output->lock)) {
    return true;
  }
  if (output->replay_accounted != 0u) {
    restir_pt_unlock(&output->lock);
    return true;
  }

  const float target = reduce_max(fabs(contribution));
  if (!(target > 0.0f) || !isfinite_safe(target)) {
    restir_pt_unlock(&output->lock);
    return true;
  }
  const float target_ratio = target / max(source->target, 1.0e-20f);
  /* This is a symmetric invertibility gate: replay must preserve both the material/topology
   * sequence and the target density to within a reciprocal bound. Hybrid reconnection replaces
   * this conservative full-replay gate when a valid earlier reconnectible vertex is available. */
  if (path_hash != source->path_hash || target_ratio < 0.5f || target_ratio > 2.0f)
  {
    restir_pt_unlock(&output->lock);
    return true;
  }
  const float weight = target * source->weight_sum * float(source_M);
  if (!(weight > 0.0f) || !isfinite_safe(weight)) {
    restir_pt_unlock(&output->lock);
    return true;
  }

  output->replay_accounted = 1u;
  output->M = min(output->M + source_M, 65535u);

  const float new_weight_sum = output->weight_sum + weight;
  /* Marginalize the random reservoir index for shading. This vector estimator is the sum of
   * F(Y_i) W_i c_i and removes color noise and rare scalar-selection fireflies. */
  output->vector_sum = output->vector_sum +
                       contribution * (source->weight_sum * float(source_M));
  const float select = hash_uint4_to_float(pixel_index,
                                           uint(output->sample),
                                           kernel_integrator_state.restir_pt_phase,
                                           output->M + source_M);
  const bool selected = !(output->target > 0.0f) ||
                        select * new_weight_sum < weight;
  output->weight_sum = new_weight_sum;
  if (selected) {
    output->contribution = contribution;
    output->target = target;
    output->rng_pixel = source->rng_pixel;
    output->sample = source->sample;
    output->path_data = (source->path_data & 0x00ffffffu) |
                        (min(((source->path_data >> 24u) & 0xffu) + 1u, 255u) << 24u);
    output->pass_diffuse_weight = pass_diffuse_weight;
    output->pass_glossy_weight = pass_glossy_weight;
    output->lightgroup = (lightgroup & 0xffu) | ((path_visibility & 0xffu) << 8u);
    output->path_flag = path_flag;
    output->path_hash = path_hash;
    output->inverse_partial_jacobian = source->inverse_partial_jacobian;
    output->rc_wi_pdf = source->rc_wi_pdf;
    output->rc_P = source->rc_P;
    output->rc_normal = source->rc_normal;
    output->rc_throughput = source->rc_throughput;
  }
  restir_pt_unlock(&output->lock);
  return true;
}

/* Stream an exact visible contribution from the initial path tree. Every NEE and forward-emitter
 * technique is a separate candidate, so the final normalization is weight_sum / target(selected)
 * without division by M. M remains the GRIS confidence used by later reuse passes. */
ccl_device_inline bool restir_pt_stream_initial(
    const uint pixel_index,
    const uint rng_pixel,
    const uint sample,
    const uint path_length,
    const uint technique,
    const Spectrum contribution,
    const Spectrum pass_diffuse_weight,
    const Spectrum pass_glossy_weight,
    const uint lightgroup,
    const uint path_visibility,
    const uint32_t path_flag,
    const uint path_hash,
    const uint rc_length,
    const float3 rc_P,
    const uint rc_normal,
    const Spectrum rc_throughput,
    const float rc_wi_pdf,
    const float inverse_partial_jacobian)
{
  if (!restir_pt_path_supported(path_flag) ||
      pixel_index >= kernel_integrator_state.restir_pt_reservoir_capacity)
  {
    return false;
  }

  /* ReSTIR PT's path-space estimator excludes direct-illumination paths. Cycles' ReSTIR DI
   * estimator handles primary NEE, while directly visible emission/background retain their exact
   * film writes. Such paths have no hybrid reconnection vertex and feeding them to temporal GRIS
   * causes confidence amplification without a valid shift. */
  if (path_length <= 2u) {
    return false;
  }

  if (kernel_integrator_state.restir_pt_phase != 0u) {
    return restir_pt_stream_replay(pixel_index,
                                   path_length,
                                   technique,
                                   contribution,
                                   pass_diffuse_weight,
                                   pass_glossy_weight,
                                   lightgroup,
                                   path_visibility,
                                   path_flag,
                                   path_hash);
  }

  const float target = reduce_max(fabs(contribution));
  if (!(target > 0.0f) || !isfinite_safe(target)) {
    return true;
  }

  ccl_global KernelReSTIRPTReservoir *reservoir =
      &kernel_integrator_state.restir_pt_initial[pixel_index];
  if (!restir_pt_try_lock(&reservoir->lock)) {
    return false;
  }

  const float new_weight_sum = reservoir->weight_sum + target;
  const uint candidate_index = reservoir->M;
  const float select = hash_uint4_to_float(
      pixel_index, sample, (path_length << 8u) | technique, candidate_index);
  const bool selected = !(reservoir->target > 0.0f) ||
                        select * new_weight_sum < target;

  reservoir->weight_sum = new_weight_sum;
  reservoir->vector_sum = reservoir->vector_sum + contribution;
  reservoir->M = min(candidate_index + 1u, 65535u);
  if (selected) {
    reservoir->contribution = contribution;
    reservoir->target = target;
    reservoir->rng_pixel = rng_pixel;
    reservoir->sample = sample;
    const uint selected_rc_length = (rc_length <= path_length) ? rc_length : path_length + 1u;
    reservoir->path_data = restir_pt_pack_path_data(
        path_length, selected_rc_length, technique, 0u);
    reservoir->pass_diffuse_weight = pass_diffuse_weight;
    reservoir->pass_glossy_weight = pass_glossy_weight;
    reservoir->lightgroup = (lightgroup & 0xffu) | ((path_visibility & 0xffu) << 8u);
    reservoir->path_flag = path_flag;
    reservoir->path_hash = path_hash;
    reservoir->inverse_partial_jacobian = (selected_rc_length <= path_length) ?
                                                inverse_partial_jacobian :
                                                0.0f;
    reservoir->rc_wi_pdf = (selected_rc_length <= path_length) ? rc_wi_pdf : 0.0f;
    reservoir->rc_P = rc_P;
    reservoir->rc_normal = rc_normal;
    reservoir->rc_throughput = rc_throughput;
  }

  restir_pt_unlock(&reservoir->lock);
  return true;
}

ccl_device_inline bool restir_pt_stream_initial(KernelGlobals /*kg*/,
                                                ConstIntegratorState state,
                                                const Spectrum contribution,
                                                const uint technique,
                                                const uint lightgroup)
{
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  Spectrum diffuse_weight = zero_spectrum();
  Spectrum glossy_weight = zero_spectrum();
  if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES) {
    diffuse_weight = INTEGRATOR_STATE(state, path, pass_diffuse_weight);
    glossy_weight = INTEGRATOR_STATE(state, path, pass_glossy_weight);
  }
  return restir_pt_stream_initial(INTEGRATOR_STATE(state, path, render_pixel_index),
                                  INTEGRATOR_STATE(state, path, rng_pixel),
                                  INTEGRATOR_STATE(state, path, sample),
                                  uint(INTEGRATOR_STATE(state, path, bounce)) + 1u,
                                  technique,
                                  contribution,
                                  diffuse_weight,
                                  glossy_weight,
                                  uint(lightgroup + 1),
                                  uint(INTEGRATOR_STATE(state, path, visibility)),
                                  path_flag,
                                  INTEGRATOR_STATE(state, path, restir_pt_path_hash),
                                  uint(INTEGRATOR_STATE(state, path, restir_pt_rc_length)),
                                  INTEGRATOR_STATE(state, path, restir_pt_rc_P),
                                  INTEGRATOR_STATE(state, path, restir_pt_rc_normal),
                                  INTEGRATOR_STATE(state, path, restir_pt_rc_throughput),
                                  INTEGRATOR_STATE(state, path, restir_pt_rc_wi_pdf),
                                  INTEGRATOR_STATE(
                                      state, path, restir_pt_inverse_partial_jacobian));
}

ccl_device_inline bool restir_pt_stream_initial(KernelGlobals /*kg*/,
                                                ConstIntegratorShadowState state,
                                                const Spectrum contribution)
{
  if (INTEGRATOR_STATE(state, shadow_path, restir_pt_reconnection) != 0u) {
    return restir_pt_stream_reconnection(
        INTEGRATOR_STATE(state, shadow_path, render_pixel_index),
        contribution,
        INTEGRATOR_STATE(state, shadow_path, restir_pt_jacobian));
  }
  const uint32_t path_flag = INTEGRATOR_STATE(state, shadow_path, flag);
  Spectrum diffuse_weight = zero_spectrum();
  Spectrum glossy_weight = zero_spectrum();
  if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES) {
    diffuse_weight = INTEGRATOR_STATE(state, shadow_path, pass_diffuse_weight);
    glossy_weight = INTEGRATOR_STATE(state, shadow_path, pass_glossy_weight);
  }
  return restir_pt_stream_initial(INTEGRATOR_STATE(state, shadow_path, render_pixel_index),
                                  INTEGRATOR_STATE(state, shadow_path, rng_pixel),
                                  INTEGRATOR_STATE(state, shadow_path, sample),
                                  uint(INTEGRATOR_STATE(state, shadow_path, bounce)) + 2u,
                                  RESTIR_PT_TECHNIQUE_NEE,
                                  contribution,
                                  diffuse_weight,
                                  glossy_weight,
                                  uint(INTEGRATOR_STATE(state, shadow_path, lightgroup)),
                                  uint(INTEGRATOR_STATE(state, shadow_path, visibility)),
                                  path_flag,
                                  INTEGRATOR_STATE(state, shadow_path, restir_pt_path_hash),
                                  uint(INTEGRATOR_STATE(state, shadow_path, restir_pt_rc_length)),
                                  INTEGRATOR_STATE(state, shadow_path, restir_pt_rc_P),
                                  INTEGRATOR_STATE(state, shadow_path, restir_pt_rc_normal),
                                  INTEGRATOR_STATE(state, shadow_path, restir_pt_rc_throughput),
                                  INTEGRATOR_STATE(state, shadow_path, restir_pt_rc_wi_pdf),
                                  INTEGRATOR_STATE(
                                      state, shadow_path, restir_pt_inverse_partial_jacobian));
}

#endif /* __KERNEL_METAL__ */

CCL_NAMESPACE_END
