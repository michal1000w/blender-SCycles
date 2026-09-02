/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

CCL_NAMESPACE_BEGIN

#ifdef __KERNEL_METAL__

/* Streaming working state for direct-light GRIS. Samples are kept in canonical light-sampling
 * coordinates instead of storing a large LightSample, which also lets moving emitters be
 * reconstructed at the current shutter time. */
struct ReSTIRDIWorkingReservoir {
  float weight_sum = 0.0f;
  float selected_target = 0.0f;
  float2 selected_rand = zero_float2();
  uint M = 0;
  uint selected_M = 0;
  uint age = 0;
  int emitter_id = -1;
};

ccl_device_inline bool restir_di_surface_nee_supported(const ccl_private ShaderData *sd,
                                                       const uint32_t path_flag)
{
  bool supported = !(path_flag & PATH_RAY_MIS_SKIP) &&
                   (sd->shader_flag &
                    ((sd->runtime_flag & SR_BACKFACING) ? SD_MIS_BACK : SD_MIS_FRONT));
#  ifdef __HAIR__
  supported &= (sd->type & PRIMITIVE_TRIANGLE);
#  endif
  return supported;
}

ccl_device_inline bool restir_di_stream(ccl_private ReSTIRDIWorkingReservoir *reservoir,
                                        const int emitter_id,
                                        const float2 rand,
                                        const float target,
                                        const float weight,
                                        const uint candidate_M,
                                        const uint age,
                                        ccl_private uint *rng)
{
  if (!(weight > 0.0f) || !isfinite_safe(weight) || !(target > 0.0f) || candidate_M == 0) {
    return false;
  }

  reservoir->weight_sum += weight;
  reservoir->M += candidate_M;
  if (!isfinite_safe(reservoir->weight_sum)) {
    return false;
  }

  const bool select = reservoir->emitter_id < 0 ||
                      lcg_step_float(rng) * reservoir->weight_sum < weight;
  if (select) {
    reservoir->emitter_id = emitter_id;
    reservoir->selected_rand = rand;
    reservoir->selected_target = target;
    reservoir->selected_M = candidate_M;
    reservoir->age = age;
  }
  return true;
}

/* Reconstruct one canonical sample at the current shading point. This evaluates the current light
 * tree selection PDF, so reuse remains valid with position-dependent many-light sampling. */
ccl_device_inline bool restir_di_reconstruct_light(KernelGlobals kg,
                                                   IntegratorState state,
                                                   const ccl_private ShaderData *sd,
                                                   const int emitter_id,
                                                   const float2 rand,
                                                   ccl_private LightSample *ls)
{
  if (emitter_id < 0 || emitter_id >= kernel_data.integrator.num_distribution) {
#  ifdef __LIGHT_TREE__
    if (!kernel_data.integrator.use_light_tree) {
      return false;
    }
#  else
    return false;
#  endif
  }

  ls->emitter_id = emitter_id;
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  const int receiver = light_link_receiver_nee(kg, sd);

#  ifdef __LIGHT_TREE__
  if (kernel_data.integrator.use_light_tree) {
    const ccl_global KernelLightTreeEmitter *kemitter = &kernel_data_fetch(light_tree_emitters,
                                                                           emitter_id);
    const bool triangle = is_triangle(kemitter);
    const int emitter_object = triangle ? kemitter->object_id : 0;
    if (triangle) {
      ls->object = emitter_object;
    }
    ls->pdf_selection = light_tree_pdf<false>(
        kg, sd->P, sd->N, 0.0f, path_flag, emitter_object, uint(emitter_id), receiver);
    if (!(ls->pdf_selection > 0.0f)) {
      return false;
    }
  }
  else
#  endif
  {
    if (emitter_id >= kernel_data.integrator.num_distribution) {
      return false;
    }
    ls->pdf_selection = kernel_data.integrator.distribution_pdf_lights;
  }

  const uint bounce = INTEGRATOR_STATE(state, path, bounce);
  return light_sample<false>(kg,
                             make_float3(rand.x, rand.y, 0.0f),
                             sd->time,
                             sd->P,
                             sd->N,
                             receiver,
                             sd->runtime_flag,
                             bounce,
                             path_flag,
                             ls);
}

/* Evaluate the emitted radiance at the sampled endpoint for a non-constant light shader. ReSTIR
 * needs this in its target: replacing a textured emitter with a unit proxy gives formally valid
 * RIS weights, but catastrophically poor finite-sample tails when texels differ by orders of
 * magnitude. This mirrors SHADE_LIGHT_NEE's endpoint setup, while visibility is still deferred
 * until after reservoir selection. */
ccl_device_inline bool restir_di_emission_eval(KernelGlobals kg,
                                               IntegratorState state,
                                               ccl_private ShaderData *sd,
                                               const ccl_private LightSample *ls,
                                               ccl_private Spectrum *eval)
{
  if (ls->type != LIGHT_TRIANGLE && ls->type != LIGHT_BACKGROUND) {
    return light_sample_shader_eval_forward(
               kg, state, ls->prim, sd->P, ls->D, ls->t, sd->time, *eval) == SHADER_EVAL_OK;
  }

  ShaderDataTinyStorage emission_sd_storage;
  ccl_private ShaderData *emission_sd = AS_SHADER_DATA(&emission_sd_storage);
  if (ls->type == LIGHT_BACKGROUND) {
    shader_setup_from_background(kg, emission_sd, sd->P, ls->D, 0.0f, sd->time);
  }
  else {
    const float2 uv = triangle_light_uv(
        kg, ls->object, ls->prim, sd->time, sd->P, ls->D);
    shader_setup_from_sample(kg,
                             emission_sd,
                             ls->P,
                             ls->Ng,
                             -ls->D,
                             ls->shader,
                             ls->object,
                             ls->prim,
                             uv.x,
                             uv.y,
                             ls->t,
                             sd->time,
                             false,
                             false);
  }

  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE_LIGHT>(
      kg, state, emission_sd, nullptr, PATH_RAY_VISIBILITY_NONE, PATH_RAY_EMISSION);
  if (emission_sd->runtime_flag & SR_CACHE_MISS) {
    return false;
  }
  *eval = (ls->type == LIGHT_BACKGROUND) ? surface_shader_background(emission_sd) :
                                           surface_shader_emission(emission_sd);
  return true;
}

/* Scalar target for reservoir selection. Visibility is deliberately excluded and evaluated once
 * for the selected sample. Every emitter uses its actual sampled radiance, including texture and
 * procedural emission shaders; a cache miss rejects only this candidate and ordinary NEE remains
 * the per-surface fallback if no trustworthy candidate survives. */
ccl_device_inline bool restir_di_target(KernelGlobals kg,
                                        IntegratorState state,
                                        ccl_private ShaderData *sd,
                                        const ccl_private LightSample *ls,
                                        ccl_private float *target)
{
  Spectrum light_eval;
  const bool is_constant = light_sample_shader_eval_nee_constant(
      kg, ls->shader, ls->prim, ls->type != LIGHT_TRIANGLE, light_eval);
  if (!is_constant && !restir_di_emission_eval(kg, state, sd, ls, &light_eval)) {
    return false;
  }

  BsdfEval bsdf_eval ccl_optional_struct_init;
  float roughness_squared = 0.0f;
  const float bsdf_pdf = surface_shader_bsdf_eval(
      kg, state, sd, ls->D, &bsdf_eval, ls->shader, roughness_squared);
  const float mis_weight = light_sample_mis_weight_nee(kg, ls->pdf, bsdf_pdf);
  const Spectrum unoccluded = fabs(bsdf_eval_sum(&bsdf_eval) * light_eval * ls->eval_fac *
                                    mis_weight);
  *target = reduce_max(unoccluded);
  return *target > 0.0f && isfinite_safe(*target);
}

ccl_device_inline bool restir_di_compatible(const ccl_private ShaderData *sd,
                                            const ccl_global KernelReSTIRDIReservoir *source,
                                            ccl_private float *score)
{
  if (!source->valid || source->emitter_id < 0 || source->object != sd->object) {
    return false;
  }

  packed_normal packed_N;
  packed_N.value = source->normal;
  const float normal_similarity = dot(sd->Ng, packed_N.decode());
  if (normal_similarity < kernel_data.integrator.restir_normal_threshold) {
    return false;
  }

  const float distance = len(sd->P - make_float3(source->P));
  const float scale = max(sd->ray_length, 1.0e-3f);
  const float max_distance = kernel_data.integrator.restir_position_threshold * scale;
  if (distance > max_distance) {
    return false;
  }

  const float position_score = expf(-distance / max(max_distance, 1.0e-6f));
  *score = position_score * powf(max(normal_similarity, 0.0f), 8.0f);
  return *score > 0.0f && isfinite_safe(*score);
}

ccl_device_inline bool restir_di_merge_history(KernelGlobals kg,
                                               IntegratorState state,
                                               ccl_private ShaderData *sd,
                                               const ccl_global KernelReSTIRDIReservoir *source,
                                               const uint source_M_limit,
                                               ccl_private ReSTIRDIWorkingReservoir *reservoir,
                                               ccl_private uint *rng)
{
  float compatibility;
  if (!restir_di_compatible(sd, source, &compatibility)) {
    return false;
  }

  LightSample ls ccl_optional_struct_init;
  const float2 rand = make_float2(source->rand_u, source->rand_v);
  if (!restir_di_reconstruct_light(kg, state, sd, source->emitter_id, rand, &ls)) {
    return false;
  }

  float target;
  if (!restir_di_target(kg, state, sd, &ls, &target)) {
    return false;
  }

  const uint source_M = min(source->M, source_M_limit);
  const float weight = target * source->weight * float(source_M);
  return restir_di_stream(reservoir,
                          source->emitter_id,
                          rand,
                          target,
                          weight,
                          source_M,
                          min(source->age + 1u, 255u),
                          rng);
}

ccl_device_inline uint restir_di_neighbor_index(const uint pixel_index, const int dx, const int dy)
{
  const int delta = int(pixel_index) - kernel_integrator_state.restir_buffer_offset;
  const int x = delta % kernel_integrator_state.restir_buffer_stride;
  const int y = delta / kernel_integrator_state.restir_buffer_stride;
  const int nx = x + dx;
  const int ny = y + dy;
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

/* Generate fresh candidates, then optionally merge temporal and compatibility-guided spatial
 * reservoirs from the immutable previous layer. Fresh RIS is the production default. Reuse is
 * deliberately bounded and opt-in because correlated screen-space samples can add finite-sample
 * bias and are most useful for very low sample-count interactive rendering. */
ccl_device_inline bool restir_di_resample(KernelGlobals kg,
                                          IntegratorState state,
                                          ccl_private ShaderData *sd,
                                          const ccl_private RNGState *rng_state,
                                          ccl_private LightSample *selected_ls,
                                          ccl_private float *selected_weight)
{
  const uint pixel_index = INTEGRATOR_STATE(state, path, render_pixel_index);
  const uint sample = INTEGRATOR_STATE(state, path, sample);
  uint rng = lcg_init(
      hash_uint3(pixel_index, sample, uint(kernel_data.integrator.seed) ^ 0x72737472u));
  ReSTIRDIWorkingReservoir reservoir;
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  const uint bounce = INTEGRATOR_STATE(state, path, bounce);
  const int receiver = light_link_receiver_nee(kg, sd);
  const float3 base_rand = path_state_rng_3D(kg, rng_state, PRNG_LIGHT);

  for (int i = 0; i < kernel_data.integrator.restir_light_candidates; i++) {
    const float3 rand = (i == 0) ? base_rand :
                                   make_float3(lcg_step_float(&rng),
                                               lcg_step_float(&rng),
                                               lcg_step_float(&rng));
    LightSample ls ccl_optional_struct_init;
    if (!light_sample_from_position(
            kg, rand, sd->time, sd->P, sd->N, receiver, sd->runtime_flag, bounce, path_flag, &ls))
    {
      reservoir.M++;
      continue;
    }
    float target;
    if (!restir_di_target(kg, state, sd, &ls, &target)) {
      reservoir.M++;
      continue;
    }
    restir_di_stream(
        &reservoir, ls.emitter_id, make_float2(rand), target, target / ls.pdf, 1u, 0u, &rng);
  }

  const bool primary = bounce == 0 && kernel_integrator_state.restir_previous &&
                       kernel_integrator_state.restir_current &&
                       pixel_index < kernel_integrator_state.restir_reservoir_capacity;
  if (primary) {
    const uint history_limit = uint(kernel_data.integrator.restir_history_length);
    if (history_limit > 0) {
      restir_di_merge_history(kg,
                              state,
                              sd,
                              &kernel_integrator_state.restir_previous[pixel_index],
                              history_limit,
                              &reservoir,
                              &rng);
    }

    uint used_indices[4] = {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
    const int radius = kernel_data.integrator.restir_spatial_radius;
    for (int neighbor = 0; neighbor < kernel_data.integrator.restir_spatial_neighbors; neighbor++)
    {
      uint chosen = UINT_MAX;
      float score_sum = 0.0f;
      for (int candidate = 0; candidate < 8; candidate++) {
        const float angle = M_2PI_F * lcg_step_float(&rng);
        const float distance = sqrtf(lcg_step_float(&rng)) * float(radius);
        const int dx = float_to_int(cosf(angle) * distance);
        const int dy = float_to_int(sinf(angle) * distance);
        const uint index = restir_di_neighbor_index(pixel_index, dx, dy);
        if (index == UINT_MAX || index == pixel_index ||
            index >= kernel_integrator_state.restir_reservoir_capacity)
        {
          continue;
        }
        bool duplicate = false;
        for (int j = 0; j < neighbor; j++) {
          duplicate |= used_indices[j] == index;
        }
        if (duplicate) {
          continue;
        }
        float score;
        if (!restir_di_compatible(sd, &kernel_integrator_state.restir_previous[index], &score)) {
          continue;
        }
        score_sum += score;
        if (chosen == UINT_MAX || lcg_step_float(&rng) * score_sum < score) {
          chosen = index;
        }
      }
      if (chosen != UINT_MAX) {
        used_indices[neighbor] = chosen;
        const uint spatial_limit = max(history_limit,
                                       uint(kernel_data.integrator.restir_light_candidates));
        restir_di_merge_history(kg,
                                state,
                                sd,
                                &kernel_integrator_state.restir_previous[chosen],
                                spatial_limit,
                                &reservoir,
                                &rng);
      }
    }
  }

  if (reservoir.emitter_id < 0 || reservoir.M == 0 || !(reservoir.selected_target > 0.0f)) {
    return false;
  }
  if (!restir_di_reconstruct_light(
          kg, state, sd, reservoir.emitter_id, reservoir.selected_rand, selected_ls))
  {
    return false;
  }

  *selected_weight = reservoir.weight_sum / (float(reservoir.M) * reservoir.selected_target);
  if (!(*selected_weight > 0.0f) || !isfinite_safe(*selected_weight)) {
    return false;
  }

  if (primary) {
    ccl_global KernelReSTIRDIReservoir *output =
        &kernel_integrator_state.restir_current[pixel_index];
    output->P = sd->P;
    output->weight = *selected_weight;
    output->normal = packed_normal(sd->Ng).value;
    output->M = min(reservoir.M,
                    max(uint(kernel_data.integrator.restir_history_length),
                        uint(kernel_data.integrator.restir_light_candidates)));
    output->emitter_id = reservoir.emitter_id;
    output->object = sd->object;
    output->rand_u = reservoir.selected_rand.x;
    output->rand_v = reservoir.selected_rand.y;
    output->age = reservoir.age;
    output->valid = 1u;
  }
  return true;
}

#endif /* __KERNEL_METAL__ */

CCL_NAMESPACE_END
