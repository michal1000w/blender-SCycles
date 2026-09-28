/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "kernel/light/coherent_util.h"
#include "kernel/light/sample.h"
#include "kernel/bvh/bvh.h"

CCL_NAMESPACE_BEGIN

/* Direct scalar source coherence, not coherent multipath transport. The field
 * scattering amplitudes have magnitude sqrt(BSDF * incident irradiance) and
 * zero additional material phase. Each RGB channel is a separate scalar field;
 * spectral path color signs are retained by the original radiometric carrier.
 * Gaussian source coherence is positive semidefinite, so the group intensity
 * is nonnegative. Reweighting all sampled members by I_coherent / I_incoherent
 * preserves the existing light-selection PDF and all light-pass bookkeeping.
 * Opaque visibility and constant point emitters are checked on the host.
 * No-inline keeps the bounded temporary arrays out of ordinary-path registers. */
#ifdef __KERNEL_METAL__
/* ccl_device_noinline intentionally has no noinline attribute on Apple Metal.
 * Keep this large field/BSDF query callable, as for the other grating workspaces. */
ccl_device __attribute__((noinline))
#else
ccl_device_noinline
#endif
Spectrum coherent_direct_light_scale(
    KernelGlobals kg, IntegratorState state, ccl_private ShaderData *sd,
    const ccl_private LightSample *selected)
{
  const ccl_global KernelLight *reference = &kernel_data_fetch(lights, selected->prim);
  if (reference->coherence_group == 0 || reference->coherence_length == 0.0f) {
    return one_spectrum();
  }
  constexpr int capacity = 16;
  Spectrum intensities[capacity];
  float3 positions[capacity];
  float phases[capacity];
  int count = 0;
  Spectrum incoherent = zero_spectrum();
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  const int bounce = INTEGRATOR_STATE(state, path, bounce);
  const int receiver = light_link_receiver_nee(kg, sd);

  for (int lamp = 0; lamp < kernel_data.integrator.num_lights; ++lamp) {
    const ccl_global KernelLight *light = &kernel_data_fetch(lights, lamp);
    if (light->coherence_group != reference->coherence_group ||
        light->type != LIGHT_POINT || light_select_reached_max_bounces(kg, lamp, bounce) ||
        !light_link_object_match(kg, receiver, light->object_id))
    {
      continue;
    }
    /* The host rejects larger groups, rather than silently dropping fields. */
    kernel_assert(count < capacity);
    if (count == capacity) {
      return one_spectrum();
    }
    LightSample sample ccl_optional_struct_init;
    if (!light_sample<false>(kg, lamp, make_float2(0.5f, 0.5f), sd->P, sd->N,
                             sd->runtime_flag, path_flag, &sample))
    {
      continue;
    }
    Ray ray ccl_optional_struct_init;
    light_sample_to_surface_shadow_ray(kg, sd, &sample, &ray);
    if (ray.self.object != OBJECT_NONE) {
      ray.P = integrate_surface_ray_offset(kg, sd, ray.P, ray.D);
    }
    Intersection isect ccl_optional_struct_init;
    uint visibility = PATH_RAY_VISIBILITY_SHADOW_OPAQUE;
#ifdef __SHADOW_CATCHER__
    visibility = SHADOW_CATCHER_PATH_VISIBILITY(path_flag, visibility);
#endif
    if (scene_intersect(kg, &ray, visibility, &isect)) {
      continue;
    }
    Spectrum emission;
    if (!light_sample_shader_eval_nee_constant(kg, sample.shader, lamp, true, emission)) {
      continue; /* Host validation rejects this configuration. */
    }
    BsdfEval bsdf;
    float roughness;
    surface_shader_bsdf_eval(kg, state, sd, sample.D, &bsdf, sample.shader, roughness);
    if (sd->runtime_flag & SR_CACHE_MISS) {
      return zero_spectrum();
    }
    const Spectrum intensity = fabs(bsdf_eval_sum(&bsdf) * emission *
                                    (sample.eval_fac / sample.pdf));
    intensities[count] = intensity;
    positions[count] = light->co;
    phases[count] = light->coherence_phase;
    incoherent += intensity;
    ++count;
  }
  Spectrum field_intensity = incoherent;
  for (int a = 0; a < count; ++a) {
    for (int b = a + 1; b < count; ++b) {
      const float difference = coherent_source_range_difference(sd->P, positions[a], positions[b]);
      const float gamma = coherent_gaussian_mutual_coherence(difference,
                                                             reference->coherence_length);
      const float cycles = coherent_phase_cycles_split(
          make_float2(difference, 0.0f),
          make_float2(reference->coherence_wavelength, reference->coherence_wavelength_low),
          (phases[a] - phases[b]) * M_1_2PI_F);
      field_intensity += 2.0f * gamma * cosf(M_2PI_F * cycles) *
                  sqrt(intensities[a] * intensities[b]);
    }
  }
  /* Only roundoff can make the PSD quadratic form slightly negative. */
  field_intensity = max(field_intensity, zero_spectrum());
  return safe_divide_color(field_intensity, incoherent);
}

CCL_NAMESPACE_END
