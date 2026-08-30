/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Progressive caustic photon mapping for the GPU integrator. Photon paths use the same scene
 * intersection, shader evaluation and closure sampling code as camera paths. The map stores only
 * the first non-specular surface following a sufficiently sharp glossy/transmission chain, so it
 * replaces the transport class which unidirectional path tracing samples particularly poorly. */

#include "kernel/bvh/bvh.h"
#include "kernel/geom/shader_data.h"
#include "kernel/integrator/path_state.h"
#include "kernel/integrator/surface_shader.h"
#include "kernel/light/distribution.h"
#include "kernel/light/sample.h"
#include "kernel/sample/lcg.h"
#include "kernel/sample/mapping.h"
#include "util/atomic.h"

CCL_NAMESPACE_BEGIN

/* Tag volume receivers without growing the compact photon record. Cycles object indices are
 * non-negative, leaving the sign bit available to distinguish the 3D volume estimator from the
 * surface estimator. */
#define PHOTON_VOLUME_RECEIVER_BIT 0x80000000u

ccl_device_inline int photon_volume_receiver_object(const int object)
{
  return int(uint(object) | PHOTON_VOLUME_RECEIVER_BIT);
}

/* Keep the photon record at 48 bytes. Half precision is more than sufficient for shutter time,
 * while 15 bits locate a wavelength to substantially better than 0.1 nm. */
ccl_device_inline uint photon_pack_time_wavelength(const float time,
                                                   const float wavelength_rand,
                                                   const bool spectral)
{
  const uint packed_time = uint(clamp(time, 0.0f, 1.0f) * 65535.0f + 0.5f);
  const uint packed_wavelength = uint(clamp(wavelength_rand, 0.0f, 1.0f) * 32767.0f + 0.5f);
  return packed_time | (packed_wavelength << 16) | (spectral ? 0x80000000u : 0u);
}

ccl_device_inline float photon_unpack_time(const uint packed)
{
  return float(packed & 0xffffu) * (1.0f / 65535.0f);
}

ccl_device_inline float photon_unpack_wavelength_rand(const uint packed)
{
  return float((packed >> 16) & 0x7fffu) * (1.0f / 32767.0f);
}

ccl_device_inline bool photon_is_spectral(const uint packed)
{
  return (packed & 0x80000000u) != 0u;
}

#ifdef __SPECTRAL__
/* Epanechnikov reconstruction over wavelength. Its support is narrow enough to preserve caustic
 * separation but wide enough for useful photon counts in an interactive map. The normalization
 * is in the same [0, 1] wavelength measure as sample_wavelength()'s returned probability. */
ccl_device_inline float photon_spectral_kernel(const float photon_wavelength,
                                               const float camera_wavelength)
{
  constexpr float bandwidth = 0.02f;
  constexpr float wavelength_range = WAVELENGTH_CIE_MAX - WAVELENGTH_CIE_MIN;
  const float x = (photon_wavelength - camera_wavelength) / bandwidth;
  return (fabsf(x) < 1.0f) ? 0.75f * (1.0f - sqr(x)) * wavelength_range / bandwidth : 0.0f;
}
#endif

ccl_device_inline uint photon_hash_cell(const int3 cell, const int time_bin, const uint hash_size)
{
  const uint x = uint(cell.x) * 0x8da6b343u;
  const uint y = uint(cell.y) * 0xd8163841u;
  const uint z = uint(cell.z) * 0xcb1ab31fu;
  const uint t = uint(time_bin) * 0x165667b1u;
  return hash_uint(x ^ y ^ z ^ t) & (hash_size - 1u);
}

ccl_device_inline int photon_time_bin(const float time)
{
  return min(float_to_int(time * float(kernel_data.integrator.photon_time_bins)),
             kernel_data.integrator.photon_time_bins - 1);
}

ccl_device_inline int3 photon_cell(const float3 P, const float radius)
{
  return make_int3(float_to_int(floorf(P.x / radius)),
                   float_to_int(floorf(P.y / radius)),
                   float_to_int(floorf(P.z / radius)));
}

ccl_device_inline void photon_state_init(IntegratorState state, const uint seed, const uint sample)
{
  INTEGRATOR_STATE_WRITE(state, path, sample) = sample;
  INTEGRATOR_STATE_WRITE(state, path, bounce) = 0;
  INTEGRATOR_STATE_WRITE(state, path, transparent_bounce) = 0;
  INTEGRATOR_STATE_WRITE(state, path, diffuse_bounce) = 0;
  INTEGRATOR_STATE_WRITE(state, path, glossy_bounce) = 0;
  INTEGRATOR_STATE_WRITE(state, path, transmission_bounce) = 0;
  INTEGRATOR_STATE_WRITE(state, path, volume_bounce) = 0;
  /* Photon transport needs real BSDF closures. PATH_RAY_EMISSION is reserved for evaluating
   * emitters and deliberately suppresses closure allocation. Classify the first light-path
   * segment as glossy so caustics disabling for camera paths does not remove the very closures
   * which this integrator is intended to sample. */
  INTEGRATOR_STATE_WRITE(state, path, flag) = PATH_RAY_MIS_SKIP;
  INTEGRATOR_STATE_WRITE(state, path, visibility) = PATH_RAY_VISIBILITY_GLOSSY;
  INTEGRATOR_STATE_WRITE(state, path, rng_pixel) = seed;
  INTEGRATOR_STATE_WRITE(state, path, rng_offset) = 0;
  INTEGRATOR_STATE_WRITE(state, path, throughput) = one_spectrum();
  INTEGRATOR_STATE_WRITE(state, path, min_ray_pdf) = FLT_MAX;
#ifdef __PATH_GUIDING__
  if (kernel_data.kernel_features & KERNEL_FEATURE_PATH_GUIDING) {
    INTEGRATOR_STATE_WRITE(state, guiding, use_surface_guiding) = false;
    INTEGRATOR_STATE_WRITE(state, guiding, path_segment) = nullptr;
  }
#endif
}

ccl_device_inline Spectrum photon_eval_triangle_emission(KernelGlobals kg,
                                                         IntegratorState state,
                                                         const float3 P,
                                                         const float3 Ng,
                                                         const float3 D,
                                                         const int shader,
                                                         const int object,
                                                         const int prim,
                                                         const float u,
                                                         const float v,
                                                         const float time)
{
  Spectrum eval = zero_spectrum();
  if (surface_shader_constant_emission(kg, shader, &eval)) {
    return eval;
  }

  ShaderDataTinyStorage storage;
  ccl_private ShaderData *sd = AS_SHADER_DATA(&storage);
  shader_setup_from_sample(
      kg, sd, P, Ng, -D, shader, object, prim, u, v, 0.0f, time, false, false);
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE_LIGHT>(
      kg, state, sd, nullptr, PATH_RAY_VISIBILITY_NONE, PATH_RAY_EMISSION);
  return (sd->runtime_flag & SR_CACHE_MISS) ? zero_spectrum() : surface_shader_emission(sd);
}

/* Sample a point spotlight proportional to its angular falloff. Sampling the full sphere and
 * rejecting rays outside the cone is catastrophically inefficient for narrow spots (a 24 degree
 * spot, for example, discards about 99% of its photon paths). Smoothstep integrates to one half
 * over the blend interval, giving the closed-form normalization below. Rejection sampling within
 * the outer cone then has at least 50% acceptance for every valid spot blend. */
ccl_device_inline float3 photon_sample_point_spot_direction(KernelGlobals kg,
                                                            ccl_global const KernelLight *klight,
                                                            ccl_private uint *rng,
                                                            ccl_private float *attenuation,
                                                            ccl_private float *pdf)
{
  const float one_minus_cos_angle = 1.0f - klight->spot.cos_half_spot_angle;
  if (!(one_minus_cos_angle > 0.0f)) {
    *attenuation = 1.0f;
    *pdf = 1.0f;
    return klight->spot.dir;
  }

  float3 D;
  do {
    float unused_cosine;
    float unused_pdf;
    D = sample_uniform_cone(klight->spot.dir,
                            one_minus_cos_angle,
                            make_float2(lcg_step_float(rng), lcg_step_float(rng)),
                            &unused_cosine,
                            &unused_pdf);
    *attenuation = spot_light_attenuation(&klight->spot, spot_light_to_local(kg, klight, D));
  } while (lcg_step_float(rng) >= *attenuation);

  const float blend_width = isfinite_safe(klight->spot.spot_smooth) ?
                                1.0f / klight->spot.spot_smooth :
                                0.0f;
  const float integrated_cosine = max(one_minus_cos_angle - 0.5f * blend_width, 1.0e-20f);
  *pdf = *attenuation / (M_2PI_F * integrated_cosine);
  return D;
}

/* Aim delta point-light photons at the aggregate sharp-caster bounds most of the time. The
 * remaining uniform-sphere component preserves support for conservatively missed or procedural
 * casters, and evaluating the full mixture PDF keeps the estimator unbiased. */
ccl_device_inline float3 photon_sample_point_direction(KernelGlobals kg,
                                                       const float3 light_P,
                                                       ccl_private uint *rng,
                                                       ccl_private float *pdf)
{
  const float3 target_center = make_float3(kernel_data.integrator.photon_target);
  const float target_radius = kernel_data.integrator.photon_target.w;
  const float3 to_target = target_center - light_P;
  const float distance2 = len_squared(to_target);
  constexpr float target_probability = 0.9f;
  constexpr float uniform_pdf = M_1_2PI_F * 0.5f;

  if (!(target_radius > 0.0f) || distance2 <= sqr(target_radius)) {
    *pdf = uniform_pdf;
    return sample_uniform_sphere(make_float2(lcg_step_float(rng), lcg_step_float(rng)));
  }

  const float inv_distance = inversesqrtf(distance2);
  const float3 target_direction = to_target * inv_distance;
  const float cos_half_angle = safe_sqrtf(1.0f - sqr(target_radius) / distance2);
  const float one_minus_cos = 1.0f - cos_half_angle;
  const float target_pdf = 1.0f / max(M_2PI_F * one_minus_cos, 1.0e-20f);

  float3 D;
  if (lcg_step_float(rng) < target_probability) {
    float unused_cosine;
    float unused_pdf;
    D = sample_uniform_cone(target_direction,
                            one_minus_cos,
                            make_float2(lcg_step_float(rng), lcg_step_float(rng)),
                            &unused_cosine,
                            &unused_pdf);
  }
  else {
    D = sample_uniform_sphere(make_float2(lcg_step_float(rng), lcg_step_float(rng)));
  }

  *pdf = (1.0f - target_probability) * uniform_pdf;
  if (dot(D, target_direction) >= cos_half_angle) {
    *pdf += target_probability * target_pdf;
  }
  return D;
}

/* Sample an emitted ray and its total flux divided by the emitter-selection and ray PDFs. */
ccl_device_inline bool photon_sample_emitter(KernelGlobals kg,
                                             IntegratorState state,
                                             ccl_private uint *rng,
                                             const float time,
                                             ccl_private Ray *ray,
                                             ccl_private Spectrum *flux,
                                             ccl_private int *emitter_object,
                                             ccl_private int *light_group = nullptr,
                                             ccl_private float *emission_pdf = nullptr,
                                             ccl_private float *direct_pdf = nullptr,
                                             ccl_private float *emission_cosine_out = nullptr,
                                             ccl_private bool *is_delta = nullptr,
                                             ccl_private bool *is_finite = nullptr,
                                             ccl_private uint *emitter_shader_flags = nullptr,
                                             ccl_private float *emitter_max_bounces = nullptr)
{
  /* MetalRT consumes the self-intersection payload unconditionally. Keep it initialized for
   * analytic emitters, and replace it below for emissive geometry. */
  ray->self.prim = PRIM_NONE;
  ray->self.object = OBJECT_NONE;
  ray->self.light_prim = PRIM_NONE;
  ray->self.light_object = OBJECT_NONE;

  if (kernel_data.integrator.num_distribution == 0) {
    return false;
  }

  const int emitter = light_distribution_sample(kg, lcg_step_float(rng));
  const ccl_global KernelLightDistribution *distribution = &kernel_data_fetch(light_distribution,
                                                                              emitter);
  const int prim_or_lamp = distribution->prim;
  *emitter_object = distribution->object_id;
  if (light_group) {
    *light_group = LIGHTGROUP_NONE;
  }
  if (emission_pdf) {
    *emission_pdf = 0.0f;
    *direct_pdf = 0.0f;
    *emission_cosine_out = 1.0f;
    *is_delta = false;
    *is_finite = true;
  }
  if (emitter_shader_flags) {
    *emitter_shader_flags = 0u;
  }
  if (emitter_max_bounces) {
    *emitter_max_bounces = FLT_MAX;
  }

  if (prim_or_lamp >= 0) {
    float3 V[3];
    triangle_world_space_vertices(kg, *emitter_object, prim_or_lamp, time, V);
    float3 Ng = cross(V[1] - V[0], V[2] - V[0]);
    const float twice_area = len(Ng);
    if (!(twice_area > 0.0f)) {
      return false;
    }
    Ng /= twice_area;
    if (kernel_data_fetch(object_flag, *emitter_object) & SD_OBJECT_NEGATIVE_SCALE) {
      Ng = -Ng;
    }

    float u = lcg_step_float(rng);
    float v = lcg_step_float(rng);
    if (v > u) {
      u *= 0.5f;
      v -= u;
    }
    else {
      v *= 0.5f;
      u -= v;
    }
    ray->P = (1.0f - u - v) * V[0] + u * V[1] + v * V[2];
    ray->self.prim = prim_or_lamp;
    ray->self.object = *emitter_object;

    const int shader = kernel_data_fetch(tri_shader, prim_or_lamp);
    if (emitter_shader_flags) {
      *emitter_shader_flags = uint(shader);
    }
    if (light_group) {
      *light_group = object_lightgroup(kg, *emitter_object);
    }
    const int shader_flags = kernel_data_fetch(shaders, shader & SHADER_MASK).flags;
    const bool front = (shader_flags & SD_MIS_FRONT) != 0;
    const bool back = (shader_flags & SD_MIS_BACK) != 0;
    if (!front && !back) {
      return false;
    }
    float side_pdf = 1.0f;
    float3 emission_N = front ? Ng : -Ng;
    if (front && back) {
      side_pdf = 0.5f;
      emission_N = (lcg_step_float(rng) < 0.5f) ? Ng : -Ng;
    }
    float direction_pdf;
    sample_cos_hemisphere(emission_N,
                          make_float2(lcg_step_float(rng), lcg_step_float(rng)),
                          &ray->D,
                          &direction_pdf);
    if (!(direction_pdf > 0.0f)) {
      return false;
    }
    if (emission_pdf) {
      *emission_pdf = kernel_data.integrator.distribution_pdf_triangles * side_pdf * direction_pdf;
      *direct_pdf = kernel_data.integrator.distribution_pdf_triangles;
      *emission_cosine_out = direction_pdf * M_PI_F;
    }

    const Spectrum Le = photon_eval_triangle_emission(
        kg, state, ray->P, Ng, ray->D, shader, *emitter_object, prim_or_lamp, u, v, time);
    /* The flat CDF chooses triangles proportional to area, hence conditional position PDF and
     * selection PDF reduce to distribution_pdf_triangles. Cosine sampling cancels the emission
     * cosine, leaving pi. */
    *flux = Le *
            (M_PI_F / max(kernel_data.integrator.distribution_pdf_triangles * side_pdf, 1.0e-20f));
  }
  else {
    const int lamp = ~prim_or_lamp;
    const ccl_global KernelLight *klight = &kernel_data_fetch(lights, lamp);
    if (emitter_shader_flags) {
      *emitter_shader_flags = uint(klight->shader_id);
    }
    if (emitter_max_bounces) {
      *emitter_max_bounces = klight->max_bounces;
    }
    if (light_group) {
      *light_group = object_lightgroup(kg, klight->object_id);
    }
    const float select_pdf = max(kernel_data.integrator.distribution_pdf_lights, 1.0e-20f);
    const LightType type = (LightType)klight->type;
    float ray_pdf = 0.0f;
    float eval_fac = 1.0f;
    float emission_cosine = 1.0f;
    float infinite_direction_pdf = 0.0f;

    if (type == LIGHT_AREA) {
      const float2 rpos = make_float2(lcg_step_float(rng), lcg_step_float(rng));
      const bool ellipse = area_light_is_ellipse(&klight->area);
      ray->P = klight->co + (ellipse ?
                                 ellipse_sample(klight->area.axis_u * klight->area.len_u * 0.5f,
                                                klight->area.axis_v * klight->area.len_v * 0.5f,
                                                rpos) :
                                 rectangle_sample(klight->area.axis_u * klight->area.len_u * 0.5f,
                                                  klight->area.axis_v * klight->area.len_v * 0.5f,
                                                  rpos));
      const float area = ellipse ? M_PI_F * klight->area.len_u * klight->area.len_v * 0.25f :
                                   klight->area.len_u * klight->area.len_v;
      float direction_pdf;
      sample_cos_hemisphere(klight->area.dir,
                            make_float2(lcg_step_float(rng), lcg_step_float(rng)),
                            &ray->D,
                            &direction_pdf);
      emission_cosine = direction_pdf * M_PI_F;
      ray_pdf = direction_pdf / max(area, 1.0e-20f);
      eval_fac = M_1_PI_F * fabsf(klight->area.invarea);
      if (klight->area.normalize_spread > 0.0f) {
        eval_fac *= area_light_spread_attenuation(
            ray->D, klight->area.dir, klight->area.tan_half_spread, klight->area.normalize_spread);
      }
    }
    else if (type == LIGHT_POINT || type == LIGHT_SPOT) {
      /* Cycles' non-sphere finite-radius point light is a receiver-facing disk impostor: its
       * position and normal depend on the point from which it is sampled. There is consequently
       * no single emitter surface (nor a reciprocal emission measure) from which a forward path
       * can be launched. Keep those lights on Cycles' regular camera-path estimator. A physical
       * sphere light has a well-defined surface and participates in every BDPT strategy. */
      if (!klight->spot.is_sphere && klight->spot.radius > 0.0f) {
        return false;
      }

      eval_fac = klight->spot.eval_fac;
      if (klight->spot.is_sphere) {
        const float3 N = sample_uniform_sphere(
            make_float2(lcg_step_float(rng), lcg_step_float(rng)));
        ray->P = klight->co + N * klight->spot.radius;
        float direction_pdf;
        sample_cos_hemisphere(
            N, make_float2(lcg_step_float(rng), lcg_step_float(rng)), &ray->D, &direction_pdf);
        emission_cosine = direction_pdf * M_PI_F;
        const float area = 4.0f * M_PI_F * sqr(klight->spot.radius);
        ray_pdf = direction_pdf / max(area, 1.0e-20f);
      }
      else {
        ray->P = klight->co;
        if (type == LIGHT_SPOT) {
          float spot_attenuation;
          ray->D = photon_sample_point_spot_direction(
              kg, klight, rng, &spot_attenuation, &ray_pdf);
          eval_fac *= spot_attenuation;
        }
        else {
          ray->D = photon_sample_point_direction(kg, ray->P, rng, &ray_pdf);
        }
      }

      if (type == LIGHT_SPOT && klight->spot.is_sphere) {
        const float3 local_ray = spot_light_to_local(kg, klight, ray->D);
        eval_fac *= spot_light_attenuation(&klight->spot, local_ray);
      }
    }
    else {
      /* Infinite emitters are launched from a disk enclosing the scene. This is the standard
       * finite-scene photon mapping construction and preserves the directional light PDF. */
      const float3 scene_center = make_float3(kernel_data.integrator.photon_scene);
      const float scene_radius = kernel_data.integrator.photon_scene.w;
      const float3 target_center = make_float3(kernel_data.integrator.photon_target);
      const float target_radius = min(kernel_data.integrator.photon_target.w, scene_radius);
      float direction_pdf;
      if (type == LIGHT_SUN) {
        if (klight->sun.angle == 0.0f) {
          ray->D = klight->co;
          direction_pdf = 1.0f;
        }
        else {
          float unused;
          ray->D = sample_uniform_cone(klight->co,
                                       klight->sun.one_minus_cosangle,
                                       make_float2(lcg_step_float(rng), lcg_step_float(rng)),
                                       &unused,
                                       &direction_pdf);
        }
        eval_fac = klight->sun.eval_fac;
      }
      else {
        ray->D = -background_light_sample(kg,
                                          scene_center,
                                          make_float2(lcg_step_float(rng), lcg_step_float(rng)),
                                          &direction_pdf);
      }
      infinite_direction_pdf = direction_pdf;

      /* Most light paths are aimed through the aggregate bounds of likely sharp caustic casters.
       * A full-scene component preserves support for procedural/OSL shaders and any conservative
       * host-side classification miss. Evaluate the complete mixture PDF to remain unbiased. */
      const float target_probability = (target_radius < 0.999f * scene_radius ||
                                        len_squared(target_center - scene_center) > 1.0e-10f) ?
                                           0.9f :
                                           0.0f;
      const bool sample_target = lcg_step_float(rng) < target_probability;
      const float3 disk_center = sample_target ? target_center : scene_center;
      const float disk_radius = sample_target ? target_radius : scene_radius;
      float3 T, B;
      make_orthonormals(ray->D, &T, &B);
      const float2 disk = sample_uniform_disk(
          make_float2(lcg_step_float(rng), lcg_step_float(rng)));
      const float3 cross_section = disk_center + disk_radius * (disk.x * T + disk.y * B);
      ray->P = cross_section - ray->D * (2.0f * scene_radius);

      const float3 scene_delta = cross_section - scene_center;
      const float3 target_delta = cross_section - target_center;
      const float scene_distance2 = len_squared(scene_delta) - sqr(dot(scene_delta, ray->D));
      const float target_distance2 = len_squared(target_delta) - sqr(dot(target_delta, ray->D));
      float position_pdf = 0.0f;
      if (scene_distance2 <= sqr(scene_radius)) {
        position_pdf += (1.0f - target_probability) / max(M_PI_F * sqr(scene_radius), 1.0e-20f);
      }
      if (target_probability > 0.0f && target_distance2 <= sqr(target_radius)) {
        position_pdf += target_probability / max(M_PI_F * sqr(target_radius), 1.0e-20f);
      }
      ray_pdf = direction_pdf * position_pdf;
    }

    if (!(ray_pdf > 0.0f) || !(eval_fac > 0.0f)) {
      return false;
    }

    if (emission_pdf) {
      *emission_pdf = select_pdf * ray_pdf;
      *emission_cosine_out = emission_cosine;
      if (type == LIGHT_AREA) {
        const bool ellipse = area_light_is_ellipse(&klight->area);
        const float area = ellipse ? M_PI_F * klight->area.len_u * klight->area.len_v * 0.25f :
                                     klight->area.len_u * klight->area.len_v;
        *direct_pdf = select_pdf / max(area, 1.0e-20f);
      }
      else if ((type == LIGHT_POINT || type == LIGHT_SPOT) && klight->spot.is_sphere) {
        const float area = 4.0f * M_PI_F * sqr(klight->spot.radius);
        *direct_pdf = select_pdf / max(area, 1.0e-20f);
      }
      else if (type == LIGHT_POINT || type == LIGHT_SPOT) {
        *direct_pdf = select_pdf;
        *is_delta = true;
      }
      else {
        /* Infinite emission additionally samples a launch disk. Direct lighting samples only the
         * direction, so its density excludes that position factor. SmallVCM deliberately treats
         * this directional density as an area density in the recursive endpoint formula. */
        *direct_pdf = select_pdf * infinite_direction_pdf;
        *is_delta = (type == LIGHT_SUN && klight->sun.angle == 0.0f);
        *is_finite = false;
      }
    }

    Spectrum Le;
    if (type == LIGHT_BACKGROUND) {
      INTEGRATOR_STATE_WRITE(state, ray, P) = ray->P;
      /* Background shaders are parameterized by the outward camera-ray direction, opposite to
       * the photon which travels inward from the enclosing launch disk. */
      INTEGRATOR_STATE_WRITE(state, ray, D) = -ray->D;
      INTEGRATOR_STATE_WRITE(state, ray, dD) = 0.0f;
      INTEGRATOR_STATE_WRITE(state, ray, time) = time;
      ShaderEvalResult result;
      Le = integrator_eval_background_shader(kg, state, nullptr, result);
      if (result != SHADER_EVAL_OK) {
        return false;
      }
    }
    else {
      const ShaderEvalResult result = light_sample_shader_eval_forward(
          kg, state, lamp, ray->P, ray->D, 0.0f, time, Le);
      if (result != SHADER_EVAL_OK) {
        return false;
      }
    }
    /* `eval_fac * Le` is radiance for finite-area lights. Photon power integrates radiance
     * against the projected-area cosine; retain that measure when dividing by the sampled
     * position/direction density. Delta and infinite lights use a unit projected cosine. */
    *flux = Le * (eval_fac * emission_cosine / (select_pdf * ray_pdf));
  }

  ray->tmin = 0.0f;
  ray->tmax = FLT_MAX;
  ray->time = time;
#ifdef __RAY_DIFFERENTIALS__
  ray->dP = differential_zero_compact();
  ray->dD = differential_zero_compact();
#endif
  return isfinite_safe(*flux) && reduce_max(*flux) > 0.0f;
}

ccl_device_inline void photon_store(KernelGlobals kg,
                                    const float3 P,
                                    const float3 N,
                                    const float3 D,
                                    const Spectrum power,
                                    const int emitter_object,
                                    const int receiver_object,
                                    const float time,
                                    const float wavelength_rand,
                                    const bool spectral,
                                    const bool volume)
{
  const uint slot = atomic_fetch_and_add_uint32(kernel_integrator_state.photon_stored, 1);
  if (slot >= kernel_integrator_state.photon_capacity) {
    return;
  }

  ccl_global KernelPhoton *photon = &kernel_integrator_state.photons[slot];
  photon->P = P;
  photon->power = power;
  photon->emitter_object = emitter_object;
  photon->direction = packed_normal(D).value;
  photon->normal = volume ? 0u : packed_normal(N).value;
  photon->time_wavelength = photon_pack_time_wavelength(time, wavelength_rand, spectral);
  photon->receiver_object = volume ? photon_volume_receiver_object(receiver_object) :
                                     receiver_object;

  const int3 cell = photon_cell(P, kernel_integrator_state.photon_radius);
  const uint bucket = photon_hash_cell(
      cell, photon_time_bin(time), kernel_integrator_state.photon_hash_size);
  photon->next = atomic_exchange_uint32(&kernel_integrator_state.photon_hash[bucket], slot + 1u);
}

ccl_device void integrator_photon_emit(KernelGlobals kg,
                                       IntegratorState state,
                                       const uint photon_index,
                                       const uint iteration)
{
  uint rng = lcg_init(
      hash_uint3(photon_index, iteration, uint(kernel_data.integrator.seed) ^ 0x70686f74u));
  photon_state_init(state, rng, iteration);

#ifdef __SPECTRAL__
  /* shader_setup_wavelength() derives this same sample from the immutable photon path state.
   * Keep it explicitly for the photon record even when the receiver shader itself does not
   * require a wavelength. */
  const float photon_wavelength_rand = path_rng_1D(
      kg, rng, iteration, PRNG_BOUNCE_NUM + PRNG_WAVELENGTH);
#else
  const float photon_wavelength_rand = 0.0f;
#endif

  Ray ray ccl_optional_struct_init;
  Spectrum throughput;
  int emitter_object = OBJECT_NONE;
  const float time = lcg_step_float(&rng);
  if (!photon_sample_emitter(kg, state, &rng, time, &ray, &throughput, &emitter_object)) {
    return;
  }
  throughput /= float(kernel_integrator_state.photon_capacity);

#ifdef __VOLUME__
  if (kernel_data.integrator.use_volumes) {
    /* Photon emitters may be inside object or world volumes. Initialize the same stack used by
     * camera paths, but with light-path visibility so holdout/camera visibility does not change
     * photon transport. */
    INTEGRATOR_STATE_WRITE(state, ray, P) = ray.P;
    INTEGRATOR_STATE_WRITE(state, ray, D) = ray.D;
    INTEGRATOR_STATE_WRITE(state, ray, tmin) = ray.tmin;
    INTEGRATOR_STATE_WRITE(state, ray, tmax) = ray.tmax;
    INTEGRATOR_STATE_WRITE(state, ray, time) = ray.time;
    integrator_volume_stack_init(kg, state, PATH_RAY_VISIBILITY_GLOSSY);
  }
#endif

  bool had_specular = false;
  for (int bounce = 0; bounce < kernel_data.integrator.photon_max_bounces; bounce++) {
    Intersection isect;
    const PathRayVisibility path_visibility = path_state_ray_visibility(state);
    const bool hit_surface = scene_intersect(kg, &ray, path_visibility, &isect);

#ifdef __VOLUME__
    if (kernel_data.integrator.use_volumes && !integrator_state_volume_stack_is_empty(kg, state)) {
      ray.tmax = hit_surface ? isect.t : FLT_MAX;
      INTEGRATOR_STATE_WRITE(state, path, throughput) = throughput;
      float3 scatter_P;
      int receiver_object = OBJECT_NONE;
      const PhotonVolumeSampleEvent volume_event = photon_volume_sample_segment(
          kg, state, &ray, &throughput, &scatter_P, &receiver_object);
      if (volume_event == PHOTON_VOLUME_CACHE_MISS) {
        return;
      }
      if (volume_event == PHOTON_VOLUME_SCATTERED) {
        if (had_specular && receiver_object != OBJECT_NONE && isfinite_safe(throughput)) {
          photon_store(kg,
                       scatter_P,
                       zero_float3(),
                       ray.D,
                       throughput,
                       emitter_object,
                       receiver_object,
                       time,
                       photon_wavelength_rand,
                       (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SPECTRAL) != 0u,
                       true);
        }
        /* The first broad event ends the caustic transport class, whether or not it was eligible
         * for storage. Ordinary volume multiple scattering remains in the path tracer. */
        return;
      }
    }
#endif

    if (!hit_surface) {
      return;
    }
    ShaderData sd;
    shader_setup_from_ray(kg, &sd, &ray, &isect);
#ifdef __SPECTRAL__
    shader_setup_wavelength(kg, &sd, state);
#endif
    surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE>(
        kg, state, &sd, nullptr, path_visibility, INTEGRATOR_STATE(state, path, flag));
    if (sd.runtime_flag & SR_CACHE_MISS) {
      return;
    }

#ifdef __VOLUME__
    /* A pure volume boundary has no BSDF closure. Camera paths pass through it transparently
     * while updating their volume stack; photon paths must do the same or emitters outside a
     * volume can never reach and scatter inside it. This is not a caustic-forming event. Keep the
     * ray origin and advance tmin past this intersection, matching
     * integrate_surface_volume_only_bounce(). */
    if (sd.shader_flag & SD_HAS_ONLY_VOLUME) {
      if (!path_state_volume_next(state)) {
        return;
      }
      volume_stack_enter_exit<false>(kg, state, &sd);
      ray.tmin = intersection_t_offset(sd.ray_length);
      ray.tmax = FLT_MAX;
      ray.self.prim = sd.prim;
      ray.self.object = sd.object;
      ray.self.light_prim = PRIM_NONE;
      ray.self.light_object = OBJECT_NONE;
      /* Volume bounds are transparent bookkeeping intersections, not transport bounces. */
      bounce--;
      continue;
    }
#endif

#ifdef __SPECTRAL__
    if (sd.runtime_flag & (SR_BSDF_HAS_DISPERSION | SR_BSDF_HAS_SPECTRAL_TRANSMISSION)) {
      /* Keep raw photon power and defer the wavelength PDF/color-matching weight until gather.
       * This permits wavelength matching when the eye path is spectral. */
      INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_SPECTRAL;
    }
#endif
    surface_shader_prepare_closures(kg, state, &sd, path_visibility);

    bool has_receiver = false;
    for (int i = 0; i < sd.num_closure; i++) {
      has_receiver |= surface_shader_photon_mapping_receiver(&sd.closure[i], sd.wi);
    }
    if (had_specular && has_receiver) {
      photon_store(kg,
                   sd.P,
                   sd.Ng,
                   ray.D,
                   throughput,
                   emitter_object,
                   sd.object,
                   time,
                   photon_wavelength_rand,
                   (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SPECTRAL) != 0u,
                   false);
      return;
    }

    float3 rand_bsdf = lcg_step_float3(&rng);
    const ccl_private ShaderClosure *sc = surface_shader_bsdf_bssrdf_pick(&sd, &rand_bsdf);
    if (!CLOSURE_IS_BSDF(sc->type) || CLOSURE_IS_RAY_PORTAL(sc->type) ||
        surface_shader_is_hair_closure(sc->type))
    {
      return;
    }

    BsdfEval eval;
    float3 wo;
    float pdf;
    float2 sampled_roughness;
    sampled_roughness = one_float2();
    float eta = 1.0f;
    float avg_roughness_squared = 0.0f;
    const int label = surface_shader_bsdf_sample_closure(
        kg, &sd, sc, rand_bsdf, &eval, &wo, &pdf, &sampled_roughness, &eta, avg_roughness_squared);
    if (!(pdf > 0.0f) || bsdf_eval_is_zero(&eval)) {
      return;
    }
    throughput *= bsdf_eval_sum(&eval) / pdf;
    if (!isfinite_safe(throughput)) {
      return;
    }

    path_state_next(kg, state, label, sd.runtime_flag);

#ifdef __VOLUME__
    if (label & LABEL_TRANSMIT) {
      volume_stack_enter_exit<false>(kg, state, &sd);
    }
#endif

    if (!(label & LABEL_TRANSPARENT)) {
      if (((label & LABEL_REFLECT) && !kernel_data.integrator.caustics_reflective) ||
          ((label & LABEL_TRANSMIT) && !kernel_data.integrator.caustics_refractive))
      {
        return;
      }
      const float roughness = max(sampled_roughness.x, sampled_roughness.y);
      const bool sharp = (label & LABEL_SINGULAR) ||
                         roughness < sqr(kernel_data.integrator.photon_roughness_threshold);
      const bool specular_label = (label & LABEL_GLOSSY) || (label & LABEL_TRANSMIT) ||
                                  (label & LABEL_SINGULAR);
      /* Keep the estimator's transport partition disjoint from camera path tracing. A caustic
       * photon may traverse any number of sharp events, but a rough glossy/transmission event
       * hands the remainder of that path back to the regular integrator. */
      if (specular_label && !sharp) {
        return;
      }
      had_specular |= sharp && specular_label;
      if (!had_specular && (label & LABEL_DIFFUSE)) {
        return;
      }
      ray.P = ray_offset(sd.P, dot(sd.Ng, wo) >= 0.0f ? sd.Ng : -sd.Ng);
      ray.tmin = 0.0f;
    }
    else {
      ray.P = sd.P;
      ray.tmin = intersection_t_offset(sd.ray_length);
    }
    ray.D = normalize(wo);
    ray.tmax = FLT_MAX;
    ray.self.prim = sd.prim;
    ray.self.object = sd.object;
    ray.self.light_prim = PRIM_NONE;
    ray.self.light_object = OBJECT_NONE;

    if (bounce >= 3) {
      const float continuation = min(saturatef(reduce_max(throughput)), 0.95f);
      if (lcg_step_float(&rng) >= continuation) {
        return;
      }
      throughput /= continuation;
    }
  }
}

/* Cheap first-pass predicate used to count valid neighbors before bounded stochastic evaluation.
 */
ccl_device_inline bool photon_mapping_matches(KernelGlobals kg,
                                              ccl_private const ShaderData *sd,
                                              ccl_global const KernelPhoton *photon,
                                              const int time_bin,
                                              const float radius2,
                                              const bool camera_spectral,
                                              const float camera_wavelength)
{
  if (photon_time_bin(photon_unpack_time(photon->time_wavelength)) != time_bin ||
      photon->receiver_object != sd->object)
  {
    return false;
  }

#ifdef __SPECTRAL__
  if (camera_spectral && photon_is_spectral(photon->time_wavelength)) {
    const float photon_wavelength = sample_wavelength(
        photon_unpack_wavelength_rand(photon->time_wavelength));
    if (photon_spectral_kernel(photon_wavelength, camera_wavelength) == 0.0f) {
      return false;
    }
  }
#else
  (void)camera_spectral;
  (void)camera_wavelength;
#endif

  const float3 delta = photon->P - sd->P;
  const float normal_distance = dot(delta, sd->Ng);
  const float tangent_distance2 = max(len_squared(delta) - sqr(normal_distance), 0.0f);
  if (tangent_distance2 > radius2 || fabsf(normal_distance) > 0.25f * sqrtf(radius2)) {
    return false;
  }

  packed_normal photon_normal;
  photon_normal.value = photon->normal;
  if (dot(sd->Ng, photon_normal.decode()) < kernel_data.integrator.photon_normal_threshold) {
    return false;
  }

#ifdef __LIGHT_LINKING__
  if ((kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_LINKING) &&
      !light_link_object_match(kg, sd->object, photon->emitter_object))
  {
    return false;
  }
#endif
  return true;
}

/* Return the photon density estimate already multiplied by the receiver BSDF. */
ccl_device_inline Spectrum photon_mapping_gather(KernelGlobals kg,
                                                 IntegratorState state,
                                                 ccl_private ShaderData *sd,
                                                 ccl_global float *ccl_restrict render_buffer)
{
  if (!kernel_data.integrator.use_photon_mapping || !kernel_integrator_state.photons ||
      kernel_integrator_state.photon_hash_size == 0)
  {
    return zero_spectrum();
  }

  bool has_receiver = false;
  for (int i = 0; i < sd->num_closure; i++) {
    has_receiver |= surface_shader_photon_mapping_receiver(&sd->closure[i], sd->wi);
  }
  if (!has_receiver) {
    return zero_spectrum();
  }

  const float radius = kernel_integrator_state.photon_radius;
  const float radius2 = sqr(radius);
  const int3 base = photon_cell(sd->P, radius);
  const int time_bin = photon_time_bin(INTEGRATOR_STATE(state, ray, time));
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  const bool camera_spectral = (path_flag & PATH_RAY_SPECTRAL) != 0u;
#ifdef __SPECTRAL__
  const float camera_wavelength_rand = path_rng_1D(kg,
                                                   INTEGRATOR_STATE(state, path, rng_pixel),
                                                   INTEGRATOR_STATE(state, path, sample),
                                                   PRNG_BOUNCE_NUM + PRNG_WAVELENGTH);
  const float camera_wavelength = sample_wavelength(camera_wavelength_rand);
#else
  const float camera_wavelength = 0.0f;
#endif
  int num_valid = 0;

  /* Counting is much cheaper than BSDF evaluation and removes the energy loss of truncating a
   * dense hash chain. The second pass evaluates a fixed-size unbiased systematic subset. */
  for (int z = -1; z <= 1; z++) {
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        const uint bucket = photon_hash_cell(
            base + make_int3(x, y, z), time_bin, kernel_integrator_state.photon_hash_size);
        uint node = kernel_integrator_state.photon_hash[bucket];
        uint traversed = 0;
        while (node != 0u && traversed++ < kernel_integrator_state.photon_capacity) {
          if (node > kernel_integrator_state.photon_capacity) {
            break;
          }
          const ccl_global KernelPhoton *photon = &kernel_integrator_state.photons[node - 1u];
          node = photon->next;
          if (photon_mapping_matches(
                  kg, sd, photon, time_bin, radius2, camera_spectral, camera_wavelength))
          {
            num_valid++;
          }
        }
      }
    }
  }

  if (num_valid == 0) {
    return zero_spectrum();
  }

  const float selection_probability = min(
      1.0f, float(kernel_data.integrator.photon_gather_max) / float(num_valid));
  const float selection_weight = 1.0f / selection_probability;
  const uint rng_pixel = INTEGRATOR_STATE(state, path, rng_pixel);
  const uint sample = uint(INTEGRATOR_STATE(state, path, sample));
  const uint bounce = uint(INTEGRATOR_STATE(state, path, bounce));
  const float selection_offset = hash_uint3_to_float(rng_pixel, sample, bounce);
  int valid_index = 0;
  Spectrum sum = zero_spectrum();

  for (int z = -1; z <= 1; z++) {
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        const uint bucket = photon_hash_cell(
            base + make_int3(x, y, z), time_bin, kernel_integrator_state.photon_hash_size);
        uint node = kernel_integrator_state.photon_hash[bucket];
        uint traversed = 0;
        while (node != 0u && traversed++ < kernel_integrator_state.photon_capacity) {
          if (node > kernel_integrator_state.photon_capacity) {
            break;
          }
          const uint photon_index = node - 1u;
          const ccl_global KernelPhoton *photon = &kernel_integrator_state.photons[photon_index];
          node = photon->next;
          if (!photon_mapping_matches(
                  kg, sd, photon, time_bin, radius2, camera_spectral, camera_wavelength))
          {
            continue;
          }

          /* Randomized systematic sampling selects exactly gather_max of N matching photons.
           * Every photon still has inclusion probability gather_max/N, while avoiding the count
           * variance and occasional empty/oversized subsets of independent Bernoulli trials. Hash
           * insertion already randomizes traversal order by emitted path index. */
          bool selected = true;
          if (selection_probability < 1.0f) {
            const float before = floorf(float(valid_index) * selection_probability +
                                        selection_offset);
            const float after = floorf(float(valid_index + 1) * selection_probability +
                                       selection_offset);
            selected = after > before;
          }
          valid_index++;
          if (!selected) {
            continue;
          }

          const float3 delta = photon->P - sd->P;
          const float normal_distance = dot(delta, sd->Ng);
          const float distance2 = max(len_squared(delta) - sqr(normal_distance), 0.0f);
          Spectrum receiver_eval = zero_spectrum();
          packed_normal photon_direction;
          photon_direction.value = photon->direction;
          const float3 light_direction = -photon_direction.decode();
          for (int i = 0; i < sd->num_closure; i++) {
            const ccl_private ShaderClosure *sc = &sd->closure[i];
            if (surface_shader_photon_mapping_receiver(sc, sd->wi)) {
              float unused_pdf;
              receiver_eval += bsdf_eval(kg, sd, sc, light_direction, &unused_pdf) * sc->weight;
            }
          }
          const float q = distance2 / radius2;
          const float kernel_weight = 2.0f * (1.0f - q) / (M_PI_F * radius2);
          Spectrum photon_power = rgb_to_spectrum(make_float3(photon->power));
#ifdef __SPECTRAL__
          if (photon_is_spectral(photon->time_wavelength)) {
            const float wavelength_rand = photon_unpack_wavelength_rand(photon->time_wavelength);
            if (camera_spectral) {
              float wavelength_pdf;
              const float wavelength = sample_wavelength(wavelength_rand, &wavelength_pdf);
              photon_power *= photon_spectral_kernel(wavelength, camera_wavelength) /
                              wavelength_pdf;
            }
            else {
              photon_power *= dispersion_throughput_weight(kg, wavelength_rand);
            }
          }
#endif
          const Spectrum photon_L = photon_power * receiver_eval *
                                    (kernel_weight * selection_weight *
                                     float(kernel_data.integrator.photon_time_bins));
          sum += photon_L;

#ifdef __PASSES__
          if (kernel_data.film.pass_lightgroup != PASS_UNUSED &&
              !(path_flag & PATH_RAY_SHADOW_CATCHER_HIT))
          {
            const int lightgroup = object_lightgroup(kg, photon->emitter_object);
            if (lightgroup != LIGHTGROUP_NONE) {
              Spectrum group_contribution = INTEGRATOR_STATE(state, path, throughput) * photon_L;
              film_clamp_light(
                  kg, &group_contribution, max(int(INTEGRATOR_STATE(state, path, bounce)), 1));
              ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);
              film_write_pass_spectrum(buffer + kernel_data.film.pass_lightgroup + 3 * lightgroup,
                                       group_contribution);
            }
          }
#endif
        }
      }
    }
  }
  return sum;
}

ccl_device_inline void photon_mapping_write(KernelGlobals kg,
                                            ConstIntegratorState state,
                                            const Spectrum L,
                                            ccl_global float *ccl_restrict render_buffer)
{
  Spectrum contribution = INTEGRATOR_STATE(state, path, throughput) * L;
  film_clamp_light(kg, &contribution, max(int(INTEGRATOR_STATE(state, path, bounce)), 1));
  if (is_zero(contribution)) {
    return;
  }

  ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);
  const PathRayVisibility visibility = INTEGRATOR_STATE(state, path, visibility);
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  const int sample = INTEGRATOR_STATE(state, path, sample);
  film_write_combined_pass(kg, visibility, path_flag, sample, contribution, buffer);

#ifdef __PASSES__
  if ((kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES) &&
      !(path_flag & PATH_RAY_SHADOW_CATCHER_HIT))
  {
    if (visibility & PATH_RAY_VISIBILITY_CAMERA) {
      if (kernel_data.film.pass_diffuse_indirect != PASS_UNUSED) {
        film_write_pass_spectrum(buffer + kernel_data.film.pass_diffuse_indirect, contribution);
      }
    }
    else if (path_flag & PATH_RAY_SURFACE_PASS) {
      const Spectrum diffuse_weight = INTEGRATOR_STATE(state, path, pass_diffuse_weight);
      const Spectrum glossy_weight = INTEGRATOR_STATE(state, path, pass_glossy_weight);
      if (kernel_data.film.pass_diffuse_indirect != PASS_UNUSED) {
        film_write_pass_spectrum(buffer + kernel_data.film.pass_diffuse_indirect,
                                 diffuse_weight * contribution);
      }
      if (kernel_data.film.pass_glossy_indirect != PASS_UNUSED) {
        film_write_pass_spectrum(buffer + kernel_data.film.pass_glossy_indirect,
                                 glossy_weight * contribution);
      }
      if (kernel_data.film.pass_transmission_indirect != PASS_UNUSED) {
        film_write_pass_spectrum(buffer + kernel_data.film.pass_transmission_indirect,
                                 (one_spectrum() - diffuse_weight - glossy_weight) * contribution);
      }
    }
  }
#endif
}

CCL_NAMESPACE_END
