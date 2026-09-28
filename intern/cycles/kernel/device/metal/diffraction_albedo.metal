/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

/* Native one-sided GGX grating albedo construction and normalization.
 * Table approximation and material support are validated by the host. */
#include "kernel/device/metal/compat.h"
#include "kernel/device/metal/globals.h"
#include "kernel/tables.h"
#include "kernel/device/metal/context_begin.h"
#include "kernel/closure/bsdf_diffraction.h"
#include "kernel/closure/bsdf_diffraction_dielectric.h"
#include "kernel/closure/diffraction_thin_sheet_model.h"
#include "kernel/device/metal/context_end.h"

kernel void diffraction_multiscatter_context_size(device uint *result [[buffer(0)]])
{
  result[0] = sizeof(KernelParamsMetal);
}

/* One SIMD group per directional cache node; group.y selects a material or
 * wavelength slice, with profile-major output. Input is (wi.xyz, unused), with
 * p=(alpha_x, alpha_y, wavelength/pitch, relief/wavelength, duty). */
kernel void diffraction_multiscatter_albedo(device const float4 *directions [[buffer(0)]],
                                            device float *albedos [[buffer(1)]],
                                            constant KernelParamsMetal &params [[buffer(2)]],
                                            constant float *parameters [[buffer(3)]],
                                            constant uint &samples [[buffer(4)]],
                                            constant uint &direction_count [[buffer(5)]],
                                            uint3 group [[threadgroup_position_in_grid]],
                                            uint lane [[thread_index_in_simdgroup]],
                                            uint width [[threads_per_simdgroup]])
{
  const uint i = group.x;
  parameters += 5 * group.y;
  MetalKernelContext context(params);
  const MetalKernelContext::DiffractionReflection p = {
      parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]};
  const float3 wi = directions[i].xyz;
  const float li = context.bsdf_aniso_lambda<MetalKernelContext::GGX>(p.alpha_x, p.alpha_y, wi);
  float sum = 0.0f;
  const int limit = context.diffraction_reflection_max_order(&p);
  const float3 axis = float3(1.0f, 0.0f, 0.0f);
  for (uint s = lane; s < samples; s += width) {
    const float u = (float(s) + 0.5f) / float(samples);
    const float v = metal::fract((float(s) + 0.5f) * 0.6180339887498949f);
    const float3 h = context.microfacet_ggx_sample_vndf(wi, p.alpha_x, p.alpha_y, float2(u, v));
    const float ci = metal::dot(wi, h);
    float nonzero_mass = 0.0f;
    for (int order = -limit; order <= limit; ++order) {
      float3 wo;
      if (order == 0 || !context.diffraction_facet_reflect(
                            wi, h, axis, float(order) * p.wavelength_over_pitch, &wo)) {
        continue;
      }
      const float mass = context.diffraction_reflection_nonzero_power(
          &p, order, ci, metal::dot(wo, h));
      nonzero_mass += mass;
      if (wo.z > 0.0f) {
        const float lo = context.bsdf_aniso_lambda<MetalKernelContext::GGX>(
            p.alpha_x, p.alpha_y, wo);
        sum += mass * (1.0f + li) / (1.0f + li + lo);
      }
    }
    const float3 wo = 2.0f * ci * h - wi;
    if (wo.z > 0.0f) {
      const float lo = context.bsdf_aniso_lambda<MetalKernelContext::GGX>(
          p.alpha_x, p.alpha_y, wo);
      sum += metal::max(0.0f, 1.0f - nonzero_mass) * (1.0f + li) / (1.0f + li + lo);
    }
  }
  const float total = metal::simd_sum(sum);
  if (lane == 0) albedos[group.y * direction_count + i] = total / float(samples);
}

/* Exact projected-solid-angle integral of the piecewise-linear mu table.
 * The mu nodes are quadratic; a plain mean of table entries is incorrect. */
kernel void diffraction_multiscatter_average(device const float *tables [[buffer(0)]],
                                             device float *averages [[buffer(1)]],
                                             constant uint2 &shape [[buffer(2)]],
                                             uint3 group [[threadgroup_position_in_grid]],
                                             uint lane [[thread_index_in_simdgroup]],
                                             uint width [[threads_per_simdgroup]])
{
  const uint mu_count = shape.x, phi_count = shape.y;
  const uint offset = group.x * mu_count * phi_count;
  float sum = 0.0f;
  for (uint index = lane; index < (mu_count - 1) * phi_count; index += width) {
    const uint i = index / phi_count, j = index % phi_count;
    const float x = float(i) / float(mu_count - 1);
    const float y = float(i + 1) / float(mu_count - 1);
    const float mu = x * x, h = y * y - mu;
    const float a = tables[offset + i * phi_count + j];
    const float delta = tables[offset + (i + 1) * phi_count + j] - a;
    sum += 2.0f * h * (mu * a + (mu * delta + h * a) * 0.5f + h * delta / 3.0f);
  }
  const float total = metal::simd_sum(sum);
  if (lane == 0) averages[group.x] = total / float(phi_count);
}

/* Two-sided dielectric single-event escape, with one SIMD group per incident
 * direction and basis/physical side/wavelength slice. The host applies the shared
 * deficit, etendue integral, and film cross-fraction finalizer to this output.
 * p = (alpha_x, alpha_y, pitch, depth, duty, ni, nt, lambda, film_ior,
 * film_thickness, generalized_f0), all lengths in nanometers. Negative F0 selects
 * physical Fresnel; nonnegative scalar F0 selects a generalized endpoint basis. */
kernel void diffraction_two_sided_albedo(device const float4 *directions [[buffer(0)]],
                                         device float2 *escapes [[buffer(1)]],
                                         constant KernelParamsMetal &params [[buffer(2)]],
                                         constant float *parameters [[buffer(3)]],
                                         constant uint &samples [[buffer(4)]],
                                         constant uint &direction_count [[buffer(5)]],
                                         uint3 group [[threadgroup_position_in_grid]],
                                         uint lane [[thread_index_in_simdgroup]],
                                         uint width [[threads_per_simdgroup]])
{
  const uint i = group.x;
  parameters += 11 * group.y;
  MetalKernelContext context(params);
  MetalKernelContext::DiffractionRoughDielectric p;
  if (!context.diffraction_dielectric_parameters(parameters[7], parameters[2], parameters[3],
                                        parameters[4], parameters[5], parameters[6],
                                        parameters[0], parameters[1], &p)) {
    if (lane == 0) escapes[group.y * direction_count + i] = float2(-1.0f);
    return;
  }
  const float3 wi = directions[i].xyz;
  const float film_ratio = parameters[9] / parameters[7];
  const float li = context.diffraction_dielectric_lambda<MetalKernelContext::GGX>(
      p.alpha_x, p.alpha_y, wi);
  const float f0 = parameters[10];
  MetalKernelContext::DiffractionDielectricGeneralizedExtra generalized{};
  generalized.base.param = p;
  generalized.generalized_f0 = float3(f0);
  generalized.generalized_reference_f0 = context.F0_from_ior(
      p.facet.transmitted_ior / p.facet.incident_ior);
  generalized.film_ior = parameters[8];
  generalized.film_thickness_over_wavelength = film_ratio;
  const bool matched = p.facet.incident_ior == p.facet.transmitted_ior;
  const bool physical = f0 < 0.0f ||
      (film_ratio > 0.0f && generalized.generalized_reference_f0 <= 1e-5f);
  if (matched && physical && film_ratio == 0.0f) {
    if (lane == 0) escapes[group.y * direction_count + i] = float2(0.0f, 1.0f);
    return;
  }
  const float continuous_budget = !physical && matched ? f0 : 1.0f;
  // The straight-through matched-index atom is added exactly once after reduction.
  float reflected = 0.0f, transmitted = 0.0f;
  for (uint s = lane; s < samples; s += width) {
    const float u = (float(s) + 0.5f) / float(samples);
    const float v = metal::fract((float(s) + 0.5f) * 0.6180339887498949f);
    const float w = metal::fract((float(s) + 0.5f) * 0.7548776662466927f);
    float3 wo;
    float mass;
    bool singular;
    const bool sampled = !physical ?
        context.diffraction_dielectric_generalized_sample_direction<MetalKernelContext::GGX>(
            &generalized, wi, float3(u, v, w), &wo, &mass, &singular) :
        context.diffraction_dielectric_sample_direction<MetalKernelContext::GGX>(
            &p, wi, float3(u, v, w), &wo, &mass, &singular, false,
            parameters[8], film_ratio);
    if (!sampled) {
      continue;
    }
    const float lo = context.diffraction_dielectric_lambda<MetalKernelContext::GGX>(
        p.alpha_x, p.alpha_y, wo);
    const float escaped = singular && matched && wo.z < 0.0f ? 1.0f :
        continuous_budget * (1.0f + li) / (1.0f + li + lo);
    if (wo.z > 0.0f) reflected += escaped;
    else transmitted += escaped;
  }
  reflected = metal::simd_sum(reflected);
  transmitted = metal::simd_sum(transmitted);
  if (lane == 0) {
    escapes[group.y * direction_count + i] =
        float2(reflected / float(samples), transmitted / float(samples) +
               (matched && !physical ? 1.0f - f0 : 0.0f));
  }
}

#include "kernel/device/metal/diffraction_thin_sheet_albedo.metal"
