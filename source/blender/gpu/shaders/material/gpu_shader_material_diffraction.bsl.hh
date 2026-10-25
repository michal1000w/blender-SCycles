/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "gpu_shader_material_interface.bsl.hh"
#include "gpu_shader_math_vector_safe.bsl.hh"

/* The spectral grating is Cycles only, preview it as a glossy reflection tinted by its color. */
[[node]]
void node_bsdf_diffraction(float4 color,
                           float roughness,
                           float3 /*pitch*/,        /* Unsupported. */
                           float3 /*groove_depth*/, /* Unsupported. */
                           float3 /*duty_cycle*/,   /* Unsupported. */
                           float3 /*rotation*/,     /* Unsupported. */
                           float3 N,
                           float3 /*T*/, /* Unsupported. */
                           float weight,
                           const float use_roughness,
                           [[resource_table]] KernelGlobals &kg,
                           ShadingData &sd,
                           Closure &result)
{
  color = max(color, float4(0.0f));
  roughness = (use_roughness != 0.0f) ? saturate(roughness) : 0.0f;
  N = safe_normalize(N);

  float3 V = coordinate_impl(kg, sd, sd.P, sd.N).incoming;
  float NV = dot(N, V);

  float3 brdf = brdf_lut(kg, color.rgb, color.rgb, NV, roughness, false);

  ClosureReflection reflection_data;
  reflection_data.color = weight * brdf;
  reflection_data.N = N;
  reflection_data.roughness = roughness;

  result = closure_eval(kg, sd, reflection_data);
}
