/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "gpu_shader_material_blackbody.glsl"

[[node]]
void node_volume_fast(float4 scatter_color,
                      float density,
                      float density_cutoff,
                      float scatter_strength,
                      float anisotropy,
                      float IOR,
                      float Backscatter,
                      float alpha,
                      float diameter,
                      float4 absorption_color,
                      float absorption_strength,
                      float emission_strength,
                      float4 emission_color,
                      float flame_cutoff,
                      float blackbody_strength,
                      float4 blackbody_tint,
                      float temperature,
                      float weight,
                      float4 density_attribute,
                      float4 color_attribute,
                      float4 scatter_attribute,
                      float4 emission_attribute,
                      float4 temperature_attribute,
                      sampler1DArray spectrummap,
                      float layer,
                      Closure &result)
{
  float3 scatter_coeff = float3(0.0f);
  float3 absorption_coeff = float3(0.0f);
  float3 emission_coeff = float3(0.0f);

  density = max(density * density_attribute.x, 0.0f);
  density = (density >= max(density_cutoff, 0.0f)) ? density : 0.0f;

  if (density > 0.0f) {
    float scatter_mask = max(scatter_attribute.x, 0.0f);
    scatter_coeff = max(scatter_color.rgb * color_attribute.rgb, float3(0.0f)) * density *
                    max(scatter_strength, 0.0f) * scatter_mask;
    absorption_coeff = max(absorption_color.rgb, float3(0.0f)) * density *
                       max(absorption_strength, 0.0f);
  }

  float flame = max(emission_attribute.x, 0.0f);
  flame = (flame > max(flame_cutoff, 0.0f)) ? flame : 0.0f;
  if (emission_strength > 0.0f && flame > 0.0f) {
    emission_coeff += max(emission_color.rgb, float3(0.0f)) * emission_strength * flame;
  }

  if (blackbody_strength > 0.0f && flame > 0.0f) {
    float T = max(temperature * max(temperature_attribute.x, 0.0f), 0.0f);
    float T2 = T * T;
    float intensity = (5.670373e-8f * 1e-6f / M_PI) * T2 * T2 *
                      max(blackbody_strength, 0.0f);
    if (intensity > 0.0f) {
      float4 bb;
      node_blackbody(T, spectrummap, layer, bb);
      emission_coeff += bb.rgb * max(blackbody_tint.rgb, float3(0.0f)) * intensity * flame;
    }
  }

  ClosureVolumeScatter volume_scatter_data;
  volume_scatter_data.scattering = scatter_coeff * weight;
  volume_scatter_data.anisotropy = clamp(anisotropy, -1.0f, 1.0f) * weight;

  ClosureVolumeAbsorption volume_absorption_data;
  volume_absorption_data.absorption = absorption_coeff * weight;

  ClosureEmission emission_data;
  emission_data.emission = emission_coeff * weight;

  result = closure_eval(volume_scatter_data, volume_absorption_data, emission_data);
}
