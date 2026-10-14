/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "gpu_shader_material_interface.bsl.hh"

/* The index of refraction of a volume bends rays, which only a ray tracer follows: nothing is
 * added to the volume here. */
[[node]]
void node_volume_ior(float ior,
                     float temperature,
                     float ambient_temperature,
                     float pressure,
                     float gas_ior,
                     float strength,
                     float weight,
                     KernelGlobals &kg,
                     ShadingData &sd,
                     Closure &result)
{
  ClosureVolumeAbsorption volume_absorption_data;
  volume_absorption_data.absorption = float3(0.0f);

  result = closure_eval(kg, sd, volume_absorption_data);
}
