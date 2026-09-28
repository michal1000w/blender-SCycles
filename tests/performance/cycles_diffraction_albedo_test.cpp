/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction_albedo.h"

#include <cmath>
#include <cstdio>
#include <limits>

using namespace ccl;

int main()
{
  int failures = 0;
  std::string error;
  DiffractionAlbedoRequest r;
  r.wavelength_count = 2;
  r.mu_count = 3;
  r.phi_count = 4;
  r.facet_samples = 8;
  failures += !diffraction_albedo_validate_request(r, error);
  DiffractionAlbedoRequest changed = r;
  changed.alpha_x = std::numeric_limits<float>::quiet_NaN();
  failures += diffraction_albedo_validate_request(changed, error);
  changed = r;
  changed.medium_ior = 0.0f;
  failures += diffraction_albedo_validate_request(changed, error);
  changed = r;
  changed.wavelength_max_nm = 700.0f;
  failures += diffraction_albedo_validate_request(changed, error);
  changed = r;
  changed.mu_count = 1000000;
  failures += diffraction_albedo_validate_request(changed, error);
  changed = r;
  changed.facet_samples = 65536;
  changed.mu_count = 256;
  changed.phi_count = 512;
  failures += diffraction_albedo_validate_request(changed, error);

  /* Synthetic slices E=mu and E=mu/2. Projected means are 2/3 and 1/3,
   * including the nonuniform first mu interval. */
  DiffractionAlbedoTable synthetic;
  synthetic.request = r;
  synthetic.values.resize(2 * 3 * 4);
  synthetic.averages = {2.0f / 3.0f, 1.0f / 3.0f};
  for (int s = 0; s < 2; ++s)
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 4; ++j)
        synthetic.values[s * 12 + i * 4 + j] = (s ? 0.5f : 1.0f) * float(i * i) / 4.0f;
  failures += !diffraction_albedo_validate_table(synthetic, error);
  const float3 seam_a = make_float3(-1.0f, 0.00001f, 0.3f);
  const float3 seam_b = make_float3(-1.0f, -0.00001f, 0.3f);
  failures += std::abs(diffraction_albedo_lookup(synthetic, 580.0f, seam_a) - 0.225f) > 1e-5f;
  failures += std::abs(diffraction_albedo_lookup(synthetic, 580.0f, seam_b) - 0.225f) > 1e-5f;
  failures += std::abs(diffraction_albedo_average(synthetic, 580.0f) - 0.5f) > 1e-6f;
  synthetic.averages[0] = 0.5f;
  failures += diffraction_albedo_validate_table(synthetic, error);

  DiffractionAlbedoTable table;
  failures += !diffraction_albedo_build_cpu(r, table, error);
  failures += !diffraction_albedo_validate_table(table, error);
  failures += !std::isfinite(diffraction_albedo_lookup(table, 550.0f,
                                                       make_float3(0.7f, 0.0f, 0.7f)));
  const auto original = table.values;
  failures += diffraction_albedo_build_cpu(r, table, error, [] { return true; });
  failures += table.values != original;

  std::printf("GGX diffraction albedo foundation: %s (%d failures)\n",
              failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
