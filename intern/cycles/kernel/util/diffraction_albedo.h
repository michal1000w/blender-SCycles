/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
CCL_NAMESPACE_BEGIN

ccl_device_inline float diffraction_albedo_angular(KernelGlobals kg, const int4 d,
                                                  const int slice, const float3 w)
{
  const float mu = saturatef(w.z), coordinate = sqrtf(mu) * (d.z - 1);
  const int i = min(int(floorf(coordinate)), d.z - 1), k = min(i + 1, d.z - 1);
  const float v = (atan2f(w.y, w.x) * M_1_2PI_F + 1.0f) * d.w - 0.5f;
  const int j = (int(floorf(v)) % d.w + d.w) % d.w, l = (j + 1) % d.w;
  const float mi = float(i) / (d.z - 1), mk = float(k) / (d.z - 1);
  const float a = i == k ? 0.0f : saturatef((mu - mi * mi) / (mk * mk - mi * mi));
  const float b = v - floorf(v);
  const int base = d.x + slice * d.z * d.w;
  const float ij = kernel_data_fetch(diffraction_albedo_values, base + i * d.w + j);
  const float kj = kernel_data_fetch(diffraction_albedo_values, base + k * d.w + j);
  const float il = kernel_data_fetch(diffraction_albedo_values, base + i * d.w + l);
  const float kl = kernel_data_fetch(diffraction_albedo_values, base + k * d.w + l);
  return mix(mix(ij, kj, a), mix(il, kl, a), b);
}

ccl_device_inline float2 diffraction_albedo_lookup(KernelGlobals kg, const int handle,
                                                  const float wavelength_nm, const float3 w)
{
  if (handle < 0 || handle >= kernel_data.tables.num_diffraction_albedo_caches) {
    return one_float2();
  }
  const int4 d = kernel_data_fetch(diffraction_albedo_descriptors, handle);
  const float4 domain = kernel_data_fetch(diffraction_albedo_domains, handle);
  const int n = int(domain.z);
  const float x = saturatef((wavelength_nm - domain.x) / (domain.y - domain.x)) * (n - 1);
  const int i = min(int(floorf(x)), n - 1), j = min(i + 1, n - 1);
  const float t = x - i;
  const float E = mix(diffraction_albedo_angular(kg, d, i, w),
                      diffraction_albedo_angular(kg, d, j, w), t);
  const float average = mix(kernel_data_fetch(diffraction_albedo_averages, d.y + i),
                            kernel_data_fetch(diffraction_albedo_averages, d.y + j), t);
  return make_float2(saturatef(E), saturatef(average));
}
CCL_NAMESPACE_END
