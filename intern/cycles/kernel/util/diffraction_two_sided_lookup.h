/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/util/dielectric_dispersion.h"
#include "kernel/util/dielectric_f0_cache.h"

CCL_NAMESPACE_BEGIN

/* Cache layout: values[basis][side][wavelength][mu][phi], Q[basis][side][wavelength],
 * cross_fraction[basis][wavelength]. descriptor.w packs mu in low8 bits and
 * basis_count-1 in the remaining bits; a physical table has one basis. The host validates all ranges before upload.
 * Direction w is in the selected side's local frame with w.z >= 0. */
ccl_device_inline float diffraction_two_sided_cache_angular(KernelGlobals kg,
                                                              const int4 descriptor,
                                                              const int phi_count,
                                                              const int wavelength_count,
                                                              const int side,
                                                              const int wavelength,
                                                              const float3 w)
{
  const int mu_count = descriptor.w & 255;
  const float mu = saturatef(w.z);
  const float coordinate = sqrtf(mu) * float(mu_count - 1);
  const int i = min(int(floorf(coordinate)), mu_count - 1);
  const int k = min(i + 1, mu_count - 1);
  const float phi = (atan2f(w.y, w.x) * M_1_2PI_F + 1.0f) * float(phi_count) - 0.5f;
  const int j = (int(floorf(phi)) % phi_count + phi_count) % phi_count;
  const int l = (j + 1) % phi_count;
  const float mi = float(i) / float(mu_count - 1);
  const float mk = float(k) / float(mu_count - 1);
  const float a = i == k ? 0.0f : saturatef((mu - mi * mi) / (mk * mk - mi * mi));
  const float b = phi - floorf(phi);
  const int base = descriptor.x + (side * wavelength_count + wavelength) * mu_count * phi_count;
  const float ij = kernel_data_fetch(diffraction_two_sided_values, base + i * phi_count + j);
  const float kj = kernel_data_fetch(diffraction_two_sided_values, base + k * phi_count + j);
  const float il = kernel_data_fetch(diffraction_two_sided_values, base + i * phi_count + l);
  const float kl = kernel_data_fetch(diffraction_two_sided_values, base + k * phi_count + l);
  return saturatef(mix(mix(ij, kj, a), mix(il, kl, a), b));
}

/* (directional deficit q, Q_exterior, Q_interior, side-cross fraction). */
ccl_device_inline float4 diffraction_two_sided_cache_lookup_basis(KernelGlobals kg,
                                                              const int handle,
                                                              const float wavelength_nm,
                                                              const int side,
                                                              const float3 w,
                                                              const int basis,
                                                              const float ior_d,
                                                              const float inv_abbe,
                                                              const float runtime_ior)
{
  if (handle < 0 || handle >= kernel_data.tables.num_diffraction_two_sided_caches ||
      (side != 0 && side != 1))
  {
    return zero_float4();
  }
  int4 descriptor = kernel_data_fetch(diffraction_two_sided_descriptors, handle);
  const float4 domain = kernel_data_fetch(diffraction_two_sided_domains, handle);
  const int wavelength_count = int(domain.z);
  const int phi_count = int(domain.w);
  const int basis_count = 1 + (descriptor.w >> 8);
  if (basis < 0 || basis >= basis_count) return zero_float4();
  descriptor.x += basis * 2 * wavelength_count * (descriptor.w & 255) * phi_count;
  descriptor.y += basis * 2 * wavelength_count;
  descriptor.z += basis * wavelength_count;
  const float coordinate = saturatef((wavelength_nm - domain.x) / (domain.y - domain.x)) *
                           float(wavelength_count - 1);
  const int i = min(int(floorf(coordinate)), wavelength_count - 1);
  const int j = min(i + 1, wavelength_count - 1);
  const float t = coordinate - float(i);
  const float q = mix(diffraction_two_sided_cache_angular(
                          kg, descriptor, phi_count, wavelength_count, side, i, w),
                      diffraction_two_sided_cache_angular(
                          kg, descriptor, phi_count, wavelength_count, side, j, w),
                      t);
  const float q0 = mix(kernel_data_fetch(diffraction_two_sided_integrals,
                                         descriptor.y + i),
                       kernel_data_fetch(diffraction_two_sided_integrals,
                                         descriptor.y + j),
                       t);
  float q1 = mix(kernel_data_fetch(diffraction_two_sided_integrals,
                                         descriptor.y + wavelength_count + i),
                       kernel_data_fetch(diffraction_two_sided_integrals,
                                         descriptor.y + wavelength_count + j),
                       t);
  if (inv_abbe != 0.0f && ior_d > 0.0f && runtime_ior > 0.0f) {
    /* q is interpolated in wavelength. Normalize its angular integral using
     * the actual path's n(lambda)^2, not an interpolation of n_node^2*q_node. */
    const float lambda_i = mix(domain.x, domain.y, float(i) / float(wavelength_count - 1));
    const float lambda_j = mix(domain.x, domain.y, float(j) / float(wavelength_count - 1));
    const float n_i = dielectric_ior_at_wavelength(ior_d, inv_abbe, dielectric_wavelength_um(lambda_i));
    const float n_j = dielectric_ior_at_wavelength(ior_d, inv_abbe, dielectric_wavelength_um(lambda_j));
    q1 = sqr(runtime_ior) * mix(
        kernel_data_fetch(diffraction_two_sided_integrals, descriptor.y + wavelength_count + i) /
            sqr(n_i),
        kernel_data_fetch(diffraction_two_sided_integrals, descriptor.y + wavelength_count + j) /
            sqr(n_j), t);
  }
  const float cross = mix(kernel_data_fetch(diffraction_two_sided_cross,
                                            descriptor.z + i),
                          kernel_data_fetch(diffraction_two_sided_cross,
                                            descriptor.z + j),
                          t);
  return make_float4(q, max(q0, 0.0f), max(q1, 0.0f), saturatef(cross));
}

/* Interpolate q, Q and conductance, then reconstruct the symmetric return
 * matrix at the call site. Blending completed matrices would lose row energy.
 * Bare generalized Fresnel powers are affine in f0 and use two endpoints.
 * Coated powers use a nonuniform bounded grid around the reference Fresnel
 * value; both interpolate deficits before angular/wavelength interpolation. */
ccl_device_inline float4 diffraction_two_sided_cache_lookup(KernelGlobals kg,
                                                           const int handle,
                                                           const float wavelength_nm,
                                                           const int side,
                                                           const float3 w,
                                                           const float f0 = 0.0f,
                                                           const float ior_d = 0.0f,
                                                           const float inv_abbe = 0.0f,
                                                           const float runtime_ior = 0.0f)
{
  if (handle < 0 || handle >= kernel_data.tables.num_diffraction_two_sided_caches) {
    return zero_float4();
  }
  const int4 descriptor = kernel_data_fetch(diffraction_two_sided_descriptors, handle);
  const int count = 1 + (descriptor.w >> 8);
  if (count == 1) {
    return diffraction_two_sided_cache_lookup_basis(kg, handle, wavelength_nm, side, w, 0, ior_d, inv_abbe, runtime_ior);
  }
  const float coordinate = dielectric_f0_cache_coordinate(ior_d, count, f0);
  const int i = min(int(floorf(coordinate)), count - 1);
  const int j = min(i + 1, count - 1);
  return mix(diffraction_two_sided_cache_lookup_basis(kg, handle, wavelength_nm, side, w, i, ior_d, inv_abbe, runtime_ior),
             diffraction_two_sided_cache_lookup_basis(kg, handle, wavelength_nm, side, w, j, ior_d, inv_abbe, runtime_ior),
             coordinate - float(i));
}

CCL_NAMESPACE_END
