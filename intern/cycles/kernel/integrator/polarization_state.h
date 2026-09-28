/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/integrator/state.h"
#include "kernel/light/polarization_math.h"
CCL_NAMESPACE_BEGIN
struct PolarizationSpectrumState { Spectrum value[4]; };
ccl_device_inline bool polarization_enabled(KernelGlobals kg)
{
#if defined(__KERNEL_METAL_TRANSPORT_FEATURES__) && !(__KERNEL_METAL_TRANSPORT_FEATURES__ & KERNEL_FEATURE_POLARIZATION)
  return false;
#else
  return (kernel_data.kernel_features & KERNEL_FEATURE_POLARIZATION)!=0;
#endif
}
ccl_device_inline PolarizationSpectrumState polarization_unpolarized()
{
  return {{one_spectrum(),zero_spectrum(),zero_spectrum(),zero_spectrum()}};
}
ccl_device_inline PolarizationSpectrumState polarization_path_read(IntegratorState state)
{
  return {{Spectrum(INTEGRATOR_STATE(state,path,polarization_i)),
           Spectrum(INTEGRATOR_STATE(state,path,polarization_q)),
           Spectrum(INTEGRATOR_STATE(state,path,polarization_u)),
           Spectrum(INTEGRATOR_STATE(state,path,polarization_v))}};
}
ccl_device_inline void polarization_path_write(IntegratorState state,
                                               const ccl_private PolarizationSpectrumState &p)
{
  INTEGRATOR_STATE_WRITE(state,path,polarization_i)=p.value[0];
  INTEGRATOR_STATE_WRITE(state,path,polarization_q)=p.value[1];
  INTEGRATOR_STATE_WRITE(state,path,polarization_u)=p.value[2];
  INTEGRATOR_STATE_WRITE(state,path,polarization_v)=p.value[3];
}
ccl_device_inline PolarizationSpectrumState polarization_spectrum_apply(
    const ccl_private PolarizationMueller &m,const ccl_private PolarizationSpectrumState &p,
    const bool adjoint)
{
  PolarizationSpectrumState r{{zero_spectrum(),zero_spectrum(),zero_spectrum(),zero_spectrum()}};
  for(int i=0;i<4;i++)for(int j=0;j<4;j++)r.value[i]+=(adjoint?m.value[j][i]:m.value[i][j])*p.value[j];
  return r;
}
ccl_device_inline Spectrum polarization_spectrum_contract(
    const ccl_private PolarizationSpectrumState &a,const ccl_private PolarizationSpectrumState &p)
{
  Spectrum r=zero_spectrum();for(int i=0;i<4;i++)r+=a.value[i]*p.value[i];return r;
}
/* Cached native directions are quantized. Rotate the polarization coordinates
 * to the decoded frame used by all cached-vertex evaluations. */
ccl_device_inline PolarizationSpectrumState polarization_reframe(
    const ccl_private PolarizationSpectrumState &p,const float3 from,const float3 to)
{
  float3 fx,fy,tx,ty; make_orthonormals(from,&fx,&fy); make_orthonormals(to,&tx,&ty);
  float c=dot(tx,fx),s=dot(ty,fx); const float norm=sqrtf(c*c+s*s);
  if (!(norm>0)) return {{p.value[0],zero_spectrum(),zero_spectrum(),zero_spectrum()}};
  c/=norm;s/=norm;
  PolarizationJones j{{{make_float2(c,0),make_float2(-s,0)},
                        {make_float2(s,0),make_float2(c,0)}}};
  return polarization_spectrum_apply(polarization_mueller_from_jones(j),p,false);
}
ccl_device_inline KernelPolarizationState polarization_pack(const ccl_private PolarizationSpectrumState &p)
{
  return {PackedSpectrum(p.value[0]),PackedSpectrum(p.value[1]),PackedSpectrum(p.value[2]),PackedSpectrum(p.value[3])};
}
ccl_device_inline PolarizationSpectrumState polarization_unpack(const KernelPolarizationState p)
{
  return {{Spectrum(p.i),Spectrum(p.q),Spectrum(p.u),Spectrum(p.v)}};
}
ccl_device_inline PolarizationSpectrumState polarization_depolarized(const ccl_private PolarizationSpectrumState &p)
{ return {{p.value[0],zero_spectrum(),zero_spectrum(),zero_spectrum()}}; }
ccl_device_inline Spectrum polarization_emission_weight(KernelGlobals kg,IntegratorState state)
{
  return polarization_enabled(kg)?Spectrum(INTEGRATOR_STATE(state,path,polarization_i)):one_spectrum();
}
CCL_NAMESPACE_END
