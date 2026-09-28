/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Included after the shared ThinSheet model in the Metal kernel context.
 * Profiles: alphaR,alphaT,lambda/pitch,phaseR,phaseT,duty,n_lambda,
 * reflectionTint,transmissionTint,filmIOR,filmThickness/lambda. */
kernel void diffraction_thin_sheet_albedo(device const float4 *directions [[buffer(0)]],
                                          device float *deficits [[buffer(1)]],
                                          constant KernelParamsMetal &params [[buffer(2)]],
                                          constant float *parameters [[buffer(3)]],
                                          constant uint &samples [[buffer(4)]],
                                          constant uint &direction_count [[buffer(5)]],
                                          uint3 group [[threadgroup_position_in_grid]],
                                          uint lane [[thread_index_in_simdgroup]],
                                          uint width [[threads_per_simdgroup]])
{
  parameters += 11 * group.y;
  MetalKernelContext context(params);
  MetalKernelContext::DiffractionThinSheetModel model;
  model.reflection = {parameters[0],parameters[2],parameters[3],parameters[5],false};
  model.transmission = {parameters[1],parameters[2],parameters[4],parameters[5],true};
  model.ior=parameters[6]; model.reflection_tint=parameters[7];
  model.transmission_tint=parameters[8]; model.film_ior=parameters[9];
  model.film_over_wavelength=parameters[10];
  const float3 wi=directions[group.x].xyz;
  float escaped=0.0f;
  for(uint s=lane;s<samples;s+=width) {
    escaped += context.diffraction_thin_sheet_model_escape_port_sample(&model,wi,0,s,samples);
    escaped += context.diffraction_thin_sheet_model_escape_port_sample(&model,wi,1,s,samples);
  }
  const float total=metal::simd_sum(escaped);
  if(lane==0) {
    const float2 coefficients=context.diffraction_thin_sheet_model_coefficients(&model,wi.z);
    /* Preserve raw q. Host validation reports an invalid estimator instead of
     * masking physical/model mistakes by clamping negative deficits. */
    deficits[group.y*direction_count+group.x]=coefficients.x+coefficients.y-total;
  }
}
