/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/light/coherent_path_field.h"
CCL_NAMESPACE_BEGIN

/* Exact one/two planar reflections on actual triangles. Smooth shading normals
 * are never used. The caller verifies native BVH visibility separately. */
ccl_device_inline bool coherent_facet_frame(const ccl_private float3 vertices[3],
                                             ccl_private CoherentGeometryInterface *patch)
{
  const float3 e0 = vertices[1] - vertices[0], e1 = vertices[2] - vertices[0];
  const float3 area = cross(e0, e1);
  if (!(len_squared(area) > 0.0f) || !(len_squared(e0) > 0.0f)) return false;
  const float3 normal = normalize(area);
  const float3 u = normalize(e0), v = normalize(cross(normal, u));
  *patch = {vertices[0], u, v, max(len(e0), fabsf(dot(e1, u))) + 1.0e-7f,
            fabsf(dot(e1, v)) + 1.0e-7f, 1.0f, 1.0f, 1.0f,
            COHERENT_GEOMETRY_REFLECT, 0};
  return true;
}

ccl_device_inline bool coherent_facet_contains(const float3 hit,
                                                const ccl_private float3 vertices[3],
                                                const uint3 indices)
{
  const float3 e0 = vertices[1] - vertices[0], e1 = vertices[2] - vertices[0];
  const float3 area = cross(e0, e1), p = hit - vertices[0];
  const float denominator = dot(area, area);
  const float b1 = dot(cross(p, e1), area) / denominator;
  const float b2 = dot(cross(e0, p), area) / denominator;
  const float b0 = 1.0f - b1 - b2;
  if (b0 < 0.0f || b1 < 0.0f || b2 < 0.0f) return false;
  /* At exact shared-edge hits, consistently oriented adjacent triangles have
   * opposite directed edges. Assign that measure-zero boundary to one facet. */
  return !((b0 == 0.0f && indices.y > indices.z) ||
           (b1 == 0.0f && indices.z > indices.x) ||
           (b2 == 0.0f && indices.x > indices.y));
}

ccl_device_inline bool coherent_facet_connect(const float3 source,
                                               const float3 receiver,
                                               const float3 receiver_normal,
                                               const ccl_private float3 vertices[3],
                                               const uint3 indices,
                                               ccl_private CoherentGeometryInterface *patch,
                                               ccl_private CoherentGeometryPath *path)
{
  if (!coherent_facet_frame(vertices, patch)) return false;
  path->count = 1;
  return coherent_geometry_connect_reflections(source, receiver, receiver_normal, patch, 1, path) &&
         coherent_facet_contains(path->point[0], vertices, indices);
}

/* Conservative side feasibility for a known endpoint and a finite triangle.
 * A triangle that touches/spans the plane is retained. This rejects only
 * impossible same-side reflection topology, never a center-ray occlusion. */
ccl_device_inline bool coherent_facet_same_side_possible(const float3 endpoint,
    const ccl_private CoherentGeometryInterface *plane,
    const ccl_private float3 other[3])
{
  const float3 normal = cross(plane->tangent_u, plane->tangent_v);
  const float d = dot(endpoint-plane->center, normal);
  float lo = FLT_MAX, hi = -FLT_MAX;
  float scale = len(endpoint)+len(plane->center);
  for (int i=0;i<3;i++) {
    const float value = dot(other[i]-plane->center, normal);
    lo=min(lo,value); hi=max(hi,value); scale+=len(other[i]);
  }
  const float epsilon=max(1.0e-8f,16.0f*FLT_EPSILON*scale);
  return !((d > epsilon && hi < -epsilon) || (d < -epsilon && lo > epsilon));
}

ccl_device_inline bool coherent_facet_pair_connect(const float3 source,
    const float3 receiver, const float3 receiver_normal,
    const ccl_private float3 first[3], const uint3 first_indices,
    const ccl_private float3 second[3], const uint3 second_indices,
    ccl_private CoherentGeometryInterface patches[2],
    ccl_private CoherentGeometryPath *path)
{
  if (!coherent_facet_frame(first,&patches[0]) ||
      !coherent_facet_frame(second,&patches[1]) ||
      !coherent_facet_same_side_possible(source,&patches[0],second) ||
      !coherent_facet_same_side_possible(receiver,&patches[1],first)) return false;
  path->count=2;
  return coherent_geometry_connect_reflections(source,receiver,receiver_normal,patches,2,path) &&
         coherent_facet_contains(path->point[0],first,first_indices) &&
         coherent_facet_contains(path->point[1],second,second_indices);
}

struct CoherentStreamComplex {
  float2 real;
  float2 imag;
};
struct CoherentStreamField {
  CoherentStreamComplex total[3][3][3]; /* RGB, independent dipole, detector coordinate. */
  CoherentStreamComplex direct[3][3][3];
};

ccl_device_inline void coherent_stream_clear(ccl_private CoherentStreamField *sum)
{
  for (int c=0;c<3;c++) for (int m=0;m<3;m++) for (int k=0;k<3;k++) {
    sum->total[c][m][k] = {zero_float2(), zero_float2()};
    sum->direct[c][m][k] = {zero_float2(), zero_float2()};
  }
}

/* Shared Gaussian phase realizes exp(-DeltaL^2/(2 Lc^2)) exactly in expectation.
 * This is the existing RGB-envelope/carrier model, not broadband materials. */
ccl_device_inline float2 coherent_stream_phase(const float2 difference,
                                                const float2 wavelength,
                                                const float coherence_length,
                                                const float gaussian,
                                                const float source_phase)
{
#if defined(__clang__)
#  pragma clang fp reassociate(off)
#  pragma clang fp contract(off)
#endif
  float stochastic = 0.0f;
  if (gaussian != 0.0f && fabsf(difference.x) <= coherence_length) {
    /* Small quotients need no modular expansion, including the infinite-Lc limit. */
    stochastic = gaussian*(difference.x/coherence_length + difference.y/coherence_length)*M_1_2PI_F;
  }
  else if (gaussian != 0.0f) {
    /* A two-float effective period retains the small phase remainder before
     * division. The low 2*pi constant is relative to M_2PI_F. */
    float2 period = coherent_geometry_product_split(make_float2(coherence_length, 0.0f), M_2PI_F);
    period = coherent_geometry_add_split(period,
        coherent_geometry_product_split(make_float2(coherence_length, 0.0f), -1.748455600074497e-7f));
    const float g = fabsf(gaussian);
    const float high = period.x / g;
    const float low = (coherent_geometry_fma(-high, g, period.x) + period.y) / g;
    period = coherent_geometry_add_split(make_float2(high, 0.0f), make_float2(low, 0.0f));
    float2 remainder = difference;
    /* Binary-scaled integer period subtraction avoids both enormous quotient
     * overflow and the integer-rounding-to-zero-coherence failure of fmod(q).
     * Twenty bits per step cover the entire normal float exponent range. */
    for (int step = 0; step < 16; step++) {
      if (fabsf(remainder.x) < 0.5f*period.x) break;
      const int shift = max(0, int(floorf(log2(fabsf(remainder.x)))) -
                                int(floorf(log2(period.x))) - 20);
      const float2 scaled = make_float2(ldexpf(period.x, shift), ldexpf(period.y, shift));
      const float q = roundf(remainder.x / scaled.x);
      const float hi = coherent_geometry_fma(-q, scaled.x, remainder.x);
      remainder = coherent_geometry_add_split(make_float2(hi, remainder.y),
          coherent_geometry_product_split(make_float2(-scaled.y, 0.0f), q));
      if (shift == 0) break;
    }
    stochastic = (remainder.x + remainder.y) / period.x;
    if (gaussian < 0.0f) stochastic = -stochastic;
  }
  const float cycles = coherent_phase_cycles_split(difference, wavelength, source_phase) + stochastic;
  /* Reduce around the nearest quarter turn. Exact 0/pi/half-pi inputs then
   * have exact zero components, rather than a sin(pi) cancellation floor. */
  const int quarter = int(roundf(cycles*4.0f));
  const float angle = M_2PI_F*(cycles-0.25f*float(quarter));
  const float c = cosf(angle), s = sinf(angle);
  switch (quarter & 3) {
    case 1: return make_float2(-s,c);
    case 2: return make_float2(-c,-s);
    case 3: return make_float2(s,-c);
    default: return make_float2(c,s);
  }
}

ccl_device_inline void coherent_stream_add_complex(ccl_private CoherentStreamComplex *sum,
                                                   const float2 value)
{
  sum->real = coherent_geometry_add_split(sum->real, make_float2(value.x, 0.0f));
  sum->imag = coherent_geometry_add_split(sum->imag, make_float2(value.y, 0.0f));
}

ccl_device_inline void coherent_stream_add(ccl_private CoherentStreamField *sum,
                                           const ccl_private CoherentCompletedPathField *field,
                                           const float2 anchor,
                                           const float2 wavelength,
                                           const float coherence_length,
                                           const float gaussian,
                                           const bool direct,
                                           const bool scalar)
{
  const float2 difference = coherent_geometry_add_split(field->optical_length_split,
                                                        make_float2(-anchor.x, -anchor.y));
  const float2 phase = coherent_stream_phase(difference, wavelength, coherence_length,
                                             gaussian, field->source_phase_cycles);
  for (int c=0;c<3;c++) {
    const float amplitude = c == 0 ? field->radiance_amplitude_rgb.x :
                            c == 1 ? field->radiance_amplitude_rgb.y :
                                     field->radiance_amplitude_rgb.z;
    for (int m=0;m<(scalar ? 1 : 3);m++) for (int k=0;k<(scalar ? 1 : 3);k++) {
      const float s = k==0 ? field->final_s_detector.x : k==1 ? field->final_s_detector.y : field->final_s_detector.z;
      const float p = k==0 ? field->final_p_detector.x : k==1 ? field->final_p_detector.y : field->final_p_detector.z;
      const float2 vector = scalar ? make_float2(1.0f, 0.0f) :
          field->world_mode[m].s*s + field->world_mode[m].p*p;
      const float2 value = coherent_field_mul(vector, phase)*amplitude;
      coherent_stream_add_complex(&sum->total[c][m][k], value);
      if (direct) coherent_stream_add_complex(&sum->direct[c][m][k], value);
    }
  }
}

ccl_device_inline float3 coherent_stream_finish(const ccl_private CoherentStreamField *sum,
                                                ccl_private float3 *direct)
{
  float result[3]={0.0f,0.0f,0.0f}, share[3]={0.0f,0.0f,0.0f};
  for (int c=0;c<3;c++) for (int m=0;m<3;m++) for (int k=0;k<3;k++) {
    const CoherentStreamComplex a=sum->total[c][m][k], d=sum->direct[c][m][k];
    const float ar=a.real.x+a.real.y, ai=a.imag.x+a.imag.y;
    result[c] += ar*ar+ai*ai;
    /* |D|^2 + Re(D conj(R)) = Re(D conj(D+R)): signed half-cross convention. */
    share[c] += (d.real.x+d.real.y)*ar + (d.imag.x+d.imag.y)*ai;
  }
  *direct=make_float3(share[0],share[1],share[2]);
  return make_float3(result[0],result[1],result[2]);
}
CCL_NAMESPACE_END
