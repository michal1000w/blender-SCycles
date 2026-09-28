/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/light/coherent_facet_stream.h"
CCL_NAMESPACE_BEGIN

/* Visibility endpoints receive native robust offsets. Any intersection in the
 * resulting open segment is a blocker; no target-distance tolerance hides it. */
ccl_device_inline bool coherent_facet_segment_visible(KernelGlobals kg,
    const ccl_private ShaderData *sd, const float3 begin, const float3 end,
    const float3 end_normal, const int self_object, const int self_prim)
{
  const float3 target = ray_offset(end, dot(end_normal, begin-end) >= 0.0f ? end_normal : -end_normal);
  const float3 delta = target-begin;
  const float distance = len(delta);
  if (!(distance > 1.0e-7f)) return false;
  Ray ray ccl_optional_struct_init;
  ray.P=begin; ray.D=delta/distance; ray.tmin=0.0f; ray.tmax=distance; ray.time=sd->time;
  ray.self.object=self_object; ray.self.prim=self_prim;
  ray.self.light_object=OBJECT_NONE; ray.self.light_prim=PRIM_NONE;
#ifdef __RAY_DIFFERENTIALS__
  ray.dP=differential_zero_compact(); ray.dD=differential_zero_compact();
#endif
  Intersection hit ccl_optional_struct_init;
  return !scene_intersect(kg, &ray, PATH_RAY_VISIBILITY_SHADOW_OPAQUE, &hit);
}

ccl_device_inline void coherent_facet_world_vertices(KernelGlobals kg, const int object,
    const int prim, ccl_private float3 vertices[3])
{
  triangle_vertices(kg,object,prim,vertices);
  if (!(kernel_data_fetch(object_flag,object)&SD_OBJECT_TRANSFORM_APPLIED)) {
    const Transform tfm=object_fetch_transform(kg,object,OBJECT_TRANSFORM);
    for (int j=0;j<3;j++) vertices[j]=transform_point(&tfm,vertices[j]);
  }
}

ccl_device_inline bool coherent_facet_pair_visible(KernelGlobals kg,
    const ccl_private ShaderData *sd, const float3 source,
    const ccl_private CoherentGeometryPath *path,
    const ccl_private CoherentGeometryInterface patches[2],
    const int first_object, const int first_prim, const int second_object, const int second_prim)
{
  const float3 n0=normalize(cross(patches[0].tangent_u,patches[0].tangent_v));
  const float3 n1=normalize(cross(patches[1].tangent_u,patches[1].tangent_v));
  if (!coherent_facet_segment_visible(kg,sd,source,path->point[0],n0,OBJECT_NONE,PRIM_NONE)) return false;
  const float3 launch0=ray_offset(path->point[0],dot(n0,path->point[1]-path->point[0])>=0 ? n0 : -n0);
  if (!coherent_facet_segment_visible(kg,sd,launch0,path->point[1],n1,first_object,first_prim)) return false;
  const float3 launch1=ray_offset(path->point[1],dot(n1,sd->P-path->point[1])>=0 ? n1 : -n1);
  return coherent_facet_segment_visible(kg,sd,launch1,sd->P,sd->Ng,second_object,second_prim);
}

#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device_noinline
#endif
Spectrum coherent_facet_stream_intensity(KernelGlobals kg, IntegratorState state,
    ccl_private ShaderData *sd, ccl_private Spectrum *primary_direct)
{
  float3 total=zero_float3(), direct_total=zero_float3();
  float3 u,v; make_orthonormals(sd->N,&u,&v);
  const CoherentPathDetectorFrame frame={u,v,sd->N};
  const float3 albedo=spectrum_to_rgb(sd->closure[0].weight);
  const bool scalar=kernel_data.integrator.coherent_polarization_mode==0;
  const int sources=kernel_data.integrator.coherent_candidate_count;
  for (int first=0;first<sources;first++) {
    const ccl_global KernelCoherentCandidate *first_source=&kernel_data_fetch(coherent_candidates,first);
    const ccl_global KernelLight *reference=&kernel_data_fetch(lights,first_source->light);
    bool seen=false;
    for (int earlier=0;earlier<first;earlier++) {
      const int lamp=kernel_data_fetch(coherent_candidates,earlier).light;
      if (kernel_data_fetch(lights,lamp).coherence_group==reference->coherence_group) { seen=true;break; }
    }
    if (seen) continue;
    const uint pixel=INTEGRATOR_STATE(state,path,rng_pixel), sample=INTEGRATOR_STATE(state,path,sample);
    const float r1=max(1.0e-7f,hash_uint3_to_float(pixel,sample,uint(reference->coherence_group)^0xa18531adu));
    const float r2=hash_uint3_to_float(pixel,sample,uint(reference->coherence_group)^0x72bd14cbu);
    const float gaussian=sqrtf(-2.0f*logf(r1))*cosf(M_2PI_F*r2);
    const float2 anchor=coherent_geometry_distance_split(float3(reference->co),sd->P);
    CoherentStreamField sum; coherent_stream_clear(&sum);
    for (int source_index=first;source_index<sources;source_index++) {
      const ccl_global KernelCoherentCandidate *source=&kernel_data_fetch(coherent_candidates,source_index);
      const ccl_global KernelLight *light=&kernel_data_fetch(lights,source->light);
      if (light->coherence_group!=reference->coherence_group) continue;
      Spectrum emission;
      if (!light_sample_shader_eval_nee_constant(kg,light->shader_id,source->light,true,emission)) continue;
      const float3 power=spectrum_to_rgb(emission)*(light->spot.eval_fac*M_4PI_F);
      const float2 wavelength=make_float2(light->coherence_wavelength,light->coherence_wavelength_low);
      const bool mirror[2]={true,true};
      CoherentGeometryInterface patch{};
      CoherentGeometryPath path;
      CoherentCompletedPathField field;
      for (int route=0;route<=kernel_data.integrator.coherent_max_interface_events;route++) {
        if (!coherent_history_candidate_within_budget(source,
            INTEGRATOR_STATE(state,path,bounce),INTEGRATOR_STATE(state,path,glossy_bounce),
            INTEGRATOR_STATE(state,path,transmission_bounce),kernel_data.integrator.max_bounce,
            kernel_data.integrator.max_glossy_bounce,kernel_data.integrator.max_transmission_bounce,
            bool(kernel_data.integrator.use_bidirectional_path_tracing),kernel_data.integrator.bdpt_max_bounces,
            light->max_bounces,route)) continue;
        if (route==0) {
          if (coherent_geometry_connect(float3(light->co),sd->P,sd->N,&patch,0,&path) &&
              coherent_facet_segment_visible(kg,sd,float3(light->co),sd->P,sd->Ng,OBJECT_NONE,PRIM_NONE) &&
              coherent_path_field_transport(float3(light->co),sd->P,frame,&path,&patch,mirror,power,albedo,
                  light->coherence_phase*M_1_2PI_F,&field))
            coherent_stream_add(&sum,&field,anchor,wavelength,light->coherence_length,gaussian,true,scalar);
          continue;
        }
        if (route==2) {
          /* Ordered pairs represent distinct physical path families. Object-qualified
           * identity permits linked instances while excluding a repeated triangle. */
          for (int oi=0;oi<kernel_data.integrator.coherent_patch_count;oi++) {
            const ccl_global KernelCoherentPatch *first_mesh=&kernel_data_fetch(coherent_patches,oi);
            for (int fi=0;fi<first_mesh->primitive_count;fi++) {
              const int first_prim=first_mesh->primitive_offset+fi;
              float3 first_vertices[3];
              coherent_facet_world_vertices(kg,first_mesh->object,first_prim,first_vertices);
              for (int oj=0;oj<kernel_data.integrator.coherent_patch_count;oj++) {
                const ccl_global KernelCoherentPatch *second_mesh=&kernel_data_fetch(coherent_patches,oj);
                for (int fj=0;fj<second_mesh->primitive_count;fj++) {
                  const int second_prim=second_mesh->primitive_offset+fj;
                  if (first_mesh->object==second_mesh->object && first_prim==second_prim) continue;
                  float3 second_vertices[3];
                  coherent_facet_world_vertices(kg,second_mesh->object,second_prim,second_vertices);
                  CoherentGeometryInterface pair[2];
                  if (!coherent_facet_pair_connect(float3(light->co),sd->P,sd->N,
                      first_vertices,kernel_data_fetch(tri_vindex,first_prim),
                      second_vertices,kernel_data_fetch(tri_vindex,second_prim),pair,&path) ||
                      !coherent_facet_pair_visible(kg,sd,float3(light->co),&path,pair,
                          first_mesh->object,first_prim,second_mesh->object,second_prim)) continue;
                  if (coherent_path_field_transport(float3(light->co),sd->P,frame,&path,pair,mirror,
                      power,albedo,light->coherence_phase*M_1_2PI_F,&field))
                    coherent_stream_add(&sum,&field,anchor,wavelength,light->coherence_length,gaussian,false,scalar);
                }
              }
            }
          }
          continue;
        }
        for (int object_index=0;object_index<kernel_data.integrator.coherent_patch_count;object_index++) {
          const ccl_global KernelCoherentPatch *mesh=&kernel_data_fetch(coherent_patches,object_index);
          for (int face=0;face<mesh->primitive_count;face++) {
            const int prim=mesh->primitive_offset+face;
            float3 vertices[3]; triangle_vertices(kg,mesh->object,prim,vertices);
            if (!(kernel_data_fetch(object_flag,mesh->object)&SD_OBJECT_TRANSFORM_APPLIED)) {
              const Transform tfm=object_fetch_transform(kg,mesh->object,OBJECT_TRANSFORM);
              for (int j=0;j<3;j++) vertices[j]=transform_point(&tfm,vertices[j]);
            }
            if (!coherent_facet_connect(float3(light->co),sd->P,sd->N,vertices,
                                        kernel_data_fetch(tri_vindex,prim),&patch,&path)) continue;
            const float3 normal=normalize(cross(patch.tangent_u,patch.tangent_v));
            if (!coherent_facet_segment_visible(kg,sd,float3(light->co),path.point[0],normal,OBJECT_NONE,PRIM_NONE)) continue;
            const float3 launch=ray_offset(path.point[0],dot(normal,sd->P-path.point[0])>=0.0f ? normal : -normal);
            if (!coherent_facet_segment_visible(kg,sd,launch,sd->P,sd->Ng,mesh->object,prim)) continue;
            if (coherent_path_field_transport(float3(light->co),sd->P,frame,&path,&patch,mirror,power,albedo,
                                             light->coherence_phase*M_1_2PI_F,&field))
              coherent_stream_add(&sum,&field,anchor,wavelength,light->coherence_length,gaussian,false,scalar);
          }
        }
      }
    }
    float3 group_direct;
    total+=coherent_stream_finish(&sum,&group_direct); direct_total+=group_direct;
  }
  if (primary_direct) *primary_direct=rgb_to_spectrum(direct_total);
  return rgb_to_spectrum(total);
}
CCL_NAMESPACE_END
