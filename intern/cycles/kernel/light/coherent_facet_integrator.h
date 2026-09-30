/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/light/coherent_facet_stream.h"
#include "kernel/light/triangle.h"
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
    const int prim, ccl_private float3 vertices[3], const float time = -1.0f)
{
  /* Moving objects: the facet at this camera sample's time, exactly as the
   * BVH intersects it. Static objects keep the direct fetch below. */
  if (kernel_data_fetch(object_flag, object) & (SD_OBJECT_MOTION | SD_OBJECT_HAS_VERTEX_MOTION)) {
    triangle_world_space_vertices(kg, object, prim, time, vertices);
    return;
  }
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

/* World vertices in the orientation used by the streamed connector. Glass
 * facets take the native outward winding: negatively scaled objects reverse it,
 * exactly as triangle_normal() does for Ng. Mirrors are two-sided. */
ccl_device_inline void coherent_facet_load(KernelGlobals kg,
    const ccl_global KernelCoherentPatch *mesh, const int prim,
    ccl_private float3 vertices[3], ccl_private uint3 *indices, const float time)
{
  coherent_facet_world_vertices(kg, mesh->object, prim, vertices, time);
  *indices = kernel_data_fetch(tri_vindex, prim);
  if (mesh->mode == 2 && (kernel_data_fetch(object_flag, mesh->object) & SD_OBJECT_NEGATIVE_SCALE)) {
    const float3 v = vertices[1];
    vertices[1] = vertices[2];
    vertices[2] = v;
    *indices = make_uint3(indices->x, indices->z, indices->y);
  }
}

ccl_device_inline bool coherent_facet_budget(IntegratorState state,
    const ccl_global KernelCoherentCandidate *source, const ccl_global KernelLight *light,
    KernelGlobals kg, const int interfaces, const int transmissions)
{
  return coherent_history_candidate_within_budget(source,
      INTEGRATOR_STATE(state,path,bounce),INTEGRATOR_STATE(state,path,glossy_bounce),
      INTEGRATOR_STATE(state,path,transmission_bounce),kernel_data.integrator.max_bounce,
      kernel_data.integrator.max_glossy_bounce,kernel_data.integrator.max_transmission_bounce,
      bool(kernel_data.integrator.use_bidirectional_path_tracing),kernel_data.integrator.bdpt_max_bounces,
      light->max_bounces,interfaces,transmissions);
}

/* Field transport arguments for a streamed interface sequence. */
ccl_device_inline void coherent_facet_materials(const ccl_global KernelCoherentPatch *mesh,
    const int index, ccl_private bool ideal[2], ccl_private bool polarizer[2],
    ccl_private float3 axes[2])
{
  ideal[index] = mesh->mode == 1;
  polarizer[index] = mesh->mode == 2 && mesh->polarizer != 0;
  axes[index] = float3(mesh->polarizer_axis);
}

/* Locate global streamed facet index g (over all declared meshes, in order). */
ccl_device_inline int coherent_facet_locate(KernelGlobals kg, int g, ccl_private int *prim)
{
  for (int index = 0; index < kernel_data.integrator.coherent_patch_count; index++) {
    const int count = kernel_data_fetch(coherent_patches, index).primitive_count;
    if (g < count) {
      *prim = kernel_data_fetch(coherent_patches, index).primitive_offset + g;
      return index;
    }
    g -= count;
  }
  return -1;
}

/* Visibility of every leg of a completed streamed route. Each launch point is
 * offset toward the side its outgoing leg leaves into; each target toward the
 * side its incoming leg arrives from. Any intersection in between blocks. */
ccl_device_inline bool coherent_facet_route_visible(KernelGlobals kg,
    const ccl_private ShaderData *sd, const float3 source,
    const ccl_private CoherentGeometryPath *path,
    const ccl_private CoherentGeometryInterface *patches,
    const ccl_private int *objects, const ccl_private int *prims, const int count)
{
  float3 begin = source;
  int self_object = OBJECT_NONE, self_prim = PRIM_NONE;
  for (int leg = 0; leg <= count; leg++) {
    const bool last = leg == count;
    const float3 end = last ? sd->P : path->point[leg];
    const float3 end_normal = last ? sd->Ng : normalize(cross(patches[leg].tangent_u, patches[leg].tangent_v));
    if (!coherent_facet_segment_visible(kg, sd, begin, end, end_normal, self_object, self_prim)) return false;
    if (last) break;
    const float3 next = leg + 1 == count ? sd->P : path->point[leg + 1];
    begin = ray_offset(end, dot(end_normal, next - end) >= 0.0f ? end_normal : -end_normal);
    self_object = objects[leg];
    self_prim = prims[leg];
  }
  return true;
}

/* Glass roles fix the side a leg arrives from and leaves into (+1 outside,
 * -1 inside); mirrors are two-sided (0). */
ccl_device_inline int coherent_facet_arrive_side(const int role)
{
  return role == COHERENT_FACET_MIRROR ? 0 :
         (role == COHERENT_FACET_GLASS_ENTER || role == COHERENT_FACET_GLASS_EXTERIOR_REFLECT) ? 1 : -1;
}
ccl_device_inline int coherent_facet_depart_side(const int role)
{
  return role == COHERENT_FACET_MIRROR ? 0 :
         (role == COHERENT_FACET_GLASS_EXIT || role == COHERENT_FACET_GLASS_EXTERIOR_REFLECT) ? 1 : -1;
}
/* Some point of the set lies strictly on the given side of the facet plane. */
ccl_device_inline bool coherent_facet_side_reachable(const ccl_private float3 *plane, const int side,
                                                     const ccl_private float3 *points, const int count)
{
  if (side == 0) return true;
  const float3 normal = cross(plane[1] - plane[0], plane[2] - plane[0]);
  for (int i = 0; i < count; i++)
    if (float(side) * dot(points[i] - plane[0], normal) > 0.0f) return true;
  return false;
}
/* A mirror keeps both legs on one side: fail only if the previous and next
 * point sets lie strictly on opposite sides. */
ccl_device_inline bool coherent_facet_mirror_sides(const ccl_private float3 *plane,
    const ccl_private float3 *before, const int nb, const ccl_private float3 *after, const int na)
{
  const float3 normal = cross(plane[1] - plane[0], plane[2] - plane[0]);
  float bmin = FLT_MAX, bmax = -FLT_MAX, amin = FLT_MAX, amax = -FLT_MAX;
  for (int i = 0; i < nb; i++) { const float d = dot(before[i] - plane[0], normal); bmin = min(bmin, d); bmax = max(bmax, d); }
  for (int i = 0; i < na; i++) { const float d = dot(after[i] - plane[0], normal); amin = min(amin, d); amax = max(amax, d); }
  return !((bmin > 0.0f && amax <= 0.0f) || (bmax < 0.0f && amin >= 0.0f));
}

/* Routes of three and four events, enumerated depth first without recursion:
 * mirror and exterior Glass reflections in air, entry into one Glass volume,
 * any number of internal (Fresnel or total) reflections on its facets, and the
 * exit back to air; routes may then continue in air. Costs O(F^events). */
#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device_noinline
#endif
void coherent_facet_stream_long_routes(KernelGlobals kg, IntegratorState state,
    ccl_private ShaderData *sd, const ccl_global KernelCoherentCandidate *source,
    const ccl_global KernelLight *light, const CoherentPathDetectorFrame frame,
    const float3 power, const float3 albedo, const float2 anchor, const float2 wavelength,
    const float gaussian, const bool scalar, ccl_private CoherentStreamField *sum)
{
  const int max_events = min(kernel_data.integrator.coherent_max_interface_events, 4);
  if (max_events < 3) return;
  int total = 0;
  for (int index = 0; index < kernel_data.integrator.coherent_patch_count; index++)
    total += kernel_data_fetch(coherent_patches, index).primitive_count;
  const float3 source_point = float3(light->co);
  int choice[4], patch_of[4], prim_of[4], object_of[4], role[4];
  float3 node_vertices[4][3];
  int medium[5]; /* Coherent patch index of the enclosing Glass, or -1 for air. */
  medium[0] = -1;
  int depth = 0;
  choice[0] = -1;
  while (depth >= 0) {
    choice[depth]++;
    if (choice[depth] >= 2 * total) { depth--; continue; }
    const int transmit = choice[depth] & 1;
    int prim;
    const int patch_index = coherent_facet_locate(kg, choice[depth] >> 1, &prim);
    const ccl_global KernelCoherentPatch *mesh = &kernel_data_fetch(coherent_patches, patch_index);
    const bool glass = mesh->mode == 2;
    int next_medium;
    if (medium[depth] < 0) {
      if (!glass && transmit) continue;
      role[depth] = !glass ? COHERENT_FACET_MIRROR :
                    transmit ? COHERENT_FACET_GLASS_ENTER : COHERENT_FACET_GLASS_EXTERIOR_REFLECT;
      next_medium = glass && transmit ? patch_index : -1;
    }
    else {
      if (patch_index != medium[depth]) continue;
      role[depth] = transmit ? COHERENT_FACET_GLASS_EXIT : COHERENT_FACET_GLASS_INTERIOR_REFLECT;
      next_medium = transmit ? -1 : medium[depth];
    }
    if (depth > 0 && patch_index == patch_of[depth - 1] && prim == prim_of[depth - 1]) continue;
    /* Incremental necessary half-space conditions; they prune whole subtrees. */
    uint3 unused_indices;
    coherent_facet_load(kg, mesh, prim, node_vertices[depth], &unused_indices, sd->time);
    {
      const ccl_private float3 *before = depth == 0 ? &source_point : node_vertices[depth - 1];
      const int nb = depth == 0 ? 1 : 3;
      if (!coherent_facet_side_reachable(node_vertices[depth], coherent_facet_arrive_side(role[depth]), before, nb))
        continue;
      if (depth > 0) {
        if (!coherent_facet_side_reachable(node_vertices[depth - 1], coherent_facet_depart_side(role[depth - 1]),
                                           node_vertices[depth], 3))
          continue;
        if (role[depth - 1] == COHERENT_FACET_MIRROR) {
          const ccl_private float3 *earlier = depth == 1 ? &source_point : node_vertices[depth - 2];
          if (!coherent_facet_mirror_sides(node_vertices[depth - 1], earlier, depth == 1 ? 1 : 3,
                                           node_vertices[depth], 3))
            continue;
        }
      }
    }
    patch_of[depth] = patch_index;
    prim_of[depth] = prim;
    object_of[depth] = mesh->object;
    medium[depth + 1] = next_medium;
    const int count = depth + 1;
    if (count >= 3 && next_medium < 0) {
      int transmissions = 0;
      for (int k = 0; k < count; k++)
        transmissions += role[k] == COHERENT_FACET_GLASS_ENTER || role[k] == COHERENT_FACET_GLASS_EXIT;
      if (coherent_facet_budget(state, source, light, kg, count, transmissions)) {
        CoherentGeometryInterface patches[4];
        float3 vertices[4][3];
        uint3 indices[4];
        bool ideal[4], polarizer[4], feasible = true, all_air_reflections = true;
        float3 axes[4];
        for (int k = 0; k < count && feasible; k++) {
          const ccl_global KernelCoherentPatch *m = &kernel_data_fetch(coherent_patches, patch_of[k]);
          coherent_facet_load(kg, m, prim_of[k], vertices[k], &indices[k], sd->time);
          feasible = coherent_facet_frame(vertices[k], &patches[k]);
          coherent_facet_set_role(&patches[k], role[k], m->inside_ior);
          ideal[k] = m->mode == 1;
          polarizer[k] = m->mode == 2 && m->polarizer != 0;
          axes[k] = float3(m->polarizer_axis);
          all_air_reflections &= role[k] == COHERENT_FACET_MIRROR ||
                                 role[k] == COHERENT_FACET_GLASS_EXTERIOR_REFLECT;
        }
        /* Endpoint side pruning: exterior roles need the source/detector outside. */
        feasible = feasible &&
            (role[0] == COHERENT_FACET_MIRROR || coherent_facet_plane_distance(source_point, &patches[0]) > 0.0f) &&
            (role[count - 1] == COHERENT_FACET_MIRROR || coherent_facet_plane_distance(sd->P, &patches[count - 1]) > 0.0f);
        CoherentGeometryPath path;
        if (feasible) {
          path.count = count;
          feasible = all_air_reflections ?
              coherent_geometry_connect_reflections(source_point, sd->P, sd->N, patches, count, &path) :
              coherent_geometry_connect_unfolded(source_point, sd->P, sd->N, patches, count, &path);
        }
        for (int k = 0; k < count && feasible; k++)
          feasible = coherent_facet_contains(path.point[k], vertices[k], indices[k]);
        CoherentCompletedPathField field;
        if (feasible && coherent_facet_route_visible(kg, sd, source_point, &path, patches, object_of, prim_of, count) &&
            coherent_path_field_transport(source_point, sd->P, frame, &path, patches, ideal, power, albedo,
                                         light->coherence_phase * M_1_2PI_F, &field, axes, polarizer))
        {
          /* Leg k (arriving at event k) lies inside the Glass of medium[k]. */
          for (int k = 1; k < count; k++) {
            if (medium[k] >= 0)
              coherent_stream_attenuate(&field, float3(kernel_data_fetch(coherent_patches, medium[k]).extinction),
                                        path.segment_length[k]);
          }
          coherent_stream_add(sum, &field, anchor, wavelength, light->coherence_length, gaussian, false, scalar);
        }
      }
    }
    /* Descend while another event fits; inside Glass one event must remain for the exit. */
    if (count < max_events && !(next_medium >= 0 && count + 1 > max_events)) {
      depth++;
      choice[depth] = -1;
    }
  }
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
  const int meshes=kernel_data.integrator.coherent_patch_count;
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
      const float3 source_point=float3(light->co);
      const float3 power=spectrum_to_rgb(emission)*(light->spot.eval_fac*M_4PI_F);
      const float2 wavelength=make_float2(light->coherence_wavelength,light->coherence_wavelength_low);
      const float source_phase=light->coherence_phase*M_1_2PI_F;
      CoherentGeometryInterface patch[2];
      CoherentGeometryPath path;
      CoherentCompletedPathField field;
      bool ideal[2], polarizer[2];
      float3 axes[2];
      for (int route=0;route<=kernel_data.integrator.coherent_max_interface_events;route++) {
        if (route==0) {
          if (!coherent_facet_budget(state,source,light,kg,0,0)) continue;
          ideal[0]=ideal[1]=true; polarizer[0]=polarizer[1]=false;
          if (coherent_geometry_connect(source_point,sd->P,sd->N,patch,0,&path) &&
              coherent_facet_segment_visible(kg,sd,source_point,sd->P,sd->Ng,OBJECT_NONE,PRIM_NONE) &&
              coherent_path_field_transport(source_point,sd->P,frame,&path,patch,ideal,power,albedo,
                  source_phase,&field))
            coherent_stream_add(&sum,&field,anchor,wavelength,light->coherence_length,gaussian,true,scalar);
          continue;
        }
        if (route==1) {
          /* One mirror reflection, or one exterior Fresnel reflection on Glass. */
          if (!coherent_facet_budget(state,source,light,kg,1,0)) continue;
          for (int object_index=0;object_index<meshes;object_index++) {
            const ccl_global KernelCoherentPatch *mesh=&kernel_data_fetch(coherent_patches,object_index);
            const int role = mesh->mode==2 ? COHERENT_FACET_GLASS_EXTERIOR_REFLECT : COHERENT_FACET_MIRROR;
            coherent_facet_materials(mesh,0,ideal,polarizer,axes);
            for (int face=0;face<mesh->primitive_count;face++) {
              const int prim=mesh->primitive_offset+face;
              float3 vertices[3]; uint3 indices;
              coherent_facet_load(kg,mesh,prim,vertices,&indices, sd->time);
              if (!coherent_facet_connect_role(source_point,sd->P,sd->N,vertices,indices,role,
                                               mesh->inside_ior,&patch[0],&path)) continue;
              const float3 normal=normalize(cross(patch[0].tangent_u,patch[0].tangent_v));
              if (!coherent_facet_segment_visible(kg,sd,source_point,path.point[0],normal,OBJECT_NONE,PRIM_NONE)) continue;
              const float3 launch=ray_offset(path.point[0],dot(normal,sd->P-path.point[0])>=0.0f ? normal : -normal);
              if (!coherent_facet_segment_visible(kg,sd,launch,sd->P,sd->Ng,mesh->object,prim)) continue;
              if (coherent_path_field_transport(source_point,sd->P,frame,&path,patch,ideal,power,albedo,
                                               source_phase,&field,axes,polarizer))
                coherent_stream_add(&sum,&field,anchor,wavelength,light->coherence_length,gaussian,false,scalar);
            }
          }
          continue;
        }
        if (route >= 3) {
          if (route == 3)
            coherent_facet_stream_long_routes(kg, state, sd, source, light, frame, power, albedo,
                                              anchor, wavelength, gaussian, scalar, &sum);
          continue;
        }
        /* Two events. Ordered reflection pairs on mirrors or exterior Glass
         * sides represent distinct path families; object-qualified identity
         * permits linked instances while excluding a repeated triangle. Entry
         * and exit transmission pairs stay within one Glass object. */
        const bool reflect_budget=coherent_facet_budget(state,source,light,kg,2,0);
        const bool transmit_budget=coherent_facet_budget(state,source,light,kg,2,2);
        if (!reflect_budget && !transmit_budget) continue;
        for (int oi=0;oi<meshes;oi++) {
          const ccl_global KernelCoherentPatch *first_mesh=&kernel_data_fetch(coherent_patches,oi);
          const bool first_glass=first_mesh->mode==2;
          for (int fi=0;fi<first_mesh->primitive_count;fi++) {
            const int first_prim=first_mesh->primitive_offset+fi;
            float3 first_vertices[3]; uint3 first_indices;
            coherent_facet_load(kg,first_mesh,first_prim,first_vertices,&first_indices, sd->time);
            for (int oj=0;oj<meshes;oj++) {
              const ccl_global KernelCoherentPatch *second_mesh=&kernel_data_fetch(coherent_patches,oj);
              const bool same_glass=first_glass && oi==oj;
              if (!reflect_budget && !(transmit_budget && same_glass)) continue;
              for (int fj=0;fj<second_mesh->primitive_count;fj++) {
                const int second_prim=second_mesh->primitive_offset+fj;
                if (first_mesh->object==second_mesh->object && first_prim==second_prim) continue;
                float3 second_vertices[3]; uint3 second_indices;
                coherent_facet_load(kg,second_mesh,second_prim,second_vertices,&second_indices, sd->time);
                for (int transmitted=0;transmitted<2;transmitted++) {
                  if (transmitted ? !(transmit_budget && same_glass) : !reflect_budget) continue;
                  const bool connected = transmitted ?
                      coherent_facet_transmit_connect(source_point,sd->P,sd->N,first_mesh->inside_ior,
                          first_vertices,first_indices,second_vertices,second_indices,patch,&path) :
                      coherent_facet_pair_connect_roles(source_point,sd->P,sd->N,
                          first_vertices,first_indices,
                          first_glass ? COHERENT_FACET_GLASS_EXTERIOR_REFLECT : COHERENT_FACET_MIRROR,
                          first_mesh->inside_ior,
                          second_vertices,second_indices,
                          second_mesh->mode==2 ? COHERENT_FACET_GLASS_EXTERIOR_REFLECT : COHERENT_FACET_MIRROR,
                          second_mesh->inside_ior,patch,&path);
                  if (!connected ||
                      !coherent_facet_pair_visible(kg,sd,source_point,&path,patch,
                          first_mesh->object,first_prim,second_mesh->object,second_prim)) continue;
                  coherent_facet_materials(first_mesh,0,ideal,polarizer,axes);
                  coherent_facet_materials(second_mesh,1,ideal,polarizer,axes);
                  if (coherent_path_field_transport(source_point,sd->P,frame,&path,patch,ideal,
                      power,albedo,source_phase,&field,axes,polarizer)) {
                    if (transmitted)
                      coherent_stream_attenuate(&field,float3(first_mesh->extinction),path.segment_length[1]);
                    coherent_stream_add(&sum,&field,anchor,wavelength,light->coherence_length,gaussian,false,scalar);
                  }
                }
              }
            }
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
