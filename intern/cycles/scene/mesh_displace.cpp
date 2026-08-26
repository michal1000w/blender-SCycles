/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <cmath>
#include <cstdlib>
#include <limits>

#include "device/device.h"

#include "integrator/shader_eval.h"

#include "scene/attribute.h"
#include "scene/devicescene.h"
#include "scene/integrator.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/shader_graph.h"
#include "scene/shader_nodes.h"

#include "util/progress.h"

CCL_NAMESPACE_BEGIN

static int pixel_displacement_cache_sample_count(const int grid)
{
  return (grid + 1) * (grid + 2) / 2;
}

static bool shader_allows_pixel_displacement_cache(const Shader *shader)
{
  /* Deterministic validation hook used by the visual regression tool to exercise the exact
   * fallback with an otherwise identical material graph. */
  if (getenv("CYCLES_PIXEL_DISPLACEMENT_DISABLE_CACHE") != nullptr) {
    return false;
  }

  if (!shader->has_displacement || shader->get_displacement_method() == DISPLACE_BUMP) {
    return false;
  }
  if (shader->has_volume || shader->has_surface_bssrdf || shader->has_surface_raytrace) {
    return false;
  }

  for (const ShaderNode *node : shader->graph->nodes) {
    const ustring node_name = node->type->name;

    if (node->special_type == SHADER_SPECIAL_TYPE_OSL ||
        node->special_type == SHADER_SPECIAL_TYPE_LIGHT_PATH ||
        node->special_type == SHADER_SPECIAL_TYPE_SCENE_TIME)
    {
      return false;
    }

    /* Per-object/path/time inputs can vary outside primitive barycentrics. */
    if (node_name == ustring("object_info")) {
      return false;
    }

    if (node_name == ustring("vector_displacement")) {
      return false;
    }
    if (node_name == ustring("image_texture")) {
      const ImageTextureNode *image_node = static_cast<const ImageTextureNode *>(node);
      if (image_node->get_animated() || image_node->get_projection() != NODE_IMAGE_PROJ_FLAT) {
        return false;
      }
      continue;
    }

    /* Procedural textures can contain arbitrarily high frequencies. Keep them exact. */
    if (node_name == ustring("noise_texture") || node_name == ustring("gabor_texture") ||
        node_name == ustring("voronoi_texture") || node_name == ustring("white_noise_texture") ||
        node_name == ustring("wave_texture") || node_name == ustring("magic_texture") ||
        node_name == ustring("checker_texture") || node_name == ustring("brick_texture") ||
        node_name == ustring("gradient_texture"))
    {
      return false;
    }
  }

  return true;
}

static const Object *single_object_for_mesh(const Scene *scene,
                                            const Mesh *mesh,
                                            int *r_object_index)
{
  const Object *object = nullptr;
  int object_index = OBJECT_NONE;

  for (int i = 0; i < int(scene->objects.size()); i++) {
    const Object *candidate = scene->objects[i];
    if (candidate->get_geometry() != mesh) {
      continue;
    }

    if (object != nullptr) {
      return nullptr;
    }
    object = candidate;
    object_index = i;
  }

  if (r_object_index) {
    *r_object_index = object_index;
  }

  return object;
}

static bool mesh_triangle_uv_extent(const Mesh *mesh, const int triangle, float *r_max_uv_edge)
{
  const Attribute *uv_attr = mesh->attributes.find(ATTR_STD_UV);
  if (uv_attr == nullptr || uv_attr->type != TypeFloat2) {
    return false;
  }

  float2 uv[3];
  if (uv_attr->element == ATTR_ELEMENT_CORNER) {
    if (triangle * 3 + 2 >= uv_attr->size) {
      return false;
    }

    const float2 *uv_data = uv_attr->data<float2>();
    uv[0] = uv_data[triangle * 3 + 0];
    uv[1] = uv_data[triangle * 3 + 1];
    uv[2] = uv_data[triangle * 3 + 2];
  }
  else if (uv_attr->element == ATTR_ELEMENT_VERTEX) {
    const Mesh::Triangle tri = mesh->get_triangle(triangle);
    if (tri.v[0] >= uv_attr->size || tri.v[1] >= uv_attr->size || tri.v[2] >= uv_attr->size) {
      return false;
    }

    const float2 *uv_data = uv_attr->data<float2>();
    uv[0] = uv_data[tri.v[0]];
    uv[1] = uv_data[tri.v[1]];
    uv[2] = uv_data[tri.v[2]];
  }
  else {
    return false;
  }

  const float2 e0 = uv[1] - uv[0];
  const float2 e1 = uv[2] - uv[0];
  const float2 e2 = uv[2] - uv[1];
  const float max_uv_edge = max(max(len(e0), len(e1)), len(e2));

  *r_max_uv_edge = max_uv_edge;
  return true;
}

static int mesh_triangle_cache_grid(const Scene *scene, const Mesh *mesh, const int triangle)
{
  const int shader_index = mesh->get_shader()[triangle];
  const array<Node *> &mesh_used_shaders = mesh->get_used_shaders();
  const Shader *shader = (shader_index < mesh_used_shaders.size()) ?
                             static_cast<const Shader *>(mesh_used_shaders[shader_index]) :
                             scene->default_surface;

  if (!shader_allows_pixel_displacement_cache(shader)) {
    return 0;
  }

  const int resolution = clamp(scene->integrator->get_pixel_displacement_steps(), 8, 128);
  float max_uv_edge;
  if (!mesh_triangle_uv_extent(mesh, triangle, &max_uv_edge)) {
    return resolution;
  }

  /* Interpret the quality setting as samples across the unit UV square, rather than samples per
   * base triangle. This keeps the sampled surface invariant when the same UV domain is split into
   * more or fewer triangles. A unit-square diagonal has length sqrt(2). */
  const float scaled_resolution = float(resolution) * max_uv_edge * 0.7071067811865476f;
  /* UVs produced by regular subdivisions should land on an integer resolution. Avoid ceilf()
   * turning harmless float round-off (for example 8.000001) into a different micromesh density,
   * which would make the rendered surface depend on the base mesh subdivision. */
  return clamp(int(ceilf(scaled_resolution - 1.0e-5f)), 1, 128);
}

bool scene_allows_pixel_displacement_metalrt(const Scene *scene)
{
  constexpr size_t max_cache_samples = 8 * 1024 * 1024;
  size_t total_samples = 0;

  for (const Geometry *geom : scene->geometry) {
    if (!geom->is_mesh() || !geom->has_true_displacement()) {
      continue;
    }

    const Mesh *mesh = static_cast<const Mesh *>(geom);
    if (mesh->get_use_motion_blur()) {
      return false;
    }
    if (single_object_for_mesh(scene, mesh, nullptr) == nullptr) {
      return false;
    }

    const array<int> &triangle_shaders = mesh->get_shader();
    const array<Node *> &used_shaders = mesh->get_used_shaders();
    for (int triangle = 0; triangle < mesh->num_triangles(); triangle++) {
      const int shader_index = triangle_shaders[triangle];
      const Shader *shader = (shader_index < used_shaders.size()) ?
                                 static_cast<const Shader *>(used_shaders[shader_index]) :
                                 scene->default_surface;
      /* The AABB path implements closest, shadow, and shadow-all queries. Meshes which need
       * volume, subsurface, AO, or bevel local intersections retain the existing BVH2 path. This
       * includes regular material slots because the whole Metal BLAS uses one geometry type. */
      if (shader->has_volume || shader->has_surface_bssrdf || shader->has_surface_raytrace) {
        return false;
      }
      if (!shader->has_displacement || shader->get_displacement_method() == DISPLACE_BUMP) {
        continue;
      }

      const int grid = mesh_triangle_cache_grid(scene, mesh, triangle);
      if (grid == 0) {
        return false;
      }
      total_samples += size_t(pixel_displacement_cache_sample_count(grid));
      if (total_samples > max_cache_samples) {
        return false;
      }
    }
  }

  return true;
}

static size_t prepare_pixel_displacement_cache_layout(const Scene *scene,
                                                      device_vector<uint> &cache_grid,
                                                      device_vector<int> &cache_offset,
                                                      size_t *r_cacheable_triangles)
{
  uint *grid_data = cache_grid.data();
  int *offset_data = cache_offset.data();
  size_t total_samples = 0;
  size_t cacheable_triangles = 0;

  for (const Geometry *geom : scene->geometry) {
    if (!geom->is_mesh()) {
      continue;
    }

    const Mesh *mesh = static_cast<const Mesh *>(geom);
    if (!mesh->use_pixel_displacement || single_object_for_mesh(scene, mesh, nullptr) == nullptr) {
      continue;
    }

    for (int triangle = 0; triangle < mesh->num_triangles(); triangle++) {
      const int grid = mesh_triangle_cache_grid(scene, mesh, triangle);
      if (grid == 0) {
        continue;
      }
      const int prim = int(mesh->prim_offset) + triangle;
      grid_data[prim] = uint(grid);
      offset_data[prim] = int(total_samples);
      total_samples += size_t(pixel_displacement_cache_sample_count(grid));
      cacheable_triangles++;
    }
  }

  *r_cacheable_triangles = cacheable_triangles;
  return total_samples;
}

static int fill_pixel_displacement_cache_input(const Scene *scene,
                                               const device_vector<uint> &pixel_displacement_grid,
                                               device_vector<int> &pixel_displacement_offset,
                                               device_vector<KernelShaderEvalInput> &d_input)
{
  KernelShaderEvalInput *d_input_data = d_input.data();
  int *offset_data = pixel_displacement_offset.data();
  int input_size = 0;

  for (const Geometry *geom : scene->geometry) {
    if (!geom->is_mesh()) {
      continue;
    }

    const Mesh *mesh = static_cast<const Mesh *>(geom);
    if (!mesh->use_pixel_displacement) {
      continue;
    }

    int object_index = OBJECT_NONE;
    const Object *object = single_object_for_mesh(scene, mesh, &object_index);
    if (object == nullptr) {
      continue;
    }
    if (object_index == OBJECT_NONE) {
      continue;
    }

    for (int tri = 0; tri < mesh->num_triangles(); tri++) {
      const int prim = int(mesh->prim_offset) + tri;
      const int grid = int(pixel_displacement_grid.data()[prim]);
      if (grid <= 0) {
        continue;
      }

      const int sample_offset = offset_data[prim];
      if (sample_offset != input_size) {
        return 0;
      }

      for (int u = 0; u <= grid; u++) {
        for (int v = 0; v <= grid - u; v++) {
          KernelShaderEvalInput in;
          in.object = object_index;
          in.prim = prim;
          in.u = float(u) / float(grid);
          in.v = float(v) / float(grid);
          d_input_data[input_size++] = in;
        }
      }
    }
  }

  return input_size;
}

static void read_pixel_displacement_cache_output(
    Scene *scene,
    const device_vector<uint> &pixel_displacement_grid,
    const device_vector<int> &pixel_displacement_offset,
    device_vector<float4> &pixel_displacement_data,
    const device_vector<float> &d_output)
{
  float4 *cache_data = pixel_displacement_data.data();
  const float *output_data = d_output.data();
  const float scale = scene->integrator->get_pixel_displacement_scale();
  const float max_distance = max(0.0f, scene->integrator->get_pixel_displacement_max_distance());

  const int num_samples = min(int(pixel_displacement_data.size()), int(d_output.size() / 3));

  for (int i = 0; i < num_samples; i++) {
    float3 D = make_float3(output_data[i * 3 + 0], output_data[i * 3 + 1], output_data[i * 3 + 2]);
    D = ensure_finite(D) * scale;

    const float distance = len(D);
    if (distance > max_distance && distance > 0.0f) {
      D *= max_distance / distance;
    }

    cache_data[i] = make_float4(D.x, D.y, D.z, 0.0f);
  }

  /* Build tight bounds from the same piecewise-linear samples used by intersection. This avoids
   * severe AABB overlap when the base mesh is highly subdivided. */
  for (Geometry *geom : scene->geometry) {
    if (!geom->is_mesh()) {
      continue;
    }
    Mesh *mesh = static_cast<Mesh *>(geom);
    mesh->pixel_displacement_bounds.resize(mesh->num_triangles());
    const packed_float3 *verts = mesh->get_position();

    for (int triangle = 0; triangle < mesh->num_triangles(); triangle++) {
      BoundBox bounds = BoundBox::empty;
      const int prim = int(mesh->prim_offset) + triangle;
      const int grid = int(pixel_displacement_grid.data()[prim]);
      const Mesh::Triangle tri = mesh->get_triangle(triangle);
      const float3 p0 = float3(verts[tri.v[0]]);
      const float3 p1 = float3(verts[tri.v[1]]);
      const float3 p2 = float3(verts[tri.v[2]]);

      if (grid > 0) {
        int sample = pixel_displacement_offset.data()[prim];
        for (int u = 0; u <= grid; u++) {
          for (int v = 0; v <= grid - u; v++, sample++) {
            const float fu = float(u) / float(grid);
            const float fv = float(v) / float(grid);
            bounds.grow(p0 + fu * (p1 - p0) + fv * (p2 - p0) + make_float3(cache_data[sample]));
          }
        }
      }
      else {
        bounds.grow(p0);
        bounds.grow(p1);
        bounds.grow(p2);
      }
      mesh->pixel_displacement_bounds[triangle] = bounds;
    }
  }
}

bool GeometryManager::device_update_pixel_displacement_cache(Device *device,
                                                             DeviceScene *dscene,
                                                             Scene *scene,
                                                             Progress &progress)
{
  device_vector<uint> &cache_info = dscene->pixel_displacement_info;
  device_vector<int> &cache_offset = dscene->pixel_displacement_offset;
  device_vector<float4> &cache_data = dscene->pixel_displacement_data;

  uint *info = cache_info.alloc(dscene->tri_shader.size());
  for (int i = 0; i < cache_info.size(); i++) {
    info[i] = 0;
  }

  int *offsets = cache_offset.alloc(dscene->tri_shader.size());
  for (int i = 0; i < cache_offset.size(); i++) {
    offsets[i] = -1;
  }

  cache_data.free();
  for (Geometry *geom : scene->geometry) {
    if (geom->is_mesh()) {
      static_cast<Mesh *>(geom)->pixel_displacement_bounds.clear();
    }
  }

  size_t cacheable_triangles = 0;
  const size_t total_samples = prepare_pixel_displacement_cache_layout(
      scene, cache_info, cache_offset, &cacheable_triangles);
  constexpr size_t max_cache_samples = 8 * 1024 * 1024;
  if (cacheable_triangles == 0 || total_samples == 0 || total_samples > max_cache_samples) {
    for (int i = 0; i < cache_info.size(); i++) {
      info[i] = 0;
    }
    for (int i = 0; i < cache_offset.size(); i++) {
      offsets[i] = -1;
    }
    cache_info.copy_to_device();
    cache_offset.copy_to_device();
    return false;
  }

  if (total_samples > size_t(std::numeric_limits<int>::max())) {
    cache_info.copy_to_device();
    cache_offset.copy_to_device();
    return false;
  }

  progress.set_status("Updating Mesh", "Computing Pixel Displacement Cache");

  float4 *cache_data_ptr = cache_data.alloc(total_samples);
  for (size_t i = 0; i < total_samples; i++) {
    cache_data_ptr[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  /* The displacement evaluation kernel uses the per-primitive grid to choose a topology-
   * independent texture footprint. Make the layout visible before launching the bake. */
  cache_info.copy_to_device();
  cache_offset.copy_to_device();

  int actual_samples = 0;
  ShaderEval shader_eval(device, progress);
  const bool success = shader_eval.eval(
      SHADER_EVAL_DISPLACE,
      int(total_samples),
      3,
      [scene, &cache_info, &cache_offset, &actual_samples](
          device_vector<KernelShaderEvalInput> &d_input) {
        actual_samples = fill_pixel_displacement_cache_input(
            scene, cache_info, cache_offset, d_input);
        return actual_samples;
      },
      [scene, &cache_info, &cache_offset, &cache_data](const device_vector<float> &d_output) {
        read_pixel_displacement_cache_output(
            scene, cache_info, cache_offset, cache_data, d_output);
      });

  if (!success || actual_samples == 0 || progress.get_cancel()) {
    for (int i = 0; i < cache_info.size(); i++) {
      info[i] = 0;
    }
    for (int i = 0; i < cache_offset.size(); i++) {
      offsets[i] = -1;
    }
    cache_data.free();
    for (Geometry *geom : scene->geometry) {
      if (geom->is_mesh()) {
        static_cast<Mesh *>(geom)->pixel_displacement_bounds.clear();
      }
    }
    cache_info.copy_to_device();
    cache_offset.copy_to_device();
    return false;
  }

  cache_info.copy_to_device();
  cache_offset.copy_to_device();
  cache_data.copy_to_device();

  return true;
}

/* Fill in coordinates for mesh displacement shader evaluation on device. */
static int fill_shader_input(const Scene *scene,
                             const Mesh *mesh,
                             const size_t object_index,
                             device_vector<KernelShaderEvalInput> &d_input)
{
  int d_input_size = 0;
  KernelShaderEvalInput *d_input_data = d_input.data();

  const array<int> &mesh_shaders = mesh->get_shader();
  const array<Node *> &mesh_used_shaders = mesh->get_used_shaders();
  const int num_verts = mesh->num_verts();

  vector<bool> done(num_verts, false);

  const int num_triangles = mesh->num_triangles();
  for (int i = 0; i < num_triangles; i++) {
    const Mesh::Triangle t = mesh->get_triangle(i);
    const int shader_index = mesh_shaders[i];
    Shader *shader = (shader_index < mesh_used_shaders.size()) ?
                         static_cast<Shader *>(mesh_used_shaders[shader_index]) :
                         scene->default_surface;

    if (!shader->has_displacement || shader->get_displacement_method() == DISPLACE_BUMP) {
      continue;
    }

    for (int j = 0; j < 3; j++) {
      if (done[t.v[j]]) {
        continue;
      }

      done[t.v[j]] = true;

      /* set up object, primitive and barycentric coordinates */
      const int object = object_index;
      const int prim = mesh->prim_offset + i;
      float u;
      float v;

      switch (j) {
        case 0:
          u = 0.0f;
          v = 0.0f;
          break;
        case 1:
          u = 1.0f;
          v = 0.0f;
          break;
        default:
          u = 0.0f;
          v = 1.0f;
          break;
      }

      /* back */
      KernelShaderEvalInput in;
      in.object = object;
      in.prim = prim;
      in.u = u;
      in.v = v;
      d_input_data[d_input_size++] = in;
    }
  }

  return d_input_size;
}

/* Read back mesh displacement shader output. */
static void read_shader_output(const Scene *scene,
                               Mesh *mesh,
                               const device_vector<float> &d_output)
{
  const array<int> &mesh_shaders = mesh->get_shader();
  const array<Node *> &mesh_used_shaders = mesh->get_used_shaders();
  packed_float3 *mesh_verts = mesh->get_position_for_write();
  const int num_verts = mesh->num_verts();
  const int num_motion_steps = mesh->get_motion_steps();
  vector<bool> done(num_verts, false);

  const float *d_output_data = d_output.data();
  int d_output_index = 0;

  Attribute *attr_P = mesh->attributes.find(ATTR_STD_POSITION);
  if (!attr_P->has_motion()) {
    attr_P = nullptr;
  }
  const int num_triangles = mesh->num_triangles();
  for (int i = 0; i < num_triangles; i++) {
    const Mesh::Triangle t = mesh->get_triangle(i);
    const int shader_index = mesh_shaders[i];
    Shader *shader = (shader_index < mesh_used_shaders.size()) ?
                         static_cast<Shader *>(mesh_used_shaders[shader_index]) :
                         scene->default_surface;

    if (!shader->has_displacement || shader->get_displacement_method() == DISPLACE_BUMP) {
      continue;
    }

    for (int j = 0; j < 3; j++) {
      if (!done[t.v[j]]) {
        done[t.v[j]] = true;
        float3 off = make_float3(d_output_data[d_output_index + 0],
                                 d_output_data[d_output_index + 1],
                                 d_output_data[d_output_index + 2]);
        d_output_index += 3;

        /* Avoid illegal vertex coordinates. */
        off = ensure_finite(off);
        mesh_verts[t.v[j]] = float3(mesh_verts[t.v[j]]) + off;
        if (attr_P != nullptr) {
          for (int step = 1; step < num_motion_steps; step++) {
            packed_float3 *mP = attr_P->data_for_write<packed_float3>(step);
            mP[t.v[j]] = float3(mP[t.v[j]]) + off;
          }
        }
      }
    }
  }
}

/* Compute unnormalized vertex normals by accumulating face normals from the
 * specified triangles. Vertices not touched by any included triangle are left
 * at zero. */
static void compute_vertex_normals(const Mesh *mesh,
                                   const packed_float3 *verts_data,
                                   const vector<bool> &tri_recompute,
                                   vector<float3> &vN)
{
  const size_t num_triangles = mesh->num_triangles();

  for (size_t i = 0; i < num_triangles; i++) {
    if (tri_recompute[i]) {
      const Mesh::Triangle triangle = mesh->get_triangle(i);
      for (size_t j = 0; j < 3; j++) {
        vN[triangle.v[j]] = zero_float3();
      }
    }
  }

  for (size_t i = 0; i < num_triangles; i++) {
    if (tri_recompute[i]) {
      const Mesh::Triangle triangle = mesh->get_triangle(i);
      const float3 fN = triangle.compute_normal(verts_data);
      for (size_t j = 0; j < 3; j++) {
        vN[triangle.v[j]] += fN;
      }
    }
  }
}

/* Store normalized vertex normals into a packed_normal attribute, applying
 * flip for negative-scaled transforms. Only vertices of included triangles
 * are written. */
static void store_vertex_normals(const Mesh *mesh,
                                 const vector<float3> &vN_float,
                                 const vector<bool> &tri_recompute,
                                 const bool flip,
                                 packed_normal *vN)
{
  const size_t num_verts = mesh->num_verts();
  vector<bool> done(num_verts, false);

  for (size_t i = 0; i < mesh->num_triangles(); i++) {
    if (tri_recompute[i]) {
      const Mesh::Triangle triangle = mesh->get_triangle(i);
      for (size_t j = 0; j < 3; j++) {
        const int vert = triangle.v[j];
        if (done[vert]) {
          continue;
        }

        float3 N = safe_normalize(vN_float[vert]);
        if (flip) {
          N = -N;
        }
        vN[vert] = packed_normal(N);
        done[vert] = true;
      }
    }
  }
}

/* Apply vertex normal delta from displacement to a set of corner normals.
 * For flat shaded triangles, use the new face normal directly. */
static void apply_corner_normal_delta(const Mesh *mesh,
                                      const packed_float3 *verts_data,
                                      const vector<float3> &post_vN,
                                      const float3 *pre_vN,
                                      const vector<bool> &tri_recompute,
                                      const bool flip,
                                      packed_normal *cN)
{
  const bool *smooth = mesh->get_smooth().data();

  for (size_t i = 0; i < mesh->num_triangles(); i++) {
    if (!tri_recompute[i]) {
      continue;
    }
    const Mesh::Triangle triangle = mesh->get_triangle(i);
    if (smooth && smooth[i]) {
      for (size_t j = 0; j < 3; j++) {
        const int vert = triangle.v[j];
        float3 post = safe_normalize(post_vN[vert]);
        if (flip) {
          post = -post;
        }
        const float3 delta = post - pre_vN[vert];
        cN[i * 3 + j] = packed_normal(safe_normalize(cN[i * 3 + j].decode() + delta));
      }
    }
    else {
      float3 post_fN = triangle.compute_normal(verts_data);
      if (flip) {
        post_fN = -post_fN;
      }
      for (size_t j = 0; j < 3; j++) {
        cN[i * 3 + j] = packed_normal(post_fN);
      }
    }
  }
}

/* Save pre-displacement vertex normals so we can compute the delta after
 * displacement and apply it to corner normals. Also saves per motion step. */
static void save_pre_displacement_normals(const Mesh *mesh,
                                          array<float3> &pre_displace_vN,
                                          vector<array<float3>> &pre_displace_motion_vN)
{
  const size_t num_verts = mesh->num_verts();
  const size_t num_triangles = mesh->num_triangles();
  const bool flip = mesh->transform_negative_scaled;
  const vector<bool> all_tris(num_triangles, true);

  auto compute_normals = [&](const packed_float3 *verts_data) {
    array<float3> result;
    result.resize(num_verts, zero_float3());
    vector<float3> vN(num_verts, zero_float3());
    compute_vertex_normals(mesh, verts_data, all_tris, vN);
    for (size_t i = 0; i < num_verts; i++) {
      float3 N = safe_normalize(vN[i]);
      if (flip) {
        N = -N;
      }
      result[i] = N;
    }
    return result;
  };

  pre_displace_vN = compute_normals(mesh->get_position());

  const Attribute *attr_P = mesh->attributes.find(ATTR_STD_POSITION);
  const Attribute *attr_cN = mesh->attributes.find(ATTR_STD_CORNER_NORMAL);
  if (mesh->has_motion_blur() && attr_P->has_motion() && attr_cN && attr_cN->has_motion()) {
    const int num_steps = mesh->get_motion_steps() - 1;
    pre_displace_motion_vN.resize(num_steps);
    for (int attr_step = 1; attr_step <= num_steps; attr_step++) {
      const packed_float3 *mP = attr_P->data<packed_float3>(attr_step);
      pre_displace_motion_vN[attr_step - 1] = compute_normals(mP);
    }
  }
}

/* Update corner normals after displacement, including motion blur steps. */
static void recompute_displaced_corner_normals(Mesh *mesh,
                                               const vector<float3> &vN_float,
                                               const array<float3> &pre_displace_vN,
                                               const vector<array<float3>> &pre_displace_motion_vN,
                                               const vector<bool> &tri_recompute,
                                               const bool flip)
{
  /* Static corner normals. */
  Attribute *attr_cN = mesh->attributes.find(ATTR_STD_CORNER_NORMAL);
  apply_corner_normal_delta(mesh,
                            mesh->get_position(),
                            vN_float,
                            pre_displace_vN.data(),
                            tri_recompute,
                            flip,
                            attr_cN->data_for_write<packed_normal>());

  /* Motion corner normals. */
  Attribute *attr_P = mesh->attributes.find(ATTR_STD_POSITION);

  if (mesh->has_motion_blur() && attr_P->has_motion() && attr_cN && attr_cN->has_motion()) {
    const size_t num_verts = mesh->num_verts();

    for (int attr_step = 1; attr_step < mesh->get_motion_steps(); attr_step++) {
      const packed_float3 *mP = attr_P->data<packed_float3>(attr_step);
      packed_normal *mcN = attr_cN->data_for_write<packed_normal>(attr_step);

      vector<float3> mN_float(num_verts, zero_float3());
      compute_vertex_normals(mesh, mP, tri_recompute, mN_float);

      apply_corner_normal_delta(mesh,
                                mP,
                                mN_float,
                                pre_displace_motion_vN[attr_step - 1].data(),
                                tri_recompute,
                                flip,
                                mcN);
    }
  }
}

/* Update vertex normals after displacement, including motion blur steps. */
static void recompute_displaced_vertex_normals(Mesh *mesh,
                                               const vector<float3> &vN_float,
                                               const vector<bool> &tri_recompute,
                                               const bool flip)
{
  const size_t num_verts = mesh->num_verts();

  /* Static vertex normals. */
  Attribute *attr_vN = mesh->attributes.find(ATTR_STD_VERTEX_NORMAL);
  store_vertex_normals(
      mesh, vN_float, tri_recompute, flip, attr_vN->data_for_write<packed_normal>());

  /* Motion vertex normals. */
  Attribute *attr_P = mesh->attributes.find(ATTR_STD_POSITION);

  if (mesh->has_motion_blur() && attr_P->has_motion() && attr_vN && attr_vN->has_motion()) {
    for (int attr_step = 1; attr_step < mesh->get_motion_steps(); attr_step++) {
      const packed_float3 *mP = attr_P->data<packed_float3>(attr_step);
      packed_normal *mN = attr_vN->data_for_write<packed_normal>(attr_step);

      vector<float3> mN_float(num_verts, zero_float3());
      compute_vertex_normals(mesh, mP, tri_recompute, mN_float);
      store_vertex_normals(mesh, mN_float, tri_recompute, flip, mN);
    }
  }
}

bool GeometryManager::displace(Device *device, Scene *scene, Mesh *mesh, Progress &progress)
{
  /* verify if we have a displacement shader */
  if (!mesh->has_true_displacement()) {
    return false;
  }

  if (mesh->use_pixel_displacement) {
    return false;
  }

  const size_t num_verts = mesh->num_verts();
  const size_t num_triangles = mesh->num_triangles();

  if (num_triangles == 0) {
    return false;
  }

  /* Corner normals for sharp edges and faces should be preserved, but we can not
   * individually displace corners as the mesh would break apart. Instead we
   * compute the delta between vertex normals before and after displacement and
   * apply the delta to corner normals. */
  bool need_recompute_vertex_normals = false;
  bool need_recompute_all_vertex_normals = false;

  const bool has_corner_normals = mesh->attributes.find(ATTR_STD_CORNER_NORMAL) != nullptr;
  array<float3> pre_displace_vN;
  vector<array<float3>> pre_displace_motion_vN;

  if (has_corner_normals) {
    need_recompute_vertex_normals = true;
    need_recompute_all_vertex_normals = true;
    save_pre_displacement_normals(mesh, pre_displace_vN, pre_displace_motion_vN);
  }

  /* Add undisplaced attributes right before doing displacement. */
  mesh->add_undisplaced(scene);

  const string msg = string_printf("Computing Displacement %s", mesh->name.c_str());
  progress.set_status("Updating Mesh", msg);

  /* find object index. todo: is arbitrary */
  size_t object_index = OBJECT_NONE;

  for (size_t i = 0; i < scene->objects.size(); i++) {
    if (scene->objects[i]->get_geometry() == mesh) {
      object_index = i;
      break;
    }
  }

  /* Evaluate shader on device. */
  ShaderEval shader_eval(device, progress);
  if (!shader_eval.eval(
          SHADER_EVAL_DISPLACE,
          num_verts,
          3,
          [scene, mesh, object_index](device_vector<KernelShaderEvalInput> &d_input) {
            return fill_shader_input(scene, mesh, object_index, d_input);
          },
          [scene, mesh](const device_vector<float> &d_output) {
            read_shader_output(scene, mesh, d_output);
          }))
  {
    return false;
  }

  /* For displacement method both, we don't need to recompute the vertex normals
   * as bump mapping in the shader will already alter the vertex normal, so we start
   * from the non-displaced vertex normals to avoid applying the perturbation twice. */
  for (Node *node : mesh->get_used_shaders()) {
    Shader *shader = static_cast<Shader *>(node);
    if (shader->has_displacement && shader->get_displacement_method() == DISPLACE_TRUE) {
      need_recompute_vertex_normals = true;
      break;
    }
  }

  if (need_recompute_vertex_normals) {
    const bool flip = mesh->transform_negative_scaled;
    vector<bool> tri_recompute(num_triangles, need_recompute_all_vertex_normals);

    if (!need_recompute_all_vertex_normals) {
      for (size_t i = 0; i < num_triangles; i++) {
        const int shader_index = mesh->shader[i];
        Shader *shader = (shader_index < mesh->used_shaders.size()) ?
                             static_cast<Shader *>(mesh->used_shaders[shader_index]) :
                             scene->default_surface;

        tri_recompute[i] = shader->has_displacement &&
                           shader->get_displacement_method() == DISPLACE_TRUE;
      }
    }

    vector<float3> vN_float(num_verts, zero_float3());
    compute_vertex_normals(mesh, mesh->get_position(), tri_recompute, vN_float);

    if (has_corner_normals) {
      recompute_displaced_corner_normals(
          mesh, vN_float, pre_displace_vN, pre_displace_motion_vN, tri_recompute, flip);
    }
    else {
      recompute_displaced_vertex_normals(mesh, vN_float, tri_recompute, flip);
    }
  }

  mesh->update_tangents(scene, false);

  return true;
}

CCL_NAMESPACE_END
