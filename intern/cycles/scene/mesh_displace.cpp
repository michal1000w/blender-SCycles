/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "device/device.h"

#include "integrator/shader_eval.h"

#include "kernel/svm/node_types.h"

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
#include "util/log.h"

CCL_NAMESPACE_BEGIN

static int pixel_displacement_cache_sample_count(const int grid)
{
  return (grid + 1) * (grid + 2) / 2;
}

static int pixel_displacement_cache_sample_index(const int grid, const int u, const int v)
{
  return u * (grid + 1) - (u * (u - 1)) / 2 + v;
}

constexpr uint PIXEL_DISPLACEMENT_CACHE_BVH_FLAG = (1u << 31);
constexpr uint PIXEL_DISPLACEMENT_CACHE_OPEN_SURFACE_FLAG = (1u << 30);
constexpr uint PIXEL_DISPLACEMENT_CACHE_GRID_MASK = (1u << 30) - 1;
constexpr int PIXEL_DISPLACEMENT_CACHE_BLOCK_SIZE = 2;
constexpr size_t PIXEL_DISPLACEMENT_MAX_CACHE_SAMPLES = 8 * 1024 * 1024;

struct PixelDisplacementBVHBlock {
  BoundBox bounds;
  int u;
  int v;
};

struct PixelDisplacementBVHNode {
  BoundBox bounds;
  int child0;
  int child1;
  int u;
  int v;
  bool leaf;
};

static bool mesh_is_closed_surface(const Mesh *mesh)
{
  vector<uint64_t> edges;
  edges.reserve(mesh->num_triangles() * 3);
  for (int triangle = 0; triangle < mesh->num_triangles(); triangle++) {
    const Mesh::Triangle tri = mesh->get_triangle(triangle);
    for (int edge = 0; edge < 3; edge++) {
      const uint v0 = uint(min(tri.v[edge], tri.v[(edge + 1) % 3]));
      const uint v1 = uint(max(tri.v[edge], tri.v[(edge + 1) % 3]));
      edges.push_back((uint64_t(v0) << 32) | uint64_t(v1));
    }
  }

  std::sort(edges.begin(), edges.end());
  for (size_t begin = 0; begin < edges.size();) {
    size_t end = begin + 1;
    while (end < edges.size() && edges[end] == edges[begin]) {
      end++;
    }
    if (end - begin != 2) {
      return false;
    }
    begin = end;
  }
  return !edges.empty();
}

static int build_pixel_displacement_bvh_recursive(std::vector<PixelDisplacementBVHBlock> &blocks,
                                                  const int begin,
                                                  const int end,
                                                  std::vector<PixelDisplacementBVHNode> &nodes)
{
  BoundBox bounds = BoundBox::empty;
  BoundBox centroids = BoundBox::empty;
  for (int i = begin; i < end; i++) {
    bounds.grow(blocks[i].bounds);
    centroids.grow(blocks[i].bounds.center());
  }

  const int node_index = int(nodes.size());
  nodes.push_back({bounds, -1, -1, 0, 0, false});
  if (end - begin == 1) {
    nodes[node_index].u = blocks[begin].u;
    nodes[node_index].v = blocks[begin].v;
    nodes[node_index].leaf = true;
    return node_index;
  }

  const float3 extent = centroids.max - centroids.min;
  const int axis = (extent.x >= extent.y && extent.x >= extent.z) ? 0 :
                   (extent.y >= extent.z)                         ? 1 :
                                                                    2;
  const int middle = (begin + end) / 2;
  std::nth_element(blocks.begin() + begin,
                   blocks.begin() + middle,
                   blocks.begin() + end,
                   [axis](const PixelDisplacementBVHBlock &a, const PixelDisplacementBVHBlock &b) {
                     return a.bounds.center()[axis] < b.bounds.center()[axis];
                   });
  const int child0 = build_pixel_displacement_bvh_recursive(blocks, begin, middle, nodes);
  const int child1 = build_pixel_displacement_bvh_recursive(blocks, middle, end, nodes);
  nodes[node_index].child0 = child0;
  nodes[node_index].child1 = child1;
  return node_index;
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
  if (shader->has_volume || shader->has_surface_raytrace) {
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

    /* Static procedural textures are deterministic functions of the shading coordinates and can
     * be sampled into the same piecewise-linear micromesh as image textures. The user-controlled
     * micromesh resolution is the spatial bandwidth limit; inputs which can change with the ray,
     * object, or time are rejected above. */
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
  /* An unbounded procedural shader has no finite cache resolution. Evaluate it at ray hits
   * instead of imposing a hidden sampling limit or allocating an unbounded micromesh. */
  if (!scene->integrator->get_use_pixel_displacement_resolution_clamp()) {
    return 0;
  }

  const int shader_index = mesh->get_shader()[triangle];
  const array<Node *> &mesh_used_shaders = mesh->get_used_shaders();
  const Shader *shader = (shader_index < mesh_used_shaders.size()) ?
                             static_cast<const Shader *>(mesh_used_shaders[shader_index]) :
                             scene->default_surface;

  if (!shader_allows_pixel_displacement_cache(shader)) {
    return 0;
  }

  const int resolution = clamp(scene->integrator->get_pixel_displacement_resolution(), 64, 16384);
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
  if (!std::isfinite(scaled_resolution)) {
    return 0;
  }
  /* Clamp before converting to int, including for very large but finite UV coordinates. */
  return int(clamp(ceilf(scaled_resolution - 1.0e-5f), 1.0f, float(resolution)));
}

bool scene_allows_pixel_displacement_metalrt(const Scene *scene)
{
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
      /* The AABB path implements closest, shadow, shadow-all, and subsurface local queries.
       * Meshes which need volume, AO, or bevel local intersections retain the BVH2 path. This
       * includes regular material slots because the whole Metal BLAS uses one geometry type. */
      if (shader->has_volume || shader->has_surface_raytrace) {
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
      if (total_samples > PIXEL_DISPLACEMENT_MAX_CACHE_SAMPLES) {
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
  *r_cacheable_triangles = 0;
  if (!scene->integrator->get_use_pixel_displacement_resolution_clamp()) {
    return 0;
  }

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
    const uint surface_flag = mesh_is_closed_surface(mesh) ?
                                  0u :
                                  PIXEL_DISPLACEMENT_CACHE_OPEN_SURFACE_FLAG;

    for (int triangle = 0; triangle < mesh->num_triangles(); triangle++) {
      const int grid = mesh_triangle_cache_grid(scene, mesh, triangle);
      if (grid == 0) {
        continue;
      }
      const size_t samples = size_t(pixel_displacement_cache_sample_count(grid));
      /* Stop before writing offsets that cannot fit the bounded cache. The caller clears the
       * partial layout and uses direct shader evaluation for the entire scene. */
      if (samples > PIXEL_DISPLACEMENT_MAX_CACHE_SAMPLES - total_samples) {
        return PIXEL_DISPLACEMENT_MAX_CACHE_SAMPLES + 1;
      }
      const int prim = int(mesh->prim_offset) + triangle;
      grid_data[prim] = uint(grid) | surface_flag;
      offset_data[prim] = int(total_samples);
      total_samples += samples;
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
      const int grid = int(pixel_displacement_grid.data()[prim] &
                           PIXEL_DISPLACEMENT_CACHE_GRID_MASK);
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
    device_vector<uint> &pixel_displacement_grid,
    const device_vector<int> &pixel_displacement_offset,
    device_vector<float4> &pixel_displacement_data,
    device_vector<int> &pixel_displacement_bvh_offset,
    device_vector<float4> &pixel_displacement_bvh_nodes,
    const device_vector<float> &d_output)
{
  float4 *cache_data = pixel_displacement_data.data();
  const float *output_data = d_output.data();
  const float scale = scene->integrator->get_pixel_displacement_scale();
  const float max_distance = max(0.0f, scene->integrator->get_pixel_displacement_max_distance());

  const int num_samples = min(int(pixel_displacement_data.size()), int(d_output.size() / 3));
  std::vector<PixelDisplacementBVHNode> bvh_nodes;

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
      const int grid = int(pixel_displacement_grid.data()[prim] &
                           PIXEL_DISPLACEMENT_CACHE_GRID_MASK);
      const Mesh::Triangle tri = mesh->get_triangle(triangle);
      const float3 p0 = float3(verts[tri.v[0]]);
      const float3 p1 = float3(verts[tri.v[1]]);
      const float3 p2 = float3(verts[tri.v[2]]);

      if (grid > 0) {
        int sample = pixel_displacement_offset.data()[prim];
        const int sample_offset = sample;
        const float3 face_normal = safe_normalize(cross(p1 - p0, p2 - p0));
        float min_height = std::numeric_limits<float>::infinity();
        float max_height = -std::numeric_limits<float>::infinity();
        float max_tangent_squared = 0.0f;
        for (int u = 0; u <= grid; u++) {
          for (int v = 0; v <= grid - u; v++, sample++) {
            const float fu = float(u) / float(grid);
            const float fv = float(v) / float(grid);
            const float3 displacement = make_float3(cache_data[sample]);
            bounds.grow(p0 + fu * (p1 - p0) + fv * (p2 - p0) + displacement);
            const float height = dot(displacement, face_normal);
            min_height = min(min_height, height);
            max_height = max(max_height, height);
            max_tangent_squared = max(max_tangent_squared,
                                      len_squared(displacement - height * face_normal));
          }
        }
        /* The fourth component is unused by displacement vectors. Keep tight height bounds in the
         * first two samples so intersection can clip the ray slab without another device array. */
        constexpr float height_epsilon = 1.0e-6f;
        cache_data[sample_offset].w = min_height - height_epsilon;
        cache_data[sample_offset + 1].w = max_height + height_epsilon;

        /* Grid DDA is exact only when displacement is parallel to the base-triangle normal. On
         * smooth or curved geometry the shading normal moves microtriangles tangentially, so build
         * an object-space block BVH and intersect the actual displaced microtriangles instead. */
        /* Shader evaluation and normal transforms can leave sub-ULP tangential residue even when
         * the displacement is mathematically parallel to the face. Keep that numerical noise on
         * the DDA path; meaningful smooth-normal displacement is orders of magnitude larger. */
        const float tangent_epsilon = max(1.0e-5f, max_distance * 1.0e-4f);
        if (max_tangent_squared > sqr(tangent_epsilon)) {
          std::vector<PixelDisplacementBVHBlock> blocks;
          for (int block_u = 0; block_u < grid; block_u += PIXEL_DISPLACEMENT_CACHE_BLOCK_SIZE) {
            for (int block_v = 0; block_u + block_v < grid;
                 block_v += PIXEL_DISPLACEMENT_CACHE_BLOCK_SIZE)
            {
              BoundBox block_bounds = BoundBox::empty;
              const int end_u = min(block_u + PIXEL_DISPLACEMENT_CACHE_BLOCK_SIZE, grid);
              for (int u = block_u; u <= end_u; u++) {
                const int end_v = min(block_v + PIXEL_DISPLACEMENT_CACHE_BLOCK_SIZE, grid - u);
                for (int v = block_v; v <= end_v; v++) {
                  const float fu = float(u) / float(grid);
                  const float fv = float(v) / float(grid);
                  const int index = sample_offset +
                                    pixel_displacement_cache_sample_index(grid, u, v);
                  block_bounds.grow(p0 + fu * (p1 - p0) + fv * (p2 - p0) +
                                    make_float3(cache_data[index]));
                }
              }
              block_bounds.grow(block_bounds.min, 1.0e-6f);
              block_bounds.grow(block_bounds.max, 1.0e-6f);
              blocks.push_back({block_bounds, block_u, block_v});
            }
          }

          if (!blocks.empty()) {
            pixel_displacement_bvh_offset.data()[prim] = int(bvh_nodes.size());
            build_pixel_displacement_bvh_recursive(blocks, 0, int(blocks.size()), bvh_nodes);
            pixel_displacement_grid.data()[prim] |= PIXEL_DISPLACEMENT_CACHE_BVH_FLAG;
          }
        }

        constexpr int block_size = 8;
        if (grid >= block_size) {
          for (int block_u = 0; block_u < grid; block_u += block_size) {
            for (int block_v = 0; block_u + block_v < grid; block_v += block_size) {
              float block_min_height = std::numeric_limits<float>::infinity();
              float block_max_height = -std::numeric_limits<float>::infinity();
              const int end_u = min(block_u + block_size, grid);
              for (int u = block_u; u <= end_u; u++) {
                const int end_v = min(block_v + block_size, grid - u);
                for (int v = block_v; v <= end_v; v++) {
                  const int index = sample_offset +
                                    pixel_displacement_cache_sample_index(grid, u, v);
                  const float height = dot(make_float3(cache_data[index]), face_normal);
                  block_min_height = min(block_min_height, height);
                  block_max_height = max(block_max_height, height);
                }
              }

              const int bounds_index = (block_u == 0 && block_v == 0) ?
                                           sample_offset + 2 :
                                           sample_offset + pixel_displacement_cache_sample_index(
                                                               grid, block_u, block_v);
              cache_data[bounds_index].w = block_min_height - height_epsilon;
              cache_data[bounds_index + 1].w = block_max_height + height_epsilon;
            }
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

  pixel_displacement_bvh_nodes.free();
  if (!bvh_nodes.empty()) {
    float4 *serialized = pixel_displacement_bvh_nodes.alloc(bvh_nodes.size() * 2);
    for (int i = 0; i < int(bvh_nodes.size()); i++) {
      const PixelDisplacementBVHNode &node = bvh_nodes[i];
      const uint meta0 = node.leaf ? (PIXEL_DISPLACEMENT_CACHE_BVH_FLAG | uint(node.u)) :
                                     uint(node.child0);
      const uint meta1 = node.leaf ? uint(node.v) : uint(node.child1);
      serialized[i * 2] = make_float4(
          node.bounds.min.x, node.bounds.min.y, node.bounds.min.z, __uint_as_float(meta0));
      serialized[i * 2 + 1] = make_float4(
          node.bounds.max.x, node.bounds.max.y, node.bounds.max.z, __uint_as_float(meta1));
    }
  }
}

/* The image fast path is restricted to constant native triangle normals. Varying
 * normals amplify arithmetic differences in the lean evaluator into visible shading changes. */
static bool pixel_displacement_has_uniform_normals(const Mesh *mesh, const int triangle)
{
  const Attribute *normals = mesh->attributes.find(ATTR_STD_CORNER_NORMAL);
  const bool corner_normals = normals != nullptr;
  if (!normals) {
    normals = mesh->attributes.find(ATTR_STD_VERTEX_NORMAL);
  }
  if (!normals) {
    return false;
  }
  const Mesh::Triangle tri = mesh->get_triangle(triangle);
  const packed_normal *values = normals->data<packed_normal>();
  const int count = normals->size;
  if (!values || count <= 0) {
    return false;
  }
  int indices[3];
  for (int corner = 0; corner < 3; corner++) {
    indices[corner] = corner_normals ? triangle * 3 + corner : tri.v[corner];
    if (indices[corner] < 0 || indices[corner] >= count) {
      return false;
    }
  }
  return values[indices[0]] == values[indices[1]] && values[indices[0]] == values[indices[2]];
}

/* Cache lookup metadata only. The original GPU evaluator still fetches and interpolates
 * native values, preserving its floating-point arithmetic and conversion paths. */
static bool pixel_displacement_cache_attribute(DeviceScene *dscene,
                                               const Shader *shader,
                                               const int object,
                                               const int prim,
                                               float4 &descriptor)
{
  const int program_offset = shader->displacement_image_offset;
  if (program_offset < 0 || size_t(program_offset) > dscene->svm_nodes.size() ||
      sizeof(SVMDisplacementImage) / sizeof(uint) >
          dscene->svm_nodes.size() - size_t(program_offset) ||
      object < 0 || size_t(object) >= dscene->objects.size() || prim < 0 ||
      size_t(prim) >= dscene->tri_vindex.size())
  {
    return false;
  }
  SVMDisplacementImage program;
  memcpy(&program, dscene->svm_nodes.data() + program_offset, sizeof(program));
  if (program.attribute.output_type != NODE_ATTR_OUTPUT_FLOAT3) {
    return false;
  }

  size_t map_offset = dscene->objects[object].attribute_map_offset;
  /* Mirror the kernel's attribute-map chaining, with bounds/cycle protection. */
  for (size_t visited = 0; visited < dscene->attributes_map.size(); visited++) {
    if (map_offset >= dscene->attributes_map.size()) {
      return false;
    }
    const AttributeMap &entry = dscene->attributes_map[map_offset];
    if (entry.id != program.attribute.attr) {
      if (entry.id == ATTR_STD_NONE) {
        if (entry.element == ATTR_ELEMENT_NONE) {
          return false;
        }
        map_offset = entry.offset;
      }
      else {
        map_offset += ATTR_PRIM_TYPES;
      }
      continue;
    }
    if ((entry.element != ATTR_ELEMENT_VERTEX && entry.element != ATTR_ELEMENT_CORNER) ||
        (entry.type != NODE_ATTR_FLOAT2 && entry.type != NODE_ATTR_FLOAT3))
    {
      return false;
    }
    const uint3 indices = dscene->tri_vindex[prim];
    for (int corner = 0; corner < 3; corner++) {
      const int64_t index = entry.element == ATTR_ELEMENT_CORNER ?
                                int64_t(prim) * 3 + corner :
                                (corner == 0 ? indices.x : (corner == 1 ? indices.y : indices.z));
      const int64_t offset = int64_t(entry.offset) + index;
      if (offset < 0) {
        return false;
      }
      if (entry.type == NODE_ATTR_FLOAT2) {
        if (uint64_t(offset) >= dscene->attributes_float2.size()) {
          return false;
        }
      }
      else {
        if (uint64_t(offset) >= dscene->attributes_float3.size()) {
          return false;
        }
      }
    }
    descriptor = make_float4(float(entry.element),
                             float(entry.type),
                             float(uint(entry.offset) & 65535u),
                             float(uint(entry.offset) >> 16));
    return true;
  }
  return false;
}

/* Accelerate the existing finite grazing fallback. No additional shader samples are
 * created; the combined samples and bounds remain inside the fallback's 16 MiB budget. */
static void build_pixel_displacement_patch_bvh(DeviceScene *dscene,
                                               Scene *scene,
                                               const int grids[3],
                                               const int num_grids)
{
  const size_t budget = 16 * 1024 * 1024;
  const size_t sample_bytes = dscene->pixel_displacement_data.size() * sizeof(float4);
  if (sample_bytes >= budget) {
    return;
  }
  const size_t max_nodes = (budget - sample_bytes) / (2 * sizeof(float4));
  std::vector<PixelDisplacementBVHNode> nodes;
  vector<pair<int, int3>> headers;
  const float scale = scene->integrator->get_pixel_displacement_scale();
  const float max_distance = scene->integrator->get_pixel_displacement_max_distance();
  for (Geometry *geom : scene->geometry) {
    if (!geom->is_mesh()) {
      continue;
    }
    const Mesh *mesh = static_cast<const Mesh *>(geom);
    int object_index;
    if (!single_object_for_mesh(scene, mesh, &object_index)) {
      continue;
    }
    const packed_float3 *verts = mesh->get_position();
    for (int triangle = 0; triangle < mesh->num_triangles(); triangle++) {
      const int prim = int(mesh->prim_offset) + triangle;
      const uint info = dscene->pixel_displacement_info[prim];
      if (!(info & (1u << 29)) || dscene->pixel_displacement_offset[prim] < 0) {
        continue;
      }
      /* All three bounded grids together need fewer than 256 nodes. */
      if (nodes.size() + 256 > max_nodes) {
        continue;
      }
      const int header = int(nodes.size());
      nodes.push_back({BoundBox::empty, 0, 0, 0, 0, false});
      int3 roots = make_int3(-1, -1, -1);
      int sample_offset = dscene->pixel_displacement_offset[prim] + ((info & (1u << 27)) ? 1 : 0);
      const Mesh::Triangle tri = mesh->get_triangle(triangle);
      const float3 p0 = float3(verts[tri.v[0]]);
      const float3 e0 = float3(verts[tri.v[1]]) - p0;
      const float3 e1 = float3(verts[tri.v[2]]) - p0;
      for (int slot = 0; slot < num_grids; slot++) {
        const int grid = grids[slot];
        const float inv_grid = 1.0f / float(grid);
        std::vector<PixelDisplacementBVHBlock> blocks;
        for (int u0 = 0; u0 < grid; u0 += 2) {
          for (int v0 = 0; u0 + v0 < grid; v0 += 2) {
            BoundBox bounds = BoundBox::empty;
            for (int u = u0; u <= min(u0 + 2, grid); u++) {
              for (int v = v0; v <= min(v0 + 2, grid - u); v++) {
                const int index = sample_offset +
                                  pixel_displacement_cache_sample_index(grid, u, v);
                float3 D = make_float3(dscene->pixel_displacement_data[index]) * scale;
                const float distance = len(D);
                if (distance > max_distance && distance > 0.0f) {
                  D *= max_distance / distance;
                }
                bounds.grow(p0 + (float(u) * inv_grid) * e0 + (float(v) * inv_grid) * e1 +
                            ensure_finite(D));
              }
            }
            /* Enclose the microtriangle edge tolerance as well as CPU/GPU roundoff. */
            const float margin = 0.002f * max(1.0f, len(bounds.max - bounds.min)) +
                                 128.0f * FLT_EPSILON * max(len(bounds.min), len(bounds.max));
            bounds.min -= make_float3(margin);
            bounds.max += make_float3(margin);
            blocks.push_back({bounds, u0, v0});
          }
        }
        roots[slot] = int(nodes.size());
        build_pixel_displacement_bvh_recursive(blocks, 0, int(blocks.size()), nodes);
        sample_offset += pixel_displacement_cache_sample_count(grid);
      }
      headers.push_back({header, roots});
      dscene->pixel_displacement_bvh_offset[prim] = header;
    }
  }
  if (nodes.empty()) {
    return;
  }
  float4 *data = dscene->pixel_displacement_bvh_nodes.alloc(nodes.size() * 2);
  for (size_t i = 0; i < nodes.size(); i++) {
    const auto &node = nodes[i];
    data[i * 2] = make_float4(
        node.bounds.min.x,
        node.bounds.min.y,
        node.bounds.min.z,
        __uint_as_float(node.leaf ? (0x80000000u | uint(node.u)) : uint(node.child0)));
    data[i * 2 + 1] = make_float4(node.bounds.max.x,
                                  node.bounds.max.y,
                                  node.bounds.max.z,
                                  __uint_as_float(node.leaf ? uint(node.v) : uint(node.child1)));
  }
  for (const auto &header : headers) {
    data[header.first * 2] = make_float4(
        float(header.second.x), float(header.second.y), float(header.second.z), 0.0f);
  }
  dscene->pixel_displacement_bvh_nodes.copy_to_device();
  dscene->pixel_displacement_bvh_offset.copy_to_device();
  LOG_DEBUG << "Displacement fallback BVH: " << headers.size() << " triangles, " << nodes.size()
            << " nodes, " << sample_bytes + nodes.size() * 2 * sizeof(float4)
            << " combined sample and bounds bytes";
}

static void build_pixel_displacement_patch_cache(Device *device,
                                                 DeviceScene *dscene,
                                                 Scene *scene,
                                                 Progress &progress)
{
  if (getenv("CYCLES_PIXEL_DISPLACEMENT_DISABLE_PATCH_CACHE") != nullptr) {
    return;
  }
  constexpr uint patch_cache_flag = 1u << 29;
  constexpr size_t max_patch_samples = 1024 * 1024;
  const bool compare_image = getenv("CYCLES_PIXEL_DISPLACEMENT_COMPARE_FUSED_IMAGE") != nullptr;
  const bool cache_inputs = !compare_image &&
                            getenv("CYCLES_PIXEL_DISPLACEMENT_DISABLE_INPUT_CACHE") == nullptr;
  const int steps = clamp(scene->integrator->get_pixel_displacement_steps(), 8, 128);
  int grids[3];
  int num_grids = 0;
  uint packed_grids = patch_cache_flag;
  size_t samples_per_triangle = 0;
  for (int factor = 1; factor <= 4; factor *= 2) {
    const int grid = clamp(int(ceilf(sqrtf(float(min(steps * factor, 128)))) + 2), 6, 14);
    if (num_grids > 0 && grid == grids[num_grids - 1]) {
      continue;
    }
    packed_grids |= uint(grid) << (4 * num_grids);
    grids[num_grids++] = grid;
    samples_per_triangle += size_t(pixel_displacement_cache_sample_count(grid));
  }
  if (compare_image) {
    samples_per_triangle = 4096;
  }

  struct PatchTriangle {
    int object;
    int prim;
    bool cache_attribute;
    float4 descriptor;
  };
  vector<PatchTriangle> triangles;
  size_t total_samples = 0;
  for (const Geometry *geom : scene->geometry) {
    if (!geom->is_mesh()) {
      continue;
    }
    const Mesh *mesh = static_cast<const Mesh *>(geom);
    if (!mesh->use_pixel_displacement || mesh->get_use_motion_blur()) {
      continue;
    }
    int object_index;
    const Object *object = single_object_for_mesh(scene, mesh, &object_index);
    if (object == nullptr || object->use_motion()) {
      continue;
    }
    for (int triangle = 0; triangle < mesh->num_triangles(); triangle++) {
      const int shader_index = mesh->get_shader()[triangle];
      const array<Node *> &shaders = mesh->get_used_shaders();
      const Shader *shader = (shader_index >= 0 && size_t(shader_index) < shaders.size()) ?
                                 static_cast<Shader *>(shaders[shader_index]) :
                                 scene->default_surface;
      if (!shader_allows_pixel_displacement_cache(shader) ||
          (compare_image && shader->displacement_image_offset < 0) ||
          samples_per_triangle > max_patch_samples - total_samples)
      {
        continue;
      }
      const int prim = int(mesh->prim_offset) + triangle;
      float4 descriptor = zero_float4();
      const bool cache_attribute = cache_inputs &&
                                   pixel_displacement_has_uniform_normals(mesh, triangle) &&
                                   pixel_displacement_cache_attribute(
                                       dscene, shader, object_index, prim, descriptor) &&
                                   uint(descriptor.y) == NODE_ATTR_FLOAT2;
      const size_t triangle_samples = samples_per_triangle + (cache_attribute ? 1 : 0);
      if (triangle_samples > max_patch_samples - total_samples) {
        continue;
      }
      dscene->pixel_displacement_info[prim] = packed_grids | (cache_attribute ? (3u << 27) : 0);
      dscene->pixel_displacement_offset[prim] = int(total_samples);
      total_samples += triangle_samples;
      triangles.push_back({object_index, prim, cache_attribute, descriptor});
    }
  }
  if (triangles.empty()) {
    if (compare_image) {
      progress.set_error("No eligible fused image program for the diagnostic comparison");
    }
    return;
  }

  /* Store raw object-space displacement. Integrator settings are uploaded after this bake;
   * scale and distance clamping are applied by the ray query, using its current settings. */
  progress.set_status("Updating Mesh", "Caching Displacement Fallback Samples");
  ShaderEval shader_eval(device, progress);
  const bool success = shader_eval.eval(
      SHADER_EVAL_DISPLACE,
      int(triangles.size() * samples_per_triangle),
      3,
      [&triangles, &grids, num_grids, compare_image](device_vector<KernelShaderEvalInput> &input) {
        int count = 0;
        for (const PatchTriangle &triangle : triangles) {
          if (compare_image) {
            for (int sample = 0; sample < 4096; sample++) {
              float u = (float(sample & 63) + 0.37f) / 64.0f;
              float v = (float(sample >> 6) + 0.61f) / 64.0f;
              if (u + v > 1.0f) {
                u = 1.0f - u;
                v = 1.0f - v;
              }
              input[count++] = {~triangle.object, ~triangle.prim, u, v};
            }
            continue;
          }
          for (int slot = 0; slot < num_grids; slot++) {
            const int grid = grids[slot];
            for (int u = 0; u <= grid; u++) {
              for (int v = 0; u + v <= grid; v++) {
                /* Negative primitive IDs request the direct shader setup in the Metal bake
                 * kernel. UVs encode integer patch coordinates to preserve GPU rounding. */
                input[count++] = {compare_image ? ~triangle.object : triangle.object,
                                  ~triangle.prim,
                                  float(grid * 16 + u),
                                  float(v)};
              }
            }
          }
        }
        return count;
      },
      [dscene, total_samples, samples_per_triangle, compare_image, &progress, &triangles](
          device_vector<float> &output) {
        if (compare_image) {
          float maximum = 0.0f;
          double squared = 0.0;
          for (size_t i = 0; i < total_samples * 3; i++) {
            maximum = max(maximum, fabsf(output[i]));
            const double value = output[i];
            squared += value * value;
          }
          LOG_INFO << "Fused displacement raw comparison: samples " << total_samples << " max "
                   << maximum << " rms " << sqrt(squared / double(total_samples * 3));
          progress.set_error("Diagnostic comparison complete; render intentionally stopped");
          return;
        }
        float4 *data = dscene->pixel_displacement_data.alloc(total_samples);
        size_t source = 0;
        for (const PatchTriangle &triangle : triangles) {
          size_t destination = dscene->pixel_displacement_offset[triangle.prim];
          if (triangle.cache_attribute) {
            data[destination++] = triangle.descriptor;
          }
          for (size_t sample = 0; sample < samples_per_triangle; sample++, source++) {
            data[destination++] = make_float4(
                output[3 * source], output[3 * source + 1], output[3 * source + 2], 0.0f);
          }
        }
        LOG_DEBUG << "Displacement fallback cache: " << triangles.size() << " triangles, "
                  << source << " baked samples, " << total_samples - source
                  << " cached attribute descriptors";
      });
  if (!success || progress.get_cancel()) {
    for (const PatchTriangle &triangle : triangles) {
      dscene->pixel_displacement_info[triangle.prim] = 0;
      dscene->pixel_displacement_offset[triangle.prim] = -1;
    }
    dscene->pixel_displacement_data.free();
  }
  if (success && !progress.get_cancel()) {
    build_pixel_displacement_patch_bvh(dscene, scene, grids, num_grids);
  }
  /* Keep the full fallback available unless every image triangle has certified inputs.
   * This scene-wide flag lets Metal omit the unused interpreter in fully eligible scenes. */
  bool needs_full_evaluator = false;
  for (size_t prim = 0; prim < dscene->tri_shader.size(); prim++) {
    const int shader_id = dscene->tri_shader[prim] & SHADER_MASK;
    if (dscene->shaders[shader_id].displacement_evaluator >= 2 &&
        ((dscene->pixel_displacement_info[prim] & (7u << 27)) != (7u << 27) ||
         dscene->pixel_displacement_offset[prim] < 0))
    {
      needs_full_evaluator = true;
      break;
    }
  }
  if (!needs_full_evaluator && success && !progress.get_cancel()) {
    dscene->data.integrator.pixel_displacement_evaluator_set &= ~32;
  }
  dscene->pixel_displacement_info.copy_to_device();
  dscene->pixel_displacement_offset.copy_to_device();
  dscene->pixel_displacement_data.copy_to_device();
}

bool GeometryManager::device_update_pixel_displacement_cache(Device *device,
                                                             DeviceScene *dscene,
                                                             Scene *scene,
                                                             Progress &progress)
{
  dscene->data.integrator.pixel_displacement_evaluator_set |= 32;
  device_vector<uint> &cache_info = dscene->pixel_displacement_info;
  device_vector<int> &cache_offset = dscene->pixel_displacement_offset;
  device_vector<float4> &cache_data = dscene->pixel_displacement_data;
  device_vector<int> &bvh_offset = dscene->pixel_displacement_bvh_offset;
  device_vector<float4> &bvh_nodes = dscene->pixel_displacement_bvh_nodes;

  uint *info = cache_info.alloc(dscene->tri_shader.size());
  for (int i = 0; i < cache_info.size(); i++) {
    info[i] = 0;
  }

  int *offsets = cache_offset.alloc(dscene->tri_shader.size());
  for (int i = 0; i < cache_offset.size(); i++) {
    offsets[i] = -1;
  }

  int *bvh_offsets = bvh_offset.alloc(dscene->tri_shader.size());
  for (int i = 0; i < bvh_offset.size(); i++) {
    bvh_offsets[i] = -1;
  }

  cache_data.free();
  bvh_nodes.free();
  for (Geometry *geom : scene->geometry) {
    if (geom->is_mesh()) {
      Mesh *mesh = static_cast<Mesh *>(geom);
      mesh->pixel_displacement_bounds.clear();
    }
  }

  size_t cacheable_triangles = 0;
  const size_t total_samples = prepare_pixel_displacement_cache_layout(
      scene, cache_info, cache_offset, &cacheable_triangles);
  if (cacheable_triangles == 0 || total_samples == 0 ||
      total_samples > PIXEL_DISPLACEMENT_MAX_CACHE_SAMPLES)
  {
    for (int i = 0; i < cache_info.size(); i++) {
      info[i] = 0;
    }
    for (int i = 0; i < cache_offset.size(); i++) {
      offsets[i] = -1;
    }
    cache_info.copy_to_device();
    cache_offset.copy_to_device();
    bvh_offset.copy_to_device();
    build_pixel_displacement_patch_cache(device, dscene, scene, progress);
    return false;
  }

  if (total_samples > size_t(std::numeric_limits<int>::max())) {
    cache_info.copy_to_device();
    cache_offset.copy_to_device();
    bvh_offset.copy_to_device();
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
      [scene, &cache_info, &cache_offset, &cache_data, &bvh_offset, &bvh_nodes](
          const device_vector<float> &d_output) {
        read_pixel_displacement_cache_output(
            scene, cache_info, cache_offset, cache_data, bvh_offset, bvh_nodes, d_output);
      });

  if (!success || actual_samples == 0 || progress.get_cancel()) {
    for (int i = 0; i < cache_info.size(); i++) {
      info[i] = 0;
    }
    for (int i = 0; i < cache_offset.size(); i++) {
      offsets[i] = -1;
    }
    cache_data.free();
    bvh_nodes.free();
    for (Geometry *geom : scene->geometry) {
      if (geom->is_mesh()) {
        Mesh *mesh = static_cast<Mesh *>(geom);
        mesh->pixel_displacement_bounds.clear();
      }
    }
    cache_info.copy_to_device();
    cache_offset.copy_to_device();
    bvh_offset.copy_to_device();
    return false;
  }

  cache_info.copy_to_device();
  cache_offset.copy_to_device();
  cache_data.copy_to_device();
  bvh_offset.copy_to_device();
  bvh_nodes.copy_to_device();

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
