/* SPDX-FileCopyrightText: 2020-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/volume.h"
#include "scene/attribute.h"
#include "scene/background.h"
#include "scene/geometry.h"
#include "scene/image_vdb.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/object.h"
#include "scene/scene.h"

#ifdef WITH_OPENVDB
#  include <openvdb/tools/GridTransformer.h>
#  include <openvdb/tools/LevelSetUtil.h>
#  include <openvdb/tools/Morphology.h>
#endif

#include "util/hash.h"
#include "util/log.h"
#include "util/nanovdb.h"
#include "util/path.h"
#include "util/progress.h"
#include "util/tbb.h"
#include "util/time.h"
#include "util/types.h"

#include "bvh/octree.h"

#include "integrator/shader_eval.h"

#include <OpenImageIO/filesystem.h>

CCL_NAMESPACE_BEGIN

NODE_DEFINE(Volume)
{
  NodeType *type = NodeType::add("volume", create, NodeType::NONE, Mesh::get_node_type());

  SOCKET_FLOAT(step_size, "Step Size", 0.0f);
  SOCKET_BOOLEAN(object_space, "Object Space", false);
  SOCKET_FLOAT(velocity_scale, "Velocity Scale", 1.0f);

  return type;
}

Volume::Volume() : Mesh(get_node_type(), Geometry::VOLUME)
{
  step_size = 0.0f;
  object_space = false;
}

void Volume::clear(bool preserve_shaders)
{
  Mesh::clear(preserve_shaders, true);
}

struct QuadData {
  int v0, v1, v2, v3;

  float3 normal;
};

struct VertHash {
  size_t operator()(const int3 &i) const
  {
    return hash_uint3(i.x, i.y, i.z);
  }
};

using VertHashMap = unordered_map<int3, int, VertHash>;

enum {
  QUAD_X_MIN = 0,
  QUAD_X_MAX = 1,
  QUAD_Y_MIN = 2,
  QUAD_Y_MAX = 3,
  QUAD_Z_MIN = 4,
  QUAD_Z_MAX = 5,
};

#if defined(WITH_OPENVDB) && defined(WITH_NANOVDB)
const int quads_indices[6][4] = {
    /* QUAD_X_MIN */
    {4, 0, 3, 7},
    /* QUAD_X_MAX */
    {1, 5, 6, 2},
    /* QUAD_Y_MIN */
    {4, 5, 1, 0},
    /* QUAD_Y_MAX */
    {3, 2, 6, 7},
    /* QUAD_Z_MIN */
    {0, 1, 2, 3},
    /* QUAD_Z_MAX */
    {5, 4, 7, 6},
};

const float3 quads_normals[6] = {
    /* QUAD_X_MIN */
    make_float3(-1.0f, 0.0f, 0.0f),
    /* QUAD_X_MAX */
    make_float3(1.0f, 0.0f, 0.0f),
    /* QUAD_Y_MIN */
    make_float3(0.0f, -1.0f, 0.0f),
    /* QUAD_Y_MAX */
    make_float3(0.0f, 1.0f, 0.0f),
    /* QUAD_Z_MIN */
    make_float3(0.0f, 0.0f, -1.0f),
    /* QUAD_Z_MAX */
    make_float3(0.0f, 0.0f, 1.0f),
};

static int add_vertex(const int3 v, vector<int3> &vertices, VertHashMap &used_verts)
{
  const VertHashMap::iterator it = used_verts.find(v);

  if (it != used_verts.end()) {
    return it->second;
  }

  const int vertex_offset = vertices.size();
  used_verts[v] = vertex_offset;
  vertices.push_back(v);
  return vertex_offset;
}

static void create_quad(const int3 corners[8],
                        vector<int3> &vertices,
                        vector<QuadData> &quads,
                        VertHashMap &used_verts,
                        const int face_index)
{
  QuadData quad;
  quad.v0 = add_vertex(corners[quads_indices[face_index][0]], vertices, used_verts);
  quad.v1 = add_vertex(corners[quads_indices[face_index][1]], vertices, used_verts);
  quad.v2 = add_vertex(corners[quads_indices[face_index][2]], vertices, used_verts);
  quad.v3 = add_vertex(corners[quads_indices[face_index][3]], vertices, used_verts);
  quad.normal = quads_normals[face_index];

  quads.push_back(quad);
}

/* Create a mesh from a volume.
 *
 * The way the algorithm works is as follows:
 *
 * - The topologies of input OpenVDB grids are merged into a temporary grid.
 * - Voxels of the temporary grid are dilated to account for the padding necessary for volume
 * sampling.
 * - A bounding box of the temporary grid is created.
 */
class VolumeMeshBuilder {
 public:
  /* use a MaskGrid to store the topology to save memory */
  openvdb::MaskGrid::Ptr topology_grid;
  bool first_grid;

  VolumeMeshBuilder();

  void add_grid(const nanovdb::GridHandle<> &grid);

  void add_padding(const int pad_size);

  void create_mesh(vector<float3> &vertices,
                   vector<int> &indices,
                   const bool ray_marching,
                   const float3 motion_displacement);

  void generate_vertices_and_quads(vector<int3> &vertices_is,
                                   vector<QuadData> &quads,
                                   const bool ray_marching,
                                   const int3 motion_pad_size);

  void convert_object_space(const vector<int3> &vertices, vector<float3> &out_vertices);

  void convert_quads_to_tris(const vector<QuadData> &quads, vector<int> &tris);

  bool empty_grid() const;
};

VolumeMeshBuilder::VolumeMeshBuilder()
{
  first_grid = true;
}

void VolumeMeshBuilder::add_grid(const nanovdb::GridHandle<> &nanogrid)
{
  /* set the transform of our grid from the first one */
  openvdb::MaskGrid::Ptr grid = nanovdb_to_openvdb_mask(nanogrid);

  if (first_grid) {
    topology_grid = grid;
    first_grid = false;
    return;
  }

  /* If the transforms do not match, we need to resample one of the grids so that
   * its index space registers with that of the other, here we resample our mask
   * grid so memory usage is kept low */
  if (topology_grid->transform() != grid->transform()) {
    const openvdb::MaskGrid::Ptr temp_grid = topology_grid->copyWithNewTree();
    temp_grid->setTransform(grid->transform().copy());
    openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(*topology_grid, *temp_grid);
    topology_grid = temp_grid;
    topology_grid->setTransform(grid->transform().copy());
  }

  topology_grid->topologyUnion(*grid);
}

void VolumeMeshBuilder::add_padding(const int pad_size)
{
  openvdb::tools::dilateActiveValues(topology_grid->tree(),
                                     pad_size,
                                     openvdb::tools::NN_FACE_EDGE_VERTEX,
                                     openvdb::tools::PRESERVE_TILES);
}

void VolumeMeshBuilder::create_mesh(vector<float3> &vertices,
                                    vector<int> &indices,
                                    const bool ray_marching,
                                    const float3 motion_displacement)
{
  /* We create vertices in index space (is), and only convert them to object
   * space when done. */
  vector<int3> vertices_is;
  vector<QuadData> quads;

  int3 motion_pad_size = make_int3(0, 0, 0);
  if (reduce_max(motion_displacement) > 0.0f) {
    /* Velocity values are object-space vectors, while the boundary mesh is built in index space.
     * Transform every corner of the symmetric displacement bound so rotated, sheared and
     * non-uniform VDB transforms remain conservative without padding unrelated axes. */
    constexpr double max_safe_padding = double(1 << 29);
    const openvdb::math::MapBase::ConstPtr map = topology_grid->transform().baseMap();
    for (int corner = 0; corner < 8; corner++) {
      const openvdb::math::Vec3d displacement(
          (corner & 1) ? motion_displacement.x : -motion_displacement.x,
          (corner & 2) ? motion_displacement.y : -motion_displacement.y,
          (corner & 4) ? motion_displacement.z : -motion_displacement.z);
      const openvdb::math::Vec3d index_displacement = map->applyInverseJacobian(displacement);
      for (int axis = 0; axis < 3; axis++) {
        const double padding = std::ceil(std::fabs(index_displacement[axis]));
        motion_pad_size[axis] = max(motion_pad_size[axis],
                                    static_cast<int>(min(padding, max_safe_padding)));
      }
    }
  }

  generate_vertices_and_quads(vertices_is, quads, ray_marching, motion_pad_size);

  convert_object_space(vertices_is, vertices);

  convert_quads_to_tris(quads, indices);
}

static bool is_non_empty_leaf(const openvdb::MaskGrid::TreeType &tree, const openvdb::Coord coord)
{
  const auto *leaf_node = tree.probeLeaf(coord);
  return (leaf_node && !leaf_node->isEmpty());
}

void VolumeMeshBuilder::generate_vertices_and_quads(vector<ccl::int3> &vertices_is,
                                                    vector<QuadData> &quads,
                                                    const bool ray_marching,
                                                    const int3 motion_pad_size)
{
  /* A velocity-swept volume can extend far beyond its active voxel topology. Building that
   * support with voxel dilation is both expensive and unnecessary: only the boundary mesh needs
   * to cover the positions that the shader may backtrack to. Use a conservative bounding box in
   * this case. Besides avoiding an O(padding^3) topology expansion, this also lets motion extend
   * beyond the original active region for arbitrarily large finite velocities. */
  if (max(max(motion_pad_size.x, motion_pad_size.y), motion_pad_size.z) > 0) {
    openvdb::CoordBBox bbox = topology_grid->evalActiveVoxelBoundingBox();
    bbox.min() = bbox.min().offsetBy(-motion_pad_size.x, -motion_pad_size.y, -motion_pad_size.z);
    bbox.max() = bbox.max().offsetBy(motion_pad_size.x, motion_pad_size.y, motion_pad_size.z);

    const int3 min = make_int3(bbox.min().x(), bbox.min().y(), bbox.min().z());
    const int3 max = make_int3(bbox.max().x(), bbox.max().y(), bbox.max().z());
    const int3 corners[8] = {
        make_int3(min[0], min[1], min[2]),
        make_int3(max[0], min[1], min[2]),
        make_int3(max[0], max[1], min[2]),
        make_int3(min[0], max[1], min[2]),
        make_int3(min[0], min[1], max[2]),
        make_int3(max[0], min[1], max[2]),
        make_int3(max[0], max[1], max[2]),
        make_int3(min[0], max[1], max[2]),
    };

    VertHashMap used_verts;
    create_quad(corners, vertices_is, quads, used_verts, QUAD_X_MIN);
    create_quad(corners, vertices_is, quads, used_verts, QUAD_X_MAX);
    create_quad(corners, vertices_is, quads, used_verts, QUAD_Y_MIN);
    create_quad(corners, vertices_is, quads, used_verts, QUAD_Y_MAX);
    create_quad(corners, vertices_is, quads, used_verts, QUAD_Z_MIN);
    create_quad(corners, vertices_is, quads, used_verts, QUAD_Z_MAX);
    return;
  }

  if (ray_marching) {
    /* Make sure we only have leaf nodes in the tree, as tiles are not handled by this algorithm */
    topology_grid->tree().voxelizeActiveTiles();

    const openvdb::MaskGrid::TreeType &tree = topology_grid->tree();

    /* Compute the active voxel bounding box to clip leaf bounding boxes. Leaf nodes are
     * aligned to 8x8x8 blocks and can extend far beyond the actual active voxels, especially
     * for low-resolution grids with large voxel sizes. */
    openvdb::CoordBBox active_bbox;
    tree.evalActiveVoxelBoundingBox(active_bbox);
    /* +1 to convert from inclusive to exclusive bounds. */
    active_bbox.max() = active_bbox.max().offsetBy(1);

    VertHashMap used_verts;
    for (auto iter = tree.cbeginLeaf(); iter; ++iter) {
      if (iter->isEmpty()) {
        continue;
      }
      openvdb::CoordBBox leaf_bbox = iter->getNodeBoundingBox();

      /* Compute the leaf center from the (unclipped) node bounding box. */
      static const int LEAF_DIM = openvdb::MaskGrid::TreeType::LeafNodeType::DIM;
      const auto center = leaf_bbox.min() + openvdb::Coord(LEAF_DIM / 2);

      /* +1 to convert from inclusive to exclusive bounds. */
      leaf_bbox.max() = leaf_bbox.max().offsetBy(1);
      /* Clip leaf bounding box to the active voxel region. */
      leaf_bbox.min() = openvdb::Coord::maxComponent(leaf_bbox.min(), active_bbox.min());
      leaf_bbox.max() = openvdb::Coord::minComponent(leaf_bbox.max(), active_bbox.max());

      const int3 min = make_int3(leaf_bbox.min().x(), leaf_bbox.min().y(), leaf_bbox.min().z());
      const int3 max = make_int3(leaf_bbox.max().x(), leaf_bbox.max().y(), leaf_bbox.max().z());
      const int3 corners[8] = {
          make_int3(min[0], min[1], min[2]),
          make_int3(max[0], min[1], min[2]),
          make_int3(max[0], max[1], min[2]),
          make_int3(min[0], max[1], min[2]),
          make_int3(min[0], min[1], max[2]),
          make_int3(max[0], min[1], max[2]),
          make_int3(max[0], max[1], max[2]),
          make_int3(min[0], max[1], max[2]),
      };
      /* Only create a quad if on the border between an active and an inactive leaf.
       *
       * We verify that a leaf exists by probing a coordinate that is at its center,
       * to do so we compute the center of the current leaf and offset this coordinate
       * by the size of a leaf in each direction.
       */
      if (!is_non_empty_leaf(tree, openvdb::Coord(center.x() - LEAF_DIM, center.y(), center.z())))
      {
        create_quad(corners, vertices_is, quads, used_verts, QUAD_X_MIN);
      }
      if (!is_non_empty_leaf(tree, openvdb::Coord(center.x() + LEAF_DIM, center.y(), center.z())))
      {
        create_quad(corners, vertices_is, quads, used_verts, QUAD_X_MAX);
      }
      if (!is_non_empty_leaf(tree, openvdb::Coord(center.x(), center.y() - LEAF_DIM, center.z())))
      {
        create_quad(corners, vertices_is, quads, used_verts, QUAD_Y_MIN);
      }
      if (!is_non_empty_leaf(tree, openvdb::Coord(center.x(), center.y() + LEAF_DIM, center.z())))
      {
        create_quad(corners, vertices_is, quads, used_verts, QUAD_Y_MAX);
      }
      if (!is_non_empty_leaf(tree, openvdb::Coord(center.x(), center.y(), center.z() - LEAF_DIM)))
      {
        create_quad(corners, vertices_is, quads, used_verts, QUAD_Z_MIN);
      }
      if (!is_non_empty_leaf(tree, openvdb::Coord(center.x(), center.y(), center.z() + LEAF_DIM)))
      {
        create_quad(corners, vertices_is, quads, used_verts, QUAD_Z_MAX);
      }
    }
    return;
  }

  openvdb::CoordBBox bbox = topology_grid->evalActiveVoxelBoundingBox();
  const int3 min = make_int3(bbox.min().x(), bbox.min().y(), bbox.min().z());
  const int3 max = make_int3(bbox.max().x(), bbox.max().y(), bbox.max().z());

  const int3 corners[8] = {
      make_int3(min[0], min[1], min[2]),
      make_int3(max[0], min[1], min[2]),
      make_int3(max[0], max[1], min[2]),
      make_int3(min[0], max[1], min[2]),
      make_int3(min[0], min[1], max[2]),
      make_int3(max[0], min[1], max[2]),
      make_int3(max[0], max[1], max[2]),
      make_int3(min[0], max[1], max[2]),
  };

  /* Create 6 quads of the bounding box. */
  VertHashMap used_verts;

  create_quad(corners, vertices_is, quads, used_verts, QUAD_X_MIN);
  create_quad(corners, vertices_is, quads, used_verts, QUAD_X_MAX);
  create_quad(corners, vertices_is, quads, used_verts, QUAD_Y_MIN);
  create_quad(corners, vertices_is, quads, used_verts, QUAD_Y_MAX);
  create_quad(corners, vertices_is, quads, used_verts, QUAD_Z_MIN);
  create_quad(corners, vertices_is, quads, used_verts, QUAD_Z_MAX);
}

void VolumeMeshBuilder::convert_object_space(const vector<int3> &vertices,
                                             vector<float3> &out_vertices)
{
  out_vertices.reserve(vertices.size());

  for (size_t i = 0; i < vertices.size(); ++i) {
    openvdb::math::Vec3d p = topology_grid->indexToWorld(
        openvdb::math::Vec3d(vertices[i].x, vertices[i].y, vertices[i].z));
    const float3 vertex = make_float3((float)p.x(), (float)p.y(), (float)p.z());
    out_vertices.push_back(vertex);
  }
}

void VolumeMeshBuilder::convert_quads_to_tris(const vector<QuadData> &quads, vector<int> &tris)
{
  /* If the grid transform has negative scale, the vertices transform by indexToWorld
   * will have reversed winding order. Here we compensate for that by flipping the
   * triangle order. */
  const bool negative_scale =
      topology_grid->transform().baseMap()->getAffineMap()->getMat4().det() < 0.0;

  int index_offset = 0;
  tris.resize(quads.size() * 6);

  for (size_t i = 0; i < quads.size(); ++i) {
    tris[index_offset++] = quads[i].v0;
    tris[index_offset++] = negative_scale ? quads[i].v1 : quads[i].v2;
    tris[index_offset++] = negative_scale ? quads[i].v2 : quads[i].v1;

    tris[index_offset++] = quads[i].v0;
    tris[index_offset++] = negative_scale ? quads[i].v2 : quads[i].v3;
    tris[index_offset++] = negative_scale ? quads[i].v3 : quads[i].v2;
  }
}

bool VolumeMeshBuilder::empty_grid() const
{
  return !topology_grid ||
         (!topology_grid->tree().hasActiveTiles() && topology_grid->tree().leafCount() == 0);
}

static float3 estimate_max_velocity_displacement(const nanovdb::GridHandle<> &grid,
                                                 const float velocity_scale)
{
  const auto *typed_grid = grid.template grid<nanovdb::Vec3f>(0);

  if (typed_grid == nullptr) {
    return zero_float3();
  }

  if (velocity_scale == 0.0f) {
    return zero_float3();
  }

  /* NanoVDB stores exact extrema in the root node. Reading them is constant time, unlike scanning
   * every active velocity voxel to estimate a percentile. The mesh is only a conservative
   * intersection bound, so discarding statistical outliers here creates visible clipping. */
  const nanovdb::Vec3f min_velocity = typed_grid->tree().root().minimum();
  const nanovdb::Vec3f max_velocity = typed_grid->tree().root().maximum();
  float3 max_displacement = zero_float3();
  for (int i = 0; i < 3; i++) {
    const double max_abs_velocity = max(std::fabs(double(min_velocity[i])),
                                        std::fabs(double(max_velocity[i])));
    const double displacement = max_abs_velocity * std::fabs(double(velocity_scale));
    max_displacement[i] = std::isfinite(displacement) ? float(displacement) : 0.0f;
  }

  return max_displacement;
}
#endif

#ifdef WITH_OPENVDB
static openvdb::FloatGrid::ConstPtr get_vdb_for_attribute(Volume *volume, AttributeStandard std)
{
  Attribute *attr = volume->attributes.find(std);
  if (!attr) {
    return nullptr;
  }

  const ImageHandle &handle = attr->data_voxel();
  VDBImageLoader *vdb_loader = handle.vdb_loader();
  if (!vdb_loader) {
    return nullptr;
  }

  const openvdb::GridBase::ConstPtr grid = vdb_loader->get_grid();
  if (!grid) {
    return nullptr;
  }

  if (!grid->isType<openvdb::FloatGrid>()) {
    return nullptr;
  }

  return openvdb::gridConstPtrCast<openvdb::FloatGrid>(grid);
}

class MergeScalarGrids {
  using ScalarTree = openvdb::FloatTree;

  openvdb::tree::ValueAccessor<const ScalarTree> m_acc_x, m_acc_y, m_acc_z;

 public:
  MergeScalarGrids(const ScalarTree *x_tree, const ScalarTree *y_tree, const ScalarTree *z_tree)
      : m_acc_x(*x_tree), m_acc_y(*y_tree), m_acc_z(*z_tree)
  {
  }

  MergeScalarGrids(const MergeScalarGrids &other)

      = default;

  void operator()(const openvdb::Vec3STree::ValueOnIter &it) const
  {
    using namespace openvdb;

    const math::Coord xyz = it.getCoord();
    const float x = m_acc_x.getValue(xyz);
    const float y = m_acc_y.getValue(xyz);
    const float z = m_acc_z.getValue(xyz);

    it.setValue(math::Vec3s(x, y, z));
  }
};

static void merge_scalar_grids_for_velocity(const Scene *scene, Volume *volume)
{
  if (volume->attributes.find(ATTR_STD_VOLUME_VELOCITY)) {
    /* A vector grid for velocity is already available. */
    return;
  }

  const openvdb::FloatGrid::ConstPtr vel_x_grid = get_vdb_for_attribute(
      volume, ATTR_STD_VOLUME_VELOCITY_X);
  const openvdb::FloatGrid::ConstPtr vel_y_grid = get_vdb_for_attribute(
      volume, ATTR_STD_VOLUME_VELOCITY_Y);
  const openvdb::FloatGrid::ConstPtr vel_z_grid = get_vdb_for_attribute(
      volume, ATTR_STD_VOLUME_VELOCITY_Z);

  if (!(vel_x_grid && vel_y_grid && vel_z_grid)) {
    return;
  }

  const openvdb::Vec3fGrid::Ptr vecgrid = openvdb::Vec3SGrid::create(openvdb::Vec3s(0.0f));

  /* Activate voxels in the vector grid based on the scalar grids to ensure thread safety during
   * the merge. */
  vecgrid->tree().topologyUnion(vel_x_grid->tree());
  vecgrid->tree().topologyUnion(vel_y_grid->tree());
  vecgrid->tree().topologyUnion(vel_z_grid->tree());

  MergeScalarGrids op(&vel_x_grid->tree(), &vel_y_grid->tree(), &vel_z_grid->tree());
  openvdb::tools::foreach(vecgrid->beginValueOn(), op, true, false);

  /* Assume all grids have the same transformation. */
  const openvdb::math::Transform::Ptr transform = openvdb::ConstPtrCast<openvdb::math::Transform>(
      vel_x_grid->transformPtr());
  vecgrid->setTransform(transform);

  /* Make an attribute for it. */
  Attribute *attr = volume->attributes.add(ATTR_STD_VOLUME_VELOCITY);
  unique_ptr<ImageLoader> loader = make_unique<VDBImageLoader>(vecgrid, "merged_velocity");
  const ImageParams params;
  attr->data_voxel_for_write() = scene->image_manager->add_image(std::move(loader), params);
}
#endif /* defined(WITH_OPENVDB) && defined(WITH_NANOVDB) */

/* ************************************************************************** */

void GeometryManager::create_volume_mesh(const Scene *scene, Volume *volume, Progress &progress)
{
  const string msg = string_printf("Computing Volume Mesh %s", volume->name.c_str());
  progress.set_status("Updating Mesh", msg);

  /* Find shader and compute padding based on volume shader interpolation settings. */
  Shader *volume_shader = nullptr;
  int pad_size = 0;
  float3 motion_displacement = zero_float3();

  for (Node *node : volume->get_used_shaders()) {
    Shader *shader = static_cast<Shader *>(node);

    if (!shader->has_volume) {
      continue;
    }

    volume_shader = shader;

    if (shader->get_volume_interpolation_method() == VOLUME_INTERPOLATION_LINEAR) {
      pad_size = max(1, pad_size);
    }
    else if (shader->get_volume_interpolation_method() == VOLUME_INTERPOLATION_CUBIC) {
      pad_size = max(2, pad_size);
    }

    break;
  }

  /* Clear existing volume mesh, done here in case we early out due to
   * empty grid or missing volume shader.
   * Also keep the shaders to avoid infinite loops when synchronizing, as this will tag the shaders
   * as having changed. */
  volume->clear(true);
  volume->need_update_rebuild = true;

  if (!volume_shader) {
    return;
  }

#if defined(WITH_OPENVDB) && defined(WITH_NANOVDB)
  /* Create volume mesh builder. */
  VolumeMeshBuilder builder;

  for (Attribute &attr : volume->attributes.attributes) {
    if (attr.element != ATTR_ELEMENT_VOXEL) {
      continue;
    }

    ImageHandle &handle = attr.data_voxel_for_write();

    if (handle.empty()) {
      continue;
    }

    /* Create NanoVDB grid handle from image memory. */
    device_image *image = handle.vdb_image_memory();
    if (image == nullptr || image->host_pointer == nullptr ||
        image->info.data_type == IMAGE_DATA_TYPE_NANOVDB_EMPTY ||
        !is_nanovdb_type(image->info.data_type))
    {
      continue;
    }

    nanovdb::GridHandle grid(
        nanovdb::HostBuffer::createFull(image->memory_size(), image->host_pointer));

    /* Add padding based on the maximum velocity vector. */
    if (attr.std == ATTR_STD_VOLUME_VELOCITY && scene->need_motion() != Scene::MOTION_NONE) {
      motion_displacement = max(
          motion_displacement,
          estimate_max_velocity_displacement(grid, volume->get_velocity_scale()));
    }

    builder.add_grid(grid);
  }

  /* If nothing to build, early out. */
  if (builder.empty_grid()) {
    LOG_DEBUG << "Memory usage volume mesh: 0 Mb. (empty grid)";
    return;
  }

  /* Interpolation padding is small and changes the sparse topology used for empty-space skipping.
   * Motion padding can be arbitrarily large and is represented only by the boundary mesh. */
  builder.add_padding(pad_size);

  /* Create mesh. */
  vector<float3> vertices;
  vector<int> indices;
  const bool ray_marching = scene->integrator->get_volume_ray_marching();
  builder.create_mesh(vertices, indices, ray_marching, motion_displacement);

  volume->resize_mesh(vertices.size(), indices.size() / 3);
  volume->used_shaders.clear();
  volume->used_shaders.push_back_slow(volume_shader);

  std::ranges::copy(vertices, volume->get_position_for_write());
  std::ranges::copy(indices, volume->triangles.data());
  std::ranges::fill(volume->get_shader(), 0);
  std::ranges::fill(volume->get_smooth(), false);

  /* Print stats. */
  LOG_DEBUG << "Memory usage volume mesh: "
            << (vertices.size() * sizeof(float3) + indices.size() * sizeof(int)) /
                   (1024.0 * 1024.0)
            << "Mb.";
#else
  (void)scene;
#endif /* defined(WITH_OPENVDB) && defined(WITH_NANOVDB) */
}

void Volume::merge_grids(const Scene *scene)
{
#ifdef WITH_OPENVDB
  merge_scalar_grids_for_velocity(scene, this);
#else
  (void)scene;
#endif
}

VolumeManager::VolumeManager()
{
  need_rebuild_ = true;
  need_update_step_size = true;
}

void VolumeManager::tag_update()
{
  need_rebuild_ = true;
  need_update_ior_ = true;
}

void VolumeManager::tag_update_ior()
{
  need_update_ior_ = true;
}

/* Remove changed objects from the list of octrees and tag for rebuild. */
void VolumeManager::tag_update(const set<Object *> &objects, uint32_t flag)
{
  /* The grids of the index of refraction are in world space, as the shaders may be, and the one
   * of the world covers all objects. */
  for (const Object *object : objects) {
    if (has_world_ior_ || object->get_geometry()->has_volume) {
      need_update_ior_ = true;
    }
  }

  if (object_octrees_.empty()) {
    /* Volume object is not in the octree, can happen when using ray marching. */
    return;
  }

  bool volume_object_updated = false;
  for (const Object *object : objects) {
    if (!object->get_geometry()->has_volume) {
      continue;
    }

    volume_object_updated = true;

    for (const Node *node : object->get_geometry()->get_used_shaders()) {
      const Shader *shader = static_cast<const Shader *>(node);
      if (shader->has_volume_spatial_varying || (flag & ObjectManager::OBJECT_REMOVED)) {
        /* TODO(weizhen): no need to update if the spatial variation is not in world space. */
        tag_update();
        object_octrees_.erase({object, shader});
      }
    }
  }

  if (volume_object_updated) {
    if (flag & ObjectManager::VISIBILITY_MODIFIED) {
      tag_update();
    }
    if (!need_rebuild_ && (flag & ObjectManager::TRANSFORM_MODIFIED)) {
      /* Octree is not tagged for rebuild but the transformation changed, so a redraw is needed. */
      update_visualization_ = true;
    }
  }
}

/* Remove object with changed shader from the list of octrees and tag for rebuild. */
void VolumeManager::tag_update(const Shader *shader)
{
  tag_update();
  for (auto it = object_octrees_.begin(); it != object_octrees_.end();) {
    if (it->first.second == shader) {
      it = object_octrees_.erase(it);
    }
    else {
      it++;
    }
  }
}

/* Remove objects with changed geometry from the list of octrees and tag for rebuild. */
void VolumeManager::tag_update(const set<Geometry *> &geometry)
{
  bool volume_geometry_updated = false;
  for (Geometry *geometry : geometry) {
    if (geometry->has_volume) {
      volume_geometry_updated = true;
    }
  }

  if (!volume_geometry_updated) {
    return;
  }

  tag_update();
  /* Tag Octree for update. */
  for (auto it = object_octrees_.begin(); it != object_octrees_.end();) {
    const Object *object = it->first.first;
    if (geometry.contains(object->get_geometry())) {
      it = object_octrees_.erase(it);
    }
    else {
      it++;
    }
  }

#ifdef WITH_OPENVDB
  /* Tag VDB map for update. */
  for (auto it = vdb_map_.begin(); it != vdb_map_.end();) {
    if (geometry.contains(const_cast<Geometry *>(it->first.first))) {
      it = vdb_map_.erase(it);
    }
    else {
      it++;
    }
  }
#endif
}

void VolumeManager::tag_update_indices()
{
  update_root_indices_ = true;
}

void VolumeManager::tag_update_algorithm()
{
  need_rebuild_ = true;
  algorithm_modified_ = true;
}

bool VolumeManager::is_homogeneous_volume(const Object *object, const Shader *shader)
{
  if (!shader->has_volume || shader->has_volume_spatial_varying) {
    return false;
  }

  if (shader->has_volume_attribute_dependency) {
    for (Attribute &attr : object->get_geometry()->attributes.attributes) {
      /* If both the shader and the object needs volume attributes, the volume is heterogeneous. */
      if (attr.element == ATTR_ELEMENT_VOXEL) {
        return false;
      }
    }
  }

  return true;
}

#ifdef WITH_OPENVDB
/* Given a mesh, check if every edge has exactly two incident triangles, and if the two triangles
 * have the same orientation. */
static bool mesh_is_closed(const std::vector<openvdb::Vec3I> &triangles)
{
  const size_t num_triangles = triangles.size();
  if (num_triangles % 2) {
    return false;
  }

  /* Store the two vertices that forms an edge. */
  std::multiset<std::pair<int, int>> edges;
  int num_edges = 0;

  for (const auto &tri : triangles) {
    for (int i = 0; i < 3; i++) {
      const std::pair<int, int> e = {tri[i], tri[(i + 1) % 3]};
      if (edges.contains(e)) {
        /* Same edge exists. */
        return false;
      }

      /* Check if an edge in the opposite order exists. */
      const auto count = edges.count({e.second, e.first});
      if (count > 1) {
        /* Edge has more than 2 incident faces. */
        return false;
      }
      if (count == 1) {
        /* If an edge in the opposite order exists, increment the count. */
        edges.insert({e.second, e.first});
      }
      else {
        /* Insert a new edge. */
        num_edges++;
        edges.insert(e);
      }
    }
  }

  /* Until this point, the count of each element in the set is at most 2; to check if they are
   * exactly 2, we just need to compare the total numbers. */
  return num_triangles * 3 == num_edges * 2;
}

openvdb::BoolGrid::ConstPtr VolumeManager::mesh_to_sdf_grid(const Mesh *mesh,
                                                            const Shader *shader,
                                                            const float half_width)
{
  const int num_verts = mesh->num_verts();
  std::vector<openvdb::Vec3f> points(num_verts);
  parallel_for(0, num_verts, [&](int i) {
    const float3 &vert = mesh->get_position()[i];
    points[i] = openvdb::Vec3f(vert.x, vert.y, vert.z);
  });

  const int max_num_triangles = mesh->num_triangles();
  std::vector<openvdb::Vec3I> triangles;
  triangles.reserve(max_num_triangles);
  for (int i = 0; i < max_num_triangles; i++) {
    /* Only push triangles with matching shader. */
    const int shader_index = mesh->get_shader()[i];
    if (static_cast<const Shader *>(mesh->get_used_shaders()[shader_index]) == shader) {
      triangles.emplace_back(mesh->get_triangles()[i * 3],
                             mesh->get_triangles()[i * 3 + 1],
                             mesh->get_triangles()[i * 3 + 2]);
    }
  }

  if (!mesh_is_closed(triangles)) {
    /* `meshToLevelSet()` requires a closed mesh, otherwise we can not determine the interior of
     * the mesh. Evaluate the whole bounding box in this case. */
    return openvdb::BoolGrid::create();
  }

  /* TODO(weizhen): Should consider object instead of mesh size. */
  const float3 mesh_size = mesh->bounds.size();
  const auto vdb_voxel_size = openvdb::Vec3d(mesh_size.x, mesh_size.y, mesh_size.z) /
                              double(1 << VOLUME_OCTREE_MAX_DEPTH);

  auto xform = openvdb::math::Transform::createLinearTransform(1.0);
  xform->postScale(vdb_voxel_size);

  auto sdf_grid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(
      *xform, points, triangles, half_width);

  return openvdb::tools::sdfInteriorMask(*sdf_grid, 0.5 * vdb_voxel_size.length());
}

openvdb::BoolGrid::ConstPtr VolumeManager::get_vdb(const Geometry *geom,
                                                   const Shader *shader) const
{
  if (geom->is_mesh()) {
    if (auto it = vdb_map_.find({geom, shader}); it != vdb_map_.end()) {
      return it->second;
    }
  }
  /* Create empty grid. */
  return openvdb::BoolGrid::create();
}
#endif

void VolumeManager::initialize_octree(const Scene *scene, Progress &progress)
{
  /* Instanced objects without spatial variation can share one octree. */
  std::map<std::pair<const Geometry *, const Shader *>, std::shared_ptr<Octree>> geometry_octrees;
  for (const auto &it : object_octrees_) {
    const Shader *shader = it.first.second;
    if (!shader->has_volume_spatial_varying) {
      if (const Object *object = it.first.first) {
        geometry_octrees[{object->get_geometry(), shader}] = it.second;
      }
    }
  }

  /* Loop through the volume objects to initialize their root nodes. */
  for (const Object *object : scene->objects) {
    const Geometry *geom = object->get_geometry();
    if (!geom->has_volume) {
      continue;
    }

    /* Create Octree. */
    for (const Node *node : geom->get_used_shaders()) {
      const Shader *shader = static_cast<const Shader *>(node);
      if (!shader->has_volume) {
        continue;
      }

      if (!object_octrees_.contains({object, shader})) {
        if (geom->is_light()) {
          const Light *light = static_cast<const Light *>(geom);
          if (light->is_background_light()) {
            /* World volume is unbounded, use some practical large number instead. */
            const float3 size = make_float3(10000.0f);
            object_octrees_[{object, shader}] = std::make_shared<Octree>(BoundBox(-size, size));
          }
        }
        else {
          const Mesh *mesh = static_cast<const Mesh *>(geom);
          if (is_zero(mesh->bounds.size())) {
            continue;
          }
          if (!shader->has_volume_spatial_varying) {
            /* TODO(weizhen): check object attribute. */
            if (auto it = geometry_octrees.find({geom, shader}); it != geometry_octrees.end()) {
              /* Share octree with other instances. */
              object_octrees_[{object, shader}] = it->second;
            }
            else {
              auto octree = std::make_shared<Octree>(mesh->bounds);
              geometry_octrees[{geom, shader}] = octree;
              object_octrees_[{object, shader}] = octree;
            }
          }
          else {
            /* TODO(weizhen): we can still share the octree if the spatial variation is in object
             * space, but that might be tricky to determine. */
            object_octrees_[{object, shader}] = std::make_shared<Octree>(mesh->bounds);
          }
        }
      }

#ifdef WITH_OPENVDB
      if (geom->is_mesh() && !VolumeManager::is_homogeneous_volume(object, shader) &&
          !vdb_map_.contains({geom, shader}))
      {
        const Mesh *mesh = static_cast<const Mesh *>(geom);
        const float3 dim = mesh->bounds.size();
        if (dim.x > 0.0f && dim.y > 0.0f && dim.z > 0.0f) {
          const char *name = object->get_asset_name().c_str();
          progress.set_substatus(string_printf("Creating SDF grid for %s", name));
          vdb_map_[{geom, shader}] = mesh_to_sdf_grid(mesh, shader, 1.0f);
        }
      }
#else
      (void)progress;
#endif
    }
  }
}

void VolumeManager::update_num_octree_nodes()
{
  num_octree_nodes_ = 0;
  num_octree_roots_ = 0;

  std::set<const Octree *> unique_octrees;
  for (const auto &it : object_octrees_) {
    const Octree *octree = it.second.get();
    if (unique_octrees.contains(octree)) {
      continue;
    }

    unique_octrees.insert(octree);

    num_octree_roots_++;
    num_octree_nodes_ += octree->get_num_nodes();
  }
}

int VolumeManager::num_octree_nodes() const
{
  return num_octree_nodes_;
}

int VolumeManager::num_octree_roots() const
{
  return num_octree_roots_;
}

void VolumeManager::build_octree(Device *device, Progress &progress)
{
  const double start_time = time_dt();

  for (auto &it : object_octrees_) {
    if (it.second->is_built()) {
      continue;
    }

    const Object *object = it.first.first;
    const Shader *shader = it.first.second;
#ifdef WITH_OPENVDB
    openvdb::BoolGrid::ConstPtr interior_mask = get_vdb(object->get_geometry(), shader);
    it.second->build(device, progress, interior_mask, object, shader);
#else
    it.second->build(device, progress, object, shader);
#endif
  }

  update_num_octree_nodes();

  const double build_time = time_dt() - start_time;

  LOG_DEBUG << object_octrees_.size() << " volume octree(s) with a total of " << num_octree_nodes()
            << " nodes are built in " << build_time << " seconds.";
}

void VolumeManager::update_root_indices(DeviceScene *dscene, const Scene *scene) const
{
  if (object_octrees_.empty()) {
    return;
  }

  /* Keep track of the root index of the unique octrees. */
  std::map<const Octree *, int> octree_root_indices;

  int *roots = dscene->volume_tree_root_ids.alloc(scene->objects.size());

  int root_index = 0;
  for (const auto &it : object_octrees_) {
    const Object *object = it.first.first;
    const int object_id = object->get_device_index();
    const Octree *octree = it.second.get();
    auto entry = octree_root_indices.find(octree);
    if (entry == octree_root_indices.end()) {
      roots[object_id] = root_index;
      octree_root_indices[octree] = root_index;

      root_index++;
    }
    else {
      /* Instances share the same octree. */
      roots[object_id] = entry->second;
    }
  }

  dscene->volume_tree_root_ids.copy_to_device();
}

void VolumeManager::flatten_octree(DeviceScene *dscene, const Scene *scene) const
{
  if (object_octrees_.empty()) {
    return;
  }

  update_root_indices(dscene, scene);

  for (const auto &it : object_octrees_) {
    /* Octrees need to be re-flattened. */
    it.second->set_flattened(false);
  }

  KernelOctreeRoot *kroots = dscene->volume_tree_roots.alloc(num_octree_roots());
  KernelOctreeNode *knodes = dscene->volume_tree_nodes.alloc(num_octree_nodes());

  int node_index = 0;
  int root_index = 0;
  for (const auto &it : object_octrees_) {
    std::shared_ptr<Octree> octree = it.second;
    if (octree->is_flattened()) {
      continue;
    }

    /* If an object has multiple shaders, the root index is overwritten, so we also write the
     * shader id, and perform a linear search in the kernel to find the correct octree. */
    kroots[root_index].shader = it.first.second->id;
    kroots[root_index].id = node_index;

    /* Transform from object space into octree space. */
    auto root = octree->get_root();
    const float3 scale = 1.0f / root->bbox.size();
    kroots[root_index].scale = scale;
    kroots[root_index].translation = -root->bbox.min * scale + 1.0f;

    root_index++;

    /* Flatten octree. */
    const uint current_index = node_index++;
    knodes[current_index].parent = -1;
    octree->flatten(knodes, current_index, root, node_index);
    octree->set_flattened();
  }

  dscene->volume_tree_nodes.copy_to_device();
  dscene->volume_tree_roots.copy_to_device();

  LOG_DEBUG << "Memory usage of volume octrees: "
            << (dscene->volume_tree_nodes.size() * sizeof(KernelOctreeNode) +
                dscene->volume_tree_roots.size() * sizeof(KernelOctreeRoot) +
                dscene->volume_tree_root_ids.size() * sizeof(int)) /
                   (1024.0 * 1024.0)
            << "Mb.";
}

/* Dump octree as python script, enabled by `CYCLES_VOLUME_OCTREE_DUMP` environment variable. */
std::string VolumeManager::visualize_octree(const char *filename) const
{
  const std::string filename_full = path_join(OIIO::Filesystem::current_path(), filename);

  std::ofstream file(filename_full);
  if (file.is_open()) {
    std::ostringstream buffer;
    file << "# Visualize volume octree.\n\n"
            "import bpy\nimport mathutils\n\n"
            "if bpy.context.active_object:\n"
            "    bpy.context.active_object.select_set(False)\n\n"
            "octree = bpy.data.collections.new(name='Octree')\n"
            "bpy.context.scene.collection.children.link(octree)\n\n";

    for (const auto &it : object_octrees_) {
      /* Draw Octree. */
      const auto octree = it.second;
      const std::string object_name = it.first.first->get_asset_name().string();
      octree->visualize(file, object_name);

      /* Apply transform. */
      const Object *object = it.first.first;
      const Geometry *geom = object->get_geometry();
      if (!geom->is_light() && !geom->transform_applied) {
        const Transform t = object->get_tfm();
        file << "obj.matrix_world = mathutils.Matrix((" << t.x << ", " << t.y << ", " << t.z
             << ", (" << 0 << "," << 0 << "," << 0 << "," << 1 << ")))\n\n";
      }
    }

    file.close();
  }

  return filename_full;
}

void VolumeManager::update_step_size(const Scene *scene, DeviceScene *dscene, Progress &progress)
{
  assert(scene->integrator->get_volume_ray_marching());

  if (!need_update_step_size && !dscene->volume_step_size.is_modified() &&
      !scene->integrator->volume_step_rate_is_modified() && !algorithm_modified_)
  {
    return;
  }

  if (dscene->volume_step_size.need_realloc()) {
    dscene->volume_step_size.alloc(scene->objects.size());
  }

  float *volume_step_size = dscene->volume_step_size.data();

  for (const Object *object : scene->objects) {
    const Geometry *geom = object->get_geometry();
    if (!geom->has_volume) {
      continue;
    }

    volume_step_size[object->index] = scene->integrator->get_volume_step_rate() *
                                      object->compute_volume_step_size(progress);
  }

  dscene->volume_step_size.copy_to_device();
  dscene->volume_step_size.clear_modified();
  need_update_step_size = false;
}

/* Volume Index of Refraction
 *
 * The volume shaders are evaluated on a grid for every object whose volume has an index of
 * refraction, for the rays it bends to look the gradient up instead of evaluating shaders at
 * every step. See kernel/integrator/volume_ior.h. */

/* Voxels along the shortest and the longest side of a grid, and in all of it. */
static const int VOLUME_IOR_MIN_RESOLUTION = 64;
static const int VOLUME_IOR_MAX_RESOLUTION = 512;
static const int64_t VOLUME_IOR_MAX_VOXELS = 16 * 1024 * 1024;
/* Voxels of a grid whose size is not set by a voxel grid of the volume nor by the shader. */
static const int64_t VOLUME_IOR_AUTO_VOXELS = 1024 * 1024;

/* Largest factor the resolution of an axis is refined by, and the resolution it may get to. */
static const int VOLUME_IOR_MAX_REFINE = 8;
static const int VOLUME_IOR_REFINED_MAX_RESOLUTION = 1024;

/* A grid that is laid out over a box, in the space of the geometry. */
struct VolumeIORGridBox {
  BoundBox box = BoundBox::empty;
  /* Voxels along the sides of the box. */
  float3 n = zero_float3();
  /* Axes along which the resolution follows the field, see volume_ior_grid_refine(), and the
   * voxels per step of a ray along every axis that come of it. */
  int3 refine = make_int3(0);
  float3 step = one_float3();
  int64_t max_voxels = 0;
};

static void volume_ior_grid_box_transform(const VolumeIORGridBox &grid_box,
                                          Transform &tfm,
                                          int3 &res)
{
  /* A voxel more on every side: lookups inside of the bounds never reach the border. */
  res = make_int3(int(grid_box.n.x), int(grid_box.n.y), int(grid_box.n.z)) + make_int3(2);
  tfm = transform_translate(make_float3(1.0f)) *
        transform_scale(grid_box.n / grid_box.box.size()) * transform_translate(-grid_box.box.min);
}

/* Resolution of the grid of the index of refraction of an object, and the transform from the
 * space of its geometry to the voxel index space of that grid. */
static bool volume_ior_grid_layout(const Scene *scene,
                                   const Object *object,
                                   const Shader *shader,
                                   Progress &progress,
                                   Transform &tfm,
                                   int3 &res,
                                   VolumeIORGridBox &grid_box)
{
  grid_box.refine = make_int3(0);

  Geometry *geom = object->get_geometry();

  /* Bounds in the space of the geometry, which is the world for the background. */
  BoundBox box = BoundBox::empty;
  if (geom->is_light()) {
    for (const Object *other : scene->objects) {
      if (!other->get_geometry()->is_light() && other->bounds.valid()) {
        box.grow(other->bounds);
      }
    }
  }
  else {
    box = geom->bounds;
  }
  if (!box.valid()) {
    return false;
  }
  float3 size = box.size();
  const float max_size = reduce_max(size);
  if (!(max_size > 0.0f) || !isfinite_safe(size)) {
    return false;
  }
  /* A flat volume has no inside, a grid needs one. */
  const float3 pad = max(make_float3(1e-4f * max_size) - size, zero_float3()) * 0.5f;
  box.min -= pad;
  box.max += pad;
  size = box.size();

  /* The lattice of the finest voxel grid of the volume: no need to resample what the shader
   * reads from it. */
  if (geom->is_volume()) {
    Volume *volume = static_cast<Volume *>(geom);
    float finest = FLT_MAX;
    Transform finest_tfm = transform_identity();
    for (Attribute &attr : volume->attributes.attributes) {
      if (attr.element != ATTR_ELEMENT_VOXEL) {
        continue;
      }
      ImageHandle &handle = attr.data_voxel_for_write();
      const ImageMetaData &metadata = handle.metadata(progress);
      if (metadata.nanovdb_byte_size == 0 || !metadata.use_transform_3d) {
        continue;
      }
      /* Geometry to index space: only lattices along the axes are followed. */
      const Transform &index_tfm = metadata.transform_3d;
      const float3 scale = make_float3(index_tfm.x.x, index_tfm.y.y, index_tfm.z.z);
      const float off_diagonal = fabsf(index_tfm.x.y) + fabsf(index_tfm.x.z) +
                                 fabsf(index_tfm.y.x) + fabsf(index_tfm.y.z) +
                                 fabsf(index_tfm.z.x) + fabsf(index_tfm.z.y);
      if (!(reduce_min(fabs(scale)) > 0.0f) || off_diagonal > 1e-5f * reduce_min(fabs(scale))) {
        continue;
      }
      const float voxel_size = 1.0f / reduce_max(fabs(scale));
      if (voxel_size < finest) {
        finest = voxel_size;
        finest_tfm = index_tfm;
      }
    }

    if (finest != FLT_MAX) {
      const float3 a = transform_point(&finest_tfm, box.min);
      const float3 b = transform_point(&finest_tfm, box.max);
      /* A voxel more on every side: lookups inside of the bounds never reach the border. */
      const float3 lo = floor(min(a, b)) - make_float3(1.0f);
      const float3 hi = ceil(max(a, b)) + make_float3(1.0f);
      const float3 count = hi - lo + make_float3(1.0f);
      /* Voxels of the grid per voxel of this one. */
      int stride = 1;
      for (;; stride++) {
        const float3 n = ceil(count / float(stride));
        if (reduce_max(n) <= float(VOLUME_IOR_MAX_RESOLUTION) &&
            double(n.x) * double(n.y) * double(n.z) <= double(VOLUME_IOR_MAX_VOXELS))
        {
          res = max(make_int3(int(n.x), int(n.y), int(n.z)), make_int3(4));
          break;
        }
      }
      /* The center of the first voxel is the value of the grid at `lo`. */
      tfm = transform_scale(make_float3(1.0f / float(stride))) *
            transform_translate(make_float3(0.5f) - lo) * finest_tfm;
      return true;
    }
  }

  /* Size in world space, which the size of a voxel is given in. */
  float3 world_size = size;
  if (!geom->is_light() && !geom->transform_applied) {
    const Transform &object_tfm = object->get_tfm();
    world_size = size * make_float3(len(transform_get_column(&object_tfm, 0)),
                                    len(transform_get_column(&object_tfm, 1)),
                                    len(transform_get_column(&object_tfm, 2)));
  }
  if (!(reduce_max(world_size) > 0.0f) || !isfinite_safe(world_size)) {
    return false;
  }

  float voxel_size = shader->volume_ior_step_size;
  const bool automatic = !(voxel_size > 0.0f);
  if (automatic) {
    const double volume = double(world_size.x) * double(world_size.y) * double(world_size.z);
    voxel_size = float(cbrt(volume / double(VOLUME_IOR_AUTO_VOXELS)));
  }
  const float3 wanted = ceil(world_size / voxel_size);
  float3 n = clamp(wanted,
                   make_float3(float(VOLUME_IOR_MIN_RESOLUTION)),
                   make_float3(float(VOLUME_IOR_MAX_RESOLUTION)));
  const double voxels = double(n.x) * double(n.y) * double(n.z);
  if (voxels > double(VOLUME_IOR_MAX_VOXELS)) {
    n = max(floor(n * float(cbrt(double(VOLUME_IOR_MAX_VOXELS) / voxels))), make_float3(4.0f));
  }

  /* The resolution along an axis follows the field where no size of a voxel was asked for, and
   * where the object is thinner than the smallest grid of such voxels: a layer of air whose
   * index changes across it within a fraction of its thickness. See volume_ior_grid_refine(). */
  grid_box.refine = make_int3(automatic || wanted.x < float(VOLUME_IOR_MIN_RESOLUTION),
                              automatic || wanted.y < float(VOLUME_IOR_MIN_RESOLUTION),
                              automatic || wanted.z < float(VOLUME_IOR_MIN_RESOLUTION));
  grid_box.max_voxels = (automatic) ? 4 * VOLUME_IOR_AUTO_VOXELS : VOLUME_IOR_MAX_VOXELS;
  grid_box.box = box;
  grid_box.n = n;
  volume_ior_grid_box_transform(grid_box, tfm, res);
  return true;
}

/* Finer voxels along the axes where the gradient of the index changes much from one voxel to
 * the next: an exponential profile above a hot road, the thin edge of a plume. Rays cost no more
 * steps for voxels that are finer across their direction, and what the lookups of the gradient
 * add in noise falls with the square of the size of a voxel. Returns whether the grid changed. */
static bool volume_ior_grid_refine(const float *voxels,
                                   VolumeIORGridBox &grid_box,
                                   Transform &tfm,
                                   int3 &res)
{
  if (grid_box.refine == make_int3(0)) {
    return false;
  }

  /* Root mean square of the differences between neighbors, and of the changes of these
   * differences along every axis. */
  const size_t strides[3] = {1, size_t(res.x), size_t(res.x) * res.y};
  struct Sums {
    double first = 0.0;
    double second[3] = {0.0, 0.0, 0.0};
  };
  const Sums sums = parallel_reduce(
      blocked_range<int>(1, res.z - 1),
      Sums(),
      [&](const blocked_range<int> &range, Sums sum) {
        for (int z = range.begin(); z < range.end(); z++) {
          for (int y = 1; y < res.y - 1; y++) {
            const float *row = voxels + (size_t(z) * res.y + y) * res.x;
            for (int x = 1; x < res.x - 1; x++) {
              for (int axis = 0; axis < 3; axis++) {
                const double ahead = double(row[x + strides[axis]]) - double(row[x]);
                const double behind = double(row[x]) - double(row[x - strides[axis]]);
                sum.first += ahead * ahead;
                sum.second[axis] += (ahead - behind) * (ahead - behind);
              }
            }
          }
        }
        return sum;
      },
      [](const Sums &a, const Sums &b) {
        Sums sum;
        sum.first = a.first + b.first;
        for (int axis = 0; axis < 3; axis++) {
          sum.second[axis] = a.second[axis] + b.second[axis];
        }
        return sum;
      });
  if (!(sums.first > 0.0)) {
    return false;
  }

  /* Change of the gradient per voxel that is fine enough, relative to the gradient. */
  const double target = 0.04;
  int factor[3] = {1, 1, 1};
  for (int axis = 0; axis < 3; axis++) {
    if (grid_box.refine[axis]) {
      const double change = std::sqrt(sums.second[axis] / sums.first);
      factor[axis] = clamp(int(std::ceil(change / target)), 1, VOLUME_IOR_MAX_REFINE);
      factor[axis] = min(factor[axis],
                         max(VOLUME_IOR_REFINED_MAX_RESOLUTION / int(grid_box.n[axis]), 1));
    }
  }
  /* Within the budget: the largest factor gives way first. */
  for (;;) {
    const double refined = double(grid_box.n.x) * factor[0] * double(grid_box.n.y) * factor[1] *
                           double(grid_box.n.z) * factor[2];
    const int largest = (factor[0] >= factor[1] && factor[0] >= factor[2]) ? 0 :
                        (factor[1] >= factor[2])                           ? 1 :
                                                                             2;
    if (refined <= double(grid_box.max_voxels) || factor[largest] == 1) {
      break;
    }
    factor[largest]--;
  }
  if (factor[0] == 1 && factor[1] == 1 && factor[2] == 1) {
    return false;
  }

  const float3 refined = make_float3(float(factor[0]), float(factor[1]), float(factor[2]));
  grid_box.n *= refined;
  /* The steps of the rays stay as long as they were. */
  grid_box.step *= refined;
  grid_box.refine = make_int3(0);
  volume_ior_grid_box_transform(grid_box, tfm, res);
  return true;
}

/* How far the gradient that lookups find is from a linear function of the position, per voxel
 * of the grid: the largest second difference of the differences of neighbors, along every
 * axis. */
static void volume_ior_build_curve(const float *voxels, const int3 res, vector<float> &curve)
{
  curve.assign(size_t(res.x) * res.y * res.z, 0.0f);
  const ptrdiff_t strides[3] = {1, ptrdiff_t(res.x), ptrdiff_t(res.x) * res.y};
  parallel_for(0, res.z, [&](const int z) {
    for (int y = 0; y < res.y; y++) {
      for (int x = 0; x < res.x; x++) {
        const int position[3] = {x, y, z};
        const int size[3] = {res.x, res.y, res.z};
        const float *voxel = voxels + (size_t(z) * res.y + y) * res.x + x;
        float largest = 0.0f;
        for (int axis = 0; axis < 3; axis++) {
          if (position[axis] + 1 >= size[axis]) {
            continue;
          }
          for (int along = 0; along < 3; along++) {
            /* The neighbors on both sides, of both voxels of the difference. */
            if (position[along] < 1 || position[along] + 1 + (along == axis) >= size[along]) {
              continue;
            }
            const auto difference = [&](const ptrdiff_t offset) {
              return voxel[offset + strides[axis]] - voxel[offset];
            };
            largest = fmaxf(largest,
                            fabsf(difference(strides[along]) - 2.0f * difference(0) +
                                  difference(-strides[along])));
          }
        }
        curve[(size_t(z) * res.y + y) * res.x + x] = 0.5f * largest;
      }
    }
  });
}

/* A coarse level of a grid, two entries for every block of `1 << shift` voxels a side. The mean
 * of the gradient that lookups in the block find, and a bound of its distance from that mean:
 * the gradient along an axis is interpolated between the differences of neighbors along that
 * axis. And a bound of its distance from a linear function of the position in the block. */
static void volume_ior_build_blocks(
    const float *voxels, const float *curve, const int3 res, const int shift, float4 *blocks)
{
  const int size = 1 << shift;
  const int3 block_res = make_int3(
      (res.x + size - 1) >> shift, (res.y + size - 1) >> shift, (res.z + size - 1) >> shift);
  const size_t strides[3] = {1, size_t(res.x), size_t(res.x) * res.y};
  parallel_for(0, block_res.z, [&](const int bz) {
    for (int by = 0; by < block_res.y; by++) {
      for (int bx = 0; bx < block_res.x; bx++) {
        /* A lookup reaches a voxel out of the block: half of one for the interpolation, and as
         * much for its random position. */
        const int lo[3] = {
            max((bx << shift) - 2, 0), max((by << shift) - 2, 0), max((bz << shift) - 2, 0)};
        const int hi[3] = {min((bx << shift) + size + 1, res.x - 1),
                           min((by << shift) + size + 1, res.y - 1),
                           min((bz << shift) + size + 1, res.z - 1)};

        float mean[3];
        float deviation2 = 0.0f;
        for (int axis = 0; axis < 3; axis++) {
          /* Differences along the axis, between all neighbors in the range. */
          int last[3] = {hi[0], hi[1], hi[2]};
          last[axis]--;
          double sum = 0.0;
          float low = FLT_MAX;
          float high = -FLT_MAX;
          size_t count = 0;
          for (int z = lo[2]; z <= last[2]; z++) {
            for (int y = lo[1]; y <= last[1]; y++) {
              const float *row = voxels + (size_t(z) * res.y + y) * res.x;
              for (int x = lo[0]; x <= last[0]; x++) {
                const float difference = 0.5f * (row[x + strides[axis]] - row[x]);
                sum += difference;
                low = fminf(low, difference);
                high = fmaxf(high, difference);
                count++;
              }
            }
          }
          mean[axis] = (count) ? float(sum / double(count)) : 0.0f;
          /* The gradient is zero behind the border of the grid. */
          const bool border = (axis == 0) ? (bx == 0 || bx == block_res.x - 1) :
                              (axis == 1) ? (by == 0 || by == block_res.y - 1) :
                                            (bz == 0 || bz == block_res.z - 1);
          if (border) {
            low = fminf(low, 0.0f);
            high = fmaxf(high, 0.0f);
          }
          const float deviation = (count) ? fmaxf(high - mean[axis], mean[axis] - low) : 0.0f;
          deviation2 += deviation * deviation;
        }

        /* A function whose second difference is no more than `c` per voxel is no further than
         * `c n^2 / 8` from a linear one over `n` voxels: of all three components, along the
         * diagonal of the block. */
        float curved = 0.0f;
        for (int z = lo[2]; z <= hi[2]; z++) {
          for (int y = lo[1]; y <= hi[1]; y++) {
            const float *row = curve + (size_t(z) * res.y + y) * res.x;
            for (int x = lo[0]; x <= hi[0]; x++) {
              curved = fmaxf(curved, row[x]);
            }
          }
        }
        const float span = float(size + 3);
        const bool at_border = bx == 0 || by == 0 || bz == 0 || bx == block_res.x - 1 ||
                               by == block_res.y - 1 || bz == block_res.z - 1;
        const float deviation = sqrtf(deviation2);
        /* The gradient ends at the border of the grid, which is not linear. */
        const float nonlinear = (at_border) ?
                                    deviation :
                                    fminf(curved * span * span * (3.0f / 8.0f) * M_SQRT3_F,
                                          deviation);

        float4 *block = blocks + 2 * ((size_t(bz) * block_res.y + by) * block_res.x + bx);
        block[0] = make_float4(mean[0], mean[1], mean[2], deviation);
        block[1] = make_float4(nonlinear, 0.0f, 0.0f, 0.0f);
      }
    }
  });
}

void VolumeManager::update_ior_grids(Device *device,
                                     DeviceScene *dscene,
                                     const Scene *scene,
                                     Progress &progress)
{
  if (!(dscene->data.kernel_features & KERNEL_FEATURE_VOLUME_IOR)) {
    if (dscene->volume_ior_object_grid.size() != 0) {
      dscene->volume_ior_object_grid.free();
      dscene->volume_ior_grids.free();
      dscene->volume_ior_voxels.free();
      dscene->volume_ior_blocks.free();
    }
    need_update_ior_ = true;
    return;
  }

  if (!need_update_ior_ && dscene->volume_ior_object_grid.size() == scene->objects.size()) {
    return;
  }

  progress.set_status("Updating Volume Refraction");
  const double time_start = time_dt();

  /* Data needed for volume shader evaluation. */
  device->const_copy_to("data", &dscene->data, sizeof(dscene->data));

  vector<int> object_grid(scene->objects.size(), -1);
  vector<KernelVolumeIORGrid> grids;
  vector<float> voxels;
  vector<float4> blocks;
  const float step_rate = clamp(scene->integrator->get_volume_ior_step_rate(), 0.01f, 100.0f);
  has_world_ior_ = false;

  for (const Object *object : scene->objects) {
    const Geometry *geom = object->get_geometry();
    if (!geom->has_volume) {
      continue;
    }
    if (geom->is_light() && !static_cast<const Light *>(geom)->is_background_light()) {
      continue;
    }
    if (!geom->is_light() && !geom->is_mesh() && !geom->is_volume()) {
      continue;
    }

    for (const Node *node : geom->get_used_shaders()) {
      const Shader *shader = static_cast<const Shader *>(node);
      if (!shader->has_volume || !shader->has_volume_ior) {
        continue;
      }

      Transform tfm;
      int3 res;
      VolumeIORGridBox grid_box;
      if (!volume_ior_grid_layout(scene, object, shader, progress, tfm, res, grid_box)) {
        continue;
      }

      const int object_id = object->index;
      const int shader_id = shader->id;
      const size_t voxel_offset = voxels.size();
      Transform itfm;
      size_t num_voxels = 0;

      /* `n^2 - 1` at the centers of the voxels, from the refractivity `n - 1` the shader
       * gives. */
      const auto evaluate = [&]() {
        itfm = transform_inverse(tfm);
        num_voxels = size_t(res.x) * res.y * res.z;
        voxels.resize(voxel_offset + num_voxels);
        float *grid_voxels = voxels.data() + voxel_offset;
        ShaderEval shader_eval(device, progress);
        return shader_eval.eval(
            SHADER_EVAL_VOLUME_DENSITY,
            int(num_voxels * 2),
            2,
            [&](device_vector<KernelShaderEvalInput> &d_input) {
              KernelShaderEvalInput *d_input_data = d_input.data();
              parallel_for(0, res.z, [&](const int z) {
                for (int y = 0; y < res.y; y++) {
                  for (int x = 0; x < res.x; x++) {
                    const size_t offset = (size_t(z) * res.y + y) * res.x + x;
                    const float3 P = transform_point(
                        &itfm, make_float3(float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f));
                    KernelShaderEvalInput in;
                    in.object = object_id;
                    in.prim = __float_as_int(P.x);
                    in.u = P.y;
                    in.v = P.z;
                    d_input_data[offset * 2] = in;
                    /* A negative voxel size asks for the refractivity. */
                    in.object = shader_id;
                    in.prim = __float_as_int(-1.0f);
                    in.u = 0.0f;
                    in.v = 0.0f;
                    d_input_data[offset * 2 + 1] = in;
                  }
                }
              });
              return int(num_voxels);
            },
            [&](device_vector<float> &d_output) {
              const float *d_output_data = d_output.data();
              parallel_for(size_t(0), num_voxels, [&](const size_t i) {
                const float value = d_output_data[i * 2];
                const float f = (isfinite_safe(value)) ? value : 0.0f;
                /* A shader may give anything: keep the index positive. */
                grid_voxels[i] = fmaxf(f * (2.0f + f), -1.0f + 1e-4f);
              });
            });
      };

      if (!evaluate()) {
        /* Canceled, or failed: without the grids, and again at the next update. */
        return;
      }
      if (volume_ior_grid_refine(voxels.data() + voxel_offset, grid_box, tfm, res)) {
        if (!evaluate()) {
          return;
        }
      }
      const float *grid_voxels = voxels.data() + voxel_offset;
      const auto n2m1 = [&](const int x, const int y, const int z) {
        return grid_voxels[(size_t(z) * res.y + y) * res.x + x];
      };

      KernelVolumeIORGrid grid;
      memset(&grid, 0, sizeof(grid));
      grid.tfm = tfm;
      grid.res_x = res.x;
      grid.res_y = res.y;
      grid.res_z = res.z;
      grid.voxel_offset = int(voxel_offset);
      grid.shader = shader_id;
      grid.step_x = step_rate * grid_box.step.x;
      grid.step_y = step_rate * grid_box.step.y;
      grid.step_z = step_rate * grid_box.step.z;

      vector<float> curve;
      volume_ior_build_curve(voxels.data() + voxel_offset, res, curve);
      for (int level = 0; level < 3; level++) {
        const int shift = (level == 0) ? VOLUME_IOR_BLOCK_SHIFT_1 :
                          (level == 1) ? VOLUME_IOR_BLOCK_SHIFT_2 :
                                         VOLUME_IOR_BLOCK_SHIFT_3;
        const int size = 1 << shift;
        const size_t num_blocks = size_t((res.x + size - 1) >> shift) *
                                  ((res.y + size - 1) >> shift) * ((res.z + size - 1) >> shift);
        const size_t block_offset = blocks.size();
        blocks.resize(block_offset + 2 * num_blocks);
        volume_ior_build_blocks(
            voxels.data() + voxel_offset, curve.data(), res, shift, blocks.data() + block_offset);
        ((level == 0) ? grid.block_offset_1 :
         (level == 1) ? grid.block_offset_2 :
                        grid.block_offset_3) = int(block_offset);
      }

      grid.next = object_grid[object->index];
      object_grid[object->index] = int(grids.size());
      grids.push_back(grid);
      if (geom->is_light()) {
        has_world_ior_ = true;
      }

      LOG_INFO << "Volume index of refraction grid for " << object->get_asset_name() << ": "
               << res.x << " x " << res.y << " x " << res.z;

      static const bool dump_grid = getenv("CYCLES_VOLUME_IOR_DUMP") != nullptr;
      if (dump_grid) {
        /* The bounds of the grid in the space of the geometry, and a row of voxels through its
         * middle along every axis. */
        const float3 lo = transform_point(&itfm, zero_float3());
        const float3 hi = transform_point(&itfm, make_float3(res));
        printf(
            "Volume IOR grid %s: %d x %d x %d, from (%g %g %g) to (%g %g %g), bounds (%g %g %g) "
            "to (%g %g %g)\n",
            object->get_asset_name().c_str(),
            res.x,
            res.y,
            res.z,
            double(lo.x),
            double(lo.y),
            double(lo.z),
            double(hi.x),
            double(hi.y),
            double(hi.z),
            double(geom->bounds.min.x),
            double(geom->bounds.min.y),
            double(geom->bounds.min.z),
            double(geom->bounds.max.x),
            double(geom->bounds.max.y),
            double(geom->bounds.max.z));
        for (int axis = 0; axis < 3; axis++) {
          printf("  n^2 - 1 along %c:", "xyz"[axis]);
          for (int i = 0; i < res[axis]; i++) {
            int3 voxel = make_int3(res.x / 2, res.y / 2, res.z / 2);
            voxel[axis] = i;
            printf(" %.5f", double(n2m1(voxel.x, voxel.y, voxel.z)));
          }
          printf("\n");
        }
      }
    }
  }

  std::copy(object_grid.begin(),
            object_grid.end(),
            dscene->volume_ior_object_grid.alloc(object_grid.size()));
  dscene->volume_ior_object_grid.copy_to_device();
  dscene->volume_ior_object_grid.clear_modified();

  /* Never empty: a device may not bind what has no memory. */
  if (grids.empty()) {
    KernelVolumeIORGrid grid;
    memset(&grid, 0, sizeof(grid));
    grid.shader = SHADER_NONE;
    grid.next = -1;
    grid.step_x = grid.step_y = grid.step_z = 1.0f;
    grids.push_back(grid);
  }
  if (voxels.empty()) {
    voxels.push_back(0.0f);
  }
  if (blocks.empty()) {
    blocks.push_back(zero_float4());
  }
  std::copy(grids.begin(), grids.end(), dscene->volume_ior_grids.alloc(grids.size()));
  dscene->volume_ior_grids.copy_to_device();
  dscene->volume_ior_grids.clear_modified();
  std::copy(voxels.begin(), voxels.end(), dscene->volume_ior_voxels.alloc(voxels.size()));
  dscene->volume_ior_voxels.copy_to_device();
  dscene->volume_ior_voxels.clear_modified();
  std::copy(blocks.begin(), blocks.end(), dscene->volume_ior_blocks.alloc(blocks.size()));
  dscene->volume_ior_blocks.copy_to_device();
  dscene->volume_ior_blocks.clear_modified();

  need_update_ior_ = false;

  LOG_INFO << "Volume index of refraction grids: " << grids.size() << " with " << voxels.size()
           << " voxels in " << time_dt() - time_start << " seconds";
  if (getenv("CYCLES_VOLUME_IOR_DUMP") != nullptr) {
    printf("Volume IOR grids: %d with %d voxels in %.3f seconds\n",
           int(grids.size()),
           int(voxels.size()),
           time_dt() - time_start);
  }
}

void VolumeManager::device_update(Device *device,
                                  DeviceScene *dscene,
                                  const Scene *scene,
                                  Progress &progress)
{
  update_ior_grids(device, dscene, scene, progress);

  if (scene->integrator->get_volume_ray_marching()) {
    /* No need to update octree for ray marching. */
    if (algorithm_modified_) {
      dscene->volume_tree_nodes.free();
      dscene->volume_tree_roots.free();
      dscene->volume_tree_root_ids.free();
    }
    update_step_size(scene, dscene, progress);
    algorithm_modified_ = false;
    return;
  }

  if (need_rebuild_) {
    /* Data needed for volume shader evaluation. */
    device->const_copy_to("data", &dscene->data, sizeof(dscene->data));

    initialize_octree(scene, progress);
    build_octree(device, progress);
    flatten_octree(dscene, scene);

    update_visualization_ = true;
    need_rebuild_ = false;
    update_root_indices_ = false;
  }
  else if (update_root_indices_) {
    update_root_indices(dscene, scene);
    update_root_indices_ = false;
  }

  if (update_visualization_) {
    static const bool dump_octree = getenv("CYCLES_VOLUME_OCTREE_DUMP") != nullptr;
    if (dump_octree) {
      const std::string octree_path = visualize_octree("octree.py");
      LOG_INFO << "Octree visualization has been written to " << octree_path;
    }
    update_visualization_ = false;
  }

  if (algorithm_modified_) {
    dscene->volume_step_size.free();
    algorithm_modified_ = false;
  }
}

void VolumeManager::device_free(DeviceScene *dscene)
{
  dscene->volume_tree_nodes.free();
  dscene->volume_tree_roots.free();
  dscene->volume_tree_root_ids.free();
  dscene->volume_step_size.free();
  dscene->volume_ior_object_grid.free();
  dscene->volume_ior_grids.free();
  dscene->volume_ior_voxels.free();
  dscene->volume_ior_blocks.free();
  need_update_ior_ = true;
}

VolumeManager::~VolumeManager()
{
#ifdef WITH_OPENVDB
  for (auto &it : vdb_map_) {
    it.second.reset();
  }
#endif
}

CCL_NAMESPACE_END
