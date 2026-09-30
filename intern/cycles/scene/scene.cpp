/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <cstdlib>
#include <climits>

#include "bvh/bvh.h"

#include "device/device.h"

#include "scene/background.h"
#include "scene/bake.h"
#include "scene/camera.h"
#include "scene/curves.h"
#include "scene/devicescene.h"
#include "scene/coherent_convex_mesh.h"
#include "scene/coherent_planar_cluster.h"
#include "scene/coherent_sphere_host.h"
#include "scene/diffraction_manager.h"
#include "scene/film.h"
#include "scene/hair.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/pass.h"
#include "scene/osl.h"
#include "scene/particles.h"
#include "scene/pointcloud.h"
#include "scene/procedural.h"
#include "scene/scene.h"
#include "scene/scene_attributes.h"
#include "scene/shader.h"
#include "scene/shader_graph.h"
#include "scene/shader_nodes.h"
#include "scene/svm.h"
#include "scene/tables.h"
#include "scene/volume.h"

#include "kernel/closure/polarizer_axis.h"
#include "kernel/light/coherent_geometry.h"

#include "session/session.h"

#include "util/guarded_allocator.h"
#include "util/log.h"
#include "util/progress.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

CCL_NAMESPACE_BEGIN

/* Detector phase belongs to the declared Lambertian receiver model. Color is
 * a passive radiometric factor, including linked color evaluated at the hit. */
static bool coherent_detector_shader(ShaderNode *node)
{
  const bool principled = node->type == PrincipledBsdfNode::get_node_type();
  if (!principled && node->type != DiffuseBsdfNode::get_node_type()) return false;
  const ShaderInput *color = node->input(principled ? "Base Color" : "Color");
  const ShaderInput *normal = node->input("Normal");
  const ShaderInput *roughness = node->input("Roughness");
  const bool geometry_normal = normal && normal->link &&
                               normal->link->parent->type == GeometryNode::get_node_type() &&
                               normal->link == normal->link->parent->output("Normal");
  if (!color || !normal || !roughness ||
      (!principled && (roughness->link || node->get_float(roughness->socket_type) != 0.0f)) ||
      (normal->link ? !geometry_normal :
                      !is_zero(node->get_float3(normal->socket_type))))
  {
    return false;
  }
  if (!color->link) {
    const float3 value = node->get_float3(color->socket_type);
    if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z) ||
        value.x < 0.0f || value.y < 0.0f || value.z < 0.0f ||
        value.x > 1.0f || value.y > 1.0f || value.z > 1.0f)
    {
      return false;
    }
  }
  if (!principled) return true;
  /* No additional lobes or rough diffuse response may participate. Linked
   * weights cannot be proved zero, even if their socket default is zero. */
  for (const char *name : {"Diffuse Roughness", "Metallic", "Transmission Weight",
                           "Specular IOR Level", "Coat Weight", "Sheen Weight",
                           "Subsurface Weight", "Emission Strength"})
  {
    const ShaderInput *input = node->input(name);
    if (!input || input->link || node->get_float(input->socket_type) != 0.0f) return false;
  }
  /* Specular IOR Level zero suppresses the bare specular lobe, but an
   * effective film recreates that lobe independently of its level. */
  const ShaderInput *film = node->input("Thin Film Thickness");
  if (!film || film->link) return false;
  const float film_thickness = node->get_float(film->socket_type);
  if (!std::isfinite(film_thickness) || film_thickness < 0.0f || film_thickness > 0.1f) {
    return false;
  }
  const ShaderInput *alpha = node->input("Alpha");
  if (!alpha || alpha->link || node->get_float(alpha->socket_type) != 1.0f) return false;
  const ShaderInput *coat_normal = node->input("Coat Normal");
  if (coat_normal && coat_normal->link &&
      !(coat_normal->link->parent->type == GeometryNode::get_node_type() &&
        coat_normal->link == coat_normal->link->parent->output("Normal")))
  {
    return false;
  }
  return true;
}

/* Only exact, single-node ideal graphs can participate. Optical phase is
 * supplied by the declared interface model, never inferred from RGB color. */
static bool coherent_interface_shader(const Shader *shader,
                                      const Object::CoherentInterface mode,
                                      float &ior)
{
  if (!shader || !shader->graph ||
      (shader->graph->output()->input("Volume")->link && mode != Object::COHERENT_INTERFACE_GLASS) ||
      shader->graph->output()->input("Displacement")->link)
  {
    return false;
  }
  const ShaderOutput *surface = shader->graph->output()->input("Surface")->link;
  if (!surface) return false;
  ShaderNode *node = surface->parent;
  if (mode == Object::COHERENT_INTERFACE_DETECTOR) return coherent_detector_shader(node);
  const ShaderInput *color = node->input("Color");
  const ShaderInput *normal = node->input("Normal");
  const ShaderInput *roughness = node->input("Roughness");
  const bool default_geometry_normal = normal && normal->link &&
                                       normal->link->parent->type == GeometryNode::get_node_type() &&
                                       normal->link == normal->link->parent->output("Normal");
  if (!color || !normal || !roughness || color->link ||
      (normal->link && !default_geometry_normal) || roughness->link ||
      !isequal(node->get_float3(color->socket_type), one_float3()) ||
      node->get_float(roughness->socket_type) != 0.0f)
  {
    return false;
  }
  if (mode == Object::COHERENT_INTERFACE_MIRROR) {
    if (node->type != GlossyBsdfNode::get_node_type()) return false;
    GlossyBsdfNode *glossy = static_cast<GlossyBsdfNode *>(node);
    return glossy->get_distribution() == CLOSURE_BSDF_MICROFACET_GGX_ID &&
           glossy->get_diffraction_weight() == 0.0f && !glossy->input("Diffraction Weight")->link &&
           glossy->get_anisotropy() == 0.0f && !glossy->input("Anisotropy")->link;
  }
  if (mode == Object::COHERENT_INTERFACE_GLASS) {
    if (node->type != GlassBsdfNode::get_node_type()) return false;
    GlassBsdfNode *glass = static_cast<GlassBsdfNode *>(node);
    if (glass->get_distribution() != CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID ||
        glass->get_diffraction_weight() != 0.0f || glass->input("Diffraction Weight")->link ||
        glass->get_thin_film_thickness() != 0.0f || glass->input("Thin Film Thickness")->link ||
        glass->input("IOR")->link || !(glass->get_IOR() > 0.0f) ||
        !std::isfinite(glass->get_IOR()))
    {
      return false;
    }
    ior = glass->get_IOR();
    return true;
  }
  return false;
}

/* Homogeneous medium inside a declared streamed Glass volume. Only constant,
 * non-emissive Absorption, Scatter and Principled Volume closures, combined by
 * Add or constant-factor Mix, are accepted; returns the RGB extinction
 * coefficient (per metre) that native volume evaluation uses for meshes. The
 * coherent field keeps the ballistic part exp(-sigma_t d / 2); light scattered
 * by the medium stays in native transport. */
static bool coherent_volume_extinction(const ShaderOutput *output, float3 &sigma, std::string &error)
{
  if (!output) {
    sigma = zero_float3();
    return true;
  }
  ShaderNode *node = output->parent;
  for (const ShaderInput *input : node->inputs) {
    if (input->link && input->type() != SocketType::CLOSURE) {
      error = string("Coherent Glass volume input is not constant: ") + input->name().c_str();
      return false;
    }
  }
  if (node->type == AddClosureNode::get_node_type() || node->type == MixClosureNode::get_node_type()) {
    float3 a, b;
    if (!coherent_volume_extinction(node->input("Closure1")->link, a, error) ||
        !coherent_volume_extinction(node->input("Closure2")->link, b, error))
      return false;
    if (node->type == AddClosureNode::get_node_type()) {
      sigma = a + b;
    }
    else {
      const float fac = clamp(static_cast<MixClosureNode *>(node)->get_fac(), 0.0f, 1.0f);
      sigma = (1.0f - fac) * a + fac * b;
    }
    return true;
  }
  if (node->type == AbsorptionVolumeNode::get_node_type()) {
    const AbsorptionVolumeNode *v = static_cast<const AbsorptionVolumeNode *>(node);
    sigma = (one_float3() - v->get_color()) * max(v->get_density(), 0.0f);
    return true;
  }
  if (node->type == ScatterVolumeNode::get_node_type()) {
    const ScatterVolumeNode *v = static_cast<const ScatterVolumeNode *>(node);
    sigma = v->get_color() * max(v->get_density(), 0.0f);
    return true;
  }
  if (node->type == PrincipledVolumeNode::get_node_type()) {
    const PrincipledVolumeNode *v = static_cast<const PrincipledVolumeNode *>(node);
    if (v->get_emission_strength() != 0.0f || v->get_blackbody_intensity() != 0.0f) {
      error = "Coherent Glass volumes cannot emit";
      return false;
    }
    const float3 color = v->get_color();
    const float3 absorption_color = max(sqrt(v->get_absorption_color()), zero_float3());
    const float3 absorption = max(one_float3() - color, zero_float3()) *
                              max(one_float3() - absorption_color, zero_float3());
    sigma = (color + absorption) * max(v->get_density(), 0.0f);
    return true;
  }
  error = "Coherent Glass volumes support constant Absorption, Scatter or Principled Volume closures";
  return false;
}

/* Polarizer orientation belongs to the physical object, never to a flipped
 * incident normal. The runtime projector uses this same world direction. */
static bool coherent_patch_polarizer(const Shader *shader,
                                    const Object *object,
                                    KernelCoherentPatch &patch,
                                    const float3 object_normal = make_float3(0.0f, 0.0f, 1.0f))
{
  if (!shader || !shader->graph) return true;
  const ShaderOutput *surface = shader->graph->output()->input("Surface")->link;
  if (!surface || surface->parent->type != GlassBsdfNode::get_node_type()) return true;
  GlassBsdfNode *glass = static_cast<GlassBsdfNode *>(surface->parent);
  if (glass->input("Polarizer")->link) return false;
  patch.polarizer = glass->get_polarizer() != 0;
  if (!patch.polarizer) return true;
  if (glass->input("Polarizer Angle")->link) return false;
  const float angle = glass->get_polarizer_angle();
  if (!std::isfinite(angle)) return false;
  const Transform tfm = object->get_tfm();
  /* Same in-film axis convention as the native Glass closure. */
  const float3 axis = transform_direction(&tfm, polarizer_object_axis(angle, object_normal));
  if (!isfinite_safe(axis) || !(len_squared(axis) > 1.0e-12f)) return false;
  patch.polarizer_axis = normalize(axis);
  return true;
}

/* These passes describe the camera-visible surface, not a decomposition of
 * incident illumination. Coherent fields do not change their native writers.
 * Light passes still need an explicit convention for cross-path pair terms. */
static bool coherent_specular_pass_supported(const PassType type)
{
  switch (type) {
    case PASS_COMBINED:
    case PASS_DIFFUSE:
    case PASS_DIFFUSE_DIRECT:
    case PASS_DIFFUSE_INDIRECT:
    case PASS_GLOSSY:
    case PASS_GLOSSY_DIRECT:
    case PASS_GLOSSY_INDIRECT:
    case PASS_TRANSMISSION:
    case PASS_TRANSMISSION_DIRECT:
    case PASS_TRANSMISSION_INDIRECT:
    case PASS_EMISSION:
    case PASS_BACKGROUND:
    case PASS_DEPTH:
    case PASS_POSITION:
    case PASS_NORMAL:
    case PASS_ROUGHNESS:
    case PASS_UV:
    case PASS_OBJECT_ID:
    case PASS_MATERIAL_ID:
    case PASS_CRYPTOMATTE:
    case PASS_AOV_COLOR:
    case PASS_AOV_VALUE:
    case PASS_SAMPLE_COUNT:
    case PASS_DIFFUSE_COLOR:
    case PASS_GLOSSY_COLOR:
    case PASS_TRANSMISSION_COLOR:
    case PASS_MIST:
    case PASS_DENOISING_ALBEDO:
    case PASS_DENOISING_SPECULAR_ALBEDO:
    case PASS_DENOISING_NORMAL:
    case PASS_DENOISING_ROUGHNESS:
    case PASS_DENOISING_DEPTH:
    case PASS_MOTION:
    case PASS_MOTION_WEIGHT:
    case PASS_DENOISING_BACKWARD_MOTION:
    case PASS_DENOISING_SPECULAR_MOTION:
    /* Internal bookkeeping passes added by viewport/adaptive sampling,
     * guiding, denoising history and volume majorants. They carry no
     * light decomposition; coherent fields enter only Combined. */
    /* Volume light passes hold only native medium scattering; coherent
     * ballistic fields never write them. */
    case PASS_VOLUME:
    case PASS_VOLUME_DIRECT:
    case PASS_VOLUME_INDIRECT:
    case PASS_VOLUME_SCATTER:
    case PASS_VOLUME_TRANSMIT:
    case PASS_ADAPTIVE_AUX_BUFFER:
    case PASS_RENDER_TIME:
    case PASS_GUIDING_COLOR:
    case PASS_GUIDING_PROBABILITY:
    case PASS_GUIDING_AVG_ROUGHNESS:
    case PASS_VOLUME_MAJORANT:
    case PASS_VOLUME_MAJORANT_SAMPLE_COUNT:
    case PASS_DENOISING_PREVIOUS:
      return true;
    default:
      return false;
  }
}

/* The deterministic detector estimator is shared kernel code. CPU renders it
 * with path tracing (BDPT and GPU guiding are Metal features and fall back to
 * PT elsewhere); Metal additionally runs the BDPT prefix ownership. Other GPU
 * backends are not validated for this transport. */
static bool coherent_specular_device_supported(const DeviceInfo &info)
{
  if (info.type == DEVICE_CPU || info.type == DEVICE_METAL) {
    return true;
  }
  if (info.type == DEVICE_MULTI && !info.multi_devices.empty()) {
    for (const DeviceInfo &subdevice : info.multi_devices) {
      if (!coherent_specular_device_supported(subdevice)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

static bool scene_prepare_coherent_specular(Scene *scene, DeviceScene *dscene, Progress &progress)
{
  KernelIntegrator &ki = dscene->data.integrator;
  ki.coherent_specular_enabled = 0;
  ki.coherent_patch_count = 0;
  ki.coherent_candidate_count = 0;
  ki.coherent_max_interface_events = 0;
  if (!scene->integrator->get_use_coherent_specular_connections()) {
    dscene->coherent_patches.free();
    dscene->coherent_patch_primitives.free();
    dscene->coherent_candidates.free();
    return true;
  }
  const bool stream_facets = scene->integrator->get_coherent_transport_mode() == 1;
  const int max_events = scene->integrator->get_coherent_max_interface_events();
  if (stream_facets && (max_events < 1 || max_events > 4)) {
    progress.set_error("Streamed Facets requires Max Interface Events 1 to 4; k events cost O(triangles^k)");
    return false;
  }
  if (max_events < 1 || max_events > 4) {
    progress.set_error("Coherent specular connections require 1 to 4 interface events");
    return false;
  }
  if (!coherent_specular_device_supported(scene->device->info)) {
    progress.set_error("Coherent specular connections require CPU or Metal devices");
    return false;
  }
  if (scene->integrator->get_max_bounce() < max_events + 1 ||
      (scene->integrator->use_bidirectional_path_tracing_on_device(scene->device) &&
       scene->integrator->get_bdpt_max_bounces() < max_events + 1))
  {
    progress.set_error("Coherent connections require sufficient global and enabled BDPT interface bounce limits");
    return false;
  }
  if (scene->integrator->get_sample_clamp_direct() != 0.0f ||
      scene->integrator->get_sample_clamp_indirect() != 0.0f || scene->has_shadow_catcher())
  {
    progress.set_error("Coherent specular connections require zero sample clamps and no shadow catcher");
    return false;
  }
  for (const Pass *pass : scene->passes) {
    if (!coherent_specular_pass_supported(pass->get_type()) || !pass->get_lightgroup().empty()) {
      progress.set_error(string("Coherent specular connections support Combined, signed surface light decomposition and surface data passes; this pass or lightgroup is unsupported: ") +
                         pass_type_as_string(pass->get_type()));
      return false;
    }
  }

  std::vector<KernelCoherentPatch> patches;
  std::vector<int> patch_primitives;
  int detector_count = 0;
  /* Streamed Glass: validated convex hulls, and world bounds of every declared
   * streamed object, for the exterior-endpoint and non-nesting requirements. */
  struct StreamedBounds {
    const Object *object;
    bool glass;
    std::array<double, 3> lower, upper;
    std::vector<CoherentPlanarTriangle> triangles; /* World, outward for Glass. */
    /* Rigid motion: the same triangles at every exported object motion step. */
    std::vector<std::vector<CoherentPlanarTriangle>> motion_triangles;
  };
  std::vector<std::pair<int, CoherentConvexHull>> glass_hulls;
  std::vector<StreamedBounds> streamed_bounds;
  auto mesh_world_bounds = [](const Object *object, const Mesh *mesh, StreamedBounds &bounds) {
    const packed_float3 *positions = mesh->get_position();
    const Transform tfm = object->get_tfm();
    if (bounds.triangles.empty()) {
      for (size_t i = 0; i < mesh->num_triangles(); i++) {
        const Mesh::Triangle tri = mesh->get_triangle(i);
        CoherentPlanarTriangle triangle{};
        for (int j = 0; j < 3; j++) {
          float3 point = positions[tri.v[j]];
          if (!mesh->transform_applied) point = transform_point(&tfm, point);
          triangle.point[j] = {double(point.x), double(point.y), double(point.z)};
          triangle.vertex[j] = tri.v[j];
        }
        bounds.triangles.push_back(triangle);
      }
    }
    bounds.lower = {DBL_MAX, DBL_MAX, DBL_MAX};
    bounds.upper = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
    bounds.motion_triangles.clear();
    if (object->use_motion() && !mesh->transform_applied) {
      /* The same triangles (vertex IDs keep any orientation swap) placed by
       * every exported object motion step. */
      for (const Transform &step : object->get_motion()) {
        std::vector<CoherentPlanarTriangle> moved = bounds.triangles;
        for (CoherentPlanarTriangle &t : moved) {
          for (int j = 0; j < 3; j++) {
            auto &q = t.point[j];
            const float3 world = transform_point(&step, float3(positions[t.vertex[j]]));
            q = {double(world.x), double(world.y), double(world.z)};
            for (int axis = 0; axis < 3; axis++) {
              bounds.lower[axis] = std::min(bounds.lower[axis], q[axis]);
              bounds.upper[axis] = std::max(bounds.upper[axis], q[axis]);
            }
          }
        }
        bounds.motion_triangles.push_back(std::move(moved));
      }
    }
    for (size_t i = 0; i < mesh->num_verts(); i++) {
      float3 point = positions[i];
      if (!mesh->transform_applied) point = transform_point(&tfm, point);
      const double p[3] = {double(point.x), double(point.y), double(point.z)};
      for (int axis = 0; axis < 3; axis++) {
        bounds.lower[axis] = std::min(bounds.lower[axis], p[axis]);
        bounds.upper[axis] = std::max(bounds.upper[axis], p[axis]);
      }
    }
  };
  for (const Object *object : scene->objects) {
    if (progress.get_cancel()) return false;
    if (object->has_light_linking()) {
      progress.set_error(string("Coherent specular connections do not support light linking on object ") + object->name.c_str());
      return false;
    }
    if (object->has_shadow_linking()) {
      progress.set_error(string("Coherent specular connections do not support shadow linking on object ") + object->name.c_str());
      return false;
    }
    if (!object->get_lightgroup().empty()) {
      progress.set_error(string("Coherent specular connections do not support light group on object ") + object->name.c_str());
      return false;
    }
    if (object->get_is_shadow_catcher() && !object->get_geometry()->is_light()) {
      progress.set_error(string("Coherent specular connections do not support shadow catcher on object ") + object->name.c_str());
      return false;
    }
    const Geometry *geometry = object->get_geometry();
    if (geometry->has_volume && !(stream_facets && object->get_coherent_interface() ==
                                                       Object::COHERENT_INTERFACE_GLASS &&
                                  geometry->is_mesh()))
    {
      progress.set_error("Coherent specular connections support volumes only inside streamed Glass meshes");
      return false;
    }
    const Object::CoherentInterface mode = object->get_coherent_interface();
    for (Node *shader_node : geometry->get_used_shaders()) {
      const Shader *shader = static_cast<const Shader *>(shader_node);
      if (mode != Object::COHERENT_INTERFACE_GLASS &&
          shader->has_surface_shadow_transparency())
      {
        progress.set_error(string("Coherent connections require opaque shadows on unmarked or non-glass object ") +
                           object->name.c_str());
        return false;
      }
    }
    if (mode == Object::COHERENT_INTERFACE_OFF) continue;
    const uint required_visibility = PATH_RAY_VISIBILITY_CAMERA |
                                     PATH_RAY_VISIBILITY_DIFFUSE |
                                     PATH_RAY_VISIBILITY_GLOSSY |
                                     PATH_RAY_VISIBILITY_SHADOW_OPAQUE;
    if ((object->get_visibility() & required_visibility) != required_visibility) {
      progress.set_error(string("Coherent interface requires camera, diffuse, glossy, and opaque-shadow visibility on object ") +
                         object->name.c_str());
      return false;
    }
    /* Streamed facets evaluate every facet at the camera sample's time, so rigid
     * object motion is supported there; the bounded inventory stores static
     * patches. A deforming Glass volume could lose closedness or orientation. */
    if (geometry->has_true_displacement() ||
        (!stream_facets && (object->use_motion() || geometry->get_use_motion_blur())) ||
        (stream_facets && mode == Object::COHERENT_INTERFACE_GLASS &&
         geometry->attributes.find(ATTR_STD_POSITION) &&
         geometry->attributes.find(ATTR_STD_POSITION)->has_motion()))
    {
      progress.set_error(stream_facets ?
                             "Coherent interfaces require undisplaced geometry; streamed Glass may move rigidly but not deform" :
                             "Coherent interfaces require static undisplaced geometry");
      return false;
    }
    if (stream_facets && mode != Object::COHERENT_INTERFACE_DETECTOR &&
        ((mode != Object::COHERENT_INTERFACE_MIRROR && mode != Object::COHERENT_INTERFACE_GLASS) ||
         !geometry->is_mesh()))
    {
      progress.set_error("Streamed Facets supports flat triangle Mirror objects, closed convex Glass meshes and Lambertian detectors only");
      return false;
    }
    if (geometry->is_pointcloud()) {
      if (mode != Object::COHERENT_INTERFACE_MIRROR && mode != Object::COHERENT_INTERFACE_GLASS) {
        progress.set_error("Coherent native spheres require ideal Mirror or Glass materials");
        return false;
      }
      if (mode == Object::COHERENT_INTERFACE_GLASS && max_events < 2) {
        progress.set_error("Coherent native Glass spheres require at least two events");
        return false;
      }
      const PointCloud *points = static_cast<const PointCloud *>(geometry);
      if (points->num_points() == 0 || patches.size() + points->num_points() > 64) {
        progress.set_error("Coherent native sphere interfaces require 1 to 64 bounded patches");
        return false;
      }
      if (points->prim_offset > size_t(INT_MAX) ||
          points->num_points() > size_t(INT_MAX) - points->prim_offset ||
          patch_primitives.size() > size_t(INT_MAX) - points->num_points())
      {
        progress.set_error("Coherent native sphere primitive IDs exceed supported integer range");
        return false;
      }
      const auto &used_shaders = points->get_used_shaders();
      const auto &shader_indices = points->get_shader();
      Shader *material = nullptr;
      for (size_t i = 0; i < points->num_points(); ++i) {
        const int index = shader_indices[i];
        if (index < 0 || size_t(index) >= used_shaders.size()) {
          progress.set_error("Coherent native sphere has an unassigned material point");
          return false;
        }
        Shader *point_material = static_cast<Shader *>(used_shaders[index]);
        if (material && material != point_material) {
          progress.set_error("Coherent native sphere points must use one material");
          return false;
        }
        material = point_material;
      }
      float sphere_ior = 1.0f;
      if (!coherent_interface_shader(material, mode, sphere_ior)) {
        progress.set_error("Coherent native sphere requires a white smooth single-node ideal shader");
        return false;
      }
      if (mode == Object::COHERENT_INTERFACE_GLASS &&
          (scene->integrator->get_coherent_polarization_mode() != 1 || !(sphere_ior > 1.0f))) {
        progress.set_error("Coherent native Glass spheres require the vector polarization model and IOR greater than one");
        return false;
      }
      const Transform tfm = object->get_tfm();
      /* A baked point primitive is exactly the world sphere stored by Cycles.
       * Unbaked general affine transforms represent ellipsoids, including
       * approximately orthogonal float matrices. Do not silently fit these. */
      float scale = 1.0f;
      if (!points->transform_applied) {
        std::array<std::array<float, 3>, 3> columns;
        for (int column = 0; column < 3; ++column) {
          const float3 axis = transform_get_column(&tfm, column);
          columns[column] = {axis.x, axis.y, axis.z};
        }
        if (!coherent_sphere_transform_scale(columns, scale)) {
          progress.set_error("Coherent instanced native spheres require an exact axis-aligned uniform transform; bake rotations and nonrepresentable scaled radii");
          return false;
        }
      }
      const packed_float3 *centers = points->get_position();
      const float *radii = points->get_radius();
      for (size_t i = 0; i < points->num_points(); ++i) {
        const float3 center = points->transform_applied ? float3(centers[i]) :
                                                        transform_point(&tfm, centers[i]);
        float radius = 0.0f;
        const bool valid_radius = coherent_sphere_world_radius(
            radii[i], scale, points->transform_applied, radius);
        if (!std::isfinite(center.x) || !std::isfinite(center.y) ||
            !std::isfinite(center.z) || !valid_radius)
        {
          progress.set_error("Coherent native spheres require finite positive exactly represented world radii; bake nonrepresentable scaled radii");
          return false;
        }
        KernelCoherentPatch patch{};
        patch.center = center;
        patch.tangent_u = make_float3(1.0f, 0.0f, 0.0f);
        patch.tangent_v = make_float3(0.0f, 1.0f, 0.0f);
        patch.outside_ior = 1.0f;
        patch.inside_ior = sphere_ior;
        patch.object = object->index;
        patch.mode = int(mode);
        if (!coherent_patch_polarizer(material, object, patch)) {
          progress.set_error("Coherent polarizers require a finite constant angle and unlinked checkbox");
          return false;
        }
        patch.primitive_offset = int(patch_primitives.size());
        patch.primitive_count = 1;
        patch.shape = 1;
        patch.radius = radius;
        patch.primitive_type = PRIMITIVE_POINT;
        patch_primitives.push_back(int(points->prim_offset + i));
        patches.push_back(patch);
      }
      continue;
    }
    if (!geometry->is_mesh()) {
      progress.set_error("Coherent interfaces require triangle meshes or declared native Mirror point spheres");
      return false;
    }
    const Mesh *mesh = static_cast<const Mesh *>(geometry);
    /* Geometry Nodes may export an empty mesh companion beside a real point
     * component. It carries the original object's optical declaration but
     * has no intersections to own. Do not reject that empty component or
     * manufacture a planar patch; the final scene guard still requires a
     * nonempty ideal interface and a nonempty Lambertian detector. */
    if (mode != Object::COHERENT_INTERFACE_DETECTOR && mesh->num_triangles() == 0) {
      continue;
    }
    if (mode == Object::COHERENT_INTERFACE_DETECTOR &&
        (object->get_shadow_terminator_shading_offset() != 0.0f ||
         mesh->attributes.find(ATTR_STD_CORNER_NORMAL)))
    {
      progress.set_error("Coherent Lambertian detectors require flat normals and zero shading terminator offset");
      return false;
    }
    if (mesh->num_triangles() == 0 || mesh->get_subdivision_type() != Mesh::SUBDIVISION_NONE) {
      progress.set_error("Coherent interfaces require nonempty ordinary triangle meshes");
      return false;
    }
    if (mode != Object::COHERENT_INTERFACE_DETECTOR &&
        mesh->attributes.find(ATTR_STD_CORNER_NORMAL)) {
      progress.set_error("Coherent ideal interfaces require geometric flat normals");
      return false;
    }
    const auto &used_shaders = mesh->get_used_shaders();
    const auto &shader_indices = mesh->get_shader();
    Shader *material = nullptr;
    for (size_t i = 0; i < mesh->num_triangles(); ++i) {
      if (mode == Object::COHERENT_INTERFACE_DETECTOR && mesh->get_smooth()[i]) {
        progress.set_error("Coherent Lambertian detectors require flat normals and zero shading terminator offset");
        return false;
      }
      if (mode != Object::COHERENT_INTERFACE_DETECTOR && mesh->get_smooth()[i]) {
        progress.set_error("Coherent ideal interfaces cannot use smooth shading normals");
        return false;
      }
      const int index = shader_indices[i];
      if (index < 0 || size_t(index) >= used_shaders.size()) {
        progress.set_error("Coherent interface has an unassigned material triangle");
        return false;
      }
      Shader *triangle_material = static_cast<Shader *>(used_shaders[index]);
      if (material && triangle_material != material) {
        progress.set_error("Coherent interface triangles must use one material");
        return false;
      }
      material = triangle_material;
    }
    float inside_ior = 1.0f;
    if (!coherent_interface_shader(material, mode, inside_ior)) {
      progress.set_error(string("Coherent interface on object ") + object->name.c_str() +
                         (mode == Object::COHERENT_INTERFACE_DETECTOR ?
                              " requires one passive Lambertian Diffuse or pure diffuse Principled shader" :
                              " requires a matching white, smooth, single-node ideal shader"));
      return false;
    }
    if (mode == Object::COHERENT_INTERFACE_DETECTOR) {
      ++detector_count;
      if (stream_facets) {
        StreamedBounds bounds{object, false, {}, {}, {}};
        mesh_world_bounds(object, mesh, bounds);
        streamed_bounds.push_back(bounds);
      }
      continue;
    }
    if (mode == Object::COHERENT_INTERFACE_GLASS) {
      if (scene->integrator->get_coherent_polarization_mode() != 1) {
        progress.set_error("Coherent Glass interfaces require the Vector Dipole Ensemble polarization model");
        return false;
      }
      if (!(inside_ior >= 1.0f)) {
        progress.set_error("Coherent planar Glass interface IOR must be at least exterior air IOR 1");
        return false;
      }
    }

    const packed_float3 *positions = mesh->get_position();
    const Transform tfm = object->get_tfm();
    std::vector<CoherentPlanarTriangle> triangles;
    triangles.reserve(mesh->num_triangles());
    /* Geometry update has finalized global primitive offsets before this scene
     * preparation. Preserve vertex IDs for topology, including instances. */
    if (mesh->prim_offset > size_t(INT_MAX) ||
        mesh->num_triangles() > size_t(INT_MAX) - mesh->prim_offset)
    {
      progress.set_error("Coherent interface primitive IDs exceed supported integer range");
      return false;
    }
    for (size_t i = 0; i < mesh->num_triangles(); ++i) {
      const Mesh::Triangle tri = mesh->get_triangle(i);
      if (!tri.valid(positions)) {
        progress.set_error("Coherent interface contains a degenerate triangle");
        return false;
      }
      CoherentPlanarTriangle triangle{};
      triangle.primitive = int(mesh->prim_offset + i);
      for (int j = 0; j < 3; ++j) {
        float3 point = positions[tri.v[j]];
        if (!mesh->transform_applied)
          point = transform_point(&tfm, point);
        if (stream_facets && !isfinite_safe(point)) {
          progress.set_error("Streamed mirror facets require finite world vertices");
          return false;
        }
        triangle.point[j] = {double(point.x), double(point.y), double(point.z)};
        triangle.vertex[j] = tri.v[j];
      }
      triangles.push_back(triangle);
    }
    if (stream_facets && mode == Object::COHERENT_INTERFACE_GLASS) {
      /* Native Cycles orients Ng by the object-space winding: world-space
       * cross products flip under a negative-determinant object transform. */
      if (transform_negative_scale(tfm)) {
        for (CoherentPlanarTriangle &triangle : triangles) {
          std::swap(triangle.point[1], triangle.point[2]);
          std::swap(triangle.vertex[1], triangle.vertex[2]);
        }
      }
      CoherentConvexHull hull;
      std::string hull_error;
      /* Concave volumes are supported: every route leg is BVH tested, so a
       * chord leaving and re-entering the volume is blocked, not assumed. */
      if (!coherent_convex_mesh_validate(triangles, hull, hull_error, false)) {
        progress.set_error(hull_error + " (object " + object->name.c_str() + ")");
        return false;
      }
      glass_hulls.emplace_back(object->index, std::move(hull));
    }
    if (stream_facets) {
      StreamedBounds bounds{object, mode == Object::COHERENT_INTERFACE_GLASS, {}, {}, {}};
      if (bounds.glass) bounds.triangles = triangles; /* Outward winding. */
      mesh_world_bounds(object, mesh, bounds);
      streamed_bounds.push_back(bounds);
      if (patches.size() >= size_t(INT_MAX)) {
        progress.set_error("Streamed mirror object count exceeds integer range");
        return false;
      }
      KernelCoherentPatch patch{};
      patch.object = object->index;
      patch.mode = int(mode);
      patch.outside_ior = 1.0f;
      patch.inside_ior = mode == Object::COHERENT_INTERFACE_GLASS ? inside_ior : 1.0f;
      if (mode == Object::COHERENT_INTERFACE_GLASS) {
        float3 sigma;
        std::string volume_error;
        if (!coherent_volume_extinction(material->graph->output()->input("Volume")->link, sigma,
                                        volume_error) ||
            !isfinite_safe(sigma))
        {
          progress.set_error(volume_error + " (object " + object->name.c_str() + ")");
          return false;
        }
        patch.extinction = sigma;
      }
      /* One axis per mesh: use the dominant object-space facet orientation,
       * which is the sheet normal of film and slab polarizers. */
      float3 area = zero_float3();
      for (size_t i = 0; i < mesh->num_triangles(); ++i) {
        const Mesh::Triangle tri = mesh->get_triangle(i);
        area += fabs(cross(positions[tri.v[1]] - positions[tri.v[0]],
                           positions[tri.v[2]] - positions[tri.v[0]]));
      }
      if (!coherent_patch_polarizer(material, object, patch, area)) {
        progress.set_error("Coherent polarizers require a finite constant angle and unlinked checkbox");
        return false;
      }
      patch.shape = 2; /* One mesh range, not a planar patch. */
      patch.primitive_type = PRIMITIVE_TRIANGLE;
      patch.primitive_offset = int(mesh->prim_offset);
      patch.primitive_count = int(mesh->num_triangles());
      patches.push_back(patch);
      continue;
    }
    std::vector<CoherentPlanarCluster> clusters;
    std::string cluster_error;
    if (!coherent_planar_cluster_build(triangles, clusters, cluster_error)) {
      progress.set_error(cluster_error);
      return false;
    }
    if (patches.size() + clusters.size() > 64) {
      progress.set_error("Coherent interfaces exceed the 64 planar patch history limit");
      return false;
    }
    for (const CoherentPlanarCluster &cluster : clusters) {
      if (patch_primitives.size() > size_t(INT_MAX) ||
          cluster.primitives.size() > size_t(INT_MAX) - patch_primitives.size())
      {
        progress.set_error("Coherent triangle membership exceeds supported integer range");
        return false;
      }
      KernelCoherentPatch patch{};
      const double center_u = 0.5 * (cluster.min_u + cluster.max_u);
      const double center_v = 0.5 * (cluster.min_v + cluster.max_v);
      patch.center = make_float3(float(cluster.origin[0] + cluster.tangent_u[0] * center_u +
                                       cluster.tangent_v[0] * center_v),
                                 float(cluster.origin[1] + cluster.tangent_u[1] * center_u +
                                       cluster.tangent_v[1] * center_v),
                                 float(cluster.origin[2] + cluster.tangent_u[2] * center_u +
                                       cluster.tangent_v[2] * center_v));
      patch.tangent_u = make_float3(
          float(cluster.tangent_u[0]), float(cluster.tangent_u[1]), float(cluster.tangent_u[2]));
      patch.tangent_v = make_float3(
          float(cluster.tangent_v[0]), float(cluster.tangent_v[1]), float(cluster.tangent_v[2]));
      patch.half_u = float(0.5 * (cluster.max_u - cluster.min_u));
      patch.half_v = float(0.5 * (cluster.max_v - cluster.min_v));
      patch.outside_ior = 1.0f;
      patch.inside_ior = inside_ior;
      patch.object = object->index;
      patch.mode = int(mode);
      if (!coherent_patch_polarizer(material, object, patch,
                                    transform_direction_transposed(
                                        &tfm, cross(patch.tangent_u, patch.tangent_v))))
      {
        progress.set_error("Coherent polarizers require a finite constant angle and unlinked checkbox");
        return false;
      }
      patch.primitive_offset = int(patch_primitives.size());
      patch.primitive_count = int(cluster.primitives.size());
      patch.shape = 0;
      patch.primitive_type = PRIMITIVE_TRIANGLE;
      patch_primitives.insert(
          patch_primitives.end(), cluster.primitives.begin(), cluster.primitives.end());
      patches.push_back(patch);
    }
  }
  if (detector_count == 0 || patches.empty()) {
    progress.set_error(
        "Coherent specular connections require a marked Lambertian detector and at least one "
        "ideal interface");
    return false;
  }
  /* Streamed Glass supports exterior-air endpoints and non-nested volumes:
   * no declared object may share bounds with a Glass volume. */
  for (const StreamedBounds &glass : streamed_bounds) {
    if (!glass.glass) continue;
    for (const StreamedBounds &other : streamed_bounds) {
      if (other.object == glass.object) continue;
      bool overlap = true;
      for (int axis = 0; axis < 3; axis++) {
        overlap &= glass.lower[axis] <= other.upper[axis] && other.lower[axis] <= glass.upper[axis];
      }
      if (!overlap) continue;
      /* Exact test at the center time and at every motion step: no surface
       * contact and no nesting (a vertex of either object inside the other
       * closed Glass volume). All step pairs are tested conservatively. */
      auto sets = [](const StreamedBounds &b) {
        std::vector<const std::vector<CoherentPlanarTriangle> *> result{&b.triangles};
        for (const auto &step : b.motion_triangles) result.push_back(&step);
        return result;
      };
      bool nested = false;
      for (const auto *g : sets(glass)) {
        for (const auto *o : sets(other)) {
          if (nested) break;
          nested = coherent_triangle_sets_touch(*g, *o);
          for (const CoherentPlanarTriangle &t : *o) {
            if (nested) break;
            nested |= !coherent_closed_mesh_strictly_outside(*g, t.point[0]);
          }
          if (!nested && other.glass) {
            nested = !coherent_closed_mesh_strictly_outside(*o, (*g)[0].point[0]);
          }
        }
      }
      if (nested) {
        progress.set_error(string("Streamed Glass object ") + glass.object->name.c_str() +
                           " touches or nests with declared object " + other.object->name.c_str() +
                           "; nested or touching coherent volumes are unsupported");
        return false;
      }
    }
  }
  bool has_mirror = false, has_glass = false;
  for (const KernelCoherentPatch &patch : patches) {
    has_mirror |= patch.mode == Object::COHERENT_INTERFACE_MIRROR;
    has_glass |= patch.mode == Object::COHERENT_INTERFACE_GLASS;
  }
  if (has_mirror && (!scene->integrator->get_caustics_reflective() ||
                     scene->integrator->get_max_glossy_bounce() < 1))
  {
    progress.set_error("Coherent Mirror interfaces require reflective caustics and at least one glossy bounce");
    return false;
  }
  if (has_glass && (!scene->integrator->get_caustics_refractive() ||
                    scene->integrator->get_max_transmission_bounce() < 1))
  {
    progress.set_error("Coherent Glass interfaces require refractive caustics and at least one transmission bounce");
    return false;
  }

  /* The kernel stores completed fields only for routes that connect at the
   * shaded detector point (at most COHERENT_SPECULAR_MAX_PATHS). Conservative
   * root-branch reservations may exceed that; overflow at a pixel is reported
   * as a render error by the kernel, never silently dropped. */
  constexpr size_t candidate_limit = 256;
  constexpr size_t search_limit = 256;
  std::vector<KernelCoherentCandidate> candidates;
  size_t searched_states = 0;
  const KernelLight *lights = dscene->lights.data();
  for (int lamp = 0; lamp < ki.num_lights; ++lamp) {
    if (progress.get_cancel()) return false;
    const KernelLight &light = lights[lamp];
    if (light.coherence_group <= 0 || light.coherence_length <= 0.0f) continue;
    if (stream_facets && light.coherence_length < 32.0f*FLT_MIN) {
      progress.set_error("Streamed Gaussian coherence length is below normal float arithmetic range");
      return false;
    }
    for (const StreamedBounds &glass : streamed_bounds) {
      bool inside = glass.glass && !coherent_closed_mesh_strictly_outside(
          glass.triangles, {double(light.co.x), double(light.co.y), double(light.co.z)});
      for (const auto &step : glass.motion_triangles) {
        inside |= glass.glass && !coherent_closed_mesh_strictly_outside(
            step, {double(light.co.x), double(light.co.y), double(light.co.z)});
      }
      if (inside)
      {
        progress.set_error("Streamed Glass sources must lie strictly outside every declared Glass volume");
        return false;
      }
    }
    for (const KernelCoherentPatch &patch : patches) {
      if (patch.shape == 1 &&
          !coherent_sphere_source_outside({light.co.x, light.co.y, light.co.z},
                                          {patch.center.x, patch.center.y, patch.center.z},
                                          patch.radius))
      {
        progress.set_error("Coherent native sphere sources must lie strictly outside every declared sphere");
        return false;
      }
    }
    if ((light.shader_id & (SHADER_EXCLUDE_DIFFUSE | SHADER_EXCLUDE_GLOSSY)) != 0 ||
        light.max_bounces < float(max_events + 1))
    {
      progress.set_error("Coherent participating lights require unrestricted visibility and sufficient light bounce limits");
      return false;
    }
    if (!stream_facets && candidates.size() >= candidate_limit) {
      progress.set_error("Coherent specular connection candidate count exceeds 256");
      return false;
    }
    KernelCoherentCandidate direct{};
    direct.light = lamp;
    candidates.push_back(direct);
    if (stream_facets) continue;
    std::function<bool(const KernelCoherentCandidate &, int, float, int, int)> enumerate =
        [&](const KernelCoherentCandidate &candidate,
            const int depth,
            const float medium_ior,
            const int reflections,
            const int transmissions) {
          if (depth == max_events) return true;
          for (size_t patch = 0; patch < patches.size(); ++patch) {
            if (progress.get_cancel()) return false;
            const KernelCoherentPatch &interface = patches[patch];
            if (depth > 0 && candidate.patch[depth - 1] == int(patch) &&
                !(interface.shape == 1 && interface.mode == Object::COHERENT_INTERFACE_GLASS &&
                  medium_ior == interface.inside_ior)) continue;
            const bool is_glass = interface.mode == Object::COHERENT_INTERFACE_GLASS;
            int incident_side = 0;
            float opposite_ior = medium_ior;
            if (is_glass) {
              /* A matched analyzer has no medium boundary and is physically
               * two-sided. Geometry still checks R/T side topology; history
               * treats expected side zero as a wildcard. */
              if (interface.inside_ior == interface.outside_ior &&
                  medium_ior == interface.outside_ior) {
                incident_side = 0;
                opposite_ior = medium_ior;
              }
              else if (fabsf(medium_ior - interface.outside_ior) <= 1e-5f) {
                incident_side = 1;
                opposite_ior = interface.inside_ior;
              }
              else if (fabsf(medium_ior - interface.inside_ior) <= 1e-5f) {
                incident_side = -1;
                opposite_ior = interface.outside_ior;
              }
              else {
                continue;
              }
            }
            for (int event = COHERENT_GEOMETRY_REFLECT;
                 event <= (is_glass ? COHERENT_GEOMETRY_TRANSMIT : COHERENT_GEOMETRY_REFLECT);
                 ++event)
            {
              const bool transmit = event == COHERENT_GEOMETRY_TRANSMIT;
              if (transmit) {
                if (!scene->integrator->get_caustics_refractive() ||
                    transmissions >= scene->integrator->get_max_transmission_bounce())
                {
                  continue;
                }
              }
              else if (!scene->integrator->get_caustics_reflective() ||
                       reflections >= scene->integrator->get_max_glossy_bounce())
              {
                continue;
              }
              /* Curved ownership is a finite declared model: exactly one exterior
               * sphere reflection or one contiguous T-R^m-T block (m<=2),
               * surrounded by planar reflections in air. Unsupported histories
               * retain their native writer, never enter coherent ownership. */
              int curved_first = -1, curved_count = 0;
              for (int k = 0; k < depth; k++) {
                if (patches[candidate.patch[k]].shape == 1) {
                  if (curved_first < 0) curved_first = k;
                  curved_count++;
                }
              }
              if (interface.shape == 1) {
                if (curved_count > 0 &&
                    !(curved_first + curved_count == depth &&
                      candidate.patch[depth - 1] == int(patch) &&
                      interface.mode == Object::COHERENT_INTERFACE_GLASS &&
                      medium_ior == interface.inside_ior &&
                      candidate.event[curved_first] == COHERENT_GEOMETRY_TRANSMIT &&
                      (transmit || curved_count < 3))) continue;
                if (curved_count == 0) {
                  bool prefix_mirrors_in_air = medium_ior == 1.0f;
                  for (int k = 0; k < depth; k++)
                    prefix_mirrors_in_air &= candidate.event[k] == COHERENT_GEOMETRY_REFLECT &&
                                             candidate.ior_before[k] == 1.0f;
                  if (!prefix_mirrors_in_air) continue;
                }
              }
              else if (curved_count > 0 && (transmit || medium_ior != 1.0f)) continue;
              if (++searched_states > search_limit) {
                progress.set_error("Coherent specular connection search exceeds 256 states");
                return false;
              }
              const float next_medium = transmit ? opposite_ior : medium_ior;
              KernelCoherentCandidate next = candidate;
              next.patch[depth] = int(patch);
              next.event[depth] = event;
              next.expected_incident_side[depth] = incident_side;
              next.ior_before[depth] = medium_ior;
              next.ior_after[depth] = next_medium;
              next.ior_opposite[depth] = opposite_ior;
              next.count = depth + 1;
              /* The declared source and Lambertian detector are in exterior air. A prefix ending
               * inside a glass medium cannot reach that detector without an exit interface. */
              if (fabsf(next_medium - 1.0f) <= 1e-5f) {
                bool sphere_transmission = false;
                for (int k = 0; k <= depth; k++)
                  sphere_transmission |= patches[next.patch[k]].shape == 1 &&
                                         next.event[k] == COHERENT_GEOMETRY_TRANSMIT;
                int sphere_events = 0;
                for (int k=0;k<=depth;k++) sphere_events += patches[next.patch[k]].shape == 1;
                /* Each of p interior chords has at most three isolated roots
                 * over its winding; reserve the proved 3*p bound, not a seed. */
                const int branches = sphere_transmission ? 3*(sphere_events-1) : 1;
                if (candidates.size() + branches > candidate_limit) {
                  progress.set_error("Coherent specular connection candidate count exceeds 256");
                  return false;
                }
                for (int branch = 0; branch < branches; branch++) {
                  next.sphere_branch = branch;
                  candidates.push_back(next);
                }
              }
              if (!enumerate(next,
                             depth + 1,
                             next_medium,
                             reflections + !transmit,
                             transmissions + transmit))
              {
                return false;
              }
            }
          }
          return true;
        };
    if (!enumerate(direct, 0, 1.0f, 0, 0)) {
      if (!progress.get_cancel() && !progress.get_error()) {
        progress.set_error("Coherent specular connection candidate count exceeds 256");
      }
      return false;
    }
  }
  if (candidates.empty()) {
    progress.set_error("Coherent specular connections require an active positive-length coherent point-light group");
    return false;
  }

  KernelCoherentPatch *device_patches = dscene->coherent_patches.alloc(patches.size());
  int *device_primitives = dscene->coherent_patch_primitives.alloc(patch_primitives.size());
  KernelCoherentCandidate *device_candidates = dscene->coherent_candidates.alloc(candidates.size());
  std::copy(patches.begin(), patches.end(), device_patches);
  std::copy(candidates.begin(), candidates.end(), device_candidates);
  std::copy(patch_primitives.begin(), patch_primitives.end(), device_primitives);
  dscene->coherent_patches.copy_to_device();
  dscene->coherent_patch_primitives.copy_to_device();
  dscene->coherent_candidates.copy_to_device();
  ki.coherent_patch_count = int(patches.size());
  ki.coherent_candidate_count = int(candidates.size());
  ki.coherent_max_interface_events = max_events;
  ki.coherent_specular_enabled = 1;
  return true;
}

static bool scene_has_true_displacement(const Scene *scene)
{
  for (const Geometry *geom : scene->geometry) {
    if (geom->has_true_displacement()) {
      return true;
    }
  }
  return false;
}

Scene ::Scene(const SceneParams &params_, Device *device)
    : name("Scene"),
      default_surface(nullptr),
      default_volume(nullptr),
      default_light(nullptr),
      default_background(nullptr),
      default_empty(nullptr),
      device(device),
      dscene(device),
      params(params_),
      update_stats(nullptr),
      kernels_loaded(false),
      /* TODO(sergey): Check if it's indeed optimal value for the split kernel.
       */
      max_closure_global(1)
{
  memset((void *)&dscene.data, 0, sizeof(dscene.data));

  osl_manager = make_unique<OSLManager>(device);
  shader_manager = ShaderManager::create(device->info.has_osl ? params.shadingsystem :
                                                                SHADINGSYSTEM_SVM);

  light_manager = make_unique<LightManager>();
  geometry_manager = make_unique<GeometryManager>();
  object_manager = make_unique<ObjectManager>();
  image_manager = make_unique<ImageManager>(device->info, params);
  particle_system_manager = make_unique<ParticleSystemManager>();
  bake_manager = make_unique<BakeManager>();
  procedural_manager = make_unique<ProceduralManager>();
  volume_manager = make_unique<VolumeManager>();

  /* Create nodes after managers, since create_node() can tag the managers. */
  camera = create_node<Camera>();
  dicing_camera = create_node<Camera>();
  lookup_tables = make_unique<LookupTables>();
  diffraction_manager = make_unique<DiffractionManager>();
  film = create_node<Film>();
  background = create_node<Background>();
  integrator = create_node<Integrator>();
  scene_attribute = create_node<SceneAttributes>();

  ccl::Film::add_default(this);
  ccl::ShaderManager::add_default(this);
}

Scene::~Scene()
{
  free_memory(true);
}

void Scene::free_memory(bool final)
{
  bvh.reset();

  /* The order of deletion is important to make sure data is freed based on
   * possible dependencies as the Nodes' reference counts are decremented in the
   * destructors:
   *
   * - Procedurals can create and hold pointers to any other types.
   * - Objects can hold pointers to Geometries and ParticleSystems
   * - Lights and Geometries can hold pointers to Shaders.
   *
   * Similarly, we first delete all nodes and their associated device data, and
   * then the managers and their associated device data.
   */
  procedurals.clear();
  objects.clear();
  geometry.clear();
  particle_systems.clear();
  passes.clear();

  if (device) {
    camera->device_free(device, &dscene, this);
    film->device_free(device, &dscene, this);
    background->device_free(device, &dscene);
    integrator->device_free(device, &dscene, true);
    scene_attribute->device_free(device, &dscene, true);
  }

  if (final) {
    cameras.clear();
    integrators.clear();
    scene_attributes.clear();
    films.clear();
    backgrounds.clear();

    camera = nullptr;
    dicing_camera = nullptr;
    integrator = nullptr;
    scene_attribute = nullptr;
    film = nullptr;
    background = nullptr;
  }

  /* Delete Shaders after every other nodes to ensure that we do not try to
   * decrement the reference count on some dangling pointer. */
  shaders.clear();

  /* Now that all nodes have been deleted, we can safely delete managers and
   * device data. */
  if (device) {
    object_manager->device_free(device, &dscene, true);
    geometry_manager->device_free(device, &dscene, true);
    shader_manager->device_free(device, &dscene, this);
    osl_manager->device_free(device, &dscene, this);
    light_manager->device_free(device, &dscene);
    dscene.coherent_patches.free();
    dscene.coherent_patch_primitives.free();
    dscene.coherent_candidates.free();
    dscene.data.integrator.coherent_specular_enabled = 0;
    dscene.data.integrator.coherent_patch_count = 0;
    dscene.data.integrator.coherent_candidate_count = 0;
    particle_system_manager->device_free(device, &dscene);
    bake_manager->device_free(device, &dscene);
    volume_manager->device_free(&dscene);

    if (final) {
      image_manager->device_free(this);
    }
    else {
      image_manager->device_free_builtin(this);
    }

    diffraction_manager->device_free(&dscene);
    lookup_tables->device_free(device, &dscene);
  }

  if (final) {
    diffraction_manager.reset();
    lookup_tables.reset();
    object_manager.reset();
    geometry_manager.reset();
    shader_manager.reset();
    osl_manager.reset();
    light_manager.reset();
    particle_system_manager.reset();
    image_manager.reset();
    bake_manager.reset();
    update_stats.reset();
    procedural_manager.reset();
    volume_manager.reset();
  }
}

void Scene::device_update(Device *device_, Progress &progress)
{
  if (!device) {
    device = device_;
  }

  const bool print_stats = need_data_update();
  bool kernels_reloaded = false;

  while (true) {
    if (update_stats) {
      update_stats->clear();
    }

    const scoped_callback_timer timer([this, print_stats](double time) {
      if (update_stats) {
        update_stats->scene.times.add_entry({"device_update", time});

        if (print_stats) {
          printf("Update statistics:\n%s\n", update_stats->full_report().c_str());
        }
      }
    });

    /* The order of updates is important, because there's dependencies between
     * the different managers, using data computed by previous managers. */

    if (film->update_lightgroups(this)) {
      light_manager->tag_update(this, ccl::LightManager::LIGHT_MODIFIED);
      object_manager->tag_update(this, ccl::ObjectManager::OBJECT_MODIFIED);
      background->tag_modified();
    }
    if (film->exposure_is_modified()) {
      integrator->tag_modified();
    }

    /* Compile shaders and get information about features they used. */
    progress.set_status("Updating Shaders");
    osl_manager->device_update_pre(device, this);
    shader_manager->device_update_pre(device, &dscene, this, progress);

    if (progress.get_cancel() || device->have_error()) {
      return;
    }

    /* Passes. After shader manager as this depends on the shaders. */
    film->update_passes(this);

    /* Update kernel features. After shaders and passes since those affect features. */
    update_kernel_features();
    if ((dscene.data.kernel_features & KERNEL_FEATURE_POLARIZATION) &&
        integrator->get_use_photon_mapping())
    {
      progress.set_error("Glass Polarizer requires Path Tracing or BDPT; Photon Mapping does not transport polarization");
      return;
    }

    device->set_scene_pixel_displacement(integrator->get_use_pixel_displacement() &&
                                             scene_has_true_displacement(this),
                                         integrator->get_pixel_displacement_scale(),
                                         integrator->get_pixel_displacement_max_distance(),
                                         scene_allows_pixel_displacement_metalrt(this));

    /* Load render kernels, before uploading most data to the GPU, and before displacement and
     * background light need to run kernels.
     *
     * Do it outside of the scene mutex since the heavy part of the loading (i.e. kernel
     * compilation) does not depend on the scene and some other functionality (like display
     * driver) might be waiting on the scene mutex to synchronize display pass.
     *
     * This does mean the scene might have gotten updated in the meantime, in which case
     * we have to redo the first part of the scene update. */
    const uint64_t kernel_features = dscene.data.kernel_features;
    scene_updated_while_loading_kernels = false;
    if (!kernels_loaded || loaded_kernel_features != kernel_features) {
      mutex.unlock();
      kernels_reloaded |= load_kernels(progress);
      mutex.lock();
    }

    if (progress.get_cancel() || device->have_error()) {
      return;
    }

    if (!scene_updated_while_loading_kernels) {
      break;
    }
  }

  /* Upload shaders to GPU and compile OSL kernels, after kernels have been loaded. */
  shader_manager->device_update_post(device, &dscene, this, progress);
  osl_manager->device_update_post(device, this, progress, kernels_reloaded);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  procedural_manager->update(this, progress);

  if (progress.get_cancel()) {
    return;
  }

  progress.set_status("Updating Background");
  background->device_update(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  progress.set_status("Updating Scene Attribute");
  scene_attribute->device_update(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Camera will be used by adaptive subdivision, so do early. */
  progress.set_status("Updating Camera");
  camera->device_update(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  geometry_manager->device_update_preprocess(device, this, progress);
  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Update objects after geometry preprocessing. */
  progress.set_status("Updating Objects");
  object_manager->device_update(device, &dscene, this, progress);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  progress.set_status("Updating Particle Systems");
  particle_system_manager->device_update(device, &dscene, this, progress);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Camera and shaders must be ready here for adaptive subdivision and displacement. */
  progress.set_status("Updating Meshes");
  geometry_manager->device_update(device, &dscene, this, progress);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Update object flags with final geometry. */
  progress.set_status("Updating Objects Flags");
  object_manager->device_update_flags(device, &dscene, this, progress);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Update BVH primitive objects with final geometry. */
  progress.set_status("Updating Primitive Offsets");
  object_manager->device_update_prim_offsets(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Images last, as they should be more likely to use host memory fallback than geometry.
   * Some images may have been uploaded early for displacement already at this point. */
  progress.set_status("Updating Images");
  image_manager->device_update(device, this, progress);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Tighten displacement bounds using statistics from images loaded above. */
  shader_manager->device_update_displacement_bounds(&dscene, this, progress);

  /* Evaluate volume shader to build volume octrees. */
  progress.set_status("Updating Volume");
  volume_manager->device_update(device, &dscene, this, progress);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  progress.set_status("Updating Camera Volume");
  camera->device_update_volume(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  progress.set_status("Updating Lookup Tables");
  diffraction_manager->device_update(&dscene);
  lookup_tables->device_update(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Light manager needs shaders and final meshes for triangles in light tree. */
  progress.set_status("Updating Lights");
  light_manager->device_update(device, &dscene, this, progress);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  if (!scene_prepare_coherent_specular(this, &dscene, progress)) {
    return;
  }

  progress.set_status("Updating Integrator");
  integrator->device_update(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  progress.set_status("Updating Film");
  film->device_update(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Update lookup tables a second time for film tables. */
  progress.set_status("Updating Lookup Tables");
  diffraction_manager->device_update(&dscene);
  lookup_tables->device_update(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  progress.set_status("Updating Baking");
  bake_manager->device_update(device, &dscene, this, progress);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  if (device->have_error() == false) {
    dscene.data.volume_stack_size = get_volume_stack_size();

    progress.set_status("Updating Device", "Writing constant memory");
    device->const_copy_to("data", &dscene.data, sizeof(dscene.data));
  }

  device->optimize_for_scene(this);

  if (need_motion() == MOTION_PASS_INTERACTIVE) {
    /* Swap current camera/object/vertex positions to previous positions for next frame. */
    camera->update_interactive_motion();
    object_manager->update_interactive_motion(this);
    geometry_manager->update_interactive_motion(this);
  }

  if (print_stats) {
    const size_t mem_used = util_guarded_get_mem_used();
    const size_t mem_peak = util_guarded_get_mem_peak();

    LOG_INFO << "System memory statistics after full device sync:\n"
             << "  Usage: " << string_human_readable_number(mem_used) << " ("
             << string_human_readable_size(mem_used) << ")\n"
             << "  Peak: " << string_human_readable_number(mem_peak) << " ("
             << string_human_readable_size(mem_peak) << ")";
  }
}

Scene::MotionType Scene::need_motion() const
{
  if (integrator->get_motion_blur()) {
    return MOTION_BLUR;
  }
  const DenoiserPassMask denoiser_motion_passes = DENOISER_PASS_MOTION |
                                                  DENOISER_PASS_BACKWARD_MOTION |
                                                  DENOISER_PASS_SPECULAR_MOTION;
  const bool denoiser_motion = (integrator->get_use_denoise()) &&
                               (integrator->get_denoiser_passes() & denoiser_motion_passes) != 0;
  if (denoiser_motion || (Pass::contains(passes, PASS_MOTION) ||
                          Pass::contains(passes, PASS_DENOISING_BACKWARD_MOTION) ||
                          Pass::contains(passes, PASS_DENOISING_SPECULAR_MOTION)))
  {
    return params.background ? MOTION_PASS : MOTION_PASS_INTERACTIVE;
  }
  return MOTION_NONE;
}

float Scene::motion_shutter_time()
{
  if (need_motion() == Scene::MOTION_PASS || need_motion() == Scene::MOTION_PASS_INTERACTIVE) {
    return 2.0f;
  }
  return camera->get_shuttertime();
}

bool Scene::need_global_attribute(AttributeStandard std) const
{
  if (std == ATTR_STD_UV) {
    return Pass::contains(passes, PASS_UV);
  }
  if (std == ATTR_STD_VOLUME_VELOCITY || std == ATTR_STD_VOLUME_VELOCITY_X ||
      std == ATTR_STD_VOLUME_VELOCITY_Y || std == ATTR_STD_VOLUME_VELOCITY_Z)
  {
    return need_motion() != MOTION_NONE;
  }

  return false;
}

void Scene::need_global_attributes(AttributeRequestSet &attributes)
{
  for (int std = ATTR_STD_NONE; std < ATTR_STD_NUM; std++) {
    if (need_global_attribute((AttributeStandard)std)) {
      attributes.add((AttributeStandard)std);
    }
  }

  for (const Shader *shader : shaders) {
    attributes.add(shader->global_attributes);
  }
}

bool Scene::need_update()
{
  return (need_reset() || film->is_modified());
}

bool Scene::need_data_update()
{
  return (background->is_modified() || image_manager->need_update() ||
          object_manager->need_update() || geometry_manager->need_update() ||
          light_manager->need_update() || lookup_tables->need_update() ||
          diffraction_manager->need_update() || integrator->is_modified() ||
          shader_manager->need_update() || particle_system_manager->need_update() ||
          bake_manager->need_update() || film->is_modified() ||
          procedural_manager->need_update() || scene_attribute->is_modified());
}

bool Scene::need_reset(const bool check_camera)
{
  return need_data_update() || (check_camera && camera->is_modified());
}

void Scene::reset()
{
  osl_manager->reset(this);
  ShaderManager::add_default(this);

  /* ensure all objects are updated */
  camera->tag_modified();
  dicing_camera->tag_modified();
  film->tag_modified();
  background->tag_modified();

  background->tag_update(this);
  integrator->tag_update(this, Integrator::UPDATE_ALL);
  scene_attribute->tag_update(this, SceneAttributes::UPDATE_ALL);
  object_manager->tag_update(this, ObjectManager::UPDATE_ALL);
  geometry_manager->tag_update(this, GeometryManager::UPDATE_ALL);
  light_manager->tag_update(this, LightManager::UPDATE_ALL);
  particle_system_manager->tag_update(this);
  procedural_manager->tag_update();
}

void Scene::device_free()
{
  free_memory(false);
}

void Scene::collect_statistics(RenderStats *stats)
{
  geometry_manager->collect_statistics(this, stats);
  image_manager->collect_statistics(stats, this);
}

void Scene::enable_update_stats()
{
  if (!update_stats) {
    update_stats = make_unique<SceneUpdateStats>();
  }
}

void Scene::update_kernel_features()
{
  if (!need_update()) {
    return;
  }

  /* These features are not being tweaked as often as shaders,
   * so could be done selective magic for the viewport as well. */
  uint64_t kernel_features = shader_manager->get_kernel_features(this);

  if (integrator->get_use_coherent_specular_connections()) {
    kernel_features |= KERNEL_FEATURE_COHERENT_SPECULAR;
  }

  const bool use_motion = need_motion() == Scene::MotionType::MOTION_BLUR;
  kernel_features |= KERNEL_FEATURE_PATH_TRACING;

  /* Track the max prim count in case the backend needs to rebuild BVHs or
   * kernels to support different limits. */
  size_t kernel_max_prim_count = 0;

  /* Figure out whether the scene will use shader ray-trace we need at least
   * one caustic light, one caustic caster and one caustic receiver to use
   * and enable the MNEE code path. */
  bool has_caustics_receiver = false;
  bool has_caustics_caster = false;
  bool has_caustics_light = false;

  for (Object *object : objects) {
    if (object->get_is_caustics_caster()) {
      has_caustics_caster = true;
    }
    else if (object->get_is_caustics_receiver()) {
      has_caustics_receiver = true;
    }
    Geometry *geom = object->get_geometry();
    if (use_motion) {
      if (object->use_motion() || geom->get_use_motion_blur()) {
        kernel_features |= KERNEL_FEATURE_OBJECT_MOTION;
      }
    }
    if (object->get_is_shadow_catcher() && !geom->is_light()) {
      kernel_features |= KERNEL_FEATURE_SHADOW_CATCHER;
    }
    if (geom->is_hair()) {
      const Hair *hair = static_cast<const Hair *>(geom);
      kernel_features |= (hair->curve_shape == CURVE_RIBBON) ? KERNEL_FEATURE_HAIR_RIBBON :
                                                               KERNEL_FEATURE_HAIR_THICK;
      kernel_max_prim_count = max(kernel_max_prim_count, hair->num_segments());
    }
    else if (geom->is_pointcloud()) {
      kernel_features |= KERNEL_FEATURE_POINTCLOUD;
      kernel_max_prim_count = max(kernel_max_prim_count,
                                  static_cast<PointCloud *>(geom)->num_points());
    }
    else if (geom->is_mesh()) {
      kernel_max_prim_count = max(kernel_max_prim_count,
                                  static_cast<Mesh *>(geom)->num_triangles());
    }
    else if (geom->is_light()) {
      const Light *light = static_cast<const Light *>(object->get_geometry());
      /* Conservatively include disabled lights too: enabled-light classification
       * is updated later. Never specialize away an active source group. */
      if (light->get_coherence_group() > 0 && light->get_coherence_length() > 0.0f) {
        kernel_features |= KERNEL_FEATURE_COHERENT_DIRECT;
      }
      if (light->get_use_caustics()) {
        has_caustics_light = true;
      }
    }
    if (object->has_light_linking()) {
      kernel_features |= KERNEL_FEATURE_LIGHT_LINKING;
    }
    if (object->has_shadow_linking()) {
      kernel_features |= KERNEL_FEATURE_SHADOW_LINKING;
    }
  }

  dscene.data.integrator.use_caustics = false;
  if (device->info.has_mnee() && has_caustics_caster && has_caustics_receiver &&
      has_caustics_light)
  {
    dscene.data.integrator.use_caustics = true;
    kernel_features |= KERNEL_FEATURE_MNEE;
  }

  if (integrator->get_guiding_params(device).use) {
    kernel_features |= KERNEL_FEATURE_PATH_GUIDING;
  }

  if (bake_manager->get_baking()) {
    kernel_features |= KERNEL_FEATURE_BAKING;
  }

  kernel_features |= film->get_kernel_features(this);
  kernel_features |= integrator->get_kernel_features();
  kernel_features |= camera->get_kernel_features();

  dscene.data.kernel_features = kernel_features;

  /* Currently viewport render is faster with higher max_closures, needs
   * investigating. */
  const uint max_closures = (params.background) ? get_max_closure_count() : MAX_CLOSURE;
  dscene.data.max_closures = max_closures;
  dscene.data.max_shaders = shaders.size();

  /* Inform the device of the BVH limits. If this returns true, all BVHs
   * and kernels need to be rebuilt. */
  if (device->set_bvh_limits(objects.size(), kernel_max_prim_count)) {
    kernels_loaded = false;
    for (Geometry *geom : geometry) {
      geom->need_update_rebuild = true;
      geom->tag_modified();
    }
  }
}

bool Scene::update(Progress &progress)
{
  if (!need_update()) {
    return false;
  }

  /* Upload scene data to the GPU. */
  progress.set_status("Updating Scene");
  MEM_GUARDED_CALL(&progress, device_update, device, progress);

  return true;
}

bool Scene::update_camera_resolution(Progress &progress, int width, int height)
{
  bool update_data = false;

  if (camera->set_screen_size(width, height)) {
    camera->device_update(device, &dscene, this);
    update_data = true;
  }

  if (integrator->get_use_pixel_jitter()) {
    integrator->tag_use_pixel_jitter_modified();

    integrator->device_update(device, &dscene, this);
    update_data = true;
  }

  if (update_data) {
    progress.set_status("Updating Device", "Writing constant memory");
    device->const_copy_to("data", &dscene.data, sizeof(dscene.data));
  }
  return update_data;
}

static void log_kernel_features(const uint64_t features)
{
  LOG_INFO << "Requested features:";
  LOG_INFO << "Use BSDF " << string_from_bool(features & KERNEL_FEATURE_NODE_BSDF);
  LOG_INFO << "Use Emission " << string_from_bool(features & KERNEL_FEATURE_NODE_EMISSION);
  LOG_INFO << "Use Volume " << string_from_bool(features & KERNEL_FEATURE_NODE_VOLUME);
  LOG_INFO << "Use Bump " << string_from_bool(features & KERNEL_FEATURE_NODE_BUMP);
  LOG_INFO << "Use Voronoi " << string_from_bool(features & KERNEL_FEATURE_NODE_VORONOI_EXTRA);
  LOG_INFO << "Use Shader Raytrace " << string_from_bool(features & KERNEL_FEATURE_NODE_RAYTRACE);
  LOG_INFO << "Use MNEE " << string_from_bool(features & KERNEL_FEATURE_MNEE);
  LOG_INFO << "Use Transparent " << string_from_bool(features & KERNEL_FEATURE_TRANSPARENT);
  LOG_INFO << "Use Denoising " << string_from_bool(features & KERNEL_FEATURE_DENOISING);
  LOG_INFO << "Use Path Tracing " << string_from_bool(features & KERNEL_FEATURE_PATH_TRACING);
  LOG_INFO << "Use Hair " << string_from_bool(features & KERNEL_FEATURE_HAIR);
  LOG_INFO << "Use Pointclouds " << string_from_bool(features & KERNEL_FEATURE_POINTCLOUD);
  LOG_INFO << "Use Object Motion " << string_from_bool(features & KERNEL_FEATURE_OBJECT_MOTION);
  LOG_INFO << "Use Baking " << string_from_bool(features & KERNEL_FEATURE_BAKING);
  LOG_INFO << "Use Subsurface " << string_from_bool(features & KERNEL_FEATURE_SUBSURFACE);
  LOG_INFO << "Use Volume " << string_from_bool(features & KERNEL_FEATURE_VOLUME);
  LOG_INFO << "Use Shadow Catcher " << string_from_bool(features & KERNEL_FEATURE_SHADOW_CATCHER);
  LOG_INFO << "Use Portal Node " << string_from_bool(features & KERNEL_FEATURE_NODE_PORTAL);
  LOG_INFO << "Use Light Linking " << string_from_bool(features & KERNEL_FEATURE_LIGHT_LINKING);
  LOG_INFO << "Use Shadow Linking " << string_from_bool(features & KERNEL_FEATURE_SHADOW_LINKING);
}

bool Scene::load_kernels(Progress &progress)
{
  progress.set_status("Loading render kernels (may take a few minutes the first time)");

  const scoped_timer timer;

  const uint64_t kernel_features = dscene.data.kernel_features;
  log_kernel_features(kernel_features);
  if (!device->load_kernels(kernel_features)) {
    string message = device->error_message();
    if (message.empty()) {
      message = "Failed loading render kernel, see console for errors";
    }

    progress.set_error(message);
    progress.set_status(message);
    progress.set_update();
    return false;
  }

  kernels_loaded = true;
  loaded_kernel_features = kernel_features;
  return true;
}

int Scene::get_max_closure_count()
{
  if (shader_manager->use_osl()) {
    /* OSL always needs the maximum as we can't predict the
     * number of closures a shader might generate. */
    return MAX_CLOSURE;
  }

  int max_closures = 0;
  for (int i = 0; i < shaders.size(); i++) {
    Shader *shader = shaders[i];
    if (shader->reference_count()) {
      const int num_closures = shader->graph->get_num_closures();
      max_closures = max(max_closures, num_closures);
    }
  }
  max_closure_global = max(max_closure_global, max_closures);

  if (max_closure_global > MAX_CLOSURE) {
    /* This is usually harmless as more complex shader tend to get many
     * closures discarded due to mixing or low weights. We need to limit
     * to MAX_CLOSURE as this is hardcoded in CPU/mega kernels, and it
     * avoids excessive memory usage for split kernels. */
    LOG_WARNING << "Maximum number of closures exceeded: " << max_closure_global << " > "
                << MAX_CLOSURE;

    max_closure_global = MAX_CLOSURE;
  }

  return max_closure_global;
}

int Scene::get_volume_stack_size() const
{
  int volume_stack_size = 0;

  /* Space for background volume and terminator.
   * Don't do optional here because camera ray initialization expects that there
   * is space for at least those elements (avoiding extra condition to check if
   * there is actual volume or not).
   */
  volume_stack_size += 2;

  /* Quick non-expensive check. Can over-estimate maximum possible nested level,
   * but does not require expensive calculation during pre-processing. */
  bool has_volume_object = false;
  for (const Object *object : objects) {
    if (!object->get_geometry()->has_volume) {
      continue;
    }

    if (object->intersects_volume) {
      /* Object intersects another volume, assume it's possible to go deeper in
       * the stack. */
      /* TODO(sergey): This might count nesting twice (A intersects B and B
       * intersects A), but can't think of a computationally cheap algorithm.
       * Dividing my 2 doesn't work because of Venn diagram example with 3
       * circles. */
      ++volume_stack_size;
    }
    else if (!has_volume_object) {
      /* Allocate space for at least one volume object. */
      ++volume_stack_size;
    }

    has_volume_object = true;

    if (volume_stack_size == MAX_VOLUME_STACK_SIZE) {
      break;
    }
  }

  volume_stack_size = min(volume_stack_size, MAX_VOLUME_STACK_SIZE);

  LOG_DEBUG << "Detected required volume stack size " << volume_stack_size;

  return volume_stack_size;
}

bool Scene::has_shadow_catcher()
{
  if (shadow_catcher_modified_) {
    has_shadow_catcher_ = false;
    for (Object *object : objects) {
      /* Shadow catcher flags on lights only controls effect on other objects, it's
       * not catching shadows itself. This is on by default, so ignore to avoid
       * performance impact when there is no actual shadow catcher. */
      if (object->get_is_shadow_catcher() && !object->get_geometry()->is_light()) {
        has_shadow_catcher_ = true;
        break;
      }
    }

    shadow_catcher_modified_ = false;
  }

  return has_shadow_catcher_;
}

void Scene::tag_shadow_catcher_modified()
{
  shadow_catcher_modified_ = true;
}

bool Scene::has_volume()
{
  has_volume_modified_ = false;
  return dscene.data.integrator.use_volumes;
}

bool Scene::has_volume_modified() const
{
  return has_volume_modified_;
}

void Scene::tag_has_volume_modified()
{
  has_volume_modified_ = true;
}

bool Scene::use_light_mis() const
{
  for (const Object *object : objects) {
    if (!object->get_geometry()->is_light()) {
      continue;
    }

    const Light *light = static_cast<const Light *>(object->get_geometry());
    if (light->get_is_enabled() && light->get_use_mis() && light->is_traceable()) {
      return true;
    }
  }

  return false;
}

template<class T> T *Scene::create_light_node()
{
  unique_ptr<T> node = make_unique<T>();
  T *node_ptr = node.get();
  node->set_owner(this);
  geometry.push_back(std::move(node));
  light_manager->tag_update(this, LightManager::LIGHT_ADDED);
  return node_ptr;
}

template<> PointLight *Scene::create_node<PointLight>()
{
  return create_light_node<PointLight>();
}

template<> SpotLight *Scene::create_node<SpotLight>()
{
  return create_light_node<SpotLight>();
}

template<> AreaLight *Scene::create_node<AreaLight>()
{
  return create_light_node<AreaLight>();
}

template<> SunLight *Scene::create_node<SunLight>()
{
  return create_light_node<SunLight>();
}

template<> BackgroundLight *Scene::create_node<BackgroundLight>()
{
  return create_light_node<BackgroundLight>();
}

template<> Mesh *Scene::create_node<Mesh>()
{
  unique_ptr<Mesh> node = make_unique<Mesh>();
  Mesh *node_ptr = node.get();
  node->set_owner(this);
  geometry.push_back(std::move(node));
  geometry_manager->tag_update(this, GeometryManager::MESH_ADDED);
  return node_ptr;
}

template<> Hair *Scene::create_node<Hair>()
{
  unique_ptr<Hair> node = make_unique<Hair>();
  Hair *node_ptr = node.get();
  node->set_owner(this);
  geometry.push_back(std::move(node));
  geometry_manager->tag_update(this, GeometryManager::HAIR_ADDED);
  return node_ptr;
}

template<> Volume *Scene::create_node<Volume>()
{
  unique_ptr<Volume> node = make_unique<Volume>();
  Volume *node_ptr = node.get();
  node->set_owner(this);
  geometry.push_back(std::move(node));
  geometry_manager->tag_update(this, GeometryManager::MESH_ADDED);
  return node_ptr;
}

template<> PointCloud *Scene::create_node<PointCloud>()
{
  unique_ptr<PointCloud> node = make_unique<PointCloud>();
  PointCloud *node_ptr = node.get();
  node->set_owner(this);
  geometry.push_back(std::move(node));
  geometry_manager->tag_update(this, GeometryManager::POINT_ADDED);
  return node_ptr;
}

template<> Object *Scene::create_node<Object>()
{
  unique_ptr<Object> node = make_unique<Object>();
  Object *node_ptr = node.get();
  node->set_owner(this);
  objects.push_back(std::move(node));
  object_manager->tag_update(this, ObjectManager::OBJECT_ADDED);
  return node_ptr;
}

template<> ParticleSystem *Scene::create_node<ParticleSystem>()
{
  unique_ptr<ParticleSystem> node = make_unique<ParticleSystem>();
  ParticleSystem *node_ptr = node.get();
  node->set_owner(this);
  particle_systems.push_back(std::move(node));
  particle_system_manager->tag_update(this);
  return node_ptr;
}

template<> Shader *Scene::create_node<Shader>()
{
  unique_ptr<Shader> node = make_unique<Shader>();
  Shader *node_ptr = node.get();
  node->set_owner(this);
  shaders.push_back(std::move(node));
  shader_manager->tag_update(this, ShaderManager::SHADER_ADDED);
  return node_ptr;
}

template<> Pass *Scene::create_node<Pass>()
{
  unique_ptr<Pass> node = make_unique<Pass>();
  Pass *node_ptr = node.get();
  node->set_owner(this);
  passes.push_back(std::move(node));
  film->tag_modified();
  return node_ptr;
}

template<> Camera *Scene::create_node<Camera>()
{
  unique_ptr<Camera> node = make_unique<Camera>();
  Camera *node_ptr = node.get();
  node->set_owner(this);
  cameras.push_back(std::move(node));
  return node_ptr;
}

template<> Integrator *Scene::create_node<Integrator>()
{
  unique_ptr<Integrator> node = make_unique<Integrator>();
  Integrator *node_ptr = node.get();
  node->set_owner(this);
  integrators.push_back(std::move(node));
  return node_ptr;
}

template<> SceneAttributes *Scene::create_node<SceneAttributes>()
{
  unique_ptr<SceneAttributes> node = make_unique<SceneAttributes>();
  SceneAttributes *node_ptr = node.get();
  node->set_owner(this);
  scene_attributes.push_back(std::move(node));
  return node_ptr;
}

template<> Background *Scene::create_node<Background>()
{
  unique_ptr<Background> node = make_unique<Background>();
  Background *node_ptr = node.get();
  node->set_owner(this);
  backgrounds.push_back(std::move(node));
  return node_ptr;
}

template<> Film *Scene::create_node<Film>()
{
  unique_ptr<Film> node = make_unique<Film>();
  Film *node_ptr = node.get();
  node->set_owner(this);
  films.push_back(std::move(node));
  return node_ptr;
}

template<> void Scene::delete_node(Light *node)
{
  assert(node->get_owner() == this);
  geometry.erase_by_swap(node);
  light_manager->tag_update(this, LightManager::LIGHT_REMOVED);
}

template<> void Scene::delete_node(Mesh *node)
{
  assert(node->get_owner() == this);
  geometry.erase_by_swap(node);
  geometry_manager->tag_update(this, GeometryManager::MESH_REMOVED);
}

template<> void Scene::delete_node(Hair *node)
{
  assert(node->get_owner() == this);
  geometry.erase_by_swap(node);
  geometry_manager->tag_update(this, GeometryManager::HAIR_REMOVED);
}

template<> void Scene::delete_node(Volume *node)
{
  assert(node->get_owner() == this);
  geometry.erase_by_swap(node);
  geometry_manager->tag_update(this, GeometryManager::MESH_REMOVED);
}

template<> void Scene::delete_node(PointCloud *node)
{
  assert(node->get_owner() == this);
  geometry.erase_by_swap(node);
  geometry_manager->tag_update(this, GeometryManager::POINT_REMOVED);
}

template<> void Scene::delete_node(Geometry *node)
{
  assert(node->get_owner() == this);

  uint flag;
  if (node->is_hair()) {
    flag = GeometryManager::HAIR_REMOVED;
  }
  else {
    flag = GeometryManager::MESH_REMOVED;
    if (node->has_volume) {
      volume_manager->tag_update({node});
    }
  }

  geometry.erase_by_swap(node);
  geometry_manager->tag_update(this, flag);
}

template<> void Scene::delete_node(Object *node)
{
  assert(node->get_owner() == this);

  uint flag = ObjectManager::OBJECT_REMOVED;
  if (node->get_geometry()->has_volume) {
    volume_manager->tag_update({node}, flag);
  }

  objects.erase_by_swap(node);
  object_manager->tag_update(this, flag);
}

template<> void Scene::delete_node(ParticleSystem *node)
{
  assert(node->get_owner() == this);
  particle_systems.erase_by_swap(node);
  particle_system_manager->tag_update(this);
}

template<> void Scene::delete_node(Shader *node)
{
  assert(node->get_owner() == this);
  /* don't delete unused shaders, not supported */
  node->clear_reference_count();
}

template<> void Scene::delete_node(Procedural *node)
{
  assert(node->get_owner() == this);
  procedurals.erase_by_swap(node);
  procedural_manager->tag_update();
}

template<> void Scene::delete_node(Pass *node)
{
  assert(node->get_owner() == this);
  passes.erase_by_swap(node);
  film->tag_modified();
}

template<typename T> static void assert_same_owner(const set<T *> &nodes, const NodeOwner *owner)
{
#ifdef NDEBUG
  (void)nodes;
  (void)owner;
#else
  for (const T *node : nodes) {
    assert(node->get_owner() == owner);
  }
#endif
}

template<> void Scene::delete_nodes(const set<Geometry *> &nodes, const NodeOwner *owner)
{
  assert_same_owner(nodes, owner);
  volume_manager->tag_update(nodes);
  geometry.erase_in_set(nodes);
  geometry_manager->tag_update(this, GeometryManager::GEOMETRY_REMOVED);
  light_manager->tag_update(this, LightManager::LIGHT_REMOVED);
}

template<> void Scene::delete_nodes(const set<Object *> &nodes, const NodeOwner *owner)
{
  assert_same_owner(nodes, owner);
  volume_manager->tag_update(nodes, ObjectManager::OBJECT_REMOVED);
  objects.erase_in_set(nodes);
  object_manager->tag_update(this, ObjectManager::OBJECT_REMOVED);
}

template<> void Scene::delete_nodes(const set<ParticleSystem *> &nodes, const NodeOwner *owner)
{
  assert_same_owner(nodes, owner);
  particle_systems.erase_in_set(nodes);
  particle_system_manager->tag_update(this);
}

template<> void Scene::delete_nodes(const set<Shader *> &nodes, const NodeOwner * /*owner*/)
{
  /* don't delete unused shaders, not supported */
  for (Shader *shader : nodes) {
    shader->clear_reference_count();
  }
}

template<> void Scene::delete_nodes(const set<Procedural *> &nodes, const NodeOwner *owner)
{
  assert_same_owner(nodes, owner);
  procedurals.erase_in_set(nodes);
  procedural_manager->tag_update();
}

template<> void Scene::delete_nodes(const set<Pass *> &nodes, const NodeOwner *owner)
{
  assert_same_owner(nodes, owner);
  passes.erase_in_set(nodes);
  film->tag_modified();
}

/* Template instantiations so we don't have to inline functions. */
template PointLight *Scene::create_light_node<PointLight>();
template SpotLight *Scene::create_light_node<SpotLight>();
template AreaLight *Scene::create_light_node<AreaLight>();
template SunLight *Scene::create_light_node<SunLight>();
template BackgroundLight *Scene::create_light_node<BackgroundLight>();

CCL_NAMESPACE_END
