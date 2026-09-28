/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <bit>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include "scene/camera.h"
#include "scene/integrator.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/pass.h"
#include "session/session.h"
#include "session/output_driver.h"
#include "session/buffers.h"

/* Actual renderer smoke fixture, separate from numerical kernel tests. */
static int diffraction_render_test(const ccl::DeviceInfo &device, const bool relief,
                                   const int seed, const int samples, const std::string &output,
                                   const std::string &transport, const bool mixed,
                                   const bool transmission, const std::string &illumination,
                                   const int quadrature_count, const std::string &angular_region)
{
  using namespace ccl;
  struct Result { std::vector<float> pixels; bool received = false; } result;
  class Driver : public OutputDriver {
   public:
    Result &result;
    explicit Driver(Result &r) : result(r) {}
    void write_render_tile(const Tile &tile) override
    {
      if (tile.size != make_int2(16, 16)) return;
      result.pixels.resize(16 * 16 * 4);
      result.received = tile.get_pass_pixels("combined", 4, result.pixels.data());
    }
  };
  SessionParams params;
  params.device = device;
  params.background = true;
  params.samples = samples;
  params.threads = 6;
  Session session(params, SceneParams{});
  session.set_output_driver(make_unique<Driver>(result));
  Scene *scene = session.scene.get();
  auto graph = make_unique<ShaderGraph>();
  auto *material = graph->create_node<DiffractionSmoothBsdfNode>();
  /* LINK_TANGENT sockets acquire geometry defaults during graph finalization.
   * An explicit connection is required to fix the grating's world-space axis. */
  auto *tangent = graph->create_node<CombineXYZNode>();
  tangent->set_x(1);
  graph->connect(tangent->output("Vector"), material->input("Tangent"));
  material->profile = {200, 0, 0.5, 1, {0.9, 6}, 1, {0.9, 6}};
  material->cache_options.bounds = {{-0.001f, -0.001f, 380}, {0.001f, 0.001f, 780}};
  material->cache_options.half_orders = 2;
  material->cache_options.retained_half_orders = 1;
  if (relief) {
    material->profile.pitch = 740;
    material->profile.depth = 150;
    material->profile.duty = 0.41;
    material->cache_options.half_orders = 16;
    material->cache_options.retained_half_orders = 3;
    material->cache_options.tolerance = 1e-4;
  }
  if (transmission) {
    /* Lossless relief in identical exterior media: uniform incident radiance
     * is preserved by reflection plus transmission, independently of RCWA. */
    material->profile.ridge_ior = {1.5, 0};
    material->profile.substrate_ior = {1, 0};
  }
  const auto profile = material->profile;
  if (mixed) {
    auto *diffuse = graph->create_node<DiffuseBsdfNode>();
    diffuse->set_color(make_float3(0.25f));
    diffuse->set_roughness(0);
    auto *mix = graph->create_node<MixClosureNode>();
    mix->set_fac(0.5f);
    graph->connect(material->output("BSDF"), mix->input("Closure1"));
    graph->connect(diffuse->output("BSDF"), mix->input("Closure2"));
    graph->connect(mix->output("Closure"), graph->output()->input("Surface"));
  }
  else {
    graph->connect(material->output("BSDF"), graph->output()->input("Surface"));
  }
  scene->default_surface->set_graph(std::move(graph));
  scene->default_surface->tag_update(scene);
  auto world = make_unique<ShaderGraph>();
  auto *background = world->create_node<BackgroundNode>();
  background->set_color(one_float3());
  background->set_strength(1);
  ShaderOutput *environment_mask = nullptr;
  if (illumination != "both") {
    /* Camera faces +Z and the surface faces -Z. Background Incoming is the
     * negative ray direction, so its +Z hemisphere supplies reflection. */
    auto *geometry = world->create_node<GeometryNode>();
    auto *dot = world->create_node<VectorMathNode>();
    dot->set_math_type(NODE_VECTOR_MATH_DOT_PRODUCT);
    dot->set_vector2(make_float3(0, 0, illumination == "reflection" ? 1 : -1));
    auto *hemisphere = world->create_node<MathNode>();
    hemisphere->set_math_type(NODE_MATH_GREATER_THAN);
    hemisphere->set_value2(0);
    world->connect(geometry->output("Incoming"), dot->input("Vector1"));
    world->connect(dot->output("Value"), hemisphere->input("Value1"));
    environment_mask = hemisphere->output("Value");
  }
  if (angular_region != "all") {
    auto *geometry = world->create_node<GeometryNode>();
    auto *dot = world->create_node<VectorMathNode>();
    dot->set_math_type(NODE_VECTOR_MATH_DOT_PRODUCT);
    dot->set_vector2(make_float3(1, 0, 0));
    auto *absolute = world->create_node<MathNode>();
    absolute->set_math_type(NODE_MATH_ABSOLUTE);
    world->connect(geometry->output("Incoming"), dot->input("Vector1"));
    world->connect(dot->output("Value"), absolute->input("Value1"));
    auto *lower = world->create_node<MathNode>();
    lower->set_math_type(NODE_MATH_GREATER_THAN);
    lower->set_value2(angular_region == "central" ? -1 :
                     (angular_region == "middle" ? 0.25f : 0.9f));
    auto *upper = world->create_node<MathNode>();
    upper->set_math_type(NODE_MATH_LESS_THAN);
    upper->set_value2(angular_region == "central" ? 0.25f :
                     (angular_region == "middle" ? 0.9f : 2));
    world->connect(absolute->output("Value"), lower->input("Value1"));
    world->connect(absolute->output("Value"), upper->input("Value1"));
    auto *region = world->create_node<MathNode>();
    region->set_math_type(NODE_MATH_MULTIPLY);
    world->connect(lower->output("Value"), region->input("Value1"));
    world->connect(upper->output("Value"), region->input("Value2"));
    if (environment_mask) {
      auto *combined = world->create_node<MathNode>();
      combined->set_math_type(NODE_MATH_MULTIPLY);
      world->connect(environment_mask, combined->input("Value1"));
      world->connect(region->output("Value"), combined->input("Value2"));
      environment_mask = combined->output("Value");
    }
    else environment_mask = region->output("Value");
  }
  if (environment_mask) world->connect(environment_mask, background->input("Strength"));
  world->connect(background->output("Background"), world->output()->input("Surface"));
  scene->default_background->set_graph(std::move(world));
  scene->default_background->tag_update(scene);
  Mesh *mesh = scene->create_node<Mesh>();
  mesh->resize_mesh(4, 2);
  const float3 positions[] = {make_float3(-10, -10, 2), make_float3(10, -10, 2),
                             make_float3(10, 10, 2), make_float3(-10, 10, 2)};
  std::copy_n(positions, 4, mesh->get_position_for_write());
  const int triangles[] = {0, 2, 1, 0, 3, 2};
  std::copy_n(triangles, 6, mesh->get_triangles().data());
  std::fill_n(mesh->get_shader().data(), 2, 0);
  std::fill_n(mesh->get_smooth().data(), 2, false);
  array<Node *> materials;
  materials.push_back_slow(scene->default_surface);
  mesh->set_used_shaders(materials);
  Object *object = scene->create_node<Object>();
  object->set_geometry(mesh);
  scene->camera->set_camera_type(CAMERA_ORTHOGRAPHIC);
  scene->camera->set_full_width(16);
  scene->camera->set_full_height(16);
  scene->camera->compute_auto_viewplane();
  scene->integrator->set_use_adaptive_sampling(false);
  scene->integrator->set_seed(seed);
  scene->integrator->set_use_bidirectional_path_tracing(transport == "bdpt" || transport == "bdpt_guided");
  scene->integrator->set_use_guiding(transport == "guided" || transport == "bdpt_guided");
  Pass *pass = scene->create_node<Pass>();
  pass->set_name(ustring("combined"));
  pass->set_type(PASS_COMBINED);
  BufferParams buffers;
  buffers.width = buffers.full_width = 16;
  buffers.height = buffers.full_height = 16;
  session.reset(params, buffers);
  session.start();
  session.wait();
  if (session.progress.get_error() || !result.received) {
    std::cerr << "Physical render failed: " << session.progress.get_error_message() << '\n';
    return 1;
  }
  const bool requested_bdpt = transport == "bdpt" || transport == "bdpt_guided";
  const bool requested_guiding = transport == "guided" || transport == "bdpt_guided";
  if (bool(scene->dscene.data.integrator.use_bidirectional_path_tracing) != requested_bdpt ||
      bool(scene->dscene.data.integrator.use_guiding) != requested_guiding) {
    std::cerr << "Requested transport mode was not enabled on this device\n";
    return 1;
  }
  float3 mean = zero_float3();
  for (int i = 0; i < 16 * 16; i++) {
    const float3 pixel = make_float3(result.pixels[4*i], result.pixels[4*i+1],
                                     result.pixels[4*i+2]);
    if (!isfinite_safe(pixel)) return 1;
    mean += pixel / 256;
  }
  /* Integrate the direct solver using the renderer's wavelength measure and
   * working-space transform, without cache interpolation. */
  KernelGlobalsCPU globals;
  globals.data = scene->dscene.data;
  globals.diffraction_descriptors.data = scene->dscene.diffraction_descriptors.data();
  globals.diffraction_domains.data = scene->dscene.diffraction_domains.data();
  globals.diffraction_nodes.data = scene->dscene.diffraction_nodes.data();
  globals.diffraction_layout.data = scene->dscene.diffraction_layout.data();
  globals.diffraction_bounds.data = scene->dscene.diffraction_bounds.data();
  globals.diffraction_ports.data = scene->dscene.diffraction_ports.data();
  globals.diffraction_active.data = scene->dscene.diffraction_active.data();
  globals.diffraction_matrices.data = scene->dscene.diffraction_matrices.data();
  Profiler profiler;
  ThreadKernelGlobalsCPU kg(globals, nullptr, profiler, 0);
  float3 expected = zero_float3();
  float3 cached_expected = zero_float3();
  /* Independent normal-incidence grating equation in equal air media. The
   * environment bins use |world direction.x|; the material tangent is +X. */
  const auto accepts_order = [&](const int order, const float wavelength_nm) {
    const double x = std::abs(order * double(wavelength_nm) / profile.pitch);
    if (angular_region == "central") return x < 0.25f;
    if (angular_region == "middle") return x > 0.25f && x < 0.9f;
    if (angular_region == "outer") return x > 0.9f;
    return true;
  };
  const DiffractionSceneData cache_data = diffraction_scene_data(&kg);
  /* This fixture creates exactly one grating material. Never silently query
   * another cache if its scene construction changes. */
  if (cache_data.cache_count != 1) {
    std::cerr << "Expected exactly one grating cache in render fixture\n";
    return 1;
  }
  for (int i = 0; i < quadrature_count; i++) {
    float probability;
    const float wavelength = sample_wavelength((i + 0.5f) / quadrature_count, &probability);
    double reflected = 0;
    if (transmission && illumination == "both" && angular_region == "all") {
      reflected = 1;
    }
    else if (relief) {
      DiffractionGratingResponse response;
      std::string error;
      if (!diffraction_grating_solve(profile, 1000.0f * wavelength, 0, 0, 16, response, error)) {
        std::cerr << "Spectral reference failed: " << error << '\n';
        return 1;
      }
      for (const auto &order : response.orders) {
        if (!accepts_order(order.order, 1000 * wavelength)) continue;
        if (illumination == "transmission" || (transmission && illumination == "both"))
          reflected += 0.5 * (order.substrate_flux[0] + order.substrate_flux[1]);
        if (illumination != "transmission")
          reflected += 0.5 * (order.reflection[0] + order.reflection[1]);
      }
    }
    else {
      reflected = std::norm((1.0 - profile.substrate_ior) / (1.0 + profile.substrate_ior));
    }
    if (mixed) reflected = 0.5 * reflected + 0.5 * 0.25;
    const float3 spectral_weight = wavelength_to_rgb_d65(&kg, wavelength) /
                                   (probability * quadrature_count);
    expected += spectral_weight * float(reflected);
    DiffractionCacheCellView view;
    int incoming_order;
    float powers[DIFFRACTION_MAX_CHANNELS / 2];
    const float lower_index = profile.substrate_ior.imag() > 0 ?
                                  profile.incident_ior : profile.substrate_ior.real();
    if (!diffraction_data_power_column(&cache_data, 0, make_float3(0, 0, -1), false,
                                      profile.incident_ior, lower_index, 1000 * wavelength,
                                      profile.pitch, &view, &incoming_order, powers)) {
      std::cerr << "Cached spectral reference failed\n";
      return 1;
    }
    double cached_power = 0;
    for (int port = 0; port < view.ports; port++) {
      const bool lower = cache_data.ports[view.port_offset + port].y;
      const int relative_order = cache_data.ports[view.port_offset + port].x - incoming_order;
      if (!accepts_order(relative_order, 1000 * wavelength)) continue;
      if (illumination == "both" || lower == (illumination == "transmission"))
        cached_power += powers[port];
    }
    if (mixed) cached_power = 0.5 * cached_power + 0.5 * 0.25;
    cached_expected += spectral_weight * float(cached_power);
  }
  std::cout << "Physical " << (transmission ? "lossless transmission furnace" :
                              (mixed ? "mixed relief" : (relief ? "relief" : "flat"))) << " renderer fixture on "
            << device.description << ": mean RGB=" << mean.x << "," << mean.y << "," << mean.z
            << " direct spectral reference=" << expected.x << "," << expected.y << ","
            << expected.z << '\n';
  std::cout << "Integrated cache RGB=" << cached_expected.x << "," << cached_expected.y
            << "," << cached_expected.z << '\n';
  if (!output.empty()) {
    const std::filesystem::path prefix(output);
    if (!prefix.parent_path().empty())
      std::filesystem::create_directories(prefix.parent_path());
    std::ofstream image(output + ".pfm", std::ios::binary);
    image << "PF\n16 16\n" << (std::endian::native == std::endian::little ? "-1.0\n" : "1.0\n");
    /* Cycles tile rows and PFM rows both start at the lower image edge. */
    for (int i = 0; i < 256; i++)
      image.write(reinterpret_cast<const char *>(&result.pixels[4*i]), 3 * sizeof(float));
    image.close();
    std::ofstream report(output + ".json");
    report << std::setprecision(9)
           << "{\n  \"device_type\": \"" << (device.type == DEVICE_METAL ? "METAL" : "CPU")
           << "\",\n  \"width\": 16, \"height\": 16, \"samples\": " << samples
           << ", \"seed\": " << seed
           << ", \"diffuse_mix_weight\": " << (mixed ? 0.5 : 0)
           << ", \"diffuse_albedo\": 0.25"
           << ",\n  \"pitch_nm\": " << profile.pitch << ", \"depth_nm\": " << profile.depth
           << ", \"duty\": " << profile.duty
           << ",\n  \"substrate_n\": " << profile.substrate_ior.real()
           << ", \"substrate_k\": " << profile.substrate_ior.imag()
           << ", \"ridge_n\": " << profile.ridge_ior.real()
           << ", \"ridge_k\": " << profile.ridge_ior.imag() << ",\n"
           << "  \"mean_rgb\": [" << mean.x << ", " << mean.y << ", " << mean.z << "],\n"
           << "  \"reference_rgb\": [" << expected.x << ", " << expected.y << ", " << expected.z << "],\n"
           << "  \"integrated_cache_rgb\": [" << cached_expected.x << ", "
           << cached_expected.y << ", " << cached_expected.z << "],\n"
           << "  \"illumination\": \"" << illumination << "\",\n"
           << "  \"angular_region\": \"" << angular_region << "\",\n"
           << "  \"angular_abs_x_boundaries\": [0.25, 0.9],\n"
           << "  \"reference_half_orders\": " << (relief && !(transmission && illumination == "both" && angular_region == "all") ? 16 : 0)
           << ", \"quadrature_count\": " << quadrature_count
           << ",\n  \"reference_kind\": \"" << (transmission && illumination == "both" && angular_region == "all" ? "analytic_lossless_furnace" :
                                                  (relief ? "same_order_direct_solver" : "analytic_fresnel"))
           << "\",\n  \"transport\": \"" << transport << "\", \"linear_image\": true, \"coherent_transport\": false\n}\n";
    report.close();
    if (!image || !report) {
      std::cerr << "Failed to save physical render artifacts\n";
      return 1;
    }
  }
  /* Smoke tolerance only; a multi-seed confidence test is still required. */
  return reduce_max(fabs(mean - expected)) < 0.03f ? 0 : 1;
}
