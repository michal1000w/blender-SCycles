/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "device/device.h"
#include "kernel/closure/bsdf_diffraction_smooth.h"
#include "kernel/integrator/surface_shader.h"
#include "kernel/svm/svm.h"
#include "kernel/osl/closures_setup.h"
#include "kernel/util/diffraction_cache_view.h"
#include "kernel/util/diffraction_scene.h"
#include "scene/devicescene.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/shader_nodes.h"
#include "scene/svm.h"
#include "util/progress.h"
#include "scene/diffraction_manager.h"
#include "util/math.h"
#include "util/profiling.h"
#include "util/stats.h"
#include <iostream>
using namespace ccl;
static bool test_microfacet_delta(Profiler &profiler)
{
  KernelGlobalsCPU globals{};
  KernelObject object{};
  globals.objects.data = &object;
  ThreadKernelGlobalsCPU kg(globals, nullptr, profiler, 0);
  int checked = 0;
  for (const ClosureType type : {CLOSURE_BSDF_MICROFACET_GGX_ID,
                                CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID,
                                CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID,
                                CLOSURE_BSDF_MICROFACET_BECKMANN_ID,
                                CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID,
                                CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID,
                                CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID}) {
    for (const float ior : {1.5f, 1.0f / 1.5f}) {
      for (const float cosine : {0.2f, 0.8f, 1.0f}) {
        for (const float alpha : {0.0f, 1e-5f}) for (const float random : {0.0f, 0.99999994f}) {
          ShaderData sd{};
          sd.N = sd.Ng = make_float3(0, 0, 1);
          sd.wi = make_float3(sqrtf(1 - cosine * cosine), 0, cosine);
          MicrofacetBsdf bsdf{};
          bsdf.N = sd.N;
          bsdf.type = type;
          bsdf.ior = ior;
          bsdf.alpha_x = bsdf.alpha_y = alpha;
          bsdf.fresnel_type = MicrofacetFresnel::DIELECTRIC;
          bsdf.energy_scale = 1;
          Spectrum sampled;
          float3 wo;
          float pdf, eta;
          float2 roughness;
          const int label = bsdf_sample(&kg, &sd, (ShaderClosure *)&bsdf,
              make_float3(.3f, .7f, random), &sampled, &wo, &pdf, &roughness, &eta);
          if (label == LABEL_NONE) continue;
          float evaluated_pdf;
          const Spectrum evaluated = bsdf_eval_delta(
              &kg, &sd, (ShaderClosure *)&bsdf, wo, &evaluated_pdf);
          if (!(label & LABEL_SINGULAR) || fabsf(pdf - evaluated_pdf) / 1e6f > 2e-6f ||
              reduce_max(fabs(sampled - evaluated)) / 1e6f > 2e-6f) return false;
          bsdf.alpha_x = bsdf.alpha_y = .1f;
          if (!is_zero(bsdf_eval_delta(&kg, &sd, (ShaderClosure *)&bsdf, wo, &evaluated_pdf)) ||
              evaluated_pdf != 0) return false;
          checked++;
        }
      }
    }
  }
  std::cout << "Passed " << checked << " smooth microfacet atom/sample comparisons\n";
  return checked > 0;
}
/* A synthetic passive two-port operator isolates routing from material-model
 * accuracy. It transfers half the incident flux to the other side/order. */
static bool test_order_routing(ccl::Device *device, Profiler &profiler,
                               const bool audit_delta_mixture = false)
{
  for (bool mirror : {false, true}) {
    for (int sx : {-1, 1}) {
      for (int sy : {-1, 1}) {
        DiffractionGratingCache cache;
        cache.bounds = {{-0.5, -1, 500}, {mirror ? 0.0 : 0.5, mirror ? 0.0 : 1.0, 600}};
        cache.mirror_symmetry = mirror;
        cache.nodes = {make_int4(-1, 0, 0, 0)};
        DiffractionGratingPackedCell cell;
        cell.bounds = cache.bounds;
        cell.ports = {{0, false}, {mirror ? -1 : sx, true}};
        cell.matrices.resize(8 * 16, zero_float2());
        for (int corner = 0; corner < 8; corner++) {
          for (int c = 0; c < 4; c++)
            cell.matrices[corner * 16 + ((c + 2) % 4) * 4 + c] = make_float2(std::sqrt(0.5f), 0);
        }
        cache.cells = {cell};
        DiffractionManager manager;
        DeviceScene scene(device);
        std::string error;
        const std::vector<DiffractionGratingCache> caches{cache};
        if (!manager.set_caches(caches, error))
          return false;
        manager.device_update(&scene);
        if (manager.need_update() || device->have_error())
          return false;
        KernelGlobalsCPU globals;
        KernelObject fixture_object{};
        globals.objects.data = &fixture_object;
        globals.data = scene.data;
        globals.diffraction_descriptors.data = scene.diffraction_descriptors.data();
        globals.diffraction_domains.data = scene.diffraction_domains.data();
        globals.diffraction_nodes.data = scene.diffraction_nodes.data();
        globals.diffraction_layout.data = scene.diffraction_layout.data();
        globals.diffraction_bounds.data = scene.diffraction_bounds.data();
        globals.diffraction_matrices.data = scene.diffraction_matrices.data();
        globals.diffraction_active.data = scene.diffraction_active.data();
        globals.diffraction_ports.data = scene.diffraction_ports.data();
        ThreadKernelGlobalsCPU kg(globals, nullptr, profiler, 0);
        for (bool below : {false, true}) {
          const float ni = below ? 1.5f : 1.0f, no = below ? 1.0f : 1.5f;
          const float x = sx * (below ? 0.6f : 0.1f) / ni, y = sy * 0.2f / ni;
          const float3 incident = make_float3(
              x, y, (below ? 1 : -1) * std::sqrt(1 - x * x - y * y));
          const float ox = sx * (below ? 0.1f : 0.6f) / no, oy = sy * 0.2f / no;
          const float oz = (below ? 1 : -1) * std::sqrt(1 - ox * ox - oy * oy);
          for (float random : {0.0f, 0.5f, 0.99999994f}) {
            DiffractionSceneSample sample;
            if (!diffraction_scene_sample<4>(
                    &kg, 0, incident, below, 1, 1.5f, 550, 1100, random, &sample) ||
                !sample.transmission || sample.relative_order != (below ? -sx : sx) ||
                sample.probability != 1 || std::abs(sample.eta - no / ni) > 2e-6f ||
                std::abs(sample.throughput - 0.5f) > 2e-6f ||
                std::abs(sample.direction.x - ox) > 2e-6f ||
                std::abs(sample.direction.y - oy) > 2e-6f ||
                std::abs(sample.direction.z - oz) > 2e-6f)
            {
              std::cerr << "Order routing failed: mirror=" << mirror << " x=" << sx << " y=" << sy
                        << " below=" << below << " random=" << random << '\n';
              return false;
            }
            /* Exercise the actual closure frame conversion, not just the
             * local-coordinate sampler. These orthonormal frames include a
             * general rotation and reverse incidence through the substrate. */
            for (const float3 fixed_normal :
                 {make_float3(0, 0, 1), make_float3(1, 0, 0), normalize(make_float3(1, 2, 3))})
            {
              const float3 X = normalize(cross(make_float3(0, 1, 0), fixed_normal));
              const float3 Y = cross(fixed_normal, X);
              DiffractionSmoothBsdf closure{};
              closure.type = CLOSURE_BSDF_DIFFRACTION_SMOOTH_ID;
              closure.N = below ? -fixed_normal : fixed_normal;
              closure.T = X;
              closure.wavelength = 550;
              closure.pitch = 1100;
              closure.upper_index = 1;
              closure.lower_index = 1.5f;
              closure.cache_handle = 0;
              closure.incoming_substrate = below;
              const float3 wi = -(incident.x * X + incident.y * Y + incident.z * fixed_normal);
              const float3 expected = ox * X + oy * Y + oz * fixed_normal;
              Spectrum eval;
              float3 wo;
              float pdf, eta;
              float2 roughness;
              const int label = bsdf_diffraction_smooth_sample(&kg,
                                                               (const ShaderClosure *)&closure,
                                                               closure.N,
                                                               wi,
                                                               make_float3(0, 0, random),
                                                               &eval,
                                                               &wo,
                                                               &pdf,
                                                               &roughness,
                                                               &eta);
              if (label != (LABEL_TRANSMIT | LABEL_SINGULAR) || len(wo - expected) > 2e-6f ||
                  std::abs(pdf - 1e6f) > 1 || std::abs(average(eval) / pdf - 0.5f) > 2e-6f ||
                  std::abs(eta - no / ni) > 2e-6f || len_squared(roughness) != 0)
              {
                std::cerr << "Smooth closure frame routing failed\n";
                return false;
              }
              float delta_power, delta_probability;
              if (!bsdf_diffraction_smooth_delta_probability(&kg,
                      (const ShaderClosure *)&closure, wi, wo, &delta_power, &delta_probability) ||
                  std::abs(delta_power - 0.5f) > 2e-6f || delta_probability != 1)
                return false;
              if (audit_delta_mixture) {
                /* Two identical deterministic closures describe the same
                 * outgoing atom. Its marginal probability must remain one. */
                ShaderData sd{};
                sd.N = sd.Ng = closure.N;
                sd.wi = wi;
                sd.num_closure = 2;
                for (int i = 0; i < 2; i++) {
                  auto *copy = reinterpret_cast<DiffractionSmoothBsdf *>(&sd.closure[i]);
                  *copy = closure;
                  copy->weight = one_spectrum();
                  copy->sample_weight = 1;
                }
                BsdfEval mixture_eval;
                bsdf_eval_init(&mixture_eval, &sd.closure[0], wo, eval);
                float average_roughness = 0;
                pdf = _surface_shader_bsdf_eval_mis(
                    &kg, &sd, wo, &sd.closure[0], &mixture_eval, pdf, 1, 0, 0,
                    average_roughness);
                if (len(wo - expected) > 2e-6f || std::abs(pdf / 1e6f - 1) > 2e-6f ||
                    std::abs(average(bsdf_eval_sum(&mixture_eval)) / 1e6f - 1) > 2e-6f)
                {
                  std::cerr << "Coincident delta mixture marginal probability=" << pdf / 1e6f
                            << "; expected 1\n";
                  return false;
                }
                float continuous_pdf;
                if (!is_zero(bsdf_eval(&kg, &sd, &sd.closure[1], wo, &continuous_pdf)) ||
                    continuous_pdf != 0) return false;
                const int mixture_label = surface_shader_bsdf_sample_closure(
                    &kg, &sd, &sd.closure[0], make_float3(0, 0, random),
                    &mixture_eval, &wo, &pdf, &roughness, &eta, average_roughness);
                if ((mixture_label & (LABEL_TRANSMIT | LABEL_SINGULAR)) !=
                        (LABEL_TRANSMIT | LABEL_SINGULAR) || len(wo - expected) > 2e-6f ||
                    std::abs(pdf / 1e6f - 1) > 2e-6f ||
                    std::abs(average(bsdf_eval_sum(&mixture_eval)) / 1e6f - 1) > 2e-6f)
                  return false;
                const float delta_pdf = surface_shader_bsdf_eval_delta(&kg, &sd, wo, &mixture_eval);
                if (std::abs(delta_pdf / 1e6f - 1) > 2e-6f ||
                    std::abs(average(bsdf_eval_sum(&mixture_eval)) / 1e6f - 1) > 2e-6f)
                  return false;
                sd.num_closure = 3;
                sd.closure[2].type = CLOSURE_BSDF_DIFFUSE_ID;
                sd.closure[2].N = sd.N;
                sd.closure[2].weight = one_spectrum();
                sd.closure[2].sample_weight = 2;
                const float diluted_pdf = surface_shader_bsdf_eval_delta(
                    &kg, &sd, wo, &mixture_eval);
                if (std::abs(diluted_pdf / 1e6f - .5f) > 2e-6f ||
                    std::abs(average(bsdf_eval_sum(&mixture_eval)) / 1e6f - 1) > 2e-6f)
                  return false;
                surface_shader_bsdf_sample_closure(
                    &kg, &sd, &sd.closure[0], make_float3(0, 0, random),
                    &mixture_eval, &wo, &pdf, &roughness, &eta, average_roughness);
                if (std::abs(pdf - diluted_pdf) / 1e6f > 2e-6f ||
                    std::abs(average(bsdf_eval_sum(&mixture_eval)) / 1e6f - 1) > 2e-6f)
                  return false;
              }
              /* Force the transmitted ray onto the wrong geometric side.
               * Rejection must leave zero contribution and sampling density. */
              if (bsdf_diffraction_smooth_sample(&kg,
                                                 (const ShaderClosure *)&closure,
                                                 -closure.N,
                                                 wi,
                                                 make_float3(0, 0, random),
                                                 &eval,
                                                 &wo,
                                                 &pdf,
                                                 &roughness,
                                                 &eta) != LABEL_NONE ||
                  pdf != 0 || len_squared(eval) != 0)
                return false;
            }
            float direction_power, direction_probability;
            if (!diffraction_scene_direction_probability<4>(&kg, 0, incident, below,
                  1, 1.5f, 550, 1100, sample.direction, &direction_power, &direction_probability) ||
                std::abs(direction_power - 0.5f) > 2e-6f || direction_probability != 1)
              return false;
            if (!diffraction_scene_direction_probability<4>(&kg, 0, incident, below,
                  1, 1.5f, 550, 1100, normalize(sample.direction + make_float3(0, 0.05f, 0)),
                  &direction_power, &direction_probability) ||
                direction_power != 0 || direction_probability != 0)
              return false;
            if (mirror) {
              float reverse_power, reverse_probability;
              if (!diffraction_scene_reverse_probability<4>(&kg,
                                                            0,
                                                            &sample,
                                                            below,
                                                            1,
                                                            1.5f,
                                                            550,
                                                            1100,
                                                            &reverse_power,
                                                            &reverse_probability) ||
                  std::abs(reverse_power - 0.5f) > 2e-6f ||
                  std::abs(reverse_probability - 1) > 2e-6f)
                return false;
            }
          }
        }
        manager.device_free(&scene);
      }
    }
  }
  std::cout << "Passed 48 CPU order-routing samples: both sides, unequal indices, mirrors and CDF "
               "endpoints; 144 rotated smooth-closure samples and hemisphere rejections\n";
  return true;
}

static bool test_physical_fresnel(ccl::Device *device, Profiler &profiler)
{
  for (bool below : {false, true}) {
    for (double angle : {0.0, 0.4, 0.9}) {
      const double ni = below ? 1.5 : 1.0, no = below ? 1.0 : 1.5;
      const float3 incident = make_float3(std::sin(angle) * std::cos(0.3),
                                          std::sin(angle) * std::sin(0.3),
                                          (below ? 1 : -1) * std::cos(angle));
      DiffractionCacheCoordinates coordinates;
      if (!diffraction_cache_coordinates(incident, ni, 550, 200, false, &coordinates))
        return false;
      const auto q = coordinates.query;
      DiffractionGratingProfile profile{200, 0, 0.41, 1, 1.5, 1, 1.5};
      DiffractionGratingCacheOptions options;
      options.bounds = {{float(q.x - 0.0001f), float(q.y - 0.0001f), double(549.9f)},
                        {float(q.x + 0.0001f), float(q.y + 0.0001f), double(550.1f)}};
      options.half_orders = 4;
      options.retained_half_orders = 3;
      options.tolerance = 1e-5;
      options.maximum_nodes = 63;
      DiffractionGratingCache cache;
      DiffractionGratingCacheStats stats;
      std::string error;
      if (!diffraction_grating_build_cache(profile, options, cache, stats, error)) {
        std::cerr << "Fresnel cache: " << error << '\n';
        return false;
      }
      DiffractionManager manager;
      DeviceScene scene(device);
      const std::vector<DiffractionGratingCache> caches{cache};
      if (!manager.set_caches(caches, error))
        return false;
      manager.device_update(&scene);
      if (manager.need_update() || device->have_error())
        return false;
      KernelGlobalsCPU globals;
      globals.data = scene.data;
      globals.diffraction_descriptors.data = scene.diffraction_descriptors.data();
      globals.diffraction_domains.data = scene.diffraction_domains.data();
      globals.diffraction_nodes.data = scene.diffraction_nodes.data();
      globals.diffraction_layout.data = scene.diffraction_layout.data();
      globals.diffraction_bounds.data = scene.diffraction_bounds.data();
      globals.diffraction_matrices.data = scene.diffraction_matrices.data();
      globals.diffraction_active.data = scene.diffraction_active.data();
      globals.diffraction_ports.data = scene.diffraction_ports.data();
      ThreadKernelGlobalsCPU kg(globals, nullptr, profiler, 0);
      const double ci = std::cos(angle);
      const std::complex<double> z = std::sqrt(
          std::complex<double>(no * no - ni * ni * (1 - ci * ci)));
      const auto rs = (ni * ci - z) / (ni * ci + z);
      const auto rp = (no * no * ci - ni * z) / (no * no * ci + ni * z);
      const double reflection = 0.5 * (std::norm(rs) + std::norm(rp));
      int reflections = 0;
      for (int j = 0; j < 1024; j++) {
        DiffractionSceneSample sample;
        if (!diffraction_scene_sample<4>(
                &kg, 0, incident, below, 1, 1.5, 550, 200, (j + 0.5f) / 1024, &sample))
        {
          std::cerr << "Fresnel sample failed: below=" << below << " angle=" << angle << '\n';
          return false;
        }
        reflections += !sample.transmission;
        const double expected = sample.transmission ? 1 - reflection : reflection;
        const double out_index = sample.transmission ? no : ni;
        const double ox = ni * incident.x / out_index, oy = ni * incident.y / out_index;
        const bool output_below = below != sample.transmission;
        const double oz = (output_below ? -1 : 1) *
                          std::sqrt(std::max(0.0, 1 - ox * ox - oy * oy));
        if (std::abs(sample.eta - (sample.transmission ? no / ni : 1)) > 2e-6 ||
            sample.relative_order != 0 || std::abs(sample.probability - expected) > 2e-5 ||
            std::abs(sample.throughput - 1) > 2e-5 || std::abs(sample.direction.x - ox) > 2e-6 ||
            std::abs(sample.direction.y - oy) > 2e-6 || std::abs(sample.direction.z - oz) > 2e-6)
          return false;
      }
      if (std::abs(reflections / 1024.0 - reflection) > 1.0 / 1024 + 2e-5)
        return false;
      manager.device_free(&scene);
    }
  }
  std::cout << "Passed 6144 solver-cache Fresnel samples, including total internal reflection\n";
  return true;
}

static bool test_physical_reverse(ccl::Device *device, Profiler &profiler)
{
  /* Published Al n,k at 516.6 nm (Rakić 1995 fixture), used as a constant
   * single-wavelength reference here. */
  const double wavelength = 516.6;
  const std::complex<double> aluminum(0.87340, 6.2418);
  const DiffractionGratingProfile profile{740, 150, 0.41, 1, aluminum, 1, aluminum};
  const float3 ray = make_float3(0.1f, 0.2f, -std::sqrt(0.95f));
  DiffractionCacheCoordinates coordinate;
  if (!diffraction_cache_coordinates(ray, 1, wavelength, 740, false, &coordinate))
    return false;
  std::vector<DiffractionGratingCache> caches(2);
  std::string error;
  for (int side = 0; side < 2; side++) {
    const float x = (side ? -1 : 1) * coordinate.query.x, y = (side ? -1 : 1) * coordinate.query.y;
    DiffractionGratingCacheOptions config;
    config.bounds = {{float(x - 0.0001f), float(y - 0.0001f), double(516.5f)},
                     {float(x + 0.0001f), float(y + 0.0001f), double(516.7f)}};
    config.half_orders = 16;
    config.retained_half_orders = 3;
    config.tolerance = 1e-6;
    DiffractionGratingCacheStats stats;
    if (!diffraction_grating_build_cache(profile, config, caches[side], stats, error)) {
      std::cerr << error;
      return false;
    }
  }
  DiffractionManager manager;
  DeviceScene scene(device);
  if (!manager.set_caches(caches, error))
    return false;
  manager.device_update(&scene);
  if (manager.need_update() || device->have_error())
    return false;
  KernelGlobalsCPU globals;
  globals.data = scene.data;
  globals.diffraction_descriptors.data = scene.diffraction_descriptors.data();
  globals.diffraction_domains.data = scene.diffraction_domains.data();
  globals.diffraction_nodes.data = scene.diffraction_nodes.data();
  globals.diffraction_layout.data = scene.diffraction_layout.data();
  globals.diffraction_bounds.data = scene.diffraction_bounds.data();
  globals.diffraction_ports.data = scene.diffraction_ports.data();
  globals.diffraction_active.data = scene.diffraction_active.data();
  globals.diffraction_matrices.data = scene.diffraction_matrices.data();
  ThreadKernelGlobalsCPU kg(globals, nullptr, profiler, 0);
  int nonzero = 0, different_probability = 0;
  for (int j = 0; j < 1024; j++) {
    DiffractionSceneSample sample;
    float reverse_power, reverse_probability;
    if (!diffraction_scene_sample<20>(
            &kg, 0, ray, false, 1, 1, wavelength, 740, (j + 0.5f) / 1024, &sample) ||
        !diffraction_scene_reverse_probability<20>(
            &kg, 1, &sample, false, 1, 1, wavelength, 740, &reverse_power, &reverse_probability))
      return false;
    if (std::abs(reverse_power - sample.probability * sample.throughput) > 2e-5 ||
        reverse_probability < 0 || reverse_probability > 1)
      return false;
    float direction_power, direction_probability;
    if (!diffraction_scene_direction_probability(&kg, 1, -sample.direction, false,
          1, 1, wavelength, 740, -ray, &direction_power, &direction_probability) ||
        std::abs(direction_power - reverse_power) > 2e-6f ||
        std::abs(direction_probability - reverse_probability) > 2e-6f)
      return false;
    nonzero += sample.relative_order != 0;
    different_probability += std::abs(reverse_probability - sample.probability) > 1e-4;
  }
  manager.device_free(&scene);
  std::cout << "Physical reverse queries: nonzero=" << nonzero
            << " differing normalized probabilities=" << different_probability << '\n';
  return nonzero > 0 && different_probability > 0;
}

static bool test_physical_shader_graph(ccl::Device *device, Profiler &profiler,
                                       const std::complex<double> substrate,
                                       const bool relief = false,
                                       const bool linked_tangent = false,
                                       const float tangent_angle = 0)
{
  Scene scene(SceneParams{}, device);
  auto graph = make_unique<ShaderGraph>();
  auto *material = graph->create_node<DiffractionSmoothBsdfNode>();
  const float3 expected_tangent = linked_tangent ?
                                     make_float3(std::cos(tangent_angle), std::sin(tangent_angle), 0) :
                                     make_float3(1, 0, 0);
  if (linked_tangent) {
    auto *vector = graph->create_node<CombineXYZNode>();
    /* Deliberately non-unit and distinct from sd.dPdu: finalization and
     * constant folding must preserve this link, and setup must normalize it. */
    vector->set_x(3 * expected_tangent.x);
    vector->set_y(3 * expected_tangent.y);
    graph->connect(vector->output("Vector"), material->input("Tangent"));
  }
  material->profile.substrate_ior = substrate;
  material->cache_options.bounds = {{-0.001f, -0.001f, 549.9f},
                                     {0.001f, 0.001f, 550.1f}};
  material->cache_options.half_orders = 2;
  material->cache_options.retained_half_orders = 1;
  if (relief) {
    material->profile.pitch = 740;
    material->profile.depth = 150;
    material->profile.duty = 0.41;
    material->profile.ridge_ior = substrate;
    material->cache_options.half_orders = 16;
    material->cache_options.retained_half_orders = 3;
    material->cache_options.tolerance = 1e-6;
  }
  const auto material_profile = material->profile;

  graph->connect(material->output("BSDF"), graph->output()->input("Surface"));
  Shader *shader = scene.default_surface;
  shader->set_graph(std::move(graph));
  shader->tag_update(&scene);
  Progress progress;
  scene.shader_manager->device_update_pre(device, &scene.dscene, &scene, progress);
  if (progress.get_error() || progress.get_cancel() ||
      scene.diffraction_manager->cache_count() != 1)
  {
    std::cerr << "Physical shader preparation failed: " << progress.get_error_message() << '\n';
    return false;
  }
  scene.shader_manager->device_update_post(device, &scene.dscene, &scene, progress);
  if (progress.get_error() || device->have_error() || !shader->has_dispersion ||
      scene.dscene.svm_nodes.size() == 0 || scene.dscene.shaders.size() <= shader->id ||
      !(scene.dscene.shaders[shader->id].flags & SD_REQUIRES_WAVELENGTH))
    return false;
  scene.diffraction_manager->device_update(&scene.dscene);
  KernelGlobalsCPU globals;
  globals.data = scene.dscene.data;
  globals.svm_nodes.data = reinterpret_cast<const uint *>(scene.dscene.svm_nodes.data());
  globals.svm_nodes.width = scene.dscene.svm_nodes.size();
  globals.shaders.data = scene.dscene.shaders.data();
  globals.shaders.width = scene.dscene.shaders.size();
#define BIND_DIFFRACTION_ARRAY(name) \
  globals.name.data = scene.dscene.name.data(); \
  globals.name.width = scene.dscene.name.size();
  BIND_DIFFRACTION_ARRAY(diffraction_descriptors)
  BIND_DIFFRACTION_ARRAY(diffraction_domains)
  BIND_DIFFRACTION_ARRAY(diffraction_nodes)
  BIND_DIFFRACTION_ARRAY(diffraction_layout)
  BIND_DIFFRACTION_ARRAY(diffraction_bounds)
  BIND_DIFFRACTION_ARRAY(diffraction_matrices)
  BIND_DIFFRACTION_ARRAY(diffraction_active)
  BIND_DIFFRACTION_ARRAY(diffraction_ports)
#undef BIND_DIFFRACTION_ARRAY
  ThreadKernelGlobalsCPU kg(globals, nullptr, profiler, 0);
  ShaderData sd{};
  sd.shader = shader->id;
  sd.shader_flag = kg.shaders.fetch(sd.shader).flags;
  sd.num_closure_left = 1;
  sd.N = sd.Ng = sd.wi = make_float3(0, 0, 1);
  sd.dPdu = make_float3(1, 0, 0);
  sd.object = OBJECT_NONE;
  /* Invert the wavelength CDF numerically to stay inside the fixture domain. */
  float lo = 0, hi = 1;
  for (int i = 0; i < 24; i++) {
    const float mid = (lo + hi) * 0.5f;
    if (sample_wavelength(mid) < 0.55f) lo = mid;
    else hi = mid;
  }
  sd.rand_wavelength = (lo + hi) * 0.5f;
  svm_eval_nodes<KERNEL_FEATURE_NODE_MASK_SURFACE, SHADER_TYPE_SURFACE>(
      &kg, IntegratorState(nullptr), &sd, nullptr, PathRayVisibility(0), 0);
  if (sd.num_closure != 1 || sd.closure[0].type != CLOSURE_BSDF_DIFFRACTION_SMOOTH_ID) {
    std::cerr << "Compiled graph did not allocate physical closure\n";
    return false;
  }
  const auto *compiled_bsdf = reinterpret_cast<const DiffractionSmoothBsdf *>(&sd.closure[0]);
  if (len(compiled_bsdf->T - expected_tangent) > 2e-6f) {
    std::cerr << "Compiled grating graph did not preserve the expected tangent\n";
    return false;
  }
  if (relief) {
    const double wavelength = 1000.0f * sample_wavelength(sd.rand_wavelength);
    DiffractionGratingResponse reference;
    std::string error;
    if (!diffraction_grating_solve(material_profile, wavelength, 0, 0, 16, reference, error))
      return false;
    double power[3] = {};
    for (const auto &order : reference.orders) {
      const double p = 0.5 * (order.reflection[0] + order.reflection[1]);
      if (order.order >= -1 && order.order <= 1)
        power[order.order + 1] = p;
      else if (p > 1e-10)
        return false;
    }
    const double total = power[0] + power[1] + power[2];
    int counts[3] = {};
    for (int i = 0; i < 4096; i++) {
      Spectrum eval;
      float3 wo;
      float pdf, eta;
      float2 roughness;
      const int label = bsdf_diffraction_smooth_sample(&kg, &sd.closure[0], sd.Ng, sd.wi,
          make_float3(0, 0, (i + 0.5f) / 4096), &eval, &wo, &pdf, &roughness, &eta);
      const int order = int(std::round(dot(wo, expected_tangent) * material_profile.pitch / wavelength));
      if (label != (LABEL_REFLECT | LABEL_SINGULAR) || order < -1 || order > 1)
        return false;
      counts[order + 1]++;
      const double x = order * wavelength / material_profile.pitch;
      if (std::abs(wo.x - x * expected_tangent.x) > 2e-6 ||
          std::abs(wo.y - x * expected_tangent.y) > 2e-6 ||
          std::abs(wo.z - std::sqrt(1 - x*x)) > 2e-6 || eta != 1 ||
          std::abs(average(eval) / pdf - total) > 2e-5 ||
          std::abs(pdf / 1e6 - power[order + 1] / total) > 2e-5)
        return false;
    }
    for (int i = 0; i < 3; i++)
      if (std::abs(counts[i] - 4096 * power[i] / total) > 2)
        return false;
    if (counts[0] == 0 || counts[2] == 0)
      return false;
    std::cout << "Passed compiled finite-depth grating: order counts " << counts[0] << ", "
              << counts[1] << ", " << counts[2] << "; reflected power " << total
              << "; linked tangent=" << linked_tangent << "; angle=" << tangent_angle << '\n';
    return true;
  }
  const bool absorbing = substrate.imag() > 0;
  const double reflection_power = std::norm((1.0 - substrate) / (1.0 + substrate));
  const float expected_throughput = absorbing ? reflection_power : 1.0;
  const float reflection_probability = absorbing ? 1.0 : reflection_power;
  int reflected = 0;
  for (int i = 0; i < 1000; i++) {
    Spectrum eval;
    float3 wo;
    float pdf, eta;
    float2 roughness;
    const int label = bsdf_diffraction_smooth_sample(&kg, &sd.closure[0], sd.Ng, sd.wi,
        make_float3(0, 0, (i + 0.5f) / 1000), &eval, &wo, &pdf, &roughness, &eta);
    const bool reflection = label == (LABEL_REFLECT | LABEL_SINGULAR);
    reflected += reflection;
    const float expected_probability = reflection ? reflection_probability :
                                                   1 - reflection_probability;
    if ((!reflection && label != (LABEL_TRANSMIT | LABEL_SINGULAR)) ||
        std::abs(pdf / 1e6f - expected_probability) > 2e-5f ||
        std::abs(average(eval) / pdf - expected_throughput) > 2e-5f ||
        std::abs(wo.z - (reflection ? 1 : -1)) > 2e-6f ||
        std::abs(eta - (reflection ? 1 : substrate.real())) > 2e-6f)
      return false;
  }
  if (std::abs(reflected - 1000 * reflection_probability) > 1)
    return false;
  std::cout << "Passed shader-manager metadata upload, physical graph SVM execution and 1000 Fresnel closure samples\n";
  return true;
}

#include "cycles_diffraction_render_test.h"

static bool test_fast_index_graph(ccl::Device *device, Profiler &profiler)
{
  Scene scene(SceneParams{}, device);
  auto graph = make_unique<ShaderGraph>();
  auto *material = graph->create_node<DiffractionSmoothBsdfNode>();
  material->use_fast_model = true;
  material->profile = {740, 150, .41, 1, {1.5, 0}, 1, {1.5, 0}};
  material->profile.ridge_spectrum = {{380, {1.4, .01}}, {780, {1.8, .09}}};
  material->profile.groove_spectrum = {{380, {1, 0}}, {780, {1.2, 0}}};
  graph->connect(material->output("BSDF"), graph->output()->input("Surface"));
  Shader *shader = scene.default_surface;
  shader->set_graph(std::move(graph));
  for (int iteration = 0; iteration < 3; ++iteration) {
    if (iteration == 1) {
      for (auto &sample : material->profile.ridge_spectrum) sample.index += .1;
    }
    if (iteration == 2) {
      material->profile.ridge_spectrum.clear();
      material->profile.groove_spectrum.clear();
    }
    shader->tag_update(&scene);
    Progress progress;
    scene.shader_manager->device_update_pre(device, &scene.dscene, &scene, progress);
    scene.shader_manager->device_update_post(device, &scene.dscene, &scene, progress);
    if (progress.get_error() || device->have_error() || shader->graph->get_num_closures() != 2 ||
        !(scene.dscene.shaders[shader->id].flags & SD_HAS_ONLY_DELTA_SURFACE)) {
      std::cerr << "Fast table graph update failed: " << progress.get_error_message() << '\n';
      return false;
    }
    KernelGlobalsCPU globals{};
    globals.data = scene.dscene.data;
    globals.svm_nodes.data = reinterpret_cast<const uint *>(scene.dscene.svm_nodes.data());
    globals.svm_nodes.width = scene.dscene.svm_nodes.size();
    globals.shaders.data = scene.dscene.shaders.data();
    globals.shaders.width = scene.dscene.shaders.size();
    globals.lookup_table.data = scene.dscene.lookup_table.data();
    globals.lookup_table.width = scene.dscene.lookup_table.size();
    ThreadKernelGlobalsCPU kg(globals, nullptr, profiler, 0);
    for (bool backface : {false, true}) {
      ShaderData sd{};
      sd.shader = shader->id;
      sd.shader_flag = kg.shaders.fetch(sd.shader).flags;
      sd.num_closure_left = 2;
      sd.N = sd.Ng = sd.wi = make_float3(0, 0, 1);
      sd.dPdu = make_float3(1, 0, 0);
      sd.object = OBJECT_NONE;
      sd.runtime_flag = backface ? SR_BACKFACING : 0;
      sd.rand_wavelength = .5f;
      svm_eval_nodes<KERNEL_FEATURE_NODE_MASK_SURFACE, SHADER_TYPE_SURFACE>(
          &kg, IntegratorState(nullptr), &sd, nullptr, PathRayVisibility(0), 0);
      if (sd.num_closure != 1 || sd.num_closure_left != 0) return false;
      const auto *bsdf = reinterpret_cast<const DiffractionSmoothBsdf *>(&sd.closure[0]);
      if (!bsdf->fast) return false;
      const auto &p = *bsdf->fast;
      const double t = (p.wavelength - 380) / 400.0;
      const double ridge_n = iteration == 2 ? 1.5 : 1.4 + .4*t + (iteration == 1 ? .1 : 0);
      const double groove_n = iteration == 2 ? 1 : 1 + .2*t;
      const double k = iteration == 2 ? 0 : .01 + .08*t;
      const double phase = 2*M_PI*150*(ridge_n-groove_n)/p.wavelength;
      const double transmission = .96*(.59+.41*std::exp(-4*M_PI*k*150/p.wavelength));
      if (std::abs(p.transmission_phase-phase)>2e-6 ||
          std::abs(p.transmission_budget-transmission)>2e-6 ||
          std::abs(p.reflection_budget-.04)>2e-6 ||
          p.incident_ior != (backface ? 1.5f : 1.f) ||
          p.transmitted_ior != (backface ? 1.f : 1.5f)) {
        std::cerr << "Fast table setup mismatch in iteration " << iteration << '\n';
        return false;
      }
    }
  }
  std::cout << "Passed six compiled Fast ridge/groove table, backface and recompile checks\n";
  for (bool mixed : {false, true}) {
    auto replacement = make_unique<ShaderGraph>();
    auto *diffuse = replacement->create_node<DiffuseBsdfNode>();
    ShaderOutput *surface = diffuse->output("BSDF");
    if (mixed) {
      auto *grating = replacement->create_node<DiffractionSmoothBsdfNode>();
      grating->use_fast_model = true;
      grating->profile = {740, 150, .41, 1, {1.5, 0}, 1, {1.5, 0}};
      auto *mix = replacement->create_node<MixClosureNode>();
      mix->set_fac(.5f);
      replacement->connect(diffuse->output("BSDF"), mix->input("Closure1"));
      replacement->connect(grating->output("BSDF"), mix->input("Closure2"));
      surface = mix->output("Closure");
    }
    replacement->connect(surface, replacement->output()->input("Surface"));
    shader->set_graph(std::move(replacement));
    shader->tag_update(&scene);
    Progress progress;
    scene.shader_manager->device_update_pre(device, &scene.dscene, &scene, progress);
    scene.shader_manager->device_update_post(device, &scene.dscene, &scene, progress);
    if (progress.get_error() || device->have_error() ||
        (scene.dscene.shaders[shader->id].flags & SD_HAS_ONLY_DELTA_SURFACE)) {
      std::cerr << "Continuous or mixed surface incorrectly classified as delta-only\n";
      return false;
    }
  }
  std::cout << "Passed continuous/mixed receiver flag and graph-replacement checks\n";
  return true;
}

int main(int argc, char **argv)
{
  const bool mixed_render = argc > 1 && std::string(argv[1]) == "--render-mixed";
  const bool transmission_render = argc > 1 && std::string(argv[1]) == "--render-transmission";
  const bool relief_render = mixed_render || transmission_render ||
                             (argc > 1 && std::string(argv[1]) == "--render-relief");
  const bool render = relief_render || (argc > 1 && std::string(argv[1]) == "--render");
  bool metal = argc == 2 && std::string(argv[1]) == "--metal";
  int seed = 0, samples = 256, reference_samples = 1024;
  std::string output, transport = "pt", illumination = "both", angular_region = "all";
  if (render) {
    for (int i = 2; i < argc; i++) {
      const std::string arg = argv[i];
      if (arg == "--metal") { metal = true; continue; }
      if ((arg != "--seed" && arg != "--samples" && arg != "--output" &&
           arg != "--transport" && arg != "--illumination" &&
           arg != "--reference-samples" && arg != "--angular-region") || i + 1 >= argc) {
        std::cerr << "Invalid render argument: " << arg << '\n';
        return 2;
      }
      const std::string value = argv[++i];
      if (arg == "--output") { output = value; continue; }
      if (arg == "--angular-region") {
        if (!transmission_render || (value != "all" && value != "central" &&
                                     value != "middle" && value != "outer")) {
          std::cerr << "Angular region must be all, central, middle or outer for --render-transmission\n";
          return 2;
        }
        angular_region = value;
        continue;
      }
      if (arg == "--illumination") {
        if (!transmission_render ||
            (value != "both" && value != "reflection" && value != "transmission")) {
          std::cerr << "Illumination must be both, reflection or transmission for --render-transmission\n";
          return 2;
        }
        illumination = value;
        continue;
      }
      if (arg == "--transport") {
        if (value != "pt" && value != "bdpt" && value != "guided" && value != "bdpt_guided") {
          std::cerr << "Invalid transport mode\n";
          return 2;
        }
        transport = value;
        continue;
      }
      int number = 0;
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
          number < (arg == "--seed" ? 0 : 1)) {
        std::cerr << "Invalid value for " << arg << '\n';
        return 2;
      }
      if (arg == "--seed") seed = number;
      else if (arg == "--reference-samples") reference_samples = number;
      else samples = number;
    }
  }
  const auto devices = ccl::Device::available_devices(metal ? DEVICE_MASK_METAL : DEVICE_MASK_CPU);
  if (devices.empty())
    return 2;
  if (render) return diffraction_render_test(devices.front(), relief_render, seed, samples, output, transport, mixed_render, transmission_render, illumination, reference_samples, angular_region);
  Stats stats;
  Profiler profiler;
  auto device = ccl::Device::create(devices.front(), stats, profiler, true);
  if (!device || device->have_error())
    return 2;
  if (argc == 2 && std::string(argv[1]) == "--fast-index-tables") {
    return test_fast_index_graph(device.get(), profiler) ? 0 : 1;
  }
  if (!test_microfacet_delta(profiler) ||
      !test_physical_shader_graph(device.get(), profiler, {1.5, 0}) ||
      !test_physical_shader_graph(device.get(), profiler, {0.9, 6}) ||
      !test_physical_shader_graph(device.get(), profiler, {0, 6}) ||
      !test_physical_shader_graph(device.get(), profiler, {0.9, 6}, true) ||
      !test_physical_shader_graph(device.get(), profiler, {0.9, 6}, true, true, M_PI_F / 4) ||
      !test_physical_shader_graph(device.get(), profiler, {0.9, 6}, true, true, M_PI_F / 2) ||
      !test_order_routing(device.get(), profiler,
                          argc == 2 && std::string(argv[1]) == "--audit-delta-mixture") ||
      !test_physical_fresnel(device.get(), profiler) ||
      !test_physical_reverse(device.get(), profiler))
    return 1;
  DeviceScene scene(device.get());
  DiffractionManager manager;
  DiffractionGratingCache cache;
  cache.bounds = {{-0.5, -1, 380}, {0.5, 1, 780}};
  cache.nodes = {make_int4(-1, 0, 0, 0)};
  DiffractionGratingPackedCell cell;
  cell.bounds = cache.bounds;
  cell.ports = {{0, false}};
  cell.active_ports = {0};
  cell.matrices.resize(32, make_float2(0.25f, -0.125f));
  cache.cells = {cell};
  std::vector<DiffractionGratingCache> caches{cache, cache};
  caches[0].cells[0].operator_chart = true;
  caches[0].cells[0].chart_rotation = make_float2(1, 0);
  std::string error;
  if (manager.add_cache(caches[0], error) != 0 || manager.add_cache(caches[1], error) != 1) {
    std::cerr << error;
    return 1;
  }
  manager.device_update(&scene);
  if (manager.need_update() || device->have_error()) {
    std::cerr << device->error_message();
    return 1;
  }
  scene.diffraction_matrices.copy_from_device();
  scene.diffraction_descriptors.copy_from_device();
  if (device->have_error() || scene.diffraction_matrices.size() != 64 ||
      scene.diffraction_matrices[63].x != 0.25f || scene.diffraction_matrices[63].y != -0.125f ||
      scene.diffraction_descriptors[2].x != 1 || scene.diffraction_descriptors[3].z != 32)
    return 1;
  DiffractionCacheCellView view;
  auto lookup = [&](int handle, float3 query) {
    return diffraction_cache_cell_view(scene.diffraction_descriptors.data(),
                                       scene.diffraction_domains.data(),
                                       scene.diffraction_nodes.data(),
                                       scene.diffraction_layout.data(),
                                       scene.diffraction_bounds.data(),
                                       int(manager.cache_count()),
                                       handle,
                                       query,
                                       &view);
  };
  if (!lookup(1, make_float3(0, 0, 580)) || view.matrix_offset != 32 || view.port_offset != 1 ||
      view.active_offset != 1 || view.ports != 1 || view.coordinate.x != 0.5f ||
      view.coordinate.y != 0.5f || view.coordinate.z != 0.5f)
    return 1;
  if (lookup(-1, make_float3(0, 0, 580)) || lookup(2, make_float3(0, 0, 580)) ||
      lookup(0, make_float3(0, 0, 800)))
    return 1;
  KernelGlobalsCPU globals;
  globals.data = scene.data;
  globals.diffraction_descriptors.data = scene.diffraction_descriptors.data();
  globals.diffraction_domains.data = scene.diffraction_domains.data();
  globals.diffraction_nodes.data = scene.diffraction_nodes.data();
  globals.diffraction_layout.data = scene.diffraction_layout.data();
  globals.diffraction_bounds.data = scene.diffraction_bounds.data();
  globals.diffraction_matrices.data = scene.diffraction_matrices.data();
  globals.diffraction_active.data = scene.diffraction_active.data();
  globals.diffraction_ports.data = scene.diffraction_ports.data();
  ThreadKernelGlobalsCPU thread_globals(globals, nullptr, profiler, 0);
  if (globals.data.tables.num_diffraction_caches != 2 ||
      !diffraction_scene_cell_view(&thread_globals, 1, make_float3(0, 0, 580), &view) ||
      view.matrix_offset != 32 ||
      diffraction_scene_cell_view(&thread_globals, 2, make_float3(0, 0, 580), &view))
    return 1;
  const float2 boundary[] = {zero_float2(), zero_float2(), make_float2(1, 1), make_float2(1, 0)};
  float power = 0;
  float2 jones[4];
  if (!diffraction_scene_cell_power<2>(&thread_globals, &view, boundary, 0, &power) ||
      power != 0.15625f ||
      diffraction_scene_cell_jones<2>(&thread_globals, &view, boundary, 0, jones))
    return 1;
  if (!diffraction_scene_cell_view(&thread_globals, 0, make_float3(0, 0, 580), &view) ||
      !diffraction_scene_cell_jones<2>(&thread_globals, &view, boundary, 0, jones) ||
      !diffraction_scene_cell_power<2>(&thread_globals, &view, boundary, 0, &power))
    return 1;
  for (int i = 0; i < 4; i++) {
    const float real = (i == 0 || i == 3) ? 24.0f / 37 : -13.0f / 37;
    if (std::abs(jones[i].x - real) > 2e-6f || std::abs(jones[i].y - 4.0f / 37) > 2e-6f)
      return 1;
  }
  if (std::abs(power - 777.0f / 1369) > 2e-6f)
    return 1;
  DiffractionSceneSample sampled;
  if (!diffraction_scene_sample<2>(
          &thread_globals, 0, make_float3(0, 0, -1), false, 1, 1, 580, 740, 0.5f, &sampled) ||
      sampled.relative_order != 0 || sampled.transmission || sampled.probability != 1 ||
      sampled.direction.x != 0 || sampled.direction.y != 0 || sampled.direction.z != 1 ||
      std::abs(sampled.throughput - 777.0f / 1369) > 2e-6f ||
      diffraction_scene_sample<2>(
          &thread_globals, 0, make_float3(0, 0, -1), true, 1, 1, 580, 740, 0.5f, &sampled) ||
      diffraction_scene_sample<2>(
          &thread_globals, 0, make_float3(0, 0, -1), false, 1, 1, 800, 740, 0.5f, &sampled))
    return 1;
  /* Execute the serialized SVM payload, including stack-connected frame and
   * mix weight. Verify skipped nodes consume exactly the same payload. */
  SVMNodeDiffractionSmoothBsdfData svm_data{};
  svm_data.cache_handle = 0;
  svm_data.pitch = 740;
  svm_data.upper_index = svm_data.lower_index = 1;
  svm_data.normal_offset = 0;
  svm_data.tangent_offset = 3;
  constexpr int payload_words = sizeof(svm_data) / sizeof(uint);
  uint encoded[payload_words];
  memcpy(encoded, &svm_data, sizeof(svm_data));
  thread_globals.svm_nodes.data = encoded;
  thread_globals.svm_nodes.width = payload_words;
  float stack[SVM_STACK_SIZE] = {};
  stack[2] = 2;
  stack[3] = 1;
  stack[5] = 1;
  stack[6] = 0.25f;
  SVMNodeClosureBsdf node{};
  node.closure_type = CLOSURE_BSDF_DIFFRACTION_SMOOTH_ID;
  node.mix_weight_offset = 6;
  ShaderData svm_sd{};
  svm_sd.num_closure_left = 1;
  svm_sd.shader_flag = SD_REQUIRES_WAVELENGTH;
  svm_sd.rand_wavelength = 0.5f;
  const auto execute = [&]() {
    return svm_node_closure_bsdf<KERNEL_FEATURE_NODE_BSDF, SHADER_TYPE_SURFACE>(
        &thread_globals, &svm_sd, stack, one_spectrum(), node, PathRayVisibility(0), 0, 0);
  };
  if (execute() != payload_words || svm_sd.num_closure != 1 ||
      svm_sd.closure[0].type != CLOSURE_BSDF_DIFFRACTION_SMOOTH_ID ||
      std::abs(average(svm_sd.closure[0].weight) - 0.25f) > 2e-6f)
    return 1;
  const auto &svm_bsdf = *(const DiffractionSmoothBsdf *)&svm_sd.closure[0];
  if (svm_bsdf.N.z != 1 || svm_bsdf.T.x != 1 || svm_bsdf.T.z != 0 || svm_bsdf.incoming_substrate)
    return 1;
  svm_sd.num_closure = 0;
  svm_sd.num_closure_left = 1;
  svm_sd.runtime_flag = SR_BACKFACING;
  if (execute() != payload_words || svm_sd.num_closure != 1 ||
      !((const DiffractionSmoothBsdf *)&svm_sd.closure[0])->incoming_substrate)
    return 1;
  svm_sd.num_closure = 0;
  svm_sd.num_closure_left = 1;
  stack[6] = 0;
  if (execute() != payload_words || svm_sd.num_closure != 0 ||
      svm_node_closure_bsdf_skip(7, node.closure_type) != 7 + payload_words)
    return 1;
  std::cout
      << "Passed physical smooth SVM payload, linked frame, mixing, backface and skip checks\n";
  DiffractionSmoothClosure osl_closure{};
  osl_closure.N = make_float3(0, 0, 2);
  osl_closure.T = make_float3(1, 0, 1);
  osl_closure.cache_handle = 0;
  osl_closure.pitch = 740;
  osl_closure.upper_index = osl_closure.lower_index = 1;
  ShaderData osl_sd{};
  osl_sd.N = osl_sd.wi = make_float3(0, 0, 1);
  osl_sd.num_closure_left = 1;
  osl_sd.rand_wavelength = 0.5f;
  float3 layer_albedo = one_float3();
  osl_closure_diffraction_smooth_setup(&thread_globals, &osl_sd, PathRayVisibility(0),
                                       0, one_float3(), &osl_closure, &layer_albedo);
  if (osl_sd.num_closure != 0 || len_squared(layer_albedo) != 0)
    return 1;
  osl_sd.shader_flag = SD_REQUIRES_WAVELENGTH;
  osl_closure_diffraction_smooth_setup(&thread_globals, &osl_sd, PathRayVisibility(0),
                                       0, make_float3(0.25f), &osl_closure, &layer_albedo);
  if (osl_sd.num_closure != 1 ||
      osl_sd.closure[0].type != CLOSURE_BSDF_DIFFRACTION_SMOOTH_ID ||
      std::abs(average(layer_albedo) - 0.25f * 777.0f / 1369) > 2e-6f)
    return 1;
  std::cout << "Passed physical OSL setup, wavelength requirement and absorbing layer albedo checks\n";
  ShaderData setup{};
  setup.num_closure_left = 1;
  setup.rand_wavelength = 0.5f;
  const float3 white = make_float3(1, 1, 1), normal = make_float3(0, 0, 2),
               tangent = make_float3(1, 0, 1);
  if (bsdf_diffraction_smooth_setup(
          &thread_globals, &setup, white, normal, tangent, 0, 740, 1, 1, false) ||
      setup.num_closure != 0)
    return 1;
  setup.shader_flag = SD_REQUIRES_WAVELENGTH;
  if (bsdf_diffraction_smooth_setup(
          &thread_globals, &setup, white, normal, normal, 0, 740, 1, 1, false) ||
      bsdf_diffraction_smooth_setup(
          &thread_globals, &setup, white, normal, tangent, 2, 740, 1, 1, false) ||
      setup.num_closure != 0)
    return 1;
  if (!bsdf_diffraction_smooth_setup(
          &thread_globals, &setup, white, normal, tangent, 0, 740, 1, 1, false) ||
      setup.num_closure != 1 || setup.num_closure_left != 0 ||
      !(setup.runtime_flag & SR_BSDF_HAS_DISPERSION) || (setup.runtime_flag & SR_BSDF_HAS_EVAL))
    return 1;
  const auto &closure = *(const DiffractionSmoothBsdf *)&setup.closure[0];
  if (closure.type != CLOSURE_BSDF_DIFFRACTION_SMOOTH_ID || closure.N.z != 1 || closure.T.x != 1 ||
      closure.T.z != 0 || closure.wavelength != 1000 * sample_wavelength(setup.rand_wavelength))
    return 1;
  if (bsdf_diffraction_smooth_setup(
          &thread_globals, &setup, white, normal, tangent, 0, 740, 1, 1, false) ||
      setup.num_closure != 1)
    return 1;
  Spectrum eval;
  float3 outgoing;
  float pdf, eta;
  float2 roughness;
  const int label = bsdf_diffraction_smooth_sample(&thread_globals,
                                                   (const ShaderClosure *)&closure,
                                                   closure.N,
                                                   closure.N,
                                                   make_float3(0.5f, 0.5f, 0.5f),
                                                   &eval,
                                                   &outgoing,
                                                   &pdf,
                                                   &roughness,
                                                   &eta);
  if (label != (LABEL_REFLECT | LABEL_SINGULAR) || pdf != 1e6f || eta != 1 || outgoing.z != 1 ||
      std::abs(average(eval) / pdf - 777.0f / 1369) > 2e-6f || len_squared(roughness) != 0)
    return 1;
  manager.device_free(&scene);
  if (!manager.need_update() || scene.data.tables.num_diffraction_caches != 0 ||
      scene.diffraction_matrices.size() != 0)
    return 1;
  manager.device_update(&scene);
  if (manager.need_update() || device->have_error() || scene.diffraction_matrices.size() != 64)
    return 1;
  /* Same-size replacement must upload new values rather than reuse stale data. */
  caches[1].cells[0].matrices[31] = make_float2(0.375f, 0.0625f);
  if (!manager.set_caches(caches, error))
    return 1;
  manager.device_update(&scene);
  scene.diffraction_matrices.copy_from_device();
  if (manager.need_update() || device->have_error() ||
      scene.diffraction_matrices[63].x != 0.375f || scene.diffraction_matrices[63].y != 0.0625f)
    return 1;
  /* A pre-existing device error must prevent commit. Recovery here means a
   * fresh Device/DeviceScene, matching the fact that Device errors are sticky.
   * This does not simulate a failure halfway through a copy. */
  caches[1].cells[0].matrices[31] = make_float2(0.5f, 0.125f);
  if (!manager.set_caches(caches, error))
    return 1;
  device->set_error("Intentional diffraction integration-test device error");
  manager.device_update(&scene);
  if (!manager.need_update() || scene.diffraction_matrices[63].x != 0.375f)
    return 1;
  manager.device_free(&scene);
  auto recovered_device = ccl::Device::create(devices.front(), stats, profiler, true);
  if (!recovered_device || recovered_device->have_error())
    return 2;
  DeviceScene recovered_scene(recovered_device.get());
  manager.device_update(&recovered_scene);
  recovered_scene.diffraction_matrices.copy_from_device();
  if (manager.need_update() || recovered_device->have_error() ||
      recovered_scene.diffraction_matrices.size() != 64 ||
      recovered_scene.diffraction_matrices[63].x != 0.5f ||
      recovered_scene.diffraction_matrices[63].y != 0.125f)
    return 1;
  if (!manager.set_caches({}, error))
    return 1;
  manager.device_update(&recovered_scene);
  if (manager.need_update() || recovered_scene.data.tables.num_diffraction_caches != 0 ||
      recovered_scene.diffraction_nodes.size() != 0 ||
      recovered_scene.diffraction_matrices.size() != 0)
    return 1;
  std::cout << "Passed upload/readback, replacement, release/reupload, device-error recovery and "
               "clear on "
            << devices.front().description << '\n';
  return 0;
}
