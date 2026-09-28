/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "kernel/util/diffraction_coordinates.h"
#include "scene/diffraction.h"
#include "util/math.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
using namespace ccl;
int main(int argc, const char **argv)
{
  if (argc != 3)
    return 2;
  std::ifstream input(argv[2]);
  std::string line;
  if (!std::getline(input, line) || (line != "wavelength_nm,n,k" && line != "wavelength_nm,n,k\r"))
    return 2;
  std::vector<DiffractionIndexSample> aluminum;
  while (std::getline(input, line)) {
    std::istringstream row(line);
    double wavelength, n, k;
    char a, b;
    if (!(row >> wavelength >> a >> n >> b >> k) || a != ',' || b != ',' ||
        !std::isfinite(wavelength) || !std::isfinite(n) || !std::isfinite(k) || wavelength <= 0 ||
        n < 0 || k <= 0 || (!aluminum.empty() && wavelength <= aluminum.back().wavelength))
      return 2;
    row >> std::ws;
    if (!row.eof())
      return 2;
    aluminum.push_back({wavelength, {n, k}});
  }
  if (aluminum.size() != 206 || aluminum.front().wavelength > 380 ||
      aluminum.back().wavelength < 780)
    return 2;
  @autoreleasepool {
    std::vector<int4> descriptors, nodes, layout;
    std::vector<float4> domains, bounds, queries, expected, reverse_checks;
    std::vector<int2> ports;
    std::vector<int> active{0};
    std::vector<float2> matrices;
    for (int handle = 0; handle < 4; handle++) {
      const bool mirror = handle >= 2;
      const int sx = handle % 2 ? 1 : -1;
      descriptors.push_back(make_int4(handle, 1, handle, 1));
      descriptors.push_back(make_int4(2 * handle, 0, 128 * handle, mirror));
      domains.push_back(make_float4(-0.5f, -1, 500, 0));
      domains.push_back(make_float4(mirror ? 0 : 0.5f, mirror ? 0 : 1, 600, 0));
      bounds.push_back(domains[2 * handle]);
      bounds.push_back(domains[2 * handle + 1]);
      nodes.push_back(make_int4(-1, 0, 0, 0));
      layout.push_back(make_int4(0, 0, 0, 0));
      layout.push_back(make_int4(2, 0, 0, 0));
      ports.push_back(make_int2(0, 0));
      ports.push_back(make_int2(mirror ? -1 : sx, 1));
      matrices.resize(128 * (handle + 1), zero_float2());
      for (int corner = 0; corner < 8; corner++)
        for (int c = 0; c < 4; c++)
          matrices[128 * handle + 16 * corner + ((c + 2) % 4) * 4 + c] = make_float2(
              std::sqrt(0.5f), 0);
      for (int sy : {-1, 1})
        for (bool below : {false, true})
          for (float random : {0.0f, 0.5f, 0.99999994f}) {
            const float ni = below ? 1.5f : 1, no = below ? 1 : 1.5f;
            const float x = sx * (below ? 0.6f : 0.1f) / ni, y = sy * 0.2f / ni;
            queries.push_back(
                make_float4(x, y, (below ? 1 : -1) * std::sqrt(1 - x * x - y * y), 1100));
            queries.push_back(make_float4(handle, below, random, 550));
            const float ox = sx * (below ? 0.1f : 0.6f) / no, oy = sy * 0.2f / no;
            expected.push_back(
                make_float4(ox, oy, (below ? 1 : -1) * std::sqrt(1 - ox * ox - oy * oy), no / ni));
            expected.push_back(make_float4(below ? -sx : sx, 1, 1, 0.5f));
          }
    }
    for (bool below : {false, true}) {
      for (double angle : {0.0, 0.4, 0.9}) {
        const double ni = below ? 1.5 : 1, no = below ? 1 : 1.5;
        const float3 ray = make_float3(std::sin(angle) * std::cos(0.3),
                                       std::sin(angle) * std::sin(0.3),
                                       (below ? 1 : -1) * std::cos(angle));
        DiffractionCacheCoordinates coordinates;
        if (!diffraction_cache_coordinates(ray, ni, 550, 200, false, &coordinates))
          return 2;
        const auto q = coordinates.query;
        DiffractionGratingProfile profile{200, 0, 0.41, 1, 1.5, 1, 1.5};
        DiffractionGratingCacheOptions config;
        config.bounds = {{float(q.x - 0.0001f), float(q.y - 0.0001f), double(549.9f)},
                         {float(q.x + 0.0001f), float(q.y + 0.0001f), double(550.1f)}};
        config.half_orders = 4;
        config.retained_half_orders = 3;
        config.tolerance = 1e-5;
        config.maximum_nodes = 63;
        DiffractionGratingCache cache;
        DiffractionGratingCacheStats stats;
        DiffractionGratingDeviceBuffers packed;
        std::string error;
        if (!diffraction_grating_build_cache(profile, config, cache, stats, error) ||
            !diffraction_grating_device_buffers(cache, packed, error))
        {
          std::cerr << error;
          return 2;
        }
        const int handle = descriptors.size() / 2;
        descriptors.push_back(make_int4(
            nodes.size(), packed.nodes.size(), layout.size() / 2, packed.layout.size() / 2));
        descriptors.push_back(make_int4(ports.size(), active.size(), matrices.size(), 0));
        domains.push_back(make_float4(
            config.bounds.lower[0], config.bounds.lower[1], config.bounds.lower[2], 0));
        domains.push_back(make_float4(
            config.bounds.upper[0], config.bounds.upper[1], config.bounds.upper[2], 0));
        nodes.insert(nodes.end(), packed.nodes.begin(), packed.nodes.end());
        layout.insert(layout.end(), packed.layout.begin(), packed.layout.end());
        bounds.insert(bounds.end(), packed.bounds.begin(), packed.bounds.end());
        ports.insert(ports.end(), packed.ports.begin(), packed.ports.end());
        active.insert(active.end(), packed.active.begin(), packed.active.end());
        matrices.insert(matrices.end(), packed.matrices.begin(), packed.matrices.end());
        const double ci = std::cos(angle);
        const auto z = std::sqrt(std::complex<double>(no * no - ni * ni * (1 - ci * ci)));
        const auto rs = (ni * ci - z) / (ni * ci + z),
                   rp = (no * no * ci - ni * z) / (no * no * ci + ni * z);
        const double R = 0.5 * (std::norm(rs) + std::norm(rp));
        for (int j = 0; j < 1024; j++) {
          const float u = (j + 0.5f) / 1024;
          const bool output_below = u >= (below ? 1 - R : R);
          const bool transmission = output_below != below;
          const double index = transmission ? no : ni;
          const double ox = ni * ray.x / index, oy = ni * ray.y / index;
          const double oz = (output_below ? -1 : 1) *
                            std::sqrt(std::max(0.0, 1 - ox * ox - oy * oy));
          queries.push_back(make_float4(ray.x, ray.y, ray.z, 200));
          queries.push_back(make_float4(handle, below, u, 550));
          expected.push_back(make_float4(ox, oy, oz, transmission ? no / ni : 1));
          expected.push_back(make_float4(0, transmission, transmission ? 1 - R : R, 1));
        }
      }
    }
    for (int material = 0; material < 8; material++) {
      const bool metal = material > 0;
      const double pitch = (material == 2 || material == 7) ? 1600 : 740;
      for (double wavelength : {520.0, 550.0, 580.0}) {
        if (material < 3 && wavelength != 550)
          continue;
        for (bool below : {false, true}) {
          if (metal && below)
            continue;
          for (double angle : {0.0, 0.4, 0.9}) {
            const double ni = below ? 1.5 : 1, no = below ? 1 : 1.5;
            const float3 ray = make_float3(std::sin(angle) * std::cos(0.3),
                                           std::sin(angle) * std::sin(0.3),
                                           (below ? 1 : -1) * std::cos(angle));
            DiffractionCacheCoordinates coordinates;
            if (!diffraction_cache_coordinates(ray, ni, wavelength, pitch, false, &coordinates))
              return 2;
            const auto q = coordinates.query;
            const std::complex<double> index = metal ? std::complex<double>(0.9, 6) :
                                                       std::complex<double>(1.5);
            DiffractionGratingProfile profile{pitch, 150, 0.41, 1, index, 1, index};
            const std::vector<DiffractionIndexSample> spectrum =
                material >= 6 ? aluminum :
                                std::vector<DiffractionIndexSample>{
                                    {500, {0.9, 6}}, {550, {1.5, 4}}, {600, {1.2, 5}}};
            if (material == 3 || material >= 5)
              profile.ridge_spectrum = spectrum;
            if (material >= 4)
              profile.absorbing_substrate_spectrum = spectrum;
            auto reference_profile = profile;
            if (material >= 3) {
              reference_profile.ridge_spectrum.clear();
              reference_profile.absorbing_substrate_spectrum.clear();
              const double t = wavelength <= 550 ? (wavelength - 500) / 50 :
                                                   (wavelength - 550) / 50;
              const auto sampled_index = wavelength <= 550 ?
                                             (1 - t) * std::complex<double>(0.9, 6) +
                                                 t * std::complex<double>(1.5, 4) :
                                             (1 - t) * std::complex<double>(1.5, 4) +
                                                 t * std::complex<double>(1.2, 5);
              if (material == 3 || material >= 5)
                reference_profile.ridge_ior = sampled_index;
              if (material >= 4)
                reference_profile.substrate_ior = sampled_index;
            }
            if (material >= 6) {
              const auto upper = std::lower_bound(
                  aluminum.begin(), aluminum.end(), wavelength, [](const auto &sample, double w) {
                    return sample.wavelength < w;
                  });
              if (upper == aluminum.begin() || upper == aluminum.end())
                return 2;
              const auto &lower = *(upper - 1);
              const double t = (wavelength - lower.wavelength) /
                               (upper->wavelength - lower.wavelength);
              reference_profile.ridge_ior = reference_profile.substrate_ior = lower.index +
                                                                              t * (upper->index -
                                                                                   lower.index);
            }
            DiffractionGratingCacheOptions config;
            config.bounds = {
                {float(q.x - 0.0001f), float(q.y - 0.0001f), double(float(wavelength - 0.1))},
                {float(q.x + 0.0001f), float(q.y + 0.0001f), double(float(wavelength + 0.1))}};
            config.half_orders = 16;
            config.retained_half_orders = 3;
            config.tolerance = 5e-7;
            config.maximum_nodes = 63;
            DiffractionGratingCache cache;
            DiffractionGratingCacheStats stats;
            DiffractionGratingDeviceBuffers packed;
            std::string error;
            if (!diffraction_grating_build_cache(profile, config, cache, stats, error) ||
                !diffraction_grating_device_buffers(cache, packed, error))
            {
              std::cerr << error;
              return 2;
            }
            const int handle = descriptors.size() / 2;
            descriptors.push_back(make_int4(
                nodes.size(), packed.nodes.size(), layout.size() / 2, packed.layout.size() / 2));
            descriptors.push_back(make_int4(ports.size(), active.size(), matrices.size(), 0));
            domains.push_back(make_float4(
                config.bounds.lower[0], config.bounds.lower[1], config.bounds.lower[2], 0));
            domains.push_back(make_float4(
                config.bounds.upper[0], config.bounds.upper[1], config.bounds.upper[2], 0));
            nodes.insert(nodes.end(), packed.nodes.begin(), packed.nodes.end());
            layout.insert(layout.end(), packed.layout.begin(), packed.layout.end());
            bounds.insert(bounds.end(), packed.bounds.begin(), packed.bounds.end());
            ports.insert(ports.end(), packed.ports.begin(), packed.ports.end());
            active.insert(active.end(), packed.active.begin(), packed.active.end());
            matrices.insert(matrices.end(), packed.matrices.begin(), packed.matrices.end());
            DiffractionGratingBlock block;
            DiffractionGratingPowerBlock power;
            if (!diffraction_grating_solve_bloch(reference_profile,
                                                 wavelength,
                                                 q.x * wavelength / pitch,
                                                 q.y,
                                                 16,
                                                 block,
                                                 error))
            {
              std::cerr << error;
              return 2;
            }
            diffraction_grating_power_block(block, power);
            int incoming = -1;
            for (int p = 0; p < int(power.ports.size()); p++)
              if (power.ports[p].order == coordinates.incoming_order &&
                  power.ports[p].substrate == below)
                incoming = p;
            if (incoming < 0)
              return 2;
            double total = 0;
            for (int p = 0; p < int(power.ports.size()); p++)
              total += power.matrix[p * power.ports.size() + incoming];
            if (metal && !(total > 0 && total < 1)) {
              std::cerr << "Metal fixture must retain absorption";
              return 2;
            }
            int reverse_handle = -1;
            DiffractionGratingPowerBlock reverse_power;
            if (material == 6 && wavelength == 550 && angle == 0.9) {
              auto reversed = config;
              for (int axis = 0; axis < 2; axis++) {
                reversed.bounds.lower[axis] = -config.bounds.upper[axis];
                reversed.bounds.upper[axis] = -config.bounds.lower[axis];
              }
              if (!diffraction_grating_build_cache(profile, reversed, cache, stats, error) ||
                  !diffraction_grating_device_buffers(cache, packed, error))
              {
                std::cerr << error;
                return 2;
              }
              reverse_handle = descriptors.size() / 2;
              descriptors.push_back(make_int4(
                  nodes.size(), packed.nodes.size(), layout.size() / 2, packed.layout.size() / 2));
              descriptors.push_back(make_int4(ports.size(), active.size(), matrices.size(), 0));
              domains.push_back(make_float4(reversed.bounds.lower[0],
                                            reversed.bounds.lower[1],
                                            reversed.bounds.lower[2],
                                            0));
              domains.push_back(make_float4(reversed.bounds.upper[0],
                                            reversed.bounds.upper[1],
                                            reversed.bounds.upper[2],
                                            0));
              nodes.insert(nodes.end(), packed.nodes.begin(), packed.nodes.end());
              layout.insert(layout.end(), packed.layout.begin(), packed.layout.end());
              bounds.insert(bounds.end(), packed.bounds.begin(), packed.bounds.end());
              ports.insert(ports.end(), packed.ports.begin(), packed.ports.end());
              active.insert(active.end(), packed.active.begin(), packed.active.end());
              matrices.insert(matrices.end(), packed.matrices.begin(), packed.matrices.end());
              DiffractionGratingBlock reverse_block;
              if (!diffraction_grating_solve_bloch(reference_profile,
                                                   wavelength,
                                                   -q.x * wavelength / pitch,
                                                   -q.y,
                                                   16,
                                                   reverse_block,
                                                   error))
                return 2;
              diffraction_grating_power_block(reverse_block, reverse_power);
            }
            for (int j = 0; j < 1024; j++) {
              const float u = (j + 0.5f) / 1024;
              int selected = -1;
              double cumulative = 0;
              for (int p = 0; p < int(power.ports.size()); p++) {
                cumulative += power.matrix[p * power.ports.size() + incoming];
                if (u * total < cumulative) {
                  selected = p;
                  break;
                }
              }
              if (selected < 0)
                return 2;
              const auto port = power.ports[selected];
              const bool output_below = port.substrate;
              const bool transmission = output_below != below;
              const int relative = port.order - coordinates.incoming_order;
              const double probability = power.matrix[selected * power.ports.size() + incoming] /
                                         total;
              const double index = transmission ? no : ni;
              const double norm = std::sqrt(double(ray.x) * ray.x + double(ray.y) * ray.y +
                                            double(ray.z) * ray.z);
              const double ox = (ni * ray.x / norm + relative * wavelength / pitch) / index,
                           oy = ni * ray.y / norm / index;
              const double oz = (output_below ? -1 : 1) *
                                std::sqrt(std::max(0.0, 1 - ox * ox - oy * oy));
              queries.push_back(make_float4(ray.x, ray.y, ray.z, pitch));
              queries.push_back(make_float4(handle, below, u, wavelength));
              expected.push_back(make_float4(ox, oy, oz, transmission ? no / ni : 1));
              expected.push_back(make_float4(relative, transmission, probability, total));
              if (reverse_handle >= 0) {
                int input = -1, output = -1;
                for (int r = 0; r < int(reverse_power.ports.size()); r++) {
                  const auto &rp = reverse_power.ports[r];
                  if (rp.order == -port.order && rp.substrate == port.substrate)
                    input = r;
                  if (rp.order == -coordinates.incoming_order && rp.substrate == below)
                    output = r;
                }
                if (input < 0 || output < 0)
                  return 2;
                double sum = 0;
                for (int r = 0; r < int(reverse_power.ports.size()); r++)
                  sum += reverse_power.matrix[r * reverse_power.ports.size() + input];
                const double reverse_mass =
                    reverse_power.matrix[output * reverse_power.ports.size() + input];
                reverse_checks.resize(queries.size() / 2, zero_float4());
                reverse_checks.back() = make_float4(
                    reverse_handle, reverse_mass, reverse_mass / sum, 1);
              }
            }
          }
        }
      }
    }
    reverse_checks.resize(queries.size() / 2, zero_float4());
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
                                                 encoding:NSUTF8StringEncoding
                                                    error:&error];
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeFast;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
      std::cerr << error.localizedDescription.UTF8String;
      return 2;
    }
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"diffraction_scene_sample_test"]
                                      error:&error];
    if (!pipeline) {
      std::cerr << "Metal sampler pipeline creation failed: "
                << error.localizedDescription.UTF8String << "\n";
      return 2;
    }
    auto upload = [&](const auto &data) {
      return [device newBufferWithBytes:data.data()
                                 length:data.size() * sizeof(data[0])
                                options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> buffers[] = {upload(descriptors),
                               upload(domains),
                               upload(nodes),
                               upload(layout),
                               upload(bounds),
                               upload(ports),
                               upload(active),
                               upload(matrices),
                               upload(queries),
                               [device newBufferWithLength:expected.size() * sizeof(float4)
                                                   options:MTLResourceStorageModeShared],
                               upload(reverse_checks)};
    for (auto buffer : buffers)
      if (!buffer)
        return 2;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    int failures = 0;
    double maximum_error = 0;
    std::vector<double> gpu_ms;
    const size_t group_size = std::min(NSUInteger(64), pipeline.maxTotalThreadsPerThreadgroup);
    for (int run = 0; run < 7; run++) {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      for (int i = 0; i < 11; i++)
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
      [encoder dispatchThreads:MTLSizeMake(queries.size() / 2, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        std::cerr << command.error.localizedDescription.UTF8String;
        return 2;
      }
      const double milliseconds = 1000 * (command.GPUEndTime - command.GPUStartTime);
      if (!std::isfinite(milliseconds) || milliseconds <= 0)
        return 2;
      if (run >= 2)
        gpu_ms.push_back(milliseconds);
      const float *actual = static_cast<const float *>(buffers[9].contents);
      const float *reference = reinterpret_cast<const float *>(expected.data());
      for (size_t i = 0; i < expected.size() * 4; i++) {
        const double difference = std::abs(actual[i] - reference[i]);
        if (std::isfinite(difference))
          maximum_error = std::max(maximum_error, difference);
        failures += !std::isfinite(actual[i]) || difference > 2e-6;
      }
    }
    auto sorted = gpu_ms;
    std::sort(sorted.begin(), sorted.end());
    size_t cache_bytes = 0;
    for (int i = 0; i < 8; i++)
      cache_bytes += buffers[i].length;
    std::cout << "{\"cases_per_run\":" << queries.size() / 2
              << ",\"runs\":7,\"warmup_runs\":2,\"compared_components\":"
              << expected.size() * 4 * 7 << ",\"failures\":" << failures
              << ",\"maximum_component_error\":" << maximum_error
              << ",\"threadgroup_size\":" << group_size << ",\"cache_bytes\":" << cache_bytes
              << ",\"device\":\"" << device.name.UTF8String << "\",\"gpu_ms\":[";
    for (size_t i = 0; i < gpu_ms.size(); i++)
      std::cout << (i ? "," : "") << gpu_ms[i];
    std::cout << "],\"median_gpu_ms\":" << sorted[sorted.size() / 2]
              << ",\"measurement\":\"hot fixed-query sampler dispatch; excludes build, upload and "
                 "rendering\"}\n";
    return failures ? 1 : 0;
  }
}
