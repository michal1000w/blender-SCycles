/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Upload audit exports and compare variable-size cache evaluation with the
 * shared CPU float matcher. This is not a Maxwell convergence test. */
#include "../performance/diffraction_packed_audit.h"
#include "kernel/util/diffraction_coordinates.h"
#include "kernel/util/diffraction_sample.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace ccl;
using Complex = std::complex<double>;

int main(int argc, const char **argv)
{
  if (argc != 4 && !(argc == 5 && std::string(argv[4]) == "--host-only"))
    return 2;
  try {
    @autoreleasepool {
      const int probes = std::stoi(argv[3]);
      if (probes < 1 || probes > 65536)
        throw std::runtime_error("Invalid probe count");
      NSString *directory = @(argv[2]);
      NSData *metadata = [NSData
          dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"cache.json"]];
      NSError *error = nil;
      NSDictionary *header = metadata ? [NSJSONSerialization JSONObjectWithData:metadata
                                                                        options:0
                                                                          error:&error] :
                                        nil;
      if (![header isKindOfClass:[NSDictionary class]] ||
          ![header[@"schema"] isEqual:@"cycles-diffraction-audit"] ||
          [header[@"version"] intValue] != 1 || ![header[@"byte_order"] isEqual:@"little"])
        throw std::runtime_error("Invalid cache metadata");
      NSData *binary = [NSData
          dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"cache.bin"]];
      if (!binary || binary.length != [header[@"binary_bytes"] unsignedLongLongValue])
        throw std::runtime_error("Invalid binary size");
      auto load = [&]<typename T>(NSString *name, int components, NSString *type) {
        NSDictionary *section = header[@"buffers"][name];
        const size_t offset = [section[@"offset"] unsignedLongLongValue];
        const size_t count = [section[@"count"] unsignedLongLongValue];
        const size_t bytes = [section[@"bytes"] unsignedLongLongValue];
        if (!section || ![section[@"type"] isEqual:type] ||
            [section[@"components"] intValue] != components || offset % 16 ||
            offset > binary.length || bytes > binary.length - offset ||
            count > binary.length / sizeof(T) || bytes != count * sizeof(T))
          throw std::runtime_error("Invalid buffer section");
        std::vector<T> data(count);
        std::memcpy(data.data(), static_cast<const char *>(binary.bytes) + offset, bytes);
        return data;
      };
      const auto nodes = load.template operator()<int4>(@"nodes", 4, @"int32");
      const auto layout = load.template operator()<int4>(@"cell_layout", 4, @"int32");
      const auto bounds = load.template operator()<float4>(@"cell_bounds", 4, @"float32");
      const auto ports = load.template operator()<int2>(@"ports", 2, @"int32");
      const auto active = load.template operator()<int>(@"active_ports", 1, @"int32");
      const auto matrices = load.template operator()<float2>(@"matrices", 2, @"float32");
      if (nodes.empty() || nodes.size() > 16777216 || layout.empty() || layout.size() % 2 ||
          bounds.size() != layout.size())
        throw std::runtime_error("Invalid cache dimensions");
      auto xyz = [](NSArray *a) {
        if (![a isKindOfClass:[NSArray class]] || a.count != 3)
          throw std::runtime_error("Invalid domain");
        return make_float3([a[0] floatValue], [a[1] floatValue], [a[2] floatValue]);
      };
      const float3 lower = xyz(header[@"lower"]), upper = xyz(header[@"upper"]);
      const bool mirror = [header[@"mirror_symmetry"] boolValue];
      NSDictionary *profile = header[@"profile"];
      const double pitch = [profile[@"pitch_nm"] doubleValue];
      const double ni = [profile[@"incident_ior"] doubleValue];
      const double ns = [profile[@"substrate_ior"][0] doubleValue];
      if (!(pitch > 0 && ni > 0 && ns > 0))
        throw std::runtime_error("Invalid exterior profile");
      std::vector<DiffractionGratingPackedCell> cells(layout.size() / 2);
      for (size_t c = 0; c < cells.size(); c++) {
        const int4 e = layout[2 * c], s = layout[2 * c + 1];
        if (e.x < 0 || e.y < 0 || e.z < 0 || e.w < 0 || e.w > 2 || s.x < 1 || s.x > 10 ||
            s.y < 0 || s.y > s.x)
          throw std::runtime_error("Invalid cell layout");
        const size_t matrix_count = size_t(e.w == 2 ? 27 : 8) * (2 * s.x) * (2 * s.x);
        if (size_t(e.x) > matrices.size() || matrix_count > matrices.size() - e.x ||
            size_t(e.y) > ports.size() || size_t(s.x) > ports.size() - e.y ||
            size_t(e.z) > active.size() || size_t(s.y) > active.size() - e.z)
          throw std::runtime_error("Cell buffer range exceeded");
        auto &cell = cells[c];
        cell.operator_chart = e.w != 0;
        cell.chart_degree = e.w;
        cell.chart_rotation = make_float2(bounds[2 * c].w, bounds[2 * c + 1].w);
        cell.matrices.assign(matrices.begin() + e.x, matrices.begin() + e.x + matrix_count);
        for (int j = 0; j < s.x; j++)
          cell.ports.push_back({ports[e.y + j].x, ports[e.y + j].y != 0});
        for (int j = 0; j < s.y; j++) {
          const int a = active[e.z + j];
          if (a < 0 || a >= s.x)
            throw std::runtime_error("Invalid active port");
          cell.active_ports.push_back(a);
        }
      }
      std::mt19937 rng(871923);
      std::uniform_real_distribution<float> uniform(0, 1);
      std::vector<float4> queries;
      std::vector<float2> boundary;
      std::vector<float> expected;
      std::vector<int> expected_leaves;
      auto pair = [](Complex z) { return make_float2(z.real(), z.imag()); };
      for (int probe = 0; probe < probes; probe++) {
#ifdef DIFFRACTION_RAY_LOOKUP
        const float cosine = uniform(rng), phi = 6.28318530718f * uniform(rng);
        const float sine = std::sqrt(1 - cosine * cosine);
        const float3 input_ray = make_float3(sine * std::cos(phi), sine * std::sin(phi), cosine);
        const float wavelength = lower.z + (upper.z - lower.z) * uniform(rng);
        DiffractionCacheCoordinates coordinates;
        if (!diffraction_cache_coordinates(input_ray, ni, wavelength, pitch, mirror, &coordinates))
          throw std::runtime_error("Ray coordinate mapping failed");
        float3 q = coordinates.query, original = q;
#else
        float3 q = make_float3(lower.x + (upper.x - lower.x) * uniform(rng),
                               lower.y + (upper.y - lower.y) * uniform(rng),
                               lower.z + (upper.z - lower.z) * uniform(rng));
        float3 original = q;
        if (mirror) {
          if (rng() % 2)
            original.x = -original.x;
          if (rng() % 2)
            original.y = -original.y;
        }
#endif
        const int leaf = diffraction_cache_lookup(
            nodes.data(), nodes.size(), lower, upper, original, mirror);
        if (leaf < 0 || size_t(leaf) >= cells.size())
          throw std::runtime_error("CPU lookup failed");
        const auto &cell = cells[leaf];
        const float4 lo = bounds[2 * leaf], hi = bounds[2 * leaf + 1];
        const float3 t = make_float3((q.x - lo.x) / (hi.x - lo.x),
                                     (q.y - lo.y) / (hi.y - lo.y),
                                     (q.z - lo.z) / (hi.z - lo.z));
        for (size_t incoming = 0; incoming < cell.ports.size(); incoming++) {
          const auto &port = cell.ports[incoming];
#ifdef DIFFRACTION_RAY_LOOKUP
          if (port.order != coordinates.incoming_order)
            continue;
#endif
          const double x = (double(q.x) + port.order) * q.z / pitch;
          if (port.substrate || ni * ni - x * x - double(q.y) * q.y <= 0)
            continue;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
#  ifdef DIFFRACTION_RAY_LOOKUP
          const float3 ray = make_float3(coordinates.reverse_orders ? -input_ray.x : input_ray.x,
                                         mirror ? -std::abs(input_ray.y) : input_ray.y,
                                         input_ray.z);
#  else
          const float3 ray = make_float3(
              x / ni, q.y / ni, std::sqrt(ni * ni - x * x - double(q.y) * q.y) / ni);
#  endif
          const double scale = ni / std::sqrt(double(ray.x) * ray.x + double(ray.y) * ray.y +
                                              double(ray.z) * ray.z);
          const double ray_x = scale * ray.x, ray_y = scale * ray.y;
          float4 inputs[20] = {};
#endif
          float2 coefficients[40] = {};
          for (size_t j = 0; j < cell.ports.size(); j++) {
            const double n = cell.ports[j].substrate ? ns : ni;
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
            const double kx = ray_x + (cell.ports[j].order - port.order) * double(q.z) / pitch;
            const double ky = ray_y;
            inputs[2 * j] = make_float4(ray.x, ray.y, ray.z, ni);
            inputs[2 * j + 1] = make_float4(n, q.z, pitch, cell.ports[j].order - port.order);
#else
            const double kx = (double(q.x) + cell.ports[j].order) * q.z / pitch, ky = q.y;
#endif
            const double q2 = n * n - kx * kx - ky * ky;
            const Complex z = q2 > 0 ? Complex(std::sqrt(q2)) : Complex(0, std::sqrt(-q2));
            const bool a = std::find(cell.active_ports.begin(), cell.active_ports.end(), int(j)) !=
                           cell.active_ports.end();
            coefficients[4 * j] = a ? pair((1.0 - z) / (1.0 + z)) : zero_float2();
            coefficients[4 * j + 1] = a ? pair((z - n * n) / (z + n * n)) : zero_float2();
            coefficients[4 * j + 2] = !a ? make_float2(1, 1) :
                                      q2 > 0 ?
                                           make_float2(2 * std::sqrt(z.real()) / (1 + z.real()),
                                                       2 * n * std::sqrt(z.real()) /
                                                           (z.real() + n * n)) :
                                           zero_float2();
            const double length = std::hypot(kx, ky);
            coefficients[4 * j + 3] = length > 0 ? make_float2(kx / length, ky / length) :
                                                   make_float2(1, 0);
          }
          float power[10] = {};
          bool valid;
          if (cell.operator_chart) {
            float2 jones[40];
            valid = diffraction_audit::chart_match(cell, coefficients, incoming, t, jones);
            if (valid)
              for (size_t j = 0; j < cell.ports.size(); j++)
                for (int k = 0; k < 4; k++)
                  power[j] += 0.5f * len_squared(jones[4 * j + k]);
          }
          else
            valid = diffraction_audit::intensity_match(cell, coefficients, incoming, t, power);
          if (!valid)
            throw std::runtime_error("CPU matching failed");
#ifdef DIFFRACTION_RAY_LOOKUP
          queries.push_back(make_float4(input_ray.x, input_ray.y, input_ray.z, wavelength));
#else
          queries.push_back(make_float4(original.x, original.y, original.z, port.order));
#endif
#ifdef DIFFRACTION_CONSTRUCT_BOUNDARIES
          float2 encoded[40];
          static_assert(sizeof(encoded) == sizeof(inputs));
          std::memcpy(encoded, inputs, sizeof(inputs));
          boundary.insert(boundary.end(), encoded, encoded + 40);
#else
          boundary.insert(boundary.end(), coefficients, coefficients + 40);
#endif
          expected.insert(expected.end(), power, power + 10);
          expected_leaves.push_back(leaf);
        }
      }
#ifdef DIFFRACTION_RAY_LOOKUP
      if (queries.size() != size_t(probes))
        throw std::runtime_error("Missing incident ray ports");
#endif
      if (queries.empty())
        throw std::runtime_error("No physical queries");
      for (float value : expected)
        if (!std::isfinite(value) || value < 0)
          throw std::runtime_error("Invalid CPU reference power");
      if (argc == 5) {
        std::cout << "{\"host_only\":true,\"probes\":" << probes
                  << ",\"physical_inputs\":" << queries.size()
                  << ",\"cells_loaded\":" << cells.size() << "}\n";
        return 0;
      }
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device)
        throw std::runtime_error("No Metal device");
      NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
                                                   encoding:NSUTF8StringEncoding
                                                      error:&error];
      MTLCompileOptions *options = [MTLCompileOptions new];
      options.mathMode = MTLMathModeFast;
      id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
      if (!library)
        throw std::runtime_error(error.localizedDescription.UTF8String);
      id<MTLComputePipelineState> pipeline = [device
          newComputePipelineStateWithFunction:[library
                                                  newFunctionWithName:@"diffraction_full_cache"]
                                        error:&error];
      if (!pipeline)
        throw std::runtime_error(error.localizedDescription.UTF8String);
      auto upload = [&](const auto &data) {
        return [device newBufferWithBytes:data.data()
                                   length:data.size() * sizeof(data[0])
                                  options:MTLResourceStorageModeShared];
      };
      const std::vector<float4> domain = {make_float4(lower.x, lower.y, lower.z, nodes.size()),
                                          make_float4(upper.x, upper.y, upper.z, mirror),
                                          make_float4(ni, ns, pitch, 0)};
      id<MTLBuffer> buffers[] = {
          upload(nodes),
          upload(layout),
          upload(bounds),
          upload(ports),
          [device newBufferWithBytes:active.empty() ? static_cast<const void *>(&probes) :
                                                      active.data()
                              length:std::max(size_t(1), active.size()) * sizeof(int)
                             options:MTLResourceStorageModeShared],
          upload(matrices),
          upload(queries),
          upload(boundary),
          [device newBufferWithLength:expected.size() * sizeof(float)
                              options:MTLResourceStorageModeShared],
          [device newBufferWithLength:queries.size() * sizeof(int)
                              options:MTLResourceStorageModeShared],
          upload(domain),
          [device newBufferWithLength:queries.size() * sizeof(float4)
                              options:MTLResourceStorageModeShared]};
      for (auto buffer : buffers)
        if (!buffer)
          throw std::runtime_error("Metal allocation failed");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      size_t failures = 0;
      double maximum = 0;
      NSMutableArray *times = [NSMutableArray array];
      for (int run = 0; run < 7; run++) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        for (int i = 0; i < 12; i++)
          [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder dispatchThreads:MTLSizeMake(queries.size(), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
          throw std::runtime_error(command.error.localizedDescription.UTF8String);
        const float *actual = static_cast<const float *>(buffers[8].contents);
        const int *leaves = static_cast<const int *>(buffers[9].contents);
        for (size_t i = 0; i < queries.size(); i++) {
          bool failed = leaves[i] != expected_leaves[i];
          for (int j = 0; j < 10; j++) {
            const double delta = std::abs(double(actual[10 * i + j]) - expected[10 * i + j]);
            failed |= !std::isfinite(delta) || delta > 2e-5;
            maximum = std::max(maximum, delta);
          }
#ifdef DIFFRACTION_SAMPLE_ORDERS
          const float random = float((unsigned(i) * 1664525u + 1013904223u) & 0xffffffu) *
                               0x1p-24f;
          DiffractionOrderSample selected;
          const bool sampled = diffraction_sample_order(
              expected.data() + 10 * i, 10, random, &selected);
          const float4 actual_sample = static_cast<const float4 *>(buffers[11].contents)[i];
          failed |= (actual_sample.w != 0) != sampled;
          if (sampled)
            failed |= actual_sample.x != selected.port || !std::isfinite(actual_sample.y) ||
                      !std::isfinite(actual_sample.z) ||
                      std::abs(actual_sample.y - selected.probability) > 2e-5f ||
                      std::abs(actual_sample.z - selected.throughput) > 2e-5f;
#endif
          failures += failed;
        }
        if (run >= 2)
          [times addObject:@(1000 * (command.GPUEndTime - command.GPUStartTime))];
      }
      NSDictionary *report = @{
        @"device" : device.name,
        @"probes" : @(probes),
        @"physical_inputs" : @(queries.size()),
        @"failures" : @(failures),
        @"maximum_power_error" : @(maximum),
        @"gpu_ms" : times,
        @"warmup_runs" : @2,
        @"validated_runs" : @7,
#ifdef DIFFRACTION_SAMPLE_ORDERS
        @"sample_orders" : @YES,
#else
        @"sample_orders" : @NO,
#endif
        @"timing_scope" :
            @"hot repeated ray batch; excludes allocation, compilation and CPU preparation",
#ifdef DIFFRACTION_INPLACE_CHART
        @"inplace_chart" : @YES,
#else
        @"inplace_chart" : @NO,
#endif
        @"reference" :
#ifdef DIFFRACTION_RAY_LOOKUP
            @"ray-derived lookup; CPU float matching with double ray-derived boundary coefficients"
#elif defined(DIFFRACTION_CONSTRUCT_BOUNDARIES)
            @"shared CPU float packed matcher; double exterior coefficients from the same "
            @"quantized ray"
#else
            @"shared CPU float packed matcher; precomputed exterior coefficients"
#endif
      };
      NSData *json = [NSJSONSerialization dataWithJSONObject:report options:0 error:&error];
      std::cout.write(static_cast<const char *>(json.bytes), json.length);
      std::cout << '\n';
      return failures ? 1 : 0;
    }
  }
  catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
