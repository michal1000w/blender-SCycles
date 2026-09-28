/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "kernel/closure/bsdf_diffraction_util.h"
#include "kernel/util/diffraction_grid.h"
#include "scene/diffraction.h"

#include <cstdio>
#include <limits>
#include <random>
#include <vector>

int main(int argc, const char **argv)
{
  if (argc != 2) {
    fprintf(stderr, "Usage: cycles_diffraction_metal expanded.metal\n");
    return 2;
  }
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "No Metal device; this test must run outside the sandbox.\n");
      return 2;
    }
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
                                                 encoding:NSUTF8StringEncoding
                                                    error:&error];
    if (!source) {
      fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 2;
    }
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeFast;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
      fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 2;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"diffraction_math"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                 error:&error];
    if (!pipeline) {
      fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 2;
    }
    constexpr unsigned count = 16384;
    constexpr size_t bytes = count * 8 * sizeof(float);
    std::vector<float> input(count * 8), expected(count * 24);
    std::mt19937 rng(736192);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (unsigned i = 0; i < count; ++i) {
      float *a = &input[8 * i];
      const ccl::float3 wi = ccl::normalize(ccl::make_float3(
          2.0f * uniform(rng) - 1.0f, 2.0f * uniform(rng) - 1.0f, 0.01f + uniform(rng)));
      a[0] = wi.x;
      a[1] = wi.y;
      a[2] = wi.z;
      a[3] = 0.1f + uniform(rng);
      a[4] = 0.5f + 1.5f * uniform(rng);
      a[5] = 20.0f * uniform(rng);
      a[6] = uniform(rng);
      a[7] = uniform(rng);
      const int order = int(i % 17) - 8;
      ccl::float3 wo;
      const bool valid = ccl::diffraction_order_direction(wi, a[3], order, a[4], i % 2, &wo);
      const ccl::float2 amp = ccl::diffraction_binary_amplitude(order, a[5], a[6], a[7]);
      float *e = &expected[24 * i];
      e[0] = wo.x;
      e[1] = wo.y;
      e[2] = wo.z;
      e[3] = valid ? 1.0f : 0.0f;
      e[4] = amp.x;
      e[5] = amp.y;
      e[6] = ccl::diffraction_binary_power(order, a[5], a[6]);
      e[7] = ccl::diffraction_relief_phase(125.0f, 550.0f, 1.0f, a[4], wi.z, wo.z);
      const ccl::float3 h = ccl::normalize(
          ccl::make_float3(2.0f * a[6] - 1.0f, 2.0f * a[7] - 1.0f, 0.1f + a[4]));
      const ccl::float3 axis = ccl::make_float3(1.0f, 0.0f, 0.0f);
      const float delta = (int(i % 5) - 2) * a[3];
      const bool reflected = ccl::diffraction_facet_reflect(wi, h, axis, delta, &wo);
      e[8] = wo.x;
      e[9] = wo.y;
      e[10] = wo.z;
      e[11] = reflected ? 1.0f : 0.0f;
      if (reflected) {
        e[12] = ccl::diffraction_reflection_jacobian(wi, wo, h, axis, delta);
        e[13] = ccl::diffraction_reflection_jacobian(wo, wi, h, axis, delta);
        ccl::DiffractionReflectionRoot roots[2];
        const int roots_count = ccl::diffraction_reflection_half_vectors(
            wi, wo, axis, delta, roots);
        e[14] = float(roots_count);
        e[15] = 2.0f;
        for (int r = 0; r < roots_count; r++) {
          e[15] = fminf(e[15], ccl::len(roots[r].h - h));
        }
      }
      const bool transmitted = ccl::diffraction_facet_transmit(wi, h, axis, a[4], delta, &wo);
      e[16] = wo.x; e[17] = wo.y; e[18] = wo.z; e[19] = transmitted;
      if (transmitted) {
        e[20] = ccl::diffraction_transmission_jacobian(wi, wo, h, axis, a[4], delta);
        ccl::DiffractionTransmissionRoot roots[2];
        const int n = ccl::diffraction_transmission_half_vectors(wi, wo, axis, a[4], delta, roots);
        e[22] = float(n); e[23] = 2.0f;
        for (int r = 0; r < n; r++) e[23] = fminf(e[23], ccl::len(roots[r].h - h));
      }

    }
    id<MTLBuffer> inputs = [device newBufferWithBytes:input.data()
                                               length:bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputs = [device newBufferWithLength:3 * bytes
                                                options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputs offset:0 atIndex:0];
    [encoder setBuffer:outputs offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
      fprintf(stderr, "GPU failure: %s\n", command.error.localizedDescription.UTF8String);
      return 2;
    }
    const float *actual = static_cast<const float *>(outputs.contents);
    float max_error = 0.0f;
    unsigned failures = 0;
    float max_geometry_relative_error = 0.0f;
    for (size_t j = 0; j < expected.size(); j++) {
      const float difference = fabsf(actual[j] - expected[j]);
      const size_t component = j % 24;
      const bool geometry = component >= 8;
      const float relative_error = difference / fmaxf(1.0f, fabsf(expected[j]));
      if (geometry) {
        max_geometry_relative_error = fmaxf(max_geometry_relative_error, relative_error);
      }
      else {
        max_error = fmaxf(max_error, difference);
      }
      if (!std::isfinite(actual[j]) || (geometry ? relative_error > 2e-4f : difference > 1e-5f) ||
          ((component == 3 || component == 11 || component == 14 || component == 19 || component == 22) && actual[j] != expected[j]))
      {
        if (failures < 8) {
          fprintf(stderr, "Mismatch %zu: CPU %.9g Metal %.9g\n", j, expected[j], actual[j]);
          const size_t base = 24 * (j / 24);
          const size_t offset = component >= 16 ? 16 : 8;
          const float *query = input.data() + 8 * (j / 24);
          fprintf(stderr, "  query=%zu component=%zu input", j / 24, component);
          for (int k = 0; k < 8; ++k) fprintf(stderr, " %.9g", query[k]);
          fprintf(stderr, "\n");
          fprintf(stderr,
                  "  forward %.9g %.9g; wo %.9g %.9g %.9g vs %.9g %.9g %.9g\n",
                  expected[base + offset + 4],
                  actual[base + offset + 4],
                  expected[base + offset],
                  expected[base + offset + 1],
                  expected[base + offset + 2],
                  actual[base + offset],
                  actual[base + offset + 1],
                  actual[base + offset + 2]);
        }
        failures++;
      }
    }
    /* Deliberately asymmetric, exactly representable values detect index and
     * side swaps. This checks packing/addressing, not electromagnetic physics. */
    constexpr int upper_first = -2, upper_count = 5, lower_first = -4, lower_count = 8;
    constexpr int port_count = upper_count + lower_count;
    std::vector<float> table(4 + port_count * port_count);
    table[0] = upper_first;
    table[1] = upper_count;
    table[2] = lower_first;
    table[3] = lower_count;
    for (int row = 0; row < port_count; row++) {
      for (int col = 0; col < port_count; col++) {
        table[4 + row * port_count + col] = float(row * port_count + col + 1) / 4096.0f;
      }
    }
    std::vector<int> ports(4 * count);
    std::vector<float> table_expected(count);
    for (unsigned i = 0; i < count; i++) {
      const int a = i % 127 == 0 ? std::numeric_limits<int>::max() : int(rng() % 21) - 10;
      const int b = i % 131 == 0 ? std::numeric_limits<int>::min() : int(rng() % 21) - 10;
      const bool a_lower = i & 1, b_lower = i & 2;
      ports[4 * i] = a;
      ports[4 * i + 1] = a_lower;
      ports[4 * i + 2] = b;
      ports[4 * i + 3] = b_lower;
      const int af = a_lower ? lower_first : upper_first, ac = a_lower ? lower_count : upper_count;
      const int bf = b_lower ? lower_first : upper_first, bc = b_lower ? lower_count : upper_count;
      if (a >= af && a < af + ac && b >= bf && b < bf + bc) {
        table_expected[i] = table[4 + (b - bf + (b_lower ? upper_count : 0)) * port_count + a -
                                  af + (a_lower ? upper_count : 0)];
      }
    }
    id<MTLFunction> table_function = [library newFunctionWithName:@"diffraction_table_test"];
    id<MTLComputePipelineState> table_pipeline = [device
        newComputePipelineStateWithFunction:table_function
                                      error:&error];
    if (!table_pipeline) {
      fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 2;
    }
    id<MTLBuffer> table_buffer = [device newBufferWithBytes:table.data()
                                                     length:table.size() * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> ports_buffer = [device newBufferWithBytes:ports.data()
                                                     length:ports.size() * sizeof(int)
                                                    options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> table_command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> table_encoder = [table_command computeCommandEncoder];
    [table_encoder setComputePipelineState:table_pipeline];
    [table_encoder setBuffer:table_buffer offset:0 atIndex:0];
    [table_encoder setBuffer:ports_buffer offset:0 atIndex:1];
    [table_encoder setBuffer:outputs offset:0 atIndex:2];
    [table_encoder dispatchThreads:MTLSizeMake(count, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(table_pipeline.threadExecutionWidth, 1, 1)];
    [table_encoder endEncoding];
    [table_command commit];
    [table_command waitUntilCompleted];
    if (table_command.status != MTLCommandBufferStatusCompleted) {
      fprintf(
          stderr, "GPU table failure: %s\n", table_command.error.localizedDescription.UTF8String);
      return 2;
    }
    unsigned table_failures = 0;
    for (unsigned i = 0; i < count; i++) {
      table_failures += actual[i] != table_expected[i];
    }
    failures += table_failures;
    const ccl::DiffractionGratingProfile profile{740.0, 150.0, 0.41, 1.0, 1.5, 1.0, 1.5};
    const ccl::DiffractionGratingGridConfig config{8, 8, 3, 500.0, 600.0, true};
    std::vector<float> grid, queries(4 * count), grid_expected(2 * count);
    std::string grid_error;
    if (!ccl::diffraction_grating_build_grid(profile, config, 8, grid, grid_error)) {
      fprintf(stderr, "%s\n", grid_error.c_str());
      return 2;
    }
    for (unsigned i = 0; i < count; i++) {
      float *q = &queries[4 * i];
      q[0] = 490.0f + uniform(rng) * 120.0f;
      q[1] = 2.8f * uniform(rng) - 1.4f;
      q[2] = 2.8f * uniform(rng) - 1.4f;
      ports[4 * i] = int(rng() % 9) - 4;
      const bool valid = ccl::diffraction_grid_power(grid.data(),
                                                     q[0],
                                                     q[1],
                                                     q[2],
                                                     ports[4 * i + 1] != 0,
                                                     ports[4 * i],
                                                     ports[4 * i + 3] != 0,
                                                     &grid_expected[2 * i]);
      grid_expected[2 * i + 1] = valid ? 1.0f : 0.0f;
    }
    id<MTLFunction> grid_function = [library newFunctionWithName:@"diffraction_grid_test"];
    id<MTLComputePipelineState> grid_pipeline = [device
        newComputePipelineStateWithFunction:grid_function
                                      error:&error];
    if (!grid_pipeline) {
      fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 2;
    }
    id<MTLBuffer> grid_buffer = [device newBufferWithBytes:grid.data()
                                                    length:grid.size() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> query_buffer = [device newBufferWithBytes:queries.data()
                                                     length:queries.size() * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    memcpy(ports_buffer.contents, ports.data(), ports.size() * sizeof(int));
    id<MTLCommandBuffer> grid_command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> grid_encoder = [grid_command computeCommandEncoder];
    [grid_encoder setComputePipelineState:grid_pipeline];
    [grid_encoder setBuffer:grid_buffer offset:0 atIndex:0];
    [grid_encoder setBuffer:query_buffer offset:0 atIndex:1];
    [grid_encoder setBuffer:ports_buffer offset:0 atIndex:2];
    [grid_encoder setBuffer:outputs offset:0 atIndex:3];
    [grid_encoder dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(grid_pipeline.threadExecutionWidth, 1, 1)];
    [grid_encoder endEncoding];
    [grid_command commit];
    [grid_command waitUntilCompleted];
    if (grid_command.status != MTLCommandBufferStatusCompleted) {
      fprintf(
          stderr, "GPU grid failure: %s\n", grid_command.error.localizedDescription.UTF8String);
      return 2;
    }
    unsigned grid_failures = 0;
    float grid_max_error = 0.0f;
    for (unsigned i = 0; i < 2 * count; i++) {
      const float difference = fabsf(actual[i] - grid_expected[i]);
      grid_max_error = fmaxf(grid_max_error, difference);
      if (!std::isfinite(actual[i]) || difference > 2e-6f ||
          (i % 2 && actual[i] != grid_expected[i]))
      {
        if (grid_failures < 8) {
          fprintf(
              stderr, "Grid mismatch %u: CPU %.9g Metal %.9g\n", i, grid_expected[i], actual[i]);
        }
        grid_failures++;
      }
    }
    failures += grid_failures;
    printf(
        "{\"device\":\"%s\",\"cases\":%u,\"max_absolute_error\":%.9g,\"geometry_max_scaled_"
        "error\":%.9g,\"table_cases\":%u,\"table_failures\":%u,"
        "\"grid_cases\":%u,\"grid_failures\":%u,\"grid_max_error\":%.9g,\"failures\":%u}\n",
        device.name.UTF8String,
        count,
        max_error,
        max_geometry_relative_error,
        count,
        table_failures,
        count,
        grid_failures,
        grid_max_error,
        failures);
    return failures ? 1 : 0;
  }
}
