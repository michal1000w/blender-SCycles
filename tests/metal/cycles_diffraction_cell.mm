/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction.h"
#include "util/math.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>

#ifndef DIFFRACTION_INPLACE_CHART
#  define DIFFRACTION_INPLACE_CHART 0
#endif

static bool run(id<MTLDevice> device,
                id<MTLLibrary> library,
                double pitch,
                std::complex<double> material,
                unsigned evaluations,
                NSMutableArray *reports)
{
  using namespace ccl;
  const double center = pitch > 1000 ? pitch / 4 : pitch;
  const DiffractionGratingCellBounds bounds{{-0.03, -0.04, center - 5}, {0.03, 0.04, center + 5}};
  const DiffractionGratingProfile profile{pitch, 150, 0.41, 1, material, 1, material};
  DiffractionGratingCell cell;
  DiffractionGratingPackedCell packed;
  std::string error;
  DiffractionGratingChartCell chart_cell;
  if (!diffraction_grating_prepare_cell(profile, bounds, 16, 5, 0.1, cell, error)) {
    fprintf(stderr, "%s\n", error.c_str());
    return false;
  }
#ifdef DIFFRACTION_CHART_CELL
#  ifdef DIFFRACTION_QUADRATIC_CELL
  const bool prepared = diffraction_grating_prepare_quadratic_chart(
      profile, cell, 16, 5, chart_cell, error);
#  else
  const bool prepared = diffraction_grating_prepare_chart_cell(cell, chart_cell, error);
#  endif
  if (!prepared || !diffraction_grating_pack_chart_cell(chart_cell, packed, error)) {
#else
  if (!diffraction_grating_pack_cell(cell, packed, error)) {
#endif
    fprintf(stderr, "%s\n", error.c_str());
    return false;
  }
  const int ports = packed.ports.size(), channels = 2 * ports;
  if (channels > 32 || cell.feedback_channels > 12)
    return false;
  std::vector<float4> inputs, queries;
  std::vector<float> expected;
  std::vector<float2> expected_complex;
  auto find_port = [&](const DiffractionGratingPort &target) {
    for (int j = 0; j < ports; j++)
      if (packed.ports[j].order == target.order && packed.ports[j].substrate == target.substrate)
        return j;
    return -1;
  };
  for (const float3 t : {make_float3(0, 0, 0),
                         make_float3(1, 1, 1),
                         make_float3(0.5f, 0.5f, 0.49f),
                         make_float3(0.5f, 0.5f, 0.5f),
                         make_float3(0.5f, 0.5f, 0.51f),
                         make_float3(0.17f, 0.73f, 0.37f)})
  {
    const double wavelength = bounds.lower[2] + t.z * 10;
    const double bloch = bounds.lower[0] + t.x * 0.06;
    const double y = bounds.lower[1] + t.y * 0.08;
    DiffractionGratingPowerBlock power;
    if (!diffraction_grating_cell_power(profile, cell, {bloch, y, wavelength}, power, error))
      return false;
    for (int col = 0; col < int(power.ports.size()); col++) {
      const auto &incoming = power.ports[col];
      const int input = find_port(incoming);
      if (input < 0)
        return false;
      queries.push_back(make_float4(t.x, t.y, t.z, float(input)));
      const double ni = incoming.substrate ? material.real() : 1;
      const double x = (bloch + incoming.order) * wavelength / pitch;
      const double z = std::sqrt(std::max(0.0, ni * ni - x * x - y * y));
      for (const auto &out : packed.ports) {
        inputs.push_back(make_float4(float(x / ni), float(y / ni), float(z / ni), float(ni)));
        inputs.push_back(make_float4(float(out.substrate ? material.real() : 1),
                                     float(wavelength),
                                     float(pitch),
                                     float(out.order - incoming.order)));
      }
      /* Match at exactly the quantized ray used by Metal. At a Rayleigh cutoff,
       * rounding a direction can open an order; comparing with the unquantized
       * ray would measure a different physical query. Corner weights are supplied
       * explicitly to this kernel and are therefore kept identical here. */
      const float4 ray = inputs[inputs.size() - 2 * ports];
      const double norm = std::sqrt(double(ray.x) * ray.x + double(ray.y) * ray.y +
                                    double(ray.z) * ray.z);
      const double wl = float(wavelength);
      const double actual_x = ni * double(ray.x) / norm - incoming.order * wl / pitch;
      const double actual_y = ni * double(ray.y) / norm;
      DiffractionGratingHybrid interpolated;
      if (packed.operator_chart) {
        auto blend = chart_cell.corners[0];
        std::fill(blend.matrix.begin(), blend.matrix.end(), std::complex<double>(0));
        for (int c = 0; c < int(chart_cell.corners.size()); c++) {
          const double coordinates[3] = {t.x, t.y, t.z};
          int index = c;
          double weight = 1;
          for (int axis = 0; axis < 3; axis++) {
            const double u = coordinates[axis];
            const int digit = index % (chart_cell.degree + 1);
            index /= chart_cell.degree + 1;
            weight *= chart_cell.degree == 1 ? (digit ? u : 1 - u) :
                      digit == 0             ? (1 - u) * (1 - u) :
                      digit == 1             ? 2 * u * (1 - u) :
                                               u * u;
          }
          for (size_t j = 0; j < blend.matrix.size(); j++)
            blend.matrix[j] += weight * chart_cell.corners[c].matrix[j];
        }
        interpolated.is_reference = chart_cell.is_reference;
        if (!diffraction_grating_chart_to_reference(blend, interpolated.scattering, error))
          return false;
      }
      std::vector<double> accumulated(ports, 0);
      std::vector<float2> complex_column(4 * ports, zero_float2());
      for (int corner = 0; corner < (packed.operator_chart ? 1 : 8); corner++) {
        const double weight = packed.operator_chart ? 1.0 :
                                                      (corner & 1 ? double(t.x) : 1.0 - t.x) *
                                                          (corner & 2 ? double(t.y) : 1.0 - t.y) *
                                                          (corner & 4 ? double(t.z) : 1.0 - t.z);
        if (weight == 0)
          continue;
        DiffractionGratingBlock matched;
        if (!diffraction_grating_match_hybrid(profile,
                                              wl,
                                              actual_x,
                                              actual_y,
                                              packed.operator_chart ? interpolated :
                                                                      cell.corners[corner],
                                              matched,
                                              error))
        {
          fprintf(stderr, "%s\n", error.c_str());
          return false;
        }
        DiffractionGratingPowerBlock matched_power;
        diffraction_grating_power_block(matched, matched_power);
        int actual_input = -1;
        for (int j = 0; j < int(matched_power.ports.size()); j++) {
          const auto &candidate = matched_power.ports[j];
          if (candidate.order == incoming.order && candidate.substrate == incoming.substrate)
            actual_input = j;
        }
        if (actual_input < 0)
          return false;
        for (int row = 0; row < int(matched_power.ports.size()); row++) {
          const int output = find_port(matched_power.ports[row]);
          if (output < 0)
            return false;
          if (packed.operator_chart) {
            const size_t n = matched.ports.size();
            for (int a = 0; a < 2; a++) {
              for (int b = 0; b < 2; b++) {
                const auto value = matched.matrix[(2 * row + a) * (2 * n) + 2 * actual_input + b];
                complex_column[4 * output + 2 * a + b] = make_float2(float(value.real()),
                                                                     float(value.imag()));
              }
            }
          }
          accumulated[output] +=
              weight * matched_power.matrix[row * matched_power.ports.size() + actual_input];
        }
      }
      std::vector<float> column(accumulated.begin(), accumulated.end());
      expected.insert(expected.end(), column.begin(), column.end());
      expected_complex.insert(
          expected_complex.end(), complex_column.begin(), complex_column.end());
    }
  }
  const unsigned cases = queries.size();
  NSError *failure = nil;
  id<MTLComputePipelineState> pipeline = [device
      newComputePipelineStateWithFunction:
          [library
              newFunctionWithName:packed.operator_chart ?
                                      [NSString
                                          stringWithFormat:@"diffraction_chart_cell_%d_%d_%d_%d",
                                                           channels,
                                                           cell.feedback_channels,
                                                           packed.chart_degree,
                                                           DIFFRACTION_INPLACE_CHART] :
                                      [NSString stringWithFormat:@"diffraction_cell_%d",
                                                                 cell.feedback_channels]]
                                    error:&failure];
  if (!pipeline) {
    fprintf(stderr, "%s\n", failure.localizedDescription.UTF8String);
    return false;
  }
  auto buffer = [&](const void *data, size_t length) {
    return [device newBufferWithBytes:data length:length options:MTLResourceStorageModeShared];
  };
  const int dummy = 0;
  id<MTLBuffer> matrices = buffer(packed.matrices.data(), packed.matrices.size() * sizeof(float2));
  id<MTLBuffer> raw = buffer(inputs.data(), inputs.size() * sizeof(float4));
  id<MTLBuffer> active = buffer(packed.active_ports.empty() ? &dummy : packed.active_ports.data(),
                                std::max(size_t(1), packed.active_ports.size()) * sizeof(int));
  id<MTLBuffer> points = buffer(queries.data(), queries.size() * sizeof(float4));
  id<MTLBuffer> outputs = [device newBufferWithLength:evaluations * ports * sizeof(float)
                                              options:MTLResourceStorageModeShared];
  id<MTLBuffer> valid = [device newBufferWithLength:evaluations * sizeof(unsigned)
                                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> amplitudes = [device newBufferWithLength:expected_complex.size() * sizeof(float2)
                                                 options:MTLResourceStorageModeShared];
  const int2 shape = make_int2(channels, cases);
  id<MTLCommandQueue> queue = [device newCommandQueue];
  NSMutableArray *times = [NSMutableArray array];
  unsigned failures = 0;
  double maximum = 0, energy_error = 0, complex_error = 0;
  for (int run = 0; run < 5; run++) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:matrices offset:0 atIndex:0];
    [encoder setBuffer:raw offset:0 atIndex:1];
    [encoder setBuffer:active offset:0 atIndex:2];
    [encoder setBuffer:points offset:0 atIndex:3];
    [encoder setBuffer:outputs offset:0 atIndex:4];
    [encoder setBuffer:valid offset:0 atIndex:5];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:6];
    if (packed.operator_chart) {
      [encoder setBytes:&packed.chart_rotation length:sizeof(float2) atIndex:7];
      [encoder setBuffer:amplitudes offset:0 atIndex:8];
    }
    [encoder dispatchThreads:MTLSizeMake(evaluations, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
      return false;
    if (run >= 2)
      [times addObject:@(1000 * (command.GPUEndTime - command.GPUStartTime))];
    if (packed.operator_chart) {
      const auto *actual = static_cast<const float2 *>(amplitudes.contents);
      for (size_t j = 0; j < expected_complex.size(); j++) {
        const double difference = std::hypot(double(actual[j].x) - expected_complex[j].x,
                                             double(actual[j].y) - expected_complex[j].y);
        complex_error = std::max(complex_error, difference);
        if ((!std::isfinite(difference) || difference > 3e-5) && failures++ < 8)
          fprintf(stderr, "Chart amplitude mismatch %zu: %.9g\n", j, difference);
      }
    }
    const auto *values = static_cast<const float *>(outputs.contents);
    const auto *flags = static_cast<const unsigned *>(valid.contents);
    for (unsigned i = 0; i < evaluations; i++) {
      bool bad = !flags[i];
      double sum = 0;
      for (int p = 0; p < ports; p++) {
        const float value = values[i * ports + p];
        const double diff = std::abs(double(value) - expected[(i % cases) * ports + p]);
        maximum = std::max(maximum, diff);
        bad |= !std::isfinite(value) || value < 0 || diff > 3e-5;
        sum += value;
      }
      if (material.imag() == 0)
        energy_error = std::max(energy_error, std::abs(sum - 1));
      bad |= sum > 1 + 3e-5 || (material.imag() == 0 && std::abs(sum - 1) > 3e-5);
      if (bad && failures++ < 8)
        fprintf(stderr, "Cell mismatch pitch %.0f case %u energy %.9g\n", pitch, i % cases, sum);
    }
  }
  [reports addObject:@{
    @"pitch_nm" : @(pitch),
    @"operator_chart" : @(packed.operator_chart),
    @"chart_degree" : @(packed.chart_degree),
    @"inplace_chart" : @(DIFFRACTION_INPLACE_CHART),
    @"lossless" : @(material.imag() == 0),
    @"cases" : @(cases),
    @"channels" : @(channels),
    @"feedback_channels" : @(cell.feedback_channels),
    @"evaluations" : @(evaluations),
    @"gpu_ms" : times,
    @"max_power_error" : @(maximum),
    @"complex_coefficients_checked_per_run" :
        @(packed.operator_chart ? expected_complex.size() : 0),
    @"max_complex_error" : @(complex_error),
    @"max_lossless_energy_error" : @(energy_error),
    @"failures" : @(failures)
  }];
  return failures == 0;
}

static bool check_lookup(id<MTLDevice> device,
                         id<MTLLibrary> library,
                         unsigned &case_count,
                         const bool mirror = false)
{
  using namespace ccl;
  const int4 nodes[] = {make_int4(0, 1, 2, __float_as_int(-0.125f)),
                        make_int4(-1, 1, 0, 0),
                        make_int4(1, 3, 4, __float_as_int(mirror ? -0.25f : 0.25f)),
                        make_int4(-1, 2, 0, 0),
                        make_int4(2, 5, 6, __float_as_int(550.0f)),
                        make_int4(-1, 3, 0, 0),
                        make_int4(-1, 4, 0, 0)};
  std::vector<float4> queries;
  std::vector<int> expected;
  auto append = [&](float x, float y, float z) {
    queries.push_back(make_float4(x, y, z, 0));
    const bool inside = std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && x >= -0.5f &&
                        x <= 0.5f && y >= -1 && y <= 1 && z >= 380 && z <= 780;
    expected.push_back(!inside     ? -1 :
                       mirror      ? (std::abs(x) > 0.125f ? 1 :
                                      std::abs(y) > 0.25f  ? 2 :
                                      z < 550              ? 3 :
                                                             4) :
                       x < -0.125f ? 1 :
                       y < 0.25f   ? 2 :
                       z < 550     ? 3 :
                                     4);
  };
  std::mt19937 rng(197341);
  std::uniform_real_distribution<float> uniform(0, 1);
  for (int i = 0; i < 1024; i++)
    append(1.2f * uniform(rng) - 0.6f, 2.4f * uniform(rng) - 1.2f, 350 + 470 * uniform(rng));
  for (float x : {-0.5f, std::nextafter(-0.125f, -INFINITY), -0.125f, 0.5f, NAN})
    for (float y : {-1.0f, std::nextafter(0.25f, -INFINITY), 0.25f, 1.0f, INFINITY})
      for (float z : {380.0f, std::nextafter(550.0f, -INFINITY), 550.0f, 780.0f, -INFINITY})
        append(x, y, z);
  case_count = queries.size();
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline = [device
      newComputePipelineStateWithFunction:[library newFunctionWithName:@"diffraction_lookup"]
                                    error:&error];
  if (!pipeline) {
    fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
    return false;
  }
  id<MTLBuffer> index = [device newBufferWithBytes:nodes
                                            length:sizeof(nodes)
                                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> points = [device newBufferWithBytes:queries.data()
                                             length:queries.size() * sizeof(float4)
                                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> outputs = [device newBufferWithLength:queries.size() * sizeof(int4)
                                              options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:index offset:0 atIndex:0];
  [encoder setBuffer:points offset:0 atIndex:1];
  [encoder setBuffer:outputs offset:0 atIndex:2];
  const unsigned use_mirror = mirror;
  [encoder setBytes:&use_mirror length:sizeof(use_mirror) atIndex:3];
  [encoder dispatchThreads:MTLSizeMake(case_count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    return false;
  const auto *actual = static_cast<const int4 *>(outputs.contents);
  for (unsigned i = 0; i < case_count; i++) {
    const int reverse = queries[i].x > 0;
    const int sign = (bool(reverse) != (queries[i].y > 0)) ? -1 : 1;
    if (actual[i].x != expected[i] || actual[i].y != reverse || actual[i].z != sign) {
      fprintf(stderr,
              "Lookup/mapping mismatch %u (mirror=%d): leaf %d != %d\n",
              i,
              int(mirror),
              actual[i].x,
              expected[i]);
      return false;
    }
  }
  return true;
}

static bool check_stored_cache(id<MTLDevice> device,
                               id<MTLLibrary> library,
                               NSMutableDictionary *report)
{
  using namespace ccl;
  using Complex = std::complex<double>;
  const DiffractionGratingProfile p{740, 150, 0.41, 1, Complex(0.9, 6), 1, Complex(0.9, 6)};
  DiffractionGratingCacheOptions settings;
  settings.bounds = {{-0.03125, -0.0625, 735}, {0.03125, 0.0625, 745}};
  settings.retained_half_orders = 1;
  settings.cutoff_margin = 100;
  settings.mirror_symmetry = true;
  settings.complex_tolerance = 0.001;
#ifdef DIFFRACTION_QUADRATIC_CELL
  settings.allow_quadratic_cells = true;
#endif
  settings.maximum_nodes = 1024;
  DiffractionGratingCache cache;
  DiffractionGratingCacheStats stats;
  std::string message;
  if (!diffraction_grating_build_cache(p, settings, cache, stats, message)) {
    fprintf(stderr, "Stored cache: %s\n", message.c_str());
    return false;
  }
  std::vector<float2> matrices, expected;
  std::vector<int2> layout;
  std::vector<float4> bounds, queries;
  for (const auto &cell : cache.cells) {
    if (!cell.operator_chart || cell.ports.size() != 3 || cell.active_ports.size() != 3 ||
        (cell.chart_degree != 1 && cell.chart_degree != 2) ||
        cell.matrices.size() != (cell.chart_degree == 1 ? 288 : 972))
      return false;
    for (int i = 0; i < 3; i++)
      if (cell.ports[i].order != i - 1 || cell.ports[i].substrate)
        return false;
    layout.push_back(make_int2(int(matrices.size()), cell.chart_degree));
    matrices.insert(matrices.end(), cell.matrices.begin(), cell.matrices.end());
    bounds.push_back(make_float4(
        cell.bounds.lower[0], cell.bounds.lower[1], cell.bounds.lower[2], cell.chart_rotation.x));
    bounds.push_back(make_float4(
        cell.bounds.upper[0], cell.bounds.upper[1], cell.bounds.upper[2], cell.chart_rotation.y));
  }
  for (int sample = 0; sample < 37; sample++) {
    const float4 q = make_float4(float(-0.03125 + 0.0625 * (sample + 0.37) / 37),
                                 float(-0.0625 + 0.125 * std::fmod(sample * 0.618 + 0.17, 1.0)),
                                 float(735 + 10 * std::fmod(sample * 0.414 + 0.29, 1.0)),
                                 0);
    DiffractionGratingBlock reference, exact;
    const double kx = double(q.x) * q.z / 740;
    if (!diffraction_grating_solve_reference(p, q.z, kx, q.y, 16, 1, reference, message) ||
        !diffraction_grating_match_reference(p, q.z, kx, q.y, reference, exact, message))
      return false;
    const int n = 2 * int(exact.ports.size());
    for (int in = 0; in < n / 2; in++) {
      queries.push_back(make_float4(q.x, q.y, q.z, float(exact.ports[in].order + 1)));
      float2 column[12] = {};
      for (int out = 0; out < n / 2; out++)
        for (int r = 0; r < 2; r++)
          for (int c = 0; c < 2; c++) {
            const Complex value = exact.matrix[(2 * out + r) * n + 2 * in + c];
            column[4 * (exact.ports[out].order + 1) + 2 * r + c] = make_float2(value.real(),
                                                                               value.imag());
          }
      expected.insert(expected.end(), column, column + 12);
    }
  }
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline = [device
      newComputePipelineStateWithFunction:[library newFunctionWithName:@"diffraction_stored_cache"]
                                    error:&error];
  if (!pipeline) {
    fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
    return false;
  }
  auto buffer = [&](const void *data, size_t size) {
    return [device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
  };
  id<MTLBuffer> layout_buffer = buffer(layout.data(), layout.size() * sizeof(int2));
  id<MTLBuffer> nodes = buffer(cache.nodes.data(), cache.nodes.size() * sizeof(int4));
  id<MTLBuffer> matrix_buffer = buffer(matrices.data(), matrices.size() * sizeof(float2));
  id<MTLBuffer> cell_bounds = buffer(bounds.data(), bounds.size() * sizeof(float4));
  id<MTLBuffer> points = buffer(queries.data(), queries.size() * sizeof(float4));
  id<MTLBuffer> output = [device newBufferWithLength:expected.size() * sizeof(float2)
                                             options:MTLResourceStorageModeShared];
  id<MTLBuffer> validity = [device newBufferWithLength:queries.size() * sizeof(unsigned)
                                               options:MTLResourceStorageModeShared];
  if (!layout_buffer || !nodes || !matrix_buffer || !cell_bounds || !points || !output ||
      !validity)
    return false;
  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:nodes offset:0 atIndex:0];
  [encoder setBuffer:matrix_buffer offset:0 atIndex:1];
  [encoder setBuffer:cell_bounds offset:0 atIndex:2];
  [encoder setBuffer:points offset:0 atIndex:3];
  [encoder setBuffer:output offset:0 atIndex:4];
  [encoder setBuffer:validity offset:0 atIndex:5];
  const unsigned count = cache.nodes.size();
  [encoder setBytes:&count length:sizeof(count) atIndex:6];
  [encoder setBuffer:layout_buffer offset:0 atIndex:7];
  [encoder dispatchThreads:MTLSizeMake(queries.size(), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    return false;
  const auto *actual = static_cast<const float2 *>(output.contents);
  const auto *valid = static_cast<const unsigned *>(validity.contents);
  double maximum = 0;
  for (size_t i = 0; i < queries.size(); i++) {
    if (!valid[i])
      return false;
    double norm = 0;
    for (int j = 0; j < 12; j++) {
      const size_t k = 12 * i + j;
      const double e = std::hypot(double(actual[k].x) - expected[k].x,
                                  double(actual[k].y) - expected[k].y);
      if (!std::isfinite(e))
        return false;
      norm = std::hypot(norm, e);
    }
    maximum = std::max(maximum, norm);
  }
  report[@"queries"] = @(queries.size());
  report[@"cells"] = @(cache.cells.size());
  report[@"quadratic_cells"] = @(stats.accepted_quadratic_cells);
  report[@"maximum_complex_column_error"] = @(maximum);
  report[@"held_out_threshold"] = @0.002;
  report[@"scope"] =
      @"Complete small cache; GPU lookup, boundary construction, float chart matching and "
      @"symmetry unfolding against same-modal-count direct Maxwell solves. Not a rendering "
      @"benchmark.";
  return maximum < 0.002;
}

int main(int argc, const char **argv)
{
  if (argc != 3)
    return 2;
  const unsigned evaluations = std::atoi(argv[2]);
  if (evaluations < 1024 || evaluations > 4194304)
    return 2;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device)
      return 2;
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
                                                 encoding:NSUTF8StringEncoding
                                                    error:&error];
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeFast;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
      fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 2;
    }
    NSMutableArray *reports = [NSMutableArray array];
    NSMutableDictionary *stored_cache = [NSMutableDictionary dictionary];
    unsigned lookup_cases = 0, mirror_lookup_cases = 0;
    bool ok = check_stored_cache(device, library, stored_cache);
    ok = check_lookup(device, library, lookup_cases) && ok;
    ok = check_lookup(device, library, mirror_lookup_cases, true) && ok;
    ok = run(device, library, 740, {0.9, 6}, evaluations, reports) && ok;
    ok = run(device, library, 1600, {0.9, 6}, evaluations, reports) && ok;
    ok = run(device, library, 740, {1.5, 0}, evaluations, reports) && ok;
    NSDictionary *report = @{
      @"device" : device.name,
      @"passed" : @(ok),
      @"cases" : reports,
      @"stored_cache" : stored_cache,
      @"lookup_cases" : @(lookup_cases),
      @"mirror_lookup_cases" : @(mirror_lookup_cases),
      @"scope" :
          @"Packed intensity or polynomial chart cells and GPU boundary construction against "
          @"double matching at identical quantized rays and supplied weights. Lookup and small "
          @"stored-cache checks are reported separately. "
          @"Hot-cell timings exclude cache search, interpolation accuracy "
          @"against direct RCWA, and rendering."
    };
    NSData *json = [NSJSONSerialization dataWithJSONObject:report options:0 error:&error];
    if (!json)
      return 2;
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
    return ok ? 0 : 1;
  }
}
