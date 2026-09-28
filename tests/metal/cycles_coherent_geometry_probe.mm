/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <simd/simd.h>

#include <cmath>
#include <cstdio>

static double mirror_range(const simd_float4 source, const simd_float4 receiver)
{
  const double ux = 0.9323273301f, uz = -0.3616154492f;
  const double nx = -uz, nz = ux;
  const double norm = std::hypot(nx, nz);
  const double x = nx / norm, z = nz / norm;
  const double height = (double(source.x) - 1000.0) * x +
                        (double(source.z) - 350.0) * z;
  const double dx = double(receiver.x) - (double(source.x) - 2.0 * height * x);
  const double dy = double(receiver.y) - double(source.y);
  const double dz = double(receiver.z) - (double(source.z) - 2.0 * height * z);
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

struct MixedReference {
  double optical_length;
  double spreading;
  double mirror_x;
};

/* Independent double Snell bisection after unfolding the interior mirror. */
static MixedReference mixed_reference(const double source_x,
                                      const double source_y,
                                      const double receiver_x,
                                      const double receiver_y,
                                      const double receiver_z,
                                      const double mirror_y)
{
  const double entry_x = double(0.015000000596046448f);
  const double exit_x = double(0.030000001192092896f);
  const double a = entry_x - source_x;
  const double t = exit_x - entry_x;
  const double b = receiver_x - exit_x;
  const double dy = 2.0 * mirror_y - receiver_y - source_y;
  const double rho = std::hypot(dy, receiver_z);
  double lo = 0.0, hi = 1.0 - 1.0e-15;
  for (int iteration = 0; iteration < 64; iteration++) {
    const double u = 0.5 * (lo + hi);
    const double ca = std::sqrt(1.0 - u * u);
    const double sg = u / 1.5;
    const double cg = std::sqrt(1.0 - sg * sg);
    const double displacement = (a + b) * u / ca + t * sg / cg;
    if (displacement < rho) lo = u;
    else hi = u;
  }
  const double u = 0.5 * (lo + hi);
  const double ca = std::sqrt(1.0 - u * u);
  const double sg = u / 1.5;
  const double cg = std::sqrt(1.0 - sg * sg);
  const double entry_y = source_y + a * u / ca * dy / rho;
  const double unfolded_exit_y = entry_y + t * sg / cg * dy / rho;
  const double dr_du = (a + b) / (ca * ca * ca) + t / (1.5 * cg * cg * cg);
  return {(a + b) / ca + 1.5 * t / cg,
          u / (rho * ca * dr_du),
          entry_x + t * (mirror_y - entry_y) / (unfolded_exit_y - entry_y)};
}

int main(int argc, const char **argv)
{
  if (argc != 2) return 2;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::fprintf(stderr, "No Metal device\n");
      return 2;
    }
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])]
                                               error:&error];
    if (!library) {
      std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 2;
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"coherent_geometry_probe"]
                                               error:&error];
    if (!pipeline) {
      std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 2;
    }
    id<MTLBuffer> output = [device newBufferWithLength:187 * sizeof(simd_float4)
                                               options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(103, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
      std::fprintf(stderr, "%s\n", command.error.localizedDescription.UTF8String);
      return 2;
    }
    const simd_float4 *value = static_cast<const simd_float4 *>(output.contents);
    const double expected_opd = mirror_range(value[3], value[5]) -
                                mirror_range(value[4], value[5]);
    const double opd_error = std::abs(double(value[0].y) - expected_opd);
    const double mirror_length = std::sqrt(1.2 * 1.2 + 0.29 * 0.29 + 3.0 * 3.0);
    const double mirror_spreading = 3.0 / (mirror_length * mirror_length * mirror_length);
    const double mirror_jac_error = std::abs(double(value[1].z) - mirror_spreading);
    const double slab_jac_error = std::abs(double(value[2].z) - 0.140625);
    int grid_solved = 0;
    double maximum_grid_opd_error = 0.0;
    double maximum_grid_jac_error = 0.0;
    for (int grid = 0; grid < 81; grid++) {
      const simd_float4 receiver = value[6 + 2 * grid];
      const simd_float4 measured = value[7 + 2 * grid];
      if (measured.x == 1.0f && measured.y == 1.0f) grid_solved++;
      const double ax = double(receiver.x) + 25.0e-6;
      const double bx = double(receiver.x) - 25.0e-6;
      const double y = double(receiver.y);
      const double ra = std::sqrt(ax * ax + y * y + 9.0);
      const double rb = std::sqrt(bx * bx + y * y + 9.0);
      const double expected = ra - rb;
      maximum_grid_opd_error = std::fmax(maximum_grid_opd_error,
                                          std::abs(double(receiver.w) - expected));
      const double expected_jac = 3.0 / (ra * ra * ra);
      maximum_grid_jac_error = std::fmax(maximum_grid_jac_error,
                                          std::abs(double(measured.z) - expected_jac));
    }
    int mixed_solved = 0;
    double mixed_opl_error = 0.0;
    double mixed_spreading_relative_error = 0.0;
    double mixed_mirror_x_error = 0.0;
    for (int grid = 0; grid < 19; grid++) {
      const bool unequal = grid == 18;
      const double sy = grid < 9 ? double(-25.0e-6f) : double(25.0e-6f);
      const double ry = unequal ? 0.0 : double((grid / 3) % 3 - 1) * double(18.0e-6f);
      const double rz = unequal ? 0.0 : double(grid % 3 - 1) * double(18.0e-6f);
      const double source_x = unequal ? double(0.0149f) : 0.0;
      const double receiver_x = unequal ? double(0.0305f) : double(0.05000000074505806f);
      const double mirror_y = unequal ? double(0.01f) : double(0.019999999552965164f);
      const MixedReference expected = mixed_reference(
          source_x, unequal ? double(-25.0e-6f) : sy,
          receiver_x, ry, rz, mirror_y);
      const simd_float4 measured = value[168 + grid];
      if (measured.x == 1.0f) mixed_solved++;
      mixed_opl_error = std::fmax(mixed_opl_error,
                                 std::abs(double(measured.y) - expected.optical_length));
      mixed_spreading_relative_error = std::fmax(
          mixed_spreading_relative_error,
          std::abs(double(measured.z) / expected.spreading - 1.0));
      mixed_mirror_x_error = std::fmax(mixed_mirror_x_error,
                                      std::abs(double(measured.w) - expected.mirror_x));
    }
    const bool success = value[0].x == 1.0f && value[0].w == 1.0f &&
                         value[1].x == 1.0f && value[2].x == 1.0f &&
                         opd_error < 2e-8 && mirror_jac_error < 2e-6 && slab_jac_error < 2e-6 &&
                         grid_solved == 81 && maximum_grid_opd_error < 2e-10 &&
                         maximum_grid_jac_error < 2e-6 && mixed_solved == 19 &&
                         mixed_opl_error < 2e-8 &&
                         mixed_spreading_relative_error < 1e-3 &&
                         mixed_mirror_x_error < 2e-6;
    std::printf("device=%s solves=%d,%d,%d,%d opd=%.12g expected=%.12g opd_error=%.9g "
                "mirror_jac=%.9g mirror_jac_error=%.9g slab_jac=%.9g slab_jac_error=%.9g "
                "grid_solved=%d/81 grid_opd_error=%.9g grid_jac_error=%.9g "
                "mixed_solved=%d/19 mixed_opl_error=%.9g mixed_spreading_rel_error=%.9g "
                "mixed_mirror_x_error=%.9g success=%d\n",
                device.name.UTF8String,
                int(value[0].x),
                int(value[0].w),
                int(value[1].x),
                int(value[2].x),
                value[0].y,
                expected_opd,
                opd_error,
                value[1].z,
                mirror_jac_error,
                value[2].z,
                slab_jac_error,
                grid_solved,
                maximum_grid_opd_error,
                maximum_grid_jac_error,
                mixed_solved,
                mixed_opl_error,
                mixed_spreading_relative_error,
                mixed_mirror_x_error,
                int(success));
    return success ? 0 : 1;
  }
}
