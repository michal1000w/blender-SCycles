/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <simd/simd.h>

#include <cmath>
#include <cstdio>
#include <cstring>

static bool close(const double value, const double expected, const double tolerance)
{
  return std::abs(value - expected) < tolerance;
}

static double transmission(const double ni, const double nt, const double ci, const bool p)
{
  const double ct = std::sqrt(1.0 - (ni / nt) * (ni / nt) * (1.0 - ci * ci));
  const double yi = p ? ni / ci : ni * ci;
  const double yt = p ? nt / ct : nt * ct;
  return 4.0 * yi * yt / ((yi + yt) * (yi + yt));
}

int main(int argc, const char **argv)
{
  if (argc != 2) return 2;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return 2;
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])]
                                               error:&error];
    if (!library) {
      std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 2;
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"coherent_path_field_probe"]
                                               error:&error];
    if (!pipeline) {
      std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 2;
    }
    id<MTLBuffer> output = [device newBufferWithLength:16 * sizeof(simd_float4)
                                               options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
      std::fprintf(stderr, "%s\n", command.error.localizedDescription.UTF8String);
      return 2;
    }
    const simd_float4 *v = static_cast<const simd_float4 *>(output.contents);
    constexpr double pi = 3.14159265358979323846;
    const double base = 1.0 / (4.0 * pi * pi);
    const double normal_t = transmission(1, 1.5, 1, false);
    const double rs = 1.0 - transmission(1, 1.5, std::sqrt(0.5), false);
    const double rp = 1.0 - transmission(1, 1.5, std::sqrt(0.5), true);
    const double mirror_expected = base / 9.0;
    const double slab_expected = base * (9.0 / 64.0) * normal_t * normal_t;
    const double corner_physical = base * rs * rp;
    const double corner_native = base * 0.25 * (rs + rp) * (rs + rp);
    const bool mirror = v[0].x == 1 && close(v[0].y, mirror_expected, 1e-7) &&
                        close(v[0].z, mirror_expected, 1e-7) && close(v[0].w, 1.0 / 9.0, 1e-6);
    const bool slab = v[1].x == 1 && close(v[1].y, slab_expected, 1e-7) &&
                      close(v[1].z, slab_expected, 1e-7) && close(v[1].w, 9.0 / 64.0, 1e-6);
    const bool corner = v[2].x == 1 && close(v[2].y, corner_physical, 1e-7) &&
                        close(v[2].z, corner_native, 1e-7);
    const bool tir = v[3].x == 1 && close(v[3].y, base, 1e-7) &&
                     close(v[3].z, base, 1e-7) && v[3].w > 0.1f;
    const bool pair = v[4].x == 1 && close(v[4].y, base, 1e-7) &&
                      close(v[4].z, 2.0 * base, 1e-7) &&
                      close(v[4].w, -2.0 * base, 1e-7) && v[5].x == 0;
    bool symmetric_slab = true;
    for (int index = 6; index < 10; index++) {
      symmetric_slab &= v[index].x == 1 && close(v[index].y, 0.2870911, 2e-5) &&
                        close(v[index].z, 1.2298075, 2e-5) &&
                        close(v[index].w, 1.15138796, 2e-5);
    }
    symmetric_slab &= close(v[6].y, v[7].y, 1e-5) &&
                      close(v[8].y, v[9].y, 1e-5);
    bool symmetric_pair = true;
    for (int index = 10; index < 12; index++) {
      const double a = v[index - 4].y;
      const double b = v[index - 2].y;
      const double bound = 2.0 * std::sqrt(a * b);
      symmetric_pair &= close(v[index].z, a + b, 1e-5) &&
                        std::abs(v[index].x) <= bound + 1e-5 &&
                        std::abs(v[index].y) <= bound + 1e-5 &&
                        v[index].z + v[index].x >= -1e-6 &&
                        v[index].z + v[index].y >= -1e-6 &&
                        close(v[index].w, 0.0, 1e-8);
    }
    symmetric_pair &= close(v[10].x, v[11].x, 1e-5) &&
                      close(v[10].y, v[11].y, 1e-5) &&
                      v[10].x > 0.57f && v[10].y < -0.57f;
    unsigned hit0_bits[4], hit1_bits[4], center0_bits[4], center1_bits[4];
    for (int i = 0; i < 4; i++) {
      const float hit0 = v[12 + i].x;
      const float hit1 = v[12 + i].y;
      const float center0 = v[12 + i].z;
      const float center1 = v[12 + i].w;
      std::memcpy(&hit0_bits[i], &hit0, sizeof(unsigned));
      std::memcpy(&hit1_bits[i], &hit1, sizeof(unsigned));
      std::memcpy(&center0_bits[i], &center0, sizeof(unsigned));
      std::memcpy(&center1_bits[i], &center1, sizeof(unsigned));
    }
    const bool success = mirror && slab && corner && tir && pair && symmetric_slab &&
                         symmetric_pair;
    std::printf("device=%s mirror=%d slab=%d corner=%d tir=%d pair=%d "
                "mirror_physical=%.9g slab_physical=%.9g corner_physical=%.9g "
                "corner_native=%.9g tir_imaginary=%.9g pair_same=%.9g pair_pi=%.9g "
                "pair_incoherent=%.9g symmetric_slab=%d symmetric_diag=[%.9g,%.9g,%.9g,%.9g] "
                "symmetric_pair=%d pair_cross=[%.9g,%.9g] pair_pi=[%.9g,%.9g] "
                "hit0_x_bits=[%08x,%08x,%08x,%08x] "
                "hit1_x_bits=[%08x,%08x,%08x,%08x] "
                "center_x_bits=[%08x,%08x] success=%d\n",
                device.name.UTF8String, int(mirror), int(slab), int(corner), int(tir), int(pair),
                v[0].y, v[1].y, v[2].y, v[2].z, v[3].w, v[4].z, v[4].w, v[5].x,
                int(symmetric_slab), v[6].y, v[7].y, v[8].y, v[9].y,
                int(symmetric_pair), v[10].x, v[11].x, v[10].y, v[11].y,
                hit0_bits[0], hit0_bits[1], hit0_bits[2], hit0_bits[3],
                hit1_bits[0], hit1_bits[1], hit1_bits[2], hit1_bits[3],
                center0_bits[0], center1_bits[0], int(success));
    return success ? 0 : 1;
  }
}
