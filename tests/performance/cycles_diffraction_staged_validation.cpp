/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
/* Run against baseline and candidate host objects; compare exported buffers. */
#include "diffraction_cache_export.h"
#include "diffraction_packed_audit.h"
#include <chrono>
#include <iostream>
#ifdef DIFFRACTION_TEST_POLAR
#  include <Eigen/Dense>
#endif
#ifdef DIFFRACTION_TEST_METAL
#  include "device/metal/diffraction/reference_pool.h"
#  include "diffraction_shader_source.h"
#endif

int main(int argc, char **argv)
{
  using namespace ccl;
  if (argc != 2) return 2;
#ifdef DIFFRACTION_TEST_METAL
  @autoreleasepool {
  auto pool = std::make_shared<DiffractionMetalReferencePool>(
      [NSString stringWithUTF8String:diffraction_metal_shader_source],
      MTLCreateSystemDefaultDevice());
#endif
  for (int test = 0; test < 4; ++test) {
    const std::complex<double> material = test < 2 ? std::complex<double>(0.9, 6) :
                                                   std::complex<double>(1.5, 0);
    DiffractionGratingProfile profile{740, 150, 0.41, 1, material, 1, material};
    DiffractionGratingCacheOptions options;
    options.bounds = {{-0.0625, -0.125, 590}, {0.0625, 0.125, 610}};
    options.half_orders = 4;
    options.retained_half_orders = 2;
    options.allow_quadratic_cells = true;
    options.curvature_adaptive_splits = true;
    options.complex_tolerance = test % 2 ? 0.001 : 0;
    options.validation_workers = 2;
    options.maximum_nodes = 1023;
#ifdef DIFFRACTION_TEST_METAL
    options.reference_backend_key = "staged-validation-metal-test";
    options.reference_solver = [pool](const auto &p, double wavelength, double kx,
                                     double ky, int n, int retained, auto &out,
                                     auto &message) {
      if (!pool->solve(p, wavelength, kx, ky, n, retained, out, message)) return false;
#ifdef DIFFRACTION_TEST_POLAR
      /* Experimental correction only for these constant-index lossless fixtures. */
      if (p.ridge_ior.imag() == 0 && p.groove_ior.imag() == 0 &&
          p.substrate_ior.imag() == 0 && p.ridge_spectrum.empty() &&
          p.groove_spectrum.empty() && p.absorbing_substrate_spectrum.empty()) {
        const int channels = 2 * out.ports.size();
        Eigen::MatrixXcd matrix(channels, channels);
        for (int r = 0; r < channels; ++r)
          for (int c = 0; c < channels; ++c)
            matrix(r,c) = out.matrix[r*channels+c];
        const auto identity = Eigen::MatrixXcd::Identity(channels, channels).eval();
        const double residual = (matrix.adjoint()*matrix-identity).norm();
        if (!std::isfinite(residual) || residual > 1e-4) {
          out = {}; message = "Lossless response exceeds experimental correction bound";
          return false;
        }
        const auto original = matrix;
        for (int i = 0; i < 2; ++i)
          matrix = (0.5*matrix*(3.0*identity-matrix.adjoint()*matrix)).eval();
        if (!matrix.allFinite() || (matrix.adjoint()*matrix-identity).norm() > 1e-12 ||
            (matrix-original).cwiseAbs().maxCoeff() > 1e-4) {
          out = {}; message = "Experimental lossless correction failed";
          return false;
        }
        for (int r = 0; r < channels; ++r)
          for (int c = 0; c < channels; ++c)
            out.matrix[r*channels+c] = matrix(r,c);
      }
#endif
      return true;
    };
#endif
    DiffractionGratingCache cache;
    DiffractionGratingCacheStats stats;
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    if (!diffraction_grating_build_cache(profile, options, cache, stats, error) ||
        !diffraction_audit::export_cache(
            std::filesystem::path(argv[1]) / std::to_string(test), profile, cache, error)) {
      std::cerr << test << ": " << error << '\n';
      return 1;
    }
    std::cout << test << " nodes=" << stats.visited_nodes
              << " cells=" << stats.accepted_cells
              << " quadratic=" << stats.accepted_quadratic_cells
              << " solves=" << stats.reference_solves
              << " seconds=" << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start).count() << '\n';
    const std::string audit = diffraction_audit::run(profile, options, cache, 1024, 617923, true);
    std::ofstream audit_file(std::filesystem::path(argv[1]) / std::to_string(test) /
                             "held_out.json");
    audit_file << audit << '\n';
    if (!audit_file || audit.find("\"passed\":true") == std::string::npos)
      return 3;
  }
#ifdef DIFFRACTION_TEST_METAL
  }
#endif
}
