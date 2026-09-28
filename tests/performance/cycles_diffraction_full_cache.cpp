/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "diffraction_cache_export.h"
#include "diffraction_packed_audit.h"
#include "scene/diffraction.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
using namespace ccl;
int main(int argc, char **argv)
{
  if (argc != 17 && argc != 18)
    return 2;
  DiffractionGratingCacheOptions options;
  options.maximum_nodes = std::strtoull(argv[1], nullptr, 10);
  options.tolerance = std::atof(argv[2]);
  options.maximum_depth = std::atoi(argv[3]);
  options.half_orders = std::atoi(argv[4]);
  options.cutoff_margin = std::atof(argv[5]);
  options.response_adaptive_splits = std::atoi(argv[6]) != 0;
  const int reference_mib = std::atoi(argv[7]);
  if (reference_mib < 0 || reference_mib > 1024)
    return 2;
  options.reference_cache_matrix_bytes = size_t(reference_mib) * 1024 * 1024;
  options.allow_chart_cells = std::atoi(argv[8]) != 0;
  options.mirror_symmetry = std::atoi(argv[9]) != 0;
  options.validation_workers = std::atoi(argv[10]);
  options.curvature_adaptive_splits = std::atoi(argv[11]) != 0;
  options.complex_tolerance = std::atof(argv[12]);
  options.allow_quadratic_cells = std::atoi(argv[13]) != 0;
  const std::filesystem::path export_directory = argv[16];
  const int audit_queries = std::atoi(argv[14]);
  const unsigned seed = std::strtoul(argv[15], nullptr, 10);
  if (audit_queries < 0 || audit_queries > 1000000)
    return 2;
  options.bounds = {{-0.5, -1, 380}, {0.5, 1, 780}};
  options.maximum_matrix_bytes = 256 * 1024 * 1024;
  std::cout << std::setprecision(12) << "{\"cases\":[";
  const bool dielectric = argc == 18 && std::string(argv[17]) == "dielectric";
  if (argc == 18 && !dielectric)
    return 2;
  bool first = true;
  for (double pitch : {740.0, 1600.0}) {
    const std::complex<double> metal(0.9, 6);
    const DiffractionGratingProfile profile{pitch, 150, 0.41, 1,
                                            dielectric ? std::complex<double>(1.5, 0) : metal,
                                            1, dielectric ? std::complex<double>(1, 0) : metal};
    std::string bound_error;
    options.retained_half_orders = diffraction_grating_reference_order_bound(profile, options, bound_error);
    if (options.retained_half_orders < 0) {
      std::cerr << bound_error << '\n';
      return 2;
    }
    options.progress = [&](const DiffractionGratingCacheStats &stats) {
      if (stats.visited_nodes % 256 == 0) {
        std::cerr << "pitch=" << pitch << " visited=" << stats.visited_nodes
                  << " accepted_domain_fraction=" << stats.accepted_domain_fraction
                  << " solves=" << stats.reference_solves << " hits=" << stats.reference_cache_hits
                  << '\n';
      }
      return true;
    };
    DiffractionGratingCache cache;
    DiffractionGratingCacheStats stats;
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    const bool complete = diffraction_grating_build_cache(profile, options, cache, stats, error);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cerr << "Full cache pitch " << pitch << " complete=" << complete
              << " visited=" << stats.visited_nodes << " " << error << '\n';
    bool exported = false;
    std::string export_error;
    if (complete && !export_directory.empty())
      exported = diffraction_audit::export_cache(
          export_directory / std::to_string(int(pitch)), profile, cache, export_error);
    const std::string validation = complete && audit_queries > 0 ?
                                       diffraction_audit::run(
                                           profile, options, cache, audit_queries, seed) :
                                       "null";
    std::cout << (first ? "" : ",") << "{\"pitch_nm\":" << pitch
              << ",\"material\":" << std::quoted(dielectric ? "dielectric_air_exteriors" : "generic_conductor")
              << ",\"complete\":" << (complete ? "true" : "false")
              << ",\"error\":" << std::quoted(error)
              << ",\"cache_exported\":" << (exported ? "true" : "false")
              << ",\"cache_export_error\":" << std::quoted(export_error)
              << ",\"held_out_validation\":" << validation
              << ",\"visited_nodes\":" << stats.visited_nodes << ",\"curvature_adaptive_splits\":"
              << (options.curvature_adaptive_splits ? "true" : "false")
              << ",\"validation_workers\":" << options.validation_workers
              << ",\"mirror_symmetry\":" << (options.mirror_symmetry ? "true" : "false")
              << ",\"allow_chart_cells\":" << (options.allow_chart_cells ? "true" : "false")
              << ",\"allow_quadratic_cells\":"
              << (options.allow_quadratic_cells ? "true" : "false")
              << ",\"accepted_quadratic_cells\":" << stats.accepted_quadratic_cells
              << ",\"accepted_chart_cells\":" << stats.accepted_chart_cells
              << ",\"accepted_cells\":" << stats.accepted_cells
              << ",\"accepted_matrix_bytes\":" << stats.matrix_bytes
              << ",\"complex_tolerance\":" << options.complex_tolerance
              << ",\"maximum_accepted_complex_error\":" << stats.maximum_accepted_complex_error
              << ",\"last_complex_error\":" << stats.last_complex_error
              << ",\"maximum_accepted_validation_error\":" << stats.maximum_accepted_error
              << ",\"accepted_domain_fraction\":" << stats.accepted_domain_fraction
              << ",\"reference_solves\":" << stats.reference_solves
              << ",\"reference_cache_hits\":" << stats.reference_cache_hits
              << ",\"peak_reference_matrix_bytes\":" << stats.peak_reference_matrix_bytes
              << ",\"reference_cache_matrix_budget\":" << options.reference_cache_matrix_bytes
              << ",\"last_validation_error\":" << stats.last_validation_error
              << ",\"last_depth\":" << stats.last_depth << ",\"last_lower\":["
              << stats.last_bounds.lower[0] << ',' << stats.last_bounds.lower[1] << ','
              << stats.last_bounds.lower[2] << ']' << ",\"last_upper\":["
              << stats.last_bounds.upper[0] << ',' << stats.last_bounds.upper[1] << ','
              << stats.last_bounds.upper[2] << ']' << ",\"returned_cells\":" << cache.cells.size()
              << ",\"seconds\":" << seconds << ",\"maximum_nodes\":" << options.maximum_nodes
              << ",\"maximum_matrix_bytes\":" << options.maximum_matrix_bytes
              << ",\"maximum_depth\":" << options.maximum_depth
              << ",\"tolerance\":" << options.tolerance << ",\"response_adaptive_splits\":"
              << (options.response_adaptive_splits ? "true" : "false")
              << ",\"half_orders\":" << options.half_orders
              << ",\"cutoff_margin\":" << options.cutoff_margin << '}';
    first = false;
  }
  std::cout << "]}\n";
}
