/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <array>
#include <complex>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "util/defines.h"
#include "util/types.h"

CCL_NAMESPACE_BEGIN

/* Host-side electromagnetic response generation. Lengths are nanometers;
 * complex indices are passive (nonnegative imaginary part). The incident
 * medium must be lossless. A single binary lamellar layer is described here.
 * The ridge is centered at x=0 in the periodic cell; this fixes the complex
 * diffraction-order phase origin as well as the intensity response. */
struct DiffractionIndexSample {
  double wavelength;
  std::complex<double> index;
  bool operator==(const DiffractionIndexSample &) const = default;
};

struct DiffractionGratingProfile {
  double pitch;
  double depth;
  double duty;
  double incident_ior;
  std::complex<double> ridge_ior;
  std::complex<double> groove_ior;
  std::complex<double> substrate_ior;
  /* Optional tabulated layer indices, piecewise linear in n and k. Empty
   * tables use the constants above. Nonempty tables must cover every queried
   * wavelength; extrapolation is rejected. Exterior indices remain constant. */
  std::vector<DiffractionIndexSample> ridge_spectrum;
  std::vector<DiffractionIndexSample> groove_spectrum;
  /* Absorbing half-space dispersion. The constant and every table entry must
   * have positive k, preserving the absence of substrate far-field ports. */
  std::vector<DiffractionIndexSample> absorbing_substrate_spectrum;
  bool operator==(const DiffractionGratingProfile &) const = default;
};

/* Resolve all tabulated indices at one wavelength with the CPU solver's exact
 * validation/interpolation rules. No extrapolation; outputs clear on failure. */
bool diffraction_grating_sample_indices(const DiffractionGratingProfile &profile,
                                        double wavelength,
                                        std::complex<double> &ridge,
                                        std::complex<double> &groove,
                                        std::complex<double> &substrate,
                                        std::string &error);

struct DiffractionGratingOrder {
  int order;
  std::array<double, 2> reflection;
  /* Flux entering the substrate; for an absorbing substrate this is not
   * transmitted far-field power. Columns are incident s, p polarizations. */
  std::array<double, 2> substrate_flux;
  /* Flux-normalized Jones matrices, row-major [ss,sp,ps,pp]. Polarization
   * s is perpendicular to the propagation/normal plane; p = cross(s,d).
   * The phase reference is the TOP interface for reflection and BOTTOM
   * interface for transmission. Evanescent orders have zero Jones matrices. */
  std::array<std::complex<double>, 4> reflection_jones;
  std::array<std::complex<double>, 4> transmission_jones;
};

struct DiffractionGratingResponse {
  std::vector<DiffractionGratingOrder> orders;
  std::array<double, 2> layer_absorption;
  bool has_transmission_jones;
  double boundary_residual;
};

struct DiffractionGratingPort {
  int order;
  bool substrate;
};

/* All propagating incident and outgoing channels for one Bloch wavevector.
 * Each port has two smooth Cartesian polarization channels. Their transverse
 * electric basis is B = I - d_t*d_t^T/(1+abs(d_z)), avoiding the s/p basis
 * singularity at normal incidence. matrix is a row-major square flux-normalized
 * scattering matrix; rows are outgoing channels, columns incoming channels.
 * Absorbing substrates have no incoming far-field ports. */
struct DiffractionGratingBlock {
  std::vector<DiffractionGratingPort> ports;
  std::vector<std::complex<double>> matrix;
  double boundary_residual;
  double minimum_power_gain;
  double maximum_power_gain;
};

/* Correct float roundoff in a known-lossless reference operator with complete
 * propagating-port coverage. Never use for absorbing materials. Rejects large
 * defects without modifying the block; this is not a general normalization. */
bool diffraction_grating_restore_lossless_reference(DiffractionGratingBlock &block,
                                                   std::string &error);

/* Compact unpolarized intensities: one real number per port pair rather than
 * a complex 2x2 polarization block. Intended for ordinary spectral rendering. */
struct DiffractionGratingPowerBlock {
  std::vector<DiffractionGratingPort> ports;
  std::vector<double> matrix;
  double maximum_column_sum;
};

struct DiffractionGratingPowerSample {
  const DiffractionGratingPowerBlock *block;
  double weight;
};

void diffraction_grating_power_block(const DiffractionGratingBlock &block,
                                     DiffractionGratingPowerBlock &power);

/* Pack a validated power block for kernel/util/diffraction_table.h. The
 * source must have contiguous, sorted upper then lower ports, as produced by
 * the Maxwell solver. No power renormalization is applied. */
bool diffraction_grating_pack_power(const DiffractionGratingPowerBlock &power,
                                    std::vector<float> &packed,
                                    std::string &error);

/* Convex interpolation in a common physical port indexing. Missing ports are
 * zero, and ports not in the requested target are projected out. Both preserve
 * the column power bound. The caller must supply the target propagating ports
 * and separately check approximation error near changing propagation cutoffs. */
bool diffraction_grating_interpolate_power(std::span<const DiffractionGratingPowerSample> samples,
                                           std::span<const DiffractionGratingPort> target_ports,
                                           DiffractionGratingPowerBlock &power,
                                           std::string &error);

struct DiffractionGratingGridConfig {
  int bloch_samples;
  int tangent_samples;
  int wavelength_samples;
  double wavelength_min;
  double wavelength_max;
  /* Numerical extension at closed channels: unit specular reflection is the
   * generic grazing limit of a reflecting interface. It is not appropriate
   * for a perfectly matched transmitting interface without separate handling.
   * Validate approximation error for the actual profile before enabling. */
  bool closed_port_reflection = false;
};

/* Generates a packed spectral/Bloch intensity grid. Resolution is explicit:
 * successful generation does not establish an interpolation accuracy bound.
 * The caller must validate/refine before using it as a material response.
 * Cancellation and every unresolved modal/cutoff error discard the result. */
bool diffraction_grating_build_grid(const DiffractionGratingProfile &profile,
                                    const DiffractionGratingGridConfig &config,
                                    int half_orders,
                                    std::vector<float> &packed,
                                    std::string &error,
                                    const std::function<bool()> &cancel = {});

/* Fourier modal Maxwell solve with inverse factorization across groove walls.
 * angle and azimuth describe the incident PROPAGATING wave toward the layer,
 * in radians, in a right-handed frame with z into the substrate.
 * This is response-generation code, not a per-shading-point kernel operation.
 * An exact external grazing order or degenerate layer mode currently returns
 * an explicit error: its limiting solution is not approximated silently. */
bool diffraction_grating_solve(const DiffractionGratingProfile &profile,
                               double wavelength,
                               double angle,
                               double azimuth,
                               int half_orders,
                               DiffractionGratingResponse &response,
                               std::string &error);

/* Reuses one modal decomposition and interface factorization for every
 * propagating incident channel, on both sides when the substrate is lossless.
 * angle/azimuth define the base wavevector, to which integer grating momenta
 * are added. Normally response generation chooses a base in one Bloch cell. */
bool diffraction_grating_solve_block(const DiffractionGratingProfile &profile,
                                     double wavelength,
                                     double angle,
                                     double azimuth,
                                     int half_orders,
                                     DiffractionGratingBlock &block,
                                     std::string &error);

/* Direct Bloch coordinates kx/k0 and ky/k0, where k0 = 2*pi/wavelength.
 * Unlike angle/azimuth, these allow a base wave evanescent in the incident
 * medium while other orders or substrate ports propagate. The truncation must
 * include every propagating port. No propagating ports yields an empty block. */
bool diffraction_grating_solve_bloch(const DiffractionGratingProfile &profile,
                                     double wavelength,
                                     double kx,
                                     double ky,
                                     int half_orders,
                                     DiffractionGratingBlock &block,
                                     std::string &error);

/* Constant-admittance reference ports include evanescent physical channels.
 * Each port carries Ex,Ey with unit flux metric, avoiding external square-root
 * cutoffs in the cached operator. High omitted channels must be evanescent.
 * Lossless substrates have reference ports on both sides; absorbing substrates
 * are terminated internally. This operator is not itself a physical BSDF. */
bool diffraction_grating_solve_reference(const DiffractionGratingProfile &profile,
                                         double wavelength,
                                         double kx,
                                         double ky,
                                         int half_orders,
                                         int retained_half_orders,
                                         DiffractionGratingBlock &reference,
                                         std::string &error);

/* Match a reference-port operator to the exact external plane-wave channels.
 * Uses bounded interface reflection coefficients, including at exact grazing.
 * Output uses the same physical Cartesian flux basis as solve_bloch. */
bool diffraction_grating_match_reference(const DiffractionGratingProfile &profile,
                                         double wavelength,
                                         double kx,
                                         double ky,
                                         const DiffractionGratingBlock &reference,
                                         DiffractionGratingBlock &physical,
                                         std::string &error);

/* A mixture of physical flux ports and unit-admittance reference ports.
 * Only orders near a cutoff need remain reference ports. Other evanescent
 * orders are terminated and removed during preparation. */
struct DiffractionGratingHybrid {
  DiffractionGratingBlock scattering;
  std::vector<unsigned char> is_reference;
};

bool diffraction_grating_prepare_hybrid(const DiffractionGratingProfile &profile,
                                        double wavelength,
                                        double kx,
                                        double ky,
                                        const DiffractionGratingBlock &reference,
                                        std::span<const unsigned char> keep_reference,
                                        DiffractionGratingHybrid &hybrid,
                                        std::string &error);

/* Query coordinates may differ from preparation coordinates only while all
 * non-reference ports retain their propagation status. Missing propagating
 * orders and physical ports crossing a cutoff are errors. */
bool diffraction_grating_match_hybrid(const DiffractionGratingProfile &profile,
                                      double wavelength,
                                      double kx,
                                      double ky,
                                      const DiffractionGratingHybrid &hybrid,
                                      DiffractionGratingBlock &physical,
                                      std::string &error);

/* Coordinates are dimensionless Bloch momentum (kx*pitch/wavelength), ky/k0,
 * and vacuum wavelength in nm. Indices are constant within this profile. */
struct DiffractionGratingCellBounds {
  std::array<double, 3> lower, upper;
};

struct DiffractionGratingCell {
  DiffractionGratingCellBounds bounds;
  std::array<DiffractionGratingHybrid, 8> corners;
  int feedback_channels;
};

/* GPU upload data; matrices are corner-major, row-major complex float pairs.
 * Cell bounds and port metadata accompany the matrices in the cache index.
 * Packing validates representation/topology, not interpolation accuracy. */
struct DiffractionGratingPackedCell {
  DiffractionGratingCellBounds bounds;
  std::vector<DiffractionGratingPort> ports;
  std::vector<int> active_ports;
  std::vector<float2> matrices;
  bool operator_chart = false;
  int chart_degree = 1;
  float2 chart_rotation{};
  /* Degree 3 identifies a matrix-anchored tensor cell (not polynomial degree).
   * matrices contains the complex anchor followed by pairs of real model
   * floats in the diffraction_tensor_chart layout. Other cells keep -1. */
  int tensor_rank = -1;
};

bool diffraction_grating_pack_cell(const DiffractionGratingCell &cell,
                                   DiffractionGratingPackedCell &packed,
                                   std::string &error);

struct DiffractionGratingCacheStats;

/* Optional reference backend. Must use the same reference-port convention as
 * diffraction_grating_solve_reference and report failure without fallback.
 * Implementations must support concurrent calls when validation_workers > 1. */
using DiffractionReferenceSolver = std::function<bool(
    const DiffractionGratingProfile &, double, double, double, int, int,
    DiffractionGratingBlock &, std::string &)>;

struct DiffractionGratingCacheOptions {
  DiffractionGratingCellBounds bounds;
  /* Empty selects the existing CPU solver. */
  DiffractionReferenceSolver reference_solver;
  /* Stable implementation/version identity required by DiffractionManager for
   * custom solvers; empty is reserved for the default CPU implementation. */
  std::string reference_backend_key;
  int half_orders = 16;
  /* Positive tolerance enables empirical modal refinement for every reference
   * query, in both fitting and validation. Zero retains fixed resolution. */
  double modal_power_tolerance = 0.0;
  double modal_complex_tolerance = 0.0;
  int maximum_half_orders = 512;
  int retained_half_orders = 5;
  double cutoff_margin = 0.1;
  double tolerance = 0.001;
  /* Zero disables complex validation. Otherwise require chart cells and bound
   * the Frobenius error of the complete physical scattering matrix. */
  double complex_tolerance = 0.0;
  bool response_adaptive_splits = true;
  bool curvature_adaptive_splits = false;
  bool allow_chart_cells = true;
  bool allow_quadratic_cells = false;
  /* Matrix-anchored Hermitian tensor cache; requires a lossless profile. */
  bool use_tensor_cells = false;
  bool mirror_symmetry = false;
  int validation_workers = 1;
  int maximum_depth = 24;
  size_t maximum_nodes = 65535;
  size_t maximum_matrix_bytes = 256 * 1024 * 1024;
  size_t reference_cache_matrix_bytes = 64 * 1024 * 1024;
  /* Return false to cancel without publishing a partial cache. */
  std::function<bool(const DiffractionGratingCacheStats &)> progress;
  /* Thread-safe cancellation query, checked between reference solves. Unlike
   * progress, this may run on parallel validation workers. */
  std::function<bool()> cancelled;
};

struct DiffractionGratingCacheStats {
  size_t visited_nodes = 0, accepted_cells = 0, matrix_bytes = 0;
  double maximum_accepted_error = 0.0;
  double maximum_accepted_complex_error = 0.0;
  double last_complex_error = 0.0;
  double last_validation_error = 0.0;
  int last_depth = 0;
  DiffractionGratingCellBounds last_bounds{};
  size_t reference_solves = 0, reference_cache_hits = 0, peak_reference_matrix_bytes = 0;
  double accepted_domain_fraction = 0.0;
  size_t accepted_chart_cells = 0;
  size_t accepted_quadratic_cells = 0;
};

/* Conservative symmetric reference-order window covering propagating and
 * near-cutoff exterior channels throughout the cache domain. Omitted channels
 * retain their exact outgoing boundary conditions in the modal solve.
 * Does not reduce the internal Fourier truncation. Returns -1 on invalid input
 * or when the required window exceeds that truncation. */
int diffraction_grating_reference_order_bound(const DiffractionGratingProfile &profile,
                                              const DiffractionGratingCacheOptions &options,
                                              std::string &error);

struct DiffractionGratingCache {
  DiffractionGratingCellBounds bounds;
  bool complex_validated = false;
  bool mirror_symmetry = false;
  /* Internal: (axis, left child, right child, split float bits).
   * Leaf: (-1, cell index, 0, 0); cell -1 has no possible physical ports. */
  std::vector<int4> nodes;
  std::vector<DiffractionGratingPackedCell> cells;
};

/* Contiguous arrays for device upload. Two layout/bounds records per cell:
 * layout: (matrix offset, port offset, active offset, degree), (ports, active,0,0).
 * degree zero denotes intensity cells; offsets count elements, not bytes.
 * bounds: (lower.xyz, rotation.real), (upper.xyz, rotation.imag).
 * ports: (order, substrate); active indices are cell-local. */
struct DiffractionGratingDeviceBuffers {
  std::vector<int4> nodes, layout;
  std::vector<float4> bounds;
  std::vector<int2> ports;
  std::vector<int> active;
  std::vector<float2> matrices;
};

bool diffraction_grating_device_buffers(const DiffractionGratingCache &cache,
                                        DiffractionGratingDeviceBuffers &buffers,
                                        std::string &error);

/* Builds every leaf with interior validation probes (27 for linear/intensity,
 * 54 for quadratic cells, including a lattice closer to the faces).
 * This is an empirical interpolation criterion, not a uniform error bound or a
 * modal-convergence guarantee. Resource exhaustion fails without a partial cache.
 * Bounds must be representable in float, as used by GPU lookup. */
bool diffraction_grating_build_cache(const DiffractionGratingProfile &profile,
                                     const DiffractionGratingCacheOptions &options,
                                     DiffractionGratingCache &cache,
                                     DiffractionGratingCacheStats &stats,
                                     std::string &error);

/* Prepare a single cell with conservative exterior-cutoff interval bounds.
 * cutoff_margin is measured in squared longitudinal wave number / k0^2.
 * Successful preparation establishes topology, not interpolation accuracy. */
bool diffraction_grating_prepare_cell(const DiffractionGratingProfile &profile,
                                      const DiffractionGratingCellBounds &bounds,
                                      int half_orders,
                                      int retained_half_orders,
                                      double cutoff_margin,
                                      DiffractionGratingCell &cell,
                                      std::string &error);

/* Match each corner to the query's reference channels, then mix powers.
 * This preserves passivity and lossless energy without mixing complex fields
 * from different parameter samples. It is for intensity transport only. */
bool diffraction_grating_cell_power(const DiffractionGratingProfile &profile,
                                    const DiffractionGratingCell &cell,
                                    const std::array<double, 3> &query,
                                    DiffractionGratingPowerBlock &power,
                                    std::string &error);

/* A local Cayley chart of the reference operator. The Hermitian part of the
 * matrix is nonnegative for a passive operator and zero for a lossless one.
 * Convex interpolation in one fixed chart preserves those properties. */
struct DiffractionGratingReferenceChart {
  std::vector<DiffractionGratingPort> ports;
  std::vector<std::complex<double>> matrix;
  std::complex<double> rotation;
  double minimum_dissipation;
  double frobenius_norm;
  double residual;
};

/* Common passive chart for the hybrid corner operators. Unlike intensity
 * blending this retains a complex response and requires one dense matching
 * solve at the query. Chart choice must be shared by reciprocal cache cells. */
struct DiffractionGratingChartCell {
  DiffractionGratingCellBounds bounds;
  std::vector<unsigned char> is_reference;
  /* Tensor Bernstein controls: 8 for degree 1, 27 for degree 2; x varies fastest. */
  std::vector<DiffractionGratingReferenceChart> corners;
  int degree = 1;
};

/* Fit quadratic Bernstein controls using a fixed hybrid topology. Rejects
 * nonpassive control matrices; successful fitting still needs error validation. */
bool diffraction_grating_prepare_quadratic_chart(const DiffractionGratingProfile &profile,
                                                 const DiffractionGratingCell &cell,
                                                 int half_orders,
                                                 int retained_half_orders,
                                                 DiffractionGratingChartCell &chart,
                                                 std::string &error);

bool diffraction_grating_pack_chart_cell(const DiffractionGratingChartCell &cell,
                                         DiffractionGratingPackedCell &packed,
                                         std::string &error);

bool diffraction_grating_prepare_chart_cell(const DiffractionGratingCell &cell,
                                            DiffractionGratingChartCell &chart_cell,
                                            std::string &error);
bool diffraction_grating_chart_cell_match(const DiffractionGratingProfile &profile,
                                          const DiffractionGratingChartCell &cell,
                                          const std::array<double, 3> &query,
                                          DiffractionGratingBlock &physical,
                                          std::string &error);

/* phase rotates the mathematical reference operator before its Cayley map.
 * A chart with a pole reports failure; choosing a well-conditioned chart and
 * validating interpolation accuracy remain caller responsibilities. */
bool diffraction_grating_reference_to_chart(const DiffractionGratingBlock &reference,
                                            double phase,
                                            DiffractionGratingReferenceChart &chart,
                                            std::string &error);
bool diffraction_grating_chart_to_reference(const DiffractionGratingReferenceChart &chart,
                                            DiffractionGratingBlock &reference,
                                            std::string &error);

/* Choose one chart for a group of operators, minimizing the worst matrix
 * norm among candidate rotations. Candidates include gaps between eigenvalue
 * pole angles. Interpolation callers should group reciprocal cells together so
 * their chart choice is identical. This controls chart conditioning, not the
 * physical resonance conditioning or interpolation error. */
bool diffraction_grating_choose_chart(std::span<const DiffractionGratingBlock> references,
                                      double &phase,
                                      std::vector<DiffractionGratingReferenceChart> &charts,
                                      std::string &error);

CCL_NAMESPACE_END
