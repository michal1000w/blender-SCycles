/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "scene/diffraction_tensor.h"
#include "scene/diffraction_convergence.h"
#include "kernel/util/diffraction_tensor.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <limits>
#include <list>
#include <map>
#include <random>
#include <algorithm>
CCL_NAMESPACE_BEGIN
namespace {
using Complex = std::complex<double>;
using Matrix = Eigen::MatrixXcd;
using RealMatrix = Eigen::MatrixXd;
using TensorReferenceSolve = std::function<bool(
    double, double, double, DiffractionGratingBlock &, size_t &, std::string &)>;
}
static bool prepare_tensor_cell_impl(const DiffractionGratingProfile &profile,
                                             const DiffractionGratingCellBounds &bounds,
                                             const DiffractionGratingCacheOptions &options,
                                             const double compression_tolerance,
                                             DiffractionGratingPackedCell &cell,
                                             std::string &error,
                                             const TensorReferenceSolve &solve)
{
  cell = {};
  error.clear();
  auto fail = [&](const char *message) { error = message; return false; };
  if (!(compression_tolerance > 0) || !std::isfinite(compression_tolerance))
    return fail("Invalid tensor compression tolerance");
  for (int a = 0; a < 3; a++) {
    if (!std::isfinite(bounds.lower[a]) || !std::isfinite(bounds.upper[a]) ||
        !(bounds.lower[a] < bounds.upper[a]) ||
        double(float(bounds.lower[a])) != bounds.lower[a] ||
        double(float(bounds.upper[a])) != bounds.upper[a])
      return fail("Tensor bounds must be finite ordered float coordinates");
  }
  std::vector<Matrix> scattering;
  std::vector<DiffractionGratingPort> ports;
  int channels = 0;
  DiffractionGratingCacheStats progress;
  progress.visited_nodes = 1;
  progress.last_bounds = bounds;
  for (int sample = 0; sample < 343; sample++) {
    if (options.progress && !options.progress(progress))
      return fail("Tensor cell construction cancelled");
    int lattice = sample;
    double q[3];
    for (int a = 0; a < 3; a++) {
      q[a] = bounds.lower[a] + double(lattice % 7) / 6 * (bounds.upper[a] - bounds.lower[a]);
      lattice /= 7;
    }
    DiffractionGratingBlock reference;
    size_t solve_count = 0;
    const bool solved = solve(
        q[2], q[0] * q[2] / profile.pitch, q[1], reference, solve_count, error);
    progress.reference_solves += solve_count;
    if (options.progress && !options.progress(progress) && solved)
      return fail("Tensor cell construction cancelled");
    if (!solved)
      return false;
    if (sample == 0) {
      ports = reference.ports;
      if (ports.empty() || ports.size() > size_t(std::numeric_limits<int>::max()) / 2)
        return fail("Invalid tensor reference port count");
      channels = int(2 * ports.size());
      if (channels > std::numeric_limits<int>::max() / channels ||
          size_t(channels) * channels > options.maximum_matrix_bytes / sizeof(float2))
        return fail("Tensor reference dimensions exceed resource limits");
    }
    if (reference.ports.size() != ports.size() ||
        reference.matrix.size() != size_t(channels) * channels)
      return fail("Inconsistent tensor reference topology");
    for (size_t p = 0; p < ports.size(); p++)
      if (ports[p].order != reference.ports[p].order || ports[p].substrate != reference.ports[p].substrate)
        return fail("Inconsistent tensor port ordering");
    Matrix s(channels, channels);
    for (int r = 0; r < channels; r++)
      for (int c = 0; c < channels; c++)
        s(r, c) = reference.matrix[size_t(r) * channels + c];
    const double residual = (s.adjoint() * s - Matrix::Identity(channels, channels)).norm();
    if (!std::isfinite(residual) || residual > 1e-8)
      return fail("Tensor Hermitian chart requires lossless unitary reference operators");
    scattering.push_back(std::move(s));
  }
  const Matrix anchor = scattering[171];
  const Matrix identity = Matrix::Identity(channels, channels);
  const int square = channels * channels, upper = channels * (channels - 1) / 2;
  RealMatrix packed(343, square);
  for (int sample = 0; sample < 343; sample++) {
    const Matrix u = anchor.adjoint() * scattering[sample];
    const Matrix left = identity + u, right = identity - u;
    Eigen::FullPivLU<Matrix> lu(left);
    if (!lu.isInvertible() || lu.rcond() < 1e-8)
      return fail("Tensor anchor chart requires subdivision near a coordinate pole");
    Matrix h = Complex(0, -1) * lu.solve(right);
    const double solve_error = (left * (Complex(0, 1) * h) - right).norm();
    const Matrix hermitian = (0.5 * (h + h.adjoint())).eval();
    if (!std::isfinite(solve_error) || solve_error > 1e-8 ||
        (h - hermitian).norm() > 1e-8)
      return fail("Tensor chart solve exceeds roundoff tolerance");
    int k = 0;
    for (int r = 0; r < channels; r++) {
      packed(sample, r) = hermitian(r, r).real();
      for (int c = r + 1; c < channels; c++, k++) {
        packed(sample, channels + k) = std::sqrt(2.0) * hermitian(r, c).real();
        packed(sample, channels + upper + k) = std::sqrt(2.0) * hermitian(r, c).imag();
      }
    }
  }
  scattering.clear();
  const Eigen::RowVectorXd mean = packed.colwise().mean();
  if (options.progress && !options.progress(progress))
    return fail("Tensor cell construction cancelled");
  RealMatrix residual = packed.rowwise() - mean;
  Eigen::JacobiSVD<RealMatrix> svd(residual, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (svd.info() != Eigen::Success)
    return fail("Tensor compression SVD failed");
  const RealMatrix coefficients = svd.matrixU() * svd.singularValues().asDiagonal();
  int rank = 0;
  while (residual.rowwise().norm().maxCoeff() > compression_tolerance) {
    if (rank >= svd.singularValues().size())
      return fail("Tensor compression cannot meet training tolerance");
    residual.noalias() -= coefficients.col(rank) * svd.matrixV().col(rank).transpose();
    rank++;
  }
  const size_t real_count = 14 + size_t(square) + size_t(rank) * (343 + size_t(square));
  const size_t pairs = size_t(square) + (real_count + 1) / 2;
  if (real_count > size_t(std::numeric_limits<int>::max()) ||
      pairs > options.maximum_matrix_bytes / sizeof(float2))
    return fail("Tensor cell exceeds matrix memory budget");
  DiffractionGratingPackedCell result;
  result.bounds = bounds;
  result.ports = ports;
  for (size_t p = 0; p < ports.size(); p++) result.active_ports.push_back(int(p));
  result.operator_chart = true;
  result.chart_degree = 3;
  result.tensor_rank = rank;
  result.matrices.resize(pairs, make_float2(0, 0));
  for (int r = 0; r < channels; r++)
    for (int c = 0; c < channels; c++)
      result.matrices[size_t(r) * channels + c] = make_float2(float(anchor(r, c).real()), float(anchor(r, c).imag()));
  const auto write = [&](const size_t i, const double value) {
    if (i & 1) result.matrices[size_t(square) + i / 2].y = float(value);
    else result.matrices[size_t(square) + i / 2].x = float(value);
  };
  const double weights[7] = {-.05, .3, -.75, 1, -.75, .3, -.05};
  for (int i = 0; i < 7; i++) { write(i, double(i) / 6); write(7 + i, weights[i]); }
  size_t offset = 14;
  for (int s = 0; s < 343; s++)
    for (int r = 0; r < rank; r++) write(offset++, coefficients(s, r));
  for (int r = 0; r < rank; r++)
    for (int i = 0; i < square; i++) write(offset++, svd.matrixV()(i, r));
  for (int i = 0; i < square; i++) write(offset++, mean[i]);
  for (const float2 value : result.matrices)
    if (!std::isfinite(value.x) || !std::isfinite(value.y))
      return fail("Nonfinite packed tensor model");
  progress.matrix_bytes = result.matrices.size() * sizeof(float2);
  if (options.progress && !options.progress(progress))
    return fail("Tensor cell construction cancelled");
  cell = std::move(result);
  return true;
}

bool diffraction_grating_prepare_tensor_cell(const DiffractionGratingProfile &profile,
                                             const DiffractionGratingCellBounds &bounds,
                                             const DiffractionGratingCacheOptions &options,
                                             const double compression_tolerance,
                                             DiffractionGratingPackedCell &cell,
                                             std::string &error)
{
  const TensorReferenceSolve solve = [&](double wavelength, double kx, double ky,
                                        DiffractionGratingBlock &reference, size_t &count,
                                        std::string &message) {
    return diffraction_grating_cache_reference(
        profile, options, wavelength, kx, ky, reference, count, message);
  };
  return prepare_tensor_cell_impl(profile, bounds, options, compression_tolerance, cell, error, solve);
}

bool diffraction_grating_tensor_reference(const DiffractionGratingPackedCell &cell,
                                          const std::array<double, 3> &query,
                                          DiffractionGratingBlock &reference,
                                          std::string &error)
{
  reference = {};
  error.clear();
  auto fail = [&](const char *message) { error = message; return false; };
  const size_t channels = 2 * cell.ports.size();
  if (!cell.operator_chart || cell.chart_degree != 3 || channels == 0 ||
      channels > size_t(std::numeric_limits<int>::max()) / channels || cell.tensor_rank < 0)
    return fail("Invalid tensor reference cell");
  const size_t square = channels * channels, rank = size_t(cell.tensor_rank);
  if (rank > square || square > size_t(std::numeric_limits<int>::max()) - 14 ||
      rank > (size_t(std::numeric_limits<int>::max()) - 14 - square) / (343 + square))
    return fail("Invalid tensor reference dimensions");
  const size_t count = 14 + square + rank * (343 + square);
  if (cell.matrices.size() != square + (count + 1) / 2)
    return fail("Invalid tensor reference payload");
  DiffractionTensorFloatView model{cell.matrices.data() + square, 0};
  float weights[3][7];
  for (int a = 0; a < 3; a++) {
    const double lo = cell.bounds.lower[a], hi = cell.bounds.upper[a];
    if (!std::isfinite(query[a]) || !std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi) ||
        query[a] < lo || query[a] > hi)
      return fail("Tensor reference query outside cell");
    const float t = (float(query[a]) - float(lo)) / (float(hi) - float(lo));
    if (!diffraction_tensor_basis(t, model, model + 7, weights[a]))
      return fail("Invalid tensor interpolation basis");
  }
  std::vector<float> coefficients(rank, 0.0f);
  for (size_t r = 0; r < rank; r++) {
    for (int z = 0; z < 7; z++) {
      float zy = 0;
      for (int y = 0; y < 7; y++) {
        float yx = 0;
        for (int x = 0; x < 7; x++)
          yx += weights[0][x] * model[14 + ((z * 7 + y) * 7 + x) * rank + r];
        zy += weights[1][y] * yx;
      }
      coefficients[r] += weights[2][z] * zy;
    }
  }
  const size_t directions = 14 + 343 * rank, mean = directions + rank * square;
  const auto value = [&](const size_t i) {
    float v = model[mean + i];
    for (size_t r = 0; r < rank; r++) v += coefficients[r] * model[directions + r * square + i];
    return v;
  };
  Eigen::MatrixXcf y = Eigen::MatrixXcf::Zero(channels, channels), anchor(channels, channels);
  size_t upper = 0;
  for (size_t row = 0; row < channels; row++) {
    y(row, row) = std::complex<float>(0, value(row));
    for (size_t col = row + 1; col < channels; col++, upper++) {
      const float re = value(channels + upper) * M_SQRT1_2F;
      const float im = value(channels + channels * (channels - 1) / 2 + upper) * M_SQRT1_2F;
      y(row, col) = std::complex<float>(-im, re);
      y(col, row) = std::complex<float>(im, re);
    }
    for (size_t col = 0; col < channels; col++) {
      const float2 a = cell.matrices[row * channels + col];
      anchor(row, col) = std::complex<float>(a.x, a.y);
    }
  }
  const Eigen::MatrixXcf identity = Eigen::MatrixXcf::Identity(channels, channels);
  const Eigen::MatrixXcf rotated = (identity + y).partialPivLu().solve(identity - y);
  const Eigen::MatrixXcf s = anchor * rotated;
  if (!s.allFinite())
    return fail("Nonfinite tensor scattering reconstruction");
  DiffractionGratingBlock result{};
  const Matrix precise = s.cast<Complex>();
  Eigen::SelfAdjointEigenSolver<Matrix> power(precise.adjoint() * precise,
                                             Eigen::EigenvaluesOnly);
  if (power.info() != Eigen::Success)
    return fail("Tensor power bound eigensystem did not converge");
  result.minimum_power_gain = power.eigenvalues().minCoeff();
  result.maximum_power_gain = power.eigenvalues().maxCoeff();
  result.boundary_residual = ((identity + y) * rotated - (identity - y)).norm();
  result.ports = cell.ports;
  result.matrix.resize(square);
  for (size_t row = 0; row < channels; row++)
    for (size_t col = 0; col < channels; col++)
      result.matrix[row * channels + col] = Complex(s(row, col).real(), s(row, col).imag());
  reference = std::move(result);
  return true;
}

bool diffraction_grating_build_tensor_cache(const DiffractionGratingProfile &profile,
                                            const DiffractionGratingCacheOptions &options,
                                            DiffractionGratingCache &cache,
                                            DiffractionGratingCacheStats &stats,
                                            std::string &error)
{
  cache = {};
  stats = {};
  error.clear();
  const auto fail = [&](const char *message) { error = message; return false; };
  if (!(options.tolerance > 0) || !std::isfinite(options.tolerance) ||
      options.complex_tolerance < 0 || !std::isfinite(options.complex_tolerance) ||
      options.maximum_depth < 0 || options.maximum_depth > 60 || options.maximum_nodes == 0 ||
      options.maximum_nodes > size_t(std::numeric_limits<int>::max()))
    return fail("Invalid tensor cache limits");
  auto domain = options.bounds;
  for (int a = 0; a < 3; a++)
    if (!std::isfinite(domain.lower[a]) || !std::isfinite(domain.upper[a]) ||
        !(domain.lower[a] < domain.upper[a]) || double(float(domain.lower[a])) != domain.lower[a] ||
        double(float(domain.upper[a])) != domain.upper[a])
      return fail("Invalid tensor cache domain");
  if (options.mirror_symmetry) {
    if (domain.lower[0] != -domain.upper[0] || domain.lower[1] != -domain.upper[1])
      return fail("Mirror tensor cache requires origin-symmetric bounds");
    domain.upper[0] = domain.upper[1] = 0;
  }
  DiffractionGratingCache result;
  result.bounds = domain;
  result.mirror_symmetry = options.mirror_symmetry;
  result.complex_validated = options.complex_tolerance > 0;
  std::vector<float> spectral_splits;
  for (const auto *spectrum : {&profile.ridge_spectrum, &profile.groove_spectrum,
                                &profile.absorbing_substrate_spectrum}) {
    for (const auto &sample : *spectrum) {
      if (!std::isfinite(sample.wavelength)) return fail("Invalid tensor material wavelength");
      if (sample.wavelength <= domain.lower[2] || sample.wavelength >= domain.upper[2]) continue;
      const float rounded = float(sample.wavelength);
      spectral_splits.push_back(rounded);
      if (double(rounded) != sample.wavelength)
        spectral_splits.push_back(std::nextafter(rounded, rounded < sample.wavelength ?
                                                  std::numeric_limits<float>::infinity() :
                                                  -std::numeric_limits<float>::infinity()));
    }
  }
  std::sort(spectral_splits.begin(), spectral_splits.end());
  spectral_splits.erase(std::unique(spectral_splits.begin(), spectral_splits.end()), spectral_splits.end());
  /* Reuse only identical queries within this material/build request. Keeping
   * the unrounded double key avoids changing physical inputs near cutoffs. */
  using Key = std::array<double, 3>;
  struct CachedReference {
    DiffractionGratingBlock block;
    std::list<Key>::iterator recent;
  };
  std::map<Key, CachedReference> references;
  std::list<Key> recent;
  size_t reference_bytes = 0;
  const TensorReferenceSolve solve = [&](double wavelength, double kx, double ky,
                                        DiffractionGratingBlock &reference, size_t &count,
                                        std::string &message) {
    const Key key{wavelength, kx, ky};
    const auto found = references.find(key);
    if (found != references.end()) {
      reference = found->second.block;
      count = 0;
      message.clear();
      recent.splice(recent.begin(), recent, found->second.recent);
      stats.reference_cache_hits++;
      return true;
    }
    if (!diffraction_grating_cache_reference(
            profile, options, wavelength, kx, ky, reference, count, message))
      return false;
    const size_t bytes = reference.matrix.size() * sizeof(Complex);
    if (bytes > 0 && bytes <= options.reference_cache_matrix_bytes) {
      while (reference_bytes > options.reference_cache_matrix_bytes - bytes) {
        const auto oldest = references.find(recent.back());
        reference_bytes -= oldest->second.block.matrix.size() * sizeof(Complex);
        references.erase(oldest);
        recent.pop_back();
      }
      recent.push_front(key);
      references.emplace(key, CachedReference{reference, recent.begin()});
      reference_bytes += bytes;
      stats.peak_reference_matrix_bytes = std::max(stats.peak_reference_matrix_bytes, reference_bytes);
    }
    return true;
  };
  std::function<bool(const DiffractionGratingCellBounds &, int, int &)> build;
  build = [&](const DiffractionGratingCellBounds &bounds, const int depth, int &index) {
    if (stats.visited_nodes >= options.maximum_nodes)
      return fail("Tensor cache exceeded node budget");
    stats.visited_nodes++;
    stats.last_bounds = bounds;
    stats.last_depth = depth;
    stats.last_validation_error = stats.last_complex_error = 0;
    if (options.progress && !options.progress(stats))
      return fail("Tensor cache construction cancelled");
    index = int(result.nodes.size());
    result.nodes.push_back(make_int4(-1, -1, 0, 0));
    const auto first_knot = std::upper_bound(spectral_splits.begin(), spectral_splits.end(), bounds.lower[2]);
    const auto last_knot = std::lower_bound(spectral_splits.begin(), spectral_splits.end(), bounds.upper[2]);
    if (first_knot != last_knot) {
      if (depth >= options.maximum_depth) return fail("Tensor material knots exceed depth budget");
      const float knot = *(first_knot + (last_knot - first_knot) / 2);
      auto left = bounds, right = bounds;
      left.upper[2] = right.lower[2] = knot;
      int l, r;
      if (!build(left, depth + 1, l) || !build(right, depth + 1, r)) return false;
      result.nodes[index] = make_int4(2, l, r, __float_as_int(knot));
      return true;
    }
    DiffractionGratingPackedCell candidate;
    auto cell_options = options;
    const size_t previous_solves = stats.reference_solves;
    cell_options.progress = [&](const DiffractionGratingCacheStats &local) {
      stats.reference_solves = previous_solves + local.reference_solves;
      return !options.progress || options.progress(stats);
    };
    bool prepared = prepare_tensor_cell_impl(
        profile, bounds, cell_options, 1e-7, candidate, error, solve);
    if (!prepared && error != "Tensor anchor chart requires subdivision near a coordinate pole")
      return false;
    error.clear();
    double maximum = 0, complex_maximum = 0;
    const auto probe = [&](std::array<double, 3> q) {
      /* Once rejected, further probes cannot make this cell admissible.
       * Refinement curvature comes from the fitted tensor, not these probes.
       * Every accepted child still runs the complete validation set. */
      if (maximum > options.tolerance * .25 ||
          (options.complex_tolerance > 0 && complex_maximum > options.complex_tolerance * .25))
        return true;
      for (int a = 0; a < 3; a++) {
        q[a] = float(q[a]);
        if (q[a] < bounds.lower[a] || q[a] > bounds.upper[a]) return true;
      }
      DiffractionGratingBlock predicted_reference, predicted, direct_reference, direct;
      if (!diffraction_grating_tensor_reference(candidate, q, predicted_reference, error) ||
          !diffraction_grating_match_reference(profile, q[2], q[0] * q[2] / profile.pitch,
                                                q[1], predicted_reference, predicted, error))
        return false;
      size_t solve_count = 0;
      const bool solved = solve(
          q[2], q[0] * q[2] / profile.pitch, q[1], direct_reference,
          solve_count, error);
      stats.reference_solves += solve_count;
      if (!solved)
        return false;
      if (!diffraction_grating_match_reference(profile, q[2], q[0] * q[2] / profile.pitch,
                                                q[1], direct_reference, direct, error))
        return false;
      DiffractionGratingPowerBlock a, b;
      diffraction_grating_power_block(predicted, a);
      diffraction_grating_power_block(direct, b);
      if (a.ports.size() != b.ports.size() || predicted.matrix.size() != direct.matrix.size())
        return fail("Inconsistent tensor validation topology");
      const size_t count = b.ports.size();
      for (size_t col = 0; col < count; col++) {
        if (a.ports[col].order != b.ports[col].order || a.ports[col].substrate != b.ports[col].substrate)
          return fail("Inconsistent tensor validation ports");
        double difference = 0;
        for (size_t row = 0; row < count; row++)
          difference += std::abs(a.matrix[row * count + col] - b.matrix[row * count + col]);
        if (!std::isfinite(difference)) return fail("Nonfinite tensor power validation");
        maximum = std::max(maximum, difference);
      }
      if (options.complex_tolerance > 0) {
        double difference = 0;
        for (size_t i = 0; i < direct.matrix.size(); i++)
          difference = std::hypot(difference, std::abs(predicted.matrix[i] - direct.matrix[i]));
        if (!std::isfinite(difference)) return fail("Nonfinite tensor complex validation");
        complex_maximum = std::max(complex_maximum, difference);
      }
      return !options.progress || options.progress(stats) || fail("Tensor cache construction cancelled");
    };
    if (prepared) {
      std::mt19937 random(190731u + unsigned(index));
      for (int sample = 0; sample < 128; sample++) {
        std::array<double, 3> q;
        for (int a = 0; a < 3; a++)
          q[a] = bounds.lower[a] + (double(random()) + .5) / 4294967296.0 *
                                     (bounds.upper[a] - bounds.lower[a]);
        if (!probe(q)) return false;
      }
      /* Include faces/corners as well as samples strictly inside each cell. */
      for (int sample = 0; sample < 125; sample++) {
        int lattice = sample;
        std::array<double, 3> q;
        for (int a = 0; a < 3; a++) {
          q[a] = bounds.lower[a] + .25 * (lattice % 5) * (bounds.upper[a] - bounds.lower[a]);
          lattice /= 5;
        }
        if (!probe(q)) return false;
      }
      /* Exact exterior matching handles topology changes. Probe adjacent
       * float ky coordinates on those changes to expose amplified fit errors. */
      for (int wi = 0; wi < 3; wi++) for (int bi = 0; bi < 3; bi++) {
        const double w = bounds.lower[2] + .5 * wi * (bounds.upper[2] - bounds.lower[2]);
        const double b = bounds.lower[0] + .5 * bi * (bounds.upper[0] - bounds.lower[0]);
        for (int order = -options.retained_half_orders; order <= options.retained_half_orders; order++) {
          const double x = (b + order) * w / profile.pitch;
          for (double exterior : {profile.incident_ior, profile.substrate_ior.real()}) {
            if (!(exterior > 0) || x * x > exterior * exterior) continue;
            const double y = std::sqrt(std::max(0.0, exterior * exterior - x * x));
            for (double sign : {-1.0, 1.0}) {
              const float center = float(sign * y);
              for (float k : {std::nextafter(center, -std::numeric_limits<float>::infinity()),
                              center, std::nextafter(center, std::numeric_limits<float>::infinity())})
                if (!probe({b, k, w})) return false;
            }
          }
        }
      }
    }
    stats.last_validation_error = maximum;
    stats.last_complex_error = complex_maximum;
    if (prepared && maximum <= options.tolerance * .25 &&
        (options.complex_tolerance == 0 || complex_maximum <= options.complex_tolerance * .25)) {
      const size_t bytes = candidate.matrices.size() * sizeof(float2);
      if (bytes > options.maximum_matrix_bytes - stats.matrix_bytes)
        return fail("Tensor cache exceeded matrix memory budget");
      result.nodes[index].y = int(result.cells.size());
      result.cells.push_back(std::move(candidate));
      stats.matrix_bytes += bytes;
      stats.accepted_cells++;
      stats.accepted_chart_cells++;
      stats.maximum_accepted_error = std::max(stats.maximum_accepted_error, maximum);
      stats.maximum_accepted_complex_error = std::max(stats.maximum_accepted_complex_error, complex_maximum);
      double fraction = 1;
      for (int a = 0; a < 3; a++) fraction *= (bounds.upper[a] - bounds.lower[a]) / (domain.upper[a] - domain.lower[a]);
      stats.accepted_domain_fraction += fraction;
      return true;
    }
    if (depth >= options.maximum_depth)
      return fail("Tensor cache failed tolerance at maximum depth");
    int axis = 0;
    double best = -1, extent_best = -1;
    for (int a = 0, stride = 1; a < 3; a++, stride *= 7) {
      double curvature = 0;
      if (prepared) {
        const int rank = candidate.tensor_rank;
        DiffractionTensorFloatView model{candidate.matrices.data() + 4 * candidate.ports.size() * candidate.ports.size(), 0};
        for (int sample = 0; sample < 343; sample++) {
          if ((sample / stride) % 7 >= 5) continue;
          for (int r = 0; r < rank; r++) {
            const double second = double(model[14 + sample * rank + r]) -
                                  2 * double(model[14 + (sample + stride) * rank + r]) +
                                  double(model[14 + (sample + 2 * stride) * rank + r]);
            curvature += second * second;
          }
        }
      }
      const double extent = (bounds.upper[a] - bounds.lower[a]) / (domain.upper[a] - domain.lower[a]);
      if (curvature > best || (curvature == best && extent > extent_best)) {
        axis = a; best = curvature; extent_best = extent;
      }
    }
    const float middle = float(.5 * (bounds.lower[axis] + bounds.upper[axis]));
    if (!(middle > bounds.lower[axis] && middle < bounds.upper[axis]))
      return fail("Tensor cache exhausted float coordinate resolution");
    auto left = bounds, right = bounds;
    left.upper[axis] = right.lower[axis] = middle;
    candidate = {};
    int l, r;
    if (!build(left, depth + 1, l) || !build(right, depth + 1, r)) return false;
    result.nodes[index] = make_int4(axis, l, r, __float_as_int(middle));
    return true;
  };
  int root;
  if (!build(domain, 0, root)) return false;
  DiffractionGratingDeviceBuffers checked;
  if (!diffraction_grating_device_buffers(result, checked, error)) return false;
  cache = std::move(result);
  return true;
}
CCL_NAMESPACE_END
