/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/diffraction.h"
#include "scene/diffraction_convergence.h"

#include "kernel/util/diffraction_grid.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <future>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <numbers>
#include <sstream>

CCL_NAMESPACE_BEGIN

namespace {
using Complex = std::complex<double>;
using Matrix = Eigen::MatrixXcd;
using RealVector = Eigen::VectorXd;
using Vector = Eigen::VectorXcd;
constexpr double pi = std::numbers::pi_v<double>;

Complex outgoing_sqrt(const Complex value)
{
  /* Preserve the exact distinction between propagating and evanescent modes
   * in lossless half spaces. Polar-form complex sqrt can leave a tiny real
   * component on the imaginary axis, which is not a propagating flux. */
  if (value.imag() == 0.0) {
    return value.real() >= 0.0 ? Complex(std::sqrt(value.real()), 0.0) :
                                 Complex(0.0, std::sqrt(-value.real()));
  }
  const Complex root = std::sqrt(value);
  return root.imag() < 0.0 ? -root : root;
}

bool valid_index(const Complex index)
{
  return std::isfinite(index.real()) && std::isfinite(index.imag()) && index.real() >= 0.0 &&
         index.imag() >= 0.0 && std::isfinite(std::norm(index)) && std::norm(index) > 0.0;
}

bool sample_layer_index(const std::vector<DiffractionIndexSample> &spectrum,
                        const Complex constant,
                        const double wavelength,
                        Complex &result)
{
  result = constant;
  if (spectrum.empty())
    return valid_index(result);
  double previous = 0;
  for (const auto &sample : spectrum) {
    if (!std::isfinite(sample.wavelength) || sample.wavelength <= previous ||
        !valid_index(sample.index))
      return false;
    previous = sample.wavelength;
  }
  if (!std::isfinite(wavelength) || wavelength < spectrum.front().wavelength ||
      wavelength > spectrum.back().wavelength)
    return false;
  const auto upper = std::lower_bound(spectrum.begin(),
                                      spectrum.end(),
                                      wavelength,
                                      [](const DiffractionIndexSample &sample, double value) {
                                        return sample.wavelength < value;
                                      });
  if (upper == spectrum.begin() || upper->wavelength == wavelength) {
    result = upper->index;
    return true;
  }
  const auto &lower = *(upper - 1);
  const double t = (wavelength - lower.wavelength) / (upper->wavelength - lower.wavelength);
  result = (1 - t) * lower.index + t * upper->index;
  return valid_index(result);
}

bool sample_substrate_index(const DiffractionGratingProfile &profile,
                            const double wavelength,
                            Complex &result)
{
  if (!profile.absorbing_substrate_spectrum.empty()) {
    if (!(profile.substrate_ior.imag() > 0))
      return false;
    for (const auto &sample : profile.absorbing_substrate_spectrum)
      if (!(sample.index.imag() > 0))
        return false;
  }
  return sample_layer_index(
      profile.absorbing_substrate_spectrum, profile.substrate_ior, wavelength, result);
}

/* Plane-wave tangential magnetic field = Y * tangential electric field.
 * Positive z points into the layer. Backward modes have magnetic field -Y*E. */
bool admittance(const Complex index,
                const RealVector &kx,
                const double ky,
                Matrix &Y,
                Vector &z,
                const int skip_first = -1,
                const int skip_count = 0)
{
  const int count = int(kx.size());
  Y = Matrix::Zero(2 * count, 2 * count);
  z.resize(count);
  const Complex epsilon = index * index;
  for (int m = 0; m < count; m++) {
    z[m] = outgoing_sqrt(epsilon - kx[m] * kx[m] - ky * ky);
    if (m >= skip_first && m < skip_first + skip_count) {
      Y(m, count + m) = -1.0;
      Y(count + m, m) = 1.0;
      continue;
    }
    if (std::abs(z[m]) < 1e-12) {
      return false;
    }
    Y(m, m) = -kx[m] * ky / z[m];
    Y(m, count + m) = (kx[m] * kx[m] - epsilon) / z[m];
    Y(count + m, m) = (epsilon - ky * ky) / z[m];
    Y(count + m, count + m) = kx[m] * ky / z[m];
  }
  return true;
}

bool store_scattering(const Matrix &scattering, DiffractionGratingBlock &block, std::string &error)
{
  if (!scattering.allFinite()) {
    block = {};
    error = "Nonfinite grating scattering matrix";
    return false;
  }
  Eigen::SelfAdjointEigenSolver<Matrix> power(scattering.adjoint() * scattering,
                                              Eigen::EigenvaluesOnly);
  if (power.info() != Eigen::Success) {
    block = {};
    error = "Grating power bound eigensystem did not converge";
    return false;
  }
  block.minimum_power_gain = power.eigenvalues().minCoeff();
  block.maximum_power_gain = power.eigenvalues().maxCoeff();
  block.matrix.resize(size_t(scattering.rows()) * scattering.cols());
  for (int r = 0; r < scattering.rows(); r++) {
    for (int c = 0; c < scattering.cols(); c++) {
      block.matrix[size_t(r) * scattering.cols() + c] = scattering(r, c);
    }
  }
  return true;
}

void flux_jones(const Matrix &electric,
                const int m,
                const double kx,
                const double ky,
                const double q,
                const double index,
                const double incident_flux,
                const double normal_sign,
                const double fallback_azimuth,
                std::array<Complex, 4> &jones)
{
  jones.fill(Complex(0.0));
  if (!(q > 0.0)) {
    return;
  }
  const int count = int(electric.rows() / 2);
  const double transverse = std::hypot(kx, ky);
  const double cx = transverse > 0.0 ? kx / transverse : std::cos(fallback_azimuth);
  const double sy = transverse > 0.0 ? ky / transverse : std::sin(fallback_azimuth);
  const double factor = std::sqrt(q / incident_flux);
  for (int input = 0; input < 2; input++) {
    const Complex ex = electric(m, input), ey = electric(count + m, input);
    jones[input] = factor * (-sy * ex + cx * ey);
    jones[2 + input] = factor * (cx * ex + sy * ey) / (normal_sign * q / index);
  }
}
}  // namespace

bool diffraction_grating_restore_lossless_reference(DiffractionGratingBlock &block,
                                                   std::string &error)
{
  error.clear();
  const size_t ports = block.ports.size();
  if (ports == 0 || ports > 2050 || block.matrix.size() != 4 * ports * ports) {
    error = "Invalid lossless reference dimensions";
    return false;
  }
  const int channels = int(2 * ports);
  Matrix original(channels, channels);
  for (int r = 0; r < channels; ++r)
    for (int c = 0; c < channels; ++c)
      original(r, c) = block.matrix[size_t(r) * channels + c];
  const Matrix identity = Matrix::Identity(channels, channels);
  const double residual = (original.adjoint() * original - identity).norm();
  if (!original.allFinite() || !std::isfinite(residual) || residual > 1e-4) {
    std::ostringstream message;
    message << std::setprecision(17)
            << "Lossless reference exceeds bounded roundoff correction: residual=" << residual;
    error = message.str();
    return false;
  }
  /* Two Newton-Schulz polar iterations in host double precision restore the
   * unitary constraint of the final float GPU response. Bounds below prevent
   * this from hiding a materially inaccurate or absorbing response. */
  Matrix corrected = original;
  for (int i = 0; i < 2; ++i)
    corrected = (0.5 * corrected *
                 (3.0 * identity - corrected.adjoint() * corrected)).eval();
  if (!corrected.allFinite() ||
      (corrected.adjoint() * corrected - identity).norm() > 1e-12 ||
      (corrected - original).cwiseAbs().maxCoeff() > 1e-4)
  {
    error = "Lossless reference roundoff correction failed";
    return false;
  }
  for (int r = 0; r < channels; ++r)
    for (int c = 0; c < channels; ++c)
      block.matrix[size_t(r) * channels + c] = corrected(r, c);
  return true;
}

bool diffraction_grating_sample_indices(const DiffractionGratingProfile &profile,
                                        const double wavelength,
                                        Complex &ridge, Complex &groove, Complex &substrate,
                                        std::string &error)
{
  ridge = groove = substrate = {};
  error.clear();
  Complex r, g, s;
  if (!(wavelength > 0) || !std::isfinite(wavelength) ||
      !sample_layer_index(profile.ridge_spectrum, profile.ridge_ior, wavelength, r) ||
      !sample_layer_index(profile.groove_spectrum, profile.groove_ior, wavelength, g)) {
    error = "Invalid or out-of-range grating layer optical-constant spectrum";
    return false;
  }
  if (!sample_substrate_index(profile, wavelength, s)) {
    error = "Invalid or out-of-range absorbing substrate spectrum";
    return false;
  }
  ridge = r; groove = g; substrate = s;
  return true;
}

static bool diffraction_grating_solve_impl(const DiffractionGratingProfile &profile,
                                           const double wavelength,
                                           const double angle,
                                           const double azimuth,
                                           const int half_orders,
                                           DiffractionGratingResponse &response,
                                           std::string &error,
                                           DiffractionGratingBlock *block,
                                           const double base_kx,
                                           const double ky,
                                           const int reference_orders = -1)
{
  response = {};
  if (block) {
    *block = {};
  }
  error.clear();
  if (!(std::isfinite(profile.pitch) && profile.pitch > 0.0 && std::isfinite(profile.depth) &&
        profile.depth >= 0.0 && profile.duty >= 0.0 && profile.duty <= 1.0 &&
        std::isfinite(profile.incident_ior) && profile.incident_ior > 0.0 &&
        valid_index(profile.ridge_ior) && valid_index(profile.groove_ior) &&
        valid_index(profile.substrate_ior) && std::isfinite(wavelength) && wavelength > 0.0 &&
        std::isfinite(angle) && std::abs(angle) < 0.5 * pi && std::isfinite(azimuth) &&
        half_orders >= 1 && half_orders <= 512 && std::isfinite(base_kx) && std::isfinite(ky) &&
        reference_orders >= -1 && reference_orders <= half_orders &&
        (reference_orders < 0 || block)))
  {
    error = "Invalid grating profile, incidence, wavelength or Fourier truncation";
    return false;
  }
  Complex ridge_index, groove_index, substrate_index;
  if (!diffraction_grating_sample_indices(profile, wavelength, ridge_index, groove_index,
                                          substrate_index, error)) {
    return false;
  }
  const int count = 2 * half_orders + 1;
  const int modes = 2 * count;
  const double ni = profile.incident_ior;
  RealVector kx(count);
  for (int m = 0; m < count; m++) {
    kx[m] = base_kx + (m - half_orders) * wavelength / profile.pitch;
  }
  /* A full scattering block must not silently omit propagating channels.
   * The interval of propagating kx is contiguous; checking both first omitted
   * orders and that the retained window straddles it catches displaced bases. */
  const bool lossless_substrate = substrate_index.imag() == 0.0 && substrate_index.real() > 0.0;
  const double external_index = lossless_substrate ? std::max(ni, substrate_index.real()) : ni;
  const double radius_squared = external_index * external_index - ky * ky;
  if (radius_squared > 0.0) {
    const double radius = std::sqrt(radius_squared);
    const double step = wavelength / profile.pitch;
    const int required_window = reference_orders >= 0 ? reference_orders : half_orders;
    if (base_kx - (required_window + 1) * step > -radius ||
        base_kx + (required_window + 1) * step < radius)
    {
      error = "Fourier truncation omits propagating grating ports";
      return false;
    }
  }
  Matrix W = Matrix::Identity(modes, modes);
  Matrix V = W;
  Vector X = Vector::Ones(modes);
  /* A zero-thickness layer has no constitutive response. Any nonsingular
   * internal field basis gives the same flat-interface boundary equations;
   * use the identity to avoid an irrelevant (possibly degenerate) eigensolve. */
  if (profile.depth > 0.0) {
    const Matrix identity = Matrix::Identity(count, count);
    const Matrix K = kx.cast<Complex>().asDiagonal();
    Matrix indicator(count, count);
    for (int i = 0; i < count; i++) {
      for (int j = 0; j < count; j++) {
        const int m = i - j;
        indicator(i, j) = m == 0 ? profile.duty : std::sin(pi * m * profile.duty) / (pi * m);
      }
    }
    const Complex er = ridge_index * ridge_index;
    const Complex eg = groove_index * groove_index;
    const Matrix E = (er - eg) * indicator + eg * identity;
    const Matrix inverse_E = E.partialPivLu().solve(identity);
    const Matrix inverse_profile = (1.0 / er - 1.0 / eg) * indicator + identity / eg;
    const Matrix normal_E = inverse_profile.partialPivLu().solve(identity);
    if (!inverse_E.allFinite() || !normal_E.allFinite() || !kx.allFinite()) {
      error = "Nonfinite grating constitutive matrix";
      return false;
    }
    Matrix P(modes, modes), Q(modes, modes);
    P.topLeftCorner(count, count) = ky * K * inverse_E;
    P.topRightCorner(count, count) = identity - K * inverse_E * K;
    P.bottomLeftCorner(count, count) = ky * ky * inverse_E - identity;
    P.bottomRightCorner(count, count) = -ky * inverse_E * K;
    Q.topLeftCorner(count, count) = -ky * K;
    Q.topRightCorner(count, count) = K * K - E;
    Q.bottomLeftCorner(count, count) = normal_E - ky * ky * identity;
    Q.bottomRightCorner(count, count) = ky * K;
    const Matrix modal = P * Q;
    if (!modal.allFinite()) {
      error = "Nonfinite grating propagation matrix";
      return false;
    }
    Vector eigenvalues(modes);
    if (ky == 0.0) {
      /* At zero groove-parallel momentum, P*Q is exactly block diagonal.
       * Preserve both full Fourier systems while avoiding a coupled 2N
       * eigensolve. No near-zero angular approximation is made. */
      W.setZero();
      for (int polarization = 0; polarization < 2; polarization++) {
        const int offset = polarization * count;
        Eigen::ComplexEigenSolver<Matrix> eigen(modal.block(offset, offset, count, count));
        if (eigen.info() != Eigen::Success) {
          error = "Grating modal eigensystem did not converge";
          return false;
        }
        W.block(offset, offset, count, count) = eigen.eigenvectors();
        eigenvalues.segment(offset, count) = eigen.eigenvalues();
      }
    }
    else {
      Eigen::ComplexEigenSolver<Matrix> eigen(modal);
      if (eigen.info() != Eigen::Success) {
        error = "Grating modal eigensystem did not converge";
        return false;
      }
      W = eigen.eigenvectors();
      eigenvalues = eigen.eigenvalues();
    }
    Vector q(modes);
    for (int i = 0; i < modes; i++) {
      q[i] = outgoing_sqrt(eigenvalues[i]);
      if (std::abs(q[i]) < 1e-12) {
        error = "A degenerate layer mode requires a limiting solution";
        return false;
      }
      X[i] = std::exp(Complex(0.0, 2.0 * pi * profile.depth / wavelength) * q[i]);
    }
    V = Q * W * q.cwiseInverse().asDiagonal();
  }
  Matrix upper, lower;
  Vector upper_z, lower_z;
  const int skip_first = reference_orders >= 0 ? half_orders - reference_orders : -1;
  const int skip_count = reference_orders >= 0 ? 2 * reference_orders + 1 : 0;
  if (!admittance(ni, kx, ky, upper, upper_z, skip_first, skip_count) ||
      !admittance(substrate_index,
                  kx,
                  ky,
                  lower,
                  lower_z,
                  skip_first,
                  lossless_substrate ? skip_count : 0))
  {
    error = "An external grazing order requires a limiting solution";
    return false;
  }
  const Matrix UW = upper * W, LW = lower * W;
  Matrix boundary(2 * modes, 2 * modes);
  boundary.topLeftCorner(modes, modes) = V + UW;
  boundary.topRightCorner(modes, modes) = (-V + UW) * X.asDiagonal();
  boundary.bottomLeftCorner(modes, modes) = (V - LW) * X.asDiagonal();
  boundary.bottomRightCorner(modes, modes) = -V - LW;
  int inputs = 2;
  if (block) {
    for (int side = 0; side < (lossless_substrate ? 2 : 1); side++) {
      const Vector &z = side == 0 ? upper_z : lower_z;
      for (int m = 0; m < count; m++) {
        if (reference_orders >= 0 ? (m >= skip_first && m < skip_first + skip_count) :
                                    (z[m].imag() == 0.0 && z[m].real() > 0.0))
        {
          block->ports.push_back({m - half_orders, side == 1});
        }
      }
    }
    inputs = 2 * int(block->ports.size());
    if (inputs == 0) {
      /* The zero-dimensional far-field scattering operator is valid. */
      return true;
    }
  }
  Matrix incident = Matrix::Zero(modes, inputs);
  Matrix incident_bottom = Matrix::Zero(modes, inputs);
  if (!block) {
    incident(half_orders, 0) = -std::sin(azimuth);
    incident(count + half_orders, 0) = std::cos(azimuth);
    incident(half_orders, 1) = std::cos(angle) * std::cos(azimuth);
    incident(count + half_orders, 1) = std::cos(angle) * std::sin(azimuth);
  }
  else {
    for (int port = 0; port < int(block->ports.size()); port++) {
      const auto &channel = block->ports[port];
      const int m = channel.order + half_orders;
      if (reference_orders >= 0) {
        Matrix &field = channel.substrate ? incident_bottom : incident;
        field(m, 2 * port) = 1.0;
        field(count + m, 2 * port + 1) = 1.0;
        continue;
      }
      const double index = channel.substrate ? substrate_index.real() : ni;
      const double z = channel.substrate ? lower_z[m].real() : upper_z[m].real();
      const double dx = kx[m] / index, dy = ky / index, cosine = z / index;
      const double amplitude = 1.0 / std::sqrt(z);
      const double bxx = 1.0 - dx * dx / (1.0 + cosine);
      const double bxy = -dx * dy / (1.0 + cosine);
      const double byy = 1.0 - dy * dy / (1.0 + cosine);
      Matrix &field = channel.substrate ? incident_bottom : incident;
      field(m, 2 * port) = amplitude * bxx;
      field(count + m, 2 * port) = amplitude * bxy;
      field(m, 2 * port + 1) = amplitude * bxy;
      field(count + m, 2 * port + 1) = amplitude * byy;
    }
  }
  Matrix rhs = Matrix::Zero(2 * modes, inputs);
  rhs.topRows(modes) = 2.0 * upper * incident;
  rhs.bottomRows(modes) = -2.0 * lower * incident_bottom;
  const Matrix solution = boundary.partialPivLu().solve(rhs);
  response.boundary_residual = (boundary * solution - rhs).norm() / rhs.norm();
  if (!solution.allFinite() || !std::isfinite(response.boundary_residual) ||
      response.boundary_residual > 1e-8)
  {
    response = {};
    if (block) {
      *block = {};
    }
    error = "Grating interface solve failed its residual check";
    return false;
  }
  const Matrix A = solution.topRows(modes), B = solution.bottomRows(modes);
  const Matrix reflected = W * (A + X.asDiagonal() * B) - incident;
  const Matrix transmitted = W * (X.asDiagonal() * A + B) - incident_bottom;
  if (block) {
    Matrix scattering(inputs, inputs);
    for (int port = 0; port < int(block->ports.size()); port++) {
      const auto &channel = block->ports[port];
      const int m = channel.order + half_orders;
      const Matrix &field = channel.substrate ? transmitted : reflected;
      if (reference_orders >= 0) {
        scattering.row(2 * port) = field.row(m);
        scattering.row(2 * port + 1) = field.row(count + m);
        continue;
      }
      const double z = channel.substrate ? lower_z[m].real() : upper_z[m].real();
      const double index = channel.substrate ? substrate_index.real() : ni;
      const double dx = kx[m] / index, dy = ky / index, cosine = z / index;
      const double factor = std::sqrt(z);
      const double denominator = cosine * (1.0 + cosine);
      const double ixx = 1.0 + dx * dx / denominator;
      const double ixy = dx * dy / denominator;
      const double iyy = 1.0 + dy * dy / denominator;
      for (int c = 0; c < inputs; c++) {
        scattering(2 * port, c) = factor * (ixx * field(m, c) + ixy * field(count + m, c));
        scattering(2 * port + 1, c) = factor * (ixy * field(m, c) + iyy * field(count + m, c));
      }
    }
    block->boundary_residual = response.boundary_residual;
    return store_scattering(scattering, *block, error);
  }
  const Matrix reflected_h = upper * reflected, transmitted_h = lower * transmitted;
  const double incident_flux = ni * std::cos(angle);
  response.layer_absorption = {1.0, 1.0};
  response.has_transmission_jones = substrate_index.imag() == 0.0 && substrate_index.real() > 0.0;
  response.orders.resize(count);
  for (int m = 0; m < count; m++) {
    DiffractionGratingOrder &order = response.orders[m];
    order.order = m - half_orders;
    for (int input = 0; input < 2; input++) {
      order.reflection[input] =
          std::real(reflected(m, input) * std::conj(reflected_h(count + m, input)) -
                    reflected(count + m, input) * std::conj(reflected_h(m, input))) /
          incident_flux;
      order.substrate_flux[input] =
          std::real(transmitted(m, input) * std::conj(transmitted_h(count + m, input)) -
                    transmitted(count + m, input) * std::conj(transmitted_h(m, input))) /
          incident_flux;
      response.layer_absorption[input] -= order.reflection[input] + order.substrate_flux[input];
    }
    flux_jones(reflected,
               m,
               kx[m],
               ky,
               upper_z[m].real(),
               ni,
               incident_flux,
               -1.0,
               azimuth,
               order.reflection_jones);
    order.transmission_jones.fill(Complex(0.0));
    if (response.has_transmission_jones) {
      flux_jones(transmitted,
                 m,
                 kx[m],
                 ky,
                 lower_z[m].real(),
                 substrate_index.real(),
                 incident_flux,
                 1.0,
                 azimuth,
                 order.transmission_jones);
    }
  }
  return true;
}

bool diffraction_grating_solve(const DiffractionGratingProfile &profile,
                               const double wavelength,
                               const double angle,
                               const double azimuth,
                               const int half_orders,
                               DiffractionGratingResponse &response,
                               std::string &error)
{
  return diffraction_grating_solve_impl(profile,
                                        wavelength,
                                        angle,
                                        azimuth,
                                        half_orders,
                                        response,
                                        error,
                                        nullptr,
                                        profile.incident_ior * std::sin(angle) * std::cos(azimuth),
                                        profile.incident_ior * std::sin(angle) *
                                            std::sin(azimuth));
}

bool diffraction_grating_solve_block(const DiffractionGratingProfile &profile,
                                     const double wavelength,
                                     const double angle,
                                     const double azimuth,
                                     const int half_orders,
                                     DiffractionGratingBlock &block,
                                     std::string &error)
{
  DiffractionGratingResponse response;
  return diffraction_grating_solve_impl(profile,
                                        wavelength,
                                        angle,
                                        azimuth,
                                        half_orders,
                                        response,
                                        error,
                                        &block,
                                        profile.incident_ior * std::sin(angle) * std::cos(azimuth),
                                        profile.incident_ior * std::sin(angle) *
                                            std::sin(azimuth));
}

bool diffraction_grating_solve_bloch(const DiffractionGratingProfile &profile,
                                     const double wavelength,
                                     const double kx,
                                     const double ky,
                                     const int half_orders,
                                     DiffractionGratingBlock &block,
                                     std::string &error)
{
  DiffractionGratingResponse response;
  return diffraction_grating_solve_impl(
      profile, wavelength, 0.0, 0.0, half_orders, response, error, &block, kx, ky);
}

int diffraction_grating_reference_order_bound(const DiffractionGratingProfile &profile,
                                              const DiffractionGratingCacheOptions &options,
                                              std::string &error)
{
  error.clear();
  const auto &bounds = options.bounds;
  if (!(std::isfinite(profile.pitch) && profile.pitch > 0 &&
        std::isfinite(profile.incident_ior) && profile.incident_ior > 0 &&
        std::isfinite(options.cutoff_margin) && options.cutoff_margin >= 0 &&
        options.half_orders >= 1 && options.half_orders <= 512)) {
    error = "Invalid grating reference-order bound parameters";
    return -1;
  }
  for (int axis = 0; axis < 3; axis++) {
    if (!std::isfinite(bounds.lower[axis]) || !std::isfinite(bounds.upper[axis]) ||
        !(bounds.lower[axis] < bounds.upper[axis])) {
      error = "Invalid grating reference-order bound domain";
      return -1;
    }
  }
  if (bounds.lower[0] < -0.5 || bounds.upper[0] > 0.5 || bounds.lower[2] <= 0) {
    error = "Invalid grating reference-order bound domain";
    return -1;
  }
  Complex substrate;
  if (!sample_substrate_index(profile, bounds.lower[2], substrate) ||
      !sample_substrate_index(profile, bounds.upper[2], substrate)) {
    error = "Invalid grating exterior index for reference-order bound";
    return -1;
  }
  const double maximum_index = substrate.imag() == 0 ?
                                   std::max(profile.incident_ior, substrate.real()) :
                                   profile.incident_ior;
  /* q^2 >= -margin requires |m+b| <= pitch/lambda * sqrt(n^2+margin).
   * Dropping ky^2 enlarges this bound, so all conical directions are covered. */
  double limit = profile.pitch / bounds.lower[2] *
                     std::sqrt(maximum_index * maximum_index + options.cutoff_margin) +
                 std::max(std::abs(bounds.lower[0]), std::abs(bounds.upper[0]));
  limit += 64 * std::numeric_limits<double>::epsilon() * (1 + limit);
  if (!std::isfinite(limit) || std::floor(limit) > options.half_orders) {
    error = "Grating reference-order domain exceeds the internal Fourier truncation";
    return -1;
  }
  return int(std::floor(limit));
}

bool diffraction_grating_solve_reference(const DiffractionGratingProfile &profile,
                                         const double wavelength,
                                         const double kx,
                                         const double ky,
                                         const int half_orders,
                                         const int retained_half_orders,
                                         DiffractionGratingBlock &reference,
                                         std::string &error)
{
  if (retained_half_orders < 0) {
    reference = {};
    error = "Invalid reference-port truncation";
    return false;
  }
  DiffractionGratingResponse response;
  return diffraction_grating_solve_impl(profile,
                                        wavelength,
                                        0.0,
                                        0.0,
                                        half_orders,
                                        response,
                                        error,
                                        &reference,
                                        kx,
                                        ky,
                                        retained_half_orders);
}

static bool diffraction_grating_match_ports(const DiffractionGratingProfile &profile,
                                            const double wavelength,
                                            const double base_kx,
                                            const double ky,
                                            const DiffractionGratingBlock &reference,
                                            const bool is_hybrid,
                                            const std::span<const unsigned char> source_flags,
                                            const std::span<const unsigned char> keep_flags,
                                            std::vector<unsigned char> *output_flags,
                                            DiffractionGratingBlock &physical,
                                            std::string &error)
{
  physical = {};
  error.clear();
  if (output_flags) {
    output_flags->clear();
  }
  Complex substrate_index;
  if (!sample_substrate_index(profile, wavelength, substrate_index)) {
    error = "Invalid or out-of-range absorbing substrate spectrum";
    return false;
  }
  const int ports = int(reference.ports.size()), channels = 2 * ports;
  if ((!is_hybrid && ports == 0) || ports > 2050 ||
      reference.matrix.size() != size_t(channels) * channels || !(wavelength > 0.0) ||
      !std::isfinite(wavelength) || !(profile.pitch > 0.0) || !std::isfinite(profile.pitch) ||
      !(profile.incident_ior > 0.0) || !valid_index(Complex(profile.incident_ior)) ||
      !valid_index(profile.substrate_ior) || !std::isfinite(base_kx) || !std::isfinite(ky))
  {
    error = "Invalid grating reference operator or matching parameters";
    return false;
  }
  if ((is_hybrid && source_flags.size() != size_t(ports)) ||
      (!keep_flags.empty() && keep_flags.size() != size_t(ports)))
  {
    error = "Invalid hybrid reference mask size";
    return false;
  }
  for (int i = 0; i < ports; i++) {
    if ((!source_flags.empty() && source_flags[i] > 1) ||
        (!keep_flags.empty() && keep_flags[i] > 1))
    {
      error = "Invalid hybrid reference mask value";
      return false;
    }
  }
  const int sides = profile.substrate_ior.imag() == 0.0 ? 2 : 1;
  const double step = wavelength / profile.pitch;
  if (!(step > 0.0) || !std::isfinite(step)) {
    error = "Invalid reference-port wavelength scale";
    return false;
  }
  if (!is_hybrid) {
    const int first = reference.ports.front().order;
    if (first < -512 || first > 0 || ports != sides * (1 - 2 * first)) {
      error = "Invalid reference-port ranges";
      return false;
    }
    const int side_ports = 1 - 2 * first;
    for (int i = 0; i < ports; i++) {
      if (reference.ports[i].order != first + i % side_ports ||
          reference.ports[i].substrate != (i >= side_ports))
      {
        error = "Reference ports must be contiguous symmetric upper then lower ranges";
        return false;
      }
    }
  }
  else {
    for (int i = 0; i < ports; i++) {
      const auto &p = reference.ports[i];
      if (p.order < -512 || p.order > 512 || (p.substrate && sides == 1) ||
          (i > 0 && (p.substrate < reference.ports[i - 1].substrate ||
                     (p.substrate == reference.ports[i - 1].substrate &&
                      p.order <= reference.ports[i - 1].order))))
      {
        error = "Hybrid ports must be distinct sorted upper then lower orders";
        return false;
      }
    }
  }
  for (int side = 0; side < sides; side++) {
    const double index = side ? profile.substrate_ior.real() : profile.incident_ior;
    const double radius2 = index * index - ky * ky;
    if (radius2 <= 0.0) {
      continue;
    }
    const double lower = (-std::sqrt(radius2) - base_kx) / step;
    const double upper = (std::sqrt(radius2) - base_kx) / step;
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower < -513 || upper > 513) {
      error = "Reference operator does not cover all propagating query channels";
      return false;
    }
    for (int order = int(std::floor(lower)) + 1; order < upper; order++) {
      const auto found = std::find_if(
          reference.ports.begin(), reference.ports.end(), [=](const DiffractionGratingPort &p) {
            return p.order == order && p.substrate == bool(side);
          });
      if (found == reference.ports.end()) {
        error = "Reference operator does not cover all propagating query channels";
        return false;
      }
    }
  }
  Matrix S(channels, channels), R = Matrix::Zero(channels, channels);
  for (int row = 0; row < channels; row++) {
    for (int col = 0; col < channels; col++) {
      S(row, col) = reference.matrix[size_t(row) * channels + col];
    }
  }
  if (!S.allFinite()) {
    error = "Nonfinite reference operator";
    return false;
  }
  std::vector<int> physical_indices;
  std::vector<Eigen::Matrix2d> couplings;
  for (int port = 0; port < ports; port++) {
    const auto &p = reference.ports[port];
    if (p.substrate &&
        (profile.substrate_ior.imag() != 0.0 || profile.substrate_ior.real() <= 0.0))
    {
      error = "Absorbing substrate cannot have external reference ports";
      return false;
    }
    const double index = p.substrate ? profile.substrate_ior.real() : profile.incident_ior;
    const double kx = base_kx + p.order * wavelength / profile.pitch;
    const double transverse2 = kx * kx + ky * ky;
    const Complex q = outgoing_sqrt(Complex(index * index - transverse2));
    const bool is_reference = !is_hybrid || source_flags[port];
    const bool keep = !keep_flags.empty() && keep_flags[port];
    if (keep && !is_reference) {
      physical = {};
      error = "A physical port cannot be promoted to a reference port";
      return false;
    }
    if (keep || !is_reference) {
      if (!keep && !(q.imag() == 0.0 && q.real() > 0.0)) {
        physical = {};
        error = "Hybrid physical port crossed a diffraction cutoff";
        return false;
      }
      physical_indices.push_back(port);
      physical.ports.push_back(p);
      couplings.push_back(Eigen::Matrix2d::Identity());
      if (output_flags) {
        output_flags->push_back(keep ? 1 : 0);
      }
      continue;
    }
    Eigen::Matrix2d parallel = Eigen::Matrix2d::Zero();
    if (transverse2 > 0.0) {
      parallel << kx * kx, kx * ky, kx * ky, ky * ky;
      parallel /= transverse2;
    }
    const Eigen::Matrix2d perpendicular = Eigen::Matrix2d::Identity() - parallel;
    const Complex r_te = (1.0 - q) / (1.0 + q);
    const Complex r_tm = (q - index * index) / (q + index * index);
    R.block<2, 2>(2 * port, 2 * port) = r_te * perpendicular + r_tm * parallel;
    if (q.imag() == 0.0 && q.real() > 0.0) {
      const double root = std::sqrt(q.real());
      couplings.push_back(2.0 * root / (1.0 + q.real()) * perpendicular +
                          2.0 * index * root / (q.real() + index * index) * parallel);
      physical_indices.push_back(port);
      physical.ports.push_back(p);
      if (output_flags) {
        output_flags->push_back(0);
      }
    }
  }
  const int inputs = 2 * int(physical_indices.size());
  if (inputs == 0) {
    return true;
  }
  Matrix T = Matrix::Zero(channels, inputs), direct = Matrix::Zero(inputs, inputs);
  for (int i = 0; i < int(physical_indices.size()); i++) {
    const int port = physical_indices[i];
    T.block<2, 2>(2 * port, 2 * i) = couplings[i].cast<Complex>();
    direct.block<2, 2>(2 * i, 2 * i) = -R.block<2, 2>(2 * port, 2 * port);
  }
  std::vector<int> active;
  for (int port = 0; port < ports; port++) {
    if (R.block<2, 2>(2 * port, 2 * port).squaredNorm() > 0.0) {
      active.push_back(2 * port);
      active.push_back(2 * port + 1);
    }
  }
  const Matrix B = S * T;
  Matrix outgoing = B;
  if (!active.empty()) {
    const int count = int(active.size());
    Matrix feedback(channels, count), boundary(count, count), rhs(count, inputs);
    for (int j = 0; j < count; j++) {
      feedback.col(j) = S.col(active[j] / 2 * 2) * R(active[j] / 2 * 2, active[j]) +
                        S.col(active[j] / 2 * 2 + 1) * R(active[j] / 2 * 2 + 1, active[j]);
    }
    for (int i = 0; i < count; i++) {
      rhs.row(i) = B.row(active[i]);
      for (int j = 0; j < count; j++) {
        boundary(i, j) = Complex(i == j ? 1.0 : 0.0) - feedback(active[i], j);
      }
    }
    const Matrix solved = boundary.partialPivLu().solve(rhs);
    physical.boundary_residual = (boundary * solved - rhs).norm() /
                                 std::max(rhs.norm(), std::numeric_limits<double>::min());
    if (!solved.allFinite() || !std::isfinite(physical.boundary_residual) ||
        physical.boundary_residual > 1e-8)
    {
      physical = {};
      error = "Reference-port matching failed its residual check";
      return false;
    }
    outgoing += feedback * solved;
  }
  return store_scattering(direct + T.transpose() * outgoing, physical, error);
}

bool diffraction_grating_match_reference(const DiffractionGratingProfile &profile,
                                         const double wavelength,
                                         const double kx,
                                         const double ky,
                                         const DiffractionGratingBlock &reference,
                                         DiffractionGratingBlock &physical,
                                         std::string &error)
{
  return diffraction_grating_match_ports(
      profile, wavelength, kx, ky, reference, false, {}, {}, nullptr, physical, error);
}

bool diffraction_grating_prepare_hybrid(const DiffractionGratingProfile &profile,
                                        const double wavelength,
                                        const double kx,
                                        const double ky,
                                        const DiffractionGratingBlock &reference,
                                        const std::span<const unsigned char> keep_reference,
                                        DiffractionGratingHybrid &hybrid,
                                        std::string &error)
{
  hybrid = {};
  if (keep_reference.size() != reference.ports.size()) {
    error = "Hybrid preparation requires one flag per reference port";
    return false;
  }
  if (!diffraction_grating_match_ports(profile,
                                       wavelength,
                                       kx,
                                       ky,
                                       reference,
                                       false,
                                       {},
                                       keep_reference,
                                       &hybrid.is_reference,
                                       hybrid.scattering,
                                       error))
  {
    hybrid = {};
    return false;
  }
  return true;
}

bool diffraction_grating_match_hybrid(const DiffractionGratingProfile &profile,
                                      const double wavelength,
                                      const double kx,
                                      const double ky,
                                      const DiffractionGratingHybrid &hybrid,
                                      DiffractionGratingBlock &physical,
                                      std::string &error)
{
  if (hybrid.is_reference.size() != hybrid.scattering.ports.size()) {
    physical = {};
    error = "Invalid hybrid reference mask";
    return false;
  }
  return diffraction_grating_match_ports(profile,
                                         wavelength,
                                         kx,
                                         ky,
                                         hybrid.scattering,
                                         true,
                                         hybrid.is_reference,
                                         {},
                                         nullptr,
                                         physical,
                                         error);
}

using ReferenceSolve =
    std::function<bool(double, double, double, DiffractionGratingBlock &, std::string &)>;

static bool prepare_cell_impl(const DiffractionGratingProfile &profile,
                              const DiffractionGratingCellBounds &bounds,
                              const double cutoff_margin,
                              DiffractionGratingCell &cell,
                              std::string &error,
                              const ReferenceSolve &solve_reference)
{
  cell = {};
  error.clear();
  for (int axis = 0; axis < 3; axis++) {
    if (!std::isfinite(bounds.lower[axis]) || !std::isfinite(bounds.upper[axis]) ||
        !(bounds.lower[axis] < bounds.upper[axis]))
    {
      error = "Invalid grating cell bounds";
      return false;
    }
  }
  if (!(bounds.lower[2] > 0.0) || bounds.lower[0] < -0.5 || bounds.upper[0] > 0.5 ||
      !(cutoff_margin >= 0.0) || !std::isfinite(cutoff_margin))
  {
    error = "Invalid grating cell wavelength, Bloch range or cutoff margin";
    return false;
  }
  auto squared = [](const double lo, const double hi) {
    return std::pair<double, double>{lo <= 0.0 && hi >= 0.0 ? 0.0 : std::min(lo * lo, hi * hi),
                                     std::max(lo * lo, hi * hi)};
  };
  const auto y2 = squared(bounds.lower[1], bounds.upper[1]);
  DiffractionGratingCell result{};
  result.bounds = bounds;
  std::vector<unsigned char> keep;
  for (int corner = 0; corner < 8; corner++) {
    const double bloch = corner & 1 ? bounds.upper[0] : bounds.lower[0];
    const double ky = corner & 2 ? bounds.upper[1] : bounds.lower[1];
    const double wavelength = corner & 4 ? bounds.upper[2] : bounds.lower[2];
    const double kx = bloch * wavelength / profile.pitch;
    DiffractionGratingBlock reference;
    if (!solve_reference(wavelength, kx, ky, reference, error)) {
      return false;
    }
    if (corner == 0) {
      keep.resize(reference.ports.size());
      for (size_t port = 0; port < keep.size(); port++) {
        const auto &p = reference.ports[port];
        const double n = p.substrate ? profile.substrate_ior.real() : profile.incident_ior;
        const std::array<double, 4> values{
            (bounds.lower[0] + p.order) * bounds.lower[2] / profile.pitch,
            (bounds.upper[0] + p.order) * bounds.lower[2] / profile.pitch,
            (bounds.lower[0] + p.order) * bounds.upper[2] / profile.pitch,
            (bounds.upper[0] + p.order) * bounds.upper[2] / profile.pitch};
        const auto extrema = std::minmax_element(values.begin(), values.end());
        const auto x2 = squared(*extrema.first, *extrema.second);
        const double lower = n * n - x2.second - y2.second;
        const double upper = n * n - x2.first - y2.first;
        /* Roundoff guard covers the few products and subtractions in these
         * interval endpoints. It expands the reference set, never removes it. */
        const double guard = 32.0 * std::numeric_limits<double>::epsilon() *
                             (n * n + x2.second + y2.second + cutoff_margin);
        keep[port] = lower <= cutoff_margin + guard && upper >= -cutoff_margin - guard;
        result.feedback_channels += 2 * keep[port];
      }
    }
    if (!diffraction_grating_prepare_hybrid(
            profile, wavelength, kx, ky, reference, keep, result.corners[corner], error))
    {
      return false;
    }
    if (corner > 0) {
      const auto &first = result.corners[0];
      const auto &current = result.corners[corner];
      if (first.is_reference != current.is_reference ||
          first.scattering.ports.size() != current.scattering.ports.size())
      {
        error = "Grating cell has inconsistent hybrid port topology";
        return false;
      }
      for (size_t p = 0; p < first.scattering.ports.size(); p++) {
        if (first.scattering.ports[p].order != current.scattering.ports[p].order ||
            first.scattering.ports[p].substrate != current.scattering.ports[p].substrate)
        {
          error = "Grating cell has inconsistent hybrid port topology";
          return false;
        }
      }
    }
  }
  cell = std::move(result);
  return true;
}

bool diffraction_grating_prepare_cell(const DiffractionGratingProfile &profile,
                                      const DiffractionGratingCellBounds &bounds,
                                      const int half_orders,
                                      const int retained_half_orders,
                                      const double cutoff_margin,
                                      DiffractionGratingCell &cell,
                                      std::string &error)
{
  const ReferenceSolve solve = [&](double wavelength,
                                   double kx,
                                   double ky,
                                   DiffractionGratingBlock &block,
                                   std::string &message) {
    return diffraction_grating_solve_reference(
        profile, wavelength, kx, ky, half_orders, retained_half_orders, block, message);
  };
  return prepare_cell_impl(profile, bounds, cutoff_margin, cell, error, solve);
}

bool diffraction_grating_pack_cell(const DiffractionGratingCell &cell,
                                   DiffractionGratingPackedCell &packed,
                                   std::string &error)
{
  packed = {};
  error.clear();
  for (int axis = 0; axis < 3; axis++) {
    if (!std::isfinite(cell.bounds.lower[axis]) || !std::isfinite(cell.bounds.upper[axis]) ||
        !(cell.bounds.upper[axis] > cell.bounds.lower[axis]))
    {
      error = "Invalid packed grating cell bounds";
      return false;
    }
  }
  DiffractionGratingPackedCell result;
  result.bounds = cell.bounds;
  result.ports = cell.corners[0].scattering.ports;
  const size_t ports = result.ports.size(), channels = 2 * ports;
  if (ports == 0 || ports > size_t(std::numeric_limits<int>::max() / 2)) {
    error = "Invalid packed grating cell port count";
    return false;
  }
  for (int corner = 0; corner < 8; corner++) {
    const auto &source = cell.corners[corner];
    if (source.scattering.ports.size() != ports || source.is_reference.size() != ports ||
        source.scattering.matrix.size() != channels * channels ||
        source.is_reference != cell.corners[0].is_reference)
    {
      error = "Inconsistent packed grating cell topology";
      return false;
    }
    for (size_t p = 0; p < ports; p++) {
      const auto &port = source.scattering.ports[p];
      if (port.order != result.ports[p].order || port.substrate != result.ports[p].substrate ||
          source.is_reference[p] > 1 ||
          (p > 0 && (int(port.substrate) < int(result.ports[p - 1].substrate) ||
                     (port.substrate == result.ports[p - 1].substrate &&
                      port.order <= result.ports[p - 1].order))))
      {
        error = "Invalid packed grating cell port mapping";
        return false;
      }
      if (corner == 0 && source.is_reference[p])
        result.active_ports.push_back(int(p));
    }
    for (const auto value : source.scattering.matrix) {
      const float real = float(value.real()), imag = float(value.imag());
      if (!std::isfinite(real) || !std::isfinite(imag)) {
        error = "Grating cell matrix cannot be represented in float";
        return false;
      }
      result.matrices.push_back(make_float2(real, imag));
    }
  }
  if (2 * int(result.active_ports.size()) != cell.feedback_channels) {
    error = "Inconsistent packed grating cell feedback count";
    return false;
  }
  packed = std::move(result);
  return true;
}

static bool prepare_quadratic_chart_impl(const DiffractionGratingProfile &profile,
                                         const DiffractionGratingCell &cell,
                                         DiffractionGratingChartCell &chart,
                                         std::string &error,
                                         const ReferenceSolve &solve)
{
  chart = {};
  error.clear();
  const auto &first = cell.corners[0];
  if (first.scattering.ports.empty() || first.is_reference.size() != first.scattering.ports.size())
  {
    error = "Invalid quadratic grating topology";
    return false;
  }
  for (int axis = 0; axis < 3; axis++) {
    if (!std::isfinite(cell.bounds.lower[axis]) || !std::isfinite(cell.bounds.upper[axis]) ||
        !(cell.bounds.upper[axis] > cell.bounds.lower[axis]))
    {
      error = "Invalid quadratic grating bounds";
      return false;
    }
  }
  std::vector<DiffractionGratingBlock> samples;
  for (int node = 0; node < 27; node++) {
    const int digits[3] = {node % 3, (node / 3) % 3, node / 9};
    DiffractionGratingHybrid hybrid;
    if (digits[0] != 1 && digits[1] != 1 && digits[2] != 1) {
      hybrid = cell.corners[(digits[0] / 2) | ((digits[1] / 2) << 1) | ((digits[2] / 2) << 2)];
    }
    else {
      std::array<double, 3> q;
      for (int axis = 0; axis < 3; axis++)
        q[axis] = cell.bounds.lower[axis] +
                  0.5 * digits[axis] * (cell.bounds.upper[axis] - cell.bounds.lower[axis]);
      const double x = q[0] * q[2] / profile.pitch;
      DiffractionGratingBlock reference;
      if (!solve(q[2], x, q[1], reference, error))
        return false;
      std::vector<unsigned char> keep(reference.ports.size(), 0);
      for (size_t i = 0; i < keep.size(); i++)
        for (size_t j = 0; j < first.scattering.ports.size(); j++)
          if (reference.ports[i].order == first.scattering.ports[j].order &&
              reference.ports[i].substrate == first.scattering.ports[j].substrate)
            keep[i] = first.is_reference[j];
      if (!diffraction_grating_prepare_hybrid(
              profile, q[2], x, q[1], reference, keep, hybrid, error))
        return false;
    }
    if (hybrid.is_reference != first.is_reference ||
        hybrid.scattering.ports.size() != first.scattering.ports.size())
    {
      error = "Inconsistent quadratic grating topology";
      return false;
    }
    for (size_t j = 0; j < first.scattering.ports.size(); j++)
      if (hybrid.scattering.ports[j].order != first.scattering.ports[j].order ||
          hybrid.scattering.ports[j].substrate != first.scattering.ports[j].substrate)
      {
        error = "Inconsistent quadratic grating port mapping";
        return false;
      }
    samples.push_back(std::move(hybrid.scattering));
  }
  DiffractionGratingChartCell result;
  result.bounds = cell.bounds;
  result.is_reference = first.is_reference;
  result.degree = 2;
  double phase;
  if (!diffraction_grating_choose_chart(samples, phase, result.corners, error))
    return false;
  auto &controls = result.corners;
  for (int axis = 0, stride = 1; axis < 3; axis++, stride *= 3)
    for (int i = 0; i < 27; i++)
      if ((i / stride) % 3 == 0)
        for (size_t j = 0; j < controls[i].matrix.size(); j++)
          controls[i + stride].matrix[j] = 2.0 * controls[i + stride].matrix[j] -
                                           0.5 * (controls[i].matrix[j] +
                                                  controls[i + 2 * stride].matrix[j]);
  const int channels = 2 * int(first.scattering.ports.size());
  for (auto &control : controls) {
    Matrix matrix(channels, channels);
    for (int r = 0; r < channels; r++)
      for (int c = 0; c < channels; c++)
        matrix(r, c) = control.matrix[r * channels + c];
    if (!matrix.allFinite()) {
      error = "Nonfinite quadratic chart control";
      return false;
    }
    Eigen::SelfAdjointEigenSolver<Matrix> eigen(0.5 * (matrix + matrix.adjoint()),
                                                Eigen::EigenvaluesOnly);
    if (eigen.info() != Eigen::Success) {
      error = "Quadratic chart passivity check failed";
      return false;
    }
    control.minimum_dissipation = eigen.eigenvalues().minCoeff();
    control.frobenius_norm = matrix.norm();
    /* Absolute numerical tolerance, not an energy-gain allowance scaled by an
     * arbitrarily ill-conditioned chart. No control is clamped or projected. */
    if (control.minimum_dissipation < -1e-9) {
      error = "Quadratic chart has a nonpassive control matrix";
      return false;
    }
  }
  chart = std::move(result);
  return true;
}

bool diffraction_grating_prepare_quadratic_chart(const DiffractionGratingProfile &profile,
                                                 const DiffractionGratingCell &cell,
                                                 const int half_orders,
                                                 const int retained_half_orders,
                                                 DiffractionGratingChartCell &chart,
                                                 std::string &error)
{
  const ReferenceSolve solve = [&](double wavelength,
                                   double x,
                                   double y,
                                   DiffractionGratingBlock &reference,
                                   std::string &message) {
    return diffraction_grating_solve_reference(
        profile, wavelength, x, y, half_orders, retained_half_orders, reference, message);
  };
  return prepare_quadratic_chart_impl(profile, cell, chart, error, solve);
}

bool diffraction_grating_build_cache(const DiffractionGratingProfile &profile,
                                     const DiffractionGratingCacheOptions &options,
                                     DiffractionGratingCache &cache,
                                     DiffractionGratingCacheStats &stats,
                                     std::string &error)
{
  cache = {};
  stats = {};
  error.clear();
  if (!(options.tolerance > 0.0) || !std::isfinite(options.tolerance) ||
      !std::isfinite(options.complex_tolerance) || options.complex_tolerance < 0 ||
      (options.complex_tolerance > 0 && !options.allow_chart_cells) || options.maximum_depth < 0 ||
      options.maximum_depth > 60 || options.maximum_nodes == 0 || options.validation_workers < 1 ||
      options.validation_workers > 64 ||
      options.maximum_nodes > size_t(std::numeric_limits<int>::max()))
  {
    error = "Invalid grating cache limits";
    return false;
  }
  for (int axis = 0; axis < 3; axis++) {
    const double lo = options.bounds.lower[axis], hi = options.bounds.upper[axis];
    if (!std::isfinite(lo) || !std::isfinite(hi) || !(hi > lo) || double(float(lo)) != lo ||
        double(float(hi)) != hi)
    {
      error = "Cache bounds must be finite increasing float-representable values";
      return false;
    }
  }
  /* Validate coverage before topology pruning: a domain with no propagating
   * ports may otherwise skip every modal solve and hide an invalid table. */
  for (const double wavelength : {options.bounds.lower[2], options.bounds.upper[2]}) {
    Complex index;
    if (!sample_layer_index(profile.ridge_spectrum, profile.ridge_ior, wavelength, index) ||
        !sample_layer_index(profile.groove_spectrum, profile.groove_ior, wavelength, index) ||
        !sample_substrate_index(profile, wavelength, index))
    {
      error = "Layer optical-constant spectra do not cover the cache wavelength domain";
      return false;
    }
  }
  auto domain = options.bounds;
  if (options.mirror_symmetry) {
    if (domain.lower[0] != -domain.upper[0] || domain.lower[1] != -domain.upper[1]) {
      error = "Mirror cache requires origin-symmetric Bloch and conical bounds";
      return false;
    }
    domain.upper[0] = domain.upper[1] = 0.0;
  }
  std::vector<float> spectral_splits;
  for (const auto *spectrum :
       {&profile.ridge_spectrum, &profile.groove_spectrum, &profile.absorbing_substrate_spectrum})
  {
    for (const auto &sample : *spectrum) {
      if (sample.wavelength <= domain.lower[2] || sample.wavelength >= domain.upper[2])
        continue;
      const float rounded = float(sample.wavelength);
      spectral_splits.push_back(rounded);
      /* Bracket a nonrepresentable knot; both adjacent float wavelengths
       * become cell boundaries rather than moving the material feature. */
      if (double(rounded) != sample.wavelength)
        spectral_splits.push_back(std::nextafter(rounded,
                                                 rounded < sample.wavelength ?
                                                     std::numeric_limits<float>::infinity() :
                                                     -std::numeric_limits<float>::infinity()));
    }
  }
  std::sort(spectral_splits.begin(), spectral_splits.end());
  spectral_splits.erase(std::unique(spectral_splits.begin(), spectral_splits.end()),
                        spectral_splits.end());
  DiffractionGratingCache result;
  result.bounds = domain;
  result.complex_validated = options.complex_tolerance > 0;
  result.mirror_symmetry = options.mirror_symmetry;
  /* Per-build bounded LRU: only identical modal solves are reused. No response
   * interpolation or rounding of cache keys is introduced by this optimization. */
  using Key = std::array<double, 3>;
  struct CachedReference {
    DiffractionGratingBlock block;
    std::list<Key>::iterator recent;
  };
  std::map<Key, CachedReference> references;
  std::mutex reference_mutex;
  std::list<Key> recent;
  size_t reference_bytes = 0;
  const ReferenceSolve solve = [&](double wavelength,
                                   double kx,
                                   double ky,
                                   DiffractionGratingBlock &block,
                                   std::string &message) {
    const Key key{wavelength, kx, ky};
    {
      std::lock_guard lock(reference_mutex);
      const auto found = references.find(key);
      if (found != references.end()) {
        stats.reference_cache_hits++;
        recent.splice(recent.begin(), recent, found->second.recent);
        block = found->second.block;
        message.clear();
        return true;
      }
    }
    size_t solve_count = 0;
    const bool solved = diffraction_grating_cache_reference(
        profile, options, wavelength, kx, ky, block, solve_count, message);
    std::lock_guard lock(reference_mutex);
    stats.reference_solves += solve_count;
    if (!solved)
      return false;
    const size_t bytes = block.matrix.size() * sizeof(Complex);
    if (bytes > 0 && bytes <= options.reference_cache_matrix_bytes &&
        references.find(key) == references.end())
    {
      while (reference_bytes > options.reference_cache_matrix_bytes - bytes) {
        const auto oldest = references.find(recent.back());
        reference_bytes -= oldest->second.block.matrix.size() * sizeof(Complex);
        references.erase(oldest);
        recent.pop_back();
      }
      recent.push_front(key);
      references.emplace(key, CachedReference{block, recent.begin()});
      reference_bytes += bytes;
      stats.peak_reference_matrix_bytes = std::max(stats.peak_reference_matrix_bytes,
                                                   reference_bytes);
    }
    return true;
  };
  std::function<bool(const DiffractionGratingCellBounds &, int, int &)> build;
  build = [&](const DiffractionGratingCellBounds &bounds, const int depth, int &node_index) {
    if (stats.visited_nodes >= options.maximum_nodes) {
      error = "Grating cache exceeded node budget";
      return false;
    }
    stats.visited_nodes++;
    stats.last_depth = depth;
    stats.last_bounds = bounds;
    stats.last_validation_error = 0;
    stats.last_complex_error = 0;
    if (options.progress && !options.progress(stats)) {
      error = "Grating cache construction cancelled";
      return false;
    }
    node_index = int(result.nodes.size());
    result.nodes.push_back(make_int4(-1, -1, 0, 0));
    const auto first_knot = std::upper_bound(
        spectral_splits.begin(), spectral_splits.end(), bounds.lower[2]);
    const auto last_knot = std::lower_bound(
        spectral_splits.begin(), spectral_splits.end(), bounds.upper[2]);
    if (first_knot < last_knot) {
      if (depth >= options.maximum_depth) {
        error = "Grating cache cannot resolve optical-constant knots at maximum depth";
        return false;
      }
      const float knot = *(first_knot + (last_knot - first_knot) / 2);
      auto left = bounds, right = bounds;
      left.upper[2] = right.lower[2] = knot;
      int l, r;
      if (!build(left, depth + 1, l) || !build(right, depth + 1, r))
        return false;
      result.nodes[node_index] = make_int4(2, l, r, __float_as_int(knot));
      return true;
    }
    DiffractionGratingCell cell;
    if (!prepare_cell_impl(profile, bounds, options.cutoff_margin, cell, error, solve))
      return false;
    DiffractionGratingChartCell chart_cell;
    double maximum_error = 0, maximum_complex_error = 0;
    std::array<double, 3> curvature{};
    struct SampleResult {
      bool valid = false;
      double maximum_error = 0;
      std::string error;
      std::vector<Complex> reference_matrix;
      double complex_error = 0;
    };
    const auto validate = [&](const bool use_chart) {
      /* Quadratic interpolation can have its largest response error near a
       * face, outside the quarter-point lattice. Probe a second tensor lattice
       * at the three-point Gauss-Legendre abscissae before accepting it. */
      const int sample_count = use_chart && chart_cell.degree == 2 ? 54 : 27;
      std::vector<SampleResult> samples(sample_count);
      const auto evaluate = [&](const int sample) {
        std::string sample_error;
        double sample_maximum = 0;
        std::array<double, 3> query;
        int lattice = sample % 27;
        for (int axis = 0; axis < 3; axis++) {
          const double t = sample < 27 ? 0.25 * (1 + lattice % 3) :
                                         0.5 + 0.5 * std::sqrt(0.6) * (lattice % 3 - 1);
          lattice /= 3;
          query[axis] = bounds.lower[axis] + t * (bounds.upper[axis] - bounds.lower[axis]);
        }
        DiffractionGratingBlock reference, direct, matched;
        DiffractionGratingPowerBlock actual, exact;
        const double kx = query[0] * query[2] / profile.pitch;
        if (use_chart) {
          if (!diffraction_grating_chart_cell_match(
                  profile, chart_cell, query, matched, sample_error))
            return SampleResult{false, 0.0, sample_error};
          diffraction_grating_power_block(matched, actual);
        }
        else if (!diffraction_grating_cell_power(profile, cell, query, actual, sample_error))
          return SampleResult{false, 0.0, sample_error};
        if (!solve(query[2], kx, query[1], reference, sample_error) ||
            !diffraction_grating_match_reference(
                profile, query[2], kx, query[1], reference, direct, sample_error))
          return SampleResult{false, 0.0, sample_error};
        diffraction_grating_power_block(direct, exact);
        if (actual.ports.size() != exact.ports.size()) {
          sample_error = "Cache validation has inconsistent physical ports";
          return SampleResult{false, 0.0, sample_error};
        }
        const size_t n = exact.ports.size();
        for (size_t col = 0; col < n; col++) {
          if (actual.ports[col].order != exact.ports[col].order ||
              actual.ports[col].substrate != exact.ports[col].substrate)
          {
            sample_error = "Cache validation has inconsistent physical port identities";
            return SampleResult{false, 0.0, sample_error};
          }
          double column_error = 0;
          for (size_t row = 0; row < n; row++)
            column_error += std::abs(actual.matrix[row * n + col] - exact.matrix[row * n + col]);
          if (!std::isfinite(column_error)) {
            sample_error = "Nonfinite cache validation error";
            return SampleResult{false, 0.0, sample_error};
          }
          sample_maximum = std::max(sample_maximum, column_error);
        }

        double complex_error = 0;
        if (options.complex_tolerance > 0 && !direct.matrix.empty()) {
          if (!use_chart || matched.matrix.size() != direct.matrix.size())
            return SampleResult{false, 0.0, "Complex cache requires matched chart operators"};
          for (size_t i = 0; i < direct.matrix.size(); i++)
            complex_error = std::hypot(complex_error,
                                       std::abs(matched.matrix[i] - direct.matrix[i]));
          if (!std::isfinite(complex_error))
            return SampleResult{false, 0.0, "Nonfinite complex cache validation error"};
        }
        return SampleResult{true,
                            sample_maximum,
                            {},
                            options.curvature_adaptive_splits ? std::move(reference.matrix) :
                                                                std::vector<Complex>{},
                            complex_error};
      };
      maximum_error = 0;
      maximum_complex_error = 0;
      for (int first = 0; first < sample_count; first += 27) {
        /* A failed quarter-point lattice already requires refinement. Keep
         * all its samples for curvature, but do not solve a second lattice
         * that cannot change rejection. Accepted quadratic cells still pass
         * both complete lattices with the same tolerances. */
        if (first > 0 &&
            (maximum_error > options.tolerance ||
             (options.complex_tolerance > 0 &&
              maximum_complex_error > options.complex_tolerance)))
          break;
        std::atomic<int> next_sample{first};
        const int end = first + 27;
        const auto worker = [&] {
          for (;;) {
            const int sample = next_sample.fetch_add(1, std::memory_order_relaxed);
            if (sample >= end)
              return;
            samples[sample] = evaluate(sample);
          }
        };
        std::vector<std::future<void>> workers;
        for (int i = 1; i < std::min(options.validation_workers, 27); i++)
          workers.push_back(std::async(std::launch::async, worker));
        worker();
        for (auto &task : workers)
          task.get();
        /* Reduce in fixed sample order, independent of thread scheduling. */
        for (int i = first; i < end; i++) {
          const auto &sample = samples[i];
          if (!sample.valid) {
            error = sample.error;
            return false;
          }
          maximum_error = std::max(maximum_error, sample.maximum_error);
          maximum_complex_error = std::max(maximum_complex_error, sample.complex_error);
        }
      }
      curvature.fill(0.0);
      if (options.curvature_adaptive_splits) {
        /* The artificial-port operator has fixed topology across exterior
         * cutoffs. Second differences target interpolation curvature rather
         * than a large, already well-represented linear slope. This only
         * chooses a split; physical-power validation still decides acceptance. */
        for (int axis = 0, stride = 1; axis < 3; axis++, stride *= 3) {
          for (int sample = 0; sample < 27; sample++) {
            if ((sample / stride) % 3 != 0)
              continue;
            const auto &a = samples[sample].reference_matrix;
            const auto &b = samples[sample + stride].reference_matrix;
            const auto &c = samples[sample + 2 * stride].reference_matrix;
            if (a.size() != b.size() || a.size() != c.size()) {
              error = "Inconsistent reference topology for cache curvature";
              return false;
            }
            for (size_t j = 0; j < a.size(); j++)
              curvature[axis] += std::norm(a[j] - 2.0 * b[j] + c[j]);
          }
        }
      }
      return true;
    };
    bool use_chart = options.complex_tolerance > 0 && !cell.corners[0].scattering.ports.empty();
    if (use_chart && !diffraction_grating_prepare_chart_cell(cell, chart_cell, error))
      return false;
    if (!validate(use_chart))
      return false;
    if (!use_chart && maximum_error > options.tolerance && options.allow_chart_cells &&
        !cell.corners[0].scattering.ports.empty())
    {
      if (!diffraction_grating_prepare_chart_cell(cell, chart_cell, error) || !validate(true))
        return false;
      use_chart = maximum_error <= options.tolerance;
    }
    if ((maximum_error > options.tolerance ||
         (options.complex_tolerance > 0 && maximum_complex_error > options.complex_tolerance)) &&
        options.allow_chart_cells && options.allow_quadratic_cells &&
        !cell.corners[0].scattering.ports.empty())
    {
      /* An inadmissible higher-order fit is not used. Refine the cell instead;
       * all children must independently pass the usual acceptance checks. */
      if (prepare_quadratic_chart_impl(profile, cell, chart_cell, error, solve)) {
        if (!validate(true))
          return false;
        use_chart = true;
      }
      else {
        error.clear();
      }
    }
    stats.last_validation_error = maximum_error;
    stats.last_complex_error = maximum_complex_error;
    if (maximum_error <= options.tolerance &&
        (options.complex_tolerance == 0 || maximum_complex_error <= options.complex_tolerance))
    {
      if (!cell.corners[0].scattering.ports.empty()) {
        DiffractionGratingPackedCell packed;
        if (!(use_chart ? diffraction_grating_pack_chart_cell(chart_cell, packed, error) :
                          diffraction_grating_pack_cell(cell, packed, error)))
          return false;
        const size_t bytes = packed.matrices.size() * sizeof(float2);
        if (bytes > options.maximum_matrix_bytes - stats.matrix_bytes) {
          error = "Grating cache exceeded matrix memory budget";
          return false;
        }
        result.nodes[node_index].y = int(result.cells.size());
        result.cells.push_back(std::move(packed));
        stats.matrix_bytes += bytes;
      }
      stats.accepted_cells++;
      stats.accepted_chart_cells += use_chart;
      stats.accepted_quadratic_cells += use_chart && chart_cell.degree == 2;
      double fraction = 1.0;
      for (int a = 0; a < 3; a++) {
        fraction *= (bounds.upper[a] - bounds.lower[a]) / (domain.upper[a] - domain.lower[a]);
      }
      stats.accepted_domain_fraction += fraction;
      stats.maximum_accepted_error = std::max(stats.maximum_accepted_error, maximum_error);
      stats.maximum_accepted_complex_error = std::max(stats.maximum_accepted_complex_error,
                                                      maximum_complex_error);
      return true;
    }
    if (depth >= options.maximum_depth) {
      error = "Grating cache failed interpolation tolerance at maximum depth";
      return false;
    }
    /* Prefer the axis with greatest corner-operator variation. All corners use
     * the same hybrid basis, so their differences are comparable. This is a
     * refinement heuristic, not the acceptance criterion: direct interior
     * validation is still required for every child. Normalized extent breaks
     * ties (including a locally constant operator). */
    int axis = 0;
    double largest = -1, largest_variation = -1;
    for (int a = 0; a < 3; a++) {
      const double extent = (bounds.upper[a] - bounds.lower[a]) /
                            (domain.upper[a] - domain.lower[a]);
      double variation = 0;
      if (options.curvature_adaptive_splits) {
        variation = curvature[a];
      }
      else if (options.response_adaptive_splits) {
        for (int corner = 0; corner < 8; corner++) {
          if (corner & (1 << a))
            continue;
          const auto &low = cell.corners[corner].scattering.matrix;
          const auto &high = cell.corners[corner | (1 << a)].scattering.matrix;
          for (size_t j = 0; j < low.size(); j++)
            variation += std::norm(high[j] - low[j]);
        }
      }
      if (variation > largest_variation || (variation == largest_variation && extent > largest)) {
        largest = extent;
        largest_variation = variation;
        axis = a;
      }
    }
    const double middle = float(0.5 * (bounds.lower[axis] + bounds.upper[axis]));
    if (!(middle > bounds.lower[axis] && middle < bounds.upper[axis])) {
      error = "Grating cache split exhausted float coordinate resolution";
      return false;
    }
    cell = {};
    auto left = bounds, right = bounds;
    left.upper[axis] = right.lower[axis] = middle;
    int l, r;
    if (!build(left, depth + 1, l) || !build(right, depth + 1, r))
      return false;
    result.nodes[node_index] = make_int4(axis, l, r, __float_as_int(float(middle)));
    return true;
  };
  int root;
  if (!build(domain, 0, root))
    return false;
  cache = std::move(result);
  return true;
}

bool diffraction_grating_cell_power(const DiffractionGratingProfile &profile,
                                    const DiffractionGratingCell &cell,
                                    const std::array<double, 3> &query,
                                    DiffractionGratingPowerBlock &power,
                                    std::string &error)
{
  power = {};
  error.clear();
  std::array<double, 3> t;
  for (int axis = 0; axis < 3; axis++) {
    const double lo = cell.bounds.lower[axis], hi = cell.bounds.upper[axis];
    if (!std::isfinite(query[axis]) || !std::isfinite(lo) || !std::isfinite(hi) || !(hi > lo) ||
        query[axis] < lo || query[axis] > hi)
    {
      error = "Grating query outside prepared cell";
      return false;
    }
    t[axis] = (query[axis] - lo) / (hi - lo);
  }
  std::array<DiffractionGratingPowerBlock, 8> powers;
  std::array<DiffractionGratingPowerSample, 8> samples;
  for (int corner = 0; corner < 8; corner++) {
    DiffractionGratingBlock matched;
    if (!diffraction_grating_match_hybrid(profile,
                                          query[2],
                                          query[0] * query[2] / profile.pitch,
                                          query[1],
                                          cell.corners[corner],
                                          matched,
                                          error))
    {
      return false;
    }
    diffraction_grating_power_block(matched, powers[corner]);
    if (corner > 0) {
      if (powers[corner].ports.size() != powers[0].ports.size()) {
        error = "Grating cell corners disagree on physical ports";
        return false;
      }
      for (size_t p = 0; p < powers[0].ports.size(); p++) {
        if (powers[corner].ports[p].order != powers[0].ports[p].order ||
            powers[corner].ports[p].substrate != powers[0].ports[p].substrate)
        {
          error = "Grating cell corners disagree on physical ports";
          return false;
        }
      }
    }
    samples[corner] = {&powers[corner],
                       (corner & 1 ? t[0] : 1.0 - t[0]) * (corner & 2 ? t[1] : 1.0 - t[1]) *
                           (corner & 4 ? t[2] : 1.0 - t[2])};
  }
  return diffraction_grating_interpolate_power(samples, powers[0].ports, power, error);
}

bool diffraction_grating_prepare_chart_cell(const DiffractionGratingCell &cell,
                                            DiffractionGratingChartCell &chart_cell,
                                            std::string &error)
{
  chart_cell = {};
  error.clear();
  for (int a = 0; a < 3; a++) {
    if (!std::isfinite(cell.bounds.lower[a]) || !std::isfinite(cell.bounds.upper[a]) ||
        !(cell.bounds.upper[a] > cell.bounds.lower[a]))
    {
      error = "Invalid chart cell bounds";
      return false;
    }
  }
  const auto &first = cell.corners[0];
  if (first.is_reference.size() != first.scattering.ports.size()) {
    error = "Invalid chart cell reference mask";
    return false;
  }
  std::array<DiffractionGratingBlock, 8> references;
  for (int c = 0; c < 8; c++) {
    const auto &corner = cell.corners[c];
    if (corner.is_reference != first.is_reference ||
        corner.scattering.ports.size() != first.scattering.ports.size())
    {
      error = "Inconsistent chart cell topology";
      return false;
    }
    for (size_t p = 0; p < first.scattering.ports.size(); p++) {
      if (corner.is_reference[p] > 1 ||
          corner.scattering.ports[p].order != first.scattering.ports[p].order ||
          corner.scattering.ports[p].substrate != first.scattering.ports[p].substrate)
      {
        error = "Inconsistent chart cell ports";
        return false;
      }
    }
    references[c] = corner.scattering;
  }
  DiffractionGratingChartCell result;
  double phase;
  if (!diffraction_grating_choose_chart(references, phase, result.corners, error))
    return false;
  result.bounds = cell.bounds;
  result.is_reference = first.is_reference;
  chart_cell = std::move(result);
  return true;
}

bool diffraction_grating_pack_chart_cell(const DiffractionGratingChartCell &cell,
                                         DiffractionGratingPackedCell &packed,
                                         std::string &error)
{
  packed = {};
  error.clear();
  const size_t controls = cell.degree == 1 ? 8 : cell.degree == 2 ? 27 : 0;
  if (controls == 0 || cell.corners.size() != controls || cell.corners[0].ports.empty() ||
      cell.is_reference.size() != cell.corners[0].ports.size())
  {
    error = "Invalid chart cell shape";
    return false;
  }
  for (int a = 0; a < 3; a++) {
    if (!std::isfinite(cell.bounds.lower[a]) || !std::isfinite(cell.bounds.upper[a]) ||
        !(cell.bounds.upper[a] > cell.bounds.lower[a]))
    {
      error = "Invalid packed chart cell bounds";
      return false;
    }
  }
  DiffractionGratingPackedCell result;
  result.bounds = cell.bounds;
  result.ports = cell.corners[0].ports;
  result.operator_chart = true;
  result.chart_degree = cell.degree;
  const Complex rotation = cell.corners[0].rotation;
  if (!std::isfinite(std::norm(rotation)) || std::abs(std::norm(rotation) - 1.0) > 1e-12) {
    error = "Invalid chart cell rotation";
    return false;
  }
  result.chart_rotation = make_float2(float(rotation.real()), float(rotation.imag()));
  const size_t ports = result.ports.size(), channels = 2 * ports;
  for (size_t p = 0; p < ports; p++) {
    if (cell.is_reference[p] > 1) {
      error = "Invalid chart cell reference mask";
      return false;
    }
    if (cell.is_reference[p])
      result.active_ports.push_back(int(p));
  }
  for (const auto &corner : cell.corners) {
    if (corner.rotation != rotation || corner.ports.size() != ports ||
        corner.matrix.size() != channels * channels)
    {
      error = "Inconsistent packed chart cell layout";
      return false;
    }
    for (size_t p = 0; p < ports; p++) {
      if (corner.ports[p].order != result.ports[p].order ||
          corner.ports[p].substrate != result.ports[p].substrate)
      {
        error = "Inconsistent packed chart cell ports";
        return false;
      }
    }
    for (Complex value : corner.matrix) {
      const float real = float(value.real()), imag = float(value.imag());
      if (!std::isfinite(real) || !std::isfinite(imag)) {
        error = "Chart cell cannot be represented in float";
        return false;
      }
      result.matrices.push_back(make_float2(real, imag));
    }
  }
  packed = std::move(result);
  return true;
}

bool diffraction_grating_chart_cell_match(const DiffractionGratingProfile &profile,
                                          const DiffractionGratingChartCell &cell,
                                          const std::array<double, 3> &query,
                                          DiffractionGratingBlock &physical,
                                          std::string &error)
{
  physical = {};
  error.clear();
  const int controls = cell.degree == 1 ? 8 : cell.degree == 2 ? 27 : 0;
  if (controls == 0 || cell.corners.size() != size_t(controls)) {
    error = "Invalid chart degree or control count";
    return false;
  }
  std::array<double, 3> t;
  for (int a = 0; a < 3; a++) {
    const double lo = cell.bounds.lower[a], hi = cell.bounds.upper[a];
    if (!std::isfinite(query[a]) || !std::isfinite(lo) || !std::isfinite(hi) || !(hi > lo) ||
        query[a] < lo || query[a] > hi)
    {
      error = "Grating query outside chart cell";
      return false;
    }
    t[a] = (query[a] - lo) / (hi - lo);
  }
  auto blend = cell.corners[0];
  std::fill(blend.matrix.begin(), blend.matrix.end(), Complex(0));
  for (int c = 0; c < controls; c++) {
    const auto &source = cell.corners[c];
    if (source.rotation != blend.rotation || source.matrix.size() != blend.matrix.size() ||
        source.ports.size() != blend.ports.size())
    {
      error = "Inconsistent chart cell matrices";
      return false;
    }
    for (size_t p = 0; p < blend.ports.size(); p++) {
      if (source.ports[p].order != blend.ports[p].order ||
          source.ports[p].substrate != blend.ports[p].substrate)
      {
        error = "Inconsistent chart cell port mapping";
        return false;
      }
    }
    double weight = 1;
    int index = c;
    for (int axis = 0; axis < 3; axis++) {
      const int digit = index % (cell.degree + 1);
      index /= cell.degree + 1;
      const double u = t[axis];
      weight *= cell.degree == 1 ? (digit ? u : 1 - u) :
                digit == 0       ? (1 - u) * (1 - u) :
                digit == 1       ? 2 * u * (1 - u) :
                                   u * u;
    }
    for (size_t j = 0; j < blend.matrix.size(); j++)
      blend.matrix[j] += weight * source.matrix[j];
  }
  DiffractionGratingHybrid hybrid;
  hybrid.is_reference = cell.is_reference;
  if (!diffraction_grating_chart_to_reference(blend, hybrid.scattering, error))
    return false;
  return diffraction_grating_match_hybrid(
      profile, query[2], query[0] * query[2] / profile.pitch, query[1], hybrid, physical, error);
}

bool diffraction_grating_reference_to_chart(const DiffractionGratingBlock &reference,
                                            const double phase,
                                            DiffractionGratingReferenceChart &chart,
                                            std::string &error)
{
  chart = {};
  error.clear();
  const size_t ports = reference.ports.size(), channels = 2 * ports;
  if (ports == 0 || ports > 2050 || reference.matrix.size() != channels * channels ||
      !std::isfinite(phase))
  {
    error = "Invalid reference operator or chart phase";
    return false;
  }
  Matrix S(channels, channels);
  for (size_t r = 0; r < channels; r++) {
    for (size_t c = 0; c < channels; c++) {
      S(r, c) = reference.matrix[r * channels + c];
    }
  }
  const Complex rotation = std::exp(Complex(0.0, phase));
  const Matrix identity = Matrix::Identity(channels, channels);
  const Matrix left = identity + rotation * S, right = identity - rotation * S;
  const Matrix Y = left.partialPivLu().solve(right);
  const double residual = (left * Y - right).norm() / std::max(1.0, right.norm());
  if (!Y.allFinite() || !std::isfinite(residual) || residual > 1e-8) {
    error = "Reference chart has a pole or failed its residual check";
    return false;
  }
  Eigen::SelfAdjointEigenSolver<Matrix> dissipation(0.5 * (Y + Y.adjoint()),
                                                    Eigen::EigenvaluesOnly);
  if (dissipation.info() != Eigen::Success) {
    error = "Reference chart dissipation check failed";
    return false;
  }
  chart.ports = reference.ports;
  chart.rotation = rotation;
  chart.minimum_dissipation = dissipation.eigenvalues().minCoeff();
  chart.frobenius_norm = Y.norm();
  chart.residual = residual;
  chart.matrix.resize(channels * channels);
  for (size_t r = 0; r < channels; r++) {
    for (size_t c = 0; c < channels; c++) {
      chart.matrix[r * channels + c] = Y(r, c);
    }
  }
  return true;
}

bool diffraction_grating_chart_to_reference(const DiffractionGratingReferenceChart &chart,
                                            DiffractionGratingBlock &reference,
                                            std::string &error)
{
  reference = {};
  error.clear();
  const size_t ports = chart.ports.size(), channels = 2 * ports;
  if (ports == 0 || ports > 2050 || chart.matrix.size() != channels * channels ||
      !std::isfinite(chart.rotation.real()) || !std::isfinite(chart.rotation.imag()) ||
      std::abs(std::norm(chart.rotation) - 1.0) > 1e-12)
  {
    error = "Invalid reference chart";
    return false;
  }
  Matrix Y(channels, channels);
  for (size_t r = 0; r < channels; r++) {
    for (size_t c = 0; c < channels; c++) {
      Y(r, c) = chart.matrix[r * channels + c];
    }
  }
  const Matrix identity = Matrix::Identity(channels, channels);
  const Matrix left = identity + Y, right = identity - Y;
  const Matrix rotated = left.partialPivLu().solve(right);
  const double residual = (left * rotated - right).norm() / std::max(1.0, right.norm());
  if (!rotated.allFinite() || !std::isfinite(residual) || residual > 1e-8) {
    error = "Reference chart inversion failed its residual check";
    return false;
  }
  reference.ports = chart.ports;
  reference.boundary_residual = residual;
  return store_scattering(std::conj(chart.rotation) * rotated, reference, error);
}

bool diffraction_grating_choose_chart(const std::span<const DiffractionGratingBlock> references,
                                      double &phase,
                                      std::vector<DiffractionGratingReferenceChart> &charts,
                                      std::string &error)
{
  charts.clear();
  phase = 0.0;
  error.clear();
  if (references.empty()) {
    error = "No reference operators for chart selection";
    return false;
  }
  std::vector<double> poles;
  for (const auto &reference : references) {
    const size_t ports = reference.ports.size(), channels = 2 * ports;
    if (ports == 0 || ports > 2050 || reference.matrix.size() != channels * channels) {
      error = "Invalid operator for reference chart selection";
      return false;
    }
    Matrix S(channels, channels);
    for (size_t r = 0; r < channels; r++) {
      for (size_t c = 0; c < channels; c++) {
        S(r, c) = reference.matrix[r * channels + c];
      }
    }
    if (!S.allFinite()) {
      error = "Nonfinite operator for reference chart selection";
      return false;
    }
    Eigen::ComplexEigenSolver<Matrix> eigen(S, false);
    if (eigen.info() != Eigen::Success) {
      error = "Reference chart pole-angle solve failed";
      return false;
    }
    for (const Complex value : eigen.eigenvalues()) {
      /* Eigenvalues far inside the unit circle cannot approach a chart pole. */
      if (std::abs(value) > 0.5) {
        poles.push_back(std::fmod(3.0 * pi - std::arg(value), 2.0 * pi));
      }
    }
  }
  std::vector<double> candidates{
      0.5 * pi, -0.5 * pi, 0.25 * pi, -0.25 * pi, 0.75 * pi, -0.75 * pi};
  if (!poles.empty()) {
    std::sort(poles.begin(), poles.end());
    std::vector<std::pair<double, double>> gaps;
    for (size_t i = 0; i < poles.size(); i++) {
      const double end = i + 1 < poles.size() ? poles[i + 1] : poles.front() + 2.0 * pi;
      gaps.emplace_back(end - poles[i], std::fmod(0.5 * (end + poles[i]), 2.0 * pi));
    }
    std::sort(
        gaps.begin(), gaps.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    for (size_t i = 0; i < std::min(size_t(8), gaps.size()); i++) {
      candidates.push_back(gaps[i].second);
    }
  }
  double best_norm = std::numeric_limits<double>::infinity();
  for (const double candidate : candidates) {
    std::vector<DiffractionGratingReferenceChart> trial;
    double worst_norm = 0.0;
    bool valid = true;
    for (const auto &reference : references) {
      DiffractionGratingReferenceChart chart;
      if (!diffraction_grating_reference_to_chart(reference, candidate, chart, error) ||
          !std::isfinite(chart.frobenius_norm))
      {
        valid = false;
        break;
      }
      worst_norm = std::max(worst_norm, chart.frobenius_norm);
      trial.push_back(std::move(chart));
    }
    if (valid && worst_norm < best_norm) {
      phase = candidate;
      best_norm = worst_norm;
      charts.swap(trial);
    }
  }
  if (charts.empty()) {
    error = "No valid reference chart candidate";
    return false;
  }
  error.clear();
  return true;
}

void diffraction_grating_power_block(const DiffractionGratingBlock &block,
                                     DiffractionGratingPowerBlock &power)
{
  power = {};
  power.ports = block.ports;
  const size_t ports = block.ports.size();
  power.matrix.resize(ports * ports);
  for (size_t col = 0; col < ports; col++) {
    double sum = 0.0;
    for (size_t row = 0; row < ports; row++) {
      double value = 0.0;
      for (size_t a = 0; a < 2; a++) {
        for (size_t b = 0; b < 2; b++) {
          value += std::norm(block.matrix[(2 * row + a) * (2 * ports) + 2 * col + b]);
        }
      }
      power.matrix[row * ports + col] = 0.5 * value;
      sum += 0.5 * value;
    }
    power.maximum_column_sum = std::max(power.maximum_column_sum, sum);
  }
}

bool diffraction_grating_pack_power(const DiffractionGratingPowerBlock &power,
                                    std::vector<float> &packed,
                                    std::string &error)
{
  packed.clear();
  error.clear();
  const size_t count = power.ports.size();
  if (count > 2050 || power.matrix.size() != count * count) {
    error = "Invalid grating power block dimensions";
    return false;
  }
  int first[2] = {0, 0}, counts[2] = {0, 0};
  for (const auto &port : power.ports) {
    const int side = port.substrate ? 1 : 0;
    if (port.order < -512 || port.order > 512 || (!side && counts[1] != 0) ||
        (counts[side] && port.order != first[side] + counts[side]))
    {
      error = "Grating power ports must be contiguous upper then lower ranges";
      return false;
    }
    if (counts[side] == 0) {
      first[side] = port.order;
    }
    counts[side]++;
  }
  for (size_t col = 0; col < count; col++) {
    double sum = 0.0;
    for (size_t row = 0; row < count; row++) {
      const double value = power.matrix[row * count + col];
      if (!std::isfinite(value) || value < 0.0) {
        error = "Invalid grating power coefficient";
        return false;
      }
      sum += value;
    }
    if (sum > 1.0 + 1e-8) {
      error = "Grating power block violates passivity";
      return false;
    }
  }
  packed.resize(DIFFRACTION_TABLE_HEADER_SIZE + count * count);
  packed[DIFFRACTION_TABLE_UPPER_FIRST] = float(first[0]);
  packed[DIFFRACTION_TABLE_UPPER_COUNT] = float(counts[0]);
  packed[DIFFRACTION_TABLE_LOWER_FIRST] = float(first[1]);
  packed[DIFFRACTION_TABLE_LOWER_COUNT] = float(counts[1]);
  for (size_t i = 0; i < power.matrix.size(); i++) {
    packed[DIFFRACTION_TABLE_HEADER_SIZE + i] = float(power.matrix[i]);
  }
  return true;
}

bool diffraction_grating_build_grid(const DiffractionGratingProfile &profile,
                                    const DiffractionGratingGridConfig &config,
                                    const int half_orders,
                                    std::vector<float> &packed,
                                    std::string &error,
                                    const std::function<bool()> &cancel)
{
  packed.clear();
  error.clear();
  const int nk = config.bloch_samples, nv = config.tangent_samples, nl = config.wavelength_samples;
  if (nk < 2 || nv < 2 || nl < 2 || nk > 1024 || nv > 1024 || nl > 1024 ||
      !(config.wavelength_min > 0.0) || !(config.wavelength_max > config.wavelength_min) ||
      !std::isfinite(config.wavelength_max) || size_t(nk) * nv * nl > (1 << 20))
  {
    error = "Invalid grating response grid dimensions or wavelength interval";
    return false;
  }
  const double lower_index = profile.substrate_ior.imag() == 0.0 ? profile.substrate_ior.real() :
                                                                   0.0;
  const double tangent_max = std::max(profile.incident_ior, lower_index);
  const size_t nodes = size_t(nk) * nv * nl;
  std::vector<float> result(DIFFRACTION_GRID_HEADER_SIZE + nodes);
  result[DIFFRACTION_GRID_BLOCH_COUNT] = float(nk);
  result[DIFFRACTION_GRID_TANGENT_COUNT] = float(nv);
  result[DIFFRACTION_GRID_WAVELENGTH_COUNT] = float(nl);
  result[DIFFRACTION_GRID_TANGENT_MAX] = float(tangent_max);
  result[DIFFRACTION_GRID_WAVELENGTH_MIN] = float(config.wavelength_min);
  result[DIFFRACTION_GRID_WAVELENGTH_MAX] = float(config.wavelength_max);
  result[DIFFRACTION_GRID_PITCH] = float(profile.pitch);
  result[DIFFRACTION_GRID_UPPER_IOR] = float(profile.incident_ior);
  result[DIFFRACTION_GRID_LOWER_IOR] = float(lower_index);
  result[DIFFRACTION_GRID_CLOSED_PORT_REFLECTION] = config.closed_port_reflection ? 1.0f : 0.0f;
  for (int i = 0; i < DIFFRACTION_GRID_HEADER_SIZE; i++) {
    if (!std::isfinite(result[i])) {
      error = "Grating response grid parameters exceed float range";
      return false;
    }
  }
  if (!(result[DIFFRACTION_GRID_PITCH] > 0.0f) || !(result[DIFFRACTION_GRID_UPPER_IOR] > 0.0f) ||
      !(result[DIFFRACTION_GRID_WAVELENGTH_MIN] > 0.0f) ||
      !(result[DIFFRACTION_GRID_WAVELENGTH_MAX] > result[DIFFRACTION_GRID_WAVELENGTH_MIN]))
  {
    error = "Grating response grid parameters are not representable as floats";
    return false;
  }
  DiffractionGratingBlock block;
  DiffractionGratingPowerBlock power;
  std::vector<float> node_data;
  for (int l = 0; l < nl; l++) {
    const double wavelength = config.wavelength_min +
                              (config.wavelength_max - config.wavelength_min) * l / (nl - 1);
    for (int v = 0; v < nv; v++) {
      const double ky = tangent_max * (2.0 * (v + 0.5) / nv - 1.0);
      for (int k = 0; k < nk; k++) {
        if (cancel && cancel()) {
          error = "Grating response generation cancelled";
          return false;
        }
        const double kx = ((k + 0.5) / nk - 0.5) * wavelength / profile.pitch;
        if (!diffraction_grating_solve_bloch(
                profile, wavelength, kx, ky, half_orders, block, error))
        {
          error += " at grid node (" + std::to_string(k) + "," + std::to_string(v) + "," +
                   std::to_string(l) + ")";
          return false;
        }
        diffraction_grating_power_block(block, power);
        if (!diffraction_grating_pack_power(power, node_data, error)) {
          return false;
        }
        if (result.size() + node_data.size() > (1 << 24)) {
          error = "Grating response grid exceeds the 64 MiB packed table limit";
          return false;
        }
        const size_t node = (size_t(l) * nv + v) * nk + k;
        result[DIFFRACTION_GRID_HEADER_SIZE + node] = float(result.size());
        result.insert(result.end(), node_data.begin(), node_data.end());
      }
    }
  }
  packed.swap(result);
  return true;
}

bool diffraction_grating_interpolate_power(
    const std::span<const DiffractionGratingPowerSample> samples,
    const std::span<const DiffractionGratingPort> target_ports,
    DiffractionGratingPowerBlock &power,
    std::string &error)
{
  power = {};
  error.clear();
  double weight_sum = 0.0;
  for (const auto &sample : samples) {
    if (!sample.block || !std::isfinite(sample.weight) || sample.weight < 0.0 ||
        sample.block->matrix.size() != sample.block->ports.size() * sample.block->ports.size())
    {
      error = "Invalid grating interpolation sample";
      return false;
    }
    weight_sum += sample.weight;
  }
  if (std::abs(weight_sum - 1.0) > 1e-12) {
    error = "Grating interpolation weights must sum to one";
    return false;
  }
  const size_t count = target_ports.size();
  for (size_t i = 0; i < count; i++) {
    for (size_t j = 0; j < i; j++) {
      if (target_ports[i].order == target_ports[j].order &&
          target_ports[i].substrate == target_ports[j].substrate)
      {
        error = "Duplicate target grating port";
        return false;
      }
    }
  }
  power.ports.assign(target_ports.begin(), target_ports.end());
  power.matrix.assign(count * count, 0.0);
  for (const auto &sample : samples) {
    const auto &source = *sample.block;
    std::vector<int> mapping(count, -1);
    for (size_t i = 0; i < count; i++) {
      for (size_t j = 0; j < source.ports.size(); j++) {
        if (target_ports[i].order == source.ports[j].order &&
            target_ports[i].substrate == source.ports[j].substrate)
        {
          if (mapping[i] >= 0) {
            power = {};
            error = "Duplicate source grating port";
            return false;
          }
          mapping[i] = int(j);
        }
      }
    }
    const double weight = sample.weight / weight_sum;
    for (size_t row = 0; row < count; row++) {
      if (mapping[row] < 0) {
        continue;
      }
      for (size_t col = 0; col < count; col++) {
        if (mapping[col] >= 0) {
          const double value =
              source.matrix[size_t(mapping[row]) * source.ports.size() + mapping[col]];
          if (!(value >= 0.0) || !std::isfinite(value)) {
            power = {};
            error = "Invalid grating power";
            return false;
          }
          power.matrix[row * count + col] += weight * value;
        }
      }
    }
  }
  for (size_t col = 0; col < count; col++) {
    double sum = 0.0;
    for (size_t row = 0; row < count; row++) {
      sum += power.matrix[row * count + col];
    }
    power.maximum_column_sum = std::max(power.maximum_column_sum, sum);
  }
  return true;
}

bool diffraction_grating_device_buffers(const DiffractionGratingCache &cache,
                                        DiffractionGratingDeviceBuffers &buffers,
                                        std::string &error)
{
  buffers = {};
  error.clear();
  auto fail = [&](const char *message) {
    error = message;
    return false;
  };
  const size_t limit = std::numeric_limits<int>::max();
  if (cache.nodes.empty() || cache.nodes.size() > limit || cache.cells.size() > limit / 2)
    return fail("Invalid diffraction cache dimensions");
  std::vector<bool> visited(cache.nodes.size(), false);
  std::vector<int> pending{0};
  while (!pending.empty()) {
    const int index = pending.back();
    pending.pop_back();
    if (index < 0 || size_t(index) >= cache.nodes.size() || visited[index])
      return fail("Invalid diffraction cache tree");
    visited[index] = true;
    const int4 node = cache.nodes[index];
    if (node.x == -1) {
      if (node.y < -1 || (node.y >= 0 && size_t(node.y) >= cache.cells.size()))
        return fail("Invalid diffraction cache leaf");
    }
    else {
      if (node.x < 0 || node.x > 2 || !std::isfinite(__int_as_float(node.w)))
        return fail("Invalid diffraction cache split");
      pending.push_back(node.y);
      pending.push_back(node.z);
    }
  }
  if (std::find(visited.begin(), visited.end(), false) != visited.end())
    return fail("Unreachable diffraction cache nodes");
  DiffractionGratingDeviceBuffers result;
  result.nodes = cache.nodes;
  for (const auto &cell : cache.cells) {
    const size_t ports = cell.ports.size();
    const int degree = cell.operator_chart ? cell.chart_degree : 0;
    const size_t controls = degree == 2 ? 27 : 8;
    if (ports == 0 || ports > limit / 2 || degree < 0 || degree > 3 ||
        (cell.operator_chart && degree == 0) || cell.active_ports.size() > ports ||
        result.ports.size() > limit - ports ||
        result.active.size() > limit - cell.active_ports.size() || cell.matrices.size() > limit ||
        result.matrices.size() > limit - cell.matrices.size())
      return fail("Invalid diffraction device cell dimensions");
    const size_t channels = 2 * ports;
    size_t tensor_floats = 0;
    if (degree == 3) {
      if (channels > limit / channels)
        return fail("Tensor channel dimensions exceed addressing capacity");
      const size_t square = channels * channels;
      if (cell.tensor_rank < 0 || size_t(cell.tensor_rank) > square || square > limit - 14 ||
          size_t(cell.tensor_rank) > (limit - 14 - square) / (343 + square))
        return fail("Invalid diffraction tensor rank or size");
      tensor_floats = 14 + square + size_t(cell.tensor_rank) * (343 + square);
      if (cell.matrices.size() != square + (tensor_floats + 1) / 2 ||
          cell.active_ports.size() != ports)
        return fail("Invalid diffraction tensor payload or reference ports");
      const auto real = [&](const size_t i) {
        const float2 pair = cell.matrices[square + i / 2];
        return (i & 1) ? pair.y : pair.x;
      };
      if (real(0) != 0.0f || real(6) != 1.0f)
        return fail("Invalid diffraction tensor node endpoints");
      for (size_t i = 0; i < 7; i++) {
        if ((i && !(real(i) > real(i - 1))) || real(7 + i) == 0.0f)
          return fail("Invalid diffraction tensor interpolation basis");
      }
      Matrix anchor(channels, channels);
      for (size_t row = 0; row < channels; row++)
        for (size_t col = 0; col < channels; col++) {
          const float2 value = cell.matrices[row * channels + col];
          anchor(row, col) = Complex(value.x, value.y);
        }
      const double residual =
          (anchor.adjoint() * anchor - Matrix::Identity(channels, channels)).norm();
      if (!std::isfinite(residual) || residual > 1e-5 * channels)
        return fail("Diffraction tensor anchor is not unitary within float precision");
    }
    else if (cell.tensor_rank != -1 || channels > limit / channels / controls ||
             cell.matrices.size() != controls * channels * channels)
      return fail("Invalid diffraction device matrix size");
    for (int axis = 0; axis < 3; axis++) {
      const float lo = cell.bounds.lower[axis], hi = cell.bounds.upper[axis];
      if (!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi))
        return fail("Invalid diffraction device bounds");
    }
    if (!std::isfinite(cell.chart_rotation.x) || !std::isfinite(cell.chart_rotation.y))
      return fail("Invalid diffraction chart rotation");
    for (size_t i = 0; i < ports; i++) {
      const auto &port = cell.ports[i];
      if (i > 0 && (port.substrate < cell.ports[i - 1].substrate ||
                    (port.substrate == cell.ports[i - 1].substrate &&
                     port.order <= cell.ports[i - 1].order)))
        return fail("Invalid diffraction device port ordering");
    }
    for (size_t i = 0; i < cell.active_ports.size(); i++)
      if (cell.active_ports[i] < 0 || size_t(cell.active_ports[i]) >= ports ||
          (i > 0 && cell.active_ports[i] <= cell.active_ports[i - 1]))
        return fail("Invalid diffraction device active ports");
    for (float2 value : cell.matrices)
      if (!std::isfinite(value.x) || !std::isfinite(value.y))
        return fail("Nonfinite diffraction device matrix");
    result.layout.push_back(
        make_int4(result.matrices.size(), result.ports.size(), result.active.size(), degree));
    result.layout.push_back(make_int4(ports,
                                     cell.active_ports.size(),
                                     degree == 3 ? cell.tensor_rank : 0,
                                     tensor_floats));
    result.bounds.push_back(make_float4(
        cell.bounds.lower[0], cell.bounds.lower[1], cell.bounds.lower[2], cell.chart_rotation.x));
    result.bounds.push_back(make_float4(
        cell.bounds.upper[0], cell.bounds.upper[1], cell.bounds.upper[2], cell.chart_rotation.y));
    for (const auto &port : cell.ports)
      result.ports.push_back(make_int2(port.order, port.substrate));
    result.active.insert(result.active.end(), cell.active_ports.begin(), cell.active_ports.end());
    result.matrices.insert(result.matrices.end(), cell.matrices.begin(), cell.matrices.end());
  }
  buffers = std::move(result);
  return true;
}

namespace {
bool modal_difference(const DiffractionGratingBlock &a,
                const DiffractionGratingBlock &b,
                double &power, double &amplitude)
{
  if (a.ports.size() != b.ports.size() || a.matrix.size() != b.matrix.size()) return false;
  for (size_t i = 0; i < a.ports.size(); i++)
    if (a.ports[i].order != b.ports[i].order || a.ports[i].substrate != b.ports[i].substrate)
      return false;
  DiffractionGratingPowerBlock pa, pb;
  diffraction_grating_power_block(a, pa);
  diffraction_grating_power_block(b, pb);
  power = amplitude = 0;
  const size_t count = a.ports.size();
  for (size_t column = 0; column < count; column++) {
    double sum = 0;
    for (size_t row = 0; row < count; row++)
      sum += std::abs(pa.matrix[row * count + column] - pb.matrix[row * count + column]);
    if (!std::isfinite(sum)) return false;
    power = std::max(power, sum);
  }
  for (size_t i = 0; i < a.matrix.size(); i++)
    amplitude = std::hypot(amplitude, std::abs(a.matrix[i] - b.matrix[i]));
  return std::isfinite(amplitude);
}
}
bool diffraction_grating_converged_reference(
    const DiffractionGratingProfile &profile, const double wavelength,
    const double kx, const double ky, const int retained_half_orders,
    const DiffractionModalOptions &options, DiffractionGratingBlock &reference,
    std::vector<DiffractionModalObservation> &observations, std::string &error)
{
  reference = {};
  observations.clear();
  error.clear();
  if (options.minimum_half_orders < 1 || options.maximum_half_orders > 512 ||
      options.maximum_half_orders < options.minimum_half_orders || retained_half_orders < 0 ||
      retained_half_orders > options.minimum_half_orders ||
      !(options.power_tolerance > 0) || !std::isfinite(options.power_tolerance) ||
      !(options.complex_tolerance >= 0) || !std::isfinite(options.complex_tolerance)) {
    error = "Invalid diffraction modal-convergence options";
    return false;
  }
  DiffractionGratingBlock previous, older;
  bool previous_pass = false;
  for (int n = options.minimum_half_orders;; n = std::min(2 * n, options.maximum_half_orders)) {
    if (options.progress && !options.progress(n)) {
      error = "Diffraction modal convergence cancelled";
      return false;
    }
    DiffractionGratingBlock candidate, physical;
    const auto &solver = options.reference_solver;
    const bool success = solver ? solver(profile, wavelength, kx, ky, n,
                                         retained_half_orders, candidate, error) :
                                  diffraction_grating_solve_reference(profile, wavelength, kx, ky, n,
                                                                       retained_half_orders, candidate, error);
    if (!success ||
        !diffraction_grating_match_reference(profile, wavelength, kx, ky, candidate,
                                             physical, error)) return false;
    DiffractionModalObservation observation;
    observation.half_orders = n;
    observation.comparisons = std::min(size_t(2), observations.size());
    if (observation.comparisons &&
        !modal_difference(physical, previous, observation.adjacent_power_difference,
                    observation.adjacent_complex_difference)) {
      error = "Inconsistent or nonfinite diffraction modal comparison";
      return false;
    }
    if (observation.comparisons == 2 &&
        !modal_difference(physical, older, observation.spanning_power_difference,
                    observation.spanning_complex_difference)) {
      error = "Inconsistent or nonfinite diffraction spanning modal comparison";
      return false;
    }
    const bool adjacent_pass = observation.comparisons &&
        observation.adjacent_power_difference <= options.power_tolerance &&
        (options.complex_tolerance == 0 ||
         observation.adjacent_complex_difference <= options.complex_tolerance);
    const bool spanning_pass = observation.comparisons == 2 &&
        observation.spanning_power_difference <= options.power_tolerance &&
        (options.complex_tolerance == 0 ||
         observation.spanning_complex_difference <= options.complex_tolerance);
    observations.push_back(observation);
    if (adjacent_pass && spanning_pass && previous_pass) {
      reference = std::move(candidate);
      return true;
    }
    previous_pass = adjacent_pass;
    older = std::move(previous);
    previous = std::move(physical);
    if (n == options.maximum_half_orders) {
      error = "Diffraction modal convergence exceeded Fourier resolution budget";
      return false;
    }
  }
}
bool diffraction_grating_cache_reference(
    const DiffractionGratingProfile &profile, const DiffractionGratingCacheOptions &options,
    const double wavelength, const double kx, const double ky, DiffractionGratingBlock &reference,
    size_t &solve_count, std::string &error)
{
  solve_count = 0;
  if (options.cancelled && options.cancelled()) {
    reference = {};
    error = "Diffraction reference construction cancelled";
    return false;
  }
  if (!(options.modal_power_tolerance >= 0) || !std::isfinite(options.modal_power_tolerance) ||
      !(options.modal_complex_tolerance >= 0) || !std::isfinite(options.modal_complex_tolerance) ||
      (options.modal_power_tolerance == 0 && options.modal_complex_tolerance != 0)) {
    reference = {};
    error = "Invalid cache modal-convergence tolerances";
    return false;
  }
  if (options.modal_power_tolerance == 0) {
    solve_count = 1;
    if (options.reference_solver) {
      reference = {};
      error.clear();
      DiffractionGratingBlock candidate, physical;
      if (!options.reference_solver(profile, wavelength, kx, ky, options.half_orders,
                                    options.retained_half_orders, candidate, error) ||
          !diffraction_grating_match_reference(profile, wavelength, kx, ky, candidate,
                                              physical, error)) {
        return false;
      }
      reference = std::move(candidate);
      return true;
    }
    return diffraction_grating_solve_reference(profile, wavelength, kx, ky, options.half_orders,
                                               options.retained_half_orders, reference, error);
  }
  DiffractionModalOptions modal;
  modal.reference_solver = options.reference_solver;
  modal.minimum_half_orders = options.half_orders;
  modal.maximum_half_orders = options.maximum_half_orders;
  modal.power_tolerance = options.modal_power_tolerance;
  modal.complex_tolerance = options.modal_complex_tolerance;
  modal.progress = [&](int) {
    if (options.cancelled && options.cancelled()) return false;
    solve_count++;
    return true;
  };
  std::vector<DiffractionModalObservation> observations;
  const bool solved = diffraction_grating_converged_reference(profile, wavelength, kx, ky,
      options.retained_half_orders, modal, reference, observations, error);
  if (!solved) {
    std::ostringstream details;
    details << std::setprecision(17) << error << "; wavelength=" << wavelength
            << " kx=" << kx << " ky=" << ky;
    for (const auto &observation : observations) {
      details << "; N=" << observation.half_orders
              << " adjacent_power=" << observation.adjacent_power_difference
              << " spanning_power=" << observation.spanning_power_difference
              << " adjacent_complex=" << observation.adjacent_complex_difference
              << " spanning_complex=" << observation.spanning_complex_difference;
    }
    error = details.str();
  }
  return solved;
}
CCL_NAMESPACE_END
