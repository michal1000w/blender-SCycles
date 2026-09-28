/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "scene/diffraction.h"
CCL_NAMESPACE_BEGIN
struct DiffractionModalOptions {
  DiffractionReferenceSolver reference_solver;
  int minimum_half_orders = 16;
  int maximum_half_orders = 512;
  double power_tolerance = 0.001;
  /* Zero disables complex-amplitude comparison. */
  double complex_tolerance = 0.0;
  /* Called before each solve. False cancels without publishing an operator. */
  std::function<bool(int)> progress;
};
struct DiffractionModalObservation {
  int half_orders = 0;
  int comparisons = 0;
  double adjacent_power_difference = 0, spanning_power_difference = 0;
  double adjacent_complex_difference = 0, spanning_complex_difference = 0;
};
/* Empirical modal convergence of the complete physical scattering operator.
 * Require two successive resolutions and their spanning comparison to agree.
 * This is a numerical stopping criterion, not a rigorous truncation bound or
 * validation of interpolation between queries. Failure leaves reference empty
 * and retains observations for diagnosis. No power normalization is applied. */
bool diffraction_grating_converged_reference(
    const DiffractionGratingProfile &profile, double wavelength, double kx, double ky,
    int retained_half_orders, const DiffractionModalOptions &options,
    DiffractionGratingBlock &reference, std::vector<DiffractionModalObservation> &observations,
    std::string &error);
/* Shared cache-builder entry point; solve_count includes attempted modal solves. */
bool diffraction_grating_cache_reference(
    const DiffractionGratingProfile &profile, const DiffractionGratingCacheOptions &options,
    double wavelength, double kx, double ky, DiffractionGratingBlock &reference,
    size_t &solve_count, std::string &error);
CCL_NAMESPACE_END
