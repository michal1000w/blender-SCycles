/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "scene/diffraction.h"
CCL_NAMESPACE_BEGIN
/* Prepare one lossless matrix-anchored cell using 7^3 direct reference solves.
 * This fits training data only: independent physical validation must decide
 * whether to accept the cell or subdivide it. Failure publishes no candidate.
 * All ports use the artificial reference basis, including evanescent orders. */
bool diffraction_grating_prepare_tensor_cell(const DiffractionGratingProfile &profile,
                                             const DiffractionGratingCellBounds &bounds,
                                             const DiffractionGratingCacheOptions &options,
                                             const double compression_tolerance,
                                             DiffractionGratingPackedCell &cell,
                                             std::string &error);
/* Evaluate a prepared cell in the artificial reference basis. Uses float
 * arithmetic for interpolation and reconstruction, matching device storage.
 * Physical exterior matching and acceptance tests remain separate. The output
 * boundary_residual measures the Cayley linear solve, not Maxwell convergence. */
bool diffraction_grating_tensor_reference(const DiffractionGratingPackedCell &cell,
                                          const std::array<double, 3> &query,
                                          DiffractionGratingBlock &reference,
                                          std::string &error);
/* Adaptive lossless tensor cache. Acceptance uses independent physical-power
 * and optional complex-amplitude probes, with a fourfold construction margin.
 * A finite validation set does not establish a uniform or modal error bound. */
bool diffraction_grating_build_tensor_cache(const DiffractionGratingProfile &profile,
                                            const DiffractionGratingCacheOptions &options,
                                            DiffractionGratingCache &cache,
                                            DiffractionGratingCacheStats &stats,
                                            std::string &error);
CCL_NAMESPACE_END
