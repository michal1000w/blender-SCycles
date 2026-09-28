/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

/* Shared between material preparation and the default kernel specialization.
 * Each diffraction port carries two polarization channels. */
#define DIFFRACTION_MAX_CHANNELS 20
#define DIFFRACTION_FAST_CACHE_HANDLE (-2)
