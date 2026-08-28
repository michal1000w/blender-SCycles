/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <gtest/gtest.h>

#include "kernel/tables.h"

#include "util/math.h"
#include "util/types.h"

CCL_NAMESPACE_BEGIN

TEST(KernelSpectralTransmission, PrimariesAreBoundedAndNeutralPreserving)
{
  for (int i = 0; i <= WAVELENGTH_RESOLUTION; i++) {
    const float *basis = cie1931_bt709_spectral_primaries[i];
    EXPECT_GE(basis[0], 0.0f);
    EXPECT_GE(basis[1], 0.0f);
    EXPECT_GE(basis[2], 0.0f);
    EXPECT_LE(basis[0], 1.0f);
    EXPECT_LE(basis[1], 1.0f);
    EXPECT_LE(basis[2], 1.0f);
    EXPECT_NEAR(basis[0] + basis[1] + basis[2], 1.0f, 1e-5f);
  }
}

TEST(KernelSpectralTransmission, PrimariesRoundTripUnderD65)
{
  /* XYZ to linear BT.709, matching ColorSpaceManager::get_xyz_to_rec709(). */
  const float3 xyz_to_rec709[3] = {
      make_float3(3.2404542f, -1.5371385f, -0.4985314f),
      make_float3(-0.9692660f, 1.8760108f, 0.0415560f),
      make_float3(0.0556434f, -0.2040259f, 1.0572252f),
  };

  for (int primary = 0; primary < 3; primary++) {
    float3 xyz = zero_float3();
    for (int i = 0; i <= WAVELENGTH_RESOLUTION; i++) {
      xyz += make_float3(cie_color_match[i][0],
                         cie_color_match[i][1],
                         cie_color_match[i][2]) *
             (cie_d65_spd[i] * cie1931_bt709_spectral_primaries[i][primary]);
    }

    /* The hero-wavelength PDF is expressed over the normalized 380--780nm interval. */
    xyz *= CIE_D65_NORMALIZATION / float(WAVELENGTH_RESOLUTION);
    const float3 rec709 = make_float3(
        dot(xyz_to_rec709[0], xyz), dot(xyz_to_rec709[1], xyz), dot(xyz_to_rec709[2], xyz));

    for (int channel = 0; channel < 3; channel++) {
      EXPECT_NEAR(rec709[channel], (channel == primary) ? 1.0f : 0.0f, 2e-4f);
    }
  }
}

CCL_NAMESPACE_END
