/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "util/defines.h"
#include "util/types.h"
#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Unit-reflector, one-sided native GGX diffraction. Lengths are nanometers.
 * Wavelengths are vacuum wavelengths; the incident medium changes the
 * wavelength at the grating. This is a geometry table, without Fresnel. */
struct DiffractionAlbedoRequest {
  float alpha_x = 0.25f;
  float alpha_y = 0.25f;
  float pitch_nm = 1600.0f;
  float depth_nm = 150.0f;
  float duty = 0.5f;
  float medium_ior = 1.0f;
  /* Thin-sheet mode stores a completed reciprocal component's missing energy
   * rather than unit-reflector albedo. alpha_x/y then mean reflection and
   * transmission isotropic alpha at the d-line respectively. The builder derives
   * transmission alpha from alpha_x and n(lambda), matching native ThinWall.
   * Runtime material blend is not
   * cached. Coefficients are passive scalar channel tints. */
  bool thin_sheet = false;
  float inside_ior = 1.5f;
  float inv_abbe = 0.0f;
  float film_ior = 1.0f;
  float film_thickness_nm = 0.0f;
  float reflection_tint = 1.0f;
  float transmission_tint = 1.0f;
  bool transmission_is_spectral = false;
  float3 transmission_bt709 = make_float3(1.0f, 1.0f, 1.0f);
  float wavelength_min_nm = 380.0f;
  float wavelength_max_nm = 780.0f;
  int wavelength_count = 16;
  int mu_count = 8;
  int phi_count = 16;
  int facet_samples = 512;
  int algorithm_revision = 1;

  bool operator==(const DiffractionAlbedoRequest &) const = default;
};

/* Values are albedo (ordinary mode), or missing energy q (thin-sheet mode).
 * Layout is [wavelength][mu][phi]. Mu nodes are (i/(mu_count-1))^2;
 * phi nodes are periodic bin centers. Averages integrate the exact linear
 * interpolation in physical mu and azimuth against 2*mu. */
struct DiffractionAlbedoTable {
  DiffractionAlbedoRequest request;
  std::vector<float> values;
  std::vector<float> averages;
};

bool diffraction_albedo_validate_request(const DiffractionAlbedoRequest &request,
                                         std::string &error);
bool diffraction_albedo_validate_table(const DiffractionAlbedoTable &table, std::string &error);
bool diffraction_albedo_build_cpu(const DiffractionAlbedoRequest &request,
                                  DiffractionAlbedoTable &table,
                                  std::string &error,
                                  const std::function<bool()> &cancelled = {});
float diffraction_albedo_lookup(const DiffractionAlbedoTable &table,
                                float wavelength_nm,
                                const float3 &direction);
float diffraction_albedo_average(const DiffractionAlbedoTable &table, float wavelength_nm);
/* Enforce the physical even azimuth symmetry of the two exterior-air ports,
 * preserving each mu row's sum (and therefore its projected normalization). */
void diffraction_albedo_symmetrize_thin_sheet(DiffractionAlbedoTable &table);
/* Exact native bounded spectral-transmission reconstruction for cache profiles. */
float diffraction_albedo_thin_sheet_transmission_tint(
    const DiffractionAlbedoRequest &request, float wavelength_nm);

/* Lossless two-sided Fast dielectric return cache. Directional
 * values are the missing single-event energy q on each incident side.
 * Averages are Q = integral q n^2 |cos(theta)| dOmega, not RGB tint. Side 0
 * is exterior air, side 1 is the material IOR. */
struct DiffractionTwoSidedAlbedoRequest {
  float alpha_x = 0.25f;
  float alpha_y = 0.25f;
  float pitch_nm = 1600.0f;
  float depth_nm = 150.0f;
  float duty = 0.5f;
  /* Native d-line IOR and Cauchy inverse Abbe parameter. */
  float inside_ior = 1.5f;
  float inv_abbe = 0.0f;
  /* 0: physical Fresnel; 2: bare generalized endpoints; 16: coated F0 grid. */
  int generalized_f0_count = 0;
  float film_ior = 1.0f;
  float film_thickness_nm = 0.0f;
  int wavelength_count = 16;
  int mu_count = 8;
  int phi_count = 16;
  int facet_samples = 512;
  int algorithm_revision = 1;

  bool operator==(const DiffractionTwoSidedAlbedoRequest &) const = default;
};

struct DiffractionTwoSidedAlbedoTable {
  DiffractionTwoSidedAlbedoRequest request;
  /* [basis][side][wavelength][mu][phi]; physical tables have one basis. */
  std::vector<float> deficits;
  /* Directional single-event albedos. Kept on host to validate R+T+q=1;
   * the renderer uploads only the deficit needed by the return lobe. */
  std::vector<float> reflectance;
  std::vector<float> transmittance;
  /* [basis][side][wavelength], etendue-integrated Q. */
  std::vector<float> integrals;
  /* [basis][wavelength], diffuse Fresnel transmission conductance normalized to
   * the smaller hemisphere's etendue. An approximate return-side mixer. */
  std::vector<float> cross_fractions;
};

bool diffraction_two_sided_albedo_build_cpu(
    const DiffractionTwoSidedAlbedoRequest &request,
    DiffractionTwoSidedAlbedoTable &table,
    std::string &error,
    const std::function<bool()> &cancelled = {});
bool diffraction_two_sided_albedo_validate_request(
    const DiffractionTwoSidedAlbedoRequest &request, std::string &error);
/* Metal and CPU builders both produce directional R/T. This shared host step
 * checks their raw energy before clamping and derives q, etendue Q and the
 * coated cross-side conductance with identical table invariants. */
bool diffraction_two_sided_albedo_complete_from_directional(
    DiffractionTwoSidedAlbedoTable &table, std::string &error);
bool diffraction_two_sided_albedo_validate(const DiffractionTwoSidedAlbedoTable &table,
                                           std::string &error);
float diffraction_two_sided_albedo_lookup(const DiffractionTwoSidedAlbedoTable &table,
                                          int side,
                                          float wavelength_nm,
                                          const float3 &direction,
                                          int basis = 0);

CCL_NAMESPACE_END
