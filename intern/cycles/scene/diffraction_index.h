/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "scene/diffraction.h"
#include "util/vector.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <cctype>
#include <sstream>

CCL_NAMESPACE_BEGIN

/* Nonuniform piecewise-linear n,k table. The real component can be stored
 * relative to the constant index so an optional table can adjust the existing
 * ridge/groove phase contrast without adding per-closure parameters.
 * Layout: knot count, then triples (wavelength_nm, n-real_origin, k).
 * Reject loss of knot ordering on conversion to device float precision. */
inline bool diffraction_pack_index_table(const std::vector<DiffractionIndexSample> &samples,
                                         const double real_origin,
                                         const bool require_lossless,
                                         const bool require_absorbing,
                                         vector<float> &packed,
                                         std::string &error)
{
  if (samples.size() < 2 || samples.size() > (1u << 20) || !std::isfinite(real_origin)) {
    error = "Optical constants require 2 to 1048576 finite, ordered samples";
    return false;
  }
  vector<float> data;
  data.reserve(1 + 3 * samples.size());
  data.push_back(float(samples.size()));
  float previous = 0;
  for (const auto &sample : samples) {
    const float wavelength = float(sample.wavelength);
    const float n = float(sample.index.real() - real_origin);
    const float k = float(sample.index.imag());
    if (!std::isfinite(sample.wavelength) || !std::isfinite(wavelength) ||
        !(wavelength > previous) || !std::isfinite(sample.index.real()) ||
        sample.index.real() < 0 || !std::isfinite(n) || !std::isfinite(sample.index.imag()) ||
        sample.index.imag() < 0 || !std::isfinite(k) ||
        (require_lossless && sample.index.imag() != 0) ||
        (require_absorbing && !(k > 0))) {
      error = "Invalid optical constants, non-passive medium, or unresolved float wavelength spacing";
      return false;
    }
    data.push_back(wavelength);
    data.push_back(n);
    data.push_back(k);
    previous = wavelength;
  }
  if (samples.front().wavelength > 380 || samples.back().wavelength < 780 ||
      data[1] > 380 || previous < 780) {
    error = "Optical constants must cover the complete 380 to 780 nm render interval";
    return false;
  }
  packed = std::move(data);
  error.clear();
  return true;
}

/* Embedded conductor data for Blender: an explicit CSV header avoids silently
 * interpreting micrometers as nanometers. Both ridge and substrate use it. */
inline bool diffraction_parse_conductor_csv(const std::string &contents,
                                            std::vector<DiffractionIndexSample> &samples,
                                            std::string &error)
{
  std::istringstream input(contents);
  std::string line;
  bool header = false;
  std::vector<DiffractionIndexSample> parsed;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (const auto comment = line.find('#'); comment != std::string::npos) {
      line.erase(comment);
    }
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    if (!header) {
      line.erase(std::remove_if(line.begin(), line.end(),
                               [](unsigned char c) { return std::isspace(c); }), line.end());
      if (line != "wavelength_nm,n,k") {
        error = "Optical constants CSV must begin with wavelength_nm,n,k";
        return false;
      }
      header = true;
      continue;
    }
    const auto malformed = [&]() {
      error = "Malformed optical constants CSV at line " + std::to_string(line_number);
      return false;
    };
    if (std::count(line.begin(), line.end(), ',') != 2) return malformed();
    const auto comma1 = line.find(','), comma2 = line.find(',', comma1 + 1);
    double values[3];
    const std::string fields[] = {line.substr(0, comma1),
                                  line.substr(comma1 + 1, comma2 - comma1 - 1),
                                  line.substr(comma2 + 1)};
    for (int i = 0; i < 3; ++i) {
      std::istringstream field(fields[i]);
      if (!(field >> values[i])) return malformed();
      field >> std::ws;
      if (!field.eof()) return malformed();
    }
    if (parsed.size() >= (1u << 20)) {
      error = "Too many optical constants samples";
      return false;
    }
    parsed.push_back({values[0], {values[1], values[2]}});
  }
  vector<float> packed;
  if (!diffraction_pack_index_table(parsed, 0, false, true, packed, error)) return false;
  samples = std::move(parsed);
  return true;
}

CCL_NAMESPACE_END
