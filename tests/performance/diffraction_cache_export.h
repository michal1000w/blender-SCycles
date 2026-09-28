/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
/* Versioned audit artifact, not a Blender scene or production disk-cache format.
 * Explicit packed buffers are directly suitable for Metal fixture uploads. */
#include "scene/diffraction.h"
#include "util/math.h"
#include <bit>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <type_traits>

namespace diffraction_audit {
using namespace ccl;
inline bool export_cache(const std::filesystem::path &directory,
                         const DiffractionGratingProfile &profile,
                         const DiffractionGratingCache &cache,
                         std::string &error)
{
  static_assert(std::endian::native == std::endian::little);
  static_assert(sizeof(float2) == 8 && sizeof(float4) == 16 && sizeof(int2) == 8 &&
                sizeof(int4) == 16);
  if (cache.nodes.empty()) {
    error = "Cannot export an empty cache";
    return false;
  }
  std::error_code ec;
  std::filesystem::create_directories(directory, ec);
  if (ec) {
    error = "Cannot create cache artifact directory: " + ec.message();
    return false;
  }
  const auto binary = directory / "cache.bin", manifest = directory / "cache.json";
  if (std::filesystem::exists(binary) || std::filesystem::exists(manifest)) {
    error = "Cache artifact already exists; choose a new output directory";
    return false;
  }
  DiffractionGratingDeviceBuffers buffers;
  if (!diffraction_grating_device_buffers(cache, buffers, error))
    return false;
  const auto &layout = buffers.layout;
  const auto &bounds = buffers.bounds;
  const auto &ports = buffers.ports;
  const auto &active = buffers.active;
  const auto &matrices = buffers.matrices;
  std::ofstream data(binary, std::ios::binary), description(manifest);
  if (!data || !description) {
    error = "Cannot open cache artifact files";
    data.close();
    description.close();
    std::filesystem::remove(binary, ec);
    std::filesystem::remove(manifest, ec);
    return false;
  }
  description << std::setprecision(17)
              << "{\"schema\":\"cycles-diffraction-audit\",\"version\":1,\"byte_order\":"
                 "\"little\",\"binary\":\"cache.bin\","
              << "\"mirror_symmetry\":" << (cache.mirror_symmetry ? "true" : "false")
              << ",\"complex_validated\":" << (cache.complex_validated ? "true" : "false")
              << ",\"profile\":{\"pitch_nm\":" << profile.pitch
              << ",\"depth_nm\":" << profile.depth << ",\"duty\":" << profile.duty
              << ",\"incident_ior\":" << profile.incident_ior << ",\"ridge_ior\":["
              << profile.ridge_ior.real() << ',' << profile.ridge_ior.imag()
              << "],\"groove_ior\":[" << profile.groove_ior.real() << ','
              << profile.groove_ior.imag() << "],\"substrate_ior\":["
              << profile.substrate_ior.real() << ',' << profile.substrate_ior.imag()
              << "]},\"lower\":[";
  for (int axis = 0; axis < 3; axis++)
    description << (axis ? "," : "") << cache.bounds.lower[axis];
  description << "],\"upper\":[";
  for (int axis = 0; axis < 3; axis++)
    description << (axis ? "," : "") << cache.bounds.upper[axis];
  description << "],\"buffers\":{";
  size_t offset = 0;
  bool first = true;
  auto write = [&](const char *name, const auto &values, const char *type, int components) {
    const size_t bytes = values.size() *
                         sizeof(typename std::decay_t<decltype(values)>::value_type);
    const size_t padding = (16 - offset % 16) % 16;
    const char zeros[16] = {};
    if (padding)
      data.write(zeros, std::streamsize(padding));
    offset += padding;
    description << (first ? "" : ",") << '"' << name << "\":{\"offset\":" << offset
                << ",\"count\":" << values.size() << ",\"components\":" << components
                << ",\"type\":\"" << type << "\",\"bytes\":" << bytes << '}';
    if (bytes)
      data.write(reinterpret_cast<const char *>(values.data()), std::streamsize(bytes));
    offset += bytes;
    first = false;
  };
  write("nodes", buffers.nodes, "int32", 4);
  write("cell_layout", layout, "int32", 4);
  write("cell_bounds", bounds, "float32", 4);
  write("ports", ports, "int32", 2);
  write("active_ports", active, "int32", 1);
  write("matrices", matrices, "float32", 2);
  description << "},\"binary_bytes\":" << offset << "}\n";
  data.close();
  description.close();
  if (!data || !description) {
    error = "Cache artifact write failed";
    std::filesystem::remove(binary, ec);
    std::filesystem::remove(manifest, ec);
    return false;
  }
  return true;
}
}  // namespace diffraction_audit
