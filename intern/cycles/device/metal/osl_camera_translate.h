/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Translation of compiled OSL camera shaders (.oso bytecode) to the Metal shading language.
 *
 * This file only depends on the C++ standard library, so the translator can also be built as a
 * standalone tool for testing, see `tests/python/cycles_osl_camera_metal.py`. The Cycles facing
 * interface is in `osl_camera.h`. */

#include <map>
#include <string>
#include <vector>

CCL_NAMESPACE_BEGIN

/* A shader parameter that has a value assigned by the camera. */
struct OSLCameraTranslateParam {
  enum Type { INT, FLOAT, STRING };
  Type type = FLOAT;
  /* Number of components for INT and FLOAT. */
  int size = 1;
  /* Value for STRING. Strings are constants of the translation. */
  std::string string_value;
};

struct OSLCameraTranslateOptions {
  /* Parameters with a value, by name. */
  std::map<std::string, OSLCameraTranslateParam> params;
  /* Interop ID of the scene linear color space, for color space conversions. */
  std::string colorspace = "lin_rec709_scene";
  /* CIE 1931 color matching functions from 380 to 780 nm in 5 nm steps, 81 XYZ triplets. */
  const float *cie_color_match = nullptr;
};

struct OSLCameraTranslateResult {
  std::string source;

  /* Parameters read at render time from an array of 32 bit words. */
  struct Slot {
    std::string name;
    int size;
    int offset;
    bool is_int;
  };
  std::vector<Slot> slots;
  int num_words = 0;
};

bool osl_camera_translate_msl(const std::string &bytecode,
                              const OSLCameraTranslateOptions &options,
                              OSLCameraTranslateResult &result,
                              std::string &error);

CCL_NAMESPACE_END
