/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Translation of compiled OSL camera shaders (.oso bytecode) to the Metal shading language.
 *
 * This file only depends on the C++ standard library, so the translator can also be built as a
 * standalone tool for testing, see `tests/python/cycles_osl_camera_metal.py`. The Cycles facing
 * interface is in `osl_camera.h`. */

#include <functional>
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

/* The dictionaries that shaders query with dict_find(): XML documents and XPath queries, which
 * the host evaluates while the shader is translated. Nodes are numbered from 1. */
class OSLCameraDictionary {
 public:
  virtual ~OSLCameraDictionary() = default;
  /* First node matching `query` in the document, 0 if there is none, -1 if the document is not
   * valid. The same query returns the same node. */
  virtual int find(const std::string &dictionary, const std::string &query) = 0;
  /* First node matching `query` relative to `node`. */
  virtual int find(int node, const std::string &query) = 0;
  /* The next node matching the query that found `node`, or 0. */
  virtual int next(int node) = 0;
  /* Text of an attribute of the node, or of the node itself for an empty name. */
  virtual bool value(int node, const std::string &attribute, std::string &text) = 0;
  /* Nodes found so far. */
  virtual int num_nodes() = 0;
};

struct OSLCameraTranslateOptions {
  /* Parameters with a value, by name. */
  std::map<std::string, OSLCameraTranslateParam> params;
  /* Interop ID of the scene linear color space, for color space conversions. */
  std::string colorspace = "lin_rec709_scene";
  /* CIE 1931 color matching functions from 380 to 780 nm in 5 nm steps, 81 XYZ triplets. */
  const float *cie_color_match = nullptr;
  /* The hash() of a string, which is the hash of the string type of the OSL runtime. */
  std::function<int(const std::string &)> string_hash;
  /* For dict_find(), dict_next() and dict_value(). */
  OSLCameraDictionary *dictionary = nullptr;
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

  /* Images that the shader looks up. The host stores the kernel ID of each image in the word
   * at `offset` of the parameter array. */
  struct Image {
    std::string filename;
    int offset;
  };
  std::vector<Image> images;

  int num_words = 0;
};

bool osl_camera_translate_msl(const std::string &bytecode,
                              const OSLCameraTranslateOptions &options,
                              OSLCameraTranslateResult &result,
                              std::string &error);

CCL_NAMESPACE_END
