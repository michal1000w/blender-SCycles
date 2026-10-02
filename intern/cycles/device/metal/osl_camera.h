/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_METAL

#  include "util/map.h"
#  include "util/param.h"
#  include "util/string.h"
#  include "util/types.h"
#  include "util/vector.h"

CCL_NAMESPACE_BEGIN

/* Custom (OSL) cameras on Metal.
 *
 * Metal has no OSL back end, so the compiled bytecode of a camera shader (.oso) is translated to
 * a Metal function. The device compiles it on its own and links it into the camera ray kernel,
 * which calls it through a visible function table, see `camera_sample_custom()`.
 *
 * Numeric shader parameters are not baked into the function: it reads them from the
 * `camera_script_params` kernel array, so animating them does not recompile anything. */

/* Name of the translated function in the library compiled from MetalOSLCameraProgram::source. */
#  define METAL_OSL_CAMERA_FUNCTION_NAME "cycles_metal_osl_camera"

/* Values passed to the function by the kernel, see `camera_sample_custom()`. */
enum MetalOSLCameraInput {
  METAL_OSL_CAMERA_INPUT_SENSOR = 0,
  METAL_OSL_CAMERA_INPUT_DSDX = 3,
  METAL_OSL_CAMERA_INPUT_DSDY = 6,
  METAL_OSL_CAMERA_INPUT_RAND_LENS = 9,
  METAL_OSL_CAMERA_INPUT_APERTURE_POSITION = 11,
  /* Number of image lookup results that follow, each five floats: four values and a status. */
  METAL_OSL_CAMERA_INPUT_NUM_RESULTS = 13,
  METAL_OSL_CAMERA_INPUT_RESULTS = 14,
};

/* The function cannot look up images itself. It returns a request instead of a ray, in the
 * output array after the ray, and the kernel calls it again with the result appended to the
 * input, see `camera_custom_image_request()`. A ray can use this many lookups. */
#  define METAL_OSL_CAMERA_MAX_IMAGE_LOOKUPS 32

/* An image lookup request, as offsets after the ray in the output array. */
enum MetalOSLCameraRequest {
  /* 0 for no request, 1 texture(), 2 texture3d(), 3 environment(), 4 gettextureinfo(), 5
   * gettextureinfo() with coordinates. */
  METAL_OSL_CAMERA_REQUEST_TYPE = 0,
  /* Kernel image ID, as the bits of an integer. */
  METAL_OSL_CAMERA_REQUEST_IMAGE = 1,
  /* s, t, ds/dx, dt/dx, ds/dy, dt/dy, or a position or direction. */
  METAL_OSL_CAMERA_REQUEST_COORDS = 2,
  /* Whether a color for missing images follows, and the color. */
  METAL_OSL_CAMERA_REQUEST_HAS_MISSING = 8,
  METAL_OSL_CAMERA_REQUEST_MISSING = 9,
  /* For gettextureinfo(): 1 resolution, 2 channels, 3 exists, 4 averagecolor. */
  METAL_OSL_CAMERA_REQUEST_INFO = 13,
  METAL_OSL_CAMERA_REQUEST_NUM = 14,
};

/* Layout is {P, dPdx, dPdy, D, dDdx, dDdy, T}, as for the other devices. */
#  define METAL_OSL_CAMERA_OUTPUT_NUM 21

using OSLCameraParams = map<ustring, pair<vector<uint8_t>, TypeDesc>>;

struct MetalOSLCameraProgram {
  /* Metal source of a library containing the camera function. It reads kernel data through
   * `OSL_KD_*` offsets, which the device defines in front of the source. */
  string source;

  /* A parameter whose value is read at render time. */
  struct Param {
    ustring name;
    /* Number of 32 bit words. */
    int size;
    /* Offset in words into the `camera_script_params` kernel array. */
    int offset;
    bool is_int;
  };
  vector<Param> params;
  int num_param_words = 0;

  /* An image that the shader looks up. Its kernel image ID is a parameter word. */
  struct Image {
    string filename;
    int offset;
  };
  vector<Image> images;

  /* Identifies everything the translation depends on besides the bytecode: the set of
   * parameters with a value, and the values of the string parameters. */
  string key;
};

/* Key of the translation for `params`, see MetalOSLCameraProgram::key. */
string metal_osl_camera_program_key(const string &bytecode_hash,
                                    const OSLCameraParams &params,
                                    const string &colorspace);

/* Translate the bytecode of a camera shader. `params` are the parameters with a value assigned
 * by the camera, all others use the defaults of the shader. Returns false and sets `error` when
 * the shader uses a feature that is not supported. */
bool metal_osl_camera_translate(const string &bytecode,
                                const OSLCameraParams &params,
                                const string &colorspace,
                                MetalOSLCameraProgram &program,
                                string &error);

/* Values of the parameters, for the `camera_script_params` kernel array. `image_ids` are the
 * kernel image IDs of MetalOSLCameraProgram::images. */
void metal_osl_camera_pack_params(const MetalOSLCameraProgram &program,
                                  const OSLCameraParams &params,
                                  const vector<int> &image_ids,
                                  vector<uint> &words);

CCL_NAMESPACE_END

#endif /* WITH_METAL */
