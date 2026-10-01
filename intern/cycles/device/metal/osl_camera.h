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
  METAL_OSL_CAMERA_INPUT_NUM = 13,
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

/* Values of the parameters, for the `camera_script_params` kernel array. */
void metal_osl_camera_pack_params(const MetalOSLCameraProgram &program,
                                  const OSLCameraParams &params,
                                  vector<uint> &words);

CCL_NAMESPACE_END

#endif /* WITH_METAL */
