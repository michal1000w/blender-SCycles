/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_METAL

#  include "device/metal/osl_camera.h"
#  include "device/metal/osl_camera_translate.h"

#  include "kernel/types.h"

#  include "kernel/tables.h"

#  include "util/md5.h"

CCL_NAMESPACE_BEGIN

static bool osl_camera_param_convert(const TypeDesc &type, OSLCameraTranslateParam &param)
{
  if (type.arraylen != 0) {
    return false;
  }
  if (type.basetype == TypeDesc::STRING) {
    param.type = OSLCameraTranslateParam::STRING;
    param.size = 0;
    return true;
  }
  if (type.basetype == TypeDesc::INT) {
    param.type = OSLCameraTranslateParam::INT;
  }
  else if (type.basetype == TypeDesc::FLOAT) {
    param.type = OSLCameraTranslateParam::FLOAT;
  }
  else {
    return false;
  }
  param.size = int(type.aggregate);
  return true;
}

string metal_osl_camera_program_key(const string &bytecode_hash,
                                    const OSLCameraParams &params,
                                    const string &colorspace)
{
  MD5Hash md5;
  md5.append(bytecode_hash);
  md5.append(colorspace);
  for (const auto &it : params) {
    OSLCameraTranslateParam param;
    if (!osl_camera_param_convert(it.second.second, param)) {
      continue;
    }
    md5.append(it.first.string());
    md5.append(string_printf(":%d:%d:", int(param.type), param.size));
    if (param.type == OSLCameraTranslateParam::STRING) {
      md5.append(string((const char *)it.second.first.data()));
    }
  }
  return md5.get_hex();
}

bool metal_osl_camera_translate(const string &bytecode,
                                const OSLCameraParams &params,
                                const string &colorspace,
                                MetalOSLCameraProgram &program,
                                string &error)
{
  OSLCameraTranslateOptions options;
  options.colorspace = colorspace;
  static_assert(sizeof(cie_color_match) == sizeof(float) * 81 * 3, "Unexpected CIE table size");
  options.cie_color_match = &cie_color_match[0][0];
  for (const auto &it : params) {
    OSLCameraTranslateParam param;
    if (!osl_camera_param_convert(it.second.second, param)) {
      continue;
    }
    if (param.type == OSLCameraTranslateParam::STRING) {
      param.string_value = (const char *)it.second.first.data();
    }
    options.params[it.first.string()] = param;
  }

  OSLCameraTranslateResult result;
  if (!osl_camera_translate_msl(bytecode, options, result, error)) {
    return false;
  }

  program.source = result.source;
  program.params.clear();
  for (const OSLCameraTranslateResult::Slot &slot : result.slots) {
    program.params.push_back({ustring(slot.name), slot.size, slot.offset, slot.is_int});
  }
  program.num_param_words = result.num_words;
  return true;
}

void metal_osl_camera_pack_params(const MetalOSLCameraProgram &program,
                                  const OSLCameraParams &params,
                                  vector<uint> &words)
{
  words.clear();
  words.resize(program.num_param_words, 0);
  for (const MetalOSLCameraProgram::Param &param : program.params) {
    const auto it = params.find(param.name);
    if (it == params.end()) {
      continue;
    }
    const vector<uint8_t> &data = it->second.first;
    const size_t size = sizeof(uint) * param.size;
    if (data.size() >= size && param.offset + param.size <= program.num_param_words) {
      memcpy(&words[param.offset], data.data(), size);
    }
  }
}

CCL_NAMESPACE_END

#endif /* WITH_METAL */
